import argparse
import hashlib
import json
import os
from typing import Any

import psycopg2
from neo4j import GraphDatabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load triplets JSON into PostgreSQL and Neo4j."
    )
    parser.add_argument("--json", required=True, help="Path to triplets JSON file")
    parser.add_argument(
        "--source-name",
        default=None,
        help="Optional source name. Defaults to JSON filename.",
    )
    parser.add_argument(
        "--stage",
        default="postprocessed",
        choices=["llm", "postprocessed"],
        help="Extraction stage label written to SQL.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def slot_to_text(slot: Any) -> str:
    if slot is None:
        return ""
    if isinstance(slot, str):
        return normalize_whitespace(slot)
    if isinstance(slot, dict):
        value = slot.get("text", "")
        if isinstance(value, str):
            return normalize_whitespace(value)
    return normalize_whitespace(str(slot))


def role_frame(slot: Any) -> dict[str, Any]:
    if isinstance(slot, dict) and "text" in slot and "frame" in slot:
        return slot
    return {"text": slot_to_text(slot), "frame": []}


def ensure_pg_schema(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,
            source_name TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS triplets (
            id BIGSERIAL PRIMARY KEY,
            document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            row_idx INTEGER NOT NULL,
            extraction_stage TEXT NOT NULL DEFAULT 'postprocessed',
            subject_text TEXT NOT NULL,
            predicate_text TEXT NOT NULL,
            object_text TEXT NOT NULL,
            sentence_text TEXT,
            confidence DOUBLE PRECISION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(document_id, row_idx, extraction_stage, subject_text, predicate_text, object_text)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS triplet_frames (
            id BIGSERIAL PRIMARY KEY,
            triplet_id BIGINT NOT NULL REFERENCES triplets(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('subject', 'predicate', 'object')),
            frame_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(triplet_id, role)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS frame_instances (
            id BIGSERIAL PRIMARY KEY,
            triplet_id BIGINT NOT NULL REFERENCES triplets(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('subject', 'predicate', 'object')),
            root_text TEXT NOT NULL,
            root_norm TEXT NOT NULL,
            sentence_text TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (triplet_id, role)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS frame_nodes (
            id BIGSERIAL PRIMARY KEY,
            frame_instance_id BIGINT NOT NULL REFERENCES frame_instances(id) ON DELETE CASCADE,
            parent_node_id BIGINT REFERENCES frame_nodes(id) ON DELETE CASCADE,
            node_text TEXT NOT NULL,
            node_norm TEXT NOT NULL,
            node_lemma TEXT,
            edge_label TEXT,
            depth INTEGER NOT NULL DEFAULT 0,
            path TEXT NOT NULL,
            ord INTEGER NOT NULL DEFAULT 0,
            is_root BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS concepts (
            id BIGSERIAL PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            canonical_norm TEXT NOT NULL UNIQUE,
            concept_type TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_aliases (
            id BIGSERIAL PRIMARY KEY,
            concept_id BIGINT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            alias_text TEXT NOT NULL,
            alias_norm TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            confidence DOUBLE PRECISION,
            status TEXT NOT NULL DEFAULT 'candidate',
            approved_at TIMESTAMPTZ,
            review_note TEXT,
            UNIQUE (concept_id, alias_norm)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS node_concept_links (
            id BIGSERIAL PRIMARY KEY,
            node_id BIGINT NOT NULL REFERENCES frame_nodes(id) ON DELETE CASCADE,
            concept_id BIGINT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
            link_type TEXT NOT NULL DEFAULT 'candidate',
            score DOUBLE PRECISION,
            method TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (node_id, concept_id, link_type)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_triplets_document_id ON triplets(document_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject_text);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object_text);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_nodes_norm ON frame_nodes(node_norm);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_nodes_lemma ON frame_nodes(node_lemma);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_nodes_parent ON frame_nodes(parent_node_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_nodes_instance ON frame_nodes(frame_instance_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_nodes_path ON frame_nodes(path);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concept_aliases_norm ON concept_aliases(alias_norm);")
    cur.execute("ALTER TABLE concept_aliases ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'candidate';")
    cur.execute("ALTER TABLE concept_aliases ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;")
    cur.execute("ALTER TABLE concept_aliases ADD COLUMN IF NOT EXISTS review_note TEXT;")


def safe_entity_name(value: str) -> str:
    value = normalize_whitespace(value)
    return value.lower()


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "triplets" not in data:
        raise ValueError("JSON must be an object with key 'triplets'")
    if not isinstance(data["triplets"], list):
        raise ValueError("'triplets' must be a list")
    return data


def content_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_pg_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5433")),
        dbname=os.getenv("PGDATABASE", "triplets"),
        user=os.getenv("PGUSER", "triplets_user"),
        password=os.getenv("PGPASSWORD", "triplets_pass"),
    )


def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4jpass")
    return GraphDatabase.driver(uri, auth=(user, password))


def upsert_document(cur, source_name: str, hash_value: str) -> int:
    cur.execute(
        """
        INSERT INTO documents(source_name, content_hash)
        VALUES (%s, %s)
        ON CONFLICT(content_hash) DO UPDATE
          SET source_name = EXCLUDED.source_name
        RETURNING id;
        """,
        (source_name, hash_value),
    )
    return cur.fetchone()[0]


def insert_triplet(cur, document_id: int, row_idx: int, stage: str, triplet: dict[str, Any]) -> int:
    subject = slot_to_text(triplet.get("subject"))
    predicate = slot_to_text(triplet.get("predicate"))
    obj = slot_to_text(triplet.get("object"))
    sentence = triplet.get("sentence", None)
    if isinstance(sentence, dict):
        sentence = sentence.get("text", "")
    if sentence is not None:
        sentence = normalize_whitespace(str(sentence))

    cur.execute(
        """
        INSERT INTO triplets(
            document_id, row_idx, extraction_stage,
            subject_text, predicate_text, object_text, sentence_text, confidence
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
        ON CONFLICT(document_id, row_idx, extraction_stage, subject_text, predicate_text, object_text)
        DO UPDATE SET sentence_text = EXCLUDED.sentence_text
        RETURNING id;
        """,
        (document_id, row_idx, stage, subject, predicate, obj, sentence),
    )
    return cur.fetchone()[0]


def upsert_frame(cur, triplet_id: int, role: str, frame: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO triplet_frames(triplet_id, role, frame_json)
        VALUES (%s, %s, %s::jsonb)
        ON CONFLICT(triplet_id, role)
        DO UPDATE SET frame_json = EXCLUDED.frame_json;
        """,
        (triplet_id, role, json.dumps(frame, ensure_ascii=False)),
    )


def insert_frame_instance(
    cur,
    triplet_id: int,
    role: str,
    frame: dict[str, Any],
    sentence_text: str,
) -> int:
    root_text = slot_to_text(frame)
    cur.execute(
        """
        INSERT INTO frame_instances(triplet_id, role, root_text, root_norm, sentence_text)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (triplet_id, role)
        DO UPDATE SET
            root_text = EXCLUDED.root_text,
            root_norm = EXCLUDED.root_norm,
            sentence_text = EXCLUDED.sentence_text
        RETURNING id;
        """,
        (triplet_id, role, root_text, safe_entity_name(root_text), sentence_text),
    )
    return cur.fetchone()[0]


def clear_frame_nodes(cur, frame_instance_id: int) -> None:
    cur.execute("DELETE FROM frame_nodes WHERE frame_instance_id = %s;", (frame_instance_id,))


def insert_frame_nodes(
    cur,
    frame_instance_id: int,
    node: dict[str, Any],
    parent_node_id: int | None = None,
    depth: int = 0,
    path: str = "0",
    ord_: int = 0,
) -> int:
    node_text = slot_to_text(node)
    edge_label = None
    if isinstance(node, dict):
        edge_label = slot_to_text(node.get("edge_label")) or None

    cur.execute(
        """
        INSERT INTO frame_nodes(
            frame_instance_id, parent_node_id, node_text, node_norm, node_lemma,
            edge_label, depth, path, ord, is_root
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (
            frame_instance_id,
            parent_node_id,
            node_text,
            safe_entity_name(node_text),
            node_text,
            edge_label,
            depth,
            path,
            ord_,
            parent_node_id is None,
        ),
    )
    node_id = cur.fetchone()[0]

    children = []
    if isinstance(node, dict):
        raw_children = node.get("frame", [])
        if isinstance(raw_children, list):
            children = raw_children

    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            child = {"text": slot_to_text(child), "frame": []}
        insert_frame_nodes(
            cur,
            frame_instance_id=frame_instance_id,
            node=child,
            parent_node_id=node_id,
            depth=depth + 1,
            path=f"{path}.{idx}",
            ord_=idx,
        )

    return node_id


def upsert_frame_structure(
    cur,
    triplet_id: int,
    role: str,
    frame: dict[str, Any],
    sentence_text: str,
) -> int:
    frame_instance_id = insert_frame_instance(cur, triplet_id, role, frame, sentence_text)
    clear_frame_nodes(cur, frame_instance_id)
    insert_frame_nodes(cur, frame_instance_id, frame)
    return frame_instance_id


def frame_has_content(frame: dict[str, Any]) -> bool:
    if slot_to_text(frame):
        return True
    children = frame.get("frame", []) if isinstance(frame, dict) else []
    return isinstance(children, list) and any(frame_has_content(role_frame(child)) for child in children)


def flatten_frame_for_neo4j(
    frame: dict[str, Any],
    path: str = "0",
    depth: int = 0,
    ord_: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node = role_frame(frame)
    node_text = slot_to_text(node)
    nodes = [
        {
            "path": path,
            "text": node_text,
            "norm": safe_entity_name(node_text),
            "depth": depth,
            "ord": ord_,
            "is_root": depth == 0,
        }
    ]
    edges: list[dict[str, Any]] = []

    children = node.get("frame", [])
    if not isinstance(children, list):
        children = []

    for idx, child in enumerate(children):
        child_path = f"{path}.{idx}"
        child_nodes, child_edges = flatten_frame_for_neo4j(
            role_frame(child),
            path=child_path,
            depth=depth + 1,
            ord_=idx,
        )
        nodes.extend(child_nodes)
        edges.append({"parent_path": path, "child_path": child_path, "ord": idx})
        edges.extend(child_edges)

    return nodes, edges


def write_to_neo4j(
    driver,
    triplet_id: int,
    document_id: int,
    source_name: str,
    triplet: dict[str, Any],
) -> bool:
    subject_text = slot_to_text(triplet.get("subject"))
    predicate_text = slot_to_text(triplet.get("predicate"))
    object_text = slot_to_text(triplet.get("object"))
    sentence_text = slot_to_text(triplet.get("sentence", ""))

    params = {
        "triplet_id": triplet_id,
        "document_id": document_id,
        "source_name": source_name,
        "subject_text": subject_text,
        "subject_norm": safe_entity_name(subject_text),
        "predicate_text": predicate_text,
        "predicate_norm": safe_entity_name(predicate_text),
        "object_text": object_text,
        "object_norm": safe_entity_name(object_text),
        "sentence": sentence_text,
    }

    root_query = """
    MERGE (doc:Document {document_id: $document_id})
      ON CREATE SET doc.source_name = $source_name
      ON MATCH SET doc.source_name = coalesce(doc.source_name, $source_name)
    MERGE (trip:Triplet {triplet_id: $triplet_id})
    SET trip.document_id = $document_id,
        trip.sentence = $sentence,
        trip.subject_text = $subject_text,
        trip.predicate_text = $predicate_text,
        trip.object_text = $object_text
    MERGE (trip)-[:IN_DOCUMENT]->(doc);
    """

    cleanup_query = """
    MATCH (trip:Triplet {triplet_id: $triplet_id})
    OPTIONAL MATCH (trip)-[ctx]-()
    WHERE type(ctx) IN ['SUBJECT', 'OBJECT', 'PREDICATE', 'HAS_FRAME']
    WITH trip, collect(DISTINCT ctx) AS ctx_rels
    OPTIONAL MATCH ()-[legacy:RELATION {triplet_id: $triplet_id}]->()
    WITH trip, ctx_rels, collect(DISTINCT legacy) AS legacy_rels
    OPTIONAL MATCH ()-[rel:RELATION_INSTANCE {triplet_id: $triplet_id}]->()
    WITH trip, ctx_rels, legacy_rels, collect(DISTINCT rel) AS relation_rels
    OPTIONAL MATCH (trip)-[:HAS_FRAME]->(occ:FrameOccurrence)
    OPTIONAL MATCH (occ)-[:HAS_ROOT]->(root:FrameNode)
    OPTIONAL MATCH (root)-[:CHILD*0..]->(node:FrameNode)
    WITH ctx_rels, legacy_rels, relation_rels, collect(DISTINCT occ) AS occs, collect(DISTINCT node) AS nodes
    FOREACH (r IN ctx_rels | DELETE r)
    FOREACH (r IN legacy_rels | DELETE r)
    FOREACH (r IN relation_rels | DELETE r)
    FOREACH (n IN nodes | DETACH DELETE n)
    FOREACH (o IN occs | DETACH DELETE o);
    """

    with driver.session() as session:
        session.run(root_query, params)
        session.run(cleanup_query, {"triplet_id": triplet_id})

        if subject_text:
            session.run(
                """
                MATCH (trip:Triplet {triplet_id: $triplet_id})
                MERGE (subj:EntityConcept {norm: $subject_norm})
                  ON CREATE SET subj.name = $subject_text
                  ON MATCH SET subj.name = coalesce(subj.name, $subject_text)
                MERGE (trip)-[:SUBJECT]->(subj)
                MERGE (legacy:Entity {name_norm: $subject_norm})
                  ON CREATE SET legacy.name = $subject_text
                  ON MATCH SET legacy.name = coalesce(legacy.name, $subject_text);
                """,
                params,
            )

        if object_text:
            session.run(
                """
                MATCH (trip:Triplet {triplet_id: $triplet_id})
                MERGE (obj:EntityConcept {norm: $object_norm})
                  ON CREATE SET obj.name = $object_text
                  ON MATCH SET obj.name = coalesce(obj.name, $object_text)
                MERGE (trip)-[:OBJECT]->(obj)
                MERGE (legacy:Entity {name_norm: $object_norm})
                  ON CREATE SET legacy.name = $object_text
                  ON MATCH SET legacy.name = coalesce(legacy.name, $object_text);
                """,
                params,
            )

        if predicate_text:
            session.run(
                """
                MATCH (trip:Triplet {triplet_id: $triplet_id})
                MERGE (pred:RelationConcept {norm: $predicate_norm})
                  ON CREATE SET pred.name = $predicate_text
                  ON MATCH SET pred.name = coalesce(pred.name, $predicate_text)
                MERGE (trip)-[:PREDICATE]->(pred);
                """,
                params,
            )

        if subject_text and object_text:
            session.run(
                """
                MERGE (legacy_s:Entity {name_norm: $subject_norm})
                  ON CREATE SET legacy_s.name = $subject_text
                  ON MATCH SET legacy_s.name = coalesce(legacy_s.name, $subject_text)
                MERGE (legacy_o:Entity {name_norm: $object_norm})
                  ON CREATE SET legacy_o.name = $object_text
                  ON MATCH SET legacy_o.name = coalesce(legacy_o.name, $object_text)
                MERGE (legacy_s)-[legacy_r:RELATION {triplet_id: $triplet_id}]->(legacy_o)
                SET legacy_r.predicate = $predicate_text,
                    legacy_r.document_id = $document_id,
                    legacy_r.sentence = $sentence
                WITH legacy_s, legacy_o
                MERGE (subj:EntityConcept {norm: $subject_norm})
                  ON CREATE SET subj.name = $subject_text
                  ON MATCH SET subj.name = coalesce(subj.name, $subject_text)
                MERGE (obj:EntityConcept {norm: $object_norm})
                  ON CREATE SET obj.name = $object_text
                  ON MATCH SET obj.name = coalesce(obj.name, $object_text)
                MERGE (subj)-[rel:RELATION_INSTANCE {triplet_id: $triplet_id}]->(obj)
                SET rel.predicate = $predicate_text,
                    rel.predicate_norm = $predicate_norm,
                    rel.document_id = $document_id,
                    rel.sentence = $sentence;
                """,
                params,
            )

        for role in ("subject", "predicate", "object"):
            frame = role_frame(triplet.get(role))
            if not frame_has_content(frame):
                continue

            concept_text = slot_to_text(frame)
            concept_norm = safe_entity_name(concept_text)
            frame_nodes, frame_edges = flatten_frame_for_neo4j(frame)
            role_params = {
                **params,
                "role": role,
                "concept_text": concept_text,
                "concept_norm": concept_norm,
                "frame_nodes": frame_nodes,
                "frame_edges": frame_edges,
            }
            concept_label = "RelationConcept" if role == "predicate" else "EntityConcept"

            session.run(
                f"""
                MATCH (trip:Triplet {{triplet_id: $triplet_id}})
                MERGE (concept:{concept_label} {{norm: $concept_norm}})
                  ON CREATE SET concept.name = $concept_text
                  ON MATCH SET concept.name = coalesce(concept.name, $concept_text)
                MERGE (occ:FrameOccurrence {{triplet_id: $triplet_id, role: $role}})
                SET occ.document_id = $document_id,
                    occ.sentence = $sentence,
                    occ.root_text = $concept_text,
                    occ.root_norm = $concept_norm
                MERGE (trip)-[:HAS_FRAME {{role: $role}}]->(occ)
                MERGE (occ)-[:OF_CONCEPT]->(concept)
                WITH occ
                UNWIND $frame_nodes AS node
                MERGE (fn:FrameNode {{triplet_id: $triplet_id, role: $role, path: node.path}})
                SET fn.text = node.text,
                    fn.norm = node.norm,
                    fn.depth = node.depth,
                    fn.ord = node.ord,
                    fn.is_root = node.is_root
                WITH occ
                MATCH (root:FrameNode {{triplet_id: $triplet_id, role: $role, path: '0'}})
                MERGE (occ)-[:HAS_ROOT]->(root);
                """,
                role_params,
            )

            if frame_edges:
                session.run(
                    """
                    UNWIND $frame_edges AS edge
                    MATCH (parent:FrameNode {triplet_id: $triplet_id, role: $role, path: edge.parent_path})
                    MATCH (child:FrameNode {triplet_id: $triplet_id, role: $role, path: edge.child_path})
                    MERGE (parent)-[rel:CHILD]->(child)
                    SET rel.ord = edge.ord;
                    """,
                    role_params,
                )

    return bool(subject_text and object_text)


def main() -> None:
    args = parse_args()
    data = read_json(args.json)
    source_name = args.source_name or os.path.basename(args.json)

    pg_conn = get_pg_conn()
    neo4j_driver = get_neo4j_driver()

    loaded = 0
    graph_loaded = 0
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                doc_id = upsert_document(cur, source_name, content_hash(data))

                for idx, triplet in enumerate(data["triplets"], start=1):
                    if not isinstance(triplet, dict):
                        continue

                    triplet_id = insert_triplet(cur, doc_id, idx, args.stage, triplet)
                    sentence_text = slot_to_text(triplet.get("sentence", ""))
                    for role in ("subject", "predicate", "object"):
                        frame = role_frame(triplet.get(role))
                        upsert_frame(cur, triplet_id, role, frame)
                        upsert_frame_structure(cur, triplet_id, role, frame, sentence_text)
                    loaded += 1

                    if write_to_neo4j(neo4j_driver, triplet_id, doc_id, source_name, triplet):
                        graph_loaded += 1

        print(f"Loaded triplets into PostgreSQL: {loaded}")
        print(f"Loaded relations into Neo4j: {graph_loaded}")
        print(f"Document ID: {doc_id}")
    finally:
        pg_conn.close()
        neo4j_driver.close()


if __name__ == "__main__":
    main()
