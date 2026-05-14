import hashlib
import io
import json
import os
import traceback
from typing import Any
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape
import pandas as pd
import streamlit as st
from pandas.errors import ParserError
from llm_triplet_extraction import extract_triplets_by_llm
from triplets_from_text_extraction import process_triplets
from postprocess_triplets import remove_me_from_triplets
from scripts.mine_alias_candidates import mine_alias_candidates
from scripts.graph_analytics import analyze_graph, get_encoder_runtime_status

def change_to_frame_format(triplets: dict[str, Any]) -> dict[str, Any]:
    
    if isinstance(triplets, str):
        triplets = json.loads(triplets)
    triplets_list = triplets["triplets"]
    new_element = {"triplets": []}
    for triplet in triplets_list:
        subject = triplet["subject"]
        predicate = triplet["predicate"]
        obj = triplet["object"]
        new_element["triplets"].append(
            {
                "subject": {"text": subject, "frame": []},
                "predicate": {"text": predicate, "frame": []},
                "object": {"text": obj, "frame": []},
            }
        )
    return new_element


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def decode_bytes_safe(value: bytes) -> str:
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def parse_uploaded_csv(csv_bytes: bytes, preferred_sep: str) -> tuple[pd.DataFrame, str]:
    text = decode_bytes_safe(csv_bytes)
    df = pd.read_csv(io.StringIO(text), sep=preferred_sep)
    print(df.to_csv())
    return df, text


def slot_to_text(slot: Any) -> str:
    if slot is None:
        return ""
    if isinstance(slot, bytes):
        return normalize_whitespace(decode_bytes_safe(slot))
    if isinstance(slot, str):
        return normalize_whitespace(slot)
    if isinstance(slot, dict):
        value = slot.get("text", "")
        if isinstance(value, bytes):
            return normalize_whitespace(decode_bytes_safe(value))
        if isinstance(value, str):
            return normalize_whitespace(value)
    return normalize_whitespace(str(slot))


def role_frame(slot: Any) -> dict[str, Any]:
    if isinstance(slot, dict) and "text" in slot and "frame" in slot:
        return slot
    return {"text": slot_to_text(slot), "frame": []}


def safe_entity_name(value: str) -> str:
    return normalize_whitespace(value).lower()


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


def connect_pg_with_fallback(pg_cfg: dict[str, Any]):
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("Missing dependency: psycopg2-binary. Install requirements-graph.txt") from exc

    tried_encodings: list[str] = []
    last_conn_exc: Exception | None = None
    pg_conn = None
    for encoding in ("UTF8", "WIN1251", "LATIN1"):
        tried_encodings.append(encoding)
        try:
            pg_conn = psycopg2.connect(
                host=pg_cfg["host"],
                port=int(pg_cfg["port"]),
                dbname=pg_cfg["dbname"],
                user=pg_cfg["user"],
                password=pg_cfg["password"],
                options=f"-c client_encoding={encoding}",
            )
            with pg_conn.cursor() as test_cur:
                test_cur.execute("SELECT 1;")
                test_cur.fetchone()
            return pg_conn
        except Exception as exc:
            last_conn_exc = exc
            if pg_conn is not None:
                pg_conn.close()
                pg_conn = None

    raise RuntimeError(
        f"Cannot connect to PostgreSQL with encodings {tried_encodings}. "
        f"Last error: {last_conn_exc!r}"
    )


def payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return decode_bytes_safe(value)
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def save_triplets_to_databases(
    triplets_payload: dict[str, Any],
    source_name: str,
    extraction_stage: str,
    pg_cfg: dict[str, Any],
    neo_cfg: dict[str, Any],
) -> dict[str, int]:
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("Missing dependency: psycopg2-binary. Install requirements-graph.txt") from exc
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: neo4j. Install requirements-graph.txt") from exc

    if "triplets" not in triplets_payload or not isinstance(triplets_payload["triplets"], list):
        raise ValueError("Invalid payload format: expected {'triplets': [...]} ")

    triplets_payload = sanitize_for_json(triplets_payload)

    tried_encodings: list[str] = []
    last_conn_exc: Exception | None = None
    pg_conn = None
    for encoding in ("UTF8", "WIN1251", "LATIN1"):
        tried_encodings.append(encoding)
        try:
            pg_conn = psycopg2.connect(
                host=pg_cfg["host"],
                port=int(pg_cfg["port"]),
                dbname=pg_cfg["dbname"],
                user=pg_cfg["user"],
                password=pg_cfg["password"],
                options=f"-c client_encoding={encoding}",
            )
            with pg_conn.cursor() as test_cur:
                test_cur.execute("SELECT 1;")
                test_cur.fetchone()
            break
        except Exception as exc:
            last_conn_exc = exc
            if pg_conn is not None:
                pg_conn.close()
                pg_conn = None

    if pg_conn is None:
        raise RuntimeError(
            f"Cannot connect to PostgreSQL with encodings {tried_encodings}. "
            f"Last error: {last_conn_exc!r}"
        )

    driver = GraphDatabase.driver(neo_cfg["uri"], auth=(neo_cfg["user"], neo_cfg["password"]))

    sql_loaded = 0
    graph_loaded = 0

    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                cur.execute(
                    """
                    INSERT INTO documents(source_name, content_hash)
                    VALUES (%s, %s)
                    ON CONFLICT(content_hash) DO UPDATE SET source_name = EXCLUDED.source_name
                    RETURNING id;
                    """,
                    (source_name, payload_hash(triplets_payload)),
                )
                document_id = cur.fetchone()[0]

                for idx, triplet in enumerate(triplets_payload["triplets"], start=1):
                    if not isinstance(triplet, dict):
                        continue

                    subject = slot_to_text(triplet.get("subject"))
                    predicate = slot_to_text(triplet.get("predicate"))
                    obj = slot_to_text(triplet.get("object"))
                    sentence = slot_to_text(triplet.get("sentence", ""))

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
                        (document_id, idx, extraction_stage, subject, predicate, obj, sentence),
                    )
                    triplet_id = cur.fetchone()[0]
                    sql_loaded += 1

                    for role in ("subject", "predicate", "object"):
                        frame = role_frame(triplet.get(role))
                        cur.execute(
                            """
                            INSERT INTO triplet_frames(triplet_id, role, frame_json)
                            VALUES (%s, %s, %s::jsonb)
                            ON CONFLICT(triplet_id, role)
                            DO UPDATE SET frame_json = EXCLUDED.frame_json;
                            """,
                            (
                                triplet_id,
                                role,
                                json.dumps(frame, ensure_ascii=True),
                            ),
                        )
                        upsert_frame_structure(cur, triplet_id, role, frame, sentence)

                    if write_to_neo4j(
                        driver,
                        triplet_id=triplet_id,
                        document_id=document_id,
                        source_name=source_name,
                        triplet=triplet,
                    ):
                        graph_loaded += 1

        return {"sql_loaded": sql_loaded, "graph_loaded": graph_loaded, "document_id": document_id}
    finally:
        if pg_conn is not None:
            pg_conn.close()
        driver.close()


def render_result(triplets_result: dict[str, Any] | None, download_button_id: int = 0) -> None:
    if st.session_state["triplets_error"]:
        st.error(f"Ошибка извлечения триплетов: {st.session_state['triplets_error']}")
    elif triplets_result is not None:
       
        st.write("Результат извлечения:")
        st.json(triplets_result)

        json_bytes = json.dumps(triplets_result, ensure_ascii=False, indent=2).encode("utf-8")

        st.download_button(
            "Скачать JSON с триплетами",
            data=json_bytes,
            file_name=f"triplets.json",
            mime="application/json",
            key=f"download_triplets_{download_button_id}",
        )


def render_storage_section() -> None:
    has_main = st.session_state["main_result"] is not None
    has_processed = st.session_state["processed_result"] is not None
    if not has_main and not has_processed:
        return

    st.subheader("Сохранение в PostgreSQL и Neo4j")
    source_choice = st.radio(
        "Какой результат сохранить",
        options=["Обработанный", "Исходный"] if has_processed else ["Исходный"],
        index=0,
        key="save_source_choice",
    )

    payload = st.session_state["processed_result"] if source_choice == "Обработанный" else st.session_state["main_result"]
    default_stage = "postprocessed" if source_choice == "Обработанный" else "llm"

    col1, col2 = st.columns(2)
    with col1:
        source_name = st.text_input(
            "Имя источника (CSV)",
            value=st.session_state.get("main_source_name") or "uploaded.csv",
            key="db_source_name",
        )
        extraction_stage = st.selectbox(
            "Этап извлечения",
            options=["llm", "postprocessed"],
            index=0 if default_stage == "llm" else 1,
            key="db_stage",
        )
        pg_host = st.text_input("Хост PostgreSQL", value="127.0.0.1", key="pg_host")
        pg_port = st.text_input("Порт PostgreSQL", value="5433", key="pg_port")
        pg_db = st.text_input("База PostgreSQL", value="triplets", key="pg_db")
    with col2:
        pg_user = st.text_input("Пользователь PostgreSQL", value="triplets_user", key="pg_user")
        pg_password = st.text_input("Пароль PostgreSQL", value="triplets_pass", type="password", key="pg_password")
        neo_uri = st.text_input("URI Neo4j", value="bolt://localhost:7687", key="neo_uri")
        neo_user = st.text_input("Пользователь Neo4j", value="neo4j", key="neo_user")
        neo_password = st.text_input("Пароль Neo4j", value="neo4jpass", type="password", key="neo_password")

    if st.button("Сохранить в БД", type="primary", key="save_to_db_btn"):
        if payload is None:
            st.error("Нет данных для сохранения.")
            return

        pg_cfg = {
            "host": pg_host,
            "port": pg_port,
            "dbname": pg_db,
            "user": pg_user,
            "password": pg_password,
        }
        neo_cfg = {"uri": neo_uri, "user": neo_user, "password": neo_password}

        try:
            with st.spinner("Сохраняю в PostgreSQL и Neo4j..."):
                stats = save_triplets_to_databases(
                    triplets_payload=payload,
                    source_name=source_name,
                    extraction_stage=extraction_stage,
                    pg_cfg=pg_cfg,
                    neo_cfg=neo_cfg,
                )
            st.success(
                f"Готово. PostgreSQL: {stats['sql_loaded']} триплетов, "
                f"Neo4j: {stats['graph_loaded']} связей, document_id={stats['document_id']}."
            )
        except Exception as exc:
            st.error(f"Ошибка сохранения в БД: {exc}")
            st.code(traceback.format_exc(), language="text")

    st.caption("Кандидаты в алиасы/слияние по embedding-похожести вершин EntityConcept")
    a1, a2, a3 = st.columns(3)
    with a1:
        alias_min_support = st.number_input(
            "Мин. встречаемость вершины",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key="alias_min_support",
            help="Вершины с меньшим числом FrameOccurrence не попадают в сравнение. Сами кандидаты ранжируются по embedding similarity вершин.",
        )
    with a2:
        alias_min_conf = st.number_input(
            "Мин. cosine similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.82,
            step=0.01,
            key="alias_min_conf",
        )
    with a3:
        alias_limit = st.number_input(
            "Лимит кандидатов",
            min_value=10,
            max_value=10000,
            value=1000,
            step=10,
            key="alias_limit",
        )

    if st.button("Найти кандидатов по вершинам", key="mine_alias_candidates_btn"):
        pg_cfg = {
            "host": pg_host,
            "port": pg_port,
            "dbname": pg_db,
            "user": pg_user,
            "password": pg_password,
        }
        neo_cfg = {"uri": neo_uri, "user": neo_user, "password": neo_password}
        try:
            with st.spinner("Сравниваю вершины EntityConcept по эмбеддингам..."):
                rows = mine_alias_candidates(
                    pg_cfg=pg_cfg,
                    neo_cfg=neo_cfg,
                    min_support=int(alias_min_support),
                    min_confidence=float(alias_min_conf),
                    limit=int(alias_limit),
                )
            st.success(f"Сохранено кандидатов в алиасы: {len(rows)}")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("По текущим порогам кандидаты не найдены.")
        except Exception as exc:
            st.error(f"Ошибка поиска алиасов: {exc}")
            st.code(traceback.format_exc(), language="text")


def fetch_pg_documents(pg_cfg: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
    pg_conn = connect_pg_with_fallback(pg_cfg)

    query = """
    SELECT
        d.id AS document_id,
        d.source_name,
        d.created_at,
        COUNT(t.id) AS triplets_count
    FROM documents d
    LEFT JOIN triplets t ON t.document_id = d.id
    GROUP BY d.id, d.source_name, d.created_at
    ORDER BY d.id DESC
    LIMIT %s;
    """
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                cur.execute(query, (int(limit),))
                rows = cur.fetchall()
                result = []
                for row in rows:
                    result.append(
                        {
                            "document_id": row[0],
                            "source_name": row[1],
                            "created_at": str(row[2]),
                            "triplets_count": int(row[3] or 0),
                        }
                    )
                return result
    finally:
        if pg_conn is not None:
            pg_conn.close()


PG_BROWSE_TABLES: dict[str, dict[str, Any]] = {
    "documents": {
        "order_by": "id DESC",
        "search_cols": ["source_name", "payload_hash"],
        "description": "Сырые загруженные документы и их хеши.",
        "supports_document_filter": True,
    },
    "triplets": {
        "order_by": "id DESC",
        "search_cols": ["subject_text", "predicate_text", "object_text", "sentence_text", "extraction_stage"],
        "description": "Извлечённые триплеты с привязкой к document_id.",
        "supports_document_filter": True,
    },
    "triplet_frames": {
        "order_by": "id DESC",
        "search_cols": ["role", "frame_json::text"],
        "description": "Сырые JSON frame для subject/predicate/object.",
        "supports_document_filter": True,
    },
    "frame_instances": {
        "order_by": "id DESC",
        "search_cols": ["role", "root_text", "root_norm", "sentence_text"],
        "description": "Нормализованные корни frame по ролям.",
        "supports_document_filter": True,
    },
    "frame_nodes": {
        "order_by": "id DESC",
        "search_cols": ["node_text", "node_norm", "node_lemma", "edge_label", "path"],
        "description": "Все узлы frame-деревьев в табличной форме.",
        "supports_document_filter": True,
    },
    "concepts": {
        "order_by": "id DESC",
        "search_cols": ["canonical_name", "canonical_norm", "concept_type"],
        "description": "Канонические концепты для схлопывания графа.",
        "supports_document_filter": False,
    },
    "concept_aliases": {
        "order_by": "id DESC",
        "search_cols": ["alias_text", "alias_norm", "source", "status", "review_note"],
        "description": "Алиасы и статусы ревью для концептов.",
        "supports_document_filter": False,
    },
    "node_concept_links": {
        "order_by": "id DESC",
        "search_cols": ["link_type", "method"],
        "description": "Связи frame-узлов с концептами.",
        "supports_document_filter": False,
    },
}


def fetch_pg_table_counts(pg_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    pg_conn = connect_pg_with_fallback(pg_cfg)
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                rows: list[dict[str, Any]] = []
                for table_name in PG_BROWSE_TABLES:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                    row_count = int(cur.fetchone()[0] or 0)
                    rows.append({"table_name": table_name, "row_count": row_count})
                return rows
    finally:
        pg_conn.close()


def fetch_pg_table_rows(
    pg_cfg: dict[str, Any],
    table_name: str,
    limit: int = 100,
    contains: str = "",
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    if table_name not in PG_BROWSE_TABLES:
        raise ValueError(f"Unsupported table for browsing: {table_name}")

    config = PG_BROWSE_TABLES[table_name]
    search_cols = config["search_cols"]
    where_parts: list[str] = []
    params: list[Any] = []

    if contains.strip():
        pattern = f"%{contains.strip()}%"
        search_sql = " OR ".join(f"CAST({col} AS TEXT) ILIKE %s" for col in search_cols)
        where_parts.append(f"({search_sql})")
        params.extend([pattern] * len(search_cols))

    if document_id is not None:
        if table_name in {"documents"}:
            where_parts.append("id = %s")
            params.append(int(document_id))
        elif table_name in {"triplets"}:
            where_parts.append("document_id = %s")
            params.append(int(document_id))
        elif table_name in {"triplet_frames", "frame_instances"}:
            where_parts.append(
                "triplet_id IN (SELECT id FROM triplets WHERE document_id = %s)"
            )
            params.append(int(document_id))
        elif table_name == "frame_nodes":
            where_parts.append(
                "frame_instance_id IN (SELECT id FROM frame_instances WHERE triplet_id IN "
                "(SELECT id FROM triplets WHERE document_id = %s))"
            )
            params.append(int(document_id))
        else:
            # concepts / aliases / links are global; keep them visible without fake filtering.
            pass

    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    query = f"""
    SELECT *
    FROM {table_name}
    {where_sql}
    ORDER BY {config['order_by']}
    LIMIT %s;
    """
    params.append(int(limit))

    pg_conn = connect_pg_with_fallback(pg_cfg)
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                cur.execute(query, tuple(params))
                col_names = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                result: list[dict[str, Any]] = []
                for row in rows:
                    item: dict[str, Any] = {}
                    for idx, col_name in enumerate(col_names):
                        value = row[idx]
                        if isinstance(value, (dict, list)):
                            item[col_name] = json.dumps(value, ensure_ascii=False)
                        elif value is None:
                            item[col_name] = None
                        else:
                            item[col_name] = str(value) if col_name.endswith("_at") else value
                    result.append(item)
                return result
    finally:
        pg_conn.close()


def delete_document_by_id(
    document_id: int,
    pg_cfg: dict[str, Any],
    delete_neo4j: bool = False,
    neo_cfg: dict[str, Any] | None = None,
) -> dict[str, int]:
    pg_conn = connect_pg_with_fallback(pg_cfg)

    pg_deleted = 0
    neo_deleted = 0
    neo_deleted_triplets = 0
    neo_deleted_frames = 0
    neo_deleted_frame_nodes = 0
    neo_deleted_documents = 0
    neo_deleted_orphans = 0
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s;", (int(document_id),))
                pg_deleted = int(cur.rowcount or 0)
    finally:
        if pg_conn is not None:
            pg_conn.close()

    if delete_neo4j:
        if neo_cfg is None:
            raise ValueError("neo_cfg is required when delete_neo4j=True")
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Missing dependency: neo4j. Install requirements-graph.txt") from exc

        driver = GraphDatabase.driver(neo_cfg["uri"], auth=(neo_cfg["user"], neo_cfg["password"]))
        try:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (t:Triplet {document_id: $document_id})
                    RETURN collect(t.triplet_id) AS triplet_ids, count(t) AS triplet_count
                    """,
                    {"document_id": int(document_id)},
                ).single()
                triplet_ids = list(record["triplet_ids"] if record else [])
                neo_deleted_triplets = int(record["triplet_count"] if record else 0)

                rel_record = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE r.document_id = $document_id
                       OR ($triplet_ids <> [] AND r.triplet_id IN $triplet_ids)
                    WITH collect(r) AS rels
                    WITH rels, size(rels) AS deleted_count
                    FOREACH (r IN rels | DELETE r)
                    RETURN deleted_count
                    """,
                    {"document_id": int(document_id), "triplet_ids": triplet_ids},
                ).single()
                neo_deleted = int(rel_record["deleted_count"] if rel_record else 0)

                frame_node_record = session.run(
                    """
                    MATCH (n:FrameNode)
                    WHERE n.triplet_id IN $triplet_ids
                    WITH collect(n) AS nodes
                    WITH nodes, size(nodes) AS deleted_count
                    FOREACH (n IN nodes | DETACH DELETE n)
                    RETURN deleted_count
                    """,
                    {"triplet_ids": triplet_ids},
                ).single()
                neo_deleted_frame_nodes = int(frame_node_record["deleted_count"] if frame_node_record else 0)

                frame_record = session.run(
                    """
                    MATCH (occ:FrameOccurrence)
                    WHERE occ.document_id = $document_id
                       OR occ.triplet_id IN $triplet_ids
                    WITH collect(occ) AS occs
                    WITH occs, size(occs) AS deleted_count
                    FOREACH (occ IN occs | DETACH DELETE occ)
                    RETURN deleted_count
                    """,
                    {"document_id": int(document_id), "triplet_ids": triplet_ids},
                ).single()
                neo_deleted_frames = int(frame_record["deleted_count"] if frame_record else 0)

                triplet_record = session.run(
                    """
                    MATCH (t:Triplet)
                    WHERE t.document_id = $document_id
                    WITH collect(t) AS triplets
                    WITH triplets, size(triplets) AS deleted_count
                    FOREACH (t IN triplets | DETACH DELETE t)
                    RETURN deleted_count
                    """,
                    {"document_id": int(document_id)},
                ).single()
                neo_deleted_triplets = int(triplet_record["deleted_count"] if triplet_record else neo_deleted_triplets)

                doc_record = session.run(
                    """
                    MATCH (d:Document {document_id: $document_id})
                    WITH collect(d) AS docs
                    WITH docs, size(docs) AS deleted_count
                    FOREACH (d IN docs | DETACH DELETE d)
                    RETURN deleted_count
                    """,
                    {"document_id": int(document_id)},
                ).single()
                neo_deleted_documents = int(doc_record["deleted_count"] if doc_record else 0)

                # Derived analytics edges become stale after document deletion; they are rebuilt by the analytics step.
                session.run(
                    """
                    MATCH ()-[r:SYNONYM_CANDIDATE|CONTEXT_BRIDGE|CONTEXT_PARENT]->()
                    DELETE r
                    """,
                )
                session.run(
                    """
                    MATCH (:GraphRoot)-[r:HAS_COMPONENT_ROOT]->()
                    DELETE r
                    """
                )

                orphan_record = session.run(
                    """
                    CALL {
                      MATCH (n:Entity)
                      WHERE NOT (n)--()
                      WITH collect(n) AS nodes
                      FOREACH (n IN nodes | DELETE n)
                      RETURN size(nodes) AS deleted_count
                      UNION ALL
                      MATCH (c:EntityConcept)
                      WHERE NOT (c)<-[:SUBJECT|OBJECT]-(:Triplet)
                        AND NOT (c)<-[:OF_CONCEPT]-(:FrameOccurrence)
                        AND NOT (c)-[:RELATION_INSTANCE]-(:EntityConcept)
                        AND NOT (c)-[:APPROVED_ALIAS_OF|APPROVED_BROADER_THAN|APPROVED_CONTEXT_LINK|MANUAL_CONTEXT_LINK|MANUAL_COMPONENT_LINK]-(:EntityConcept)
                      WITH collect(c) AS nodes
                      FOREACH (c IN nodes | DETACH DELETE c)
                      RETURN size(nodes) AS deleted_count
                      UNION ALL
                      MATCH (p:RelationConcept)
                      WHERE NOT (p)<-[:PREDICATE]-(:Triplet)
                        AND NOT (p)<-[:OF_CONCEPT]-(:FrameOccurrence)
                      WITH collect(p) AS nodes
                      FOREACH (p IN nodes | DETACH DELETE p)
                      RETURN size(nodes) AS deleted_count
                    }
                    RETURN sum(deleted_count) AS deleted_count
                    """
                ).single()
                neo_deleted_orphans = int(orphan_record["deleted_count"] if orphan_record else 0)
        finally:
            driver.close()

    return {
        "pg_deleted_documents": pg_deleted,
        "neo_deleted_relations": neo_deleted,
        "neo_deleted_triplets": neo_deleted_triplets,
        "neo_deleted_frames": neo_deleted_frames,
        "neo_deleted_frame_nodes": neo_deleted_frame_nodes,
        "neo_deleted_documents": neo_deleted_documents,
        "neo_deleted_orphans": neo_deleted_orphans,
    }


def fetch_concept_aliases(
    pg_cfg: dict[str, Any],
    status_filter: str = "candidate",
    limit: int = 200,
) -> list[dict[str, Any]]:
    pg_conn = connect_pg_with_fallback(pg_cfg)
    query = """
    SELECT
        ca.id AS alias_id,
        c.id AS concept_id,
        c.canonical_name,
        c.canonical_norm,
        c.concept_type,
        ca.alias_text,
        ca.alias_norm,
        ca.source,
        ca.confidence,
        ca.status,
        ca.approved_at,
        ca.review_note
    FROM concept_aliases ca
    JOIN concepts c ON c.id = ca.concept_id
    WHERE (%s = 'all' OR ca.status = %s)
    ORDER BY
        CASE ca.status WHEN 'candidate' THEN 0 WHEN 'approved' THEN 1 ELSE 2 END,
        COALESCE(ca.confidence, 0) DESC,
        c.canonical_name,
        ca.alias_text
    LIMIT %s;
    """
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                cur.execute(query, (status_filter, status_filter, int(limit)))
                rows = cur.fetchall()
                result = []
                for row in rows:
                    result.append(
                        {
                            "alias_id": int(row[0]),
                            "concept_id": int(row[1]),
                            "canonical_name": row[2] or "",
                            "canonical_norm": row[3] or "",
                            "concept_type": row[4] or "",
                            "alias_text": row[5] or "",
                            "alias_norm": row[6] or "",
                            "source": row[7] or "",
                            "confidence": float(row[8]) if row[8] is not None else None,
                            "status": row[9] or "",
                            "approved_at": str(row[10]) if row[10] is not None else "",
                            "review_note": row[11] or "",
                        }
                    )
                return result
    finally:
        pg_conn.close()


def fetch_concepts(pg_cfg: dict[str, Any], limit: int = 500) -> list[dict[str, Any]]:
    pg_conn = connect_pg_with_fallback(pg_cfg)
    query = """
    SELECT
        c.id,
        c.canonical_name,
        c.canonical_norm,
        c.concept_type,
        COUNT(ca.id) AS alias_count,
        COUNT(*) FILTER (WHERE ca.status = 'approved') AS approved_aliases
    FROM concepts c
    LEFT JOIN concept_aliases ca ON ca.concept_id = c.id
    GROUP BY c.id, c.canonical_name, c.canonical_norm, c.concept_type
    ORDER BY c.canonical_name, c.id
    LIMIT %s;
    """
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                cur.execute(query, (int(limit),))
                rows = cur.fetchall()
                result = []
                for row in rows:
                    result.append(
                        {
                            "concept_id": int(row[0]),
                            "canonical_name": row[1] or "",
                            "canonical_norm": row[2] or "",
                            "concept_type": row[3] or "",
                            "alias_count": int(row[4] or 0),
                            "approved_aliases": int(row[5] or 0),
                        }
                    )
                return result
    finally:
        pg_conn.close()


def update_concept_alias_status(
    pg_cfg: dict[str, Any],
    alias_id: int,
    status: str,
    review_note: str = "",
) -> dict[str, Any]:
    pg_conn = connect_pg_with_fallback(pg_cfg)
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                cur.execute(
                    """
                    UPDATE concept_aliases
                    SET
                        status = %s,
                        approved_at = CASE
                            WHEN %s = 'approved' THEN COALESCE(approved_at, NOW())
                            ELSE NULL
                        END,
                        review_note = NULLIF(%s, ''),
                        source = CASE
                            WHEN %s = 'approved' THEN 'manual_review'
                            ELSE source
                        END
                    WHERE id = %s
                    RETURNING concept_id, alias_text, alias_norm;
                    """,
                    (status, status, review_note.strip(), status, int(alias_id)),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"Alias with id={alias_id} not found.")

                concept_id = int(row[0])
                if status == "approved":
                    cur.execute(
                        """
                        UPDATE concepts
                        SET concept_type = 'approved'
                        WHERE id = %s;
                        """,
                        (concept_id,),
                    )

                return {
                    "concept_id": concept_id,
                    "alias_text": row[1] or "",
                    "alias_norm": row[2] or "",
                    "status": status,
                }
    finally:
        pg_conn.close()


def merge_concepts(
    pg_cfg: dict[str, Any],
    source_concept_id: int,
    target_concept_id: int,
    review_note: str = "",
) -> dict[str, Any]:
    if int(source_concept_id) == int(target_concept_id):
        raise ValueError("Source and target concept must be different.")

    pg_conn = connect_pg_with_fallback(pg_cfg)
    try:
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_pg_schema(cur)
                cur.execute(
                    """
                    SELECT id, canonical_name, canonical_norm, COALESCE(concept_type, '')
                    FROM concepts
                    WHERE id IN (%s, %s)
                    ORDER BY id;
                    """,
                    (int(source_concept_id), int(target_concept_id)),
                )
                rows = cur.fetchall()
                concept_map = {
                    int(row[0]): {
                        "concept_id": int(row[0]),
                        "canonical_name": row[1] or "",
                        "canonical_norm": row[2] or "",
                        "concept_type": row[3] or "",
                    }
                    for row in rows
                }
                if int(source_concept_id) not in concept_map or int(target_concept_id) not in concept_map:
                    raise ValueError("One of the selected concepts was not found.")

                source = concept_map[int(source_concept_id)]
                target = concept_map[int(target_concept_id)]

                if source["canonical_norm"] != target["canonical_norm"]:
                    cur.execute(
                        """
                        INSERT INTO concept_aliases(
                            concept_id, alias_text, alias_norm, source, confidence, status, approved_at, review_note
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, NOW(), NULLIF(%s, ''))
                        ON CONFLICT (concept_id, alias_norm)
                        DO UPDATE SET
                            alias_text = EXCLUDED.alias_text,
                            source = 'merge_manual',
                            confidence = GREATEST(COALESCE(concept_aliases.confidence, 0), COALESCE(EXCLUDED.confidence, 0)),
                            status = 'approved',
                            approved_at = COALESCE(concept_aliases.approved_at, EXCLUDED.approved_at),
                            review_note = COALESCE(EXCLUDED.review_note, concept_aliases.review_note);
                        """,
                        (
                            target["concept_id"],
                            source["canonical_name"],
                            source["canonical_norm"],
                            "merge_manual",
                            1.0,
                            "approved",
                            review_note.strip(),
                        ),
                    )

                cur.execute(
                    """
                    INSERT INTO concept_aliases(
                        concept_id, alias_text, alias_norm, source, confidence, status, approved_at, review_note
                    )
                    SELECT
                        %s,
                        alias_text,
                        alias_norm,
                        source,
                        confidence,
                        CASE WHEN status = 'rejected' THEN 'candidate' ELSE status END,
                        approved_at,
                        review_note
                    FROM concept_aliases
                    WHERE concept_id = %s
                    ON CONFLICT (concept_id, alias_norm)
                    DO UPDATE SET
                        alias_text = EXCLUDED.alias_text,
                        confidence = GREATEST(COALESCE(concept_aliases.confidence, 0), COALESCE(EXCLUDED.confidence, 0)),
                        status = CASE
                            WHEN concept_aliases.status = 'approved' OR EXCLUDED.status = 'approved' THEN 'approved'
                            WHEN concept_aliases.status = 'candidate' THEN EXCLUDED.status
                            ELSE concept_aliases.status
                        END,
                        approved_at = COALESCE(concept_aliases.approved_at, EXCLUDED.approved_at),
                        review_note = COALESCE(concept_aliases.review_note, EXCLUDED.review_note);
                    """,
                    (target["concept_id"], source["concept_id"]),
                )

                cur.execute(
                    """
                    INSERT INTO node_concept_links(node_id, concept_id, link_type, score, method)
                    SELECT node_id, %s, link_type, score, method
                    FROM node_concept_links
                    WHERE concept_id = %s
                    ON CONFLICT (node_id, concept_id, link_type)
                    DO UPDATE SET
                        score = GREATEST(COALESCE(node_concept_links.score, 0), COALESCE(EXCLUDED.score, 0)),
                        method = COALESCE(node_concept_links.method, EXCLUDED.method);
                    """,
                    (target["concept_id"], source["concept_id"]),
                )
                links_moved = int(cur.rowcount or 0)

                cur.execute("DELETE FROM concept_aliases WHERE concept_id = %s;", (source["concept_id"],))
                deleted_aliases = int(cur.rowcount or 0)
                cur.execute("DELETE FROM node_concept_links WHERE concept_id = %s;", (source["concept_id"],))
                cur.execute("DELETE FROM concepts WHERE id = %s;", (source["concept_id"],))
                cur.execute(
                    """
                    UPDATE concepts
                    SET concept_type = CASE
                        WHEN COALESCE(concept_type, '') = '' THEN 'approved'
                        ELSE concept_type
                    END
                    WHERE id = %s;
                    """,
                    (target["concept_id"],),
                )

                return {
                    "source_concept_id": source["concept_id"],
                    "source_name": source["canonical_name"],
                    "target_concept_id": target["concept_id"],
                    "target_name": target["canonical_name"],
                    "deleted_aliases": deleted_aliases,
                    "links_moved": links_moved,
                }
    finally:
        pg_conn.close()


def render_pg_browser_section() -> None:
    st.subheader("Просмотр PostgreSQL")
    c1, c2, c3 = st.columns(3)
    with c1:
        host = st.text_input("Хост PostgreSQL (просмотр БД)", value=st.session_state.get("pg_host", "127.0.0.1"), key="pg_browser_host")
    with c2:
        port = st.text_input("Порт PostgreSQL (просмотр БД)", value=st.session_state.get("pg_port", "5433"), key="pg_browser_port")
    with c3:
        dbname = st.text_input("База PostgreSQL (просмотр БД)", value=st.session_state.get("pg_db", "triplets"), key="pg_browser_db")

    c4, c5, c6 = st.columns(3)
    with c4:
        user = st.text_input("Пользователь PostgreSQL (просмотр БД)", value=st.session_state.get("pg_user", "triplets_user"), key="pg_browser_user")
    with c5:
        password = st.text_input(
            "Пароль PostgreSQL (просмотр БД)",
            value=st.session_state.get("pg_password", "triplets_pass"),
            type="password",
            key="pg_browser_password",
        )
    with c6:
        limit = st.slider("Макс. строк из таблицы", min_value=10, max_value=1000, value=100, step=10, key="pg_browser_limit")

    pg_cfg = {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password,
    }

    if "pg_browser_counts" not in st.session_state:
        st.session_state["pg_browser_counts"] = []
    if "pg_browser_rows" not in st.session_state:
        st.session_state["pg_browser_rows"] = []
    if "pg_browser_selected_doc_label" not in st.session_state:
        st.session_state["pg_browser_selected_doc_label"] = "Без фильтра"

    c7, c8, c9, c10 = st.columns(4)
    with c7:
        table_name = st.selectbox("Таблица", options=list(PG_BROWSE_TABLES.keys()), key="pg_browser_table")
    with c8:
        contains = st.text_input("Поиск по тексту", value="", key="pg_browser_contains")
    with c9:
        limit_hint = ", ".join(PG_BROWSE_TABLES[table_name]["search_cols"][:4])
        st.caption(f"Поиск работает по полям: {limit_hint}")
    with c10:
        if st.button("Сбросить фильтры", key="pg_browser_reset_filters_btn"):
            st.session_state["pg_browser_contains"] = ""
            st.session_state["pg_browser_document_id"] = ""
            st.session_state["pg_browser_selected_doc_label"] = "Без фильтра"
            st.rerun()

    table_config = PG_BROWSE_TABLES[table_name]
    st.info(f"Таблица `{table_name}`: {table_config['description']}")

    doc_filter_text = st.session_state.get("pg_browser_document_id", "")
    if table_config.get("supports_document_filter"):
        doc_rows = st.session_state.get("pg_docs_rows", [])
        doc_options = ["Без фильтра"] + [
            f"{row['document_id']} | {row['source_name']} | triplets={row['triplets_count']}"
            for row in doc_rows
        ]
        selected_doc_label = st.selectbox(
            "Документ",
            options=doc_options,
            key="pg_browser_selected_doc_label",
        )
        if selected_doc_label != "Без фильтра":
            selected_doc_id = selected_doc_label.split("|", 1)[0].strip()
            if st.session_state.get("pg_browser_document_id") != selected_doc_id:
                st.session_state["pg_browser_document_id"] = selected_doc_id
                doc_filter_text = selected_doc_id
        doc_filter_text = st.text_input("Фильтр по document_id", value=st.session_state.get("pg_browser_document_id", ""), key="pg_browser_document_id")
    else:
        st.caption("Для этой таблицы фильтр по document_id не применяется.")

    document_id: int | None = None
    if doc_filter_text.strip():
        try:
            document_id = int(doc_filter_text.strip())
        except ValueError:
            st.warning("Фильтр по document_id должен быть целым числом.")

    summary_tab, data_tab = st.tabs(["Сводка таблиц", "Данные таблицы"])

    with summary_tab:
        if st.button("Загрузить сводку таблиц", key="pg_browser_load_counts_btn"):
            try:
                with st.spinner("Считаю строки по таблицам PostgreSQL..."):
                    st.session_state["pg_browser_counts"] = fetch_pg_table_counts(pg_cfg)
                st.success("Сводка по таблицам загружена.")
            except Exception as exc:
                st.error(f"Ошибка загрузки сводки PostgreSQL: {exc}")
                st.code(traceback.format_exc(), language="text")
        if st.session_state["pg_browser_counts"]:
            counts_df = pd.DataFrame(st.session_state["pg_browser_counts"])
            st.dataframe(counts_df, use_container_width=True)
            non_empty = counts_df[counts_df["row_count"] > 0]
            if not non_empty.empty:
                st.caption(
                    "Непустые таблицы: " + ", ".join(
                        f"{row.table_name}={row.row_count}" for row in non_empty.itertuples()
                    )
                )

    with data_tab:
        if st.button("Загрузить данные таблицы", key="pg_browser_load_rows_btn"):
            try:
                with st.spinner(f"Загружаю таблицу {table_name}..."):
                    st.session_state["pg_browser_rows"] = fetch_pg_table_rows(
                        pg_cfg=pg_cfg,
                        table_name=table_name,
                        limit=limit,
                        contains=contains,
                        document_id=document_id,
                    )
                st.success(
                    f"Загружено строк из {table_name}: {len(st.session_state['pg_browser_rows'])}"
                )
            except Exception as exc:
                st.error(f"Ошибка загрузки таблицы PostgreSQL: {exc}")
                st.code(traceback.format_exc(), language="text")

        if st.session_state["pg_browser_rows"]:
            st.caption(f"Предпросмотр таблицы: {table_name}")
            st.dataframe(pd.DataFrame(st.session_state["pg_browser_rows"]), use_container_width=True)
        else:
            st.info("Загрузите данные таблицы, чтобы посмотреть содержимое PostgreSQL.")


def render_documents_list_section() -> None:
    st.subheader("Документы в PostgreSQL")
    c1, c2, c3 = st.columns(3)
    with c1:
        host = st.text_input("Хост PostgreSQL (документы)", value=st.session_state.get("pg_host", "127.0.0.1"), key="pg_docs_host")
    with c2:
        port = st.text_input("Порт PostgreSQL (документы)", value=st.session_state.get("pg_port", "5433"), key="pg_docs_port")
    with c3:
        dbname = st.text_input("База PostgreSQL (документы)", value=st.session_state.get("pg_db", "triplets"), key="pg_docs_db")

    c4, c5, c6 = st.columns(3)
    with c4:
        user = st.text_input("Пользователь PostgreSQL (документы)", value=st.session_state.get("pg_user", "triplets_user"), key="pg_docs_user")
    with c5:
        password = st.text_input(
            "Пароль PostgreSQL (документы)",
            value=st.session_state.get("pg_password", "triplets_pass"),
            type="password",
            key="pg_docs_password",
        )
    with c6:
        limit = st.slider("Макс. документов", min_value=10, max_value=500, value=100, step=10, key="pg_docs_limit")

    pg_cfg = {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password,
    }

    if "pg_docs_rows" not in st.session_state:
        st.session_state["pg_docs_rows"] = []

    if st.button("Загрузить список документов", key="pg_docs_load_btn"):
        try:
            with st.spinner("Загружаю документы из PostgreSQL..."):
                rows = fetch_pg_documents(pg_cfg, limit=limit)
            st.session_state["pg_docs_rows"] = rows
            if not rows:
                st.info("В PostgreSQL документы не найдены.")
            else:
                st.success(f"Загружено документов: {len(rows)}")
        except Exception as exc:
            st.error(f"Ошибка просмотра документов PostgreSQL: {exc}")
            st.code(traceback.format_exc(), language="text")

    rows = st.session_state["pg_docs_rows"]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        labels = [
            f"{row['document_id']} | {row['source_name']} | triplets={row['triplets_count']}"
            for row in rows
        ]
        selected_label = st.selectbox("Документ для удаления", options=labels, key="pg_docs_delete_select")
        selected_doc_id = int(selected_label.split("|", 1)[0].strip())

        c7, c8, c9 = st.columns(3)
        with c7:
            del_neo = st.checkbox("Удалить также из Neo4j", value=True, key="pg_docs_delete_neo")
        with c8:
            neo_uri = st.text_input("URI Neo4j (удаление)", value=st.session_state.get("neo_uri", "bolt://localhost:7687"), key="pg_docs_neo_uri")
        with c9:
            neo_user = st.text_input("Пользователь Neo4j (удаление)", value=st.session_state.get("neo_user", "neo4j"), key="pg_docs_neo_user")
        neo_password = st.text_input(
            "Пароль Neo4j (удаление)",
            value=st.session_state.get("neo_password", "neo4jpass"),
            type="password",
            key="pg_docs_neo_password",
        )

        confirm = st.checkbox(
            f"Подтвердить удаление document_id={selected_doc_id}",
            value=False,
            key="pg_docs_delete_confirm",
        )
        if st.button("Удалить документ", type="secondary", key="pg_docs_delete_btn"):
            if not confirm:
                st.warning("Перед удалением включите подтверждение.")
            else:
                try:
                    with st.spinner(f"Удаляю document_id={selected_doc_id}..."):
                        stats = delete_document_by_id(
                            document_id=selected_doc_id,
                            pg_cfg=pg_cfg,
                            delete_neo4j=del_neo,
                            neo_cfg={"uri": neo_uri, "user": neo_user, "password": neo_password},
                        )
                    st.success(
                        f"Удалено в PostgreSQL: {stats['pg_deleted_documents']} документ(ов), "
                        f"в Neo4j: связей={stats['neo_deleted_relations']}, "
                        f"триплетов={stats.get('neo_deleted_triplets', 0)}, "
                        f"frame-occurrence={stats.get('neo_deleted_frames', 0)}, "
                        f"frame-узлов={stats.get('neo_deleted_frame_nodes', 0)}, "
                        f"сиротских вершин={stats.get('neo_deleted_orphans', 0)}."
                    )
                    # Обновляем список после удаления.
                    st.session_state["pg_docs_rows"] = fetch_pg_documents(pg_cfg, limit=limit)
                except Exception as exc:
                    st.error(f"Ошибка удаления: {exc}")
                    st.code(traceback.format_exc(), language="text")


# def render_concept_aliases_section() -> None:
#     st.subheader("Алиасы концептов")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         host = st.text_input(
#             "Хост PostgreSQL (концепты)",
#             value=st.session_state.get("pg_host", "127.0.0.1"),
#             key="pg_concepts_host",
#         )
#     with c2:
#         port = st.text_input(
#             "Порт PostgreSQL (концепты)",
#             value=st.session_state.get("pg_port", "5433"),
#             key="pg_concepts_port",
#         )
#     with c3:
#         dbname = st.text_input(
#             "База PostgreSQL (концепты)",
#             value=st.session_state.get("pg_db", "triplets"),
#             key="pg_concepts_db",
#         )

#     c4, c5, c6 = st.columns(3)
#     with c4:
#         user = st.text_input(
#             "Пользователь PostgreSQL (концепты)",
#             value=st.session_state.get("pg_user", "triplets_user"),
#             key="pg_concepts_user",
#         )
#     with c5:
#         password = st.text_input(
#             "Пароль PostgreSQL (концепты)",
#             value=st.session_state.get("pg_password", "triplets_pass"),
#             type="password",
#             key="pg_concepts_password",
#         )
#     with c6:
#         alias_limit = st.slider(
#             "Макс. алиасов",
#             min_value=20,
#             max_value=1000,
#             value=200,
#             step=20,
#             key="pg_concepts_limit",
#         )

#     c7, c8 = st.columns(2)
#     with c7:
#         status_filter = st.selectbox(
#             "Статус алиаса",
#             options=["candidate", "approved", "rejected", "all"],
#             index=0,
#             key="concept_alias_status_filter",
#         )
#     with c8:
#         concepts_limit = st.slider(
#             "Макс. концептов",
#             min_value=20,
#             max_value=2000,
#             value=500,
#             step=20,
#             key="concepts_limit",
#         )

#     pg_cfg = {
#         "host": host,
#         "port": port,
#         "dbname": dbname,
#         "user": user,
#         "password": password,
#     }

#     if "concept_alias_rows" not in st.session_state:
#         st.session_state["concept_alias_rows"] = []
#     if "concept_rows" not in st.session_state:
#         st.session_state["concept_rows"] = []

#     def refresh_concept_state() -> None:
#         st.session_state["concept_alias_rows"] = fetch_concept_aliases(
#             pg_cfg,
#             status_filter=status_filter,
#             limit=alias_limit,
#         )
#         st.session_state["concept_rows"] = fetch_concepts(pg_cfg, limit=concepts_limit)

#     if st.button("Загрузить алиасы концептов", key="load_concept_aliases_btn"):
#         try:
#             with st.spinner("Загружаю концепты и алиасы..."):
#                 refresh_concept_state()
#             st.success(
#                 f"Загружено алиасов: {len(st.session_state['concept_alias_rows'])}, "
#                 f"концептов: {len(st.session_state['concept_rows'])}"
#             )
#         except Exception as exc:
#             st.error(f"Ошибка загрузки алиасов концептов: {exc}")
#             st.code(traceback.format_exc(), language="text")

#     alias_rows = st.session_state["concept_alias_rows"]
#     concept_rows = st.session_state["concept_rows"]

#     if alias_rows:
#         st.dataframe(pd.DataFrame(alias_rows), use_container_width=True)

#         alias_labels = [
#             (
#                 f"{row['alias_id']} | concept={row['concept_id']} | "
#                 f"{row['canonical_name']} <- {row['alias_text']} | "
#                 f"status={row['status']} | conf={row['confidence'] if row['confidence'] is not None else 'n/a'}"
#             )
#             for row in alias_rows
#         ]
#         selected_alias_label = st.selectbox(
#             "Алиас для проверки",
#             options=alias_labels,
#             key="concept_alias_select",
#         )
#         selected_alias_id = int(selected_alias_label.split("|", 1)[0].strip())
#         review_note = st.text_input("Комментарий ревью", value="", key="concept_alias_review_note")

#         b1, b2, b3 = st.columns(3)
#         with b1:
#             approve_clicked = st.button("Одобрить алиас", key="approve_alias_btn")
#         with b2:
#             reject_clicked = st.button("Отклонить алиас", key="reject_alias_btn")
#         with b3:
#             reset_clicked = st.button("Вернуть в кандидаты", key="reset_alias_btn")

#         try:
#             if approve_clicked:
#                 with st.spinner("Одобряю алиас..."):
#                     result = update_concept_alias_status(
#                         pg_cfg=pg_cfg,
#                         alias_id=selected_alias_id,
#                         status="approved",
#                         review_note=review_note,
#                     )
#                     refresh_concept_state()
#                 st.success(f"Алиас одобрен: {result['alias_text']} -> concept_id={result['concept_id']}")
#             if reject_clicked:
#                 with st.spinner("Отклоняю алиас..."):
#                     result = update_concept_alias_status(
#                         pg_cfg=pg_cfg,
#                         alias_id=selected_alias_id,
#                         status="rejected",
#                         review_note=review_note,
#                     )
#                     refresh_concept_state()
#                 st.success(f"Алиас отклонён: {result['alias_text']}")
#             if reset_clicked:
#                 with st.spinner("Сбрасываю статус алиаса..."):
#                     result = update_concept_alias_status(
#                         pg_cfg=pg_cfg,
#                         alias_id=selected_alias_id,
#                         status="candidate",
#                         review_note=review_note,
#                     )
#                     refresh_concept_state()
#                 st.success(f"Алиас возвращён в кандидаты: {result['alias_text']}")
#         except Exception as exc:
#             st.error(f"Ошибка ревью алиаса: {exc}")
#             st.code(traceback.format_exc(), language="text")
#     else:
#         st.info("Загрузите алиасы концептов, чтобы проверить кандидатные соответствия.")

#     if concept_rows:
#         st.caption("Слияние концептов")
#         st.dataframe(pd.DataFrame(concept_rows), use_container_width=True)
#         concept_labels = [
#             (
#                 f"{row['concept_id']} | {row['canonical_name']} | "
#                 f"aliases={row['alias_count']} | approved={row['approved_aliases']}"
#             )
#             for row in concept_rows
#         ]
#         source_label = st.selectbox("Исходный концепт", options=concept_labels, key="merge_source_concept")
#         target_label = st.selectbox("Целевой концепт", options=concept_labels, key="merge_target_concept")
#         merge_note = st.text_input("Комментарий к слиянию", value="", key="merge_concept_note")
#         merge_confirm = st.checkbox("Подтвердить слияние концептов", value=False, key="merge_concept_confirm")
#         if st.button("Слить концепты", key="merge_concepts_btn"):
#             if not merge_confirm:
#                 st.warning("Включите подтверждение перед слиянием концептов.")
#             else:
#                 source_concept_id = int(source_label.split("|", 1)[0].strip())
#                 target_concept_id = int(target_label.split("|", 1)[0].strip())
#                 try:
#                     with st.spinner("Сливаю концепты..."):
#                         stats = merge_concepts(
#                             pg_cfg=pg_cfg,
#                             source_concept_id=source_concept_id,
#                             target_concept_id=target_concept_id,
#                             review_note=merge_note,
#                         )
#                         refresh_concept_state()
#                     st.success(
#                         f"Концепт '{stats['source_name']}' слит в '{stats['target_name']}'. "
#                         f"Удалено алиасов исходного концепта: {stats['deleted_aliases']}, "
#                         f"перенесено ссылок на узлы: {stats['links_moved']}."
#                     )
#                 except Exception as exc:
#                     st.error(f"Ошибка слияния концептов: {exc}")
#                     st.code(traceback.format_exc(), language="text")
#     else:
#         st.info("Загрузите концепты, чтобы проверить их или слить.")


def format_concept_option(row: dict[str, Any]) -> str:
    return (
        f"{row['name']} | norm={row['norm']} | "
        f"in={row['in_degree']} out={row['out_degree']} frame={row['frame_count']}"
    )


def render_context_merge_tools(uri: str, user: str, password: str, analytics_result: dict[str, Any] | None) -> None:
    st.subheader("Ревью и ручное связывание концептов")
    st.caption(
        "Одобренные связи не удаляют вершины: они фиксируют решение ревью и сохраняют исходные frame."
    )

    if analytics_result:
        legacy_root_rows = analytics_result.get("legacy_root_rows") or []
        legacy_root_groups: list[dict[str, Any]] = []
        groups_by_root: dict[tuple[str, str], dict[str, Any]] = {}
        for row in legacy_root_rows:
            root_name = row.get("root_name") or row.get("root_norm") or ""
            root_norm = row.get("root_norm") or safe_entity_name(root_name)
            key = (root_norm, root_name)
            group = groups_by_root.setdefault(
                key,
                {
                    "root_norm": root_norm,
                    "root_name": root_name,
                    "cluster_score": row.get("cluster_score", 0.0),
                    "cluster_size": row.get("cluster_size", 0),
                    "shared_examples": row.get("shared_examples", ""),
                    "children": [],
                },
            )
            group["children"].append(
                {
                    "child_norm": row.get("child_norm", ""),
                    "child_name": row.get("child_name") or row.get("child_norm", ""),
                    "cluster_score": row.get("cluster_score", 0.0),
                    "shared_examples": row.get("shared_examples", ""),
                }
            )
            group["cluster_score"] = max(float(group.get("cluster_score") or 0.0), float(row.get("cluster_score") or 0.0))
            group["cluster_size"] = max(int(group.get("cluster_size") or 0), int(row.get("cluster_size") or 0))
        legacy_root_groups = sorted(
            groups_by_root.values(),
            key=lambda item: (-float(item.get("cluster_score") or 0.0), -len(item.get("children") or []), item.get("root_name", "")),
        )
        candidate_groups = {
            "Синонимия по контексту": legacy_root_groups,
            "Мост между компонентами": analytics_result.get("bridge_rows") or [],
        }
        candidate_kind = st.selectbox(
            "Тип кандидата",
            options=list(candidate_groups.keys()),
            key="context_merge_candidate_kind",
        )
        candidates = candidate_groups[candidate_kind]
        if candidates:
            if candidate_kind == "Синонимия по контексту":
                labels = [
                    (
                        f"{idx + 1}. {row['root_name']} -> {len(row.get('children') or [])} сущн. | "
                        f"score={row.get('cluster_score', '')}"
                    )
                    for idx, row in enumerate(candidates)
                ]
                selected_label = st.selectbox(
                    "Кластер из общих legacy-сущностей",
                    options=labels,
                    key="context_merge_legacy_root_select",
                )
                selected_idx = labels.index(selected_label)
                selected_row = candidates[selected_idx]
                child_rows = selected_row.get("children") or []
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "root_name": selected_row.get("root_name", ""),
                                "child_name": child.get("child_name", ""),
                                "cluster_score": child.get("cluster_score", selected_row.get("cluster_score", "")),
                                "shared_examples": child.get("shared_examples", selected_row.get("shared_examples", "")),
                            }
                            for child in child_rows
                        ]
                    ),
                    use_container_width=True,
                )
                root_name = st.text_input(
                    "Название новой компоненты",
                    value=selected_row.get("root_name", ""),
                    key=f"context_merge_root_name_{selected_idx}",
                    help="Это изначальное название из root_name. Его можно поправить перед записью.",
                )
                relation_text = st.text_input(
                    "Подпись связей от компоненты",
                    value="синонимия по контексту",
                    key="context_merge_legacy_relation_text",
                )
                review_note = st.text_input("Комментарий ревью", value="", key="context_merge_legacy_note")
                if st.button("Создать компоненту и связать сущности", key="approve_context_legacy_root_btn"):
                    if not root_name.strip():
                        st.warning("Заполните название новой компоненты.")
                    elif not child_rows:
                        st.warning("В выбранном кластере нет дочерних сущностей.")
                    else:
                        try:
                            created = create_context_synonym_component(
                                uri=uri,
                                user=user,
                                password=password,
                                root_name=root_name,
                                child_rows=child_rows,
                                relation_text=relation_text.strip() or "синонимия по контексту",
                                note=review_note.strip(),
                            )
                            st.success(
                                f"Компонента '{created['root_name']}' записана. "
                                f"Связей создано/обновлено: {created['linked_count']}."
                            )
                        except Exception as exc:
                            st.error(f"Не удалось записать компоненту синонимии: {exc}")
                            st.code(traceback.format_exc(), language="text")
            else:
                labels = [
                    (
                        f"{idx + 1}. {row['left_name']} -> {row['right_name']} | "
                        f"score={row.get('combined_score', row.get('bridge_score', ''))}"
                    )
                    for idx, row in enumerate(candidates)
                ]
                selected_label = st.selectbox(
                    "Кандидат из последней аналитики",
                    options=labels,
                    key="context_merge_candidate_select",
                )
                selected_idx = labels.index(selected_label)
                selected_row = candidates[selected_idx]
                st.dataframe(pd.DataFrame([selected_row]), use_container_width=True)
                review_note = st.text_input("Комментарий ревью", value="", key="context_merge_candidate_note")
                default_relation = "контекстная связь"
                relation_text = st.text_input(
                    "Подпись одобренной связи",
                    value=default_relation,
                    key="context_merge_candidate_relation_text",
                )
                if st.button("Одобрить выбранную связь", key="approve_context_candidate_btn"):
                    try:
                        create_approved_concept_relation(
                            uri=uri,
                            user=user,
                            password=password,
                            left_norm=selected_row["left_norm"],
                            right_norm=selected_row["right_norm"],
                            relation_kind="context",
                            relation_text=relation_text.strip() or default_relation,
                            note=review_note.strip(),
                        )
                        st.success("Одобренная связь записана в Neo4j.")
                    except Exception as exc:
                        st.error(f"Не удалось записать одобренную связь: {exc}")
                        st.code(traceback.format_exc(), language="text")
        else:
            st.info("В последней аналитике нет кандидатов выбранного типа.")
    else:
        st.info("Запустите аналитику, чтобы одобрять найденные кандидаты.")

    with st.expander("Добавить собственную вершину или ручную связь между компонентами", expanded=False):
        manual_filter = st.text_input("Фильтр концептов для ручной связи", value="", key="manual_context_filter")
        manual_limit = st.number_input(
            "Сколько концептов загрузить",
            min_value=20,
            max_value=1000,
            value=200,
            step=20,
            key="manual_context_limit",
        )
        if st.button("Загрузить концепты для ручной связи", key="load_manual_context_concepts_btn"):
            try:
                st.session_state["manual_context_concepts"] = fetch_neo4j_entity_concept_options(
                    uri=uri,
                    user=user,
                    password=password,
                    limit=int(manual_limit),
                    contains=manual_filter,
                )
            except Exception as exc:
                st.error(f"Не удалось загрузить концепты: {exc}")
                st.code(traceback.format_exc(), language="text")

        concepts = st.session_state.get("manual_context_concepts", [])
        if concepts:
            labels = [format_concept_option(row) for row in concepts]
            c1, c2 = st.columns(2)
            with c1:
                left_label = st.selectbox("Левая вершина", options=labels, key="manual_context_left")
            with c2:
                right_label = st.selectbox("Правая вершина", options=labels, key="manual_context_right")
            left_row = concepts[labels.index(left_label)]
            right_row = concepts[labels.index(right_label)]

            relation_text = st.text_input(
                "Текст связи",
                value="связано по контексту",
                key="manual_context_relation_text",
            )
            use_connector = st.checkbox(
                "Добавить собственную вершину-связку между компонентами",
                value=True,
                key="manual_context_use_connector",
            )
            auto_connector_name = st.checkbox(
                "Выделить автоматически",
                value=True,
                key="manual_context_auto_connector",
                help="Имя вершины-связки будет собрано из текста связи, как служебный родитель/коннектор.",
            )
            connector_name = ""
            if use_connector and not auto_connector_name:
                connector_name = st.text_input(
                    "Название собственной вершины",
                    value="",
                    key="manual_context_connector_name",
                )
            manual_note = st.text_input("Комментарий", value="", key="manual_context_note")

            if st.button("Записать ручную связь", key="save_manual_context_link_btn"):
                if left_row["norm"] == right_row["norm"]:
                    st.warning("Выберите две разные вершины.")
                elif not relation_text.strip():
                    st.warning("Заполните текст связи.")
                else:
                    try:
                        if use_connector:
                            connector = create_manual_component_connector(
                                uri=uri,
                                user=user,
                                password=password,
                                left_norm=left_row["norm"],
                                right_norm=right_row["norm"],
                                connector_name=connector_name,
                                relation_text=relation_text.strip(),
                                auto_name=auto_connector_name,
                                note=manual_note.strip(),
                            )
                            st.success(f"Добавлена вершина-связка: {connector['connector_name']}.")
                        else:
                            create_approved_concept_relation(
                                uri=uri,
                                user=user,
                                password=password,
                                left_norm=left_row["norm"],
                                right_norm=right_row["norm"],
                                relation_kind="manual",
                                relation_text=relation_text.strip(),
                                note=manual_note.strip(),
                            )
                            st.success("Ручная связь записана в Neo4j.")
                    except Exception as exc:
                        st.error(f"Не удалось записать ручную связь: {exc}")
                        st.code(traceback.format_exc(), language="text")
        else:
            st.caption("Загрузите список концептов, чтобы выбрать вершины.")


def render_graph_analytics_section() -> None:
    st.subheader("Аналитика графа и синонимов")
    c1, c2, c3 = st.columns(3)
    with c1:
        uri = st.text_input(
            "URI Neo4j (аналитика)",
            value=st.session_state.get("neo_uri", "bolt://localhost:7687"),
            key="analytics_neo_uri",
        )
    with c2:
        user = st.text_input(
            "Пользователь Neo4j (аналитика)",
            value=st.session_state.get("neo_user", "neo4j"),
            key="analytics_neo_user",
        )
    with c3:
        password = st.text_input(
            "Пароль Neo4j (аналитика)",
            value=st.session_state.get("neo_password", "neo4jpass"),
            type="password",
            key="analytics_neo_password",
        )

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        min_combined = st.number_input(
            "Мин. итоговый score",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            key="analytics_min_combined",
        )
    with c5:
        min_structural = st.number_input(
            "Мин. структурный score",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
            key="analytics_min_structural",
        )
    with c6:
        synonym_limit = st.number_input(
            "Лимит синонимов",
            min_value=10,
            max_value=5000,
            value=200,
            step=10,
            key="analytics_synonym_limit",
        )
    with c7:
        sync_syn_edges = st.checkbox(
            "Записать SYNONYM_CANDIDATE в Neo4j",
            value=True,
            key="analytics_sync_syn_edges",
        )

    c8, c9, c10 = st.columns(3)
    with c8:
        min_bridge = st.number_input(
            "Мин. score для мостов",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            key="analytics_min_bridge",
        )
    with c9:
        bridge_limit = st.number_input(
            "Лимит мостов",
            min_value=10,
            max_value=5000,
            value=200,
            step=10,
            key="analytics_bridge_limit",
        )
    with c10:
        sync_context_bridges = st.checkbox(
            "Записать CONTEXT_BRIDGE в Neo4j",
            value=True,
            key="analytics_sync_context_bridges",
        )

    c11, c12, c13 = st.columns(3)
    with c11:
        min_legacy_root = st.number_input(
            "Мин. score для общих legacy-сущностей",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            key="analytics_min_legacy_root",
        )
    with c12:
        legacy_root_limit = st.number_input(
            "Лимит общих legacy-сущностей",
            min_value=10,
            max_value=5000,
            value=200,
            step=10,
            key="analytics_legacy_root_limit",
        )
    with c13:
        sync_legacy_roots = st.checkbox(
            "Записать CONTEXT_PARENT в Neo4j",
            value=True,
            key="analytics_sync_legacy_roots",
        )

    sync_root = st.checkbox(
        "Выделить глобального родителя графа в Neo4j",
        value=True,
        key="analytics_sync_root",
    )
    include_frame_context = st.checkbox(
        "Учитывать frame-контекст при поиске синонимов и мостов",
        value=True,
        key="analytics_include_frame_context",
    )
    include_table_tests = st.checkbox(
        "Проверить граф по CSV-таблицам",
        value=True,
        key="analytics_include_table_tests",
    )
    table_tests_dir = st.text_input(
        "Папка с тестовыми таблицами",
        value=st.session_state.get("analytics_table_tests_dir", "tables_for_test"),
        key="analytics_table_tests_dir",
        disabled=not include_table_tests,
    )

    st.caption("Слой эмбеддингов")
    emb1, emb2 = st.columns(2)
    with emb1:
        embedding_mode_label = st.selectbox(
            "Режим эмбеддингов",
            options=[
                "Авто: encoder -> fallback",
                "Только encoder",
                "Только fallback",
            ],
            index=0,
            key="analytics_embedding_mode",
        )
    with emb2:
        encoder_model_name = st.text_input(
            "Encoder-модель или локальный путь",
            value=st.session_state.get("analytics_encoder_model", "intfloat/multilingual-e5-small"),
            key="analytics_encoder_model",
        )

    emb3, emb4, emb5 = st.columns(3)
    with emb3:
        encoder_local_only = st.checkbox(
            "Только локальные файлы модели",
            value=False,
            key="analytics_encoder_local_only",
        )
    with emb4:
        encoder_batch_size = st.number_input(
            "Размер batch для encoder",
            min_value=1,
            max_value=128,
            value=16,
            step=1,
            key="analytics_encoder_batch_size",
        )
    with emb5:
        encoder_max_length = st.number_input(
            "Макс. длина текста для encoder",
            min_value=32,
            max_value=2048,
            value=256,
            step=32,
            key="analytics_encoder_max_length",
        )

    embedding_mode_map = {
        "Авто: encoder -> fallback": "auto",
        "Только encoder": "encoder",
        "Только fallback": "fallback",
    }
    embedding_mode = embedding_mode_map[embedding_mode_label]
    encoder_status = get_encoder_runtime_status()

    if encoder_status["available"]:
        st.caption("Encoder runtime: доступен.")
    else:
        st.warning(
            f"Encoder runtime сейчас недоступен. {encoder_status['reason']}. "
            "Используйте режим 'Авто: encoder -> fallback' или 'Только fallback', "
            "либо установите недостающие пакеты."
        )

    if "graph_analytics_result" not in st.session_state:
        st.session_state["graph_analytics_result"] = None

    if st.button("Запустить аналитику графа", key="run_graph_analytics_btn"):
        if embedding_mode == "encoder" and not encoder_status["available"]:
            st.error(
                "Режим 'Только encoder' недоступен: не установлены зависимости для encoder-эмбеддингов. "
                "Переключите режим на 'Авто: encoder -> fallback' или 'Только fallback'."
            )
            return
        try:
            with st.spinner("Считаю аналитику графа и ранжирование синонимов..."):
                st.session_state["graph_analytics_result"] = analyze_graph(
                    neo_cfg={"uri": uri, "user": user, "password": password},
                    min_combined_score=float(min_combined),
                    min_structural_score=float(min_structural),
                    synonym_limit=int(synonym_limit),
                    min_bridge_score=float(min_bridge),
                    bridge_limit=int(bridge_limit),
                    min_legacy_root_score=float(min_legacy_root),
                    legacy_root_limit=int(legacy_root_limit),
                    sync_synonym_edges=bool(sync_syn_edges),
                    sync_context_bridges=bool(sync_context_bridges),
                    sync_legacy_roots=bool(sync_legacy_roots),
                    sync_graph_root_node=bool(sync_root),
                    embedding_backend=embedding_mode,
                    encoder_model_name=encoder_model_name.strip() or "intfloat/multilingual-e5-small",
                    encoder_local_files_only=bool(encoder_local_only),
                    encoder_batch_size=int(encoder_batch_size),
                    encoder_max_length=int(encoder_max_length),
                    include_frame_context=bool(include_frame_context),
                    table_tests_dir=os.path.abspath(table_tests_dir.strip() or "tables_for_test"),
                    include_table_tests=bool(include_table_tests),
                )
            st.success("Аналитика графа завершена.")
        except Exception as exc:
            error_text = str(exc)
            if "transformers package" in error_text or "encoder model" in error_text:
                st.error(
                    "Ошибка аналитики графа: encoder-слой сейчас недоступен. "
                    "Переключите режим эмбеддингов на 'Авто: encoder -> fallback' "
                    "или 'Только fallback', либо установите пакет transformers и модель."
                )
            else:
                st.error(f"Ошибка аналитики графа: {exc}")
            st.code(traceback.format_exc(), language="text")

    result = st.session_state["graph_analytics_result"]
    if not result:
        st.info("Запустите аналитику, чтобы увидеть степени вершин, циклы, корневые вершины и синонимические связи.")
        render_context_merge_tools(uri=uri, user=user, password=password, analytics_result=None)
        return

    embedding_info = result.get("embedding_info") or {}
    backend_label_map = {
        "transformers_encoder": "encoder-модель через transformers",
        "hashed_char_ngrams_fallback": "резервный hashed char n-gram",
        "none": "не использовался",
    }
    backend_label = backend_label_map.get(embedding_info.get("backend"), embedding_info.get("backend", "неизвестно"))
    model_name = embedding_info.get("model_name") or "n/a"
    st.info(f"Слой эмбеддингов: {backend_label}. Модель: {model_name}.")
    if embedding_info.get("warning"):
        st.warning(embedding_info["warning"])

    metrics = result["metrics"]
    st.caption("Сводные метрики")
    metrics_rows = [
        {
            "nodes_count": metrics["nodes_count"],
            "edges_count": metrics["edges_count"],
            "components_count": metrics["components_count"],
            "is_cyclic": metrics["is_cyclic"],
            "cycles_found": metrics["cycles_found"],
            "max_in_degree": metrics["max_in_degree"],
            "max_out_degree": metrics["max_out_degree"],
            "avg_in_degree": metrics["avg_in_degree"],
            "avg_out_degree": metrics["avg_out_degree"],
            "bridge_edges_count": metrics.get("bridge_edges_count", 0),
            "components_after_bridges": metrics.get("components_after_bridges", metrics["components_count"]),
            "components_merged_by_bridges": metrics.get("components_merged_by_bridges", 0),
            "table_triplets_total": metrics.get("table_triplets_total", 0),
            "table_triplets_tp": metrics.get("table_triplets_tp", 0),
            "table_triplets_fn": metrics.get("table_triplets_fn", 0),
            "table_triplets_recall": metrics.get("table_triplets_recall", 0.0),
        }
    ]
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

    table_rows = result.get("table_triplet_rows") or []
    st.caption("Проверка триплетов из таблиц")
    if table_rows:
        st.write(
            f"Папка: `{result.get('table_tests_dir') or ''}`. "
            f"TP={metrics.get('table_triplets_tp', 0)}, "
            f"FN={metrics.get('table_triplets_fn', 0)}, "
            f"recall={metrics.get('table_triplets_recall', 0.0)}."
        )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    elif result.get("table_tests_dir"):
        st.info("Тестовые CSV-триплеты не найдены или проверка отключена.")

    global_root = result.get("global_root")
    if global_root:
        if global_root["root_type"] == "concept":
            st.info(f"Глобальный родитель графа: {global_root['root_name']} ({global_root['root_norm']})")
        else:
            children = ", ".join(global_root.get("children", []))
            st.info(f"Глобальный виртуальный родитель: {global_root['root_name']}. Корни компонентов: {children}")

    if result["component_rows"]:
        st.caption("Корни компонент")
        st.dataframe(pd.DataFrame(result["component_rows"]), use_container_width=True)

    if result["degree_rows"]:
        st.caption("Степени вершин")
        st.dataframe(pd.DataFrame(result["degree_rows"][:200]), use_container_width=True)

    if result["cycle_rows"]:
        st.caption("Найденные циклы")
        st.dataframe(pd.DataFrame(result["cycle_rows"]), use_container_width=True)
    else:
        st.caption("Найденные циклы")
        st.info("Циклы не обнаружены.")

    if result["synonym_rows"]:
        st.caption("Кандидаты на синонимические связи")
        st.dataframe(pd.DataFrame(result["synonym_rows"]), use_container_width=True)
    else:
        st.caption("Кандидаты на синонимические связи")
        st.info("По текущим порогам кандидаты на синонимические связи не найдены.")

    if result.get("bridge_rows"):
        st.caption("Контекстные мосты между компонентами")
        st.dataframe(pd.DataFrame(result["bridge_rows"]), use_container_width=True)
    else:
        st.caption("Контекстные мосты между компонентами")
        st.info("По текущим порогам мосты между разрозненными компонентами не найдены.")

    if result.get("legacy_root_rows"):
        st.caption("Общие legacy-сущности по контексту")
        st.dataframe(pd.DataFrame(result["legacy_root_rows"]), use_container_width=True)
    else:
        st.caption("Общие legacy-сущности по контексту")
        st.info("По текущим порогам общие legacy-сущности не найдены.")

    render_context_merge_tools(uri=uri, user=user, password=password, analytics_result=result)


def escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\"", "\\\"")


def format_graph_label(value: str, max_len: int = 48) -> str:
    text = " ".join((value or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def format_triplet_option_label(triplet: dict[str, Any], max_len: int = 96) -> str:
    triplet_id = triplet.get("triplet_id", "?")
    subject = format_graph_label(triplet.get("subject", "") or "[пустой субъект]", max_len=28)
    predicate = format_graph_label(triplet.get("predicate", "") or "[пустой предикат]", max_len=24)
    obj = format_graph_label(triplet.get("object", "") or "[пустой объект]", max_len=28)
    sentence = format_graph_label(triplet.get("sentence", "") or "", max_len=max_len)
    base = f"{triplet_id} | {subject} -[{predicate}]-> {obj}"
    return f"{base} | {sentence}" if sentence else base


def get_neo4j_driver(uri: str, user: str, password: str):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: neo4j. Install requirements-graph.txt") from exc
    return GraphDatabase.driver(uri, auth=(user, password))


def payload_triplet_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    rows: list[dict[str, Any]] = []
    for idx, triplet in enumerate(payload.get("triplets", []), start=1):
        if not isinstance(triplet, dict):
            continue
        rows.append(
            {
                "triplet_id": idx,
                "subject": slot_to_text(triplet.get("subject")),
                "predicate": slot_to_text(triplet.get("predicate")),
                "object": slot_to_text(triplet.get("object")),
                "sentence": slot_to_text(triplet.get("sentence", "")),
            }
        )
    return rows


def collect_frame_edges(
    node: dict[str, Any],
    parent_id: str | None,
    role: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    prefix: str,
) -> None:
    current = role_frame(node)
    node_id = prefix
    node_text = slot_to_text(current) or "[empty]"
    nodes.append({"id": node_id, "label": node_text, "role": role})

    if parent_id is not None:
        edges.append({"source": parent_id, "target": node_id, "label": role})

    children = current.get("frame", [])
    if not isinstance(children, list):
        return

    for idx, child in enumerate(children):
        collect_frame_edges(
            role_frame(child),
            parent_id=node_id,
            role=role,
            nodes=nodes,
            edges=edges,
            prefix=f"{prefix}.{idx}",
        )


def build_payload_graph_data(
    payload: dict[str, Any] | None,
    include_frames: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    semantic_rows = payload_triplet_rows(payload)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    def add_node(node_id: str, label: str, kind: str) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({"id": node_id, "label": label or "[empty]", "kind": kind})

    for row in semantic_rows:
        subject = row["subject"] or "[empty subject]"
        obj = row["object"] or "[empty object]"
        predicate = row["predicate"] or "[empty predicate]"
        subj_id = f"entity::{subject}"
        obj_id = f"entity::{obj}"
        add_node(subj_id, subject, "entity")
        add_node(obj_id, obj, "entity")
        edges.append(
            {
                "source": subj_id,
                "target": obj_id,
                "label": predicate,
                "kind": "semantic",
                "triplet_id": row["triplet_id"],
            }
        )

    frame_rows: list[dict[str, Any]] = []
    if include_frames and isinstance(payload, dict):
        for idx, triplet in enumerate(payload.get("triplets", []), start=1):
            if not isinstance(triplet, dict):
                continue
            for role in ("subject", "predicate", "object"):
                frame = role_frame(triplet.get(role))
                frame_nodes: list[dict[str, Any]] = []
                frame_edges: list[dict[str, Any]] = []
                collect_frame_edges(
                    frame,
                    parent_id=None,
                    role=role,
                    nodes=frame_nodes,
                    edges=frame_edges,
                    prefix=f"t{idx}.{role}.0",
                )
                frame_rows.append(
                    {
                        "triplet_id": idx,
                        "role": role,
                        "root_text": slot_to_text(frame),
                        "nodes": frame_nodes,
                        "edges": frame_edges,
                    }
                )

    return semantic_rows, nodes, edges, frame_rows


def render_graphviz_from_edges(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    show_edge_labels: bool,
    max_label_len: int,
    graph_direction: str = "LR",
) -> None:
    dot_lines = [
        "digraph G {",
        f"rankdir={graph_direction};",
        "layout=dot;",
        "overlap=false;",
        "splines=true;",
        "nodesep=0.5;",
        "ranksep=1.0;",
        "node [shape=box, style=\"rounded,filled\", fillcolor=\"#F5F7FA\", color=\"#D0D7DE\", fontname=\"Helvetica\", fontsize=10, margin=\"0.10,0.06\"];",
        "edge [color=\"#6B7280\", fontname=\"Helvetica\", fontsize=9, arrowsize=0.7];",
    ]

    for node in nodes:
        node_id = escape_dot(node["id"])
        label = escape_dot(format_graph_label(node["label"], max_len=max_label_len))
        fill = "#E6F4EA" if node.get("kind") == "entity" else "#FFF4E5"
        if node.get("kind") == "frame":
            fill = "#E8F0FE"
        if node.get("kind") == "triplet":
            fill = "#FCE8E6"
        dot_lines.append(f"\"{node_id}\" [label=\"{label}\", fillcolor=\"{fill}\"];")

    for edge in edges:
        source = escape_dot(edge["source"])
        target = escape_dot(edge["target"])
        label = escape_dot(format_graph_label(edge.get("label", ""), max_len=max_label_len))
        attrs = []
        if show_edge_labels and label:
            attrs.append(f"label=\"{label}\"")
        if edge.get("color"):
            attrs.append(f"color=\"{escape_dot(edge['color'])}\"")
        if edge.get("style"):
            attrs.append(f"style=\"{escape_dot(edge['style'])}\"")
        if attrs:
            dot_lines.append(f"\"{source}\" -> \"{target}\" [{', '.join(attrs)}];")
        else:
            dot_lines.append(f"\"{source}\" -> \"{target}\";")

    dot_lines.append("}")
    st.graphviz_chart("\n".join(dot_lines))


def render_payload_graph_section(
    title: str,
    payload: dict[str, Any] | None,
    key_prefix: str,
) -> None:
    if not isinstance(payload, dict):
        return

    st.caption(title)
    semantic_rows, semantic_nodes, semantic_edges, frame_rows = build_payload_graph_data(payload, include_frames=True)
    if not semantic_rows:
        st.info("Нет триплетов для визуализации.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        local_graph_mode = st.selectbox(
            "Режим графа",
            options=["Семантические триплеты", "Frame-дерево", "Смешанный граф: триплет + frame", "Таблица триплетов"],
            index=0,
            key=f"{key_prefix}_graph_mode",
        )
    with c2:
        show_edge_labels = st.checkbox(
            "Показывать подписи",
            value=True,
            key=f"{key_prefix}_graph_labels",
        )
    with c3:
        max_label_len = st.slider(
            "Макс. длина подписи",
            min_value=20,
            max_value=100,
            value=48,
            step=2,
            key=f"{key_prefix}_graph_label_len",
        )

    if local_graph_mode == "Таблица триплетов":
        st.dataframe(pd.DataFrame(semantic_rows), use_container_width=True)
        return

    if local_graph_mode == "Семантические триплеты":
        st.write(f"Семантических рёбер: {len(semantic_edges)}")
        st.dataframe(pd.DataFrame(semantic_rows), use_container_width=True)
        render_graphviz_from_edges(
            nodes=semantic_nodes,
            edges=semantic_edges,
            show_edge_labels=show_edge_labels,
            max_label_len=max_label_len,
            graph_direction="LR",
        )
        return

    if local_graph_mode == "Смешанный граф: триплет + frame":
        triplet_labels = [
            f"triplet={row['triplet_id']} | {row['subject']} -[{row['predicate']}]-> {row['object']}"
            for row in semantic_rows
        ]
        selected_triplet_label = st.selectbox(
            "Триплет для просмотра",
            options=triplet_labels,
            key=f"{key_prefix}_mixed_triplet_select",
        )
        selected_triplet_idx = triplet_labels.index(selected_triplet_label)
        selected_triplet = semantic_rows[selected_triplet_idx]
        mixed_role_mode = st.selectbox(
            "Роли frame",
            options=["all", "subject", "predicate", "object"],
            index=0,
            key=f"{key_prefix}_mixed_role_select",
        )

        mixed_nodes: list[dict[str, Any]] = []
        mixed_edges: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        def add_mixed_node(node_id: str, label: str, kind: str) -> None:
            if node_id in seen_nodes:
                return
            seen_nodes.add(node_id)
            mixed_nodes.append({"id": node_id, "label": label, "kind": kind})

        triplet_id = selected_triplet["triplet_id"]
        triplet_node_id = f"payload_triplet::{triplet_id}"
        add_mixed_node(triplet_node_id, f"Triplet {triplet_id}", "triplet")

        for role_name, text, kind in (
            ("subject", selected_triplet["subject"], "entity"),
            ("predicate", selected_triplet["predicate"], "predicate"),
            ("object", selected_triplet["object"], "entity"),
        ):
            role_node_id = f"payload_role::{triplet_id}::{role_name}"
            add_mixed_node(role_node_id, text or f"[empty {role_name}]", kind)
            mixed_edges.append(
                {
                    "source": triplet_node_id,
                    "target": role_node_id,
                    "label": role_name,
                    "color": "#B3261E",
                }
            )

        selected_frame_rows = [
            row
            for row in frame_rows
            if row["triplet_id"] == triplet_id and (mixed_role_mode == "all" or row["role"] == mixed_role_mode)
        ]
        for frame_row in selected_frame_rows:
            if not frame_row["nodes"]:
                continue
            root_node_id = frame_row["nodes"][0]["id"]
            for node in frame_row["nodes"]:
                add_mixed_node(node["id"], node["label"], "frame")
            mixed_edges.append(
                {
                    "source": triplet_node_id,
                    "target": root_node_id,
                    "label": f"{frame_row['role']} frame",
                    "color": "#1A73E8",
                    "style": "dashed",
                }
            )
            for edge in frame_row["edges"]:
                mixed_edges.append(
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "label": edge["label"],
                        "color": "#1A73E8",
                    }
                )

        st.write(
            f"Триплет {triplet_id}: семантических ролей=3, "
            f"frame-групп={len(selected_frame_rows)}, узлов={len(mixed_nodes)}, рёбер={len(mixed_edges)}"
        )
        st.dataframe(pd.DataFrame([selected_triplet]), use_container_width=True)
        render_graphviz_from_edges(
            nodes=mixed_nodes,
            edges=mixed_edges,
            show_edge_labels=show_edge_labels,
            max_label_len=max_label_len,
            graph_direction="LR",
        )
        return

    frame_labels = [
        f"triplet={row['triplet_id']} | role={row['role']} | root={row['root_text'] or '[пусто]'}"
        for row in frame_rows
    ]
    selected_frame_label = st.selectbox(
        "Frame для просмотра",
        options=frame_labels,
        key=f"{key_prefix}_frame_select",
    )
    selected_idx = frame_labels.index(selected_frame_label)
    selected_frame = frame_rows[selected_idx]
    frame_nodes = [
        {"id": node["id"], "label": node["label"], "kind": "frame"}
        for node in selected_frame["nodes"]
    ]
    st.write(
        f"Узлов frame: {len(selected_frame['nodes'])}, "
        f"рёбер: {len(selected_frame['edges'])}, "
        f"triplet_id={selected_frame['triplet_id']}, role={selected_frame['role']}"
    )
    render_graphviz_from_edges(
        nodes=frame_nodes,
        edges=selected_frame["edges"],
        show_edge_labels=False,
        max_label_len=max_label_len,
        graph_direction="TB",
    )


def fetch_neo4j_document_ids(uri: str, user: str, password: str) -> list[dict[str, int]]:
    query = """
    MATCH (t:Triplet)-[:IN_DOCUMENT]->(d:Document)
    RETURN d.document_id AS document_id, count(DISTINCT t) AS triplet_count
    ORDER BY document_id DESC
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            result = session.run(query)
            rows = []
            for record in result:
                rows.append(
                    {
                        "document_id": int(record["document_id"]),
                        "triplet_count": int(record["triplet_count"]),
                    }
                )
            return rows
    finally:
        driver.close()


def fetch_neo4j_relations(
    uri: str,
    user: str,
    password: str,
    limit: int,
    contains: str,
    document_id: int | None = None,
) -> list[dict[str, str]]:
    query = """
    CALL {
      MATCH (a:Entity)-[r:RELATION]->(b:Entity)
      WHERE ($contains = "" OR toLower(a.name) CONTAINS toLower($contains) OR toLower(b.name) CONTAINS toLower($contains))
        AND ($document_id IS NULL OR r.document_id = $document_id)
      RETURN a.name AS subject, r.predicate AS predicate, b.name AS object, r.triplet_id AS triplet_id
      UNION ALL
      MATCH (a:Entity)-[r:CONTEXT_PARENT]->(b:Entity)
      WHERE ($contains = "" OR toLower(a.name) CONTAINS toLower($contains) OR toLower(b.name) CONTAINS toLower($contains))
        AND $document_id IS NULL
      RETURN a.name AS subject, 'контекстный корень' AS predicate, b.name AS object, r.example_triplet_id AS triplet_id
    }
    RETURN subject, predicate, object, triplet_id
    LIMIT $limit
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            result = session.run(
                query,
                {
                    "contains": contains,
                    "limit": int(limit),
                    "document_id": document_id,
                },
            )
            rows = []
            for record in result:
                rows.append(
                    {
                        "subject": record["subject"] or "",
                        "predicate": record["predicate"] or "",
                        "object": record["object"] or "",
                        "triplet_id": int(record["triplet_id"]) if record["triplet_id"] is not None else None,
                    }
                )
            return rows
    finally:
        driver.close()


def fetch_neo4j_concept_relations(
    uri: str,
    user: str,
    password: str,
    limit: int,
    contains: str,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    query = """
    MATCH (s:EntityConcept)-[r:RELATION_INSTANCE]->(o:EntityConcept)
    WHERE ($contains = "" OR toLower(s.name) CONTAINS toLower($contains) OR toLower(o.name) CONTAINS toLower($contains))
      AND ($document_id IS NULL OR r.document_id = $document_id)
    RETURN
      s.name AS subject,
      r.predicate AS predicate,
      o.name AS object,
      r.triplet_id AS triplet_id,
      r.document_id AS document_id
    LIMIT $limit
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            result = session.run(
                query,
                {
                    "contains": contains,
                    "limit": int(limit),
                    "document_id": document_id,
                },
            )
            return [
                {
                    "subject": record["subject"] or "",
                    "predicate": record["predicate"] or "",
                    "object": record["object"] or "",
                    "triplet_id": int(record["triplet_id"]) if record["triplet_id"] is not None else None,
                    "document_id": int(record["document_id"]) if record["document_id"] is not None else None,
                }
                for record in result
            ]
    finally:
        driver.close()


def fetch_neo4j_context_bridges(
    uri: str,
    user: str,
    password: str,
    limit: int,
    contains: str,
) -> list[dict[str, Any]]:
    query = """
    MATCH (s:EntityConcept)-[r:CONTEXT_BRIDGE]->(o:EntityConcept)
    WHERE (
      $contains = ""
      OR toLower(coalesce(s.name, s.norm, "")) CONTAINS toLower($contains)
      OR toLower(coalesce(o.name, o.norm, "")) CONTAINS toLower($contains)
      OR toLower(coalesce(r.shared_predicates, "")) CONTAINS toLower($contains)
      OR toLower(coalesce(r.shared_children, "")) CONTAINS toLower($contains)
    )
    RETURN
      coalesce(s.name, s.norm) AS subject,
      ('bridge ' + toString(round(coalesce(r.bridge_score, 0.0) * 100.0) / 100.0)) AS predicate,
      coalesce(o.name, o.norm) AS object,
      coalesce(r.bridge_score, 0.0) AS bridge_score,
      coalesce(r.structural_score, 0.0) AS structural_score,
      coalesce(r.embedding_score, 0.0) AS embedding_score,
      coalesce(r.support, 0) AS support,
      coalesce(r.left_component_id, 0) AS left_component_id,
      coalesce(r.right_component_id, 0) AS right_component_id,
      coalesce(r.left_document_ids, '') AS left_document_ids,
      coalesce(r.right_document_ids, '') AS right_document_ids,
      coalesce(r.shared_predicates, '') AS shared_predicates,
      coalesce(r.shared_children, '') AS shared_children
    ORDER BY bridge_score DESC, structural_score DESC, embedding_score DESC
    LIMIT $limit
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            result = session.run(query, {"contains": contains, "limit": int(limit)})
            return [dict(record) for record in result]
    finally:
        driver.close()


def fetch_neo4j_approved_context_links(
    uri: str,
    user: str,
    password: str,
    limit: int,
    contains: str,
) -> list[dict[str, Any]]:
    query = """
    MATCH (s:EntityConcept)-[r]->(o:EntityConcept)
    WHERE type(r) IN [
      'APPROVED_ALIAS_OF',
      'APPROVED_BROADER_THAN',
      'APPROVED_CONTEXT_LINK',
      'MANUAL_CONTEXT_LINK',
      'MANUAL_COMPONENT_LINK'
    ]
      AND (
        $contains = ""
        OR toLower(coalesce(s.name, s.norm, "")) CONTAINS toLower($contains)
        OR toLower(coalesce(o.name, o.norm, "")) CONTAINS toLower($contains)
        OR toLower(coalesce(r.relation_text, "")) CONTAINS toLower($contains)
      )
    RETURN
      coalesce(s.name, s.norm) AS subject,
      coalesce(r.relation_text, type(r)) AS predicate,
      coalesce(o.name, o.norm) AS object,
      type(r) AS relation_type,
      coalesce(r.note, '') AS note
    ORDER BY relation_type, subject, object
    LIMIT $limit
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            return [dict(record) for record in session.run(query, {"contains": contains, "limit": int(limit)})]
    finally:
        driver.close()


def normalize_graph_export_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [normalize_graph_export_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_graph_export_value(item) for key, item in value.items()}
    return str(value)


def fetch_neo4j_full_graph_export(
    uri: str,
    user: str,
    password: str,
    node_limit: int,
    relation_limit: int,
    contains: str = "",
) -> dict[str, Any]:
    node_query = """
    MATCH (n)
    WHERE $contains = ""
       OR any(label IN labels(n) WHERE toLower(label) CONTAINS toLower($contains))
       OR any(value IN [key IN keys(n) | toString(n[key])] WHERE toLower(value) CONTAINS toLower($contains))
    WITH n
    ORDER BY elementId(n)
    LIMIT $node_limit
    RETURN
      elementId(n) AS id,
      id(n) AS neo4j_id,
      labels(n) AS labels,
      properties(n) AS properties
    """
    rel_query = """
    MATCH (a)-[r]->(b)
    WHERE $contains = ""
       OR toLower(type(r)) CONTAINS toLower($contains)
       OR any(value IN [key IN keys(r) | toString(r[key])] WHERE toLower(value) CONTAINS toLower($contains))
       OR any(value IN [key IN keys(a) | toString(a[key])] WHERE toLower(value) CONTAINS toLower($contains))
       OR any(value IN [key IN keys(b) | toString(b[key])] WHERE toLower(value) CONTAINS toLower($contains))
    WITH a, r, b
    ORDER BY elementId(r)
    LIMIT $relation_limit
    RETURN
      elementId(r) AS id,
      id(r) AS neo4j_id,
      type(r) AS type,
      elementId(a) AS source,
      elementId(b) AS target,
      properties(r) AS properties
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            nodes = [
                {
                    "id": record["id"],
                    "neo4j_id": int(record["neo4j_id"]),
                    "labels": list(record["labels"] or []),
                    "properties": normalize_graph_export_value(dict(record["properties"] or {})),
                }
                for record in session.run(
                    node_query,
                    {
                        "contains": contains,
                        "node_limit": int(node_limit),
                    },
                )
            ]
            node_ids = {node["id"] for node in nodes}
            edges = []
            skipped_edges = 0
            for record in session.run(
                rel_query,
                {
                    "contains": contains,
                    "relation_limit": int(relation_limit),
                },
            ):
                edge = {
                    "id": record["id"],
                    "neo4j_id": int(record["neo4j_id"]),
                    "type": record["type"] or "",
                    "source": record["source"],
                    "target": record["target"],
                    "properties": normalize_graph_export_value(dict(record["properties"] or {})),
                }
                if edge["source"] not in node_ids or edge["target"] not in node_ids:
                    skipped_edges += 1
                    continue
                edges.append(edge)
            return {
                "format": "neo4j_graph_export",
                "nodes": nodes,
                "edges": edges,
                "meta": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "skipped_edges_outside_node_limit": skipped_edges,
                    "node_limit": int(node_limit),
                    "relation_limit": int(relation_limit),
                    "filter": contains,
                },
            }
    finally:
        driver.close()


def graph_export_to_json_bytes(graph_data: dict[str, Any]) -> bytes:
    return json.dumps(graph_data, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def graph_export_to_graphml_bytes(graph_data: dict[str, Any]) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="neo4j_element_id" for="node" attr.name="neo4j_element_id" attr.type="string"/>',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="labels" for="node" attr.name="labels" attr.type="string"/>',
        '  <key id="properties_json" for="node" attr.name="properties_json" attr.type="string"/>',
        '  <key id="edge_neo4j_element_id" for="edge" attr.name="neo4j_element_id" attr.type="string"/>',
        '  <key id="type" for="edge" attr.name="type" attr.type="string"/>',
        '  <key id="edge_properties_json" for="edge" attr.name="properties_json" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    graphml_node_ids = {
        str(node["id"]): f"n{idx}"
        for idx, node in enumerate(graph_data.get("nodes", []))
    }
    for node in graph_data.get("nodes", []):
        original_node_id = str(node["id"])
        node_id = graphml_node_ids[original_node_id]
        props = node.get("properties") or {}
        label = props.get("name") or props.get("text") or props.get("norm") or node_id
        props_json = json.dumps(props, ensure_ascii=False, sort_keys=True, default=str)
        labels_text = ", ".join(node.get("labels") or [])
        lines.extend(
            [
                f'    <node id="{node_id}">',
                f'      <data key="neo4j_element_id">{xml_escape(original_node_id)}</data>',
                f'      <data key="label">{xml_escape(str(label))}</data>',
                f'      <data key="labels">{xml_escape(labels_text)}</data>',
                f'      <data key="properties_json">{xml_escape(props_json)}</data>',
                "    </node>",
            ]
        )
    for idx, edge in enumerate(graph_data.get("edges", [])):
        edge_id = f"e{idx}"
        original_edge_id = str(edge["id"])
        source = graphml_node_ids.get(str(edge["source"]))
        target = graphml_node_ids.get(str(edge["target"]))
        if source is None or target is None:
            continue
        props_json = json.dumps(edge.get("properties") or {}, ensure_ascii=False, sort_keys=True, default=str)
        lines.extend(
            [
                f'    <edge id="{edge_id}" source="{source}" target="{target}">',
                f'      <data key="edge_neo4j_element_id">{xml_escape(original_edge_id)}</data>',
                f'      <data key="type">{xml_escape(str(edge.get("type") or ""))}</data>',
                f'      <data key="edge_properties_json">{xml_escape(props_json)}</data>',
                "    </edge>",
            ]
        )
    lines.extend(["  </graph>", "</graphml>"])
    return "\n".join(lines).encode("utf-8")


def sanitize_cypher_identifier(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value or "").strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned[:120]


def sanitize_node_label(value: str) -> str:
    return sanitize_cypher_identifier(value, "ImportedNode")


def sanitize_relation_type(value: str) -> str:
    return sanitize_cypher_identifier(value, "IMPORTED_RELATION").upper()


def coerce_neo4j_property_value(value: Any) -> Any:
    value = normalize_graph_export_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
            return [item for item in value if item is not None]
        return json.dumps(value, ensure_ascii=False, default=str)
    return json.dumps(value, ensure_ascii=False, default=str)


def coerce_neo4j_properties(properties: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (properties or {}).items():
        clean_key = sanitize_cypher_identifier(str(key), "property")
        clean_value = coerce_neo4j_property_value(value)
        if clean_value is not None:
            result[clean_key] = clean_value
    return result


def parse_graph_json_upload(file_bytes: bytes) -> dict[str, Any]:
    raw = json.loads(decode_bytes_safe(file_bytes))
    if isinstance(raw, dict) and "nodes" in raw and ("edges" in raw or "relationships" in raw):
        nodes = raw.get("nodes") or []
        edges = raw.get("edges") or raw.get("relationships") or []
    elif isinstance(raw, list):
        raise ValueError("JSON должен быть объектом с полями nodes и edges, а не списком.")
    else:
        raise ValueError("JSON должен содержать поля nodes и edges.")

    parsed_nodes = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or node.get("neo4j_id") or f"node_{idx}")
        labels = node.get("labels") or node.get("label") or ["ImportedNode"]
        if isinstance(labels, str):
            labels = [labels]
        props = node.get("properties") or {}
        if not isinstance(props, dict):
            props = {"value": props}
        parsed_nodes.append(
            {
                "id": node_id,
                "labels": [str(label) for label in labels if str(label).strip()],
                "properties": props,
            }
        )

    parsed_edges = []
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        source = edge.get("source") or edge.get("from") or edge.get("start")
        target = edge.get("target") or edge.get("to") or edge.get("end")
        if source is None or target is None:
            continue
        props = edge.get("properties") or {}
        if not isinstance(props, dict):
            props = {"value": props}
        parsed_edges.append(
            {
                "id": str(edge.get("id") or edge.get("neo4j_id") or f"edge_{idx}"),
                "source": str(source),
                "target": str(target),
                "type": str(edge.get("type") or edge.get("label") or "IMPORTED_RELATION"),
                "properties": props,
            }
        )
    return {"nodes": parsed_nodes, "edges": parsed_edges, "meta": {"source_format": "json"}}


def parse_graphml_upload(file_bytes: bytes) -> dict[str, Any]:
    root = ET.fromstring(file_bytes)
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0][1:]
    ns = {"g": namespace} if namespace else {}
    prefix = "g:" if namespace else ""

    key_names: dict[str, str] = {}
    for key_el in root.findall(f"{prefix}key", ns):
        key_id = key_el.attrib.get("id", "")
        key_name = key_el.attrib.get("attr.name") or key_id
        key_names[key_id] = key_name

    def read_data(element: ET.Element) -> dict[str, str]:
        data: dict[str, str] = {}
        for data_el in element.findall(f"{prefix}data", ns):
            key = key_names.get(data_el.attrib.get("key", ""), data_el.attrib.get("key", ""))
            data[key] = data_el.text or ""
        return data

    graph_el = root.find(f"{prefix}graph", ns)
    if graph_el is None:
        raise ValueError("GraphML не содержит элемента graph.")

    parsed_nodes = []
    for node_el in graph_el.findall(f"{prefix}node", ns):
        node_id = node_el.attrib.get("id", "")
        data = read_data(node_el)
        props: dict[str, Any] = {}
        if data.get("properties_json"):
            try:
                props.update(json.loads(data["properties_json"]))
            except json.JSONDecodeError:
                props["properties_json"] = data["properties_json"]
        for key, value in data.items():
            if key in {"properties_json", "labels"}:
                continue
            props.setdefault(key, value)
        labels_text = data.get("labels", "")
        labels = [item.strip() for item in labels_text.split(",") if item.strip()] or ["ImportedNode"]
        parsed_nodes.append({"id": node_id, "labels": labels, "properties": props})

    parsed_edges = []
    for idx, edge_el in enumerate(graph_el.findall(f"{prefix}edge", ns)):
        data = read_data(edge_el)
        props: dict[str, Any] = {}
        if data.get("properties_json"):
            try:
                props.update(json.loads(data["properties_json"]))
            except json.JSONDecodeError:
                props["properties_json"] = data["properties_json"]
        for key, value in data.items():
            if key in {"properties_json", "type"}:
                continue
            props.setdefault(key, value)
        parsed_edges.append(
            {
                "id": edge_el.attrib.get("id", f"edge_{idx}"),
                "source": edge_el.attrib.get("source", ""),
                "target": edge_el.attrib.get("target", ""),
                "type": data.get("type") or edge_el.attrib.get("label") or "IMPORTED_RELATION",
                "properties": props,
            }
        )
    return {"nodes": parsed_nodes, "edges": parsed_edges, "meta": {"source_format": "graphml"}}


def parse_graph_upload(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    lower_name = file_name.lower()
    if lower_name.endswith(".json"):
        return parse_graph_json_upload(file_bytes)
    if lower_name.endswith(".graphml") or lower_name.endswith(".xml"):
        return parse_graphml_upload(file_bytes)
    raise ValueError("Поддерживаются только .json, .graphml и .xml.")


def import_graph_to_neo4j(
    uri: str,
    user: str,
    password: str,
    graph_data: dict[str, Any],
    namespace: str,
) -> dict[str, int]:
    namespace = safe_entity_name(namespace or "imported_graph") or "imported_graph"
    nodes = graph_data.get("nodes") or []
    edges = graph_data.get("edges") or []
    driver = get_neo4j_driver(uri, user, password)
    created_nodes = 0
    updated_nodes = 0
    created_edges = 0
    skipped_edges = 0
    try:
        with driver.session() as session:
            for node in nodes:
                original_id = str(node.get("id") or "")
                if not original_id:
                    continue
                labels = [sanitize_node_label(label) for label in (node.get("labels") or ["ImportedNode"])]
                label_clause = ":".join(dict.fromkeys(["ImportedGraphNode", *labels]))
                import_key = f"{namespace}::{original_id}"
                props = coerce_neo4j_properties(node.get("properties") or {})
                props.update(
                    {
                        "import_namespace": namespace,
                        "import_id": original_id,
                        "import_key": import_key,
                        "import_labels_json": json.dumps(node.get("labels") or [], ensure_ascii=False),
                    }
                )
                record = session.run(
                    f"""
                    MERGE (n:{label_clause} {{import_key: $import_key}})
                    ON CREATE SET n.created_at = datetime(), n.import_created = true
                    ON MATCH SET n.import_created = false
                    SET n += $props,
                        n.updated_at = datetime()
                    RETURN n.import_created AS created
                    """,
                    {"import_key": import_key, "props": props},
                ).single()
                if record and record["created"]:
                    created_nodes += 1
                else:
                    updated_nodes += 1

            for edge in edges:
                source_id = str(edge.get("source") or "")
                target_id = str(edge.get("target") or "")
                if not source_id or not target_id:
                    skipped_edges += 1
                    continue
                rel_type = sanitize_relation_type(str(edge.get("type") or "IMPORTED_RELATION"))
                props = coerce_neo4j_properties(edge.get("properties") or {})
                props.update(
                    {
                        "import_namespace": namespace,
                        "import_id": str(edge.get("id") or ""),
                        "import_type": str(edge.get("type") or "IMPORTED_RELATION"),
                    }
                )
                record = session.run(
                    f"""
                    MATCH (source:ImportedGraphNode {{import_key: $source_key}})
                    MATCH (target:ImportedGraphNode {{import_key: $target_key}})
                    MERGE (source)-[r:{rel_type} {{import_namespace: $namespace, import_id: $import_id}}]->(target)
                    ON CREATE SET r.created_at = datetime(), r.import_created = true
                    ON MATCH SET r.import_created = false
                    SET r += $props,
                        r.updated_at = datetime()
                    RETURN r.import_created AS created
                    """,
                    {
                        "source_key": f"{namespace}::{source_id}",
                        "target_key": f"{namespace}::{target_id}",
                        "namespace": namespace,
                        "import_id": str(edge.get("id") or f"{source_id}->{target_id}:{rel_type}"),
                        "props": props,
                    },
                ).single()
                if record is None:
                    skipped_edges += 1
                elif record["created"]:
                    created_edges += 1
        return {
            "created_nodes": created_nodes,
            "updated_nodes": updated_nodes,
            "created_edges": created_edges,
            "skipped_edges": skipped_edges,
        }
    finally:
        driver.close()


def fetch_neo4j_entity_concept_options(
    uri: str,
    user: str,
    password: str,
    limit: int,
    contains: str = "",
) -> list[dict[str, Any]]:
    query = """
    MATCH (c:EntityConcept)
    WHERE $contains = ""
       OR toLower(coalesce(c.name, c.norm, "")) CONTAINS toLower($contains)
       OR toLower(coalesce(c.norm, "")) CONTAINS toLower($contains)
    OPTIONAL MATCH (c)-[out:RELATION_INSTANCE]->()
    OPTIONAL MATCH ()-[inn:RELATION_INSTANCE]->(c)
    OPTIONAL MATCH (occ:FrameOccurrence)-[:OF_CONCEPT]->(c)
    RETURN
      c.norm AS norm,
      coalesce(c.name, c.norm) AS name,
      count(DISTINCT out) AS out_degree,
      count(DISTINCT inn) AS in_degree,
      count(DISTINCT occ) AS frame_count
    ORDER BY (out_degree + in_degree) DESC, frame_count DESC, name
    LIMIT $limit
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            return [
                {
                    "norm": record["norm"] or "",
                    "name": record["name"] or "",
                    "out_degree": int(record["out_degree"] or 0),
                    "in_degree": int(record["in_degree"] or 0),
                    "frame_count": int(record["frame_count"] or 0),
                }
                for record in session.run(query, {"contains": contains, "limit": int(limit)})
            ]
    finally:
        driver.close()


def create_approved_concept_relation(
    uri: str,
    user: str,
    password: str,
    left_norm: str,
    right_norm: str,
    relation_kind: str,
    relation_text: str,
    note: str = "",
) -> None:
    relation_type_map = {
        "alias": "APPROVED_ALIAS_OF",
        "broader": "APPROVED_BROADER_THAN",
        "context": "APPROVED_CONTEXT_LINK",
        "manual": "MANUAL_CONTEXT_LINK",
    }
    rel_type = relation_type_map.get(relation_kind, "MANUAL_CONTEXT_LINK")
    query = f"""
    MATCH (left:EntityConcept {{norm: $left_norm}})
    MATCH (right:EntityConcept {{norm: $right_norm}})
    MERGE (left)-[rel:{rel_type}]->(right)
    SET rel.relation_text = $relation_text,
        rel.note = $note,
        rel.approved = true,
        rel.updated_at = datetime()
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            session.run(
                query,
                {
                    "left_norm": left_norm,
                    "right_norm": right_norm,
                    "relation_text": relation_text,
                    "note": note,
                },
            )
    finally:
        driver.close()


def create_context_synonym_component(
    uri: str,
    user: str,
    password: str,
    root_name: str,
    child_rows: list[dict[str, Any]],
    relation_text: str,
    note: str = "",
) -> dict[str, Any]:
    root_name = normalize_whitespace(root_name)
    root_norm = safe_entity_name(root_name)
    children = []
    seen_child_norms = set()
    for row in child_rows:
        child_name = normalize_whitespace(row.get("child_name") or row.get("child_norm") or "")
        child_norm = normalize_whitespace(row.get("child_norm") or safe_entity_name(child_name))
        if not child_norm or child_norm == root_norm or child_norm in seen_child_norms:
            continue
        seen_child_norms.add(child_norm)
        children.append(
            {
                "child_norm": child_norm,
                "child_name": child_name or child_norm,
                "cluster_score": float(row.get("cluster_score") or 0.0),
                "shared_examples": row.get("shared_examples", ""),
            }
        )

    query = """
    MERGE (root:EntityConcept {norm: $root_norm})
      ON CREATE SET root.name = $root_name,
                    root.manual = true,
                    root.context_component = true,
                    root.created_at = datetime()
      ON MATCH SET root.name = $root_name,
                   root.manual = true,
                   root.context_component = true,
                   root.updated_at = datetime()
    WITH root
    UNWIND $children AS child_data
    MERGE (child:EntityConcept {norm: child_data.child_norm})
      ON CREATE SET child.name = child_data.child_name,
                    child.created_from_legacy_context = true,
                    child.created_at = datetime()
      ON MATCH SET child.name = coalesce(child.name, child_data.child_name)
    MERGE (root)-[rel:APPROVED_ALIAS_OF]->(child)
    SET rel.relation_text = $relation_text,
        rel.note = $note,
        rel.approved = true,
        rel.source = "legacy_context_root",
        rel.cluster_score = child_data.cluster_score,
        rel.shared_examples = child_data.shared_examples,
        rel.updated_at = datetime()
    RETURN root.norm AS root_norm, root.name AS root_name, count(rel) AS linked_count
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            record = session.run(
                query,
                {
                    "root_norm": root_norm,
                    "root_name": root_name,
                    "children": children,
                    "relation_text": relation_text,
                    "note": note,
                },
            ).single()
            return {
                "root_norm": record["root_norm"] if record else root_norm,
                "root_name": record["root_name"] if record else root_name,
                "linked_count": int(record["linked_count"]) if record else 0,
            }
    finally:
        driver.close()


def create_manual_component_connector(
    uri: str,
    user: str,
    password: str,
    left_norm: str,
    right_norm: str,
    connector_name: str,
    relation_text: str,
    auto_name: bool = False,
    note: str = "",
) -> dict[str, str]:
    connector_name = normalize_whitespace(connector_name or "")
    if not connector_name:
        connector_name = f"Связка: {relation_text.strip() or 'контекстная связь'}"
    connector_norm = safe_entity_name(connector_name)
    query = """
    MATCH (left:EntityConcept {norm: $left_norm})
    MATCH (right:EntityConcept {norm: $right_norm})
    MERGE (connector:EntityConcept {norm: $connector_norm})
      ON CREATE SET connector.name = $connector_name,
                    connector.manual = true,
                    connector.connector = true,
                    connector.created_at = datetime()
      ON MATCH SET connector.name = coalesce(connector.name, $connector_name),
                   connector.manual = true,
                   connector.connector = true
    MERGE (left)-[lr:MANUAL_COMPONENT_LINK]->(connector)
    SET lr.relation_text = $relation_text,
        lr.note = $note,
        lr.auto_name = $auto_name,
        lr.updated_at = datetime()
    MERGE (connector)-[rr:MANUAL_COMPONENT_LINK]->(right)
    SET rr.relation_text = $relation_text,
        rr.note = $note,
        rr.auto_name = $auto_name,
        rr.updated_at = datetime()
    RETURN connector.norm AS connector_norm, connector.name AS connector_name
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            record = session.run(
                query,
                {
                    "left_norm": left_norm,
                    "right_norm": right_norm,
                    "connector_norm": connector_norm,
                    "connector_name": connector_name,
                    "relation_text": relation_text,
                    "auto_name": bool(auto_name),
                    "note": note,
                },
            ).single()
            return {
                "connector_norm": record["connector_norm"] if record else connector_norm,
                "connector_name": record["connector_name"] if record else connector_name,
            }
    finally:
        driver.close()


def fetch_neo4j_triplets(
    uri: str,
    user: str,
    password: str,
    limit: int,
    contains: str,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    query = """
    MATCH (t:Triplet)
    OPTIONAL MATCH (t)-[:SUBJECT]->(s:EntityConcept)
    OPTIONAL MATCH (t)-[:PREDICATE]->(p:RelationConcept)
    OPTIONAL MATCH (t)-[:OBJECT]->(o:EntityConcept)
    WHERE ($document_id IS NULL OR t.document_id = $document_id)
      AND (
        $contains = ""
        OR toLower(coalesce(s.name, "")) CONTAINS toLower($contains)
        OR toLower(coalesce(p.name, "")) CONTAINS toLower($contains)
        OR toLower(coalesce(o.name, "")) CONTAINS toLower($contains)
        OR toLower(coalesce(t.sentence, "")) CONTAINS toLower($contains)
      )
    RETURN
      t.triplet_id AS triplet_id,
      t.document_id AS document_id,
      coalesce(s.name, t.subject_text, "") AS subject,
      coalesce(p.name, t.predicate_text, "") AS predicate,
      coalesce(o.name, t.object_text, "") AS object,
      coalesce(t.sentence, "") AS sentence
    ORDER BY t.triplet_id DESC
    LIMIT $limit
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            result = session.run(
                query,
                {
                    "contains": contains,
                    "limit": int(limit),
                    "document_id": document_id,
                },
            )
            return [
                {
                    "triplet_id": int(record["triplet_id"]),
                    "document_id": int(record["document_id"]) if record["document_id"] is not None else None,
                    "subject": record["subject"] or "",
                    "predicate": record["predicate"] or "",
                    "object": record["object"] or "",
                    "sentence": record["sentence"] or "",
                }
                for record in result
            ]
    finally:
        driver.close()


def fetch_neo4j_triplet_by_id(
    uri: str,
    user: str,
    password: str,
    triplet_id: int,
) -> dict[str, Any] | None:
    query = """
    MATCH (t:Triplet {triplet_id: $triplet_id})
    OPTIONAL MATCH (t)-[:SUBJECT]->(s:EntityConcept)
    OPTIONAL MATCH (t)-[:PREDICATE]->(p:RelationConcept)
    OPTIONAL MATCH (t)-[:OBJECT]->(o:EntityConcept)
    RETURN
      t.triplet_id AS triplet_id,
      t.document_id AS document_id,
      coalesce(s.name, t.subject_text, "") AS subject,
      coalesce(p.name, t.predicate_text, "") AS predicate,
      coalesce(o.name, t.object_text, "") AS object,
      coalesce(t.sentence, "") AS sentence
    LIMIT 1
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            record = session.run(query, {"triplet_id": int(triplet_id)}).single()
            if record is None:
                return None
            return {
                "triplet_id": int(record["triplet_id"]),
                "document_id": int(record["document_id"]) if record["document_id"] is not None else None,
                "subject": record["subject"] or "",
                "predicate": record["predicate"] or "",
                "object": record["object"] or "",
                "sentence": record["sentence"] or "",
            }
    finally:
        driver.close()


def fetch_neo4j_frame_tree(
    uri: str,
    user: str,
    password: str,
    triplet_id: int,
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query = """
    MATCH (occ:FrameOccurrence {triplet_id: $triplet_id, role: $role})-[:HAS_ROOT]->(root:FrameNode)
    OPTIONAL MATCH (parent:FrameNode {triplet_id: $triplet_id, role: $role})-[rel:CHILD]->(child:FrameNode {triplet_id: $triplet_id, role: $role})
    RETURN
      collect(DISTINCT {
        id: root.triplet_id + ':' + root.role + ':' + root.path,
        path: root.path,
        text: root.text,
        depth: root.depth,
        ord: root.ord,
        is_root: root.is_root
      }) +
      collect(DISTINCT {
        id: parent.triplet_id + ':' + parent.role + ':' + parent.path,
        path: parent.path,
        text: parent.text,
        depth: parent.depth,
        ord: parent.ord,
        is_root: parent.is_root
      }) +
      collect(DISTINCT {
        id: child.triplet_id + ':' + child.role + ':' + child.path,
        path: child.path,
        text: child.text,
        depth: child.depth,
        ord: child.ord,
        is_root: child.is_root
      }) AS raw_nodes,
      collect(DISTINCT {
        source: parent.triplet_id + ':' + parent.role + ':' + parent.path,
        target: child.triplet_id + ':' + child.role + ':' + child.path,
        ord: rel.ord
      }) AS raw_edges
    """
    driver = get_neo4j_driver(uri, user, password)
    try:
        with driver.session() as session:
            record = session.run(query, {"triplet_id": int(triplet_id), "role": role}).single()
            if record is None:
                return [], []

            node_map: dict[str, dict[str, Any]] = {}
            for item in record["raw_nodes"]:
                if not item or item.get("id") is None:
                    continue
                node_map[item["id"]] = {
                    "id": item["id"],
                    "label": item.get("text") or "[empty]",
                    "kind": "frame",
                    "depth": item.get("depth"),
                    "path": item.get("path"),
                    "ord": item.get("ord"),
                    "is_root": item.get("is_root"),
                }

            edges = []
            for item in record["raw_edges"]:
                if not item or item.get("source") is None or item.get("target") is None:
                    continue
                edges.append(
                    {
                        "source": item["source"],
                        "target": item["target"],
                        "label": str(item.get("ord", "")),
                    }
                )

            nodes = sorted(
                node_map.values(),
                key=lambda item: (int(item.get("depth") or 0), str(item.get("path") or "")),
            )
            return nodes, edges
    finally:
        driver.close()


def render_neo4j_graph_section() -> None:
    st.subheader("Просмотр графа Neo4j")
    c1, c2, c3 = st.columns(3)
    with c1:
        uri = st.text_input("URI Neo4j (просмотр)", value="bolt://localhost:7687", key="neo_view_uri")
    with c2:
        user = st.text_input("Пользователь Neo4j (просмотр)", value="neo4j", key="neo_view_user")
    with c3:
        password = st.text_input("Пароль Neo4j (просмотр)", value="neo4jpass", type="password", key="neo_view_password")

    c4, c5 = st.columns(2)
    with c4:
        contains = st.text_input("Фильтр", value="", key="neo_view_filter")
    with c5:
        limit = st.slider("Макс. строк", min_value=10, max_value=500, value=100, step=10, key="neo_view_limit")

    c6, c7, c8 = st.columns(3)
    with c6:
        show_edge_labels = st.checkbox("Показывать подписи рёбер", value=True, key="neo_view_show_edge_labels")
    with c7:
        deduplicate_edges = st.checkbox("Объединять дубликаты рёбер", value=True, key="neo_view_dedup")
    with c8:
        max_label_len = st.slider("Макс. длина подписи", min_value=20, max_value=100, value=48, step=2, key="neo_view_max_label")

    graph_mode = st.radio(
        "Режим просмотра графа",
        options=[
            "Legacy-связи",
            # "Связи концептов",
            "Контекстные мосты",
            "Одобренные/ручные связи",
            "Триплеты",
            "Frame-дерево",
            "Смешанный граф: триплет + frame",
        ],
        index=0,
        key="neo_view_mode",
    )
    selected_doc_id: int | None = None
    selected_triplet_id: int | None = None
    selected_role = "subject"

    use_document_filter = graph_mode in ("Legacy-связи", "Триплеты") #"Связи концептов"
    if use_document_filter:
        doc_filter_mode = st.radio(
            "Режим фильтрации по документу",
            options=["Все документы", "По document_id"],
            index=0,
            key="neo_view_doc_filter_mode",
        )
    else:
        doc_filter_mode = "Все документы"

    if doc_filter_mode == "По document_id":
        try:
            doc_rows = fetch_neo4j_document_ids(uri=uri, user=user, password=password)
            if not doc_rows:
                st.info("В Neo4j пока нет значений document_id у триплетов.")
                return

            labels = [f"{row['document_id']} (триплетов: {row['triplet_count']})" for row in doc_rows]
            selected_label = st.selectbox("Выберите document_id", options=labels, key="neo_view_doc_select")
            selected_doc_id = int(selected_label.split(" ", 1)[0])
        except Exception as exc:
            st.error(f"Не удалось загрузить список документов: {exc}")
            st.code(traceback.format_exc(), language="text")
            return

    if graph_mode in ("Frame-дерево", "Смешанный граф: триплет + frame"):
        role_key = "neo_view_frame_role_frame_tree" if graph_mode == "Frame-дерево" else "neo_view_frame_role_mixed"
        triplet_pick_mode = st.radio(
            "Режим выбора триплета",
            options=["Выбрать триплет", "Ввести triplet_id вручную"],
            index=0,
            key="neo_view_triplet_pick_mode",
        )

        if triplet_pick_mode == "Выбрать триплет":
            try:
                triplet_options = fetch_neo4j_triplets(
                    uri=uri,
                    user=user,
                    password=password,
                    limit=limit,
                    contains=contains,
                    document_id=selected_doc_id,
                )
            except Exception as exc:
                st.error(f"Не удалось загрузить список триплетов: {exc}")
                st.code(traceback.format_exc(), language="text")
                return

            if not triplet_options:
                st.info("Для текущих фильтров триплеты не найдены.")
                return

            option_labels = [format_triplet_option_label(row) for row in triplet_options]
            f1, f2 = st.columns(2)
            with f1:
                selected_triplet_label = st.selectbox(
                    "Триплет",
                    options=option_labels,
                    key="neo_view_triplet_select",
                )
                selected_triplet_idx = option_labels.index(selected_triplet_label)
                selected_triplet_id = int(triplet_options[selected_triplet_idx]["triplet_id"])
            with f2:
                selected_role = st.selectbox(
                    "Роль frame",
                    options=["subject", "predicate", "object"] if graph_mode == "Frame-дерево" else ["all", "subject", "predicate", "object"],
                    index=0,
                    key=role_key,
                )

            with st.expander("Выбранный триплет", expanded=False):
                st.dataframe(pd.DataFrame([triplet_options[selected_triplet_idx]]), use_container_width=True)
        else:
            f1, f2 = st.columns(2)
            with f1:
                selected_triplet_id = int(
                    st.number_input(
                        "Triplet ID",
                        min_value=1,
                        value=1,
                        step=1,
                        key="neo_view_triplet_id",
                    )
                )
            with f2:
                selected_role = st.selectbox(
                    "Роль frame",
                    options=["subject", "predicate", "object"] if graph_mode == "Frame-дерево" else ["all", "subject", "predicate", "object"],
                    index=0,
                    key=role_key,
                )

    if st.button("Загрузить граф из Neo4j", key="neo_view_load_btn"):
        try:
            with st.spinner("Загружаю граф из Neo4j..."):
                if graph_mode == "Legacy-связи":
                    rows = fetch_neo4j_relations(
                        uri=uri,
                        user=user,
                        password=password,
                        limit=limit,
                        contains=contains,
                        document_id=selected_doc_id,
                    )
                elif graph_mode == "Связи концептов":
                    rows = fetch_neo4j_concept_relations(
                        uri=uri,
                        user=user,
                        password=password,
                        limit=limit,
                        contains=contains,
                        document_id=selected_doc_id,
                    )
                elif graph_mode == "Контекстные мосты":
                    rows = fetch_neo4j_context_bridges(
                        uri=uri,
                        user=user,
                        password=password,
                        limit=limit,
                        contains=contains,
                    )
                elif graph_mode == "Одобренные/ручные связи":
                    rows = fetch_neo4j_approved_context_links(
                        uri=uri,
                        user=user,
                        password=password,
                        limit=limit,
                        contains=contains,
                    )
                elif graph_mode == "Триплеты":
                    rows = fetch_neo4j_triplets(
                        uri=uri,
                        user=user,
                        password=password,
                        limit=limit,
                        contains=contains,
                        document_id=selected_doc_id,
                    )
                elif graph_mode == "Frame-дерево":
                    frame_nodes, frame_edges = fetch_neo4j_frame_tree(
                        uri=uri,
                        user=user,
                        password=password,
                        triplet_id=selected_triplet_id or 1,
                        role=selected_role,
                    )
                else:
                    triplet_row = fetch_neo4j_triplet_by_id(
                        uri=uri,
                        user=user,
                        password=password,
                        triplet_id=selected_triplet_id or 1,
                    )
                    selected_roles = ["subject", "predicate", "object"] if selected_role == "all" else [selected_role]
                    mixed_frame_payload = [
                        (role_name, *fetch_neo4j_frame_tree(
                            uri=uri,
                            user=user,
                            password=password,
                            triplet_id=selected_triplet_id or 1,
                            role=role_name,
                        ))
                        for role_name in selected_roles
                    ]

            if graph_mode == "Frame-дерево":
                if not frame_nodes:
                    st.info("Для выбранных triplet_id/role frame-дерево не найдено.")
                    return
                st.write(
                    f"Загружено frame-дерево: triplet_id={selected_triplet_id}, "
                    f"role={selected_role}, узлов={len(frame_nodes)}, рёбер={len(frame_edges)}"
                )
                st.dataframe(pd.DataFrame(frame_nodes), use_container_width=True)
                render_graphviz_from_edges(
                    nodes=frame_nodes,
                    edges=frame_edges,
                    show_edge_labels=False,
                    max_label_len=max_label_len,
                    graph_direction="TB",
                )
                return

            if graph_mode == "Смешанный граф: триплет + frame":
                if triplet_row is None:
                    st.info("Триплет с выбранным triplet_id не найден.")
                    return

                mixed_nodes: list[dict[str, Any]] = []
                mixed_edges: list[dict[str, Any]] = []
                seen_nodes: set[str] = set()

                def add_mixed_node(node_id: str, label: str, kind: str) -> None:
                    if node_id in seen_nodes:
                        return
                    seen_nodes.add(node_id)
                    mixed_nodes.append({"id": node_id, "label": label, "kind": kind})

                triplet_node_id = f"neo_triplet::{triplet_row['triplet_id']}"
                add_mixed_node(triplet_node_id, f"Triplet {triplet_row['triplet_id']}", "triplet")

                for role_name, text, kind in (
                    ("subject", triplet_row["subject"], "entity"),
                    ("predicate", triplet_row["predicate"], "predicate"),
                    ("object", triplet_row["object"], "entity"),
                ):
                    role_node_id = f"neo_role::{triplet_row['triplet_id']}::{role_name}"
                    add_mixed_node(role_node_id, text or f"[empty {role_name}]", kind)
                    mixed_edges.append(
                        {
                            "source": triplet_node_id,
                            "target": role_node_id,
                            "label": role_name,
                            "color": "#B3261E",
                        }
                    )

                frame_group_count = 0
                for role_name, role_nodes, role_edges in mixed_frame_payload:
                    if not role_nodes:
                        continue
                    frame_group_count += 1
                    for node in role_nodes:
                        add_mixed_node(node["id"], node["label"], "frame")
                    mixed_edges.append(
                        {
                            "source": triplet_node_id,
                            "target": role_nodes[0]["id"],
                            "label": f"{role_name} frame",
                            "color": "#1A73E8",
                            "style": "dashed",
                        }
                    )
                    for edge in role_edges:
                        mixed_edges.append(
                            {
                                "source": edge["source"],
                                "target": edge["target"],
                                "label": edge["label"],
                                "color": "#1A73E8",
                            }
                        )

                st.write(
                    f"Загружен смешанный граф: triplet_id={triplet_row['triplet_id']}, "
                    f"frame-групп={frame_group_count}, узлов={len(mixed_nodes)}, рёбер={len(mixed_edges)}"
                )
                st.dataframe(pd.DataFrame([triplet_row]), use_container_width=True)
                render_graphviz_from_edges(
                    nodes=mixed_nodes,
                    edges=mixed_edges,
                    show_edge_labels=show_edge_labels,
                    max_label_len=max_label_len,
                    graph_direction="LR",
                )
                return

            if not rows:
                st.info("Для текущего фильтра строки не найдены.")
                return

            if graph_mode in ("Legacy-связи", "Связи концептов", "Контекстные мосты", "Одобренные/ручные связи"):
                if deduplicate_edges:
                    seen = set()
                    unique_rows = []
                    for row in rows:
                        key = (row["subject"], row["predicate"], row["object"])
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_rows.append(row)
                    rows = unique_rows

                st.write(f"Загружено связей: {len(rows)}")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                graph_nodes: list[dict[str, Any]] = []
                graph_edges: list[dict[str, Any]] = []
                seen_nodes: set[str] = set()
                for row in rows:
                    subj_id = f"neo::{row['subject']}"
                    obj_id = f"neo::{row['object']}"
                    if subj_id not in seen_nodes:
                        seen_nodes.add(subj_id)
                        graph_nodes.append({"id": subj_id, "label": row["subject"], "kind": "entity"})
                    if obj_id not in seen_nodes:
                        seen_nodes.add(obj_id)
                        graph_nodes.append({"id": obj_id, "label": row["object"], "kind": "entity"})
                    graph_edges.append(
                        {
                            "source": subj_id,
                            "target": obj_id,
                            "label": row["predicate"],
                            "color": "#0B8043" if graph_mode == "Контекстные мосты" else "#7B1FA2" if graph_mode == "Одобренные/ручные связи" else None,
                            "style": "dashed" if graph_mode in ("Контекстные мосты", "Одобренные/ручные связи") else None,
                        }
                    )
                render_graphviz_from_edges(
                    nodes=graph_nodes,
                    edges=graph_edges,
                    show_edge_labels=show_edge_labels,
                    max_label_len=max_label_len,
                    graph_direction="LR",
                )
                return

            st.write(f"Загружено триплетов: {len(rows)}")
            st.session_state["neo_view_triplet_rows"] = rows
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            triplet_option_labels = [format_triplet_option_label(row) for row in rows]
            selected_triplet_label = st.selectbox(
                "Открыть триплет в смешанном графе",
                options=triplet_option_labels,
                key="neo_view_triplets_open_select",
            )
            selected_triplet_idx = triplet_option_labels.index(selected_triplet_label)
            selected_triplet_for_open = rows[selected_triplet_idx]
            if st.button("Открыть выбранный в смешанном графе", key="neo_view_open_selected_mixed_btn"):
                st.session_state["neo_view_mode"] = "Смешанный граф: триплет + frame"
                st.session_state["neo_view_triplet_pick_mode"] = "Ввести triplet_id вручную"
                st.session_state["neo_view_triplet_id"] = int(selected_triplet_for_open["triplet_id"])
                st.session_state["neo_view_frame_role_mixed"] = "all"
                st.rerun()
            triplet_nodes: list[dict[str, Any]] = []
            triplet_edges: list[dict[str, Any]] = []
            seen_nodes: set[str] = set()
            for row in rows:
                triplet_id = row["triplet_id"]
                triplet_node_id = f"triplet::{triplet_id}"
                if triplet_node_id not in seen_nodes:
                    seen_nodes.add(triplet_node_id)
                    triplet_nodes.append(
                        {
                            "id": triplet_node_id,
                            "label": f"Triplet {triplet_id}",
                            "kind": "frame",
                        }
                    )
                for role_name, text, kind in (
                    ("subject", row["subject"], "entity"),
                    ("predicate", row["predicate"], "predicate"),
                    ("object", row["object"], "entity"),
                ):
                    node_id = f"{role_name}::{triplet_id}::{text}"
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        triplet_nodes.append({"id": node_id, "label": text or f"[empty {role_name}]", "kind": kind})
                    triplet_edges.append(
                        {
                            "source": triplet_node_id,
                            "target": node_id,
                            "label": role_name,
                        }
                    )
            render_graphviz_from_edges(
                nodes=triplet_nodes,
                edges=triplet_edges,
                show_edge_labels=show_edge_labels,
                max_label_len=max_label_len,
                graph_direction="LR",
            )
        except Exception as exc:
            st.error(f"Ошибка просмотра Neo4j: {exc}")
            st.code(traceback.format_exc(), language="text")


def render_graph_export_section() -> None:
    st.subheader("Выгрузить / загрузить граф")
    c1, c2, c3 = st.columns(3)
    with c1:
        uri = st.text_input(
            "URI Neo4j (выгрузка)",
            value=st.session_state.get("neo_uri", "bolt://localhost:7687"),
            key="neo_export_uri",
        )
    with c2:
        user = st.text_input(
            "Пользователь Neo4j (выгрузка)",
            value=st.session_state.get("neo_user", "neo4j"),
            key="neo_export_user",
        )
    with c3:
        password = st.text_input(
            "Пароль Neo4j (выгрузка)",
            value=st.session_state.get("neo_password", "neo4jpass"),
            type="password",
            key="neo_export_password",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        contains = st.text_input(
            "Фильтр по узлам/связям",
            value="",
            key="neo_export_filter",
            help="Пустой фильтр выгружает граф целиком в пределах заданных лимитов.",
        )
    with c5:
        node_limit = st.number_input(
            "Лимит вершин",
            min_value=1,
            max_value=1_000_000,
            value=50_000,
            step=1000,
            key="neo_export_node_limit",
        )
    with c6:
        relation_limit = st.number_input(
            "Лимит связей",
            min_value=1,
            max_value=2_000_000,
            value=100_000,
            step=1000,
            key="neo_export_relation_limit",
        )

    st.caption(
        "JSON сохраняет labels/properties для узлов и type/properties для связей. "
        "GraphML подходит для Gephi, yEd и NetworkX; свойства упакованы в properties_json."
    )

    export_tab, import_tab = st.tabs(["Выгрузка", "Загрузка"])

    with import_tab:
        st.caption(
            "Импорт создаёт отдельные ImportedGraphNode-узлы с import_namespace/import_id. "
            "Существующий граф не удаляется."
        )
        uploaded_graph = st.file_uploader(
            "Загрузите JSON/GraphML граф",
            type=["json", "graphml", "xml"],
            key="neo_graph_import_upload",
            accept_multiple_files=False,
        )
        import_namespace = st.text_input(
            "Namespace импорта",
            value="imported_graph",
            key="neo_graph_import_namespace",
            help="Одинаковый namespace обновляет ранее импортированные узлы/связи с теми же id.",
        )
        with st.expander("Ожидаемый формат JSON/GraphML", expanded=False):
            st.markdown(
                """
JSON:
```json
{
  "nodes": [
    {
      "id": "node-1",
      "labels": ["EntityConcept"],
      "properties": {"name": "угол", "norm": "угол"}
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "source": "node-1",
      "target": "node-2",
      "type": "RELATION_INSTANCE",
      "properties": {"predicate": "характеризует"}
    }
  ]
}
```

GraphML:
```xml
<graphml>
  <graph edgedefault="directed">
    <node id="n0">
      <data key="labels">EntityConcept</data>
      <data key="properties_json">{"name": "угол"}</data>
    </node>
    <edge id="e0" source="n0" target="n1">
      <data key="type">RELATION_INSTANCE</data>
      <data key="properties_json">{"predicate": "характеризует"}</data>
    </edge>
  </graph>
</graphml>
```
                """
            )
        if uploaded_graph is not None:
            try:
                parsed_graph = parse_graph_upload(uploaded_graph.name, uploaded_graph.getvalue())
                st.write(
                    f"Файл прочитан: вершин={len(parsed_graph.get('nodes', []))}, "
                    f"связей={len(parsed_graph.get('edges', []))}."
                )
                preview_nodes = parsed_graph.get("nodes", [])[:10]
                preview_edges = parsed_graph.get("edges", [])[:10]
                p1, p2 = st.tabs(["Пример вершин", "Пример связей"])
                with p1:
                    st.dataframe(pd.DataFrame(preview_nodes), use_container_width=True)
                with p2:
                    st.dataframe(pd.DataFrame(preview_edges), use_container_width=True)
                if st.button("Загрузить граф в Neo4j", type="primary", key="neo_graph_import_btn"):
                    with st.spinner("Создаю граф в Neo4j..."):
                        stats = import_graph_to_neo4j(
                            uri=uri,
                            user=user,
                            password=password,
                            graph_data=parsed_graph,
                            namespace=import_namespace,
                        )
                    st.success(
                        f"Импорт завершён: создано вершин={stats['created_nodes']}, "
                        f"обновлено вершин={stats['updated_nodes']}, "
                        f"создано связей={stats['created_edges']}, "
                        f"пропущено связей={stats['skipped_edges']}."
                    )
            except Exception as exc:
                st.error(f"Не удалось прочитать или загрузить граф: {exc}")
                st.code(traceback.format_exc(), language="text")

    with export_tab:
        if "neo_graph_export_data" not in st.session_state:
            st.session_state["neo_graph_export_data"] = None

        if st.button("Выгрузить граф из Neo4j", type="primary", key="neo_export_btn"):
            try:
                with st.spinner("Выгружаю граф из Neo4j..."):
                    st.session_state["neo_graph_export_data"] = fetch_neo4j_full_graph_export(
                        uri=uri,
                        user=user,
                        password=password,
                        node_limit=int(node_limit),
                        relation_limit=int(relation_limit),
                        contains=contains.strip(),
                    )
                st.success("Граф подготовлен к скачиванию.")
            except Exception as exc:
                st.error(f"Ошибка выгрузки графа: {exc}")
                st.code(traceback.format_exc(), language="text")
                return

        graph_data = st.session_state.get("neo_graph_export_data")
        if not graph_data:
            st.info("Нажмите кнопку выгрузки, чтобы подготовить JSON/GraphML.")
            return

        meta = graph_data.get("meta") or {}
        st.write(
            f"Выгружено: вершин={meta.get('node_count', 0)}, "
            f"связей={meta.get('edge_count', 0)}, "
            f"пропущено связей вне лимита вершин={meta.get('skipped_edges_outside_node_limit', 0)}."
        )
        if meta.get("skipped_edges_outside_node_limit", 0):
            st.warning(
                "Часть связей не попала в экспорт, потому что один из концов связи оказался вне лимита вершин. "
                "Увеличьте лимит вершин или сузьте фильтр."
            )

        sample_nodes = graph_data.get("nodes", [])[:20]
        sample_edges = graph_data.get("edges", [])[:20]
        preview_tab_nodes, preview_tab_edges = st.tabs(["Первые вершины", "Первые связи"])
        with preview_tab_nodes:
            if sample_nodes:
                st.dataframe(pd.DataFrame(sample_nodes), use_container_width=True)
            else:
                st.info("Вершины не найдены.")
        with preview_tab_edges:
            if sample_edges:
                st.dataframe(pd.DataFrame(sample_edges), use_container_width=True)
            else:
                st.info("Связи не найдены.")

        json_bytes = graph_export_to_json_bytes(graph_data)
        graphml_bytes = graph_export_to_graphml_bytes(graph_data)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Скачать JSON",
                data=json_bytes,
                file_name="neo4j_graph_export.json",
                mime="application/json",
                key="download_graph_json",
            )
        with d2:
            st.download_button(
                "Скачать GraphML",
                data=graphml_bytes,
                file_name="neo4j_graph_export.graphml",
                mime="application/graphml+xml",
                key="download_graph_graphml",
            )


def main() -> None:
    st.set_page_config(page_title="Triplet Extraction", layout="wide")
    st.title("Извлечение триплетов из CSV-таблиц")

    if "main_result" not in st.session_state:
        st.session_state["main_result"] = None
    if "processed_result" not in st.session_state:
        st.session_state["processed_result"] = None
    if "triplets_error" not in st.session_state:
        st.session_state["triplets_error"] = None
    if "main_source_name" not in st.session_state:
        st.session_state["main_source_name"] = None

    separator = st.radio(
        "Знак разделителя (для ллм по умолчанию описана точка с запятой из-за наличия текстов с запятыми)",
        options=["Запятая", "Точка с запятой"],
        index=0,
        key="separator_choice",
    )
    sep = "," if separator == "Запятая" else ";"

    uploaded = st.file_uploader(
        "Загрузите файл",
        type=["csv"],
        key="main_upload",
        accept_multiple_files=False,
    )

    uploaded_df = None
    if uploaded is not None:
        uploaded_df, uploaded_text = parse_uploaded_csv(uploaded.getvalue(), preferred_sep=sep)
        st.text(uploaded_text)

    run = st.button(
        "Извлечь триплеты",
        type="primary",
        disabled=uploaded is None,
        key="extract_triplets_btn",
    )

    if run and uploaded is not None:
        try:
            with st.spinner("Извлекаю триплеты..."):
                triplets_result = extract_triplets_by_llm(uploaded_df)
            triplets_result = change_to_frame_format(triplets_result)
            triplets_result = {"triplets": remove_me_from_triplets(triplets_result["triplets"])}
            print(triplets_result)
            st.session_state["triplets_error"] = None
            st.session_state["main_result"] = triplets_result
            st.session_state["processed_result"] = None
            st.session_state["main_source_name"] = uploaded.name
        except Exception as exc:
            st.session_state["main_result"] = None
            st.session_state["processed_result"] = None
            st.session_state["triplets_error"] = str(exc)

    if st.session_state["main_result"] is not None:
        st.subheader("Исходный результат")
        render_result(st.session_state["main_result"], download_button_id=1)
        render_payload_graph_section(
            title="Граф исходного результата",
            payload=st.session_state["main_result"],
            key_prefix="main_result",
        )

        mode_choice = st.radio(
            "Способ обработки предложений в триплетах",
            options=[
                "Извлечь общие SPO из всего текста (без учета текущей структуры S-P-O) каждого триплета",
                "Извлечь элементы отдельно с сохранением структуры (просто добавить frame)"
            ],
            index=0,
            key="mode_choice",
        )

        if mode_choice == "Извлечь общие SPO из всего текста (без учета текущей структуры S-P-O) каждого триплета":
            mode = "concat"
        else:
            mode = "separate"

        process_text = st.button("Обработать текст в триплетах", key="process_triplets_btn")

        if process_text:
            st.session_state["processed_result"] = process_triplets(st.session_state["main_result"], mode=mode)

        if st.session_state["processed_result"] is not None:
            st.subheader("Обработанный результат")
            render_result(st.session_state["processed_result"], download_button_id=2)
            render_payload_graph_section(
                title="Граф обработанного результата",
                payload=st.session_state["processed_result"],
                key_prefix="processed_result",
            )

    # tab_storage, tab_pg, tab_concepts, tab_analytics, tab_export, tab_neo = st.tabs(
    #     ["Сохранение", "PostgreSQL", "Концепты", "Аналитика", "Выгрузить граф", "Neo4j"]
    # )
    tab_storage, tab_pg, tab_analytics, tab_export, tab_neo = st.tabs(
        ["Сохранение", "PostgreSQL", "Аналитика", "Выгрузить граф", "Neo4j"]
    )
    with tab_storage:
        render_storage_section()
    with tab_pg:
        render_documents_list_section()
        render_pg_browser_section()
    # with tab_concepts:
    #     render_concept_aliases_section()
    with tab_analytics:
        render_graph_analytics_section()
    with tab_export:
        render_graph_export_section()
    with tab_neo:
        render_neo4j_graph_section()


if __name__ == "__main__":
    main()
