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


def write_to_neo4j(driver, triplet_id: int, document_id: int, triplet: dict[str, Any]) -> None:
    subject_text = slot_to_text(triplet.get("subject"))
    predicate_text = slot_to_text(triplet.get("predicate"))
    object_text = slot_to_text(triplet.get("object"))

    if not subject_text or not object_text:
        return

    sentence_text = triplet.get("sentence", "")
    sentence_text = slot_to_text(sentence_text)

    query = """
    MERGE (s:Entity {name_norm: $subject_norm})
      ON CREATE SET s.name = $subject_raw
      ON MATCH SET s.name = coalesce(s.name, $subject_raw)
    MERGE (o:Entity {name_norm: $object_norm})
      ON CREATE SET o.name = $object_raw
      ON MATCH SET o.name = coalesce(o.name, $object_raw)
    MERGE (s)-[r:RELATION {triplet_id: $triplet_id}]->(o)
    SET r.predicate = $predicate,
        r.document_id = $document_id,
        r.sentence = $sentence;
    """
    params = {
        "triplet_id": triplet_id,
        "document_id": document_id,
        "subject_raw": subject_text,
        "object_raw": object_text,
        "subject_norm": safe_entity_name(subject_text),
        "object_norm": safe_entity_name(object_text),
        "predicate": predicate_text,
        "sentence": sentence_text,
    }

    with driver.session() as session:
        session.run(query, params)


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
                doc_id = upsert_document(cur, source_name, content_hash(data))

                for idx, triplet in enumerate(data["triplets"], start=1):
                    if not isinstance(triplet, dict):
                        continue

                    triplet_id = insert_triplet(cur, doc_id, idx, args.stage, triplet)
                    for role in ("subject", "predicate", "object"):
                        upsert_frame(cur, triplet_id, role, role_frame(triplet.get(role)))
                    loaded += 1

                    write_to_neo4j(neo4j_driver, triplet_id, doc_id, triplet)
                    graph_loaded += 1

        print(f"Loaded triplets into PostgreSQL: {loaded}")
        print(f"Loaded relations into Neo4j: {graph_loaded}")
        print(f"Document ID: {doc_id}")
    finally:
        pg_conn.close()
        neo4j_driver.close()


if __name__ == "__main__":
    main()
