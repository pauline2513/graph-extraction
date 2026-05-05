import argparse
import os
from collections import defaultdict
from typing import Any

QUERY = """
MATCH (p1:FrameNode)-[:CHILD]->(n1:FrameNode)
MATCH (p2:FrameNode)-[:CHILD]->(n2:FrameNode)
WHERE n1.norm <> n2.norm
  AND p1.norm = p2.norm
  AND n1.depth = n2.depth
WITH
  p1.norm AS parent_norm,
  n1.norm AS norm_a,
  n2.norm AS norm_b,
  n1.text AS text_a,
  n2.text AS text_b,
  count(*) AS support
WHERE support >= $min_support
RETURN parent_norm, norm_a, norm_b, text_a, text_b, support
ORDER BY support DESC
LIMIT $limit
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine alias candidates from Neo4j frame context and store them in PostgreSQL."
    )
    parser.add_argument("--min-support", type=int, default=3, help="Minimum support for a candidate pair.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence to persist a candidate alias.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of Neo4j candidate rows to inspect.")
    return parser.parse_args()


def get_pg_conn(pg_cfg: dict[str, Any] | None = None):
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("Missing dependency: psycopg2-binary. Install requirements-graph.txt") from exc

    cfg = pg_cfg or {
        "host": os.getenv("PGHOST", "localhost"),
        "port": int(os.getenv("PGPORT", "5433")),
        "dbname": os.getenv("PGDATABASE", "triplets"),
        "user": os.getenv("PGUSER", "triplets_user"),
        "password": os.getenv("PGPASSWORD", "triplets_pass"),
    }
    return psycopg2.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
    )


def get_neo4j_driver(neo_cfg: dict[str, Any] | None = None):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: neo4j. Install requirements-graph.txt") from exc

    cfg = neo_cfg or {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4jpass"),
    }
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def ensure_candidate_schema(cur) -> None:
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concept_aliases_norm ON concept_aliases(alias_norm);")
    cur.execute("ALTER TABLE concept_aliases ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'candidate';")
    cur.execute("ALTER TABLE concept_aliases ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;")
    cur.execute("ALTER TABLE concept_aliases ADD COLUMN IF NOT EXISTS review_note TEXT;")


def norm_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def confidence_from_support(support: int) -> float:
    if support >= 10:
        return 0.9
    if support >= 7:
        return 0.8
    if support >= 5:
        return 0.7
    return 0.6


def ensure_concept(cur, canonical_name: str, canonical_norm: str) -> int:
    cur.execute(
        """
        INSERT INTO concepts(canonical_name, canonical_norm, concept_type)
        VALUES (%s, %s, %s)
        ON CONFLICT (canonical_norm)
        DO UPDATE SET
            canonical_name = EXCLUDED.canonical_name,
            concept_type = EXCLUDED.concept_type
        RETURNING id;
        """,
        (canonical_name, canonical_norm, "candidate"),
    )
    return cur.fetchone()[0]


def upsert_alias(cur, concept_id: int, alias_text: str, alias_norm: str, confidence: float) -> None:
    cur.execute(
        """
        INSERT INTO concept_aliases(concept_id, alias_text, alias_norm, source, confidence)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (concept_id, alias_norm)
        DO UPDATE SET
            alias_text = EXCLUDED.alias_text,
            source = EXCLUDED.source,
            confidence = GREATEST(COALESCE(concept_aliases.confidence, 0), COALESCE(EXCLUDED.confidence, 0));
        """,
        (concept_id, alias_text, alias_norm, "neo4j_candidate", confidence),
    )


def choose_canonical_variant(norm_a: str, text_a: str, norm_b: str, text_b: str) -> tuple[str, str, str, str]:
    if len(norm_a) < len(norm_b):
        return norm_a, text_a, norm_b, text_b
    if len(norm_b) < len(norm_a):
        return norm_b, text_b, norm_a, text_a
    if norm_a <= norm_b:
        return norm_a, text_a, norm_b, text_b
    return norm_b, text_b, norm_a, text_a


def mine_alias_candidates(
    pg_cfg: dict[str, Any] | None = None,
    neo_cfg: dict[str, Any] | None = None,
    min_support: int = 3,
    min_confidence: float = 0.55,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    driver = get_neo4j_driver(neo_cfg)
    pg_conn = get_pg_conn(pg_cfg)

    try:
        with driver.session() as session:
            rows = list(session.run(QUERY, {"min_support": int(min_support), "limit": int(limit)}))

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = norm_pair(row["norm_a"], row["norm_b"])
            grouped[key].append(
                {
                    "parent_norm": row["parent_norm"],
                    "norm_a": row["norm_a"],
                    "norm_b": row["norm_b"],
                    "text_a": row["text_a"],
                    "text_b": row["text_b"],
                    "support": int(row["support"]),
                }
            )

        saved_rows: list[dict[str, Any]] = []
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_candidate_schema(cur)
                for pair_rows in grouped.values():
                    support = max(item["support"] for item in pair_rows)
                    confidence = confidence_from_support(support)
                    if confidence < float(min_confidence):
                        continue

                    sample = pair_rows[0]
                    canonical_norm, canonical_name, alias_norm, alias_text = choose_canonical_variant(
                        sample["norm_a"],
                        sample["text_a"],
                        sample["norm_b"],
                        sample["text_b"],
                    )
                    concept_id = ensure_concept(cur, canonical_name, canonical_norm)
                    upsert_alias(cur, concept_id, alias_text, alias_norm, confidence)

                    saved_rows.append(
                        {
                            "concept_id": concept_id,
                            "canonical_name": canonical_name,
                            "canonical_norm": canonical_norm,
                            "alias_text": alias_text,
                            "alias_norm": alias_norm,
                            "support": support,
                            "confidence": confidence,
                            "parent_examples": ", ".join(sorted({item["parent_norm"] for item in pair_rows})[:5]),
                        }
                    )

        saved_rows.sort(key=lambda item: (-item["support"], item["canonical_name"], item["alias_text"]))
        return saved_rows
    finally:
        pg_conn.close()
        driver.close()


def main() -> None:
    args = parse_args()
    results = mine_alias_candidates(
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        limit=args.limit,
    )
    print(f"Saved alias candidates: {len(results)}")
    for row in results[:50]:
        print(
            f"{row['canonical_name']} <- {row['alias_text']} "
            f"(support={row['support']}, confidence={row['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
