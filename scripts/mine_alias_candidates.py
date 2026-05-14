import argparse
import os
from itertools import combinations
from typing import Any

from scripts.graph_analytics import (
    DEFAULT_ENCODER_BATCH_SIZE,
    DEFAULT_ENCODER_MAX_LENGTH,
    DEFAULT_ENCODER_MODEL,
    cosine_dense,
    embed_texts,
)

QUERY = """
MATCH (c:EntityConcept)
WHERE coalesce(c.norm, '') <> ''
OPTIONAL MATCH (occ:FrameOccurrence)-[:OF_CONCEPT]->(c)
WITH c, count(DISTINCT occ) AS occurrence_count
RETURN
  c.norm AS norm,
  coalesce(c.name, c.norm) AS name,
  occurrence_count
ORDER BY occurrence_count DESC, name
LIMIT $limit
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine alias candidates from Neo4j EntityConcept vertex embeddings and store them in PostgreSQL."
    )
    parser.add_argument("--min-support", type=int, default=1, help="Minimum EntityConcept occurrence count.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.82,
        help="Minimum embedding cosine similarity to persist a candidate alias.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of Neo4j candidate rows to inspect.")
    parser.add_argument("--embedding-backend", choices=["auto", "encoder", "fallback"], default="auto")
    parser.add_argument("--encoder-model-name", default=DEFAULT_ENCODER_MODEL)
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
        (concept_id, alias_text, alias_norm, "neo4j_vertex_embedding", confidence),
    )


def choose_canonical_variant(norm_a: str, text_a: str, norm_b: str, text_b: str) -> tuple[str, str, str, str]:
    if len(norm_a) < len(norm_b):
        return norm_a, text_a, norm_b, text_b
    if len(norm_b) < len(norm_a):
        return norm_b, text_b, norm_a, text_a
    if norm_a <= norm_b:
        return norm_a, text_a, norm_b, text_b
    return norm_b, text_b, norm_a, text_a


def vertex_embedding_text(row: dict[str, Any]) -> str:
    return (row.get("name") or row.get("norm") or "").strip()


def mine_alias_candidates(
    pg_cfg: dict[str, Any] | None = None,
    neo_cfg: dict[str, Any] | None = None,
    min_support: int = 1,
    min_confidence: float = 0.82,
    limit: int = 1000,
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
) -> list[dict[str, Any]]:
    driver = get_neo4j_driver(neo_cfg)
    pg_conn = get_pg_conn(pg_cfg)

    try:
        with driver.session() as session:
            rows = [
                {
                    "norm": record["norm"] or "",
                    "name": record["name"] or record["norm"] or "",
                    "occurrence_count": int(record["occurrence_count"] or 0),
                }
                for record in session.run(QUERY, {"limit": int(limit)})
            ]

        rows = [
            row
            for row in rows
            if row["norm"]
            and vertex_embedding_text(row)
            and max(int(row.get("occurrence_count") or 0), 1) >= int(min_support)
        ]
        if not rows:
            return []
        embedding_data = embed_texts(
            texts=[vertex_embedding_text(row) for row in rows],
            embedding_backend=embedding_backend,
            encoder_model_name=encoder_model_name,
            encoder_local_files_only=encoder_local_files_only,
            encoder_batch_size=encoder_batch_size,
            encoder_max_length=encoder_max_length,
        )
        vectors = embedding_data["vectors"]
        candidate_pairs: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for left_idx, right_idx in combinations(range(len(rows)), 2):
            left = rows[left_idx]
            right = rows[right_idx]
            if left["norm"] == right["norm"]:
                continue
            pair_key = norm_pair(left["norm"], right["norm"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            similarity = cosine_dense(vectors[left_idx], vectors[right_idx])
            if similarity < float(min_confidence):
                continue
            candidate_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "similarity": similarity,
                    "support": max(left["occurrence_count"], right["occurrence_count"], 1),
                }
            )

        candidate_pairs.sort(
            key=lambda item: (
                -float(item["similarity"]),
                -int(item["support"]),
                item["left"]["name"],
                item["right"]["name"],
            )
        )

        saved_rows: list[dict[str, Any]] = []
        with pg_conn:
            with pg_conn.cursor() as cur:
                ensure_candidate_schema(cur)
                for pair in candidate_pairs[: int(limit)]:
                    left = pair["left"]
                    right = pair["right"]
                    confidence = float(pair["similarity"])
                    canonical_norm, canonical_name, alias_norm, alias_text = choose_canonical_variant(
                        left["norm"],
                        left["name"],
                        right["norm"],
                        right["name"],
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
                            "support": int(pair["support"]),
                            "confidence": confidence,
                            "parent_examples": (
                                f"vertex embedding: {embedding_data['backend']}; "
                                f"model={embedding_data['model_name']}"
                            ),
                        }
                    )

        saved_rows.sort(key=lambda item: (-item["confidence"], -item["support"], item["canonical_name"], item["alias_text"]))
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
        embedding_backend=args.embedding_backend,
        encoder_model_name=args.encoder_model_name,
    )
    print(f"Saved alias candidates: {len(results)}")
    for row in results[:50]:
        print(
            f"{row['canonical_name']} <- {row['alias_text']} "
            f"(support={row['support']}, confidence={row['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
