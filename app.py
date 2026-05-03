import hashlib
import io
import json
import traceback
from typing import Any
import re
import pandas as pd
import streamlit as st
from pandas.errors import ParserError
from llm_triplet_extraction import extract_triplets_by_llm
from triplets_from_text_extraction import process_triplets
from postprocess_triplets import remove_me_from_triplets

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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_triplets_document_id ON triplets(document_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject_text);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object_text);")


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

                relation_query = """
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

                with driver.session() as session:
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
                                    json.dumps(role_frame(triplet.get(role)), ensure_ascii=True),
                                ),
                            )

                        if subject and obj:
                            session.run(
                                relation_query,
                                {
                                    "triplet_id": triplet_id,
                                    "document_id": document_id,
                                    "subject_raw": subject,
                                    "subject_norm": subject.lower(),
                                    "object_raw": obj,
                                    "object_norm": obj.lower(),
                                    "predicate": predicate,
                                    "sentence": sentence,
                                },
                            )
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

    st.subheader("Save To PostgreSQL And Neo4j")
    source_choice = st.radio(
        "Result to save",
        options=["Processed", "Original"] if has_processed else ["Original"],
        index=0,
        key="save_source_choice",
    )

    payload = st.session_state["processed_result"] if source_choice == "Processed" else st.session_state["main_result"]
    default_stage = "postprocessed" if source_choice == "Processed" else "llm"

    col1, col2 = st.columns(2)
    with col1:
        source_name = st.text_input(
            "Source name (CSV)",
            value=st.session_state.get("main_source_name") or "uploaded.csv",
            key="db_source_name",
        )
        extraction_stage = st.selectbox(
            "Extraction stage",
            options=["llm", "postprocessed"],
            index=0 if default_stage == "llm" else 1,
            key="db_stage",
        )
        pg_host = st.text_input("PG host", value="127.0.0.1", key="pg_host")
        pg_port = st.text_input("PG port", value="5433", key="pg_port")
        pg_db = st.text_input("PG database", value="triplets", key="pg_db")
    with col2:
        pg_user = st.text_input("PG user", value="triplets_user", key="pg_user")
        pg_password = st.text_input("PG password", value="triplets_pass", type="password", key="pg_password")
        neo_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687", key="neo_uri")
        neo_user = st.text_input("Neo4j user", value="neo4j", key="neo_user")
        neo_password = st.text_input("Neo4j password", value="neo4jpass", type="password", key="neo_password")

    if st.button("Save To DB", type="primary", key="save_to_db_btn"):
        if payload is None:
            st.error("No data to save.")
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
            with st.spinner("Saving to PostgreSQL and Neo4j..."):
                stats = save_triplets_to_databases(
                    triplets_payload=payload,
                    source_name=source_name,
                    extraction_stage=extraction_stage,
                    pg_cfg=pg_cfg,
                    neo_cfg=neo_cfg,
                )
            st.success(
                f"Done. PostgreSQL: {stats['sql_loaded']} triplets, "
                f"Neo4j: {stats['graph_loaded']} relations, document_id={stats['document_id']}."
            )
        except Exception as exc:
            st.error(f"DB save error: {exc}")
            st.code(traceback.format_exc(), language="text")


def fetch_pg_documents(pg_cfg: dict[str, Any], limit: int = 200) -> list[dict[str, Any]]:
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


def delete_document_by_id(
    document_id: int,
    pg_cfg: dict[str, Any],
    delete_neo4j: bool = False,
    neo_cfg: dict[str, Any] | None = None,
) -> dict[str, int]:
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

    pg_deleted = 0
    neo_deleted = 0
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
                result = session.run(
                    """
                    MATCH ()-[r:RELATION]->()
                    WHERE r.document_id = $document_id
                    DELETE r
                    RETURN count(r) AS deleted_count
                    """,
                    {"document_id": int(document_id)},
                )
                record = result.single()
                neo_deleted = int(record["deleted_count"] if record else 0)

                # Remove orphaned entity nodes to keep graph clean.
                session.run(
                    """
                    MATCH (n:Entity)
                    WHERE NOT (n)--()
                    DELETE n
                    """
                )
        finally:
            driver.close()

    return {"pg_deleted_documents": pg_deleted, "neo_deleted_relations": neo_deleted}


def render_documents_list_section() -> None:
    st.subheader("Documents In PostgreSQL")
    c1, c2, c3 = st.columns(3)
    with c1:
        host = st.text_input("PG host (docs)", value=st.session_state.get("pg_host", "127.0.0.1"), key="pg_docs_host")
    with c2:
        port = st.text_input("PG port (docs)", value=st.session_state.get("pg_port", "5433"), key="pg_docs_port")
    with c3:
        dbname = st.text_input("PG database (docs)", value=st.session_state.get("pg_db", "triplets"), key="pg_docs_db")

    c4, c5, c6 = st.columns(3)
    with c4:
        user = st.text_input("PG user (docs)", value=st.session_state.get("pg_user", "triplets_user"), key="pg_docs_user")
    with c5:
        password = st.text_input(
            "PG password (docs)",
            value=st.session_state.get("pg_password", "triplets_pass"),
            type="password",
            key="pg_docs_password",
        )
    with c6:
        limit = st.slider("Max documents", min_value=10, max_value=500, value=100, step=10, key="pg_docs_limit")

    pg_cfg = {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password,
    }

    if "pg_docs_rows" not in st.session_state:
        st.session_state["pg_docs_rows"] = []

    if st.button("Load documents list", key="pg_docs_load_btn"):
        try:
            with st.spinner("Loading documents from PostgreSQL..."):
                rows = fetch_pg_documents(pg_cfg, limit=limit)
            st.session_state["pg_docs_rows"] = rows
            if not rows:
                st.info("No documents found in PostgreSQL.")
            else:
                st.success(f"Loaded documents: {len(rows)}")
        except Exception as exc:
            st.error(f"PostgreSQL documents view error: {exc}")
            st.code(traceback.format_exc(), language="text")

    rows = st.session_state["pg_docs_rows"]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        labels = [
            f"{row['document_id']} | {row['source_name']} | triplets={row['triplets_count']}"
            for row in rows
        ]
        selected_label = st.selectbox("Document to delete", options=labels, key="pg_docs_delete_select")
        selected_doc_id = int(selected_label.split("|", 1)[0].strip())

        c7, c8, c9 = st.columns(3)
        with c7:
            del_neo = st.checkbox("Also delete from Neo4j", value=True, key="pg_docs_delete_neo")
        with c8:
            neo_uri = st.text_input("Neo4j URI (delete)", value=st.session_state.get("neo_uri", "bolt://localhost:7687"), key="pg_docs_neo_uri")
        with c9:
            neo_user = st.text_input("Neo4j user (delete)", value=st.session_state.get("neo_user", "neo4j"), key="pg_docs_neo_user")
        neo_password = st.text_input(
            "Neo4j password (delete)",
            value=st.session_state.get("neo_password", "neo4jpass"),
            type="password",
            key="pg_docs_neo_password",
        )

        confirm = st.checkbox(
            f"Confirm delete document_id={selected_doc_id}",
            value=False,
            key="pg_docs_delete_confirm",
        )
        if st.button("Delete document", type="secondary", key="pg_docs_delete_btn"):
            if not confirm:
                st.warning("Please enable confirmation checkbox before deletion.")
            else:
                try:
                    with st.spinner(f"Deleting document_id={selected_doc_id}..."):
                        stats = delete_document_by_id(
                            document_id=selected_doc_id,
                            pg_cfg=pg_cfg,
                            delete_neo4j=del_neo,
                            neo_cfg={"uri": neo_uri, "user": neo_user, "password": neo_password},
                        )
                    st.success(
                        f"Deleted in PostgreSQL: {stats['pg_deleted_documents']} document(s), "
                        f"in Neo4j: {stats['neo_deleted_relations']} relation(s)."
                    )
                    # Refresh list after deletion.
                    st.session_state["pg_docs_rows"] = fetch_pg_documents(pg_cfg, limit=limit)
                except Exception as exc:
                    st.error(f"Delete error: {exc}")
                    st.code(traceback.format_exc(), language="text")


def escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\"", "\\\"")


def format_graph_label(value: str, max_len: int = 48) -> str:
    text = " ".join((value or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def fetch_neo4j_document_ids(uri: str, user: str, password: str) -> list[dict[str, int]]:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: neo4j. Install requirements-graph.txt") from exc

    query = """
    MATCH ()-[r:RELATION]->()
    WHERE r.document_id IS NOT NULL
    RETURN r.document_id AS document_id, count(*) AS rel_count
    ORDER BY document_id DESC
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(query)
            rows = []
            for record in result:
                rows.append(
                    {
                        "document_id": int(record["document_id"]),
                        "rel_count": int(record["rel_count"]),
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
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: neo4j. Install requirements-graph.txt") from exc

    query = """
    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
    WHERE ($contains = "" OR toLower(a.name) CONTAINS toLower($contains) OR toLower(b.name) CONTAINS toLower($contains))
      AND ($document_id IS NULL OR r.document_id = $document_id)
    RETURN a.name AS subject, r.predicate AS predicate, b.name AS object
    LIMIT $limit
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
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
                    }
                )
            return rows
    finally:
        driver.close()


def render_neo4j_graph_section() -> None:
    st.subheader("Neo4j Graph View")
    c1, c2, c3 = st.columns(3)
    with c1:
        uri = st.text_input("Neo4j URI (view)", value="bolt://localhost:7687", key="neo_view_uri")
    with c2:
        user = st.text_input("Neo4j user (view)", value="neo4j", key="neo_view_user")
    with c3:
        password = st.text_input("Neo4j password (view)", value="neo4jpass", type="password", key="neo_view_password")

    c4, c5 = st.columns(2)
    with c4:
        contains = st.text_input("Filter (subject/object contains)", value="", key="neo_view_filter")
    with c5:
        limit = st.slider("Max relations", min_value=10, max_value=300, value=100, step=10, key="neo_view_limit")

    c6, c7, c8 = st.columns(3)
    with c6:
        show_edge_labels = st.checkbox("Show edge labels", value=True, key="neo_view_show_edge_labels")
    with c7:
        deduplicate_edges = st.checkbox("Merge duplicate edges", value=True, key="neo_view_dedup")
    with c8:
        max_label_len = st.slider("Max label length", min_value=20, max_value=100, value=48, step=2, key="neo_view_max_label")

    graph_mode = st.radio(
        "Graph selection mode",
        options=["All relations", "By document_id"],
        index=0,
        key="neo_view_mode",
    )
    selected_doc_id: int | None = None
    if graph_mode == "By document_id":
        try:
            doc_rows = fetch_neo4j_document_ids(uri=uri, user=user, password=password)
            if not doc_rows:
                st.info("No document_id values found in Neo4j relations yet.")
                return

            labels = [f"{row['document_id']} (relations: {row['rel_count']})" for row in doc_rows]
            selected_label = st.selectbox("Choose document_id", options=labels, key="neo_view_doc_select")
            selected_doc_id = int(selected_label.split(" ", 1)[0])
        except Exception as exc:
            st.error(f"Cannot load document list: {exc}")
            st.code(traceback.format_exc(), language="text")
            return

    if st.button("Load graph from Neo4j", key="neo_view_load_btn"):
        try:
            with st.spinner("Loading graph from Neo4j..."):
                rows = fetch_neo4j_relations(
                    uri=uri,
                    user=user,
                    password=password,
                    limit=limit,
                    contains=contains,
                    document_id=selected_doc_id,
                )

            if not rows:
                st.info("No relations found for current filter.")
                return

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

            st.write(f"Loaded relations: {len(rows)}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            dot_lines = [
                "digraph G {",
                "rankdir=LR;",
                "layout=dot;",
                "overlap=false;",
                "splines=true;",
                "nodesep=0.5;",
                "ranksep=1.0;",
                "node [shape=box, style=\"rounded,filled\", fillcolor=\"#F5F7FA\", color=\"#D0D7DE\", fontname=\"Helvetica\", fontsize=10, margin=\"0.10,0.06\"];",
                "edge [color=\"#6B7280\", fontname=\"Helvetica\", fontsize=9, arrowsize=0.7];",
            ]
            for row in rows:
                subj = escape_dot(format_graph_label(row["subject"], max_len=max_label_len))
                obj = escape_dot(format_graph_label(row["object"], max_len=max_label_len))
                pred = escape_dot(format_graph_label(row["predicate"], max_len=max_label_len))
                if show_edge_labels:
                    dot_lines.append(f"\"{subj}\" -> \"{obj}\" [label=\"{pred}\"];")
                else:
                    dot_lines.append(f"\"{subj}\" -> \"{obj}\";")
            dot_lines.append("}")
            st.graphviz_chart("\n".join(dot_lines))
        except Exception as exc:
            st.error(f"Neo4j view error: {exc}")
            st.code(traceback.format_exc(), language="text")


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

    render_storage_section()
    render_documents_list_section()
    render_neo4j_graph_section()


if __name__ == "__main__":
    main()
