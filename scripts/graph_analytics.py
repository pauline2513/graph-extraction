import csv
import hashlib
import itertools
import math
import os
import re
from collections import Counter
from functools import lru_cache
from importlib.util import find_spec
from typing import Any
import random

import networkx as nx


DEFAULT_ENCODER_MODEL = os.getenv("GRAPH_ENCODER_MODEL", "intfloat/multilingual-e5-small")
DEFAULT_ENCODER_BATCH_SIZE = max(int(os.getenv("GRAPH_ENCODER_BATCH_SIZE", "16")), 1)
DEFAULT_ENCODER_MAX_LENGTH = max(int(os.getenv("GRAPH_ENCODER_MAX_LENGTH", "256")), 32)
GLOBAL_ROOT_NORM = "__graph_root__"
GLOBAL_ROOT_NAME = "Global graph root"


def get_encoder_runtime_status() -> dict[str, Any]:
    missing: list[str] = []
    if find_spec("torch") is None:
        missing.append("torch")
    if find_spec("transformers") is None:
        missing.append("transformers")

    return {
        "available": not missing,
        "missing": missing,
        "reason": (
            None
            if not missing
            else "Для encoder-эмбеддингов не хватает пакетов: " + ", ".join(missing)
        ),
    }


def get_neo4j_driver(neo_cfg: dict[str, Any] | None = None):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Missing dependency: neo4j. Install the package from requirements.txt.") from exc

    cfg = neo_cfg or {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4jpass"),
    }
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split()).strip()


def canonical_match_text(value: Any) -> str:
    text = normalize_text(str(value or "").replace('"', "").replace("\ufeff", ""))
    text = text.replace("ё", "е")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return text


def canonical_contains(haystack: str, needle: str) -> bool:
    haystack = canonical_match_text(haystack)
    needle = canonical_match_text(needle)
    return bool(needle) and (needle == haystack or needle in haystack)


def detect_table_delimiter(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except csv.Error:
        counts = {delimiter: sample.count(delimiter) for delimiter in (";", ",", "\t")}
        return max(counts, key=counts.get) if sample else ";"


def read_table_csv(filepath: str, sep: str | None = None) -> list[list[str]]:
    delimiter = sep or detect_table_delimiter(filepath)
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter=delimiter, quotechar='"', skipinitialspace=True))


def fix_decimal_commas(row: list[str], expected_len: int) -> list[str]:
    row = list(row)
    while len(row) > expected_len:
        for i in range(1, len(row) - 1):
            if re.fullmatch(r"[+-]?\d+", row[i].strip()) and re.fullmatch(r"\d+", row[i + 1].strip()):
                row[i] = row[i].strip() + "," + row[i + 1].strip()
                del row[i + 1]
                break
        else:
            break
    return row


def table_cell_text(value: Any) -> str:
    return " ".join(str(value or "").replace('"', "").split()).strip()


def extract_ordinary_table_triplets(filepath: str) -> list[dict[str, Any]]:
    rows = [row for row in read_table_csv(filepath) if any(table_cell_text(cell) for cell in row)]
    if not rows:
        return []

    headers = [table_cell_text(cell) for cell in rows[0][1:]]
    expected_len = len(headers) + 1
    triplets: list[dict[str, Any]] = []
    for row_idx, raw_row in enumerate(rows[1:], start=2):
        row = fix_decimal_commas(raw_row, expected_len)
        subject = table_cell_text(row[0] if row else "")
        if not subject:
            continue
        values = row[1:]
        for header, value in zip(headers, values):
            predicate = table_cell_text(value)
            obj = table_cell_text(header)
            if not predicate and not obj:
                continue
            triplets.append(
                {
                    "table_file": os.path.basename(filepath),
                    "table_type": "ordinary",
                    "row_idx": row_idx,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                }
            )
    return triplets


def extract_name_value_horizontal_table_triplets(filepath: str) -> list[dict[str, Any]]:
    rows = [row for row in read_table_csv(filepath) if any(table_cell_text(cell) for cell in row)]
    if len(rows) < 2:
        return []

    headers = [table_cell_text(cell) for cell in rows[0]]
    values = [table_cell_text(cell) for cell in rows[1]]
    triplets: list[dict[str, Any]] = []
    for idx, subject in enumerate(headers):
        if not subject:
            continue
        predicate = values[idx] if idx < len(values) else ""
        triplets.append(
            {
                "table_file": os.path.basename(filepath),
                "table_type": "name_value_horizontal",
                "row_idx": 2,
                "subject": subject,
                "predicate": predicate,
                "object": "",
            }
        )
    return triplets


def extract_table_test_triplets(tables_dir: str) -> list[dict[str, Any]]:
    if not tables_dir or not os.path.isdir(tables_dir):
        return []

    triplets: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(tables_dir)):
        if not filename.lower().endswith(".csv"):
            continue
        filepath = os.path.join(tables_dir, filename)
        if "1.7" in filename:
            triplets.extend(extract_name_value_horizontal_table_triplets(filepath))
        else:
            triplets.extend(extract_ordinary_table_triplets(filepath))
    return triplets


def build_relation_match_index(rel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    for row in rel_rows:
        if row.get("triplet_id") is None:
            continue
        index_rows.append(
            {
                "source_keys": {
                    canonical_match_text(row.get("source_norm")),
                    canonical_match_text(row.get("source_name")),
                },
                "target_keys": {
                    canonical_match_text(row.get("target_norm")),
                    canonical_match_text(row.get("target_name")),
                },
                "predicate_keys": {
                    canonical_match_text(row.get("predicate_norm")),
                    canonical_match_text(row.get("predicate")),
                },
                "source_name": row.get("source_name") or row.get("source_norm") or "",
                "target_name": row.get("target_name") or row.get("target_norm") or "",
                "predicate": row.get("predicate") or row.get("predicate_norm") or "",
                "triplet_id": row.get("triplet_id"),
                "document_id": row.get("document_id"),
            }
        )
    return index_rows


def build_triplet_match_index(triplet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    for row in triplet_rows:
        frame_terms_by_role = row.get("frame_terms_by_role") or {}
        index_rows.append(
            {
                "subject_keys": {
                    canonical_match_text(row.get("subject")),
                    canonical_match_text(row.get("subject_norm")),
                },
                "predicate_keys": {
                    canonical_match_text(row.get("predicate")),
                    canonical_match_text(row.get("predicate_norm")),
                },
                "object_keys": {
                    canonical_match_text(row.get("object")),
                    canonical_match_text(row.get("object_norm")),
                },
                "frame_terms_by_role": {
                    role: {canonical_match_text(term) for term in terms if canonical_match_text(term)}
                    for role, terms in frame_terms_by_role.items()
                },
                "subject": row.get("subject") or "",
                "predicate": row.get("predicate") or "",
                "object": row.get("object") or "",
                "triplet_id": row.get("triplet_id"),
                "document_id": row.get("document_id"),
            }
        )
    return index_rows


def slot_matches_keys(value: Any, keys: set[str], allow_contains: bool = False) -> bool:
    needle = canonical_match_text(value)
    if not needle:
        return "" in keys
    if needle in keys:
        return True
    return allow_contains and any(canonical_contains(key, needle) or canonical_contains(needle, key) for key in keys)


def triplet_matches_direct(index_row: dict[str, Any], table_triplet: dict[str, Any]) -> bool:
    return (
        slot_matches_keys(table_triplet.get("subject"), index_row["subject_keys"])
        and slot_matches_keys(table_triplet.get("predicate"), index_row["predicate_keys"])
        and slot_matches_keys(table_triplet.get("object"), index_row["object_keys"])
    )


def triplet_matches_frame(index_row: dict[str, Any], table_triplet: dict[str, Any]) -> bool:
    return (
        slot_matches_keys(
            table_triplet.get("subject"),
            index_row["subject_keys"] | index_row["frame_terms_by_role"].get("subject", set()),
            allow_contains=True,
        )
        and slot_matches_keys(
            table_triplet.get("predicate"),
            index_row["predicate_keys"] | index_row["frame_terms_by_role"].get("predicate", set()),
            allow_contains=True,
        )
        and slot_matches_keys(
            table_triplet.get("object"),
            index_row["object_keys"] | index_row["frame_terms_by_role"].get("object", set()),
            allow_contains=True,
        )
    )


def relation_matches_table_triplet(index_row: dict[str, Any], table_triplet: dict[str, Any]) -> bool:
    subject = canonical_match_text(table_triplet.get("subject"))
    predicate = canonical_match_text(table_triplet.get("predicate"))
    obj = canonical_match_text(table_triplet.get("object"))
    if not subject or not predicate:
        return False
    if subject not in index_row["source_keys"]:
        return False
    if obj and obj not in index_row["target_keys"]:
        return False
    return predicate in index_row["predicate_keys"]


def evaluate_table_triplets_against_graph(
    rel_rows: list[dict[str, Any]],
    tables_dir: str,
    triplet_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    table_triplets = extract_table_test_triplets(tables_dir)
    triplet_index = build_triplet_match_index(triplet_rows or [])
    relation_index = build_relation_match_index(rel_rows)

    detail_rows: list[dict[str, Any]] = []
    true_positive = 0
    false_negative = 0

    for table_triplet in table_triplets:
        match_type = ""
        match = next((row for row in triplet_index if triplet_matches_direct(row, table_triplet)), None)
        if match:
            match_type = "direct_triplet"
        if not match:
            match = next((row for row in relation_index if relation_matches_table_triplet(row, table_triplet)), None)
            if match:
                match_type = "direct_relation"
        if not match:
            match = next((row for row in triplet_index if triplet_matches_frame(row, table_triplet)), None)
            if match:
                match_type = "frame_fallback"
        if match:
            true_positive += 1
            status = "TP"
        else:
            false_negative += 1
            status = "FN"
        detail_rows.append(
            {
                **table_triplet,
                "status": status,
                "match_type": match_type,
                "matched_triplet_id": match.get("triplet_id") if match else None,
                "matched_document_id": match.get("document_id") if match else None,
                "matched_subject": match.get("source_name") or match.get("subject") if match else "",
                "matched_predicate": match.get("predicate") if match else "",
                "matched_object": match.get("target_name") or match.get("object") if match else "",
            }
        )

    total = true_positive + false_negative
    recall = round(true_positive / total, 6) if total else 0.0
    return {
        "metrics": {
            "table_triplets_total": total,
            "table_triplets_tp": true_positive,
            "table_triplets_fn": false_negative,
            "table_triplets_recall": recall,
        },
        "table_triplet_rows": detail_rows,
        "tables_dir": tables_dir,
    }


def char_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> list[str]:
    clean = f" {normalize_text(text)} "
    if not clean.strip():
        return []
    grams: list[str] = []
    for n_size in range(n_min, n_max + 1):
        if len(clean) < n_size:
            continue
        grams.extend(clean[idx: idx + n_size] for idx in range(len(clean) - n_size + 1))
    return grams


def stable_hash_pair(token: str, dims: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dims, -1.0 if value & 1 else 1.0


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def cosine_dense(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def build_hashed_ngram_embeddings(texts: list[str], dims: int = 768) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dims
        for token, tf in Counter(char_ngrams(text)).items():
            idx, sign = stable_hash_pair(token, dims)
            vec[idx] += float(tf) * sign
        vectors.append(l2_normalize(vec))
    return vectors


def mean_pool_last_hidden_state(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_sum = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return masked_sum / counts


@lru_cache(maxsize=4)
def load_transformer_encoder(model_name: str, local_files_only: bool):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Encoder embeddings require the transformers package. "
            "Install it or switch to auto/fallback mode."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def adapt_text_for_encoder(text: str, model_name: str) -> str:
    if "e5" in (model_name or "").lower():
        return f"passage: {text}"
    return text


def encode_texts_with_transformer(
    texts: list[str],
    model_name: str,
    local_files_only: bool = False,
    batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
) -> dict[str, Any]:
    import torch

    tokenizer, model, device = load_transformer_encoder(model_name, local_files_only)
    prepared = [adapt_text_for_encoder(text, model_name) for text in texts]
    vectors: list[list[float]] = []

    for start in range(0, len(prepared), max(batch_size, 1)):
        batch = prepared[start: start + max(batch_size, 1)]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            pooled = mean_pool_last_hidden_state(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.extend(pooled.detach().cpu().tolist())

    return {
        "backend": "transformers_encoder",
        "model_name": model_name,
        "warning": None,
        "vectors": vectors,
    }


def embed_texts(
    texts: list[str],
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
) -> dict[str, Any]:
    backend = (embedding_backend or "auto").strip().lower()
    if backend not in {"auto", "encoder", "fallback"}:
        raise ValueError(f"Unknown embedding mode: {embedding_backend}")

    if backend != "fallback":
        try:
            return encode_texts_with_transformer(
                texts=texts,
                model_name=encoder_model_name,
                local_files_only=encoder_local_files_only,
                batch_size=encoder_batch_size,
                max_length=encoder_max_length,
            )
        except Exception as exc:
            if backend == "encoder":
                raise RuntimeError(
                    f"Failed to load encoder model '{encoder_model_name}': {exc}"
                ) from exc
            fallback_warning = (
                f"Encoder model '{encoder_model_name}' is unavailable, so a hashed char n-gram "
                f"fallback embedding was used. Reason: {exc}"
            )
        else:
            fallback_warning = None
    else:
        fallback_warning = "Fallback embedding mode is enabled without an encoder model."

    return {
        "backend": "hashed_char_ngrams_fallback",
        "model_name": "local-hash-fallback",
        "warning": fallback_warning,
        "vectors": build_hashed_ngram_embeddings(texts),
    }


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def compute_pagerank(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}

    nodes = list(graph.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    node_count = len(nodes)
    ranks = [1.0 / node_count] * node_count
    out_degrees = [graph.out_degree(node) for node in nodes]
    successors = [[node_to_idx[target] for target in graph.successors(node)] for node in nodes]
    base_score = (1.0 - alpha) / node_count

    for _ in range(max_iter):
        next_ranks = [base_score] * node_count
        dangling_share = alpha * sum(ranks[idx] for idx, degree in enumerate(out_degrees) if degree == 0) / node_count
        if dangling_share:
            next_ranks = [rank + dangling_share for rank in next_ranks]

        for src_idx, degree in enumerate(out_degrees):
            if degree == 0:
                continue
            weight = alpha * ranks[src_idx] / degree
            for dst_idx in successors[src_idx]:
                next_ranks[dst_idx] += weight

        norm = sum(next_ranks)
        if norm > 0:
            next_ranks = [rank / norm for rank in next_ranks]

        error = sum(abs(next_ranks[idx] - ranks[idx]) for idx in range(node_count))
        ranks = next_ranks
        if error < tol:
            break

    return {node: float(ranks[idx]) for node, idx in node_to_idx.items()}


def compute_graph_fragility(graph: nx.DiGraph, pagerank: dict[str, float], limit: int = 50) -> dict[str, Any]:
    """Estimate how easily the graph breaks when important vertices or edges disappear."""
    if graph.number_of_nodes() == 0:
        return {
            "metrics": {
                "fragility_score": 0.0,
                "articulation_points_count": 0,
                "articulation_points_share": 0.0,
                "bridge_edges_count_structural": 0,
                "bridge_edges_share_structural": 0.0,
                "max_node_removal_component_gain": 0,
                "max_node_removal_largest_component_loss": 0.0,
            },
            "fragility_rows": [],
        }

    undirected = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        undirected.add_node(node, **attrs)
    undirected.add_edges_from(graph.to_undirected().edges())

    base_components = nx.number_connected_components(undirected)
    base_largest = max((len(component) for component in nx.connected_components(undirected)), default=0)
    node_count = undirected.number_of_nodes()
    edge_count = undirected.number_of_edges()

    articulation_points = set(nx.articulation_points(undirected)) if node_count > 1 else set()
    bridge_edges = list(nx.bridges(undirected)) if node_count > 1 else []

    fragility_rows: list[dict[str, Any]] = []
    for node in undirected.nodes:
        reduced = undirected.copy()
        reduced.remove_node(node)
        components_after = nx.number_connected_components(reduced) if reduced.number_of_nodes() else 0
        largest_after = max((len(component) for component in nx.connected_components(reduced)), default=0)
        component_gain = max(components_after - base_components, 0)
        largest_loss = (base_largest - largest_after) / base_largest if base_largest else 0.0
        total_degree = int(graph.in_degree(node) + graph.out_degree(node))
        impact_score = (
            0.45 * min(component_gain / max(base_components, 1), 1.0)
            + 0.35 * largest_loss
            + 0.20 * min(total_degree / max(node_count - 1, 1), 1.0)
        )
        fragility_rows.append(
            {
                "concept_norm": node,
                "concept_name": graph.nodes[node].get("label", node),
                "is_articulation_point": node in articulation_points,
                "component_gain_if_removed": component_gain,
                "largest_component_loss_if_removed": round(largest_loss, 6),
                "total_degree": total_degree,
                "pagerank": round(float(pagerank.get(node, 0.0)), 6),
                "fragility_impact_score": round(impact_score, 6),
            }
        )

    fragility_rows.sort(
        key=lambda item: (
            -item["fragility_impact_score"],
            -int(item["is_articulation_point"]),
            -item["component_gain_if_removed"],
            -item["largest_component_loss_if_removed"],
            -item["total_degree"],
            item["concept_name"],
        )
    )

    articulation_share = len(articulation_points) / node_count if node_count else 0.0
    bridge_share = len(bridge_edges) / edge_count if edge_count else 0.0
    max_component_gain = max((row["component_gain_if_removed"] for row in fragility_rows), default=0)
    max_largest_loss = max((row["largest_component_loss_if_removed"] for row in fragility_rows), default=0.0)
    top_impact = max((row["fragility_impact_score"] for row in fragility_rows), default=0.0)
    fragility_score = 0.35 * articulation_share + 0.25 * bridge_share + 0.40 * top_impact

    return {
        "metrics": {
            "fragility_score": round(fragility_score, 6),
            "articulation_points_count": len(articulation_points),
            "articulation_points_share": round(articulation_share, 6),
            "bridge_edges_count_structural": len(bridge_edges),
            "bridge_edges_share_structural": round(bridge_share, 6),
            "max_node_removal_component_gain": int(max_component_gain),
            "max_node_removal_largest_component_loss": round(float(max_largest_loss), 6),
        },
        "fragility_rows": fragility_rows[:limit],
    }


def fetch_concept_relations(driver) -> list[dict[str, Any]]:
    query = """
    CALL {
      MATCH (s:EntityConcept)-[r:RELATION_INSTANCE]->(o:EntityConcept)
      RETURN
        s.norm AS source_norm,
        coalesce(s.name, s.norm) AS source_name,
        o.norm AS target_norm,
        coalesce(o.name, o.norm) AS target_name,
        coalesce(r.predicate_norm, r.predicate, '') AS predicate_norm,
        coalesce(r.predicate, '') AS predicate,
        r.triplet_id AS triplet_id,
        r.document_id AS document_id
      UNION ALL
      MATCH (s:EntityConcept)-[r]->(o:EntityConcept)
      WHERE type(r) IN [
        'APPROVED_ALIAS_OF',
        'APPROVED_BROADER_THAN',
        'APPROVED_CONTEXT_LINK',
        'MANUAL_CONTEXT_LINK',
        'MANUAL_COMPONENT_LINK'
      ]
      RETURN
        s.norm AS source_norm,
        coalesce(s.name, s.norm) AS source_name,
        o.norm AS target_norm,
        coalesce(o.name, o.norm) AS target_name,
        type(r) AS predicate_norm,
        coalesce(r.relation_text, type(r)) AS predicate,
        null AS triplet_id,
        null AS document_id
    }
    RETURN source_norm, source_name, target_norm, target_name, predicate_norm, predicate, triplet_id, document_id
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]


def fetch_triplet_search_rows(driver) -> list[dict[str, Any]]:
    query = """
    MATCH (t:Triplet)
    OPTIONAL MATCH (t)-[:SUBJECT]->(subj)
    OPTIONAL MATCH (t)-[:PREDICATE]->(pred)
    OPTIONAL MATCH (t)-[:OBJECT]->(obj)
    OPTIONAL MATCH (t)-[:HAS_FRAME]->(occ:FrameOccurrence)-[:HAS_ROOT]->(root:FrameNode)
    OPTIONAL MATCH (root)-[:CHILD*0..]->(node:FrameNode)
    WITH
      t,
      subj,
      pred,
      obj,
      occ.role AS role,
      collect(DISTINCT coalesce(node.text, node.norm, '')) +
      collect(DISTINCT coalesce(node.norm, node.text, '')) AS frame_terms
    WITH
      t,
      subj,
      pred,
      obj,
      collect({role: role, terms: [term IN frame_terms WHERE term <> '']}) AS role_frames
    RETURN
      t.triplet_id AS triplet_id,
      t.document_id AS document_id,
      coalesce(t.subject_text, subj.name, subj.norm, '') AS subject,
      coalesce(subj.norm, '') AS subject_norm,
      coalesce(t.predicate_text, pred.name, pred.norm, '') AS predicate,
      coalesce(pred.norm, '') AS predicate_norm,
      coalesce(t.object_text, obj.name, obj.norm, '') AS object,
      coalesce(obj.norm, '') AS object_norm,
      role_frames
    """
    with driver.session() as session:
        rows: list[dict[str, Any]] = []
        for record in session.run(query):
            frame_terms_by_role: dict[str, list[str]] = {"subject": [], "predicate": [], "object": []}
            for role_frame in record["role_frames"] or []:
                role = role_frame.get("role")
                if role in frame_terms_by_role:
                    frame_terms_by_role[role].extend(role_frame.get("terms") or [])
            rows.append(
                {
                    "triplet_id": record["triplet_id"],
                    "document_id": record["document_id"],
                    "subject": record["subject"] or "",
                    "subject_norm": record["subject_norm"] or "",
                    "predicate": record["predicate"] or "",
                    "predicate_norm": record["predicate_norm"] or "",
                    "object": record["object"] or "",
                    "object_norm": record["object_norm"] or "",
                    "frame_terms_by_role": frame_terms_by_role,
                }
            )
        return rows


def fetch_concept_contexts(driver) -> list[dict[str, Any]]:
    query = """
    MATCH (c:EntityConcept)
    OPTIONAL MATCH (c)-[out:RELATION_INSTANCE]->()
    WITH c, collect(DISTINCT coalesce(out.predicate_norm, out.predicate, '')) AS out_preds
    OPTIONAL MATCH ()-[inn:RELATION_INSTANCE]->(c)
    WITH c, out_preds, collect(DISTINCT coalesce(inn.predicate_norm, inn.predicate, '')) AS in_preds
    OPTIONAL MATCH (occ:FrameOccurrence)-[:OF_CONCEPT]->(c)
    OPTIONAL MATCH (occ)-[:HAS_ROOT]->(:FrameNode)-[:CHILD]->(child:FrameNode)
    WITH
      c,
      [item IN out_preds WHERE item <> ''] AS out_preds,
      [item IN in_preds WHERE item <> ''] AS in_preds,
      collect(DISTINCT coalesce(child.norm, '')) AS child_terms,
      count(DISTINCT occ) AS occ_count
    RETURN
      c.norm AS concept_norm,
      coalesce(c.name, c.norm) AS concept_name,
      out_preds,
      in_preds,
      [item IN child_terms WHERE item <> ''] AS child_terms,
      occ_count
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]


def fetch_legacy_entity_contexts(driver) -> list[dict[str, Any]]:
    query = """
    MATCH (e:Entity)
    OPTIONAL MATCH (e)-[out:RELATION]->(out_target:Entity)
    WITH
      e,
      collect(DISTINCT coalesce(out.predicate, '')) AS out_preds,
      collect(DISTINCT coalesce(out_target.name, '')) AS out_neighbors
    OPTIONAL MATCH (in_source:Entity)-[inn:RELATION]->(e)
    WITH
      e,
      [item IN out_preds WHERE item <> ''] AS out_preds,
      [item IN out_neighbors WHERE item <> ''] AS out_neighbors,
      collect(DISTINCT coalesce(inn.predicate, '')) AS in_preds,
      collect(DISTINCT coalesce(in_source.name, '')) AS in_neighbors
    RETURN
      e.name_norm AS entity_norm,
      coalesce(e.name, e.name_norm) AS entity_name,
      [item IN in_preds WHERE item <> ''] AS in_preds,
      [item IN in_neighbors WHERE item <> ''] AS in_neighbors,
      out_preds,
      out_neighbors
    """
    with driver.session() as session:
        result = session.run(query)
        return [dict(record) for record in result]


def build_graph_metrics(rel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    graph = nx.DiGraph()
    node_documents: dict[str, set[int]] = {}
    for row in rel_rows:
        graph.add_node(row["source_norm"], label=row["source_name"])
        graph.add_node(row["target_norm"], label=row["target_name"])
        graph.add_edge(
            row["source_norm"],
            row["target_norm"],
            predicate=row["predicate"],
            predicate_norm=row["predicate_norm"],
            triplet_id=row["triplet_id"],
            document_id=row["document_id"],
        )
        if row.get("document_id") is not None:
            doc_id = int(row["document_id"])
            node_documents.setdefault(row["source_norm"], set()).add(doc_id)
            node_documents.setdefault(row["target_norm"], set()).add(doc_id)

    pagerank = compute_pagerank(graph)
    fragility_data = compute_graph_fragility(graph, pagerank)
    degree_rows = []
    for node in graph.nodes:
        label = graph.nodes[node].get("label", node)
        in_degree = int(graph.in_degree(node))
        out_degree = int(graph.out_degree(node))
        degree_rows.append(
            {
                "concept_norm": node,
                "concept_name": label,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "pagerank": round(float(pagerank.get(node, 0.0)), 6),
            }
        )
    degree_rows.sort(key=lambda item: (-item["total_degree"], -item["pagerank"], item["concept_name"]))

    is_cyclic = not nx.is_directed_acyclic_graph(graph) if graph.number_of_nodes() else False
    cycle_rows = []
    if is_cyclic:
        for cycle in itertools.islice(nx.simple_cycles(graph), 20):
            cycle_rows.append(
                {
                    "cycle_length": len(cycle),
                    "cycle_path": " -> ".join(graph.nodes[node].get("label", node) for node in cycle),
                }
            )

    component_rows = []
    component_roots = []
    component_by_norm: dict[str, int] = {}
    for comp_idx, component in enumerate(nx.weakly_connected_components(graph), start=1):
        comp_graph = graph.subgraph(component).copy()
        for node in component:
            component_by_norm[node] = comp_idx
        zero_in = [node for node in comp_graph.nodes if comp_graph.in_degree(node) == 0]
        candidate_nodes = zero_in or list(comp_graph.nodes)

        def reachable_count(node: str) -> int:
            return len(nx.descendants(comp_graph, node))

        root_node = sorted(
            candidate_nodes,
            key=lambda node: (
                -reachable_count(node),
                -float(pagerank.get(node, 0.0)),
                int(comp_graph.in_degree(node)),
                -int(comp_graph.out_degree(node)),
                graph.nodes[node].get("label", node),
            ),
        )[0]
        root_label = graph.nodes[root_node].get("label", root_node)
        component_row = {
            "component_id": comp_idx,
            "nodes_count": comp_graph.number_of_nodes(),
            "edges_count": comp_graph.number_of_edges(),
            "root_norm": root_node,
            "root_name": root_label,
            "zero_in_degree_nodes": len(zero_in),
        }
        component_rows.append(component_row)
        component_roots.append(component_row)

    if not component_roots:
        global_root = None
    elif len(component_roots) == 1:
        global_root = {
            "root_type": "concept",
            "root_norm": component_roots[0]["root_norm"],
            "root_name": component_roots[0]["root_name"],
            "components_count": 1,
        }
    else:
        global_root = {
            "root_type": "virtual",
            "root_norm": GLOBAL_ROOT_NORM,
            "root_name": GLOBAL_ROOT_NAME,
            "components_count": len(component_roots),
            "children": [row["root_name"] for row in component_roots],
        }

    metrics = {
        "nodes_count": graph.number_of_nodes(),
        "edges_count": graph.number_of_edges(),
        "components_count": nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0,
        "is_cyclic": is_cyclic,
        "cycles_found": len(cycle_rows),
        "max_in_degree": max((row["in_degree"] for row in degree_rows), default=0),
        "max_out_degree": max((row["out_degree"] for row in degree_rows), default=0),
        "avg_in_degree": round(sum(row["in_degree"] for row in degree_rows) / len(degree_rows), 4) if degree_rows else 0.0,
        "avg_out_degree": round(sum(row["out_degree"] for row in degree_rows) / len(degree_rows), 4) if degree_rows else 0.0,
        **fragility_data["metrics"],
    }

    return {
        "graph": graph,
        "metrics": metrics,
        "degree_rows": degree_rows,
        "fragility_rows": fragility_data["fragility_rows"],
        "cycle_rows": cycle_rows,
        "component_rows": component_rows,
        "global_root": global_root,
        "component_by_norm": component_by_norm,
        "node_documents": node_documents,
    }


def compose_context_text(row: dict[str, Any], include_frame_context: bool = True) -> str:
    name = row.get("concept_name") or row.get("concept_norm") or ""
    out_preds = ", ".join(sorted(set(row.get("out_preds", [])))) or "none"
    in_preds = ", ".join(sorted(set(row.get("in_preds", [])))) or "none"
    occ_count = max(int(row.get("occ_count") or 0), 0)
    parts = [
        f"concept: {name}",
        f"outgoing predicates: {out_preds}",
        f"incoming predicates: {in_preds}",
        f"occurrence count: {occ_count}",
    ]
    if include_frame_context:
        child_terms = ", ".join(sorted(set(row.get("child_terms", [])))) or "none"
        parts.insert(3, f"frame children: {child_terms}")
    return "\n".join(parts)


def tokenize_entity_name(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[0-9A-Za-zА-Яа-яЁё-]+", normalize_text(text)) if len(token) > 1]


def compose_legacy_context_text(row: dict[str, Any]) -> str:
    name = row.get("entity_name") or row.get("entity_norm") or ""
    out_preds = ", ".join(sorted(set(row.get("out_preds", [])))) or "none"
    in_preds = ", ".join(sorted(set(row.get("in_preds", [])))) or "none"
    out_neighbors = ", ".join(sorted(set(row.get("out_neighbors", [])))) or "none"
    in_neighbors = ", ".join(sorted(set(row.get("in_neighbors", [])))) or "none"
    return "\n".join(
        [
            f"entity: {name}",
            f"outgoing predicates: {out_preds}",
            f"incoming predicates: {in_preds}",
            f"outgoing neighbors: {out_neighbors}",
            f"incoming neighbors: {in_neighbors}",
        ]
    )


def prepare_legacy_embeddings(
    legacy_rows: list[dict[str, Any]],
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
) -> tuple[list[list[float]], dict[str, Any]]:
    if not legacy_rows:
        return [], {"backend": "none", "model_name": None, "warning": None}

    texts = [compose_legacy_context_text(row) for row in legacy_rows]
    embedding_data = embed_texts(
        texts=texts,
        embedding_backend=embedding_backend,
        encoder_model_name=encoder_model_name,
        encoder_local_files_only=encoder_local_files_only,
        encoder_batch_size=encoder_batch_size,
        encoder_max_length=encoder_max_length,
    )
    return embedding_data["vectors"], {
        "backend": embedding_data["backend"],
        "model_name": embedding_data["model_name"],
        "warning": embedding_data["warning"],
    }


def score_legacy_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    left_vector: list[float],
    right_vector: list[float],
) -> dict[str, Any]:
    left_predicates = set(left.get("out_preds", [])) | set(left.get("in_preds", []))
    right_predicates = set(right.get("out_preds", [])) | set(right.get("in_preds", []))
    left_neighbors = set(left.get("out_neighbors", [])) | set(left.get("in_neighbors", []))
    right_neighbors = set(right.get("out_neighbors", [])) | set(right.get("in_neighbors", []))
    left_tokens = set(tokenize_entity_name(left.get("entity_name", "")))
    right_tokens = set(tokenize_entity_name(right.get("entity_name", "")))

    predicate_overlap = jaccard_similarity(left_predicates, right_predicates)
    neighbor_overlap = jaccard_similarity(left_neighbors, right_neighbors)
    token_overlap = jaccard_similarity(left_tokens, right_tokens)
    lexical_core = sorted(left_tokens & right_tokens)
    structural_score = 0.45 * predicate_overlap + 0.35 * neighbor_overlap + 0.20 * token_overlap
    embedding_score = cosine_dense(left_vector, right_vector)
    combined_score = 0.60 * structural_score + 0.40 * embedding_score

    return {
        "predicate_overlap": predicate_overlap,
        "neighbor_overlap": neighbor_overlap,
        "token_overlap": token_overlap,
        "lexical_core": lexical_core,
        "structural_score": structural_score,
        "embedding_score": embedding_score,
        "combined_score": combined_score,
    }


def infer_cluster_root_name(cluster_rows: list[dict[str, Any]]) -> str:
    token_sets = [set(tokenize_entity_name(row.get("entity_name", ""))) for row in cluster_rows]
    token_sets = [tokens for tokens in token_sets if tokens]
    if not token_sets:
        return ""
    shared_tokens = set.intersection(*token_sets) if len(token_sets) > 1 else set(token_sets[0])
    if not shared_tokens:
        return ""

    shortest_name = min(
        (row.get("entity_name", "") for row in cluster_rows if row.get("entity_name")),
        key=lambda value: (len(value.split()), len(value), value),
        default="",
    )
    ordered_tokens = tokenize_entity_name(shortest_name)
    root_tokens = [token for token in ordered_tokens if token in shared_tokens]
    if not root_tokens:
        root_tokens = sorted(shared_tokens, key=lambda token: (len(token), token))
    return " ".join(root_tokens).strip()


def rank_legacy_context_roots(
    legacy_rows: list[dict[str, Any]],
    min_group_score: float = 0.45,
    min_structural_score: float = 0.15,
    limit: int = 200,
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors, embedding_info = prepare_legacy_embeddings(
        legacy_rows=legacy_rows,
        embedding_backend=embedding_backend,
        encoder_model_name=encoder_model_name,
        encoder_local_files_only=encoder_local_files_only,
        encoder_batch_size=encoder_batch_size,
        encoder_max_length=encoder_max_length,
    )
    if not legacy_rows:
        return [], embedding_info

    cluster_graph = nx.Graph()
    for row in legacy_rows:
        cluster_graph.add_node(row["entity_norm"])

    row_by_norm = {row["entity_norm"]: row for row in legacy_rows}
    pair_scores: list[dict[str, Any]] = []
    for left_idx in range(len(legacy_rows)):
        left = legacy_rows[left_idx]
        for right_idx in range(left_idx + 1, len(legacy_rows)):
            right = legacy_rows[right_idx]
            if left["entity_norm"] == right["entity_norm"]:
                continue
            score = score_legacy_pair(left, right, vectors[left_idx], vectors[right_idx])
            if not score["lexical_core"]:
                continue
            if score["combined_score"] < min_group_score or score["structural_score"] < min_structural_score:
                continue
            cluster_graph.add_edge(left["entity_norm"], right["entity_norm"])
            pair_scores.append(
                {
                    "left_norm": left["entity_norm"],
                    "right_norm": right["entity_norm"],
                    "combined_score": score["combined_score"],
                }
            )

    pair_score_map = {
        tuple(sorted((item["left_norm"], item["right_norm"]))): float(item["combined_score"])
        for item in pair_scores
    }
    root_rows: list[dict[str, Any]] = []
    for component in nx.connected_components(cluster_graph):
        if len(component) < 2:
            continue
        cluster_rows = [row_by_norm[norm] for norm in component if norm in row_by_norm]
        root_name = infer_cluster_root_name(cluster_rows)
        if not root_name:
            continue
        root_norm = normalize_text(root_name)
        component_scores = [
            score for pair, score in pair_score_map.items()
            if pair[0] in component and pair[1] in component
        ]
        cluster_score = sum(component_scores) / len(component_scores) if component_scores else 0.0
        example_pair = sorted(cluster_rows, key=lambda row: row["entity_name"])
        example_triplet_id = None
        for row in cluster_rows:
            if row.get("example_triplet_id") is not None:
                example_triplet_id = row["example_triplet_id"]
                break
        for row in cluster_rows:
            if row["entity_norm"] == root_norm:
                continue
            root_rows.append(
                {
                    "child_norm": row["entity_norm"],
                    "child_name": row["entity_name"],
                    "root_norm": root_norm,
                    "root_name": root_name,
                    "cluster_size": len(cluster_rows),
                    "cluster_score": round(cluster_score, 6),
                    "shared_examples": ", ".join(item["entity_name"] for item in example_pair[:5]),
                    "example_triplet_id": example_triplet_id,
                }
            )

    root_rows.sort(
        key=lambda item: (-item["cluster_score"], -item["cluster_size"], item["root_name"], item["child_name"])
    )
    return root_rows[:limit], embedding_info

def attach_context_metadata(
    context_rows: list[dict[str, Any]],
    component_by_norm: dict[str, int],
    node_documents: dict[str, set[int]],
) -> list[dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    for row in context_rows:
        concept_norm = row["concept_norm"]
        document_ids = sorted(node_documents.get(concept_norm, set()))
        enriched = dict(row)
        enriched["component_id"] = component_by_norm.get(concept_norm)
        enriched["document_ids"] = document_ids
        enriched["document_count"] = len(document_ids)
        enriched_rows.append(enriched)
    return enriched_rows


def prepare_context_embeddings(
    context_rows: list[dict[str, Any]],
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
    include_frame_context: bool = True,
) -> tuple[list[list[float]], dict[str, Any]]:
    if not context_rows:
        return [], {"backend": "none", "model_name": None, "warning": None}

    texts = [compose_context_text(row, include_frame_context=include_frame_context) for row in context_rows]
    embedding_data = embed_texts(
        texts=texts,
        embedding_backend=embedding_backend,
        encoder_model_name=encoder_model_name,
        encoder_local_files_only=encoder_local_files_only,
        encoder_batch_size=encoder_batch_size,
        encoder_max_length=encoder_max_length,
    )
    return embedding_data["vectors"], {
        "backend": embedding_data["backend"],
        "model_name": embedding_data["model_name"],
        "warning": embedding_data["warning"],
    }


def score_context_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    left_vector: list[float],
    right_vector: list[float],
    include_frame_context: bool = True,
) -> dict[str, Any]:
    left_predicates = set(left.get("out_preds", [])) | set(left.get("in_preds", []))
    right_predicates = set(right.get("out_preds", [])) | set(right.get("in_preds", []))
    left_children = set(left.get("child_terms", []))
    right_children = set(right.get("child_terms", []))
    left_occ = max(int(left.get("occ_count") or 0), 1)
    right_occ = max(int(right.get("occ_count") or 0), 1)
    left_docs = set(left.get("document_ids", []))
    right_docs = set(right.get("document_ids", []))

    predicate_overlap = jaccard_similarity(left_predicates, right_predicates)
    child_overlap = jaccard_similarity(left_children, right_children) if include_frame_context else 0.0
    occ_balance = min(left_occ, right_occ) / max(left_occ, right_occ)
    if include_frame_context:
        structural_score = 0.45 * predicate_overlap + 0.45 * child_overlap + 0.10 * occ_balance
    else:
        structural_score = 0.85 * predicate_overlap + 0.15 * occ_balance
    embedding_score = cosine_dense(left_vector, right_vector)
    combined_score = 0.55 * structural_score + 0.45 * embedding_score
    doc_overlap = jaccard_similarity({str(item) for item in left_docs}, {str(item) for item in right_docs})
    doc_separation = 1.0 - doc_overlap if left_docs or right_docs else 0.0
    bridge_score = 0.40 * structural_score + 0.40 * embedding_score + 0.20 * doc_separation

    shared_predicates = sorted(left_predicates & right_predicates)
    shared_children = sorted(left_children & right_children)
    shared_documents = sorted(left_docs & right_docs)
    support = len(shared_predicates) + (len(shared_children) if include_frame_context else 0)

    return {
        "support": support,
        "shared_predicates": shared_predicates,
        "shared_children": shared_children,
        "shared_documents": shared_documents,
        "structural_score": structural_score,
        "embedding_score": embedding_score,
        "combined_score": combined_score,
        "bridge_score": bridge_score,
        "left_component_id": left.get("component_id"),
        "right_component_id": right.get("component_id"),
        "left_document_ids": sorted(left_docs),
        "right_document_ids": sorted(right_docs),
        "doc_separation": doc_separation,
    }


def rank_synonym_candidates(
    context_rows: list[dict[str, Any]],
    min_combined_score: float = 0.45,
    min_structural_score: float = 0.10,
    limit: int = 200,
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
    embedding_vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    include_frame_context: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if embedding_vectors is None or embedding_info is None:
        embedding_vectors, embedding_info = prepare_context_embeddings(
            context_rows=context_rows,
            embedding_backend=embedding_backend,
            encoder_model_name=encoder_model_name,
            encoder_local_files_only=encoder_local_files_only,
            encoder_batch_size=encoder_batch_size,
            encoder_max_length=encoder_max_length,
            include_frame_context=include_frame_context,
        )
    if not context_rows:
        return [], embedding_info

    results: list[dict[str, Any]] = []

    for left_idx in range(len(context_rows)):
        left = context_rows[left_idx]
        for right_idx in range(left_idx + 1, len(context_rows)):
            right = context_rows[right_idx]
            if left["concept_norm"] == right["concept_norm"]:
                continue

            score = score_context_pair(
                left,
                right,
                embedding_vectors[left_idx],
                embedding_vectors[right_idx],
                include_frame_context=include_frame_context,
            )

            if score["combined_score"] < min_combined_score or score["structural_score"] < min_structural_score:
                continue

            results.append(
                {
                    "left_norm": left["concept_norm"],
                    "left_name": left["concept_name"],
                    "right_norm": right["concept_norm"],
                    "right_name": right["concept_name"],
                    "support": score["support"],
                    "shared_predicates": ", ".join(score["shared_predicates"][:10]),
                    "shared_children": ", ".join(score["shared_children"][:10]),
                    "structural_score": round(score["structural_score"], 6),
                    "embedding_score": round(score["embedding_score"], 6),
                    "combined_score": round(score["combined_score"], 6),
                }
            )

    results.sort(
        key=lambda item: (
            -item["combined_score"],
            -item["structural_score"],
            -item["embedding_score"],
            -item["support"],
            item["left_name"],
            item["right_name"],
        )
    )
    return results[:limit], embedding_info


def rank_context_bridges(
    context_rows: list[dict[str, Any]],
    min_bridge_score: float = 0.35,
    min_structural_score: float = 0.10,
    limit: int = 200,
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
    embedding_vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    include_frame_context: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if embedding_vectors is None or embedding_info is None:
        embedding_vectors, embedding_info = prepare_context_embeddings(
            context_rows=context_rows,
            embedding_backend=embedding_backend,
            encoder_model_name=encoder_model_name,
            encoder_local_files_only=encoder_local_files_only,
            encoder_batch_size=encoder_batch_size,
            encoder_max_length=encoder_max_length,
            include_frame_context=include_frame_context,
        )
    if not context_rows:
        return [], embedding_info

    results: list[dict[str, Any]] = []
    for left_idx in range(len(context_rows)):
        left = context_rows[left_idx]
        if left.get("component_id") is None:
            continue
        for right_idx in range(left_idx + 1, len(context_rows)):
            right = context_rows[right_idx]
            if right.get("component_id") is None:
                continue
            if left["concept_norm"] == right["concept_norm"]:
                continue
            if left["component_id"] == right["component_id"]:
                continue

            score = score_context_pair(
                left,
                right,
                embedding_vectors[left_idx],
                embedding_vectors[right_idx],
                include_frame_context=include_frame_context,
            )
            if score["bridge_score"] < min_bridge_score or score["structural_score"] < min_structural_score:
                continue

            results.append(
                {
                    "left_norm": left["concept_norm"],
                    "left_name": left["concept_name"],
                    "right_norm": right["concept_norm"],
                    "right_name": right["concept_name"],
                    "left_component_id": score["left_component_id"],
                    "right_component_id": score["right_component_id"],
                    "left_document_ids": ", ".join(str(item) for item in score["left_document_ids"][:10]),
                    "right_document_ids": ", ".join(str(item) for item in score["right_document_ids"][:10]),
                    "shared_documents": ", ".join(str(item) for item in score["shared_documents"][:10]),
                    "support": score["support"],
                    "shared_predicates": ", ".join(score["shared_predicates"][:10]),
                    "shared_children": ", ".join(score["shared_children"][:10]),
                    "doc_separation": round(score["doc_separation"], 6),
                    "structural_score": round(score["structural_score"], 6),
                    "embedding_score": round(score["embedding_score"], 6),
                    "combined_score": round(score["combined_score"], 6),
                    "bridge_score": round(score["bridge_score"], 6),
                }
            )

    results.sort(
        key=lambda item: (
            -item["bridge_score"],
            -item["combined_score"],
            -item["structural_score"],
            -item["embedding_score"],
            -item["support"],
            item["left_name"],
            item["right_name"],
        )
    )
    return results[:limit], embedding_info


def project_components_with_bridges(graph: nx.DiGraph, bridge_rows: list[dict[str, Any]]) -> dict[str, int]:
    projected = graph.to_undirected().copy()
    before_count = nx.number_connected_components(projected) if projected.number_of_nodes() else 0
    for row in bridge_rows:
        projected.add_edge(row["left_norm"], row["right_norm"])
    after_count = nx.number_connected_components(projected) if projected.number_of_nodes() else 0
    return {
        "components_before_bridges": before_count,
        "components_after_bridges": after_count,
        "bridge_edges_count": len(bridge_rows),
        "components_merged_by_bridges": max(before_count - after_count, 0),
    }


def sync_graph_root(driver, global_root: dict[str, Any] | None, component_rows: list[dict[str, Any]]) -> None:
    with driver.session() as session:
        session.run(
            f"""
            MERGE (root:GraphRoot {{norm: '{GLOBAL_ROOT_NORM}'}})
            SET root.name = $root_name;
            """,
            {"root_name": GLOBAL_ROOT_NAME},
        )
        session.run(
            f"""
            MATCH (root:GraphRoot {{norm: '{GLOBAL_ROOT_NORM}'}})-[rel:HAS_COMPONENT_ROOT]->()
            DELETE rel;
            """
        )
        if not global_root:
            return

        if global_root["root_type"] == "concept":
            session.run(
                f"""
                MATCH (root:GraphRoot {{norm: '{GLOBAL_ROOT_NORM}'}})
                MATCH (concept:EntityConcept {{norm: $concept_norm}})
                MERGE (root)-[:HAS_COMPONENT_ROOT {{score: 1.0}}]->(concept);
                """,
                {"concept_norm": global_root["root_norm"]},
            )
            return

        for row in component_rows:
            session.run(
                f"""
                MATCH (root:GraphRoot {{norm: '{GLOBAL_ROOT_NORM}'}})
                MATCH (concept:EntityConcept {{norm: $concept_norm}})
                MERGE (root)-[:HAS_COMPONENT_ROOT {{component_id: $component_id}}]->(concept);
                """,
                {
                    "concept_norm": row["root_norm"],
                    "component_id": int(row["component_id"]),
                },
            )


def sync_synonym_candidate_edges(driver, synonym_rows: list[dict[str, Any]]) -> None:
    with driver.session() as session:
        session.run("MATCH ()-[rel:SYNONYM_CANDIDATE]->() DELETE rel;")
        for row in synonym_rows:
            session.run(
                """
                MATCH (left:EntityConcept {norm: $left_norm})
                MATCH (right:EntityConcept {norm: $right_norm})
                MERGE (left)-[rel:SYNONYM_CANDIDATE]->(right)
                SET
                    rel.structural_score = $structural_score,
                    rel.embedding_score = $embedding_score,
                    rel.combined_score = $combined_score,
                    rel.support = $support,
                    rel.shared_predicates = $shared_predicates,
                    rel.shared_children = $shared_children;
                """,
                row,
            )


def sync_context_bridge_edges(driver, bridge_rows: list[dict[str, Any]]) -> None:
    with driver.session() as session:
        session.run("MATCH ()-[rel:CONTEXT_BRIDGE]->() DELETE rel;")
        for row in bridge_rows:
            session.run(
                """
                MATCH (left:EntityConcept {norm: $left_norm})
                MATCH (right:EntityConcept {norm: $right_norm})
                MERGE (left)-[rel:CONTEXT_BRIDGE]->(right)
                SET
                    rel.bridge_score = $bridge_score,
                    rel.structural_score = $structural_score,
                    rel.embedding_score = $embedding_score,
                    rel.combined_score = $combined_score,
                    rel.doc_separation = $doc_separation,
                    rel.support = $support,
                    rel.left_component_id = $left_component_id,
                    rel.right_component_id = $right_component_id,
                    rel.left_document_ids = $left_document_ids,
                    rel.right_document_ids = $right_document_ids,
                    rel.shared_documents = $shared_documents,
                    rel.shared_predicates = $shared_predicates,
                    rel.shared_children = $shared_children;
                """,
                row,
            )


def sync_legacy_context_parent_edges(driver, root_rows: list[dict[str, Any]]) -> None:
    with driver.session() as session:
        session.run("MATCH ()-[rel:CONTEXT_PARENT]->() DELETE rel;")
        for row in root_rows:
            session.run(
                """
                MATCH (child:Entity {name_norm: $child_norm})
                MERGE (root:Entity {name_norm: $root_norm})
                ON CREATE SET root.name = $root_name
                ON MATCH SET root.name = coalesce(root.name, $root_name)
                MERGE (child)-[rel:CONTEXT_PARENT]->(root)
                SET
                    rel.cluster_score = $cluster_score,
                    rel.cluster_size = $cluster_size,
                    rel.shared_examples = $shared_examples,
                    rel.example_triplet_id = $example_triplet_id;
                """,
                row,
            )


def analyze_graph(
    neo_cfg: dict[str, Any] | None = None,
    min_combined_score: float = 0.45,
    min_structural_score: float = 0.10,
    synonym_limit: int = 200,
    min_bridge_score: float = 0.35,
    bridge_limit: int = 200,
    min_legacy_root_score: float = 0.45,
    legacy_root_limit: int = 200,
    sync_synonym_edges: bool = True,
    sync_context_bridges: bool = True,
    sync_legacy_roots: bool = True,
    sync_graph_root_node: bool = True,
    embedding_backend: str = "auto",
    encoder_model_name: str = DEFAULT_ENCODER_MODEL,
    encoder_local_files_only: bool = False,
    encoder_batch_size: int = DEFAULT_ENCODER_BATCH_SIZE,
    encoder_max_length: int = DEFAULT_ENCODER_MAX_LENGTH,
    include_frame_context: bool = True,
    table_tests_dir: str | None = None,
    include_table_tests: bool = True,
) -> dict[str, Any]:
    driver = get_neo4j_driver(neo_cfg)
    try:
        rel_rows = fetch_concept_relations(driver)
        triplet_rows = fetch_triplet_search_rows(driver)
        graph_data = build_graph_metrics(rel_rows)
        table_test_data = {"metrics": {}, "table_triplet_rows": [], "tables_dir": table_tests_dir}
        if include_table_tests:
            resolved_tables_dir = table_tests_dir or os.path.join(os.getcwd(), "tables_for_test")
            table_test_data = evaluate_table_triplets_against_graph(rel_rows, resolved_tables_dir, triplet_rows)
        context_rows = attach_context_metadata(
            context_rows=fetch_concept_contexts(driver),
            component_by_norm=graph_data["component_by_norm"],
            node_documents=graph_data["node_documents"],
        )
        embedding_vectors, embedding_info = prepare_context_embeddings(
            context_rows=context_rows,
            embedding_backend=embedding_backend,
            encoder_model_name=encoder_model_name,
            encoder_local_files_only=encoder_local_files_only,
            encoder_batch_size=encoder_batch_size,
            encoder_max_length=encoder_max_length,
            include_frame_context=include_frame_context,
        )
        synonym_rows, embedding_info = rank_synonym_candidates(
            context_rows=context_rows,
            min_combined_score=min_combined_score,
            min_structural_score=min_structural_score,
            limit=synonym_limit,
            embedding_backend="fallback",
            embedding_vectors=embedding_vectors,
            embedding_info=embedding_info,
            include_frame_context=include_frame_context,
        )
        bridge_rows, _ = rank_context_bridges(
            context_rows=context_rows,
            min_bridge_score=min_bridge_score,
            min_structural_score=min_structural_score,
            limit=bridge_limit,
            embedding_backend="fallback",
            embedding_vectors=embedding_vectors,
            embedding_info=embedding_info,
            include_frame_context=include_frame_context,
        )
        bridge_metrics = project_components_with_bridges(graph_data["graph"], bridge_rows)
        legacy_root_rows, legacy_embedding_info = rank_legacy_context_roots(
            legacy_rows=fetch_legacy_entity_contexts(driver),
            min_group_score=min_legacy_root_score,
            min_structural_score=min_structural_score,
            limit=legacy_root_limit,
            embedding_backend=embedding_backend,
            encoder_model_name=encoder_model_name,
            encoder_local_files_only=encoder_local_files_only,
            encoder_batch_size=encoder_batch_size,
            encoder_max_length=encoder_max_length,
        )

        if sync_synonym_edges:
            sync_synonym_candidate_edges(driver, synonym_rows)
        if sync_context_bridges:
            sync_context_bridge_edges(driver, bridge_rows)
        if sync_legacy_roots:
            sync_legacy_context_parent_edges(driver, legacy_root_rows)
        if sync_graph_root_node:
            sync_graph_root(driver, graph_data["global_root"], graph_data["component_rows"])

        return {
            "metrics": {**graph_data["metrics"], **bridge_metrics, **table_test_data["metrics"]},
            "degree_rows": graph_data["degree_rows"],
            "fragility_rows": graph_data["fragility_rows"],
            "cycle_rows": graph_data["cycle_rows"],
            "component_rows": graph_data["component_rows"],
            "global_root": graph_data["global_root"],
            "synonym_rows": synonym_rows,
            "bridge_rows": bridge_rows,
            "legacy_root_rows": legacy_root_rows,
            "table_triplet_rows": table_test_data["table_triplet_rows"],
            "table_tests_dir": table_test_data["tables_dir"],
            "embedding_info": embedding_info,
            "legacy_embedding_info": legacy_embedding_info,
            "include_frame_context": include_frame_context,
        }
    finally:
        driver.close()
