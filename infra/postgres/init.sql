CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS triplet_frames (
    id BIGSERIAL PRIMARY KEY,
    triplet_id BIGINT NOT NULL REFERENCES triplets(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('subject', 'predicate', 'object')),
    frame_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(triplet_id, role)
);

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

CREATE TABLE IF NOT EXISTS concepts (
    id BIGSERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    canonical_norm TEXT NOT NULL UNIQUE,
    concept_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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

CREATE INDEX IF NOT EXISTS idx_triplets_document_id ON triplets(document_id);
CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject_text);
CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object_text);
CREATE INDEX IF NOT EXISTS idx_frame_nodes_norm ON frame_nodes(node_norm);
CREATE INDEX IF NOT EXISTS idx_frame_nodes_lemma ON frame_nodes(node_lemma);
CREATE INDEX IF NOT EXISTS idx_frame_nodes_parent ON frame_nodes(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_frame_nodes_instance ON frame_nodes(frame_instance_id);
CREATE INDEX IF NOT EXISTS idx_frame_nodes_path ON frame_nodes(path);
CREATE INDEX IF NOT EXISTS idx_concept_aliases_norm ON concept_aliases(alias_norm);
