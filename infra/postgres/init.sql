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

CREATE INDEX IF NOT EXISTS idx_triplets_document_id ON triplets(document_id);
CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject_text);
CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object_text);
