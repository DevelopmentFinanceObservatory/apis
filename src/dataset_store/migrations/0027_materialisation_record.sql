-- One row per materialisation run: which release, catalogue, coverage and
-- graph generation produced a version, what the run depended on, and what it
-- selected. A version keeps many rows, one per run that produced it.
--
-- `dependencies` is the run's dependency fingerprint
-- (`materialise.dependencies.Dependencies.as_json`). `selection` holds the
-- run's `trace.facets` and `trace.mint_facets`, and `conversion` its
-- `trace.conversion` entries, which carry the rate and deflator table row
-- digests, or NULL for an unconverted run. `graph_generation` is the graph
-- service's serving revision at the run; it moves on every graph restart, so
-- no freshness comparison reads it.
--
-- `subject_head` gains `materialisation_id`, the run that produced the head's
-- version, and that run's `dependencies`. `LiveHeadStore.advance_head` inserts
-- the run's row and moves the head in one transaction, so a head names the run
-- that produced it. `subject_head.materialised_at` (0009) already holds the
-- run's instant.
--
-- `signal` gets a default so a writer that stops writing it still satisfies
-- NOT NULL. An empty member list matches no derived signal, so a reader that
-- still compares signals rebuilds such a head once.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES covers the new table and its
-- identity sequence, so this migration needs no GRANT statement of its own.

CREATE TABLE datasets.materialisation (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subject_id text NOT NULL,
    content_version text NOT NULL,
    materialised_at timestamptz NOT NULL,
    "trigger" text NOT NULL
        CHECK ("trigger" IN ('mint', 'get', 'options', 'refresh', 'worker')),
    materialiser_build text NOT NULL,
    frame_grammar_version text NOT NULL,
    code_index_stamp text NOT NULL,
    origin_catalogue_build_id text NOT NULL,
    coverage_stamp text,
    graph_generation text,
    dependencies jsonb NOT NULL,
    selection jsonb NOT NULL,
    conversion jsonb,
    trace_id text,
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);

CREATE INDEX materialisation_subject_version_idx
    ON datasets.materialisation (subject_id, content_version, materialised_at DESC);

ALTER TABLE datasets.subject_head
    ADD COLUMN materialisation_id bigint REFERENCES datasets.materialisation (id),
    ADD COLUMN dependencies jsonb,
    ALTER COLUMN signal SET DEFAULT '{"members": []}'::jsonb;
