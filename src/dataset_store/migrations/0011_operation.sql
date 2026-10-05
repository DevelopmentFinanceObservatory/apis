-- The async worker's own durable job record: one row per operation,
-- keyed on an id the mint door derives deterministically from the
-- request identity and the coverage stamp the sync-admission estimate
-- ran against (never from `options`, which never hashes into identity
-- anywhere else on this surface either -- see `datasets_api.schema
-- .MintRequest`'s own docstring). A repeated submit of the same request
-- against the same coverage stamp recomputes the same id and finds the
-- row `create_or_get` already wrote, rather than starting a second
-- operation for work already pending, running, or done.
--
-- `request_identity` and `options` together carry enough for a worker
-- to reproduce the mint with no access to the original HTTP request:
-- `request_identity` is the mint input's own identity payload (the same
-- shape a `spec_id`/`col_id` hashes over), `options` the rendering-only
-- block (format, bundle, pivot, columns, ...) a mint never folds into
-- identity. `coverage_stamp` is the `series_coverage` artifact stamp the
-- estimate that routed this request to the async path ran against,
-- carried here for the same lineage reason `datasets.version_record`
-- (migration 0004) carries `series_coverage_stamp` on a manifest.
--
-- `state` starts `pending`, moves to `running` once a worker claims the
-- lease, and ends at `done` or `failed`, never reversing. `lease_until`
-- is null until first claimed; a worker claims or reclaims the lease with
-- one conditional `UPDATE ... WHERE state IN ('pending', 'running') AND
-- (lease_until IS NULL OR lease_until <= now())`, so a crashed worker's
-- expired lease is retaken without an explicit unlock step, and a live
-- lease held by another attempt blocks a concurrent claim. `attempt_count`
-- increments on every successful claim, both the first and any retake.
--
-- `subject_id`/`content_version` are set together, once, on completion:
-- the manifest reference a `done` operation's own poll re-signs through
-- exactly the same path `GET /api/datasets/{id}/{content_version}`
-- (`datasets_api.doors.snapshot.read_snapshot_manifest`) already uses --
-- this table stores no manifest body or signed URL of its own, since a
-- stored signed URL would go stale between completion and the next poll
-- (see that door's own module docstring on why signing always happens at
-- read time). The foreign key onto `datasets.version_record` only checks
-- once both columns are set (a composite foreign key with either side
-- NULL is not enforced), so a pending/running row references nothing yet.
-- `refusal_code` is set together with a `failed` state and stays null
-- otherwise.
--
-- 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES already covers this table.
CREATE TABLE datasets.operation (
    id text PRIMARY KEY,
    request_identity jsonb NOT NULL,
    options jsonb NOT NULL,
    coverage_stamp text NOT NULL,
    state text NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'running', 'done', 'failed')),
    estimate_rows bigint NOT NULL,
    estimate_requests bigint NOT NULL,
    rows_written bigint NOT NULL DEFAULT 0,
    subject_id text,
    content_version text,
    refusal_code text,
    attempt_count int NOT NULL DEFAULT 0,
    lease_until timestamptz,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CHECK ((subject_id IS NULL) = (content_version IS NULL)),
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
