-- The mutable head: the current (content_version, confirmed_at,
-- materialised_at, signal) a live door serves for one subject, keyed by
-- `subject_id` alone. `datasets.version_record` (migration 0004) is
-- written once per (subject_id, content_version) pair and stays immutable
-- after that, with nowhere to say which of its own content_versions is
-- the one a GET should currently serve. This table is that pointer.
--
-- `confirmed_at` is the last instant a probe or a materialisation
-- confirmed the stored `content_version` current. `materialised_at` is
-- the last instant a materialisation wrote that `content_version`: it
-- only moves alongside `content_version` itself, while `confirmed_at`
-- also moves on a probe that finds the existing `content_version` still
-- current. A maximum-age check measures staleness from `materialised_at`,
-- since a `confirmed_at` that moves on every cheap probe would never let
-- a head go stale.
--
-- `signal` is the currency signal captured at `confirmed_at`, as a JSON
-- object `{"members": [...]}`: one entry per member
-- (`dispatch_facet_id`, `obs_count`, `earliest_date`, `latest_date`, the
-- fields `materialise.facets.probe_facets` already computes), wrapped in an
-- object because a bare Python `list` has no `jsonb` mapping through this
-- store's driver. Only `dict` does (see
-- `datasets_api.versions.encode_signal`). A single-frame group
-- carries exactly one entry. A later currency check compares a fresh probe
-- against this stored value to decide whether the served frame is still
-- current, without re-materialising.
--
-- `(subject_id, content_version)` FK-references `datasets.version_record`:
-- a head can only ever point at a version that was actually recorded, so
-- the head write always runs after that version's own `record_version`
-- call has committed. 0001_bootstrap.sql's ALTER DEFAULT PRIVILEGES
-- already covers this table.
CREATE TABLE datasets.subject_head (
    subject_id text PRIMARY KEY,
    content_version text NOT NULL,
    confirmed_at timestamptz NOT NULL,
    materialised_at timestamptz NOT NULL,
    signal jsonb NOT NULL,
    FOREIGN KEY (subject_id, content_version)
        REFERENCES datasets.version_record (subject_id, content_version)
);
