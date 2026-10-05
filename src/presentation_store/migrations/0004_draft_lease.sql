-- Draft leases: the single session-ownership authority for a component or
-- composition draft. `subject_id` is a `component_id` or `composition_id`
-- depending on `subject_kind`, never both at once, so one table (not two)
-- covers both -- store.py's lease queries always filter on `subject_kind`
-- alongside `subject_id`. A component's lease is born in the same
-- transaction that creates its `component` row (store.py's `begin_write`,
-- create path) and is renewed on every successful write; nothing in this
-- migration series ever deletes a row here -- `list_expired_leases`/
-- `list_orphaned_leases` (store.py) are read-only reclaim queries with no
-- accompanying sweep job: the reclaim queries and their tests ship, the
-- sweep job itself doesn't. 0001_bootstrap.sql's
-- ALTER DEFAULT PRIVILEGES already covers this table, so this migration
-- needs no GRANT statement of its own.

CREATE TABLE presentation.draft_lease (
    subject_id text PRIMARY KEY,
    subject_kind text NOT NULL CHECK (subject_kind IN ('component', 'composition')),
    session_id text NOT NULL,
    renewed_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL
);

-- Every lease predicate in store.py includes `expires_at > now()` (a live
-- ownership check) or `expires_at <= now()` (`list_expired_leases`'s own
-- reclaim scan); `subject_id`'s own PK index covers the former's point
-- lookups but not a scan ordered by expiry, which this index covers.
CREATE INDEX draft_lease_expires_at_idx ON presentation.draft_lease (expires_at);
