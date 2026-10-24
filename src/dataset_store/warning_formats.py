"""Count the stored warnings that name refs in their message, by old format.

Migration `0039_warning_counted_refs.sql` rewrites these warnings of
`datasets.version_record.warnings` into counted messages with structured fields. This command
counts each old format it rewrites, and checks the `member_count` a group's
`coverage_below_scope` already stores against the length of its trace's
`resolved_stat_vars`, the count 0039 gives the warnings it rewrites. It reads in a READ ONLY
transaction, refuses to read when the session is not read-only, and rolls back.

It connects with the migration job's settings (`dataset_store.migrate`), so it runs as an
execution of that job with its args overridden. The deployed image omits `scripts/`, so this
command lives in the package. Run it with `python -m dataset_store.warning_formats`.
"""

from __future__ import annotations

import sys
from typing import cast

from dataset_store import connection
from dataset_store.dbapi import DBAPIConnection
from dataset_store.errors import DatasetStoreConfigError, DatasetStoreError

# Each format and the SQL condition on one stored warning that selects it. A warning reads as
# `code`, `message`, `body` (the whole warning) and `trace_members`, the length of its row's
# `trace.resolved_stat_vars`, NULL when the trace holds no such list. The patterns are the
# sentences 0039 parses.
_MEMBERS = "which hold none at the observation period the other members serve"
_UNSERVED = r" No observations were served for [^ ,]+(, [^ ,]+)*( and [0-9]+ more)?\."
_FALLBACK = "No place was named, so the answer covers the countries with data"
_NAMED = " [0-9]+ entities: "
_TRUNCATED = r" [0-9]+ entities: .* \(truncated\)$"
FORMATS: tuple[tuple[str, str], ...] = (
    (
        "coverage_below_scope, unserved entities named",
        f"code = 'coverage_below_scope' AND message ~ '{_UNSERVED}'",
    ),
    (
        "coverage_below_scope, members named, window not a range",
        rf"code = 'coverage_below_scope' AND message ~ '{_MEMBERS}\.'",
    ),
    (
        "coverage_below_scope, members named, range window",
        rf"code = 'coverage_below_scope' AND message ~ '{_MEMBERS} in the requested years\.'",
    ),
    (
        "coverage_below_scope, members named, trace holds no resolved_stat_vars",
        f"code = 'coverage_below_scope' AND message ~ '{_MEMBERS}' AND trace_members IS NULL",
    ),
    (
        "coverage_below_scope, named refs beside the countries fallback",
        f"code = 'coverage_below_scope' AND (message ~ '{_UNSERVED}' OR message ~ '{_MEMBERS}') "
        f"AND strpos(message, '{_FALLBACK}') > 0",
    ),
    (
        "currency_unmapped_omitted, entities named",
        f"code = 'currency_unmapped_omitted' AND message ~ '{_NAMED}'",
    ),
    (
        "currency_unmapped_omitted, entities named, truncated",
        f"code = 'currency_unmapped_omitted' AND message ~ '{_TRUNCATED}'",
    ),
    (
        "deflator_missing_omitted, entities named",
        f"code = 'deflator_missing_omitted' AND message ~ '{_NAMED}'",
    ),
    (
        "deflator_missing_omitted, entities named, truncated",
        f"code = 'deflator_missing_omitted' AND message ~ '{_TRUNCATED}'",
    ),
    (
        "currency or deflator, neither named nor counted",
        "code IN ('currency_unmapped_omitted', 'deflator_missing_omitted') "
        f"AND message !~ '{_NAMED}' "
        "AND NOT (body ? 'omitted_entities' OR body ? 'omitted_entity_years')",
    ),
    (
        "member_count equals the trace's resolved_stat_vars",
        "body ? 'member_count' AND (body ->> 'member_count')::integer = trace_members",
    ),
    (
        "member_count differs from the trace's resolved_stat_vars",
        "body ? 'member_count' AND (body ->> 'member_count')::integer <> trace_members",
    ),
    (
        "member_count beside a trace holding no resolved_stat_vars",
        "body ? 'member_count' AND trace_members IS NULL",
    ),
)

# The formats 0039 rewrites, whose rows it updates.
_REWRITTEN = (
    f"code = 'coverage_below_scope' AND (message ~ '{_UNSERVED}' OR message ~ '{_MEMBERS}') "
    f"OR code IN ('currency_unmapped_omitted', 'deflator_missing_omitted') "
    f"AND message ~ '{_NAMED}'"
)
REWRITTEN_ROWS = "version rows 0039 rewrites"


def _query() -> str:
    """One row: for each of `FORMATS`, its warnings and the version rows holding them, then
    the version rows holding any warning 0039 rewrites."""
    counts = ",\n".join(
        f"count(*) FILTER (WHERE {condition}), "
        f"count(DISTINCT (subject_id, content_version)) FILTER (WHERE {condition})"
        for _name, condition in FORMATS
    )
    return f"""
        WITH warning AS (
            SELECT
                r.subject_id,
                r.content_version,
                w.value ->> 'code' AS code,
                w.value ->> 'message' AS message,
                w.value AS body,
                CASE
                    WHEN jsonb_typeof(r.trace -> 'resolved_stat_vars') = 'array'
                        THEN jsonb_array_length(r.trace -> 'resolved_stat_vars')
                END AS trace_members
            FROM datasets.version_record AS r
            CROSS JOIN LATERAL jsonb_array_elements(r.warnings) AS w (value)
            WHERE jsonb_typeof(r.warnings) = 'array'
        )
        SELECT
            {counts},
            count(DISTINCT (subject_id, content_version)) FILTER (WHERE {_REWRITTEN})
        FROM warning
    """


def count_formats(conn: DBAPIConnection) -> list[tuple[str, int | None, int]]:
    """Each of `FORMATS` with its warnings and the version rows holding them, then
    `REWRITTEN_ROWS` with its version rows alone. Reads in a READ ONLY transaction and rolls
    back. Raises `DatasetStoreError` when the session does not report the transaction
    read-only."""
    cursor = conn.cursor()
    try:
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SHOW transaction_read_only")
        read_only = cursor.fetchone()
        if read_only is None or read_only[0] != "on":
            raise DatasetStoreError(f"transaction_read_only is {read_only!r}, not 'on'")
        cursor.execute(_query())
        row = cursor.fetchone()
    finally:
        cursor.close()
        conn.rollback()
    assert row is not None
    values = cast("tuple[int, ...]", row)
    counts: list[tuple[str, int | None, int]] = [
        (name, values[2 * index], values[2 * index + 1])
        for index, (name, _condition) in enumerate(FORMATS)
    ]
    return [*counts, (REWRITTEN_ROWS, None, values[-1])]


def main() -> int:
    """Run `python -m dataset_store.warning_formats`. Prints one line per format; a
    configuration or store error exits 1."""
    try:
        settings = connection.ConnectionSettings.from_env(
            user_var="DATASET_STORE_MIGRATOR_DB_USER",
            password_var="DATASET_STORE_MIGRATOR_DB_PASSWORD",
            enable_iam_auth=False,
        )
    except DatasetStoreConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    connector = connection.build_connector() if settings.instance_connection_name else None
    conn = settings.connect_callable(connector=connector)()
    try:
        counts = count_formats(conn)
    except DatasetStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    for name, warnings, rows in counts:
        if warnings is None:
            print(f"{name}: {rows}")
        else:
            print(f"{name}: {warnings} warnings in {rows} version rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
