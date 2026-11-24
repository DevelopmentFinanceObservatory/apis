"""The stored warning formats that name refs in their message, their rewrite, and the count of
version rows still holding one.

`OLD_FORMATS` holds the only definition of each format: one detection pattern per warning
code, valid in both Python `re` and Postgres ARE. Detection is loose: it asks whether a
message names something, so a message no rewrite can parse still counts. The rewrite parses
exactly. `rewrite_old_formats` rewrites every format but one, with the functions of
`warning_rewrite.sql`: the old member sentence (`MEMBER_SENTENCE`), whose split into
off-cadence and out-of-window members needs graph reads, is left to
`datasets_api.warning_backfill`. `datasets-maintenance rewrite-warnings` runs the rewrite on
every deploy run whose target image serves, once the services have moved, scanning
`datasets.version_record`, so the next deploy rewrites a row an old revision wrote in an old
format while one ran.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import cast

from dataset_store.bounds import TransactionBounds
from dataset_store.store import DatasetStore, Transaction

# The detection pattern of each code's format.
OLD_FORMATS: dict[str, str] = {
    "coverage_below_scope": r"No observations were served for (?![0-9]+ requested entit)",
    "currency_unmapped_omitted": r" entit(?:y|ies): ",
    "deflator_missing_omitted": r" entit(?:y|ies): ",
    "withheld_by_registry": r" served no observations for ",
    "un_system_unavailable": r"No UN data was read for (?![0-9]+ entit)",
    "deflator_projected": r", for example: ",
}

# The old sentence naming the members of a group that served nothing: the first five `dcid:`
# refs, a count of the rest, and the range-window clause.
MEMBER_SENTENCE_CODE = "coverage_below_scope"
MEMBER_SENTENCE = (
    r" No observations were served for (dcid:[^ ,]+(?:, dcid:[^ ,]+)*)(?: and ([0-9]+) more)?, "
    r"which hold none at the observation period the other members serve"
    r"( in the requested years)?\."
)

REWRITE_SQL = Path(__file__).with_name("warning_rewrite.sql")

REWRITE_LOCK_TIMEOUT = "5s"
"""How long `rewrite_old_formats` waits on a row lock a running writer holds."""


def count_timeout(bounds: TransactionBounds) -> timedelta:
    """How long `count_old_format_rows` may scan `version_record` before Postgres cancels it:
    five sixths of the store's socket bound (`TransactionBounds.round_trip`), 25 s at the
    default 30 s. Postgres cancels the scan and replies before the socket read gives up, so a
    slow count raises `query_canceled` (SQLSTATE 57014) and the connection stays usable."""
    return bounds.round_trip * 5 / 6


# The signature of each function `REWRITE_SQL` defines; an argument list holds no parenthesis.
_DEFINED = re.compile(r"^CREATE OR REPLACE FUNCTION (pg_temp\.\w+\([^)]*\))", re.MULTILINE)


def _drop_functions() -> list[str]:
    """A `DROP FUNCTION` of each function `REWRITE_SQL` defines, the last defined first."""
    return [
        f"DROP FUNCTION {signature}"
        for signature in reversed(_DEFINED.findall(REWRITE_SQL.read_text()))
    ]


_COLUMN = "CASE WHEN jsonb_typeof(v.warnings) = 'array' THEN v.warnings ELSE '[]'::jsonb END"
_BODY = (
    "CASE WHEN jsonb_typeof(v.immutable_body -> 'warnings') = 'array' "
    "THEN v.immutable_body -> 'warnings' ELSE '[]'::jsonb END"
)

# Whether the row `v` holds a warning in its code's old format, in its `warnings` column or in
# the `warnings` of its `immutable_body`. It takes `OLD_FORMATS`' codes and patterns as two
# parameters, and calls no function of `warning_rewrite.sql`.
_HOLDS_OLD_FORMAT = (
    f"EXISTS (SELECT 1 FROM jsonb_array_elements({_COLUMN} || {_BODY}) AS w "
    "JOIN unnest(%s::text[], %s::text[]) AS f (code, pattern) ON f.code = w ->> 'code' "
    "WHERE w ->> 'message' ~ f.pattern)"
)

_OLD_FORMAT_ROWS = (
    f"SELECT count(*) AS count FROM datasets.version_record AS v WHERE {_HOLDS_OLD_FORMAT}"
)

_FOUND = (
    "SELECT v.subject_id, v.content_version FROM datasets.version_record AS v "
    f"WHERE {_HOLDS_OLD_FORMAT}"
)

REWRITE_BATCH_ROWS = 500
"""The most rows one page of `rewrite_old_formats` finds and rewrites, in one transaction that
holds their row locks until it commits. A batch of 500 found rows with 9 KB bodies, half of them
changed, runs in 0.13 to 0.16 s against a local Postgres 18, so the 30 s each statement may run
(`dataset_store.bounds.CONNECT_TIMEOUT_SECONDS`) leaves room for bodies about a hundred times
larger or a far slower disk."""

# Of the found rows keyed by the two arrays, each is updated only where the rewrite changes its
# column or its body's `warnings`, each rewritten from its own value as it stands once the row
# is locked; a row holding only a member sentence, or a list the rewrite cannot parse, is
# neither updated nor locked. Each key is probed through the primary key.
_REWRITE = (
    "WITH rewritten AS ("
    "UPDATE datasets.version_record AS v SET "
    "warnings = pg_temp.warnings_counted_refs(v.warnings), "
    "immutable_body = CASE WHEN jsonb_typeof(v.immutable_body -> 'warnings') = 'array' "
    "THEN jsonb_set(v.immutable_body, '{warnings}', "
    "pg_temp.warnings_counted_refs(v.immutable_body -> 'warnings')) "
    "ELSE v.immutable_body END "
    "FROM unnest(%s::text[], %s::text[]) AS found (subject_id, content_version) "
    "WHERE v.subject_id = found.subject_id AND v.content_version = found.content_version "
    "AND (pg_temp.warnings_counted_refs(v.warnings) IS DISTINCT FROM v.warnings "
    "OR pg_temp.warnings_counted_refs(v.immutable_body -> 'warnings') "
    "IS DISTINCT FROM v.immutable_body -> 'warnings') "
    "RETURNING 1) "
    "SELECT count(*) AS rewritten FROM rewritten"
)


def _formats() -> tuple[list[str], list[str]]:
    return list(OLD_FORMATS), list(OLD_FORMATS.values())


def rewrite_old_formats(store: DatasetStore) -> Iterator[int]:
    """Rewrite every old format but the member sentence in the version rows holding one, in
    the `warnings` column and in the `warnings` of `immutable_body`, each from its own value,
    and yield the rows each page rewrote once its transaction has committed.

    The scan runs a page at a time: each page finds, after the last key the page before it
    reached, up to `REWRITE_BATCH_ROWS` rows holding an old format in primary key order (a
    keyset), and rewrites them by key, in one transaction of its own. A failure loses only its
    own page. A later run scans from the start, and the rows earlier pages rewrote no longer
    hold an old format, so it rewrites only what is left. The page that finds fewer rows than a
    full page is the last. `count_old_format_rows` counts the rows left in an old format: a row
    a concurrent rewrite cleared between a page's find and its update is found but not
    rewritten, so the pages cannot state that count.

    Each page defines the functions of `REWRITE_SQL` and drops them before it commits. The file
    is sent without parameters, so through the simple query protocol, which accepts its several
    statements, and each of its patterns holding a backslash is an escape string, so it reads
    the same whatever `standard_conforming_strings` is. A row lock a running writer holds for
    more than `REWRITE_LOCK_TIMEOUT` fails the page, which a later run retries."""
    after: tuple[str, str] | None = None
    while True:
        with store.transaction() as tx:
            found, rewritten = _rewrite_page(tx, after)
        yield rewritten
        if len(found) < REWRITE_BATCH_ROWS:
            return
        after = found[-1]


def _rewrite_page(
    tx: Transaction, after: tuple[str, str] | None
) -> tuple[list[tuple[str, str]], int]:
    """The keys one page found after `after` and the rows it rewrote."""
    tx.execute(f"SET LOCAL lock_timeout = '{REWRITE_LOCK_TIMEOUT}'")
    tx.execute(REWRITE_SQL.read_text())
    keyset = "" if after is None else " AND (v.subject_id, v.content_version) > (%s, %s)"
    found = [
        (cast(str, row["subject_id"]), cast(str, row["content_version"]))
        for row in tx.fetch_all(
            f"{_FOUND}{keyset} ORDER BY v.subject_id, v.content_version LIMIT %s",
            (*_formats(), *(after or ()), REWRITE_BATCH_ROWS),
        )
    ]
    rewritten = 0
    if found:
        row = tx.fetch_one(
            _REWRITE,
            (
                [subject_id for subject_id, _ in found],
                [content_version for _, content_version in found],
            ),
        )
        # A `SELECT` of aggregates without `GROUP BY` returns exactly one row.
        assert row is not None
        rewritten = int(cast(int, row["rewritten"]))
    for statement in _drop_functions():
        tx.execute(statement)
    return found, rewritten


def count_old_format_rows(tx: Transaction, *, bounds: TransactionBounds) -> int:
    """The version rows holding a warning in its code's old format, in the `warnings` column
    or in the `warnings` of `immutable_body`. The count scans the table, so Postgres cancels
    it after `count_timeout(bounds)` and it raises. `bounds` are those of the store `tx`
    belongs to."""
    # A statement_timeout of 0 turns the timeout off, so the bound is at least 1 ms.
    timeout_ms = max(1, int(count_timeout(bounds) / timedelta(milliseconds=1)))
    tx.execute(f"SET LOCAL statement_timeout = '{timeout_ms}ms'")
    row = tx.fetch_one(_OLD_FORMAT_ROWS, _formats())
    # `count(*)` without `GROUP BY` returns exactly one row.
    assert row is not None
    return int(cast(int, row["count"]))
