"""The stored warning formats that name refs in their message, and a dry run of their rewrite.

`OLD_FORMATS` holds the only definition of each format: one detection pattern per warning
code, valid in both Python `re` and Postgres ARE. Detection is loose: it asks whether a
message names something, so a message the rewrite cannot parse still counts. The rewrite
parses exactly. `warning_rewrite.sql` rewrites the unserved-entity, currency and deflator
lists; the backfill task rewrites the old member sentence (`MEMBER_SENTENCE`) and the codes in
`TASK_CODES`.

`python -m dataset_store.warning_formats --dry-run` opens a transaction, executes
`warning_rewrite.sql`, which creates temporary functions only, sets the transaction READ ONLY,
refuses unless the session reports it read-only, reads `datasets.version_record.warnings`
and rolls back. It prints the version rows the SQL rewrite changes, the rows left for the
task, and the residual rows, each by format; then up to `SAMPLES_PER_FORMAT` messages of each
format before and after the SQL rewrite; then a census by code of the rows whose message holds
a `dcid:` or slash-shaped ref, which finds codes no format lists. A residual row holds a
warning that, after the SQL rewrite and with its member sentences removed, still matches a
format the task does not rewrite. Any residual row exits 1.

It connects with the migration job's settings (`dataset_store.migrate`), so it runs as an
execution of that job with its args overridden. The deployed image omits `scripts/`, so this
command lives in the package.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dataset_store import connection
from dataset_store.dbapi import DBAPIConnection, DBAPICursor
from dataset_store.errors import DatasetStoreConfigError, DatasetStoreError

# (name, code, detection pattern). Each code has one format.
OLD_FORMATS: tuple[tuple[str, str, str], ...] = (
    (
        "coverage_below_scope, entities or members named",
        "coverage_below_scope",
        r"No observations were served for (?![0-9]+ requested entit)",
    ),
    (
        "currency_unmapped_omitted, entities named",
        "currency_unmapped_omitted",
        r" entit(?:y|ies): ",
    ),
    (
        "deflator_missing_omitted, entities named",
        "deflator_missing_omitted",
        r" entit(?:y|ies): ",
    ),
    (
        "withheld_by_registry, entities and provenances named",
        "withheld_by_registry",
        r" served no observations for ",
    ),
    (
        "un_system_unavailable, places named",
        "un_system_unavailable",
        r"No UN data was read for (?![0-9]+ entit)",
    ),
    (
        "deflator_projected, examples named",
        "deflator_projected",
        r", for example: ",
    ),
)

# The old sentence naming the members of a group that served nothing, which the backfill task
# rewrites: the first five `dcid:` refs, a count of the rest, and the range-window clause.
MEMBER_SENTENCE_CODE = "coverage_below_scope"
MEMBER_SENTENCE = (
    r" No observations were served for (dcid:[^ ,]+(?:, dcid:[^ ,]+)*)(?: and ([0-9]+) more)?, "
    r"which hold none at the observation period the other members serve"
    r"( in the requested years)?\."
)

# The codes whose every detected warning the backfill task rewrites.
TASK_CODES = frozenset({"withheld_by_registry", "un_system_unavailable", "deflator_projected"})

REWRITE_SQL = Path(__file__).with_name("warning_rewrite.sql")
SAMPLES_PER_FORMAT = 20
VERSION_TABLE = "datasets.version_record"

_REF = r"dcid:|[A-Za-z][A-Za-z0-9_]*/[A-Za-z0-9_]"
_TABLE = re.compile(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*")
_ROW = "(subject_id, content_version)"


@dataclass(frozen=True)
class DryRun:
    """What the rewrite would change. Counts are version rows, by format name."""

    read_only: str
    rewritten_rows: int
    rewritten: dict[str, int]
    task_rows: int
    task: dict[str, int]
    residual_rows: int
    residual: dict[str, int]
    samples: dict[str, list[tuple[str, str]]]
    census: list[tuple[str | None, int, int]]


def _literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _matches(message: str, formats: Sequence[tuple[str, str, str]]) -> str:
    """A CASE on `code`, true when `message` matches the pattern of its code's format."""
    branches = " ".join(
        f"WHEN {_literal(code)} THEN {message} ~ {_literal(pattern)}"
        for _name, code, pattern in formats
    )
    return f"CASE code {branches} ELSE false END"


def _classified(table: str) -> str:
    """`classified`: each stored warning of `table` with its message before and after the SQL
    rewrite, whether that rewrite changes it, whether the task rewrites it next, and whether it
    is residual."""
    task = _matches("after", [f for f in OLD_FORMATS if f[1] in TASK_CODES])
    residual = _matches(
        f"regexp_replace(after, {_literal(MEMBER_SENTENCE)}, '', 'g')",
        [f for f in OLD_FORMATS if f[1] not in TASK_CODES],
    )
    return f"""
        WITH warning AS (
            SELECT
                r.subject_id,
                r.content_version,
                w.position,
                w.value ->> 'code' AS code,
                w.value ->> 'message' AS before,
                rewritten.value ->> 'message' AS after,
                rewritten.value IS DISTINCT FROM w.value AS rewritten
            FROM {table} AS r
            CROSS JOIN LATERAL jsonb_array_elements(r.warnings)
                WITH ORDINALITY AS w (value, position)
            CROSS JOIN LATERAL (SELECT pg_temp.warning_counted_refs_0046(w.value) AS value)
                AS rewritten
            WHERE jsonb_typeof(r.warnings) = 'array'
        ),
        classified AS (
            SELECT
                *,
                {task} OR code = {_literal(MEMBER_SENTENCE_CODE)}
                    AND after ~ {_literal(MEMBER_SENTENCE)} AS by_task,
                {residual} AS residual
            FROM warning
        )
    """


def dry_run(conn: DBAPIConnection, *, table: str = VERSION_TABLE) -> DryRun:
    """Read what `warning_rewrite.sql` and the task would change in `table`'s `warnings`.

    Creates the rewrite's temporary functions, sets the transaction READ ONLY, reads, and rolls
    back. Raises `DatasetStoreError` when the session does not report the transaction
    read-only."""
    if _TABLE.fullmatch(table) is None:
        raise ValueError(f"not a schema-qualified table name: {table!r}")
    classified = _classified(table)
    cursor = conn.cursor()
    try:
        cursor.execute(REWRITE_SQL.read_text())
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SHOW transaction_read_only")
        read_only = cursor.fetchone()
        if read_only is None or read_only[0] != "on":
            raise DatasetStoreError(f"transaction_read_only is {read_only!r}, not 'on'")
        counts = _fetch_all(
            cursor,
            f"""{classified}
            SELECT
                GROUPING(code) = 1,
                code,
                count(DISTINCT {_ROW}) FILTER (WHERE rewritten),
                count(DISTINCT {_ROW}) FILTER (WHERE by_task),
                count(DISTINCT {_ROW}) FILTER (WHERE residual)
            FROM classified
            GROUP BY GROUPING SETS ((code), ())
            """,
        )
        samples = _fetch_all(
            cursor,
            f"""{classified}
            SELECT code, before, after
            FROM (
                SELECT
                    code,
                    before,
                    after,
                    row_number() OVER (
                        PARTITION BY code ORDER BY subject_id, content_version, position
                    ) AS n
                FROM classified
                WHERE {_matches("before", OLD_FORMATS)}
            ) AS sampled
            WHERE n <= {SAMPLES_PER_FORMAT}
            ORDER BY code, n
            """,
        )
        census = _fetch_all(
            cursor,
            f"""{classified}
            SELECT
                code,
                count(DISTINCT {_ROW}),
                count(DISTINCT {_ROW}) FILTER (WHERE before ~ {_literal(_REF)})
            FROM classified
            GROUP BY code
            ORDER BY code NULLS LAST
            """,
        )
    finally:
        cursor.close()
        conn.rollback()

    names = {code: name for name, code, _pattern in OLD_FORMATS}
    totals = (0, 0, 0)
    by_code: dict[str, tuple[int, int, int]] = {}
    for total, code, *row in counts:
        if total:
            totals = cast("tuple[int, int, int]", tuple(row))
        elif code in names:
            by_code[cast(str, code)] = cast("tuple[int, int, int]", tuple(row))

    def _by_format(index: int) -> dict[str, int]:
        return {name: by_code.get(code, (0, 0, 0))[index] for code, name in names.items()}

    sampled: dict[str, list[tuple[str, str]]] = {name: [] for name in names.values()}
    for code, before, after in samples:
        sampled[names[cast(str, code)]].append((cast(str, before), cast(str, after)))
    return DryRun(
        read_only=cast(str, read_only[0]),
        rewritten_rows=totals[0],
        rewritten=_by_format(0),
        task_rows=totals[1],
        task=_by_format(1),
        residual_rows=totals[2],
        residual=_by_format(2),
        samples=sampled,
        census=[
            (cast("str | None", code), cast(int, rows), cast(int, refs))
            for code, rows, refs in census
        ],
    )


def _fetch_all(cursor: DBAPICursor, sql: str) -> list[tuple[object, ...]]:
    cursor.execute(sql)
    return cursor.fetchall()


def _print(report: DryRun) -> None:
    print(f"transaction read only: {report.read_only}")
    for title, rows, by_format in (
        ("rows the SQL rewrite changes", report.rewritten_rows, report.rewritten),
        ("rows the task will change", report.task_rows, report.task),
        ("residual rows", report.residual_rows, report.residual),
    ):
        print(f"{title}: {rows}")
        for name, count in by_format.items():
            print(f"  {name}: {count}")
    for name, samples in report.samples.items():
        print(f"samples, {name}: {len(samples)}")
        for before, after in samples:
            print(f"  before: {before}")
            if after != before:
                print(f"  after:  {after}")
    print("rows by code, then rows whose message holds a dcid: or slash-shaped ref:")
    for code, rows, refs in report.census:
        print(f"  {code}: {rows}, {refs}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run `python -m dataset_store.warning_formats --dry-run`. A configuration or store error,
    or a residual row, exits 1."""
    parser = argparse.ArgumentParser(prog="python -m dataset_store.warning_formats")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="report what the rewrite would change, then roll back",
    )
    parser.parse_args(argv)
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
        report = dry_run(conn)
    except DatasetStoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    _print(report)
    return 1 if report.residual_rows else 0


if __name__ == "__main__":
    sys.exit(main())
