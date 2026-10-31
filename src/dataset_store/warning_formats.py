"""The stored warning formats that name refs in their message, and a dry run of their rewrite.

`OLD_FORMATS` holds the only definition of each format: one detection pattern per warning
code, valid in both Python `re` and Postgres ARE. Detection is loose: it asks whether a
message names something, so a message the rewrite cannot parse still counts. The rewrite
parses exactly. `warning_rewrite.sql` rewrites the unserved-entity, currency and deflator
lists. It leaves two kinds of warning for a rewrite outside SQL: the old member sentence
(`MEMBER_SENTENCE`), whose split into off-cadence and out-of-window members needs graph reads,
and the codes in `CODES_LEFT_BY_SQL`, whose new sentences and fields come from their writers.
The serving image still writes the codes in `CODES_STILL_WRITTEN` in their old shape, so their
rows are reported under a heading of their own.

`python -m dataset_store.warning_formats` refuses unless every statement of
`warning_rewrite.sql` is `CREATE OR REPLACE FUNCTION pg_temp.`, then opens a transaction,
executes each statement on its own through the extended query protocol, which refuses a
statement holding a second command, sets the transaction READ ONLY, refuses unless the
session reports it read-only, reads `datasets.version_record` in one statement and rolls
back. Its patterns are bound parameters. It prints, each by format, the version rows the SQL
rewrite changes, the rows it leaves for a rewrite outside SQL, the rows holding a code still
written in its old shape, and the residual rows; then up
to `SAMPLES_PER_FORMAT` messages of each format before and after the SQL rewrite; then the
rows whose `immutable_body` warnings differ from their `warnings` column, and of those the
rows whose body warnings the rewrite changes; then the rows whose stored `member_count`
equals, differs from or has no trace `resolved_stat_vars` to compare with, and the rows
holding a currency or deflator warning that neither names nor counts its entities; then a
census by code of the rows naming a `dcid:` or slash-shaped ref.

A residual row holds a warning that still matches its format after the SQL rewrite, once the
member sentences of a row whose trace holds `resolved_stat_vars` are removed: a member
sentence on a trace holding none has no member count to state. An unlisted row holds a
warning that names a ref after the SQL rewrite and with its member sentences removed, but
matches no format. Residual rows and unlisted rows of a code outside `CENSUS_ALLOWED` are
findings and exit 1. The rows holding a code still written in its old shape, the body figures,
and the `member_count` and conversion figures are reported only. A run that cannot complete
exits 2.

It connects with the migration job's settings (`dataset_store.migrate`), so it runs as an
execution of that job with its args overridden. The deployed image omits `scripts/`, so this
command lives in the package.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pg8000.exceptions

from dataset_store import connection
from dataset_store.dbapi import DBAPIConnection
from dataset_store.errors import DatasetStoreConfigError, DatasetStoreError
from dataset_store.migrate import _VALID_BOOKKEEPING_TABLE

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

# The old sentence naming the members of a group that served nothing: the first five `dcid:`
# refs, a count of the rest, and the range-window clause.
MEMBER_SENTENCE_CODE = "coverage_below_scope"
MEMBER_SENTENCE = (
    r" No observations were served for (dcid:[^ ,]+(?:, dcid:[^ ,]+)*)(?: and ([0-9]+) more)?, "
    r"which hold none at the observation period the other members serve"
    r"( in the requested years)?\."
)

# The codes `warning_rewrite.sql` does not rewrite, whose new sentences and structured fields
# come from their writers.
CODES_LEFT_BY_SQL = frozenset(
    {"withheld_by_registry", "un_system_unavailable", "deflator_projected"}
)

# The codes of `CODES_LEFT_BY_SQL` that the serving image still writes in their old shape
# (`datasets_api.origin`), so their rows keep growing until those writers count.
CODES_STILL_WRITTEN = frozenset({"withheld_by_registry", "un_system_unavailable"})

# The structured field that counts the entities of each conversion code.
CONVERSION_FIELDS = {
    "currency_unmapped_omitted": "omitted_entities",
    "deflator_missing_omitted": "omitted_entity_years",
}

# Codes whose messages may name a ref while matching no format.
CENSUS_ALLOWED: frozenset[str] = frozenset()

REWRITE_SQL = Path(__file__).with_name("warning_rewrite.sql")
SAMPLES_PER_FORMAT = 20
VERSION_TABLE = "datasets.version_record"

_REF = r"dcid:|[A-Za-z][A-Za-z0-9_]*/[A-Za-z0-9_]"
_DEFINITION = "CREATE OR REPLACE FUNCTION pg_temp."
_SQL_TOKEN = re.compile(
    r"(?P<comment>--[^\n]*|/\*.*?\*/)|[Ee]'(?:[^'\\]|\\.|'')*'|'(?:[^']|'')*'"
    r"|\$(?P<tag>[A-Za-z_]*)\$.*?\$(?P=tag)\$|(?P<end>;)|[^-/';$Ee]+|.",
    re.DOTALL,
)
_ROW = "(subject_id, content_version)"


class _ExtendedQuery(Protocol):
    """A pg8000 connection's extended query protocol: one Parse message, one command."""

    def execute_unnamed(self, statement: str) -> object: ...


@dataclass(frozen=True)
class DryRun:
    """What the rewrite would change. Counts are version rows; the by-format counts are keyed
    by format name, and `census` holds each code's rows, rows naming a ref and unlisted rows."""

    read_only: str
    rewritten_rows: int
    rewritten: dict[str, int]
    left_rows: int
    left: dict[str, int]
    still_written_rows: int
    still_written: dict[str, int]
    residual_rows: int
    residual: dict[str, int]
    samples: dict[str, list[tuple[str, str]]]
    body_mismatch_rows: int
    body_rewritten_rows: int
    member_count_equal_rows: int
    member_count_differs_rows: int
    member_count_without_trace_rows: int
    uncounted_conversion_rows: int
    census: list[tuple[str | None, int, int, int]]


def definitions(sql: str) -> list[str]:
    """The statements of `sql`, comments removed. Raises `DatasetStoreError` unless each is
    `CREATE OR REPLACE FUNCTION pg_temp.`."""
    statements: list[str] = []
    current: list[str] = []
    for token in _SQL_TOKEN.finditer(sql):
        if token["comment"] is not None:
            continue
        if token["end"] is None:
            current.append(token[0])
            continue
        statements.append("".join(current).strip())
        current = []
    statements.append("".join(current).strip())
    statements = [statement for statement in statements if statement]
    refused = [statement for statement in statements if not statement.startswith(_DEFINITION)]
    if not statements or refused:
        first = refused[0].splitlines()[0] if refused else "no statement"
        raise DatasetStoreError(
            f"{REWRITE_SQL.name} must define only `{_DEFINITION}` functions: {first!r}"
        )
    return statements


def _statement(table: str) -> str:
    """One row: the totals, the counts by format, the samples, the census, and the body
    figures, all read from one materialised scan of `table`."""
    return f"""
        WITH param AS (
            SELECT
                %s::text AS member_code,
                %s::text AS member,
                %s::text AS ref,
                %s::text[] AS left_codes,
                %s::text[] AS still_written_codes,
                %s::jsonb AS conversion_fields
        ),
        format AS (
            SELECT * FROM unnest(%s::text[], %s::text[], %s::text[]) AS format (name, code, pattern)
        ),
        version AS MATERIALIZED (
            SELECT
                subject_id,
                content_version,
                warnings,
                immutable_body -> 'warnings' AS body_warnings,
                CASE
                    WHEN jsonb_typeof(trace -> 'resolved_stat_vars') = 'array'
                        THEN jsonb_array_length(trace -> 'resolved_stat_vars')
                END AS trace_members
            FROM {table}
        ),
        warning AS MATERIALIZED (
            SELECT
                v.subject_id,
                v.content_version,
                v.trace_members IS NOT NULL AS has_members,
                v.trace_members,
                CASE
                    WHEN jsonb_typeof(w.value -> 'member_count') = 'number'
                        THEN (w.value ->> 'member_count')::numeric
                END AS member_count,
                p.conversion_fields ? (w.value ->> 'code')
                    AND NOT w.value ? (p.conversion_fields ->> (w.value ->> 'code'))
                    AS conversion_uncounted,
                w.position,
                w.value ->> 'code' AS code,
                f.name AS format,
                f.pattern,
                coalesce(f.code = ANY(p.left_codes), false) AS code_left,
                coalesce(f.code = ANY(p.still_written_codes), false) AS code_still_written,
                w.value ->> 'message' AS before,
                r.value ->> 'message' AS after,
                r.value IS DISTINCT FROM w.value AS rewritten,
                regexp_replace(r.value ->> 'message', p.member, '', 'g') AS without_members,
                w.value ->> 'code' = p.member_code AND r.value ->> 'message' ~ p.member
                    AS has_member_sentence,
                p.ref
            FROM version AS v
            CROSS JOIN param AS p
            CROSS JOIN LATERAL jsonb_array_elements(v.warnings)
                WITH ORDINALITY AS w (value, position)
            CROSS JOIN LATERAL (SELECT pg_temp.warning_counted_refs(w.value) AS value) AS r
            LEFT JOIN format AS f ON f.code = w.value ->> 'code'
            WHERE jsonb_typeof(v.warnings) = 'array'
        ),
        classified AS MATERIALIZED (
            SELECT
                subject_id,
                content_version,
                position,
                code,
                format,
                before,
                after,
                rewritten,
                coalesce(before ~ pattern, false) AS detected,
                coalesce(has_members AND has_member_sentence, false)
                    OR code_left AND NOT code_still_written AND coalesce(after ~ pattern, false)
                    AS left_by_sql,
                code_still_written AND coalesce(after ~ pattern, false) AS still_written,
                NOT code_left AND coalesce(
                    CASE WHEN has_members THEN without_members ELSE after END ~ pattern, false
                ) AS residual,
                coalesce(before ~ ref, false) AS names_ref,
                coalesce(without_members ~ ref, false)
                    AND NOT coalesce(without_members ~ pattern, false) AS unlisted,
                CASE
                    WHEN member_count IS NULL THEN NULL
                    WHEN trace_members IS NULL THEN 'without_trace'
                    WHEN member_count = trace_members THEN 'equal'
                    ELSE 'differs'
                END AS member_count_check,
                coalesce(conversion_uncounted AND NOT before ~ pattern, false)
                    AS conversion_uncounted
            FROM warning
        )
        SELECT
            (
                SELECT jsonb_build_array(
                    count(DISTINCT {_ROW}) FILTER (WHERE rewritten),
                    count(DISTINCT {_ROW}) FILTER (WHERE left_by_sql),
                    count(DISTINCT {_ROW}) FILTER (WHERE still_written),
                    count(DISTINCT {_ROW}) FILTER (WHERE residual)
                )
                FROM classified
            ),
            (
                SELECT coalesce(jsonb_object_agg(format, counts), '{{}}'::jsonb)
                FROM (
                    SELECT
                        format,
                        jsonb_build_array(
                            count(DISTINCT {_ROW}) FILTER (WHERE rewritten),
                            count(DISTINCT {_ROW}) FILTER (WHERE left_by_sql),
                            count(DISTINCT {_ROW}) FILTER (WHERE still_written),
                            count(DISTINCT {_ROW}) FILTER (WHERE residual)
                        ) AS counts
                    FROM classified
                    WHERE format IS NOT NULL
                    GROUP BY format
                ) AS by_format
            ),
            (
                SELECT coalesce(
                    jsonb_agg(jsonb_build_array(format, before, after) ORDER BY format, n),
                    '[]'::jsonb
                )
                FROM (
                    SELECT
                        format,
                        before,
                        after,
                        row_number() OVER (
                            PARTITION BY format ORDER BY subject_id, content_version, position
                        ) AS n
                    FROM classified
                    WHERE detected
                ) AS sampled
                WHERE n <= {SAMPLES_PER_FORMAT:d}
            ),
            (
                SELECT jsonb_build_array(
                    count(*),
                    count(*) FILTER (
                        WHERE pg_temp.warnings_counted_refs(body_warnings)
                            IS DISTINCT FROM body_warnings
                    )
                )
                FROM version
                WHERE body_warnings IS DISTINCT FROM warnings
            ),
            (
                SELECT jsonb_build_array(
                    count(DISTINCT {_ROW}) FILTER (WHERE member_count_check = 'equal'),
                    count(DISTINCT {_ROW}) FILTER (WHERE member_count_check = 'differs'),
                    count(DISTINCT {_ROW}) FILTER (WHERE member_count_check = 'without_trace'),
                    count(DISTINCT {_ROW}) FILTER (WHERE conversion_uncounted)
                )
                FROM classified
            ),
            (
                SELECT coalesce(
                    jsonb_agg(
                        jsonb_build_array(code, rows, refs, unlisted) ORDER BY code NULLS LAST
                    ),
                    '[]'::jsonb
                )
                FROM (
                    SELECT
                        code,
                        count(DISTINCT {_ROW}) AS rows,
                        count(DISTINCT {_ROW}) FILTER (WHERE names_ref) AS refs,
                        count(DISTINCT {_ROW}) FILTER (WHERE unlisted) AS unlisted
                    FROM classified
                    GROUP BY code
                ) AS by_code
            )
    """


def _params() -> tuple[object, ...]:
    names, codes, patterns = (list(column) for column in zip(*OLD_FORMATS, strict=True))
    return (
        MEMBER_SENTENCE_CODE,
        MEMBER_SENTENCE,
        _REF,
        sorted(CODES_LEFT_BY_SQL),
        sorted(CODES_STILL_WRITTEN),
        json.dumps(CONVERSION_FIELDS),
        names,
        codes,
        patterns,
    )


def dry_run(conn: DBAPIConnection, *, table: str = VERSION_TABLE) -> DryRun:
    """Read what `warning_rewrite.sql` would change in `table`, and what it leaves.

    Creates the rewrite's temporary functions, sets the transaction READ ONLY, reads in one
    statement, and rolls back. Raises `DatasetStoreError` when the file defines anything but
    `pg_temp` functions or the session does not report the transaction read-only."""
    if not _VALID_BOOKKEEPING_TABLE.fullmatch(table):
        raise ValueError(f"invalid table {table!r}: must match {_VALID_BOOKKEEPING_TABLE.pattern}")
    statements = definitions(REWRITE_SQL.read_text())
    extended = cast(_ExtendedQuery, conn)
    cursor = conn.cursor()
    try:
        extended.execute_unnamed("BEGIN")
        for statement in statements:
            extended.execute_unnamed(statement)
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute("SHOW transaction_read_only")
        read_only = cursor.fetchone()
        if read_only is None or read_only[0] != "on":
            raise DatasetStoreError(f"transaction_read_only is {read_only!r}, not 'on'")
        cursor.execute(_statement(table), _params())
        row = cursor.fetchone()
    finally:
        cursor.close()
        conn.rollback()
    assert row is not None
    totals, by_format, samples, body, checks, census = cast(
        "tuple[list[int], dict[str, list[int]], list[list[str]], list[int], list[int], list[list]]",
        row,
    )
    names = [name for name, _code, _pattern in OLD_FORMATS]

    def _column(index: int) -> dict[str, int]:
        return {name: by_format.get(name, [0, 0, 0, 0])[index] for name in names}

    sampled: dict[str, list[tuple[str, str]]] = {name: [] for name in names}
    for name, before, after in samples:
        sampled[name].append((before, after))
    return DryRun(
        read_only=cast(str, read_only[0]),
        rewritten_rows=totals[0],
        rewritten=_column(0),
        left_rows=totals[1],
        left=_column(1),
        still_written_rows=totals[2],
        still_written=_column(2),
        residual_rows=totals[3],
        residual=_column(3),
        samples=sampled,
        body_mismatch_rows=body[0],
        body_rewritten_rows=body[1],
        member_count_equal_rows=checks[0],
        member_count_differs_rows=checks[1],
        member_count_without_trace_rows=checks[2],
        uncounted_conversion_rows=checks[3],
        census=[(code, rows, refs, unlisted) for code, rows, refs, unlisted in census],
    )


def findings(report: DryRun) -> list[str]:
    """What stops the rewrite: residual rows, and unlisted rows of a code outside
    `CENSUS_ALLOWED`."""
    found = []
    if report.residual_rows:
        found.append(f"{report.residual_rows} residual rows")
    found.extend(
        f"{unlisted} rows of {code} name a ref in no format"
        for code, _rows, _refs, unlisted in report.census
        if unlisted and code not in CENSUS_ALLOWED
    )
    return found


def _print(report: DryRun) -> None:
    print(f"transaction read only: {report.read_only}")
    for title, rows, by_format in (
        ("rows the SQL rewrite changes", report.rewritten_rows, report.rewritten),
        (
            "rows holding a member sentence or a code the SQL leaves to its writer",
            report.left_rows,
            report.left,
        ),
        (
            "rows holding a code the serving image still writes in its old shape",
            report.still_written_rows,
            report.still_written,
        ),
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
    print(
        "rows whose immutable_body warnings differ from their warnings column: "
        f"{report.body_mismatch_rows}"
    )
    print(
        f"  of these, rows whose body warnings the SQL rewrite changes: {report.body_rewritten_rows}"
    )
    print(
        f"rows whose member_count equals the trace's resolved_stat_vars: {report.member_count_equal_rows}"
    )
    print(
        f"rows whose member_count differs from the trace's resolved_stat_vars: {report.member_count_differs_rows}"
    )
    print(
        "rows holding a member_count beside a trace holding no resolved_stat_vars: "
        f"{report.member_count_without_trace_rows}"
    )
    print(
        "rows holding a currency or deflator warning that neither names nor counts its "
        f"entities: {report.uncounted_conversion_rows}"
    )
    print("by code: rows, rows naming a dcid: or slash-shaped ref, rows naming one in no format")
    for code, rows, refs, unlisted in report.census:
        print(f"  {code}: {rows}, {refs}, {unlisted}")


def main() -> int:
    """Run `python -m dataset_store.warning_formats`. Exits 0 when nothing stops the rewrite,
    1 on a finding (`findings`), and 2 when the run cannot complete."""
    try:
        settings = connection.ConnectionSettings.from_env(
            user_var="DATASET_STORE_MIGRATOR_DB_USER",
            password_var="DATASET_STORE_MIGRATOR_DB_PASSWORD",
            enable_iam_auth=False,
        )
        connector = connection.build_connector() if settings.instance_connection_name else None
        conn = settings.connect_callable(connector=connector)()
        try:
            report = dry_run(conn)
        finally:
            conn.close()
        _print(report)
        found = findings(report)
    except (DatasetStoreConfigError, DatasetStoreError, pg8000.exceptions.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # A broken run exits 2 so that 1 means only a finding.
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for finding in found:
        print(f"FINDING: {finding}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
