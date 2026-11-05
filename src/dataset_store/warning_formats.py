"""The stored warning formats that name refs in their message, and the count of version rows
holding one.

`OLD_FORMATS` holds the only definition of each format: one detection pattern per warning
code, valid in both Python `re` and Postgres ARE. Detection is loose: it asks whether a
message names something, so a message no rewrite can parse still counts. Migration 0047
(`migrations/0047_warning_counted_refs.sql`) rewrites every format but one: the old member
sentence (`MEMBER_SENTENCE`), whose split into off-cadence and out-of-window members needs
graph reads, is left to `datasets_api.warning_backfill`. `datasets-maintenance settle` counts
the rows still holding any format (`count_old_format_rows`) and fails the deploy when one
exists.
"""

from __future__ import annotations

from typing import cast

from dataset_store.store import Transaction

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

_OLD_FORMAT_ROWS = (
    "SELECT count(*) AS count FROM datasets.version_record AS v WHERE EXISTS ("
    "SELECT 1 FROM jsonb_array_elements("
    "CASE WHEN jsonb_typeof(v.warnings) = 'array' THEN v.warnings ELSE '[]'::jsonb END "
    "|| CASE WHEN jsonb_typeof(v.immutable_body -> 'warnings') = 'array' "
    "THEN v.immutable_body -> 'warnings' ELSE '[]'::jsonb END) AS w "
    "JOIN unnest(%s::text[], %s::text[]) AS f (code, pattern) ON f.code = w ->> 'code' "
    "WHERE w ->> 'message' ~ f.pattern)"
)


def count_old_format_rows(tx: Transaction) -> int:
    """The version rows holding a warning in its code's old format, in the `warnings` column
    or in the `warnings` of `immutable_body`."""
    row = tx.fetch_one(_OLD_FORMAT_ROWS, (list(OLD_FORMATS), list(OLD_FORMATS.values())))
    # `count(*)` without `GROUP BY` returns exactly one row.
    assert row is not None
    return int(cast(int, row["count"]))
