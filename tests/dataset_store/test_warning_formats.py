"""`dataset_store.warning_formats`'s definitions against the messages old and current writers
store. Migration 0047 and the count `settle` reads are tested against Postgres in
`test_warning_formats_live.py`."""

from __future__ import annotations

import re
from typing import cast

import pytest

from dataset_store.warning_formats import MEMBER_SENTENCE, OLD_FORMATS
from materialise.finish import _coverage_below_scope_warning
from tests.dataset_store._warning_refs import (
    COVERAGE_DETAIL,
    MANY_MEMBERS,
    MEMBERS,
    UNSERVED,
    current_warnings,
    new_coverage,
    old_member_sentence,
    rewrite_cases,
)

_CURRENT_CASES = {"current", "unlisted", "revision_unavailable"}


def _matches(code: str, message: str) -> bool:
    return code in OLD_FORMATS and re.search(OLD_FORMATS[code], message) is not None


def _current_shapes() -> list[tuple[str, str]]:
    """Messages in the shapes current writers store."""
    coverage = [
        cast(str, warning["message"])
        for warning in (
            new_coverage(unserved=UNSERVED[:5]),
            new_coverage(unserved=UNSERVED[:1]),
            new_coverage(members=MEMBERS, member_count=4),
            _coverage_below_scope_warning(
                UNSERVED,
                countries_fallback=True,
                off_cadence=MEMBERS[:1],
                out_of_window=MEMBERS[1:2],
                unserved_members=MEMBERS[2:],
                member_count=7,
                years=({"start": 2020, "end": 2030}, {"start": 2020, "end": 2024}),
            ),
        )
    ]
    composite = f"{coverage[0]} Of these, the donor component leaves out 3."
    return [
        *(("coverage_below_scope", message) for message in [*coverage, composite]),
        *(
            (cast(str, warning["code"]), cast(str, warning["message"]))
            for warning in current_warnings()
        ),
    ]


@pytest.mark.parametrize(("code", "message"), _current_shapes())
def test_a_current_shape_matches_no_format(code, message):
    assert not _matches(code, message)
    assert re.search(MEMBER_SENTENCE, message) is None


@pytest.mark.parametrize(
    ("case", "warning"),
    [
        (case, warning)
        for case, (stored, _rewritten) in rewrite_cases().items()
        if case not in _CURRENT_CASES
        for warning in stored
        if warning["code"] != "empty_result"
    ],
)
def test_each_old_format_matches_its_code_s_format(case, warning):
    """Every old message matches its code's format, a name holding a comma included, which
    migration 0047 leaves as stored."""
    assert _matches(cast(str, warning["code"]), cast(str, warning["message"])), case


@pytest.mark.parametrize(
    ("members", "range_window", "named", "more"),
    [
        (MEMBERS, False, MEMBERS, None),
        (MANY_MEMBERS, True, MANY_MEMBERS[:5], "1"),
    ],
)
def test_the_member_sentence_reads_the_named_members_and_the_rest(
    members, range_window, named, more
):
    message = (
        COVERAGE_DETAIL
        + " No observations were served for 7 requested entities."
        + old_member_sentence(members, range_window=range_window)
    )
    match = re.search(MEMBER_SENTENCE, message)
    assert match is not None
    assert match[1].split(", ") == named
    assert match[2] == more
    assert (match[3] is not None) == range_window
