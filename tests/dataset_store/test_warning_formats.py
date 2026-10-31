"""`dataset_store.warning_formats`'s definitions against the messages old and current writers
store. The dry run itself is tested against Postgres in `test_warning_formats_live.py`."""

from __future__ import annotations

import re
from typing import cast

import pytest

from dataset_store.warning_formats import MEMBER_SENTENCE, OLD_FORMATS
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
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

_CURRENT_CASES = {"current", "unlisted"}


def _formats(code: str, message: str) -> list[str]:
    return [
        name
        for name, format_code, pattern in OLD_FORMATS
        if format_code == code and re.search(pattern, message)
    ]


def _current_shapes() -> list[tuple[str, str]]:
    """Messages in the shapes current writers store, and those the writers of
    `withheld_by_registry`, `un_system_unavailable` and `deflator_projected` will store."""
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
    withheld, un_unread, projected = (
        DATASETS_WARNING_CATALOGUE[code][1]
        for code in ("withheld_by_registry", "un_system_unavailable", "deflator_projected")
    )
    return [
        *(("coverage_below_scope", message) for message in [*coverage, composite]),
        *(
            (cast(str, warning["code"]), cast(str, warning["message"]))
            for warning in current_warnings()
        ),
        (
            "withheld_by_registry",
            f"{withheld} 3 entities served no observations; "
            "2 withheld provenances hold data there.",
        ),
        (
            "withheld_by_registry",
            f"{withheld} 1 entity served no observations; 1 withheld provenance holds data there.",
        ),
        ("un_system_unavailable", f"{un_unread} No UN data was read for 3 entities."),
        ("un_system_unavailable", f"{un_unread} No UN data was read for 1 entity."),
        ("deflator_projected", f"{projected} It applies to 4 rows."),
    ]


def test_each_code_has_one_format():
    codes = [code for _name, code, _pattern in OLD_FORMATS]
    assert len(set(codes)) == len(codes)


@pytest.mark.parametrize(("code", "message"), _current_shapes())
def test_a_current_shape_matches_no_format(code, message):
    assert [name for name, _code, pattern in OLD_FORMATS if re.search(pattern, message)] == []
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
    the SQL rewrite leaves as stored."""
    assert len(_formats(cast(str, warning["code"]), cast(str, warning["message"]))) == 1, case


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
