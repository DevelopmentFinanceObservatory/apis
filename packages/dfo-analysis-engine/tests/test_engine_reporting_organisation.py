"""Every verb reads `reporting_organisation`, the member provenance that reported a row of a
declared dataset, as an attribute: it names no entity, an output row keeps the member its input
rows share and holds null where it folds rows of two members, and a join's row keeps its own
side's member. A frame without the column builds the output it built before the column existed.

The root is a two-member Ledger: `dataset/Ledger` holds the series of `provenance/LedgerNorth`
and `provenance/LedgerSouth`, one member per donor.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
import pytest
from dfo_frame_grammar import Frame, build_frame_from_arrow
from dfo_frame_grammar.grammar import REPORTING_ORGANISATION_COLUMN

from dfo_analysis_engine.engine.aggregate import UnknownGroupByColumnError, run_aggregate
from dfo_analysis_engine.engine.compare import run_compare
from dfo_analysis_engine.engine.filter import run_filter
from dfo_analysis_engine.engine.growth import run_growth
from dfo_analysis_engine.engine.join import run_join
from dfo_analysis_engine.engine.paired import run_paired_sum_of
from dfo_analysis_engine.engine.quotient_of import run_quotient_of
from dfo_analysis_engine.engine.rank import run_rank
from dfo_analysis_engine.engine.share_of import run_share_of
from dfo_analysis_engine.engine.sum_of import run_sum_of

_LEDGER = "dataset/Ledger"
_NORTH = "provenance/LedgerNorth"
_SOUTH = "provenance/LedgerSouth"
_MEMBER_OF = {"dcid:A": _NORTH, "dcid:B": _SOUTH}


def _root(
    stat_var: str = "dcid:X", *, members: bool = True, member_of: dict[str, str] = _MEMBER_OF
) -> Frame:
    """A root over the Ledger, as the observation read builds one, each donor's rows reported
    by its `member_of` member, or, without `members`, the same rows of a plain table, which
    carry no member column."""
    donors, periods, values = [], [], []
    for donor in _MEMBER_OF:
        for period, value in (("2020", 1.0), ("2021", 2.0)):
            donors.append(donor)
            periods.append(period)
            values.append(value)
    count = len(donors)
    columns: dict[str, list[object]] = {
        "stat_var": [stat_var] * count,
        "donor": donors,
        "time_period": periods,
        "value": values,
        "unit": ["dcid:USDollar"] * count,
        "provenance": [_LEDGER if members else "provenance/Plain"] * count,
    }
    if members:
        columns[REPORTING_ORGANISATION_COLUMN] = [member_of[donor] for donor in donors]
    return build_frame_from_arrow(pa.table(columns), role_keys=["donor"])


def _members(frame: Frame) -> list[object]:
    return frame.table.column(REPORTING_ORGANISATION_COLUMN).to_pylist()


def _members_by_donor(frame: Frame) -> dict[str, object]:
    return dict(
        zip(
            frame.table.column("donor").to_pylist(),
            _members(frame),
            strict=True,
        )
    )


_A, _B = {"donor": "dcid:A"}, {"donor": "dcid:B"}

# Each verb over the Ledger root, and the member every output row holds: the donor's own
# where a row reads one donor's rows, and null where it reads both donors'.
_VERBS: dict[str, tuple[Callable[[Frame], Frame], list[object] | dict[str, object]]] = {
    "filter": (lambda f: run_filter(f, predicates=[]).frame, [_NORTH, _NORTH, _SOUTH, _SOUTH]),
    "aggregate": (
        lambda f: run_aggregate(f, group_by=["donor"], aggregation="sum").frame,
        _MEMBER_OF,
    ),
    "rank": (
        lambda f: (
            run_rank(
                f,
                rank_by=["donor"],
                aggregation="sum",
                window_start=None,
                window_end=None,
                ascending=False,
            ).frame
        ),
        _MEMBER_OF,
    ),
    "rank per period": (
        lambda f: (
            run_rank(
                f,
                rank_by=["donor"],
                aggregation="sum",
                window_start=None,
                window_end=None,
                ascending=False,
                per_period=True,
            ).frame
        ),
        [_NORTH, _NORTH, _SOUTH, _SOUTH],
    ),
    "growth": (
        lambda f: run_growth(f, window_start="2020", window_end="2021", entity_by=["donor"]).frame,
        _MEMBER_OF,
    ),
    "compare across members": (
        lambda f: run_compare(f, f, stat_var="dcid:X", left_entity=_A, right_entity=_B).frame,
        [None, None],
    ),
    "sum_of within a member": (
        lambda f: run_sum_of(f, f, stat_var="dcid:X", left_entity=_A, right_entity=_A).frame,
        [_NORTH, _NORTH],
    ),
    "share_of over every member": (
        lambda f: (
            run_share_of(f, f, stat_var="dcid:X", numerator_entity=_A, denominator_entity={}).frame
        ),
        [None, None],
    ),
    "share_of within a member": (
        lambda f: (
            run_share_of(f, f, stat_var="dcid:X", numerator_entity=_B, denominator_entity=_B).frame
        ),
        [_SOUTH, _SOUTH],
    ),
    "quotient_of keyed by donor": (
        lambda f: (
            run_quotient_of(
                f,
                f,
                stat_var="dcid:X",
                numerator_entity={},
                denominator_entity={},
                numerator_by=["donor"],
                denominator_by={"donor": "donor"},
            ).frame
        ),
        [_NORTH, _NORTH, _SOUTH, _SOUTH],
    ),
    "paired sum_of": (
        lambda f: (
            run_paired_sum_of(
                f, f, stat_var="dcid:X", on={"donor": "donor"}, left_entity={}, right_entity={}
            ).frame
        ),
        [_NORTH, _NORTH, _SOUTH, _SOUTH],
    ),
}


@pytest.mark.parametrize("verb", list(_VERBS))
def test_a_verb_over_a_ledger_root_carries_each_row_s_shared_member(verb: str) -> None:
    run, expected = _VERBS[verb]

    output = run(_root())

    assert (
        REPORTING_ORGANISATION_COLUMN not in output.columns[: output.columns.index("time_period")]
    )
    if isinstance(expected, dict):
        assert _members_by_donor(output) == expected
    else:
        assert _members(output) == expected


@pytest.mark.parametrize("verb", list(_VERBS))
def test_a_verb_over_a_plain_table_writes_no_member_column(verb: str) -> None:
    run, _ = _VERBS[verb]

    assert REPORTING_ORGANISATION_COLUMN not in run(_root(members=False)).columns


@pytest.mark.parametrize(
    ("group_by", "expected"),
    [((), [None]), (("donor",), [_NORTH, _SOUTH]), (("time_period",), [None, None])],
)
def test_an_aggregate_keeps_a_member_only_within_it(
    group_by: tuple[str, ...], expected: list[object]
) -> None:
    output = run_aggregate(_root(), group_by=group_by, aggregation="sum").frame

    assert _members(output) == expected


def test_the_member_column_keys_no_group() -> None:
    with pytest.raises(UnknownGroupByColumnError) as raised:
        run_aggregate(_root(), group_by=[REPORTING_ORGANISATION_COLUMN], aggregation="sum")

    assert raised.value.columns == (REPORTING_ORGANISATION_COLUMN,)


@pytest.mark.parametrize(
    ("right", "expected_right"),
    [
        (_root("dcid:Y", member_of={"dcid:A": _SOUTH, "dcid:B": _NORTH}), [_SOUTH, _NORTH]),
        (_root("dcid:Y", members=False), [None, None]),
    ],
    ids=["other members", "plain table"],
)
def test_a_join_row_keeps_its_own_side_s_member(right: Frame, expected_right: list[object]) -> None:
    output = run_join(_root(), right).frame

    members: dict[tuple[str, str], set[object]] = {}
    for stat_var, donor, member in zip(
        output.table.column("stat_var").to_pylist(),
        output.table.column("donor").to_pylist(),
        _members(output),
        strict=True,
    ):
        members.setdefault((stat_var, donor), set()).add(member)
    assert members == {
        ("dcid:X", "dcid:A"): {_NORTH},
        ("dcid:X", "dcid:B"): {_SOUTH},
        ("dcid:Y", "dcid:A"): {expected_right[0]},
        ("dcid:Y", "dcid:B"): {expected_right[1]},
    }
