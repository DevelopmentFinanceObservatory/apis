"""`groupby.codes` sums the selected groups: the checks before the folds read every row, and
the folds and their preconditions read the selected groups alone. Where the render without a
selection answers, the selected render gives its collapse cut to the rows whose kept code is
selected, frame for frame and byte for byte: rows, values, units, methods and provenance. A
fold fault in a group the selection leaves out refuses nothing; the same fault in a selected
group refuses as the render without a selection does. The frame fixture mixes units and party
totals (`test_render_collapse_frame_columnar`)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from dfo_frame_grammar import Frame, build_frame_from_arrow, canonical_frame_bytes

from code_index import CodeIndex
from datasets_api import render
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.query_options import QueryOptionsInvalid
from materialise.enrichment import EnrichmentLookup
from materialise.groupby_columnar import ColumnarFoldOutcome
from materialise.reshape import RollupNotSummableError

from ._graph_doubles import FakeMintGraphReader
from .test_render_collapse_frame_columnar import (
    _ALL,
    _DEVELOPING,
    _ENTITY_COLUMNS,
    _FINANCES,
    _GHANA,
    _INDEX,
    _KENYA,
    _MODALITY,
    _RESOLVED,
    _STAT_VARS,
    _frame,
    _purpose,
    _render,
)

_P111, _P120, _P122 = "dcid:CRSPurposeCode111", "dcid:CRSPurposeCode120", "dcid:CRSPurposeCode122"
_LOAN = "dcid:CRSTypeOfFinanceCode421"
# A purpose code of the index that no StatVar holds.
_ABSENT = "dcid:CRSPurposeCode130"


def _options(
    by: list[str],
    *,
    codes: dict[str, list[str]] | None = None,
    levels: dict[str, int] | None = None,
    cut: bool = False,
) -> MintOptions:
    return MintOptions(
        format="csv",
        bundle=False,
        groupby=GroupbyOptions(by=by, op="sum", levels=levels, codes=codes),
        without_party_totals=cut,
    )


def _collapse(options: MintOptions) -> render.GroupbyCollapseRendering:
    rendered, _body = _render(_ALL, options)
    assert rendered.collapse is not None
    return rendered.collapse


def _cut_to(collapse: render.GroupbyCollapseRendering, codes: dict[str, list[str]]) -> Frame:
    """The collapse's frame with only the rows whose `stat_var_<axis>` cell is listed."""
    table = collapse.frame.table
    mask = pa.array([True] * table.num_rows)
    for axis, listed in codes.items():
        mask = pc.and_(mask, pc.is_in(table[f"stat_var_{axis}"], value_set=pa.array(listed)))
    # The build derives `status` itself, so the canonical table's column goes back out.
    kept = table.filter(mask).drop_columns(["status"])
    return build_frame_from_arrow(kept, role_keys=collapse.role_columns)


@pytest.mark.parametrize(
    ("by", "codes", "levels", "cut", "rows"),
    [
        pytest.param(
            ["crsPurpose", "time_period"],
            {"crsPurpose": [_P122, _P111]},
            None,
            False,
            24,
            id="kept-axis",
        ),
        pytest.param(
            ["crsPurpose", "time_period"],
            {"crsPurpose": [_P120]},
            {"crsPurpose": 0},
            False,
            12,
            id="level-roll-up",
        ),
        pytest.param(
            ["crsPurpose", "crsTypeOfFinance", "time_period"],
            {"crsTypeOfFinance": [_LOAN], "crsPurpose": [_P111, _P120]},
            None,
            True,
            20,
            id="party-totals-cut-two-axes",
        ),
        pytest.param(
            ["crsPurpose", "time_period"],
            {"crsPurpose": [_P111, _ABSENT]},
            None,
            False,
            12,
            id="absent-code",
        ),
        pytest.param(
            ["crsPurpose", "time_period"],
            {"crsPurpose": [_ABSENT]},
            None,
            False,
            0,
            id="only-absent-codes",
        ),
    ],
)
def test_a_selection_before_the_sum_equals_the_sum_cut_to_the_selected_codes(
    by: list[str],
    codes: dict[str, list[str]],
    levels: dict[str, int] | None,
    cut: bool,
    rows: int,
):
    selected = _collapse(_options(by, codes=codes, levels=levels, cut=cut))
    expected = _cut_to(_collapse(_options(by, levels=levels, cut=cut)), codes)

    assert selected.frame.num_rows == rows
    assert selected.frame == expected
    assert selected.frame.table.equals(expected.table)
    assert canonical_frame_bytes(selected.frame) == canonical_frame_bytes(expected)
    assert selected.frame_sha256 != _collapse(_options(by, levels=levels, cut=cut)).frame_sha256


def test_a_selected_render_folds_only_the_selected_rows(monkeypatch):
    """The first fold reads only the selected rows; the render without a selection reads
    every row."""
    folded: list[int] = []
    fold = render.fold_groupby_sum_table

    def counted(table: pa.Table, **kwargs: Any) -> ColumnarFoldOutcome:
        folded.append(table.num_rows)
        return fold(table, **kwargs)

    monkeypatch.setattr(render, "fold_groupby_sum_table", counted)
    by = ["crsPurpose", "time_period"]

    _collapse(_options(by))
    _collapse(_options(by, codes={"crsPurpose": [_P111]}))

    every_row, selected = folded
    assert every_row == _ALL.num_rows
    assert 0 < selected < every_row


_BIG = 1.7e308
# Each fold fault sits in purpose 120's groups. The parameters name the fault, the selection
# that leaves purpose 120 out, and the axis whose fold the fault refuses.
_FOLD_FAULTS = [
    # Purpose 120's loan rows hold another method, which the finance fold adds to its grant
    # rows.
    pytest.param(
        {"methods": {"SV_120_421": "Other"}},
        None,
        "stat_var_crsTypeOfFinance",
        "facet_homogeneity",
        id="facet-later-fold",
    ),
    # Purpose 122 rolls up to 120, so the first fold adds their loan rows together.
    pytest.param(
        {"methods": {"SV_122_421": "Other"}},
        {"crsPurpose": 0},
        "stat_var",
        "facet_homogeneity",
        id="facet-first-fold",
    ),
    # Purpose 120's grant and loan totals pass the largest double when the finance fold adds
    # them.
    pytest.param(
        {"values": {"SV_120_110": _BIG, "SV_120_421": _BIG}},
        None,
        "stat_var_crsTypeOfFinance",
        "finite_total",
        id="not-finite-later-fold",
    ),
    pytest.param(
        {"values": {"SV_120_110": _BIG, "SV_122_110": _BIG}},
        {"crsPurpose": 0},
        "stat_var",
        "finite_total",
        id="not-finite-first-fold",
    ),
]


@pytest.mark.parametrize(("fault", "levels", "axis", "precondition"), _FOLD_FAULTS)
def test_a_fold_fault_in_a_group_the_selection_leaves_out_refuses_nothing(
    fault: dict, levels: dict[str, int] | None, axis: str, precondition: str
):
    """The render without a selection refuses; the render of purpose 111 alone answers with
    the rows the faultless frame gives purpose 111."""
    frame = _frame((_KENYA, _GHANA, _DEVELOPING), **fault)
    by = ["crsPurpose", "time_period"]
    codes = {"crsPurpose": [_P111]}

    with pytest.raises(RollupNotSummableError) as raised:
        _render(frame, _options(by, levels=levels))
    rendered, _body = _render(frame, _options(by, codes=codes, levels=levels))

    assert (raised.value.axis, raised.value.precondition) == (axis, precondition)
    assert rendered.collapse is not None
    assert rendered.collapse.frame == _cut_to(_collapse(_options(by, levels=levels)), codes)


@pytest.mark.parametrize(("fault", "levels", "axis", "precondition"), _FOLD_FAULTS)
def test_a_collapse_of_rows_in_memory_selects_before_its_folds(
    fault: dict, levels: dict[str, int] | None, axis: str, precondition: str
):
    """A collapse of a table held as rows, with no frame slices to stream, also folds the
    selected rows alone, and gives the streamed collapse's frame."""
    frame = _frame((_KENYA, _GHANA, _DEVELOPING), **fault)
    by = ["crsPurpose", "time_period"]
    codes = {"crsPurpose": [_P111]}
    options = _options(by, codes=codes, levels=levels)

    in_memory = render._reshape(
        frame,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        reads=render._RenderReads(
            enrichment=EnrichmentLookup(FakeMintGraphReader()),
            parties=_INDEX.parties,
            known={},
            deadline=None,
        ),
        pivot_max_output_cells=5_000_000,
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED,
        provenance_refs=["dcid:CRS"],
        subject_id="spec_columnar",
    )

    assert in_memory.collapse is not None
    assert in_memory.collapse.frame == _collapse(options).frame


@pytest.mark.parametrize(("fault", "levels", "axis", "precondition"), _FOLD_FAULTS)
def test_a_fold_fault_in_a_selected_group_refuses_as_the_unselected_render(
    fault: dict, levels: dict[str, int] | None, axis: str, precondition: str
):
    frame = _frame((_KENYA, _GHANA, _DEVELOPING), **fault)
    by = ["crsPurpose", "time_period"]
    refusals = []
    for codes in (None, {"crsPurpose": [_P120, _P111]}):
        with pytest.raises(RollupNotSummableError) as raised:
            _render(frame, _options(by, codes=codes, levels=levels))
        refusals.append((raised.value.code, raised.value.axis, raised.value.precondition))

    assert refusals[0] == refusals[1]
    assert refusals[0][1:] == (axis, precondition)


# Purpose 140's parent, 150, has no code row, so a roll-up to depth 0 finds no code for it.
_DANGLING = _purpose("140", "150")
_DANGLING_STAT_VARS = tuple((f"SV_140_{f.code}", _DANGLING, f) for f in _FINANCES)
_DANGLING_INDEX = CodeIndex(
    stamp=_INDEX.stamp,
    declarations=tuple(_INDEX.declarations.values()),
    datasets=tuple(_INDEX.datasets.values()),
    codes=(*_INDEX.codes, _DANGLING),
    statvars=(
        *_INDEX.statvars,
        *(
            dataclasses.replace(
                _INDEX.statvars[0],
                dcid=dcid,
                constraint_properties={"crsPurpose": purpose.dcid, "crsTypeOfFinance": f.dcid},
            )
            for dcid, purpose, f in _DANGLING_STAT_VARS
        ),
    ),
    parties=tuple(_INDEX.parties.values()),
)


@pytest.mark.parametrize(
    ("render_arguments", "levels", "precondition"),
    [
        pytest.param(
            {
                "frame": _frame(
                    (_KENYA, _GHANA, _DEVELOPING), stat_vars=(*_STAT_VARS, *_DANGLING_STAT_VARS)
                ),
                "code_index": _DANGLING_INDEX,
                "resolved": (*_RESOLVED, *(f"dcid:{dcid}" for dcid, *_ in _DANGLING_STAT_VARS)),
            },
            {"crsPurpose": 0},
            "index_gap",
            id="level-roll-up",
        ),
        pytest.param(
            {"frame": _ALL, "resolved": (*_RESOLVED, "dcid:SV_Unindexed")},
            None,
            "index_gap",
            id="decode-gap",
        ),
        pytest.param(
            {"frame": _ALL, "provenance_refs": ("dcid:CRS", "dcid:OtherSource")},
            None,
            "provenance_disagreement",
            id="provenance",
        ),
    ],
)
def test_a_check_before_the_folds_refuses_a_selected_render(
    render_arguments: dict, levels: dict[str, int] | None, precondition: str
):
    """The checks before the folds read every StatVar, so a fault the selection leaves out
    still refuses."""
    refusals = []
    for codes in (None, {"crsPurpose": [_P111]}):
        options = _options(["crsPurpose", "time_period"], codes=codes, levels=levels)
        with pytest.raises(RollupNotSummableError) as raised:
            _render(options=options, **render_arguments)
        refusals.append((raised.value.code, raised.value.axis, raised.value.precondition))

    assert refusals[0] == refusals[1]
    assert refusals[0][2] == precondition


def test_a_selection_on_an_axis_the_frame_does_not_decode_refuses():
    """No StatVar of the frame carries `crsCooperationModality`, so its rows hold no code to
    select on: the selection refuses, where it would otherwise keep every row."""
    options = _options(
        ["crsCooperationModality", "crsPurpose", "time_period"],
        codes={"crsCooperationModality": [_MODALITY.dcid]},
    )

    with pytest.raises(QueryOptionsInvalid) as raised:
        _render(_ALL, options)

    assert raised.value.pointer == "/options/groupby/codes/crsCooperationModality"
