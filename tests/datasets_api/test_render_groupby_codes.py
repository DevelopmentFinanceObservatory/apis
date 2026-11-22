"""`groupby.codes` keeps the summed rows of the selected codes after every fold has read every
row, so a selected render refuses exactly when the unselected one does. Each render here
checks that the collapse it gives equals the collapse of the same render without a selection
then cut to the rows whose kept code is selected, frame for frame and byte for byte: rows,
values, units, methods and provenance. The frame fixture mixes units and party totals
(`test_render_collapse_frame_columnar`)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from dfo_frame_grammar import Frame, build_frame_from_arrow, canonical_frame_bytes

from datasets_api import render
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.query_options import QueryOptionsInvalid
from materialise.reshape import RollupNotSummableError

from .test_render_collapse_frame_columnar import (
    _ALL,
    _DEVELOPING,
    _GHANA,
    _KENYA,
    _MODALITY,
    _frame,
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


@pytest.mark.parametrize(
    ("methods", "levels", "axis"),
    [
        # Purpose 120's loan rows hold another method, which the finance fold adds to its
        # grant rows.
        pytest.param({"SV_120_421": "Other"}, None, "stat_var_crsTypeOfFinance", id="later-fold"),
        # Purpose 122 rolls up to 120, so the first fold adds their loan rows together.
        pytest.param({"SV_122_421": "Other"}, {"crsPurpose": 0}, "stat_var", id="first-fold"),
    ],
)
@pytest.mark.parametrize("codes", [None, {"crsPurpose": [_P111]}], ids=["unselected", "selected"])
def test_a_sum_refused_over_unselected_codes_refuses_with_a_selection(
    methods: dict[str, str],
    levels: dict[str, int] | None,
    axis: str,
    codes: dict[str, list[str]] | None,
):
    """Every fold reads every row, so a refusal in a group the selection leaves out refuses
    the selected render as it does the unselected one."""
    frame = _frame((_KENYA, _GHANA, _DEVELOPING), methods=methods)

    with pytest.raises(RollupNotSummableError) as raised:
        _render(frame, _options(["crsPurpose", "time_period"], codes=codes, levels=levels))

    assert (raised.value.axis, raised.value.precondition) == (axis, "facet_homogeneity")


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
