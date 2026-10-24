"""Tests `groupby.levels`: a kept CRS purpose axis rolled up to one depth before the sum.

The fixture mirrors CRS: `crsPurpose` is ragged and disjoint from its children, so the
parent-coded StatVar `SV_SOCIAL` holds records none of its child-coded StatVars hold. At depth 0
social infrastructure is 4 + 10 + 5 = 19, economic infrastructure 7, and unallocated 2.
"""

from __future__ import annotations

import dfo_frame_grammar
import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import ReshapeResult, _RenderReads, _reshape
from materialise.content_version import mint_version
from materialise.enrichment import EnrichmentLookup
from materialise.frame_build import build_frame
from materialise.reshape import Table

from ._graph_doubles import FakeMintGraphReader

_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-10-05T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)


def _purpose(code: str, label: str, parent: str | None) -> CodeRow:
    return CodeRow(
        dcid=f"CRSPurposeCode{code}",
        code_system="crsPurpose",
        code=code,
        label=label,
        synonyms=(),
        parent=None if parent is None else f"CRSPurposeCode{parent}",
    )


def _finance(code: str, label: str) -> CodeRow:
    return CodeRow(
        dcid=f"CRSTypeOfFinance{code}",
        code_system="crsTypeOfFinance",
        code=code,
        label=label,
        synonyms=(),
        parent=None,
    )


_CODES = (
    _purpose("100", "Social infrastructure", None),
    _purpose("120", "Health", "100"),
    _purpose("12250", "Malaria control", "120"),
    _purpose("200", "Economic infrastructure", None),
    _purpose("210", "Transport", "200"),
    _purpose("998", "Unallocated", None),
    _finance("110", "Grant"),
    _finance("421", "Loan"),
)

_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="sector",
    ),
    CodeSystemDeclaration(
        code_system="crsTypeOfFinance",
        hierarchical=False,
        disjoint_from_children=False,
        summable=True,
        absent_reading="not_declared",
        plain_name="instrument",
    ),
)


def _statvar(dcid: str, *, purpose: str, finance: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties={
            "crsPurpose": f"CRSPurposeCode{purpose}",
            "crsTypeOfFinance": f"CRSTypeOfFinance{finance}",
        },
    )


_INDEX = CodeIndex(
    stamp=_STAMP,
    declarations=_DECLARATIONS,
    codes=_CODES,
    statvars=(
        _statvar("SV_SOCIAL", purpose="100", finance="110"),
        _statvar("SV_HEALTH", purpose="120", finance="110"),
        _statvar("SV_MALARIA", purpose="12250", finance="421"),
        _statvar("SV_TRANSPORT", purpose="210", finance="110"),
        _statvar("SV_UNALLOCATED", purpose="998", finance="110"),
    ),
)

_ENTITY_COLUMNS = ("financingProvider", "financingRecipient")
# Rows sit in frame order, sorted by `stat_var`, as the streamed fold requires.
_TABLE = Table(
    columns=(
        "stat_var",
        "financingProvider",
        "financingRecipient",
        "time_period",
        "value",
        "unit",
        "provenance",
    ),
    rows=(
        ("dcid:SV_HEALTH", "country/USA", "country/KEN", "2024", 10.0, "USD", "p1"),
        ("dcid:SV_MALARIA", "country/USA", "country/KEN", "2024", 5.0, "USD", "p1"),
        ("dcid:SV_SOCIAL", "country/USA", "country/KEN", "2024", 4.0, "USD", "p1"),
        ("dcid:SV_TRANSPORT", "country/USA", "country/KEN", "2024", 7.0, "USD", "p1"),
        ("dcid:SV_UNALLOCATED", "country/USA", "country/KEN", "2024", 2.0, "USD", "p1"),
    ),
)
_SUBJECT_ID = "spec_991000"


def _options(levels: dict[str, int] | None) -> MintOptions:
    return MintOptions(
        groupby=GroupbyOptions(by=["time_period", "crsPurpose"], op="sum", levels=levels)
    )


def _reshape_with(
    options: MintOptions, *, chunked: bool = False, table: Table = _TABLE
) -> ReshapeResult:
    return _reshape(
        table,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        reads=_RenderReads(
            enrichment=EnrichmentLookup(FakeMintGraphReader(), parties=lambda: {}),
            known={},
            deadline=None,
        ),
        code_index=_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        subject_id=_SUBJECT_ID,
        frame_chunks=iter((table,)) if chunked else None,
        pivot_max_output_cells=5_000_000,
    )


def _by_sector(result: ReshapeResult) -> dict[str, float]:
    table = result.table
    sector_pos = table.columns.index("stat_var_crsPurpose")
    value_pos = table.columns.index("value")
    return {row[sector_pos]: row[value_pos] for row in table.rows}


@pytest.mark.parametrize("chunked", [False, True], ids=["whole-table", "streamed"])
def test_level_zero_sums_parent_and_child_coded_stat_vars_into_their_root_sector(chunked: bool):
    result = _reshape_with(_options({"crsPurpose": 0}), chunked=chunked)

    assert _by_sector(result) == {
        "Social infrastructure": 19.0,
        "Economic infrastructure": 7.0,
        "Unallocated": 2.0,
    }


def test_a_code_shallower_than_the_level_stays_as_itself_under_its_own_label():
    result = _reshape_with(_options({"crsPurpose": 1}))

    # 100 and 998 are root codes, so at depth 1 each keeps its own records and label.
    assert _by_sector(result) == {
        "Social infrastructure": 4.0,
        "Health": 15.0,
        "Transport": 7.0,
        "Unallocated": 2.0,
    }


def test_a_level_roll_up_is_a_grammar_frame_with_the_family_ref_and_a_sector_role():
    result = _reshape_with(_options({"crsPurpose": 0}))

    assert result.collapse is not None
    frame = result.collapse.frame
    assert isinstance(frame, dfo_frame_grammar.Frame)
    assert frame.columns == (
        "stat_var",
        "financingProvider",
        "financingRecipient",
        "stat_var_crsPurpose",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    stat_var_pos = frame.columns.index("stat_var")
    sector_pos = frame.columns.index("stat_var_crsPurpose")
    assert {row[stat_var_pos] for row in frame.rows} == {f"dcid:{_SUBJECT_ID}"}
    assert all(isinstance(row[sector_pos], str) and row[sector_pos] for row in frame.rows)
    assert len(frame.rows) == 3


@pytest.mark.parametrize("levels", [None, {"crsPurpose": 0}], ids=["kept", "level"])
def test_a_sole_contributor_frame_that_keeps_a_code_axis_names_the_family(
    levels: dict[str, int] | None,
):
    # Only SV_MALARIA has rows. The frame keeps a sector column, so it is a new frame and
    # names the family.
    sole = Table(columns=_TABLE.columns, rows=(_TABLE.rows[1],))
    result = _reshape_with(_options(levels), table=sole)

    assert result.collapse is not None
    assert result.collapse.stat_var_ref == f"dcid:{_SUBJECT_ID}"
    frame = result.collapse.frame
    stat_var_pos = frame.columns.index("stat_var")
    assert {row[stat_var_pos] for row in frame.rows} == {f"dcid:{_SUBJECT_ID}"}
    assert result.warnings == ()


def test_each_level_mints_its_own_content_version_apart_from_the_separate_frame():
    separate = build_frame(
        role_keys=_ENTITY_COLUMNS,
        rows=[dict(zip(_TABLE.columns, row, strict=True)) for row in _TABLE.rows],
    )
    separate_version = mint_version(separate)

    versions = set()
    for levels in (None, {"crsPurpose": 0}, {"crsPurpose": 1}):
        collapse = _reshape_with(_options(levels)).collapse
        assert collapse is not None
        versions.add(collapse.content_version)

    assert len(versions) == 3
    assert separate_version not in versions
