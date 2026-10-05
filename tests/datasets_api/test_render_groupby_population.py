"""Tests that groupby sums refuse incomplete populations, including missing rows, index gaps,
and completeness checks across sequential rollups.

The fixtures use a synthetic `CodeIndex` with two summable axes.
"""

from __future__ import annotations

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import _reshape
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import RollupNotSummableError, Table

from ._graph_doubles import FakeMintGraphReader

_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-09T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)

_PURPOSE_HEALTH = CodeRow(
    dcid="dcid:CRSPurposeCode122",
    code_system="crsPurpose",
    code="122",
    label="Basic health care",
    synonyms=(),
    parent=None,
)
_PURPOSE_MALARIA = CodeRow(
    dcid="dcid:CRSPurposeCode12250",
    code_system="crsPurpose",
    code="12250",
    label="Malaria control",
    synonyms=(),
    parent=None,
)
_FINANCE_GRANT = CodeRow(
    dcid="dcid:CRSTypeOfFinanceCode110",
    code_system="crsTypeOfFinance",
    code="110",
    label="Standard grant",
    synonyms=(),
    parent=None,
)
_FINANCE_LOAN = CodeRow(
    dcid="dcid:CRSTypeOfFinanceCode421",
    code_system="crsTypeOfFinance",
    code="421",
    label="Standard loan",
    synonyms=(),
    parent=None,
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
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="instrument",
    ),
    # Declared summable but never decoded by any StatVar below -- naming it
    # in `by` keeps every real axis collapsing (`collapse_axes`'s own
    # no-op-for-an-undecoded-name reading) without `by` ever being empty
    # (`GroupbyOptions.by` requires at least one entry).
    CodeSystemDeclaration(
        code_system="crsCooperationModality",
        hierarchical=False,
        disjoint_from_children=False,
        summable=True,
        absent_reading="not_declared",
        plain_name="modality",
    ),
)


def _index(*, statvars: tuple[StatVarRow, ...]) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT, _FINANCE_LOAN),
        statvars=statvars,
    )


def _statvar(dcid: str, *, purpose: str, finance: str | None = None) -> StatVarRow:
    constraint_properties = {"crsPurpose": purpose}
    if finance is not None:
        constraint_properties["crsTypeOfFinance"] = finance
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties=constraint_properties,
    )


_ENTITY_COLUMNS = ("financingProvider", "financingRecipient")
_TABLE_COLUMNS = (
    "stat_var",
    "financingProvider",
    "financingRecipient",
    "time_period",
    "value",
    "unit",
    "provenance",
)


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader())


def _single_axis_by() -> GroupbyOptions:
    # Keeping the undecoded modality axis makes `stat_var` and `crsPurpose` collapse in sequence.
    return GroupbyOptions(by=["crsCooperationModality"], op="sum")


# Zero-row resolved member.


def test_zero_row_resolved_member_refuses_and_names_it():
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),),
    )
    index = _index(
        statvars=(
            _statvar("SV_health", purpose=_PURPOSE_HEALTH.dcid),
            _statvar("SV_malaria", purpose=_PURPOSE_MALARIA.dcid),
        )
    )
    options = MintOptions(groupby=_single_axis_by())

    try:
        _reshape(
            table,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=index,
            # `dcid:SV_malaria` never contributed a row (an entirely
            # zero-row resolved member), but the request still resolved it.
            resolved_stat_var_refs=("dcid:SV_health", "dcid:SV_malaria"),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "missing_members"
        assert "Malaria control" in str(exc)
    else:
        raise AssertionError("expected RollupNotSummableError")


def test_without_the_zero_row_member_resolved_the_same_frame_passes():
    # Isolates the refusal above to the population, not the table shape:
    # when the request never resolved `SV_malaria` at all, the identical
    # table and code index pass cleanly.
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),),
    )
    index = _index(statvars=(_statvar("SV_health", purpose=_PURPOSE_HEALTH.dcid),))
    options = MintOptions(groupby=_single_axis_by())

    reshaped = _reshape(
        table,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=index,
        resolved_stat_var_refs=("dcid:SV_health",),
        provenance_refs=("p1",),
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.warnings == ()
    value_pos = reshaped.table.columns.index("value")
    assert [row[value_pos] for row in reshaped.table.rows] == [10.0]


# Index-gap StatVars.


def test_two_index_gap_statvars_refuse_before_merging():
    # Neither gap has a `StatVarRow`; both would decode to the same empty value and merge.
    # `SV_health` keeps `crsPurpose` present in the frame.
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_health", "country/USA", "country/KEN", "2022", 20.0, "USD", "p1"),
            ("dcid:SV_gap1", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
            ("dcid:SV_gap2", "country/USA", "country/KEN", "2022", 5.0, "USD", "p1"),
        ),
    )
    index = _index(statvars=(_statvar("SV_health", purpose=_PURPOSE_HEALTH.dcid),))
    options = MintOptions(groupby=_single_axis_by())

    try:
        _reshape(
            table,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=index,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "index_gap"
        assert "dcid:SV_gap1" in str(exc)
        assert "dcid:SV_gap2" in str(exc)
    else:
        raise AssertionError("expected RollupNotSummableError")


# Member rows missing for one recipient.


def test_reviewer_repro_one_member_row_removed_refuses():
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
            ("dcid:SV_health", "country/USA", "country/TZA", "2022", 8.0, "USD", "p1"),
            ("dcid:SV_malaria", "country/USA", "country/KEN", "2022", 4.0, "USD", "p1"),
            # SV_malaria has a Kenya row but no Tanzania row.
        ),
    )
    index = _index(
        statvars=(
            _statvar("SV_health", purpose=_PURPOSE_HEALTH.dcid),
            _statvar("SV_malaria", purpose=_PURPOSE_MALARIA.dcid),
        )
    )
    options = MintOptions(groupby=_single_axis_by())

    try:
        _reshape(
            table,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=index,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "missing_members"
    else:
        raise AssertionError("expected RollupNotSummableError")


# Completeness across sequential rollups.

_TWO_AXIS_INDEX = _index(
    statvars=(
        _statvar("SV_A", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_GRANT.dcid),
        _statvar("SV_B", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_LOAN.dcid),
        _statvar("SV_C", purpose=_PURPOSE_MALARIA.dcid, finance=_FINANCE_GRANT.dcid),
    )
)


def test_composition_two_rollups_passes_on_a_complete_frame():
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
            ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0, "USD", "p1"),
            ("dcid:SV_C", "country/USA", "country/KEN", "2022", 7.0, "USD", "p1"),
            ("dcid:SV_A", "country/USA", "country/TZA", "2022", 3.0, "USD", "p1"),
            ("dcid:SV_B", "country/USA", "country/TZA", "2022", 2.0, "USD", "p1"),
            ("dcid:SV_C", "country/USA", "country/TZA", "2022", 1.0, "USD", "p1"),
        ),
    )
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))

    reshaped = _reshape(
        table,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_TWO_AXIS_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.warnings == ()
    purpose_pos = reshaped.table.columns.index("stat_var_crsPurpose")
    recipient_pos = reshaped.table.columns.index("financingRecipient")
    value_pos = reshaped.table.columns.index("value")
    totals = {(row[recipient_pos], row[purpose_pos]): row[value_pos] for row in reshaped.table.rows}
    assert totals == {
        ("country/KEN", "Basic health care"): 15.0,
        ("country/KEN", "Malaria control"): 7.0,
        ("country/TZA", "Basic health care"): 5.0,
        ("country/TZA", "Malaria control"): 1.0,
    }


def test_composition_two_rollups_refuses_when_a_wholly_absent_finer_cell_surfaces_at_the_second():
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
            ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0, "USD", "p1"),
            ("dcid:SV_C", "country/USA", "country/KEN", "2022", 7.0, "USD", "p1"),
            ("dcid:SV_A", "country/USA", "country/TZA", "2022", 3.0, "USD", "p1"),
            # The missing health-loan row forms no group in rollup 0; rollup 1 exposes the gap.
            ("dcid:SV_C", "country/USA", "country/TZA", "2022", 1.0, "USD", "p1"),
        ),
    )
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))

    try:
        _reshape(
            table,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_TWO_AXIS_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "missing_members"
    else:
        raise AssertionError("expected RollupNotSummableError")
