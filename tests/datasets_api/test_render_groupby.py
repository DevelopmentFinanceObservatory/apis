"""Tests how `_reshape` keeps requested groupby axes and collapses the others, including
`stat_var`.

The fixture has three StatVars. `SV_A` and `SV_B` sum to 15 under health. Malaria remains 7.
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
    # Declared summable, but no fixture StatVar carries it. `refuse_unknown_by_axes` must accept
    # it, unlike a name the deployment never declares.
    CodeSystemDeclaration(
        code_system="crsCooperationModality",
        hierarchical=False,
        disjoint_from_children=False,
        summable=True,
        absent_reading="not_declared",
        plain_name="modality",
    ),
)


def _statvar(
    dcid: str,
    *,
    purpose: str,
    finance: str,
    measurement_qualifier: str | None = None,
) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=measurement_qualifier,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties={"crsPurpose": purpose, "crsTypeOfFinance": finance},
    )


def _index(*, statvars: tuple[StatVarRow, ...]) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT, _FINANCE_LOAN),
        statvars=statvars,
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

# SV_A and SV_B sum to 15 under health. SV_C remains separate under malaria.
_INDEX = _index(
    statvars=(
        _statvar("SV_A", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_GRANT.dcid),
        _statvar("SV_B", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_LOAN.dcid),
        _statvar("SV_C", purpose=_PURPOSE_MALARIA.dcid, finance=_FINANCE_GRANT.dcid),
    )
)
_TABLE = Table(
    columns=_TABLE_COLUMNS,
    rows=(
        ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
        ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0, "USD", "p1"),
        ("dcid:SV_C", "country/USA", "country/KEN", "2022", 7.0, "USD", "p1"),
    ),
)


_SUBJECT_ID = "spec_990700"


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader(), type_kinds={}, parties={})


def test_naming_one_axis_to_keep_collapses_the_other_into_a_frame_keeping_its_column():
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))

    reshaped = _reshape(
        _TABLE,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        subject_id=_SUBJECT_ID,
        pivot_max_output_cells=5_000_000,
    )
    table = reshaped.table

    assert [warning["code"] for warning in reshaped.warnings] == ["stat_vars_summed"]
    assert reshaped.collapse is not None
    assert reshaped.collapse.frame.columns == (
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
    assert "stat_var_crsTypeOfFinance" not in table.columns
    stat_var_pos = table.columns.index("stat_var")
    purpose_pos = table.columns.index("stat_var_crsPurpose")
    value_pos = table.columns.index("value")
    assert {row[stat_var_pos] for row in table.rows} == {f"dcid:{_SUBJECT_ID}"}
    by_purpose = {row[purpose_pos]: row[value_pos] for row in table.rows}
    assert by_purpose == {"Basic health care": 15.0, "Malaria control": 7.0}
    assert len(table.rows) == 2


def test_naming_every_declared_summable_axis_collapses_nothing():
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose", "crsTypeOfFinance"], op="sum"))

    reshaped = _reshape(
        _TABLE,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        pivot_max_output_cells=5_000_000,
    )
    table = reshaped.table

    assert reshaped.warnings == ()
    # `stat_var` rides along unchanged -- the identity reading, matching
    # a caller who already sees every axis on the frame.
    assert "stat_var" in table.columns
    assert len(table.rows) == 3
    value_pos = table.columns.index("value")
    assert sorted(row[value_pos] for row in table.rows) == [5.0, 7.0, 10.0]


def test_an_unsafe_collapse_refuses_rollup_not_summable_naming_it():
    # SV_A and SV_B share a purpose but differ in unit, so the sum must refuse.
    mixed_unit_table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
            ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0, "EUR", "p1"),
            ("dcid:SV_C", "country/USA", "country/KEN", "2022", 7.0, "USD", "p1"),
        ),
    )
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))

    try:
        _reshape(
            mixed_unit_table,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "facet_homogeneity"
    else:
        raise AssertionError("expected RollupNotSummableError")


def test_a_frame_carrying_decoded_columns_collapses_without_double_counting():
    # Keep `stat_var` in the collapse keys to conserve the total across decoded axes.
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))

    table = _reshape(
        _TABLE,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        subject_id=_SUBJECT_ID,
        pivot_max_output_cells=5_000_000,
    ).table

    purpose_pos = table.columns.index("stat_var_crsPurpose")
    value_pos = table.columns.index("value")
    total = sum(row[value_pos] for row in table.rows)
    assert total == 22.0  # 10 + 5 + 7, conserved across the collapse
    by_purpose = {row[purpose_pos]: row[value_pos] for row in table.rows}
    assert by_purpose["Basic health care"] == 15.0


def test_distinct_measures_sharing_every_decoded_axis_value_refuse_before_merging():
    # Gross and net StatVars share decoded axes but differ in `measurement_qualifier`.
    # The frame omits that field, so collapsing them would merge distinct measures.
    gross_net_index = _index(
        statvars=(
            _statvar(
                "SV_gross",
                purpose=_PURPOSE_HEALTH.dcid,
                finance=_FINANCE_GRANT.dcid,
                measurement_qualifier="gross",
            ),
            _statvar(
                "SV_net",
                purpose=_PURPOSE_HEALTH.dcid,
                finance=_FINANCE_GRANT.dcid,
                measurement_qualifier="net",
            ),
        )
    )
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_gross", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
            ("dcid:SV_net", "country/USA", "country/KEN", "2022", 4.0, "USD", "p1"),
        ),
    )
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))

    try:
        _reshape(
            table,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=gross_net_index,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.axis == "stat_var"
        assert exc.precondition == "stat_var_ambiguous"
    else:
        raise AssertionError("expected RollupNotSummableError")


def test_a_by_entry_naming_an_axis_this_frame_never_declares_is_a_no_op():
    # The declared modality axis is absent from this frame, so it has no effect on grouping.
    options = MintOptions(
        groupby=GroupbyOptions(by=["crsPurpose", "crsCooperationModality"], op="sum")
    )

    table = _reshape(
        _TABLE,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        subject_id=_SUBJECT_ID,
        pivot_max_output_cells=5_000_000,
    ).table

    purpose_pos = table.columns.index("stat_var_crsPurpose")
    value_pos = table.columns.index("value")
    by_purpose = {row[purpose_pos]: row[value_pos] for row in table.rows}
    assert by_purpose == {"Basic health care": 15.0, "Malaria control": 7.0}


def test_a_by_entry_this_deployment_never_declares_at_all_refuses():
    # `crsTypoedAxis` is declared nowhere, unlike the modality axis above, so it is refused.
    options = MintOptions(groupby=GroupbyOptions(by=["crsTypoedAxis"], op="sum"))

    try:
        _reshape(
            _TABLE,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.axis == "crsTypoedAxis"
        assert exc.precondition == "not_declared_summable"
    else:
        raise AssertionError("expected RollupNotSummableError")
