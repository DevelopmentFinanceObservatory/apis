"""Tests refusal when `stat_var` is omitted and no decoded summable axis can collapse multiple
contributors. A sole contributor remains a no-op, and a decoded summable axis still collapses.
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


def _statvar(dcid: str, *, purpose: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties={"crsPurpose": purpose},
    )


# SV_A and SV_B have different purposes, so a declared summable axis can collapse them.
_STATVARS = (
    _statvar("SV_A", purpose=_PURPOSE_HEALTH.dcid),
    _statvar("SV_B", purpose=_PURPOSE_MALARIA.dcid),
)
_CODES = (_PURPOSE_HEALTH, _PURPOSE_MALARIA)

_NO_SUMMABLE_AXIS_INDEX = CodeIndex(
    stamp=_STAMP,
    # The StatVars carry `crsPurpose`, but this index declares no summable code system.
    # `axes_to_collapse` is empty for every `by` value.
    declarations=(),
    codes=_CODES,
    statvars=_STATVARS,
)

_TWO_CONTRIBUTOR_TABLE = Table(
    columns=_TABLE_COLUMNS,
    rows=(
        ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
        ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0, "USD", "p1"),
    ),
)

_SOLE_CONTRIBUTOR_TABLE = Table(
    columns=_TABLE_COLUMNS,
    rows=(("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),),
)


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader())


def test_by_omitting_stat_var_refuses_when_no_summable_axis_and_multiple_contributors():
    options = MintOptions(groupby=GroupbyOptions(by=["financingProvider"], op="sum"))

    try:
        _reshape(
            _TWO_CONTRIBUTOR_TABLE,
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_NO_SUMMABLE_AXIS_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            subject_id="spec_family",
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.axis == "stat_var"
        assert exc.precondition == "no_summable_axis"
    else:
        raise AssertionError("expected RollupNotSummableError")


def test_a_sole_contributor_with_no_summable_axis_stays_a_no_op():
    # A single contributing StatVar needs no axis to collapse.
    options = MintOptions(groupby=GroupbyOptions(by=["financingProvider"], op="sum"))

    reshaped = _reshape(
        _SOLE_CONTRIBUTOR_TABLE,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_NO_SUMMABLE_AXIS_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        subject_id="spec_family",
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.warnings == ()
    stat_var_pos = reshaped.table.columns.index("stat_var")
    assert {row[stat_var_pos] for row in reshaped.table.rows} == {"dcid:SV_A"}


def test_by_naming_stat_var_explicitly_never_refuses_even_with_multiple_contributors():
    # The caller explicitly kept `stat_var`, so more than one contributing
    # StatVar staying on the frame is exactly what they asked for.
    options = MintOptions(groupby=GroupbyOptions(by=["stat_var"], op="sum"))

    reshaped = _reshape(
        _TWO_CONTRIBUTOR_TABLE,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_NO_SUMMABLE_AXIS_INDEX,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        subject_id="spec_family",
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.warnings == ()
    stat_var_pos = reshaped.table.columns.index("stat_var")
    assert {row[stat_var_pos] for row in reshaped.table.rows} == {"dcid:SV_A", "dcid:SV_B"}


def test_a_decoded_summable_axis_still_collapses_instead_of_refusing():
    # A decoded summable axis enters the collapse path, so this request sums successfully.
    declared_index = CodeIndex(
        stamp=_STAMP,
        declarations=(
            CodeSystemDeclaration(
                code_system="crsPurpose",
                hierarchical=True,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="sector",
            ),
        ),
        codes=_CODES,
        statvars=_STATVARS,
    )
    options = MintOptions(groupby=GroupbyOptions(by=["financingProvider"], op="sum"))

    result = _reshape(
        _TWO_CONTRIBUTOR_TABLE,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=declared_index,
        resolved_stat_var_refs=(),
        provenance_refs=("p1",),
        subject_id="spec_family",
        pivot_max_output_cells=5_000_000,
    )

    assert result.collapse is not None
    assert result.collapse.contributing_member_count == 2
