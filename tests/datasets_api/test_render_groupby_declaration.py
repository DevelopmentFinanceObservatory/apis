"""Tests how `_reshape` uses the injected `CodeIndex` to handle absent members in groupby
sums.

A source declaring absence as zero permits a missing member. An undeclared source or mixed
provenance refuses.
"""

from __future__ import annotations

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, DatasetDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import _RenderReads, _reshape
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import RollupNotSummableError, Table

from ._graph_doubles import FakeMintGraphReader

_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-15T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)

_CRS_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_OTHER_PROVENANCE = "provenance/SomeOtherSource"

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

_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="sector",
    ),
    # This summable axis is absent from the fixture.
    # Naming it in `by` leaves other axes free to collapse.
    CodeSystemDeclaration(
        code_system="crsCooperationModality",
        hierarchical=False,
        disjoint_from_children=False,
        summable=True,
        absent_reading="not_declared",
        plain_name="modality",
    ),
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


def _index(*, declares_crs_zero: bool) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA),
        statvars=(
            _statvar("SV_health", purpose=_PURPOSE_HEALTH.dcid),
            _statvar("SV_malaria", purpose=_PURPOSE_MALARIA.dcid),
        ),
        datasets=(
            (
                DatasetDeclaration(
                    provenance=_CRS_PROVENANCE,
                    label="OECD Creditor Reporting System (CRS)",
                    description="A test table.",
                    opt_in=False,
                    absent_observation="zero",
                    account="provider",
                    rank=0,
                ),
            )
            if declares_crs_zero
            else ()
        ),
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

# `SV_malaria` is resolved (named in `resolved_stat_var_refs` below) but
# contributes zero rows to the table at all -- the same "entirely absent
# member" shape `test_render_groupby_population.py`'s own
# `test_zero_row_resolved_member_refuses_and_names_it` refuses on HEAD.
_TABLE = Table(
    columns=_TABLE_COLUMNS,
    rows=(("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", _CRS_PROVENANCE),),
)
_RESOLVED_REFS = ("dcid:SV_health", "dcid:SV_malaria")


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader(), parties={})


def _options() -> MintOptions:
    return MintOptions(groupby=GroupbyOptions(by=["crsCooperationModality"], op="sum"))


# --- A declared-zero source sums and publishes -----------------------------


def test_a_zero_row_member_from_a_declared_zero_source_sums_and_publishes():
    reshaped = _reshape(
        _TABLE,
        options=_options(),
        entity_columns=_ENTITY_COLUMNS,
        reads=_RenderReads(enrichment=_enrichment(), known={}, deadline=None),
        code_index=_index(declares_crs_zero=True),
        resolved_stat_var_refs=_RESOLVED_REFS,
        provenance_refs=(_CRS_PROVENANCE,),
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.warnings == ()
    # The declared zero reading permits the absent malaria member. Health is the only
    # contributor, so the full collapse restores its StatVar id.
    assert "stat_var_crsPurpose" not in reshaped.table.columns
    stat_var_pos = reshaped.table.columns.index("stat_var")
    value_pos = reshaped.table.columns.index("value")
    assert [row[stat_var_pos] for row in reshaped.table.rows] == ["dcid:SV_health"]
    assert [row[value_pos] for row in reshaped.table.rows] == [10.0]


# --- The same frame under an undeclared provenance still refuses -----------


def test_the_same_zero_row_member_under_an_undeclared_provenance_still_refuses():
    try:
        _reshape(
            _TABLE,
            options=_options(),
            entity_columns=_ENTITY_COLUMNS,
            reads=_RenderReads(enrichment=_enrichment(), known={}, deadline=None),
            code_index=_index(declares_crs_zero=False),
            resolved_stat_var_refs=_RESOLVED_REFS,
            provenance_refs=(_CRS_PROVENANCE,),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "missing_members"
        assert "Malaria control" in str(exc)
    else:
        raise AssertionError("expected RollupNotSummableError")


# --- Two provenance refs on one frame refuse before picking one ------------


def test_two_distinct_provenance_refs_refuse_before_any_declaration_lookup():
    # A complete population isolates the provenance disagreement from missing-member errors.
    complete_table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", _CRS_PROVENANCE),
            ("dcid:SV_malaria", "country/USA", "country/KEN", "2022", 4.0, "USD", _CRS_PROVENANCE),
        ),
    )

    try:
        _reshape(
            complete_table,
            options=_options(),
            entity_columns=_ENTITY_COLUMNS,
            reads=_RenderReads(enrichment=_enrichment(), known={}, deadline=None),
            code_index=_index(declares_crs_zero=True),
            resolved_stat_var_refs=(),
            provenance_refs=(_CRS_PROVENANCE, _OTHER_PROVENANCE),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.axis == "provenance"
        assert exc.precondition == "provenance_disagreement"
    else:
        raise AssertionError("expected RollupNotSummableError")


# --- Declaration lookup uses the injected index -----------------------------


def test_declaration_follows_the_injected_index_even_when_the_singleton_disagrees():
    import code_index.shared as shared

    # The singleton carries the opposite declaration. A fallback to it changes the result.
    with shared._singleton.poisoned_for_test(_index(declares_crs_zero=False)):
        reshaped = _reshape(
            _TABLE,
            options=_options(),
            entity_columns=_ENTITY_COLUMNS,
            reads=_RenderReads(enrichment=_enrichment(), known={}, deadline=None),
            code_index=_index(declares_crs_zero=True),
            resolved_stat_var_refs=_RESOLVED_REFS,
            provenance_refs=(_CRS_PROVENANCE,),
            pivot_max_output_cells=5_000_000,
        )
        assert reshaped.warnings == ()
        value_pos = reshaped.table.columns.index("value")
        assert [row[value_pos] for row in reshaped.table.rows] == [10.0]

    with shared._singleton.poisoned_for_test(_index(declares_crs_zero=True)):
        try:
            _reshape(
                _TABLE,
                options=_options(),
                entity_columns=_ENTITY_COLUMNS,
                reads=_RenderReads(enrichment=_enrichment(), known={}, deadline=None),
                code_index=_index(declares_crs_zero=False),
                resolved_stat_var_refs=_RESOLVED_REFS,
                provenance_refs=(_CRS_PROVENANCE,),
                pivot_max_output_cells=5_000_000,
            )
        except RollupNotSummableError as exc:
            assert exc.precondition == "missing_members"
        else:
            raise AssertionError("expected RollupNotSummableError")

        # Confirms both calls ran while the singleton held the poisoned index.
        assert shared.get_index().absent_observation(_CRS_PROVENANCE) == "zero"
