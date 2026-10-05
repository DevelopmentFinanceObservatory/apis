"""Compares streamed and whole-table groupby results, including collapses that cross
chunk boundaries.

`resolved_stat_var_refs` includes each fixture StatVar, including members absent from a
chunk.
"""

from __future__ import annotations

import pytest

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
        constraint_properties={"crsPurpose": purpose, "crsTypeOfFinance": finance},
    )


_INDEX = CodeIndex(
    stamp=_STAMP,
    declarations=_DECLARATIONS,
    codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT, _FINANCE_LOAN),
    statvars=(
        _statvar("SV_A", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_GRANT.dcid),
        _statvar("SV_B", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_LOAN.dcid),
        _statvar("SV_C", purpose=_PURPOSE_MALARIA.dcid, finance=_FINANCE_GRANT.dcid),
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
_ROWS = (
    ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
    ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0, "USD", "p1"),
    ("dcid:SV_C", "country/USA", "country/KEN", "2022", 7.0, "USD", "p1"),
)
_RESOLVED_STAT_VAR_REFS = ("dcid:SV_A", "dcid:SV_B", "dcid:SV_C")


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader())


def _chunked(bounds: list[int]) -> list[Table]:
    return [
        Table(columns=_TABLE_COLUMNS, rows=_ROWS[bounds[i] : bounds[i + 1]])
        for i in range(len(bounds) - 1)
    ]


def test_a_collapse_group_spanning_a_chunk_boundary_matches_the_whole_table_result():
    # SV_A (chunk 1) and SV_B (chunk 2) share `crsPurpose`, so collapsing
    # `crsTypeOfFinance`+`stat_var` must merge their rows (10 + 5 = 15)
    # even though the fold only ever sees one chunk at a time.
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))
    whole_table = Table(columns=_TABLE_COLUMNS, rows=_ROWS)

    whole = _reshape(
        whole_table,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED_STAT_VAR_REFS,
        provenance_refs=("p1",),
        subject_id="family-spec",
        pivot_max_output_cells=5_000_000,
    )
    streamed = _reshape(
        _chunked([0, 1, 2, 3])[0],
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED_STAT_VAR_REFS,
        provenance_refs=("p1",),
        subject_id="family-spec",
        frame_chunks=iter(_chunked([0, 1, 2, 3])),
        pivot_max_output_cells=5_000_000,
    )

    assert streamed.table == whole.table
    assert streamed.warnings == whole.warnings
    assert streamed.collapse == whole.collapse

    purpose_pos = streamed.table.columns.index("stat_var_crsPurpose")
    value_pos = streamed.table.columns.index("value")
    by_purpose = {row[purpose_pos]: row[value_pos] for row in streamed.table.rows}
    assert by_purpose == {"Basic health care": 15.0, "Malaria control": 7.0}


def test_a_full_collapse_across_chunks_restores_the_family_dcid_the_same_way():
    # Naming no CRS axis at all is a full collapse: every declared-summable
    # axis this frame decoded joins `stat_var`'s own collapse, restoring
    # one family dcid across all three (streamed) contributors.
    options = MintOptions(groupby=GroupbyOptions(by=["financingProvider"], op="sum"))
    whole_table = Table(columns=_TABLE_COLUMNS, rows=_ROWS)

    whole = _reshape(
        whole_table,
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED_STAT_VAR_REFS,
        provenance_refs=("p1",),
        subject_id="family-spec",
        pivot_max_output_cells=5_000_000,
    )
    streamed = _reshape(
        _chunked([0, 2, 3])[0],
        options=options,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=_enrichment(),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED_STAT_VAR_REFS,
        provenance_refs=("p1",),
        subject_id="family-spec",
        frame_chunks=iter(_chunked([0, 2, 3])),
        pivot_max_output_cells=5_000_000,
    )

    assert streamed.table == whole.table
    assert streamed.collapse is not None and whole.collapse is not None
    assert streamed.collapse.stat_var_ref == whole.collapse.stat_var_ref == "dcid:family-spec"
    assert (
        streamed.collapse.contributing_member_count == whole.collapse.contributing_member_count == 3
    )
    value_pos = streamed.table.columns.index("value")
    assert sum(row[value_pos] for row in streamed.table.rows) == 22.0


def test_distinct_measures_sharing_every_decoded_axis_value_still_refuse_when_streamed():
    # The stat_var_ambiguous refusal (`refuse_stat_var_collision`) is
    # deferred to run after axis-0's fold on the streaming path (see
    # `_reshape`'s own docstring) -- this proves it still fires, from the
    # distinct-dcid set the fold collects, split across two chunks.
    gross_net_index = CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT, _FINANCE_LOAN),
        statvars=(
            StatVarRow(
                dcid="SV_gross",
                population_type="Amount",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier="gross",
                measurement_denominator=None,
                observation_properties=("financingProvider", "financingRecipient"),
                constraint_properties={
                    "crsPurpose": _PURPOSE_HEALTH.dcid,
                    "crsTypeOfFinance": _FINANCE_GRANT.dcid,
                },
            ),
            StatVarRow(
                dcid="SV_net",
                population_type="Amount",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier="net",
                measurement_denominator=None,
                observation_properties=("financingProvider", "financingRecipient"),
                constraint_properties={
                    "crsPurpose": _PURPOSE_HEALTH.dcid,
                    "crsTypeOfFinance": _FINANCE_GRANT.dcid,
                },
            ),
        ),
    )
    columns = _TABLE_COLUMNS
    rows = (
        ("dcid:SV_gross", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),
        ("dcid:SV_net", "country/USA", "country/KEN", "2022", 4.0, "USD", "p1"),
    )
    chunks = [Table(columns=columns, rows=(rows[0],)), Table(columns=columns, rows=(rows[1],))]
    options = MintOptions(groupby=GroupbyOptions(by=["crsPurpose"], op="sum"))

    with pytest.raises(RollupNotSummableError) as exc_info:
        _reshape(
            chunks[0],
            options=options,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=gross_net_index,
            resolved_stat_var_refs=("dcid:SV_gross", "dcid:SV_net"),
            provenance_refs=("p1",),
            frame_chunks=iter(chunks),
            pivot_max_output_cells=5_000_000,
        )
    assert exc_info.value.axis == "stat_var"
    assert exc_info.value.precondition == "stat_var_ambiguous"
