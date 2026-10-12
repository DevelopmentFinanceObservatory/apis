"""Tests full-collapse identity and content version when groupby keeps no decoded CRS axis.

The fixture has three members; their values sum to 22 under the family StatVar.
"""

from __future__ import annotations

from typing import cast

import dfo_frame_grammar
import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import (
    StatVarsSummedWarning,
    _reshape,
)
from materialise.content_version import mint_version
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import Table

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


def _index(*, statvars: tuple[StatVarRow, ...]) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT, _FINANCE_LOAN),
        statvars=statvars,
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

_BY_NO_CRS_AXIS = MintOptions(groupby=GroupbyOptions(by=["financingProvider"], op="sum"))


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader(), type_kinds={}, parties={})


class TestFullCollapseMultiMember:
    def test_stat_var_restored_as_the_family_ref_and_every_decoded_column_gone(self):
        result = _reshape(
            _TABLE,
            options=_BY_NO_CRS_AXIS,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            subject_id="spec_990600",
            pivot_max_output_cells=5_000_000,
        )

        assert result.collapse is not None
        assert result.collapse.stat_var_ref == "dcid:spec_990600"
        assert "stat_var_crsPurpose" not in result.table.columns
        assert "stat_var_crsTypeOfFinance" not in result.table.columns
        stat_var_pos = result.table.columns.index("stat_var")
        value_pos = result.table.columns.index("value")
        assert {row[stat_var_pos] for row in result.table.rows} == {"dcid:spec_990600"}
        assert sum(row[value_pos] for row in result.table.rows) == 22.0  # 10 + 5 + 7

    def test_contributing_count_and_stat_vars_summed_warning(self):
        result = _reshape(
            _TABLE,
            options=_BY_NO_CRS_AXIS,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            subject_id="spec_990600",
            pivot_max_output_cells=5_000_000,
        )

        assert result.collapse is not None
        assert result.collapse.contributing_member_count == 3
        assert len(result.warnings) == 1
        warning = cast(StatVarsSummedWarning, result.warnings[0])
        assert warning["code"] == "stat_vars_summed"
        assert warning["contributing_member_count"] == 3
        assert warning["stat_var_ref"] == "dcid:spec_990600"

    def test_the_collapsed_frame_is_grammar_valid_and_carries_the_family_ref(self):
        result = _reshape(
            _TABLE,
            options=_BY_NO_CRS_AXIS,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            subject_id="spec_990600",
            pivot_max_output_cells=5_000_000,
        )

        assert result.collapse is not None
        frame = result.collapse.frame
        assert isinstance(frame, dfo_frame_grammar.Frame)
        assert frame.columns[0] == "stat_var"
        assert all(row[0] == "dcid:spec_990600" for row in frame.rows)

    def test_content_version_is_minted_from_the_collapsed_frame_alone(self):
        result = _reshape(
            _TABLE,
            options=_BY_NO_CRS_AXIS,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            subject_id="spec_990600",
            pivot_max_output_cells=5_000_000,
        )

        assert result.collapse is not None
        assert result.collapse.content_version == mint_version(result.collapse.frame)

    def test_more_than_one_contributor_with_no_subject_id_refuses(self):
        with pytest.raises(ValueError, match="subject_id"):
            _reshape(
                _TABLE,
                options=_BY_NO_CRS_AXIS,
                entity_columns=_ENTITY_COLUMNS,
                enrichment=_enrichment(),
                code_index=_INDEX,
                resolved_stat_var_refs=(),
                provenance_refs=("p1",),
                pivot_max_output_cells=5_000_000,
            )


class TestFullCollapseSoleContributor:
    _SOLE_TABLE = Table(
        columns=_TABLE_COLUMNS,
        rows=(("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", "p1"),),
    )

    def test_stat_var_keeps_the_sole_members_own_dcid_and_no_warning_fires(self):
        result = _reshape(
            self._SOLE_TABLE,
            options=_BY_NO_CRS_AXIS,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )

        assert result.collapse is not None
        assert result.collapse.stat_var_ref == "dcid:SV_A"
        assert result.collapse.contributing_member_count == 1
        assert result.warnings == ()

    def test_a_sole_contributor_needs_no_subject_id(self):
        # A sole contributor keeps its dcid and needs no family-ref fallback.
        result = _reshape(
            self._SOLE_TABLE,
            options=_BY_NO_CRS_AXIS,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=(),
            provenance_refs=("p1",),
            pivot_max_output_cells=5_000_000,
        )
        assert result.collapse is not None
        assert result.collapse.stat_var_ref == "dcid:SV_A"


class TestFullCollapseEmpty:
    """An empty full collapse emits no `stat_vars_summed` warning.

    Pass a resolved StatVar absent from the table to activate full-collapse decoding with
    zero contributing members.
    """

    _NO_ROWS = Table(columns=_TABLE_COLUMNS, rows=())

    def test_an_empty_full_collapse_emits_no_stat_vars_summed(self):
        result = _reshape(
            self._NO_ROWS,
            options=_BY_NO_CRS_AXIS,
            entity_columns=_ENTITY_COLUMNS,
            enrichment=_enrichment(),
            code_index=_INDEX,
            resolved_stat_var_refs=("dcid:SV_A",),
            provenance_refs=("p1",),
            subject_id="spec_990600",
            pivot_max_output_cells=5_000_000,
        )

        assert result.collapse is not None
        assert result.collapse.contributing_member_count == 0
        assert result.collapse.stat_var_ref == "dcid:spec_990600"
        assert result.warnings == ()

    def test_an_empty_full_collapse_with_no_subject_id_still_refuses(self):
        # Zero contributors require the family ref to stand in for `stat_var`.
        with pytest.raises(ValueError, match="subject_id"):
            _reshape(
                self._NO_ROWS,
                options=_BY_NO_CRS_AXIS,
                entity_columns=_ENTITY_COLUMNS,
                enrichment=_enrichment(),
                code_index=_INDEX,
                resolved_stat_var_refs=("dcid:SV_A",),
                provenance_refs=("p1",),
                pivot_max_output_cells=5_000_000,
            )
