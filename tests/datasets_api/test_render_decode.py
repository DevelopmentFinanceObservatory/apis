"""Tests `_reshape` decoding order and `columns` validation with and without a `CodeIndex`."""

from __future__ import annotations

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import MintOptions
from datasets_api.render import _reshape
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import ColumnUnknownError, Table

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
_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=False,
        absent_reading="not_declared",
        plain_name="sector",
    ),
)


def _index() -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH,),
        statvars=(
            StatVarRow(
                # Bare, matching `StatVarRow.dcid`'s real convention -- the
                # table's own `stat_var` cell below stays `dcid:`-prefixed,
                # exercising the same boundary `build_decode_plan` bridges
                # in production (`datasets_api.decode`'s own module
                # docstring).
                dcid="SV_A",
                population_type="Amount",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("financingProvider", "financingRecipient"),
                constraint_properties={"crsPurpose": _PURPOSE_HEALTH.dcid},
            ),
        ),
    )


_TABLE_COLUMNS = ("stat_var", "financingProvider", "financingRecipient", "time_period", "value")
_TABLE = Table(
    columns=_TABLE_COLUMNS, rows=(("dcid:SV_A", "country/USA", "country/KEN", "2020", 10.0),)
)


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader())


def test_code_index_none_skips_decoding_entirely():
    reshaped = _reshape(
        _TABLE,
        options=MintOptions(columns=None, pivot=None),
        entity_columns=("financingProvider", "financingRecipient"),
        enrichment=_enrichment(),
        code_index=None,
        resolved_stat_var_refs=(),
        provenance_refs=(),
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.table.columns == _TABLE_COLUMNS
    assert reshaped.warnings == ()


def test_a_supplied_code_index_decodes_unconditionally_even_with_no_columns_request():
    reshaped = _reshape(
        _TABLE,
        options=MintOptions(columns=None, pivot=None),
        entity_columns=("financingProvider", "financingRecipient"),
        enrichment=_enrichment(),
        code_index=_index(),
        resolved_stat_var_refs=(),
        provenance_refs=(),
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.table.columns == (*_TABLE_COLUMNS, "stat_var_crsPurpose")
    assert reshaped.table.rows == (
        ("dcid:SV_A", "country/USA", "country/KEN", "2020", 10.0, "Basic health care"),
    )
    assert reshaped.warnings == ()


def test_columns_naming_a_decoded_column_parses_against_the_post_decode_column_set():
    # Without decode, `parse_column_tokens` would see only `_TABLE_COLUMNS`
    # and refuse `column_unknown` for `stat_var_crsPurpose` -- a column the
    # response will actually carry once decode has run.
    reshaped = _reshape(
        _TABLE,
        options=MintOptions(columns=[*_TABLE_COLUMNS, "stat_var_crsPurpose"], pivot=None),
        entity_columns=("financingProvider", "financingRecipient"),
        enrichment=_enrichment(),
        code_index=_index(),
        resolved_stat_var_refs=(),
        provenance_refs=(),
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.table.columns == (*_TABLE_COLUMNS, "stat_var_crsPurpose")
    assert reshaped.warnings == ()


def test_a_decoded_column_named_in_columns_without_a_code_index_still_refuses_column_unknown():
    with pytest.raises(ColumnUnknownError) as excinfo:
        _reshape(
            _TABLE,
            options=MintOptions(columns=["stat_var_crsPurpose"], pivot=None),
            entity_columns=("financingProvider", "financingRecipient"),
            enrichment=_enrichment(),
            code_index=None,
            resolved_stat_var_refs=(),
            provenance_refs=(),
            pivot_max_output_cells=5_000_000,
        )

    assert excinfo.value.pointer == "/options/columns"


def test_a_decoded_column_can_be_dropped_via_columns_selection():
    table = _reshape(
        _TABLE,
        options=MintOptions(columns=list(_TABLE_COLUMNS), pivot=None),
        entity_columns=("financingProvider", "financingRecipient"),
        enrichment=_enrichment(),
        code_index=_index(),
        resolved_stat_var_refs=(),
        provenance_refs=(),
        pivot_max_output_cells=5_000_000,
    ).table

    assert table.columns == _TABLE_COLUMNS
