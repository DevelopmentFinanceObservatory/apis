"""A `columns` list that names only enrichment fields keeps every default column, decoded ones
included, and adds the enrichments. A list that mixes a table column with an enrichment selects.

The frame carries a decoded `stat_var_crsPurpose` column and one place column, `observationAbout`.
The whole-table path (`_reshape`) and the streamed bare-pivot path (`_stream_bare_pivot`) agree.
"""

from __future__ import annotations

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import MintOptions, PivotOptions
from datasets_api.render import _reshape, _stream_bare_pivot
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import ColumnUnknownError, Table

from ._graph_doubles import FakeMintGraphReader

_GRAPH = FakeMintGraphReader(
    arcs={
        "country/KEN": {"typeOf": ("Country",)},
        "country/TGO": {"typeOf": ("Country",)},
    }
)
_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://example.test/core/api",
        exported_at="2026-09-09T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="deadbeef",
    ),
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
    codes=(
        CodeRow(
            dcid="dcid:CRSPurposeCode122",
            code_system="crsPurpose",
            code="122",
            label="Basic health care",
            synonyms=(),
            parent=None,
        ),
    ),
    statvars=(
        StatVarRow(
            dcid="SV_A",
            population_type="Amount",
            measured_property="amount",
            stat_type=None,
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=("observationAbout",),
            constraint_properties={"crsPurpose": "dcid:CRSPurposeCode122"},
        ),
    ),
)
_COLUMNS = ("stat_var", "observationAbout", "time_period", "value")
_ROWS = (
    ("dcid:SV_A", "country/KEN", "2020", 10.0),
    ("dcid:SV_A", "country/TGO", "2020", 20.0),
)
_DEFAULTS = (*_COLUMNS, "stat_var_crsPurpose")
_COMMON = {
    "entity_columns": ("observationAbout",),
    "code_index": _INDEX,
    "resolved_stat_var_refs": (),
    "pivot_max_output_cells": 5_000_000,
}


def _whole(columns: list[str]):
    return _reshape(
        Table(columns=_COLUMNS, rows=_ROWS),
        options=MintOptions(format="csv", bundle=False, columns=columns),
        enrichment=EnrichmentLookup(_GRAPH),
        provenance_refs=(),
        **_COMMON,
    )


def _pivot_options(columns: list[str]) -> MintOptions:
    return MintOptions(
        format="csv",
        bundle=False,
        columns=columns,
        pivot=PivotOptions(index="observationAbout", columns="stat_var_crsPurpose"),
    )


def _streamed(columns: list[str], *, chunk_size: int):
    chunks = iter(
        [Table(columns=_COLUMNS, rows=_ROWS[i : i + chunk_size]) for i in range(0, 2, chunk_size)]
    )
    return _stream_bare_pivot(
        chunks,
        options=_pivot_options(columns),
        enrichment=EnrichmentLookup(_GRAPH),
        provenance_refs=(),
        subject_id="enrichment-only-columns-test",
        groupby_preimage=None,
        **_COMMON,
    )


def test_an_enrichment_only_list_keeps_every_default_column_on_the_whole_table():
    result = _whole(["observationAbout_type"])

    assert result.table.columns == (*_DEFAULTS, "observationAbout_type")
    assert [row[-1] for row in result.table.rows] == ["Country", "Country"]


def test_a_mixed_list_selects_the_table_column_and_adds_the_enrichment_on_the_whole_table():
    result = _whole([*_COLUMNS, "observationAbout_type"])

    assert result.table.columns == (*_COLUMNS, "observationAbout_type")


def test_an_enrichment_only_list_keeps_the_decoded_column_a_streamed_pivot_names():
    whole = _reshape(
        Table(columns=_COLUMNS, rows=_ROWS[:2]),
        options=_pivot_options(["observationAbout_type"]),
        enrichment=EnrichmentLookup(_GRAPH),
        provenance_refs=(),
        **_COMMON,
    )

    assert whole.table.columns == ("observationAbout", "Basic health care")
    for chunk_size in (2, 1):
        assert _streamed(["observationAbout_type"], chunk_size=chunk_size).table == whole.table


def test_a_mixed_list_drops_the_decoded_column_a_streamed_pivot_names():
    with pytest.raises(ColumnUnknownError):
        _streamed([*_COLUMNS, "observationAbout_type"], chunk_size=1)
