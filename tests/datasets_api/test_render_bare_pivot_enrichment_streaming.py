"""Streamed bare-pivot output and warnings match whole-table results across one- and four-chunk inputs.

The graph supplies `iso3` for Kenya alone. Uganda and Togo each contribute one unavailable value.
"""

from __future__ import annotations

from datasets_api.mint_models import MintOptions, PivotOptions
from datasets_api.render import _RenderReads, _reshape, _stream_bare_pivot
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import Table

from ._graph_doubles import FakeMintGraphReader

_GRAPH = FakeMintGraphReader(arcs={"country/KEN": {"countryAlpha3Code": ("KEN",)}})
_COLUMNS = ("stat_var", "recipient", "time_period", "value")
_ROWS = (
    ("dcid:MalariaGrantDisbursements", "country/KEN", "2020", 10.0),
    ("dcid:MalariaGrantDisbursements", "country/UGA", "2020", 20.0),
    ("dcid:MalariaGrantDisbursements", "country/TGO", "2021", 15.0),
    ("dcid:MalariaGrantDisbursements", "country/KEN", "2021", 5.0),
)


def _options() -> MintOptions:
    return MintOptions(
        format="csv",
        bundle=False,
        columns=["stat_var", "recipient", "time_period", "value", "recipient_iso3"],
        pivot=PivotOptions(index="recipient", columns="time_period"),
    )


def _chunks(bounds: list[int]) -> list[Table]:
    return [
        Table(columns=_COLUMNS, rows=_ROWS[bounds[i] : bounds[i + 1]])
        for i in range(len(bounds) - 1)
    ]


def _whole():
    whole_table = Table(columns=_COLUMNS, rows=_ROWS)
    return _reshape(
        whole_table,
        options=_options(),
        entity_columns=("recipient",),
        reads=_RenderReads(
            enrichment=EnrichmentLookup(_GRAPH, parties={}), known={}, deadline=None
        ),
        code_index=None,
        resolved_stat_var_refs=(),
        provenance_refs=(),
        pivot_max_output_cells=5_000_000,
    )


def _streamed(bounds: list[int]):
    return _stream_bare_pivot(
        iter(_chunks(bounds)),
        options=_options(),
        entity_columns=("recipient",),
        reads=_RenderReads(
            enrichment=EnrichmentLookup(_GRAPH, parties={}), known={}, deadline=None
        ),
        code_index=None,
        resolved_stat_var_refs=(),
        provenance_refs=(),
        subject_id="bare-pivot-enrichment-test",
        pivot_max_output_cells=5_000_000,
    )


def test_a_bare_pivot_with_an_enrichment_column_matches_the_whole_table_table():
    whole = _whole()
    streamed_one_chunk = _streamed([0, len(_ROWS)])
    streamed_four_chunks = _streamed([0, 1, 2, 3, len(_ROWS)])

    assert streamed_one_chunk.table == whole.table
    assert streamed_four_chunks.table == whole.table


def test_a_bare_pivot_with_an_enrichment_column_matches_the_whole_table_warning():
    whole = _whole()
    streamed_one_chunk = _streamed([0, len(_ROWS)])
    streamed_four_chunks = _streamed([0, 1, 2, 3, len(_ROWS)])

    assert whole.warnings, "the fixture must actually trip enrichment_unavailable"
    assert streamed_one_chunk.warnings == whole.warnings
    assert streamed_four_chunks.warnings == whole.warnings
    warning = whole.warnings[0]
    assert warning["code"] == "enrichment_unavailable"
    assert warning["column"] == "recipient_iso3"
    # Uganda and Togo each contribute one row missing iso3.
    # country/KEN's two rows both resolve.
    assert warning["count"] == 2
