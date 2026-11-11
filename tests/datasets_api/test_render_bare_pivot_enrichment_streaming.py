"""Streamed bare-pivot output and warnings match whole-table results across one- and four-chunk inputs.

The graph supplies `iso3` for Kenya alone. Uganda and Togo each contribute one unavailable value.
"""

from __future__ import annotations

from collections.abc import Sequence

from code_index.types import PartyRow
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
            enrichment=EnrichmentLookup(_GRAPH), parties={}, known={}, deadline=None
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
            enrichment=EnrichmentLookup(_GRAPH), parties={}, known={}, deadline=None
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


_D1, _D2 = "provenance/D1", "provenance/D2"
_PARTY_COLUMNS = ("stat_var", "financingProvider", "time_period", "provenance", "value")
_PARTIES = {
    # A total in D1 and a party in D2, both on the provider side.
    "org/X": PartyRow(
        dcid="org/X",
        kind="party",
        published=((_D1, "origin"), (_D2, "origin")),
        dataset_kinds=((_D1, "total"),),
    ),
    "org/Y": PartyRow(
        dcid="org/Y",
        kind="party",
        published=((_D1, "origin"), (_D2, "origin")),
        dataset_kinds=((_D1, "total"),),
    ),
    "org/Z": PartyRow(dcid="org/Z", kind="party", published=((_D1, "origin"),)),
}


def _provider_pivot(
    rows: Sequence[tuple[object, ...]], *, columns: Sequence[str]
) -> list[dict[str, object]]:
    """The warnings of a streamed bare pivot of `rows`, a provider per row and a dataset per
    column."""
    streamed = _stream_bare_pivot(
        iter((Table(columns=_PARTY_COLUMNS, rows=tuple(rows)),)),
        options=MintOptions(
            format="csv",
            bundle=False,
            columns=list(columns),
            pivot=PivotOptions(index="financingProvider", columns="provenance"),
        ),
        entity_columns=("financingProvider",),
        reads=_RenderReads(
            enrichment=EnrichmentLookup(_GRAPH), parties=_PARTIES, known={}, deadline=None
        ),
        code_index=None,
        resolved_stat_var_refs=(),
        provenance_refs=(),
        subject_id="bare-pivot-party-kind-test",
        pivot_max_output_cells=5_000_000,
    )
    return list(streamed.warnings)


def test_a_bare_pivot_counts_a_party_missing_where_the_datasets_serving_it_differ_on_its_kind():
    """org/X is served from D1 and D2, where its kinds differ, so its one pivot cell takes no
    kind and its two rows count missing. org/Y, the same party row served from D1 alone, takes
    D1's kind, and org/Z has one kind everywhere."""
    rows = (
        ("dcid:SV", "org/X", "2020", _D1, 1.0),
        ("dcid:SV", "org/X", "2020", _D2, 2.0),
        ("dcid:SV", "org/Y", "2020", _D1, 3.0),
        ("dcid:SV", "org/Z", "2020", _D1, 5.0),
    )

    warnings = _provider_pivot(rows, columns=(*_PARTY_COLUMNS, "financingProvider_party_kind"))

    assert [(w["code"], w["column"], w["count"]) for w in warnings] == [
        ("enrichment_unavailable", "financingProvider_party_kind", 2)
    ]


class _CountedDataset(str):
    """A `provenance` cell that counts the times it is hashed, as a set of datasets does."""

    hashed = 0

    def __hash__(self) -> int:
        type(self).hashed += 1
        return super().__hash__()


def test_a_bare_pivot_reads_no_dataset_of_a_column_no_party_field_is_asked_of():
    """A graph field asks for no party kind, so the stream gathers no dataset for its column:
    the provenance cells are read as often as with no field asked at all."""
    rows = [
        ("dcid:SV", ref, "2020", _CountedDataset(_D1), 1.0) for ref in ("org/X", "org/Y", "org/Z")
    ]

    def hashed(columns: Sequence[str]) -> int:
        _CountedDataset.hashed = 0
        _provider_pivot(rows, columns=columns)
        return _CountedDataset.hashed

    assert hashed((*_PARTY_COLUMNS, "financingProvider_iso3")) == hashed(_PARTY_COLUMNS)
