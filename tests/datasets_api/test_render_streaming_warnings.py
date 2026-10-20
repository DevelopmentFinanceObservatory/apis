"""Streaming reports one enrichment warning per column with whole-result counts and the
same order as the whole-table path.

Uganda and Togo each lack `countryAlpha3Code`, so two chunks produce one
`recipient_iso3` warning with count 2. The two-column case puts misses in a different order
from `options.columns` to check warning order.
"""

from __future__ import annotations

from datasets_api.mint_models import MintOptions
from datasets_api.render import render_and_store
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import Table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_STAT_VAR = "dcid:MalariaGrantDisbursements"
_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit")

_GRAPH = FakeMintGraphReader(arcs={"country/KEN": {"countryAlpha3Code": ("KEN",)}})
"""The graph supplies an `iso3` for Kenya. Uganda and Togo each lack a value."""

_ROWS_BY_CHUNK = (
    (
        (_STAT_VAR, "country/KEN", "2015", 100.0, "USD"),
        (_STAT_VAR, "country/UGA", "2015", 200.0, "USD"),
    ),
    (
        (_STAT_VAR, "country/KEN", "2016", 300.0, "USD"),
        (_STAT_VAR, "country/TGO", "2016", 400.0, "USD"),
    ),
)


def _options() -> MintOptions:
    return MintOptions(
        format="csv",
        bundle=False,
        columns=["stat_var", "recipient", "time_period", "value", "unit", "recipient_iso3"],
    )


def _render(chunks: tuple[Table, ...]) -> tuple:
    rendered = render_and_store(
        frame_chunks=lambda: iter(chunks),
        options=_options(),
        spec={"spec_id": "streaming-warnings-test"},
        provenance_refs=[],
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=FakeArtifactStore(),
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
        resolved_stat_var_refs=(),
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered.artifact, rendered.warnings


def test_enrichment_warnings_merge_across_chunks_to_the_whole_table_answer():
    one_chunk = (Table(columns=_COLUMNS, rows=_ROWS_BY_CHUNK[0] + _ROWS_BY_CHUNK[1]),)
    two_chunks = tuple(Table(columns=_COLUMNS, rows=rows) for rows in _ROWS_BY_CHUNK)

    _one_artifact, one_chunk_warnings = _render(one_chunk)
    _two_artifact, two_chunk_warnings = _render(two_chunks)

    one_chunk_enrichment = [w for w in one_chunk_warnings if w["code"] == "enrichment_unavailable"]
    two_chunk_enrichment = [w for w in two_chunk_warnings if w["code"] == "enrichment_unavailable"]

    # The graph names and kinds nothing, so the default name and kind columns miss on every
    # row.
    assert [(w["column"], w["count"]) for w in one_chunk_enrichment] == [
        ("recipient_iso3", 2),
        ("stat_var_label", 4),
        ("recipient_label", 4),
        ("recipient_kind", 4),
    ]
    assert two_chunk_enrichment == one_chunk_enrichment


_TWO_COLUMN_GRAPH = FakeMintGraphReader(
    arcs={
        "country/UGA": {"countryAlpha2Code": ("UG",)},
        "country/TGO": {"countryAlpha3Code": ("TGO",)},
    }
)
"""Uganda supplies `iso2` and Togo supplies `iso3`, so each row misses one requested field.
The misses land in separate chunks.
"""

_TWO_COLUMN_OPTIONS_COLUMNS = (
    "stat_var",
    "recipient",
    "time_period",
    "value",
    "unit",
    "recipient_iso2",
    "recipient_iso3",
)
"""`recipient_iso2` is requested before `recipient_iso3` -- the fixed
request order `_reshape`'s own `warning_order` return value carries, and
the order the whole-table path's single `_fill_lookup_columns` call reports
misses in."""

_TWO_COLUMN_ROWS_BY_CHUNK = (
    # Chunk 1: UGA has iso2, misses iso3 -- only the LATER-requested column
    # (`recipient_iso3`) misses here.
    ((_STAT_VAR, "country/UGA", "2015", 100.0, "USD"),),
    # Chunk 2: TGO has iso3, misses iso2 -- the EARLIER-requested column
    # (`recipient_iso2`) misses for the first time only in this later chunk.
    ((_STAT_VAR, "country/TGO", "2016", 400.0, "USD"),),
)


def _render_two_columns(chunks: tuple[Table, ...]) -> tuple:
    rendered = render_and_store(
        frame_chunks=lambda: iter(chunks),
        options=MintOptions(format="csv", bundle=False, columns=list(_TWO_COLUMN_OPTIONS_COLUMNS)),
        spec={"spec_id": "streaming-warnings-order-test"},
        provenance_refs=[],
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=FakeArtifactStore(),
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_TWO_COLUMN_GRAPH, parties=lambda: {}),
        resolved_stat_var_refs=(),
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered.artifact, rendered.warnings


def test_streamed_warning_order_matches_whole_table_order_across_two_columns():
    """The two columns first miss in reverse order to `options.columns`. The whole-table path
    reports request order, and streamed warnings must match it.
    """
    one_chunk = (
        Table(columns=_COLUMNS, rows=_TWO_COLUMN_ROWS_BY_CHUNK[0] + _TWO_COLUMN_ROWS_BY_CHUNK[1]),
    )
    two_chunks = tuple(Table(columns=_COLUMNS, rows=rows) for rows in _TWO_COLUMN_ROWS_BY_CHUNK)

    _whole_artifact, whole_table_warnings = _render_two_columns(one_chunk)
    _streamed_artifact, streamed_warnings = _render_two_columns(two_chunks)

    whole_table_enrichment = [
        w for w in whole_table_warnings if w["code"] == "enrichment_unavailable"
    ]
    streamed_enrichment = [w for w in streamed_warnings if w["code"] == "enrichment_unavailable"]

    assert [w["column"] for w in whole_table_enrichment] == [
        "recipient_iso2",
        "recipient_iso3",
        "stat_var_label",
        "recipient_label",
        "recipient_kind",
    ]
    assert streamed_enrichment == whole_table_enrichment
