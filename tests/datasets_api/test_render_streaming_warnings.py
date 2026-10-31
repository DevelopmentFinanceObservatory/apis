"""Streaming reports one enrichment warning per column with whole-result counts and the
same order as the whole-table path.

Uganda and Togo each lack `countryAlpha3Code`, so two chunks produce one
`recipient_iso3` warning with count 2. The two-column case puts misses in a different order
from `options.columns` to check warning order. A streamed render cuts the frame into chunks of
`_RESHAPE_CHUNK_ROWS` rows, set here to the rows of a test's chunk.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from dfo_frame_grammar import Frame

from datasets_api import render as render_module
from datasets_api.mint_models import MintOptions
from datasets_api.render import render_and_store
from materialise.enrichment import EnrichmentLookup

from ._artifact_doubles import FakeArtifactStore
from ._frame_chunks import frame_chunks_of
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_STAT_VAR = "dcid:MalariaGrantDisbursements"
_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit")

_GRAPH = FakeMintGraphReader(arcs={"country/KEN": {"countryAlpha3Code": ("KEN",)}})
"""The graph supplies an `iso3` for Kenya. Uganda and Togo each lack a value."""

_ROWS_BY_CHUNK = (
    (
        (_STAT_VAR, "country/KEN", "2015", 100.0, "USD"),
        (_STAT_VAR, "country/KEN", "2016", 300.0, "USD"),
    ),
    (
        (_STAT_VAR, "country/TGO", "2016", 400.0, "USD"),
        (_STAT_VAR, "country/UGA", "2015", 200.0, "USD"),
    ),
)
"""In the frame's row order."""


def _options() -> MintOptions:
    return MintOptions(
        format="csv",
        bundle=False,
        columns=[
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "provenance",
            "recipient_iso3",
        ],
    )


def _render(chunks: Callable[[], Iterator[Frame]]) -> tuple:
    rendered = render_and_store(
        frame_chunks=chunks,
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
        groupby_fold_max_rows=2_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered.artifact, rendered.warnings


def test_enrichment_warnings_merge_across_chunks_to_the_whole_table_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    _one_artifact, one_chunk_warnings = _render(frame_chunks_of(_COLUMNS, _ROWS_BY_CHUNK))
    monkeypatch.setattr(render_module, "_RESHAPE_CHUNK_ROWS", 2)
    _two_artifact, two_chunk_warnings = _render(frame_chunks_of(_COLUMNS, _ROWS_BY_CHUNK))

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
        "country/TGO": {"countryAlpha2Code": ("TG",)},
        "country/UGA": {"countryAlpha3Code": ("UGA",)},
    }
)
"""Togo supplies `iso2` and Uganda supplies `iso3`, so each row misses one requested field.
The misses land in separate chunks.
"""

_TWO_COLUMN_OPTIONS_COLUMNS = (
    "stat_var",
    "recipient",
    "time_period",
    "value",
    "unit",
    "provenance",
    "recipient_iso2",
    "recipient_iso3",
)
"""`recipient_iso2` is requested before `recipient_iso3` -- the fixed
request order `_reshape`'s own `warning_order` return value carries, and
the order the whole-table path's single `_fill_lookup_columns` call reports
misses in."""

_TWO_COLUMN_ROWS_BY_CHUNK = (
    # Chunk 1: TGO has iso2, misses iso3 -- only the LATER-requested column
    # (`recipient_iso3`) misses here.
    ((_STAT_VAR, "country/TGO", "2015", 100.0, "USD"),),
    # Chunk 2: UGA has iso3, misses iso2 -- the EARLIER-requested column
    # (`recipient_iso2`) misses for the first time only in this later chunk.
    ((_STAT_VAR, "country/UGA", "2016", 400.0, "USD"),),
)


def _render_two_columns(chunks: Callable[[], Iterator[Frame]]) -> tuple:
    rendered = render_and_store(
        frame_chunks=chunks,
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
        groupby_fold_max_rows=2_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered.artifact, rendered.warnings


def test_streamed_warning_order_matches_whole_table_order_across_two_columns(
    monkeypatch: pytest.MonkeyPatch,
):
    """The two columns first miss in reverse order to `options.columns`. The whole-table path
    reports request order, and streamed warnings must match it.
    """
    chunks = frame_chunks_of(_COLUMNS, _TWO_COLUMN_ROWS_BY_CHUNK)
    _whole_artifact, whole_table_warnings = _render_two_columns(chunks)
    monkeypatch.setattr(render_module, "_RESHAPE_CHUNK_ROWS", 1)
    _streamed_artifact, streamed_warnings = _render_two_columns(chunks)

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
