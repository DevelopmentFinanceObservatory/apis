"""Rendering immutability (I11): every rendering made from a version's stored canonical frame has
the bytes the mint's own render of the fresh frame wrote, whatever chunks the materialiser
yielded, and rendering it again from the stored frame gives the same bytes. The render converts
each frame row to Python objects once."""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pytest
from dfo_frame_grammar import Frame

from artifact_store import Tier
from datasets_api import render
from datasets_api.mint_models import MintOptions
from datasets_api.render import render_and_store, render_and_store_canonical_frame
from materialise.enrichment import EnrichmentLookup
from materialise.formats import CANONICAL_FRAME_MEDIA_TYPE, read_stored_frame
from materialise.frame_build import build_frame
from tests.materialise._canonical_frame import frame_of_document, stored_canonical_frame_959f8bc

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_ROWS = 9_000
"""More rows than one 8,192-row render slice, so the fresh chunks and the stored frame split
differently."""

_FIRST_CHUNK_ROWS = 3_000
"""The fresh frame's first chunk, whose rows all leave `measurementMethod` null, so a writer that
typed its columns from that chunk alone would type the column apart from a whole-frame read."""

_GRAPH = FakeMintGraphReader(
    arcs={"country/R00001": {"name": ("Recipient one",), "typeOf": ("Country",)}}
)


def _frame() -> Frame:
    return build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:Amount_FinancialFlow",
                "recipient": f"country/R{index:05d}",
                "time_period": "2020",
                "value": index + 0.5,
                "unit": "USD",
                "measurementMethod": None if index < _FIRST_CHUNK_ROWS else "GrantEquivalent",
                "provenance": "OECD_CRS",
            }
            for index in range(_ROWS)
        ],
    )


def _fresh_chunks(frame: Frame) -> list[Frame]:
    table = frame.table
    starts = [0, _FIRST_CHUNK_ROWS, 6_000]
    ends = [*starts[1:], _ROWS]
    return [
        Frame(columns=frame.columns, table=table.slice(start, end - start))
        for start, end in zip(starts, ends, strict=True)
    ]


def _render_digest(chunks, *, options: MintOptions, store: FakeArtifactStore) -> str:
    rendered = render_and_store(
        frame_chunks=lambda: iter(chunks),
        options=options,
        spec={"spec_id": "spec_immutable"},
        provenance_refs=[],
        citations={},
        reproducing_call=None,
        content_version="cv_immutable",
        prepared_at=PREPARED_AT,
        site=SITE,
        artifact_store=store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
        pivot_max_output_cells=5_000_000,
        groupby_fold_max_rows=2_000_000,
        groupby_row_fold_max_rows=700_000,
        resolved_stat_var_refs=(),
    )
    return rendered.artifact.artifact_digest


_OPTIONS = pytest.mark.parametrize(
    "options",
    [
        MintOptions(format="csv"),
        MintOptions(format="xlsx"),
        MintOptions(format="parquet"),
        MintOptions(format="parquet", bundle=False, layout="frame"),
        MintOptions(format="csv", bundle=True),
    ],
    ids=["csv", "xlsx", "parquet", "frame-layout", "bundle"],
)


@_OPTIONS
def test_a_rendering_from_the_stored_frame_has_the_bytes_of_the_fresh_render(
    options: MintOptions, monkeypatch: pytest.MonkeyPatch
):
    frame = _frame()
    store = FakeArtifactStore()
    record = render_and_store_canonical_frame(
        frame_chunks=_fresh_chunks(frame),
        subject_id="spec_immutable",
        content_version="cv_immutable",
        artifact_store=store,
    )
    data = store.get(artifact_digest=record.artifact_digest, tier=Tier.working)

    fresh = _render_digest(_fresh_chunks(frame), options=options, store=store)
    converted = _count_converted_rows(monkeypatch)
    stored = _render_digest(
        (read_stored_frame(data, media_type=record.media_type),), options=options, store=store
    )
    converted_by_stored = converted[0]
    again = _render_digest(
        (read_stored_frame(data, media_type=record.media_type),), options=options, store=store
    )

    assert stored == fresh
    assert again == stored
    assert converted_by_stored == _ROWS


@_OPTIONS
def test_a_frame_stored_before_unit_recipes_renders_as_its_rows_do(options: MintOptions):
    """A canonical frame stored under frame grammar kit 959f8bc, whose unit `request` has no
    `recipe`, renders the bytes of the same rows built by this release."""
    data, _frame_sha256, document = stored_canonical_frame_959f8bc()
    store = FakeArtifactStore()

    stored = _render_digest(
        (read_stored_frame(data, media_type=CANONICAL_FRAME_MEDIA_TYPE),),
        options=options,
        store=store,
    )

    assert stored == _render_digest((frame_of_document(document),), options=options, store=store)


def _count_converted_rows(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts the frame rows converted to Python objects, through `Frame.rows` or
    `canonical_frame_rows`, in a one-element list."""
    converted = [0]
    frame_rows = Frame.rows.fget
    rows_of = render.canonical_frame_rows

    def counted_frame_rows(frame: Frame) -> tuple[tuple[object, ...], ...]:
        converted[0] += frame.table.num_rows
        return frame_rows(frame)

    def counted_rows_of(table: pa.Table) -> Iterator[tuple[object, ...]]:
        converted[0] += table.num_rows
        return rows_of(table)

    monkeypatch.setattr(Frame, "rows", property(counted_frame_rows))
    monkeypatch.setattr(render, "canonical_frame_rows", counted_rows_of)
    return converted
