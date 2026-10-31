"""Streaming writes abort without recording an artifact when rendering fails mid-write.

Tests use `LiveArtifactStore` with fake GCS and inventory, and set the spill threshold to
8 bytes to exercise staged uploads. Failures come from the chunk iterator, during the long
output's period-grain pass or partway through its write, the CSV writer, or a deferred
`no_summable_axis` refusal.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from dfo_frame_grammar import Frame

from artifact_store.store import LiveArtifactStore, Tier
from datasets_api import render as render_module
from datasets_api.mint_models import MintOptions
from datasets_api.render import (
    render_and_store,
)
from materialise.enrichment import EnrichmentLookup

from ..artifact_store.conftest import FakeGCSClient, RecordingArtifactInventory
from ._frame_chunks import frame_of_rows
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_BUCKETS = {
    Tier.working: "dev-fin-observatory-datasets-working",
    Tier.pinned: "dev-fin-observatory-datasets-pinned",
}
_GRAPH = FakeMintGraphReader()
_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit")


def _live_store(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LiveArtifactStore, FakeGCSClient, RecordingArtifactInventory]:
    """A `LiveArtifactStore` wired to fresh fakes, with `SPILL_THRESHOLD_BYTES`
    patched down so every write here crosses it -- see this module's own
    docstring.
    """
    monkeypatch.setattr("artifact_store.store.SPILL_THRESHOLD_BYTES", 8)
    client = FakeGCSClient()
    inventory = RecordingArtifactInventory()
    store = LiveArtifactStore(buckets=_BUCKETS, inventory=inventory, client=client)
    return store, client, inventory


def _assert_nothing_landed(client: FakeGCSClient, inventory: RecordingArtifactInventory) -> None:
    assert not any(bucket.objects for bucket in client.buckets.values()), (
        "no object (staged or final) should exist under any digest name"
    )
    assert inventory.calls == [], "no inventory row should have been recorded"


def _rows(count: int) -> list[tuple[object, ...]]:
    """`count` rows, one a recipient."""
    return [
        ("dcid:MalariaGrantDisbursements", f"country/R{index:04d}", "2020", 1.0, "USD")
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("failing_pass", "rows", "staged"),
    [
        # Enough rows to pass `TextIOWrapper`'s buffer, so the write pass opens a staging writer
        # and writes to it before the iterator raises.
        pytest.param(2, 2_000, True, id="write-pass"),
        # Long output reads the frame once for its period grain before writing anything.
        pytest.param(1, 1, False, id="grain-pre-pass"),
    ],
)
def test_render_and_store_streaming_write_aborts_when_the_chunk_iterator_raises(
    monkeypatch, failing_pass, rows, staged
):
    store, client, inventory = _live_store(monkeypatch)
    # The render passes each 1,000 rows on as they arrive, before the iterator raises.
    monkeypatch.setattr(render_module, "_RESHAPE_CHUNK_ROWS", 1_000)
    passes: list[int] = []
    frame = frame_of_rows(_COLUMNS, _rows(rows))

    class _UpstreamFailure(Exception):
        pass

    def _chunks() -> Iterator[Frame]:
        passes.append(len(passes) + 1)
        yield frame
        if len(passes) == failing_pass:
            raise _UpstreamFailure("fetch died mid-stream")

    with pytest.raises(_UpstreamFailure):
        render_and_store(
            frame_chunks=_chunks,
            options=MintOptions(format="csv", bundle=False),
            spec={"spec_id": "streaming-write-abort"},
            provenance_refs=[],
            citations={},
            reproducing_call=None,
            content_version="sha256:" + "0" * 64,
            artifact_store=store,
            entity_columns=("recipient",),
            enrichment=EnrichmentLookup(_GRAPH, parties=lambda: {}),
            resolved_stat_var_refs=(),
            pivot_max_output_cells=5_000_000,
            groupby_fold_max_rows=2_000_000,
            prepared_at=PREPARED_AT,
            site=SITE,
        )

    assert passes == list(range(1, failing_pass + 1))
    writers = [bucket.last_writer for bucket in client.buckets.values() if bucket.last_writer]
    assert [writer.terminated for writer in writers] == ([True] if staged else [])
    _assert_nothing_landed(client, inventory)
