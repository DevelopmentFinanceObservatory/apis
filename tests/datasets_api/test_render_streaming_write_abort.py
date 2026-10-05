"""Streaming writes abort without recording an artifact when rendering fails mid-write.

Tests use `LiveArtifactStore` with fake GCS and inventory, and set the spill threshold to
8 bytes to exercise staged uploads. Failures come from the chunk iterator, CSV writer, or a
deferred `no_summable_axis` refusal; the iterator case writes too little to open a staging
writer because `TextIOWrapper` buffers the row.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import datasets_api.render as render_module
from artifact_store.store import LiveArtifactStore, Tier
from code_index import CodeIndex, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import (
    CollectionMemberRendering,
    render_and_store,
    render_and_store_collection,
)
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import RollupNotSummableError, Table

from ..artifact_store.conftest import FakeGCSClient, RecordingArtifactInventory
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


def _assert_staging_terminated(client: FakeGCSClient) -> None:
    working_bucket = client.buckets.get(_BUCKETS[Tier.working])
    assert working_bucket is not None and working_bucket.last_writer is not None, (
        "expected this write to cross the (patched, tiny) spill threshold and open a staging writer"
    )
    assert working_bucket.last_writer.terminated is True
    assert working_bucket.last_writer.closed is True


def test_render_and_store_streaming_write_aborts_when_the_chunk_iterator_raises(monkeypatch):
    store, client, inventory = _live_store(monkeypatch)

    class _UpstreamFailure(Exception):
        pass

    def _chunks() -> Iterator[Table]:
        yield Table(
            columns=_COLUMNS,
            rows=(("dcid:MalariaGrantDisbursements", "country/KEN", "2020", 1.0, "USD"),),
        )
        raise _UpstreamFailure("fetch died mid-stream")

    with pytest.raises(_UpstreamFailure):
        render_and_store(
            frame_chunks=_chunks(),
            options=MintOptions(format="csv", bundle=False),
            spec={"spec_id": "streaming-write-abort"},
            provenance_refs=[],
            citations={},
            reproducing_call=None,
            content_version="sha256:" + "0" * 64,
            artifact_store=store,
            entity_columns=("recipient",),
            enrichment=EnrichmentLookup(_GRAPH),
            resolved_stat_var_refs=(),
            pivot_max_output_cells=5_000_000,
            prepared_at=PREPARED_AT,
            site=SITE,
        )

    # `TextIOWrapper` buffers this row, so the iterator raises before `ArtifactSink` opens a
    # staging writer.
    _assert_nothing_landed(client, inventory)


def test_collection_bare_path_aborts_the_failing_members_own_write_on_a_writer_error(
    monkeypatch,
):
    store, client, inventory = _live_store(monkeypatch)

    class _WriterFailure(Exception):
        pass

    def _raising_writer(chunks, *, rendering, sink):
        # Writes enough bytes to cross the spill threshold, then raises before the writer finishes.
        sink.write(next(iter(chunks)).rows[0][0].encode())
        raise _WriterFailure("simulated writer failure")

    monkeypatch.setitem(render_module._FORMAT_WRITERS, "csv", _raising_writer)

    def _member(spec_id: str) -> CollectionMemberRendering:
        def _chunks() -> Iterator[Table]:
            yield Table(
                columns=_COLUMNS,
                rows=((f"dcid:{spec_id}", "country/KEN", "2020", 1.0, "USD"),),
            )

        return CollectionMemberRendering(
            spec_id=spec_id,
            frame_chunks=_chunks,
            provenance_refs=("p1",),
            cited_refs=("p1",),
            content_version=f"cv_{spec_id}",
            prepared_at=PREPARED_AT,
            entity_columns=("recipient",),
            resolved_stat_var_refs=(),
        )

    with pytest.raises(_WriterFailure):
        render_and_store_collection(
            members=[_member("member_a"), _member("member_b")],
            options=MintOptions(format="csv", bundle=False),
            spec={"collection_id": "col_bare_abort"},
            citations={},
            reproducing_call=None,
            content_version="sha256:" + "1" * 64,
            artifact_store=store,
            enrichment=EnrichmentLookup(_GRAPH),
            pivot_max_output_cells=5_000_000,
            prepared_at=PREPARED_AT,
            site=SITE,
        )

    # Rendering is sequential, so the writer raises on the first member before the second starts.
    _assert_nothing_landed(client, inventory)
    _assert_staging_terminated(client)


_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-09T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)


def _statvar(dcid: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("recipient",),
        constraint_properties={},
    )


_INDEX_TWO_CONTRIBUTORS = CodeIndex(
    stamp=_STAMP, declarations=(), codes=(), statvars=(_statvar("SV_A"), _statvar("SV_B"))
)


def test_collection_bundle_path_aborts_the_whole_archive_on_a_deferred_refusal(monkeypatch):
    """A groupby-collapses-nothing member (no pivot, no declared-summable
    axis at all) whose own two contributing StatVars trip the deferred
    `no_summable_axis` refusal (`_reshape_stream`'s own docstring) only
    once the writer has fully drained that member's own chunks -- proving
    a refusal discovered at the END of one member's own pass still leaves
    the WHOLE bundle unrecorded, not just that member's own entry.
    """
    store, client, inventory = _live_store(monkeypatch)

    def _good_member(spec_id: str) -> CollectionMemberRendering:
        def _chunks() -> Iterator[Table]:
            yield Table(
                columns=_COLUMNS,
                rows=((f"dcid:{spec_id}", "country/KEN", "2020", 1.0, "USD"),),
            )

        return CollectionMemberRendering(
            spec_id=spec_id,
            frame_chunks=_chunks,
            provenance_refs=("p1",),
            cited_refs=("p1",),
            content_version=f"cv_{spec_id}",
            prepared_at=PREPARED_AT,
            entity_columns=("recipient",),
            resolved_stat_var_refs=("dcid:SV_A",),
        )

    def _failing_member(spec_id: str) -> CollectionMemberRendering:
        def _chunks() -> Iterator[Table]:
            yield Table(
                columns=_COLUMNS,
                rows=(
                    ("dcid:SV_A", "country/KEN", "2020", 1.0, "USD"),
                    ("dcid:SV_B", "country/KEN", "2021", 2.0, "USD"),
                ),
            )

        return CollectionMemberRendering(
            spec_id=spec_id,
            frame_chunks=_chunks,
            provenance_refs=("p1",),
            cited_refs=("p1",),
            content_version=f"cv_{spec_id}",
            prepared_at=PREPARED_AT,
            entity_columns=("recipient",),
            resolved_stat_var_refs=("dcid:SV_A", "dcid:SV_B"),
        )

    options = MintOptions(
        format="csv", bundle=True, groupby=GroupbyOptions(by=["recipient"], op="sum")
    )

    with pytest.raises(RollupNotSummableError) as exc_info:
        render_and_store_collection(
            members=[_good_member("member_a"), _failing_member("member_b")],
            options=options,
            spec={"collection_id": "col_bundle_abort", "members": ["member_a", "member_b"]},
            citations={},
            reproducing_call=None,
            content_version="sha256:" + "2" * 64,
            artifact_store=store,
            enrichment=EnrichmentLookup(_GRAPH),
            code_index=_INDEX_TWO_CONTRIBUTORS,
            pivot_max_output_cells=5_000_000,
            prepared_at=PREPARED_AT,
            site=SITE,
        )
    assert exc_info.value.precondition == "no_summable_axis"

    # Both members share one archive. Member A's entry streams before member B's refusal
    # aborts the ZIP, so neither artifact is recorded.
    _assert_nothing_landed(client, inventory)
    _assert_staging_terminated(client)
