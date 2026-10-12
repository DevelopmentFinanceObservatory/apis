"""Every mint stores the version's canonical frame (Parquet, rendering `{"format": "frame"}`) on its
record, off mint responses and signatures, and the head names a version only once that frame is
durable (I4). Tests read rows from the SQLite-shaped `datasets.version_artifact` store.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
from collections.abc import Buffer, Callable
from contextlib import contextmanager

import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import Frame

from artifact_store import (
    ArtifactWriteError,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from datasets_api.render import CANONICAL_FRAME_RENDERING, render_and_store_canonical_frame
from datasets_api.serve import CANONICAL_FRAME_STORE_FAILED_MARKER
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.formats import (
    CANONICAL_FRAME_MEDIA_TYPE,
    FRAME1_MEDIA_TYPE,
    read_canonical_frame,
    write_canonical_frame,
)
from materialise.frame_build import build_frame
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.artifact_store.conftest import FakeArtifactInventoryDatasetStore, FakeGCSClient

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore, _BufferedSink
from ._frame_doubles import (
    _BASE_URL,
    _body,
    _canonical_frame_row,
    _coverage_service,
    _graph,
    _sdmx_transport,
    _statistics_row,
    _version_artifact_rows,
)
from ._snapshot_doubles import FakeSnapshotReader
from ._store_fault_doubles import PoolBoundedFakeDatasetStore
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter


def _client(*, artifact_store, version_writer=None, head_writer=None):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=artifact_store,
        version_writer=version_writer,
        head_writer=head_writer,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )


def _fake_url(*, digest: str, tier: str = "working") -> str:
    return f"fake://artifact/{tier}/{digest}"


@pytest.fixture
def version_store():
    store = FakeVersionRecordDatasetStore()
    try:
        yield store
    finally:
        store.close()


@pytest.mark.parametrize("format", ["csv", "xlsx", "parquet"])
@pytest.mark.parametrize("bundle", [False, True])
def test_a_mint_returns_only_the_requested_rendering_and_records_the_canonical_frame_too(
    format: str, bundle: bool, version_store: FakeVersionRecordDatasetStore
):
    artifact_store = FakeArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    resp = client.post("/api/datasets", json=_body(format=format, bundle=bundle))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()

    # C79 returns the requested artifact and its `series-statistics/1` artifact.
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 2
    requested_artifact = next(a for a in artifacts if a["rendering"] != {"statistics": "series"})
    assert requested_artifact["rendering"]["format"] == format
    assert requested_artifact["rendering"]["bundle"] is bundle
    assert all(a["rendering"] != CANONICAL_FRAME_RENDERING for a in artifacts)

    # The version record carries all three: the requested rendering, statistics, and the
    # canonical frame. Nothing writes frame1.
    rows = _version_artifact_rows(
        version_store,
        subject_id=manifest["definition"]["spec_id"],
        content_version=manifest["content_version"],
    )
    assert len(rows) == 3
    assert {row["artifact_digest"] for row in rows} >= {requested_artifact["artifact_digest"]}
    frame_row = _canonical_frame_row(rows)
    assert frame_row["media_type"] == CANONICAL_FRAME_MEDIA_TYPE
    assert not any(row["media_type"] == FRAME1_MEDIA_TYPE for row in rows)
    _statistics_row(rows)


def test_c79_a_mint_and_a_definition_retrieval_each_return_exactly_the_requested_plus_statistics(
    version_store: FakeVersionRecordDatasetStore,
):
    """C79 returns the requested artifact and statistics, while keeping the stored canonical
    frame off both mint and definition responses.
    """
    artifact_store = FakeArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    mint_resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert mint_resp.status_code == 200, mint_resp.text
    mint_manifest = mint_resp.json()
    assert len(mint_manifest["artifacts"]) == 2
    assert all(a["rendering"] != CANONICAL_FRAME_RENDERING for a in mint_manifest["artifacts"])

    spec_id = mint_manifest["definition"]["spec_id"]
    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=mint_manifest["content_version"]
    )
    _canonical_frame_row(rows)

    # GET uses default mint options and returns the requested rendering plus statistics.
    retrieval_resp = client.get(f"/api/datasets/{spec_id}")
    assert retrieval_resp.status_code == 200, retrieval_resp.text
    retrieval_manifest = retrieval_resp.json()
    assert len(retrieval_manifest["artifacts"]) == 2
    assert all(a["rendering"] != CANONICAL_FRAME_RENDERING for a in retrieval_manifest["artifacts"])


def test_the_canonical_frame_bytes_read_back_as_the_minted_frame(
    version_store: FakeVersionRecordDatasetStore,
):
    artifact_store = FakeArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    rows = _version_artifact_rows(
        version_store,
        subject_id=manifest["definition"]["spec_id"],
        content_version=manifest["content_version"],
    )
    frame_row = _canonical_frame_row(rows)

    # The canonical frame is unsigned. Fetch its digest-keyed object directly from the fake store.
    frame = read_canonical_frame(
        artifact_store.fetch(_fake_url(digest=frame_row["artifact_digest"]))
    )
    assert frame.columns == (
        "stat_var",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    assert frame.rows == (
        (
            "dcid:MalariaGrantDisbursements",
            "country/KEN",
            "2015",
            1000.0,
            "USD",
            "GrantEquivalent",
            "OECD_DAC2A",
            None,
        ),
    )


def test_a_repeat_mint_reuses_the_same_canonical_frame_digest(
    version_store: FakeVersionRecordDatasetStore,
):
    artifact_store = FakeArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)
    body = _body(format="csv", bundle=False)

    first = client.post("/api/datasets", json=body)
    second = client.post("/api/datasets", json=body)

    assert first.status_code == 200 and second.status_code == 200
    first_manifest, second_manifest = first.json(), second.json()
    assert first_manifest["content_version"] == second_manifest["content_version"]

    rows = _version_artifact_rows(
        version_store,
        subject_id=first_manifest["definition"]["spec_id"],
        content_version=first_manifest["content_version"],
    )
    # The unique rendering key keeps repeated frame and statistics writes to one row each.
    assert len(rows) == 3
    _canonical_frame_row(rows)
    _statistics_row(rows)


def _write_error(data: bytes) -> ArtifactWriteError:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return ArtifactWriteError(object_name=digest, digest=digest)


def _parquet_columns(data: bytes) -> list[str] | None:
    if not data.startswith(b"PAR1"):
        return None
    return pq.read_schema(io.BytesIO(data)).names


def _is_canonical_frame_payload(data: bytes) -> bool:
    """Identifies the canonical frame by its grammar `time_period` column, which the statistics
    Parquet does not carry."""
    columns = _parquet_columns(data)
    return columns is not None and "time_period" in columns and "first_period" not in columns


def _is_statistics_payload(data: bytes) -> bool:
    columns = _parquet_columns(data)
    return columns is not None and "first_period" in columns


def _is_requested_render_payload(data: bytes) -> bool:
    """Identifies the requested rendering's write, a CSV for these mints."""
    return not data.startswith(b"PAR1")


class _CanonicalFramePutFailsArtifactStore(FakeArtifactStore):
    """Fails the selected canonical frame write while passing all other writes through.
    Payload classification makes the failure deterministic despite concurrent write order.
    """

    def __init__(
        self, *, fail_on_call: int, error_factory: Callable[[bytes], BaseException]
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._frame_calls = 0
        self._fail_on_call = fail_on_call
        self._error_factory = error_factory

    def put(self, data: bytes, *, tier: Tier):
        if _is_canonical_frame_payload(data):
            with self._lock:
                self._frame_calls += 1
                call_number = self._frame_calls
            if call_number == self._fail_on_call:
                raise self._error_factory(data)
        return super().put(data, tier=tier)


def _table_rows(version_store: FakeVersionRecordDatasetStore, table: str) -> list[dict]:
    with version_store.transaction() as tx:
        return tx.fetch_all(f"SELECT * FROM datasets.{table}", ())


def test_i4_a_failed_canonical_frame_upload_refuses_the_mint_and_leaves_the_head_alone(
    version_store: FakeVersionRecordDatasetStore,
):
    """I4: the head names a version only once its canonical frame is durable. A failed frame
    upload refuses the mint retryably, writes no materialisation row and advances no head."""
    artifact_store = _CanonicalFramePutFailsArtifactStore(
        fail_on_call=1, error_factory=_write_error
    )
    client = _client(
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=LiveHeadStore(store=version_store),
    )

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "service_unavailable"
    assert _table_rows(version_store, "subject_head") == []
    assert _table_rows(version_store, "materialisation") == []


def test_a_mint_after_a_failed_canonical_frame_upload_writes_the_frame_and_the_head(
    version_store: FakeVersionRecordDatasetStore,
):
    artifact_store = _CanonicalFramePutFailsArtifactStore(
        fail_on_call=1, error_factory=_write_error
    )
    head_store = LiveHeadStore(store=version_store)
    client = _client(
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=head_store,
    )
    body = _body(format="csv", bundle=False)

    first = client.post("/api/datasets", json=body)
    second = client.post("/api/datasets", json=body)

    assert first.status_code == 503, first.text
    assert second.status_code == 200, second.text
    manifest = second.json()
    spec_id = manifest["definition"]["spec_id"]
    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=manifest["content_version"]
    )
    _canonical_frame_row(rows)
    head = head_store.read_head(spec_id)
    assert head is not None and head.content_version == manifest["content_version"]


def test_a_non_artifact_store_exception_from_the_canonical_frame_write_fails_the_mint():
    artifact_store = _CanonicalFramePutFailsArtifactStore(
        fail_on_call=1, error_factory=lambda _data: TypeError("not an artifact-store failure")
    )
    client = _client(artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 500, resp.text


class _BarrierSynchronizedArtifactStore(FakeArtifactStore):
    """Blocks the canonical frame write and the requested rendering's write at a two-party
    barrier to prove overlap. The statistics write bypasses the barrier.
    """

    def __init__(self, *, timeout: float = 5.0) -> None:
        super().__init__()
        self._barrier = threading.Barrier(2, timeout=timeout)

    def put(self, data: bytes, *, tier: Tier):
        if not _is_statistics_payload(data):
            self._barrier.wait()
        return super().put(data, tier=tier)


def test_the_canonical_frame_and_requested_rendering_writes_overlap(
    version_store: FakeVersionRecordDatasetStore,
):
    artifact_store = _BarrierSynchronizedArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text


class _RenderFailsWhileFrameStoresArtifactStore(FakeArtifactStore):
    """Fails the requested rendering's write, then lets the canonical frame write finish and
    records its completion. The statistics write passes through.

    The frame write waits for the failure first, so `frame_stored` proves the door joined it.
    """

    def __init__(self, *, error_factory: Callable[[bytes], BaseException]) -> None:
        super().__init__()
        self.render_failed = threading.Event()
        self.frame_stored = threading.Event()
        self._error_factory = error_factory

    def put(self, data: bytes, *, tier: Tier):
        if _is_canonical_frame_payload(data):
            assert self.render_failed.wait(timeout=5.0), "the rendering's write never failed"
            stored = super().put(data, tier=tier)
            self.frame_stored.set()
            return stored
        if _is_statistics_payload(data):
            return super().put(data, tier=tier)
        try:
            raise self._error_factory(data)
        finally:
            self.render_failed.set()


def test_a_requested_rendering_store_failure_still_joins_the_canonical_frame_write():
    artifact_store = _RenderFailsWhileFrameStoresArtifactStore(error_factory=_write_error)
    client = _client(artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    # The frame write completes before the rendering's failure reaches the caller.
    assert artifact_store.frame_stored.is_set()


class _AllPutsFailArtifactStore(FakeArtifactStore):
    """`put()` raises `error_factory(data)` for every call, whether it writes the statistics, the
    requested rendering or the canonical frame. This models a correlated store outage.
    """

    def __init__(self, *, error_factory: Callable[[bytes], BaseException]) -> None:
        super().__init__()
        self._error_factory = error_factory

    def put(self, data: bytes, *, tier: Tier):
        raise self._error_factory(data)


@pytest.mark.parametrize(
    "artifact_store_factory",
    [
        pytest.param(
            lambda: _AllPutsFailArtifactStore(error_factory=_write_error), id="every-write-fails"
        ),
        pytest.param(
            lambda: _CanonicalFramePutFailsArtifactStore(
                fail_on_call=1, error_factory=_write_error
            ),
            id="the-frame-write-alone-fails",
        ),
    ],
)
def test_a_canonical_frame_store_failure_logs_its_marker_once(
    artifact_store_factory: Callable[[], FakeArtifactStore],
    caplog: pytest.LogCaptureFixture,
):
    """The frame write's failure logs its marker once, whether it alone fails the mint or
    fails beside the failure the response reports, the statistics write's, the first write the
    mint makes outside the frame worker.
    """
    client = _client(artifact_store=artifact_store_factory())

    with caplog.at_level(logging.ERROR, logger="datasets_api.serve"):
        resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    marker_records = [r for r in errors if CANONICAL_FRAME_STORE_FAILED_MARKER in r.getMessage()]
    assert len(marker_records) == 1


class _RenderFailsWhileFrameRaisesArtifactStore(FakeArtifactStore):
    """Raises separate exceptions for the canonical frame and requested-rendering payloads. The
    statistics write passes through."""

    def __init__(
        self,
        *,
        frame_error_factory: Callable[[bytes], BaseException],
        render_error_factory: Callable[[bytes], BaseException],
    ) -> None:
        super().__init__()
        self._frame_error_factory = frame_error_factory
        self._render_error_factory = render_error_factory

    def put(self, data: bytes, *, tier: Tier):
        if _is_canonical_frame_payload(data):
            raise self._frame_error_factory(data)
        if _is_requested_render_payload(data):
            raise self._render_error_factory(data)
        return super().put(data, tier=tier)


def test_a_canonical_frame_bug_stays_visible_alongside_a_requested_rendering_store_failure(
    caplog: pytest.LogCaptureFixture,
):
    """A frame programming error is logged when the requested rendering's write also fails.
    The response reports the rendering's failure.
    """
    artifact_store = _RenderFailsWhileFrameRaisesArtifactStore(
        frame_error_factory=lambda _data: TypeError("not an artifact-store failure"),
        render_error_factory=_write_error,
    )
    client = _client(artifact_store=artifact_store)

    with caplog.at_level(logging.ERROR, logger="datasets_api.serve"):
        resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    # Filter to this module's logger. The mint door logs the rendering's `ArtifactWriteError`
    # separately.
    errors = [
        r for r in caplog.records if r.name == "datasets_api.serve" and r.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[0] is TypeError


class _BarrierGatedTransaction:
    """Gates inventory inserts at a shared barrier.
    Only the first two inserts wait.
    """

    def __init__(
        self, delegate: object, *, barrier: threading.Barrier, should_wait: Callable[[], bool]
    ) -> None:
        self._delegate = delegate
        self._barrier = barrier
        self._should_wait = should_wait

    def execute(self, sql: str, params: object = ()) -> None:
        if "INSERT INTO datasets.artifact_inventory" in sql and self._should_wait():
            self._barrier.wait()
        self._delegate.execute(sql, params)

    def fetch_one(self, sql: str, params: object = ()) -> object:
        return self._delegate.fetch_one(sql, params)

    def fetch_all(self, sql: str, params: object = ()) -> object:
        return self._delegate.fetch_all(sql, params)


class _BarrierBoundedArtifactInventoryStore:
    """Bounds open inventory transactions and gates the first two inserts to prove concurrent
    writes. Each thread uses a separate SQLite-backed fake store.
    """

    def __init__(self, *, pool_size: int, barrier: threading.Barrier) -> None:
        self._semaphore = threading.Semaphore(pool_size)
        self._barrier = barrier
        self._local = threading.local()
        self._lock = threading.Lock()
        self._insert_calls = 0

    def _should_wait(self) -> bool:
        with self._lock:
            self._insert_calls += 1
            return self._insert_calls <= 2

    def _thread_local_store(self) -> FakeArtifactInventoryDatasetStore:
        store = getattr(self._local, "store", None)
        if store is None:
            store = FakeArtifactInventoryDatasetStore()
            self._local.store = store
        return store

    @contextmanager
    def transaction(self):
        if not self._semaphore.acquire(blocking=False):
            raise RuntimeError(
                "_BarrierBoundedArtifactInventoryStore: no free connection -- more "
                "transactions were opened concurrently than the configured pool size allows"
            )
        try:
            with self._thread_local_store().transaction() as tx:
                yield _BarrierGatedTransaction(
                    tx, barrier=self._barrier, should_wait=self._should_wait
                )
        finally:
            self._semaphore.release()


def test_the_write_pair_never_needs_more_than_two_connections(
    version_store: FakeVersionRecordDatasetStore, monkeypatch: pytest.MonkeyPatch
):
    """A barrier proves the canonical frame's inventory write overlaps the mint's first other
    one, the statistics write. `pool_size=2` suffices for both transactions. A third connection
    raises `RuntimeError`.
    """
    # The canonical frame is unsigned. Signing the statistics and the requested rendering
    # requires IAM credentials. Stub the signer.
    monkeypatch.setattr(
        "artifact_store.store.sign_get_url",
        lambda **kwargs: "https://storage.googleapis.com/fake-signed-url",
    )
    barrier = threading.Barrier(2, timeout=5.0)
    inventory_store = _BarrierBoundedArtifactInventoryStore(pool_size=2, barrier=barrier)
    artifact_store = LiveArtifactStore(
        buckets={Tier.working: "fake-working-bucket", Tier.pinned: "fake-pinned-bucket"},
        inventory=LiveArtifactInventory(store=inventory_store),
        client=FakeGCSClient(),
    )
    registry_store = PoolBoundedFakeDatasetStore(pool_size=2)
    writer = LiveVersionRecordWriter(store=version_store)
    client = make_client(
        registry=LiveSpecRegistry(store=registry_store),
        spec_reader=LiveSpecReader(store=registry_store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=artifact_store,
        version_writer=writer,
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text


class _WriteSizeRecordingSink(_BufferedSink):
    def __init__(self, write_sizes: list[int]) -> None:
        super().__init__()
        self._write_sizes = write_sizes

    def write(self, data: Buffer, /) -> int:
        self._write_sizes.append(memoryview(data).nbytes)
        return super().write(data)


class _StreamOnlyArtifactStore(FakeArtifactStore):
    """Records each `open_put` write size and refuses whole-object `put` calls."""

    def __init__(self) -> None:
        super().__init__()
        self.write_sizes: list[int] = []

    def put(self, data: bytes, *, tier: Tier):
        raise AssertionError("the canonical frame must stream through open_put")

    @contextmanager
    def open_put(self, *, tier: Tier):
        sink = _WriteSizeRecordingSink(self.write_sizes)
        yield sink
        sink._set_result(FakeArtifactStore.put(self, sink.getvalue(), tier=tier))


def test_render_and_store_canonical_frame_streams_every_chunk_through_open_put():
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": f"country/R{index:04d}",
                "time_period": "2020",
                "value": float(index),
            }
            for index in range(120)
        ],
    )
    chunks = [
        Frame(columns=frame.columns, table=frame.table.slice(start, 25))
        for start in range(0, 120, 25)
    ]
    whole = io.BytesIO()
    write_canonical_frame((frame,), sink=whole)
    artifact_store = _StreamOnlyArtifactStore()

    record = render_and_store_canonical_frame(
        frame_chunks=chunks,
        subject_id="spec_streamed",
        content_version="cv_streamed",
        artifact_store=artifact_store,
    )

    assert record.artifact_digest == "sha256:" + hashlib.sha256(whole.getvalue()).hexdigest()
    assert record.media_type == CANONICAL_FRAME_MEDIA_TYPE
    assert record.rendering == CANONICAL_FRAME_RENDERING == {"format": "frame"}
    assert sum(artifact_store.write_sizes) == len(whole.getvalue())


def test_a_version_is_not_recorded_when_its_frame_and_requested_rendering_both_fail_to_store(
    version_store: FakeVersionRecordDatasetStore,
):
    """A version on record always holds its canonical frame, so a mint whose frame write and
    requested rendering write both fail records no version."""
    artifact_store = _RenderFailsWhileFrameRaisesArtifactStore(
        frame_error_factory=_write_error, render_error_factory=_write_error
    )
    client = _client(
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=LiveHeadStore(store=version_store),
    )

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    assert _table_rows(version_store, "version_record") == []
    assert _table_rows(version_store, "subject_head") == []
