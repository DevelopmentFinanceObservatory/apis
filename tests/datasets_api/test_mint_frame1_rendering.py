"""Frame1 is stored on definition version records but excluded from mint responses and signatures.
Tests read its row from the SQLite-shaped `datasets.version_artifact` store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Buffer, Callable
from contextlib import contextmanager

import pytest
from dfo_frame_grammar import Frame, canonical_frame_bytes

import materialise.formats as materialise_formats
from artifact_store import (
    ArtifactSigningError,
    ArtifactWriteError,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from datasets_api.render import render_and_store_frame1
from datasets_api.revision import SentinelRevisionReader
from datasets_api.serve import FRAME1_STORE_FAILED_MARKER
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.frame_build import build_frame
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.artifact_store.conftest import FakeArtifactInventoryDatasetStore, FakeGCSClient

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore, _BufferedSink
from ._frame1_doubles import (
    _BASE_URL,
    _body,
    _coverage_service,
    _frame1_row,
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


def _client(*, artifact_store, version_writer=None):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        revision_reader=SentinelRevisionReader(),
        artifact_store=artifact_store,
        version_writer=version_writer,
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
def test_a_mint_returns_only_the_requested_rendering_and_records_frame1_too(
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
    assert requested_artifact["media_type"] != FRAME1_MEDIA_TYPE

    # The version record carries all three: the requested rendering,
    # statistics, and frame1.
    rows = _version_artifact_rows(
        version_store,
        subject_id=manifest["definition"]["spec_id"],
        content_version=manifest["content_version"],
    )
    assert len(rows) == 3
    assert {row["artifact_digest"] for row in rows} >= {requested_artifact["artifact_digest"]}
    frame1_row = _frame1_row(rows)
    assert frame1_row["media_type"] == FRAME1_MEDIA_TYPE == "application/vnd.dfo.frame1+json"
    _statistics_row(rows)


def test_frame1_is_not_written_above_the_d8_row_gate(
    monkeypatch, version_store: FakeVersionRecordDatasetStore
):
    """Patching the `FRAME1_MAX_ROWS` ceiling to 0 makes one positive row hit the same gate.
    The content version and requested artifacts are still written.
    """
    monkeypatch.setattr("datasets_api.serve.FRAME1_MAX_ROWS", 0)
    artifact_store = FakeArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert manifest["content_version"]

    rows = _version_artifact_rows(
        version_store,
        subject_id=manifest["definition"]["spec_id"],
        content_version=manifest["content_version"],
    )
    # The requested (CSV) artifact's own row and the statistics artifact's
    # own row -- no frame1 row. The row gate bounds frame1 alone.
    assert len(rows) == 2
    assert not any(row["media_type"] == FRAME1_MEDIA_TYPE for row in rows)
    _statistics_row(rows)


def test_c79_a_mint_and_a_definition_retrieval_each_return_exactly_the_requested_plus_statistics(
    version_store: FakeVersionRecordDatasetStore,
):
    """C79 returns the requested artifact and statistics, while keeping the stored frame1 artifact
    off both mint and definition responses.
    """
    artifact_store = FakeArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    mint_resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert mint_resp.status_code == 200, mint_resp.text
    mint_manifest = mint_resp.json()
    assert len(mint_manifest["artifacts"]) == 2
    assert not any(a["media_type"] == FRAME1_MEDIA_TYPE for a in mint_manifest["artifacts"])

    # The version record carries frame1 for this content version. The response omits it.
    spec_id = mint_manifest["definition"]["spec_id"]
    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=mint_manifest["content_version"]
    )
    _frame1_row(rows)

    # GET uses default mint options and returns the requested rendering plus statistics.
    retrieval_resp = client.get(f"/api/datasets/{spec_id}")
    assert retrieval_resp.status_code == 200, retrieval_resp.text
    retrieval_manifest = retrieval_resp.json()
    assert len(retrieval_manifest["artifacts"]) == 2
    assert not any(a["media_type"] == FRAME1_MEDIA_TYPE for a in retrieval_manifest["artifacts"])


def test_the_frame1_rendering_bytes_decode_as_the_canonical_frame1_document(
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
    frame1_row = _frame1_row(rows)

    # Frame1 is unsigned. Fetch its digest-keyed object directly from the fake store.
    fetched = artifact_store.fetch(_fake_url(digest=frame1_row["artifact_digest"]))
    doc = json.loads(fetched)
    assert doc["frame_version"] == "frame1"
    assert doc["columns"] == [
        "stat_var",
        "recipient",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    ]
    assert len(doc["rows"]) == 1


def test_a_repeat_mint_reuses_the_same_frame1_artifact_digest(
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
    # The unique rendering key keeps repeated frame1 and statistics writes to one row each.
    assert len(rows) == 3
    _frame1_row(rows)
    _statistics_row(rows)


def test_content_version_and_the_frame1_artifact_digest_are_different_hashes(
    version_store: FakeVersionRecordDatasetStore,
):
    """The content version hashes a lineage preimage. The frame1 digest hashes canonical bytes.
    The formats are `cv_` plus 32 hex digits and `sha256:` plus 64.
    """
    artifact_store = FakeArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    content_version = manifest["content_version"]
    rows = _version_artifact_rows(
        version_store,
        subject_id=manifest["definition"]["spec_id"],
        content_version=content_version,
    )
    frame1_row = _frame1_row(rows)
    frame1_digest = frame1_row["artifact_digest"]

    assert content_version.startswith("cv_")
    assert len(content_version) == len("cv_") + 32
    assert frame1_digest.startswith("sha256:")
    assert len(frame1_digest) == len("sha256:") + 64
    assert content_version != frame1_digest
    assert content_version.removeprefix("cv_") != frame1_digest.removeprefix("sha256:")


def _write_error(data: bytes) -> ArtifactWriteError:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    return ArtifactWriteError(object_name=digest, digest=digest)


def _is_frame1_payload(data: bytes) -> bool:
    """Identifies frame1 by its canonical JSON `frame_version` field."""
    try:
        doc = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(doc, dict) and doc.get("frame_version") == "frame1"


def _is_statistics_payload(data: bytes) -> bool:
    """Identifies the series-statistics artifact, the only Parquet write of a CSV mint."""
    return data.startswith(b"PAR1")


def _is_requested_render_payload(data: bytes) -> bool:
    """Identifies the requested rendering's write, which is neither frame1 nor the statistics."""
    return not _is_frame1_payload(data) and not _is_statistics_payload(data)


class _Frame1PutFailsArtifactStore(FakeArtifactStore):
    """Fails the selected frame1 payload write while passing all other writes through.
    Payload classification makes the failure deterministic despite concurrent write order.
    `fail_on_call` counts frame1 writes across all mints that share the instance.
    """

    def __init__(
        self, *, fail_on_call: int, error_factory: Callable[[bytes], BaseException]
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._frame1_calls = 0
        self._fail_on_call = fail_on_call
        self._error_factory = error_factory

    def put(self, data: bytes, *, tier: Tier):
        if _is_frame1_payload(data):
            with self._lock:
                self._frame1_calls += 1
                call_number = self._frame1_calls
            if call_number == self._fail_on_call:
                raise self._error_factory(data)
        return super().put(data, tier=tier)


def test_a_frame1_only_store_failure_still_succeeds_the_callers_requested_mint(
    caplog: pytest.LogCaptureFixture, version_store: FakeVersionRecordDatasetStore
):
    artifact_store = _Frame1PutFailsArtifactStore(fail_on_call=1, error_factory=_write_error)
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    with caplog.at_level(logging.ERROR, logger="datasets_api.serve"):
        resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 2
    requested_artifact = next(a for a in artifacts if a["rendering"] != {"statistics": "series"})
    assert requested_artifact["rendering"]["format"] == "csv"

    spec_id = manifest["definition"]["spec_id"]
    content_version = manifest["content_version"]
    rows = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=content_version
    )
    # The version record carries requested and statistics artifacts.
    # The failed frame1 write leaves no row.
    assert len(rows) == 2
    assert not any(row["rendering"] == {"format": "frame1"} for row in rows)
    _statistics_row(rows)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(
        FRAME1_STORE_FAILED_MARKER in r.getMessage()
        and spec_id in r.getMessage()
        and content_version in r.getMessage()
        for r in errors
    )


def test_a_non_artifact_store_exception_from_the_frame1_write_still_fails_the_mint():
    """A bug-class exception (anything other than `ArtifactStoreError`)
    from `render_and_store_frame1` must still fail the mint -- only the
    artifact-store failure family is absorbed as best-effort.
    """
    artifact_store = _Frame1PutFailsArtifactStore(
        fail_on_call=1, error_factory=lambda _data: TypeError("not an artifact-store failure")
    )
    client = _client(artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 500, resp.text


def test_a_signing_failure_on_the_requested_artifact_still_fails_the_whole_mint():
    """A signing failure on a requested artifact fails the mint.
    Frame1 is unsigned.
    """

    class _SigningFailsArtifactStore(FakeArtifactStore):
        def signed_get_url(
            self, digest: str, *, tier: Tier, ttl, response_type: str, response_disposition: str
        ):
            raise ArtifactSigningError(
                bucket="dev-fin-observatory-datasets-working", object_name=digest, digest=digest
            )

    client = _client(artifact_store=_SigningFailsArtifactStore())

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "service_unavailable"


def test_a_repeat_mint_heals_a_frame1_write_that_failed_the_first_time(
    caplog: pytest.LogCaptureFixture, version_store: FakeVersionRecordDatasetStore
):
    artifact_store = _Frame1PutFailsArtifactStore(fail_on_call=1, error_factory=_write_error)
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)
    body = _body(format="csv", bundle=False)

    with caplog.at_level(logging.ERROR, logger="datasets_api.serve"):
        first = client.post("/api/datasets", json=body)

    assert first.status_code == 200, first.text
    first_manifest = first.json()
    assert len(first_manifest["artifacts"]) == 2

    spec_id = first_manifest["definition"]["spec_id"]
    content_version = first_manifest["content_version"]
    rows_after_first = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=content_version
    )
    assert len(rows_after_first) == 2
    assert not any(row["rendering"] == {"format": "frame1"} for row in rows_after_first)
    _statistics_row(rows_after_first)

    second = client.post("/api/datasets", json=body)
    assert second.status_code == 200, second.text

    rows_after_second = _version_artifact_rows(
        version_store, subject_id=spec_id, content_version=content_version
    )
    assert len(rows_after_second) == 3
    _frame1_row(rows_after_second)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(FRAME1_STORE_FAILED_MARKER in r.getMessage() for r in errors)


class _BarrierSynchronizedArtifactStore(FakeArtifactStore):
    """Blocks the frame1 write and the requested rendering's write at a two-party barrier to
    prove overlap. The statistics write bypasses the barrier.
    """

    def __init__(self, *, timeout: float = 5.0) -> None:
        super().__init__()
        self._barrier = threading.Barrier(2, timeout=timeout)

    def put(self, data: bytes, *, tier: Tier):
        if not _is_statistics_payload(data):
            self._barrier.wait()
        return super().put(data, tier=tier)


def test_the_frame1_and_requested_rendering_writes_overlap(
    version_store: FakeVersionRecordDatasetStore,
):
    artifact_store = _BarrierSynchronizedArtifactStore()
    writer = LiveVersionRecordWriter(store=version_store)
    client = _client(artifact_store=artifact_store, version_writer=writer)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text


class _RenderFailsWhileFrame1StoresArtifactStore(FakeArtifactStore):
    """Fails the requested rendering's write, then lets the frame1 write finish and records its
    completion. The statistics write passes through.

    Frame1 waits for the failure first, so `frame1_stored` proves the door joined that write.
    """

    def __init__(self, *, error_factory: Callable[[bytes], BaseException]) -> None:
        super().__init__()
        self.render_failed = threading.Event()
        self.frame1_stored = threading.Event()
        self._error_factory = error_factory

    def put(self, data: bytes, *, tier: Tier):
        if _is_frame1_payload(data):
            assert self.render_failed.wait(timeout=5.0), "the rendering's write never failed"
            stored = super().put(data, tier=tier)
            self.frame1_stored.set()
            return stored
        if _is_statistics_payload(data):
            return super().put(data, tier=tier)
        try:
            raise self._error_factory(data)
        finally:
            self.render_failed.set()


def test_a_requested_rendering_store_failure_still_joins_the_frame1_write():
    artifact_store = _RenderFailsWhileFrame1StoresArtifactStore(error_factory=_write_error)
    client = _client(artifact_store=artifact_store)

    resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    # The frame1 write completes before the rendering's failure reaches the caller.
    assert artifact_store.frame1_stored.is_set()


class _AllPutsFailArtifactStore(FakeArtifactStore):
    """`put()` raises `error_factory(data)` for every call, whether it writes the statistics, the
    requested rendering or frame1. This models a correlated store outage.
    """

    def __init__(self, *, error_factory: Callable[[bytes], BaseException]) -> None:
        super().__init__()
        self._error_factory = error_factory

    def put(self, data: bytes, *, tier: Tier):
        raise self._error_factory(data)


def test_a_correlated_store_outage_still_logs_the_frame1_failure_marker_once(
    caplog: pytest.LogCaptureFixture,
):
    """The frame1 failure logs its marker once. The response reports the statistics write's
    failure, the first write the mint makes outside frame1.
    """
    artifact_store = _AllPutsFailArtifactStore(error_factory=_write_error)
    client = _client(artifact_store=artifact_store)

    with caplog.at_level(logging.ERROR, logger="datasets_api.serve"):
        resp = client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 503, resp.text
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    marker_records = [r for r in errors if FRAME1_STORE_FAILED_MARKER in r.getMessage()]
    assert len(marker_records) == 1


class _RenderFailsWhileFrame1RaisesArtifactStore(FakeArtifactStore):
    """Raises separate exceptions for the frame1 and requested-rendering payloads. The
    statistics write passes through."""

    def __init__(
        self,
        *,
        frame1_error_factory: Callable[[bytes], BaseException],
        render_error_factory: Callable[[bytes], BaseException],
    ) -> None:
        super().__init__()
        self._frame1_error_factory = frame1_error_factory
        self._render_error_factory = render_error_factory

    def put(self, data: bytes, *, tier: Tier):
        if _is_frame1_payload(data):
            raise self._frame1_error_factory(data)
        if _is_requested_render_payload(data):
            raise self._render_error_factory(data)
        return super().put(data, tier=tier)


def test_a_frame1_bug_stays_visible_alongside_a_requested_rendering_store_failure(
    caplog: pytest.LogCaptureFixture,
):
    """A frame1 programming error is logged when the requested rendering's write also fails.
    The response reports the rendering's failure.
    """
    artifact_store = _RenderFailsWhileFrame1RaisesArtifactStore(
        frame1_error_factory=lambda _data: TypeError("not an artifact-store failure"),
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
    """A barrier proves frame1's inventory write overlaps the mint's first other one, the
    statistics write. `pool_size=2` suffices for both transactions. A third connection raises
    `RuntimeError`.
    """
    # Frame1 is unsigned. Signing the statistics and the requested rendering requires IAM
    # credentials.
    # Stub the signer.
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
        revision_reader=SentinelRevisionReader(),
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
        raise AssertionError("frame1 must stream through open_put, not put one bytes object")

    @contextmanager
    def open_put(self, *, tier: Tier):
        sink = _WriteSizeRecordingSink(self.write_sizes)
        yield sink
        sink._set_result(FakeArtifactStore.put(self, sink.getvalue(), tier=tier))


def test_render_and_store_frame1_streams_every_chunk_through_open_put(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(materialise_formats, "_FRAME1_WRITE_BYTES", 512)
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
    whole = canonical_frame_bytes(frame)
    artifact_store = _StreamOnlyArtifactStore()

    record = render_and_store_frame1(
        frame_chunks=chunks,
        subject_id="spec_streamed",
        content_version="cv_streamed",
        artifact_store=artifact_store,
    )

    assert record.artifact_digest == "sha256:" + hashlib.sha256(whole).hexdigest()
    assert record.media_type == FRAME1_MEDIA_TYPE
    assert record.rendering == {"format": "frame1"}
    assert sum(artifact_store.write_sizes) == len(whole)
    assert len(artifact_store.write_sizes) > 5
    assert max(artifact_store.write_sizes) < 1024 < len(whole)
