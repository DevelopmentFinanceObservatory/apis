"""Four concurrent mint requests share a real size-2 `ConnectionPool` beneath the head store,
the snapshot reader and the version writer, whose one write transaction per mint records the
inventory rows `LiveArtifactStore` defers. The test proves the pool caps concurrent borrows
during overlapping writes and all three artifact records per mint commit.

SQLite shared-cache tables lock overlapping writes, so `_SQL_LOCK` serializes
SQL statements while `_SERVICE_TIME_SECONDS` simulates overlapping borrows.
`_BorrowTracker` measures those borrows before SQL locking. Requests share one
`TestClient` portal to avoid FastAPI's unsynchronised first-use route-cache
rebuild; `_LoopThreadRecorder` checks that they use one event loop.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence

import httpx
import pytest
from fastapi.testclient import TestClient

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store import ConnectionPool, LiveDatasetStore
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.operation_store import NullOperationStore
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.artifact_store.conftest import FakeGCSClient
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_app
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import (
    _VERSION_RECORD_SCHEMA,
    _decode_row,
    _sqlite_param,
    _to_sqlite_paramstyle,
)
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"

# Match the pool default of 2.
_POOL_SIZE = 2
# Four matches the default mint concurrency and oversubscribes the pool. `make_client` widens
# the materialise limit (default 2) so all four mints reach the SDMX barrier together.
_CONCURRENT_MINTS = 4
# Forces queuing while keeping the serial case under a second.
_SERVICE_TIME_SECONDS = 0.05

_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_SCOPE = {
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}


class _LoopThreadRecorder:
    """Records the serving thread for each HTTP request."""

    def __init__(self, app, *, threads: set[int]) -> None:
        self._app = app
        self._threads = threads

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            self._threads.add(threading.get_ident())
        await self._app(scope, receive, send)


class _BorrowTracker:
    """Tracks simultaneous busy connections and their peak count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self.max_concurrent = 0

    def enter(self) -> None:
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)

    def exit(self) -> None:
        with self._lock:
            self._current -= 1


class _TrackingCursor:
    """Runs each SQL statement through the connection's borrow tracker."""

    def __init__(self, cursor: sqlite3.Cursor, *, connection: _TrackingConnection) -> None:
        self._cursor = cursor
        self._connection = connection

    @property
    def description(self):
        return self._cursor.description

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._connection._before_statement()
        if sql.strip().upper().startswith("SELECT PG_ADVISORY_XACT_LOCK"):
            # `_sql_lock` already serialises every transaction.
            return
        translated_params = tuple(
            p.value if isinstance(p, Tier) else _sqlite_param(p) for p in params
        )
        self._cursor.execute(_to_sqlite_paramstyle(sql), translated_params)

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None

    def fetchall(self):
        return [tuple(_decode_row(row).values()) for row in self._cursor.fetchall()]

    def close(self) -> None:
        self._cursor.close()


class _TrackingConnection:
    """Wraps a SQLite connection as a `DBAPIConnection` for the pool test.

    The simulated service delay runs before `_sql_lock`, so borrows overlap;
    the lock serializes SQLite writes until commit or rollback.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        tracker: _BorrowTracker,
        service_time: float,
        sql_lock: threading.Lock,
    ) -> None:
        self._conn = conn
        self._tracker = tracker
        self._service_time = service_time
        self._sql_lock = sql_lock
        self._busy = False
        self._holding_sql_lock = False

    def cursor(self) -> _TrackingCursor:
        return _TrackingCursor(self._conn.cursor(), connection=self)

    def _before_statement(self) -> None:
        if not self._busy:
            self._busy = True
            self._tracker.enter()
            time.sleep(self._service_time)
        if not self._holding_sql_lock:
            self._sql_lock.acquire()
            self._holding_sql_lock = True

    def commit(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._release_sql_lock()
        self._release_busy()

    def rollback(self) -> None:
        try:
            self._conn.rollback()
        finally:
            self._release_sql_lock()
        self._release_busy()

    def close(self) -> None:
        self._conn.close()

    def _release_sql_lock(self) -> None:
        if self._holding_sql_lock:
            self._holding_sql_lock = False
            self._sql_lock.release()

    def _release_busy(self) -> None:
        if self._busy:
            self._busy = False
            self._tracker.exit()


def _setup_shared_schema(schema_name: str) -> sqlite3.Connection:
    """Creates shared-cache tables and keeps the in-memory database alive."""
    keepalive = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
    keepalive.execute(f"ATTACH DATABASE 'file:{schema_name}?mode=memory&cache=shared' AS datasets")
    keepalive.executescript(_VERSION_RECORD_SCHEMA)
    keepalive.commit()
    return keepalive


def _connect_factory(
    *, tracker: _BorrowTracker, service_time: float, schema_name: str, sql_lock: threading.Lock
):
    def _connect() -> _TrackingConnection:
        conn = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(f"ATTACH DATABASE 'file:{schema_name}?mode=memory&cache=shared' AS datasets")
        return _TrackingConnection(
            conn, tracker=tracker, service_time=service_time, sql_lock=sql_lock
        )

    return _connect


def _stat_var(i: int) -> str:
    return f"MintPoolContentionStatVar{i}"


def _csv_body(stat_var: str) -> str:
    return (
        _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{_RECIPIENT},USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    )


def _coverage_service(stat_vars: list[str]) -> FakeCoverageService:
    """Builds one 2020 coverage facet per StatVar for `_RECIPIENT`."""
    index = build_coverage_index(
        [(stat_var, "facet1", ABSENT_ENTITY, _RECIPIENT, [2020]) for stat_var in stat_vars],
        role_names={
            stat_var: RoleNames(donor=None, recipient="recipient") for stat_var in stat_vars
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _graph_for(stat_vars: list[str]) -> FakeMintGraphReader:
    arcs = {
        stat_var: {
            "typeOf": ("StatisticalVariable",),
            "observationProperties": ("recipient",),
            "measuredProperty": ("amount",),
        }
        for stat_var in stat_vars
    }
    facets = {(stat_var, _RECIPIENT): [_FACET] for stat_var in stat_vars}
    return FakeMintGraphReader(
        arcs=arcs, facets=facets, labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"}
    )


def _sdmx_transport(
    bodies_by_stat_var: dict[str, str], *, barrier: threading.Barrier
) -> httpx.MockTransport:
    """Releases SDMX reads together so all mints reach artifact writes concurrently."""

    def handler(request: httpx.Request) -> httpx.Response:
        stat_var = str(request.url.params.get("c[variableMeasured]"))
        barrier.wait(timeout=5.0)
        return httpx.Response(
            200,
            text=bodies_by_stat_var[stat_var],
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _mint_body(*, stat_var: str) -> dict:
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [f"dcid:{stat_var}"], "scope": _SCOPE},
        "options": {"bundle": False},
    }


def test_concurrent_mints_never_oversubscribe_the_shared_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four concurrent mints reach a real size-2 pool beneath their store work.

    Each mint commits three inventory entries in its version write. The test verifies the
    peak borrow count reaches 2 without exceeding it. URL signing is faked; the store writes
    and mint call chain remain real.
    """
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(
        "artifact_store.store.sign_get_url",
        lambda **kwargs: f"https://storage.googleapis.com/signed/{kwargs['object_name']}",
    )
    stat_vars = [_stat_var(i) for i in range(_CONCURRENT_MINTS)]
    bodies = {stat_var: _csv_body(stat_var) for stat_var in stat_vars}
    barrier = threading.Barrier(_CONCURRENT_MINTS, timeout=5.0)
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(bodies, barrier=barrier))
    graph = _graph_for(stat_vars)

    schema_name = f"mint_pool_contention_{uuid.uuid4().hex}"
    keepalive = _setup_shared_schema(schema_name)
    tracker = _BorrowTracker()
    sql_lock = threading.Lock()
    pool = ConnectionPool(
        connect=_connect_factory(
            tracker=tracker,
            service_time=_SERVICE_TIME_SECONDS,
            schema_name=schema_name,
            sql_lock=sql_lock,
        ),
        size=_POOL_SIZE,
    )
    try:
        store = LiveDatasetStore(connections=pool)
        inventory = LiveArtifactInventory(store=store)
        artifact_store = LiveArtifactStore(
            buckets={
                Tier.working: "contention-working-bucket",
                Tier.pinned: "contention-pinned-bucket",
            },
            inventory=inventory,
            client=FakeGCSClient(),
        )
        spec_store = FakeDatasetStore()
        app = make_app(
            registry=LiveSpecRegistry(store=spec_store),
            spec_reader=LiveSpecReader(store=spec_store),
            graph=graph,
            sdmx=sdmx,
            artifact_store=artifact_store,
            work_max_concurrency=_CONCURRENT_MINTS,
            materialise_max_concurrency=_CONCURRENT_MINTS,
            snapshot_reader=LiveSnapshotReader(store=store),
            version_writer=LiveVersionRecordWriter(store=store),
            head_writer=LiveHeadStore(store=store),
            reference_writer=FakeArtifactReferenceWriter(),
            coverage=_coverage_service(stat_vars),
            operation_store=NullOperationStore(),
        )
        loop_threads: set[int] = set()
        app.add_middleware(_LoopThreadRecorder, threads=loop_threads)
        client = TestClient(app, raise_server_exceptions=False)

        responses: list[httpx.Response | None] = [None] * _CONCURRENT_MINTS

        def _mint(i: int) -> None:
            responses[i] = client.post("/api/datasets", json=_mint_body(stat_var=stat_vars[i]))

        with client:
            threads = [threading.Thread(target=_mint, args=(i,)) for i in range(_CONCURRENT_MINTS)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10.0)

        # Read the shared-cache tables before closing their keepalive connection.
        inventory_digests = {
            row[0]
            for row in keepalive.execute("SELECT artifact_digest FROM datasets.artifact_inventory")
        }
        location_digests = {
            row[0]
            for row in keepalive.execute("SELECT artifact_digest FROM datasets.artifact_location")
        }
    finally:
        pool.close_all()
        keepalive.close()

    assert len(loop_threads) == 1

    requested_digests: set[str] = set()
    for i, resp in enumerate(responses):
        assert resp is not None, f"mint {i} never returned"
        assert resp.status_code == 200, resp.text
        requested_digests.add(resp.json()["artifacts"][0]["artifact_digest"])

    assert tracker.max_concurrent == _POOL_SIZE

    # Each mint records the requested, statistics, and canonical frame artifacts.
    assert len(inventory_digests) == 3 * _CONCURRENT_MINTS
    assert location_digests == inventory_digests
    assert requested_digests <= inventory_digests
