"""Tests that mint and definition requests pass the head they read before materialising as
the compare-and-set expectation (its `materialisation_id`). A hook races a second writer inside the SDMX
fetch, so the first writer loses the compare-and-set. The composed app and materialiser are
real. Graph, SDMX, artifact, and head stores are fakes.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from artifact_store import LiveArtifactInventory
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import HeadFence, SubjectHead
from dc_sdmx import SdmxClient
from origin_catalogue.context import current_source_scope
from series_coverage.schema import ABSENT_ENTITY, CoverageManifest, FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from ._version_record_store import FakeVersionRecordDatasetStore, materialisation_record
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"
_STAT_VAR = "MalariaGrantDisbursements"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"

_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "observationProperties": ("recipient",),
    "measuredProperty": ("amount",),
}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_ROW = (
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
)
_CSV_BODY = _CSV_HEADER + _CSV_ROW

# The real fetch reports 2021 while the stored head reflects 2020 coverage.
# This forces rematerialisation.
# Role-bearing facet probes read coverage. This body is used for the fetch.
_CSV_BODY_CHANGED = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_RECIPIENT},USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2021,1000,,facet1\r\n"
)

_SCOPE = {
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"ref": "OECD_DAC2A"},
        "measurementMethod": {"ref": "GrantEquivalent"},
        "unit": {"ref": "USD"},
    },
}

_MINT_BODY = {
    "schema_version": 1,
    "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
    "options": {"bundle": False},
}

_PAST_MAX_AGE = timedelta(hours=25)  # past the 86400s default maximum age
_PAST_RECHECK_INTERVAL = timedelta(minutes=20)  # past the 900s default recheck interval


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


class _SdmxProbeHook:
    """Mutable SDMX response handler with a one-shot hook before each selected response.
    Role-bearing facet probes read coverage, so only real observation fetches reach this handler.
    """

    def __init__(self, body: str = _CSV_BODY) -> None:
        self.body = body
        self.on_probe: Callable[[], None] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.on_probe is not None:
            hook, self.on_probe = self.on_probe, None
            hook()
        return httpx.Response(
            200,
            text=self.body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={_STAT_VAR: _ARCS},
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def _coverage_index(*, year: int = 2020):
    """One-facet coverage matching `_CSV_ROW`.
    Setting `year=2021` moves the series to 2021.
    """
    return build_coverage_index(
        [(_STAT_VAR, "facet1", ABSENT_ENTITY, _RECIPIENT, [year])],
        role_names={_STAT_VAR: RoleNames(donor=None, recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )


class _MutableCoverageService:
    """Coverage double whose index can change between requests."""

    def __init__(self, index, *, stamp: str = "test-coverage-stamp") -> None:
        self._index = index
        self.stamp = stamp

    def current(self):
        return self._index

    def replace(self, index) -> None:
        self._index = index

    def serves(self, snapshot: CoverageSnapshot) -> bool:
        return snapshot.index is self._index

    def register_load_guard(self, guard: object) -> Callable[[], None]:
        """This fake never loads a build, so no guard is ever entered."""
        return lambda: None

    def snapshot(self) -> CoverageSnapshot:
        return CoverageSnapshot(
            index=self._index,
            manifest=CoverageManifest(
                stamp=self.stamp,
                build_id=f"{self.stamp}-build",
                built_at="t",
                source="fake",
                row_counts={},
                files={},
                horizon=self._index.horizon,
            ),
            stamp=self.stamp,
        )


@dataclass(frozen=True)
class _RecordedAdvanceCall:
    subject_id: str
    content_version: str
    expected_materialisation_id: int | None
    applied: bool
    fence: HeadFence | None


@dataclass
class _FakeHeadStore:
    """In-memory head store with compare-and-set writes and ordered call recording.
    `age_head` changes stored timestamps without waiting for real time.
    """

    _heads: dict[str, SubjectHead] = field(default_factory=dict)
    advance_head_calls: list[_RecordedAdvanceCall] = field(default_factory=list)
    _runs: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def read_head(self, subject_id: str) -> SubjectHead | None:
        return self._heads.get(subject_id)

    def advance_head(
        self,
        *,
        materialisation,
        expected_materialisation_id: int | None,
        fence=None,
    ) -> SubjectHead | None:
        subject_id = materialisation.subject_id
        content_version = materialisation.content_version
        existing = self._heads.get(subject_id)
        stored_run = existing.materialisation_id if existing is not None else None
        applied = stored_run == expected_materialisation_id
        self.advance_head_calls.append(
            _RecordedAdvanceCall(
                subject_id=subject_id,
                content_version=content_version,
                expected_materialisation_id=expected_materialisation_id,
                applied=applied,
                fence=fence,
            )
        )
        if not applied:
            return None
        now = datetime.now(UTC)
        written = SubjectHead(
            subject_id=subject_id,
            content_version=content_version,
            confirmed_at=now,
            materialised_at=now,
            materialisation_id=next(self._runs),
            dependencies=materialisation.dependencies,
        )
        self._heads[subject_id] = written
        return written

    def confirm_head(self, *, subject_id: str, content_version: str) -> SubjectHead | None:
        existing = self._heads.get(subject_id)
        if existing is None or existing.content_version != content_version:
            return None
        written = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=datetime.now(UTC),
            materialised_at=existing.materialised_at,
            materialisation_id=existing.materialisation_id,
            dependencies=existing.dependencies,
        )
        self._heads[subject_id] = written
        return written

    def age_head(self, subject_id: str, *, age: timedelta) -> None:
        existing = self._heads[subject_id]
        self._heads[subject_id] = SubjectHead(
            subject_id=existing.subject_id,
            content_version=existing.content_version,
            confirmed_at=datetime.now(UTC) - age,
            materialised_at=datetime.now(UTC) - age,
            materialisation_id=existing.materialisation_id,
            dependencies=existing.dependencies,
        )


def _client(
    *,
    graph,
    sdmx_transport: httpx.MockTransport | None = None,
    head_writer: _FakeHeadStore | None = None,
    store: FakeDatasetStore | None = None,
    coverage: _MutableCoverageService | None = None,
):
    store = store if store is not None else FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            transport=sdmx_transport if sdmx_transport is not None else _sdmx_transport(),
        ),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        head_writer=head_writer,
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage if coverage is not None else _MutableCoverageService(_coverage_index()),
    )


def test_definition_get_expects_the_head_it_read_before_its_own_fetch() -> None:
    """A second writer advances the head inside the fetch for this GET. The GET must retain its earlier
    observed head, lose the compare-and-set, and leave the second writer's head intact.
    """
    hook = _SdmxProbeHook()
    head_store = _FakeHeadStore()
    coverage = _MutableCoverageService(_coverage_index())
    client = _client(
        graph=_graph(),
        sdmx_transport=httpx.MockTransport(hook),
        head_writer=head_store,
        coverage=coverage,
    )

    minted = client.post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    initial = head_store.read_head(spec_id)

    # Age past the maximum age, which forces rematerialisation, and change the data.
    head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    hook.body = _CSV_BODY_CHANGED
    coverage.replace(_coverage_index(year=2021))

    def _second_writer() -> None:
        current = head_store.read_head(spec_id)
        head_store.advance_head(
            materialisation=materialisation_record(
                subject_id=spec_id, content_version="racer-version"
            ),
            expected_materialisation_id=current.materialisation_id if current is not None else None,
        )

    hook.on_probe = _second_writer

    got = client.get(f"/api/datasets/{spec_id}?bundle=false")

    assert got.status_code == 200, got.text
    assert got.json()["trace"]["served_from"] == "materialise"
    lost_calls = [
        call
        for call in head_store.advance_head_calls
        if call.expected_materialisation_id == initial.materialisation_id
    ]
    assert lost_calls, head_store.advance_head_calls
    assert lost_calls[-1].applied is False
    assert head_store.read_head(spec_id).content_version == "racer-version"
    # A request that is not an operation run fences nothing.
    assert all(call.fence is None for call in head_store.advance_head_calls)


def test_mint_reprime_expects_the_head_it_read_before_its_own_fetch() -> None:
    """A remint must carry the head version it read through the SDMX fetch.
    A racing writer advances the head first, so the remint loses its compare-and-set.
    """
    hook = _SdmxProbeHook()
    head_store = _FakeHeadStore()
    coverage = _MutableCoverageService(_coverage_index())
    client = _client(
        graph=_graph(),
        sdmx_transport=httpx.MockTransport(hook),
        head_writer=head_store,
        coverage=coverage,
    )

    minted = client.post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    initial = head_store.read_head(spec_id)
    # A current head answers a remint with no fetch, so age it past the maximum age to rebuild.
    head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    hook.body = _CSV_BODY_CHANGED
    coverage.replace(_coverage_index(year=2021))

    def _second_writer() -> None:
        current = head_store.read_head(spec_id)
        head_store.advance_head(
            materialisation=materialisation_record(
                subject_id=spec_id, content_version="racer-version"
            ),
            expected_materialisation_id=current.materialisation_id if current is not None else None,
        )

    hook.on_probe = _second_writer

    remint = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id},
            "options": {"bundle": False},
        },
    )

    assert remint.status_code == 200, remint.text
    lost_calls = [
        call
        for call in head_store.advance_head_calls
        if call.expected_materialisation_id == initial.materialisation_id
    ]
    assert lost_calls, head_store.advance_head_calls
    assert lost_calls[-1].applied is False
    assert head_store.read_head(spec_id).content_version == "racer-version"
    # A request that is not an operation run fences nothing.
    assert all(call.fence is None for call in head_store.advance_head_calls)


def test_two_concurrent_rebuilds_from_one_head_read_advance_it_once() -> None:
    """Two processes rebuild one subject from the same stale head read. Both fetch, and the
    head compare-and-set applies one advance and refuses the other, which returns `None`."""
    head_store = _LockedHeadStore()
    store = FakeDatasetStore()
    coverage = _MutableCoverageService(_coverage_index())
    both_fetching = threading.Barrier(2, timeout=10)
    # Armed after the mint: the first fetch of each rebuild waits for the other's.
    fetches: list[int] | None = None
    count_lock = threading.Lock()

    def _respond(request: httpx.Request) -> httpx.Response:
        with count_lock:
            waits = fetches is not None and len(fetches) < 2
            if fetches is not None:
                fetches.append(1)
        if waits:
            both_fetching.wait()
        return httpx.Response(
            200,
            text=_CSV_BODY if fetches is None else _CSV_BODY_CHANGED,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    processes = [
        _client(
            graph=_graph(),
            sdmx_transport=httpx.MockTransport(_respond),
            head_writer=head_store,
            store=store,
            coverage=coverage,
        )
        for _ in range(2)
    ]
    minted = processes[0].post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    initial = head_store.read_head(spec_id)
    head_store.age_head(spec_id, age=_PAST_MAX_AGE)
    coverage.replace(_coverage_index(year=2021))
    fetches = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(lambda client: client.get(f"/api/datasets/{spec_id}?bundle=false"), processes)
        )

    assert [response.status_code for response in responses] == [200, 200]
    rebuilds = [
        call
        for call in head_store.advance_head_calls
        if call.expected_materialisation_id == initial.materialisation_id
    ]
    assert sorted(call.applied for call in rebuilds) == [False, True]
    assert head_store.read_head(spec_id).content_version != initial.content_version


@dataclass
class _LockedHeadStore(_FakeHeadStore):
    """`_FakeHeadStore` whose compare-and-set runs under one lock, as the database row's does."""

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def advance_head(self, **kwargs) -> SubjectHead | None:
        with self._lock:
            return super().advance_head(**kwargs)


def test_a_pinned_definition_materialises_and_refreshes_inside_its_provenance_scope() -> None:
    """The mint's scope pins `facets.provenance`, so the materialisation's SDMX fetch and,
    past the recheck window, the head's fingerprint read both run with the catalogue scoped to
    that provenance (`adapter.pinned_provenance_scope`)."""
    fetch_scopes: list[frozenset[str] | None] = []
    refresh_scopes: list[frozenset[str] | None] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        fetch_scopes.append(current_source_scope())
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    class _ScopeRecordingGraph(FakeMintGraphReader):
        def node_arcs_many(self, *args, **kwargs):
            refresh_scopes.append(current_source_scope())
            return super().node_arcs_many(*args, **kwargs)

    head_store = _FakeHeadStore()
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_ScopeRecordingGraph(
            arcs={_STAT_VAR: _ARCS},
            labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
        ),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(_respond)),
        artifact_store=_InventoryRecordingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        head_writer=head_store,
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_MutableCoverageService(_coverage_index()),
    )
    minted = client.post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    head_store.age_head(spec_id, age=_PAST_RECHECK_INTERVAL)
    refresh_scopes.clear()

    got = client.get(f"/api/datasets/{spec_id}?bundle=false")

    assert got.status_code == 200, got.text
    assert got.json()["trace"]["served_from"] == "head"
    pinned = frozenset({"OECD_DAC2A"})
    assert fetch_scopes and set(fetch_scopes) == {pinned}
    assert refresh_scopes and set(refresh_scopes) == {pinned}
