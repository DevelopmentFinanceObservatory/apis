"""Tests that mint and definition requests pass the head version they read before probing as
`expected_content_version`. A hook races a second writer inside the SDMX fetch, so the first
writer loses the compare-and-set. Revision snapshots must be read before coverage probes.
The composed app and materialiser are real. Graph, SDMX, artifact, and head stores are fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from datasets_api.adapter import prime_head_signal
from datasets_api.doors import definition as definition_door
from datasets_api.revision import SentinelRevisionReader
from datasets_api.versions import SubjectHead
from dc_sdmx import SdmxClient
from materialise import MaterialisationRequest
from materialise.revision import RevisionReader, RevisionSnapshot, SourceRevision
from materialise.window import RequestedWindow
from series_coverage.schema import CoverageManifest, FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

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
    Setting `year=2021` changes the signal from 2020.
    """
    return build_coverage_index(
        [(_STAT_VAR, "facet1", "country/USA", _RECIPIENT, [year])],
        role_names={_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
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
    expected_content_version: str | None
    applied: bool


@dataclass
class _FakeHeadStore:
    """In-memory head store with compare-and-set writes and ordered call recording.
    `age_head` changes stored timestamps without waiting for real time.
    """

    _heads: dict[str, SubjectHead] = field(default_factory=dict)
    advance_head_calls: list[_RecordedAdvanceCall] = field(default_factory=list)

    def read_head(self, subject_id: str) -> SubjectHead | None:
        return self._heads.get(subject_id)

    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal,
        expected_content_version: str | None,
    ) -> SubjectHead | None:
        existing = self._heads.get(subject_id)
        stored_content_version = existing.content_version if existing is not None else None
        applied = stored_content_version == expected_content_version
        self.advance_head_calls.append(
            _RecordedAdvanceCall(
                subject_id=subject_id,
                content_version=content_version,
                expected_content_version=expected_content_version,
                applied=applied,
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
            signal=tuple(signal),
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
            signal=existing.signal,
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
            signal=existing.signal,
        )


def _client(
    *,
    graph,
    sdmx_transport: httpx.MockTransport | None = None,
    head_writer: _FakeHeadStore | None = None,
    revision_reader: RevisionReader | None = None,
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
        revision_reader=revision_reader
        if revision_reader is not None
        else SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        head_writer=head_writer,
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage if coverage is not None else _MutableCoverageService(_coverage_index()),
    )


def test_definition_get_expects_the_head_it_read_before_its_own_probe() -> None:
    """A second writer advances the head inside the fetch for this GET. The GET must retain its earlier
    `expected_content_version`, lose the compare-and-set, and leave the second writer's head intact.
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
    initial_version = head_store.read_head(spec_id).content_version

    # Age past the recheck interval and change coverage before the race.
    # Both changes force rematerialisation.
    head_store.age_head(spec_id, age=_PAST_RECHECK_INTERVAL)
    hook.body = _CSV_BODY_CHANGED
    coverage.replace(_coverage_index(year=2021))

    def _second_writer() -> None:
        current = head_store.read_head(spec_id)
        head_store.advance_head(
            subject_id=spec_id,
            content_version="racer-version",
            signal=(),
            expected_content_version=current.content_version if current is not None else None,
        )

    hook.on_probe = _second_writer

    got = client.get(f"/api/datasets/{spec_id}?bundle=false")

    assert got.status_code == 200, got.text
    assert got.json()["trace"]["served_from"] == "materialise"
    lost_calls = [
        call
        for call in head_store.advance_head_calls
        if call.expected_content_version == initial_version
    ]
    assert lost_calls, head_store.advance_head_calls
    assert lost_calls[-1].applied is False
    assert head_store.read_head(spec_id).content_version == "racer-version"


def test_mint_reprime_expects_the_head_it_read_before_its_own_probe() -> None:
    """A remint must carry the head version read before its coverage probe through the SDMX fetch.
    A racing writer advances the head first, so the remint loses its compare-and-set.
    """
    hook = _SdmxProbeHook()
    head_store = _FakeHeadStore()
    client = _client(
        graph=_graph(), sdmx_transport=httpx.MockTransport(hook), head_writer=head_store
    )

    minted = client.post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    initial_version = head_store.read_head(spec_id).content_version

    def _second_writer() -> None:
        current = head_store.read_head(spec_id)
        head_store.advance_head(
            subject_id=spec_id,
            content_version="racer-version",
            signal=(),
            expected_content_version=current.content_version if current is not None else None,
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
        if call.expected_content_version == initial_version
    ]
    assert lost_calls, head_store.advance_head_calls
    assert lost_calls[-1].applied is False
    assert head_store.read_head(spec_id).content_version == "racer-version"


def test_prime_head_signal_reads_the_revision_before_its_own_probe() -> None:
    """`prime_head_signal` stamps the revision snapshot read before its first coverage lookup."""
    order: list[str] = []

    class _OrderRecordingRevisionReader:
        def __init__(self, snapshot: RevisionSnapshot) -> None:
            self._snapshot = snapshot

        def read(self) -> RevisionSnapshot:
            order.append("revision")
            return self._snapshot

    class _OrderMarkingCoverage:
        """Records the first coverage lookup, then delegates to the real index."""

        def __init__(self, index) -> None:
            self._index = index

        @property
        def role_names(self):
            return self._index.role_names

        @property
        def facet_metadata(self):
            return self._index.facet_metadata

        def filter_candidates(self, **kwargs):
            order.append("probe")
            return self._index.filter_candidates(**kwargs)

        def series_counts(self, **kwargs):
            return self._index.series_counts(**kwargs)

    snapshot = RevisionSnapshot(
        graph_revision="rev-1", source_revisions=(SourceRevision(source="oecd", revision="42"),)
    )
    request = MaterialisationRequest(
        stat_var=_STAT_VAR_REF,
        role_bindings={"recipient": [_RECIPIENT]},
        declared_roles=("recipient",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
    )

    priming = prime_head_signal(
        (request,),
        subject_id="spec_test",
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(_SdmxProbeHook())),
        head_store=_FakeHeadStore(),
        materialise_wait_seconds=30,
        revision_reader=_OrderRecordingRevisionReader(snapshot),
        coverage=_OrderMarkingCoverage(_coverage_index()),
    )

    assert priming.probe.revision == snapshot
    assert order == ["revision", "probe"]


def test_resolve_currency_probe_carries_the_revision_readers_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_resolve_currency` passes its revision reader to the facet probe before rematerialisation."""
    recorded: list[RevisionReader | None] = []
    real_probe = definition_door.probe_collection_facets

    def _recording_probe(requests, *, graph, revision_reader=None, **kwargs):
        recorded.append(revision_reader)
        return real_probe(requests, graph=graph, revision_reader=revision_reader, **kwargs)

    monkeypatch.setattr(definition_door, "probe_collection_facets", _recording_probe)

    revision_reader = SentinelRevisionReader()
    client = _client(graph=_graph(), revision_reader=revision_reader)

    minted = client.post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]

    # The default null head writer returns no head, so each GET reaches the probe.
    got = client.get(f"/api/datasets/{spec_id}?bundle=false")

    assert got.status_code == 200, got.text
    assert recorded and recorded[-1] is revision_reader
