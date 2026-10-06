"""Checks `LiveFrameStore` with in-memory snapshot, artifact, graph, and reference fakes."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from artifact_store.errors import (
    ArtifactIntegrityError,
    ArtifactInventoryError,
    ArtifactNotFoundError,
    ArtifactWriteError,
)
from artifact_store.store import ArtifactSink, StoredArtifact, Tier
from assembler.ports import FrameRef, FrameUnavailable
from assembler.testing import (
    build_frame_doc,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_rows,
)
from code_index.query import CodeIndex
from code_index.types import CodeRow, IndexStamp
from dataset_store.snapshot import (
    Revocation,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
)
from datasets_api.codes_snapshot import ArtifactNotFound
from datasets_api.reference_store import ReferenceWrite
from materialise import provenance_doc
from materialise.graph import GraphReadError
from materialise.provenance_doc import SourceDocumentNotFound
from presentation_store.frame_store import (
    FRAME1_MEDIA_TYPE,
    PRESENTATION_PIN_REFERENCE_KIND,
    LiveFrameStore,
    frame_reads_bounded_by,
)
from spec_registry.errors import IdentityQuarantined, SpecRegistryUnavailable
from spec_registry.reader import StoredDefinition

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# -- Fakes ------------------------------------------------------------------


@dataclass
class _FakeSnapshotReader:
    versions: dict[tuple[str, str], StoredVersion] = field(default_factory=dict)
    locations: dict[tuple[str, str], list[StoredArtifactLocation]] = field(default_factory=dict)
    revocations: dict[tuple[str, str], Revocation] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        self.calls.append("read_version")
        return self.versions.get((subject_id, content_version))

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        self.calls.append("read_artifact_locations")
        return list(self.locations.get((subject_id, content_version), []))

    def read_revocation(self, subject_id: str, content_version: str) -> Revocation | None:
        self.calls.append("read_revocation")
        return self.revocations.get((subject_id, content_version))


class _BufferedSink(io.BytesIO):
    """An in-memory sink that stores the result when `open_put` commits."""

    def __init__(self) -> None:
        super().__init__()
        self._result: StoredArtifact | None = None

    def _set_result(self, result: StoredArtifact) -> None:
        self._result = result

    def result(self) -> StoredArtifact:
        if self._result is None:
            raise RuntimeError("ArtifactSink.result() called before open_put committed")
        return self._result


class _FakeArtifactStore:
    """Implement the artifact operations used by `LiveFrameStore`."""

    def __init__(
        self,
        *,
        corrupt: frozenset[str] = frozenset(),
        copy_raises: Exception | None = None,
    ) -> None:
        self.objects: dict[tuple[Tier, str], bytes] = {}
        self.calls: list[tuple[str, Tier, str]] = []
        self._corrupt = corrupt
        self.copy_raises = copy_raises

    def seed(self, *, tier: Tier, data: bytes) -> str:
        """Test setup helper, not part of the `ArtifactStore` Protocol:
        stores `data` under its own digest, returns the digest."""
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        self.objects[(tier, digest)] = data
        return digest

    def put(self, data: bytes, *, tier: Tier) -> StoredArtifact:
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        key = (tier, digest)
        deduplicated = key in self.objects
        self.objects[key] = data
        return StoredArtifact(
            digest=digest,
            tier=tier,
            bucket="fake-bucket",
            object_name=digest,
            size_bytes=len(data),
            deduplicated=deduplicated,
        )

    @contextmanager
    def open_put(self, *, tier: Tier) -> Generator[ArtifactSink, None, None]:
        sink = _BufferedSink()
        yield sink
        sink._set_result(self.put(sink.getvalue(), tier=tier))

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        self.calls.append(("get", tier, artifact_digest))
        if artifact_digest in self._corrupt:
            raise ArtifactIntegrityError(
                digest=artifact_digest, tier=tier, computed_digest="sha256:corrupted"
            )
        key = (tier, artifact_digest)
        if key not in self.objects:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        return self.objects[key]

    def signed_get_url(
        self,
        digest: str,
        *,
        tier: Tier,
        ttl: timedelta,
        response_type: str,
        response_disposition: str,
    ) -> str:
        raise NotImplementedError("not exercised by LiveFrameStore")

    def copy(self, digest: str, *, from_tier: Tier, to_tier: Tier) -> StoredArtifact:
        self.calls.append(("copy", from_tier, digest))
        if self.copy_raises is not None:
            raise self.copy_raises
        source = (from_tier, digest)
        if source not in self.objects:
            raise ArtifactNotFoundError(digest=digest, tier=from_tier)
        data = self.objects[source]
        dest = (to_tier, digest)
        deduplicated = dest in self.objects
        self.objects[dest] = data
        return StoredArtifact(
            digest=digest,
            tier=to_tier,
            bucket="fake-bucket",
            object_name=digest,
            size_bytes=len(data),
            deduplicated=deduplicated,
        )

    def list(self, *, tier: Tier) -> frozenset[str]:
        return frozenset(digest for (t, digest) in self.objects if t == tier)


class _FakeGraphReader:
    """Record batched node reads and optionally fail the whole batch. A dcid in `arcs`
    answers with those arcs, any other with its `labels` entry as its `name`."""

    def __init__(
        self,
        *,
        labels: dict[str, str] | None = None,
        arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
        raise_for: frozenset[str] = frozenset(),
    ) -> None:
        self._labels = labels or {}
        self._arcs = arcs or {}
        self.raise_for = raise_for
        self.calls = 0
        self.node_arcs_many_calls: list[tuple[str, ...]] = []
        self.node_arcs_many_requests: list[
            tuple[tuple[str, ...], tuple[str, ...], float | None]
        ] = []

    def node_label(self, dcid: str) -> str | None:
        raise NotImplementedError("not exercised by LiveFrameStore")

    def observation_facets_batch(self, *, stat_var: str, entities, deadline: float | None = None):
        raise NotImplementedError("not exercised by LiveFrameStore")

    def declares_facet_defaults(self) -> bool:
        raise NotImplementedError("not exercised by LiveFrameStore")

    def default_facet_policy(self, *, stat_var: str, axis: str):
        raise NotImplementedError("not exercised by LiveFrameStore")

    def node_arcs(self, dcid: str, *, properties):
        raise NotImplementedError("not exercised by LiveFrameStore")

    def node_subjects(self, dcid: str, *, property, filters=None, max_results: int):
        raise NotImplementedError("not exercised by LiveFrameStore")

    def node_arcs_many(
        self,
        dcids,
        *,
        properties,
        deadline: float | None = None,
    ):
        self.calls += 1
        dcids = tuple(dcids)
        self.node_arcs_many_calls.append(dcids)
        self.node_arcs_many_requests.append((dcids, tuple(properties), deadline))
        if any(dcid in self.raise_for for dcid in dcids):
            raise GraphReadError(f"transport error looking up batch {dcids!r}")
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            if dcid in self._arcs:
                result[dcid] = self._arcs[dcid]
                continue
            label = self._labels.get(dcid)
            result[dcid] = {"name": (label,)} if label is not None else None
        return result

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        return {
            dcid: ((label,) if (label := self._labels.get(dcid)) is not None else ())
            for dcid in dict.fromkeys(dcids)
        }


@dataclass
class _FakeReferenceWriter:
    records: dict[tuple[str, str, str, str, str], datetime] = field(default_factory=dict)
    revoked: set[tuple[str, str]] = field(default_factory=set)
    # (subject_id, content_version) pairs whose link a re-render replaced.
    unlinked: set[tuple[str, str]] = field(default_factory=set)
    calls: list[dict[str, object]] = field(default_factory=list)

    def record_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        key = (principal, artifact_digest, subject_id, content_version, reference_kind)
        self.calls.append({"method": "record_reference", "key": key, "now": now})
        if key not in self.records:
            self.records[key] = now
        return self.records[key]

    def record_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        self.calls.append(
            {
                "method": "record_reference_unless_revoked",
                "key": (principal, artifact_digest, subject_id, content_version, reference_kind),
                "now": now,
            }
        )
        if (subject_id, content_version) in self.revoked:
            raise SnapshotRevoked(id=subject_id, content_version=content_version, reason="test")
        if (subject_id, content_version) in self.unlinked:
            raise ArtifactNotFound(
                id=subject_id, content_version=content_version, artifact_digest=artifact_digest
            )
        return self.record_reference(
            principal=principal,
            artifact_digest=artifact_digest,
            subject_id=subject_id,
            content_version=content_version,
            reference_kind=reference_kind,
            now=now,
        )

    def claim_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> ReferenceWrite:
        key = (principal, artifact_digest, subject_id, content_version, reference_kind)
        changed = key not in self.records
        created_at = self.record_reference_unless_revoked(
            principal=principal,
            artifact_digest=artifact_digest,
            subject_id=subject_id,
            content_version=content_version,
            reference_kind=reference_kind,
            now=now,
        )
        return ReferenceWrite(created_at=created_at, changed=changed)

    def remove_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> None:
        key = (principal, artifact_digest, subject_id, content_version, reference_kind)
        self.calls.append({"method": "remove_reference", "key": key, "now": now})
        # A no-op, not an error, when no live row matches -- mirrors the
        # live store's own soft-delete contract (a plain `UPDATE ...
        # WHERE removed_at IS NULL` that matches zero rows commits fine).
        self.records.pop(key, None)


class _LegacyReferenceWriter:
    """Writer with `record_reference_unless_revoked` but no claim method."""

    def __init__(self) -> None:
        self.inner = _FakeReferenceWriter()

    def record_reference_unless_revoked(self, **kwargs) -> datetime:
        return self.inner.record_reference_unless_revoked(**kwargs)

    def remove_reference(self, **kwargs) -> None:
        self.inner.remove_reference(**kwargs)


# -- Fixtures -----------------------------------------------------------


def _frame_bytes() -> bytes:
    doc = build_frame_doc(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    return json.dumps(doc).encode("utf-8")


def _frame_bytes_with_converted_unit() -> bytes:
    """Build frame bytes with a converted unit cell as a JSON object."""
    doc = {
        "frame_version": "frame1",
        "columns": [
            "stat_var",
            "recipient",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
            "status",
        ],
        "rows": [
            [
                "dcid:MalariaGrantDisbursements",
                "country/ETH",
                period,
                value,
                {"currency": "USD", "prices": "constant", "base_year": 2015},
                None,
                "src_oecd_dac",
                None,
            ]
            for period, value in {"2015": 412.3, "2016": 455.1}.items()
        ],
    }
    return json.dumps(doc).encode("utf-8")


def _location(
    *,
    digest: str,
    tier: str,
    media_type: str = FRAME1_MEDIA_TYPE,
    stored_at: datetime = _NOW,
    created_at: datetime = _NOW,
    rendering_key: str = "frame1",
) -> StoredArtifactLocation:
    return StoredArtifactLocation(
        artifact_digest=digest,
        rendering_key=rendering_key,
        rendering={},
        media_type=media_type,
        size_bytes=0,
        created_at=created_at,
        stored_at=stored_at,
        tier=tier,
        bucket="fake-bucket",
        object_name=digest,
    )


def _version(frame_ref: FrameRef) -> StoredVersion:
    return StoredVersion(
        subject_id=frame_ref.spec_id,
        subject_kind="spec",
        content_version=frame_ref.content_version,
        schema_version=1,
        immutable_body={"content_version": frame_ref.content_version, "coverage": {}},
    )


def _collapsed_version(frame_ref: FrameRef, *, base_content_version: str) -> StoredVersion:
    """Build a stored version that names its source content version."""
    return StoredVersion(
        subject_id=frame_ref.spec_id,
        subject_kind="spec",
        content_version=frame_ref.content_version,
        schema_version=1,
        immutable_body={
            "content_version": frame_ref.content_version,
            "coverage": {},
        },
        base_content_version=base_content_version,
    )


def _store(
    *,
    snapshots: _FakeSnapshotReader | None = None,
    artifacts: _FakeArtifactStore | None = None,
    graph=None,
    reference_writer: _FakeReferenceWriter | None = None,
    **kwargs,
) -> LiveFrameStore:
    return LiveFrameStore(
        snapshots=snapshots if snapshots is not None else _FakeSnapshotReader(),
        artifacts=artifacts if artifacts is not None else _FakeArtifactStore(),
        graph=graph,
        reference_writer=reference_writer
        if reference_writer is not None
        else _FakeReferenceWriter(),
        now=lambda: _NOW,
        **kwargs,
    )


def _happy_path_fixture() -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef, str]:
    frame_ref = malaria_ethiopia_frame_ref()
    data = _frame_bytes()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    return snapshots, artifacts, frame_ref, digest


def _published_ops_frame_ref() -> FrameRef:
    """Return the identifier pair used by a published derived frame."""
    return FrameRef(spec_id="ops_" + "c" * 32, content_version="ocv_" + "d" * 32)


def _ops_happy_path_fixture() -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef, str]:
    """Build a working-tier frame fixture with a published `ops_`/`ocv_` pair."""
    frame_ref = _published_ops_frame_ref()
    data = _frame_bytes()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    return snapshots, artifacts, frame_ref, digest


# -- load(): happy path + cache -------------------------------------------


def test_load_resolves_a_frame_with_view_manifest_and_labels():
    snapshots, artifacts, frame_ref, _digest = _happy_path_fixture()
    graph = _FakeGraphReader(
        labels={
            "MalariaGrantDisbursements": "Malaria Grant Disbursements",
            "country/ETH": "Ethiopia",
            "src_oecd_dac": "OECD DAC",
        }
    )
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.frame_ref == frame_ref
    assert resolved.frame["frame_version"] == "frame1"
    assert resolved.structural_view.series_count == 1
    assert resolved.manifest == {"content_version": frame_ref.content_version, "coverage": {}}
    assert resolved.labels["dcid:MalariaGrantDisbursements"] == "Malaria Grant Disbursements"
    assert resolved.labels["country/ETH"] == "Ethiopia"
    assert resolved.labels["src_oecd_dac"] == "OECD DAC"
    assert graph.calls == 1  # one batched call covers every distinct ref
    assert sorted(graph.node_arcs_many_calls[0]) == sorted(
        ["MalariaGrantDisbursements", "country/ETH", "src_oecd_dac"]
    )


def test_load_dates_the_frame_by_its_canonical_artifact_s_first_storage_in_utc():
    """Convert the canonical artifact creation time to a UTC date."""
    frame_ref = malaria_ethiopia_frame_ref()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=_frame_bytes())
    created_at = datetime(2026, 3, 4, 23, 30, tzinfo=timezone(timedelta(hours=-5)))
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value, created_at=created_at)
            ]
        },
    )
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=None)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.retrieved_on == date(2026, 3, 5)


def test_load_resolves_a_frame_whose_unit_column_carries_a_converted_unit():
    """A converted unit dictionary survives frame loading and structural-view construction.

    `compute_structural_view` once raised `TypeError: unhashable type: 'dict'` on such a frame.
    """
    frame_ref = malaria_ethiopia_frame_ref()
    data = _frame_bytes_with_converted_unit()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    store = _store(snapshots=snapshots, artifacts=artifacts)

    resolved = store.load(frame_ref=frame_ref)

    unit = next(d for d in resolved.structural_view.dimensions if d.name == "unit")
    assert unit.values == ("USD/constant/2015",)
    assert unit.plural is False
    assert resolved.structural_view.series_count == 1


def test_load_cache_hit_skips_the_byte_fetch_but_still_rechecks_state():
    """A cache hit skips immutable bytes but rechecks revocation and tier state."""
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts)

    first = store.load(frame_ref=frame_ref)
    assert snapshots.calls  # something ran on the first, real resolution
    assert artifacts.calls
    snapshot_calls_after_first = len(snapshots.calls)
    artifact_calls_after_first = len(artifacts.calls)

    second = store.load(frame_ref=frame_ref)

    assert second is first
    assert len(snapshots.calls) == 2 * snapshot_calls_after_first
    assert len(artifacts.calls) == artifact_calls_after_first


def test_load_cache_hit_refuses_once_the_frame_is_revoked_after_the_first_load():
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts)

    first = store.load(frame_ref=frame_ref)
    assert first is not None

    snapshots.revocations[(frame_ref.spec_id, frame_ref.content_version)] = Revocation(
        reason="takedown", revoked_at=_NOW
    )

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)
    assert excinfo.value.reason == "revoked"


def test_load_with_no_graph_returns_no_labels():
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=None)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {}


type _LabelFailure = tuple[LiveFrameStore, FrameRef, Callable[[], None], dict[str, str]]


def _graph_label_failure() -> _LabelFailure:
    """A store whose graph label batch fails until `recover` runs."""
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    graph = _FakeGraphReader(
        labels={"country/ETH": "Ethiopia"},
        raise_for=frozenset({"MalariaGrantDisbursements"}),
    )
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    def recover() -> None:
        graph.raise_for = frozenset()

    return store, frame_ref, recover, {"country/ETH": "Ethiopia"}


def _family_label_failure() -> _LabelFailure:
    """A store whose spec registry is unavailable until `recover` runs."""
    family_spec_id = "spec_" + "c" * 32
    frame_ref = _family_ref_frame_ref()
    data = json.dumps(
        build_frame_doc(role_keys=(), rows=_family_ref_rows(family_spec_id=family_spec_id))
    ).encode()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    spec_reader = _FakeSpecReader(
        definitions={
            family_spec_id: _stored_definition(
                spec_id=family_spec_id,
                slots=[
                    {
                        "property": "crsPurpose",
                        "binding": {"state": "value", "ref": "CRSPurposeCode11220"},
                    }
                ],
            )
        },
        unavailable=True,
    )
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        spec_reader=spec_reader,
        code_index=_code_index([_EDUCATION_ROW]),
    )

    def recover() -> None:
        spec_reader.unavailable = False

    return store, frame_ref, recover, {f"dcid:{family_spec_id}": "Education"}


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(_graph_label_failure, id="graph-label-batch"),
        pytest.param(_family_label_failure, id="spec-registry-unavailable"),
    ],
)
def test_load_tolerates_a_label_batch_failure_and_reads_labels_again_next_load(
    failure: Callable[[], _LabelFailure],
):
    """A failed label read omits those labels while the frame still loads, and is not cached
    in place of them."""
    store, frame_ref, recover, labels = failure()

    first = store.load(frame_ref=frame_ref)
    recover()
    second = store.load(frame_ref=frame_ref)

    assert first.labels == {}
    assert second.labels == labels


def test_load_caps_the_distinct_refs_it_batches_for_labels(monkeypatch):
    """Label lookup sends only the first capped distinct refs in row order."""
    import presentation_store.frame_store as frame_store_module

    monkeypatch.setattr(frame_store_module, "_LABEL_RESOLUTION_CAP", 1)
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    graph = _FakeGraphReader(
        labels={
            "MalariaGrantDisbursements": "Malaria Grant Disbursements",
            "country/ETH": "Ethiopia",
            "src_oecd_dac": "OECD DAC",
        }
    )
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    resolved = store.load(frame_ref=frame_ref)

    assert len(resolved.labels) == 1
    assert len(graph.node_arcs_many_calls[0]) == 1


def test_load_labels_unit_refs_under_their_own_cap_after_the_dimension_refs(monkeypatch):
    """Unit dcid refs join the label batch after the capped dimension refs, so they cannot
    crowd those out. A structured unit is skipped, and units past their cap stay raw."""
    import presentation_store.frame_store as frame_store_module

    monkeypatch.setattr(frame_store_module, "_LABEL_RESOLUTION_CAP", 1)
    monkeypatch.setattr(frame_store_module, "_UNIT_LABEL_CAP", 1)
    frame_ref = malaria_ethiopia_frame_ref()
    units = {
        "2015": {"currency": "USD", "prices": "constant", "base_year": 2015},
        "2016": "CurrencyUSD",
        "2017": "USDollar",
    }
    doc = {
        "frame_version": "frame1",
        "columns": ["stat_var", "recipient", "time_period", "value", "unit", "provenance"],
        "rows": [
            ["dcid:MalariaGrantDisbursements", "country/ETH", period, 1.0, unit, "src_oecd_dac"]
            for period, unit in units.items()
        ],
    }
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=json.dumps(doc).encode("utf-8"))
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    graph = _FakeGraphReader(
        labels={
            "MalariaGrantDisbursements": "Malaria Grant Disbursements",
            "country/ETH": "Ethiopia",
            "CurrencyUSD": "US Dollar",
            "USDollar": "US Dollar (legacy)",
        }
    )
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    resolved = store.load(frame_ref=frame_ref)

    assert graph.node_arcs_many_calls == [("MalariaGrantDisbursements", "CurrencyUSD")]
    assert resolved.labels == {
        "dcid:MalariaGrantDisbursements": "Malaria Grant Disbursements",
        "CurrencyUSD": "US Dollar",
    }


# -- load(): family-ref labels ----------------------------------------------


def _family_ref_frame_ref(*, spec_id: str = "spec_" + "e" * 32) -> FrameRef:
    return FrameRef(spec_id=spec_id, content_version="cv_" + "f" * 32)


def _family_ref_rows(*, family_spec_id: str) -> tuple[dict[str, object], ...]:
    """Build a collapsed row with no role column."""
    return (
        {
            "stat_var": f"dcid:{family_spec_id}",
            "time_period": "2020",
            "value": 1_361_983_900.0,
            "provenance": "src_oecd_dac",
        },
    )


@dataclass
class _FakeSpecReader:
    """Record batched definition reads and optionally raise for quarantined IDs or an
    unavailable registry."""

    definitions: dict[str, StoredDefinition] = field(default_factory=dict)
    quarantined: frozenset[str] = frozenset()
    unavailable: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def read_definition(
        self, spec_id: str, *, schema_version: int | None = None
    ) -> StoredDefinition | None:
        return self.read_definitions((spec_id,)).get(spec_id)

    def read_definitions(self, spec_ids) -> dict[str, StoredDefinition]:
        spec_ids = tuple(spec_ids)
        self.calls.append(spec_ids)
        if self.unavailable:
            raise SpecRegistryUnavailable("spec registry pool borrow timed out")
        hit = self.quarantined.intersection(spec_ids)
        if hit:
            raise IdentityQuarantined(subject_id=sorted(hit)[0], id_kind="spec")
        return {
            spec_id: definition
            for spec_id, definition in self.definitions.items()
            if spec_id in spec_ids
        }


def _stored_definition(*, spec_id: str, slots: list[dict[str, object]]) -> StoredDefinition:
    return StoredDefinition(
        spec_id=spec_id,
        schema_version=1,
        identity={
            "identity_version": "spec1",
            "shape": {"kind": "statvar_schema", "population_type": "developmentFinanceFlow"},
            "slots": slots,
            "entities": {"recipient": ["country/FRA"]},
            "requested_window": {"kind": "all"},
        },
        identity_bytes=b"{}",
        identity_digest="a" * 64,
    )


def _code_index(rows: Sequence[CodeRow]) -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=rows,
        statvars=(),
    )


_EDUCATION_ROW = CodeRow(
    dcid="CRSPurposeCode11220",
    code_system="crsPurpose",
    code="11220",
    label="Education",
    synonyms=(),
    parent=None,
)
_ODA_GRANTS_ROW = CodeRow(
    dcid="CRSTypeOfFlow10",
    code_system="crsTypeOfFlow",
    code="10",
    label="ODA Grants",
    synonyms=(),
    parent=None,
)


def test_load_labels_a_family_ref_from_the_stored_definition_through_the_code_index():
    """A collapsed family label joins the bound slot labels from the code index."""
    family_spec_id = "spec_" + "c" * 32
    frame_ref = _family_ref_frame_ref()
    data = json.dumps(
        build_frame_doc(role_keys=(), rows=_family_ref_rows(family_spec_id=family_spec_id))
    ).encode()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    spec_reader = _FakeSpecReader(
        definitions={
            family_spec_id: _stored_definition(
                spec_id=family_spec_id,
                slots=[
                    {
                        "property": "crsPurpose",
                        "binding": {"state": "value", "ref": "CRSPurposeCode11220"},
                    },
                    {
                        "property": "crsTypeOfFlow",
                        "binding": {"state": "value", "ref": "CRSTypeOfFlow10"},
                    },
                ],
            )
        }
    )
    graph = _FakeGraphReader()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        spec_reader=spec_reader,
        code_index=_code_index([_EDUCATION_ROW, _ODA_GRANTS_ROW]),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {f"dcid:{family_spec_id}": "Education, ODA Grants"}
    assert spec_reader.calls == [(family_spec_id,)]
    # No graph node names a family -- `_resolve_labels` never sends the
    # family ref itself to `node_arcs_many`, only the frame's other ref
    # (its provenance column).
    assert graph.node_arcs_many_calls == [("src_oecd_dac",)]


def test_load_labels_a_family_ref_on_a_published_ops_frame():
    """Family labels do not depend on the frame ID prefix."""
    family_spec_id = "spec_" + "d" * 32
    frame_ref = _published_ops_frame_ref()
    data = json.dumps(
        build_frame_doc(role_keys=(), rows=_family_ref_rows(family_spec_id=family_spec_id))
    ).encode()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    spec_reader = _FakeSpecReader(
        definitions={
            family_spec_id: _stored_definition(
                spec_id=family_spec_id,
                slots=[
                    {
                        "property": "crsPurpose",
                        "binding": {"state": "value", "ref": "CRSPurposeCode11220"},
                    }
                ],
            )
        }
    )
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        spec_reader=spec_reader,
        code_index=_code_index([_EDUCATION_ROW]),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {f"dcid:{family_spec_id}": "Education"}


def test_load_family_ref_falls_back_to_the_raw_ref_when_the_spec_id_is_unknown():
    """An unknown family ID stays unlabeled without failing the load."""
    family_spec_id = "spec_" + "9" * 32
    frame_ref = _family_ref_frame_ref()
    data = json.dumps(
        build_frame_doc(role_keys=(), rows=_family_ref_rows(family_spec_id=family_spec_id))
    ).encode()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        spec_reader=_FakeSpecReader(),
        code_index=_code_index([_EDUCATION_ROW]),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {}


def test_load_family_ref_falls_back_when_no_spec_reader_is_configured():
    """Without a spec reader, family refs remain unlabeled."""
    family_spec_id = "spec_" + "8" * 32
    frame_ref = _family_ref_frame_ref()
    data = json.dumps(
        build_frame_doc(role_keys=(), rows=_family_ref_rows(family_spec_id=family_spec_id))
    ).encode()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    store = _store(snapshots=snapshots, artifacts=artifacts)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {}


def test_load_family_ref_falls_back_when_the_spec_id_is_quarantined():
    family_spec_id = "spec_" + "7" * 32
    frame_ref = _family_ref_frame_ref()
    data = json.dumps(
        build_frame_doc(role_keys=(), rows=_family_ref_rows(family_spec_id=family_spec_id))
    ).encode()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        spec_reader=_FakeSpecReader(quarantined=frozenset({family_spec_id})),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {}


def test_load_labels_a_family_ref_and_a_graph_ref_from_the_same_frame_independently():
    """A graph batch failure does not affect separately resolved family labels."""
    family_spec_id = "spec_" + "6" * 32
    frame_ref = _family_ref_frame_ref()
    rows = (
        {
            "stat_var": f"dcid:{family_spec_id}",
            "recipient": "country/FRA",
            "time_period": "2020",
            "value": 1_361_983_900.0,
            "provenance": "src_oecd_dac",
        },
    )
    data = json.dumps(build_frame_doc(role_keys=("recipient",), rows=rows)).encode()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    spec_reader = _FakeSpecReader(
        definitions={
            family_spec_id: _stored_definition(
                spec_id=family_spec_id,
                slots=[
                    {
                        "property": "crsPurpose",
                        "binding": {"state": "value", "ref": "CRSPurposeCode11220"},
                    }
                ],
            )
        }
    )
    graph = _FakeGraphReader(raise_for=frozenset({"country/FRA"}))
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        spec_reader=spec_reader,
        code_index=_code_index([_EDUCATION_ROW]),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {f"dcid:{family_spec_id}": "Education"}


# -- load(): refusals -----------------------------------------------------


def test_load_resolves_a_published_ops_frame():
    """A published `ops_`/`ocv_` frame resolves through the normal load path."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.frame_ref == frame_ref
    assert resolved.frame["frame_version"] == "frame1"


def _citing_graph() -> _FakeGraphReader:
    """A graph holding the fixture frame's `src_oecd_dac` provenance and its licence."""
    return _FakeGraphReader(
        arcs={
            "src_oecd_dac": {
                "typeOf": ("Provenance",),
                "name": ("OECD Creditor Reporting System",),
                "url": ("https://example.org/crs",),
                "licenseType": ("license/cc_by_4",),
            },
            "license/cc_by_4": {"name": ("CC BY 4.0",)},
        }
    )


_OECD_CITATION = {
    "text": "OECD Creditor Reporting System",
    "sources": [
        {
            "ref": "src_oecd_dac",
            "label": "OECD Creditor Reporting System",
            "processing_level": None,
            "license": "CC BY 4.0",
            "license_type": "license/cc_by_4",
            "license_url": None,
            "url": "https://example.org/crs",
            "source": None,
            "data_commons": None,
            "documentation": {
                "publisher": None,
                "description": None,
                "source_data_url": None,
                "attribution": None,
                "update_cadence": None,
                "last_update": None,
                "years_covered": None,
                "granularity": None,
                "currencies": [],
                "transformation_log": [],
                "version_history": [],
            },
        }
    ],
}


def test_load_cites_the_distinct_upstream_provenances_of_a_published_ops_frame():
    """Nine rows sharing one provenance ref give one cited source, beside the lineage body."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=_citing_graph())

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.manifest == {
        **_version(frame_ref).immutable_body,
        "citation": _OECD_CITATION,
    }


def test_load_caches_an_ops_frame_s_citation_with_the_frame():
    """A cache hit serves the citation without reading the graph again."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)
    store.load(frame_ref=frame_ref)
    reads_after_first_load = graph.calls

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.manifest["citation"] == _OECD_CITATION
    assert graph.calls == reads_after_first_load


def test_load_leaves_an_ops_citation_out_when_the_graph_fails_and_reads_it_next_load(caplog):
    """A failed citation read is logged, still loads the frame, and is not cached in place of one."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    graph = _citing_graph()
    graph.raise_for = frozenset({"src_oecd_dac"})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    with caplog.at_level(logging.WARNING, logger="presentation_store.frame_store"):
        first = store.load(frame_ref=frame_ref)
    graph.raise_for = frozenset()
    second = store.load(frame_ref=frame_ref)

    [record] = [r for r in caplog.records if r.name == "presentation_store.frame_store"]
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None
    assert frame_ref.spec_id in record.getMessage()
    assert frame_ref.content_version in record.getMessage()
    assert "citation" not in first.manifest
    assert second.manifest["citation"] == _OECD_CITATION


@pytest.mark.parametrize(
    ("raised", "cached"),
    [(SourceDocumentNotFound("src_oecd_dac"), True), (GraphReadError("transport"), False)],
    ids=["nothing-to-document", "documentation-read-failed"],
)
def test_load_caches_an_ops_frame_unless_a_documentation_read_failed(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, cached: bool
):
    """A source the graph documents nothing for is a lasting answer and caches; a failed
    documentation read is read again on the next load."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    def read(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(provenance_doc, "read_provenance_doc", read)
    first = store.load(frame_ref=frame_ref)
    monkeypatch.undo()
    reads_after_first_load = graph.calls
    second = store.load(frame_ref=frame_ref)

    undocumented = json.loads(json.dumps(_OECD_CITATION))
    undocumented["sources"][0]["documentation"] = None
    assert first.manifest["citation"] == undocumented
    assert (graph.calls == reads_after_first_load) is cached
    assert second.manifest["citation"] == (undocumented if cached else _OECD_CITATION)


def _thin_spec_fixture() -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef, dict]:
    """A root frame whose stored citation predates licence types, licence URLs and
    documentation: those three fields are null and its label is the stored one."""
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    thin = json.loads(json.dumps(_OECD_CITATION))
    thin["sources"][0].update(
        label="Stored label", license_type=None, license_url=None, documentation=None
    )
    version = _version(frame_ref)
    snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)] = replace(
        version, immutable_body={**version.immutable_body, "citation": thin}
    )
    return snapshots, artifacts, frame_ref, thin


_HEALED_THIN_CITATION = {
    **_OECD_CITATION,
    "sources": [{**_OECD_CITATION["sources"][0], "label": "Stored label"}],
}


def test_load_fills_a_root_frame_s_empty_citation_fields_from_the_graph():
    """A card export cites the stored citation with its null licence type and documentation
    filled live; the stored label stands, and the filled citation caches with the frame."""
    snapshots, artifacts, frame_ref, thin = _thin_spec_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    first = store.load(frame_ref=frame_ref)
    reads_after_first_load = graph.calls
    second = store.load(frame_ref=frame_ref)

    assert first.manifest["citation"] == _HEALED_THIN_CITATION
    assert second.manifest["citation"] == _HEALED_THIN_CITATION
    assert graph.calls == reads_after_first_load
    stored = snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)]
    assert stored.immutable_body["citation"] == thin


def test_load_keeps_a_root_frame_s_stored_citation_when_the_read_fails_and_reads_it_next_load():
    snapshots, artifacts, frame_ref, thin = _thin_spec_fixture()
    graph = _citing_graph()
    graph.raise_for = frozenset({"src_oecd_dac"})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    first = store.load(frame_ref=frame_ref)
    graph.raise_for = frozenset()
    second = store.load(frame_ref=frame_ref)

    assert first.manifest["citation"] == thin
    assert second.manifest["citation"] == _HEALED_THIN_CITATION


def test_a_failed_documentation_read_caches_the_frame_and_retries_only_the_fill(
    monkeypatch: pytest.MonkeyPatch,
):
    """The labels answer but the documentation read fails: later loads fetch no artifact and
    retry only the citation fill, until one answers and the frame serves from the cache."""
    snapshots, artifacts, frame_ref, _ = _thin_spec_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)
    read_documentation = provenance_doc.read_provenance_doc
    attempts: list[str] = []

    def failing(*_args, **_kwargs):
        attempts.append("documentation")
        raise GraphReadError("transport")

    monkeypatch.setattr(provenance_doc, "read_provenance_doc", failing)
    first = store.load(frame_ref=frame_ref)
    second = store.load(frame_ref=frame_ref)
    failed_attempts = len(attempts)
    monkeypatch.setattr(provenance_doc, "read_provenance_doc", read_documentation)
    third = store.load(frame_ref=frame_ref)
    graph_calls_after_third = graph.calls
    fourth = store.load(frame_ref=frame_ref)

    assert first.manifest["citation"] == second.manifest["citation"]
    assert first.manifest["citation"] != _HEALED_THIN_CITATION
    assert third.manifest["citation"] == fourth.manifest["citation"] == _HEALED_THIN_CITATION
    assert failed_attempts == 2
    assert [call for call in artifacts.calls if call[0] == "get"] == artifacts.calls[:1]
    assert graph.calls == graph_calls_after_third


def _two_source_ops_fixture() -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef]:
    """A published `ops_` frame whose rows carry `src_oecd_dac`, then `src_imf`."""
    frame_ref = _published_ops_frame_ref()
    rows = [
        *malaria_ethiopia_rows(),
        {**malaria_ethiopia_rows()[0], "recipient": "country/KEN", "provenance": "src_imf"},
    ]
    data = json.dumps(build_frame_doc(role_keys=("recipient",), rows=rows)).encode("utf-8")
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    return snapshots, artifacts, frame_ref


def _citation_requests(graph: _FakeGraphReader) -> list[tuple[tuple[str, ...], float | None]]:
    """The citation reads `load` sent, told apart from the label read by asking for licences."""
    return [
        (dcids, deadline)
        for dcids, properties, deadline in graph.node_arcs_many_requests
        if "licenseType" in properties
    ]


def test_load_reads_citations_for_capped_refs_and_cites_the_rest_by_ref(monkeypatch):
    """Refs past the cap are never sent to the graph and are cited by their ref alone."""
    import presentation_store.frame_store as frame_store_module

    monkeypatch.setattr(frame_store_module, "_CITATION_READ_CAP", 1)
    snapshots, artifacts, frame_ref = _two_source_ops_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    resolved = store.load(frame_ref=frame_ref)

    requested = {dcid for dcids, _ in _citation_requests(graph) for dcid in dcids}
    assert "src_oecd_dac" in requested
    assert "src_imf" not in requested
    assert resolved.manifest["citation"] == {
        "text": "OECD Creditor Reporting System; src_imf",
        "sources": [
            *_OECD_CITATION["sources"],
            {
                "ref": "src_imf",
                "label": "src_imf",
                "processing_level": None,
                "license": None,
                "license_type": None,
                "license_url": None,
                "url": None,
                "source": None,
                "data_commons": None,
                "documentation": None,
            },
        ],
    }


def test_load_bounds_its_label_and_citation_reads_by_its_read_budget_less_grace():
    """The label read and every citation read carry one absolute deadline, the load's budget
    less its grace."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, read_budget_seconds=5.0)

    before = time.monotonic()
    store.load(frame_ref=frame_ref)
    after = time.monotonic()

    assert _citation_requests(graph)
    assert len(graph.node_arcs_many_requests) > len(_citation_requests(graph))
    deadlines = {deadline for _, _, deadline in graph.node_arcs_many_requests}
    assert len(deadlines) == 1
    [deadline] = deadlines
    assert deadline is not None
    assert before + 4.5 <= deadline <= after + 4.5


def test_load_bounds_its_graph_reads_by_an_earlier_door_deadline():
    """A door deadline earlier than the store's own budget is the one the graph reads carry."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, read_budget_seconds=30.0)

    before = time.monotonic()
    with frame_reads_bounded_by(timeout_seconds=5.0):
        after = time.monotonic()
        store.load(frame_ref=frame_ref)

    deadlines = {deadline for _, _, deadline in graph.node_arcs_many_requests}
    assert len(deadlines) == 1
    [deadline] = deadlines
    assert deadline is not None
    assert before + 4.5 <= deadline <= after + 4.5


def test_load_refuses_an_unpublished_ops_prefixed_spec_id_as_ops_unsupported():
    """An unpublished `ops_` pair returns `ops_unsupported` after one version lookup."""
    snapshots = _FakeSnapshotReader()
    artifacts = _FakeArtifactStore()
    store = _store(snapshots=snapshots, artifacts=artifacts)
    frame_ref = FrameRef(spec_id="ops_deadbeef", content_version="ocv_deadbeef")

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "ops_unsupported"
    assert "not published" in excinfo.value.detail
    assert snapshots.calls == ["read_version"]
    assert artifacts.calls == []


def test_load_refuses_unknown_when_no_version_record_exists():
    store = _store()
    frame_ref = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "unknown"


@pytest.mark.parametrize(
    "frame_ref_factory",
    [
        pytest.param(malaria_ethiopia_frame_ref, id="ordinary-frame"),
        pytest.param(
            _published_ops_frame_ref,
            id="published-ops-frame",
        ),
    ],
)
def test_load_refuses_a_revoked_version(frame_ref_factory):
    """A revocation blocks ordinary and published `ops_` frames."""
    frame_ref = frame_ref_factory()
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        revocations={
            (frame_ref.spec_id, frame_ref.content_version): Revocation(
                reason="takedown", revoked_at=_NOW
            )
        },
    )
    store = _store(snapshots=snapshots)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "revoked"
    assert "takedown" in excinfo.value.detail


def test_load_refuses_a_collapsed_frame_whose_base_version_is_revoked():
    """Revoking the base version also blocks a collapsed version."""
    frame_ref = malaria_ethiopia_frame_ref()
    snapshots = _FakeSnapshotReader(
        versions={
            (frame_ref.spec_id, frame_ref.content_version): _collapsed_version(
                frame_ref, base_content_version="cv_base"
            )
        },
        revocations={
            (frame_ref.spec_id, "cv_base"): Revocation(reason="base-takedown", revoked_at=_NOW)
        },
    )
    store = _store(snapshots=snapshots)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "revoked"
    assert "base-takedown" in excinfo.value.detail
    # Names the ref the caller actually asked for, not the base version.
    assert repr(frame_ref.content_version) in excinfo.value.detail


def test_load_resolves_a_collapsed_frame_whose_base_version_is_not_revoked():
    """A non-revoked base version remains available."""
    frame_ref = malaria_ethiopia_frame_ref()
    data = _frame_bytes()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={
            (frame_ref.spec_id, frame_ref.content_version): _collapsed_version(
                frame_ref, base_content_version="cv_base"
            )
        },
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    store = _store(snapshots=snapshots, artifacts=artifacts)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.frame_ref == frame_ref


def test_load_refuses_when_no_location_is_servable():
    frame_ref = malaria_ethiopia_frame_ref()
    long_ago = _NOW - timedelta(days=365)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest="sha256:aged-out", tier=Tier.working.value, stored_at=long_ago)
            ]
        },
    )
    store = _store(snapshots=snapshots)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "expired"


def test_load_refuses_a_published_ops_frame_aged_out_of_the_working_tier():
    """An expired working-tier `ops_` frame returns `expired` when no pinned copy exists."""
    frame_ref = _published_ops_frame_ref()
    long_ago = _NOW - timedelta(days=365)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest="sha256:aged-out", tier=Tier.working.value, stored_at=long_ago)
            ]
        },
    )
    store = _store(snapshots=snapshots)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "expired"


def test_load_refuses_when_no_rendering_carries_the_frame1_media_type():
    frame_ref = malaria_ethiopia_frame_ref()
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(
                    digest="sha256:csv-only",
                    tier=Tier.working.value,
                    media_type="text/csv",
                    rendering_key="csv",
                )
            ]
        },
    )
    store = _store(snapshots=snapshots)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "missing_rendering"


def test_load_refuses_on_digest_mismatch():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    corrupt_artifacts = _FakeArtifactStore(corrupt=frozenset({digest}))
    corrupt_artifacts.objects.update(artifacts.objects)
    store = _store(snapshots=snapshots, artifacts=corrupt_artifacts)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "digest_mismatch"


def test_load_refuses_expired_when_artifact_bytes_are_gone():
    frame_ref = malaria_ethiopia_frame_ref()
    artifacts = _FakeArtifactStore()  # nothing ever `seed`ed
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest="sha256:never-stored", tier=Tier.working.value)
            ]
        },
    )
    store = _store(snapshots=snapshots, artifacts=artifacts)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "expired"


# -- Cache: byte-bound eviction --------------------------------------------


def _seed_frame_ref(
    *, snapshots: _FakeSnapshotReader, artifacts: _FakeArtifactStore, spec_suffix: str
) -> FrameRef:
    frame_ref = FrameRef(spec_id=f"spec_{spec_suffix * 32}", content_version="cv_" + "1" * 32)
    digest = artifacts.seed(tier=Tier.working, data=_frame_bytes())
    snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)] = _version(frame_ref)
    snapshots.locations[(frame_ref.spec_id, frame_ref.content_version)] = [
        _location(digest=digest, tier=Tier.working.value)
    ]
    return frame_ref


def test_cache_evicts_oldest_insertion_once_over_the_byte_budget():
    entry_bytes = len(_frame_bytes())
    snapshots = _FakeSnapshotReader()
    artifacts = _FakeArtifactStore()
    ref_a = _seed_frame_ref(snapshots=snapshots, artifacts=artifacts, spec_suffix="a")
    ref_b = _seed_frame_ref(snapshots=snapshots, artifacts=artifacts, spec_suffix="b")

    # A budget under two full entries: loading a second frame must evict
    # the first (oldest-inserted).
    store = LiveFrameStore(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=None,
        reference_writer=_FakeReferenceWriter(),
        now=lambda: _NOW,
        cache_max_bytes=entry_bytes + entry_bytes // 2,
    )

    store.load(frame_ref=ref_a)
    store.load(frame_ref=ref_b)
    snapshot_calls_before_a_reload = len(snapshots.calls)

    # `ref_a` is evicted, so reloading it reads the version again.
    store.load(frame_ref=ref_a)
    assert len(snapshots.calls) > snapshot_calls_before_a_reload


def test_cache_put_does_not_double_count_bytes_on_a_same_key_overwrite():
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts)
    cache_key = (frame_ref.spec_id, frame_ref.content_version)
    resolved = store.load(frame_ref=frame_ref)

    # A forced same-key double-put -- e.g. two concurrent resolves of the
    # same frame racing this store's own cache -- must not leave
    # `_cache_bytes` counting the evicted-by-overwrite prior entry too.
    store._cache_put(cache_key, resolved, size_bytes=1_000, citation_complete=True)

    assert store._cache_bytes == sum(entry.size_bytes for entry in store._cache.values())


# -- pin() ------------------------------------------------------------------


def test_pin_promotes_working_and_writes_a_reference():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    reference_writer = _FakeReferenceWriter()
    store = _store(snapshots=snapshots, artifacts=artifacts, reference_writer=reference_writer)

    store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert ("copy", Tier.working, digest) in artifacts.calls
    assert (Tier.pinned, digest) in artifacts.objects
    key = (
        "presentation:cmp_test",
        digest,
        frame_ref.spec_id,
        frame_ref.content_version,
        PRESENTATION_PIN_REFERENCE_KIND,
    )
    assert key in reference_writer.records


def test_pin_promotes_working_for_a_published_ops_frame():
    """`write_component` (`assembler.write.write_component`) calls `pin()`
    unconditionally on every create whose candidate resolved a frame,
    including one resolved under a published `ops_`/`ocv_` pair. `pin()`
    needs nothing an `ops_` pair lacks: the publish route gave it a real
    `artifact_digest` and `tier` exactly like a definition mint's
    `StoredArtifactLocation`, so this promotion sequence runs
    unchanged."""
    snapshots, artifacts, frame_ref, digest = _ops_happy_path_fixture()
    reference_writer = _FakeReferenceWriter()
    store = _store(snapshots=snapshots, artifacts=artifacts, reference_writer=reference_writer)

    store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert ("copy", Tier.working, digest) in artifacts.calls
    assert (Tier.pinned, digest) in artifacts.objects
    key = (
        "presentation:cmp_test",
        digest,
        frame_ref.spec_id,
        frame_ref.content_version,
        PRESENTATION_PIN_REFERENCE_KIND,
    )
    assert key in reference_writer.records


def test_pin_maps_a_copy_write_failure_to_frame_unavailable_io():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    artifacts.copy_raises = ArtifactWriteError(object_name=digest, digest=digest)
    store = _store(snapshots=snapshots, artifacts=artifacts)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert excinfo.value.reason == "io"


def test_pin_maps_a_copy_inventory_failure_to_frame_unavailable_io():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    artifacts.copy_raises = ArtifactInventoryError(digest=digest)
    store = _store(snapshots=snapshots, artifacts=artifacts)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert excinfo.value.reason == "io"


def test_pin_skips_the_copy_when_already_pinned():
    frame_ref = malaria_ethiopia_frame_ref()
    data = _frame_bytes()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.pinned, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.pinned.value)
            ]
        },
    )
    store = _store(snapshots=snapshots, artifacts=artifacts)

    store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert all(call[0] != "copy" for call in artifacts.calls)


def test_pin_is_idempotent_on_repeat_calls():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    reference_writer = _FakeReferenceWriter()
    store = _store(snapshots=snapshots, artifacts=artifacts, reference_writer=reference_writer)

    assert store.pin(frame_ref=frame_ref, component_id="cmp_test") is True
    key = (
        "presentation:cmp_test",
        digest,
        frame_ref.spec_id,
        frame_ref.content_version,
        PRESENTATION_PIN_REFERENCE_KIND,
    )
    first_created_at = reference_writer.records[key]

    assert store.pin(frame_ref=frame_ref, component_id="cmp_test") is False

    assert len(reference_writer.records) == 1
    assert reference_writer.records[key] == first_created_at  # unchanged, not re-stamped


def test_pin_uses_record_reference_when_claim_reference_is_unavailable():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    reference_writer = _LegacyReferenceWriter()
    store = _store(snapshots=snapshots, artifacts=artifacts, reference_writer=reference_writer)

    assert store.pin(frame_ref=frame_ref, component_id="cmp_test") is True
    key = (
        "presentation:cmp_test",
        digest,
        frame_ref.spec_id,
        frame_ref.content_version,
        PRESENTATION_PIN_REFERENCE_KIND,
    )
    assert key in reference_writer.inner.records


@pytest.mark.parametrize(
    ("refusing_writer", "reason"),
    [
        (lambda pair: _FakeReferenceWriter(revoked=pair), "revoked"),
        (lambda pair: _FakeReferenceWriter(unlinked=pair), "expired"),
    ],
    ids=["revocation", "link-replaced-by-a-re-render"],
)
def test_pin_maps_a_write_time_refusal_to_frame_unavailable(
    refusing_writer: Callable[[set[tuple[str, str]]], _FakeReferenceWriter], reason: str
):
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    reference_writer = refusing_writer({(frame_ref.spec_id, frame_ref.content_version)})
    store = _store(snapshots=snapshots, artifacts=artifacts, reference_writer=reference_writer)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert excinfo.value.reason == reason
    assert reference_writer.records == {}


def test_pin_refuses_a_collapsed_frame_whose_base_version_is_revoked():
    """Same base-revocation check as `load()`'s own -- `_resolve_canonical
    _location` is shared by both."""
    frame_ref = malaria_ethiopia_frame_ref()
    snapshots = _FakeSnapshotReader(
        versions={
            (frame_ref.spec_id, frame_ref.content_version): _collapsed_version(
                frame_ref, base_content_version="cv_base"
            )
        },
        revocations={
            (frame_ref.spec_id, "cv_base"): Revocation(reason="base-takedown", revoked_at=_NOW)
        },
    )
    store = _store(snapshots=snapshots)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert excinfo.value.reason == "revoked"
    assert "base-takedown" in excinfo.value.detail


def test_pin_promotes_a_collapsed_frame_whose_base_version_is_not_revoked():
    """A non-revoked base version remains available."""
    frame_ref = malaria_ethiopia_frame_ref()
    data = _frame_bytes()
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={
            (frame_ref.spec_id, frame_ref.content_version): _collapsed_version(
                frame_ref, base_content_version="cv_base"
            )
        },
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    store = _store(snapshots=snapshots, artifacts=artifacts)

    assert store.pin(frame_ref=frame_ref, component_id="cmp_test") is True


# -- release_pin() ------------------------------------------------------------


def test_release_pin_removes_the_reference_row_and_a_repeat_call_is_a_no_op():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    reference_writer = _FakeReferenceWriter()
    store = _store(snapshots=snapshots, artifacts=artifacts, reference_writer=reference_writer)
    store.pin(frame_ref=frame_ref, component_id="cmp_test")
    key = (
        "presentation:cmp_test",
        digest,
        frame_ref.spec_id,
        frame_ref.content_version,
        PRESENTATION_PIN_REFERENCE_KIND,
    )
    assert key in reference_writer.records

    store.release_pin(frame_ref=frame_ref, component_id="cmp_test")

    assert key not in reference_writer.records

    store.release_pin(frame_ref=frame_ref, component_id="cmp_test")

    assert key not in reference_writer.records


def test_release_pin_leaves_the_pinned_copy_in_place():
    """`release_pin` removes only the reference row -- the pinned artifact
    copy stays, the same shape `datasets_api.reconciliation`'s own
    unreferenced-pinned-digest leg already exists to catch."""
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts)
    store.pin(frame_ref=frame_ref, component_id="cmp_test")

    store.release_pin(frame_ref=frame_ref, component_id="cmp_test")

    assert (Tier.pinned, digest) in artifacts.objects
