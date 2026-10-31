"""Checks `LiveFrameStore` with in-memory snapshot, artifact, graph, and reference fakes."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any, Never, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import Frame, canonical_frame_bytes
from dfo_frame_grammar.grammar import FrameError, derive_columns

from artifact_store.errors import (
    ArtifactIntegrityError,
    ArtifactInventoryError,
    ArtifactNotFoundError,
    ArtifactWriteError,
)
from artifact_store.store import ArtifactSink, StoredArtifact, Tier
from assembler.fakes import FakeReleaseRegistry, FakeRevisionStore
from assembler.limits import LABEL_MAX
from assembler.pipeline import write_component
from assembler.ports import (
    Continuation,
    FrameRef,
    FrameUnavailable,
    LeftOutSeries,
    ResolvedFrame,
    RollingAnchor,
)
from assembler.resolve import resolve_selectors
from assembler.testing import (
    build_frame_doc,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_rows,
)
from code_index.members import canonical_identity
from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import CodeRow, IndexStamp
from dataset_store.snapshot import (
    ArtifactNotFound,
    FrameLineage,
    ManifestFacts,
    Revocation,
    SnapshotRevoked,
    StoredArtifactLocation,
    StoredVersion,
    VersionFacts,
    VersionState,
)
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    LiveAllowanceLedger,
)
from datasets_api.manifest import version_document
from datasets_api.recipe_store import RecipeRecord
from datasets_api.reference_store import ReferenceWrite
from datasets_api.retention_admission import OpenPinBudgetExhausted, PinBudgetExhausted
from datasets_api.store_errors import DatasetsStoreUnavailable
from description.citation import citation_line
from description.facts import (
    Builds,
    ComposedVariable,
    ComputedFacts,
    DescriptionRecord,
    Evidence,
    FixedValue,
    InputVersion,
    SeriesFacts,
    StatusCounts,
    VariableFacts,
)
from description.steps import Step
from description.vocabulary import SourceDocumentFacts
from dfo_platform.caller import Caller
from dfo_platform.timings import PhaseTimings, collecting
from graph_generation import FakeGeneration, GenerationStamp
from graph_release.context import bound_release, current_release
from materialise import provenance_doc
from materialise.continued_series import ContinuedSeries
from materialise.formats import CANONICAL_FRAME_MEDIA_TYPE, read_frame1
from materialise.graph import GraphReadError
from materialise.provenance_doc import SourceDocumentNotFound
from presentation_store import frame_store
from presentation_store.frame_store import (
    FRAME1_MEDIA_TYPE,
    PRESENTATION_PIN_REFERENCE_KIND,
    LiveFrameStore,
    _CachedFrame,
    frame_reads_bounded_by,
)
from spec_registry import recompute_identity
from spec_registry.errors import (
    IdentityQuarantined,
    SpecRegistryStoreError,
    SpecRegistryUnavailable,
)
from spec_registry.reader import StoredDefinition
from tests.code_index.aid_identity import IMPUTED_PROVENANCE, imputed_education_aid_identity
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.graph_release.support import release_at_revision
from tests.materialise._canonical_frame import (
    canonical_frame_parquet,
    stored_canonical_frame_959f8bc,
)
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

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

    def read_version_state(
        self, subject_id: str, content_version: str, *, revoked_under: Sequence[str] = ()
    ) -> VersionState | None:
        self.calls.append("read_version_state")
        version = self.versions.get((subject_id, content_version))
        if version is None:
            return None
        revocation = next(
            (
                revoked
                for cv in (content_version, version.base_content_version)
                if cv is not None
                for subject in dict.fromkeys((subject_id, *revoked_under))
                if (revoked := self.revocations.get((subject, cv))) is not None
            ),
            None,
        )
        return VersionState(
            version=version,
            revocation=revocation,
            locations=list(self.locations.get((subject_id, content_version), [])),
        )


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

    def signed_location_url(
        self,
        digest: str,
        *,
        tier: Tier,
        bucket: str,
        object_name: str,
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
    """Record batched node reads and optionally fail the whole batch: one holding a dcid in
    `raise_for`, or one asking for exactly `raise_for_properties`. A dcid in `arcs` answers
    with those arcs, any other with its `labels` entry as its `name`. Its generation stamp is
    `generation`'s, unknown by default."""

    def __init__(
        self,
        *,
        labels: dict[str, str] | None = None,
        arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
        raise_for: frozenset[str] = frozenset(),
        raise_for_properties: tuple[str, ...] | None = None,
        generation: FakeGeneration | None = None,
    ) -> None:
        self._labels = labels or {}
        self._arcs = arcs or {}
        self._generation = generation if generation is not None else FakeGeneration()
        self.raise_for = raise_for
        self.raise_for_properties = raise_for_properties
        self.calls = 0
        self.node_arcs_many_calls: list[tuple[str, ...]] = []
        self.node_arcs_many_requests: list[
            tuple[tuple[str, ...], tuple[str, ...], float | None]
        ] = []

    def generation_stamp(self) -> GenerationStamp:
        return self._generation.stamp()

    def node_label(self, dcid: str) -> str | None:
        raise NotImplementedError("not exercised by LiveFrameStore")

    def observation_facets_batch(self, *, stat_var: str, entities, deadline: float | None = None):
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
        if any(dcid in self.raise_for for dcid in dcids) or (
            tuple(properties) == self.raise_for_properties
        ):
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
        revoked_under: Sequence[str] = (),
    ) -> datetime:
        self.calls.append(
            {
                "method": "record_reference_unless_revoked",
                "key": (principal, artifact_digest, subject_id, content_version, reference_kind),
                "now": now,
            }
        )
        if any(
            (subject, content_version) in self.revoked for subject in (subject_id, *revoked_under)
        ):
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
        revoked_under: Sequence[str] = (),
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
            revoked_under=revoked_under,
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

    def holds_reference(self, *, subject_id: str, content_version: str) -> bool:
        return any(key[2:4] == (subject_id, content_version) for key in self.records)


class _LegacyReferenceWriter:
    """Writer with `record_reference_unless_revoked` but no claim method."""

    def __init__(self) -> None:
        self.inner = _FakeReferenceWriter()

    def record_reference_unless_revoked(self, **kwargs) -> datetime:
        return self.inner.record_reference_unless_revoked(**kwargs)

    def remove_reference(self, **kwargs) -> None:
        self.inner.remove_reference(**kwargs)


# -- Fixtures -----------------------------------------------------------


def _frame1_with(**members: object) -> bytes:
    """The malaria frame1 document with `members` replaced; a `None` member is left out."""
    doc = build_frame_doc(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    doc.update(members)
    return json.dumps({key: value for key, value in doc.items() if value is not None}).encode()


def _frame1_without(column: str) -> bytes:
    """The malaria frame1 document without `column`."""
    doc = build_frame_doc(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    columns = cast("list[str]", doc["columns"])
    index = columns.index(column)
    doc["columns"] = [name for name in columns if name != column]
    doc["rows"] = [
        [cell for position, cell in enumerate(row) if position != index]
        for row in cast("list[list[object]]", doc["rows"])
    ]
    return json.dumps(doc).encode()


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
    rendering: dict[str, object] | None = None,
) -> StoredArtifactLocation:
    return StoredArtifactLocation(
        artifact_digest=digest,
        rendering_key=rendering_key,
        rendering=rendering if rendering is not None else {},
        media_type=media_type,
        size_bytes=0,
        created_at=created_at,
        stored_at=stored_at,
        tier=tier,
        bucket="fake-bucket",
        object_name=digest,
    )


def _facts(frame_ref: FrameRef) -> VersionFacts:
    """A published frame's lineage, or a dataset version's facts with no cited source."""
    if frame_ref.spec_id.startswith("ops_"):
        return FrameLineage(input_content_versions=[])
    return ManifestFacts(
        prepared_at=datetime(2026, 1, 1, tzinfo=UTC),
        definition={"schema_version": 1, "spec_id": frame_ref.spec_id, "identity": {}},
        coverage={"rows": 0, "dimensions": []},
        citation={"text": "", "sources": []},
        trace={},
        warnings=[],
    )


def _version(frame_ref: FrameRef) -> StoredVersion:
    return StoredVersion(
        subject_id=frame_ref.spec_id,
        subject_kind="spec",
        content_version=frame_ref.content_version,
        schema_version=1,
        facts=_facts(frame_ref),
    )


def _collapsed_version(frame_ref: FrameRef, *, base_content_version: str) -> StoredVersion:
    """Build a stored version that names its source content version."""
    return StoredVersion(
        subject_id=frame_ref.spec_id,
        subject_kind="spec",
        content_version=frame_ref.content_version,
        schema_version=1,
        facts=_facts(frame_ref),
        base_content_version=base_content_version,
    )


@dataclass
class _Clock:
    """A store's clock, which a test moves past the wait before a description record that is
    not final is read again."""

    now: datetime = _NOW

    def __call__(self) -> datetime:
        return self.now

    def pass_recheck(self) -> None:
        self.now += frame_store._RECHECK


def _store(
    *,
    snapshots: _FakeSnapshotReader | None = None,
    artifacts: _FakeArtifactStore | None = None,
    graph=None,
    reference_writer: _FakeReferenceWriter | None = None,
    allowances: LiveAllowanceLedger | None = None,
    now: Callable[[], datetime] = lambda: _NOW,
    **kwargs,
) -> LiveFrameStore:
    return LiveFrameStore(
        snapshots=snapshots if snapshots is not None else _FakeSnapshotReader(),
        artifacts=artifacts if artifacts is not None else _FakeArtifactStore(),
        graph=graph,
        reference_writer=reference_writer
        if reference_writer is not None
        else _FakeReferenceWriter(),
        allowances=allowances,
        now=now,
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


def test_load_reads_a_version_s_canonical_frame_ahead_of_its_frame1_document():
    """The canonical frame loads as the parsed frame1 document of its frame, and a frame1
    document beside it is never read."""
    frame_ref = malaria_ethiopia_frame_ref()
    frame = Frame.build(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    artifacts = _FakeArtifactStore()
    canonical = artifacts.seed(tier=Tier.working, data=canonical_frame_parquet((frame,)))
    frame1 = artifacts.seed(tier=Tier.working, data=b"never read")
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=frame1, tier=Tier.working.value),
                _location(
                    digest=canonical,
                    tier=Tier.working.value,
                    media_type=CANONICAL_FRAME_MEDIA_TYPE,
                    rendering_key="frame",
                    rendering={"format": "frame"},
                ),
            ]
        },
    )

    resolved = _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    assert resolved.frame == json.loads(canonical_frame_bytes(frame))
    assert [call for call in artifacts.calls if call[0] == "get"] == [
        ("get", Tier.working, canonical)
    ]


def _canonical_frame_fixture() -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef, Frame]:
    """A version whose only stored frame is the canonical frame of the malaria rows."""
    frame = Frame.build(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    snapshots, artifacts, frame_ref = _canonical_frame_version(canonical_frame_parquet((frame,)))
    return snapshots, artifacts, frame_ref, frame


def _canonical_frame_version(
    data: bytes,
) -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef]:
    """A version whose only stored frame is the canonical frame `data`."""
    frame_ref = malaria_ethiopia_frame_ref()
    artifacts = _FakeArtifactStore()
    canonical = artifacts.seed(tier=Tier.working, data=data)
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(
                    digest=canonical,
                    tier=Tier.working.value,
                    media_type=CANONICAL_FRAME_MEDIA_TYPE,
                    rendering_key="frame",
                    rendering={"format": "frame"},
                )
            ]
        },
    )
    return snapshots, artifacts, frame_ref


def test_load_refuses_a_canonical_frame_over_the_row_bound_from_its_footer(monkeypatch):
    """A canonical frame with more rows than `frame_max_rows` refuses `missing_rendering` on
    its Parquet footer's row count, and its rows are never decoded."""
    snapshots, artifacts, frame_ref, frame = _canonical_frame_fixture()

    def never_decoded(*_args, **_kwargs):
        raise AssertionError("an oversized frame's rows were decoded")

    monkeypatch.setattr(frame_store, "read_canonical_frame_table", never_decoded)
    store = _store(
        snapshots=snapshots, artifacts=artifacts, frame_max_rows=frame.table.num_rows - 1
    )

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "missing_rendering"


def test_load_refuses_a_canonical_frame_over_the_row_bound_before_fetching_it():
    """A version whose facts record more rows than `frame_max_rows` refuses
    `missing_rendering` on the stored count, and the artifact store is never asked for the
    frame."""
    snapshots, artifacts, frame_ref, frame = _canonical_frame_fixture()
    key = (frame_ref.spec_id, frame_ref.content_version)
    version = snapshots.versions[key]
    facts = cast(ManifestFacts, version.facts)
    snapshots.versions[key] = replace(
        version,
        facts=replace(facts, coverage={**facts.coverage, "rows": frame.table.num_rows}),
    )
    store = _store(
        snapshots=snapshots, artifacts=artifacts, frame_max_rows=frame.table.num_rows - 1
    )

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "missing_rendering"
    assert [call for call in artifacts.calls if call[0] == "get"] == []


def _parquet_bytes() -> bytes:
    sink = io.BytesIO()
    table = pa.table({"name": [f"row {index}" for index in range(5000)]})
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()


def _footer_zeroed(data: bytes) -> bytes:
    footer = int.from_bytes(data[-8:-4], "little")
    return data[: -8 - footer] + bytes(footer) + data[-8:]


def _pages_flipped(data: bytes) -> bytes:
    flipped = bytearray(data)
    for index in range(40, 2000, 3):
        flipped[index] ^= 0xFF
    return bytes(flipped)


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda _: b"not a parquet file", id="not-parquet"),
        pytest.param(lambda _: b"", id="empty"),
        pytest.param(_footer_zeroed, id="footer-zeroed"),
        pytest.param(_pages_flipped, id="pages-flipped"),
        pytest.param(lambda data: data[: len(data) // 2], id="body-truncated"),
        pytest.param(lambda data: data[:4] + data[len(data) // 2 :], id="tail-kept"),
    ],
)
def test_load_refuses_a_canonical_frame_whose_bytes_do_not_decode(corrupt):
    """Stored bytes that match their digest and do not decode as a frame never load, so the
    load refuses `missing_rendering`."""
    snapshots, artifacts, frame_ref = _canonical_frame_version(corrupt(_parquet_bytes()))
    store = _store(snapshots=snapshots, artifacts=artifacts)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)

    assert excinfo.value.reason == "missing_rendering"


def test_load_refuses_a_frame_of_another_schema():
    """A Parquet file that is no canonical frame decodes, and the grammar refuses its schema."""
    snapshots, artifacts, frame_ref = _canonical_frame_version(_parquet_bytes())

    with pytest.raises(FrameUnavailable) as excinfo:
        _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    assert excinfo.value.reason == "missing_rendering"


def _frame_of_every_unit() -> Frame:
    """The malaria rows, holding every form a `unit` cell stores in turn."""
    units = (
        None,
        "USDollar",
        {"currency": "USD", "prices": "current"},
        {"currency": "EUR", "prices": "constant", "base_year": 2020},
        {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
        {
            "numerator": {"currency": "USD", "prices": "current"},
            "denominator": {"currency": "EUR", "prices": "current"},
        },
    )
    rows = (
        {**row, "unit": units[index % len(units)]}
        for index, row in enumerate(malaria_ethiopia_rows())
    )
    return Frame.build(role_keys=("recipient",), rows=rows)


def test_every_bit_flip_of_a_canonical_frame_loads_or_refuses_missing_rendering():
    """A stored frame's bytes can break anywhere, and each break is a permanent refusal."""
    data = canonical_frame_parquet((_frame_of_every_unit(),))
    rng = random.Random(7)
    for _ in range(1000):
        mutant = bytearray(data)
        mutant[rng.randrange(len(mutant))] ^= 1 << rng.randrange(8)
        snapshots, artifacts, frame_ref = _canonical_frame_version(bytes(mutant))
        store = _store(snapshots=snapshots, artifacts=artifacts)
        try:
            store.load(frame_ref=frame_ref)
        except FrameUnavailable as refused:
            assert refused.reason == "missing_rendering"


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(MemoryError(), id="python-out-of-memory"),
        pytest.param(pa.ArrowMemoryError("exemplar"), id="arrow-out-of-memory"),
    ],
)
def test_load_refuses_io_when_decoding_runs_out_of_memory(
    monkeypatch: pytest.MonkeyPatch, raised: Exception
):
    def fail(_data: bytes, *, columns: object = None) -> Never:
        raise raised

    snapshots, artifacts, frame_ref, _ = _canonical_frame_fixture()
    monkeypatch.setattr(frame_store, "read_canonical_frame_table", fail)

    with pytest.raises(FrameUnavailable) as excinfo:
        _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    assert excinfo.value.reason == "io"
    assert excinfo.value.retry_after_s == frame_store._DECODE_RETRY_AFTER_S


def _raises(raised: Exception) -> Callable[..., Never]:
    def fail(*_args: object, **_kwargs: object) -> Never:
        raise raised

    return fail


def _logged_traceback_names(exc: BaseException, digest: str) -> bool:
    """Whether the traceback the platform logs for `exc` names `digest`."""
    return digest in logging.Formatter().formatException((type(exc), exc, exc.__traceback__))


@pytest.mark.parametrize(
    "site", ["read_canonical_frame_table", "canonical_frame_document", "compute_structural_view"]
)
@pytest.mark.parametrize(
    "raised", [pytest.param(TypeError, id="type"), pytest.param(KeyError, id="key")]
)
def test_a_defect_reading_a_stored_frame_reaches_internal_error_naming_its_digest(
    monkeypatch: pytest.MonkeyPatch, site: str, raised: type[Exception]
):
    """A refusal comes only from a check of the stored content, so a raised defect stays one
    whatever its type."""
    snapshots, artifacts, frame_ref, _ = _canonical_frame_fixture()
    monkeypatch.setattr(frame_store, site, _raises(raised("exemplar")))

    with pytest.raises(raised) as excinfo:
        _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    ((location,),) = snapshots.locations.values()
    assert _logged_traceback_names(excinfo.value, location.artifact_digest)


def test_a_structural_view_out_of_memory_refuses_io(monkeypatch: pytest.MonkeyPatch):
    snapshots, artifacts, frame_ref, _ = _canonical_frame_fixture()
    monkeypatch.setattr(frame_store, "compute_structural_view", _raises(MemoryError()))

    with pytest.raises(FrameUnavailable) as excinfo:
        _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    assert excinfo.value.reason == "io"


def _with_column(data: bytes, name: str, array: pa.Array) -> bytes:
    """The canonical frame `data` with column `name` replaced, as a damaged artifact holds it."""
    table = pq.read_table(pa.BufferReader(data))
    sink = io.BytesIO()
    pq.write_table(table.set_column(table.column_names.index(name), name, array), sink)
    return sink.getvalue()


def _stored_cell(data: bytes, name: str, cell: object) -> bytes:
    """`data` with column `name` holding `cell` in every row."""
    table = pq.read_table(pa.BufferReader(data))
    return _with_column(
        data, name, pa.array([cell] * table.num_rows, table.schema.field(name).type)
    )


@pytest.mark.parametrize(
    ("name", "cell"),
    [
        pytest.param("unit", {"ref": None, "request": None, "composed": None}, id="no-variant"),
        pytest.param("unit", {"ref": "USD", "composed": '{"a":1}'}, id="two-variants"),
        pytest.param("unit", {"ref": ""}, id="empty-ref"),
        pytest.param("unit", {"request": {"currency": None, "prices": None}}, id="empty-request"),
        pytest.param(
            "unit", {"request": {"currency": "USD", "prices": "constant"}}, id="no-base-year"
        ),
        pytest.param("unit", {"composed": "not json"}, id="composed-not-json"),
        pytest.param("unit", {"composed": "[" * 100_000}, id="composed-past-the-parser"),
        pytest.param(
            "unit",
            {"composed": '{"numerator":' + "9" * 5000 + ',"denominator":"USD"}'},
            id="composed-past-the-digit-limit",
        ),
        pytest.param(
            "unit",
            {"composed": '{"numerator":"a","numerator":"b","denominator":"USD"}'},
            id="composed-repeats-a-member",
        ),
        pytest.param("unit", {"composed": '"USD"'}, id="composed-a-ref"),
        pytest.param("unit", {"composed": "null"}, id="composed-null"),
        pytest.param("unit", {"composed": '{"numerator":"USD"}'}, id="composed-one-side"),
        pytest.param(
            "unit",
            {"composed": '{"numerator":"USD","denominator":{"stat_var":3}}'},
            id="composed-bad-side",
        ),
        pytest.param("time_period", None, id="null-period"),
        pytest.param("time_period", "20:0", id="unparseable-period"),
    ],
)
def test_a_malformed_stored_cell_refuses_missing_rendering_naming_its_digest(
    caplog: pytest.LogCaptureFixture, name: str, cell: object
):
    data = canonical_frame_parquet((_frame_of_every_unit(),))
    snapshots, artifacts, frame_ref = _canonical_frame_version(data)
    assert _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    snapshots, artifacts, frame_ref = _canonical_frame_version(_stored_cell(data, name, cell))
    with caplog.at_level(logging.ERROR), pytest.raises(FrameUnavailable) as excinfo:
        _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    assert excinfo.value.reason == "missing_rendering"
    ((location,),) = snapshots.locations.values()
    (record,) = caplog.records
    assert record.args[0] == location.artifact_digest
    assert record.exc_info[0] is FrameError


@pytest.mark.parametrize(
    ("column", "cell"),
    [
        pytest.param("unit", ["USD"], id="unit-a-list"),
        pytest.param("unit", {"currency": "USD"}, id="unit-no-prices"),
        pytest.param("unit", {"numerator": "USD"}, id="unit-one-side"),
        pytest.param("time_period", None, id="period-null"),
        pytest.param("time_period", 2020, id="period-a-number"),
        pytest.param("time_period", ["2020"], id="period-a-list"),
        pytest.param("time_period", "20:0", id="period-unparseable"),
        pytest.param("recipient", None, id="role-null"),
        pytest.param("value", None, id="value-null"),
        pytest.param("value", "1.5", id="value-a-numeric-string"),
        pytest.param("value", "abc", id="value-a-string"),
        pytest.param("value", [1], id="value-a-list"),
        pytest.param("value", 10**400, id="value-past-the-double-range"),
        pytest.param("value", True, id="value-a-boolean"),
    ],
)
def test_a_malformed_frame1_cell_refuses_missing_rendering(column: str, cell: object):
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    document = json.loads(artifacts.objects[(Tier.working, digest)])
    index = document["columns"].index(column)
    document["rows"][0][index] = cell
    artifacts.objects[(Tier.working, digest)] = json.dumps(document).encode()

    with pytest.raises(FrameUnavailable) as excinfo:
        _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    assert excinfo.value.reason == "missing_rendering"


def test_a_stored_frame_that_does_not_decode_logs_its_digest_and_cause(
    caplog: pytest.LogCaptureFixture,
):
    snapshots, artifacts, frame_ref = _canonical_frame_version(b"not a parquet file")

    with caplog.at_level(logging.ERROR), pytest.raises(FrameUnavailable):
        _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref)

    (record,) = caplog.records
    (digest, cause, _) = record.args
    ((location,),) = snapshots.locations.values()
    assert digest == location.artifact_digest
    assert cause == "ArrowInvalid"
    assert record.exc_info[0] is pa.ArrowInvalid


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe", id="not-utf8"),
        pytest.param(b"[" * 100_000, id="nested-past-the-parser"),
        pytest.param(b"null", id="null"),
        pytest.param(b"[]", id="list"),
        pytest.param(b"{}", id="no-columns"),
        pytest.param(b'{"columns": 3, "rows": []}', id="columns-not-a-list"),
        pytest.param(b'{"columns": ["a", "b"], "rows": [[1]]}', id="short-row"),
        pytest.param(b"[" + b"9" * 5000 + b"]", id="integer-past-the-digit-limit"),
        pytest.param(_frame1_with(frame_version="frame2"), id="another-version"),
        pytest.param(_frame1_with(frame_version=None), id="no-version"),
        pytest.param(_frame1_with(columns=["stat_var", 3], rows=[]), id="a-column-not-a-name"),
        pytest.param(
            json.dumps(
                {"frame_version": "frame1", "columns": ["stat_var", "\ud800"], "rows": []}
            ).encode(),
            id="a-column-name-with-a-lone-surrogate",
        ),
        pytest.param(_frame1_with(rows=[["dcid:X"]]), id="a-short-row"),
        pytest.param(_frame1_without("unit"), id="no-unit-column"),
        pytest.param(_frame1_without("provenance"), id="no-provenance-column"),
        pytest.param(
            _frame1_with(columns=["stat_var", "stat_var", "time_period", "value", "unit"], rows=[]),
            id="a-repeated-column",
        ),
    ],
)
def test_a_frame1_document_that_is_not_a_frame_refuses_missing_rendering(data: bytes):
    with pytest.raises(FrameUnavailable) as excinfo:
        frame_store._decode_frame(data, location=_location(digest="d", tier="working"), max_rows=9)

    assert excinfo.value.reason == "missing_rendering"


def _frame1_with_value(value: int) -> bytes:
    """The malaria document with `value` in its first row, as `write_frame1` writes a whole
    number: a JSON integer."""
    doc = build_frame_doc(role_keys=("recipient",), rows=malaria_ethiopia_rows())
    columns = cast("list[str]", doc["columns"])
    cast("list[list[object]]", doc["rows"])[0][columns.index("value")] = value
    return json.dumps(doc).encode()


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(_frame1_with(), id="whole"),
        pytest.param(_frame1_without("measurementMethod"), id="before-measurementMethod"),
        pytest.param(_frame1_without("status"), id="before-status"),
        pytest.param(_frame1_with_value(20_900_000_000_000_000), id="a-value-past-2-to-the-53"),
    ],
)
def test_a_frame1_document_decodes_as_the_canonical_frame_of_its_rows(data: bytes):
    """A frame1 document, of any age and past the row bound, decodes to the document of the
    canonical frame of its rows, with its unit cells shared, and counts in the cache at that
    frame's measure whatever its byte length. The malaria document holds 9 rows."""
    doc, size = frame_store._decode_frame(
        data, location=_location(digest="d", tier="working"), max_rows=1
    )
    canonical = canonical_frame_parquet((read_frame1(data),))
    canonical_doc, canonical_size = frame_store._decode_frame(
        canonical,
        location=_location(digest="c", tier="working", media_type=CANONICAL_FRAME_MEDIA_TYPE),
        max_rows=9,
    )

    rows, columns = cast("list[list[object]]", doc["rows"]), cast("list[str]", doc["columns"])
    unit = columns.index("unit")
    assert doc == canonical_doc
    assert all(row[unit] is rows[0][unit] for row in rows)
    assert size == canonical_size == frame_store._document_bytes(rows=9, columns=len(columns))
    assert size != len(data)


def test_load_decodes_a_canonical_frame_at_the_row_bound():
    snapshots, artifacts, frame_ref, frame = _canonical_frame_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts, frame_max_rows=frame.table.num_rows)

    assert store.load(frame_ref=frame_ref).frame == json.loads(canonical_frame_bytes(frame))


def test_load_reads_a_canonical_frame_stored_before_unit_recipes():
    """A canonical frame stored under frame grammar kit 959f8bc, whose unit `request` has no
    `recipe`, loads as the frame1 document its release read from it."""
    data, _frame_sha256, document = stored_canonical_frame_959f8bc()
    snapshots, artifacts, frame_ref = _canonical_frame_version(data)

    assert _store(snapshots=snapshots, artifacts=artifacts).load(frame_ref=frame_ref).frame == (
        document
    )


def test_load_resolves_a_frame_with_view_manifest_and_labels(bind_catalogue):
    snapshots, artifacts, frame_ref, _digest = _happy_path_fixture()
    graph = _FakeGraphReader(
        labels={
            "MalariaGrantDisbursements": "Malaria Grant Disbursements",
            "country/ETH": "Ethiopia",
            "src_oecd_dac": "OECD DAC",
        },
        generation=FakeGeneration(token="gen-1"),
    )
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)
    bind_catalogue(FakeCatalogue(build_id="build-1"))

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.frame_ref == frame_ref
    assert resolved.frame["frame_version"] == "frame1"
    assert resolved.structural_view.series_count == 1
    assert resolved.manifest == version_document(_version(frame_ref))
    assert resolved.labels["dcid:MalariaGrantDisbursements"] == "Malaria Grant Disbursements"
    assert resolved.labels["country/ETH"] == "Ethiopia"
    assert resolved.labels["src_oecd_dac"] == "OECD DAC"
    assert graph.calls == 1  # one batched call covers every distinct ref
    assert sorted(graph.node_arcs_many_calls[0]) == sorted(
        ["MalariaGrantDisbursements", "country/ETH", "src_oecd_dac"]
    )
    assert (resolved.graph_generation, resolved.catalogue_build) == (
        "gen-1",
        f"build-1@{current_release().release_id}",
    )


@pytest.mark.parametrize(
    "names",
    [
        ("Ethiopia", "Federal Democratic Republic of Ethiopia"),
        ("Federal Democratic Republic of Ethiopia", "Ethiopia"),
    ],
)
def test_load_labels_a_node_of_several_names_as_the_node_door_does(bind_catalogue, names):
    """A card's label is the node's `display_name`, whatever order the graph gives its names."""
    snapshots, artifacts, frame_ref, _digest = _happy_path_fixture()
    graph = _FakeGraphReader(labels={}, arcs={"country/ETH": {"name": names}})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)
    bind_catalogue(FakeCatalogue(build_id="build-1"))

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels["country/ETH"] == "Federal Democratic Republic of Ethiopia"


def test_load_labels_the_entities_and_members_a_coverage_warning_counts():
    """An entity or member that served nothing holds no row, so the refs a
    `coverage_below_scope` warning carries join the frame's one label batch."""
    snapshots, artifacts, frame_ref, _digest = _happy_path_fixture()
    version = snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)]
    warning = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": "Short of scope.",
        "unserved_entities": {"count": 1, "refs": ["dcid:country/SEN"]},
        "member_count": 3,
        "off_cadence_members": {"count": 1, "refs": ["dcid:YouthRate"]},
        "out_of_window_members": {"count": 1, "refs": ["dcid:AdultRate"]},
    }
    snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)] = replace(
        version, facts=replace(version.facts, warnings=[warning])
    )
    graph = _FakeGraphReader(
        labels={
            "MalariaGrantDisbursements": "Malaria Grant Disbursements",
            "YouthRate": "Youth rate",
            "AdultRate": "Adult rate",
            "country/SEN": "Senegal",
        }
    )
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    resolved = store.load(frame_ref=frame_ref)

    assert (
        resolved.labels["dcid:country/SEN"],
        resolved.labels["dcid:YouthRate"],
        resolved.labels["dcid:AdultRate"],
    ) == ("Senegal", "Youth rate", "Adult rate")
    assert graph.calls == 1
    assert {"country/SEN", "YouthRate", "AdultRate"} <= set(graph.node_arcs_many_calls[0])


_WDI_GDP = "worldBank/NY_GDP_MKTP_CD"


def _world_bank_fixture() -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef]:
    """`_happy_path_fixture` over a frame whose rows hold one World Development Indicators
    variable."""
    snapshots, artifacts, frame_ref, _digest = _happy_path_fixture()
    rows = [{**row, "stat_var": f"dcid:{_WDI_GDP}"} for row in malaria_ethiopia_rows()]
    digest = artifacts.seed(
        tier=Tier.working,
        data=json.dumps(build_frame_doc(role_keys=("recipient",), rows=rows)).encode("utf-8"),
    )
    snapshots.locations[(frame_ref.spec_id, frame_ref.content_version)] = [
        _location(digest=digest, tier=Tier.working.value)
    ]
    return snapshots, artifacts, frame_ref


def _world_bank_record(frame_ref: FrameRef, *, variable: str | None) -> _RecordReader:
    """A reader answering a dataset version's record whose definition resolved to
    `variable`, or to several variables when it is `None`."""
    record = _input_record(
        (frame_ref.spec_id, frame_ref.content_version), variable=_WDI_GDP, recipient="country/ETH"
    )
    assert record.variables is not None
    return _RecordReader(
        record.model_copy(
            update={"variables": record.variables.model_copy(update={"variable": variable})}
        )
    )


def test_load_reads_the_description_of_the_versions_one_world_bank_variable():
    """The description of the World Development Indicators variable a version resolved to
    defines its measure, so the load reads that one variable's description, by bare dcid."""
    snapshots, artifacts, frame_ref = _world_bank_fixture()
    graph = _FakeGraphReader(
        arcs={_WDI_GDP: {"name": ("GDP (current US$)",), "description": ("GDP is output.",)}}
    )
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_world_bank_record(frame_ref, variable=_WDI_GDP),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.descriptions == {_WDI_GDP: "GDP is output."}
    assert [(dcids, properties) for dcids, properties, _ in graph.node_arcs_many_requests][-1] == (
        (_WDI_GDP,),
        ("description",),
    )


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(None, id="version-of-several-variables"),
        pytest.param("Amount_Oda", id="version-of-another-variable"),
    ],
)
def test_load_reads_no_description_unless_the_version_resolved_to_one_world_bank_variable(
    variable: str | None,
):
    """A frame holding a World Development Indicators variable reads no description when its
    version's measure is not that one variable."""
    snapshots, artifacts, frame_ref = _world_bank_fixture()
    graph = _FakeGraphReader(arcs={_WDI_GDP: {"name": ("GDP",), "description": ("GDP.",)}})
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_world_bank_record(frame_ref, variable=variable),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.descriptions == {}
    assert [properties for _, properties, _ in graph.node_arcs_many_requests] == [("name",)]


def test_a_failed_description_read_is_logged_not_cached_and_read_again(caplog):
    """Only the description read fails. The labels resolve, the failure is logged with the
    variable, and the next load reads the frame and its description again."""
    snapshots, artifacts, frame_ref = _world_bank_fixture()
    graph = _FakeGraphReader(
        arcs={_WDI_GDP: {"name": ("GDP",), "description": ("GDP is output.",)}},
        raise_for_properties=("description",),
    )
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_world_bank_record(frame_ref, variable=_WDI_GDP),
    )

    with caplog.at_level(logging.WARNING, logger="presentation_store.frame_store"):
        first = store.load(frame_ref=frame_ref)
    graph.raise_for_properties = None
    second = store.load(frame_ref=frame_ref)

    [logged] = [r for r in caplog.records if r.name == "presentation_store.frame_store"]
    assert _WDI_GDP in logged.getMessage()
    assert first.labels[f"dcid:{_WDI_GDP}"] == "GDP"
    assert first.descriptions == {}
    assert second is not first
    assert second.descriptions == {_WDI_GDP: "GDP is output."}


@pytest.mark.parametrize(
    ("change", "reread"),
    [
        pytest.param("none", False, id="same-stamps-within-a-day"),
        pytest.param("catalogue", True, id="catalogue-build-changed"),
        pytest.param("generation", True, id="graph-generation-changed"),
        pytest.param("release", True, id="release-changed-under-the-origin-flag"),
        pytest.param("age", True, id="a-day-old"),
    ],
)
def test_a_cached_frame_rereads_its_words_once_its_stamps_change_or_age(
    bind_catalogue, request, change, reread
):
    """A cached frame serves its labels, citation and source documents while the graph
    generation and catalogue build they were read under hold and they are younger than that
    generation's cache lifetime. A new build, a new generation, a new release or a day's age
    re-reads them from the cached bytes, so a card composed after a source's refresh dates
    change shows the new dates, under the stamps of the read they came from."""
    first_release, second_release = release_at_revision("rev-1"), release_at_revision("rev-2")
    stack = ExitStack()
    request.addfinalizer(stack.close)
    stack.enter_context(bound_release(first_release))
    snapshots, artifacts, frame_ref, digest = _full_spec_fixture()
    generation = FakeGeneration(token="gen-1")
    graph = _documenting_graph()
    graph._generation = generation
    clock = _Clock()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_RecordReader(
            DescriptionRecord.model_construct(computed=None, variables=None)
        ),
        now=clock,
    )
    bind_catalogue(FakeCatalogue(build_id="build-1"))
    first = store.load(frame_ref=frame_ref)
    graph._arcs["src_oecd_dac"] |= {
        "lastDataRefreshDate": ("2026-11-02",),
        "nextDataRefreshDate": ("2027-03",),
    }
    if change == "catalogue":
        bind_catalogue(FakeCatalogue(build_id="build-2"))
    elif change == "generation":
        generation.advance("gen-2")
    elif change == "release":
        stack.enter_context(bound_release(second_release))
    elif change == "age":
        clock.now += timedelta(days=1)
    calls = graph.calls

    second = store.load(frame_ref=frame_ref)

    dates = second.source_documents["src_oecd_dac"]
    assert first.source_documents["src_oecd_dac"].next_release == "2026-12"
    assert (dates.last_downloaded, dates.next_release) == (
        ("2026-11-02", "2027-03") if reread else ("2026-10-08", "2026-12")
    )
    assert (second is first) is not reread
    assert (graph.calls > calls) is reread
    assert (second.graph_generation, second.catalogue_build) == {
        "none": ("gen-1", f"build-1@{first_release.release_id}"),
        "catalogue": ("gen-1", f"build-2@{first_release.release_id}"),
        "generation": ("gen-2", f"build-1@{first_release.release_id}"),
        "release": ("gen-1", f"build-1@{second_release.release_id}"),
        "age": ("gen-1", f"build-1@{first_release.release_id}"),
    }[change]
    assert artifacts.calls.count(("get", Tier.working, digest)) == 1


@pytest.mark.parametrize("failing", ["label-read", "documentation-read"])
def test_a_failed_refresh_of_stale_words_keeps_the_cached_words_until_a_later_retry(
    monkeypatch: pytest.MonkeyPatch, failing: str
):
    """After a generation change, a refresh whose label or documentation read fails serves the
    cached labels, source dates and stamps, and caches no fallback. A load within the retry
    wait reads nothing; the first load past it, once the reads recover, takes the new words."""
    snapshots, artifacts, frame_ref, digest = _full_spec_fixture()
    generation = FakeGeneration(token="gen-1")
    graph = _documenting_graph()
    graph._generation = generation
    graph._labels["country/ETH"] = "Ethiopia"
    clock = _Clock()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_RecordReader(
            DescriptionRecord.model_construct(computed=None, variables=None)
        ),
        now=clock,
    )
    first = store.load(frame_ref=frame_ref)
    graph._labels["country/ETH"] = "Federal Democratic Republic of Ethiopia"
    graph._arcs["src_oecd_dac"] |= {"nextDataRefreshDate": ("2027-03",)}
    generation.advance("gen-2")
    if failing == "label-read":
        graph.raise_for = frozenset({"country/ETH"})
    else:

        def read(*_args, **_kwargs):
            raise GraphReadError("transport")

        monkeypatch.setattr(provenance_doc, "read_provenance_doc", read)

    failed = store.load(frame_ref=frame_ref)
    graph.raise_for = frozenset()
    monkeypatch.undo()
    calls = graph.calls
    within = store.load(frame_ref=frame_ref)
    calls_within = graph.calls - calls
    clock.pass_recheck()
    recovered = store.load(frame_ref=frame_ref)

    assert failed is first
    assert failed.labels["country/ETH"] == "Ethiopia"
    assert failed.source_documents["src_oecd_dac"].next_release == "2026-12"
    assert failed.graph_generation == "gen-1"
    assert within is first
    assert calls_within == 0
    assert recovered.labels["country/ETH"] == "Federal Democratic Republic of Ethiopia"
    assert recovered.source_documents["src_oecd_dac"].next_release == "2027-03"
    assert recovered.graph_generation == "gen-2"
    assert artifacts.calls.count(("get", Tier.working, digest)) == 1


def test_a_failed_refresh_keeps_its_retry_wait_through_a_citation_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    """A frame whose documentation read failed retries its citation on each load. After a
    generation change during a graph outage, the failed label refresh is not run again by the
    loads that retry only the citation before the retry wait passes."""
    snapshots, artifacts, frame_ref, _ = _full_spec_fixture()
    generation = FakeGeneration(token="gen-1")
    graph = _documenting_graph()
    graph._generation = generation
    graph._labels["country/ETH"] = "Ethiopia"
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_RecordReader(
            DescriptionRecord.model_construct(computed=None, variables=None)
        ),
        now=_Clock(),
    )

    def read(*_args, **_kwargs):
        raise GraphReadError("transport")

    monkeypatch.setattr(provenance_doc, "read_provenance_doc", read)
    first = store.load(frame_ref=frame_ref)
    generation.advance("gen-2")
    graph.raise_for = frozenset({"country/ETH"})

    def label_reads() -> int:
        return sum(properties == ("name",) for _, properties, _ in graph.node_arcs_many_requests)

    before = label_reads()
    loads = [store.load(frame_ref=frame_ref) for _ in range(4)]

    assert label_reads() - before == 1
    assert all(load.labels["country/ETH"] == "Ethiopia" for load in loads)
    assert first.graph_generation == "gen-1"


def _stale_words_store(
    *, failing: bool
) -> tuple[LiveFrameStore, FrameRef, _FakeGraphReader, ResolvedFrame]:
    """A store holding a frame loaded under generation `gen-1`, whose words the generation's
    move to `gen-2` leaves stale, and whose label read fails from then on when `failing`."""
    snapshots, artifacts, frame_ref, _ = _full_spec_fixture()
    generation = FakeGeneration(token="gen-1")
    graph = _documenting_graph()
    graph._generation = generation
    graph._labels["country/ETH"] = "Ethiopia"
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_RecordReader(
            DescriptionRecord.model_construct(computed=None, variables=None)
        ),
        now=_Clock(),
    )
    first = store.load(frame_ref=frame_ref)
    generation.advance("gen-2")
    if failing:
        graph.raise_for = frozenset({"country/ETH"})
    return store, frame_ref, graph, first


def _gate_graph_reads(graph: _FakeGraphReader) -> tuple[threading.Event, threading.Event]:
    """Hold every graph read of `graph` until the returned release event is set; the entered
    event is set once a read is held."""
    entered, release = threading.Event(), threading.Event()
    read = graph.node_arcs_many

    def gated(*args, **kwargs):
        entered.set()
        release.wait(5)
        return read(*args, **kwargs)

    graph.node_arcs_many = gated
    return entered, release


@pytest.mark.parametrize("failing", [False, True], ids=["refresh-succeeds", "refresh-fails"])
def test_concurrent_loads_of_a_stale_frame_refresh_it_once(failing: bool):
    """Four loads that find a frame's words stale at once refresh it once: the first refreshes
    while the others wait, then serve what it cached, so they read the graph as often as one
    load does. A refresh that fails leaves every load the cached words and their stamps, and
    a refresh that succeeds leaves the cache its words, never the words it replaced. The test
    waits a fifth of a second for the waiting loads to reach the refresh."""
    single, frame_ref, graph, _ = _stale_words_store(failing=failing)
    calls = graph.calls
    single.load(frame_ref=frame_ref)
    one_refresh = graph.calls - calls

    store, frame_ref, graph, first = _stale_words_store(failing=failing)
    entered, release = _gate_graph_reads(graph)
    calls = graph.calls
    with ThreadPoolExecutor(max_workers=4) as pool:
        loads = [pool.submit(copy_context().run, store.load, frame_ref=frame_ref) for _ in range(4)]
        assert entered.wait(5)
        time.sleep(0.2)
        release.set()
        served = [load.result(timeout=5) for load in loads]
    after = store.load(frame_ref=frame_ref)

    assert graph.calls - calls == one_refresh
    assert all(frame is served[0] for frame in served)
    assert after is served[0]
    assert (served[0] is first) is failing
    assert after.graph_generation == ("gen-1" if failing else "gen-2")


def test_a_load_waiting_on_a_refresh_past_its_deadline_serves_the_cached_words():
    """A load that waits on another load's refresh of the same frame until its door deadline
    passes serves the cached words, reading nothing."""
    store, frame_ref, graph, first = _stale_words_store(failing=False)
    entered, release = _gate_graph_reads(graph)
    with ThreadPoolExecutor(max_workers=1) as pool:
        refresh = pool.submit(copy_context().run, store.load, frame_ref=frame_ref)
        assert entered.wait(5)
        calls = graph.calls
        with frame_reads_bounded_by(timeout_seconds=0.2):
            waited = store.load(frame_ref=frame_ref)
        reads_while_waiting = graph.calls - calls
        release.set()
        refreshed = refresh.result(timeout=5)

    assert waited is first
    assert reads_while_waiting == 0
    assert refreshed.graph_generation == "gen-2"
    assert store.load(frame_ref=frame_ref) is refreshed


def test_a_load_caches_its_frame_only_over_the_entry_it_read():
    """A first load whose frame bytes arrive after another load cached the frame, and a refresh
    re-read its words, leaves the cache the refreshed entry."""
    snapshots, artifacts, frame_ref, _ = _full_spec_fixture()
    generation = FakeGeneration(token="gen-1")
    graph = _documenting_graph()
    graph._generation = generation
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=_RecordReader(
            DescriptionRecord.model_construct(computed=None, variables=None)
        ),
        now=_Clock(),
    )
    entered, release = threading.Event(), threading.Event()
    get = artifacts.get

    def gated(**kwargs):
        if not entered.is_set():
            entered.set()
            release.wait(5)
        return get(**kwargs)

    artifacts.get = gated
    with ThreadPoolExecutor(max_workers=1) as pool:
        late = pool.submit(copy_context().run, store.load, frame_ref=frame_ref)
        assert entered.wait(5)
        store.load(frame_ref=frame_ref)
        generation.advance("gen-2")
        refreshed = store.load(frame_ref=frame_ref)
        release.set()
        late_frame = late.result(timeout=5)

    assert late_frame is not refreshed
    assert store.load(frame_ref=frame_ref) is refreshed


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


def test_load_states_the_rendering_the_versions_linked_files_record_at_each_load():
    """A cut file alone makes the version that rendering. A plain file linked later makes it
    ambiguous, and the next load says so though the frame is cached."""
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    key = (frame_ref.spec_id, frame_ref.content_version)
    (stored_frame,) = snapshots.locations[key]
    snapshots.locations[key] = [replace(stored_frame, rendering={"format": "frame1"})]
    csv = {"format": "csv", "bundle": False}
    snapshots.locations[key].append(
        _location(
            digest=digest,
            tier=Tier.working.value,
            media_type="text/csv",
            rendering_key="cut",
            rendering={**csv, "without_party_totals": True},
        )
    )
    store = _store(snapshots=snapshots, artifacts=artifacts)

    cut = store.load(frame_ref=frame_ref).rendering
    snapshots.locations[key].append(
        _location(
            digest=digest,
            tier=Tier.working.value,
            media_type="text/csv",
            rendering_key="plain",
            rendering=csv,
        )
    )
    ambiguous = store.load(frame_ref=frame_ref).rendering

    minting = {"groupby": None, "without_party_totals": True}
    assert cut == {
        "base_content_version": frame_ref.content_version,
        "minting": minting,
        "ambiguous": False,
        "collapsed": False,
        "mintings": [minting],
    }
    assert ambiguous == {**cut, "ambiguous": True}


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
    assert (resolved.graph_generation, resolved.catalogue_build) == (None, None)


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
        "columns": list(derive_columns(["recipient"])),
        "rows": [
            [
                "dcid:MalariaGrantDisbursements",
                "country/ETH",
                period,
                1.0,
                unit,
                None,
                "src_oecd_dac",
                None,
            ]
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


@pytest.mark.parametrize(
    ("caps", "graph_refs", "expected_labels"),
    [
        pytest.param(
            {},
            (
                "Count_Person",
                "MalariaGrantDisbursements",
                "country/ETH",
                "src_oecd_dac",
                "USDollar",
                "CurrencyEUR",
            ),
            {
                "dcid:MalariaGrantDisbursements": "Malaria Grant Disbursements",
                "dcid:Count_Person": "Total population",
                "dcid:spec_" + "c" * 32: "Education",
                "USDollar": "US Dollar",
            },
            id="every-side-labelled",
        ),
        pytest.param(
            {"_LABEL_RESOLUTION_CAP": 3, "_UNIT_LABEL_CAP": 1},
            ("Count_Person", "MalariaGrantDisbursements", "USDollar"),
            {
                "dcid:Count_Person": "Total population",
                "dcid:spec_" + "c" * 32: "Education",
                "dcid:MalariaGrantDisbursements": "Malaria Grant Disbursements",
                "USDollar": "US Dollar",
            },
            id="stat-var-sides-precede-column-refs-under-the-cap",
        ),
        pytest.param(
            {"_LABEL_RESOLUTION_CAP": 1, "_UNIT_LABEL_CAP": 1},
            ("Count_Person", "USDollar"),
            {"dcid:Count_Person": "Total population", "USDollar": "US Dollar"},
            id="sides-past-each-cap-stay-raw",
        ),
    ],
)
def test_load_labels_each_side_of_a_composed_unit(monkeypatch, caps, graph_refs, expected_labels):
    """A StatVar side joins the stat_var refs ahead of the columns' refs, so a family ref
    takes its stored definition and column refs cannot crowd a side out. A string side joins
    the unit refs. A side the graph does not name stays unlabelled."""
    import presentation_store.frame_store as frame_store_module

    for name, cap in caps.items():
        monkeypatch.setattr(frame_store_module, name, cap)
    family_spec_id = "spec_" + "c" * 32
    frame_ref = malaria_ethiopia_frame_ref()
    units = {
        "2015": {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}},
        "2016": {
            "numerator": "CurrencyEUR",
            "denominator": {"stat_var": f"dcid:{family_spec_id}"},
        },
    }
    doc = {
        "frame_version": "frame1",
        "columns": list(derive_columns(["recipient"])),
        "rows": [
            [
                "dcid:MalariaGrantDisbursements",
                "country/ETH",
                period,
                1.0,
                unit,
                None,
                "src_oecd_dac",
                None,
            ]
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
            "Count_Person": "Total population",
            "USDollar": "US Dollar",
        }
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
        graph=graph,
        spec_reader=spec_reader,
        code_index=_code_index([_EDUCATION_ROW]),
    )

    resolved = store.load(frame_ref=frame_ref)

    assert graph.node_arcs_many_calls == [graph_refs]
    assert resolved.labels == expected_labels


_FAMILY_ON_A_SIDE = "spec_" + "c" * 32
_SUM_ENTRY = ComposedVariable(
    variable="ops_sum", relation="sum", first=[_FAMILY_ON_A_SIDE], second=["Count_Person"]
)
_QUOTIENT_ENTRY = ComposedVariable(
    variable="ops_quotient", relation="quotient", first=["ops_sum"], second=["Count_Person"]
)
_SUM_LABEL = "Education plus Total population"
_QUOTIENT_LABEL = f"{_SUM_LABEL} per Total population"


def _computed_record(
    frame_ref: FrameRef,
    *,
    composed: list[ComposedVariable] | None = None,
    steps: list[dict[str, object]] | None = None,
) -> DescriptionRecord:
    """A computed version's record holding `composed` and `steps`."""
    return DescriptionRecord(
        id=frame_ref.spec_id,
        content_version=frame_ref.content_version,
        schema_version=1,
        basis="publish",
        index_stamp=None,
        variables=None,
        computed=ComputedFacts(
            inputs=[InputVersion(id=None, content_version="cv_" + "d" * 32)], composed=composed
        ),
        series=SeriesFacts(
            series=1,
            rows=1,
            periods=1,
            span=None,
            grain="year",
            latest_period="2015",
            series_reaching_latest=1,
            entities_by_role=None,
            by_source_unit=None,
            status=StatusCounts(estimated=None, provisional=None),
        ),
        steps=None if steps is None else [Step.model_validate(step) for step in steps],
        caveats=[],
        evidence=Evidence(
            prepared=date(2026, 9, 28),
            title=None,
            sources=[],
            source_release=None,
            labels=[],
            membership=None,
            builds=Builds(
                origin_catalogue=None, code_index=None, coverage=None, method_status=None
            ),
        ),
        late_fields=[],
    )


class _RecordReader:
    """Returns one record for every version, and `inputs` for a batched read, which it notes
    in `batches` and fails with `DatasetsStoreUnavailable` while `failing_batches` is above 0."""

    def __init__(self, record: DescriptionRecord | None) -> None:
        self.record = record
        self.inputs: dict[tuple[str, str], DescriptionRecord] = {}
        self.batches: list[list[tuple[str, str]]] = []
        self.failing_batches = 0

    def read_description(self, *, id: str, content_version: str) -> DescriptionRecord | None:
        return self.record

    def read_descriptions(
        self, *, versions: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], DescriptionRecord]:
        self.batches.append(list(versions))
        if self.failing_batches:
            self.failing_batches -= 1
            raise DatasetsStoreUnavailable("reading descriptions failed", sqlstate="08006")
        return {version: self.inputs[version] for version in versions if version in self.inputs}


_COMPUTED_COLUMNS = ["stat_var", "recipient", "time_period", "value", "unit", "provenance"]

# The columns every frame holds, beside its role columns.
_FRAME_COLUMNS = frozenset(derive_columns(()))


def _computed_frame_store(
    *,
    rows: list[list[object]],
    columns: list[str] = _COMPUTED_COLUMNS,
    composed: list[ComposedVariable] | None = None,
    steps: list[dict[str, object]] | None = None,
    raise_for: frozenset[str] = frozenset(),
    graph_labels: Mapping[str, str] | None = None,
    generation: FakeGeneration | None = None,
    clock: _Clock | None = None,
    stored_composed: list[object] | None = None,
    continuations: list[dict[str, object]] | None = None,
) -> tuple[LiveFrameStore, FrameRef, _FakeGraphReader, _RecordReader]:
    """A store over one computed frame of `rows` under `columns`, whose version's record holds
    `composed` and `steps` and whose stored lineage holds `stored_composed` and
    `continuations`, over a graph that also names `graph_labels` and stamps `generation`'s
    token, on `clock`."""
    frame_ref = FrameRef(spec_id="ops_" + "a" * 32, content_version="ocv_" + "b" * 32)
    # The stored document holds the derived column order, each column `columns` leaves out null.
    derived = derive_columns([column for column in columns if column not in _FRAME_COLUMNS])
    doc = {
        "frame_version": "frame1",
        "columns": list(derived),
        "rows": [
            [dict(zip(columns, row, strict=True)).get(column) for column in derived] for row in rows
        ],
    }
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=json.dumps(doc).encode("utf-8"))
    version = dataclasses.replace(
        _version(frame_ref),
        facts=FrameLineage(
            input_content_versions=["cv_" + "d" * 32],
            composed_stat_vars=cast("list[Mapping[str, object]] | None", stored_composed),
            continuations=continuations,
        ),
    )
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): version},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
        },
    )
    graph = _FakeGraphReader(
        labels={
            "Count_Person": "Total population",
            "country/ETH": "Ethiopia",
            **(graph_labels or {}),
        },
        raise_for=raise_for,
        generation=generation,
    )
    spec_reader = _FakeSpecReader(
        definitions={
            _FAMILY_ON_A_SIDE: _stored_definition(
                spec_id=_FAMILY_ON_A_SIDE,
                slots=[
                    {
                        "property": "crsPurpose",
                        "binding": {"state": "value", "ref": "CRSPurposeCode11220"},
                    }
                ],
            )
        }
    )
    reader = _RecordReader(_computed_record(frame_ref, composed=composed, steps=steps))
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        spec_reader=spec_reader,
        code_index=_code_index([_EDUCATION_ROW]),
        description_reader=reader,
        now=clock if clock is not None else _Clock(),
    )
    return store, frame_ref, graph, reader


_COMPOSED_ROWS: list[list[object]] = [
    [
        "dcid:ops_quotient",
        "country/ETH",
        "2015",
        1.0,
        {"numerator": "USDollar", "denominator": {"stat_var": "dcid:ops_sum"}},
        "src_oecd_dac",
    ]
]


def _version_input(label: str, digit: str) -> dict[str, object]:
    return {"id": "spec_" + digit * 32, "content_version": "cv_" + digit * 32, "label": label}


_RETRIEVE_AID = {
    "id": "s1",
    "op": "retrieve",
    "input": _version_input("Gross ODA disbursements", "1"),
}
_RETRIEVE_INCOME = {
    "id": "s2",
    "op": "retrieve",
    "input": _version_input("Gross national income", "2"),
}
_RETRIEVE_POPULATION = {"id": "s4", "op": "retrieve", "input": _version_input("Population", "4")}


_AID_VERSION = ("spec_" + "1" * 32, "cv_" + "1" * 32)
_INCOME_VERSION = ("spec_" + "2" * 32, "cv_" + "2" * 32)


def _input_record(version: tuple[str, str], *, variable: str, recipient: str) -> DescriptionRecord:
    """A dataset version's record whose title names `variable` and the fixed `recipient`."""
    return DescriptionRecord(
        id=version[0],
        content_version=version[1],
        schema_version=1,
        basis="mint",
        index_stamp=None,
        variables=VariableFacts(
            shape=[],
            variable=variable,
            fixed=[FixedValue(property="financingRecipient", value=recipient)],
            dimensions=None,
            admitted_at_mint=1,
            admitted_from="resolved_variables",
            with_data=1,
        ),
        computed=None,
        series=SeriesFacts(
            series=1,
            rows=1,
            periods=1,
            span=None,
            grain="year",
            latest_period="2015",
            series_reaching_latest=1,
            entities_by_role=None,
            by_source_unit=None,
            status=StatusCounts(estimated=None, provisional=None),
        ),
        steps=[],
        caveats=[],
        evidence=Evidence(
            prepared=date(2026, 9, 28),
            title=None,
            sources=[],
            source_release=None,
            labels=[],
            membership=None,
            builds=Builds(
                origin_catalogue=None, code_index=None, coverage=None, method_status=None
            ),
        ),
        late_fields=[],
    )


def _quotient_of_inputs_store(
    *, failing_batches: int = 0, clock: _Clock | None = None
) -> tuple[
    LiveFrameStore, FrameRef, _FakeGraphReader, _RecordReader, tuple[DescriptionRecord, ...]
]:
    """A store over a quotient of the aid and income versions, whose records the reader holds
    and whose title refs the graph names; its first `failing_batches` input reads fail."""
    steps: list[dict[str, object]] = [
        _RETRIEVE_AID,
        _RETRIEVE_INCOME,
        {
            "id": "s3",
            "op": "quotient",
            "inputs": {"numerator": "s1", "denominator": "s2"},
            "stat_var": "ops_quotient",
        },
    ]
    store, frame_ref, graph, reader = _computed_frame_store(
        rows=[["dcid:ops_quotient", "country/ETH", "2015", 1.0, None, "src_oecd_dac"]],
        steps=steps,
        graph_labels={"Amount_Oda": "Gross ODA", "GNI": "Gross national income"},
        clock=clock,
    )
    record = reader.record
    assert record is not None and record.computed is not None
    reader.record = record.model_copy(
        update={
            "computed": record.computed.model_copy(
                update={
                    "inputs": [
                        InputVersion(id=_AID_VERSION[0], content_version=_AID_VERSION[1]),
                        InputVersion(id=_INCOME_VERSION[0], content_version=_INCOME_VERSION[1]),
                    ]
                }
            )
        }
    )
    inputs = (
        _input_record(_AID_VERSION, variable="Amount_Oda", recipient="country/ETH"),
        _input_record(_INCOME_VERSION, variable="GNI", recipient="country/ETH"),
    )
    reader.inputs = {(held.id, held.content_version): held for held in inputs}
    reader.failing_batches = failing_batches
    return store, frame_ref, graph, reader, inputs


@pytest.mark.parametrize(
    ("cap", "label_read", "quotient_label"),
    [
        pytest.param(
            None,
            ("Amount_Oda", "GNI", "country/ETH", "src_oecd_dac"),
            "Gross ODA to Ethiopia per gross national income to Ethiopia",
            id="every-ref-labelled",
        ),
        pytest.param(
            2,
            ("Amount_Oda", "GNI"),
            "Gross ODA disbursements per gross national income",
            id="title-refs-count-under-the-cap",
        ),
    ],
)
def test_load_reads_a_computed_versions_inputs_in_one_batch_with_their_title_labels(
    monkeypatch, cap, label_read, quotient_label
):
    """The load that reads a computed version's record reads its inputs' records in one batch,
    and the refs their titles name lead the frame's one label read, within its cap. The
    variable a step writes is named from the inputs' titles, as the card's description names
    them; an input whose title today's words cannot compose keeps the label stored with its
    step. A cache hit reads neither again."""
    if cap is not None:
        monkeypatch.setattr(frame_store, "_LABEL_RESOLUTION_CAP", cap)
    store, frame_ref, graph, reader, inputs = _quotient_of_inputs_store()

    first = store.load(frame_ref=frame_ref)
    graph_reads = list(graph.node_arcs_many_calls)
    second = store.load(frame_ref=frame_ref)

    assert first.input_records == inputs
    assert reader.batches == [[_AID_VERSION, _INCOME_VERSION]]
    assert graph_reads[0] == label_read
    assert first.labels["Amount_Oda"] == "Gross ODA"
    assert first.labels["GNI"] == "Gross national income"
    assert first.labels["dcid:ops_quotient"] == quotient_label
    assert second is first
    assert reader.batches == [[_AID_VERSION, _INCOME_VERSION]]
    assert graph.node_arcs_many_calls == graph_reads


def test_a_failed_input_read_fails_the_load_and_the_next_load_reads_again():
    """A failed input read raises, so no frame loads naming its inputs by id; the next load
    reads the inputs and labels their titles, and the load after is a hit."""
    store, frame_ref, _, reader, inputs = _quotient_of_inputs_store(failing_batches=1)

    with pytest.raises(DatasetsStoreUnavailable):
        store.load(frame_ref=frame_ref)
    retried = store.load(frame_ref=frame_ref)
    cached = store.load(frame_ref=frame_ref)

    assert retried.input_records == inputs
    assert retried.labels["Amount_Oda"] == "Gross ODA"
    assert len(reader.batches) == 2
    assert cached is retried


def test_a_failed_input_read_on_a_cached_frame_fails_the_load_and_keeps_the_cache():
    """A failed input read on the re-read of a cached frame whose record is not final raises,
    spends no graph read and leaves the cached frame, which the next load reads again."""
    clock = _Clock()
    store, frame_ref, graph, reader, inputs = _quotient_of_inputs_store(clock=clock)
    assert reader.record is not None
    reader.record = reader.record.model_copy(update={"steps": None})

    first = store.load(frame_ref=frame_ref)
    graph_reads = list(graph.node_arcs_many_calls)
    clock.pass_recheck()
    reader.failing_batches = 1
    with pytest.raises(DatasetsStoreUnavailable):
        store.load(frame_ref=frame_ref)
    retried = store.load(frame_ref=frame_ref)

    assert first.input_records == inputs
    assert retried is first
    assert graph.node_arcs_many_calls == graph_reads
    assert len(reader.batches) == 3


@pytest.mark.parametrize("late", ["no-record", "steps-null"])
def test_an_input_not_final_yet_is_read_again(late: str):
    """An input whose record is not written yet, or a computed input whose record holds
    `steps: null`, leaves the frame's record not final, so a load past the recheck wait reads
    the inputs again and takes the late record; the load after is a hit."""
    clock = _Clock()
    store, frame_ref, _, reader, inputs = _quotient_of_inputs_store(clock=clock)
    aid, income = inputs
    unsettled = income.model_copy(
        update={
            "computed": ComputedFacts(
                inputs=[InputVersion(id=None, content_version="cv_" + "e" * 32)], composed=None
            ),
            "steps": None,
        }
    )
    reader.inputs = (
        {_AID_VERSION: aid}
        if late == "no-record"
        else {_AID_VERSION: aid, _INCOME_VERSION: unsettled}
    )

    first = store.load(frame_ref=frame_ref)
    reader.inputs = {_AID_VERSION: aid, _INCOME_VERSION: income}
    clock.pass_recheck()
    second = store.load(frame_ref=frame_ref)
    third = store.load(frame_ref=frame_ref)

    assert first.input_records == ((aid,) if late == "no-record" else (aid, unsettled))
    if late == "no-record":
        assert "GNI" not in first.labels
    assert second.input_records == inputs
    assert second.labels["GNI"] == "Gross national income"
    assert len(reader.batches) == 2
    assert third is second


_ZAMBIA_VERSION = ("spec_" + "5" * 32, "cv_" + "5" * 32)
# Growth of IMF WEO gross national saving in Zambia: the root reports 2015-2024 under a null
# method and its source's estimates continue it for 2025-2026, so the growth rows keep the
# root rows' methods.
_ZAMBIA_GROWTH_ROWS: list[list[object]] = [
    [
        "dcid:ops_growth",
        "country/ZMB",
        str(year),
        float(year - 2015),
        "Percent",
        "Estimated" if year >= 2025 else None,
        "provenance/IMFWorldEconomicOutlook",
    ]
    for year in range(2016, 2027)
]


# The continuation the Zambia growth version stores, inherited from its root: the null-method
# series of IMF WEO continued by its estimates.
_ZAMBIA_CONTINUATION = ContinuedSeries(
    method="Estimated", continues=None, provenance="provenance/IMFWorldEconomicOutlook"
)


def _zambia_growth_store(
    *, clock: _Clock, continuations: list[dict[str, object]] | None
) -> tuple[LiveFrameStore, FrameRef, _RecordReader]:
    store, frame_ref, _, reader = _computed_frame_store(
        rows=_ZAMBIA_GROWTH_ROWS,
        columns=[
            "stat_var",
            "observationAbout",
            "time_period",
            "value",
            "unit",
            "measurementMethod",
            "provenance",
        ],
        steps=[
            {"id": "s1", "op": "retrieve", "input": _version_input("Gross saving", "5")},
            {"id": "s2", "op": "growth", "input": "s1", "stat_var": "ops_growth"},
        ],
        clock=clock,
        continuations=continuations,
    )
    record = reader.record
    assert record is not None and record.computed is not None
    reader.record = record.model_copy(
        update={
            "computed": record.computed.model_copy(
                update={
                    "inputs": [
                        InputVersion(id=_ZAMBIA_VERSION[0], content_version=_ZAMBIA_VERSION[1])
                    ]
                }
            )
        }
    )
    return store, frame_ref, reader


def _zambia_root(params: object) -> DescriptionRecord:
    """The root's record, holding a `continue_series` step with `params`, built unvalidated
    as a record stored before the publish door checked them."""
    root = _input_record(_ZAMBIA_VERSION, variable="GrossSaving", recipient="country/ZMB")
    step = Step.model_validate(
        {"id": "s1", "op": "continue_series", "policy": "continue_with_source_estimates"}
    )
    return root.model_copy(update={"steps": [step.model_copy(update={"params": params})]})


@pytest.mark.parametrize(
    "root_params",
    [
        pytest.param({"method": "Estimated", "periods": ["2025", "2026"]}, id="well-formed"),
        pytest.param({}, id="params-without-method"),
        pytest.param({"method": []}, id="a-method-that-is-no-string"),
    ],
)
def test_a_derived_frame_over_a_continued_series_groups_it_as_its_root_does(
    root_params: object,
) -> None:
    """A derived frame keeps its root's method cells and stores the continuation it inherited,
    so it holds one series from its first load, the view leaves `measurementMethod` not
    plural, and an unbound select draws every year. The root's record arriving later, whatever
    its `continue_series` step holds, changes nothing. Issue: the Zambia chart that drew only
    2025-2026."""
    clock = _Clock()
    store, frame_ref, reader = _zambia_growth_store(
        clock=clock, continuations=[_ZAMBIA_CONTINUATION.as_entry()]
    )
    select = [{"bindings": [{"dimension": "observationAbout", "values": ["country/ZMB"]}]}]

    first = store.load(frame_ref=frame_ref)
    reader.inputs = {_ZAMBIA_VERSION: _zambia_root(root_params)}
    clock.pass_recheck()
    after = store.load(frame_ref=frame_ref)

    for frame in (first, after):
        assert frame.continuation == Continuation(series=(_ZAMBIA_CONTINUATION,))
        assert frame.structural_view.series_count == 1
        (method,) = (
            one for one in frame.structural_view.dimensions if one.name == "measurementMethod"
        )
        assert (method.values, method.plural) == (("Estimated",), False)
        (series,) = resolve_selectors(resolved_frame=frame, selectors=select, window=None)
        assert [row["time_period"] for row in series.rows] == [
            str(year) for year in range(2016, 2027)
        ]
    assert after.input_records == (reader.inputs[_ZAMBIA_VERSION],)


@pytest.mark.parametrize(
    ("stored", "steps_late"),
    [
        # A version published without steps stores the continuations its lineage gives, here
        # none; the publish filling its steps replaces them with its root's.
        pytest.param([], True, id="late-steps"),
        # A version recorded before versions stored continuations, whose record is final,
        # holds them as unknown until the backfill fills them.
        pytest.param(None, False, id="backfilled"),
    ],
)
def test_a_cached_frame_regroups_once_its_stored_continuations_arrive(
    stored: list[dict[str, object]] | None, steps_late: bool
) -> None:
    """A computed frame whose stored continuations are none yet or unknown loads ungrouped and
    is not final; once its root's continuation is stored, the next load past the recheck wait
    groups the cached frame as one series."""
    clock = _Clock()
    store, frame_ref, reader = _zambia_growth_store(clock=clock, continuations=stored)
    reader.inputs = {
        _ZAMBIA_VERSION: _input_record(_ZAMBIA_VERSION, variable="GrossSaving", recipient="ZMB")
    }
    filled = reader.record
    assert filled is not None
    if steps_late:
        reader.record = filled.model_copy(update={"steps": None})
    snapshots = cast("_FakeSnapshotReader", store._snapshots)
    key = (frame_ref.spec_id, frame_ref.content_version)

    first = store.load(frame_ref=frame_ref)
    version = snapshots.versions[key]
    assert isinstance(version.facts, FrameLineage)
    snapshots.versions[key] = dataclasses.replace(
        version,
        facts=dataclasses.replace(version.facts, continuations=[_ZAMBIA_CONTINUATION.as_entry()]),
    )
    reader.record = filled
    clock.pass_recheck()
    after = store.load(frame_ref=frame_ref)

    assert (first.continuation, first.structural_view.series_count) == (None, 2)
    assert after.continuation == Continuation(series=(_ZAMBIA_CONTINUATION,))
    assert after.structural_view.series_count == 1


@pytest.mark.parametrize(
    ("raised", "failure"),
    [
        pytest.param(MemoryError(), FrameUnavailable, id="out-of-memory"),
        pytest.param(TypeError("exemplar"), TypeError, id="defect"),
    ],
)
def test_a_cached_frame_regroups_under_the_read_of_its_stored_cells(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, failure: type[Exception]
) -> None:
    clock = _Clock()
    store, frame_ref, _reader = _zambia_growth_store(clock=clock, continuations=None)
    snapshots = cast("_FakeSnapshotReader", store._snapshots)
    key = (frame_ref.spec_id, frame_ref.content_version)
    store.load(frame_ref=frame_ref)
    version = snapshots.versions[key]
    assert isinstance(version.facts, FrameLineage)
    snapshots.versions[key] = dataclasses.replace(
        version,
        facts=dataclasses.replace(version.facts, continuations=[_ZAMBIA_CONTINUATION.as_entry()]),
    )
    clock.pass_recheck()
    monkeypatch.setattr(frame_store, "compute_structural_view", _raises(raised))

    with pytest.raises(failure) as excinfo:
        store.load(frame_ref=frame_ref)

    (location, *_) = snapshots.locations[key]
    if failure is FrameUnavailable:
        assert excinfo.value.reason == "io"
    else:
        assert _logged_traceback_names(excinfo.value, location.artifact_digest)


@pytest.mark.parametrize(
    ("continuations", "series_count"),
    [
        pytest.param(None, 2, id="recorded-before-continuations"),
        pytest.param([], 2, id="nothing-continues"),
        pytest.param(
            [
                ContinuedSeries(
                    method="Estimated", continues=None, provenance="provenance/OtherSource"
                ).as_entry()
            ],
            2,
            id="another-sources-continuation",
        ),
        pytest.param(
            [ContinuedSeries(method="Estimated", continues="Reported", provenance=None).as_entry()],
            2,
            id="a-continuation-of-another-method",
        ),
        pytest.param([_ZAMBIA_CONTINUATION.as_entry()], 1, id="its-roots-continuation"),
    ],
)
def test_a_derived_frame_groups_only_the_series_its_stored_continuation_names(
    continuations: list[dict[str, object]] | None, series_count: int
) -> None:
    """A stored continuation joins the estimates only to the series of the method and source
    it continued in its root; a version storing none splits on method."""
    store, frame_ref, _ = _zambia_growth_store(clock=_Clock(), continuations=continuations)

    frame = store.load(frame_ref=frame_ref)

    assert frame.structural_view.series_count == series_count


def test_load_names_each_variable_of_a_join_of_two_quotients_from_the_step_that_wrote_it():
    """Each quotient's variable takes the name of that quotient's output, and the graph read
    leaves both out."""
    rows: list[list[object]] = [
        [stat_var, "country/ETH", "2015", 1.0, None, "src_oecd_dac"]
        for stat_var in ("dcid:ops_per_income", "dcid:ops_per_person")
    ]
    steps: list[dict[str, object]] = [
        _RETRIEVE_AID,
        _RETRIEVE_INCOME,
        {
            "id": "s3",
            "op": "quotient",
            "inputs": {"numerator": "s1", "denominator": "s2"},
            "stat_var": "ops_per_income",
        },
        _RETRIEVE_POPULATION,
        {
            "id": "s5",
            "op": "quotient",
            "inputs": {"numerator": "s1", "denominator": "s4"},
            "stat_var": "ops_per_person",
        },
        {"id": "s6", "op": "join", "inputs": {"left": "s3", "right": "s5"}},
    ]
    store, frame_ref, graph, _ = _computed_frame_store(rows=rows, steps=steps)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.labels == {
        "dcid:ops_per_income": "Gross ODA disbursements per gross national income",
        "dcid:ops_per_person": "Gross ODA disbursements per population",
        "country/ETH": "Ethiopia",
    }
    assert graph.node_arcs_many_calls[0] == ("country/ETH", "src_oecd_dac")


def test_load_names_a_comparison_of_one_variable_by_the_entities_its_sides_hold():
    """A compare frame holds no entity column, so the label read takes the entities from the
    step's sides, and the variable reads in the order compare subtracts."""
    rows: list[list[object]] = [["dcid:ops_difference", "2015", 1.0, None, "src_oecd_dac"]]
    steps: list[dict[str, object]] = [
        _RETRIEVE_AID,
        {**_RETRIEVE_AID, "id": "s2"},
        {
            "id": "s3",
            "op": "compare",
            "inputs": {"left": "s1", "right": "s2"},
            "sides": {
                "left": {"donor": "dcid:country/GBR"},
                "right": {"donor": "dcid:country/FRA"},
            },
            "stat_var": "ops_difference",
        },
    ]
    store, frame_ref, _, _ = _computed_frame_store(
        rows=rows,
        columns=["stat_var", "time_period", "value", "unit", "provenance"],
        steps=steps,
        graph_labels={"country/GBR": "United Kingdom", "country/FRA": "France"},
    )

    assert store.load(frame_ref=frame_ref).labels["dcid:ops_difference"] == (
        "Gross ODA disbursements, France minus United Kingdom"
    )


_STEPS_WITHOUT_STAT_VAR: list[dict[str, object]] = [
    _RETRIEVE_AID,
    _RETRIEVE_INCOME,
    {"id": "s3", "op": "quotient", "inputs": {"numerator": "s1", "denominator": "s2"}},
]


@pytest.mark.parametrize(
    ("cap", "graph_refs", "expected_labels"),
    [
        pytest.param(
            None,
            ("Count_Person", "country/ETH", "src_oecd_dac", "USDollar"),
            {
                "dcid:Count_Person": "Total population",
                f"dcid:{_FAMILY_ON_A_SIDE}": "Education",
                "country/ETH": "Ethiopia",
                "dcid:ops_sum": _SUM_LABEL,
                "dcid:ops_quotient": _QUOTIENT_LABEL,
            },
            id="every-side-labelled",
        ),
        pytest.param(
            2,
            ("Count_Person", "USDollar"),
            {
                "dcid:Count_Person": "Total population",
                f"dcid:{_FAMILY_ON_A_SIDE}": "Education",
                "dcid:ops_sum": _SUM_LABEL,
                "dcid:ops_quotient": _QUOTIENT_LABEL,
            },
            id="side-refs-precede-column-refs-under-the-cap",
        ),
    ],
)
@pytest.mark.parametrize(
    "steps",
    [None, _STEPS_WITHOUT_STAT_VAR],
    ids=["no-steps", "steps-naming-no-variable"],
)
def test_load_words_a_variable_no_step_names_from_its_composed_entry(
    monkeypatch, cap, graph_refs, expected_labels, steps
):
    """A record whose steps name no variable, or that holds none, words each composed entry
    from its sides. The sides join the label read ahead of the column refs, a family ref on a
    side takes its stored definition, and a variable an entry names is left out of the read."""
    if cap is not None:
        monkeypatch.setattr(frame_store, "_LABEL_RESOLUTION_CAP", cap)
    store, frame_ref, graph, _ = _computed_frame_store(
        rows=_COMPOSED_ROWS, composed=[_SUM_ENTRY, _QUOTIENT_ENTRY], steps=steps
    )

    resolved = store.load(frame_ref=frame_ref)

    assert graph.node_arcs_many_calls[0] == graph_refs
    assert resolved.labels == expected_labels


_STORED_COMPOSED: list[object] = [
    {
        "stat_var": "dcid:ops_sum",
        "relation": "sum",
        "first": [f"dcid:{_FAMILY_ON_A_SIDE}"],
        "second": ["dcid:Count_Person"],
    },
    {
        "stat_var": "dcid:ops_quotient",
        "relation": "quotient",
        "first": ["dcid:ops_sum"],
        "second": ["dcid:Count_Person"],
    },
]


@pytest.mark.parametrize(
    ("stored_composed", "expected"),
    [
        pytest.param(
            _STORED_COMPOSED,
            {"dcid:ops_sum": _SUM_LABEL, "dcid:ops_quotient": _QUOTIENT_LABEL},
            id="stored-lineage-names-them",
        ),
        pytest.param([{"stat_var": "dcid:ops_sum"}], {}, id="unreadable-lineage-keeps-refs"),
    ],
)
def test_a_version_with_no_record_names_its_composed_variables_from_its_stored_lineage(
    stored_composed, expected
):
    """A computed version that holds no record yet, as before the backfill describes it or
    after its publish could not, words its composed variables from the list its stored lineage
    holds, as its record would. A list that does not read keeps the refs and loads."""
    store, frame_ref, _, reader = _computed_frame_store(
        rows=_COMPOSED_ROWS, stored_composed=stored_composed
    )
    reader.record = None

    labels = store.load(frame_ref=frame_ref).labels

    assert {ref: labels[ref] for ref in ("dcid:ops_sum", "dcid:ops_quotient") if ref in labels} == (
        expected
    )


def test_load_words_a_composed_entry_from_bare_refs_when_the_graph_read_fails():
    """A failed graph read leaves the graph side as its bare dcid and the load uncached, so
    the next load reads again."""
    store, frame_ref, graph, _ = _computed_frame_store(
        rows=_COMPOSED_ROWS, composed=[_SUM_ENTRY], raise_for=frozenset({"Count_Person"})
    )

    labels = store.load(frame_ref=frame_ref).labels
    store.load(frame_ref=frame_ref)

    assert labels["dcid:ops_sum"] == "Education plus Count_Person"
    assert sum("Count_Person" in call for call in graph.node_arcs_many_calls) == 2


def test_load_leaves_a_variable_no_step_and_no_entry_names_with_its_ref():
    """An entry whose relation has no wording, and a variable the record does not name, keep
    their refs."""
    store, frame_ref, _, _ = _computed_frame_store(
        rows=_COMPOSED_ROWS,
        composed=[_SUM_ENTRY.model_copy(update={"relation": "product"})],
        steps=_STEPS_WITHOUT_STAT_VAR,
    )

    labels = store.load(frame_ref=frame_ref).labels

    assert "dcid:ops_sum" not in labels
    assert "dcid:ops_quotient" not in labels


_LONG_LABELS = {"LongA": "A" * 150, "LongB": "B" * 150, "GDP": "Nominal GDP", "Longer": "B" * 250}


@pytest.mark.parametrize(
    ("composed", "steps", "expected"),
    [
        pytest.param(
            [
                ComposedVariable(
                    variable="ops_a", relation="quotient", first=["LongA"], second=["LongB"]
                )
            ],
            None,
            "A" * 97 + " per " + "B" * 98,
            id="two-long-sides-take-half-each",
        ),
        pytest.param(
            [
                ComposedVariable(
                    variable="ops_a", relation="quotient", first=["GDP"], second=["Longer"]
                )
            ],
            None,
            "Nominal GDP per " + "B" * 184,
            id="a-short-side-keeps-its-text",
        ),
        pytest.param(
            None,
            [
                {"id": "s1", "op": "retrieve", "input": _version_input("A" * 150, "1")},
                {"id": "s2", "op": "retrieve", "input": _version_input("B" * 150, "2")},
                {
                    "id": "s3",
                    "op": "quotient",
                    "inputs": {"numerator": "s1", "denominator": "s2"},
                    "stat_var": "ops_a",
                },
            ],
            "A" * 97 + " per " + "B" * 98,
            id="a-step-name-of-two-long-sides-takes-half-each",
        ),
    ],
)
def test_load_keeps_both_sides_of_a_long_name_within_label_max(composed, steps, expected):
    """A name of two sides shortens the longer sides to fit `LABEL_MAX`, so both sides and the
    word between them stay."""
    rows: list[list[object]] = [["dcid:ops_a", "country/ETH", "2015", 1.0, None, None]]
    store, frame_ref, _, _ = _computed_frame_store(
        rows=rows, composed=composed, steps=steps, graph_labels=_LONG_LABELS
    )

    label = store.load(frame_ref=frame_ref).labels["dcid:ops_a"]

    assert label == expected
    assert len(label) <= LABEL_MAX


def test_load_names_a_variable_once_a_record_written_after_the_first_load_arrives(
    bind_catalogue,
):
    """A frame cached before its record existed takes the step names on the load that reads
    the record. That load alone re-reads the frame's labels, in one read stamped with the
    generation and build it runs under; a cache hit around it reads no labels."""
    rows: list[list[object]] = [["dcid:ops_per_income", "country/ETH", "2015", 1.0, None, None]]
    steps: list[dict[str, object]] = [
        *_STEPS_WITHOUT_STAT_VAR[:2],
        {**_STEPS_WITHOUT_STAT_VAR[2], "stat_var": "ops_per_income"},
    ]
    generation = FakeGeneration(token="gen-1")
    clock = _Clock()
    store, frame_ref, graph, reader = _computed_frame_store(
        rows=rows, steps=steps, generation=generation, clock=clock
    )
    record, reader.record = reader.record, None

    def label_reads() -> int:
        return sum(properties == ("name",) for _, properties, _ in graph.node_arcs_many_requests)

    bind_catalogue(FakeCatalogue(build_id="build-1"))
    first = store.load(frame_ref=frame_ref)
    before_record = label_reads()
    store.load(frame_ref=frame_ref)
    reads_without_record = label_reads() - before_record

    generation.advance("gen-2")
    bind_catalogue(FakeCatalogue(build_id="build-2"))
    reader.record = record
    clock.pass_recheck()
    second = store.load(frame_ref=frame_ref)
    reads_on_arrival = label_reads() - before_record

    cached = store.load(frame_ref=frame_ref)

    assert "dcid:ops_per_income" not in first.labels
    release_id = current_release().release_id
    assert (first.graph_generation, first.catalogue_build) == ("gen-1", f"build-1@{release_id}")
    assert second.labels["dcid:ops_per_income"] == (
        "Gross ODA disbursements per gross national income"
    )
    assert second.labels["country/ETH"] == "Ethiopia"
    assert (second.graph_generation, second.catalogue_build) == ("gen-2", f"build-2@{release_id}")
    assert (reads_without_record, reads_on_arrival) == (0, 1)
    assert label_reads() - before_record == 1
    assert cached == second


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
    """Record batched definition reads, the label read, and optionally raise for quarantined
    IDs or an unavailable registry. `served` holds the served ids bound to a posted id, newest
    first. `unavailable` fails the batched read, and `definition_error` is raised by the single
    reads, the served-id read the revocation scope makes among them."""

    definitions: dict[str, StoredDefinition] = field(default_factory=dict)
    served: dict[str, tuple[str, ...]] = field(default_factory=dict)
    quarantined: frozenset[str] = frozenset()
    unavailable: bool = False
    definition_error: Exception | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def read_definition(
        self, spec_id: str, *, schema_version: int | None = None
    ) -> StoredDefinition | None:
        if self.definition_error is not None:
            raise self.definition_error
        if spec_id in self.quarantined:
            raise IdentityQuarantined(subject_id=spec_id, id_kind="spec")
        return self.definitions.get(spec_id)

    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None:
        if self.definition_error is not None:
            raise self.definition_error
        served = self.served.get(spec_id, ())
        hit = self.quarantined.intersection((spec_id, *served))
        if hit:
            raise IdentityQuarantined(subject_id=sorted(hit)[0], id_kind="spec")
        return served if spec_id in self.definitions else None

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
    "text": citation_line(retrieved=_NOW.date(), sources=["OECD Creditor Reporting System"]),
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


_OPS_CITATION = {
    **_OECD_CITATION,
    "text": citation_line(
        retrieved=_NOW.date(), sources=["OECD Creditor Reporting System"], processed=True
    ),
}
"""`_OECD_CITATION` as a computed version cites it, as processed by the DFO."""


def test_load_cites_the_distinct_upstream_provenances_of_a_published_ops_frame():
    """Nine rows sharing one provenance ref give one cited source, beside the lineage body."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=_citing_graph())

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.manifest == {
        **version_document(_version(frame_ref)),
        "citation": _OPS_CITATION,
    }


def test_load_caches_an_ops_frame_s_citation_with_the_frame():
    """A cache hit serves the citation without reading the graph again."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)
    store.load(frame_ref=frame_ref)
    reads_after_first_load = graph.calls

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.manifest["citation"] == _OPS_CITATION
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
    assert second.manifest["citation"] == _OPS_CITATION


class _FailingOnceDescriptionReader:
    """Raises `raise_first()` on its first read and returns `record` after."""

    def __init__(self, record: DescriptionRecord, *, raise_first: Callable[[], None]) -> None:
        self.record = record
        self.raise_first = raise_first
        self.reads = 0

    def read_description(self, *, id: str, content_version: str) -> DescriptionRecord | None:
        self.reads += 1
        if self.reads == 1:
            self.raise_first()
        return self.record


def _store_unavailable() -> None:
    raise DatasetsStoreUnavailable("reading description failed", sqlstate="08006")


def _invalid_stored_record() -> None:
    DescriptionRecord.model_validate({"schema_version": 2})


def test_a_failed_record_read_fails_the_load_before_the_frame_is_fetched():
    """A store outage on the record read raises, so no frame loads without the record its
    version holds; the next load reads the record and fetches the frame once."""
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    record = DescriptionRecord.model_construct(computed=None, variables=None)
    reader = _FailingOnceDescriptionReader(record, raise_first=_store_unavailable)
    store = _store(snapshots=snapshots, artifacts=artifacts, description_reader=reader)

    with pytest.raises(DatasetsStoreUnavailable):
        store.load(frame_ref=frame_ref)
    second = store.load(frame_ref=frame_ref)

    assert second.description_record is record
    assert reader.reads == 2
    assert artifacts.calls.count(("get", Tier.working, digest)) == 1


def test_load_caches_a_frame_with_no_record_when_its_stored_record_fails_validation(caplog):
    """A stored record that fails validation is logged once as an error, and the frame loads
    and is cached with none, since reading it again returns the same record."""
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    record = DescriptionRecord.model_construct(computed=None, variables=None)
    reader = _FailingOnceDescriptionReader(record, raise_first=_invalid_stored_record)
    store = _store(snapshots=snapshots, artifacts=artifacts, description_reader=reader)

    with caplog.at_level(logging.WARNING, logger="presentation_store.frame_store"):
        first = store.load(frame_ref=frame_ref)
    second = store.load(frame_ref=frame_ref)

    [logged] = [r for r in caplog.records if r.name == "presentation_store.frame_store"]
    assert logged.levelno == logging.ERROR
    assert frame_ref.spec_id in logged.getMessage()
    assert frame_ref.content_version in logged.getMessage()
    assert first.description_record is None
    assert second is first
    assert reader.reads == 1


def test_a_failed_record_read_on_a_cached_unsettled_frame_fails_the_load_and_keeps_the_cache():
    """A computed record holding `steps: null` is read again once the recheck wait passes. A
    store outage on that read raises without a label read and leaves the cached frame, so the
    next load reads the record again and serves the cached record and composed labels."""
    clock = _Clock()
    store, frame_ref, graph, reader = _computed_frame_store(
        rows=_COMPOSED_ROWS, composed=[_SUM_ENTRY, _QUOTIENT_ENTRY], clock=clock
    )
    first = store.load(frame_ref=frame_ref)
    label_reads = len(graph.node_arcs_many_calls)
    record = reader.record
    assert record is not None
    failing = _FailingOnceDescriptionReader(record, raise_first=_store_unavailable)
    reader.read_description = failing.read_description

    clock.pass_recheck()
    with pytest.raises(DatasetsStoreUnavailable):
        store.load(frame_ref=frame_ref)
    retried = store.load(frame_ref=frame_ref)

    assert retried is first
    assert retried.labels["dcid:ops_quotient"] == _QUOTIENT_LABEL
    assert len(graph.node_arcs_many_calls) == label_reads
    assert failing.reads == 2


@pytest.mark.parametrize(
    ("raised", "cached"),
    [(SourceDocumentNotFound("src_oecd_dac"), True), (GraphReadError("transport"), False)],
    ids=["nothing-to-document", "documentation-read-failed"],
)
def test_load_caches_an_ops_frame_unless_a_documentation_read_failed(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, cached: bool
):
    """A source the graph documents nothing for is a lasting answer and caches; a failed
    documentation read is read again on the first load past the retry wait."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    graph = _citing_graph()
    clock = _Clock()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, now=clock)

    def read(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr(provenance_doc, "read_provenance_doc", read)
    first = store.load(frame_ref=frame_ref)
    monkeypatch.undo()
    clock.pass_recheck()
    reads_after_first_load = graph.calls
    second = store.load(frame_ref=frame_ref)

    undocumented = json.loads(json.dumps(_OPS_CITATION))
    undocumented["sources"][0]["documentation"] = None
    assert first.manifest["citation"] == undocumented
    assert (graph.calls == reads_after_first_load) is cached
    assert second.manifest["citation"] == (undocumented if cached else _OPS_CITATION)


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
        version, facts=replace(version.facts, citation=thin)
    )
    return snapshots, artifacts, frame_ref, thin


_HEALED_THIN_CITATION = {
    **_OECD_CITATION,
    "sources": [{**_OECD_CITATION["sources"][0], "label": "Stored label"}],
}


def _documenting_graph() -> _FakeGraphReader:
    """`_citing_graph`, with the pipeline's last and next refresh dates on `src_oecd_dac`."""
    graph = _citing_graph()
    graph._arcs["src_oecd_dac"] |= {
        "lastDataRefreshDate": ("2026-10-08",),
        "nextDataRefreshDate": ("2026-12",),
    }
    return graph


@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(_thin_spec_fixture, id="dataset-version"),
        pytest.param(_ops_happy_path_fixture, id="computed-version"),
    ],
)
def test_load_carries_what_each_cited_source_s_document_says_and_caches_it(fixture):
    """The citation read documents each source once, and the frame carries the source's last
    download and next release with its cached citation."""
    snapshots, artifacts, frame_ref, _ = fixture()
    graph = _documenting_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    first = store.load(frame_ref=frame_ref)
    reads_after_first_load = graph.calls
    second = store.load(frame_ref=frame_ref)

    assert first.source_documents == {
        "src_oecd_dac": SourceDocumentFacts(
            publisher=None, last_downloaded="2026-10-08", next_release="2026-12"
        )
    }
    assert second is first
    assert graph.calls == reads_after_first_load


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
    assert stored.facts.citation == thin


def _full_spec_fixture() -> tuple[_FakeSnapshotReader, _FakeArtifactStore, FrameRef, str]:
    """A root frame whose stored citation sets every field a live read fills, and cites its one
    source at two processing levels."""
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    full = json.loads(json.dumps(_OECD_CITATION))
    source = full["sources"][0]
    source.update(
        license_url="https://example.org/licence",
        source={"ref": "src_oecd", "label": "OECD"},
        data_commons={"id": "dfo", "label": "DFO"},
    )
    full["sources"].append({**source, "processing_level": "derived"})
    version = _version(frame_ref)
    snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)] = replace(
        version, facts=replace(version.facts, citation=full)
    )
    return snapshots, artifacts, frame_ref, digest


def test_a_root_frame_s_full_citation_is_read_only_once_the_frame_holds_a_record():
    """A stored citation with every field set reads nothing while the frame holds no record.
    A record arriving on the cached frame re-runs the citation read, which documents each
    distinct source once, under the call's `citation` phase, so a described card words what
    its document says."""
    snapshots, artifacts, frame_ref, _ = _full_spec_fixture()
    graph = _documenting_graph()
    reader = _RecordReader(None)
    clock = _Clock()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        graph=graph,
        description_reader=reader,
        now=clock,
    )

    undescribed = store.load(frame_ref=frame_ref)
    reads_undescribed = list(_citation_requests(graph))
    reader.record = DescriptionRecord.model_construct(computed=None, variables=None)
    clock.pass_recheck()
    timings = PhaseTimings()
    with collecting(timings):
        described = store.load(frame_ref=frame_ref)

    assert reads_undescribed == []
    assert undescribed.source_documents == {}
    assert {dcids for dcids, _ in _citation_requests(graph)} == {("src_oecd_dac",)}
    assert "citation" in timings.snapshot()
    assert described.source_documents["src_oecd_dac"].next_release == "2026-12"


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
    """The labels answer but the documentation read fails: the frame caches with the citation
    that read left, and loads within the retry wait serve it with no citation read. Each load
    past the wait retries only the citation fill, fetching no artifact, until one answers and
    the frame serves from the cache."""
    snapshots, artifacts, frame_ref, _ = _thin_spec_fixture()
    graph = _citing_graph()
    clock = _Clock()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, now=clock)
    read_documentation = provenance_doc.read_provenance_doc
    attempts: list[str] = []

    def failing(*_args, **_kwargs):
        attempts.append("documentation")
        raise GraphReadError("transport")

    monkeypatch.setattr(provenance_doc, "read_provenance_doc", failing)
    first = store.load(frame_ref=frame_ref)
    graph_calls_after_first = graph.calls
    within = [store.load(frame_ref=frame_ref) for _ in range(3)]
    graph_calls_within = graph.calls - graph_calls_after_first
    clock.pass_recheck()
    second = store.load(frame_ref=frame_ref)
    failed_attempts = len(attempts)
    monkeypatch.setattr(provenance_doc, "read_provenance_doc", read_documentation)
    clock.pass_recheck()
    third = store.load(frame_ref=frame_ref)
    graph_calls_after_third = graph.calls
    fourth = store.load(frame_ref=frame_ref)

    assert all(load is first for load in within)
    assert graph_calls_within == 0
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
        "text": citation_line(
            retrieved=_NOW.date(),
            sources=["OECD Creditor Reporting System", "src_imf"],
            processed=True,
        ),
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


def test_load_reads_a_root_frame_s_capped_sources_and_keeps_the_rest_as_stored(monkeypatch):
    """A root frame's stored sources past the cap are never read, neither their citation nor
    their documentation, and stand as stored; the sources within it are filled."""
    import presentation_store.frame_store as frame_store_module

    monkeypatch.setattr(frame_store_module, "_CITATION_READ_CAP", 1)
    snapshots, artifacts, frame_ref, thin = _thin_spec_fixture()
    imf = {**thin["sources"][0], "ref": "src_imf", "label": "src_imf"}
    thin["sources"].append(imf)
    version = snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)]
    snapshots.versions[(frame_ref.spec_id, frame_ref.content_version)] = replace(
        version, facts=replace(version.facts, citation=thin)
    )
    read_documentation = provenance_doc.read_provenance_doc
    documented: list[str] = []

    def recording(graph, dcid, **kwargs):
        documented.append(dcid)
        return read_documentation(graph, dcid, **kwargs)

    monkeypatch.setattr(provenance_doc, "read_provenance_doc", recording)
    graph = _citing_graph()
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph)

    resolved = store.load(frame_ref=frame_ref)

    requested = {dcid for dcids, _ in _citation_requests(graph) for dcid in dcids}
    assert requested == {"src_oecd_dac"}
    assert documented == ["src_oecd_dac"]
    assert resolved.manifest["citation"]["sources"] == [
        *_HEALED_THIN_CITATION["sources"],
        imf,
    ]


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
    assert snapshots.calls == ["read_version_state"]
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


@pytest.mark.parametrize(
    ("media_type", "rendering"),
    [
        ("text/csv", {"format": "csv"}),
        (CANONICAL_FRAME_MEDIA_TYPE, {"format": "parquet", "bundle": False, "layout": "frame"}),
    ],
    ids=["csv", "requested-parquet"],
)
def test_load_refuses_a_version_holding_no_stored_frame(
    media_type: str, rendering: dict[str, object]
):
    """A requested Parquet file shares the canonical frame's media type and is no frame."""
    frame_ref = malaria_ethiopia_frame_ref()
    snapshots = _FakeSnapshotReader(
        versions={(frame_ref.spec_id, frame_ref.content_version): _version(frame_ref)},
        locations={
            (frame_ref.spec_id, frame_ref.content_version): [
                _location(
                    digest="sha256:requested-only",
                    tier=Tier.working.value,
                    media_type=media_type,
                    rendering_key="requested",
                    rendering=rendering,
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
        allowances=None,
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
    store._cache_put(
        cache_key,
        _CachedFrame(
            size_bytes=1_000,
            resolved=resolved,
            citation_retry_at=None,
            anchor_final=True,
            description_recheck_at=None,
            words_read_at=_NOW,
        ),
        expected=store._cache[cache_key],
    )

    assert store._cache_bytes == sum(entry.size_bytes for entry in store._cache.values())


# -- load(): a derived frame's rolling anchor -------------------------------


class _FakeRecipeReader:
    """Holds recipes by frame pair and counts reads."""

    def __init__(self) -> None:
        self.recipes: dict[tuple[str, str], RecipeRecord] = {}
        self.reads = 0

    def put(
        self,
        frame_ref: FrameRef,
        *,
        anchor: dict[str, object] | None,
        roots: list[dict[str, object]] | None = None,
    ) -> None:
        self.recipes[(frame_ref.spec_id, frame_ref.content_version)] = RecipeRecord(
            spec_id=frame_ref.spec_id,
            content_version=frame_ref.content_version,
            normal_form_hash="rnf_" + "1" * 32,
            replay_key="rpk_" + "2" * 32,
            body={"schema_version": 1, "tree": {}, "roots": roots or [], "anchor": anchor},
        )

    def read_recipe(self, *, spec_id: str, content_version: str) -> RecipeRecord | None:
        self.reads += 1
        return self.recipes.get((spec_id, content_version))


_CHAD = {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/TCD"}
_MALI = {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/MLI"}
_ANCHOR = {
    "period": "2024",
    "rule": "half1",
    "left_out": [{"root": 0, "coordinates": _CHAD}, {"root": 1, "coordinates": _MALI}],
}


def test_load_attaches_a_derived_frame_s_recipe_anchor_and_labels_its_left_out_series():
    """Chad and Mali are in no row of the frame. The graph names Chad and holds no Mali."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    recipes.put(frame_ref, anchor=_ANCHOR)
    graph = _FakeGraphReader(labels={"country/ETH": "Ethiopia", "country/TCD": "Chad"})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, recipe_reader=recipes)

    resolved = store.load(frame_ref=frame_ref)

    assert resolved.anchor == RollingAnchor(
        period="2024",
        rule="half1",
        left_out=(
            LeftOutSeries(root=0, coordinates=_CHAD, label="Chad"),
            LeftOutSeries(root=1, coordinates=_MALI, label="country/MLI"),
        ),
    )
    assert "country/TCD" not in resolved.labels


def test_load_keeps_a_label_the_recipe_states_for_a_left_out_series():
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    stated = {"root": 0, "coordinates": _CHAD, "label": "Tchad"}
    recipes.put(frame_ref, anchor={**_ANCHOR, "left_out": [stated]})
    graph = _FakeGraphReader(labels={"country/TCD": "Chad"})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, recipe_reader=recipes)

    resolved = store.load(frame_ref=frame_ref)

    assert [one.label for one in resolved.anchor.left_out] == ["Tchad"]


@pytest.mark.parametrize(
    ("left_out", "label"),
    [
        pytest.param({"root": 0, "coordinates": _CHAD, "label": None}, "Chad", id="null-label"),
        pytest.param(
            {"root": 0, "coordinates": {"stat_var": "dcid:MalariaGrantDisbursements"}},
            "dcid:MalariaGrantDisbursements",
            id="stat-var-alone",
        ),
        pytest.param({"root": 0, "coordinates": {}}, "", id="empty-coordinates"),
        pytest.param(
            {"root": 0, "coordinates": {"unit": "USD", "provenance": "src_oecd"}},
            "unit=USD, provenance=src_oecd",
            id="coordinates-naming-no-series",
        ),
    ],
)
def test_load_labels_every_stored_left_out_series_whatever_its_coordinates_hold(
    left_out: dict[str, object], label: str
):
    """The recipe store keeps a pair's first recipe, so a load labels what is stored. The
    write routes refuse the last two shapes."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    recipes.put(frame_ref, anchor={**_ANCHOR, "left_out": [left_out]})
    graph = _FakeGraphReader(labels={"country/TCD": "Chad"})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, recipe_reader=recipes)

    resolved = store.load(frame_ref=frame_ref)

    assert [one.label for one in resolved.anchor.left_out] == [label]


def test_load_caches_the_anchor_with_the_frame():
    """A pair keeps its first recipe, so a cache hit reads neither the recipe nor the graph."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    recipes.put(frame_ref, anchor=_ANCHOR)
    graph = _FakeGraphReader(labels={"country/TCD": "Chad"})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, recipe_reader=recipes)
    first = store.load(frame_ref=frame_ref)
    recipe_reads, graph_reads = recipes.reads, graph.calls

    second = store.load(frame_ref=frame_ref)

    assert second.anchor == first.anchor
    assert (recipes.reads, graph.calls) == (recipe_reads, graph_reads)


def test_load_reads_the_left_out_labels_again_after_a_failed_graph_read():
    """The frame's own refs are labelled by the first batch, and the anchor's by a second
    batch that fails, so the load serves raw refs and is not cached."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    recipes.put(frame_ref, anchor=_ANCHOR)
    graph = _FakeGraphReader(labels={"country/TCD": "Chad"}, raise_for=frozenset({"country/TCD"}))
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, recipe_reader=recipes)

    failed = store.load(frame_ref=frame_ref)
    graph.raise_for = frozenset()
    retried = store.load(frame_ref=frame_ref)

    assert [one.label for one in failed.anchor.left_out] == ["country/TCD", "country/MLI"]
    assert [one.label for one in retried.anchor.left_out] == ["Chad", "country/MLI"]


@pytest.mark.parametrize(
    ("anchor_recorded", "replay_key", "rendering"),
    [
        (True, "rpk_" + "2" * 32, {"normal_form_hash": "rnf_" + "1" * 32}),
        (False, None, None),
    ],
    ids=["null-anchor", "no-recipe"],
)
def test_a_derived_frame_without_an_anchor_loads_as_it_does_with_no_recipe_reader(
    anchor_recorded, replay_key, rendering
):
    """A recipe with a null anchor still names the frame's replay key and its rendering."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    if anchor_recorded:
        recipes.put(frame_ref, anchor=None)
    plain = _store(snapshots=snapshots, artifacts=artifacts, graph=_citing_graph())
    store = _store(
        snapshots=snapshots, artifacts=artifacts, graph=_citing_graph(), recipe_reader=recipes
    )

    assert store.load(frame_ref=frame_ref) == replace(
        plain.load(frame_ref=frame_ref), replay_key=replay_key, rendering=rendering
    )


def test_a_frame_cached_before_its_recipe_was_written_gains_the_anchor_without_a_refetch():
    """A recipe can be written after its frame is published. The cached bytes are kept."""
    snapshots, artifacts, frame_ref, digest = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    store = _store(snapshots=snapshots, artifacts=artifacts, recipe_reader=recipes)
    before = store.load(frame_ref=frame_ref)
    fetches = artifacts.calls.count(("get", Tier.working, digest))

    recipes.put(frame_ref, anchor={**_ANCHOR, "left_out": []})
    after = store.load(frame_ref=frame_ref)

    assert (before.anchor, after.anchor) == (None, RollingAnchor(period="2024", rule="half1"))
    assert artifacts.calls.count(("get", Tier.working, digest)) == fetches == 1


_WDI_SOURCE = {
    **_OECD_CITATION["sources"][0],
    "ref": "dc/base/WorldDevelopmentIndicators",
    "label": "World Development Indicators",
    "url": "https://example.org/wdi",
}


def _derived_over_cited_roots() -> tuple[
    _FakeSnapshotReader, _FakeArtifactStore, FrameRef, _FakeRecipeReader
]:
    """`_derived_over_a_root`, whose root version cites the frame's `src_oecd_dac` with its
    licence type unset and a WDI source no row of the frame names, and a recipe reader that
    holds no recipe yet."""
    snapshots, artifacts, frame_ref, _, root_ref, _ = _derived_over_a_root()
    oecd = {**_OECD_CITATION["sources"][0], "label": "Stored label", "license_type": None}
    root = snapshots.versions[(root_ref.spec_id, root_ref.content_version)]
    snapshots.versions[(root_ref.spec_id, root_ref.content_version)] = replace(
        root,
        facts=replace(root.facts, citation={"text": "Root text.", "sources": [oecd, _WDI_SOURCE]}),
    )
    return snapshots, artifacts, frame_ref, _FakeRecipeReader()


def _put_recipe_over_the_root(recipes: _FakeRecipeReader, frame_ref: FrameRef) -> None:
    root_ref = malaria_ethiopia_frame_ref()
    pair = {"spec_id": root_ref.spec_id, "content_version": root_ref.content_version}
    recipes.put(frame_ref, anchor=None, roots=[{"dataset": pair, "options": {}}])


def _cited(manifest: Mapping[str, object]) -> list[tuple[str, str, str | None]]:
    citation = cast("Mapping[str, Any]", manifest["citation"])
    return [
        (source["ref"], source["label"], source["license_type"]) for source in citation["sources"]
    ]


def test_load_cites_a_derived_frame_with_its_recipe_roots_sources_filled_from_the_graph():
    """The frame cites its root's two sources, as the derived manifest does, the root's WDI
    source among them though no row names it; the stored label stands and the empty licence
    type is filled from the graph."""
    snapshots, artifacts, frame_ref, recipes = _derived_over_cited_roots()
    _put_recipe_over_the_root(recipes, frame_ref)
    store = _store(
        snapshots=snapshots, artifacts=artifacts, graph=_citing_graph(), recipe_reader=recipes
    )

    resolved = store.load(frame_ref=frame_ref)

    assert _cited(resolved.manifest) == [
        ("src_oecd_dac", "Stored label", "license/cc_by_4"),
        (_WDI_SOURCE["ref"], "World Development Indicators", "license/cc_by_4"),
    ]
    assert cast("Mapping[str, Any]", resolved.manifest["citation"])["text"] == "Root text."


def test_a_derived_frame_cached_before_its_recipe_was_written_cites_its_roots_once_it_is():
    """Before its recipe, the frame cites its provenance refs from the graph; the load that
    first reads the recipe cites the roots' sources instead."""
    snapshots, artifacts, frame_ref, recipes = _derived_over_cited_roots()
    store = _store(
        snapshots=snapshots, artifacts=artifacts, graph=_citing_graph(), recipe_reader=recipes
    )
    before = store.load(frame_ref=frame_ref)

    _put_recipe_over_the_root(recipes, frame_ref)
    after = store.load(frame_ref=frame_ref)

    assert _cited(before.manifest) == [
        ("src_oecd_dac", "OECD Creditor Reporting System", "license/cc_by_4")
    ]
    assert [ref for ref, _, _ in _cited(after.manifest)] == ["src_oecd_dac", _WDI_SOURCE["ref"]]


def test_a_rolling_card_on_a_derived_frame_is_created_at_the_recipe_s_anchor():
    """The frame's newest period is 2024 and its recipe anchors `latest` at 2023, so the create
    reads the anchor through `load`: the card's period is 2023 and it names Chad as left out.
    The select names only the StatVar, so it admits Chad's series. Its provenance names the
    recipe's replay key as `execution_ref`."""
    snapshots, artifacts, frame_ref, _ = _ops_happy_path_fixture()
    recipes = _FakeRecipeReader()
    recipes.put(
        frame_ref,
        anchor={"period": "2023", "rule": "half1", "left_out": [{"root": 0, "coordinates": _CHAD}]},
    )
    graph = _FakeGraphReader(labels={"country/ETH": "Ethiopia", "country/TCD": "Chad"})
    store = _store(snapshots=snapshots, artifacts=artifacts, graph=graph, recipe_reader=recipes)

    created = write_component(
        intent={
            "type": "stat_card",
            "frame_ref": {
                "spec_id": frame_ref.spec_id,
                "content_version": frame_ref.content_version,
            },
            "select": {"stat_var": "dcid:MalariaGrantDisbursements"},
            "window": {
                "start": {"anchor": "latest", "offset": -1},
                "end": {"anchor": "latest", "offset": 0},
            },
        },
        frame_store=store,
        revision_store=FakeRevisionStore(),
        release_registry=FakeReleaseRegistry(),
    )

    payload = created.resolved_payload
    assert payload["component"]["period"] == "2023"
    assert payload["provenance"]["execution_ref"] == "rpk_" + "2" * 32
    assert payload["provenance"]["warnings"][0] == {
        "kind": "coverage_gap",
        "detail": "1 series has no 2023 value: Chad",
    }


def test_a_dataset_frame_never_reads_a_recipe():
    snapshots, artifacts, frame_ref, _ = _happy_path_fixture()
    recipes = _FakeRecipeReader()
    store = _store(snapshots=snapshots, artifacts=artifacts, recipe_reader=recipes)

    resolved = store.load(frame_ref=frame_ref)

    assert (resolved.anchor, recipes.reads) == (None, 0)


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


def _derived_over_a_root() -> tuple[
    _FakeSnapshotReader, _FakeArtifactStore, FrameRef, str, FrameRef, str
]:
    """A published `ops_` frame and the dataset frame its recipe's one root names, each with
    its own working-tier artifact."""
    snapshots, artifacts, frame_ref, digest = _ops_happy_path_fixture()
    root_ref = malaria_ethiopia_frame_ref()
    root_digest = artifacts.seed(tier=Tier.working, data=_frame_bytes() + b"\n")
    snapshots.versions[(root_ref.spec_id, root_ref.content_version)] = _version(root_ref)
    snapshots.locations[(root_ref.spec_id, root_ref.content_version)] = [
        _location(digest=root_digest, tier=Tier.working.value)
    ]
    return snapshots, artifacts, frame_ref, digest, root_ref, root_digest


def test_pin_of_a_derived_frame_pins_the_root_versions_its_recipe_names():
    snapshots, artifacts, frame_ref, digest, root_ref, root_digest = _derived_over_a_root()
    recipes = _FakeRecipeReader()
    recipes.put(
        frame_ref,
        anchor=None,
        roots=[
            {
                "frame": {"spec_id": "spec_analysis_side", "content_version": "cv_other"},
                "dataset": {
                    "spec_id": root_ref.spec_id,
                    "content_version": root_ref.content_version,
                },
            }
        ],
    )
    reference_writer = _FakeReferenceWriter()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        reference_writer=reference_writer,
        recipe_reader=recipes,
    )

    assert store.pin(frame_ref=frame_ref, component_id="cmp_test") is True

    assert {(Tier.pinned, digest), (Tier.pinned, root_digest)} <= set(artifacts.objects)
    assert {
        (key[2], key[3]) for key in reference_writer.records if key[0] == "presentation:cmp_test"
    } == {
        (frame_ref.spec_id, frame_ref.content_version),
        (root_ref.spec_id, root_ref.content_version),
    }


def _recipe_naming(root_ref: FrameRef) -> list[dict[str, object]]:
    return [
        {
            "frame": {"spec_id": "spec_analysis_side", "content_version": "cv_other"},
            "dataset": {"spec_id": root_ref.spec_id, "content_version": root_ref.content_version},
        }
    ]


@pytest.mark.parametrize(
    ("recipe_roots", "reason"),
    [
        (None, "recipe_unreadable"),
        ([{"frame": {"spec_id": "spec_x", "content_version": "cv_x"}}], "recipe_unreadable"),
        ([], "recipe_pending"),
    ],
    ids=["no-recipe-reader", "root-without-dataset", "no-recipe-yet"],
)
def test_pin_of_a_derived_frame_whose_roots_cannot_be_named_refuses_and_pins_nothing(
    recipe_roots, reason
):
    snapshots, artifacts, frame_ref, digest, _, _ = _derived_over_a_root()
    recipes = None
    if recipe_roots is not None:
        recipes = _FakeRecipeReader()
        if recipe_roots:
            recipes.put(frame_ref, anchor=None, roots=recipe_roots)
    reference_writer = _FakeReferenceWriter()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        reference_writer=reference_writer,
        recipe_reader=recipes,
    )

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert excinfo.value.reason == reason
    assert (Tier.pinned, digest) not in artifacts.objects
    assert reference_writer.records == {}


def test_pin_of_a_derived_frame_with_no_recipe_whose_bytes_are_pinned_still_refuses():
    """Another pair holding the same bytes pinned them, and no reference names this pair, so
    its recipe may still be on its way."""
    snapshots, artifacts, frame_ref, digest, _, _ = _derived_over_a_root()
    artifacts.seed(tier=Tier.pinned, data=artifacts.objects[(Tier.working, digest)])
    key = (frame_ref.spec_id, frame_ref.content_version)
    snapshots.locations[key] = [_location(digest=digest, tier=Tier.pinned.value)]
    reference_writer = _FakeReferenceWriter()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        reference_writer=reference_writer,
        recipe_reader=_FakeRecipeReader(),
    )

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert excinfo.value.reason == "recipe_pending"
    assert reference_writer.records == {}


class _RecipeWrittenAfterFirstRead(_FakeRecipeReader):
    """Answers no recipe on its first read and the stored one after, as a recipe write
    landing between two reads would."""

    def read_recipe(self, *, spec_id: str, content_version: str) -> RecipeRecord | None:
        first = self.reads == 0
        recipe = super().read_recipe(spec_id=spec_id, content_version=content_version)
        return None if first else recipe


def test_pin_of_a_referenced_derived_frame_whose_recipe_lands_meanwhile_pins_its_roots():
    """A reference names the frame, and its recipe is written after the first read, so the
    second read names the root."""
    snapshots, artifacts, frame_ref, digest, root_ref, root_digest = _derived_over_a_root()
    recipes = _RecipeWrittenAfterFirstRead()
    recipes.put(frame_ref, anchor=None, roots=_recipe_naming(root_ref))
    reference_writer = _FakeReferenceWriter()
    reference_writer.records[
        (
            "presentation:cmp_earlier",
            digest,
            frame_ref.spec_id,
            frame_ref.content_version,
            PRESENTATION_PIN_REFERENCE_KIND,
        )
    ] = _NOW
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        reference_writer=reference_writer,
        recipe_reader=recipes,
    )

    store.pin(frame_ref=frame_ref, component_id="cmp_test")

    assert (Tier.pinned, root_digest) in artifacts.objects


def _root_aged_out(snapshots: _FakeSnapshotReader, root_ref: FrameRef) -> None:
    key = (root_ref.spec_id, root_ref.content_version)
    snapshots.locations[key] = [
        dataclasses.replace(location, stored_at=_NOW - timedelta(days=31))
        for location in snapshots.locations[key]
    ]


def _root_revoked(snapshots: _FakeSnapshotReader, root_ref: FrameRef) -> None:
    snapshots.revocations[(root_ref.spec_id, root_ref.content_version)] = Revocation(
        reason="takedown", revoked_at=_NOW
    )


@pytest.mark.parametrize("lapse", [_root_aged_out, _root_revoked], ids=["expired", "revoked"])
def test_pin_of_a_derived_frame_pins_the_frame_when_a_root_is_no_longer_held(lapse):
    snapshots, artifacts, frame_ref, digest, root_ref, root_digest = _derived_over_a_root()
    lapse(snapshots, root_ref)
    recipes = _FakeRecipeReader()
    recipes.put(frame_ref, anchor=None, roots=_recipe_naming(root_ref))
    reference_writer = _FakeReferenceWriter()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        reference_writer=reference_writer,
        recipe_reader=recipes,
    )

    assert store.pin(frame_ref=frame_ref, component_id="cmp_test") is True

    assert (Tier.pinned, digest) in artifacts.objects
    assert (Tier.pinned, root_digest) not in artifacts.objects
    assert {(key[2], key[3]) for key in reference_writer.records} == {
        (frame_ref.spec_id, frame_ref.content_version)
    }


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


_OPEN_CALLER = Caller(kind="address", key="ip:" + "0" * 32)


def _ledger(**limits: int) -> LiveAllowanceLedger:
    """A ledger over a fresh store, 500 of each pin kind unless `limits` names it."""
    return LiveAllowanceLedger(
        store=FakeVersionRecordDatasetStore(),
        limits={
            kind: limits.get(kind.value, 500)
            for kind in (AllowanceKind.pin_budget, AllowanceKind.open_pin_budget, AllowanceKind.pin)
        },
    )


def _units(ledger: LiveAllowanceLedger) -> dict[tuple[str, str], int]:
    """Each `(key, kind)` the ledger holds a count for, with the count."""
    with ledger._store.transaction() as tx:
        rows = tx.fetch_all("SELECT caller_key, kind, count FROM datasets.caller_allowance")
    return {
        (cast(str, row["caller_key"]), cast(str, row["kind"])): cast(int, row["count"])
        for row in rows
    }


def _derived_store(ledger: LiveAllowanceLedger):
    """A store spending from `ledger` over a derived frame and its one root, both working-tier,
    with `_derived_over_a_root`'s fixtures."""
    snapshots, artifacts, frame_ref, digest, root_ref, root_digest = _derived_over_a_root()
    recipes = _FakeRecipeReader()
    recipes.put(frame_ref, anchor=None, roots=_recipe_naming(root_ref))
    store = _store(
        snapshots=snapshots, artifacts=artifacts, recipe_reader=recipes, allowances=ledger
    )
    return store, snapshots, artifacts, frame_ref, digest, root_ref, root_digest


def _set_of_units(caller: Caller | None, count: int) -> dict[tuple[str, str], int]:
    units = {(BUDGET_KEY, AllowanceKind.pin_budget.value): count}
    if caller is not None:
        units[(BUDGET_KEY, AllowanceKind.open_pin_budget.value)] = count
        units[(caller.key, AllowanceKind.pin.value)] = count
    return units


@pytest.mark.parametrize("caller", [_OPEN_CALLER, None], ids=["open-caller", "no-caller"])
@pytest.mark.parametrize("reads_pinned", [True, False], ids=["reads-pinned", "races-to-a-copy"])
def test_pin_spends_one_set_of_units_per_new_copy_and_none_for_a_frame_already_pinned(
    caller: Caller | None, reads_pinned: bool
):
    """The derived frame and its root are two new copies. A second pin spends nothing, whether
    it reads both pinned or still reads them working and finds the pinned bytes made."""
    ledger = _ledger()
    store, snapshots, _, frame_ref, _, root_ref, _ = _derived_store(ledger)

    store.pin(frame_ref=frame_ref, component_id="cmp_a", caller=caller)
    assert _units(ledger) == _set_of_units(caller, 2)
    if reads_pinned:
        for ref in (frame_ref, root_ref):
            key = (ref.spec_id, ref.content_version)
            snapshots.locations[key] = [
                dataclasses.replace(location, tier=Tier.pinned.value)
                for location in snapshots.locations[key]
            ]
    store.pin(frame_ref=frame_ref, component_id="cmp_b", caller=caller)

    assert _units(ledger) == _set_of_units(caller, 2)


@pytest.mark.parametrize(
    ("limit", "refusal"),
    [
        ("pin", AllowanceExhausted),
        ("open_pin_budget", OpenPinBudgetExhausted),
        ("pin_budget", PinBudgetExhausted),
    ],
)
def test_pin_refuses_at_the_root_a_spent_budget_stops_and_the_frames_copy_keeps_its_units(
    limit: str, refusal: type[Exception]
):
    ledger = _ledger(**{limit: 1})
    store, _, artifacts, frame_ref, digest, _, root_digest = _derived_store(ledger)

    with pytest.raises(refusal):
        store.pin(frame_ref=frame_ref, component_id="cmp_a", caller=_OPEN_CALLER)

    assert (Tier.pinned, digest) in artifacts.objects
    assert (Tier.pinned, root_digest) not in artifacts.objects
    assert _units(ledger) == _set_of_units(_OPEN_CALLER, 1)


def test_pin_keeps_the_units_of_a_copy_whose_reference_write_then_fails():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    ledger = _ledger()
    reference_writer = _FakeReferenceWriter(
        revoked={(frame_ref.spec_id, frame_ref.content_version)}
    )
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        reference_writer=reference_writer,
        allowances=ledger,
    )

    with pytest.raises(FrameUnavailable):
        store.pin(frame_ref=frame_ref, component_id="cmp_a", caller=_OPEN_CALLER)

    assert (Tier.pinned, digest) in artifacts.objects
    assert _units(ledger) == _set_of_units(_OPEN_CALLER, 1)


def test_pin_gives_back_the_units_of_a_copy_whose_working_bytes_are_gone():
    snapshots, artifacts, frame_ref, digest = _happy_path_fixture()
    del artifacts.objects[(Tier.working, digest)]
    ledger = _ledger()
    store = _store(snapshots=snapshots, artifacts=artifacts, allowances=ledger)

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_a", caller=_OPEN_CALLER)

    assert excinfo.value.reason == "expired"
    assert _units(ledger) == {}


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


def _posted_frame_fixture():
    """A frame recorded under a posted, non-canonical id and under its served id, with a spec
    reader registering the posted definition and binding the served id to it. Returns the
    snapshots, artifacts, the frame ref naming the posted id, the served id and the spec
    reader."""
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    for slot in identity["slots"]:
        if slot["property"] in ("crsTypeOfFlow", "dac2aAidType"):
            slot["binding"] = {"state": "unbound"}
    posted = recompute_identity(identity)
    served_id = recompute_identity(canonical_identity(get_index(), posted.identity)).spec_id
    posted_id = posted.spec_id
    assert served_id != posted_id
    spec_reader = _FakeSpecReader(
        definitions={
            posted_id: StoredDefinition(
                spec_id=posted_id,
                schema_version=1,
                identity=posted.identity,
                identity_bytes=b"",
                identity_digest="",
            )
        },
        served={posted_id: (served_id,)},
    )
    frame_ref = FrameRef(spec_id=posted_id, content_version="cv_" + "e" * 32)
    artifacts = _FakeArtifactStore()
    digest = artifacts.seed(tier=Tier.working, data=_frame_bytes())
    snapshots = _FakeSnapshotReader(
        versions={
            (subject_id, frame_ref.content_version): replace(
                _version(frame_ref), subject_id=subject_id
            )
            for subject_id in (served_id, posted_id)
        },
        locations={
            (subject_id, frame_ref.content_version): [
                _location(digest=digest, tier=Tier.working.value)
            ]
            for subject_id in (served_id, posted_id)
        },
    )
    return snapshots, artifacts, frame_ref, served_id, spec_reader


def _load_or_pin(store: LiveFrameStore, call: str, frame_ref: FrameRef) -> object:
    if call == "load":
        return store.load(frame_ref=frame_ref)
    return store.pin(frame_ref=frame_ref, component_id="cmp_test")


@pytest.mark.parametrize("revoked_under", ["served", "posted", None])
def test_a_frame_ref_naming_a_posted_id_honours_a_revocation_under_its_served_id(revoked_under):
    """A frame ref naming a posted id honours revocations under the posted and the served id,
    as a Datasets request naming it does, so a takedown under the served id stops a frame
    recorded under the posted id."""
    snapshots, artifacts, frame_ref, served_id, spec_reader = _posted_frame_fixture()
    if revoked_under is not None:
        subject_id = served_id if revoked_under == "served" else frame_ref.spec_id
        snapshots.revocations[(subject_id, frame_ref.content_version)] = Revocation(
            reason="takedown", revoked_at=_NOW
        )
    store = _store(snapshots=snapshots, artifacts=artifacts, spec_reader=spec_reader)

    if revoked_under is None:
        assert store.load(frame_ref=frame_ref) is not None
        return
    with pytest.raises(FrameUnavailable) as excinfo:
        store.load(frame_ref=frame_ref)
    assert excinfo.value.reason == "revoked"


@pytest.mark.parametrize(
    "error",
    [
        SpecRegistryUnavailable("spec registry pool borrow timed out"),
        SpecRegistryStoreError("spec registry read failed"),
    ],
    ids=["unavailable", "store-error"],
)
@pytest.mark.parametrize("call", ["load", "pin"])
def test_a_frame_read_refuses_when_the_registry_cannot_scope_its_revocation_check(call, error):
    """The served id comes from the registry. A registry that cannot answer fails the load and
    the pin rather than check revocation under the ref's own id alone."""
    snapshots, artifacts, frame_ref, _served_id, spec_reader = _posted_frame_fixture()
    spec_reader.definition_error = error
    references = _FakeReferenceWriter()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        spec_reader=spec_reader,
        reference_writer=references,
    )

    with pytest.raises(type(error)):
        _load_or_pin(store, call, frame_ref)
    assert references.records == {}


@pytest.mark.parametrize("quarantined", ["posted", "served"])
@pytest.mark.parametrize("call", ["load", "pin"])
def test_a_frame_ref_naming_a_quarantined_id_is_unavailable(call, quarantined):
    """A frame ref naming a quarantined id, or a posted id bound to a quarantined served id,
    refuses `quarantined` rather than load or pin."""
    snapshots, artifacts, frame_ref, served_id, spec_reader = _posted_frame_fixture()
    spec_reader.quarantined = frozenset(
        {frame_ref.spec_id if quarantined == "posted" else served_id}
    )
    references = _FakeReferenceWriter()
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        spec_reader=spec_reader,
        reference_writer=references,
    )

    with pytest.raises(FrameUnavailable) as excinfo:
        _load_or_pin(store, call, frame_ref)
    assert excinfo.value.reason == "quarantined"
    assert references.records == {}


def test_a_pin_of_a_posted_frame_rechecks_a_served_id_revocation_at_the_write():
    """A revocation under the served id that lands after the read refuses the pin's reference
    write, as the Datasets pin does."""
    snapshots, artifacts, frame_ref, served_id, spec_reader = _posted_frame_fixture()
    references = _FakeReferenceWriter(revoked={(served_id, frame_ref.content_version)})
    store = _store(
        snapshots=snapshots,
        artifacts=artifacts,
        spec_reader=spec_reader,
        reference_writer=references,
    )

    with pytest.raises(FrameUnavailable) as excinfo:
        store.pin(frame_ref=frame_ref, component_id="cmp_test")
    assert excinfo.value.reason == "revoked"
    assert references.records == {}
