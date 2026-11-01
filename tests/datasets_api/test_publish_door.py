"""Tests `publish_frame` and the publish door through shared SQLite-backed stores.
One `FakeVersionRecordDatasetStore` backs the snapshot reader, version writer, and artifact
inventory, so artifact-location joins exercise the same coupling as production. HTTP tests
cover refusal mapping through `make_client_for_door`.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jsonschema
import pyarrow.parquet as pq
import pytest
import rfc8785
from dfo_frame_grammar import frame_sha256
from pydantic import BaseModel, Field, ValidationError, field_validator

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import (
    FrameLineage,
    LiveSnapshotReader,
    SnapshotRevoked,
    select_servable_locations,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.doors.publish import (
    PublishBytesDiverge,
    PublishFrameInvalid,
    PublishLineageDiverge,
    PublishRecordMissing,
    build_publish_door,
    publish_frame,
)
from datasets_api.publish_models import (
    MAX_CELL_CHARS,
    MAX_INPUT_CONTENT_VERSIONS,
    MAX_ROLE_KEYS,
    MAX_ROW_WIDTH,
    MAX_ROWS,
    MAX_UNIT_ENTRIES,
    MAX_UNIT_STAT_VAR_CHARS,
    PublishFrameRequest,
)
from datasets_api.render import (
    CANONICAL_FRAME_RENDERING,
    STORED_FRAME_COPY_PASSED_OVER,
    frame_file_rendering,
    render_and_store_canonical_frame,
)
from datasets_api.store_errors import StoredFrameUnreadable
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, NullVersionRecordWriter, VersionRecord
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from materialise.errors import OverCeilingError
from materialise.formats import (
    CANONICAL_FRAME_MEDIA_TYPE,
    FRAME1_MEDIA_TYPE,
    canonical_frame_document,
    read_canonical_frame,
    read_canonical_frame_table,
    write_frame1,
)
from materialise.frame_build import build_frame
from tests.artifact_store.conftest import FakeGCSClient

from ._deps_doubles import fake_deps_cell
from ._frame_doubles import canonical_frame_digest, partial_frame1_document
from ._render_inputs import ENRICHMENT, PREPARED_AT, SITE
from ._snapshot_doubles import make_client_for_door
from ._version_facts import facts_from_body, golden_cases
from ._version_record_store import FakeVersionRecordDatasetStore

_SPEC_ID = "ops_abc123"
_CONTENT_VERSION = "ocv_abc123"
_ROLE_KEYS = ("recipient",)
_PRINCIPAL = "publisher-principal"
_AUTH_HEADERS = {"x-dfo-principal": _PRINCIPAL}


def _rows(value: float = 1.0) -> list[dict[str, object]]:
    return [
        {
            "stat_var": "dcid:Exemplar",
            "time_period": "2020",
            "value": value,
            "recipient": "country/KEN",
        }
    ]


def _distinct_grain_rows(count: int) -> list[dict[str, object]]:
    """Builds `count` rows with distinct years so `Frame.build` reaches the ceiling check."""
    return [
        {
            "stat_var": "dcid:Exemplar",
            "time_period": str(2000 + offset),
            "value": 1.0,
            "recipient": "country/KEN",
        }
        for offset in range(count)
    ]


def _publish_body(**overrides: object) -> dict[str, object]:
    """Builds a valid publish body with optional field overrides."""
    body: dict[str, object] = {
        "schema_version": 1,
        "spec_id": _SPEC_ID,
        "content_version": _CONTENT_VERSION,
        "role_keys": list(_ROLE_KEYS),
        "rows": _rows(),
        "input_content_versions": ["cv_00000000000000000000000000abc123"],
    }
    body.update(overrides)
    return body


def _revoke(
    store: FakeVersionRecordDatasetStore,
    *,
    subject_id: str,
    content_version: str,
    reason: str | None = None,
) -> None:
    """Inserts a revocation row into the shared store used by the snapshot reader."""
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, trace_id, revoked_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (subject_id, content_version, reason, None, datetime.now(UTC)),
        )


class _Deps:
    """Wires inventory, snapshot, and version stores over one SQLite-backed store."""

    def __init__(self) -> None:
        self.store = FakeVersionRecordDatasetStore()
        self.artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=self.store),
            client=FakeGCSClient(),
        )
        self.snapshot_reader = LiveSnapshotReader(store=self.store)
        self.version_writer = LiveVersionRecordWriter(store=self.store)

    def publish(
        self,
        *,
        spec_id: str = _SPEC_ID,
        content_version: str = _CONTENT_VERSION,
        role_keys=_ROLE_KEYS,
        rows=None,
        input_content_versions=("cv_00000000000000000000000000abc123",),
        ceiling: int | None = None,
        artifact_store=None,
    ):
        kwargs: dict[str, object] = {}
        if ceiling is not None:
            kwargs["ceiling"] = ceiling
        return publish_frame(
            spec_id=spec_id,
            content_version=content_version,
            role_keys=role_keys,
            rows=rows if rows is not None else _rows(),
            input_content_versions=input_content_versions,
            artifact_store=artifact_store if artifact_store is not None else self.artifact_store,
            snapshot_reader=self.snapshot_reader,
            version_writer=self.version_writer,
            **kwargs,
            file_enrichment=ENRICHMENT,
            site=SITE,
            prepared_at=PREPARED_AT,
        )


@pytest.fixture
def deps() -> _Deps:
    return _Deps()


def _canonical_frames(deps: _Deps):
    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    return [loc for loc in locations if loc.rendering == CANONICAL_FRAME_RENDERING]


def _frame_files(deps: _Deps):
    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    return [loc for loc in locations if loc.rendering == frame_file_rendering(SITE)]


def test_fresh_publish_records_version_row_and_its_parquet_canonical_frame(deps: _Deps) -> None:
    result = deps.publish()

    assert result.created is True
    assert result.spec_id == _SPEC_ID
    assert result.content_version == _CONTENT_VERSION

    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert stored.subject_kind == "spec"
    assert stored.facts == FrameLineage(
        input_content_versions=["cv_00000000000000000000000000abc123"], continuations=[]
    )

    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    assert sorted(
        ((loc.rendering, loc.media_type, loc.tier) for loc in locations), key=str
    ) == sorted(
        [
            (CANONICAL_FRAME_RENDERING, CANONICAL_FRAME_MEDIA_TYPE, "working"),
            (frame_file_rendering(SITE), CANONICAL_FRAME_MEDIA_TYPE, "working"),
        ],
        key=str,
    )
    # The result names the frame file, the flat file a ref names.
    assert [loc.artifact_digest for loc in _frame_files(deps)] == [result.artifact_digest]

    (canonical,) = _canonical_frames(deps)
    raw = deps.artifact_store.get(artifact_digest=canonical.artifact_digest, tier=Tier.working)
    assert read_canonical_frame(raw) == build_frame(role_keys=_ROLE_KEYS, rows=_rows())


def test_repeat_publish_with_identical_bytes_is_idempotent(deps: _Deps) -> None:
    first = deps.publish()
    second = deps.publish()

    assert first.created is True
    assert second.created is False
    assert second.artifact_digest == first.artifact_digest
    assert len(_canonical_frames(deps)) == 1


def _frame1_document(value: float) -> bytes:
    sink = _BytesSink()
    write_frame1((build_frame(role_keys=_ROLE_KEYS, rows=_rows(value)),), sink=sink)
    return sink.data


def _snappy_parquet_frame(value: float) -> bytes:
    """The frame's table in Parquet under another compression and row-group size than
    `write_canonical_frame` uses, as another PyArrow version may encode it."""
    buffer = io.BytesIO()
    table = build_frame(role_keys=_ROLE_KEYS, rows=_rows(value)).table
    pq.write_table(table, buffer, compression="snappy", row_group_size=1)
    return buffer.getvalue()


def _record_stored_pair(
    deps: _Deps,
    *,
    data: bytes,
    media_type: str,
    rendering: dict[str, object],
    frame_sha256: str | None = None,
) -> str:
    """Record the pair holding `data` as its only stored frame, with `frame_sha256` as the
    frame hash on its record, and return the frame's digest."""
    stored = deps.artifact_store.put(data, tier=Tier.working)
    deps.version_writer.record_version(
        VersionRecord(
            subject_id=_SPEC_ID,
            subject_kind="spec",
            content_version=_CONTENT_VERSION,
            schema_version=1,
            facts=FrameLineage(input_content_versions=["cv_00000000000000000000000000abc123"]),
            artifacts=(
                ArtifactRecord(
                    artifact_digest=stored.digest, rendering=rendering, media_type=media_type
                ),
            ),
            frame_sha256=frame_sha256,
        )
    )
    return stored.digest


class _BytesSink:
    def __init__(self) -> None:
        self.data = b""

    def write(self, data: bytes, /) -> int:
        self.data += bytes(data)
        return len(data)


@pytest.mark.parametrize(
    ("encode", "media_type", "rendering", "records_hash"),
    [
        (_frame1_document, FRAME1_MEDIA_TYPE, {"format": "frame1"}, False),
        (_snappy_parquet_frame, CANONICAL_FRAME_MEDIA_TYPE, dict(CANONICAL_FRAME_RENDERING), True),
        (_snappy_parquet_frame, CANONICAL_FRAME_MEDIA_TYPE, dict(CANONICAL_FRAME_RENDERING), False),
    ],
    ids=[
        "frame1-document",
        "canonical-frame-other-parquet-encoding",
        "canonical-frame-without-hash",
    ],
)
def test_repeat_publish_compares_the_stored_frame_not_its_encoding(
    deps: _Deps, encode, media_type, rendering, records_hash
) -> None:
    """A repeat publish matches a stored frame holding the same rows whatever bytes encode it,
    and refuses one holding other rows: by the frame hash on a record that holds one, else by
    the hash of the stored frame read (`render.stored_frame_sha256`). It links one canonical
    frame and its frame file, and answers the frame file's digest."""
    _record_stored_pair(
        deps,
        data=encode(1.0),
        media_type=media_type,
        rendering=rendering,
        frame_sha256=(
            frame_sha256(build_frame(role_keys=_ROLE_KEYS, rows=_rows(1.0)))
            if records_hash
            else None
        ),
    )

    repeat = deps.publish(rows=_rows(1.0))

    assert repeat.created is False
    assert len(_canonical_frames(deps)) == 1
    assert [loc.artifact_digest for loc in _frame_files(deps)] == [repeat.artifact_digest]
    with pytest.raises(PublishBytesDiverge):
        deps.publish(rows=_rows(2.0))


def _age_stored_frame(deps: _Deps) -> None:
    """Put every stored location of the pair's frame past working-tier retention, its objects
    still held, as they are until the sweep deletes them."""
    with deps.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (datetime.now(UTC) - timedelta(days=366),),
        )


class _ReadCountingArtifactStore:
    """`store`, counting each `get` that returns bytes."""

    def __init__(self, store: LiveArtifactStore) -> None:
        self._store = store
        self.reads = 0

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        data = self._store.get(artifact_digest=artifact_digest, tier=tier)
        self.reads += 1
        return data

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)


def test_a_hashless_pair_past_retention_matches_by_reading_its_held_frame(deps: _Deps) -> None:
    """A pair recording no frame hash, whose only link is a partial frame1 document past
    retention with its object still held, is named by the hash of the frame read from it, not
    by the document's digest."""
    _record_stored_pair(
        deps,
        data=partial_frame1_document(_frame1_document(1.0)),
        media_type=FRAME1_MEDIA_TYPE,
        rendering={"format": "frame1"},
    )
    _age_stored_frame(deps)
    counting = _ReadCountingArtifactStore(deps.artifact_store)

    repeat = deps.publish(rows=_rows(1.0), artifact_store=counting)

    assert repeat.created is False
    assert counting.reads > 0


@pytest.mark.parametrize("unreadable", ["corrupt-pinned-object", "undecodable-canonical-frame"])
def test_a_hashless_pair_passes_over_an_unreadable_copy_to_one_that_reads(
    deps: _Deps, unreadable: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A hashless pair's frame is read from the first linked copy that reads: past a servable
    pinned object that does not hash to its digest, or past a canonical frame that does not
    decode, to the frame1 document. Each copy passed over logs a WARNING naming it."""
    digest = _record_stored_pair(
        deps,
        data=partial_frame1_document(_frame1_document(1.0)),
        media_type=FRAME1_MEDIA_TYPE,
        rendering={"format": "frame1"},
    )
    if unreadable == "corrupt-pinned-object":
        deps.artifact_store.copy(digest, from_tier=Tier.working, to_tier=Tier.pinned)
        pinned = deps.artifact_store._client.buckets["pinned-bucket"].objects
        for name in pinned:
            pinned[name] = b"{not a frame1 document"
    else:
        undecodable = deps.artifact_store.put(b"not a parquet file", tier=Tier.working)
        deps.version_writer.record_artifact(
            subject_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            artifact=ArtifactRecord(
                artifact_digest=undecodable.digest,
                rendering=dict(CANONICAL_FRAME_RENDERING),
                media_type=CANONICAL_FRAME_MEDIA_TYPE,
            ),
        )

    with caplog.at_level(logging.WARNING, logger="datasets_api.render"):
        repeat = deps.publish(rows=_rows(1.0))

    assert repeat.created is False
    (passed_over,) = _passed_over(caplog)
    if unreadable == "corrupt-pinned-object":
        assert passed_over["artifact_digest"] == digest
        assert (passed_over["tier"], passed_over["reason"]) == ("pinned", "integrity")
        assert passed_over["computed_digest"] not in (None, digest)
    else:
        assert passed_over == {
            "event": STORED_FRAME_COPY_PASSED_OVER,
            "artifact_digest": undecodable.digest,
            "tier": "working",
            "reason": "undecodable",
            "computed_digest": None,
        }


def _passed_over(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    """The structured fields of each stored frame copy `linked_stored_frame` passed over."""
    fields = [getattr(record, JSON_FIELDS_ATTR, {}) for record in caplog.records]
    return [field for field in fields if field.get("event") == STORED_FRAME_COPY_PASSED_OVER]


def test_a_hashless_pair_whose_only_frame_copy_does_not_decode_refuses_unreadable(
    deps: _Deps, caplog: pytest.LogCaptureFixture
) -> None:
    """With no copy that reads, the pair has no frame to name it by: the copy that did not
    decode refuses, and through the door answers `internal_error`, the answer of every stored
    frame that does not decode."""
    _record_stored_pair(
        deps,
        data=b"not a parquet file",
        media_type=CANONICAL_FRAME_MEDIA_TYPE,
        rendering=dict(CANONICAL_FRAME_RENDERING),
    )
    client = _build_client(deps)

    with caplog.at_level(logging.WARNING, logger="datasets_api.render"):
        with pytest.raises(StoredFrameUnreadable):
            deps.publish(rows=_rows(1.0))
        response = client.post(
            "/api/datasets/frames/publish", json=_publish_body(), headers=_AUTH_HEADERS
        )

    assert response.status_code == 500, response.text
    assert (response.json()["code"], response.json()["retryable"]) == ("internal_error", False)
    assert [held["reason"] for held in _passed_over(caplog)] == ["undecodable", "undecodable"]


def test_a_hashless_pair_whose_frame1_object_is_gone_is_named_by_its_digest(deps: _Deps) -> None:
    """With no bytes left to read, a whole frame1 document's digest, the frame hash of the rows
    it held, names the pair: the same rows match without a read, and other rows refuse."""
    _record_stored_pair(
        deps,
        data=_frame1_document(1.0),
        media_type=FRAME1_MEDIA_TYPE,
        rendering={"format": "frame1"},
    )
    _expire_stored_frame(deps)
    counting = _ReadCountingArtifactStore(deps.artifact_store)

    repeat = deps.publish(rows=_rows(1.0), artifact_store=counting)

    assert repeat.created is False
    assert counting.reads == 0
    with pytest.raises(PublishBytesDiverge):
        deps.publish(rows=_rows(2.0))


def _expire_stored_frame(deps: _Deps) -> None:
    """Put every stored location of the pair's frame past working-tier retention and delete
    its objects, as the sweep leaves an unpinned frame."""
    with deps.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.artifact_location SET stored_at = %s",
            (datetime.now(UTC) - timedelta(days=366),),
        )
    for bucket in deps.artifact_store._client.buckets.values():
        bucket.objects.clear()


class _ReEncodingArtifactStore:
    """Stores a canonical frame under another Parquet encoding than the door writes, as an
    image on another PyArrow version does, and counts `get` calls."""

    def __init__(self, store: LiveArtifactStore) -> None:
        self._store = store
        self.gets = 0

    @contextmanager
    def open_put(self, *, tier: Tier):
        with self._store.open_put(tier=tier) as sink:
            written = _ReEncodingSink(sink)
            yield written
            written.flush()

    def get(self, *, artifact_digest: str, tier: Tier) -> bytes:
        self.gets += 1
        return self._store.get(artifact_digest=artifact_digest, tier=tier)


class _ReEncodingSink:
    def __init__(self, sink) -> None:
        self._sink = sink
        self._written = io.BytesIO()

    def write(self, data: bytes, /) -> int:
        return self._written.write(data)

    def flush(self) -> None:
        buffer = io.BytesIO()
        table = pq.read_table(io.BytesIO(self._written.getvalue()))
        pq.write_table(table, buffer, compression="snappy", row_group_size=1)
        self._sink.write(buffer.getvalue())

    def result(self):
        return self._sink.result()


def test_republishing_an_expired_frame_under_another_encoding_makes_it_servable(
    deps: _Deps,
) -> None:
    """A repeat publish matches by the frame hash the version records, with no stored object
    left to read, and links its own canonical frame in place of the expired one."""
    deps.publish()
    (first,) = _canonical_frames(deps)
    _expire_stored_frame(deps)
    assert select_servable_locations(_canonical_frames(deps), now=datetime.now(UTC)) == []
    re_encoding = _ReEncodingArtifactStore(deps.artifact_store)

    second = deps.publish(artifact_store=re_encoding)

    assert second.created is False
    assert re_encoding.gets == 0
    (servable,) = select_servable_locations(_canonical_frames(deps), now=datetime.now(UTC))
    assert servable.artifact_digest != first.artifact_digest
    (servable_file,) = select_servable_locations(_frame_files(deps), now=datetime.now(UTC))
    assert servable_file.artifact_digest == second.artifact_digest
    raw = deps.artifact_store.get(artifact_digest=servable.artifact_digest, tier=Tier.working)
    assert read_canonical_frame(raw) == build_frame(role_keys=_ROLE_KEYS, rows=_rows())


def test_republishing_other_rows_over_an_expired_frame_conflicts_without_reading_it(
    deps: _Deps,
) -> None:
    deps.publish()
    (first,) = _canonical_frames(deps)
    _expire_stored_frame(deps)
    re_encoding = _ReEncodingArtifactStore(deps.artifact_store)

    with pytest.raises(PublishBytesDiverge):
        deps.publish(rows=_rows(2.0), artifact_store=re_encoding)

    assert re_encoding.gets == 0
    assert {loc.artifact_digest for loc in _canonical_frames(deps)} == {first.artifact_digest}


def test_http_republishing_other_rows_over_an_expired_frame_maps_to_publish_conflict_409() -> None:
    deps = _Deps()
    client = _build_client(deps)
    first = client.post("/api/datasets/frames/publish", json=_publish_body(), headers=_AUTH_HEADERS)
    assert first.status_code == 200, first.text
    _expire_stored_frame(deps)

    response = client.post(
        "/api/datasets/frames/publish", json=_publish_body(rows=_rows(2.0)), headers=_AUTH_HEADERS
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "publish_conflict"


def test_a_pair_recording_no_frame_hash_whose_stored_frame_is_gone_refuses(deps: _Deps) -> None:
    """With no hash on its record and no stored frame to read, the pair's frame has no
    identity to match."""
    _record_stored_pair(
        deps,
        data=_snappy_parquet_frame(1.0),
        media_type=CANONICAL_FRAME_MEDIA_TYPE,
        rendering=dict(CANONICAL_FRAME_RENDERING),
    )
    _expire_stored_frame(deps)

    with pytest.raises(PublishBytesDiverge):
        deps.publish(rows=_rows(1.0))


_COMPOSED_STAT_VARS = [
    {
        "stat_var": "dcid:Exemplar",
        "relation": "quotient",
        "first": ["dcid:Amount_GDP"],
        "second": ["dcid:Count_Person"],
    }
]


def test_repeat_publish_keeps_the_composed_stat_vars_a_stored_body_holds(deps: _Deps) -> None:
    """A version stored with `composed_stat_vars` keeps them through a repeat publish, which
    compares the frame and its lineage alone."""
    assert deps.publish().created is True
    with deps.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET composed_stat_vars = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps(_COMPOSED_STAT_VARS), _SPEC_ID, _CONTENT_VERSION),
        )

    assert deps.publish().created is False

    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION).facts == FrameLineage(
        input_content_versions=["cv_00000000000000000000000000abc123"],
        composed_stat_vars=_COMPOSED_STAT_VARS,
        continuations=[],
    )


@pytest.mark.parametrize(
    "input_content_versions",
    [("cv_00000000000000000000000000abc123",), ("cv_00000000000000000000000000bbbbbb",)],
)
def test_repeat_publish_with_different_bytes_refuses_rather_than_overwrites(
    deps: _Deps, input_content_versions: tuple[str, ...]
) -> None:
    """Different rows refuse as a byte conflict, and a call whose lineage differs too reports
    the byte conflict, since lineage is compared only once the frames match."""
    deps.publish(rows=_rows(1.0))
    (first,) = _canonical_frames(deps)

    with pytest.raises(PublishBytesDiverge) as excinfo:
        deps.publish(rows=_rows(2.0), input_content_versions=input_content_versions)

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert [loc.artifact_digest for loc in _canonical_frames(deps)] == [first.artifact_digest]


def _canonical_frame_only(deps: _Deps) -> str:
    """Publishes `_rows()`, which leaves the pair linking its canonical frame and no frame1
    document. Returns the canonical frame's digest."""
    assert deps.publish().created is True
    return canonical_frame_digest(deps.store, subject_id=_SPEC_ID, content_version=_CONTENT_VERSION)


def test_repeat_publish_of_the_frame_a_canonical_frame_holds_answers_its_linked_digest(
    deps: _Deps,
) -> None:
    canonical_digest = _canonical_frame_only(deps)
    (frame_file,) = _frame_files(deps)

    repeat = deps.publish()

    assert repeat.created is False
    assert repeat.artifact_digest == frame_file.artifact_digest
    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    assert {loc.artifact_digest for loc in locations} == {
        canonical_digest,
        frame_file.artifact_digest,
    }


def test_repeat_publish_of_another_frame_than_a_canonical_frame_holds_refuses(
    deps: _Deps,
) -> None:
    _canonical_frame_only(deps)

    with pytest.raises(PublishBytesDiverge):
        deps.publish(rows=_rows(2.0))


def test_repeat_publish_with_identical_bytes_but_different_lineage_refuses(deps: _Deps) -> None:
    """Identical bytes with different lineage raise `PublishLineageDiverge` before writing."""
    deps.publish(input_content_versions=("cv_00000000000000000000000000aaaaaa",))
    (first,) = _canonical_frames(deps)

    with pytest.raises(PublishLineageDiverge) as excinfo:
        deps.publish(input_content_versions=("cv_00000000000000000000000000bbbbbb",))

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    # The original lineage stays on record; the second call's own claimed
    # lineage never got recorded.
    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert stored.facts == FrameLineage(
        input_content_versions=["cv_00000000000000000000000000aaaaaa"], continuations=[]
    )
    assert _canonical_frames(deps)[0].artifact_digest == first.artifact_digest


def test_repeat_publish_over_a_pair_recording_no_lineage_refuses(deps: _Deps) -> None:
    """A pair holding this call's frame but no `input_content_versions` (a dataset version's
    facts) refuses `PublishLineageDiverge`: lineage is compared, never assumed. The op-spec
    identity is the row's `subject_id`, so the lookup has already matched it."""
    frame = render_and_store_canonical_frame(
        frame_chunks=(build_frame(role_keys=_ROLE_KEYS, rows=_rows()),),
        subject_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        artifact_store=deps.artifact_store,
    )
    deps.version_writer.record_version(
        VersionRecord(
            subject_id=_SPEC_ID,
            subject_kind="spec",
            content_version=_CONTENT_VERSION,
            schema_version=1,
            facts=facts_from_body(golden_cases()["definition"]["body"]),
            artifacts=(frame,),
            frame_sha256=frame_sha256(build_frame(role_keys=_ROLE_KEYS, rows=_rows())),
        )
    )

    with pytest.raises(PublishLineageDiverge):
        deps.publish()


def test_repeat_publish_of_a_revoked_pair_raises_snapshot_revoked_not_replay(deps: _Deps) -> None:
    """Republishing a revoked pair raises `SnapshotRevoked` even when bytes and lineage match."""
    deps.publish()
    _revoke(deps.store, subject_id=_SPEC_ID, content_version=_CONTENT_VERSION)

    with pytest.raises(SnapshotRevoked) as excinfo:
        deps.publish()

    assert excinfo.value.id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION


def test_revocation_outranks_bytes_divergence(deps: _Deps) -> None:
    """Revocation takes precedence over byte divergence."""
    deps.publish(rows=_rows(1.0))
    _revoke(deps.store, subject_id=_SPEC_ID, content_version=_CONTENT_VERSION)

    with pytest.raises(SnapshotRevoked):
        deps.publish(rows=_rows(2.0))


def test_revocation_outranks_lineage_divergence(deps: _Deps) -> None:
    """Revocation takes precedence over lineage divergence."""
    deps.publish(input_content_versions=("cv_00000000000000000000000000aaaaaa",))
    _revoke(deps.store, subject_id=_SPEC_ID, content_version=_CONTENT_VERSION)

    with pytest.raises(SnapshotRevoked):
        deps.publish(input_content_versions=("cv_00000000000000000000000000bbbbbb",))


class _CheckBlindSnapshotReader:
    """Hides the first version read to simulate a concurrent publish between check and verify.

    Later reads delegate to the real snapshot reader.
    """

    def __init__(self, reader) -> None:
        self._reader = reader
        self._read_version_calls = 0

    def read_version(self, subject_id: str, content_version: str):
        self._read_version_calls += 1
        if self._read_version_calls == 1:
            return None
        return self._reader.read_version(subject_id, content_version)

    def read_artifact_locations(self, subject_id: str, content_version: str):
        return self._reader.read_artifact_locations(subject_id, content_version)

    def read_revocation(self, subject_id: str, content_version: str):
        return self._reader.read_revocation(subject_id, content_version)

    def read_version_state(self, subject_id: str, content_version: str):
        return self._reader.read_version_state(subject_id, content_version)

    def read_versions_of(self, content_versions):
        return self._reader.read_versions_of(content_versions)


def test_verify_phase_catches_a_concurrent_publish_with_different_lineage(deps: _Deps) -> None:
    """Post-write verification catches the losing publish when concurrent calls claim
    different `input_content_versions` for identical bytes.
    """
    winner = deps.publish(input_content_versions=("cv_00000000000000000000000000aaaaaa",))

    blind_reader = _CheckBlindSnapshotReader(deps.snapshot_reader)
    with pytest.raises(PublishLineageDiverge) as excinfo:
        publish_frame(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            role_keys=_ROLE_KEYS,
            rows=_rows(),
            input_content_versions=("cv_00000000000000000000000000bbbbbb",),
            artifact_store=deps.artifact_store,
            snapshot_reader=blind_reader,
            version_writer=deps.version_writer,
            file_enrichment=ENRICHMENT,
            site=SITE,
            prepared_at=PREPARED_AT,
        )

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert stored.facts == FrameLineage(
        input_content_versions=["cv_00000000000000000000000000aaaaaa"], continuations=[]
    )
    assert winner.created is True


def test_verify_phase_derives_created_from_the_insert_not_from_reaching_it(deps: _Deps) -> None:
    """Post-write verification reports `created=False` when a concurrent call inserted
    the same bytes and lineage first.
    """
    deps.publish()  # the call that actually lands first

    blind_reader = _CheckBlindSnapshotReader(deps.snapshot_reader)
    result = publish_frame(
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        role_keys=_ROLE_KEYS,
        rows=_rows(),
        input_content_versions=("cv_00000000000000000000000000abc123",),
        artifact_store=deps.artifact_store,
        snapshot_reader=blind_reader,
        version_writer=deps.version_writer,
        file_enrichment=ENRICHMENT,
        site=SITE,
        prepared_at=PREPARED_AT,
    )

    assert result.created is False


def test_verify_phase_answers_the_digest_a_concurrent_publish_linked(deps: _Deps) -> None:
    """A publish that loses the insert to another encoding of the same frame keeps the
    winner's canonical frame and answers the digest of the frame file the pair links.
    """
    _record_stored_pair(
        deps,
        data=_snappy_parquet_frame(1.0),
        media_type=CANONICAL_FRAME_MEDIA_TYPE,
        rendering=dict(CANONICAL_FRAME_RENDERING),
        frame_sha256=frame_sha256(build_frame(role_keys=_ROLE_KEYS, rows=_rows(1.0))),
    )
    [linked] = _canonical_frames(deps)

    result = publish_frame(
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        role_keys=_ROLE_KEYS,
        rows=_rows(1.0),
        input_content_versions=("cv_00000000000000000000000000abc123",),
        artifact_store=deps.artifact_store,
        snapshot_reader=_CheckBlindSnapshotReader(deps.snapshot_reader),
        version_writer=deps.version_writer,
        file_enrichment=ENRICHMENT,
        site=SITE,
        prepared_at=PREPARED_AT,
    )

    assert result.created is False
    assert [loc.artifact_digest for loc in _canonical_frames(deps)] == [linked.artifact_digest]
    assert [loc.artifact_digest for loc in _frame_files(deps)] == [result.artifact_digest]


def test_verify_phase_catches_a_revocation_landing_between_check_and_verify(deps: _Deps) -> None:
    """Post-write verification detects a revocation that lands after the initial check,
    even when this call's insert loses to an existing row.
    """
    deps.publish()
    _revoke(deps.store, subject_id=_SPEC_ID, content_version=_CONTENT_VERSION)

    blind_reader = _CheckBlindSnapshotReader(deps.snapshot_reader)
    with pytest.raises(SnapshotRevoked) as excinfo:
        publish_frame(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            role_keys=_ROLE_KEYS,
            rows=_rows(),
            input_content_versions=("cv_00000000000000000000000000abc123",),
            artifact_store=deps.artifact_store,
            snapshot_reader=blind_reader,
            version_writer=deps.version_writer,
            file_enrichment=ENRICHMENT,
            site=SITE,
            prepared_at=PREPARED_AT,
        )

    assert excinfo.value.id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION


def test_publish_composed_with_a_null_version_writer_raises_publish_record_missing(
    deps: _Deps,
) -> None:
    """A null version writer leaves no row for post-write verification, raising
    `PublishRecordMissing` after the artifact has been stored.
    """
    with pytest.raises(PublishRecordMissing) as excinfo:
        publish_frame(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            role_keys=_ROLE_KEYS,
            rows=_rows(),
            input_content_versions=("cv_00000000000000000000000000abc123",),
            artifact_store=deps.artifact_store,
            snapshot_reader=deps.snapshot_reader,
            version_writer=NullVersionRecordWriter(),
            file_enrichment=ENRICHMENT,
            site=SITE,
            prepared_at=PREPARED_AT,
        )

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None

    # Re-rendering the orphaned frame recovers its content-addressed digest.
    frame = build_frame(role_keys=_ROLE_KEYS, rows=_rows())
    artifact = render_and_store_canonical_frame(
        frame_chunks=(frame,),
        subject_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        artifact_store=deps.artifact_store,
    )
    raw = deps.artifact_store.get(artifact_digest=artifact.artifact_digest, tier=Tier.working)
    assert read_canonical_frame(raw) == frame


def test_malformed_frame_refuses_before_any_write(deps: _Deps) -> None:
    with pytest.raises(PublishFrameInvalid):
        deps.publish(rows=[{"stat_var": "dcid:Exemplar", "time_period": "2020"}])  # missing value

    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_different_content_versions_of_the_same_spec_id_publish_independently(deps: _Deps) -> None:
    first = deps.publish(content_version="ocv_000001", rows=_rows(1.0))
    second = deps.publish(content_version="ocv_000002", rows=_rows(2.0))

    assert first.created is True
    assert second.created is True
    assert first.artifact_digest != second.artifact_digest


def test_ceiling_over_the_configured_row_count_refuses_before_any_write(deps: _Deps) -> None:
    with pytest.raises(OverCeilingError) as excinfo:
        deps.publish(rows=_distinct_grain_rows(2), ceiling=1)

    assert excinfo.value.ceiling_kind == "rows"
    assert excinfo.value.count == 2
    assert excinfo.value.cap == 1
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_ceiling_at_exactly_the_configured_row_count_succeeds(deps: _Deps) -> None:
    result = deps.publish(rows=_distinct_grain_rows(2), ceiling=2)

    assert result.created is True


def _build_client(
    deps: _Deps,
    *,
    allowed_principals: frozenset[str] | None = frozenset({_PRINCIPAL}),
    ceiling: int | None = None,
):
    door_kwargs: dict[str, object] = {"allowed_principals": allowed_principals}
    if ceiling is not None:
        door_kwargs["ceiling"] = ceiling
    door = build_publish_door(
        deps=fake_deps_cell(
            artifact_store=deps.artifact_store,
            snapshot_reader=deps.snapshot_reader,
            version_writer=deps.version_writer,
        ),
        settings=DatasetsBuildSettings.from_env(),
        **door_kwargs,
    )
    return make_client_for_door(door)


def test_http_fresh_publish_returns_200_with_created_true() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish", json=_publish_body(), headers=_AUTH_HEADERS
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["published"]["created"] is True
    assert body["published"]["spec_id"] == _SPEC_ID
    assert body["published"]["content_version"] == _CONTENT_VERSION


def test_http_empty_rows_and_role_keys_publish_successfully() -> None:
    """The frame grammar accepts empty rows and role keys for subject-only frames."""
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(role_keys=[], rows=[]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_http_non_hex_spec_id_is_a_pydantic_request_invalid() -> None:
    """The publish request rejects non-hex `spec_id` suffixes at validation."""
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(spec_id="ops_exemplar"),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"


def test_http_malformed_frame_maps_to_request_invalid_400() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[{"stat_var": "dcid:Exemplar", "time_period": "2020"}]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["retryable"] is False


def test_http_divergent_bytes_maps_to_publish_conflict_409() -> None:
    deps = _Deps()
    client = _build_client(deps)
    first = _publish_body(rows=_rows(1.0))
    second = {**first, "rows": _rows(2.0)}

    assert (
        client.post("/api/datasets/frames/publish", json=first, headers=_AUTH_HEADERS).status_code
        == 200
    )
    response = client.post("/api/datasets/frames/publish", json=second, headers=_AUTH_HEADERS)

    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "publish_conflict"
    assert body["retryable"] is False


def test_http_revoked_pair_maps_to_snapshot_revoked_410() -> None:
    deps = _Deps()
    client = _build_client(deps)
    body = _publish_body()
    first = client.post("/api/datasets/frames/publish", json=body, headers=_AUTH_HEADERS)
    assert first.status_code == 200, first.text

    _revoke(deps.store, subject_id=_SPEC_ID, content_version=_CONTENT_VERSION)

    response = client.post("/api/datasets/frames/publish", json=body, headers=_AUTH_HEADERS)
    assert response.status_code == 410, response.text
    response_body = response.json()
    assert response_body["code"] == "snapshot_revoked"
    assert response_body["retryable"] is False


def test_http_malformed_spec_id_is_a_pydantic_request_invalid() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(spec_id="spec_not_an_ops_id"),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"


def test_http_over_ceiling_maps_to_over_ceiling_422() -> None:
    deps = _Deps()
    client = _build_client(deps, ceiling=1)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=_distinct_grain_rows(2)),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "over_ceiling"
    assert body["retryable"] is False


def test_http_publish_without_a_principal_header_refuses_publish_not_permitted() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post("/api/datasets/frames/publish", json=_publish_body())
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "publish_not_permitted"
    # No version row: an unauthorized publish never writes.
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_http_publish_with_an_unlisted_principal_refuses_publish_not_permitted() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(),
        headers={"x-dfo-principal": "some-other-caller"},
    )
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "publish_not_permitted"


def test_http_publish_with_no_allowed_principals_configured_refuses_everyone() -> None:
    deps = _Deps()
    client = _build_client(deps, allowed_principals=frozenset())
    response = client.post(
        "/api/datasets/frames/publish", json=_publish_body(), headers=_AUTH_HEADERS
    )
    assert response.status_code == 403, response.text
    assert response.json()["code"] == "publish_not_permitted"


def test_http_publish_with_two_principal_header_values_refuses_regardless_of_order() -> None:
    deps = _Deps()
    client = _build_client(deps)
    body = _publish_body()

    allowed_first = client.post(
        "/api/datasets/frames/publish",
        json=body,
        headers=[("x-dfo-principal", _PRINCIPAL), ("x-dfo-principal", "some-other-caller")],
    )
    assert allowed_first.status_code == 403, allowed_first.text
    assert allowed_first.json()["code"] == "publish_not_permitted"

    allowed_second = client.post(
        "/api/datasets/frames/publish",
        json=body,
        headers=[("x-dfo-principal", "some-other-caller"), ("x-dfo-principal", _PRINCIPAL)],
    )
    assert allowed_second.status_code == 403, allowed_second.text
    assert allowed_second.json()["code"] == "publish_not_permitted"


def test_http_missing_version_row_after_write_maps_to_internal_error_500() -> None:
    """A publish door explicitly composed with a `version_writer` that does
    not share `snapshot_reader`'s own store reproduces
    `PublishRecordMissing` on the wire as `internal_error`/500.
    """
    deps = _Deps()
    door = build_publish_door(
        deps=fake_deps_cell(
            artifact_store=deps.artifact_store,
            snapshot_reader=deps.snapshot_reader,
            version_writer=NullVersionRecordWriter(),
        ),
        settings=DatasetsBuildSettings.from_env(),
        allowed_principals=frozenset({_PRINCIPAL}),
    )
    client = make_client_for_door(door)
    response = client.post(
        "/api/datasets/frames/publish", json=_publish_body(), headers=_AUTH_HEADERS
    )
    assert response.status_code == 500, response.text
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["retryable"] is False


# `MAX_ROWS` is a schema backstop above the publish door's `PUBLISH_MAX_ROWS`.
# Pydantic failures return `request_invalid` before route execution.


def _no_artifact_store_puts(deps: _Deps) -> bool:
    """Checks whether request validation prevented all artifact writes."""
    return not deps.artifact_store._client.buckets


def _wide_role_keys(count: int) -> list[str]:
    return [f"role{i}" for i in range(count)]


def _row_at_max_width(role_keys: list[str]) -> dict[str, object]:
    """Builds a row at `MAX_ROW_WIDTH` when `role_keys` has `MAX_ROLE_KEYS` entries."""
    row: dict[str, object] = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "unit": "USD",
        "measurementMethod": "dcAggregate/method",
        "provenance": "dcid:Provenance",
    }
    for index, role in enumerate(role_keys):
        row[role] = f"country/K{index:02d}"
    return row


def test_http_role_keys_over_the_cap_is_request_invalid_with_a_pointer() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(role_keys=_wide_role_keys(MAX_ROLE_KEYS + 1), rows=[]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/role_keys"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_http_role_keys_at_exactly_the_cap_succeeds() -> None:
    deps = _Deps()
    client = _build_client(deps)
    role_keys = _wide_role_keys(MAX_ROLE_KEYS)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(role_keys=role_keys, rows=[_row_at_max_width(role_keys)]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_http_row_width_over_the_cap_is_request_invalid_and_writes_nothing() -> None:
    """Request validation rejects a row one key above `MAX_ROW_WIDTH` before frame building."""
    deps = _Deps()
    client = _build_client(deps)
    row = {f"k{i}": "v" for i in range(MAX_ROW_WIDTH + 1)}
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows/0"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_http_row_width_at_exactly_the_cap_succeeds() -> None:
    deps = _Deps()
    client = _build_client(deps)
    role_keys = _wide_role_keys(MAX_ROLE_KEYS)
    row = _row_at_max_width(role_keys)
    assert len(row) == MAX_ROW_WIDTH
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(role_keys=role_keys, rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_outer_rows_cap_short_circuits_the_row_cell_validator() -> None:
    """Pydantic enforces outer `max_length` before the field's `mode="after"` validator.
    A small mirror tests this without allocating the `MAX_ROWS` limit of 1,000,000 rows.
    """
    assert PublishFrameRequest.model_fields["rows"].metadata[0].max_length == MAX_ROWS

    validator_calls: list[int] = []

    class _RowsCapMirror(BaseModel):
        rows: list[Annotated[dict[str, object], Field(max_length=3)]] = Field(max_length=5)

        @field_validator("rows", mode="after")
        @classmethod
        def _record_call(cls, value: list[dict[str, object]]) -> list[dict[str, object]]:
            validator_calls.append(len(value))
            return value

    with pytest.raises(ValidationError) as exc_info:
        _RowsCapMirror(rows=[{"a": 1}] * 10)

    assert validator_calls == []
    assert any(error["type"] == "too_long" for error in exc_info.value.errors())


def test_http_input_content_versions_over_the_cap_is_request_invalid_with_a_pointer() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(
            input_content_versions=["cv_00000000000000000000000000abc123"]
            * (MAX_INPUT_CONTENT_VERSIONS + 1)
        ),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input_content_versions"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


@pytest.mark.parametrize(
    ("token", "status"),
    [
        pytest.param("cv_abc123", 400, id="a-short-cv"),
        pytest.param("cv_" + "a" * 33, 400, id="a-long-cv"),
        pytest.param("cv_" + "a" * 32, 200, id="control-a-32-hex-cv"),
        pytest.param("ocv_abc123", 200, id="control-a-short-ocv"),
    ],
)
def test_http_an_input_content_version_outside_the_frame_grammar_is_request_invalid(
    token: str, status: int
) -> None:
    """A `cv_` input holds exactly 32 hex digits, as the frame grammar and the description
    record require; an `ocv_` input keeps the publish door's 6 to 64."""
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(input_content_versions=["cv_" + "b" * 32, token]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == status, response.text
    if status == 400:
        body = response.json()
        assert body["code"] == "request_invalid"
        assert body["pointer"] == "/input_content_versions/1"
        assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_http_input_content_versions_at_exactly_the_cap_succeeds() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(
            input_content_versions=["cv_00000000000000000000000000abc123"]
            * MAX_INPUT_CONTENT_VERSIONS
        ),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_http_role_keys_entry_over_max_cell_chars_is_request_invalid() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(role_keys=["x" * (MAX_CELL_CHARS + 1)], rows=[]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/role_keys/0"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_http_role_keys_entry_at_exactly_max_cell_chars_succeeds() -> None:
    deps = _Deps()
    client = _build_client(deps)
    role = "x" * MAX_CELL_CHARS
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(
            role_keys=[role],
            rows=[
                {
                    "stat_var": "dcid:Exemplar",
                    "time_period": "2020",
                    "value": 1.0,
                    role: "country/KEN",
                }
            ],
        ),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_http_row_key_over_max_cell_chars_is_request_invalid_and_writes_nothing() -> None:
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "z" * (MAX_CELL_CHARS + 1): "recipient-value",
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(role_keys=[], rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_http_row_string_value_over_max_cell_chars_is_request_invalid_and_writes_nothing() -> None:
    """The scalar type reports the oversized value at `/rows/0/recipient`, not `/rows`."""
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "x" * (MAX_CELL_CHARS + 1),
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows/0/recipient"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_http_row_string_value_at_exactly_max_cell_chars_succeeds() -> None:
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "x" * MAX_CELL_CHARS,
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_http_unit_mapping_value_over_max_cell_chars_is_request_invalid_and_writes_nothing() -> (
    None
):
    """`_UnitMapping` bounds `currency` at `MAX_CELL_CHARS` and reports `/rows/0/unit/currency`."""
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
        "unit": {"currency": "x" * (MAX_CELL_CHARS + 1), "prices": "current"},
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows/0/unit/currency"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_unit_mapping_value_at_exactly_max_cell_chars_parses() -> None:
    """The model accepts a currency of `MAX_CELL_CHARS`; the publish route also requires ISO 4217.
    This boundary is checked on `PublishFrameRequest` alone.
    """
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
        "unit": {"currency": "x" * MAX_CELL_CHARS, "prices": "current"},
    }
    request = PublishFrameRequest(
        schema_version=1,
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        role_keys=list(_ROLE_KEYS),
        rows=[row],
        input_content_versions=["cv_00000000000000000000000000abc123"],
    )

    assert request.rows[0]["unit"] == row["unit"]


def test_http_unit_entries_over_the_cap_is_request_invalid_and_writes_nothing() -> None:
    """A fourth `unit` member exceeds the three-key grammar and fails request validation."""
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
        "unit": {"currency": "USD", "prices": "current", "base_year": 2020, "extra": "x"},
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows/0/unit"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_http_unit_entries_at_exactly_the_cap_succeeds() -> None:
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
        "unit": {"currency": "USD", "prices": "constant", "base_year": 2015},
    }
    assert len(row["unit"]) == MAX_UNIT_ENTRIES
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True


def test_http_unit_key_over_max_cell_chars_is_request_invalid_and_writes_nothing() -> None:
    """The validator bounds unit keys and truncates the reported key to 80 characters."""
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
        "unit": {"x" * (MAX_CELL_CHARS + 1): "USD", "prices": "current"},
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows"
    assert "x" * 80 in body["detail"]
    assert "x" * (MAX_CELL_CHARS + 1) not in body["detail"]
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_unit_key_at_exactly_max_cell_chars_parses() -> None:
    """The model accepts a `MAX_CELL_CHARS`-length key; frame grammar validates unit keys later."""
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
        "unit": {"x" * MAX_CELL_CHARS: "USD"},
    }
    request = PublishFrameRequest(
        schema_version=1,
        spec_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        role_keys=list(_ROLE_KEYS),
        rows=[row],
        input_content_versions=["cv_00000000000000000000000000abc123"],
    )

    assert request.rows[0]["unit"] == row["unit"]


def test_http_row_cell_nested_list_is_request_invalid_and_writes_nothing() -> None:
    """Pydantic rejects nested lists before `Frame.build` receives the row."""
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": ["dcid:Exemplar"],
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows/0/stat_var"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_http_mapping_cell_under_non_unit_key_is_request_invalid_and_writes_nothing() -> None:
    """The row validator restricts mapping cells to the `unit` key."""
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": {"currency": "USD", "prices": "current"},
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows"
    assert "recipient" in body["detail"]
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_http_unit_mapping_nested_a_second_level_is_request_invalid_and_writes_nothing() -> None:
    """Pydantic rejects a mapping nested inside a `unit` value."""
    deps = _Deps()
    client = _build_client(deps)
    row = {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
        "unit": {"currency": {"role": "recipient"}, "prices": "current"},
    }
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[row]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/rows/0/unit/currency"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


_PER_PERSON = {"stat_var": "dcid:Count_Person"}
_LONGEST_STAT_VAR = "dcid:" + "x" * (MAX_UNIT_STAT_VAR_CHARS - len("dcid:"))


def _base_year(year: int) -> dict[str, object]:
    return {"currency": "EUR", "prices": "constant", "base_year": year}


@pytest.mark.parametrize(
    "unit",
    [
        {"numerator": "USDollar", "denominator": _PER_PERSON},
        {
            "numerator": {"currency": "EUR", "prices": "constant", "base_year": 2020},
            "denominator": _PER_PERSON,
        },
        {"numerator": "USDollar", "denominator": "Year"},
        {"numerator": "USDollar", "denominator": {"stat_var": _LONGEST_STAT_VAR}},
        {"numerator": _base_year(1900), "denominator": _base_year(2100)},
    ],
    ids=[
        "ref_over_stat_var",
        "converted_over_stat_var",
        "ref_over_ref",
        "longest_stat_var",
        "base_years_at_the_catalogue_bounds",
    ],
)
def test_http_composed_unit_publishes_and_the_canonical_frame_holds_its_json(
    unit: dict[str, object],
) -> None:
    """The canonical frame holds the composed unit as its grammar cell, and the frame file the
    response names holds it as JCS text."""
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[{**_rows()[0], "unit": unit}]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    (canonical,) = _canonical_frames(deps)
    raw = deps.artifact_store.get(artifact_digest=canonical.artifact_digest, tier=Tier.working)
    document = canonical_frame_document(read_canonical_frame_table(raw))
    assert [row[document["columns"].index("unit")] for row in document["rows"]] == [unit]
    flat = deps.artifact_store.get(
        artifact_digest=response.json()["published"]["artifact_digest"], tier=Tier.working
    )
    assert pq.read_table(io.BytesIO(flat)).column("unit").to_pylist() == [
        rfc8785.dumps(unit).decode()
    ]


# U+0958 is two code points in NFC, so each text is within its bound as sent and over it as
# `Frame.build` stores it.
_NFC_GROWING_REF = "dcid:" + "\u0958" * 195
_NFC_GROWING_TEXT = "\u0958" * MAX_CELL_CHARS


@pytest.mark.parametrize(
    ("cells", "pointer"),
    [
        (
            {"unit": {"numerator": "USDollar", "denominator": {"stat_var": {"ref": "dcid:X"}}}},
            "/rows/0/unit/denominator/stat_var",
        ),
        (
            {"unit": {"numerator": "USDollar", "denominator": {"stat_var": "Count_Person"}}},
            "/rows/0/unit/denominator/stat_var",
        ),
        (
            {"unit": {"numerator": {"stat_var": _LONGEST_STAT_VAR + "x"}, "denominator": "Year"}},
            "/rows/0/unit/numerator/stat_var",
        ),
        (
            {"unit": {"numerator": "USDollar", "denominator": {**_PER_PERSON, "label": "People"}}},
            "/rows/0/unit/denominator/label",
        ),
        (
            {"unit": {"numerator": "USDollar", "denominator": _PER_PERSON, "scale": "Year"}},
            "/rows/0/unit",
        ),
        ({"unit": {"numerator": "USDollar", "denominator": None}}, "/rows/0/unit/denominator"),
        ({"unit": {"numerator": "USDollar", "per": "Year"}}, None),
        ({"recipient": {"numerator": "USDollar", "denominator": _PER_PERSON}}, "/rows"),
        (
            {"unit": {"numerator": "USDollar", "denominator": {"stat_var": _NFC_GROWING_REF}}},
            "/rows",
        ),
        ({"unit": {"numerator": _NFC_GROWING_TEXT, "denominator": _PER_PERSON}}, "/rows"),
        ({"unit": _NFC_GROWING_TEXT}, "/rows"),
        ({"unit": _base_year(1899)}, "/rows"),
        ({"unit": {"numerator": _base_year(2101), "denominator": _PER_PERSON}}, "/rows"),
        ({"unit": _base_year(int("9" * 4_000))}, "/rows"),
    ],
    ids=[
        "three_levels",
        "stat_var_without_dcid",
        "stat_var_over_200",
        "stat_var_part_extra_member",
        "third_member",
        "null_side",
        "unknown_member_refused_by_the_grammar",
        "composed_under_a_role_key",
        "stat_var_over_200_in_nfc",
        "side_over_the_cell_bound_in_nfc",
        "unit_ref_over_the_cell_bound_in_nfc",
        "base_year_below_the_catalogue_range",
        "side_base_year_above_the_catalogue_range",
        "base_year_of_4000_digits",
    ],
)
def test_http_malformed_composed_unit_is_request_invalid_and_writes_nothing(
    cells: dict[str, object], pointer: str | None
) -> None:
    """The request model bounds a composed unit's size and depth and names the member at
    fault. `Frame.build` refuses a member name the model admits, with no pointer."""
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[{**_rows()[0], **cells}]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body.get("pointer") == pointer
    assert len(response.text) < 2_000
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


_LONG_KEY = "k" * 50_000


def _one_row(cells: dict[str, object]) -> dict[str, object]:
    return {"rows": [{**_rows()[0], **cells}]}


@pytest.mark.parametrize(
    ("overrides", "pointer"),
    [
        (
            _one_row({"unit": {"numerator": "USDollar", "denominator": {_LONG_KEY: "USD"}}}),
            "/rows",
        ),
        (
            _one_row(
                {"unit": {"numerator": {"stat_var": "dcid:a", _LONG_KEY: "y"}, "denominator": "b"}}
            ),
            "/rows",
        ),
        (_one_row({"unit": {"numerator": {_LONG_KEY: [1]}, "denominator": "b"}}), "/rows"),
        (_one_row({"unit": {_LONG_KEY: [1]}}), "/rows"),
        (_one_row({"unit": {"numerator": "a", "denominator": "b", _LONG_KEY: "y"}}), "/rows"),
        (_one_row({_LONG_KEY: [1]}), "/rows"),
        (_one_row({"recipient": {_LONG_KEY: [1]}}), "/rows"),
        (
            _one_row({"recipient": {"numerator": {_LONG_KEY: [1]}, "denominator": "b"}}),
            "/rows",
        ),
        (_one_row({"recipient": {"numerator": "a", _LONG_KEY: [1]}}), "/rows"),
        (
            _one_row({"unit": {"numerator": {"stat_var": {_LONG_KEY: [1]}}, "denominator": "b"}}),
            "/rows/0/unit/numerator/stat_var",
        ),
    ],
    ids=[
        "request_side_scalar_value",
        "stat_var_side_extra_member",
        "request_side_list_value",
        "unit_list_value",
        "composed_third_member",
        "row_list_value",
        "role_mapping_list_value",
        "role_composed_side_list_value",
        "role_composed_third_member",
        "third_level_of_a_unit",
    ],
)
def test_http_overlong_key_is_refused_without_echoing_it(
    overrides: dict[str, object], pointer: str
) -> None:
    """Key lengths are checked before a cell is parsed, so a cell that fails its type cannot
    carry the key above it into the pointer. A key at the third level of a unit is below the
    member the model reports."""
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish", json=_publish_body(**overrides), headers=_AUTH_HEADERS
    )
    assert response.status_code == 400, response.text[:500]
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == pointer
    assert len(response.text) < 2_000
    assert _no_artifact_store_puts(deps)


def test_publish_request_schema_admits_every_unit_form() -> None:
    """The published schema lists a cell's forms under `anyOf`, since a converted unit also
    fits the composed arm's bounds and `oneOf` would refuse it."""
    schema = PublishFrameRequest.model_json_schema()
    validator = jsonschema.Draft202012Validator(schema)
    for unit in (
        None,
        "USDollar",
        {"currency": "EUR", "prices": "constant", "base_year": 2020},
        {"numerator": "USDollar", "denominator": _PER_PERSON},
    ):
        validator.validate(_publish_body(rows=[{**_rows()[0], "unit": unit}]))


_STEP = {
    "id": "s1",
    "op": "retrieve",
    "input": {
        "id": "spec_" + "a" * 32,
        "content_version": "cv_00000000000000000000000000abc123",
        "label": "GDP",
    },
    "options": {},
}


@pytest.mark.parametrize(
    ("posted", "carried"),
    [
        pytest.param({"steps": [_STEP]}, [_STEP], id="steps"),
        pytest.param({}, None, id="no-steps-member"),
    ],
)
def test_http_posted_steps_travel_to_the_description_record(
    posted: dict[str, object], carried: list[dict[str, object]] | None
) -> None:
    deps = _Deps()
    client = _build_client(deps)

    response = client.post(
        "/api/datasets/frames/publish", json=_publish_body(**posted), headers=_AUTH_HEADERS
    )

    assert response.status_code == 200, response.text
    record = LiveDescriptionStore(store=deps.store).read_description(
        id=_SPEC_ID, content_version=_CONTENT_VERSION
    )
    assert record is not None
    steps = record.steps
    assert (None if steps is None else [step.model_dump(mode="json") for step in steps]) == carried


@pytest.mark.parametrize(
    "composed_stat_vars",
    [_COMPOSED_STAT_VARS, [], [{_LONG_KEY: [1]}]],
    ids=["an_entry", "empty", "overlong_key"],
)
def test_http_composed_stat_vars_is_an_unknown_member_and_writes_nothing(
    composed_stat_vars: list[object],
) -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(composed_stat_vars=composed_stat_vars),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text[:500]
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/composed_stat_vars"
    assert len(response.text) < 2_000
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)
