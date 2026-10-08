"""Tests `publish_frame` and the publish door through shared SQLite-backed stores.
One `FakeVersionRecordDatasetStore` backs the snapshot reader, version writer, and artifact
inventory, so artifact-location joins exercise the same coupling as production. HTTP tests
cover refusal mapping through `make_client_for_door`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated

import jsonschema
import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import LiveSnapshotReader, SnapshotRevoked
from datasets_api.config import DatasetsConfigError
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
    MAX_COMPOSED_STAT_VAR_SIDE_REFS,
    MAX_COMPOSED_STAT_VARS,
    MAX_INPUT_CONTENT_VERSIONS,
    MAX_ROLE_KEYS,
    MAX_ROW_WIDTH,
    MAX_ROWS,
    MAX_UNIT_ENTRIES,
    MAX_UNIT_STAT_VAR_CHARS,
    PublishFrameRequest,
)
from datasets_api.render import render_and_store_frame1
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import NullVersionRecordWriter
from materialise.errors import OverCeilingError
from materialise.formats import FRAME1_MEDIA_TYPE
from materialise.frame_build import build_frame
from tests.artifact_store.conftest import FakeGCSClient

from ._snapshot_doubles import make_client_for_door
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
        "input_content_versions": ["cv_abc123"],
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
        input_content_versions=("cv_abc123",),
        composed_stat_vars=(),
        ceiling: int | None = None,
    ):
        kwargs: dict[str, object] = {"composed_stat_vars": composed_stat_vars}
        if ceiling is not None:
            kwargs["ceiling"] = ceiling
        return publish_frame(
            spec_id=spec_id,
            content_version=content_version,
            role_keys=role_keys,
            rows=rows if rows is not None else _rows(),
            input_content_versions=input_content_versions,
            artifact_store=self.artifact_store,
            snapshot_reader=self.snapshot_reader,
            version_writer=self.version_writer,
            **kwargs,
        )


@pytest.fixture
def deps() -> _Deps:
    return _Deps()


def test_fresh_publish_records_version_row_and_readable_frame1_bytes(deps: _Deps) -> None:
    result = deps.publish()

    assert result.created is True
    assert result.spec_id == _SPEC_ID
    assert result.content_version == _CONTENT_VERSION

    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert stored.subject_kind == "spec"
    assert stored.immutable_body["op_spec_hash"] == _SPEC_ID
    assert stored.immutable_body["input_content_versions"] == ["cv_abc123"]

    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    frame1 = [loc for loc in locations if loc.media_type == FRAME1_MEDIA_TYPE]
    assert len(frame1) == 1
    assert frame1[0].artifact_digest == result.artifact_digest
    assert frame1[0].tier == "working"

    raw = deps.artifact_store.get(artifact_digest=result.artifact_digest, tier=Tier.working)
    document = json.loads(raw)
    assert document["frame_version"] == "frame1"
    assert document["columns"]
    # Column order is outside this test.
    assert len(document["rows"]) == 1


def test_repeat_publish_with_identical_bytes_is_idempotent(deps: _Deps) -> None:
    first = deps.publish()
    second = deps.publish()

    assert first.created is True
    assert second.created is False
    assert second.artifact_digest == first.artifact_digest

    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    frame1 = [loc for loc in locations if loc.media_type == FRAME1_MEDIA_TYPE]
    assert len(frame1) == 1


_COMPOSED_STAT_VARS = [
    {
        "stat_var": "dcid:Exemplar",
        "relation": "quotient",
        "first": ["dcid:Amount_GDP"],
        "second": ["dcid:Count_Person"],
    }
]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_COMPOSED_STAT_VARS, _COMPOSED_STAT_VARS),
        (_COMPOSED_STAT_VARS, ()),
        ((), _COMPOSED_STAT_VARS),
        ((), ()),
        (_COMPOSED_STAT_VARS, [{**_COMPOSED_STAT_VARS[0], "relation": "sum"}]),
    ],
    ids=[
        "with-onto-with",
        "without-onto-with",
        "with-onto-without",
        "without-onto-without",
        "another-list-onto-with",
    ],
)
def test_repeat_publish_keeps_the_body_first_written_whatever_entries_it_carries(
    deps: _Deps, first, second
) -> None:
    """`composed_stat_vars` is no part of a pair's lineage, so a repeat publish succeeds with or
    without it and the stored body is the first publish's."""
    assert deps.publish(composed_stat_vars=first).created is True
    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION).immutable_body

    assert deps.publish(composed_stat_vars=second).created is False

    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION).immutable_body == stored
    assert stored == {
        "op_spec_hash": _SPEC_ID,
        "input_content_versions": ["cv_abc123"],
        **({"composed_stat_vars": _COMPOSED_STAT_VARS} if first else {}),
    }


def test_repeat_publish_with_different_bytes_refuses_rather_than_overwrites(deps: _Deps) -> None:
    first = deps.publish(rows=_rows(1.0))

    with pytest.raises(PublishBytesDiverge) as excinfo:
        deps.publish(rows=_rows(2.0))

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    frame1 = [loc for loc in locations if loc.media_type == FRAME1_MEDIA_TYPE]
    assert len(frame1) == 1
    assert frame1[0].artifact_digest == first.artifact_digest


def test_repeat_publish_with_identical_bytes_but_different_lineage_refuses(deps: _Deps) -> None:
    """Identical bytes with different lineage raise `PublishLineageDiverge` before writing."""
    first = deps.publish(input_content_versions=("cv_aaaaaa",))

    with pytest.raises(PublishLineageDiverge) as excinfo:
        deps.publish(input_content_versions=("cv_bbbbbb",))

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    # The original lineage stays on record; the second call's own claimed
    # lineage never got recorded.
    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert stored.immutable_body["input_content_versions"] == ["cv_aaaaaa"]
    locations = deps.snapshot_reader.read_artifact_locations(_SPEC_ID, _CONTENT_VERSION)
    frame1 = [loc for loc in locations if loc.media_type == FRAME1_MEDIA_TYPE]
    assert frame1[0].artifact_digest == first.artifact_digest


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
    deps.publish(input_content_versions=("cv_aaaaaa",))
    _revoke(deps.store, subject_id=_SPEC_ID, content_version=_CONTENT_VERSION)

    with pytest.raises(SnapshotRevoked):
        deps.publish(input_content_versions=("cv_bbbbbb",))


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


def test_verify_phase_catches_a_concurrent_publish_with_different_lineage(deps: _Deps) -> None:
    """Post-write verification catches the losing publish when concurrent calls claim
    different `input_content_versions` for identical bytes.
    """
    winner = deps.publish(input_content_versions=("cv_aaaaaa",))

    blind_reader = _CheckBlindSnapshotReader(deps.snapshot_reader)
    with pytest.raises(PublishLineageDiverge) as excinfo:
        publish_frame(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            role_keys=_ROLE_KEYS,
            rows=_rows(),
            input_content_versions=("cv_bbbbbb",),
            artifact_store=deps.artifact_store,
            snapshot_reader=blind_reader,
            version_writer=deps.version_writer,
        )

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert stored.immutable_body["input_content_versions"] == ["cv_aaaaaa"]
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
        input_content_versions=("cv_abc123",),
        artifact_store=deps.artifact_store,
        snapshot_reader=blind_reader,
        version_writer=deps.version_writer,
    )

    assert result.created is False


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
            input_content_versions=("cv_abc123",),
            artifact_store=deps.artifact_store,
            snapshot_reader=blind_reader,
            version_writer=deps.version_writer,
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
            input_content_versions=("cv_abc123",),
            artifact_store=deps.artifact_store,
            snapshot_reader=deps.snapshot_reader,
            version_writer=NullVersionRecordWriter(),
        )

    assert excinfo.value.spec_id == _SPEC_ID
    assert excinfo.value.content_version == _CONTENT_VERSION

    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None

    # Re-rendering the orphaned frame recovers its content-addressed digest.
    frame = build_frame(role_keys=_ROLE_KEYS, rows=_rows())
    artifact = render_and_store_frame1(
        frame_chunks=(frame,),
        subject_id=_SPEC_ID,
        content_version=_CONTENT_VERSION,
        artifact_store=deps.artifact_store,
    )
    raw = deps.artifact_store.get(artifact_digest=artifact.artifact_digest, tier=Tier.working)
    assert json.loads(raw)["frame_version"] == "frame1"


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
    kwargs: dict[str, object] = {
        "artifact_store": deps.artifact_store,
        "snapshot_reader": deps.snapshot_reader,
        "version_writer": deps.version_writer,
        "allowed_principals": allowed_principals,
    }
    if ceiling is not None:
        kwargs["ceiling"] = ceiling
    door = build_publish_door(**kwargs)
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


@pytest.mark.parametrize(
    "missing",
    [
        ("artifact_store",),
        ("snapshot_reader",),
        ("version_writer",),
        ("artifact_store", "snapshot_reader"),
        ("artifact_store", "version_writer"),
        ("snapshot_reader", "version_writer"),
    ],
)
def test_build_publish_door_refuses_partial_injection(missing: tuple[str, ...]) -> None:
    """Partial injection of the three publish stores refuses and names each missing store."""
    deps = _Deps()
    every_dep = {
        "artifact_store": deps.artifact_store,
        "snapshot_reader": deps.snapshot_reader,
        "version_writer": deps.version_writer,
    }
    kwargs = {name: value for name, value in every_dep.items() if name not in missing}

    with pytest.raises(DatasetsConfigError) as excinfo:
        build_publish_door(**kwargs)

    # Check the clause listing missing dependencies, not the general explanation.
    detail = str(excinfo.value)
    named_clause = detail.split("left uninjected", 1)[0]
    for name in missing:
        assert name in named_clause
    for name in every_dep:
        if name not in missing:
            assert name not in named_clause


def test_build_publish_door_with_every_dependency_none_still_builds() -> None:
    """Leaving all three dependencies `None` defers their live construction to lifespan."""
    door = build_publish_door()
    assert door.router is not None


def test_http_missing_version_row_after_write_maps_to_internal_error_500() -> None:
    """A publish door explicitly composed with a `version_writer` that does
    not share `snapshot_reader`'s own store reproduces
    `PublishRecordMissing` on the wire as `internal_error`/500.
    """
    deps = _Deps()
    door = build_publish_door(
        artifact_store=deps.artifact_store,
        snapshot_reader=deps.snapshot_reader,
        version_writer=NullVersionRecordWriter(),
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
        json=_publish_body(input_content_versions=["cv_abc123"] * (MAX_INPUT_CONTENT_VERSIONS + 1)),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input_content_versions"
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_http_input_content_versions_at_exactly_the_cap_succeeds() -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(input_content_versions=["cv_abc123"] * MAX_INPUT_CONTENT_VERSIONS),
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
        input_content_versions=["cv_abc123"],
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
        input_content_versions=["cv_abc123"],
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
def test_http_composed_unit_publishes_and_frame1_holds_its_json(unit: dict[str, object]) -> None:
    deps = _Deps()
    client = _build_client(deps)
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(rows=[{**_rows()[0], "unit": unit}]),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    raw = deps.artifact_store.get(
        artifact_digest=response.json()["published"]["artifact_digest"], tier=Tier.working
    )
    document = json.loads(raw)
    assert [row[document["columns"].index("unit")] for row in document["rows"]] == [unit]


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


def _one_entry(**members: object) -> dict[str, object]:
    return {"composed_stat_vars": [{**_COMPOSED_STAT_VARS[0], **members}]}


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
        (_one_entry(**{_LONG_KEY: [1]}), "/composed_stat_vars"),
        (_one_entry(label="GDP per person"), "/composed_stat_vars"),
        (_one_entry(first=[{_LONG_KEY: [1]}]), "/composed_stat_vars/0/first/0"),
        (_one_entry(stat_var={_LONG_KEY: [1]}), "/composed_stat_vars/0/stat_var"),
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
        "composed_stat_var_entry_key",
        "composed_stat_var_unknown_member",
        "composed_stat_var_side_ref_mapping",
        "composed_stat_var_ref_mapping",
    ],
)
def test_http_overlong_key_is_refused_without_echoing_it(
    overrides: dict[str, object], pointer: str
) -> None:
    """Key lengths are checked before a cell is parsed, so a cell that fails its type cannot
    carry the key above it into the pointer. A key at the third level of a unit, or inside a
    composed StatVar's ref, is below the member the model reports. An entry of
    `composed_stat_vars` takes its four keys and no other."""
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


@pytest.mark.parametrize("relation", ["sum", "share"])
def test_http_composed_stat_vars_are_stored_in_nfc_with_the_version(relation: str) -> None:
    deps = _Deps()
    client = _build_client(deps)
    entries = [
        {
            "stat_var": "dcid:Exemplar",
            "relation": relation,
            "first": ["dcid:Cafe\u0301"],
            "second": ["dcid:Count_Person", "dcid:spec_" + "a" * 32],
        }
    ]
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(composed_stat_vars=entries),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION).immutable_body
    assert stored["composed_stat_vars"] == [{**entries[0], "first": ["dcid:Caf\u00e9"]}]


def _entries(count: int, *, refs_per_side: int = 1) -> list[dict[str, object]]:
    return [
        {
            "stat_var": f"dcid:ops_{position}",
            "relation": "sum",
            "first": [f"dcid:A{position}_{ref}" for ref in range(refs_per_side)],
            "second": [f"dcid:B{position}_{ref}" for ref in range(refs_per_side)],
        }
        for position in range(count)
    ]


def _entry(**members: object) -> list[dict[str, object]]:
    return [{**_COMPOSED_STAT_VARS[0], **members}]


@pytest.mark.parametrize(
    ("composed_stat_vars", "pointer"),
    [
        (_entries(MAX_COMPOSED_STAT_VARS + 1), "/composed_stat_vars"),
        (_entries(11, refs_per_side=3), "/composed_stat_vars"),
        (_entry(first=[]), "/composed_stat_vars/0/first"),
        (_entry(second=["Count_Person"]), "/composed_stat_vars/0/second/0"),
        (_entry(stat_var="dcid:" + "x" * 196), "/composed_stat_vars/0/stat_var"),
        (_entry(first=["dcid:" + "\u0958" * 195]), "/composed_stat_vars"),
        (_entry(relation="difference"), "/composed_stat_vars/0/relation"),
        (_entry(second=None), "/composed_stat_vars/0/second"),
        ([{"stat_var": "dcid:Exemplar", "relation": "sum"}], "/composed_stat_vars/0/first"),
        (["dcid:Exemplar"], "/composed_stat_vars/0"),
        (_COMPOSED_STAT_VARS * 2, "/composed_stat_vars"),
        (_entry(first=["dcid:Exemplar"]), "/composed_stat_vars"),
        (
            [
                {**_COMPOSED_STAT_VARS[0], "first": ["dcid:ops_later"]},
                {**_COMPOSED_STAT_VARS[0], "stat_var": "dcid:ops_later"},
            ],
            "/composed_stat_vars",
        ),
        (_entry(first=["dcid:" + "9" * 100_000]), "/composed_stat_vars/0/first/0"),
        (_entry(stat_var="dcid:a\u0000b"), "/composed_stat_vars"),
        (_entry(second=["dcid:a\u0000b"]), "/composed_stat_vars"),
    ],
    ids=[
        "seventeen_entries",
        "sixty_six_side_refs",
        "empty_side",
        "ref_without_dcid",
        "ref_over_200",
        "ref_over_200_in_nfc",
        "unknown_relation",
        "null_side",
        "missing_sides",
        "entry_not_a_mapping",
        "repeated_stat_var",
        "side_names_its_own_entry",
        "side_names_a_later_entry",
        "ref_of_100000_characters",
        "stat_var_holding_nul",
        "side_ref_holding_nul",
    ],
)
def test_http_malformed_composed_stat_vars_is_request_invalid_and_writes_nothing(
    composed_stat_vars: list[object], pointer: str
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
    assert body["pointer"] == pointer
    assert len(response.text) < 2_000
    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None
    assert _no_artifact_store_puts(deps)


def test_http_composed_stat_vars_at_the_caps_publish() -> None:
    """Sixteen entries and 64 side refs, with a side naming an earlier entry and a ref no
    entry describes."""
    deps = _Deps()
    client = _build_client(deps)
    entries = _entries(MAX_COMPOSED_STAT_VARS, refs_per_side=2)
    entries[1]["first"] = ["dcid:ops_0", "dcid:Count_Person"]
    assert sum(len(one["first"]) + len(one["second"]) for one in entries) == (
        MAX_COMPOSED_STAT_VAR_SIDE_REFS
    )
    response = client.post(
        "/api/datasets/frames/publish",
        json=_publish_body(composed_stat_vars=entries),
        headers=_AUTH_HEADERS,
    )
    assert response.status_code == 200, response.text
    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION).immutable_body
    assert stored["composed_stat_vars"] == entries
