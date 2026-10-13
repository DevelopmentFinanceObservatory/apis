"""Tests recipes at the publish door and the recipe door's four routes.

One `FakeVersionRecordDatasetStore` backs the version writer, snapshot reader, recipe store, and
version lister, so the recipe SQL runs against the rows the publish door wrote. HTTP tests use
the whole Datasets surface, which also proves no other door's route answers a recipe path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import FrameLineage, LiveSnapshotReader
from datasets_api.doors.publish import PublishRecipeStoreMissing, publish_frame
from datasets_api.publish_models import MAX_RECIPE_BYTES, Recipe
from datasets_api.recipe_store import LiveRecipeStore, ReplayedFrame
from datasets_api.version_store import LiveVersionLister, LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from spec_registry import LiveSpecRegistry
from tests.artifact_store.conftest import FakeGCSClient

from ._app_factories import make_client
from ._snapshot_doubles import FakeSpecReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_SPEC_ID = "ops_abc123"
_CONTENT_VERSION = "ocv_abc123"
_OTHER_SPEC_ID = "ops_def456"
_OTHER_CONTENT_VERSION = "ocv_def456"
_NORMAL_FORM_HASH = "rnf_" + "a" * 32
_REPLAY_KEY = "rpk_" + "b" * 32
_PRINCIPAL = "publisher-principal"
_AUTH_HEADERS = {"x-dfo-principal": _PRINCIPAL}

_RECIPE_PATH = f"/api/datasets/frames/{_SPEC_ID}/{_CONTENT_VERSION}/recipe"
_REPLAY_PATH = f"/api/datasets/recipes/replays/{_REPLAY_KEY}"
_VERSIONS_PATH = "/api/datasets/spec_a/versions"


def _recipe(**overrides: object) -> dict[str, object]:
    recipe: dict[str, object] = {
        "schema_version": 1,
        "normal_form_hash": _NORMAL_FORM_HASH,
        "replay_key": _REPLAY_KEY,
        "execution_identity": "engine-3|grammar-1.0.0|reduction-2",
        "tree": {"verb": "growth", "inputs": [{"root": 0}], "args": {"periods": 1}},
        "roots": [
            {
                "spec_id": "spec_a",
                "content_version": "cv_00000000000000000000000000abc123",
                "groupby": None,
            }
        ],
        "anchor": None,
    }
    recipe.update(overrides)
    return recipe


def _anchor(**overrides: object) -> dict[str, object]:
    anchor: dict[str, object] = {
        "period": "2024",
        "rule": "half1",
        "left_out": [{"root": 0, "coordinates": {"recipient": "country/KEN"}, "label": "Kenya"}],
    }
    anchor.update(overrides)
    return anchor


_ROWS: list[dict[str, object]] = [
    {
        "stat_var": "dcid:Exemplar",
        "time_period": "2020",
        "value": 1.0,
        "recipient": "country/KEN",
    }
]


def _publish_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "spec_id": _SPEC_ID,
        "content_version": _CONTENT_VERSION,
        "role_keys": ["recipient"],
        "rows": _ROWS,
        "input_content_versions": ["cv_00000000000000000000000000abc123"],
    }
    body.update(overrides)
    return body


class _Deps:
    """Wires every store the publish and recipe doors use over one SQLite-backed store."""

    def __init__(self) -> None:
        self.store = FakeVersionRecordDatasetStore()
        self.artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=self.store),
            client=FakeGCSClient(),
        )
        self.snapshot_reader = LiveSnapshotReader(store=self.store)
        self.now = datetime(2026, 6, 1, tzinfo=UTC)
        self.version_writer = LiveVersionRecordWriter(store=self.store, now=lambda: self.now)
        self.recipe_store = LiveRecipeStore(store=self.store)
        self.version_lister = LiveVersionLister(store=self.store)

    def client(self, *, allowed_principals: frozenset[str] = frozenset({_PRINCIPAL})) -> TestClient:
        return make_client(
            registry=LiveSpecRegistry(store=FakeDatasetStore()),
            spec_reader=FakeSpecReader(),
            reference_writer=FakeArtifactReferenceWriter(),
            artifact_store=self.artifact_store,
            snapshot_reader=self.snapshot_reader,
            version_writer=self.version_writer,
            recipe_store=self.recipe_store,
            version_lister=self.version_lister,
            publish_allowed_principals=allowed_principals,
        )

    def publish(self, client: TestClient, **overrides: object):
        return client.post(
            "/api/datasets/frames/publish", json=_publish_body(**overrides), headers=_AUTH_HEADERS
        )

    def stored_recipe(self, spec_id: str = _SPEC_ID, content_version: str = _CONTENT_VERSION):
        record = self.recipe_store.read_recipe(spec_id=spec_id, content_version=content_version)
        return None if record is None else dict(record.body)

    def record_version(
        self,
        *,
        subject_id: str,
        content_version: str,
        base_content_version: str | None = None,
        renderings: tuple[dict[str, object], ...] = (),
    ) -> None:
        """Record a version with one stored file linked under it per entry of `renderings`."""
        artifacts = tuple(
            ArtifactRecord(
                artifact_digest=self.artifact_store.put(
                    f"{content_version}:{index}".encode(), tier=Tier.working
                ).digest,
                rendering=rendering,
                media_type="application/vnd.apache.parquet",
            )
            for index, rendering in enumerate(renderings)
        )
        self.version_writer.record_version(
            VersionRecord(
                subject_id=subject_id,
                subject_kind="spec",
                content_version=content_version,
                schema_version=1,
                facts=FrameLineage(input_content_versions=[]),
                artifacts=artifacts,
                base_content_version=base_content_version,
            )
        )


@pytest.fixture
def deps() -> _Deps:
    return _Deps()


def test_publish_with_a_recipe_stores_it_outside_the_version_facts(deps: _Deps) -> None:
    response = deps.publish(deps.client(), recipe=_recipe(anchor=_anchor()))

    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is True
    assert deps.stored_recipe() == _recipe(anchor=_anchor())
    assert deps.recipe_store.read_replay(replay_key=_REPLAY_KEY) == ReplayedFrame(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION
    )
    stored = deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION)
    assert stored is not None
    assert stored.facts == FrameLineage(
        input_content_versions=["cv_00000000000000000000000000abc123"], continuations=[]
    )


def test_republish_with_the_same_recipe_answers_as_a_republish(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client, recipe=_recipe())

    response = deps.publish(client, recipe=_recipe())

    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is False
    assert deps.stored_recipe() == _recipe()


def test_republish_without_a_recipe_leaves_the_stored_recipe_in_place(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client, recipe=_recipe())

    response = deps.publish(client)

    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is False
    assert deps.stored_recipe() == _recipe()


def test_republish_with_a_recipe_gives_one_to_a_frame_published_without(deps: _Deps) -> None:
    client = deps.client()
    assert deps.publish(client).status_code == 200
    assert deps.stored_recipe() is None

    response = deps.publish(client, recipe=_recipe())

    assert response.status_code == 200, response.text
    assert response.json()["published"]["created"] is False
    assert deps.stored_recipe() == _recipe()
    assert deps.recipe_store.read_replay(replay_key=_REPLAY_KEY) is not None


def test_republish_with_a_different_normal_form_hash_refuses_and_keeps_the_first(
    deps: _Deps,
) -> None:
    client = deps.client()
    deps.publish(client, recipe=_recipe())

    response = deps.publish(client, recipe=_recipe(normal_form_hash="rnf_" + "c" * 32))

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "publish_conflict"
    assert response.json()["constraint_id"] == "recipe_requires_matching_normal_form_hash"
    assert deps.stored_recipe() == _recipe()


def test_publish_with_a_recipe_and_no_recipe_store_refuses_before_any_write(deps: _Deps) -> None:
    with pytest.raises(PublishRecipeStoreMissing):
        publish_frame(
            spec_id=_SPEC_ID,
            content_version=_CONTENT_VERSION,
            role_keys=("recipient",),
            rows=_ROWS,
            input_content_versions=("cv_00000000000000000000000000abc123",),
            artifact_store=deps.artifact_store,
            snapshot_reader=deps.snapshot_reader,
            version_writer=deps.version_writer,
            recipe=Recipe.model_validate(_recipe()),
        )

    assert deps.snapshot_reader.read_version(_SPEC_ID, _CONTENT_VERSION) is None


def test_put_recipe_is_idempotent_and_readable(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client)

    first = client.put(_RECIPE_PATH, json=_recipe(anchor=_anchor()), headers=_AUTH_HEADERS)
    second = client.put(_RECIPE_PATH, json=_recipe(anchor=_anchor()), headers=_AUTH_HEADERS)
    read = client.get(_RECIPE_PATH, headers=_AUTH_HEADERS)
    replay = client.get(_REPLAY_PATH, headers=_AUTH_HEADERS)

    assert first.status_code == 200, first.text
    assert first.json() == {
        "spec_id": _SPEC_ID,
        "content_version": _CONTENT_VERSION,
        "created": True,
    }
    assert second.status_code == 200, second.text
    assert second.json()["created"] is False
    assert read.status_code == 200, read.text
    assert read.json() == _recipe(anchor=_anchor())
    assert replay.status_code == 200, replay.text
    assert replay.json() == {"spec_id": _SPEC_ID, "content_version": _CONTENT_VERSION}


def test_a_left_out_series_validates_with_and_without_a_label(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client)
    labelled = {"root": 0, "coordinates": {"recipient": "country/KEN"}, "label": "Kenya"}
    unlabelled = {"root": 1, "coordinates": {"recipient": "country/UGA"}}

    written = client.put(
        _RECIPE_PATH,
        json=_recipe(anchor=_anchor(left_out=[labelled, unlabelled])),
        headers=_AUTH_HEADERS,
    )
    read = client.get(_RECIPE_PATH, headers=_AUTH_HEADERS)

    assert written.status_code == 200, written.text
    assert read.status_code == 200, read.text
    # An absent label reads back as null, and the rest of each entry as sent.
    assert read.json()["anchor"]["left_out"] == [labelled, {**unlabelled, "label": None}]


@pytest.mark.parametrize(
    ("field", "value", "constraint_id"),
    [
        ("normal_form_hash", "rnf_" + "c" * 32, "recipe_requires_matching_normal_form_hash"),
        ("replay_key", "rpk_" + "c" * 32, "recipe_requires_matching_replay_key"),
    ],
)
def test_put_recipe_with_a_different_identity_field_is_a_conflict(
    deps: _Deps, field: str, value: str, constraint_id: str
) -> None:
    client = deps.client()
    deps.publish(client, recipe=_recipe())

    response = client.put(_RECIPE_PATH, json=_recipe(**{field: value}), headers=_AUTH_HEADERS)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "publish_conflict"
    assert response.json()["constraint_id"] == constraint_id
    assert deps.stored_recipe() == _recipe()


def test_put_recipe_keeps_the_first_body_when_only_opaque_fields_differ(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client, recipe=_recipe())

    response = client.put(_RECIPE_PATH, json=_recipe(tree={"verb": "rank"}), headers=_AUTH_HEADERS)

    assert response.status_code == 200, response.text
    assert response.json()["created"] is False
    assert deps.stored_recipe() == _recipe()


def test_put_recipe_for_an_unpublished_frame_is_not_found_and_stores_nothing(deps: _Deps) -> None:
    response = deps.client().put(_RECIPE_PATH, json=_recipe(), headers=_AUTH_HEADERS)

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "snapshot_not_found"
    assert deps.stored_recipe() is None
    assert deps.recipe_store.read_replay(replay_key=_REPLAY_KEY) is None


def test_a_second_frame_under_a_held_replay_key_is_refused(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client, recipe=_recipe())
    other = deps.publish(client, spec_id=_OTHER_SPEC_ID, content_version=_OTHER_CONTENT_VERSION)
    assert other.status_code == 200, other.text

    response = client.put(
        f"/api/datasets/frames/{_OTHER_SPEC_ID}/{_OTHER_CONTENT_VERSION}/recipe",
        json=_recipe(),
        headers=_AUTH_HEADERS,
    )

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "publish_conflict"
    assert response.json()["constraint_id"] == "replay_key_requires_one_frame"
    assert deps.stored_recipe(_OTHER_SPEC_ID, _OTHER_CONTENT_VERSION) is None
    assert deps.recipe_store.read_replay(replay_key=_REPLAY_KEY) == ReplayedFrame(
        spec_id=_SPEC_ID, content_version=_CONTENT_VERSION
    )


def test_recipe_and_replay_reads_are_not_found_when_nothing_is_stored(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client)

    recipe = client.get(_RECIPE_PATH, headers=_AUTH_HEADERS)
    replay = client.get(_REPLAY_PATH, headers=_AUTH_HEADERS)

    assert recipe.status_code == 404, recipe.text
    assert recipe.json()["code"] == "recipe_not_found"
    assert replay.status_code == 404, replay.text
    assert replay.json()["code"] == "recipe_not_found"


_FRAME1_RENDERING = {"format": "frame1"}
_STATISTICS_RENDERING = {"statistics": "series"}
_PLAIN_RENDERING = {"format": "parquet", "bundle": False, "layout": "frame"}
_GROUPBY = {"by": ["recipient"], "op": "sum"}
_LEVELLED_GROUPBY = {"by": ["recipient", "sector"], "op": "sum", "levels": {"sector": 0}}


def test_versions_list_is_newest_first_with_each_version_s_rendering(deps: _Deps) -> None:
    deps.record_version(
        subject_id="spec_a",
        content_version="cv_00000000000000000000000000000001",
        base_content_version="cv_00000000000000000000000000000000",
        renderings=(_FRAME1_RENDERING, {**_PLAIN_RENDERING, "groupby": _LEVELLED_GROUPBY}),
    )
    deps.now += timedelta(hours=1)
    # A raw version lists as plain whatever is linked under it.
    deps.record_version(
        subject_id="spec_a",
        content_version="cv_00000000000000000000000000000003",
        renderings=(_FRAME1_RENDERING, _PLAIN_RENDERING, {**_PLAIN_RENDERING, "groupby": _GROUPBY}),
    )
    deps.now += timedelta(hours=1)
    deps.record_version(
        subject_id="spec_a",
        content_version="cv_00000000000000000000000000000002",
        base_content_version="cv_00000000000000000000000000000003",
        renderings=(
            _STATISTICS_RENDERING,
            _FRAME1_RENDERING,
            {**_PLAIN_RENDERING, "groupby": _GROUPBY},
        ),
    )
    deps.now += timedelta(hours=1)
    # A collapse holding only its frame1 and statistics files records no groupby.
    deps.record_version(
        subject_id="spec_a",
        content_version="cv_00000000000000000000000000000004",
        base_content_version="cv_00000000000000000000000000000003",
        renderings=(_FRAME1_RENDERING, _STATISTICS_RENDERING),
    )
    deps.record_version(
        subject_id="spec_b",
        content_version="cv_00000000000000000000000000000009",
        base_content_version="cv_00000000000000000000000000000008",
        renderings=({**_PLAIN_RENDERING, "groupby": {"by": ["donor"], "op": "sum"}},),
    )

    response = deps.client().get(_VERSIONS_PATH, headers=_AUTH_HEADERS)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "schema_version": 1,
        "spec_id": "spec_a",
        "versions": [
            {
                "content_version": "cv_00000000000000000000000000000004",
                "base_content_version": "cv_00000000000000000000000000000003",
                "rendering": "not_recorded",
                "groupby": None,
            },
            {
                "content_version": "cv_00000000000000000000000000000002",
                "base_content_version": "cv_00000000000000000000000000000003",
                "rendering": "groupby",
                "groupby": _GROUPBY,
            },
            {
                "content_version": "cv_00000000000000000000000000000003",
                "base_content_version": None,
                "rendering": "plain",
                "groupby": None,
            },
            {
                "content_version": "cv_00000000000000000000000000000001",
                "base_content_version": "cv_00000000000000000000000000000000",
                "rendering": "groupby",
                "groupby": _LEVELLED_GROUPBY,
            },
        ],
    }


def test_versions_list_is_empty_for_a_dataset_with_no_stored_version(deps: _Deps) -> None:
    response = deps.client().get(_VERSIONS_PATH, headers=_AUTH_HEADERS)

    assert response.status_code == 200, response.text
    assert response.json()["versions"] == []


def test_versions_list_refuses_an_id_outside_the_dataset_grammar(deps: _Deps) -> None:
    response = deps.client().get("/api/datasets/ops_abc123/versions", headers=_AUTH_HEADERS)

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "not_found"


_GUARDED_REQUESTS = [
    ("PUT", _RECIPE_PATH),
    ("GET", _RECIPE_PATH),
    ("GET", _REPLAY_PATH),
    ("GET", _VERSIONS_PATH),
]


@pytest.mark.parametrize(("method", "path"), _GUARDED_REQUESTS)
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [("x-dfo-principal", "some-other-caller")],
        [("x-dfo-principal", _PRINCIPAL), ("x-dfo-principal", "some-other-caller")],
    ],
    ids=["no-principal", "unlisted-principal", "two-principals"],
)
def test_recipe_routes_refuse_a_caller_the_publish_door_refuses(
    deps: _Deps, method: str, path: str, headers: list[tuple[str, str]]
) -> None:
    client = deps.client()
    deps.publish(client, recipe=_recipe())
    # A body the route would refuse as invalid: the caller check comes first.
    body = {"not": "a recipe"} if method == "PUT" else None

    response = client.request(method, path, json=body, headers=headers)

    assert response.status_code == 403, response.text
    assert response.json()["code"] == "publish_not_permitted"


@pytest.mark.parametrize(("method", "path"), _GUARDED_REQUESTS)
def test_recipe_routes_refuse_everyone_when_no_principal_is_configured(
    deps: _Deps, method: str, path: str
) -> None:
    client = deps.client(allowed_principals=frozenset())

    response = client.request(
        method, path, json=_recipe() if method == "PUT" else None, headers=_AUTH_HEADERS
    )

    assert response.status_code == 403, response.text
    assert response.json()["code"] == "publish_not_permitted"


@pytest.mark.parametrize(
    "recipe",
    [
        _recipe(extra_field=1),
        _recipe(normal_form_hash="rnf_short"),
        _recipe(replay_key="rnf_" + "b" * 32),
        _recipe(execution_identity=""),
        _recipe(tree=["not", "an", "object"]),
        _recipe(roots={"not": "a list"}),
        _recipe(anchor=_anchor(rule="newest")),
        _recipe(anchor=_anchor(extra_field=1)),
        _recipe(anchor=_anchor(left_out=[{"root": 0, "label": "Kenya"}])),
        _recipe(anchor=_anchor(left_out=[{"root": 0, "coordinates": {}, "label": ""}])),
        _recipe(anchor=_anchor(left_out=[{"root": -1, "coordinates": {}, "label": "Kenya"}])),
        _recipe(anchor=_anchor(left_out=[{"root": 0, "coordinates": {"r": 1}, "label": "K"}])),
        _recipe(anchor=_anchor(left_out=[{"root": 0, "coordinates": {}, "label": "Kenya"}])),
        _recipe(
            anchor=_anchor(
                left_out=[{"root": 0, "coordinates": {"unit": "USD", "provenance": "src_oecd"}}]
            )
        ),
        {key: value for key, value in _recipe().items() if key != "tree"},
    ],
    ids=[
        "extra-top-level-field",
        "normal-form-hash-pattern",
        "replay-key-prefix",
        "empty-execution-identity",
        "tree-not-an-object",
        "roots-not-a-list",
        "anchor-rule",
        "anchor-extra-field",
        "left-out-missing-coordinates",
        "left-out-empty-label",
        "left-out-negative-root",
        "left-out-coordinate-not-a-string",
        "left-out-empty-coordinates",
        "left-out-coordinates-name-no-series",
        "missing-tree",
    ],
)
def test_an_invalid_recipe_is_refused_on_both_write_routes(
    deps: _Deps, recipe: dict[str, object]
) -> None:
    client = deps.client()
    deps.publish(client)

    put = client.put(_RECIPE_PATH, json=recipe, headers=_AUTH_HEADERS)
    published = deps.publish(client, recipe=recipe)

    assert (put.status_code, put.json()["code"]) == (400, "request_invalid"), put.text
    assert (published.status_code, published.json()["code"]) == (400, "request_invalid")
    assert deps.stored_recipe() is None


def test_an_unsupported_recipe_schema_version_is_refused_on_both_write_routes(deps: _Deps) -> None:
    client = deps.client()
    deps.publish(client)

    put = client.put(_RECIPE_PATH, json=_recipe(schema_version=2), headers=_AUTH_HEADERS)
    published = deps.publish(client, recipe=_recipe(schema_version=2))

    # The platform answers a body's own top-level `schema_version` with its own code.
    assert (put.status_code, put.json()["code"]) == (422, "unsupported_schema_version")
    assert (published.status_code, published.json()["code"]) == (400, "request_invalid")
    assert deps.stored_recipe() is None


def test_tree_and_roots_are_stored_as_sent_whatever_their_inner_shape(deps: _Deps) -> None:
    opaque = _recipe(
        tree={"any": [1, 2.5, None, True, {"nested": {"deeper": ["x"]}}], "unknown_key": "kept"},
        roots=["a string", 7, {"spec_id": "spec_a"}, [1, 2]],
    )
    client = deps.client()
    deps.publish(client)

    response = client.put(_RECIPE_PATH, json=opaque, headers=_AUTH_HEADERS)

    assert response.status_code == 200, response.text
    assert client.get(_RECIPE_PATH, headers=_AUTH_HEADERS).json() == opaque


def _recipe_of_compact_size(size: int) -> dict[str, object]:
    base = _recipe(tree={"pad": ""})
    padding = size - len(json.dumps(base, separators=(",", ":")).encode("utf-8"))
    return _recipe(tree={"pad": "x" * padding})


def test_a_recipe_at_the_size_cap_parses_and_one_byte_over_is_refused(deps: _Deps) -> None:
    Recipe.model_validate(_recipe_of_compact_size(MAX_RECIPE_BYTES))

    with pytest.raises(ValidationError):
        Recipe.model_validate(_recipe_of_compact_size(MAX_RECIPE_BYTES + 1))

    client = deps.client()
    deps.publish(client)
    response = client.put(
        _RECIPE_PATH, json=_recipe_of_compact_size(MAX_RECIPE_BYTES + 1), headers=_AUTH_HEADERS
    )
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
    assert deps.stored_recipe() is None


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("PUT", "/api/datasets/frames/spec_abc123/cv_00000000000000000000000000abc123/recipe"),
        ("GET", "/api/datasets/frames/ops_abc123/cv_00000000000000000000000000abc123/recipe"),
        ("GET", "/api/datasets/recipes/replays/not-a-replay-key"),
    ],
)
def test_recipe_routes_refuse_ids_outside_their_grammar(
    deps: _Deps, method: str, path: str
) -> None:
    response = deps.client().request(
        method, path, json=_recipe() if method == "PUT" else None, headers=_AUTH_HEADERS
    )

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
