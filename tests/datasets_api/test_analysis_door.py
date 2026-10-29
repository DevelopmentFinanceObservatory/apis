"""The analysis door through the app factory: the ready body and its stored frame, a rerun
answered from its stored body, a request joining a run of its key, the pending answer, refusals
with their fix, the anchor route, and a full analysis pool.

The door's roots come from a double source (`_analysis_doubles.FakeRootSource`) in place of
`LiveRootSource`; the frame, its version and its recipe go through the live store SQL."""

from __future__ import annotations

import contextvars
import threading
from typing import Any

import pytest
from dfo_frame_grammar import Frame
from fastapi.testclient import TestClient

import datasets_api.doors.analysis as analysis_door
import presentation_api.newest_frames as newest_frames
from assembler.ports import FrameRef
from datasets_api.analysis import ResolvedRoot, RootExpired
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.doors.analysis import RunningAnalyses
from datasets_api.version_store import LiveVersionLister
from presentation_api.newest_frames import DatasetsNewestFrames, NewestPending
from spec_registry import LiveSpecRegistry

from ._analysis_doubles import (
    AID,
    AID_V1,
    POPULATION,
    POPULATION_V1,
    FakeRootSource,
    Stores,
    dataset,
    ref_of,
    source,
    step,
)
from ._app_factories import make_client
from .conftest import FakeDatasetStore

_PATH = "/api/datasets/analysis"
_TREE = step(
    "quotient_of",
    dataset(AID, AID_V1),
    dataset(POPULATION, POPULATION_V1),
    on={"destination": "place"},
)


def _client(
    stores: Stores,
    roots: FakeRootSource,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared: tuple[DepsCell, RunningAnalyses] | None = None,
    **settings: Any,
) -> TestClient:
    monkeypatch.setattr(analysis_door, "LiveRootSource", lambda **_: roots)
    resolved = DatasetsBuildSettings.from_env().model_copy(update=settings)
    cell, running = shared if shared is not None else (None, None)
    return make_client(
        deps_cell=cell,
        running_analyses=running,
        registry=LiveSpecRegistry(store=FakeDatasetStore()),
        settings=resolved,
        artifact_store=stores.artifact_store,
        snapshot_reader=stores.snapshot_reader,
        version_writer=stores.version_writer,
        recipe_store=stores.recipe_store,
        version_lister=LiveVersionLister(store=stores.store),
        now=lambda: stores.now,
    )


def test_a_tree_answers_its_stored_frame_and_a_rerun_answers_the_same_reading_no_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    roots = source(stores)
    with _client(stores, roots, monkeypatch) as client:
        first = client.post(_PATH, json={"tree": _TREE})
        versions = stores.version_count()
        reads = roots.reads
        again = client.post(_PATH, json={"tree": _TREE})

    assert first.status_code == 200, first.text
    body = first.json()
    assert body["status"] == "ready"
    member = body["ref"]["members"][0]
    assert member["output_digest"] == body["output_digest"]
    assert body["summary"]["verb"] == "quotient_of"
    assert body["rows"] == body["summary"]["rows"] == 4
    assert body["role_keys"] == ["financingRecipient", "financingRecipient_party_kind"]
    assert body["execution_identity"]["engine_version"] == "0.9.0"
    assert first.headers["cache-control"] == "no-store"
    assert stores.snapshot_reader.read_version(member["spec"]["spec_id"], member["content_version"])
    assert again.json() == body
    assert stores.version_count() == versions
    assert reads == 2
    assert roots.reads == reads


def test_a_root_still_building_answers_202_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    roots = source(stores, pending={AID: "op_aid"})
    with _client(stores, roots, monkeypatch) as client:
        response = client.post(
            _PATH, json={"tree": step("aggregate", {"spec_id": AID}, aggregation="sum")}
        )

    assert response.status_code == 202, response.text
    assert response.json() == {"status": "pending", "retry_after_s": 2.0, "operations": ["op_aid"]}
    assert response.headers["retry-after"] == "2"


def test_a_dataset_of_several_readings_refuses_with_its_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    tree = step(
        "aggregate", {"ref": ref_of((AID, AID_V1), (POPULATION, POPULATION_V1))}, aggregation="sum"
    )
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.post(_PATH, json={"tree": tree})

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "several_members"
    assert problem["retry_mode"] == "correct_and_retry"
    assert problem["fix"]["spec_ids"] == [AID, POPULATION]
    assert response.headers["content-type"] == "application/problem+json"


def test_an_engine_refusal_answers_its_code_retry_mode_and_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    tree = step("aggregate", dataset(AID, AID_V1), aggregation="median_of_means")
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.post(_PATH, json={"tree": tree})

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["code"] == "operation_uncovered"
    assert problem["retry_mode"] == "correct_and_retry"
    assert "sum" in problem["fix"]["supported"]
    assert stores.version_count() == 0


def test_the_anchor_route_answers_the_latest_period_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    body = {"roots": [dataset(AID, AID_V1)], "populations": [{"spec_id": AID, "bindings": {}}]}
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.post(f"{_PATH}/anchor", json=body)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "period": "2022",
        "rule": "half1",
        "left_out": [],
        "roots": [{"spec_id": AID, "content_version": AID_V1}],
    }
    assert stores.version_count() == 0


def test_a_rolling_step_over_a_two_sided_verb_stores_its_slots_and_replays_on_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stored recipe names each frame slot in its op-specs, so a `latest` replay walks both
    sides of the `quotient_of` as the first run did and, with no newer version of either
    dataset, answers the same frame."""
    stores = Stores()
    roots = source(stores, newest={AID: AID_V1, POPULATION: POPULATION_V1})
    tree = step(
        "growth",
        _TREE,
        entity_by=["destination"],
        window_start={"anchor": "latest", "offset": -1},
        window_end={"anchor": "latest", "offset": 0},
    )
    with _client(stores, roots, monkeypatch) as client:
        first = client.post(_PATH, json={"tree": tree})
        assert first.status_code == 200, first.text
        member = first.json()["ref"]["members"][0]
        stored = {
            "spec_id": member["spec"]["spec_id"],
            "content_version": member["content_version"],
        }
        replayed = client.post(_PATH, json={"tree": stored, "latest": True})

    recipe = stores.recipe_store.read_recipe(**stored)
    assert recipe is not None
    growth = recipe.body["tree"]["op_spec"]
    quotient = recipe.body["tree"]["inputs"]["frame_ref"]["op_spec"]
    assert "frame_ref" in growth
    assert "frame_ref" in quotient
    assert "denominator_frame_ref" in quotient["parameters"]
    assert growth["parameters"]["rolling"]["population"] == [
        {
            "pair": [[{"spec_id": AID, "bindings": {}}], [{"spec_id": POPULATION, "bindings": {}}]],
            "keys": {"financingRecipient": "observationAbout"},
        }
    ]
    assert replayed.status_code == 200, replayed.text
    assert replayed.json()["ref"] == first.json()["ref"]


def test_the_anchor_route_refuses_roots_past_the_root_row_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    body = {"roots": [dataset(AID, AID_V1)], "populations": [{"spec_id": AID, "bindings": {}}]}
    with _client(stores, source(stores), monkeypatch, analysis_max_root_rows=1) as client:
        response = client.post(f"{_PATH}/anchor", json=body)

    assert response.status_code == 422, response.text
    problem = response.json()
    assert problem["retry_mode"] == "correct_and_retry"
    assert problem["fix"]["ceiling_kind"] == "analysis_root_rows"
    assert problem["fix"]["cap"] == 1


class _BlockingSource(FakeRootSource):
    """Holds every resolve until `release` is set."""

    def __init__(self, base: FakeRootSource) -> None:
        super().__init__(frames=base.frames, newest=base.newest, recipes=base.recipes)
        self.entered = threading.Event()
        self.release = threading.Event()

    def resolve(self, root, *, latest, trace_id):
        self.entered.set()
        assert self.release.wait(5)
        return super().resolve(root, latest=latest, trace_id=trace_id)


def test_a_full_analysis_pool_refuses_503(monkeypatch: pytest.MonkeyPatch) -> None:
    stores = Stores()
    roots = _BlockingSource(source(stores))
    with _client(stores, roots, monkeypatch, analysis_max_concurrency=1) as client:
        held = threading.Thread(target=lambda: client.post(_PATH, json={"tree": _TREE}))
        held.start()
        assert roots.entered.wait(5)
        refused = client.post(_PATH, json={"tree": _TREE})
        roots.release.set()
        held.join(5)

    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"


class _ReadBlockingSource(FakeRootSource):
    """Holds every frame read until `release` is set."""

    def __init__(self, base: FakeRootSource) -> None:
        super().__init__(frames=base.frames, newest=base.newest, recipes=base.recipes)
        self.entered = threading.Event()
        self.release = threading.Event()

    def read(self, root: ResolvedRoot) -> Frame:
        self.entered.set()
        assert self.release.wait(5)
        return super().read(root)


def test_a_request_whose_key_is_running_joins_it_with_202_reading_no_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = Stores()
    roots = _ReadBlockingSource(source(stores))
    with _client(stores, roots, monkeypatch) as client:
        answers: list[Any] = []
        held = threading.Thread(
            target=lambda: answers.append(client.post(_PATH, json={"tree": _TREE}))
        )
        held.start()
        assert roots.entered.wait(5)
        joined = client.post(_PATH, json={"tree": _TREE})
        reads = roots.reads
        roots.release.set()
        held.join(5)
        stored = client.post(_PATH, json={"tree": _TREE})

    assert joined.status_code == 202, joined.text
    assert joined.json() == {"status": "pending", "retry_after_s": 5.0, "operations": []}
    assert joined.headers["retry-after"] == "5"
    assert reads == 0
    assert answers[0].status_code == 200, answers[0].text
    assert stored.json() == answers[0].json()


@pytest.mark.parametrize("first", ["post", "check"])
def test_a_live_check_and_a_post_of_one_key_compute_it_once(
    monkeypatch: pytest.MonkeyPatch, first: str
) -> None:
    """A live share slot check and `POST /api/datasets/analysis` share the app's
    `RunningAnalyses`, so whichever arrives second while the key computes answers pending."""
    stores = Stores()
    roots = _ReadBlockingSource(source(stores))
    roots.release.set()
    shared = (DepsCell(), RunningAnalyses())
    monkeypatch.setattr(newest_frames, "LiveRootSource", lambda **_: roots)
    check = DatasetsNewestFrames(shared[0], running=shared[1], budget_seconds=5)
    with _client(stores, roots, monkeypatch, shared=shared) as client:
        stored = client.post(_PATH, json={"tree": _TREE})
        assert stored.status_code == 200, stored.text
        member = stored.json()["ref"]["members"][0]
        frame = FrameRef(
            spec_id=member["spec"]["spec_id"], content_version=member["content_version"]
        )
        # The request a check makes of the stored frame: its recipe on the newest versions.
        rerun = {
            "tree": {"spec_id": frame.spec_id, "content_version": frame.content_version},
            "latest": True,
        }

        def post() -> Any:
            return client.post(_PATH, json=rerun)

        def newest() -> Any:
            return check.newest(frame, rendering=None)

        held_call, second_call = (post, newest) if first == "post" else (newest, post)
        roots.entered.clear()
        roots.release.clear()
        answers: list[Any] = []
        # The check runs as the share door runs it, under the caller's bound release.
        held = threading.Thread(
            target=contextvars.copy_context().run, args=(lambda: answers.append(held_call()),)
        )
        held.start()
        assert roots.entered.wait(5)
        reads = roots.reads
        second = second_call()
        reads_while_held = roots.reads - reads
        roots.release.set()
        held.join(5)

    if first == "post":
        assert second == NewestPending(retry_after_s=5.0)
        assert answers[0].status_code == 200, answers[0].text
    else:
        assert second.status_code == 202, second.text
        assert second.json() == {"status": "pending", "retry_after_s": 5.0, "operations": []}
        assert isinstance(answers[0], FrameRef)
    assert reads_while_held == 0


class _FailingOnceSource(FakeRootSource):
    """Refuses the first frame read as expired."""

    def __init__(self, base: FakeRootSource) -> None:
        super().__init__(frames=base.frames, newest=base.newest, recipes=base.recipes)
        self.failed = False

    def read(self, root: ResolvedRoot) -> Frame:
        if not self.failed:
            self.failed = True
            raise RootExpired(
                spec_id=root.dataset.spec_id,
                content_version=root.dataset.content_version,
                reason="expired",
            )
        return super().read(root)


def test_a_run_that_fails_releases_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    stores = Stores()
    roots = _FailingOnceSource(source(stores))
    with _client(stores, roots, monkeypatch) as client:
        failed = client.post(_PATH, json={"tree": _TREE})
        retried = client.post(_PATH, json={"tree": _TREE})

    assert failed.json()["code"] == "artifact_expired", failed.text
    assert retried.status_code == 200, retried.text
    assert roots.reads == 2


def _member(spec_id: str, content_version: str) -> dict[str, Any]:
    ref = ref_of((AID, AID_V1))
    ref["members"][0]["spec"]["spec_id"] = spec_id
    ref["members"][0]["content_version"] = content_version
    return {"ref": ref}


@pytest.mark.parametrize(
    "root",
    [
        pytest.param(_member("spec_BAD", AID_V1), id="spec-id-off-grammar"),
        pytest.param(_member(AID, "cv_BAD"), id="version-off-grammar"),
        pytest.param(_member(AID, "ocv_" + "1" * 32), id="spec-id-with-a-derived-version"),
        pytest.param(_member("ops_" + "1" * 32, AID_V1), id="derived-id-with-a-dataset-version"),
    ],
)
def test_a_ref_member_off_the_id_grammar_refuses_the_request(
    monkeypatch: pytest.MonkeyPatch, root: dict[str, Any]
) -> None:
    """As a dataset or derived root off its grammar does, before any root is read."""
    stores = Stores()
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.post(_PATH, json={"tree": step("aggregate", root, aggregation="sum")})

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"


@pytest.mark.parametrize("path", [_PATH, f"{_PATH}/anchor"])
def test_a_wrong_method_on_an_analysis_route_answers_405(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    stores = Stores()
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.get(path)

    assert response.status_code == 405, response.text
