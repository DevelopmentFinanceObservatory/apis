"""The analysis door through the app factory: the ready body and its stored frame, a rerun
answered from its stored body, a request joining a run of its key, the pending answer, refusals
with their fix, the anchor route, and a full analysis pool.

The door's roots come from a double source (`_analysis_doubles.FakeRootSource`) in place of
`LiveRootSource`; the frame, its version and its recipe go through the live store SQL."""

from __future__ import annotations

import contextvars
import hashlib
import logging
import threading
from typing import Any

import pytest
from dfo_frame_grammar import Frame
from fastapi.testclient import TestClient

import datasets_api.analysis as analysis
import datasets_api.doors.analysis as analysis_door
import presentation_api.newest_frames as newest_frames
from assembler.ports import FrameRef
from dataset_store.snapshot import SnapshotRevoked
from datasets_api.analysis import DatasetRoot, LocatedRoot, ResolvedRoot, RootExpired, SpecRoot
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.doors.analysis import RunningAnalyses
from datasets_api.doors.publish import PublishRecordMissing
from datasets_api.version_store import LiveVersionLister
from dfo_analysis_engine.frame_ref import FrameRef as EngineFrameRef
from materialise.errors import OverCeilingError
from presentation_api.newest_frames import DatasetsNewestFrames, NewestPending
from spec_registry import LiveSpecRegistry

from ._analysis_doubles import (
    AID,
    AID_V1,
    POPULATION,
    POPULATION_V1,
    FakeRootSource,
    Stores,
    aid_frame,
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
        capabilities = client.get("/api/datasets/capabilities").json()

    assert first.status_code == 200, first.text
    body = first.json()
    assert body["execution_identity"] == capabilities["analysis_execution_identity"]
    assert body["status"] == "ready"
    member = body["ref"]["members"][0]
    assert member["output_digest"] == body["output_digest"]
    assert body["summary"]["verb"] == "quotient_of"
    assert body["rows"] == body["summary"]["rows"] == 4
    assert body["role_keys"] == ["financingRecipient", "financingRecipient_party_kind"]
    assert body["execution_identity"]["engine_version"] == "0.9.1"
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


class _GatedSource(FakeRootSource):
    """Holds every resolve until `release` is set, counting each that starts."""

    def __init__(self, base: FakeRootSource) -> None:
        super().__init__(frames=base.frames, newest=base.newest, recipes=base.recipes)
        self.started = threading.Semaphore(0)
        self.release = threading.Event()

    def resolve(self, root, *, latest, trace_id):
        self.started.release()
        assert self.release.wait(5)
        return super().resolve(root, latest=latest, trace_id=trace_id)


def test_a_verification_leaves_an_analysis_thread_free_for_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two analysis threads: a verification holding one refuses another 503, and a run still
    starts beside it."""
    stores = Stores()
    roots = _GatedSource(source(stores))
    roots.release.set()
    with _client(
        stores,
        roots,
        monkeypatch,
        analysis_max_concurrency=2,
        verify_principal=_PROBE_PRINCIPAL,
    ) as client:
        stored = _stored_pair(client)
        roots.release.clear()
        roots.started = threading.Semaphore(0)
        verify = {"tree": stored, "verify": True}
        answers: list[Any] = []
        verifying = threading.Thread(
            target=lambda: answers.append(client.post(_PATH, json=verify, headers=_PROBE))
        )
        verifying.start()
        assert roots.started.acquire(timeout=5)
        refused = client.post(_PATH, json=verify, headers=_PROBE)
        running = threading.Thread(
            target=lambda: answers.append(client.post(_PATH, json={"tree": _TREE}))
        )
        running.start()
        run_started = roots.started.acquire(timeout=5)
        roots.release.set()
        verifying.join(5)
        running.join(5)

    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"
    assert run_started
    assert [answer.status_code for answer in answers] == [200, 200]


def test_a_verification_leaves_the_last_materialise_permit_to_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two materialise permits, one held: a verification refuses 503 without taking the
    other, and a run takes it."""
    stores = Stores()
    roots = source(stores)
    shared = (DepsCell(), RunningAnalyses())
    with _client(
        stores,
        roots,
        monkeypatch,
        shared=shared,
        materialise_max_concurrency=2,
        verify_principal=_PROBE_PRINCIPAL,
    ) as client:
        permits = shared[0].get().materialise_semaphore
        assert permits.acquire_leaving(0)
        try:
            refused = client.post(_PATH, json={"tree": _TREE, "verify": True}, headers=_PROBE)
            ran = client.post(_PATH, json={"tree": _TREE})
        finally:
            permits.release()

    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"
    assert ran.status_code == 200, ran.text


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

    def read(self, root: LocatedRoot) -> Frame:
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


def test_a_vouched_request_without_a_caller_header_still_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The door spends no allowance, so a vouched request missing `x-dfo-caller` logs its
    caller as unresolved and is served."""
    monkeypatch.setenv("DFO_CALLER_PRINCIPAL", "website-secret")
    stores = Stores()
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.post(
            _PATH, json={"tree": _TREE}, headers={"x-dfo-caller-principal": "website-secret"}
        )

    assert response.status_code == 200, response.text


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

    def read(self, root: LocatedRoot) -> Frame:
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


class _PublishFailingSource(FakeRootSource):
    """Refuses every frame read with a server fault the route renders itself."""

    def __init__(self, base: FakeRootSource) -> None:
        super().__init__(frames=base.frames, newest=base.newest, recipes=base.recipes)

    def read(self, root: LocatedRoot) -> Frame:
        raise PublishRecordMissing(
            spec_id=root.dataset.spec_id, content_version=root.dataset.content_version
        )


def test_a_server_fault_the_route_renders_logs_its_traceback_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stores = Stores()
    with (
        _client(stores, _PublishFailingSource(source(stores)), monkeypatch) as client,
        caplog.at_level(logging.ERROR),
    ):
        response = client.post(_PATH, json={"tree": _TREE})

    assert response.status_code == 500, response.text
    assert response.json()["code"] == "internal_error"
    logged = [r for r in caplog.records if r.exc_info is not None]
    assert len(logged) == 1, [r.getMessage() for r in caplog.records]
    assert isinstance(logged[0].exc_info[1], PublishRecordMissing)


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


_CONTRADICTING_CUT = {"party_totals": True, "without_party_totals": True}


def _ref_contradicting_the_cut() -> dict[str, Any]:
    ref = ref_of((AID, AID_V1))
    ref["members"][0]["artifact"]["rendering"].update(_CONTRADICTING_CUT)
    return {"ref": ref}


@pytest.mark.parametrize(
    ("root", "pointer_tail"),
    [
        pytest.param(
            {**dataset(AID, AID_V1), "rendering": _CONTRADICTING_CUT},
            "/rendering/party_totals",
            id="dataset-root",
        ),
        pytest.param(
            _ref_contradicting_the_cut(), "/artifact/rendering/party_totals", id="ref-member"
        ),
    ],
)
def test_a_root_whose_party_totals_contradicts_without_party_totals_refuses(
    monkeypatch: pytest.MonkeyPatch, root: dict[str, Any], pointer_tail: str
) -> None:
    stores = Stores()
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.post(_PATH, json={"tree": step("aggregate", root, aggregation="sum")})

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
    assert response.json()["pointer"].endswith(pointer_tail)


@pytest.mark.parametrize("path", [_PATH, f"{_PATH}/anchor"])
def test_a_wrong_method_on_an_analysis_route_answers_405(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    stores = Stores()
    with _client(stores, source(stores), monkeypatch) as client:
        response = client.get(path)

    assert response.status_code == 405, response.text


_PROBE_PRINCIPAL = "probe-verify-secret"
_PROBE = {"x-dfo-principal": _PROBE_PRINCIPAL}


def _writes(stores: Stores) -> tuple[int, int]:
    """Rows written to the dataset store so far, and objects held in the artifact buckets."""
    objects = sum(len(bucket.objects) for bucket in stores.gcs.buckets.values())
    return stores.store.total_changes, objects


class _ExpiringSource(FakeRootSource):
    """`base`, with each version in `expired` refused `RootExpired` by the step it maps to
    (`locate` or `read`), refused `SnapshotRevoked` as it resolves (`revoked`), or refused
    `OverCeilingError` as it is located, a frame1 document past the decode bound (`decode`)."""

    def __init__(self, base: FakeRootSource) -> None:
        super().__init__(frames=base.frames, newest=base.newest, recipes=base.recipes)
        self.expired: dict[tuple[str, str], str] = {}

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
    ) -> ResolvedRoot:
        resolved = super().resolve(root, latest=latest, trace_id=trace_id)
        pair = (resolved.dataset.spec_id, resolved.dataset.content_version)
        if self.expired.get(pair) == "revoked":
            raise SnapshotRevoked(id=pair[0], content_version=pair[1], reason="taken down")
        return resolved

    def locate(self, root: ResolvedRoot) -> LocatedRoot:
        pair = (root.dataset.spec_id, root.dataset.content_version)
        if self.expired.get(pair) == "locate":
            raise RootExpired(spec_id=pair[0], content_version=pair[1], reason="aged out")
        if self.expired.get(pair) == "decode":
            raise OverCeilingError("frame1 too large", ceiling_kind="bytes", count=2, cap=1)
        return super().locate(root)

    def read(self, root: LocatedRoot) -> Frame:
        pair = (root.dataset.spec_id, root.dataset.content_version)
        if self.expired.get(pair) == "read":
            raise RootExpired(spec_id=pair[0], content_version=pair[1], reason="file gone")
        return super().read(root)


class _RenamingSource(FakeRootSource):
    """`base`, naming each root's frame under another pair once `renamed` is set, as an
    earlier derivation of `analysis_roots.frame_ref_for` named it."""

    def __init__(self, base: FakeRootSource) -> None:
        super().__init__(frames=base.frames, newest=base.newest, recipes=base.recipes)
        self.renamed = False

    def resolve(
        self, root: SpecRoot | DatasetRoot, *, latest: bool, trace_id: str | None
    ) -> ResolvedRoot:
        resolved = super().resolve(root, latest=latest, trace_id=trace_id)
        if not self.renamed:
            return resolved
        digest = hashlib.sha256(resolved.dataset.content_version.encode()).hexdigest()
        frame = EngineFrameRef(
            spec_id=resolved.dataset.spec_id, content_version=f"cv_{digest[:32]}"
        )
        record = {**resolved.record, "frame": frame.model_dump()}
        return ResolvedRoot(
            frame=frame,
            dataset=resolved.dataset,
            record=record,
            rendering=resolved.rendering,
            rows=resolved.rows,
        )


def _stored_pair(client: TestClient) -> dict[str, str]:
    first = client.post(_PATH, json={"tree": _TREE})
    assert first.status_code == 200, first.text
    member = first.json()["ref"]["members"][0]
    return {"spec_id": member["spec"]["spec_id"], "content_version": member["content_version"]}


def test_verify_computes_a_stored_frame_again_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verification reads its roots past the stored answer, names the frame stored under the
    tree's pair, and writes no row and no object."""
    stores = Stores()
    roots = source(stores)
    with _client(stores, roots, monkeypatch, verify_principal=_PROBE_PRINCIPAL) as client:
        stored = _stored_pair(client)
        stored_cv = client.post(_PATH, json={"tree": _TREE}).json()["frame_content_version"]
        reads, writes = roots.reads, _writes(stores)
        response = client.post(_PATH, json={"tree": stored, "verify": True}, headers=_PROBE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "verified"
    assert body["outcome"] == "same"
    assert body["frame"] == body["stored"] == stored
    assert body["frame_content_version"] == body["stored_frame_content_version"] == stored_cv
    assert body["stored_execution_identity"] == "|".join(body["execution_identity"].values())
    assert body["unavailable_roots"] == []
    assert response.headers["cache-control"] == "no-store"
    assert roots.reads == reads + 2
    assert _writes(stores) == writes


def test_verify_reports_a_frame_the_pinned_roots_no_longer_compute_as_different(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stored answer is skipped: the pinned root now holds other bytes, so the frame
    computed again differs from the stored one."""
    stores = Stores()
    roots = source(stores)
    with _client(stores, roots, monkeypatch, verify_principal=_PROBE_PRINCIPAL) as client:
        stored = _stored_pair(client)
        stored_cv = client.post(_PATH, json={"tree": _TREE}).json()["frame_content_version"]
        roots.frames[(AID, AID_V1)] = aid_frame(scale=3.0)
        writes = _writes(stores)
        response = client.post(_PATH, json={"tree": stored, "verify": True}, headers=_PROBE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "different"
    assert body["stored_frame_content_version"] == stored_cv
    assert body["frame_content_version"] != stored_cv
    assert _writes(stores) == writes


_CODES = {"locate": "artifact_expired", "read": "artifact_expired", "revoked": "snapshot_revoked"}


@pytest.mark.parametrize(
    ("aid_at", "population_at"),
    [
        ("locate", "locate"),
        ("read", "read"),
        ("revoked", "revoked"),
        ("revoked", "read"),
        ("locate", "read"),
    ],
)
def test_verify_names_every_root_whose_pinned_version_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, aid_at: str, population_at: str
) -> None:
    """Every root that cannot be read is named, whether its frame aged out before it is read,
    its file went as it is read, or its version was taken down, and a root that fails as it
    resolves leaves the others still read."""
    stores = Stores()
    roots = _ExpiringSource(source(stores))
    with _client(stores, roots, monkeypatch, verify_principal=_PROBE_PRINCIPAL) as client:
        stored = _stored_pair(client)
        roots.expired = {(AID, AID_V1): aid_at, (POPULATION, POPULATION_V1): population_at}
        writes = _writes(stores)
        response = client.post(_PATH, json={"tree": stored, "verify": True}, headers=_PROBE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "roots_unavailable"
    assert body["frame"] is None and body["frame_content_version"] is None
    assert body["stored"] == stored and body["stored_frame_content_version"] is not None
    assert sorted(
        (root["spec_id"], root["content_version"], root["code"])
        for root in body["unavailable_roots"]
    ) == [(AID, AID_V1, _CODES[aid_at]), (POPULATION, POPULATION_V1, _CODES[population_at])]
    assert _writes(stores) == writes


@pytest.mark.parametrize(
    ("aid_at", "outcome", "unavailable"),
    [
        (None, "different", []),
        ("revoked", "roots_unavailable", [(AID, AID_V1, "snapshot_revoked")]),
    ],
)
def test_verify_records_a_root_past_the_decode_bound_as_over_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    aid_at: str | None,
    outcome: str,
    unavailable: list[tuple[str, str, str]],
) -> None:
    """A root refused as located for a frame1 document past the decode bound ends the
    verification with `over_ceiling` recorded, and a root found unreadable before it is still
    named."""
    stores = Stores()
    roots = _ExpiringSource(source(stores))
    with _client(stores, roots, monkeypatch, verify_principal=_PROBE_PRINCIPAL) as client:
        stored = _stored_pair(client)
        roots.expired = {(POPULATION, POPULATION_V1): "decode"}
        if aid_at is not None:
            roots.expired[(AID, AID_V1)] = aid_at
        response = client.post(_PATH, json={"tree": stored, "verify": True}, headers=_PROBE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == outcome
    assert body["refusal_code"] == "over_ceiling"
    assert body["refusal_reason"].startswith("bytes: ")
    assert [
        (root["spec_id"], root["content_version"], root["code"])
        for root in body["unavailable_roots"]
    ] == unavailable


def test_verify_reports_a_stored_frame_whose_roots_it_named_under_older_pairs_as_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored run named its roots' frames under pairs this runner no longer derives, so the
    `quotient_of` step's rows carry another `ops_` id and the frame `cv_`s differ. The output
    digest names that variable by what was computed, so the answer reproduces."""
    stores = Stores()
    roots = _RenamingSource(source(stores))
    with _client(stores, roots, monkeypatch, verify_principal=_PROBE_PRINCIPAL) as client:
        stored = _stored_pair(client)
        roots.renamed = True
        response = client.post(_PATH, json={"tree": stored, "verify": True}, headers=_PROBE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["frame_content_version"] != body["stored_frame_content_version"]
    assert body["outcome"] == "same"
    assert body["stored_output_digest"] == body["output_digest"]


@pytest.mark.parametrize("refusal", ["output-cap", "engine"])
def test_verify_reports_a_refused_computation_as_different(
    monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    """The computation is refused, its output past the rows the publish door stores or the
    engine declining a pinned root that now holds other data: the answer does not reproduce,
    and the refusal is recorded, so the probe reads a result."""
    stores = Stores()
    roots = source(stores)
    with _client(stores, roots, monkeypatch, verify_principal=_PROBE_PRINCIPAL) as client:
        stored = _stored_pair(client)
        if refusal == "output-cap":
            monkeypatch.setattr(analysis, "PUBLISH_MAX_ROWS", 1)
        else:
            roots.frames[(POPULATION, POPULATION_V1)] = aid_frame()
        writes = _writes(stores)
        response = client.post(_PATH, json={"tree": stored, "verify": True}, headers=_PROBE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "different"
    assert body["frame"] is None
    assert body["stored_frame_content_version"] is not None
    assert body["refusal_code"] == ("over_ceiling" if refusal == "output-cap" else "role_ambiguous")
    assert body["refusal_reason"]
    assert _writes(stores) == writes


@pytest.mark.parametrize(
    "headers",
    [{}, {"x-dfo-principal": "someone-else"}, {"x-dfo-principal": "refresh-secret"}],
    ids=["no-principal", "another-principal", "refresh-principal"],
)
def test_verify_from_a_caller_without_the_verify_grant_refuses_403(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    stores = Stores()
    roots = source(stores)
    with _client(
        stores,
        roots,
        monkeypatch,
        verify_principal=_PROBE_PRINCIPAL,
        live_refresh_principal="refresh-secret",
    ) as client:
        stored = _stored_pair(client)
        reads = roots.reads
        response = client.post(_PATH, json={"tree": stored, "verify": True}, headers=headers)

    assert response.status_code == 403, response.text
    assert response.json()["code"] == "verify_not_permitted"
    assert roots.reads == reads


@pytest.mark.parametrize(
    "request_body",
    [
        {"tree": _TREE, "verify": True, "latest": True},
        {"tree": step("aggregate", {"spec_id": AID}, aggregation="sum"), "verify": True},
    ],
    ids=["latest", "spec-root"],
)
def test_verify_of_a_tree_naming_no_version_refuses_400(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, Any]
) -> None:
    stores = Stores()
    with _client(stores, source(stores), monkeypatch, verify_principal=_PROBE_PRINCIPAL) as client:
        response = client.post(_PATH, json=request_body, headers=_PROBE)

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
