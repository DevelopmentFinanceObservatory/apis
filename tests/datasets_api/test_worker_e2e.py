"""Runs the API and worker ASGI apps with a fake task queue and shared stores.
Async and inline mints return the same manifest shape, apart from fresh version stamps."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from artifact_store import LiveArtifactInventory
from code_index.shared import get_index
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import kinds
from datasets_api import serve as serve_module
from datasets_api.admission import MaterialisePermits
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.oidc import FakeOidcVerifier
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker_app import build_worker_app
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup
from materialise.rate_tables import RateTableCache
from series_coverage.shared import CoverageSnapshot
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from statvar_members import MemberEnumerator
from tests.engine._fake_coverage import FakeCoverageService

from ._app_factories import make_client
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_materialise_semaphore import _CoverageBuilds, _PermitLog, _reloading_permit
from .test_mint_statvar_set import _CSV_BODY, _HAPPY_PATH_BODY, _coverage_service, _graph
from .test_pin_door import FakeArtifactReferenceWriter
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore


class _WorkerCoverageService(FakeCoverageService):
    """Expose the worker's `snapshot_current()` read on the static fixture."""

    def snapshot_current(self) -> CoverageSnapshot:
        return self.snapshot()


_BASE_URL = "https://datacommons.one.org/core/api"
_WORKER_URL = "https://worker.example.org"
_INVOKER = "worker-invoker@example.iam.gserviceaccount.com"
_PRINCIPAL = "e2e-test-principal"


def _sdmx_transport(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def api_and_worker(
    *,
    graph,
    sdmx: Callable[[], SdmxClient],
    coverage: Callable[[], FakeCoverageService],
    worker_coverage: object | None = None,
    worker_permits: MaterialisePermits | None = None,
) -> tuple[TestClient, LiveOperationStore, list[tuple[str, int]]]:
    """The API app and a worker app sharing stores, head store and graph, as the two services
    do in deployment. The fake queue runs each enqueued operation on the worker inline and
    records it in the returned list. `worker_coverage` and `worker_permits` replace the
    worker's static coverage and its one permit."""
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    head_store = LiveHeadStore(store=version_store)
    # One database holds operations and heads, so the worker's fenced head advance reads the row.
    operation_store = LiveOperationStore(store=version_store)
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    worker_deps = DefinitionDeps(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx(),
        artifact_store=artifact_store,
        materialiser_build="test-build",
        capability_version="dset_test",
        ceiling=1_000_000,
        member_concurrency=4,
        enumerator=MemberEnumerator(),
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        code_index=get_index(),
        rate_cache=RateTableCache(),
        # The worker probes role-bearing facets from this index.
        coverage=(
            worker_coverage
            if worker_coverage is not None
            else _WorkerCoverageService(coverage().current())
        ),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        head_store=head_store,
        # One permit, as the worker's lifespan builds it: a nested acquire would stall the run.
        materialise_semaphore=(
            worker_permits if worker_permits is not None else MaterialisePermits(1)
        ),
        head_recheck_interval=timedelta(seconds=DatasetsBuildSettings().head_recheck_seconds),
        head_max_age=timedelta(seconds=DatasetsBuildSettings().head_max_age_seconds),
    )
    worker_client = TestClient(
        build_worker_app(
            operation_store=operation_store,
            deps=worker_deps,
            oidc_verifier=FakeOidcVerifier(),
            worker_url=_WORKER_URL,
            worker_invoker_service_account=_INVOKER,
        )
    )

    forwarded: list[tuple[str, int]] = []

    def _forward_to_worker(operation_id: str, submission_count: int) -> None:
        forwarded.append((operation_id, submission_count))
        resp = worker_client.post(
            f"/internal/datasets/operations/{operation_id}/run",
            json={"operation_id": operation_id, "submission_count": submission_count},
            headers={"Authorization": "Bearer t"},
        )
        assert resp.status_code == 200, resp.text

    api_client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx(),
        artifact_store=artifact_store,
        snapshot_reader=LiveSnapshotReader(store=version_store),
        version_writer=LiveVersionRecordWriter(store=version_store),
        head_writer=head_store,
        reference_writer=FakeArtifactReferenceWriter(),
        operation_store=operation_store,
        task_client=FakeTaskClient(on_enqueue=_forward_to_worker),
        pin_allowed_principals=frozenset({_PRINCIPAL}),
        coverage=coverage(),
    )
    return api_client, operation_store, forwarded


def test_local_end_to_end_over_budget_mint_completes_through_the_worker(
    monkeypatch: pytest.MonkeyPatch,
):
    api_client, operation_store, forwarded = api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport([])),
        coverage=_coverage_service,
    )

    # Force this mint over budget so it takes the worker path.
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)

    submitted = api_client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    assert submitted.status_code == 202, submitted.text
    operation_id = submitted.json()["operation"]["operation_id"]
    assert submitted.headers["location"] == f"/api/datasets/operations/{operation_id}"

    # The fake queue runs the worker inline. Poll the API route as a client would.
    assert forwarded == [(operation_id, 0)]
    stored = operation_store.get(operation_id)
    assert stored is not None
    assert stored.state == "done"
    assert stored.dispatched_at is not None

    polled = api_client.get(f"/api/datasets/operations/{operation_id}")
    assert polled.status_code == 200, polled.text
    body = polled.json()
    assert body["operation"]["status"] == "done"
    assert "progress" not in body["operation"]
    manifest = body["manifest"]
    assert manifest["content_version"].startswith("cv_")
    assert manifest["definition"]["spec_id"].startswith("spec_")
    assert len(manifest["artifacts"]) >= 1
    assert stored.confirmed_at is not None
    assert (manifest["trace"]["served_from"], manifest["trace"]["confirmed_at"]) == (
        "materialise",
        stored.confirmed_at.isoformat(),
    )

    # Pin the completed operation's `(subject_id, content_version)` as for a sync mint.
    pin_resp = api_client.post(
        f"/api/datasets/{manifest['definition']['spec_id']}/{manifest['content_version']}/pin",
        json={"schema_version": 1, "artifact_digest": manifest["artifacts"][0]["artifact_digest"]},
        headers={"x-dfo-principal": _PRINCIPAL},
    )
    assert pin_resp.status_code == 200, pin_resp.text


def _rendered(manifest: dict) -> list[tuple[str, str]]:
    return sorted(
        (json.dumps(artifact["rendering"], sort_keys=True), artifact["artifact_digest"])
        for artifact in manifest["artifacts"]
    )


@pytest.mark.parametrize(
    ("options", "other_options"),
    [
        pytest.param({"bundle": False}, {"bundle": False, "format": "parquet"}, id="bare"),
        pytest.param({}, {"bundle": False}, id="bundle"),
    ],
)
def test_worker_completed_poll_lists_the_artifacts_a_synchronous_mint_lists(
    monkeypatch: pytest.MonkeyPatch, options: dict, other_options: dict
):
    """With another rendering stored under the same version, the done operation's poll and the
    synchronous mint of the same request list the same `(rendering, artifact_digest)` pairs."""
    api_client, _operation_store, _forwarded = api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport([])),
        coverage=_coverage_service,
    )
    body = {**_HAPPY_PATH_BODY, "options": options}

    other = api_client.post("/api/datasets", json={**_HAPPY_PATH_BODY, "options": other_options})
    assert other.status_code == 200, other.text

    with monkeypatch.context() as mp:
        mp.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
        submitted = api_client.post("/api/datasets", json=body)
    assert submitted.status_code == 202, submitted.text
    polled = api_client.get(submitted.headers["location"])
    assert polled.status_code == 200, polled.text
    async_manifest = polled.json()["manifest"]

    synchronous = api_client.post("/api/datasets", json=body)
    assert synchronous.status_code == 200, synchronous.text
    sync_manifest = synchronous.json()

    assert async_manifest["content_version"] == other.json()["content_version"]
    assert sync_manifest["content_version"] == async_manifest["content_version"]
    assert _rendered(async_manifest) == _rendered(sync_manifest)


def test_a_get_after_the_worker_ran_an_over_budget_mint_is_a_head_hit(
    monkeypatch: pytest.MonkeyPatch,
):
    """The worker advances the head inside `serve_current`, so the next GET of a dataset too
    large to mint inline serves the head with no SDMX request."""
    requests: list[httpx.Request] = []
    api_client, _operation_store, forwarded = api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(requests)),
        coverage=_coverage_service,
    )
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)

    submitted = api_client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    assert submitted.status_code == 202, submitted.text
    assert len(forwarded) == 1
    manifest = api_client.get(submitted.headers["location"]).json()["manifest"]
    fetched = len(requests)

    got = api_client.get(f"/api/datasets/{manifest['definition']['spec_id']}?bundle=false")

    assert got.status_code == 200, got.text
    assert got.json()["trace"]["served_from"] == "head"
    assert got.json()["content_version"] == manifest["content_version"]
    assert len(requests) == fetched


def test_a_worker_run_whose_snapshot_a_reload_replaced_materialises_on_the_served_build(
    monkeypatch: pytest.MonkeyPatch,
):
    """The run read build 1 and a reload to build 2 completed before its permit. The run gives
    the permit back, starts again from the pointer read, and fetches against build 2's index,
    after build 1's index has left memory, as its first SDMX request observes."""
    log = _PermitLog()
    builds = _CoverageBuilds(monkeypatch, make_index=lambda: _coverage_service().current())
    permit, schedule = _reloading_permit(log, builds)
    # The worker's lifespan registers the same guard.
    builds.service.register_load_guard(permit.every_permit)
    old_index_resident: list[bool] = []
    fetched_on_served_index: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        old_index_resident.append(builds.resident(1))
        log.record("sdmx")
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    real_prewrite = serve_module.prewrite_definition

    def observing_prewrite(**kwargs):
        fetched_on_served_index.append(kwargs["coverage_index"] is builds.indexes[2]())
        return real_prewrite(**kwargs)

    monkeypatch.setattr(serve_module, "prewrite_definition", observing_prewrite)
    api_client, operation_store, _forwarded = api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)),
        coverage=_coverage_service,
        worker_coverage=builds.service,
        worker_permits=permit,
    )
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
    schedule(1)

    submitted = api_client.post("/api/datasets", json=_HAPPY_PATH_BODY)

    assert submitted.status_code == 202, submitted.text
    stored = operation_store.get(submitted.json()["operation"]["operation_id"])
    assert stored is not None and stored.state == "done"
    assert fetched_on_served_index == [True]
    assert old_index_resident and not any(old_index_resident)
    kinds_logged = [kind for kind in log.kinds() if kind != "write"]
    assert kinds_logged[:5] == ["reload", "acquire", "release", "acquire", "sdmx"]
    assert kinds_logged.count("acquire") == kinds_logged.count("release") == 2
