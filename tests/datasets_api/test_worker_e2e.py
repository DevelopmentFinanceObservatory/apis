"""Runs the API and worker ASGI apps with a fake task queue and shared stores.
Async and inline mints return the same manifest shape, apart from fresh version stamps."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from artifact_store import LiveArtifactInventory
from code_index.shared import get_index
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import kinds
from datasets_api.admission import MaterialisePermits
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.definition import DefinitionDeps
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.oidc import FakeOidcVerifier
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.worker_app import build_worker_app
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from graph_release.service import ReleaseService
from materialise import EnrichmentLookup
from materialise.rate_tables import RateTableCache
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from statvar_members import MemberEnumerator
from tests.engine._fake_coverage import FakeCoverageService

from ._app_factories import make_client
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_materialise_semaphore import (
    _PermitLog,
    _SwapBeforePermit,
    _swapping_release,
)
from .test_mint_statvar_set import _CSV_BODY, _HAPPY_PATH_BODY, _coverage_service, _graph
from .test_pin_door import FakeArtifactReferenceWriter
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

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
    worker_permits: MaterialisePermits | None = None,
    worker_release: ReleaseService | None = None,
    now: Clock = utc_now,
    deferred: list[Callable[[], None]] | None = None,
) -> tuple[TestClient, LiveOperationStore, list[tuple[str, int]]]:
    """The API app and a worker app sharing stores, head store and graph, as the two services
    do in deployment. The fake queue runs each enqueued operation on the worker inline and
    records it in the returned list. `worker_permits` replaces the worker's one permit.
    `worker_release` is the graph release service each run binds from. Both apps and the
    the head and operation stores read `now`. With `deferred` the queue appends each run there for the test
    to call, so the 202 answers before the worker runs."""
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    head_store = LiveHeadStore(store=version_store, now=now)
    # One database holds operations and heads, so the worker's fenced head advance reads the row.
    operation_store = LiveOperationStore(store=version_store, now=now)
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
        coverage=FakeCoverageService(coverage().current()),
        version_writer=LiveVersionRecordWriter(store=version_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        head_store=head_store,
        # One permit, as the worker's lifespan builds it: a nested acquire would stall the run.
        materialise_semaphore=(
            worker_permits if worker_permits is not None else MaterialisePermits(1)
        ),
        head_recheck_interval=timedelta(seconds=DatasetsBuildSettings().head_recheck_seconds),
        head_max_age=timedelta(seconds=DatasetsBuildSettings().head_max_age_seconds),
        now=now,
    )
    worker_client = TestClient(
        build_worker_app(
            operation_store=operation_store,
            deps=worker_deps,
            oidc_verifier=FakeOidcVerifier(),
            worker_url=_WORKER_URL,
            worker_invoker_service_account=_INVOKER,
            release=worker_release,
            now=now,
        )
    )

    forwarded: list[tuple[str, int]] = []

    def _run(operation_id: str, submission_count: int) -> None:
        resp = worker_client.post(
            f"/internal/datasets/operations/{operation_id}/run",
            json={"operation_id": operation_id, "submission_count": submission_count},
            headers={"Authorization": "Bearer t"},
        )
        assert resp.status_code == 200, resp.text

    def _forward_to_worker(operation_id: str, submission_count: int) -> None:
        forwarded.append((operation_id, submission_count))
        if deferred is None:
            _run(operation_id, submission_count)
        else:
            deferred.append(lambda: _run(operation_id, submission_count))

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
        now=now,
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


def test_a_resubmit_after_the_head_aged_out_restarts_the_done_operation(
    monkeypatch: pytest.MonkeyPatch,
):
    """A done operation whose head then passes `head_max_age` answers the next submit with a
    fresh 202 that reads pending, and its poll reaches done with a manifest. Before, the submit
    returned the done row unchanged, so the 202 read done with nothing behind it."""
    clock = {"t": datetime(2026, 10, 24, tzinfo=UTC)}
    runs: list[Callable[[], None]] = []
    api_client, operation_store, forwarded = api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport([])),
        coverage=_coverage_service,
        now=lambda: clock["t"],
        deferred=runs,
    )
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)

    first = api_client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    assert first.status_code == 202, first.text
    runs.pop()()
    operation_id = first.json()["operation"]["operation_id"]
    done = operation_store.get(operation_id)
    assert done is not None and done.state == "done"

    clock["t"] += timedelta(seconds=DatasetsBuildSettings().head_max_age_seconds + 1)
    resubmitted = api_client.post("/api/datasets", json=_HAPPY_PATH_BODY)

    assert resubmitted.status_code == 202, resubmitted.text
    assert resubmitted.json()["operation"] == {
        "operation_id": operation_id,
        "status": "pending",
        "poll": f"/api/datasets/operations/{operation_id}",
        "retry_after_ms": 1_500,
    }
    assert forwarded == [(operation_id, 0), (operation_id, 1)]
    runs.pop()()
    polled = api_client.get(resubmitted.headers["location"])
    assert polled.status_code == 200, polled.text
    assert polled.json()["operation"]["status"] == "done"
    assert polled.json()["manifest"]["content_version"].startswith("cv_")
    rebuilt = operation_store.get(operation_id)
    assert rebuilt is not None and rebuilt.confirmed_at == clock["t"]


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


def test_a_worker_run_bound_to_a_swapped_out_release_takes_no_permit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """The run bound the first release at delivery, and the service swapped to the second
    before the run's permit. The run resolved its members from the first release's code index,
    so it takes no permit and fetches nothing, and the operation is left for a later delivery,
    which binds the served release."""
    log = _PermitLog()
    permit = _SwapBeforePermit(1, log=log)
    release, swap = _swapping_release(monkeypatch, tmp_path)
    # The worker's lifespan registers the same guard.
    release.register_swap_guard(permit.every_permit)
    requests: list[httpx.Request] = []
    api_client, operation_store, forwarded = api_and_worker(
        graph=_graph(),
        sdmx=lambda: SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(requests)),
        coverage=_coverage_service,
        worker_permits=permit,
        worker_release=release,
    )
    monkeypatch.setattr(kinds.spec_id, "within_sync_budget", lambda estimate, **_: False)
    permit.before_acquire = swap

    submitted = api_client.post("/api/datasets", json=_HAPPY_PATH_BODY)

    assert submitted.status_code == 202, submitted.text
    stored = operation_store.get(submitted.json()["operation"]["operation_id"])
    # Released for redelivery, as a transient failure before the final attempt is.
    assert stored is not None and stored.state == "pending"
    assert len(forwarded) == 1
    assert log.kinds() == ["swap"]
    assert requests == []
