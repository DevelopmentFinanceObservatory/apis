"""Tests over-budget submissions through the mint and definition doors. They return 202,
record one operation, and skip inline materialisation. The fixture StatVar estimates zero
rows and requests, so tests patch each door's imported `within_sync_budget` directly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from datasets_api import kinds
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import definition as definition_door_module
from datasets_api.mint_models import MintOptions
from datasets_api.operation_store import (
    MAX_REDISPATCHES,
    REDISPATCH_GRACE_SECONDS,
    LiveOperationStore,
    compute_operation_id,
    never_claimed_stale_after_seconds,
)
from datasets_api.operations.submit import OperationSubmitted, submit_over_budget
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient
from materialise.errors import OverCeilingError
from origin_catalogue.service import get_catalogue_service
from series_coverage.index import Estimate
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._operation_store import FakeOperationDatasetStore
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_mint_statvar_set import (
    _CSV_BODY,
    _HAPPY_PATH_BODY,
    _STAT_VAR_REF,
    _coverage_service,
    _graph,
)
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_SETTINGS = DatasetsBuildSettings()


def _sdmx_transport(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _force_mint_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches each mint kind's imported `within_sync_budget` binding."""
    for module in (kinds.statvar_set, kinds.spec_id, kinds.collection_id):
        monkeypatch.setattr(module, "within_sync_budget", lambda estimate, **_: False)


def _force_definition_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(definition_door_module, "within_sync_budget", lambda estimate, **_: False)


def _operation_store() -> LiveOperationStore:
    return LiveOperationStore(store=FakeOperationDatasetStore())


def _client(*, operation_store, task_client, store: FakeDatasetStore | None = None):
    store = store if store is not None else FakeDatasetStore()
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport([]))
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        operation_store=operation_store,
        task_client=task_client,
    )


def test_over_budget_mint_returns_202_with_location_and_pending_status(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_mint_over_budget(monkeypatch)
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    client = _client(operation_store=operation_store, task_client=task_client)

    resp = client.post("/api/datasets", json=_HAPPY_PATH_BODY)

    assert resp.status_code == 202, resp.text
    body = resp.json()
    operation_id = body["operation"]["operation_id"]
    assert operation_id.startswith("op_")
    assert body["operation"]["status"] == "pending"
    assert body["operation"]["poll"] == f"/api/datasets/operations/{operation_id}"
    assert resp.headers["location"] == f"/api/datasets/operations/{operation_id}"
    assert "manifest" not in body

    # Enqueued exactly once, as submission 0's task.
    assert task_client.enqueued == [(operation_id, 0)]

    # The registry write (spec_id registration) still happened, only
    # materialisation was skipped.
    stored = operation_store.get(operation_id)
    assert stored is not None
    assert stored.state == "pending"


def test_over_budget_role_bearing_mint_with_no_bound_entity_is_submitted(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unbound role means every entity, so admission submits the mint as an operation."""
    _force_mint_over_budget(monkeypatch)
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    client = _client(operation_store=operation_store, task_client=task_client)

    resp = client.post(
        "/api/datasets",
        json={"schema_version": 1, "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF]}},
    )

    assert resp.status_code == 202, resp.text
    assert task_client.enqueued == [(resp.json()["operation"]["operation_id"], 0)]


def test_repeat_submit_returns_the_same_operation_and_does_not_enqueue_a_second_task(
    monkeypatch: pytest.MonkeyPatch,
):
    _force_mint_over_budget(monkeypatch)
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    client = _client(operation_store=operation_store, task_client=task_client)

    first = client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    second = client.post("/api/datasets", json=_HAPPY_PATH_BODY)

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["operation"]["operation_id"] == second.json()["operation"]["operation_id"]
    # create_or_get found the existing pending row rather than minting a
    # second operation, and the first submit already owns its dispatch.
    assert task_client.enqueued == [(first.json()["operation"]["operation_id"], 0)]


def test_a_submit_under_a_new_catalogue_build_starts_a_new_operation(
    monkeypatch: pytest.MonkeyPatch,
):
    """A completed operation is reused only under the catalogue build its submitting request
    was bound to, and so under the same admission. Each row records that build."""
    _force_mint_over_budget(monkeypatch)
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    client = _client(operation_store=operation_store, task_client=task_client)
    first = client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    first_id = first.json()["operation"]["operation_id"]
    operation_store.claim_lease(first_id, submission_count=0, lease_seconds=60)
    operation_store.complete(first_id, attempt=1, subject_id="spec_done", content_version="cv_1")
    rebuilt = FakeCatalogue(build_id="20261006T000000Z-ef567890")
    monkeypatch.setattr(get_catalogue_service(), "snapshot", lambda: rebuilt)

    second = client.post("/api/datasets", json=_HAPPY_PATH_BODY)

    assert second.status_code == 202, second.text
    second_id = second.json()["operation"]["operation_id"]
    assert second_id != first_id
    assert second.json()["operation"]["status"] == "pending"
    assert task_client.enqueued == [(first_id, 0), (second_id, 0)]
    first_row, second_row = operation_store.get(first_id), operation_store.get(second_id)
    assert first_row is not None and second_row is not None
    assert first_row.origin_catalogue_build_id == FakeCatalogue().build_id
    assert second_row.origin_catalogue_build_id == "20261006T000000Z-ef567890"


def test_repeat_submit_after_claim_does_not_enqueue_again(monkeypatch: pytest.MonkeyPatch):
    """A resubmit of a running operation reuses its row without another enqueue."""
    _force_mint_over_budget(monkeypatch)
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    client = _client(operation_store=operation_store, task_client=task_client)

    first = client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    operation_id = first.json()["operation"]["operation_id"]
    claimed = operation_store.claim_lease(operation_id, submission_count=0, lease_seconds=1_700)
    assert claimed is not None and claimed.state == "running"

    second = client.post("/api/datasets", json=_HAPPY_PATH_BODY)

    assert second.status_code == 202
    assert second.json()["operation"]["operation_id"] == operation_id
    assert second.json()["operation"]["status"] == "running"
    # Only the first (pre-claim) submit ever enqueued.
    assert task_client.enqueued == [(operation_id, 0)]


def test_over_budget_definition_retrieval_returns_202(monkeypatch: pytest.MonkeyPatch):
    """An over-budget GET returns the same 202 operation envelope as a mint."""
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    store = FakeDatasetStore()

    # First mint inline, with the budget honoured for real (within_sync_budget
    # is not patched yet), to register a spec_id. A real mint resolves a
    # real facet, so this client needs the same coverage double
    # test_mint_statvar_set._coverage_service() gives _graph()'s StatVar
    # (facet_id/provenance/measurement_method/unit matching _graph()'s own
    # facet), not the process-wide default, which carries no row for it.
    inline_client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=_graph(),
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport([])),
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        operation_store=operation_store,
        task_client=task_client,
        coverage=_coverage_service(),
    )
    minted = inline_client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]

    # Now patch the definition door's own within_sync_budget and GET it
    # back through a fresh client, forcing every retrieval, not just
    # this one, over budget.
    _force_definition_over_budget(monkeypatch)
    client = _client(operation_store=operation_store, task_client=task_client, store=store)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["operation"]["operation_id"].startswith("op_")
    assert resp.headers["location"] == body["operation"]["poll"]


class _RaisingTaskClient:
    """Simulates a Cloud Tasks enqueue with an unknown outcome."""

    def __init__(self) -> None:
        self.calls = 0

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        self.calls += 1
        raise RuntimeError("cloud tasks unavailable")


def _submit(
    *, operation_store, task_client, options: MintOptions, stamp: str
) -> OperationSubmitted:
    with pytest.raises(OperationSubmitted) as excinfo:
        submit_over_budget(
            subject_id="spec_abc",
            options=options,
            estimate=Estimate(rows=10, requests=1),
            coverage_stamp=stamp,
            operation_store=operation_store,
            task_client=task_client,
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
        )
    return excinfo.value


def test_a_successful_enqueue_confirms_the_dispatch() -> None:
    operation_store = _operation_store()
    task_client = FakeTaskClient()

    submitted = _submit(
        operation_store=operation_store,
        task_client=task_client,
        options=MintOptions(),
        stamp="cov_stamp_1",
    )

    stored = operation_store.get(submitted.operation.id)
    assert stored is not None
    assert stored.dispatch_count == 1
    assert stored.dispatch_confirmed == 1
    assert task_client.enqueued == [(stored.id, 0)]


def test_enqueue_failure_leaves_the_operation_pending_and_answers_202() -> None:
    """An enqueue error leaves the row pending and unconfirmed because the task may exist."""
    operation_store = _operation_store()
    task_client = _RaisingTaskClient()

    submitted = _submit(
        operation_store=operation_store,
        task_client=task_client,
        options=MintOptions(),
        stamp="cov_stamp_1",
    )

    assert task_client.calls == 1
    assert submitted.operation.state == "pending"
    stored = operation_store.get(submitted.operation.id)
    assert stored is not None
    assert stored.state == "pending"
    assert stored.dispatch_count == 1
    assert stored.dispatch_confirmed == 0


class _ClaimedMeanwhileTaskClient:
    """Claims the task before simulating a timeout from Cloud Tasks."""

    def __init__(self, operation_store: LiveOperationStore) -> None:
        self._operation_store = operation_store

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        claimed = self._operation_store.claim_lease(
            operation_id, submission_count=submission_count, lease_seconds=60
        )
        assert claimed is not None
        raise TimeoutError("create_task deadline exceeded")


def test_enqueue_failure_never_fails_an_operation_a_worker_is_running() -> None:
    operation_store = _operation_store()

    submitted = _submit(
        operation_store=operation_store,
        task_client=_ClaimedMeanwhileTaskClient(operation_store),
        options=MintOptions(),
        stamp="cov_stamp_1",
    )

    stored = operation_store.get(submitted.operation.id)
    assert stored is not None
    assert stored.state == "running"
    assert stored.refusal_code is None


def test_resubmit_redispatches_an_unconfirmed_enqueue_only_after_the_grace() -> None:
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = LiveOperationStore(
        store=FakeOperationDatasetStore(), now=lambda: now_box["t"]
    )
    options = MintOptions()
    _submit(
        operation_store=operation_store,
        task_client=_RaisingTaskClient(),
        options=options,
        stamp="cov_stamp_1",
    )

    # Inside the grace the first create may still be in flight.
    task_client = FakeTaskClient()
    _submit(
        operation_store=operation_store,
        task_client=task_client,
        options=options,
        stamp="cov_stamp_1",
    )
    assert task_client.enqueued == []

    now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
    submitted = _submit(
        operation_store=operation_store,
        task_client=task_client,
        options=options,
        stamp="cov_stamp_1",
    )
    assert task_client.enqueued == [(submitted.operation.id, 0)]
    stored = operation_store.get(submitted.operation.id)
    assert stored is not None
    assert stored.dispatch_count == 2
    assert stored.dispatch_confirmed == 2


def test_resubmit_of_a_never_dispatched_pending_row_enqueues_it() -> None:
    """A pending row with `dispatch_count = 0` receives its first task on resubmit."""
    operation_store = _operation_store()
    options = MintOptions()
    estimate = Estimate(rows=10, requests=1)
    stamp = "cov_stamp_1"

    # Stand in for the row existing with no task ever enqueued for it, by
    # calling create_or_get directly rather than through submit_over_budget.
    operation = operation_store.create_or_get(
        request_identity={"subject_id": "spec_abc"},
        options=options.model_dump(mode="json"),
        coverage_stamp=stamp,
        origin_catalogue_build_id="fake-catalogue",
        estimate=estimate,
    )
    assert operation.state == "pending"

    task_client = FakeTaskClient()
    _submit(operation_store=operation_store, task_client=task_client, options=options, stamp=stamp)

    assert task_client.enqueued == [(operation.id, 0)]


def test_resubmit_after_a_retryable_failure_resets_the_row_and_enqueues_under_a_new_name() -> None:
    """A retryable failure resets the row and increments `submission_count`.

    The new count gives the task a new name, so Cloud Tasks deduplication cannot swallow it.
    """
    operation_store = _operation_store()
    options = MintOptions()
    stamp = "cov_stamp_1"
    first = _submit(
        operation_store=operation_store, task_client=FakeTaskClient(), options=options, stamp=stamp
    )
    operation_id = first.operation.id
    operation_store.claim_lease(operation_id, submission_count=0, lease_seconds=60)
    operation_store.fail(operation_id, attempt=1, refusal_code="internal_error", retryable=True)

    task_client = FakeTaskClient()
    resubmitted = _submit(
        operation_store=operation_store, task_client=task_client, options=options, stamp=stamp
    ).operation

    assert resubmitted.id == operation_id
    assert resubmitted.state == "pending"
    assert resubmitted.submission_count == 1
    # submission_count 1, not 0, a different task name than the first
    # submission's own "op_id-0".
    assert task_client.enqueued == [(operation_id, 1)]


def test_resubmit_of_a_deterministically_failed_operation_stays_failed() -> None:
    """A non-retryable failure remains failed after resubmission."""
    operation_store = _operation_store()
    options = MintOptions()
    estimate = Estimate(rows=10, requests=1)
    stamp = "cov_stamp_1"

    operation = operation_store.create_or_get(
        request_identity={"subject_id": "spec_abc"},
        options=options.model_dump(mode="json"),
        coverage_stamp=stamp,
        origin_catalogue_build_id="fake-catalogue",
        estimate=estimate,
    )
    operation_store.claim_lease(operation.id, submission_count=0, lease_seconds=60)
    operation_store.fail(operation.id, attempt=1, refusal_code="excel_limit", retryable=False)

    task_client = FakeTaskClient()
    with pytest.raises(OperationSubmitted) as excinfo:
        submit_over_budget(
            subject_id="spec_abc",
            options=options,
            estimate=estimate,
            coverage_stamp=stamp,
            operation_store=operation_store,
            task_client=task_client,
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
        )

    assert excinfo.value.operation.state == "failed"
    assert excinfo.value.operation.refusal_code == "excel_limit"
    assert excinfo.value.operation.submission_count == 0
    # Deterministically failed and unchanged, never enqueued again.
    assert task_client.enqueued == []


def _submit_members(operation_store, *, members: list[str] | None):
    with pytest.raises(OperationSubmitted) as excinfo:
        submit_over_budget(
            subject_id="spec_abc",
            options=MintOptions(),
            estimate=Estimate(rows=10, requests=1),
            coverage_stamp="cov_stamp_1",
            operation_store=operation_store,
            task_client=FakeTaskClient(),
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
            members=members,
        )
    return excinfo.value.operation


def test_a_failure_recorded_for_one_member_set_is_not_returned_to_another() -> None:
    """A definition's resolved members are part of its operation id."""
    operation_store = _operation_store()
    unseeded = _submit_members(operation_store, members=["dcid:sv_b", "dcid:sv_c"])
    operation_store.claim_lease(unseeded.id, submission_count=0, lease_seconds=60)
    operation_store.fail(unseeded.id, attempt=1, refusal_code="over_ceiling", retryable=False)

    seeded = _submit_members(operation_store, members=["dcid:sv_root"])
    subject_only = _submit_members(operation_store, members=None)

    assert seeded.id != unseeded.id
    assert seeded.state == "pending"
    assert seeded.request_identity == {"subject_id": "spec_abc", "member_dcids": ["sv_root"]}
    # A row recorded with the subject alone has its own id as well.
    assert subject_only.id not in {seeded.id, unseeded.id}
    assert subject_only.request_identity == {"subject_id": "spec_abc"}
    assert _submit_members(operation_store, members=["dcid:sv_b", "dcid:sv_c"]).state == "failed"


def test_the_same_members_share_one_operation_in_any_order_or_form() -> None:
    operation_store = _operation_store()

    first = _submit_members(operation_store, members=["dcid:sv_b", "dcid:sv_a"])
    second = _submit_members(operation_store, members=["sv_a", "sv_b", "sv_a"])

    assert second.id == first.id
    assert first.request_identity["member_dcids"] == ["sv_a", "sv_b"]


def _operation_id(*, subject_id: str, stamp: str, options: MintOptions) -> str:
    return compute_operation_id(
        request_identity={"subject_id": subject_id},
        coverage_stamp=stamp,
        origin_catalogue_build_id="fake-catalogue",
        options=options.model_dump(mode="json"),
    )


def test_over_row_ceiling_refuses_before_any_operation_or_enqueue() -> None:
    """An estimate above the row ceiling refuses before creating an operation or task."""
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    options = MintOptions()
    estimate = Estimate(rows=_SETTINGS.async_max_rows + 1, requests=1)
    stamp = "cov_stamp_async_rows"

    with pytest.raises(OverCeilingError) as excinfo:
        submit_over_budget(
            subject_id="spec_async_rows",
            options=options,
            estimate=estimate,
            coverage_stamp=stamp,
            operation_store=operation_store,
            task_client=task_client,
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
        )

    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.status == 422
    assert excinfo.value.ceiling_kind == "operation_rows"
    assert excinfo.value.count == _SETTINGS.async_max_rows + 1
    assert excinfo.value.cap == _SETTINGS.async_max_rows
    operation_id = _operation_id(subject_id="spec_async_rows", stamp=stamp, options=options)
    assert operation_store.get(operation_id) is None
    assert task_client.enqueued == []


def test_row_ceiling_at_the_cap_is_admitted() -> None:
    """An estimate at the row ceiling is admitted and enqueued."""
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    options = MintOptions()
    estimate = Estimate(rows=_SETTINGS.async_max_rows, requests=1)
    stamp = "cov_stamp_async_rows_at_cap"

    with pytest.raises(OperationSubmitted) as excinfo:
        submit_over_budget(
            subject_id="spec_async_rows_at_cap",
            options=options,
            estimate=estimate,
            coverage_stamp=stamp,
            operation_store=operation_store,
            task_client=task_client,
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
        )

    operation = excinfo.value.operation
    assert operation.state == "pending"
    assert task_client.enqueued == [(operation.id, 0)]


def test_over_request_ceiling_refuses_before_any_operation_or_enqueue() -> None:
    """An estimate above the request ceiling refuses before operation creation or enqueue."""
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    options = MintOptions()
    estimate = Estimate(rows=1, requests=_SETTINGS.async_max_requests + 1)
    stamp = "cov_stamp_async_requests"

    with pytest.raises(OverCeilingError) as excinfo:
        submit_over_budget(
            subject_id="spec_async_requests",
            options=options,
            estimate=estimate,
            coverage_stamp=stamp,
            operation_store=operation_store,
            task_client=task_client,
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
        )

    assert excinfo.value.ceiling_kind == "operation_requests"
    assert excinfo.value.count == _SETTINGS.async_max_requests + 1
    assert excinfo.value.cap == _SETTINGS.async_max_requests
    operation_id = _operation_id(subject_id="spec_async_requests", stamp=stamp, options=options)
    assert operation_store.get(operation_id) is None
    assert task_client.enqueued == []


def test_request_ceiling_at_the_cap_is_admitted() -> None:
    """An estimate at the request ceiling is admitted and enqueued."""
    operation_store = _operation_store()
    task_client = FakeTaskClient()
    options = MintOptions()
    estimate = Estimate(rows=1, requests=_SETTINGS.async_max_requests)
    stamp = "cov_stamp_async_requests_at_cap"

    with pytest.raises(OperationSubmitted) as excinfo:
        submit_over_budget(
            subject_id="spec_async_requests_at_cap",
            options=options,
            estimate=estimate,
            coverage_stamp=stamp,
            operation_store=operation_store,
            task_client=task_client,
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
        )

    operation = excinfo.value.operation
    assert operation.state == "pending"
    assert task_client.enqueued == [(operation.id, 0)]


def test_over_row_ceiling_takes_priority_over_request_ceiling_reporting() -> None:
    """The row ceiling takes precedence when both limits are exceeded."""
    estimate = Estimate(
        rows=_SETTINGS.async_max_rows + 1, requests=_SETTINGS.async_max_requests + 1
    )
    with pytest.raises(OverCeilingError) as excinfo:
        submit_over_budget(
            subject_id="spec_async_both",
            options=MintOptions(),
            estimate=estimate,
            coverage_stamp="cov_stamp_async_both",
            operation_store=_operation_store(),
            task_client=FakeTaskClient(),
            within_budget=False,
            async_max_rows=_SETTINGS.async_max_rows,
            async_max_requests=_SETTINGS.async_max_requests,
        )
    assert excinfo.value.ceiling_kind == "operation_rows"


def test_a_resubmit_after_every_dispatch_was_lost_starts_a_new_submission() -> None:
    """After all dispatches expire, a resubmit starts a new submission and task name."""
    now_box = {"t": datetime(2026, 1, 1, tzinfo=UTC)}
    operation_store = LiveOperationStore(
        store=FakeOperationDatasetStore(), now=lambda: now_box["t"]
    )
    options = MintOptions()
    raising = _RaisingTaskClient()
    for _ in range(MAX_REDISPATCHES + 1):
        _submit(
            operation_store=operation_store, task_client=raising, options=options, stamp="cov_1"
        )
        now_box["t"] = now_box["t"] + timedelta(seconds=REDISPATCH_GRACE_SECONDS + 1)
    assert raising.calls == MAX_REDISPATCHES + 1
    now_box["t"] = now_box["t"] + timedelta(
        seconds=never_claimed_stale_after_seconds(
            backlog_max=_SETTINGS.operation_backlog_max,
            dispatch_concurrency=_SETTINGS.operation_dispatch_concurrency,
            max_retry_duration_seconds=_SETTINGS.operation_max_retry_duration_seconds,
        )
    )

    task_client = FakeTaskClient()
    submitted = _submit(
        operation_store=operation_store, task_client=task_client, options=options, stamp="cov_1"
    )

    assert submitted.operation.state == "pending"
    assert submitted.operation.submission_count == 1
    assert task_client.enqueued == [(submitted.operation.id, 1)]
