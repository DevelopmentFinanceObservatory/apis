"""Pins worker-route authentication, claim outcomes, submission-count validation, and status mapping.
Route tests stub `run_operation`. Dependency-construction tests exercise `_live_deps_factory`.
A successful route response reports the operation's stored terminal state."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from fastapi.testclient import TestClient

from artifact_store import ArtifactStore
from code_index.shared import get_index
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.admission import MaterialisePermits
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.meta import _resolve_operation
from datasets_api.operation_store import LiveOperationStore, Operation, OperationStore
from datasets_api.operations.oidc import FakeOidcVerifier, OidcVerifier
from datasets_api.operations.worker import (
    RUN_DEADLINE_SECONDS,
    OperationSuperseded,
    OperationTransientFailure,
)
from datasets_api.operations.worker_app import (
    RUN_ROUTE_PATH,
    _live_deps_factory,
    build_worker_app,
)
from datasets_api.publish_models import MAX_ROWS as _MATERIALISE_RAW_FETCH_GUARD_ROWS
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dc_sdmx import SdmxClient
from materialise.graph import GraphReader
from origin_catalogue import Catalogue, current_catalogue
from origin_catalogue.service import CatalogueService
from series_coverage.index import Estimate
from series_coverage.shared import ServedCoverage
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from ._operation_store import FakeOperationDatasetStore

_CONFIRMED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_WORKER_URL = "https://worker.example.org"
_INVOKER = "worker-invoker@example.iam.gserviceaccount.com"
_SUBJECT_ID = "spec_abc123"


def _operation_store() -> LiveOperationStore:
    return LiveOperationStore(store=FakeOperationDatasetStore())


def _submit(operation_store: OperationStore, *, subject_id: str = _SUBJECT_ID) -> str:
    operation = operation_store.create_or_get(
        request_identity={"subject_id": subject_id},
        options={"format": "csv"},
        coverage_stamp="cov_1",
        origin_catalogue_build_id="fake-catalogue",
        estimate=Estimate(rows=200_000, requests=5),
    )
    return operation.id


def _run_path(operation_id: str) -> str:
    return RUN_ROUTE_PATH.format(operation_id=operation_id)


def _task_body(submission_count: int = 0) -> dict[str, object]:
    return {"operation_id": "ignored", "submission_count": submission_count}


def _complete_run(
    *, operation: Operation, deps: object, operation_store: OperationStore, **_: object
) -> None:
    operation_store.complete(
        operation.id,
        attempt=operation.attempt_count,
        subject_id=_SUBJECT_ID,
        content_version="cv_1",
        confirmed_at=_CONFIRMED_AT,
    )


def _client(
    *,
    operation_store: OperationStore,
    run_in_thread: Callable[..., None] = _complete_run,
    oidc_verifier: OidcVerifier | None = None,
    max_attempts: int = 5,
    catalogue: CatalogueService | None = None,
) -> TestClient:
    app = build_worker_app(
        operation_store=operation_store,
        deps=object(),  # never read: every test injects its own run_in_thread
        oidc_verifier=oidc_verifier if oidc_verifier is not None else FakeOidcVerifier(),
        worker_url=_WORKER_URL,
        worker_invoker_service_account=_INVOKER,
        max_attempts=max_attempts,
        run_in_thread=run_in_thread,
        catalogue=catalogue,
    )
    return TestClient(app)


def _catalogue_service(*builds: Catalogue | Exception) -> CatalogueService:
    """A service whose first load, at construction, and each later load serve `builds` in
    order. Its `fake://` pointer never resolves, so `snapshot_current` always reloads."""
    queue = list(builds)

    def load(_uri: str) -> Catalogue:
        build = queue.pop(0)
        if isinstance(build, Exception):
            raise build
        return build

    return CatalogueService(artifact_uri="fake://origin-catalogue", recheck_seconds=0, loader=load)


class TestAuthorization:
    def test_missing_bearer_token_is_401(self):
        client = _client(operation_store=_operation_store())

        resp = client.post(_run_path("op_unknown"))

        assert resp.status_code == 401

    def test_rejected_token_is_401(self):
        client = _client(
            operation_store=_operation_store(), oidc_verifier=FakeOidcVerifier.rejecting()
        )

        resp = client.post(
            _run_path("op_unknown"), json=_task_body(), headers={"Authorization": "Bearer anything"}
        )

        assert resp.status_code == 401

    def test_token_outside_the_accepted_set_is_401(self):
        client = _client(
            operation_store=_operation_store(),
            oidc_verifier=FakeOidcVerifier(valid_tokens=frozenset({"good-token"})),
        )

        resp = client.post(
            _run_path("op_unknown"),
            json=_task_body(),
            headers={"Authorization": "Bearer wrong-token"},
        )

        assert resp.status_code == 401

    def test_accepted_token_passes_through_to_lease_logic(self):
        operation_store = _operation_store()
        client = _client(
            operation_store=operation_store,
            oidc_verifier=FakeOidcVerifier(valid_tokens=frozenset({"good-token"})),
        )

        # The accepted token reaches lease handling and returns 404 for the unknown operation.
        resp = client.post(
            _run_path("op_unknown"),
            json=_task_body(),
            headers={"Authorization": "Bearer good-token"},
        )

        assert resp.status_code == 404


class TestOriginCatalogue:
    def _bound_build(
        self, operation_store: OperationStore, operation_id: str, catalogue: CatalogueService
    ) -> list[str]:
        bound: list[str] = []

        def run(*, operation, deps, operation_store, **_) -> None:
            bound.append(current_catalogue().build_id)
            _complete_run(operation=operation, deps=deps, operation_store=operation_store)

        client = _client(operation_store=operation_store, run_in_thread=run, catalogue=catalogue)
        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )
        assert resp.status_code == 200
        return bound

    def test_operation_runs_under_the_loaded_build_its_row_names_without_reading_the_pointer(
        self,
    ):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        # A second load raises, so reaching `snapshot_current` would answer 503.
        catalogue = _catalogue_service(
            FakeCatalogue(build_id="fake-catalogue"), OSError("bucket unreachable")
        )

        assert self._bound_build(operation_store, operation_id, catalogue) == ["fake-catalogue"]

    def test_operation_naming_another_build_runs_under_the_published_one(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        catalogue = _catalogue_service(
            FakeCatalogue(build_id="loaded-build"), FakeCatalogue(build_id="published-build")
        )

        assert self._bound_build(operation_store, operation_id, catalogue) == ["published-build"]

    def test_a_row_without_a_build_runs_under_the_published_one(self):
        dataset_store = FakeOperationDatasetStore()
        operation_store = LiveOperationStore(store=dataset_store)
        operation_id = _submit(operation_store)
        with dataset_store.transaction() as tx:
            tx.execute(
                "UPDATE datasets.operation SET origin_catalogue_build_id = NULL WHERE id = %s",
                (operation_id,),
            )
        catalogue = _catalogue_service(
            FakeCatalogue(build_id="fake-catalogue"), FakeCatalogue(build_id="published-build")
        )

        assert self._bound_build(operation_store, operation_id, catalogue) == ["published-build"]

    @pytest.mark.parametrize(
        ("retry_count", "status", "state", "retryable"),
        [
            pytest.param("0", 503, "pending", None, id="released-for-the-retry"),
            pytest.param("4", 200, "failed", True, id="final-attempt-fails-retryable"),
        ],
    )
    def test_an_unreadable_published_catalogue_is_a_transient_failure_of_the_claim(
        self, retry_count: str, status: int, state: str, retryable: bool | None
    ):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        ran: list[int] = []
        client = _client(
            operation_store=operation_store,
            run_in_thread=lambda **kw: ran.append(1),
            catalogue=_catalogue_service(
                FakeCatalogue(build_id="loaded-build"), OSError("bucket unreachable")
            ),
        )

        resp = client.post(
            _run_path(operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t", "X-CloudTasks-TaskRetryCount": retry_count},
        )

        assert resp.status_code == status
        assert ran == []
        stored = operation_store.get(operation_id)
        assert stored is not None
        assert (stored.state, stored.attempt_count, stored.retryable) == (state, 1, retryable)


class TestLeaseSemantics:
    def test_unknown_operation_is_404(self):
        client = _client(operation_store=_operation_store())

        resp = client.post(
            _run_path("op_unknown"), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 404

    def test_pending_operation_claims_the_lease_and_runs(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        ran: list[str] = []

        def run(*, operation, deps, operation_store, **_) -> None:
            ran.append(operation.id)
            assert operation.state == "running"  # already claimed by the route
            operation_store.complete(
                operation.id,
                attempt=operation.attempt_count,
                subject_id=_SUBJECT_ID,
                content_version="cv_1",
                confirmed_at=_CONFIRMED_AT,
            )

        client = _client(operation_store=operation_store, run_in_thread=run)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"status": "done"}
        assert ran == [operation_id]
        assert operation_store.get(operation_id).state == "done"

    def test_live_lease_from_a_concurrent_attempt_is_non_2xx(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        claimed = operation_store.claim(
            operation_id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        assert claimed is not None
        ran: list[int] = []
        client = _client(operation_store=operation_store, run_in_thread=lambda **kw: ran.append(1))

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code >= 300
        assert ran == []  # never ran: the lease was never claimed by this call

    def test_done_operation_is_200_and_does_not_rerun(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        operation_store.claim(
            operation_id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        operation_store.complete(
            operation_id,
            attempt=1,
            subject_id=_SUBJECT_ID,
            content_version="cv_1",
            confirmed_at=_CONFIRMED_AT,
        )
        ran: list[int] = []
        client = _client(operation_store=operation_store, run_in_thread=lambda **kw: ran.append(1))

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"status": "done"}
        assert ran == []

    def test_failed_operation_is_200_and_does_not_rerun(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        operation_store.claim(
            operation_id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        operation_store.fail(
            operation_id, attempt=1, refusal_code="member_not_resolvable", retryable=False
        )
        ran: list[int] = []
        client = _client(operation_store=operation_store, run_in_thread=lambda **kw: ran.append(1))

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"status": "failed"}
        assert ran == []

    def test_a_task_from_an_earlier_submission_is_superseded_and_claims_nothing(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        operation_store.claim(
            operation_id, submission_count=0, lease_seconds=1_500, final_delivery=False
        )
        operation_store.fail(operation_id, attempt=1, refusal_code="internal_error", retryable=True)
        assert _submit(operation_store) == operation_id  # resets to submission 1
        ran: list[int] = []
        client = _client(operation_store=operation_store, run_in_thread=lambda **kw: ran.append(1))

        resp = client.post(
            _run_path(operation_id), json=_task_body(0), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"status": "superseded"}
        assert ran == []
        current = operation_store.get(operation_id)
        assert current.state == "pending"
        assert current.submission_count == 1


class _ClaimSpy:
    """Wraps an `OperationStore`, counting `claim` calls. Everything
    else forwards straight through to `inner`.
    """

    def __init__(self, inner: OperationStore) -> None:
        self._inner = inner
        self.claim_calls = 0

    def claim(
        self, id: str, *, submission_count: int, lease_seconds: int, final_delivery: bool
    ) -> Operation | None:
        self.claim_calls += 1
        return self._inner.claim(
            id,
            submission_count=submission_count,
            lease_seconds=lease_seconds,
            final_delivery=final_delivery,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class TestMalformedSubmissionCount:
    """Missing, non-integer, negative, and future counts return 400. Older counts are superseded.

    Missing or invalid fields fail before `claim`. A future count fails its claim."""

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"operation_id": "ignored"}, id="missing_field"),
            pytest.param({"operation_id": "ignored", "submission_count": "0"}, id="non_integer"),
            pytest.param({"operation_id": "ignored", "submission_count": -1}, id="negative"),
        ],
    )
    def test_malformed_submission_count_is_400_and_claims_nothing(
        self, body: dict[str, object], caplog: pytest.LogCaptureFixture
    ) -> None:
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        spy = _ClaimSpy(operation_store)
        ran: list[int] = []
        client = _client(
            operation_store=cast(OperationStore, spy), run_in_thread=lambda **kw: ran.append(1)
        )

        with caplog.at_level(logging.ERROR):
            resp = client.post(
                _run_path(operation_id), json=body, headers={"Authorization": "Bearer t"}
            )

        assert resp.status_code == 400
        assert ran == []
        assert spy.claim_calls == 0
        assert operation_store.get(operation_id).state == "pending"
        assert operation_id in caplog.text

    def test_a_task_naming_a_submission_ahead_of_the_row_is_400_not_superseded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        operation_store = _operation_store()
        operation_id = _submit(operation_store)  # submission_count starts at 0
        ran: list[int] = []
        client = _client(operation_store=operation_store, run_in_thread=lambda **kw: ran.append(1))

        with caplog.at_level(logging.ERROR):
            resp = client.post(
                _run_path(operation_id), json=_task_body(5), headers={"Authorization": "Bearer t"}
            )

        assert resp.status_code == 400
        assert ran == []
        current = operation_store.get(operation_id)
        assert current.state == "pending"
        assert current.submission_count == 0
        assert operation_id in caplog.text


class TestRunOutcomes:
    def test_transient_failure_is_non_2xx_and_leaves_the_claim_to_its_runner(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)

        def run(*, operation, deps, operation_store, **_) -> None:
            raise OperationTransientFailure("upstream SDMX call failed")

        client = _client(operation_store=operation_store, run_in_thread=run)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code >= 300
        # `run_operation` releases a transient failure itself; this stub never did.
        assert operation_store.get(operation_id).state == "running"

    def test_deterministic_failure_calls_fail_and_returns_200(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)

        def run(*, operation, deps, operation_store, **_) -> None:
            operation_store.fail(
                operation.id,
                attempt=operation.attempt_count,
                refusal_code="member_not_resolvable",
                retryable=False,
            )

        client = _client(operation_store=operation_store, run_in_thread=run)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"status": "failed"}
        stored = operation_store.get(operation_id)
        assert stored.state == "failed"
        assert stored.refusal_code == "member_not_resolvable"

    def test_a_run_that_lost_its_fence_is_200_superseded(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)

        def run(*, operation, deps, operation_store, **_) -> None:
            raise OperationSuperseded("a newer attempt owns the row")

        client = _client(operation_store=operation_store, run_in_thread=run)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"status": "superseded"}

    def test_unexpected_crash_is_500_and_leaves_the_claim_until_its_lease_ends(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)

        def run(*, operation, deps, operation_store, **_) -> None:
            raise RuntimeError("unexpected bug")

        client = _client(operation_store=operation_store, run_in_thread=run)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 500
        assert operation_store.get(operation_id).state == "running"


class TestFinalAttempt:
    """The route's own `is_final_attempt`/`deadline` wiring into
    `run_in_thread` and the `final_delivery` it stores with the claim, both
    computed from `X-CloudTasks-TaskRetryCount` against `max_attempts`.
    `run_operation`'s own behavior once it receives `is_final_attempt` is
    `test_worker.py`'s concern, not this module's.
    """

    def test_missing_retry_count_header_is_attempt_zero_not_final(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        captured: dict[str, object] = {}

        def run(*, operation, deps, operation_store, deadline, is_final_attempt) -> None:
            captured["deadline"] = deadline
            captured["is_final_attempt"] = is_final_attempt
            captured["final_delivery"] = operation.final_delivery
            operation_store.complete(
                operation.id,
                attempt=operation.attempt_count,
                subject_id=_SUBJECT_ID,
                content_version="cv_1",
                confirmed_at=_CONFIRMED_AT,
            )

        client = _client(operation_store=operation_store, run_in_thread=run, max_attempts=5)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert captured["is_final_attempt"] is False
        assert captured["final_delivery"] is False
        remaining = cast(float, captured["deadline"]) - time.monotonic()
        assert RUN_DEADLINE_SECONDS - 60 < remaining <= RUN_DEADLINE_SECONDS

    def test_the_work_deadline_counts_from_the_claim_time(self):
        """A claim stored 100 s before the route reads the clock leaves 100 s less to run."""
        operation_store = LiveOperationStore(
            store=FakeOperationDatasetStore(),
            now=lambda: datetime.now(UTC) - timedelta(seconds=100),
        )
        operation_id = _submit(operation_store)
        captured: dict[str, float] = {}

        def run(*, operation, deps, operation_store, deadline, is_final_attempt) -> None:
            captured["remaining"] = deadline - time.monotonic()
            _complete_run(operation=operation, deps=deps, operation_store=operation_store)

        client = _client(operation_store=operation_store, run_in_thread=run)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert resp.status_code == 200
        assert RUN_DEADLINE_SECONDS - 160 < captured["remaining"] <= RUN_DEADLINE_SECONDS - 100

    def test_a_poll_past_the_work_deadline_leaves_the_run_its_lease_for_the_final_writes(self):
        claimed_at = datetime.now(UTC)
        now_box = {"t": claimed_at}
        operation_store = LiveOperationStore(
            store=FakeOperationDatasetStore(), now=lambda: now_box["t"]
        )
        operation_id = _submit(operation_store)
        polled: list[str] = []

        def run(*, operation, deps, operation_store, deadline, is_final_attempt) -> None:
            # The poll and the store's fenced abandon both read the clock past the deadline.
            now_box["t"] = claimed_at + timedelta(seconds=RUN_DEADLINE_SECONDS + 1)
            resource = _resolve_operation(
                operation_id=operation.id,
                operation_store=operation_store,
                task_client=None,
                spec_reader=None,
                snapshot_reader=None,
                now=lambda: now_box["t"],
                settings=DatasetsBuildSettings(),
            )
            polled.append(resource.operation.status)
            _complete_run(operation=operation, deps=deps, operation_store=operation_store)

        client = _client(operation_store=operation_store, run_in_thread=run)

        resp = client.post(
            _run_path(operation_id), json=_task_body(), headers={"Authorization": "Bearer t"}
        )

        assert polled == ["running"]
        assert (resp.status_code, resp.json()) == (200, {"status": "done"})

    def test_retry_count_below_max_minus_one_is_not_final(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        captured: dict[str, object] = {}

        def run(*, operation, deps, operation_store, deadline, is_final_attempt) -> None:
            captured["is_final_attempt"] = is_final_attempt
            operation_store.complete(
                operation.id,
                attempt=operation.attempt_count,
                subject_id=_SUBJECT_ID,
                content_version="cv_1",
                confirmed_at=_CONFIRMED_AT,
            )

        client = _client(operation_store=operation_store, run_in_thread=run, max_attempts=5)

        resp = client.post(
            _run_path(operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t", "X-CloudTasks-TaskRetryCount": "3"},
        )

        assert resp.status_code == 200
        assert captured["is_final_attempt"] is False

    def test_retry_count_at_max_minus_one_is_final(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        captured: dict[str, object] = {}

        def run(*, operation, deps, operation_store, deadline, is_final_attempt) -> None:
            captured["is_final_attempt"] = is_final_attempt
            captured["final_delivery"] = operation.final_delivery
            operation_store.complete(
                operation.id,
                attempt=operation.attempt_count,
                subject_id=_SUBJECT_ID,
                content_version="cv_1",
                confirmed_at=_CONFIRMED_AT,
            )

        client = _client(operation_store=operation_store, run_in_thread=run, max_attempts=5)

        resp = client.post(
            _run_path(operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t", "X-CloudTasks-TaskRetryCount": "4"},
        )

        assert resp.status_code == 200
        assert captured["is_final_attempt"] is True
        assert captured["final_delivery"] is True

    def test_retry_count_past_max_minus_one_is_still_final(self):
        """Cloud Tasks itself never dispatches past max_attempts, but a
        malformed or stale header should not read as "not final" and
        answer 503 for a retry that will never come."""
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        captured: dict[str, object] = {}

        def run(*, operation, deps, operation_store, deadline, is_final_attempt) -> None:
            captured["is_final_attempt"] = is_final_attempt
            operation_store.complete(
                operation.id,
                attempt=operation.attempt_count,
                subject_id=_SUBJECT_ID,
                content_version="cv_1",
                confirmed_at=_CONFIRMED_AT,
            )

        client = _client(operation_store=operation_store, run_in_thread=run, max_attempts=5)

        resp = client.post(
            _run_path(operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t", "X-CloudTasks-TaskRetryCount": "99"},
        )

        assert resp.status_code == 200
        assert captured["is_final_attempt"] is True

    def test_unparseable_retry_count_header_reads_as_zero(self):
        operation_store = _operation_store()
        operation_id = _submit(operation_store)
        captured: dict[str, object] = {}

        def run(*, operation, deps, operation_store, deadline, is_final_attempt) -> None:
            captured["is_final_attempt"] = is_final_attempt
            operation_store.complete(
                operation.id,
                attempt=operation.attempt_count,
                subject_id=_SUBJECT_ID,
                content_version="cv_1",
                confirmed_at=_CONFIRMED_AT,
            )

        client = _client(operation_store=operation_store, run_in_thread=run, max_attempts=5)

        resp = client.post(
            _run_path(operation_id),
            json=_task_body(),
            headers={"Authorization": "Bearer t", "X-CloudTasks-TaskRetryCount": "not-a-number"},
        )

        assert resp.status_code == 200
        assert captured["is_final_attempt"] is False


class TestLiveDefinitionDepsConstruction:
    """`_live_deps_factory` is the pure assembly step the
    worker's own `lifespan` calls once its live sub-dependencies (graph,
    SDMX, store, artifact store, coverage) are built.
    Exercised here directly, with cheap stand-ins for those sub-
    dependencies, since none of them do any I/O merely by being
    constructed or referenced."""

    _semaphore = MaterialisePermits(1)

    def _factory(self):
        return _live_deps_factory(
            store=FakeOperationDatasetStore(),
            graph=cast(GraphReader, object()),
            sdmx=SdmxClient(base_url="https://example.org"),
            artifact_store=cast(ArtifactStore, object()),
            coverage=cast(ServedCoverage, object()),
            materialise_semaphore=self._semaphore,
        )

    def _deps(self):
        return self._factory()(code_index=get_index())

    def test_ceiling_is_the_configured_raw_fetch_guard(self):
        """The same `datasets_api.publish_models.MAX_ROWS` the mint/
        definition doors build their own `DefinitionDeps` from, not the
        worker's own hardcoded `1_000_000`."""
        assert self._deps().ceiling == _MATERIALISE_RAW_FETCH_GUARD_ROWS

    def test_member_concurrency_is_the_configured_setting(self):
        """The same `DATASETS_MINT_MEMBER_MAX_CONCURRENCY` `sync_estimate`
        prices `lanes` from for the same request, not the worker's own
        hardcoded `8`, so the worker's own dispatch concurrency can never
        diverge from what admission priced."""
        assert (
            self._deps().member_concurrency == DatasetsBuildSettings().mint_member_max_concurrency
        )

    def test_materialisation_takes_the_process_semaphore(self):
        """The worker takes a materialise permit, as the doors do."""
        assert self._deps().materialise_semaphore is self._semaphore

    def test_the_factory_holds_no_code_index(self):
        """Each run completes the factory with the index its own bindings read, so the
        long-lived factory pins no index across a release swap."""
        factory = self._factory()
        index = get_index()

        assert "code_index" not in factory.keywords
        assert factory(code_index=index).code_index is index

    def test_runs_share_the_process_rebuild_flights(self):
        """One head-advancing rebuild per subject at a time holds across the process's runs."""
        factory = self._factory()
        index = get_index()

        assert factory(code_index=index).rebuilds is factory(code_index=index).rebuilds

    def test_snapshot_reader_is_live(self):
        """`run_operation` reads artifact locations through it before completing a row."""
        assert isinstance(self._deps().snapshot_reader, LiveSnapshotReader)

    def test_runs_record_the_surface_build_and_capability_version(self, monkeypatch):
        """A worker run records and stamps the release that produced it, as the API does."""
        monkeypatch.setenv("DFO_BUILD_COMMIT", "abc123")

        deps = self._deps()

        assert (deps.materialiser_build, deps.capability_version) == (
            "abc123",
            DATASETS_CAPABILITY_VERSION,
        )
