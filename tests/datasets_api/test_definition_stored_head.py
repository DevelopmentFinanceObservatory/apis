"""`GET /api/datasets/{id}` answers at once from a stored head that holds the requested
rendering, past the recheck window too, up to `head_serve_stale_max_age_seconds`, and submits
the head's check to the worker (`current.serve_stored_head`). Its 200 and 304 carry
`x-dfo-freshness`, `x-dfo-content-version`, a strong ETag over the body and the freshness, and
`no-cache`.

The door, the materialiser and the operation store are real. The graph, SDMX and head store are
the currency tests' doubles, and the task queue records its tasks. The rig's task create pool is
the one the lifespan would build, held so a test waits for its creates (`_Rig.settle`).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from datetime import UTC, datetime, timedelta

import pytest

from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.dispatch import ENQUEUE_RECOVERY_SECONDS
from datasets_api.operations.head_check import HEAD_CHECK_OPTIONS
from datasets_api.operations.task_client import ENQUEUE_TIMEOUT_SECONDS, FakeTaskClient
from datasets_api.operations.task_create_pool import TaskCreatePool
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from graph_release.context import current_release

from ._operation_store import FakeOperationDatasetStore
from .test_definition_door import _CSV_HEADER, _RECIPIENT, _STAT_VAR, _STAT_VAR_REF, _mint_one_spec
from .test_definition_door_currency import (
    _FACET,
    _PAST_INTERVAL,
    _PAST_MAX_AGE,
    _client,
    _counting_transport,
    _CoverageFromFacets,
    _FakeHeadStore,
    _mutable_graph,
    _row,
)

# Past the 14-day `head_serve_stale_max_age_seconds` default.
_PAST_SERVE_STALE_CEILING = timedelta(days=15)

# Past the 7-day `head_imported_data_max_age_seconds` default, inside the ceiling.
_PAST_IMPORT_MAX_AGE = timedelta(days=10)

_IDENTITY = {"accept-encoding": "identity"}


class _RaisingTaskClient:
    """Fails every create, counting them."""

    def __init__(self) -> None:
        self.calls = 0

    def enqueue(
        self,
        *,
        operation_id: str,
        submission_count: int,
        timeout: float = ENQUEUE_TIMEOUT_SECONDS,
    ) -> None:
        self.calls += 1
        raise RuntimeError("cloud tasks unavailable")


class _SwitchableTaskClient(FakeTaskClient):
    """Records its tasks, or fails every create while `failing`."""

    failing = False

    def enqueue(
        self,
        *,
        operation_id: str,
        submission_count: int,
        timeout: float = ENQUEUE_TIMEOUT_SECONDS,
    ) -> None:
        if self.failing:
            raise RuntimeError("cloud tasks unavailable")
        super().enqueue(
            operation_id=operation_id, submission_count=submission_count, timeout=timeout
        )


class _GatedTaskClient(FakeTaskClient):
    """Records its tasks once `gate` is set, holding every create until then."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()

    def enqueue(
        self,
        *,
        operation_id: str,
        submission_count: int,
        timeout: float = ENQUEUE_TIMEOUT_SECONDS,
    ) -> None:
        assert self.gate.wait(5)
        super().enqueue(
            operation_id=operation_id, submission_count=submission_count, timeout=timeout
        )


class _Rig:
    """A minted spec behind the GET, with its head store, SDMX count and operation queue."""

    def __init__(self, *, operation_store=None, task_client=None) -> None:
        facets = {(_STAT_VAR, _RECIPIENT): [_FACET]}
        self.graph = _mutable_graph(facets)
        body_ref = {"text": _CSV_HEADER + _row(stat_var_ref=_STAT_VAR_REF, obs_value="1000")}
        transport, self.sdmx_calls = _counting_transport(body_ref)
        self.head_store = _FakeHeadStore()
        self.operation_store = (
            operation_store
            if operation_store is not None
            else LiveOperationStore(store=FakeOperationDatasetStore())
        )
        self.task_client = task_client if task_client is not None else FakeTaskClient()
        self.task_creates = TaskCreatePool()
        self.client = _client(
            graph=self.graph,
            sdmx_transport=transport,
            head_writer=self.head_store,
            coverage=_CoverageFromFacets(facets),
            operation_store=self.operation_store,
            task_client=self.task_client,
            task_creates=self.task_creates,
        )
        self.spec_id = _mint_one_spec(self.client)
        self.content_version = self.head_store.head_for(self.spec_id).content_version

    def get(self, query: str = "?bundle=false", **headers: str):
        """A GET of the rendering the mint stored, unless `query` names another, once the
        task creates earlier reads handed off have run."""
        self.settle()
        return self.client.get(f"/api/datasets/{self.spec_id}{query}", headers=headers)

    def settle(self) -> None:
        """Wait until every task create a read handed off has run."""
        assert self.task_creates.wait_idle(5)


@pytest.mark.parametrize(
    ("age", "materialised_age"),
    [(_PAST_INTERVAL, None), (_PAST_INTERVAL, _PAST_MAX_AGE)],
    ids=["past_the_recheck_window", "past_the_maximum_age"],
)
def test_a_get_past_the_window_serves_the_head_at_once_and_submits_its_check(
    age: timedelta, materialised_age: timedelta | None
):
    rig = _Rig()
    rig.head_store.age_head(rig.spec_id, age=age, materialised_age=materialised_age)
    confirmed_at = rig.head_store.head_for(rig.spec_id).confirmed_at
    graph_calls, sdmx_calls = dict(rig.graph.calls), rig.sdmx_calls.n
    confirms, advances = rig.head_store.confirm_head_calls, rig.head_store.advance_head_calls

    resp = rig.get()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trace"]["served_from"] == "head"
    assert body["trace"]["confirmed_at"] == confirmed_at.isoformat()
    assert body["content_version"] == rig.content_version
    assert resp.headers["x-dfo-freshness"] == "pending"
    assert resp.headers["x-dfo-content-version"] == rig.content_version
    assert resp.headers["cache-control"] == "no-cache"
    # No graph read, no fetch, no head write: the worker judges the head.
    assert (dict(rig.graph.calls), rig.sdmx_calls.n) == (graph_calls, sdmx_calls)
    assert rig.head_store.confirm_head_calls == confirms
    assert rig.head_store.advance_head_calls == advances
    rig.settle()
    ((operation_id, submission),) = rig.task_client.enqueued
    check = rig.operation_store.get(operation_id)
    assert check is not None
    assert (check.request_identity, check.state, submission) == (
        {"subject_id": rig.spec_id, "head_check": True},
        "pending",
        0,
    )


def test_a_get_inside_the_window_answers_current_and_submits_nothing():
    rig = _Rig()

    resp = rig.get()

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert resp.headers["x-dfo-freshness"] == "current"
    assert rig.task_client.enqueued == []


def test_a_get_past_the_serve_stale_ceiling_rebuilds_in_the_request(
    caplog: pytest.LogCaptureFixture,
):
    rig = _Rig()
    rig.head_store.age_head(rig.spec_id, age=_PAST_SERVE_STALE_CEILING)
    sdmx_calls = rig.sdmx_calls.n

    with caplog.at_level(logging.INFO):
        resp = rig.get()

    # The timings line labels the freshness on the work executor's path too.
    (timings,) = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("datasets.timings door=definition ")
    ]
    assert " freshness=current " in timings

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"
    assert resp.headers["x-dfo-freshness"] == "current"
    assert rig.sdmx_calls.n == sdmx_calls + 1
    assert rig.task_client.enqueued == []


def test_an_import_keyed_head_past_its_maximum_age_serves_at_once_and_is_checked_on_the_worker():
    rig = _Rig()
    head = rig.head_store.head_for(rig.spec_id)
    rig.head_store._heads[rig.spec_id] = dataclasses.replace(
        head,
        dependencies={
            **(head.dependencies or {}),
            "imported_data": current_release().imported_data,
        },
    )
    rig.head_store.age_head(rig.spec_id, age=_PAST_IMPORT_MAX_AGE)
    sdmx_calls, advances = rig.sdmx_calls.n, rig.head_store.advance_head_calls

    resp = rig.get()

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert resp.headers["x-dfo-freshness"] == "pending"
    assert (rig.sdmx_calls.n, rig.head_store.advance_head_calls) == (sdmx_calls, advances)
    rig.settle()
    assert len(rig.task_client.enqueued) == 1


def test_a_get_past_the_window_for_a_rendering_the_head_lacks_checks_in_the_request():
    """The read executor serves only a rendering the head holds, so the work executor judges the
    head and renders the missing one, as before."""
    rig = _Rig()
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)
    confirms = rig.head_store.confirm_head_calls

    resp = rig.get("?format=parquet&bundle=false")

    assert resp.status_code == 200, resp.text
    assert resp.headers["x-dfo-freshness"] == "current"
    assert rig.head_store.confirm_head_calls == confirms + 1
    assert rig.task_client.enqueued == []


def test_a_get_whose_check_cannot_be_recorded_serves_the_head_stale():
    """Half of the default backlog of 20 is the check's headroom, so a backlog of 10 refuses
    every check."""
    rig = _Rig(
        operation_store=LiveOperationStore(store=FakeOperationDatasetStore(), backlog_max=10)
    )
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)

    resp = rig.get()

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert resp.json()["content_version"] == rig.content_version
    assert resp.headers["x-dfo-freshness"] == "stale"
    assert resp.headers["cache-control"] == "no-cache"
    rig.settle()
    assert rig.task_client.enqueued == []


def test_a_get_answers_before_its_checks_task_is_created():
    """The GET records the check and answers `pending` while the task create is still held on
    the pool, and the create completes afterwards, once."""
    task_client = _GatedTaskClient()
    rig = _Rig(task_client=task_client)
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)

    resp = rig.client.get(f"/api/datasets/{rig.spec_id}?bundle=false")

    assert resp.status_code == 200, resp.text
    assert resp.headers["x-dfo-freshness"] == "pending"
    assert task_client.enqueued == []
    task_client.gate.set()
    rig.settle()
    ((operation_id, submission),) = task_client.enqueued
    check = rig.operation_store.get(operation_id)
    assert check is not None
    assert (check.state, check.dispatched_at is not None, submission) == ("pending", True, 0)


def test_stale_reads_against_a_failing_task_queue_all_answer_and_create_once():
    """Inside the recovery interval only the read that wrote the check hands off its task
    create, so a Cloud Tasks outage costs one short create per spec, and every read still
    answers. The first read answers before its create fails, and the rest find the row
    undispatched."""
    task_client = _RaisingTaskClient()
    rig = _Rig(task_client=task_client)
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)

    answers = [rig.get() for _ in range(5)]

    assert [resp.status_code for resp in answers] == [200] * 5
    assert [resp.headers["x-dfo-freshness"] for resp in answers] == ["pending"] + ["stale"] * 4
    rig.settle()
    assert task_client.calls == 1


def test_readers_of_every_rendering_share_one_check():
    """The check carries one rendering whatever the read asked for, so a reader of another
    rendering the head holds joins the same row and creates no second task."""
    rig = _Rig()
    held = rig.get("?format=parquet&bundle=false")
    assert held.status_code == 200, held.text
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)

    csv = rig.get()
    parquet = rig.get("?format=parquet&bundle=false")

    for resp in (csv, parquet):
        assert resp.status_code == 200, resp.text
        assert resp.json()["trace"]["served_from"] == "head"
        assert resp.headers["x-dfo-freshness"] == "pending"
    rig.settle()
    ((operation_id, _),) = rig.task_client.enqueued
    check = rig.operation_store.get(operation_id)
    assert check is not None
    assert check.options == HEAD_CHECK_OPTIONS.model_dump(mode="json")


def test_a_get_whose_check_failed_for_good_judges_the_head_in_the_request():
    """A check that ended on a refusal stays failed, so the GET takes the in-request path, which
    judges the head and answers what it finds, refusal included, in place of `stale`."""
    rig = _Rig()
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)
    assert rig.get().headers["x-dfo-freshness"] == "pending"
    rig.settle()
    ((operation_id, submission),) = rig.task_client.enqueued
    rig.operation_store.claim(
        operation_id, submission_count=submission, lease_seconds=60, final_delivery=False
    )
    rig.operation_store.fail(operation_id, attempt=1, refusal_code="over_ceiling", retryable=False)
    confirms = rig.head_store.confirm_head_calls

    resp = rig.get()

    assert resp.status_code == 200, resp.text
    assert resp.headers["x-dfo-freshness"] == "current"
    assert rig.head_store.confirm_head_calls == confirms + 1
    assert len(rig.task_client.enqueued) == 1


def test_a_revalidation_naming_the_etag_answers_304_with_the_same_headers(
    caplog: pytest.LogCaptureFixture,
):
    rig = _Rig()
    with caplog.at_level(logging.INFO, logger="datasets_api.delivery"):
        first = rig.get(**_IDENTITY)
        etag = first.headers["etag"]
        assert etag.startswith('"'), etag

        revalidated = rig.get(**_IDENTITY, **{"if-none-match": etag})

    assert revalidated.status_code == 304
    assert revalidated.content == b""
    for name in ("etag", "x-dfo-freshness", "x-dfo-content-version", "cache-control"):
        assert revalidated.headers[name] == first.headers[name], name
    # Each read logs its delivery, and the 304's line says it sent no body.
    lines = [
        fields
        for record in caplog.records
        if (fields := getattr(record, JSON_FIELDS_ATTR, None)) and fields.get("door")
    ]
    assert [(line["door"], line["revalidated"]) for line in lines] == [
        ("definition", False),
        ("definition", True),
    ]


def test_a_gzip_answer_carries_the_etag_weak_and_revalidates_on_it():
    rig = _Rig()
    strong = rig.get(**_IDENTITY).headers["etag"]
    gzipped = rig.get()
    assert gzipped.headers["etag"] == f"W/{strong}"

    revalidated = rig.get(**{"if-none-match": gzipped.headers["etag"]})

    assert revalidated.status_code == 304


def test_the_etag_moves_with_the_freshness_while_the_body_stays():
    """A cache holding the `stale` answer revalidates to a 200 once the check is submitted, so it
    never keeps `stale` for a head whose check is under way."""
    task_client = _SwitchableTaskClient()
    clock = {"t": datetime(2026, 11, 20, 9, tzinfo=UTC)}
    rig = _Rig(
        task_client=task_client,
        operation_store=LiveOperationStore(
            store=FakeOperationDatasetStore(), now=lambda: clock["t"]
        ),
    )
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)
    task_client.failing = True
    # The first read hands off the create, which fails, so the next read finds it undispatched.
    assert rig.get(**_IDENTITY).headers["x-dfo-freshness"] == "pending"
    stale = rig.get(**_IDENTITY)
    assert stale.headers["x-dfo-freshness"] == "stale"
    task_client.failing = False
    # The next read past the recovery interval retries the lost create.
    clock["t"] += timedelta(seconds=ENQUEUE_RECOVERY_SECONDS + 1)

    pending = rig.get(**_IDENTITY, **{"if-none-match": stale.headers["etag"]})

    assert pending.status_code == 200
    assert pending.headers["x-dfo-freshness"] == "pending"
    assert pending.content == stale.content
    assert pending.headers["etag"] != stale.headers["etag"]
