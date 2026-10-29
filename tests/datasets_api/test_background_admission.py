"""Background mint admission (`datasets_api.background_admission`): a mint posted with
`X-DFO-Mint-Admission: background` is admitted only without waiting and only while it leaves a
place for foreground callers, and is refused 503 `mint_background_declined` otherwise. Once
admitted it runs as a foreground mint does, so a foreground caller of the same subject joins
it. A foreground caller takes every limit as usual.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from datetime import UTC, datetime
from typing import cast

import httpx
import pytest

import datasets_api.render as render_module
from datasets_api.admission import MaterialisePermits, RebuildFlights, bounded_materialise_slot
from datasets_api.background_admission import (
    BACKGROUND_HEADER,
    HeadroomSemaphore,
    leaving_headroom,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.door_executor import build_door_executor
from datasets_api.doors.mint_errors import MintBackgroundDeclined, MintBackpressure
from datasets_api.operation_store import LiveOperationStore, OperationBacklogFull
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.version_store import LiveHeadStore
from graph_release.context import bound_release
from graph_release.service import Release
from series_coverage.index import Estimate

from ._operation_store import FakeOperationDatasetStore
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client as _head_client
from .test_materialise_semaphore import _ServedRelease
from .test_mint_concurrency import (
    _BODY,
    _client,
    _coverage_service,
    _default_graph,
    _sdmx_transport,
)
from .test_mint_statvar_set import _HAPPY_PATH_BODY
from .test_operation_submit_door import _client as _operations_client
from .test_operation_submit_door import _force_mint_over_budget
from .test_serve_groupby_collapse import _CSV_BODY as _GROUPBY_CSV_BODY
from .test_serve_groupby_collapse import _code_index as _groupby_code_index
from .test_serve_groupby_collapse import _coverage_service as _groupby_coverage_service
from .test_serve_groupby_collapse import _graph as _groupby_graph
from .test_serve_groupby_collapse import _spec_id_and_sidecar

_BACKGROUND = {BACKGROUND_HEADER: "background"}


@pytest.mark.parametrize(
    ("units", "held", "headroom", "taken"),
    [
        pytest.param(2, 0, 1, True, id="spare-unit"),
        pytest.param(2, 1, 1, False, id="last-unit-kept"),
        pytest.param(1, 0, 1, False, id="single-unit-kept"),
        pytest.param(1, 0, 0, True, id="no-headroom-takes-the-last"),
    ],
)
def test_acquire_leaving_takes_a_unit_only_above_the_headroom(units, held, headroom, taken):
    semaphore = HeadroomSemaphore(units)
    for _ in range(held):
        assert semaphore.acquire(blocking=False)

    assert semaphore.acquire_leaving(headroom) is taken
    # A refused attempt takes nothing, so the units it left are still there to take.
    free = units - held - (1 if taken else 0)
    assert [semaphore.acquire(blocking=False) for _ in range(free)] == [True] * free
    assert semaphore.acquire(blocking=False) is False


@pytest.mark.parametrize(("held", "admitted"), [(0, True), (1, False)])
def test_the_work_slot_admits_a_background_job_only_with_a_slot_to_spare(held, admitted):
    """With two slots and `held` taken, a background job runs only when one stays free after
    it, and a refused one is never submitted."""
    executor = build_door_executor(2, "background")
    for _ in range(held):
        assert executor.semaphore.acquire(blocking=False)
    ran: list[bool] = []

    async def _run() -> None:
        await executor.run(lambda: ran.append(True), refuse=MintBackgroundDeclined, headroom=1)

    try:
        if admitted:
            asyncio.run(_run())
        else:
            with pytest.raises(MintBackgroundDeclined):
                asyncio.run(_run())
        assert ran == ([True] if admitted else [])
        # A foreground job takes the last slot either way.
        asyncio.run(executor.run(lambda: ran.append(True), refuse=MintBackpressure))
    finally:
        executor.executor.shutdown(wait=True)


@pytest.mark.parametrize(
    ("headroom", "held", "outcome"),
    [
        pytest.param(1, 0, None, id="background-spare-permit"),
        pytest.param(1, 1, MintBackgroundDeclined, id="background-last-permit"),
        pytest.param(0, 1, None, id="foreground-last-permit"),
    ],
)
def test_the_materialise_permit_admits_a_background_mint_only_with_a_permit_to_spare(
    headroom, held, outcome
):
    """Of two permits with `held` taken, a background mint enters only when one stays free,
    and its refusal comes at once although the wait setting is ten seconds."""
    permits = MaterialisePermits(2)
    for _ in range(held):
        assert permits.acquire(timeout=0)
    entered: list[bool] = []
    started = time.monotonic()
    with leaving_headroom(headroom):
        if outcome is None:
            with bounded_materialise_slot(permits, wait_seconds=10):
                entered.append(True)
        else:
            with pytest.raises(outcome), bounded_materialise_slot(permits, wait_seconds=10):
                entered.append(True)
    assert time.monotonic() - started < 1.0
    assert entered == ([True] if outcome is None else [])
    # The slot released what it took: every permit not held by the test is free again.
    assert [permits.acquire(timeout=0) for _ in range(2 - held)] == [True] * (2 - held)


@pytest.mark.parametrize(
    ("background", "status"),
    [(True, 503), (False, 200)],
)
def test_a_one_slot_work_executor_refuses_a_background_mint_and_serves_a_foreground_one(
    background, status
):
    """A work executor of one has no slot to spare, so a background mint answers 503 with the
    background constraint at once, and a foreground mint is served."""
    client = _client(
        graph=_default_graph(),
        work_max_concurrency=1,
        settings=DatasetsBuildSettings(mint_timeout_seconds=30),
    )
    response = client.post("/api/datasets", json=_BODY, headers=_BACKGROUND if background else None)

    assert response.status_code == status, response.text
    if background:
        problem = response.json()
        assert problem["code"] == "service_unavailable"
        assert problem["constraint_id"] == "mint_background_declined"
        assert problem["retryable"] is True
        assert response.headers["retry-after"] == "1"


def test_a_background_mint_declined_at_the_permit_is_served_from_the_head_a_foreground_wrote():
    """With two work slots and one materialise permit, a background mint holds its work slot
    but has no permit to spare, so it is declined. Once a foreground mint wrote the head, the
    same background mint is answered from the head, which needs no permit."""
    version_store = FakeVersionRecordDatasetStore()
    client = _head_client(
        graph=_default_graph(),
        sdmx_transport=_sdmx_transport(),
        coverage=_coverage_service(),
        version_store=version_store,
        head_writer=LiveHeadStore(store=version_store),
        settings=DatasetsBuildSettings(work_max_concurrency=2, materialise_max_concurrency=1),
    )

    declined = client.post("/api/datasets", json=_BODY, headers=_BACKGROUND)
    assert declined.status_code == 503, declined.text
    assert declined.json()["constraint_id"] == "mint_background_declined"

    foreground = client.post("/api/datasets", json=_BODY)
    assert foreground.status_code == 200, foreground.text

    served = client.post("/api/datasets", json=_BODY, headers=_BACKGROUND)
    assert served.status_code == 200, served.text
    assert served.json()["trace"]["served_from"] == "head"


@pytest.mark.parametrize(
    ("swapped_after", "looks"),
    [pytest.param(0, 1, id="before-the-permit"), pytest.param(1, 2, id="once-it-is-held")],
)
def test_a_replaced_release_declines_a_background_mint(swapped_after, looks):
    """A background mint whose bound release is replaced, seen before it takes a permit or once
    it holds one, answers the background refusal, takes nothing and gives back what it took."""
    permits = MaterialisePermits(2)
    release = _ServedRelease(swapped_after=swapped_after)
    with (
        bound_release(cast(Release, release)),
        leaving_headroom(1),
        pytest.raises(MintBackgroundDeclined),
        bounded_materialise_slot(permits, wait_seconds=10),
    ):
        pytest.fail("entered under a replaced release")
    assert len(release.looks) == looks
    assert [permits.acquire(timeout=0) for _ in range(2)] == [True, True]


_SUBJECT = "spec_" + "b" * 32


def test_a_background_mint_declines_at_once_while_another_rebuild_of_the_subject_runs():
    """A background mint waits for no rebuild: while one runs it is declined at once although
    its deadline is ten seconds out, and the running rebuild remains the one a foreground
    caller joins. With no rebuild running it takes the turn as any caller does."""
    flights = RebuildFlights()
    with flights.turn(_SUBJECT, deadline=None, wait_seconds=10):
        started = time.monotonic()
        with (
            leaving_headroom(1),
            pytest.raises(MintBackgroundDeclined),
            flights.turn(_SUBJECT, deadline=time.monotonic() + 10, wait_seconds=10),
        ):
            pytest.fail("entered while the rebuild ran")
        assert time.monotonic() - started < 1.0
        assert flights._running[_SUBJECT] is not None
    with leaving_headroom(1), flights.turn(_SUBJECT, deadline=None, wait_seconds=10) as waited:
        assert waited is False


def _operation_store(*, backlog_max: int) -> LiveOperationStore:
    return LiveOperationStore(
        store=FakeOperationDatasetStore(),
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        backlog_max=backlog_max,
    )


def _create(store: LiveOperationStore, spec_id: str, *, headroom: int):
    return store.create_or_get(
        request_identity={"input": {"spec_id": spec_id}},
        options={"format": "csv"},
        coverage_stamp="stamp",
        origin_catalogue_build_id="build",
        estimate=Estimate(rows=1, requests=1),
        headroom=headroom,
    )


def test_a_new_operation_leaves_headroom_places_of_the_backlog_free():
    """Of a backlog of two holding one live row, a new operation that must leave one place
    free is refused, a foreground one takes the last place, and a live operation is returned
    to either at a full backlog."""
    store = _operation_store(backlog_max=2)
    first = _create(store, "spec_a", headroom=0)

    with pytest.raises(OperationBacklogFull):
        _create(store, "spec_b", headroom=1)
    _create(store, "spec_b", headroom=0)
    assert _create(store, "spec_a", headroom=1).id == first.id


def test_an_over_budget_background_mint_records_an_operation_only_with_a_backlog_place_to_spare(
    monkeypatch: pytest.MonkeyPatch,
):
    """With a backlog of one, an over-budget background mint is declined with nothing recorded
    or enqueued. A foreground mint records the operation, and the same background mint then
    joins it under the same operation id."""
    _force_mint_over_budget(monkeypatch)
    task_client = FakeTaskClient()
    client = _operations_client(
        operation_store=_operation_store(backlog_max=1), task_client=task_client
    )

    declined = client.post("/api/datasets", json=_HAPPY_PATH_BODY, headers=_BACKGROUND)
    assert declined.status_code == 503, declined.text
    assert declined.json()["constraint_id"] == "mint_background_declined"
    assert task_client.enqueued == []

    foreground = client.post("/api/datasets", json=_HAPPY_PATH_BODY)
    assert foreground.status_code == 202, foreground.text
    joined = client.post("/api/datasets", json=_HAPPY_PATH_BODY, headers=_BACKGROUND)
    assert joined.status_code == 202, joined.text
    assert (
        joined.json()["operation"]["operation_id"] == foreground.json()["operation"]["operation_id"]
    )


class _GatedTransport(httpx.MockTransport):
    """Answers `_GROUPBY_CSV_BODY`, counting requests. With `gate`, the first request signals
    `entered` and waits for `gate` before it answers."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.requests = 0
        self.entered = threading.Event()

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests += 1
            if gate is not None and not self.entered.is_set():
                self.entered.set()
                assert gate.wait(timeout=10)
            return httpx.Response(
                200,
                text=_GROUPBY_CSV_BODY,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )

        super().__init__(handler)


class _WatchedFoldPermit(threading.BoundedSemaphore):
    """One fold permit, held at first, that signals `asked` when a render asks for it."""

    def __init__(self) -> None:
        super().__init__(1)
        self.asked = threading.Event()
        assert super().acquire(blocking=False)

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        self.asked.set()
        return super().acquire(blocking, timeout)


def _groupby_client(transport: _GatedTransport):
    version_store = FakeVersionRecordDatasetStore()
    return _head_client(
        graph=_groupby_graph(),
        sdmx_transport=transport,
        version_store=version_store,
        head_writer=LiveHeadStore(store=version_store),
        code_index=_groupby_code_index(),
        coverage=_groupby_coverage_service(),
    )


def _post_in_thread(client, body: dict, headers: dict | None, into: list) -> threading.Thread:
    thread = threading.Thread(
        target=contextvars.copy_context().run,
        args=(lambda: into.append(client.post("/api/datasets", json=body, headers=headers)),),
    )
    thread.start()
    return thread


@pytest.mark.parametrize(
    ("held_at", "options"),
    [
        pytest.param("fetch", {"bundle": False}, id="during-the-fetch"),
        pytest.param(
            "fold",
            {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
            id="at-the-fold-permit",
        ),
    ],
)
def test_a_foreground_mint_joins_an_admitted_background_mint_with_no_second_fetch(
    monkeypatch: pytest.MonkeyPatch, held_at, options
):
    """An admitted background mint is held during its SDMX fetch, or after it at the fold
    permit another render holds, where it waits as a foreground mint does. A foreground mint of
    the same subject arrives, joins its rebuild and is served the version it records, with the
    SDMX requests of one mint."""
    spec_id, sidecar = _spec_id_and_sidecar()
    body = {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
        "options": options,
    }
    alone = _GatedTransport()
    alone_response = _groupby_client(alone).post("/api/datasets", json=body)
    assert alone_response.status_code == 200, alone_response.text

    gate = threading.Event()
    transport = _GatedTransport(gate if held_at == "fetch" else None)
    permit = _WatchedFoldPermit()
    monkeypatch.setattr(render_module, "_fold_permits", lambda: permit)
    client = _groupby_client(transport)
    background: list = []
    foreground: list = []

    background_thread = _post_in_thread(client, body, _BACKGROUND, background)
    held = transport.entered if held_at == "fetch" else permit.asked
    assert held.wait(timeout=10), "the background mint never reached the point it is held at"
    foreground_thread = _post_in_thread(client, body, None, foreground)
    # Long enough for the foreground mint to reach the rebuild it joins.
    time.sleep(0.3)
    gate.set()
    permit.release()
    background_thread.join(timeout=10)
    foreground_thread.join(timeout=10)

    assert background[0].status_code == 200, background[0].text
    assert foreground[0].status_code == 200, foreground[0].text
    assert transport.requests == alone.requests
    assert (
        foreground[0].json()["content_version"]
        == background[0].json()["content_version"]
        == alone_response.json()["content_version"]
    )
