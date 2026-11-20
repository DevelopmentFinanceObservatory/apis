"""`GET /api/datasets/{id}` answers at once from a stored head that holds the requested
rendering, past the recheck window too, up to `head_serve_stale_max_age_seconds`, and submits
the head's check to the worker (`current.serve_stored_head`). Its 200 and 304 carry
`x-dfo-freshness`, `x-dfo-content-version`, a strong ETag over the body and the freshness, and
`no-cache`.

The door, the materialiser and the operation store are real. The graph, SDMX and head store are
the currency tests' doubles, and the task queue records its tasks.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.task_client import FakeTaskClient

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

# Past the 7-day `head_serve_stale_max_age_seconds` default.
_PAST_SERVE_STALE_CEILING = timedelta(days=8)

_IDENTITY = {"accept-encoding": "identity"}


class _RaisingTaskClient:
    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        raise RuntimeError("cloud tasks unavailable")


class _SwitchableTaskClient(FakeTaskClient):
    """Records its tasks, or fails every create while `failing`."""

    failing = False

    def enqueue(self, *, operation_id: str, submission_count: int) -> None:
        if self.failing:
            raise RuntimeError("cloud tasks unavailable")
        super().enqueue(operation_id=operation_id, submission_count=submission_count)


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
        self.client = _client(
            graph=self.graph,
            sdmx_transport=transport,
            head_writer=self.head_store,
            coverage=_CoverageFromFacets(facets),
            operation_store=self.operation_store,
            task_client=self.task_client,
        )
        self.spec_id = _mint_one_spec(self.client)
        self.content_version = self.head_store.head_for(self.spec_id).content_version

    def get(self, query: str = "?bundle=false", **headers: str):
        """A GET of the rendering the mint stored, unless `query` names another."""
        return self.client.get(f"/api/datasets/{self.spec_id}{query}", headers=headers)


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


def test_a_get_past_the_serve_stale_ceiling_rebuilds_in_the_request():
    rig = _Rig()
    rig.head_store.age_head(rig.spec_id, age=_PAST_SERVE_STALE_CEILING)
    sdmx_calls = rig.sdmx_calls.n

    resp = rig.get()

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"
    assert resp.headers["x-dfo-freshness"] == "current"
    assert rig.sdmx_calls.n == sdmx_calls + 1
    assert rig.task_client.enqueued == []


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


@pytest.mark.parametrize(
    "rig",
    [
        # Half of the default backlog of 20 is the check's headroom, so a backlog of 10 refuses
        # every check.
        pytest.param(
            lambda: _Rig(
                operation_store=LiveOperationStore(
                    store=FakeOperationDatasetStore(), backlog_max=10
                )
            ),
            id="backlog_full",
        ),
        pytest.param(lambda: _Rig(task_client=_RaisingTaskClient()), id="enqueue_error"),
    ],
)
def test_a_get_whose_check_cannot_be_submitted_serves_the_head_stale(rig):
    rig = rig()
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)

    resp = rig.get()

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert resp.json()["content_version"] == rig.content_version
    assert resp.headers["x-dfo-freshness"] == "stale"
    assert resp.headers["cache-control"] == "no-cache"


def test_a_revalidation_naming_the_etag_answers_304_with_the_same_headers():
    rig = _Rig()
    first = rig.get(**_IDENTITY)
    etag = first.headers["etag"]
    assert etag.startswith('"'), etag

    revalidated = rig.get(**_IDENTITY, **{"if-none-match": etag})

    assert revalidated.status_code == 304
    assert revalidated.content == b""
    for name in ("etag", "x-dfo-freshness", "x-dfo-content-version", "cache-control"):
        assert revalidated.headers[name] == first.headers[name], name


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
    rig = _Rig(task_client=task_client)
    rig.head_store.age_head(rig.spec_id, age=_PAST_INTERVAL)
    task_client.failing = True
    stale = rig.get(**_IDENTITY)
    assert stale.headers["x-dfo-freshness"] == "stale"
    task_client.failing = False

    pending = rig.get(**_IDENTITY, **{"if-none-match": stale.headers["etag"]})

    assert pending.status_code == 200
    assert pending.headers["x-dfo-freshness"] == "pending"
    assert pending.content == stale.content
    assert pending.headers["etag"] != stale.headers["etag"]
