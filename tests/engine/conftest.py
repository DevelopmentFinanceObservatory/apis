"""Shared engine-test fixtures. The pinned date keeps FakeLLM prompts deterministic."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from concurrent.futures import Future
from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

from dfo_platform import trace as _trace_mod
from search.engine.region import related_read
from tests.engine._request_log import checked_requests
from tests.fixtures import PINNED_DATE


@pytest.fixture(autouse=True)
def _pin_extract_date():
    """Pin the prompt date so recorded FakeLLM fixtures keep matching, and the clock a
    mint's open-ended election ends at (`materialise.window.current_year`) with it."""
    with (
        patch("search.engine.extract.date") as mock_date,
        patch("materialise.window.date") as election_date,
    ):
        mock_date.today.return_value = PINNED_DATE
        election_date.today.return_value = PINNED_DATE
        # Keep date(year, month, day) construction backed by the real class.
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        yield


@pytest.fixture(autouse=True)
def _reset_trace_id():
    """Clear the trace ContextVar before direct-call tests."""
    token = _trace_mod._current.set(None)
    try:
        yield
    finally:
        _trace_mod._current.reset(token)


@pytest.fixture
def request_log(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Fail the test unless it makes the graph and SDMX requests stored for it
    (`tests.engine._request_log`)."""
    yield from checked_requests(request, monkeypatch)


class _InlineReads:
    """Runs each related read on the thread that starts it, under a copy of its context, and
    returns it settled."""

    def start(self, read: Callable[[], Any]) -> asyncio.Future[Any]:
        return _settled(asyncio.get_running_loop().create_future(), read)

    def submit(self, read: Callable[[], Any]) -> Future[Any]:
        return _settled(Future(), read)


def _settled[F: Future | asyncio.Future](future: F, read: Callable[[], Any]) -> F:
    try:
        future.set_result(contextvars.copy_context().run(read))
    except Exception as exc:
        future.set_exception(exc)
    return future


@pytest.fixture(autouse=True)
def _related_reads_inline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Run each related-measures read on the thread that starts it, so it has ended when the
    answer takes it and the answer states it however long the machine stalls. On its own
    thread a read the doubles serve in milliseconds can miss its 1 s wall-clock deadline when
    a garbage collection pauses the worker. A test marked `related_read_threads` keeps the
    threads, as one of the deadline or of the wait for the read does."""
    if request.node.get_closest_marker("related_read_threads") is None:
        monkeypatch.setattr(related_read, "read_runner", _InlineReads())
