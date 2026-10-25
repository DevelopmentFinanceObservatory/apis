"""Tests request-scoped concurrency and owned tasks (#624).
A mixed resubmit shares one semaphore across promote and parsed fan-out work.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from search.engine.binders import GeminiBinder
from search.engine.concurrency import (
    MissingConcurrencyScope,
    bounded_semaphore,
    create_owned_task,
    scoped_concurrency,
)
from search.engine.core import resolve_events
from search.engine.fanout import _FanOutResult, _run_variables_streaming
from search.models import (
    DatasetResubmitInput,
    ParsedEntity,
    ParsedPart,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    ResubmitParsedEntry,
    Slot,
)
from tests.engine._base_identity import GDP_SHAPE, make_identity, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_INDIA = "country/IND"

# Same recipe test_dataset_resubmit.py's own `_gdp_entry` uses (its own
# `_ACTIVITY_SOURCE_SLOT`): GDP_SHAPE's graph arcs already assert
# `activitySource`, the one axis this identity declares.
_ACTIVITY_SOURCE_SLOT = Slot.model_validate(
    {
        "key": {
            "axis": "how",
            "property": {"ref": "activitySource", "label": "activity source"},
            "label": "activity source",
        },
        "binding": {"kind": "unbound"},
    }
)


def _gdp_identity_entry(entity_dcid: str, variable_text: str) -> ResubmitIdentityEntry:
    entities = [subject_entity(entity_dcid)]
    identity = make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities)
    return ResubmitIdentityEntry(
        variable_text=variable_text,
        identity=identity,
    )


# ---------------------------------------------------------------------------
# Calls that need a semaphore require a scope.
# ---------------------------------------------------------------------------


def test_bounded_semaphore_without_a_scope_raises():
    """`bounded_semaphore()` requires an open concurrency scope."""
    with pytest.raises(MissingConcurrencyScope):
        bounded_semaphore()


def test_create_owned_task_without_a_scope_raises_and_never_schedules_the_coroutine():
    """The missing-scope error must precede task creation.
    An unchanged task count and cooperative yield catch leaked work.
    """

    async def _scenario() -> None:
        before = len(asyncio.all_tasks())
        ran = False

        async def _coro() -> None:
            nonlocal ran
            ran = True

        with pytest.raises(MissingConcurrencyScope):
            create_owned_task(_coro())

        # One cooperative yield: if create_owned_task had scheduled the
        # coroutine despite raising, it would have had a chance to run here.
        await asyncio.sleep(0)
        assert not ran, "the coroutine ran despite MissingConcurrencyScope -- it was scheduled"
        assert len(asyncio.all_tasks()) == before, (
            "a task was created despite MissingConcurrencyScope"
        )

    asyncio.run(_scenario())


def test_run_variables_streaming_use_semaphore_without_a_scope_raises():
    """The N>=2 streaming path requires `core.resolve_events` to open its scope."""

    async def _scenario() -> None:
        graph = FakeGraph()
        resolver = make_resolver(graph)
        admission = Admission(graph=graph)
        sdmx = FakeSdmxClient(observations={}, availability={})
        llm = FakeLLM(responses={})
        out = _FanOutResult()

        with pytest.raises(MissingConcurrencyScope):
            async for _ev in _run_variables_streaming(
                ["v0", "v1"],
                entities=[],
                entity_texts={},
                date_request=None,
                query="",
                pac=False,
                graph=graph,
                resolver=resolver,
                admission=admission,
                llm=llm,
                binder=GeminiBinder(),
                sdmx=sdmx,
                base_timing={},
                pre_resolved=None,
                deadline=time.monotonic() + 3600,
                degrade_on_error=True,
                use_semaphore=True,
                out=out,
                coverage=empty_coverage_snapshot(),
            ):
                pass

    asyncio.run(_scenario())


def test_scoped_concurrency_lets_the_same_call_succeed():
    """The same streaming call succeeds inside `scoped_concurrency()`."""

    async def _scenario() -> None:
        graph = FakeGraph()
        resolver = make_resolver(graph)
        admission = Admission(graph=graph)
        sdmx = FakeSdmxClient(observations={}, availability={})
        llm = FakeLLM(responses={})
        out = _FanOutResult()

        async with scoped_concurrency():
            async for _ev in _run_variables_streaming(
                ["v0", "v1"],
                entities=[],
                entity_texts={},
                date_request=None,
                query="",
                pac=False,
                graph=graph,
                resolver=resolver,
                admission=admission,
                llm=llm,
                binder=GeminiBinder(),
                sdmx=sdmx,
                base_timing={},
                pre_resolved=None,
                deadline=time.monotonic() + 3600,
                degrade_on_error=True,
                use_semaphore=True,
                out=out,
                coverage=empty_coverage_snapshot(),
            ):
                pass
        assert len(out.regions) == 2

    asyncio.run(_scenario())


# ---------------------------------------------------------------------------
# A mixed dataset_resubmit request shares one semaphore.
# ---------------------------------------------------------------------------


class _Overlap:
    """Tracks concurrent test legs and waits briefly for a peer.
    A shared semaphore keeps `peak` at 1. Independent semaphores reach 2.
    The bounded wait makes serial runs fail without hanging.
    """

    def __init__(self, *, settle: float = 0.2) -> None:
        self._settle = settle
        self._lock = threading.Lock()
        self._active = 0
        self._paired = threading.Event()
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            if self._active >= 2:
                self._paired.set()

    def leave(self) -> None:
        with self._lock:
            self._active -= 1

    def hold(self) -> None:
        self._paired.wait(timeout=self._settle)


def test_mixed_dataset_resubmit_request_shares_one_semaphore(search_settings):
    """Two identity promotes and two parsed legs share one semaphore at limit 1 (#624).
    A rendezvous distinguishes the shared limit from independent semaphores, which reach 2.
    `max_variables=1` satisfies SearchSettings' invariant. DatasetResubmitInput does not use it.
    """
    from search.engine.region import RegionResult

    overlap = _Overlap()

    def _resolve_spec_resubmit(*args, **kwargs):
        overlap.enter()
        try:
            overlap.hold()
            return RegionResult(
                variable_text="identity",
                status="no_data",
                specs=(),
                no_data_reason="variable_not_resolved",
                warnings=(),
                timing_by_step={},
            )
        finally:
            overlap.leave()

    async def _slow_resolve_variable(variable, *args, **kwargs):
        overlap.enter()
        try:
            # This leg runs on the event loop, so the bounded hold is taken
            # off-loop: blocking inline would stall every other leg's turn
            # at arriving.
            await asyncio.to_thread(overlap.hold)
            return RegionResult(
                variable_text=variable,
                status="no_data",
                specs=(),
                no_data_reason="variable_not_resolved",
                warnings=(),
                timing_by_step={},
            )
        finally:
            overlap.leave()

    entries = [
        _gdp_identity_entry(_INDIA, "GDP one"),
        _gdp_identity_entry(_INDIA, "GDP two"),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="life expectancy", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=entries),
        response_options=ResolveOptions(),
    )

    async def _drain() -> None:
        graph = FakeGraph()
        async for _ev in resolve_events(
            request,
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
        ):
            pass

    search_settings(max_variables=1, max_variable_concurrency=1)
    with (
        patch(
            "search.engine.entry.dataset_resubmit.resolve_spec_resubmit",
            _resolve_spec_resubmit,
        ),
        patch("search.engine.fanout.resolve_variable", _slow_resolve_variable),
    ):
        asyncio.run(_drain())

    assert overlap.peak == 1, (
        f"peak concurrent legs was {overlap.peak}, expected 1 -- a mixed dataset_resubmit "
        f"request must share one semaphore across identity/candidate promotes and "
        f"the parsed fan-out (#624), not reach {overlap.peak}x SEARCH_MAX_VARIABLE_CONCURRENCY."
    )
