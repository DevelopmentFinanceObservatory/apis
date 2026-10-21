"""A dataset resubmit's trailing stream and its `DoneEvent` follow the parts it resolved.

The trailing stream keeps the request deadline unless a promote's member walk ran the full
read, and a part whose answer stands on less than a later request may read marks the
`DoneEvent` `degraded_leg`, as the parsed door does. `resolve_spec_resubmit` and the
trailing stream are patched at their call sites.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from search.engine.binders import GeminiBinder
from search.engine.concurrency import scoped_concurrency
from search.engine.config import settings
from search.engine.entry import dataset_resubmit
from search.engine.entry.params import EntryParams
from search.engine.events import DoneEvent
from search.models import DatasetResubmitInput, ResubmitIdentityEntry
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_entry_rate_cache import _gdp_identity, _stub_region
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def _run(regions, *, deadline: float, stream: bool) -> tuple[list[float], list[DoneEvent]]:
    """Resubmit one identity entry per region in `regions`, each resolving to its region.
    Returns the deadlines the trailing stream ran to and the `DoneEvent`s."""
    pending = list(regions)
    trailing: list[float] = []

    def _fake_resolve_spec_resubmit(**kwargs):
        return pending.pop(0)

    async def _fake_stream_slot_options(regions, *, deadline, **kwargs):
        trailing.append(deadline)
        return
        yield

    graph = FakeGraph()
    entries = [
        ResubmitIdentityEntry(variable_text=f"GDP {i}", identity=_gdp_identity())
        for i in range(len(regions))
    ]
    params = EntryParams(
        deadline=deadline,
        max_candidates=5,
        pac=False,
        stream_option_coverage=stream,
        include_option_coverage=False,
        stream_observations=False,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        coverage=empty_coverage_snapshot(),
        binder=GeminiBinder(),
    )

    async def _drain() -> list[DoneEvent]:
        async with scoped_concurrency():
            events = dataset_resubmit.handle(
                DatasetResubmitInput(entries=entries),
                graph=graph,
                resolver=make_resolver(graph),
                admission=Admission(graph=graph),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                materialise_graph=None,
                materialise_sdmx=None,
                start_ms=0,
                params=params,
            )
            return [event async for event in events if isinstance(event, DoneEvent)]

    with (
        patch(
            "search.engine.entry.dataset_resubmit.resolve_spec_resubmit",
            _fake_resolve_spec_resubmit,
        ),
        patch(
            "search.engine.entry.dataset_resubmit._stream_slot_options", _fake_stream_slot_options
        ),
        patch("search.engine.entry.multi._stream_slot_options", _fake_stream_slot_options),
    ):
        done = asyncio.run(_drain())
    return trailing, done


@pytest.mark.parametrize("parts", [1, 2], ids=["one_part", "combined"])
@pytest.mark.parametrize("full_read", [False, True], ids=["first_page", "full_read"])
def test_the_trailing_stream_runs_to_the_full_read_bound_only_after_a_full_read(
    parts: int, full_read: bool
):
    deadline = time.monotonic() + 30
    extended = deadline + settings().resubmit_full_read_timeout_s - settings().request_timeout_s
    regions = [
        dataclasses.replace(_stub_region(None), reads_deadline=deadline) for _ in range(parts)
    ]
    if full_read:
        regions[-1] = dataclasses.replace(regions[-1], reads_deadline=extended)

    trailing, _done = _run(regions, deadline=deadline, stream=True)

    assert trailing == [extended if full_read else deadline]


@pytest.mark.parametrize(
    ("parts", "stream"),
    [(1, False), (1, True), (2, True)],
    ids=["buffered", "one_part_streamed", "combined"],
)
@pytest.mark.parametrize("degraded", [False, True], ids=["settled", "degraded"])
def test_a_degraded_part_marks_the_done_event_degraded_leg(
    parts: int, stream: bool, degraded: bool
):
    regions = [_stub_region(None) for _ in range(parts)]
    regions[0] = dataclasses.replace(regions[0], degraded=degraded)

    _trailing, done = _run(regions, deadline=time.monotonic() + 30, stream=stream)

    assert [event.degraded_leg for event in done] == [degraded]
