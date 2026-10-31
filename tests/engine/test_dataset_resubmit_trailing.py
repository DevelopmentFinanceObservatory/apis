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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from entity_resolution import Admission
from search.engine.binders import GeminiBinder
from search.engine.concurrency import scoped_concurrency
from search.engine.config import settings
from search.engine.entry import dataset_resubmit
from search.engine.entry.dataset_entries import _build_candidates_region
from search.engine.entry.params import EntryParams
from search.engine.events import DoneEvent
from search.engine.region import RegionResult
from search.models import (
    DatasetResubmitInput,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_entry_rate_cache import _gdp_identity, _stub_region
from tests.fixtures import FakeGraph, FakeLLM, make_resolver


def _run(
    regions, *, deadline: float, stream: bool, candidates: bool = False
) -> tuple[list[float], list[DoneEvent]]:
    """Resubmit one identity entry per region in `regions`, each resolving to its region, or
    with `candidates` one candidates entry promoting one candidate to each. Returns the
    deadlines the trailing stream ran to and the `DoneEvent`s."""
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
    if candidates:
        entries = [
            ResubmitCandidatesEntry(
                variable_text="GDP",
                ordering="broadest_first",
                truncated=False,
                candidates=[
                    ResubmitCandidate(identity=_gdp_identity(), entity_groups={}) for _ in regions
                ],
            )
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
            "search.engine.entry.dataset_entries.resolve_spec_resubmit",
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


def test_a_degraded_candidate_marks_the_done_event_degraded_leg():
    regions = [_stub_region(None), dataclasses.replace(_stub_region(None), degraded=True)]

    _trailing, done = _run(regions, deadline=time.monotonic() + 30, stream=False, candidates=True)

    assert [event.degraded_leg for event in done] == [True]


def _promote(status: str, spec_id: str = "") -> RegionResult:
    if status == "no_data":
        return _stub_region(None)
    spec = SimpleNamespace(spec_id=spec_id)
    return dataclasses.replace(
        _stub_region(None), status=status, specs=(spec,), no_data_reason=None
    )


@pytest.mark.parametrize("survivors", [0, 1, 2])
def test_a_candidates_part_keeps_the_read_health_of_a_discarded_promote(survivors: int):
    """Each branch of `_build_candidates_region` keeps it: rebuilding a set, collapsing to a
    survivor and collapsing to no_data."""
    promoted = [
        *(_promote("definite", f"spec-{i}") for i in range(survivors)),
        dataclasses.replace(_promote("no_data"), degraded=True),
    ]
    entry = SimpleNamespace(
        variable_text="GDP",
        candidates=[SimpleNamespace(assumptions=None) for _ in promoted],
        ordering="broadest_first",
        truncated=False,
        separate=False,
    )

    region = _build_candidates_region(entry, promoted, index=0)

    assert region.degraded is True
