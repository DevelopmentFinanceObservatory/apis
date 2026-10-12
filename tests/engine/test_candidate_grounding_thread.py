"""`assemble_candidates` grounds its candidates on a worker thread, so the event loop keeps
running while grounding makes its synchronous graph reads."""

from __future__ import annotations

import asyncio
import threading
from typing import cast

import pytest

from entity_resolution import Admission, EntityResolver
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.region import candidate_collapse
from search.engine.region.candidate_admission import _AlternatesAdmission
from search.engine.region.state import ResolveState
from tests.engine._fake_coverage import empty_coverage_snapshot


class _GroundingStopped(Exception):
    """Raised by the grounding double once it has recorded where it ran."""


def _state() -> ResolveState:
    """A state with no elected result, so `assemble_candidates` grounds an empty list."""
    return ResolveState(
        variable="aid",
        entities=[],
        date_request=None,
        detect_query="aid",
        role_query="aid",
        pac=True,
        graph=cast(EngineGraphClient, None),
        resolver=cast(EntityResolver, None),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
    )


def test_assemble_candidates_grounds_off_the_event_loop(monkeypatch) -> None:
    """Grounding runs on a thread other than the loop's, and a coroutine on the loop runs
    while grounding waits. The double waits for an event that only the loop sets."""
    loop_progressed = threading.Event()
    recorded: dict[str, object] = {}

    def _ground(state, candidates):
        recorded["thread"] = threading.get_ident()
        recorded["loop_progressed"] = loop_progressed.wait(5)
        raise _GroundingStopped

    monkeypatch.setattr(candidate_collapse, "ground_candidates", _ground)

    async def _set_progress() -> None:
        loop_progressed.set()

    async def _run() -> int:
        setter = asyncio.ensure_future(_set_progress())
        try:
            with pytest.raises(_GroundingStopped):
                await candidate_collapse.assemble_candidates(
                    _state(), _AlternatesAdmission(), rate_cache=None
                )
        finally:
            await setter
        return threading.get_ident()

    loop_thread = asyncio.run(_run())

    assert recorded["thread"] != loop_thread
    assert recorded["loop_progressed"] is True
