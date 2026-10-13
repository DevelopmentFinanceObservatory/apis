"""Departures the part emits for its requirements, for tests that run one phase."""

from __future__ import annotations

from search.engine.region.requirements import judge
from search.engine.region.state import ResolveState


def depart_requirements(state: ResolveState) -> None:
    """Depart on `state` what its requirements depart when no final answer is judged
    (`requirements.Judgement.departures`), as the region pipeline's exit does, so a test of
    one phase reads them among `state.warnings` and `state.departure_facts`."""
    for warning, fact in judge(state.requirements, ()).departures():
        state.depart(warning, fact)
