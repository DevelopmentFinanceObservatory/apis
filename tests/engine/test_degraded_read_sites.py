"""Each read an answer swallows notes the lost read (`degraded_reads.note_degraded_read`), so
the region is marked degraded and no response cache stores it. One row per site, each driven
to the failure it swallows."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

import search.engine.place as place
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphDeadlineFailure, GraphInfraError, RoleSignatureError
from search.engine.region import grounding
from search.engine.region import state as region_state
from search.engine.region.phase_rank_bind import _stray_requirements
from search.engine.region.state import ResolveState
from search.engine.retrieval import derive
from search.engine.unit import refs
from tests.engine._fail_open_deadline import (
    FAIL_OPEN_REQUESTS,
    assert_fail_open_deadline,
    request_deadline,
)


def _raise_graph_failure(*_args, **_kwargs):
    raise GraphInfraError("the read failed")


def _availability_probe() -> None:
    derive._probe_place_availability(
        {"Count_Person": {}},
        place_dcids=["country/KEN"],
        full_facets_max_candidates=None,
        graph=SimpleNamespace(observation_facets_batch=_raise_graph_failure),
        deadline=None,
        coverage=None,
    )


def _currency_units() -> None:
    refs._graph_currency(
        SimpleNamespace(currency_units=_raise_graph_failure), "Kenya_Shilling", deadline=None
    )


def _relaxation_ctx(**changes: object) -> SimpleNamespace:
    roles = SimpleNamespace(resolved_entities={}, origin_dcids=[], roles=[], open_axes=frozenset())
    return SimpleNamespace(
        **{
            "graph": None,
            "sdmx": None,
            "shape_draft": SimpleNamespace(sv_arc_facts={"Count_Person": {}}),
            "place": SimpleNamespace(
                roles=roles,
                default_scope="countries",
                probe_places=["country/KEN"],
                presence_for=lambda presence: presence,
            ),
            "unsettled": None,
            "presence": SimpleNamespace(scopes={}),
            "date_request": None,
            "deadline": None,
            "role_class_walks": object(),
            **changes,
        }
    )


def _no_role_keys(*_args, **_kwargs) -> SimpleNamespace:
    return SimpleNamespace(bound=None, open=None)


def _relaxation_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grounding, "shape_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grounding, "entity_role_keys", _no_role_keys)
    monkeypatch.setattr(grounding, "elected_role_class_shape", _raise_graph_failure)
    assert grounding._walked_relaxations(_relaxation_ctx(), [[]]) == [None]


def _relaxations_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(region_state, "presence_stat_vars", lambda *_args: ["Count_Person"])
    monkeypatch.setattr(grounding, "entity_role_keys", _no_role_keys)
    monkeypatch.setattr(place, "read_default_presence", _raise_graph_failure)
    ctx = _relaxation_ctx()
    assert grounding._relaxations_presence(ctx, [(ctx.shape_draft, [])]) is ctx.presence


def _relaxed_suggestion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grounding, "is_role_bearing", lambda _shape: False)
    monkeypatch.setattr(grounding, "standard_resolve", _raise_graph_failure)
    ctx = _relaxation_ctx()
    suggestion = grounding._relaxed_binding_suggestion(ctx, ctx.shape_draft, [], None)
    assert suggestion == (None, None, None, None)


def _stray_value_labels() -> None:
    """A stray value no menu shape or code index labels, named in words from its dcid when
    the graph's label read fails, so no dcid reaches the prose that discloses it."""
    stray = SlotBindingDraft(
        axis="how", property_dcid="gender", kind="value", value_dcids=["dc/g/SexFemale"]
    )
    state = SimpleNamespace(
        capped_shapes=[],
        label_dcids=_raise_graph_failure,
        admitted_qualifiers=(),
        query_qualifiers=[],
        variable="population",
        query_text=None,
        dropped_qualifier_phrases=(),
    )
    (requirement,) = asyncio.run(_stray_requirements(cast(ResolveState, state), [stray]))
    assert requirement.term.text == "sex female"
    assert requirement.departure is not None
    assert requirement.departure[1].asked == "sex female"


@pytest.mark.parametrize(
    ("swallow", "read"),
    [
        (lambda _mp: _availability_probe(), "place-availability probe"),
        (lambda _mp: _currency_units(), "currency units"),
        (_relaxation_walk, "nearest-real relaxation walk"),
        (_relaxations_presence, "nearest-real presence"),
        (_relaxed_suggestion, "nearest-real suggestion"),
        (lambda _mp: _stray_value_labels(), "stray value labels"),
    ],
    ids=["availability", "currency", "walk", "presence", "suggestion", "stray-labels"],
)
def test_a_swallowed_read_is_noted(
    swallow: Callable[[pytest.MonkeyPatch], None], read: str, monkeypatch
) -> None:
    with recording_degraded_reads() as lost:
        swallow(monkeypatch)
    assert lost.reads == [read]


def test_a_refused_relaxation_walk_notes_nothing(monkeypatch) -> None:
    """A walk the mint refuses is deterministic, so the answer without it can be stored."""

    def _refused(*_args, **_kwargs):
        raise RoleSignatureError("role sets the election cannot order")

    monkeypatch.setattr(grounding, "shape_query", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grounding, "entity_role_keys", _no_role_keys)
    monkeypatch.setattr(grounding, "elected_role_class_shape", _refused)
    with recording_degraded_reads() as lost:
        assert grounding._walked_relaxations(_relaxation_ctx(), [[]]) == [None]
    assert lost.reads == []


class _FacetGraph:
    """Records the deadline of each facet read it serves, and answers none."""

    def __init__(self) -> None:
        self.deadlines: list[float | None] = []

    def observation_facets_batch(self, svs, places, _needs_dates, *, deadline=None):
        self.deadlines.append(deadline)
        return {sv: {place: [] for place in places} for sv in svs}


@FAIL_OPEN_REQUESTS
def test_the_place_availability_probe_reads_under_the_fail_open_budget(request_in_s) -> None:
    graph = _FacetGraph()
    requested = request_deadline(request_in_s)
    started = time.monotonic()

    derive._probe_place_availability(
        {"Count_Person": {}},
        place_dcids=["country/KEN"],
        full_facets_max_candidates=None,
        graph=cast(EngineGraphClient, graph),
        deadline=requested,
        coverage=None,
    )

    (deadline,) = graph.deadlines
    assert_fail_open_deadline(deadline, requested=requested, started=started)


def test_a_probe_read_past_its_deadline_is_a_lost_read_and_no_request_timeout() -> None:
    """The client's deadline error is an `Exception` the probe catches. Its class sits outside
    `TimeoutError`, which the request-level classifiers read as the request's 504."""
    assert not issubclass(GraphDeadlineFailure, TimeoutError)

    def expired(*_args, **_kwargs):
        raise GraphDeadlineFailure("the read ran past its deadline")

    with recording_degraded_reads() as lost:
        read = derive._graph_facets_at(
            ["Count_Person"],
            ["country/KEN"],
            full_facets_max_candidates=None,
            graph=cast(EngineGraphClient, SimpleNamespace(observation_facets_batch=expired)),
            deadline=None,
        )

    assert read is None
    assert lost.reads == ["place-availability probe"]
