"""Each read an answer swallows notes the lost read (`degraded_reads.note_degraded_read`), so
the region is marked degraded and no response cache stores it. One row per site, each driven
to the failure it swallows."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

import search.engine.place as place
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphInfraError, RoleSignatureError
from search.engine.region import grounding
from search.engine.retrieval import derive
from search.engine.unit import refs


def _raise_graph_failure(*_args, **_kwargs):
    raise GraphInfraError("the read failed")


def _availability_probe() -> None:
    derive._probe_place_availability(
        {"Count_Person": {}},
        place_dcids=["country/KEN"],
        full_facets_max_candidates=None,
        graph=SimpleNamespace(observation_facets_batch=_raise_graph_failure),
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
    monkeypatch.setattr(grounding, "is_role_bearing", lambda _shape: False)
    monkeypatch.setattr(place, "read_default_presence", _raise_graph_failure)
    ctx = _relaxation_ctx()
    assert grounding._relaxations_presence(ctx, [ctx.shape_draft]) is ctx.presence


def _relaxed_suggestion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grounding, "is_role_bearing", lambda _shape: False)
    monkeypatch.setattr(grounding, "standard_resolve", _raise_graph_failure)
    ctx = _relaxation_ctx()
    suggestion = grounding._relaxed_binding_suggestion(ctx, ctx.shape_draft, [], None)
    assert suggestion == (None, None, None, None)


@pytest.mark.parametrize(
    ("swallow", "read"),
    [
        (lambda _mp: _availability_probe(), "place-availability probe"),
        (lambda _mp: _currency_units(), "currency units"),
        (_relaxation_walk, "nearest-real relaxation walk"),
        (_relaxations_presence, "nearest-real presence"),
        (_relaxed_suggestion, "nearest-real suggestion"),
    ],
    ids=["availability", "currency", "walk", "presence", "suggestion"],
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
