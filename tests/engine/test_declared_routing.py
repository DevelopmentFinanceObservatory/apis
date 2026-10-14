"""Tests routing from each shape's declared entity roles.

Fixtures use a neutral namespace because live StatVars lack ``observationProperties``."""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.departures import warning_key
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import DIRECTION_DISCARDED
from search.engine.region.pipeline import resolve_variable
from search.heading.departure_fact import DepartureFact
from search.models import TimeWindow
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver


def _plain_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }


def _declaring_arcs() -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
    }


def _unplaceable_arcs() -> dict:
    arcs = _declaring_arcs()
    arcs["observationProperties"] = {"nodes": [{"dcid": "contributor"}]}
    return arcs


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


_PLAIN_SV = "test/ns/PlainSubjectCount"
_DECLARING_SV = "test/ns/CreditorDebtorFlow"
_FRA = "country/FRA"
_KEN = "country/KEN"


def _run(
    *,
    detect_svs: list[str],
    sv_scores: dict[str, float],
    graph_nodes: dict,
    obs: dict,
    role_query: str,
    date_request: DateRequest | None = None,
    sdmx: FakeSdmxClient | None = None,
    variable: str = "aid",
):
    graph = FakeGraph(
        nodes=graph_nodes,
        detect={
            role_query: {"svs": detect_svs, "cosine_scores": [sv_scores[s] for s in detect_svs]}
        },
        resolve={"France": _FRA, "Kenya": _KEN},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            variable,
            entities=["France", "Kenya"],
            date_request=date_request,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["France", "Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx or FakeSdmxClient(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def _assert_role_refusal_before_bind(result) -> None:
    """No reading can place the query's places, so the part refuses `role_incompatible`
    before bind and flags nothing (`admissibility.judge_measure_and_roles`)."""
    assert result.status == "no_data"
    assert result.no_data_reason == "role_incompatible"
    assert result.gate is None
    assert not result.flagged_match
    assert "bind" not in result.timing_by_step


def test_directional_query_on_a_declaring_recall_below_the_floor_answers_as_subjects():
    """A declaring sibling, a role-bearing shape the index does not carry, scoring below
    the relevance floor outranks nothing, so the plain shape is elected and answers both
    places as its subjects. Each flow end the request named is disclosed as a discarded
    direction."""
    graph_nodes = {
        _FRA: {"label": "France", "type": "Country"},
        _KEN: {"label": "Kenya", "type": "Country"},
        _PLAIN_SV: {"label": "Plain population count", "arcs": _plain_arcs()},
        _DECLARING_SV: {"label": "Creditor/debtor flow", "arcs": _declaring_arcs()},
    }
    facet = [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}]
    result = _run(
        detect_svs=[_PLAIN_SV, _DECLARING_SV],
        sv_scores={_PLAIN_SV: 0.9, _DECLARING_SV: 0.5},
        graph_nodes=graph_nodes,
        obs={f"{_PLAIN_SV}|{place}": facet for place in (_FRA, _KEN)},
        role_query="aid from France to Kenya",
    )
    assert result.status == "definite", result
    (spec,) = result.specs
    assert [stat_var.ref.ref for stat_var in spec.resolved.members] == [_PLAIN_SV]
    assert spec.definition.identity.entities == {"observationAbout": [_FRA, _KEN]}
    warnings = [w for w in result.warnings if w.code == DIRECTION_DISCARDED]
    assert [result.departure_facts[warning_key(w)] for w in warnings] == [
        DepartureFact(
            code="direction_discarded", binding_ref="where:", asked="France", over="origin"
        ),
        DepartureFact(
            code="direction_discarded", binding_ref="where:", asked="Kenya", over="destination"
        ),
    ]


def test_directional_query_on_a_declaring_recall_elects_the_declaring_shape():
    """A plain shape cannot hold both flow ends, so the declaring sibling is elected although
    the plain shape outranks it, and binds each place to its own role. The declaring shape
    scores above the relevance floor, which holds a role-bearing shape the index does not
    carry."""
    graph_nodes = {
        _FRA: {"label": "France", "type": "Country"},
        _KEN: {"label": "Kenya", "type": "Country"},
        _PLAIN_SV: {"label": "Plain population count", "arcs": _plain_arcs()},
        _DECLARING_SV: {"label": "Creditor/debtor flow", "arcs": _declaring_arcs()},
    }
    window = TimeWindow(start_year=2015, end_year=2020)
    result = _run(
        detect_svs=[_PLAIN_SV, _DECLARING_SV],
        sv_scores={_PLAIN_SV: 0.9, _DECLARING_SV: 0.7},
        graph_nodes=graph_nodes,
        obs={},
        role_query="aid from France to Kenya",
        date_request=DateRequest(window=window, latest=False),
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.elected_stat_vars is not None
    assert [stat_var.ref for stat_var in result.elected_stat_vars] == [_DECLARING_SV]
    assert result.flagged_match is not None
    assert result.flagged_match[0].definition.identity.entities == {
        "origin": [_FRA],
        "destination": [_KEN],
    }


def test_unplaceable_role_refuses_before_bind(monkeypatch):
    """A declaring shape whose role is on neither flow axis is no reading of a question
    naming both ends."""
    graph_nodes = {
        _FRA: {"label": "France", "type": "Country"},
        _KEN: {"label": "Kenya", "type": "Country"},
        _DECLARING_SV: {"label": "Unplaceable flow", "arcs": _unplaceable_arcs()},
    }

    def _materialise_was_not_called(*args, **kwargs):
        raise AssertionError("role_incompatible must not materialise or probe availability")

    monkeypatch.setattr(
        "search.engine.region.materialise_step.materialise", _materialise_was_not_called
    )
    window = TimeWindow(start_year=2015, end_year=2020)
    result = _run(
        detect_svs=[_DECLARING_SV],
        sv_scores={_DECLARING_SV: 0.9},
        graph_nodes=graph_nodes,
        obs={},
        role_query="aid from France to Kenya",
        date_request=DateRequest(window=window, latest=False),
    )
    _assert_role_refusal_before_bind(result)


def test_directional_query_with_no_declaring_shape_answers_as_subjects():
    """A standard shape answers when no recalled group declares flow roles."""
    graph_nodes = {
        _FRA: {"label": "France", "type": "Country"},
        _KEN: {"label": "Kenya", "type": "Country"},
        _PLAIN_SV: {"label": "Plain population count", "arcs": _plain_arcs()},
    }
    obs = {
        f"{_PLAIN_SV}|{_FRA}": [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}],
    }
    result = _run(
        detect_svs=[_PLAIN_SV],
        sv_scores={_PLAIN_SV: 0.9},
        graph_nodes=graph_nodes,
        obs=obs,
        role_query="aid from France to Kenya",
    )
    assert result.no_data_reason != "role_incompatible"


def test_single_anchor_destination_on_a_declaring_recall_answers_as_subject():
    """The missing door role leaves Kenya as a subject under the plain shape."""
    graph_nodes = {
        _KEN: {"label": "Kenya", "type": "Country"},
        _PLAIN_SV: {"label": "Plain population count", "arcs": _plain_arcs()},
        _DECLARING_SV: {"label": "Creditor/debtor flow", "arcs": _declaring_arcs()},
    }
    obs = {
        f"{_PLAIN_SV}|{_KEN}": [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}],
    }
    role_query = "aid to Kenya"
    graph = FakeGraph(
        nodes=graph_nodes,
        detect={role_query: {"svs": [_PLAIN_SV, _DECLARING_SV], "cosine_scores": [0.9, 0.5]}},
        resolve={"Kenya": _KEN},
        obs=obs,
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Kenya"],
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=FakeSdmxClient(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert {sv.ref.ref for sv in result.specs[0].resolved.members} == {_PLAIN_SV}


def test_single_anchor_source_on_a_declaring_recall_answers_as_subject():
    """A single source anchor leaves Kenya as a subject under the plain shape.

    This is a known residual of the opposing-anchors rule.
    """
    graph_nodes = {
        _KEN: {"label": "Kenya", "type": "Country"},
        _PLAIN_SV: {"label": "Plain population count", "arcs": _plain_arcs()},
        _DECLARING_SV: {"label": "Creditor/debtor flow", "arcs": _declaring_arcs()},
    }
    obs = {
        f"{_PLAIN_SV}|{_KEN}": [{"earliestDate": "2015", "latestDate": "2020", "obsCount": 5}],
    }
    role_query = "aid from Kenya"
    graph = FakeGraph(
        nodes=graph_nodes,
        detect={role_query: {"svs": [_PLAIN_SV, _DECLARING_SV], "cosine_scores": [0.9, 0.5]}},
        resolve={"Kenya": _KEN},
        obs=obs,
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Kenya"],
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=FakeSdmxClient(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "definite"
    assert len(result.specs) == 1
    assert {sv.ref.ref for sv in result.specs[0].resolved.members} == {_PLAIN_SV}


def test_no_confirmed_statvar_returns_variable_not_resolved():
    """Missing candidate nodes produce ``variable_not_resolved`` despite flow wording."""
    graph = FakeGraph(
        nodes={},
        detect={"aid flows": {"svs": ["test/ns/Unconfirmable"], "cosine_scores": [0.9]}},
        resolve={},
        obs={},
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=[],
            date_request=None,
            detect_query="aid flows",
            role_query="aid flows",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=FakeSdmxClient(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"
