"""A query naming both flow ends elects among role-bearing shapes only.

A subject-only shape has one subject slot, so it cannot hold an origin and a destination
at once, and `place_roles.bind_place_roles` refuses it ``role_incompatible``. When the
query's places name both ends and a role-bearing shape survives the score floor,
`_phase_prune_and_rank` drops subject-only shapes before the bind menu is built
(`subject_shapes_refused`). A query
naming one end or none keeps every shape on the menu. The fixtures mirror "Kenya's debt to
China" on the dev graph: a subject-only government debt series outranks the by-creditor
series that declares both roles.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.region import phase_rank_bind
from search.engine.region.pipeline import resolve_variable
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.fixtures import FakeGraph, make_resolver

_KEN = "country/KEN"
_CHN = "country/CHN"
_SUBJECT_SV = "TESTFIXTURE/TwoEndedMenu/DebtGovernment"
_FLOW_SV = "TESTFIXTURE/TwoEndedMenu/DebtByCreditor"


def _debt_arcs(qualifier: str, roles: tuple[str, ...] = ()) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": qualifier}]},
    }
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return arcs


_NODES = {
    _SUBJECT_SV: {"label": "Government debt", "arcs": _debt_arcs("Government")},
    _FLOW_SV: {
        "label": "Debt by creditor",
        "arcs": _debt_arcs("ByCreditor", ("creditor", "debtor")),
    },
    _KEN: {"label": "Kenya", "type": "Country"},
    _CHN: {"label": "China", "type": "Country"},
}


class _ElectFirst:
    """Elects the first shape on the menu with no bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _resolve(monkeypatch, query: str, entities: list[str]):
    """Resolve `query` and return the result with the StatVars of each bind-menu shape."""
    menus: list[list[tuple[str, ...]]] = []
    real_bind = phase_rank_bind.bind_universal

    async def _spy(variable, shapes, **kwargs):
        menus.append([tuple(shape.sv_arc_facts or ()) for shape in shapes])
        return await real_bind(variable, shapes, **kwargs)

    monkeypatch.setattr(phase_rank_bind, "bind_universal", _spy)
    graph = FakeGraph(
        nodes=_NODES,
        detect={query: {"svs": [_SUBJECT_SV, _FLOW_SV], "cosine_scores": [0.9, 0.8]}},
        resolve={"Kenya": _KEN, "China": _CHN},
        obs={},
    )
    coverage = FakeCoverageService(
        build_coverage_index(
            [(_FLOW_SV, "f1", _CHN, _KEN, [2020, 2021, 2022])],
            role_names={_FLOW_SV: RoleNames(donor="creditor", recipient="debtor")},
        )
    )
    result = asyncio.run(
        resolve_variable(
            "debt",
            entities=entities,
            date_request=None,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirst(),
            sdmx=FakeSdmxClient(observations={}),
            base_steps=[],
            base_timing={},
            coverage=coverage.snapshot(),
        )
    )
    assert len(menus) == 1, menus
    return result, menus[0]


def test_a_two_ended_query_elects_the_role_bearing_shape(monkeypatch) -> None:
    """With China as origin and Kenya as destination the menu holds only the by-creditor
    series, so the top-ranked government debt series cannot be elected and refuse."""
    result, menu = _resolve(monkeypatch, "debt from China to Kenya", ["China", "Kenya"])

    assert menu == [(_FLOW_SV,)]
    assert result.status == "definite", result
    spec = result.specs[0]
    assert [item.ref for item in spec.resolved.stat_vars.items] == [_FLOW_SV]
    assert spec.definition.identity.entities == {"origin": [_CHN], "destination": [_KEN]}


@pytest.mark.parametrize(
    ("query", "entities"),
    [
        pytest.param("debt to Kenya", ["Kenya"], id="one-flow-end"),
        pytest.param("debt of Kenya", ["Kenya"], id="no-flow-end"),
        pytest.param("debt of Kenya and China", ["Kenya", "China"], id="two-undirected-places"),
    ],
)
def test_a_query_naming_one_flow_end_or_none_keeps_every_shape(
    monkeypatch, query: str, entities: list[str]
) -> None:
    """The subject-only series stays on the menu, ranked first by its recall score."""
    _result, menu = _resolve(monkeypatch, query, entities)

    assert menu == [(_SUBJECT_SV,), (_FLOW_SV,)]
