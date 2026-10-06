"""A candidate keeps the place binding it was admitted with after the role-class election
narrows it.

The election (`bound_values.elected_role_class_shape`) can narrow a debt shape holding a
recipient-only total and its by-creditor variants to the by-creditor class, which requires
both roles. Kenya, named with no direction, binds as the recipient on the shape as admitted
and would bind to no role on the narrowed one. Each candidate carries the place context it
was bound with into grounding (`candidates._Candidate.place`), and an alternate binds the
undirected place on the axis of the elected shape's binding as admitted
(`ResolveState.elected_place_roles`).
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.region.pipeline import resolve_variable
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_candidate_place_roles import _GNI_PPP
from tests.engine.test_candidate_place_roles import _NODES as _GNI_NODES
from tests.engine.test_role_class_shapes import (
    _BY_CREDITOR,
    _BY_CREDITOR_GUARANTEED,
    _DEBT_NODES,
    _PLACES,
    _TOTAL,
    _node,
)
from tests.fixtures import FakeGraph, make_resolver

_KEN = "country/KEN"
_CHN = "country/CHN"
_QUERY = "guaranteed debt of Kenya"
_NODES = {
    **_PLACES,
    **{dcid: _DEBT_NODES[dcid] for dcid in (_TOTAL, _BY_CREDITOR, _BY_CREDITOR_GUARANTEED)},
    _GNI_PPP: _GNI_NODES[_GNI_PPP],
}
_PAIR = RoleNames(donor="financingProvider", recipient="financingRecipient")
_YEARS = list(range(2015, 2023))
_COVERAGE_ROWS = [
    (_TOTAL, "f_total", "", _KEN, _YEARS),
    (_BY_CREDITOR, "f_by_creditor", _CHN, _KEN, _YEARS),
    (_BY_CREDITOR_GUARANTEED, "f_by_creditor_guaranteed", _CHN, _KEN, _YEARS),
]
_ROLE_NAMES = {
    _TOTAL: RoleNames(donor=None, recipient="financingRecipient"),
    _BY_CREDITOR: _PAIR,
    _BY_CREDITOR_GUARANTEED: _PAIR,
}
_GUARANTEED = SlotBindingDraft(
    axis="how",
    property_dcid="guarantee",
    kind="value",
    value_dcids=["Public"],
    evidence="mentioned",
)


class _BindLLM:
    """Elects the first menu shape, binding `guarantee` when asked, and lists the second as
    also plausible."""

    def __init__(self, *, bind_guarantee: bool) -> None:
        self._bindings = [_GUARANTEED] if bind_guarantee else []

    async def generate_structured(self, *, prompt, system, schema):
        return (
            _UniversalBindOutput(
                chosen_shape_index=0, bindings=self._bindings, also_plausible=[1], ask=None
            ),
            None,
        )


def _resolve(recalled: list[str], *, bind_guarantee: bool):
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": recalled, "cosine_scores": [0.9, 0.89, 0.88, 0.87]}},
        resolve={"Kenya": _KEN, "China": _CHN},
        obs={
            f"{_GNI_PPP}|{_KEN}": [{"earliestDate": "1990", "latestDate": "2024", "obsCount": 35}]
        },
    )
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(bind_guarantee=bind_guarantee),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            base_timing={},
            coverage=FakeCoverageService(
                build_coverage_index(_COVERAGE_ROWS, role_names=_ROLE_NAMES)
            ).snapshot(),
        )
    )


@pytest.mark.parametrize(
    ("recalled", "bind_guarantee", "debt_member"),
    [
        pytest.param(
            [_TOTAL, _BY_CREDITOR, _BY_CREDITOR_GUARANTEED, _GNI_PPP],
            True,
            _BY_CREDITOR_GUARANTEED,
            id="the-elected-shape-narrowed-by-a-bound-guarantee",
        ),
        pytest.param(
            [_GNI_PPP, _BY_CREDITOR_GUARANTEED, _BY_CREDITOR, _TOTAL],
            False,
            _BY_CREDITOR_GUARANTEED,
            id="an-alternate-narrowed-by-its-first-member",
        ),
        pytest.param(
            [_GNI_PPP, _TOTAL, _BY_CREDITOR, _BY_CREDITOR_GUARANTEED],
            False,
            _TOTAL,
            id="an-alternate-the-election-leaves-on-the-total",
        ),
    ],
)
def test_a_narrowed_candidate_answers_under_its_admitted_places(
    recalled: list[str], bind_guarantee: bool, debt_member: str
):
    result = _resolve(recalled, bind_guarantee=bind_guarantee)

    assert result.status == "candidates"
    by_member = {
        tuple(sv.ref.dcid for sv in spec.resolved.members): spec.definition.identity.entities
        for spec in result.specs
    }
    assert by_member == {
        (debt_member,): {"destination": [_KEN]},
        (_GNI_PPP,): {"observationAbout": [_KEN]},
    }


_SERVICE = "DebtService_ByCreditor_Guaranteed"
_SERVICE_NODE = _node(
    "Guaranteed debt service, by creditor",
    ["financingProvider", "financingRecipient"],
    population_type="DebtServicePayment",
    positionType="AmountOutstanding",
    guarantee="Public",
)


def test_a_two_role_alternate_binds_the_place_where_the_elected_shape_bound_it():
    """The bound guarantee narrows the elected debt shape to its by-creditor class, which
    requires both roles. The by-creditor debt-service alternate takes Kenya on the axis the
    elected shape bound it under before the narrowing, the destination, and stays a
    candidate."""
    recalled = [_TOTAL, _BY_CREDITOR, _BY_CREDITOR_GUARANTEED, _SERVICE]
    graph = FakeGraph(
        nodes={**_NODES, _SERVICE: _SERVICE_NODE},
        detect={_QUERY: {"svs": recalled, "cosine_scores": [0.9, 0.89, 0.88, 0.87]}},
        resolve={"Kenya": _KEN, "China": _CHN},
        obs={},
    )
    coverage = FakeCoverageService(
        build_coverage_index(
            [*_COVERAGE_ROWS, (_SERVICE, "f_service", _CHN, _KEN, _YEARS)],
            role_names={**_ROLE_NAMES, _SERVICE: _PAIR},
        )
    ).snapshot()

    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(bind_guarantee=True),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            base_timing={},
            coverage=coverage,
        )
    )

    assert result.status == "candidates"
    assert {
        tuple(sv.ref.dcid for sv in spec.resolved.members): spec.definition.identity.entities
        for spec in result.specs
    } == {
        (_BY_CREDITOR_GUARANTEED,): {"destination": [_KEN]},
        (_SERVICE,): {"destination": [_KEN]},
    }
