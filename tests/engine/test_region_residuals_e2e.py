"""End-to-end tests for a recipient region read as one sum.

France's aid to Africa goes through the parsed door, elects the CRS's member, and finds no
Africa total there (DAC2a publishes it), so Africa binds its members. Read as one sum it
binds the seven residual buckets DAC's Africa total counts beside them, which the CRS
publishes as recipients, and states the region choice. A ranking and a request for
"African countries" bind the members alone. The graph holds Africa, its members and its
residuals, as the dev graph does; coverage holds France's series to every member.
"""

from __future__ import annotations

from typing import Any

import pytest

from code_index.shared import get_index
from entity_resolution.shared import shared_membership_policy
from search.engine.departures import spec_departure_facts
from search.models import (
    ChoiceOrigin,
    DatasetResubmitInput,
    DefinitePart,
    Entry,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.code_index.aid_identity import CRS_PROVENANCE
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _CRS_MEMBER, _ScriptedLLM
from tests.engine.test_whole_total_bind_e2e import _FRA, _VARIABLE, _graph, _parsed
from tests.fixtures import FakeGraph

_AFRICA_MEMBERS = shared_membership_policy().members("africa").members
_AFRICA_RESIDUALS = get_index().totals_by_group["africa"].residuals
_ALL = "region:members_and_residuals"
_MEMBERS = "region:members_only"
_FROM_FRANCE = {"role": "from", "text": "France"}


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_coverage(
        monkeypatch,
        [(_CRS_MEMBER, "crs", _FRA, member, [2019, 2020, 2021]) for member in _AFRICA_MEMBERS],
        role_names={
            _CRS_MEMBER: RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
        facet_metadata={"crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD")},
    )


def _africa_graph() -> FakeGraph:
    graph = _graph(members=(_CRS_MEMBER,))
    nodes = graph._nodes
    nodes["africa"] = {"label": "Africa", "type": "Continent"}
    for member in _AFRICA_MEMBERS:
        nodes.setdefault(member, {"label": member, "type": "Country"})
    for residual in _AFRICA_RESIDUALS:
        nodes[residual] = {"label": get_index().parties[residual].label, "type": "Organization"}
    graph._resolve["Africa"] = "africa"
    return graph


def _run(request: ResolveRequest) -> DefinitePart:
    response = offline_resolve(
        request,
        graph=_africa_graph(),
        llm=_ScriptedLLM(_VARIABLE),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part


def _ask(africa: dict[str, Any], *operations: dict[str, str]) -> DefinitePart:
    return _run(
        _parsed(
            {
                "variable": _VARIABLE,
                "entities": [_FROM_FRANCE, {"role": "to", "text": "Africa", **africa}],
                "operations": list(operations),
                "time": "2021",
            }
        )
    )


def _destination(spec: Spec) -> list[str]:
    return spec.definition.identity.entities["destination"]


def _region(spec: Spec) -> tuple[str, str, str | None]:
    """The region choice's active option, provenance and the residual option's code."""
    scope = spec.resolved.scope_choices
    assert scope is not None
    (choice,) = [c for c in scope.choices if c.axis == "region"]
    residuals = next(o for o in choice.options if o.option_id == _ALL)
    return choice.active, choice.provenance, residuals.availability.code


@pytest.mark.parametrize(
    "operations",
    [(), ({"phrase": "total aid", "verb": "aggregate", "over": "destination"},)],
    ids=["no-operation", "aggregate-over-the-recipients"],
)
def test_a_summed_region_binds_its_residuals_and_states_the_choice(operations):
    part = _ask({}, *operations)

    spec = part.spec
    assert _destination(spec) == sorted((*_AFRICA_MEMBERS, *_AFRICA_RESIDUALS))
    assert _region(spec) == (_ALL, "default", None)
    assert spec.resolved.entity_groups is not None
    assert spec.resolved.entity_groups["destination"].ref == "africa"
    # The residuals hold no series and depart as no member; the group counts its members.
    codes = [d.code for d in part.departures]
    assert "group_read_as_members" in codes
    assert "entity_unavailable" not in codes
    (fact,) = spec_departure_facts(spec, operations=())
    assert (fact.group_size, fact.excluded, len(fact.included)) == (
        len(_AFRICA_MEMBERS),
        (),
        len(_AFRICA_RESIDUALS),
    )


@pytest.mark.parametrize(
    ("africa", "operation", "region"),
    [
        pytest.param(
            {},
            {"phrase": "top recipients", "verb": "rank"},
            (_MEMBERS, "default", "residuals_need_summed_region"),
            id="a-ranking",
        ),
        pytest.param(
            {"members": True},
            {"phrase": "which African countries received the most", "verb": "rank"},
            (_MEMBERS, "explicit", "residuals_need_summed_region"),
            id="a-ranking-of-african-countries",
        ),
        pytest.param(
            {"members": True},
            {"phrase": "total aid", "verb": "aggregate"},
            (_MEMBERS, "explicit", None),
            id="african-countries-summed",
        ),
    ],
)
def test_a_ranking_or_a_members_ask_binds_the_members_alone(africa, operation, region):
    part = _ask(africa, {**operation, "over": "destination"})

    spec = part.spec
    assert _destination(spec) == sorted(_AFRICA_MEMBERS)
    assert _region(spec) == region
    entries = spec.resolved.scope_choices.entries if spec.resolved.scope_choices else []
    assert [(e.option_id, e.origin.kind) for e in entries] == (
        [(_MEMBERS, "text")] if region[1] == "explicit" else []
    )


def test_a_switch_round_trips_through_dataset_resubmit():
    """Switching the summed region to its members and back, each posted over the scope's
    base identity with its entries, returns the first answer's spec_id."""
    first = _ask({}).spec
    scope = first.resolved.scope_choices
    assert scope is not None

    def switched(option: str) -> Spec:
        switch = Entry(
            axis="region",
            option_id=option,
            provenance="explicit",
            origin=ChoiceOrigin(choice_id=f"{scope.id}:region"),
        )
        entry = ResubmitIdentityEntry(
            variable_text=_VARIABLE,
            identity=scope.base.identity,
            entity_groups=first.resolved.entity_groups,
            scope_entries=[*scope.entries, switch],
        )
        return _run(ResolveRequest(input=DatasetResubmitInput(entries=[entry]))).spec

    members = switched(_MEMBERS)
    back = switched(_ALL)

    assert _destination(members) == sorted(_AFRICA_MEMBERS)
    assert _region(members)[:2] == (_MEMBERS, "explicit")
    assert members.spec_id != first.spec_id
    assert back.spec_id == first.spec_id
    assert back.resolved.entity_groups == members.resolved.entity_groups
    assert back.resolved.entity_groups == first.resolved.entity_groups
    assert _region(back)[:2] == (_ALL, "explicit")
