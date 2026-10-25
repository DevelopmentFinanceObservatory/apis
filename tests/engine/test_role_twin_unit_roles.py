"""Test that a unit decision on an IDS-style twin shape hands SDMX the survivors' roles.

The total declares `financingRecipient`; its by-creditor twin declares
`financingProvider` and `financingRecipient`. The shape carries the union, but a
query that binds only the recipient resolves to the total, so SDMX must see
`("financingRecipient",)`.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from materialise import MaterialisationRequest
from materialise.errors import MaterialisationError
from materialise.facets import FacetSelection
from materialise.member_conversion import decide_member_conversion
from materialise.recipes import RecipeNotDeclaredError
from search.engine.door_names import own_texts
from search.engine.region import candidate_grounding
from search.engine.region.candidate_grounding import _prepare_candidate_context
from search.engine.region.candidates import _Candidate
from search.engine.region.convertibility import survivor_declared_roles
from search.engine.region.phase_answer import _decide_bound_unit_convertibility
from search.engine.region.place_context import PlaceContext, place_context
from search.engine.region.promote_ground import _decide_unit_convertibility
from search.engine.region.promote_state import _PromoteState
from search.engine.region.state import ResolveState
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.retrieve import Materialised, materialise
from search.engine.shape import ShapeDraft
from search.models import (
    CoverageBare,
    IdentityCurrencyRole,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
)
from tests.engine._builders import flow_place_roles, place_sets
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_role_twin_resolution import (
    _BY_CREDITOR,
    _TOTAL,
    _shape,
    _twin_coverage,
    _twin_nodes,
)
from tests.fixtures import FakeGraph
from tests.materialise._rate_table_support import CONVERTIBLE_STAT_VAR_ARCS, conversion_arcs
from tests.materialise.conftest import FakeGraphReader

_DESTINATION = "country/KEN"
_TOTAL_ROLES = ("financingRecipient",)


def _served_context(shape, roles, **kwargs):
    """`place_context` over every StatVar `shape` holds, the members a mint serves for it."""
    return place_context(shape, roles, stat_vars=list(shape.sv_arc_facts or {}), **kwargs)


class _StopAfterRequest(Exception):
    """Ends a unit decision once its first materialisation request is captured."""


def _destination_place(shape: ShapeDraft) -> PlaceContext:
    """The place context of a resubmit posting `_DESTINATION` as its one destination."""
    return _served_context(
        shape,
        flow_place_roles(shape, destination=[_DESTINATION]),
        presence=None,
    )


@pytest.fixture
def unit_request() -> IdentityUnitRequest:
    return IdentityUnitRequest(currency="USD", prices="current")


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[MaterialisationRequest]:
    """Capture requests reaching `decide_member_unit` from the one unit decision
    (`convertibility.decide_part_unit`), which the candidate fan-out calls too."""
    requests: list[MaterialisationRequest] = []

    def _capture(request: MaterialisationRequest, **_: object) -> None:
        requests.append(request)
        raise _StopAfterRequest

    monkeypatch.setattr("search.engine.region.convertibility.decide_member_unit", _capture)
    return requests


@pytest.fixture
def twin_shape() -> ShapeDraft:
    return _shape(_twin_nodes(), [_TOTAL, _BY_CREDITOR])


@pytest.fixture
def total_result(twin_shape: ShapeDraft) -> Materialised:
    """Resolve the twin shape with only the recipient bound, which admits the total."""
    result = materialise(
        twin_shape,
        [],
        {"financingRecipient": [_DESTINATION]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=_twin_coverage().snapshot(),
    )
    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_TOTAL]
    # The shape's union differs from what the survivor declares.
    assert twin_shape.declared_roles == ("financingProvider", "financingRecipient")
    return result


def _resolve_state(shape: ShapeDraft, unit_request: IdentityUnitRequest) -> ResolveState:
    def _none(*_: object) -> Any:
        return None

    return ResolveState(
        variable="debt",
        # One undirected place, which binds to the twin shape's one required role.
        entities=["Kenya"],
        entity_texts=own_texts(["Kenya"]),
        place_sets=place_sets({"Kenya": _DESTINATION}),
        date_request=None,
        detect_query="debt",
        role_query="debt",
        pac=True,
        graph=_none(),
        resolver=_none(),
        admission=_none(),
        llm=_none(),
        sdmx=_none(),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=cast(Any, object()),
        materialise_sdmx=cast(Any, object()),
        timing={},
        warnings=[],
        shape_draft=shape,
        place=_destination_place(shape),
        coverage=empty_coverage_snapshot(),
        unit_request=unit_request,
    )


def test_survivor_roles_come_from_the_resolved_members(total_result: Materialised) -> None:
    assert survivor_declared_roles(total_result) == _TOTAL_ROLES


def test_result_without_member_facts_refuses() -> None:
    bare = Materialised(sv_dcids=["a"], has_data=True, coverage=CoverageBare(has_data=True))
    with pytest.raises(ValueError, match="member_facts"):
        survivor_declared_roles(bare)


def test_agreeing_survivors_share_their_signature() -> None:
    arcs = {"observationProperties": {"nodes": [{"dcid": "financingRecipient"}]}}
    result = Materialised(
        sv_dcids=["a", "b"],
        has_data=True,
        coverage=CoverageBare(has_data=True),
        member_facts={"a": ArcsMemberFacts(arcs=arcs), "b": ArcsMemberFacts(arcs=arcs)},
    )

    assert survivor_declared_roles(result) == _TOTAL_ROLES


def test_text_path_unit_decision_hands_sdmx_the_survivors_roles(
    twin_shape: ShapeDraft,
    total_result: Materialised,
    unit_request: IdentityUnitRequest,
    captured: list[MaterialisationRequest],
) -> None:
    state = _resolve_state(twin_shape, unit_request)

    with pytest.raises(_StopAfterRequest):
        asyncio.run(_decide_bound_unit_convertibility(state, total_result))

    assert [r.declared_roles for r in captured] == [_TOTAL_ROLES]


def test_promote_path_unit_decision_hands_sdmx_the_survivors_roles(
    twin_shape: ShapeDraft,
    total_result: Materialised,
    unit_request: IdentityUnitRequest,
    captured: list[MaterialisationRequest],
) -> None:
    inp = SimpleNamespace(
        identity=SimpleNamespace(facets={"unit": IdentityFacetValueRequest(request=unit_request)})
    )
    state = _PromoteState(
        inp=cast(Any, inp),
        graph=cast(Any, None),
        sdmx=cast(Any, None),
        deadline=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        materialise_graph=cast(Any, object()),
        materialise_sdmx=cast(Any, object()),
        coverage=empty_coverage_snapshot(),
    )
    state.shape_draft = twin_shape
    state.entity_dcid = _DESTINATION
    state.place = _destination_place(twin_shape)

    with pytest.raises(_StopAfterRequest):
        _decide_unit_convertibility(state, total_result)

    assert [r.declared_roles for r in captured] == [_TOTAL_ROLES]


def test_candidate_path_unit_decision_hands_sdmx_the_survivors_roles(
    twin_shape: ShapeDraft,
    total_result: Materialised,
    unit_request: IdentityUnitRequest,
    captured: list[MaterialisationRequest],
) -> None:
    state = _resolve_state(twin_shape, unit_request)

    with pytest.raises(_StopAfterRequest):
        _prepare_candidate_context(
            state, [_Candidate(twin_shape, [], total_result, _destination_place(twin_shape), None)]
        )

    assert [r.declared_roles for r in captured] == [_TOTAL_ROLES]


def test_candidate_unit_decision_binds_the_candidates_own_probe_places(
    monkeypatch: pytest.MonkeyPatch,
    twin_shape: ShapeDraft,
    total_result: Materialised,
    unit_request: IdentityUnitRequest,
) -> None:
    """A candidate's unit election binds the places its own context probes, not those of the
    shape in play."""
    seen: list[dict[str, list[str]]] = []

    def _capture(sv_dcids, *, role_bindings: dict[str, list[str]], **_: object):
        seen.append(role_bindings)
        raise _StopAfterRequest

    monkeypatch.setattr(candidate_grounding, "_candidate_unit_elections", _capture)
    state = _resolve_state(twin_shape, unit_request)
    own_places = dataclasses.replace(
        _destination_place(twin_shape), probe_dcids=(_DESTINATION, "country/UGA")
    )

    with pytest.raises(_StopAfterRequest):
        _prepare_candidate_context(
            state, [_Candidate(twin_shape, [], total_result, own_places, None)]
        )

    assert seen == [{"financingRecipient": [_DESTINATION, "country/UGA"]}]


_CREDITOR = "country/FRA"
_DEBT_PROVENANCE = "dc/base/RoleTwinUnitRolesSrc"


def _role_currency_decision(
    monkeypatch: pytest.MonkeyPatch, currency_role: Literal["origin", "destination"]
) -> tuple[Mapping[str, MaterialisationError], FakeGraphReader]:
    """Run the candidate path's unit election over the twin's elected total for a role-shaped
    currency request, and return the candidate's refusals and the materialise graph.

    Facet selection is stubbed to the total's one USD facet. The conversion decision is the
    real one, over a graph that maps both the debtor and the creditor to a national currency,
    so a currency read could succeed for either role."""
    graph = FakeGraphReader(
        arcs={
            _TOTAL: dict(CONVERTIBLE_STAT_VAR_ARCS),
            _BY_CREDITOR: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id="ConversionRecipeIDS",
                provenance_id=_DEBT_PROVENANCE,
                currencies={
                    "CurrencyUSD": (),
                    "CurrencyKES": (_DESTINATION,),
                    "CurrencyEUR": (_CREDITOR,),
                },
            ),
        }
    )

    def _decide(request: MaterialisationRequest, *, graph: Any, **_: object):
        unit_request = request.facet_bindings["unit"].request
        selections = {
            axis: FacetSelection(
                axis=axis,
                ref=ref,
                basis="bound",
                default_policy=None,
                default_policy_version=None,
                selection_reason=None,
                dispatch_facet_id="f1",
                request=unit_request if axis == "unit" else None,
            )
            for axis, ref in (
                ("unit", "CurrencyUSD"),
                ("provenance", _DEBT_PROVENANCE),
                ("measurementMethod", None),
            )
        }
        decide_member_conversion(request=request, selections=selections, graph=graph)
        return selections

    monkeypatch.setattr("search.engine.region.convertibility.decide_member_unit", _decide)
    twin_shape = _shape(_twin_nodes(), [_TOTAL, _BY_CREDITOR])
    total = materialise(
        twin_shape,
        [],
        {"financingRecipient": [_DESTINATION]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=_twin_coverage().snapshot(),
    )
    assert isinstance(total, Materialised)
    request = IdentityUnitRequest(
        currency=IdentityCurrencyRole(role=currency_role), prices="current"
    )
    state = _resolve_state(twin_shape, request)
    state.materialise_graph = graph
    prep = _prepare_candidate_context(
        state, [_Candidate(twin_shape, [], total, _destination_place(twin_shape), None)]
    )
    return prep.unit_refusals[0], graph


def test_candidate_role_currency_decision_admits_a_role_the_elected_class_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the debtor's currency is a role the elected total declares, so the candidate's
    election converts. The decision reads no national currency; conversion reads it from the
    rows the elected members serve."""
    refusals, graph = _role_currency_decision(monkeypatch, "destination")
    assert refusals == {}
    assert graph.calls["node_subjects"] == 0


def test_candidate_role_currency_decision_refuses_a_role_only_the_unelected_class_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The role-class election runs before the role-currency read. The creditor's currency is
    a role only the by-creditor twin declares, and the shape's union holds it, but a query
    binding the debtor alone elects the total. Its election refuses the role from the total's
    declared roles before any `nationalCurrencyOf` read, and keeps the refusal for the
    candidates door's answer."""
    refusals, graph = _role_currency_decision(monkeypatch, "origin")
    assert list(refusals) == [_TOTAL]
    refusal = refusals[_TOTAL]
    assert isinstance(refusal, RecipeNotDeclaredError)
    assert refusal.cause == "role_undeclared"
    assert graph.calls["node_subjects"] == 0
