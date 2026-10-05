"""A role-bearing shape holding several role sets answers through the class its bound roles
elect.

A recipient-only debt total and its by-creditor variant share one shape. A place named
without a direction binds as the recipient, the plain total answers it, and the
by-creditor rows answer a named creditor. The member election is the mint's
(`statvar_members.elect_role_class`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import time

import pytest

from code_index.shared import get_index, get_measure_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.departures import warning_key
from search.engine.errors import FamilyNotResolvableError, RoleSignatureError
from search.engine.fanout import _success_directional
from search.engine.options.population_memo import _population_key
from search.engine.qualifier_match import QualifierMatch
from search.engine.region import candidate_collapse, resolve_spec_resubmit, resolve_variable
from search.engine.region.phase_rank_bind import _phase_prune_and_rank
from search.engine.region.result import with_runnable_operations
from search.engine.region.state import EngineOptions
from search.engine.retrieval.bound_values import (
    MembershipResolution,
    RoleClassWalks,
    bind_shared_values,
    collapse_measure_families,
    elect_five_tuple_family,
    elected_role_class_shape,
    entity_role_keys,
    five_tuple_member_facts,
    index_carries_shape,
    measure_family_cache_key,
    measure_family_partitions,
    posted_members_outside_role_class,
    shape_member_facts,
    shape_query,
)
from search.engine.retrieval.derive import derive_shapes
from search.engine.role_text import DoorRole
from search.engine.shape import (
    ShapeDraft,
    required_role_names,
)
from search.heading.departure_fact import DepartureFact
from search.models import BaseIdentity, Operation, ResubmitIdentityPayload
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from series_coverage.shared import CoverageSnapshot
from spec_registry.identity import recompute_identity
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_bind_slot_reserve import _state as _prune_state
from tests.fixtures import FakeGraph, make_resolver

_RECIPIENT = ["financingRecipient"]
_PAIR = ["financingProvider", "financingRecipient"]

_TOTAL = "Debt_Total"
_TOTAL_GUARANTEED = "Debt_Total_Guaranteed"
_BY_CREDITOR = "Debt_ByCreditor"
_BY_CREDITOR_GUARANTEED = "Debt_ByCreditor_Guaranteed"


def _arcs(
    label: str,
    roles: list[str],
    *,
    population_type: str = "FinancialPosition",
    qualifier: str | None = "Nominal",
    **constraints: str,
) -> dict:
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": population_type}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }
    if qualifier is not None:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": qualifier}]}
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return arcs


def _node(label: str, roles: list[str], **kwargs: str) -> dict:
    return {"label": label, "type": "StatisticalVariable", "arcs": _arcs(label, roles, **kwargs)}


_DEBT_NODES = {
    _TOTAL: _node("External debt stocks", _RECIPIENT, positionType="AmountOutstanding"),
    _TOTAL_GUARANTEED: _node(
        "Guaranteed external debt stocks",
        _RECIPIENT,
        positionType="AmountOutstanding",
        guarantee="Public",
    ),
    _BY_CREDITOR: _node(
        "External debt stocks, by creditor", _PAIR, positionType="AmountOutstanding"
    ),
    _BY_CREDITOR_GUARANTEED: _node(
        "Guaranteed external debt stocks, by creditor",
        _PAIR,
        positionType="AmountOutstanding",
        guarantee="Public",
    ),
}


def _one_role_coverage(nodes: dict, obs: dict) -> CoverageSnapshot:
    """The series coverage index of the recipient-only members among `nodes`, with a row at
    each place `obs` holds them at. The index holds one-role series with no donor
    (`series_coverage.schema.ABSENT_ENTITY`), so Search reads their facets there."""
    recipient_only = {
        dcid
        for dcid, node in nodes.items()
        if [
            n["dcid"]
            for n in node.get("arcs", {}).get("observationProperties", {}).get("nodes", [])
        ]
        == ["financingRecipient"]
    }
    rows = [
        (dcid, f"f_{dcid}", ABSENT_ENTITY, place, list(range(2015, 2023)))
        for key in obs
        for dcid, place in [key.split("|", 1)]
        if dcid in recipient_only
    ]
    role_names = {
        dcid: RoleNames(donor=None, recipient="financingRecipient") for dcid in recipient_only
    }
    return FakeCoverageService(build_coverage_index(rows, role_names=role_names)).snapshot()


def _debt_shape(*members: str) -> ShapeDraft:
    graph = FakeGraph(nodes=_DEBT_NODES, obs={}, detect={}, resolve={})
    (shape,) = derive_shapes(confirmed_svs=list(members or _DEBT_NODES), graph=graph)
    return shape


# ---------------------------------------------------------------------------
# The place a shape binds without a direction
# ---------------------------------------------------------------------------


def test_an_undirected_place_binds_the_role_every_member_declares():
    """A shape holding a recipient-only total beside its by-creditor variant requires the
    recipient alone, so a place named without a direction binds as the recipient."""
    assert required_role_names(_debt_shape()) == ("financingRecipient",)
    assert required_role_names(_debt_shape(_TOTAL, _TOTAL_GUARANTEED)) == ("financingRecipient",)


def test_an_undirected_place_is_ambiguous_when_every_member_declares_two_roles():
    assert required_role_names(_debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)) == (
        "financingProvider",
        "financingRecipient",
    )


def test_a_question_naming_no_place_offers_a_one_role_shape_to_the_election():
    """The recipient-only shape scores highest, and bind is offered it beside the
    subject-only shape: coverage holds one-role series, so with no place named its
    recipients are discovered there."""
    subject = "Gdp_Subject"
    recipient = "Gni_Received"
    nodes = {
        subject: _node("Gross domestic product", [], population_type="EconomicActivity"),
        recipient: _node("Gross national income", _RECIPIENT, population_type="EconomicActivity"),
    }
    offered: list[list[str]] = []

    class _RecordingLLM:
        async def generate_structured(self, *, prompt, system, schema):
            offered.append(
                [
                    label
                    for label in ("Gross domestic product", "Gross national income")
                    if label in prompt
                ]
            )
            return _UniversalBindOutput(
                chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
            ), None

    query = "national income"
    graph = FakeGraph(
        nodes=nodes,
        detect={query: {"svs": [recipient, subject], "cosine_scores": [0.92, 0.9]}},
        resolve={},
        obs={},
    )
    asyncio.run(
        resolve_variable(
            query,
            entities=[],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_RecordingLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )

    assert offered == [["Gross domestic product", "Gross national income"]]


# ---------------------------------------------------------------------------
# The role class the bound roles elect
# ---------------------------------------------------------------------------


def _resolved(
    shape: ShapeDraft, *role_keys: str, bindings=(), **kwargs
) -> MembershipResolution[ShapeDraft]:
    return elected_role_class_shape(shape, shape_query(shape, list(bindings), role_keys), **kwargs)


def _narrowed(shape: ShapeDraft, *role_keys: str, bindings=(), **kwargs) -> ShapeDraft:
    return _resolved(shape, *role_keys, bindings=bindings, **kwargs).value


def test_a_bound_recipient_narrows_the_shape_to_the_recipient_only_members():
    narrowed = _narrowed(_debt_shape(), "destination")

    assert list(narrowed.sv_arc_facts or {}) == [_TOTAL, _TOTAL_GUARANTEED]
    assert narrowed.declared_roles == ("financingRecipient",)
    assert narrowed.label == "External debt stocks"


def test_a_named_creditor_narrows_the_shape_to_the_by_creditor_members():
    narrowed = _narrowed(_debt_shape(), "origin", "destination")

    assert list(narrowed.sv_arc_facts or {}) == [_BY_CREDITOR, _BY_CREDITOR_GUARANTEED]
    assert set(narrowed.declared_roles or ()) == {"financingProvider", "financingRecipient"}
    assert narrowed.label == "External debt stocks, by creditor"


def test_a_shape_of_one_role_set_is_returned_as_it_is():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    assert _narrowed(shape, "destination") is shape


def test_the_member_facts_of_a_recalled_shape_are_the_elected_role_class():
    shape = _debt_shape()

    recipient, index_sourced = shape_member_facts(shape, shape_query(shape, [], ["destination"]))
    creditor, _ = shape_member_facts(shape, shape_query(shape, [], ["origin", "destination"]))

    assert index_sourced is False
    assert list(recipient) == [_TOTAL, _TOTAL_GUARANTEED]
    assert list(creditor) == [_BY_CREDITOR, _BY_CREDITOR_GUARANTEED]


def test_the_election_runs_over_the_members_that_match_the_bindings():
    """Only the by-creditor class holds this guarantee value, so it answers a recipient."""
    nodes = {
        _TOTAL: _DEBT_NODES[_TOTAL],
        _BY_CREDITOR: _DEBT_NODES[_BY_CREDITOR],
        _BY_CREDITOR_GUARANTEED: _DEBT_NODES[_BY_CREDITOR_GUARANTEED],
    }
    (shape,) = derive_shapes(
        confirmed_svs=list(nodes), graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    )
    guaranteed = SlotBindingDraft(
        axis="how", property_dcid="guarantee", kind="value", value_dcids=["Public"]
    )

    narrowed = _narrowed(shape, "destination", bindings=[guaranteed])

    assert list(narrowed.sv_arc_facts or {}) == [_BY_CREDITOR, _BY_CREDITOR_GUARANTEED]


def test_entity_role_keys_are_the_keys_the_identity_carries():
    shape = _debt_shape()

    assert entity_role_keys(shape, {"financingRecipient": ["country/KEN"]}) == {"destination"}
    assert entity_role_keys(
        shape, {"financingRecipient": ["country/KEN"], "financingProvider": ["country/CHN"]}
    ) == {"destination", "origin"}
    assert entity_role_keys(shape, {"observationAbout": ["country/KEN"]}) == {"observationAbout"}
    assert entity_role_keys(shape, {"financingRecipient": []}) == frozenset()


def _walk_arcs(node: dict) -> dict[str, tuple[str, ...]]:
    return {
        prop: tuple(n["dcid"] for n in arc["nodes"] if "dcid" in n)
        for prop, arc in node["arcs"].items()
    }


def test_the_mints_walk_settles_a_role_class_the_recall_sample_misses():
    """The sample holds by-creditor members only. With the recipient alone bound, the walk
    finds the recipient-only total the mint elects, and the shape takes its members and name."""
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)
    walk = FakeMintGraphReader(arcs={dcid: _walk_arcs(node) for dcid, node in _DEBT_NODES.items()})
    arcs_graph = FakeGraph(nodes=_DEBT_NODES, obs={}, detect={}, resolve={})

    settled = _narrowed(shape, "destination", walks=RoleClassWalks(walk), arcs_graph=arcs_graph)

    assert list(settled.sv_arc_facts or {}) == [_TOTAL, _TOTAL_GUARANTEED]
    assert settled.declared_roles == ("financingRecipient",)
    assert settled.label == "External debt stocks"


def test_a_walk_that_elects_the_samples_class_with_every_role_bound_changes_nothing():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)
    walk = FakeMintGraphReader(arcs={dcid: _walk_arcs(node) for dcid, node in _DEBT_NODES.items()})
    arcs_graph = FakeGraph(nodes=_DEBT_NODES, obs={}, detect={}, resolve={})

    settled = _narrowed(
        shape, "origin", "destination", walks=RoleClassWalks(walk), arcs_graph=arcs_graph
    )

    assert settled is shape
    assert walk.calls["node_subjects"] == 1


# A recipient-only total that carries a guarantee status its by-creditor sibling does not.
_CODED_TOTAL = "Debt_Total_Coded"
_CODED_NODES = {
    _CODED_TOTAL: _node(
        "Guaranteed external debt stocks",
        _RECIPIENT,
        positionType="AmountOutstanding",
        guarantee="Public",
    ),
    _BY_CREDITOR: _DEBT_NODES[_BY_CREDITOR],
}


def test_a_less_constrained_class_outside_a_sample_with_every_role_bound_answers():
    """The sample holds the guaranteed recipient-only total, whose one role the debtor
    binds. The by-creditor series covers the debtor too and asserts fewer constraints, so
    the mint's election keeps it and drops the guaranteed total."""
    (shape,) = derive_shapes(
        confirmed_svs=[_CODED_TOTAL],
        graph=FakeGraph(nodes=_CODED_NODES, obs={}, detect={}, resolve={}),
    )

    settled = _narrowed(
        shape,
        "destination",
        walks=RoleClassWalks(_walk_reader(_CODED_NODES)),
        arcs_graph=FakeGraph(nodes=_CODED_NODES, obs={}, detect={}, resolve={}),
    )

    assert list(settled.sv_arc_facts or {}) == [_BY_CREDITOR]
    assert set(settled.declared_roles or ()) == {"financingProvider", "financingRecipient"}
    assert settled.label == "External debt stocks, by creditor"


def test_a_posted_member_with_every_role_bound_is_outside_a_less_constrained_family():
    (shape,) = derive_shapes(
        confirmed_svs=[_CODED_TOTAL],
        graph=FakeGraph(nodes=_CODED_NODES, obs={}, detect={}, resolve={}),
    )
    posted = {_CODED_TOTAL: _CODED_NODES[_CODED_TOTAL]["arcs"]}

    outside = posted_members_outside_role_class(
        posted,
        shape_query(shape, [], ["destination"]),
        walks=RoleClassWalks(_walk_reader(_CODED_NODES)),
    ).value

    assert outside == (_CODED_TOTAL,)


def test_a_walk_that_finds_the_samples_role_class_changes_nothing():
    by_creditor = {_BY_CREDITOR: _DEBT_NODES[_BY_CREDITOR]}
    shape = _debt_shape(_BY_CREDITOR)
    walk = FakeMintGraphReader(arcs={dcid: _walk_arcs(node) for dcid, node in by_creditor.items()})
    arcs_graph = FakeGraph(nodes=by_creditor, obs={}, detect={}, resolve={})

    assert (
        _narrowed(shape, "destination", walks=RoleClassWalks(walk), arcs_graph=arcs_graph) is shape
    )


def _walk_reader(nodes: dict) -> FakeMintGraphReader:
    return FakeMintGraphReader(arcs={dcid: _walk_arcs(node) for dcid, node in nodes.items()})


_TIE_NODES = {
    "Tie_ByCreditor": _node("Debt, by creditor", _PAIR, positionType="AmountOutstanding"),
    "Tie_ByCreditor_G": _node(
        "Guaranteed debt, by creditor", _PAIR, positionType="AmountOutstanding", guarantee="Public"
    ),
    "Tie_Total_G": _node("Guaranteed debt total", _RECIPIENT, guarantee="Public"),
}


def _tie_sample_shape() -> ShapeDraft:
    sample = {dcid: _TIE_NODES[dcid] for dcid in ("Tie_ByCreditor", "Tie_ByCreditor_G")}
    (shape,) = derive_shapes(
        confirmed_svs=list(sample), graph=FakeGraph(nodes=sample, obs={}, detect={}, resolve={})
    )
    return shape


def test_a_tie_only_the_walk_sees_is_not_answered_from_the_sample():
    """The sample holds by-creditor members. The graph also holds a recipient-only member
    with other constraint keys, which the election cannot order against them. The mint
    refuses the identity, so the sample does not answer it."""
    shape = _tie_sample_shape()
    arcs_graph = FakeGraph(nodes=_TIE_NODES, obs={}, detect={}, resolve={})

    with pytest.raises(RoleSignatureError):
        _narrowed(
            shape,
            "destination",
            walks=RoleClassWalks(_walk_reader(_TIE_NODES)),
            arcs_graph=arcs_graph,
        )


def test_the_walk_runs_once_for_a_query_asked_twice():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)
    reader = _walk_reader(_DEBT_NODES)
    walks = RoleClassWalks(reader)
    arcs_graph = FakeGraph(nodes=_DEBT_NODES, obs={}, detect={}, resolve={})

    first = _narrowed(shape, "destination", walks=walks, arcs_graph=arcs_graph)
    second = _narrowed(shape, "destination", walks=walks, arcs_graph=arcs_graph)

    assert (
        list(first.sv_arc_facts or {})
        == list(second.sv_arc_facts or {})
        == [
            _TOTAL,
            _TOTAL_GUARANTEED,
        ]
    )
    assert reader.calls["node_subjects"] == 1


def test_a_walk_past_its_bound_leaves_the_samples_answer():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)
    reader = _walk_reader(_DEBT_NODES)
    arcs_graph = FakeGraph(nodes=_DEBT_NODES, obs={}, detect={}, resolve={})

    settled = _narrowed(
        shape,
        "destination",
        walks=RoleClassWalks(reader),
        arcs_graph=arcs_graph,
        deadline=time.monotonic() - 1.0,
    )

    assert settled is shape
    assert reader.calls["node_subjects"] == 0


def test_a_walk_that_meets_a_member_at_fault_is_not_answered_from_the_sample():
    """A member with two values on one constraint makes the mint refuse the identity
    `member_not_resolvable`, so the sample does not answer it."""
    nodes = {
        **_DEBT_NODES,
        "Debt_TwoValues": {
            **_DEBT_NODES[_TOTAL],
            "arcs": {
                **_DEBT_NODES[_TOTAL]["arcs"],
                "positionType": {"nodes": [{"dcid": "AmountOutstanding"}, {"dcid": "Arrears"}]},
            },
        },
    }
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    with pytest.raises(FamilyNotResolvableError):
        _narrowed(
            shape,
            "destination",
            walks=RoleClassWalks(_walk_reader(nodes)),
            arcs_graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={}),
        )


def _posted(*members: str) -> dict[str, dict]:
    return {dcid: _DEBT_NODES[dcid]["arcs"] for dcid in members}


def test_a_posted_by_creditor_member_is_outside_the_family_of_a_recipient_only_identity():
    """The identity binds the debtor alone, so the mint resolves the recipient-only total
    and refuses the posted by-creditor row."""
    shape = _debt_shape()
    query = shape_query(shape, [], ["destination"])
    walks = RoleClassWalks(_walk_reader(_DEBT_NODES))

    assert posted_members_outside_role_class(_posted(_BY_CREDITOR), query, walks=walks).value == (
        _BY_CREDITOR,
    )
    assert posted_members_outside_role_class(_posted(_TOTAL), query, walks=walks).value == ()
    creditor_named = shape_query(shape, [], ["origin", "destination"])
    assert (
        posted_members_outside_role_class(_posted(_BY_CREDITOR), creditor_named, walks=walks).value
        == ()
    )


def test_a_posted_set_inside_the_walked_family_is_answered_as_posted():
    shape = _debt_shape()
    reader = _walk_reader(_DEBT_NODES)

    outside = posted_members_outside_role_class(
        _posted(_TOTAL),
        shape_query(shape, [], ["destination"]),
        walks=RoleClassWalks(reader),
    ).value

    assert outside == ()
    assert reader.calls["node_subjects"] == 1


def _resubmit(identity: BaseIdentity, posted: list[str], *, member_graph: bool = True):
    graph = FakeGraph(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES},
    )
    document = identity.model_dump(mode="json", exclude_none=True)
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
            base_spec_id=recompute_identity(document).spec_id,
            base_identity=identity,
            resolved_stat_var_dcids=posted,
        ),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=_one_role_coverage(
            {**_PLACES, **_DEBT_NODES}, {f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES}
        ),
        options=EngineOptions(
            materialise_graph=_walk_reader(_DEBT_NODES) if member_graph else None
        ),
    )


def test_a_resubmit_posting_a_member_the_mint_refuses_is_not_answered_definite():
    """The first pass answers the recipient-only family under `destination: KEN`. A
    resubmit of that identity with the by-creditor row posted names a member the mint
    refuses `MemberSelectionMismatch`, so Search reports no matching slice."""
    first = _run(entities=["Kenya"], obs={f"{_TOTAL}|country/KEN": _OBS})
    identity = first.specs[0].definition.identity

    refused = _resubmit(identity, [_BY_CREDITOR])
    kept = _resubmit(identity, [_TOTAL])

    assert (refused.status, refused.no_data_reason) == ("no_data", "no_matching_slice")
    assert kept.status == "definite"
    assert [sv.ref for sv in kept.specs[0].resolved.stat_vars.items] == [_TOTAL]


def test_a_resubmit_posting_two_role_sets_names_the_member_the_mint_refuses():
    """The identity binds the debtor alone. The mint's walk keeps the recipient-only
    total, so the mint refuses the posted by-creditor row `MemberSelectionMismatch` and
    Search reports no matching slice."""
    first = _run(entities=["Kenya"], obs={f"{_TOTAL}|country/KEN": _OBS})

    region = _resubmit(first.specs[0].definition.identity, [_TOTAL, _BY_CREDITOR])

    assert (region.status, region.no_data_reason) == ("no_data", "no_matching_slice")


def test_without_a_walk_the_posted_members_elect_their_role_class():
    query = shape_query(_debt_shape(), [], ["destination"])

    assert posted_members_outside_role_class(_posted(_TOTAL, _BY_CREDITOR), query).value == (
        _BY_CREDITOR,
    )
    with pytest.raises(RoleSignatureError):
        posted_members_outside_role_class(
            {dcid: _TIE_NODES[dcid]["arcs"] for dcid in ("Tie_ByCreditor", "Tie_Total_G")},
            shape_query(_tie_sample_shape(), [], ["destination"]),
        )


def test_a_place_whose_walk_meets_a_tie_reports_the_role_signature_refusal():
    query = "guaranteed debt of Kenya"
    sample = [dcid for dcid in _TIE_NODES if dcid != "Tie_Total_G"]
    graph = FakeGraph(
        nodes={**_PLACES, **_TIE_NODES},
        detect={query: {"svs": sample, "cosine_scores": [0.9, 0.88]}},
        resolve={"Kenya": "country/KEN"},
        obs={},
    )
    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            entity_roles={"Kenya": [DoorRole(axis="destination", span="Kenya")]},
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(materialise_graph=_walk_reader(_TIE_NODES)),
        )
    )

    assert (result.status, result.no_data_reason) == ("no_data", "role_signature_incompatible")


# ---------------------------------------------------------------------------
# A family the index does not export on a five-tuple it carries
# ---------------------------------------------------------------------------

_SERVICE = "DebtService_Total"
_SERVICE_BY_CREDITOR = "DebtService_ByCreditor"
_SERVICE_NODES = {
    _SERVICE: _node(
        "Debt service",
        _RECIPIENT,
        population_type="FinancialFlow",
        financialFlowType="ServicingPayment",
        sourceIndicatorCode="DT.TDS.DECT.CD",
    ),
    _SERVICE_BY_CREDITOR: _node(
        "Debt service, by creditor",
        _PAIR,
        population_type="FinancialFlow",
        financialFlowType="ServicingPayment",
        sourceIndicatorCode="DT.TDS.DECT.CD",
    ),
}
"""The twins hold one indicator code, as every IDS total and by-creditor pair on the dev
graph does. The graph read drops `sourceIndicatorCode` from their constraint list
(`dc_graph.IMPORT_ADDED_CONSTRAINT_PROPERTIES`), so they differ only in roles and the
role-twin collapse pairs them."""


def _service_shape() -> ShapeDraft:
    graph = FakeGraph(nodes=_SERVICE_NODES, obs={}, detect={}, resolve={})
    (shape,) = derive_shapes(confirmed_svs=list(_SERVICE_NODES), graph=graph)
    return shape


def test_a_shape_none_of_whose_members_the_index_holds_takes_no_slot_key_from_it():
    """Debt service shares the CRS five-tuple. Its slot keys stay its own members', so its
    identity's slot keys cover no index row and the mint walks the graph for it."""
    shape = _service_shape()

    assert index_carries_shape(shape) is False
    member_facts, index_sourced = five_tuple_member_facts(shape)
    assert index_sourced is False
    assert set(member_facts) == set(_SERVICE_NODES)
    assert {slot.property_dcid for slot in shape.slot_keys if slot.property_dcid} == {
        "financialFlowType"
    }


def _crs_row():
    """One packaged CRS row of the five-tuple debt service shares, as a recalled node."""
    row = next(
        row
        for row in get_index().statvars
        if row.population_type == "FinancialFlow"
        and row.measured_property == "amount"
        and "crsTypeOfFinance" in row.constraint_properties
    )
    node = _node(
        "Gross ODA disbursements",
        list(row.observation_properties),
        population_type="FinancialFlow",
        **dict(row.constraint_properties),
    )
    return row, node


@pytest.mark.parametrize("crs_dcid", [None, "Crs_Row_The_Index_Does_Not_Hold"])
def test_members_the_index_decides_and_members_it_does_not_are_separate_shapes(crs_dcid):
    """A recalled CRS StatVar and the debt-service members share a five-tuple and both are
    role-bearing. The first's keys cover an index row, so the index decides its family,
    whether or not it holds that very row, as the mint does for its identity. The graph walk
    decides the second's, so each is its own shape with its own label, slot keys and
    members."""
    row, crs_node = _crs_row()
    row = dataclasses.replace(row, dcid=crs_dcid or row.dcid)
    nodes = {**_SERVICE_NODES, row.dcid: crs_node}
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

    by_id = {s.shape_id: s for s in derive_shapes(confirmed_svs=list(nodes), graph=graph)}

    debt = by_id["financialflow_amount_measuredvalue_nominal_by_role"]
    crs = by_id["financialflow_amount_measuredvalue_nominal"]
    assert len(by_id) == 2
    assert list(debt.sv_arc_facts or {}) == list(_SERVICE_NODES)
    assert debt.label == "Debt service"
    assert index_carries_shape(debt) is False
    assert {slot.property_dcid for slot in debt.slot_keys if slot.property_dcid} == {
        "financialFlowType"
    }
    assert list(crs.sv_arc_facts or {}) == [row.dcid]
    assert index_carries_shape(crs) is True
    assert "crsTypeOfFinance" in {slot.property_dcid for slot in crs.slot_keys}


def test_a_recalled_index_row_does_not_take_the_debt_shapes_family():
    """ "Debt service of Kenya" with one CRS row in the recall answers with the debt total.
    The CRS row is another shape, which the model did not elect."""
    row, crs_node = _crs_row()
    nodes = {**_SERVICE_NODES, row.dcid: crs_node}
    query = "debt service of Kenya"
    graph = FakeGraph(
        nodes={**_PLACES, **nodes},
        detect={query: {"svs": list(nodes), "cosine_scores": [0.9, 0.88, 0.7]}},
        resolve={"Kenya": "country/KEN"},
        obs={f"{_SERVICE}|country/KEN": _OBS, f"{_SERVICE_BY_CREDITOR}|country/KEN": _OBS},
    )

    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(
                nodes,
                {f"{_SERVICE}|country/KEN": _OBS, f"{_SERVICE_BY_CREDITOR}|country/KEN": _OBS},
            ),
        )
    )

    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.shape.label == "Debt service"
    assert [sv.ref for sv in spec.resolved.stat_vars.items] == [_SERVICE]
    assert spec.definition.identity.entities == {"destination": ["country/KEN"]}
    assert {
        slot.property: getattr(slot.binding, "ref", None) for slot in spec.definition.identity.slots
    } == {"financialFlowType": "ServicingPayment"}


def test_an_unindexed_shape_on_a_five_tuple_the_index_carries_keeps_its_own_slot_keys():
    """By-creditor debt members declare the roles CRS rows declare. The index holds none of
    them, so the shape is the debt family and reads no index row."""
    nodes = {_SERVICE_BY_CREDITOR: _SERVICE_NODES[_SERVICE_BY_CREDITOR]}
    (shape,) = derive_shapes(
        confirmed_svs=list(nodes), graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    )

    assert shape.shape_id == "financialflow_amount_measuredvalue_nominal_by_role"
    assert index_carries_shape(shape) is False
    assert {slot.property_dcid for slot in shape.slot_keys if slot.property_dcid} == {
        "financialFlowType"
    }


def test_the_unexported_family_binds_the_value_every_member_shares():
    shape = _service_shape()

    bindings, added = bind_shared_values(shape, [])

    assert [(b.property_dcid, b.kind, b.value_dcids, b.evidence) for b in added] == [
        ("financialFlowType", "value", ["ServicingPayment"], "sole_option")
    ]
    assert bindings == added


def test_a_bound_slot_and_a_slot_asked_in_full_keep_their_bindings():
    shape = _service_shape()
    asked_in_full = SlotBindingDraft(
        axis="how",
        property_dcid="financialFlowType",
        kind="unbound",
        value_dcids=[],
        evidence="requested_all",
        requested_all_phrase="all flow types",
    )

    bindings, added = bind_shared_values(shape, [asked_in_full])

    assert added == []
    assert bindings == [asked_in_full]


def test_a_shape_on_a_five_tuple_the_index_does_not_carry_binds_nothing():
    assert bind_shared_values(_debt_shape(), []) == ([], [])


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

_QUERY = "external debt stocks of Kenya"
_PLACES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    "country/CHN": {"label": "China", "type": "Country"},
}
_OBS = [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 8}]


class _BindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
        ), None


def _run(*, entities: list[str], entity_roles=None, obs: dict, query: str = _QUERY):
    members = list(_DEBT_NODES)
    graph = FakeGraph(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={query: {"svs": members, "cosine_scores": [0.9, 0.88, 0.87, 0.86]}},
        resolve={"Kenya": "country/KEN", "China": "country/CHN"},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=entities,
            entity_roles=entity_roles,
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(_DEBT_NODES, obs),
        )
    )


def test_a_place_named_without_a_direction_answers_with_the_recipient_only_members():
    """The undirected place binds as the recipient, the recipient-only class is elected, and
    its series are confirmed through the graph's facets for that place. The total dominates
    its guaranteed part, so the answer lists the total alone, as the mint serves it."""
    result = _run(
        entities=["Kenya"],
        obs={
            f"{_TOTAL}|country/KEN": _OBS,
            f"{_TOTAL_GUARANTEED}|country/KEN": _OBS,
            f"{_BY_CREDITOR}|country/KEN": _OBS,
        },
    )

    assert result.status == "definite"
    spec = result.specs[0]
    assert [sv.ref for sv in spec.resolved.stat_vars.items] == [_TOTAL]
    assert spec.definition.identity.entities == {"destination": ["country/KEN"]}


def test_a_recipient_with_no_observation_of_the_plain_total_reports_no_observations():
    result = _run(entities=["Kenya"], obs={f"{_BY_CREDITOR}|country/KEN": _OBS})

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


def test_a_named_creditor_leaves_the_recipient_only_total_out():
    """The by-creditor class covers the creditor. Its two-role series are confirmed against
    the coverage artefact, which this fixture leaves empty, so the plain total's data at
    the recipient answers nothing."""
    result = _run(
        entities=["Kenya", "China"],
        entity_roles={
            "Kenya": [DoorRole(axis="destination", span="Kenya")],
            "China": [DoorRole(axis="origin", span="China")],
        },
        obs={f"{_TOTAL}|country/KEN": _OBS},
        query="external debt stocks Kenya owes China",
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"


_OWED_TO_CREDITOR = "Debt_OwedToCreditor"
_CREDITOR_NODES = {
    _OWED_TO_CREDITOR: _node(
        "External debt owed to a creditor",
        ["financingProvider"],
        population_type="FinancialClaim",
        positionType="AmountOutstanding",
    ),
}


def _run_one_role(
    query: str,
    *,
    nodes: dict,
    place: str,
    entity_roles=None,
    obs: dict,
    coverage: CoverageSnapshot | None = None,
):
    graph = FakeGraph(
        nodes={**_PLACES, **nodes},
        detect={query: {"svs": list(nodes), "cosine_scores": [0.9] * len(nodes)}},
        resolve={"Kenya": "country/KEN", "China": "country/CHN"},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=[place],
            entity_roles=entity_roles,
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=coverage or _one_role_coverage(nodes, obs),
        )
    )


@pytest.mark.parametrize("directed", [True, False])
def test_a_creditor_only_series_is_not_resolved(directed):
    """The series declares the creditor alone. The coverage index holds one-role series on
    the recipient axis only, and the mint reads every role-bearing StatVar's facets from
    that index, so no facet source serves it and Search reports the role unresolved."""
    result = _run_one_role(
        "external debt lent by China",
        nodes=_CREDITOR_NODES,
        place="China",
        entity_roles={"China": [DoorRole(axis="origin", span="China")]} if directed else None,
        obs={f"{_OWED_TO_CREDITOR}|country/CHN": _OBS},
    )

    assert (result.status, result.no_data_reason) == ("no_data", "role_not_resolved")


def test_a_one_role_series_the_coverage_index_names_is_confirmed_there():
    """The coverage index names the recipient-only total and holds no row at Kenya, so the
    mint reads its candidates there and finds none, whatever the graph holds at Kenya."""
    coverage = FakeCoverageService(
        build_coverage_index(
            [(_TOTAL, "f1", "country/CHN", "country/GHA", [2020])],
            role_names={
                _TOTAL: RoleNames(donor="financingProvider", recipient="financingRecipient")
            },
        )
    ).snapshot()

    result = _run_one_role(
        "external debt stocks of Kenya",
        nodes={_TOTAL: _DEBT_NODES[_TOTAL]},
        place="Kenya",
        obs={f"{_TOTAL}|country/KEN": _OBS},
        coverage=coverage,
    )

    assert (result.status, result.no_data_reason) == ("no_data", "no_observations")


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


def _debt_identity(entities: dict[str, list[str]]) -> BaseIdentity:
    return BaseIdentity.model_validate(
        {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                "population_type": "FinancialPosition",
                "measured_property": "amount",
                "stat_type": "measuredValue",
                "measurement_qualifier": "Nominal",
            },
            "slots": [{"property": "positionType", "binding": {"state": "unbound"}}],
            "entities": entities,
            "requested_window": {"kind": "all"},
        }
    )


def test_the_population_memo_keys_on_role_keys_and_not_on_the_entities_under_them():
    recipient = _population_key(_debt_identity({"destination": ["country/KEN"]}))

    assert recipient == _population_key(_debt_identity({"destination": ["country/GHA"]}))
    assert recipient != _population_key(
        _debt_identity({"destination": ["country/KEN"], "origin": ["country/CHN"]})
    )
    assert recipient != _population_key(_debt_identity({}))


def test_the_measure_family_cache_keys_on_role_keys():
    shape = _debt_shape()
    axes = get_measure_index().axes

    assert measure_family_cache_key(
        shape, axes, shape_query(shape, [], ["destination"])
    ) != measure_family_cache_key(shape, axes, shape_query(shape, [], ["origin", "destination"]))


# ---------------------------------------------------------------------------
# A candidate of another role class
# ---------------------------------------------------------------------------

_GNI_RECEIVED = "Gni_Received_Nominal"
_GNI_PPP = "Gni_Ppp"
_GNI_NODES = {
    _GNI_RECEIVED: _node(
        "Gross national income (current US$)", _RECIPIENT, population_type="EconomicActivity"
    ),
    _GNI_PPP: _node(
        "Gross national income at purchasing power parity",
        [],
        population_type="EconomicActivity",
        qualifier="PurchasingPowerParity",
    ),
}


def test_a_subject_only_candidate_beside_a_role_bearing_election_binds_its_place_as_subject():
    """The elected shape binds Kenya as the recipient. The subject-only candidate's identity
    binds Kenya under `observationAbout`, the role its member declares."""

    class _TwoReadingsLLM:
        async def generate_structured(self, *, prompt, system, schema):
            return _UniversalBindOutput(
                chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
            ), None

    query = "national income of Kenya"
    graph = FakeGraph(
        nodes={**_PLACES, **_GNI_NODES},
        detect={query: {"svs": [_GNI_RECEIVED, _GNI_PPP], "cosine_scores": [0.9, 0.88]}},
        resolve={"Kenya": "country/KEN"},
        obs={f"{_GNI_RECEIVED}|country/KEN": _OBS, f"{_GNI_PPP}|country/KEN": _OBS},
    )
    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_TwoReadingsLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(
                _GNI_NODES, {f"{_GNI_RECEIVED}|country/KEN": _OBS, f"{_GNI_PPP}|country/KEN": _OBS}
            ),
        )
    )

    assert result.status == "candidates"
    entities_by_member = {
        spec.resolved.stat_vars.items[0].ref: spec.definition.identity.entities
        for spec in result.specs
    }
    assert entities_by_member == {
        _GNI_RECEIVED: {"destination": ["country/KEN"]},
        _GNI_PPP: {"observationAbout": ["country/KEN"]},
    }


# ---------------------------------------------------------------------------
# An operation the answer cannot run
# ---------------------------------------------------------------------------

_RANK_CREDITORS = Operation(verb="rank", phrase="top creditors", over="origin")
_GROWTH = Operation(verb="growth", phrase="growth")


def _run_with_operations(*, obs: dict, operations: tuple[Operation, ...]):
    query = "top creditors of external debt stocks of Kenya"
    graph = FakeGraph(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={query: {"svs": list(_DEBT_NODES), "cosine_scores": [0.9, 0.88, 0.87, 0.86]}},
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            operations=operations,
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(_DEBT_NODES, obs),
        )
    )


def test_a_rank_over_creditors_is_left_off_a_recipient_only_answer_with_a_warning():
    """The identity binds the recipient alone, so the answer is the plain total the mint
    resolves. The rank has no creditor to run over, and the part says so."""
    result = _run_with_operations(
        obs={f"{_TOTAL}|country/KEN": _OBS}, operations=(_RANK_CREDITORS, _GROWTH)
    )

    assert result.status == "definite"
    assert {sv.ref for sv in result.specs[0].resolved.stat_vars.items} == {_TOTAL}
    assert result.answered_roles == ("financingRecipient",)
    stamped = with_runnable_operations(result, (_RANK_CREDITORS, _GROWTH))
    assert stamped.operations == (_GROWTH,)
    dropped = [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]
    assert [w.message for w in dropped] == [
        'The requested "top creditors" was not applied: the answer is the total over every '
        "provider, and a breakdown by provider needs a named provider."
    ]
    assert stamped.departure_facts == {
        warning_key(dropped[0]): DepartureFact(
            code="breakdown_not_applied",
            binding_ref="origin:",
            asked=_RANK_CREDITORS.phrase,
            over="origin",
        )
    }
    assert with_runnable_operations(stamped, (_RANK_CREDITORS, _GROWTH)).warnings == (
        stamped.warnings
    )


def test_an_operation_over_no_flow_axis_stays_on_a_recipient_only_answer():
    result = _run_with_operations(obs={f"{_TOTAL}|country/KEN": _OBS}, operations=(_GROWTH,))

    stamped = with_runnable_operations(result, (_GROWTH,))

    assert stamped.operations == (_GROWTH,)
    assert not [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]


def test_a_resubmitted_identity_drops_a_posted_operation_its_family_cannot_run():
    """The resubmit door stamps a posted operation through the same rule: the identity
    binds the debtor alone, and a posted rank over creditors leaves its part with a warning."""
    first = _run(entities=["Kenya"], obs={f"{_TOTAL}|country/KEN": _OBS})
    region = _resubmit(first.specs[0].definition.identity, [_TOTAL])

    stamped = with_runnable_operations(region, (_RANK_CREDITORS,))

    assert region.status == "definite"
    assert region.answered_roles == ("financingRecipient",)
    assert stamped.operations == ()
    assert [w.code for w in stamped.warnings if w.code == "query_qualifier_dropped"] == [
        "query_qualifier_dropped"
    ]


def test_a_value_the_recalled_members_share_on_their_own_property_stays_open():
    """Two recalled debt-service members both hold `guarantee=Public`, and the family's
    total, not recalled, holds none. Only the flow type, which CRS rows carry too, is
    bound, so the mint's walk still reaches the total."""
    nodes = {
        "Service_Guaranteed_A": _node(
            "Guaranteed debt service to bondholders",
            _RECIPIENT,
            population_type="FinancialFlow",
            financialFlowType="ServicingPayment",
            guarantee="Public",
            idsCreditorType="Bondholders",
        ),
        "Service_Guaranteed_B": _node(
            "Guaranteed debt service to banks",
            _RECIPIENT,
            population_type="FinancialFlow",
            financialFlowType="ServicingPayment",
            guarantee="Public",
            idsCreditorType="Banks",
        ),
    }
    (shape,) = derive_shapes(
        confirmed_svs=list(nodes), graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    )

    _bindings, added = bind_shared_values(shape, [])

    assert [(b.property_dcid, b.value_dcids) for b in added] == [
        ("financialFlowType", ["ServicingPayment"])
    ]


def test_a_role_bearing_candidate_beside_a_subject_only_election_answers_its_own_family():
    """The subject-only shape is elected and the recipient-only shape is an alternate. The
    alternate is probed with Kenya as its recipient, and its identity binds Kenya there."""

    class _SubjectFirstLLM:
        async def generate_structured(self, *, prompt, system, schema):
            return _UniversalBindOutput(
                chosen_shape_index=0, bindings=[], also_plausible=[1], ask=None
            ), None

    query = "national income of Kenya"
    graph = FakeGraph(
        nodes={**_PLACES, **_GNI_NODES},
        detect={query: {"svs": [_GNI_PPP, _GNI_RECEIVED], "cosine_scores": [0.9, 0.88]}},
        resolve={"Kenya": "country/KEN"},
        obs={f"{_GNI_RECEIVED}|country/KEN": _OBS, f"{_GNI_PPP}|country/KEN": _OBS},
    )
    result = asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_SubjectFirstLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(
                _GNI_NODES, {f"{_GNI_RECEIVED}|country/KEN": _OBS, f"{_GNI_PPP}|country/KEN": _OBS}
            ),
        )
    )

    assert result.status == "candidates"
    assert {
        spec.resolved.stat_vars.items[0].ref: spec.definition.identity.entities
        for spec in result.specs
    } == {
        _GNI_PPP: {"observationAbout": ["country/KEN"]},
        _GNI_RECEIVED: {"destination": ["country/KEN"]},
    }


# ---------------------------------------------------------------------------
# One family of a five-tuple reaches the election
# ---------------------------------------------------------------------------


def _rival_shapes(*, debt_score: float, crs_score: float) -> list[ShapeDraft]:
    row, crs_node = _crs_row()
    nodes = {**_SERVICE_NODES, row.dcid: crs_node}
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    scores = {_SERVICE: debt_score, _SERVICE_BY_CREDITOR: debt_score - 0.01, row.dcid: crs_score}
    return derive_shapes(confirmed_svs=list(nodes), graph=graph, sv_scores=scores)


def _named(code_system: str, code_dcid: str) -> QualifierMatch:
    return QualifierMatch(
        code_system=code_system,
        code_dcid=code_dcid,
        label=code_dcid,
        source="label",
        matched_text=code_dcid,
    )


def test_the_family_with_the_stronger_recall_score_reaches_the_election():
    debt_first = elect_five_tuple_family(_rival_shapes(debt_score=0.9, crs_score=0.7), [])
    crs_first = elect_five_tuple_family(_rival_shapes(debt_score=0.7, crs_score=0.9), [])

    assert [index_carries_shape(shape) for shape in debt_first] == [False]
    assert [index_carries_shape(shape) for shape in crs_first] == [True]


def test_the_family_holding_more_of_the_values_the_query_names_outranks_the_score():
    """The debt family scores higher, and the query names a netting basis only the CRS
    rows hold."""
    shapes = _rival_shapes(debt_score=0.9, crs_score=0.7)

    kept = elect_five_tuple_family(shapes, [_named("nettingBasis", "Net")])

    assert [index_carries_shape(shape) for shape in kept] == [True]


def test_families_of_equal_score_and_named_values_both_stay():
    assert len(elect_five_tuple_family(_rival_shapes(debt_score=0.8, crs_score=0.8), [])) == 2


def test_shapes_of_other_five_tuples_and_role_classes_are_not_rivals():
    shapes = [_debt_shape(), *_rival_shapes(debt_score=0.9, crs_score=0.7)]

    kept = elect_five_tuple_family(shapes, [])

    assert [shape.pop_type_dcid for shape in kept] == ["FinancialPosition", "FinancialFlow"]


def test_an_unindexed_shape_has_no_measure_family_and_passes_the_collapse():
    """A by-creditor debt member shares its five-tuple and its two roles with CRS flows.
    The measure family machinery reads index rows, so it leaves the debt shape alone."""
    row, crs_node = _crs_row()
    nodes = {_SERVICE_BY_CREDITOR: _SERVICE_NODES[_SERVICE_BY_CREDITOR], row.dcid: crs_node}
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    debt, crs = derive_shapes(confirmed_svs=list(nodes), graph=graph)
    index, axes = get_index(), get_measure_index().axes

    assert index_carries_shape(crs) and not index_carries_shape(debt)
    assert debt.declared_roles == crs.declared_roles
    assert measure_family_partitions(debt, index, axes, shape_query(debt, [])) == ()
    assert measure_family_partitions(crs, index, axes, shape_query(crs, [])) != ()
    assert [s.shape_id for s in collapse_measure_families([debt, crs], index, axes)] == [
        debt.shape_id,
        crs.shape_id,
    ]


# ---------------------------------------------------------------------------
# The unknown-score floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("debt_score", "kept_authorities"),
    [
        # Both under the floor: the index vouches for the CRS shape's members, and nothing
        # vouches for the debt shape's beyond its recall score.
        (0.5, [True]),
        # The debt shape clears the floor on its own score, and as the stronger of the two
        # families of the five-tuple it is the one that reaches the bind.
        (0.66, [False]),
    ],
)
def test_the_unknown_score_floor_exempts_only_a_role_bearing_shape_the_index_carries(
    debt_score, kept_authorities
):
    """ "Commitments to Kenya" recalls debt commitments at 0.56. A CRS shape under the floor
    still reaches the bind, since the index holds its family whatever the sample scored."""
    row, crs_node = _crs_row()
    nodes = {_SERVICE_BY_CREDITOR: _SERVICE_NODES[_SERVICE_BY_CREDITOR], row.dcid: crs_node}
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    shapes = derive_shapes(
        confirmed_svs=list(nodes),
        graph=graph,
        sv_scores={_SERVICE_BY_CREDITOR: debt_score, row.dcid: 0.5},
    )
    state = _prune_state(shapes)

    asyncio.run(_phase_prune_and_rank(state))

    assert sorted(index_carries_shape(shape) for shape in state.eligible_shapes) == sorted(
        kept_authorities
    )


# ---------------------------------------------------------------------------
# The roles of the Spec that answers
# ---------------------------------------------------------------------------

_RECEIVED = "Debt_Received_Total"
_LOANS_BY_CREDITOR = "Loans_ByCreditor"
_GRANTS_BY_CREDITOR = "Grants_ByCreditor"
_ANSWER_NODES = {
    _RECEIVED: _node("External debt received", _RECIPIENT, positionType="AmountOutstanding"),
    _LOANS_BY_CREDITOR: _node("Loans, by creditor", _PAIR, population_type="LoanFlow"),
    _GRANTS_BY_CREDITOR: _node("Grants, by creditor", _PAIR, population_type="GrantFlow"),
}


class _ElectFirstLLM:
    """Elects the recipient-only total and offers the by-creditor shapes beside it."""

    def __init__(self, also_plausible: list[int]) -> None:
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=self._also_plausible, ask=None
        ), None


def _run_beside_by_creditor(*, obs: dict, also_plausible: list[int]):
    query = "top creditors of external debt of Kenya"
    graph = FakeGraph(
        nodes={**_PLACES, **_ANSWER_NODES},
        detect={query: {"svs": list(_ANSWER_NODES), "cosine_scores": [0.9, 0.86, 0.85]}},
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    pair_roles = RoleNames(donor="financingProvider", recipient="financingRecipient")
    coverage = FakeCoverageService(
        build_coverage_index(
            [
                (_LOANS_BY_CREDITOR, "f1", "country/CHN", "country/KEN", [2020]),
                (_GRANTS_BY_CREDITOR, "f1", "country/CHN", "country/KEN", [2020]),
                *(
                    (_RECEIVED, "f_received", ABSENT_ENTITY, "country/KEN", [2020])
                    for key in obs
                    if key == f"{_RECEIVED}|country/KEN"
                ),
            ],
            role_names={
                _LOANS_BY_CREDITOR: pair_roles,
                _GRANTS_BY_CREDITOR: pair_roles,
                _RECEIVED: RoleNames(donor=None, recipient="financingRecipient"),
            },
        )
    ).snapshot()
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            operations=(_RANK_CREDITORS,),
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectFirstLLM(also_plausible),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=coverage,
        )
    )


def test_a_candidate_set_answers_under_the_roles_of_its_candidates():
    """The recipient-only total has no data at Kenya and two by-creditor shapes do. The
    candidates hold a row per creditor, so a rank over creditors runs on them."""
    result = _run_beside_by_creditor(obs={}, also_plausible=[1, 2])

    assert result.status == "candidates"
    assert {spec.resolved.stat_vars.items[0].ref for spec in result.specs} == {
        _LOANS_BY_CREDITOR,
        _GRANTS_BY_CREDITOR,
    }
    assert result.answered_roles == tuple(_PAIR)
    stamped = with_runnable_operations(result, (_RANK_CREDITORS,))
    assert stamped.operations == (_RANK_CREDITORS,)
    assert not [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]


def test_a_definite_answer_from_an_alternate_answers_under_the_alternates_roles(monkeypatch):
    """A bound unit can leave the elected Spec out of the candidate set
    (`candidate_collapse._apply_bound_unit_filter`), and one by-creditor alternate then
    answers alone. The answer holds a row per creditor, so a rank over creditors runs, and
    the leg is directional: Kenya is the recipient of a two-role series."""

    def _drop_the_elected(state, unique_specs, grounded):
        return [spec for spec in unique_specs if spec.spec_id != grounded.elected_spec_id]

    monkeypatch.setattr(candidate_collapse, "_apply_bound_unit_filter", _drop_the_elected)

    result = _run_beside_by_creditor(obs={f"{_RECEIVED}|country/KEN": _OBS}, also_plausible=[1])

    assert result.status == "definite"
    assert [sv.ref for sv in result.specs[0].resolved.stat_vars.items] == [_LOANS_BY_CREDITOR]
    assert result.answered_roles == tuple(_PAIR)
    stamped = with_runnable_operations(result, (_RANK_CREDITORS,))
    assert stamped.operations == (_RANK_CREDITORS,)
    assert not [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]
    assert _success_directional(result, role_query="top creditors of external debt of Kenya")
