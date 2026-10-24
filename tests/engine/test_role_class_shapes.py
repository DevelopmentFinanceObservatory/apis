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
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from code_index import CodeIndex, IndexStamp, StatVarRow
from code_index import shared as code_index_shared
from code_index.members import DatasetScope
from code_index.shared import get_index, get_measure_index
from code_index.types import CodeRow, CodeSystemDeclaration
from datasets_api.adapter import resolve_definition_members
from entity_resolution import Admission
from origin_catalogue import BASE
from search.engine import core as search_core
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.binders import Binder, GeminiBinder
from search.engine.departures import warning_key
from search.engine.entry.parsed import _open_axes_for_part
from search.engine.errors import EngineInputError, FamilyNotResolvableError, RoleSignatureError
from search.engine.fanout import _FanOutResult, _run_variables_streaming, _success_directional
from search.engine.options.population_memo import _population_key
from search.engine.qualifier_match import QualifierMatch, read_by_vocabulary
from search.engine.quantity_read import QuantityOutput
from search.engine.region import (
    candidate_collapse,
    phase_election,
    phase_place_materialise,
    promote_materialise,
    resolve_spec_resubmit,
    resolve_variable,
)
from search.engine.region.candidates import _Alternate, _materialise_alt_shapes
from search.engine.region.phase_place_materialise import (
    _declined_open_roles,
    _members_holding_bound_values,
)
from search.engine.region.phase_rank_bind import _phase_prune_and_rank
from search.engine.region.promote_materialise import _phase_materialise
from search.engine.region.promote_state import _PromoteState
from search.engine.region.result import with_runnable_operations
from search.engine.region.state import EngineOptions, shape_vocabulary
from search.engine.region.unrealisable import unbind_unrealisable
from search.engine.retrieval.bound_values import (
    MembershipAuthority,
    MembershipResolution,
    RoleClassWalks,
    RoleKeys,
    bind_shared_values,
    collapse_measure_families,
    elect_five_tuple_family,
    elected_role_class_shape,
    entity_role_keys,
    five_tuple_member_facts,
    held_codes,
    index_carries_shape,
    measure_family_cache_key,
    measure_family_partitions,
    query_phrases_read,
    shape_five_tuple,
    shape_member_facts,
    shape_population_arcs,
    shape_query,
)
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.retrieve import Materialised
from search.engine.role_text import DoorRole
from search.engine.shape import (
    ShapeDraft,
    required_role_names,
)
from search.engine.stated_words_read import StatedWordsOutput, restricting
from search.heading.departure_fact import DepartureFact
from search.models import (
    BaseIdentity,
    CoverageBare,
    DatasetResubmitInput,
    IdentityBindingValue,
    IdentitySlot,
    Operation,
    ParsedEntity,
    ParsedPart,
    ResolveRequest,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
    RoleAxis,
)
from series_coverage.schema import ABSENT_ENTITY, RoleNames
from series_coverage.shared import CoverageSnapshot
from statvar_members import MemberEnumerator, member_digest
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._binders import RatingBinder, elected_reply
from tests.engine._builders import bound_place_context, flow_place_roles
from tests.engine._fake_coverage import (
    FIXTURE_PROVENANCE,
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.engine.test_bind_slot_reserve import _state as _prune_state
from tests.engine.test_two_ended_menu import _index_row_node
from tests.fixtures import FakeGraph, make_resolver
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue
from tests.origin_catalogue.support import build_arc_tables

_WDI = "dc/base/WorldDevelopmentIndicators"
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
    """The series coverage index of the one-role members among `nodes`, with a row at each
    place `obs` holds them at. The index holds a one-role series on its own role's axis,
    with `series_coverage.schema.ABSENT_ENTITY` on the other, so Search reads its facets
    there."""
    lone_role = {
        dcid: roles[0]
        for dcid, node in nodes.items()
        if len(
            roles := [
                n["dcid"]
                for n in node.get("arcs", {}).get("observationProperties", {}).get("nodes", [])
            ]
        )
        == 1
        and roles[0] in ("financingRecipient", "financingProvider")
    }
    rows = [
        (dcid, f"f_{dcid}", place, ABSENT_ENTITY, list(range(2015, 2023)))
        if lone_role[dcid] == "financingProvider"
        else (dcid, f"f_{dcid}", ABSENT_ENTITY, place, list(range(2015, 2023)))
        for key in obs
        for dcid, place in [key.split("|", 1)]
        if dcid in lone_role
    ]
    role_names = {
        dcid: RoleNames(donor=role, recipient=None)
        if role == "financingProvider"
        else RoleNames(donor=None, recipient=role)
        for dcid, role in lone_role.items()
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
            if schema is StatedWordsOutput:
                return NO_RATING
            if schema is QuantityOutput:
                return NO_QUANTITY
            offered.append(
                [
                    label
                    for label in ("Gross domestic product", "Gross national income")
                    if label in prompt
                ]
            )
            return elected_reply(0, also_plausible=None), None

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


def test_a_held_family_elects_its_role_class_among_every_member_it_holds(bind_catalogue):
    """Recall finds the by-creditor members alone, of one role set. The catalogue holds the
    whole family, so the shape declares the family's roles, the creditor optional, and a
    bound recipient elects the recipient-only class among its members, as the mint does, and the shape answers through its least-constrained member, the total,
    under the family's label."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={dcid: ([_WDI], []) for dcid in _DEBT_NODES},
            arcs=build_arc_tables(
                dfo={},
                base={dcid: node["arcs"] for dcid, node in _DEBT_NODES.items()},
                schema={},
            ),
        )
    )
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    narrowed = _narrowed(shape, "destination")

    assert set(shape.declared_roles or ()) == {"financingProvider", "financingRecipient"}
    assert required_role_names(shape) == ("financingRecipient",)
    assert narrowed.declared_roles == ("financingRecipient",)
    assert list(narrowed.sv_arc_facts or {}) == [_TOTAL]
    assert list(shape_population_arcs(narrowed)) == [_TOTAL, _TOTAL_GUARANTEED]
    assert narrowed.label == shape.label
    assert narrowed.held_family == shape.held_family is not None


def test_a_declined_reading_states_the_open_roles_of_held_members_outside_recall(
    bind_catalogue,
):
    """Recall finds the unguaranteed members alone; the catalogue holds the guaranteed ones
    too. Bound to a public guarantee, the declined reading's members are all held past
    recall, and the by-creditor one among them leaves the origin open."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row(_WDI, BASE)],
            base_variables={dcid: ([_WDI], []) for dcid in _DEBT_NODES},
            arcs=build_arc_tables(
                dfo={},
                base={dcid: node["arcs"] for dcid, node in _DEBT_NODES.items()},
                schema={},
            ),
        )
    )
    shape = _debt_shape(_TOTAL, _BY_CREDITOR)
    holding = _members_holding_bound_values(shape, {"guarantee": ["Public"]})
    state = SimpleNamespace(
        shape_draft=shape,
        frame=SimpleNamespace(open_roles=SimpleNamespace(value=("origin",))),
    )

    assert set(holding) == {_TOTAL_GUARANTEED, _BY_CREDITOR_GUARANTEED}
    assert not set(holding) & set(shape.sv_arc_facts or {})
    assert _declined_open_roles(cast(Any, state), holding) == ("origin",)


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


@pytest.mark.parametrize(
    ("entities", "open_axes", "expected"),
    [
        ({"financingRecipient": ["country/KEN"]}, (), RoleKeys(bound=frozenset({"destination"}))),
        (
            {"financingRecipient": ["country/KEN"], "financingProvider": ["country/CHN"]},
            (),
            RoleKeys(bound=frozenset({"destination", "origin"})),
        ),
        (
            {"observationAbout": ["country/KEN"]},
            (),
            RoleKeys(bound=frozenset({"observationAbout"})),
        ),
        ({"financingRecipient": []}, (), RoleKeys()),
        (
            {"financingRecipient": ["country/KEN"]},
            ("origin",),
            RoleKeys(bound=frozenset({"destination"}), open=frozenset({"origin"})),
        ),
        (
            {"financingRecipient": ["country/KEN"]},
            ("destination",),
            RoleKeys(bound=frozenset({"destination"})),
        ),
    ],
    ids=["recipient", "both", "subject", "empty", "open-creditor", "open-but-bound"],
)
def test_entity_role_keys_are_the_keys_the_identity_carries(entities, open_axes, expected):
    assert entity_role_keys(entities, open_axes=open_axes) == expected


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


_BY_CREDITOR_NODES = {
    "Tie_ByCreditor": _node("Debt, by creditor", _PAIR, positionType="AmountOutstanding"),
    "Tie_ByCreditor_G": _node(
        "Guaranteed debt, by creditor", _PAIR, positionType="AmountOutstanding", guarantee="Public"
    ),
}
# A recipient-only total holding another value of the least-constrained by-creditor
# member's one constraint key: its role set lies inside the by-creditor members', so a
# question naming no creditor reads it.
_NESTED_NODES = {
    **_BY_CREDITOR_NODES,
    "Tie_Total_Arrears": _node("Debt arrears total", _RECIPIENT, positionType="Arrears"),
}
# A recipient-only total with other constraint keys: its role set lies inside the
# by-creditor members', and it is the total of none of them.
_APART_NODES = {
    **_BY_CREDITOR_NODES,
    "Tie_Total_G": _node("Guaranteed debt total", _RECIPIENT, guarantee="Public"),
}
# A member placing the destination under a role the by-creditor members do not declare:
# neither role set holds the other.
_TIE_NODES = {
    **_BY_CREDITOR_NODES,
    "Tie_Owed_G": _node("Guaranteed debt owed", ["debtor"], guarantee="Public"),
}
_TIES = [
    pytest.param(_TIE_NODES, id="role sets that do not nest"),
    pytest.param(_APART_NODES, id="nested role sets on other constraint keys"),
]


def _by_creditor_sample_shape() -> ShapeDraft:
    (shape,) = derive_shapes(
        confirmed_svs=list(_BY_CREDITOR_NODES),
        graph=FakeGraph(nodes=_BY_CREDITOR_NODES, obs={}, detect={}, resolve={}),
    )
    return shape


def test_a_total_of_another_constraint_value_only_the_walk_sees_answers():
    """The sample holds by-creditor members. The graph also holds a recipient-only total
    at another value of their constraint key. With the recipient alone bound, the mint
    reads the total, the series published without a creditor, and the shape takes its
    member and name."""
    settled = _narrowed(
        _by_creditor_sample_shape(),
        "destination",
        walks=RoleClassWalks(_walk_reader(_NESTED_NODES)),
        arcs_graph=FakeGraph(nodes=_NESTED_NODES, obs={}, detect={}, resolve={}),
    )

    assert list(settled.sv_arc_facts or {}) == ["Tie_Total_Arrears"]
    assert settled.declared_roles == ("financingRecipient",)
    assert settled.label == "Debt arrears total"


@pytest.mark.parametrize("nodes", _TIES)
def test_a_tie_only_the_walk_sees_is_not_answered_from_the_sample(nodes: dict):
    """The sample holds by-creditor members. The graph also holds a member that places
    the recipient under a role of its own, where neither role set holds the other, or a
    recipient-only total with other constraint keys. The mint refuses the identity, so
    the sample does not answer it."""
    with pytest.raises(RoleSignatureError):
        _narrowed(
            _by_creditor_sample_shape(),
            "destination",
            walks=RoleClassWalks(_walk_reader(nodes)),
            arcs_graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={}),
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


def _resubmit(
    identity: BaseIdentity,
    *,
    member_graph: bool = True,
    coverage: CoverageSnapshot | None = None,
    nodes: dict | None = None,
):
    nodes = nodes or _DEBT_NODES
    graph = FakeGraph(
        nodes={**_PLACES, **nodes},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in nodes},
    )
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=coverage
        or _one_role_coverage(
            {**_PLACES, **nodes}, {f"{dcid}|country/KEN": _OBS for dcid in nodes}
        ),
        options=EngineOptions(materialise_graph=_walk_reader(nodes) if member_graph else None),
    )


def test_a_resubmit_of_a_recipient_only_identity_answers_the_mints_family():
    """The first pass answers the recipient-only family under `destination: KEN`. A
    resubmit of that identity elects the same family, the recipient-only total, as the
    mint's walk does, whatever by-creditor rows the graph holds."""
    first = _run(entities=["Kenya"], obs={f"{_TOTAL}|country/KEN": _OBS})

    region = _resubmit(first.specs[0].definition.identity)

    assert region.status == "definite"
    assert [sv.ref.ref for sv in region.specs[0].resolved.members] == [_TOTAL]


def _guaranteed_debt_of_kenya(
    nodes: dict, *, obs: dict | None = None, binder: Binder | None = None
):
    """Resolve "guaranteed debt of Kenya" with the by-creditor members recalled and the
    mint's walk reading `nodes`."""
    query = "guaranteed debt of Kenya"
    obs = obs or {}
    graph = FakeGraph(
        nodes={**_PLACES, **nodes},
        detect={query: {"svs": list(_BY_CREDITOR_NODES), "cosine_scores": [0.9, 0.88]}},
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    return asyncio.run(
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
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(nodes, obs),
            options=EngineOptions(materialise_graph=_walk_reader(nodes)),
            binder=binder,
        )
    )


@pytest.mark.parametrize("nodes", _TIES)
def test_a_place_whose_walk_meets_a_tie_reports_the_role_signature_refusal(nodes: dict):
    result = _guaranteed_debt_of_kenya(nodes)

    assert (result.status, result.no_data_reason) == ("no_data", "role_signature_incompatible")


def test_a_place_whose_walk_finds_a_total_of_another_constraint_value_elects_the_total():
    """The walk settles the shape on the arrears total, the recipient-only total of another
    constraint value. That total does not state "guaranteed", as the binder rates it, so
    the part refuses on it, flagging the total under `destination: KEN`."""
    result = _guaranteed_debt_of_kenya(
        _NESTED_NODES,
        obs={"Tie_Total_Arrears|country/KEN": _OBS},
        binder=RatingBinder(GeminiBinder(), {"guaranteed": 0.0}),
    )

    assert (result.status, result.no_data_reason) == ("no_data", "qualifier_not_bound")
    assert [(q.text, q.disposition, q.evidence) for q in result.unbound_qualifiers] == [
        ("guaranteed", "no_vocabulary", "uncovered")
    ]
    (flagged,) = result.flagged_match or ()
    assert [sv.ref.ref for sv in flagged.resolved.members] == ["Tie_Total_Arrears"]
    assert flagged.definition.identity.entities == {"destination": ["country/KEN"]}


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
    """One fixture CRS row of the five-tuple debt service shares, as a recalled node."""
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
    assert [sv.ref.ref for sv in spec.resolved.members] == [_SERVICE]
    assert spec.definition.identity.entities == {"destination": ["country/KEN"]}
    assert {
        slot.property: getattr(slot.binding, "ref", None) for slot in spec.definition.identity.slots
    } == {"financialFlowType": "ServicingPayment"}


_IDS_SERVICE = "Amount_FinancialFlow_ServicingPayment_Nominal"
_IDS_INTEREST = "Amount_FinancialFlow_InterestPayment_Nominal"


def _ids_flow(label: str, flow_type: str) -> dict:
    return _node(label, _RECIPIENT, population_type="FinancialFlow", financialFlowType=flow_type)


@pytest.mark.parametrize(
    "draft",
    [
        SlotBindingDraft(
            axis="how",
            property_dcid="financialFlowType",
            kind="value",
            value_dcids=["ServicingPayment"],
            evidence="sole_option",
        ),
        SlotBindingDraft(axis="how", property_dcid="financialFlowType", kind="unbound"),
    ],
    ids=["claimed-by-the-reply", "claimed-at-settle"],
)
def test_debt_service_keeps_the_flow_type_its_family_is_told_apart_by(draft):
    """ "Total debt service of Kenya", as dq-ids-001 asks of Ghana, recalls the IDS debt service total
    alone, and the reply binds its one flow type as the slot's sole option, or leaves the
    slot unbound and settle claims it. The recalled sample is not the whole family, so the
    claim rests on it. The family shares its five-tuple with the CRS rows the index
    exports, and the flow type is what tells it apart from them (`bind_shared_values`), so
    the walk reads the flow type bound. Read open, the walk lists the IDS interest payments
    total beside it, refutes the claim and answers every flow to Kenya."""
    recalled = {_IDS_SERVICE: _ids_flow("Debt service on external debt", "ServicingPayment")}
    interest = {_IDS_INTEREST: _ids_flow("Interest payments on external debt", "InterestPayment")}
    walked = {**recalled, **interest}
    obs = {f"{dcid}|country/KEN": _OBS for dcid in walked}

    class _DraftLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema is StatedWordsOutput:
                return NO_RATING
            if schema is QuantityOutput:
                return NO_QUANTITY
            return elected_reply(0, bindings=[draft], also_plausible=None), None

    result = _run(
        entities=["Kenya"],
        obs=obs,
        query="total debt service of Kenya",
        nodes=recalled,
        value_nodes=interest,
        llm=_DraftLLM(),
        options=EngineOptions(materialise_graph=_walk_reader(walked)),
        coverage=_one_role_coverage(walked, obs),
    )

    assert result.status == "definite"
    (spec,) = result.specs
    assert [sv.ref.ref for sv in spec.resolved.members] == [_IDS_SERVICE]
    assert {
        slot.property: getattr(slot.binding, "ref", None) for slot in spec.definition.identity.slots
    } == {"financialFlowType": "ServicingPayment"}
    assert "membership_unverified" not in {w.code for w in result.warnings}


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
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, also_plausible=None), None


class _AskNearestLLM:
    """Declines the empty menu and names the first not-electable reading as nearest."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, ask="no reading divides by people", nearest_refused_index=0
        ), None


def _run(
    *,
    entities: list[str],
    entity_roles=None,
    obs: dict,
    query: str = _QUERY,
    open_axes: frozenset[RoleAxis] = frozenset(),
    coverage: CoverageSnapshot | None = None,
    llm=None,
    nodes: dict = _DEBT_NODES,
    options: EngineOptions | None = None,
    value_nodes: dict | None = None,
    binder: Binder | None = None,
):
    members = list(nodes)
    graph = FakeGraph(
        nodes={**_PLACES, **(value_nodes or {}), **nodes},
        detect={query: {"svs": members, "cosine_scores": [0.9, 0.88, 0.87, 0.86]}},
        resolve={"Kenya": "country/KEN", "China": "country/CHN"},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=entities,
            entity_roles=entity_roles,
            open_axes=open_axes,
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm or _BindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=coverage or _one_role_coverage(nodes, obs),
            options=options,
            binder=binder,
        )
    )


def _twin_coverage() -> CoverageSnapshot:
    """The coverage index of the recipient-only total at Kenya and its by-creditor twin at
    Kenya from China."""
    return FakeCoverageService(
        build_coverage_index(
            [
                (_TOTAL, f"f_{_TOTAL}", ABSENT_ENTITY, "country/KEN", list(range(2015, 2023))),
                (_BY_CREDITOR, "f_by_creditor", "country/CHN", "country/KEN", [2020, 2021]),
            ],
            role_names={
                _TOTAL: RoleNames(donor=None, recipient="financingRecipient"),
                _BY_CREDITOR: RoleNames(donor="financingProvider", recipient="financingRecipient"),
            },
        )
    ).snapshot()


def test_an_open_creditor_answers_with_the_by_creditor_members():
    """Kenya with every creditor asked for. The place binds the recipient, the creditor
    axis stays open, and the by-creditor class covers both, so it displaces the total and
    the answer reads its series at each creditor."""
    result = _run(
        entities=["Kenya"],
        obs={f"{_TOTAL}|country/KEN": _OBS},
        open_axes=frozenset({"origin"}),
        coverage=_twin_coverage(),
    )

    assert result.status == "definite"
    spec = result.specs[0]
    assert [sv.ref.ref for sv in spec.resolved.members] == [_BY_CREDITOR]
    assert spec.definition.identity.entities == {"destination": ["country/KEN"]}
    assert spec.definition.identity.open_roles == ["origin"]


def _run_leg(
    *,
    nodes: dict,
    coverage: CoverageSnapshot,
    operations: tuple[Operation, ...] = (),
    open_axes: frozenset[RoleAxis] = frozenset(),
    query: str = "top creditors of external debt stocks of Kenya",
    recalled: list[str] | None = None,
):
    """Resolve one fan-out leg over Kenya, as a door hands it its operations and its own
    open roles. Recall finds `recalled`, every member of `nodes` by default. The default
    `query` is the raw-text door's for a rank over creditors (`_RANK_CREDITORS`); a leg
    without that rank passes the text its door sends."""
    recalled = list(nodes) if recalled is None else recalled
    graph = FakeGraph(
        nodes={**_PLACES, **nodes},
        detect={query: {"svs": recalled, "cosine_scores": [0.9] * len(recalled)}},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in nodes},
    )
    out = _FanOutResult()

    async def drain() -> None:
        async for _event in _run_variables_streaming(
            [query],
            entities=["Kenya"],
            date_request=None,
            query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            admission=Admission(graph=graph),
            llm=_BindLLM(),
            binder=GeminiBinder(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            base_timing={},
            pre_resolved=None,
            deadline=time.monotonic() + 3600,
            degrade_on_error=False,
            use_semaphore=False,
            out=out,
            per_variable_operations=[operations],
            per_variable_open_axes=[open_axes],
            coverage=coverage,
        ):
            pass

    asyncio.run(drain())
    (region,) = out.regions
    return region


_RECIPIENT_ONLY_NODES = {dcid: _DEBT_NODES[dcid] for dcid in (_TOTAL, _TOTAL_GUARANTEED)}

# The variable a parsed part asking a role open sends (`ParsedPart.open_roles`): its
# variable text, with the open role and no operation.
_PARSED_VARIABLE = "external debt stocks"


def test_a_rank_over_creditors_on_a_twin_family_keeps_the_operation_and_the_open_creditor():
    """The rank runs over creditors, so the leg leaves the creditor open. The by-creditor
    twin covers it and displaces the total, the answer holds a row per creditor, and the
    identity states the open creditor beside the bound debtor."""
    region = _run_leg(nodes=_DEBT_NODES, coverage=_twin_coverage(), operations=(_RANK_CREDITORS,))

    assert region.status == "definite"
    spec = region.specs[0]
    assert [sv.ref.ref for sv in spec.resolved.members] == [_BY_CREDITOR]
    assert spec.definition.identity.entities == {"destination": ["country/KEN"]}
    assert spec.definition.identity.open_roles == ["origin"]
    assert region.operations == (_RANK_CREDITORS,)
    assert not [w for w in region.warnings if w.code == "query_qualifier_dropped"]


def test_a_parsed_part_asking_every_creditor_answers_as_a_rank_over_creditors_does():
    part = ParsedPart(
        variable_text="external debt stocks",
        entities=[ParsedEntity(text="Kenya")],
        open_roles=["creditor"],
    )
    open_axes = _open_axes_for_part(part, pointer_prefix="/input/parts/0")

    opened = _run_leg(
        nodes=_DEBT_NODES,
        coverage=_twin_coverage(),
        open_axes=open_axes,
        query=part.variable_text,
    )
    ranked = _run_leg(nodes=_DEBT_NODES, coverage=_twin_coverage(), operations=(_RANK_CREDITORS,))

    assert open_axes == {"origin"}
    assert opened.specs[0].spec_id == ranked.specs[0].spec_id
    assert opened.specs[0].definition.identity.open_roles == ["origin"]
    assert opened.parsed_form is not None
    assert opened.parsed_form.open_roles == ["origin"]


def _debt_index(*dcids: str) -> CodeIndex:
    """The code index holding the members `dcids` of `_DEBT_NODES`, so it decides the debt
    family's members."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system="positionType",
                hierarchical=False,
                disjoint_from_children=False,
                summable=True,
                absent_reading="not_declared",
                plain_name="position",
            ),
        ),
        codes=(
            CodeRow(
                dcid="AmountOutstanding",
                code_system="positionType",
                code="AmountOutstanding",
                label="amount outstanding",
                synonyms=(),
                parent=None,
            ),
        ),
        statvars=[
            StatVarRow(
                dcid=dcid,
                population_type="FinancialPosition",
                measured_property="amount",
                stat_type="measuredValue",
                measurement_qualifier="Nominal",
                measurement_denominator=None,
                observation_properties=tuple(
                    n["dcid"] for n in _DEBT_NODES[dcid]["arcs"]["observationProperties"]["nodes"]
                ),
                constraint_properties={"positionType": "AmountOutstanding"},
                provenances=(FIXTURE_PROVENANCE,),
            )
            for dcid in dcids
        ],
    )


def _use_index(monkeypatch: pytest.MonkeyPatch, index: CodeIndex) -> None:
    """Serve `index` wherever a loaded module reads the fixture index."""
    packaged = code_index_shared.get_index
    for module in list(sys.modules.values()):
        if vars(module).get("get_index") is packaged:
            monkeypatch.setattr(module, "get_index", lambda: index)


def test_an_open_creditor_elects_the_index_familys_twin_that_recall_missed(monkeypatch):
    """Recall finds only the recipient-only total, so the recalled shape declares no creditor
    role, and the code index decides the family, which holds the by-creditor twin. The open
    creditor reaches the election, the twin answers, and the part states the creditor it
    serves. The identity leaves it open, the rank over creditors runs, and nothing warns
    that the breakdown was dropped."""
    coverage = _twin_coverage()
    _use_index(monkeypatch, _debt_index(_TOTAL, _BY_CREDITOR))

    region = _run_leg(
        nodes=_DEBT_NODES,
        coverage=coverage,
        operations=(_RANK_CREDITORS,),
        open_axes=frozenset({"origin"}),
        recalled=[_TOTAL],
    )

    assert region.status == "definite"
    spec = region.specs[0]
    assert [sv.ref.ref for sv in spec.resolved.members] == [_BY_CREDITOR]
    assert spec.definition.identity.open_roles == ["origin"]
    assert region.operations == (_RANK_CREDITORS,)
    assert not [w for w in region.warnings if w.code == "query_qualifier_dropped"]


def test_an_open_creditor_on_an_index_family_with_no_twin_answers_the_total_and_warns(
    monkeypatch,
):
    """No member of the index family declares a creditor, so the total answers, its identity
    leaves no role open, and the part warns that the breakdown was not applied."""
    coverage = _twin_coverage()
    _use_index(monkeypatch, _debt_index(_TOTAL))

    region = _run_leg(
        nodes=_DEBT_NODES,
        coverage=coverage,
        open_axes=frozenset({"origin"}),
        query=_PARSED_VARIABLE,
        recalled=[_TOTAL],
    )

    assert region.status == "definite"
    spec = region.specs[0]
    assert [sv.ref.ref for sv in spec.resolved.members] == [_TOTAL]
    assert spec.definition.identity.open_roles is None
    assert [w.code for w in region.warnings if w.code == "query_qualifier_dropped"] == [
        "query_qualifier_dropped"
    ]


def test_a_resubmitted_open_creditor_identity_keeps_its_open_role_and_spec_id():
    first = _run_leg(nodes=_DEBT_NODES, coverage=_twin_coverage(), operations=(_RANK_CREDITORS,))
    spec = first.specs[0]

    region = _resubmit(spec.definition.identity, coverage=_twin_coverage())

    assert region.status == "definite"
    assert region.spec.definition.identity.open_roles == ["origin"]
    assert region.spec.spec_id == spec.spec_id
    assert [sv.ref.ref for sv in region.spec.resolved.members] == [_BY_CREDITOR]


def test_an_open_creditor_identity_states_the_members_the_mint_resolves_for_it():
    """A resubmit of the open-creditor identity states the served count and family digest
    the Datasets mint resolves from the same identity: the by-creditor members."""
    first = _run_leg(nodes=_DEBT_NODES, coverage=_twin_coverage(), operations=(_RANK_CREDITORS,))
    identity = first.specs[0].definition.identity
    assert identity.open_roles == ["origin"]

    region = _resubmit(identity, coverage=_twin_coverage())
    (minted,) = resolve_definition_members(
        identity.model_dump(mode="json", exclude_none=True),
        spec_id=first.specs[0].spec_id,
        graph=_walk_reader(_DEBT_NODES),
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    ).shapes

    assert minted.stat_var == f"dcid:{_BY_CREDITOR}"
    assert minted.family is not None
    assert minted.family.digest == member_digest([_BY_CREDITOR, _BY_CREDITOR_GUARANTEED])
    assert region.spec.resolved.stat_vars.served_count == 1
    assert region.spec.resolved.shape.member_count == minted.family.count == 2
    assert region.spec.resolved.shape.member_digest == minted.family.digest


def _resubmit_entry(
    identity: BaseIdentity,
    *,
    coverage: CoverageSnapshot,
    nodes: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """Resolve `identity` through the `dataset_resubmit` door as its one entry."""
    graph = FakeGraph(
        nodes={**_PLACES, **nodes},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in nodes},
    )
    monkeypatch.setattr(
        search_core, "get_coverage_service", lambda: SimpleNamespace(snapshot=lambda: coverage)
    )
    entry = ResubmitIdentityEntry(variable_text="external debt stocks", identity=identity)
    return offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=_walk_reader(nodes),
    )


def _recipient_only_identity() -> tuple[BaseIdentity, str, CoverageSnapshot]:
    """The recipient-only total's identity at Kenya, its spec_id, and its coverage."""
    coverage = _one_role_coverage(_RECIPIENT_ONLY_NODES, {f"{_TOTAL}|country/KEN": _OBS})
    first = _run_leg(nodes=_RECIPIENT_ONLY_NODES, coverage=coverage, query=_QUERY)
    return first.specs[0].definition.identity, first.specs[0].spec_id, coverage


def test_a_resubmit_naming_an_open_creditor_no_member_serves_refuses(monkeypatch):
    """The posted identity asks for every creditor, and no member of the family declares
    one, so the resubmit cannot give a series per creditor and refuses. The identity is the
    whole definition: an answer for the total would carry the spec_id of another identity."""
    identity, _, coverage = _recipient_only_identity()
    opened = identity.model_copy(update={"open_roles": ["origin"]})

    with pytest.raises(EngineInputError) as exc_info:
        _resubmit_entry(
            opened, coverage=coverage, nodes=_RECIPIENT_ONLY_NODES, monkeypatch=monkeypatch
        )

    assert (exc_info.value.code, exc_info.value.status, exc_info.value.pointer) == (
        "open_role_unavailable",
        422,
        "/input/entries/0/identity/open_roles",
    )


def test_a_resubmit_opening_a_creditor_the_twin_serves_answers_the_by_creditor_members(
    monkeypatch,
):
    """The recipient-only identity, edited to ask for every creditor, over a family that
    holds the by-creditor twin. The twin serves the open role, so the part answers with it
    under the identity posted and nothing warns that the breakdown was dropped."""
    first = _run_leg(nodes=_DEBT_NODES, coverage=_twin_coverage(), query=_QUERY)
    identity = first.specs[0].definition.identity
    assert identity.open_roles is None
    opened = identity.model_copy(update={"open_roles": ["origin"]})

    result = _resubmit_entry(
        opened, coverage=_twin_coverage(), nodes=_DEBT_NODES, monkeypatch=monkeypatch
    )

    (part,) = result.parts
    assert part.status == "definite"
    assert part.spec.definition.identity.open_roles == ["origin"]
    assert part.spec.spec_id != first.specs[0].spec_id
    assert part.spec.resolved.shape.member_digest == member_digest(
        [_BY_CREDITOR, _BY_CREDITOR_GUARANTEED]
    )
    assert part.spec.resolved.stat_vars.served_count == 1
    assert not [w for w in part.warnings if w.code == "query_qualifier_dropped"]


def _resubmit_candidates(
    identity: BaseIdentity,
    *,
    coverage: CoverageSnapshot,
    nodes: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """Resolve `identity` posted twice as a candidates entry of the `dataset_resubmit` door."""
    graph = FakeGraph(
        nodes={**_PLACES, **nodes},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in nodes},
    )
    monkeypatch.setattr(
        search_core, "get_coverage_service", lambda: SimpleNamespace(snapshot=lambda: coverage)
    )
    candidate = ResubmitCandidate(identity=identity)
    entry = ResubmitCandidatesEntry(
        variable_text="external debt stocks",
        ordering="broadest_first",
        truncated=False,
        candidates=[candidate, candidate],
    )
    return offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=_walk_reader(nodes),
    )


def test_a_candidate_whose_open_creditor_no_member_serves_refuses(monkeypatch):
    """A candidate's open role is a promise, like an identity entry's. No member of the
    family declares a creditor, so the candidate set refuses."""
    identity, _, coverage = _recipient_only_identity()
    opened = identity.model_copy(update={"open_roles": ["origin"]})

    with pytest.raises(EngineInputError) as exc_info:
        _resubmit_candidates(
            opened, coverage=coverage, nodes=_RECIPIENT_ONLY_NODES, monkeypatch=monkeypatch
        )

    # Both candidates refuse, and whichever promote raises first names the pointer.
    assert (exc_info.value.code, exc_info.value.status) == ("open_role_unavailable", 422)
    assert exc_info.value.pointer in {
        f"/input/entries/0/candidates/{j}/identity/open_roles" for j in (0, 1)
    }


def test_a_candidate_whose_open_creditor_its_twin_serves_keeps_its_spec_id(monkeypatch):
    first = _run_leg(nodes=_DEBT_NODES, coverage=_twin_coverage(), operations=(_RANK_CREDITORS,))
    spec = first.specs[0]

    result = _resubmit_candidates(
        spec.definition.identity,
        coverage=_twin_coverage(),
        nodes=_DEBT_NODES,
        monkeypatch=monkeypatch,
    )

    # The two posted candidates rebuild to one spec_id, so the part is that reading's
    # definite answer, and its warning says both resolved.
    (part,) = result.parts
    assert part.status == "definite"
    assert part.spec.spec_id == spec.spec_id
    assert part.spec.definition.identity.open_roles == ["origin"]
    (warning,) = (w for w in part.warnings if w.code == "candidates_collapsed_definite")
    assert warning.message.startswith(
        "2 candidates were posted; 2 still resolve and rebuild to 1 reading"
    )


@pytest.mark.parametrize(
    ("open_axes", "open_roles"),
    [
        pytest.param(frozenset({"origin"}), ["origin"], id="open-creditor"),
        pytest.param(frozenset(), None, id="no-open-axis"),
    ],
)
def test_a_parsed_unsupported_definition_keeps_the_open_axes_the_part_asks_for(
    open_axes, open_roles
):
    """No election confirms an open axis on an unsupported part, so its definition keeps
    the axes the part asks for, as the resubmit door's keeps the posted ones."""
    from search.engine.region.phase_shared import _unsupported_definition

    shape = _debt_shape(_TOTAL)
    roles = dataclasses.replace(
        flow_place_roles(shape, destination=["country/KEN"]), open_axes=open_axes
    )
    state = SimpleNamespace(
        shape_draft=shape, bindings=[], place=bound_place_context(roles), date_request=None
    )

    definition = _unsupported_definition(cast(Any, state))

    assert cast(dict, definition["identity"]).get("open_roles") == open_roles


def _open_creditor_promote_state(
    monkeypatch: pytest.MonkeyPatch, *, posted_open: list[RoleAxis], result: Materialised
) -> _PromoteState:
    """A resubmit state at materialisation over the debt family at Kenya, whose identity
    names `posted_open` as its open roles. The materialise returns `result`."""
    monkeypatch.setattr(promote_materialise, "run_materialise", lambda *_, **__: result)
    shape = _debt_shape()
    roles = dataclasses.replace(
        flow_place_roles(shape, destination=["country/KEN"]), open_axes=frozenset(posted_open)
    )
    state = _PromoteState(
        inp=cast(
            Any, SimpleNamespace(identity=SimpleNamespace(facets=None, open_roles=posted_open))
        ),
        graph=cast(Any, FakeGraph(raise_on_call=True)),
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
    state.shape_draft = shape
    state.place = bound_place_context(roles)
    state.bindings = []
    return state


def _elected(sv: str, *, has_data: bool, open_axes: frozenset[RoleAxis]) -> Materialised:
    return Materialised(
        sv_dcids=[sv],
        member_facts={sv: ArcsMemberFacts(arcs=_debt_shape().sv_arc_facts[sv])},
        has_data=has_data,
        coverage=CoverageBare(has_data=has_data),
        open_axes=open_axes,
    )


def test_an_answer_whose_members_serve_the_open_creditor_keeps_it_and_settles_the_shape(
    monkeypatch,
):
    """The by-creditor member answers and serves the creditor the identity leaves open. The
    place keeps the axis open, the shape declares the member's two roles, and the resubmit
    answers without refusing."""
    state = _open_creditor_promote_state(
        monkeypatch,
        posted_open=["origin"],
        result=_elected(_BY_CREDITOR, has_data=True, open_axes=frozenset({"origin"})),
    )

    assert _phase_materialise(state) is None

    assert state.mat_result.open_axes == frozenset({"origin"})
    assert state.place.roles.open_axes == frozenset({"origin"})
    assert frozenset(state.shape_draft.declared_roles) == frozenset(_PAIR)


def test_an_answer_whose_members_serve_no_open_creditor_refuses_the_identitys_open_role(
    monkeypatch,
):
    """The guaranteed total, which declares no creditor, answers an identity that asks for
    every creditor. Its members give no series per creditor, so the phase refuses."""
    state = _open_creditor_promote_state(
        monkeypatch,
        posted_open=["origin"],
        result=_elected(_TOTAL_GUARANTEED, has_data=True, open_axes=frozenset()),
    )

    with pytest.raises(EngineInputError) as exc_info:
        _phase_materialise(state)

    assert (exc_info.value.code, exc_info.value.status, exc_info.value.pointer) == (
        "open_role_unavailable",
        422,
        "/input/identity/open_roles",
    )


def test_a_recipient_only_answer_settles_the_shape_on_the_recipient_alone(monkeypatch):
    """The identity leaves no role open and the guaranteed total answers. The place leaves
    no axis open and the shape declares the recipient alone, as the answer's members do."""
    state = _open_creditor_promote_state(
        monkeypatch,
        posted_open=[],
        result=_elected(_TOTAL_GUARANTEED, has_data=True, open_axes=frozenset()),
    )

    assert _phase_materialise(state) is None

    assert state.mat_result.open_axes == frozenset()
    assert state.place.roles.open_axes == frozenset()
    assert state.shape_draft.declared_roles == ("financingRecipient",)


def test_an_open_role_a_named_place_binds_drops_and_warns():
    """Kenya, named without a direction, binds the recipient, so the open recipient the
    part asks for drops. The bound role wins and the part says so."""
    region = _run_leg(
        nodes=_DEBT_NODES,
        coverage=_twin_coverage(),
        open_axes=frozenset({"destination"}),
        query=_PARSED_VARIABLE,
    )

    assert region.status == "definite"
    spec = region.specs[0]
    assert spec.definition.identity.entities == {"destination": ["country/KEN"]}
    assert spec.definition.identity.open_roles is None
    assert [w.message for w in region.warnings if w.code == "query_qualifier_dropped"] == [
        "The requested breakdown over every recipient was not applied: the part names a "
        "recipient, and the answer is for that recipient."
    ]


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
    assert [sv.ref.ref for sv in spec.resolved.members] == [_TOTAL]
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
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=coverage or _one_role_coverage(nodes, obs),
        )
    )


@pytest.mark.parametrize("directed", [True, False])
def test_a_creditor_only_series_is_confirmed_at_its_creditor(directed):
    """The series declares the creditor alone. The coverage index holds a one-role series on
    its own role's axis, so the creditor named, with or without a stated direction, binds
    the origin and Search confirms the series there."""
    result = _run_one_role(
        "external debt lent by China",
        nodes=_CREDITOR_NODES,
        place="China",
        entity_roles={"China": [DoorRole(axis="origin", span="China")]} if directed else None,
        obs={f"{_OWED_TO_CREDITOR}|country/CHN": _OBS},
    )

    assert (result.status, result.no_data_reason) == ("definite", None)
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_OWED_TO_CREDITOR]


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


def _debt_identity(
    entities: dict[str, list[str]], open_roles: list[RoleAxis] | None = None
) -> BaseIdentity:
    return BaseIdentity.model_validate(
        {
            **({"open_roles": open_roles} if open_roles else {}),
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
    assert recipient != _population_key(
        _debt_identity({"destination": ["country/KEN"]}, open_roles=["origin"])
    )


def test_the_measure_family_cache_keys_on_role_keys_and_scope():
    shape = _debt_shape()
    axes = get_measure_index().axes
    query = shape_query(shape, [], ["destination"])

    assert measure_family_cache_key(shape, axes, query) != measure_family_cache_key(
        shape, axes, shape_query(shape, [], ["origin", "destination"])
    )
    assert measure_family_cache_key(shape, axes, query) != measure_family_cache_key(
        shape,
        axes,
        dataclasses.replace(
            query, scope=DatasetScope(allowed=frozenset({"provenance/Allowed"}), named=False)
        ),
    )
    assert measure_family_cache_key(shape, axes, query) != measure_family_cache_key(
        shape, axes, shape_query(shape, [], ["destination"], open_role_keys=["origin"])
    )


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


def test_a_candidate_of_another_role_class_binds_its_place_in_its_own_role():
    """The elected PPP reading binds Kenya under `observationAbout`, the role its member
    declares. The question names no price basis and the elected reading is off the default
    one, so the current-dollar reading is offered beside it (status rule 2(a)), and its
    identity binds Kenya as the recipient, the role its member declares."""

    class _TwoReadingsLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema is StatedWordsOutput:
                return NO_RATING
            if schema is QuantityOutput:
                return NO_QUANTITY
            return elected_reply(1, also_plausible=[0]), None

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
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(
                _GNI_NODES, {f"{_GNI_RECEIVED}|country/KEN": _OBS, f"{_GNI_PPP}|country/KEN": _OBS}
            ),
        )
    )

    assert result.status == "candidates"
    entities_by_member = {
        spec.resolved.members[0].ref.ref: spec.definition.identity.entities for spec in result.specs
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
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(_DEBT_NODES, obs),
        )
    )


def test_a_rank_over_creditors_refuses_a_recipient_only_family_before_bind():
    """A rank over creditors asks the creditor axis, and no member of the family declares
    the creditor, so no reading of the part is admissible: the part refuses
    `role_incompatible` without a bind call and flags nothing
    (`admissibility.judge_measure_and_roles`)."""
    stamped = _run_leg(
        nodes=_RECIPIENT_ONLY_NODES,
        coverage=_one_role_coverage(_RECIPIENT_ONLY_NODES, {f"{_TOTAL}|country/KEN": _OBS}),
        operations=(_RANK_CREDITORS, _GROWTH),
    )

    assert stamped.status == "no_data"
    assert stamped.no_data_reason == "role_incompatible"
    assert not stamped.flagged_match
    assert "bind" not in stamped.timing_by_step


_OPEN_CREDITOR_DROPPED = (
    "The requested breakdown over every provider was not applied: the answer is the total "
    "over every provider, and a breakdown by provider needs a named provider."
)


def test_an_open_creditor_on_a_recipient_only_answer_warns_once():
    """No member declares the creditor, so the open creditor the part asks for elects as it
    would unasked and the answer is the total over every creditor. An axis asked member by
    member is a preference among readings, never a refusal, and the part warns once for
    it."""
    stamped = _run_leg(
        nodes=_RECIPIENT_ONLY_NODES,
        coverage=_one_role_coverage(_RECIPIENT_ONLY_NODES, {f"{_TOTAL}|country/KEN": _OBS}),
        open_axes=frozenset({"origin"}),
        query=_PARSED_VARIABLE,
    )

    assert stamped.status == "definite"
    assert stamped.specs[0].definition.identity.open_roles is None
    dropped = [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]
    assert [w.message for w in dropped] == [_OPEN_CREDITOR_DROPPED]
    assert stamped.departure_facts == {
        warning_key(dropped[0]): DepartureFact(
            code="breakdown_not_applied", binding_ref="origin:", asked=None, over="origin"
        )
    }


def test_an_operation_over_no_flow_axis_stays_on_a_recipient_only_answer():
    result = _run_with_operations(obs={f"{_TOTAL}|country/KEN": _OBS}, operations=(_GROWTH,))

    stamped = with_runnable_operations(result, (_GROWTH,))

    assert stamped.operations == (_GROWTH,)
    assert not [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]


def test_a_resubmitted_identity_drops_a_posted_operation_its_family_cannot_run():
    """The resubmit door stamps a posted operation through the same rule: the identity
    binds the debtor alone, and a posted rank over creditors leaves its part with a warning."""
    first = _run(entities=["Kenya"], obs={f"{_TOTAL}|country/KEN": _OBS})
    region = _resubmit(first.specs[0].definition.identity)

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
            if schema is StatedWordsOutput:
                return NO_RATING
            if schema is QuantityOutput:
                return NO_QUANTITY
            return elected_reply(0, also_plausible=[1]), None

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
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=_one_role_coverage(
                _GNI_NODES, {f"{_GNI_RECEIVED}|country/KEN": _OBS, f"{_GNI_PPP}|country/KEN": _OBS}
            ),
        )
    )

    assert result.status == "candidates"
    assert {
        spec.resolved.members[0].ref.ref: spec.definition.identity.entities for spec in result.specs
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


def _reads(*named: QualifierMatch) -> Callable[[ShapeDraft], frozenset[str]]:
    """The phrases each shape's family reads among `named`."""
    return lambda shape: query_phrases_read(held_codes(shape), named, ())


def test_the_family_with_the_stronger_recall_score_reaches_the_election():
    debt_first = elect_five_tuple_family(_rival_shapes(debt_score=0.9, crs_score=0.7), _reads())
    crs_first = elect_five_tuple_family(_rival_shapes(debt_score=0.7, crs_score=0.9), _reads())

    assert [index_carries_shape(shape) for shape in debt_first] == [False]
    assert [index_carries_shape(shape) for shape in crs_first] == [True]


def test_the_family_holding_more_of_the_values_the_query_names_outranks_the_score():
    """The debt family scores higher, and the query names a netting basis only the CRS
    rows hold."""
    shapes = _rival_shapes(debt_score=0.9, crs_score=0.7)

    kept = elect_five_tuple_family(shapes, _reads(_named("nettingBasis", "Net")))

    assert [index_carries_shape(shape) for shape in kept] == [True]


def test_families_of_equal_score_and_named_values_both_stay():
    assert len(elect_five_tuple_family(_rival_shapes(debt_score=0.8, crs_score=0.8), _reads())) == 2


def test_shapes_of_other_five_tuples_and_role_classes_are_not_rivals():
    shapes = [_debt_shape(), *_rival_shapes(debt_score=0.9, crs_score=0.7)]

    kept = elect_five_tuple_family(shapes, _reads())

    assert [shape.pop_type_dcid for shape in kept] == ["FinancialPosition", "FinancialFlow"]


_DAC2A_IMPUTED = "AttributedAmount_FinancialFlow_Gross_Disbursement_DAC2aAidType106_Nominal"
_SECTORS_PRIMARY = (
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode11220_Nominal"
)
_SECTORS_TERTIARY = (
    "AttributedAmount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_DAC2aAidType106_"
    "CRSPurposeCode11420_Nominal"
)
_DAC2A_ODA = "Amount_FinancialFlow_Gross_Disbursement_DAC2aAidType206_Nominal"
_CRS_EMERGENCY_FOOD = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode72040_Nominal"
)
_CRS_PRIMARY = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode11220_Nominal"
)


def _index_shapes(*dcids: str) -> list[ShapeDraft]:
    """The shapes of the fixture index rows `dcids`, recalled together."""
    graph = FakeGraph(
        nodes={dcid: _index_row_node(dcid) for dcid in dcids}, obs={}, detect={}, resolve={}
    )
    return derive_shapes(confirmed_svs=list(dcids), graph=graph)


def _read_in_own_vocabulary(query: str) -> Callable[[ShapeDraft], frozenset[str]]:
    """The phrases of `query` each shape's family reads, in the shape's own vocabulary."""

    def phrases_read(shape: ShapeDraft) -> frozenset[str]:
        vocabulary = shape_vocabulary(shape)
        (reading,) = read_by_vocabulary(
            [query], get_index(), get_measure_index(), vocabularies=[vocabulary]
        )
        own = reading.by_vocabulary[vocabulary]
        return query_phrases_read(held_codes(shape), own.matches, own.ties)

    return phrases_read


@pytest.mark.parametrize(
    ("query", "recalled", "kept"),
    [
        # Both tables are indexed families of the attributed-amount five-tuple, and only the
        # sectors table holds a code for "basic education".
        pytest.param(
            "imputed multilateral ODA to basic education",
            (_DAC2A_IMPUTED, _SECTORS_PRIMARY),
            {_SECTORS_PRIMARY},
            id="two-indexed-families",
        ),
        # CRS reads "ODA" in its flow type and DAC2a in its aid type, so the tables compare
        # by the words their codes read, and CRS alone reads the sector.
        pytest.param(
            "ODA for basic education",
            (_DAC2A_ODA, _CRS_PRIMARY),
            {_CRS_PRIMARY},
            id="words-read-across-code-systems",
        ),
        # "Tertiary education" names CRS purposes 114 and 11420 alike, a tie the sectors
        # table reads through either code.
        pytest.param(
            "imputed multilateral ODA for tertiary education",
            (_DAC2A_IMPUTED, _SECTORS_TERTIARY),
            {_SECTORS_TERTIARY},
            id="tie-as-read",
        ),
        # DAC2a's ODA code reads "ODA" as CRS's flow type does, so neither table leaves.
        pytest.param("ODA to Kenya", (_DAC2A_ODA, _CRS_PRIMARY), None, id="synonym-parity"),
        # Neither table reads a word of the query, so neither reads fewer than the other.
        pytest.param("aid to Kenya", (_DAC2A_ODA, _CRS_PRIMARY), None, id="vacuous-words"),
        # DAC2a's developmental food aid reads "food aid" through its concept group, and CRS
        # reads the longer "emergency food aid" (purpose 72040), so DAC2a reads fewer.
        pytest.param(
            "emergency food aid to Sudan",
            (_DAC2A_ODA, _CRS_EMERGENCY_FOOD),
            {_CRS_EMERGENCY_FOOD},
            id="longer-phrase-across-code-systems",
        ),
    ],
)
def test_the_family_reading_more_of_the_query_reaches_the_election(
    query: str, recalled: tuple[str, ...], kept: set[str] | None
):
    """Rival families of one five-tuple and role class compare by the query phrases their
    codes read, whether or not the index carries them. `kept` None keeps both."""
    shapes = _index_shapes(*recalled)
    assert len(shapes) == 2 and all(index_carries_shape(shape) for shape in shapes)

    elected = elect_five_tuple_family(shapes, _read_in_own_vocabulary(query))

    assert {sv for shape in elected for sv in shape.sv_arc_facts or ()} == (kept or set(recalled))


def test_a_family_reading_fewer_leaves_before_recall_scores_compare():
    """Three families of one five-tuple and role class: CRS reads {"oda", "basic
    education"} at 0.3, the unindexed debt series {"debt"} at 0.5 and DAC2a {"oda"} at 0.8.
    DAC2a reads fewer than CRS and leaves first. The debt series and CRS read
    incomparably, so the recall score keeps the debt series. Scored against all three at
    once, each would drop another and none would stay."""
    dac2a, crs = _index_shapes(_DAC2A_ODA, _CRS_PRIMARY)
    (debt,) = [
        shape
        for shape in _rival_shapes(debt_score=0.5, crs_score=0.3)
        if not index_carries_shape(shape)
    ]
    crs = dataclasses.replace(crs, representative_score=0.3)
    dac2a = dataclasses.replace(dac2a, representative_score=0.8)
    reads = {
        id(crs): frozenset({"oda", "basic education"}),
        id(debt): frozenset({"debt"}),
        id(dac2a): frozenset({"oda"}),
    }
    assert len({shape_five_tuple(shape) for shape in (crs, debt, dac2a)}) == 1

    kept = elect_five_tuple_family([crs, debt, dac2a], lambda shape: reads[id(shape)])

    assert kept == [debt]


_DAC1_BILATERAL_ODA = "Amount_FinancialFlow_Gross_Disbursement_DAC1AidType1015_Nominal"


def _election_family(name: str) -> ShapeDraft:
    """A shape of the family `name` among the rivals of one five-tuple and role class: the
    DAC1, DAC2a and CRS tables, and an unindexed debt series. Each has the same recall
    score, so only the phrases it reads and the words it states separate them."""
    if name == "debt":
        (shape,) = [
            shape
            for shape in _rival_shapes(debt_score=0.5, crs_score=0.5)
            if not index_carries_shape(shape)
        ]
    else:
        dcid = {"dac1": _DAC1_BILATERAL_ODA, "dac2a": _DAC2A_ODA, "crs": _CRS_PRIMARY}[name]
        (shape,) = _index_shapes(dcid)
    return dataclasses.replace(shape, representative_score=0.5)


def _words(*words: str) -> frozenset[str]:
    return frozenset(words)


@pytest.mark.parametrize(
    ("families", "reads_stated_words", "kept"),
    [
        # DAC1 codes "bilateral ODA" as one aid type. The CRS reads "ODA" and states
        # "bilateral", which its rival already reads in a phrase, so the CRS reads fewer.
        pytest.param(
            {
                "dac1": (_words("bilateral oda"), _words()),
                "crs": (_words("oda"), _words("bilateral")),
            },
            True,
            {"dac1"},
            id="a-coding-family-beats-a-stating-one",
        ),
        # DAC2a reads "ODA" and states nothing here, and the CRS reads "ODA" and states
        # "bilateral", which DAC2a reads nowhere.
        pytest.param(
            {"dac2a": (_words("oda"), _words()), "crs": (_words("oda"), _words("bilateral"))},
            True,
            {"crs"},
            id="a-stating-family-beats-one-that-neither-states-nor-codes-the-word",
        ),
        pytest.param(
            {
                "dac2a": (_words("oda"), _words("bilateral")),
                "crs": (_words("oda"), _words("bilateral")),
            },
            True,
            {"dac2a", "crs"},
            id="two-stating-families-read-alike",
        ),
        # A debt series codes "bilateral" as its creditor type and the CRS reads nothing.
        pytest.param(
            {"debt": (_words("bilateral"), _words()), "crs": (_words(), _words("bilateral"))},
            True,
            {"debt"},
            id="a-family-coding-the-word-beats-a-stating-one-reading-nothing",
        ),
        # Each reads a phrase the other lacks: the debt series "bilateral", DAC2a "ODA".
        pytest.param(
            {
                "debt": (_words("bilateral"), _words()),
                "dac2a": (_words("oda"), _words("bilateral")),
            },
            True,
            {"debt", "dac2a"},
            id="a-stating-family-adds-nothing-beside-one-that-codes-the-word",
        ),
        # The index records no phrase for a debt series, whose creditor type is a graph
        # value, so nothing shows whether it reads the word.
        pytest.param(
            {"debt": (_words(), _words()), "crs": (_words(), _words("bilateral"))},
            True,
            {"debt", "crs"},
            id="a-stated-word-adds-nothing-beside-a-family-the-index-does-not-carry",
        ),
        pytest.param(
            {"dac2a": (_words("oda"), _words()), "crs": (_words("oda"), _words("bilateral"))},
            False,
            {"dac2a", "crs"},
            id="stated-words-unread-without-a-reader",
        ),
    ],
)
def test_a_family_stating_a_query_word_reads_it_in_the_election(
    families: dict[str, tuple[frozenset[str], frozenset[str]]],
    reads_stated_words: bool,
    kept: set[str],
):
    """A family reads beside a rival the phrases its codes read and, where the index
    carries the rival, each word its tables state that no phrase of the rival reads. A
    coding family beats a stating one, a stating family beats one that neither states nor
    codes the word, and two stating families read alike."""
    shapes = {name: _election_family(name) for name in families}
    assert len({shape_five_tuple(shape) for shape in shapes.values()}) == 1
    reads = {id(shape): families[name] for name, shape in shapes.items()}

    elected = elect_five_tuple_family(
        list(shapes.values()),
        lambda shape: reads[id(shape)][0],
        stated_words=(lambda shape: reads[id(shape)][1]) if reads_stated_words else None,
    )

    assert {name for name, shape in shapes.items() if any(shape is e for e in elected)} == kept


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
    assert measure_family_partitions(debt, index, axes, shape_query(debt, [], ())) == ()
    assert measure_family_partitions(crs, index, axes, shape_query(crs, [], ())) != ()
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
_DISBURSED_BY_CREDITOR = "Disbursed_ByCreditor"
_REPAID_BY_CREDITOR = "Repaid_ByCreditor"
# Two by-creditor families of external debt on different five-tuples, each named for the
# debt it measures.
_ANSWER_NODES = {
    _RECEIVED: _node("External debt received", _RECIPIENT, positionType="AmountOutstanding"),
    _DISBURSED_BY_CREDITOR: _node(
        "External debt disbursements, by creditor", _PAIR, population_type="DisbursementFlow"
    ),
    _REPAID_BY_CREDITOR: _node(
        "External debt repayments, by creditor", _PAIR, population_type="RepaymentFlow"
    ),
}


class _ElectFirstLLM:
    """Elects the recipient-only total and offers the by-creditor shapes beside it."""

    def __init__(self, also_plausible: list[int]) -> None:
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        if schema is StatedWordsOutput:
            return NO_RATING
        if schema is QuantityOutput:
            return NO_QUANTITY
        return elected_reply(0, also_plausible=self._also_plausible), None


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
                (_DISBURSED_BY_CREDITOR, "f1", "country/CHN", "country/KEN", [2020]),
                (_REPAID_BY_CREDITOR, "f1", "country/CHN", "country/KEN", [2020]),
                *(
                    (_RECEIVED, "f_received", ABSENT_ENTITY, "country/KEN", [2020])
                    for key in obs
                    if key == f"{_RECEIVED}|country/KEN"
                ),
            ],
            role_names={
                _DISBURSED_BY_CREDITOR: pair_roles,
                _REPAID_BY_CREDITOR: pair_roles,
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
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=coverage,
        )
    )


def test_a_candidate_set_answers_under_the_roles_of_its_candidates():
    """A rank over creditors needs the origin axis, so the recipient-only total, which
    declares none, is no candidate (`lacks_query_axis`), and the two by-creditor shapes
    are the menu. The candidates hold a row per creditor, so a rank over creditors runs on
    them."""
    result = _run_beside_by_creditor(obs={}, also_plausible=[1])

    assert result.status == "candidates"
    assert {spec.resolved.members[0].ref.ref for spec in result.specs} == {
        _DISBURSED_BY_CREDITOR,
        _REPAID_BY_CREDITOR,
    }
    assert result.answered_roles == tuple(_PAIR)
    stamped = with_runnable_operations(result, (_RANK_CREDITORS,))
    assert stamped.operations == (_RANK_CREDITORS,)
    assert not [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]


def test_a_definite_answer_from_an_alternate_answers_under_the_alternates_roles(monkeypatch):
    """A bound unit can leave the elected Spec out of the candidate set
    (`candidate_collapse._apply_bound_unit_filter`), and the other by-creditor shape then
    answers alone as the alternate. The answer holds a row per creditor, so a rank over
    creditors runs, and the leg is directional: Kenya is the recipient of a two-role
    series."""

    def _drop_the_elected(state, unique_specs, grounded):
        return [spec for spec in unique_specs if spec.spec_id != grounded.elected_spec_id]

    monkeypatch.setattr(candidate_collapse, "_apply_bound_unit_filter", _drop_the_elected)

    result = _run_beside_by_creditor(obs={f"{_RECEIVED}|country/KEN": _OBS}, also_plausible=[1])

    assert result.status == "definite"
    assert [sv.ref.ref for sv in result.specs[0].resolved.members] == [_REPAID_BY_CREDITOR]
    assert result.answered_roles == tuple(_PAIR)
    stamped = with_runnable_operations(result, (_RANK_CREDITORS,))
    assert stamped.operations == (_RANK_CREDITORS,)
    assert not [w for w in stamped.warnings if w.code == "query_qualifier_dropped"]
    assert _success_directional(result, role_query="top creditors of external debt of Kenya")


@pytest.mark.parametrize(
    ("entities", "coverage", "bound"),
    [
        (["Kenya"], _twin_coverage, {"destination": ["country/KEN"]}),
        (["Kenya"], lambda: _one_role_coverage(_DEBT_NODES, {}), {"destination": ["country/KEN"]}),
        ([], lambda: _one_role_coverage(_DEBT_NODES, {}), {}),
    ],
    ids=["data-at-kenya", "no-data-at-kenya", "no-data-anywhere"],
)
def test_a_per_capita_refusal_keeps_the_open_creditor_on_its_flagged_match(
    entities, coverage, bound
):
    """External debt per capita by every creditor. No debt series divides by people, so
    every reading fails the form the question names and none is admissible. Bind declines
    the empty menu and names the nearest not-electable reading, which the part refuses
    with as its flagged match. The flagged definition keeps the creditor axis the frame
    confirms open, as the definite answer would, whether or not Kenya holds data."""
    result = _run(
        entities=entities,
        obs={},
        query="external debt stocks per capita of Kenya",
        open_axes=frozenset({"origin"}),
        coverage=coverage(),
        llm=_AskNearestLLM(),
    )

    assert (result.status, result.no_data_reason) == ("no_data", "denominator_not_available")
    assert result.flagged_match is not None
    (flagged,) = result.flagged_match
    assert flagged.definition.identity.entities == bound
    assert flagged.definition.identity.open_roles == ["origin"]


# ---------------------------------------------------------------------------
# Drafted values no member holds together
# ---------------------------------------------------------------------------

_RELAX_BONDS = "Debt_Bonds"
_RELAX_GUARANTEED = "Debt_Guaranteed"
_RELAX_LONG_TERM = "Debt_LongTerm"
_RELAX_TOTAL = "Debt_Stocks"
_RELAX_NODES = {
    _RELAX_BONDS: _node(
        "External debt stocks, bonds",
        _RECIPIENT,
        positionType="AmountOutstanding",
        instrument="Bond",
    ),
    _RELAX_GUARANTEED: _node(
        "Guaranteed external debt stocks",
        _RECIPIENT,
        positionType="AmountOutstanding",
        guarantee="Public",
    ),
    _RELAX_LONG_TERM: _node(
        "Long-term external debt stocks",
        _RECIPIENT,
        positionType="AmountOutstanding",
        maturity="LongTerm",
    ),
    _RELAX_TOTAL: _node("External debt stocks", _RECIPIENT, positionType="AmountOutstanding"),
}
_RELAX_OBS = {f"{dcid}|country/KEN": _OBS for dcid in _RELAX_NODES}


def _relax_llm(**drafts: tuple[str, str]):
    """A bind double drafting each property's (value, evidence) on the first shape."""

    class _DraftLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema is StatedWordsOutput:
                return NO_RATING
            if schema is QuantityOutput:
                return NO_QUANTITY
            return elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid=prop,
                        kind="value",
                        value_dcids=[value],
                        evidence=cast(Any, evidence),
                    )
                    for prop, (value, evidence) in drafts.items()
                ],
                also_plausible=None,
            ), None

    return _DraftLLM()


def _relax_outcome(result) -> tuple:
    """The status, the refusal reason, the members and bound slots of the answer or of the
    read a refusal flags, and each departure code."""
    (spec,) = result.specs or result.flagged_match or (None,)
    return (
        result.status,
        result.no_data_reason,
        spec and [sv.ref.ref for sv in spec.resolved.members],
        spec
        and {
            slot.property: slot.binding.ref
            for slot in spec.definition.identity.slots
            if slot.binding.state == "value"
        },
        sorted(fact.code for fact in result.departure_facts.values()),
    )


class _RelaxUnreadableWalk(FakeMintGraphReader):
    """A walk reader over the relax members whose reverse query times out."""

    def __init__(self) -> None:
        super().__init__(arcs={dcid: _walk_arcs(node) for dcid, node in _RELAX_NODES.items()})

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        raise TimeoutError("the read timed out")


@pytest.mark.parametrize(
    ("member_graph", "unverified"),
    [
        pytest.param(lambda: _walk_reader(_RELAX_NODES), False, id="walked"),
        pytest.param(_RelaxUnreadableWalk, True, id="unreadable-walk"),
        pytest.param(lambda: None, True, id="no-walk"),
    ],
)
def test_sole_option_drafts_no_member_holds_together_give_way_and_the_named_value_stays(
    member_graph, unverified
):
    """ "Principal repayments" names the flow type alone, and the reply also drafts one
    value on each of three slots because each offered no other. The recalled family holds
    a total, which asserts none of the three, so each claim falls when it is checked
    against the sample (`_validate_sole_option_bindings`): the total answers under the
    named value alone, the three slots shown unbound and no departure (dq-ids-029). The
    family with no total, where the drafts reach the read and give way there, is
    `test_a_family_with_no_total_refuses_once_its_sole_option_drafts_go`.

    A walk that settles the family leaves no `membership_unverified`; a walk that cannot
    read it, and a request with no walk, answer the recalled members under that warning."""
    result = _run(
        entities=["Kenya"],
        obs=_RELAX_OBS,
        nodes=_RELAX_NODES,
        options=EngineOptions(materialise_graph=member_graph()),
        llm=_relax_llm(
            positionType=("AmountOutstanding", "mentioned"),
            instrument=("Bond", "sole_option"),
            guarantee=("Public", "sole_option"),
            maturity=("LongTerm", "sole_option"),
        ),
    )

    assert _relax_outcome(result) == (
        "definite",
        None,
        [_RELAX_TOTAL],
        {"positionType": "AmountOutstanding"},
        [],
    )
    assert not result.unbound_qualifiers
    assert ("membership_unverified" in {w.code for w in result.warnings}) is unverified
    assert (result.specs[0].resolved.shape.member_digest is None) is unverified


@pytest.mark.parametrize("walked", [True, False], ids=["walked", "no-walk"])
def test_a_named_value_gives_way_only_after_every_sole_option_draft_and_is_disclosed(walked):
    """The request names bonds and the reply drafts the guarantee as a sole option. The
    guaranteed member would answer by dropping the named instrument, and the bonds member
    by dropping the draft, so the draft goes and the bonds member answers. With both
    values named, one must go: the first member's conflict is unbound. No word of the
    question names that guarantee, which has no name, so the bonds member answers and
    departs `measure_not_applied` under the guarantee slot's name. The order is the same
    where the walk exposes the conflict and where the read does."""
    pair = {
        _RELAX_GUARANTEED: _RELAX_NODES[_RELAX_GUARANTEED],
        _RELAX_BONDS: _RELAX_NODES[_RELAX_BONDS],
    }

    def options(nodes: dict) -> EngineOptions:
        return EngineOptions(materialise_graph=_walk_reader(nodes) if walked else None)

    drafted = _run(
        entities=["Kenya"],
        obs=_RELAX_OBS,
        nodes=pair,
        options=options(pair),
        llm=_relax_llm(instrument=("Bond", "mentioned"), guarantee=("Public", "sole_option")),
    )
    named = _run(
        entities=["Kenya"],
        obs=_RELAX_OBS,
        nodes=_RELAX_NODES,
        options=options(_RELAX_NODES),
        llm=_relax_llm(instrument=("Bond", "mentioned"), guarantee=("Public", "mentioned")),
    )

    assert _relax_outcome(drafted) == ("definite", None, [_RELAX_BONDS], {"instrument": "Bond"}, [])
    assert _relax_outcome(named) == (
        "definite",
        None,
        [_RELAX_BONDS],
        {"instrument": "Bond"},
        ["measure_not_applied"],
    )
    assert [fact.binding_ref for fact in named.departure_facts.values()] == ["how:guarantee"]


_PPG = "Debt_Ppg"
_PUBLIC_SECTOR = "Debt_PublicSector"
_PPG_NODES = {
    _PPG: _node(
        "Public and publicly guaranteed external debt stocks",
        _RECIPIENT,
        positionType="AmountOutstanding",
        guaranteeStatus="PublicOrPubliclyGuaranteed",
        idsDebtorType="IDSDebtorTypeGG",
    ),
    _PUBLIC_SECTOR: _node(
        "Private nonguaranteed external debt stocks of the public sector",
        _RECIPIENT,
        positionType="AmountOutstanding",
        guaranteeStatus="PrivateNonguaranteed",
        idsDebtorType="IDSDebtorTypePS",
    ),
}
_PPG_VALUE_NODES = {
    "PublicOrPubliclyGuaranteed": {"label": "public and publicly guaranteed", "type": "Thing"},
    "PrivateNonguaranteed": {"label": "private nonguaranteed", "type": "Thing"},
    "IDSDebtorTypePS": {"label": "public sector", "type": "Thing"},
    "IDSDebtorTypeGG": {"label": "general government", "type": "Thing"},
}


@pytest.mark.parametrize(
    ("query", "refuses"),
    [
        ("public and publicly guaranteed debt of Kenya", False),
        ("public and publicly guaranteed public sector debt of Kenya", True),
    ],
    ids=["model-only", "words-name-it"],
)
def test_a_debtor_type_no_member_holds_with_the_guarantee_refuses_only_when_words_name_it(
    query, refuses
):
    """The bind binds the guarantee status the words name and a public-sector debtor type,
    and no member holds both, so the debtor type is unbound and the guaranteed series read.
    No recorded term supports the debtor type when the words do not name it, so the series
    answers and departs `measure_not_applied` under its name. When the words name it, the
    part refuses `qualifier_not_bound` on it, flagging the same series."""
    result = _run(
        entities=["Kenya"],
        obs={f"{dcid}|country/KEN": _OBS for dcid in _PPG_NODES},
        query=query,
        nodes=_PPG_NODES,
        value_nodes=_PPG_VALUE_NODES,
        llm=_relax_llm(
            guaranteeStatus=("PublicOrPubliclyGuaranteed", "mentioned"),
            idsDebtorType=("IDSDebtorTypePS", "mentioned"),
        ),
    )

    (spec,) = result.specs or result.flagged_match or (None,)
    assert spec is not None
    assert [sv.ref.ref for sv in spec.resolved.members] == [_PPG]
    if refuses:
        assert (result.status, result.no_data_reason) == ("no_data", "qualifier_not_bound")
        assert [term.slot for term in result.unbound_qualifiers] == ["idsDebtorType"]
    else:
        assert result.status == "definite"
        assert [
            (fact.code, fact.binding_ref, fact.asked) for fact in result.departure_facts.values()
        ] == [("measure_not_applied", "how:idsDebtorType", "public sector")]


@pytest.mark.parametrize("walked", [True, False], ids=["walked", "no-walk"])
def test_bound_values_no_unbinding_can_admit_a_member_under_refuse_binding_mismatch(walked):
    """The one member asserting the bound instrument carries a guarantee that resolves to
    no node, and the other's instrument resolves to none, so the mint admits neither
    under any bindings. Nothing is unbound and the part refuses `binding_mismatch`."""
    unresolved = {"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]}
    bonds = _node(
        "External debt stocks, bonds",
        _RECIPIENT,
        positionType="AmountOutstanding",
        instrument="Bond",
    )
    guaranteed = _node(
        "Guaranteed external debt stocks, bonds",
        _RECIPIENT,
        positionType="AmountOutstanding",
        instrument="Bond",
        guarantee="Public",
    )
    bonds["arcs"]["instrument"] = unresolved
    guaranteed["arcs"]["guarantee"] = unresolved

    nodes = {_RELAX_BONDS: bonds, _RELAX_GUARANTEED: guaranteed}

    result = _run(
        entities=["Kenya"],
        obs=_RELAX_OBS,
        nodes=nodes,
        options=EngineOptions(materialise_graph=_walk_reader(nodes) if walked else None),
        llm=_relax_llm(instrument=("Bond", "mentioned")),
    )

    assert (result.status, result.no_data_reason) == ("no_data", "binding_mismatch")
    assert result.flagged_match is None
    assert not result.unbound_qualifiers


def test_an_alternate_whose_values_no_member_holds_together_unbinds_a_minimal_set():
    """An alternate's values come from its representative, none from the request, so the
    candidate read weighs no class: the smallest conflict goes, the first member's on a
    tie, and the alternate answers from that member with its value still bound."""
    graph = FakeGraph(nodes={**_PLACES, **_RELAX_NODES}, obs=_RELAX_OBS, detect={}, resolve={})
    (shape,) = derive_shapes(confirmed_svs=list(_RELAX_NODES), graph=graph)
    bindings = [
        SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])
        for prop, value in (
            ("guarantee", "Public"),
            ("instrument", "Bond"),
            ("maturity", "LongTerm"),
        )
    ]

    read = _materialise_alt_shapes(
        [_Alternate(shape, bindings)],
        places=[
            bound_place_context(flow_place_roles(shape, destination=["country/KEN"]), "country/KEN")
        ],
        named_place_dcids=["country/KEN"],
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=_one_role_coverage(_RELAX_NODES, _RELAX_OBS),
    )

    (candidate,) = read.candidates
    assert candidate.mat.sv_dcids == [_RELAX_BONDS]
    assert {
        b.property_dcid: b.value_dcids for b in candidate.bindings if b.property_dcid is not None
    } == {"guarantee": [], "instrument": ["Bond"], "maturity": []}


def test_a_resubmit_posting_values_no_member_holds_together_refuses_and_unbinds_nothing():
    """A posted identity is what the mint resolves, and it resolves no member for a bond
    that is also guaranteed. The resubmit refuses `no_matching_slice` before any read: it
    answers the identity posted or none, so it relaxes nothing and flags no other read."""
    first = _run(
        entities=["Kenya"],
        obs=_RELAX_OBS,
        nodes=_RELAX_NODES,
        options=EngineOptions(materialise_graph=_walk_reader(_RELAX_NODES)),
        llm=_relax_llm(instrument=("Bond", "mentioned")),
    )
    identity = first.specs[0].definition.identity
    posted = identity.model_copy(
        update={
            "slots": [
                IdentitySlot(property=slot.property, binding=IdentityBindingValue(ref="Public"))
                if slot.property == "guarantee"
                else slot
                for slot in identity.slots
            ]
        }
    )

    region = _resubmit(posted, nodes=_RELAX_NODES)

    assert (region.status, region.no_data_reason) == ("no_data", "no_matching_slice")
    assert region.flagged_match is None
    assert not region.unbound_qualifiers


def test_a_walk_whose_reverse_read_holds_no_candidate_finds_no_member():
    """The reverse read under a bond binding holds no candidate, while more recalled seeds
    are silent on the instrument than the walk re-reads, so it leaves some undismissed. The
    mint walks the reverse read alone, with no seed, and serves no member for the identity,
    so the walk reports no member, the fact the place-binding relaxation reads, and not a
    family it could not read."""
    nodes = {
        f"Debt_Seed_{i}": _node(f"Debt stocks {i}", _RECIPIENT, positionType=f"Position{i}")
        for i in range(40)
    }
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    (shape,) = derive_shapes(confirmed_svs=list(nodes), graph=graph)
    bond = SlotBindingDraft(
        axis="how", property_dcid="instrument", kind="value", value_dcids=["Bond"]
    )

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [bond], ["destination"]),
        walks=RoleClassWalks(_walk_reader(nodes)),
        arcs_graph=graph,
    )

    assert resolution.authority is MembershipAuthority.FAILED
    assert resolution.no_member is True


@pytest.mark.parametrize("walked", [True, False], ids=["walked", "no-walk"])
def test_a_standard_part_is_not_relaxed_where_its_walk_finds_no_member(walked):
    """A subject-only family of the total and its male part, with female bound. The walk
    finds no member under it, and the mint's rule, which counts the total's silence on
    gender as a conflict, would unbind female and refuse on it. A standard part reads
    its members by its own rule, under which the silent total admits, so the walk changes
    nothing: both answers read the total and find no observation."""
    nodes = {
        "Pop_Total": _node("People", [], positionType="Stock"),
        "Pop_Male": _node("Male people", [], positionType="Stock", gender="Male"),
    }

    result = _run(
        entities=["Kenya"],
        obs={f"{dcid}|country/KEN": _OBS for dcid in nodes},
        nodes=nodes,
        query="female people in Kenya",
        options=EngineOptions(materialise_graph=_walk_reader(nodes) if walked else None),
        llm=_relax_llm(gender=("Female", "mentioned")),
    )

    assert _relax_outcome(result) == ("no_data", "no_observations", ["Pop_Total"], {}, [])
    assert not result.unbound_qualifiers


@pytest.mark.parametrize(
    ("query", "rated", "outcome"),
    [
        ("aged X in Kenya", [], ("definite", None, ["Pop_B"], {"age": "X"}, [])),
        (
            "female aged X in Kenya",
            [(("female",), {"age": ("X",)})],
            ("no_data", "qualifier_not_bound", ["Pop_B"], {"age": "X"}, []),
        ),
    ],
    ids=["unnamed", "named"],
)
def test_a_standard_part_unbinds_a_sole_option_value_before_a_named_one(
    monkeypatch, query, rated, outcome
):
    """Female is drafted as the gender slot's sole option and age X is named. No member
    holds both: A is female at age Y, B male at age X. The sole-option claim gives way
    silently and B answers, where unbinding the named age would refuse on it. Validation
    of the claim against the population (`_validate_sole_option_bindings`) would drop it
    first here, so it is held off to reach the read. A query naming "female" leaves the
    word on no bound code once Female is unbound, so the binder rates it on B under the
    bindings left, and the part refuses on its rating, as it would had the bind never
    drafted Female."""
    nodes = {
        "Pop_A": _node("Females aged Y", [], positionType="Stock", gender="Female", age="Y"),
        "Pop_B": _node("Males aged X", [], positionType="Stock", gender="Male", age="X"),
    }
    monkeypatch.setattr(
        phase_election, "_validate_sole_option_bindings", lambda bindings, shape: bindings
    )
    binder = RatingBinder(GeminiBinder(), {"female": 0.0})

    result = _run(
        entities=["Kenya"],
        obs={f"{dcid}|country/KEN": _OBS for dcid in nodes},
        query=query,
        nodes=nodes,
        llm=_relax_llm(gender=("Female", "sole_option"), age=("X", "mentioned")),
        binder=binder,
    )

    assert [(request.words, restricting(request.bindings)) for request in binder.requests] == rated
    assert _relax_outcome(result) == outcome
    assert bool(result.unbound_qualifiers) == bool(rated)


@pytest.mark.parametrize("walked", [True, False], ids=["walked", "no-walk"])
def test_a_family_with_no_total_refuses_once_its_sole_option_drafts_go(monkeypatch, walked):
    """The family holds no member asserting the named value alone, and its bonds member
    declares a creditor where the others declare the debtor alone. Every member conflicts
    on a sole-option draft, so all three go, and the members left under the named value
    declare role sets the election cannot order: the part refuses
    `role_signature_incompatible`, as the mint refuses the identity, and no value is refused
    on. The recalled sample does not refute the drafts, so each stands as a sampled claim.
    The walk reads every sampled claim open, meets those role sets at once and refuses
    with nothing unbound. With no walk the read unbinds all three drafts, leaving no draft
    bound."""
    nodes = {
        _RELAX_BONDS: _node(
            "External debt stocks, bonds",
            _PAIR,
            positionType="AmountOutstanding",
            instrument="Bond",
        ),
        _RELAX_GUARANTEED: _RELAX_NODES[_RELAX_GUARANTEED],
        _RELAX_LONG_TERM: _RELAX_NODES[_RELAX_LONG_TERM],
    }
    states = []

    def unbind(state, unrealisable):
        unbind_unrealisable(state, unrealisable)
        states.append(state)

    monkeypatch.setattr(phase_place_materialise, "unbind_unrealisable", unbind)

    result = _run(
        entities=["Kenya"],
        obs=_RELAX_OBS,
        nodes=nodes,
        options=EngineOptions(materialise_graph=_walk_reader(nodes) if walked else None),
        llm=_relax_llm(
            positionType=("AmountOutstanding", "mentioned"),
            instrument=("Bond", "sole_option"),
            guarantee=("Public", "sole_option"),
            maturity=("LongTerm", "sole_option"),
        ),
    )

    assert (result.status, result.no_data_reason) == ("no_data", "role_signature_incompatible")
    assert result.flagged_match is None
    assert not result.unbound_qualifiers
    assert [
        {
            b.property_dcid: b.value_dcids
            for b in state.bindings
            if b.kind == "value" and b.property_dcid is not None
        }
        for state in states
    ] == ([] if walked else [{"positionType": ["AmountOutstanding"]}])


@pytest.mark.parametrize("walked", [True, False], ids=["walk-point", "read"])
def test_the_served_dataset_is_settled_again_under_the_bindings_an_unbinding_leaves(
    monkeypatch, walked
):
    """The dataset a part is served from is named from its bindings, so each point that
    unbinds values settles it again on what remains, and a part no table then serves
    answers with that refusal."""
    settled = []

    def settle(state, role_keys):
        settled.append(
            {b.property_dcid for b in state.bindings if b.kind == "value" and b.property_dcid}
        )
        return state.no_data("variable_not_resolved")

    monkeypatch.setattr(phase_place_materialise, "_settle_served_dataset", settle)
    nodes = {dcid: node for dcid, node in _RELAX_NODES.items() if dcid != _RELAX_TOTAL}

    result = _run(
        entities=["Kenya"],
        obs=_RELAX_OBS,
        nodes=nodes,
        options=EngineOptions(materialise_graph=_walk_reader(nodes) if walked else None),
        llm=_relax_llm(
            positionType=("AmountOutstanding", "mentioned"),
            instrument=("Bond", "sole_option"),
            guarantee=("Public", "sole_option"),
            maturity=("LongTerm", "sole_option"),
        ),
    )

    assert settled == [{"positionType"}]
    assert (result.status, result.no_data_reason) == ("no_data", "variable_not_resolved")
