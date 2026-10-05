"""Search's membership resolution states how an answer's members were decided.

For an identity the code index does not decide, the mint's walk, seeded with the recalled or
posted members, settles the members (`MembershipAuthority.AUTHORITATIVE` or `PROVEN`), and
its family replaces any members that differ from it, whatever their roles. A walk that does
not run (`SAMPLED`) or settles nothing (`FAILED`) leaves the recalled or posted members to
answer, under a `membership_unverified` warning the response cache will not store.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from dc_graph import GraphDeadlineExceededError, GraphReadError
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.region.codes import MEMBERSHIP_UNVERIFIED
from search.engine.region.grounding import carrier_flag_warnings
from search.engine.region.place_roles import PlaceRoles
from search.engine.region.state import EngineOptions, served_stat_vars
from search.engine.response_cache_keys import is_storable
from search.engine.retrieval.bound_values import (
    MembershipAuthority,
    RoleClassWalks,
    elected_role_class_shape,
    posted_members_outside_role_class,
    shape_query,
)
from search.engine.retrieval.derive import derive_shapes
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from search.models import BaseIdentity, ResubmitIdentityPayload
from series_coverage.shared import CoverageSnapshot
from spec_registry.identity import recompute_identity
from statvar_members import MemberEnumerator, five_tuple_from_identity
from tests.datasets_api._graph_doubles import FakeMintGraphReader, _FakeSubjectPage
from tests.engine._builders import bound_place_context, materialise_alternates, subject_place_roles
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, member_walk_reader, offline_resolve
from tests.engine.test_place_scope_default import (
    _DETECT_QUERY,
    _EARTH_CHILDREN,
    _TOTAL_AND_PART_NODES,
    _scope_sdmx,
)
from tests.engine.test_place_scope_default import _SV as _POPULATION
from tests.engine.test_place_scope_default import _SV_FEMALE as _POPULATION_FEMALE
from tests.engine.test_place_scope_default import _run as _run_place_less
from tests.engine.test_role_class_shapes import (
    _BY_CREDITOR,
    _BY_CREDITOR_GUARANTEED,
    _DEBT_NODES,
    _OBS,
    _PLACES,
    _RECIPIENT,
    _TOTAL,
    _TOTAL_GUARANTEED,
    _BindLLM,
    _debt_shape,
    _node,
    _one_role_coverage,
    _posted,
    _walk_arcs,
    _walk_reader,
)
from tests.fixtures import FakeGraph, make_resolver

_KENYA = {"Kenya": [DoorRole(axis="destination", span="Kenya")]}


class _FailingWalkReader(FakeMintGraphReader):
    """A walk reader over `nodes` whose reverse query raises `error`."""

    def __init__(self, error: Exception, nodes: dict = _DEBT_NODES) -> None:
        super().__init__(arcs={dcid: _walk_arcs(node) for dcid, node in nodes.items()})
        self._error = error

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        raise self._error


class _TruncatedWalkReader(FakeMintGraphReader):
    """A walk reader whose reverse query reports more candidates past its window, which no
    root among the debt members proves irrelevant."""

    def __init__(self) -> None:
        super().__init__(arcs={dcid: _walk_arcs(node) for dcid, node in _DEBT_NODES.items()})

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        return _FakeSubjectPage(dcids=page.dcids, truncated=True)


_UNSETTLED_WALKS = {
    "timeout": lambda: _FailingWalkReader(GraphDeadlineExceededError("the read timed out")),
    "graph_failure": lambda: _FailingWalkReader(GraphReadError("the graph returned HTTP 503")),
    "incomplete_enumeration": _TruncatedWalkReader,
}

# A total population and its female part, which declare no flow role. `gender` reads
# `aggregate`, so the total dominates the part.
_UNSETTLED_POPULATION_WALKS = {
    "no_member_graph": lambda: None,
    "timeout": lambda: _FailingWalkReader(
        GraphDeadlineExceededError("the read timed out"), _TOTAL_AND_PART_NODES
    ),
    "graph_failure": lambda: _FailingWalkReader(
        GraphReadError("the graph returned HTTP 503"), _TOTAL_AND_PART_NODES
    ),
}


def _debt_coverage() -> CoverageSnapshot:
    """The coverage index holding each recipient-only debt series at Kenya, where Search
    reads a one-role series' facets."""
    return _one_role_coverage(_DEBT_NODES, {f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES})


def _arcs_graph() -> FakeGraph:
    return FakeGraph(nodes=_DEBT_NODES, obs={}, detect={}, resolve={})


def _run(recalled: list[str], *, member_graph: FakeMintGraphReader | None):
    query = "external debt stocks of Kenya"
    graph = FakeGraph(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={query: {"svs": recalled, "cosine_scores": [0.9] * len(recalled)}},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES},
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=["Kenya"],
            entity_roles=_KENYA,
            date_request=None,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=_debt_coverage(),
            options=EngineOptions(materialise_graph=member_graph),
        )
    )


def _resubmit(identity: BaseIdentity, posted: list[str], *, member_graph: FakeMintGraphReader):
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
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=_debt_coverage(),
        options=EngineOptions(materialise_graph=member_graph),
    )


def _minted_members(identity: dict, *, seed: list[str] | None = None) -> list[str]:
    graph = _walk_reader(_DEBT_NODES)
    resolved = resolve_definition_members(
        identity,
        spec_id="spec_membership",
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
        seed=seed,
    )
    return [shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes]


def _codes(result) -> list[str]:
    return [warning.code for warning in result.warnings]


def _population_graph(*recalled: str) -> FakeGraph:
    return FakeGraph(
        nodes=_TOTAL_AND_PART_NODES,
        detect={_DETECT_QUERY: {"svs": list(recalled), "cosine_scores": [0.9] * len(recalled)}},
        resolve={},
        obs={
            f"{dcid}|country/A1": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
            for dcid in _TOTAL_AND_PART_NODES
        },
        children=_EARTH_CHILDREN,
    )


def _subject_only_shape(*members: str, posted: bool = False) -> ShapeDraft:
    (shape,) = derive_shapes(confirmed_svs=list(members), graph=_population_graph())
    return dataclasses.replace(shape, sv_arc_facts_complete=posted)


# ---------------------------------------------------------------------------
# A settled walk replaces recalled members that differ from its family
# ---------------------------------------------------------------------------


def test_a_walk_that_finds_more_of_the_samples_role_class_replaces_the_sample():
    """The sample holds the guaranteed part alone. The walk elects the same recipient-only
    class and finds the total beside it, so the shape takes the walked family."""
    shape = _debt_shape(_TOTAL_GUARANTEED)

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [], ["destination"]),
        walks=RoleClassWalks(_walk_reader(_DEBT_NODES)),
        arcs_graph=_arcs_graph(),
    )

    assert resolution.authority is MembershipAuthority.AUTHORITATIVE
    assert set(resolution.value.sv_arc_facts or {}) == {_TOTAL, _TOTAL_GUARANTEED}
    assert resolution.value.label == "External debt stocks"


def test_the_answer_and_its_emitted_identity_mint_the_same_members():
    """Recall holds only the guaranteed debt part and the guarantee slot stays unbound. The
    mint resolves the emitted identity to the unrestricted total, so the answer lists it."""
    result = _run([_TOTAL_GUARANTEED], member_graph=_walk_reader(_DEBT_NODES))

    assert result.status == "definite"
    spec = result.specs[0]
    guarantee = next(
        slot for slot in spec.definition.identity.slots if slot.property == "guarantee"
    )
    assert guarantee.binding.state == "unbound"
    answered = [sv.ref for sv in spec.resolved.stat_vars.items]
    assert answered == _minted_members(
        spec.definition.identity.model_dump(mode="json", exclude_none=True)
    )
    assert answered == [_TOTAL]
    assert MEMBERSHIP_UNVERIFIED not in _codes(result)


# ---------------------------------------------------------------------------
# A walk that does not run, or settles nothing, leaves the sample under a warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("walk", list(_UNSETTLED_WALKS))
def test_an_unsettled_walk_leaves_the_sample_and_says_so(walk):
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [], ["destination"]),
        walks=RoleClassWalks(_UNSETTLED_WALKS[walk]()),
        arcs_graph=_arcs_graph(),
    )

    assert resolution.authority is MembershipAuthority.FAILED
    assert resolution.value is shape


def test_a_shape_with_no_walk_is_answered_from_its_sample():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    resolution = elected_role_class_shape(shape, shape_query(shape, [], ["destination"]))

    assert resolution.authority is MembershipAuthority.SAMPLED
    assert resolution.value is shape


def test_a_failed_read_of_the_walked_familys_arcs_leaves_the_sample():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [], ["destination"]),
        walks=RoleClassWalks(_walk_reader(_DEBT_NODES)),
        arcs_graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}, raise_on_call=True),
    )

    assert resolution.authority is MembershipAuthority.FAILED
    assert resolution.value is shape


@pytest.mark.parametrize("walk", list(_UNSETTLED_WALKS))
def test_an_answer_whose_walk_settles_nothing_warns_and_is_not_cached(walk):
    """The answer still comes from the recalled members with data, carries
    `membership_unverified`, and the response cache refuses it."""
    result = _run([_TOTAL, _TOTAL_GUARANTEED], member_graph=_UNSETTLED_WALKS[walk]())

    assert result.status == "definite"
    (warning,) = [w for w in result.warnings if w.code == MEMBERSHIP_UNVERIFIED]
    base = offline_resolve(
        make_request("total population India"), materialise_graph=member_walk_reader()
    )
    assert is_storable(base, timed_out=False, degraded_leg=False)
    tainted = base.model_copy(
        update={"parts": [base.parts[0].model_copy(update={"warnings": [warning]})]}
    )
    assert not is_storable(tainted, timed_out=False, degraded_leg=False)


def test_an_answer_with_no_member_graph_warns():
    result = _run([_TOTAL, _TOTAL_GUARANTEED], member_graph=None)

    assert result.status == "definite"
    assert MEMBERSHIP_UNVERIFIED in _codes(result)


def test_an_answer_whose_walk_settles_its_members_does_not_warn():
    result = _run([_TOTAL, _TOTAL_GUARANTEED], member_graph=_walk_reader(_DEBT_NODES))

    assert result.status == "definite"
    assert MEMBERSHIP_UNVERIFIED not in _codes(result)


# ---------------------------------------------------------------------------
# A posted set
# ---------------------------------------------------------------------------


def test_a_posted_set_checked_by_a_settled_walk_is_authoritative():
    query = shape_query(_debt_shape(), [], ["destination"])

    resolution = posted_members_outside_role_class(
        _posted(_TOTAL), query, walks=RoleClassWalks(_walk_reader(_DEBT_NODES))
    )

    assert (resolution.value, resolution.authority) == ((), MembershipAuthority.AUTHORITATIVE)


@pytest.mark.parametrize("walk", list(_UNSETTLED_WALKS))
def test_a_posted_set_whose_walk_settles_nothing_is_checked_among_itself(walk):
    query = shape_query(_debt_shape(), [], ["destination"])

    resolution = posted_members_outside_role_class(
        _posted(_TOTAL, _BY_CREDITOR), query, walks=RoleClassWalks(_UNSETTLED_WALKS[walk]())
    )

    assert (resolution.value, resolution.authority) == (
        (_BY_CREDITOR,),
        MembershipAuthority.FAILED,
    )


def test_a_posted_set_with_no_walk_is_sampled():
    query = shape_query(_debt_shape(), [], ["destination"])

    resolution = posted_members_outside_role_class(_posted(_TOTAL), query)

    assert (resolution.value, resolution.authority) == ((), MembershipAuthority.SAMPLED)


def test_posted_members_that_declare_no_flow_role_are_checked_by_the_walk():
    shape = _subject_only_shape(_POPULATION_FEMALE)
    reader = member_walk_reader(_population_graph())

    resolution = posted_members_outside_role_class(
        dict(shape.sv_arc_facts or {}), shape_query(shape, [], []), walks=RoleClassWalks(reader)
    )

    assert (resolution.value, resolution.authority) == ((), MembershipAuthority.AUTHORITATIVE)
    assert reader.calls["node_subjects"] == 1


@pytest.mark.parametrize("walk", list(_UNSETTLED_WALKS))
def test_a_resubmit_whose_walk_settles_nothing_warns(walk):
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity

    settled = _resubmit(identity, [_TOTAL], member_graph=_walk_reader(_DEBT_NODES))
    unsettled = _resubmit(identity, [_TOTAL], member_graph=_UNSETTLED_WALKS[walk]())

    assert settled.status == unsettled.status == "definite"
    assert MEMBERSHIP_UNVERIFIED not in _codes(settled)
    assert MEMBERSHIP_UNVERIFIED in _codes(unsettled)


def test_a_resubmit_posting_the_guaranteed_part_answers_the_total_the_mint_serves():
    """The identity leaves the guarantee open and the resubmit posts the guaranteed part. The
    mint seeds its walk with the part and serves the total, so the answer lists the total."""
    identity = (
        _run([_TOTAL_GUARANTEED], member_graph=_walk_reader(_DEBT_NODES))
        .specs[0]
        .definition.identity
    )

    result = _resubmit(identity, [_TOTAL_GUARANTEED], member_graph=_walk_reader(_DEBT_NODES))

    assert result.status == "definite"
    (spec,) = result.specs
    assert spec.definition.identity == identity
    answered = [sv.ref for sv in spec.resolved.stat_vars.items]
    assert answered == _minted_members(
        identity.model_dump(mode="json", exclude_none=True), seed=[_TOTAL_GUARANTEED]
    )
    assert answered == [_TOTAL]
    assert MEMBERSHIP_UNVERIFIED not in _codes(result)


# ---------------------------------------------------------------------------
# A shape that declares no flow role
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("posted", [False, True])
def test_a_walk_that_finds_the_total_of_a_subject_only_part_replaces_the_part(posted):
    """The shape holds the female part alone, recalled or posted, and `gender` is open. The
    walk finds the total beside it, so the shape holds both and the mint's total is served."""
    shape = _subject_only_shape(_POPULATION_FEMALE, posted=posted)

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [], []),
        walks=RoleClassWalks(member_walk_reader(_population_graph())),
        arcs_graph=_population_graph(),
    )

    assert resolution.authority is MembershipAuthority.AUTHORITATIVE
    assert set(resolution.value.sv_arc_facts or {}) == {_POPULATION, _POPULATION_FEMALE}
    assert resolution.value.sv_arc_facts_complete is posted
    assert served_stat_vars(resolution.value, []) == [_POPULATION]


def test_a_walked_family_keeps_the_order_of_the_members_the_shape_held():
    """Later readers rank a shape's members by their order, so the recalled part stays first
    and the total the walk adds follows it."""
    shape = _subject_only_shape(_POPULATION_FEMALE)

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [], []),
        walks=RoleClassWalks(member_walk_reader(_population_graph())),
        arcs_graph=_population_graph(),
    )

    assert list(resolution.value.sv_arc_facts or {}) == [_POPULATION_FEMALE, _POPULATION]


def test_a_shape_with_an_unresolved_denominator_is_walked_under_its_identitys_five_tuple():
    """An identity states an unresolved denominator with no ref, and the mint walks that
    five-tuple, so the shape's query drops the ref the shape read from the graph."""
    shape = dataclasses.replace(
        _subject_only_shape(_POPULATION), meas_denom_dcid="PerCapita", meas_denom_unresolved=True
    )
    identity = {
        "kind": "statvar_schema",
        "population_type": shape.pop_type_dcid,
        "measured_property": shape.meas_prop_dcid,
        "stat_type": shape.stat_type_dcid,
        "measurement_denominator": {"unresolved": True},
    }

    assert shape_query(shape, []).five_tuple == five_tuple_from_identity(identity)


def test_a_posted_set_that_is_the_walked_family_is_kept():
    shape = _subject_only_shape(_POPULATION, _POPULATION_FEMALE, posted=True)

    resolution = elected_role_class_shape(
        shape,
        shape_query(shape, [], []),
        walks=RoleClassWalks(member_walk_reader(_population_graph())),
        arcs_graph=_population_graph(),
    )

    assert (resolution.value, resolution.authority) == (
        shape,
        MembershipAuthority.AUTHORITATIVE,
    )


@pytest.mark.parametrize("walk", list(_UNSETTLED_POPULATION_WALKS))
def test_a_subject_only_answer_whose_walk_settles_nothing_answers_its_recall_and_warns(walk):
    """Recall holds the female part alone. With no walk, or one that settles nothing, the part
    answers, and the answer carries `membership_unverified`."""
    graph = _population_graph(_POPULATION_FEMALE)

    result = _run_place_less(
        graph=graph,
        sdmx=_scope_sdmx(graph),
        materialise_graph=_UNSETTLED_POPULATION_WALKS[walk](),
    )

    assert result.status == "definite"
    assert [sv.ref for sv in result.specs[0].resolved.stat_vars.items] == [_POPULATION_FEMALE]
    assert MEMBERSHIP_UNVERIFIED in _codes(result)


# ---------------------------------------------------------------------------
# An alternate
# ---------------------------------------------------------------------------


def _alternates(
    member_graph: FakeMintGraphReader,
    shape: ShapeDraft | None = None,
    *,
    nodes: dict = _DEBT_NODES,
):
    """Walk and materialise `shape` (the debt total and its guaranteed part by default) as an
    alternate bound at Kenya, as candidate admission does."""
    roles = PlaceRoles(
        destination_dcids=["country/KEN"],
        origin_dcids=[],
        resolved_entities={"financingRecipient": ["country/KEN"]},
        roles=[("country/KEN", "destination")],
        where_dcids=["country/KEN"],
        obs_destination_dcid="country/KEN",
    )
    return materialise_alternates(
        [shape if shape is not None else _debt_shape(_TOTAL, _TOTAL_GUARANTEED)],
        places=[bound_place_context(roles, "country/KEN")],
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=["country/KEN"],
        graph=FakeGraph(
            nodes={**_PLACES, **nodes},
            detect={},
            resolve={},
            obs={f"{dcid}|country/KEN": _OBS for dcid in nodes},
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=None,
        coverage=_one_role_coverage(nodes, {f"{dcid}|country/KEN": _OBS for dcid in nodes}),
        role_class_walks=RoleClassWalks(member_graph),
    )


@pytest.mark.parametrize("walk", ["timeout", "graph_failure"])
def test_an_alternate_whose_walk_settles_nothing_carries_the_flag(walk):
    (settled,), _ = _alternates(_walk_reader(_DEBT_NODES))
    (unsettled,), _ = _alternates(_UNSETTLED_WALKS[walk]())

    assert settled.mat.membership_unverified is False
    assert unsettled.mat.membership_unverified is True
    warnings = carrier_flag_warnings(
        role_declaration_unverified=False,
        open_origin_defaulted=False,
        membership_unverified=unsettled.mat.membership_unverified,
    )
    assert [warning.code for warning in warnings] == [MEMBERSHIP_UNVERIFIED]


def test_an_alternate_whose_truncated_walk_a_root_proves_complete_is_settled():
    """The alternate binds its representative's position type, and the plain total is the
    root that proves no unread candidate changes the family, so the walk is `PROVEN`."""
    (proven,), _ = _alternates(_TruncatedWalkReader())

    assert proven.mat.membership_unverified is False


def test_an_alternate_a_settled_walk_narrows_is_scoped_and_answered_over_the_walked_family():
    """The alternate's representative, the guaranteed by-creditor series, binds the public
    guarantee, so the mint's walk under the recipient alone elects the guaranteed total. The
    recall also holds the plain total, which the walk does not confirm, so the alternate's
    served members and its answer read the guaranteed total alone."""
    shape = _debt_shape(_BY_CREDITOR_GUARANTEED, _BY_CREDITOR, _TOTAL_GUARANTEED, _TOTAL)

    (answered,), _ = _alternates(_walk_reader(_DEBT_NODES), shape)

    assert served_stat_vars(answered.shape, answered.bindings) == [_TOTAL_GUARANTEED]
    assert answered.mat.sv_dcids == [_TOTAL_GUARANTEED]
    assert set(answered.shape.sv_arc_facts or {}) == {_TOTAL_GUARANTEED}


_TOTAL_PRIVATE = "Debt_Total_Private"
_PRIVATE_NODES = {
    **_DEBT_NODES,
    _TOTAL_PRIVATE: _node(
        "Private external debt stocks",
        _RECIPIENT,
        positionType="AmountOutstanding",
        guarantee="Private",
    ),
}


@pytest.mark.parametrize("walk", ["timeout", "graph_failure"])
def test_an_alternate_whose_walk_settles_nothing_is_scoped_over_the_members_it_answers(walk):
    """The guaranteed total's public guarantee is bound, and no walk settles the members, so
    the alternate answers from its recalled members that hold that guarantee, and its served
    members are the same: the private-guarantee total is in neither."""
    (shape,) = derive_shapes(
        confirmed_svs=[_TOTAL_GUARANTEED, _TOTAL_PRIVATE],
        graph=FakeGraph(nodes=_PRIVATE_NODES, obs={}, detect={}, resolve={}),
    )

    (answered,), _ = _alternates(_UNSETTLED_WALKS[walk](), shape, nodes=_PRIVATE_NODES)

    assert answered.mat.membership_unverified is True
    assert answered.mat.sv_dcids == [_TOTAL_GUARANTEED]
    assert served_stat_vars(answered.shape, answered.bindings) == [_TOTAL_GUARANTEED]


# ---------------------------------------------------------------------------
# An alternate that declares no flow role
# ---------------------------------------------------------------------------


def _population_alternate(reader: FakeMintGraphReader, *members: str):
    """Walk and materialise a place-less alternate of `members`, the female part by
    default, probed at A1."""
    graph = _population_graph()
    (candidate,), _ = materialise_alternates(
        [_subject_only_shape(*(members or (_POPULATION_FEMALE,)))],
        places=[bound_place_context(subject_place_roles(), "country/A1")],
        elected_bindings=[],
        elected_has_data=False,
        named_place_dcids=[],
        graph=graph,
        sdmx=_scope_sdmx(graph),
        date_request=None,
        coverage=empty_coverage_snapshot(),
        role_class_walks=RoleClassWalks(reader),
    )
    return candidate


def test_an_offered_alternate_that_declares_no_flow_role_is_walked_and_flagged_like_a_role_bearing_one():
    """It is walked before it is offered, so it lists the members its own resubmit serves,
    and a walk that settles nothing flags it."""
    settled = member_walk_reader(_population_graph())

    walked = _population_alternate(settled)
    unsettled = _population_alternate(_UNSETTLED_POPULATION_WALKS["graph_failure"]())

    assert settled.calls["node_subjects"] == 1
    assert walked.mat.membership_unverified is False
    assert unsettled.mat.membership_unverified is True


def test_an_offered_alternate_listing_a_member_the_mint_does_not_serve_answers_the_walked_family():
    """The alternate holds the total and the female part with `gender` open, and the mint
    serves the total, so the alternate is offered over the walked family."""
    candidate = _population_alternate(
        member_walk_reader(_population_graph()), _POPULATION, _POPULATION_FEMALE
    )

    assert candidate.mat.sv_dcids == [_POPULATION]
    assert candidate.mat.membership_unverified is False
