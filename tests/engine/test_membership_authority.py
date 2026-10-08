"""Search's membership resolution states how an answer's members were decided.

For an identity the code index does not decide, the mint's walk, seeded on a first answer
with the recalled members, settles the members (`MembershipAuthority.AUTHORITATIVE` or
`PROVEN`), and its family replaces any members that differ from it, whatever their roles. On
a first answer, a walk that does not run (`SAMPLED`) or settles nothing (`FAILED`) leaves the
recalled members to answer, under a `membership_unverified` warning the response cache will
not store. A resubmit has no recalled members: a walk past its bound or a failed read fails
the request retryably, and a family past the ceiling answers `scope_too_broad`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from dc_graph import GraphDeadlineExceededError, GraphReadError
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphInfraError
from search.engine.region import promote_reread, resolve_spec_resubmit, resolve_variable
from search.engine.region.codes import MEMBERSHIP_UNVERIFIED
from search.engine.region.grounding import carrier_flag_warnings
from search.engine.region.place_context import PlaceContext
from search.engine.region.place_roles import PlaceRoles
from search.engine.region.promote_state import _PromoteState
from search.engine.region.state import EngineOptions, served_stat_vars
from search.engine.response_cache_keys import is_storable
from search.engine.retrieval.bound_values import (
    MembershipAuthority,
    ResubmitWalk,
    RoleClassWalks,
    RoleKeys,
    elected_role_class_shape,
    entity_role_keys,
    shape_query,
)
from search.engine.retrieval.derive import derive_shapes
from search.engine.role_text import DoorRole
from search.engine.shape import ShapeDraft
from search.models import BaseIdentity, EntityGroup, ResubmitIdentityPayload
from series_coverage.shared import CoverageSnapshot
from statvar_members import Member, MemberEnumerator, MemberGroup, five_tuple_from_identity
from tests.datasets_api._graph_doubles import FakeMintGraphReader, _FakeSubjectPage
from tests.datasets_api._member_bounds import mint_bounds
from tests.engine._builders import bound_place_context, materialise_alternates, subject_place_roles
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, member_walk_reader, offline_resolve
from tests.engine._search_app import search_app
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
    _walk_arcs,
    _walk_reader,
)
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

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
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=_debt_coverage(),
            options=EngineOptions(materialise_graph=member_graph),
        )
    )


def _resubmit(identity: BaseIdentity, *, member_graph: FakeMintGraphReader):
    graph = FakeGraph(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES},
    )
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=_debt_coverage(),
        options=EngineOptions(materialise_graph=member_graph),
    )


def _minted_members(identity: dict) -> list[str]:
    graph = _walk_reader(_DEBT_NODES)
    resolved = resolve_definition_members(
        identity,
        spec_id="spec_membership",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
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
    answered = [sv.ref.dcid for sv in spec.resolved.members]
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
    """A walk that fails a read or runs past its bound settles nothing and marks the
    request's walks degraded; one that truncates without proof settles nothing, cleanly."""
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)
    walks = RoleClassWalks(_UNSETTLED_WALKS[walk]())

    with recording_degraded_reads() as lost:
        resolution = elected_role_class_shape(
            shape, shape_query(shape, [], ["destination"]), walks=walks, arcs_graph=_arcs_graph()
        )

    assert resolution.authority is MembershipAuthority.FAILED
    assert resolution.value is shape
    assert lost.degraded is (walk != "incomplete_enumeration")


def test_a_shape_with_no_walk_is_answered_from_its_sample():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    resolution = elected_role_class_shape(shape, shape_query(shape, [], ["destination"]))

    assert resolution.authority is MembershipAuthority.SAMPLED
    assert resolution.value is shape


def test_a_failed_read_of_the_walked_familys_arcs_leaves_the_sample():
    shape = _debt_shape(_BY_CREDITOR, _BY_CREDITOR_GUARANTEED)

    walks = RoleClassWalks(_walk_reader(_DEBT_NODES))

    with recording_degraded_reads() as lost:
        resolution = elected_role_class_shape(
            shape,
            shape_query(shape, [], ["destination"]),
            walks=walks,
            arcs_graph=FakeGraph(nodes={}, obs={}, detect={}, resolve={}, raise_on_call=True),
        )

    assert resolution.authority is MembershipAuthority.FAILED
    assert resolution.value is shape
    assert lost.degraded


@pytest.mark.parametrize("walk", list(_UNSETTLED_WALKS))
def test_an_answer_whose_walk_settles_nothing_warns_and_is_not_cached(walk):
    """The answer still comes from the recalled members with data, carries
    `membership_unverified`, and the response cache refuses it."""
    result = _run([_TOTAL, _TOTAL_GUARANTEED], member_graph=_UNSETTLED_WALKS[walk]())

    assert result.status == "definite"
    assert result.degraded is (walk != "incomplete_enumeration")
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
# A resubmit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("walk", "failure"),
    [("timeout", TimeoutError), ("graph_failure", GraphInfraError)],
)
def test_a_resubmit_whose_walk_fails_fails_retryably(walk, failure):
    """A resubmit holds no recalled member to answer from, so a walk that runs past its
    bound fails the request with a timeout (504) and one whose read fails with an
    upstream error (503), never a silent no-data part."""
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity

    with pytest.raises(failure):
        _resubmit(identity, member_graph=_UNSETTLED_WALKS[walk]())


def test_a_resubmit_whose_family_is_past_the_ceiling_answers_scope_too_broad():
    """A filtered set the full read cannot finish under the subject ceiling is a family the
    mint refuses as too broad; the part says so, pointing at the identity."""
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity

    result = _resubmit(identity, member_graph=_TruncatedWalkReader())

    assert result.status == "unsupported"
    assert result.unsupported is not None
    assert (result.unsupported.code, result.unsupported.pointer) == (
        "scope_too_broad",
        "/definition/identity",
    )
    assert result.unsupported.definition["identity"] == identity.model_dump(
        mode="json", exclude_none=True
    )


class _WithholdingWalkReader(FakeMintGraphReader):
    """A walk reader over the debt members whose forward reads never return `withheld`'s
    arcs, so the walk cannot dismiss it."""

    def __init__(self, withheld: str) -> None:
        super().__init__(arcs={dcid: _walk_arcs(node) for dcid, node in _DEBT_NODES.items()})
        self._withheld = withheld

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        arcs = super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )
        return {dcid: None if dcid == self._withheld else value for dcid, value in arcs.items()}


def test_a_resubmit_whose_walk_leaves_a_candidate_undismissed_refuses_as_the_mint_does():
    """The mint refuses a walk that leaves a candidate undismissed `member_not_resolvable`,
    so the resubmit answers `variable_not_resolved`, not a settled `no_matching_slice`."""
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity

    result = _resubmit(identity, member_graph=_WithholdingWalkReader(_BY_CREDITOR))

    assert (result.status, result.no_data_reason) == ("no_data", "variable_not_resolved")


def test_a_resubmit_whose_walk_settles_does_not_warn():
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity

    settled = _resubmit(identity, member_graph=_walk_reader(_DEBT_NODES))

    assert settled.status == "definite"
    assert not settled.degraded
    assert MEMBERSHIP_UNVERIFIED not in _codes(settled)


@pytest.mark.parametrize(
    ("walk", "status", "code"),
    [("timeout", 504, "timeout"), ("graph_failure", 503, "service_unavailable")],
)
def test_a_resubmit_whose_walk_fails_answers_a_retryable_problem(walk, status, code):
    """Over HTTP, the walk's failure reaches the client as a retryable problem, never as a
    no-data part a cache could keep."""
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity
    graph = FakeGraph(nodes={**_PLACES, **_DEBT_NODES}, detect={}, resolve={})
    client = TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            materialise_graph=_UNSETTLED_WALKS[walk](),
        ),
        raise_server_exceptions=False,
    )

    resp = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "external debt stocks",
                        "identity": identity.model_dump(mode="json"),
                    }
                ],
            }
        },
    )

    body = resp.json()
    assert (resp.status_code, body["code"], body["retryable"]) == (status, code, True)


class _FullReadEnumerator:
    """Records each walk's arguments and answers with a family its full read settled."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def enumerate(self, **kwargs) -> MemberGroup:
        self.calls.append(kwargs)
        member = Member(stat_var=_TOTAL, constraints={}, declared_roles=("debtor",))
        return MemberGroup(
            five_tuple=kwargs["five_tuple"],
            members=(member,),
            family=(_TOTAL,),
            candidates_considered=1,
            candidate_cap=kwargs["subject_ceiling"],
            truncated=False,
            proven_complete=True,
            root_confirmed=True,
            full_read=True,
        )


def test_a_resubmit_walk_reads_unseeded_under_the_ceiling_and_its_full_read_deadline():
    """A resubmit's walk takes no seed, reads its first page under the request deadline and
    its full read under the full-read deadline, and the family it settles carries that
    later bound for every read after it."""
    enumerator = _FullReadEnumerator()
    walks = RoleClassWalks(
        _walk_reader(_DEBT_NODES),
        enumerator=cast(MemberEnumerator, enumerator),
        resubmit=ResubmitWalk(subject_ceiling=20_000, full_read_deadline=1_090.0),
    )

    resolution = walks.family(
        shape_query(_debt_shape(), [], ["destination"]), (_BY_CREDITOR,), deadline=1_045.0
    )

    (call,) = enumerator.calls
    assert (call["extra_candidates"], call["subject_ceiling"]) == ((), 20_000)
    assert (call["deadline"], call["full_read_deadline"]) == (1_045.0, 1_090.0)
    assert resolution.value is not None
    assert resolution.value.bound == 1_090.0


def test_a_resubmit_whose_walk_ran_the_full_read_moves_its_deadline_to_the_full_read_bound():
    """The phases after the election share the bound the full read ran to, so grounding and
    observations are not cut off at the request deadline the full read passed."""
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity
    walks = RoleClassWalks(
        _walk_reader(_DEBT_NODES),
        enumerator=cast(MemberEnumerator, _FullReadEnumerator()),
        resubmit=ResubmitWalk(subject_ceiling=20_000, full_read_deadline=1_090.0),
    )
    state = SimpleNamespace(
        inp=ResubmitIdentityPayload(identity=identity),
        deadline=1_045.0,
        role_class_walks=walks,
        variable_text="external debt stocks",
    )

    elected = promote_reread._elected_members(cast(_PromoteState, state))

    assert elected == [_TOTAL]
    assert state.deadline == 1_090.0


@pytest.mark.parametrize("full_read", [False, True], ids=["first_page", "full_read"])
def test_a_resubmits_region_reads_to_the_bound_its_walk_ran_to(
    monkeypatch: pytest.MonkeyPatch, full_read: bool
):
    """The region states the bound its reads ran to, which the trailing stream takes: the
    full-read bound after a full read, else the request deadline."""
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity
    enumerator = _FullReadEnumerator() if full_read else MemberEnumerator()
    monkeypatch.setattr(
        "search.engine.retrieval.bound_values.shared_enumerator", lambda: enumerator
    )
    graph = FakeGraph(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES},
    )
    deadline = time.monotonic() + 30

    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=_debt_coverage(),
        options=EngineOptions(materialise_graph=_walk_reader(_DEBT_NODES), deadline=deadline),
        full_read_deadline=deadline + 45,
    )

    assert region.status == "definite"
    assert region.reads_deadline == (deadline + 45 if full_read else deadline)


def test_a_resubmits_walk_is_named_by_the_posted_identitys_spec_id(
    monkeypatch: pytest.MonkeyPatch,
):
    """The walk's log line names the spec id of the identity the resubmit posts, so a slow
    walk ties to its request."""
    spec = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0]
    enumerator = _FullReadEnumerator()
    monkeypatch.setattr(
        "search.engine.retrieval.bound_values.shared_enumerator", lambda: enumerator
    )

    _resubmit(spec.definition.identity, member_graph=_walk_reader(_DEBT_NODES))

    assert [call["spec_id"] for call in enumerator.calls] == [spec.spec_id]


def test_a_resubmit_dropping_a_group_for_a_failed_read_is_degraded():
    """A named group whose members read fails drops, and the answer says it stands on less
    than it read (`RegionResult.degraded`), so no cache keeps it."""
    identity = _run([_TOTAL], member_graph=_walk_reader(_DEBT_NODES)).specs[0].definition.identity
    groups = {"destination": EntityGroup(dcid="africa", label="Africa")}

    class _GroupReadFails(FakeGraph):
        def node_subjects_many(self, *args, **kwargs):
            raise GraphInfraError("members read failed")

    graph = _GroupReadFails(
        nodes={**_PLACES, **_DEBT_NODES},
        detect={},
        resolve={"Kenya": "country/KEN"},
        obs={f"{dcid}|country/KEN": _OBS for dcid in _DEBT_NODES},
    )
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity, entity_groups=groups),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=_debt_coverage(),
        options=EngineOptions(materialise_graph=_walk_reader(_DEBT_NODES)),
    )

    assert region.status == "definite"
    assert region.degraded
    assert region.specs[0].resolved.entity_groups is None


def test_a_resubmit_leaving_the_guarantee_open_answers_the_total_the_mint_serves():
    """The identity leaves the guarantee open. The first answer recalled the guaranteed part
    alone; the resubmit's walk, as the mint's, serves the total."""
    identity = (
        _run([_TOTAL_GUARANTEED], member_graph=_walk_reader(_DEBT_NODES))
        .specs[0]
        .definition.identity
    )

    result = _resubmit(identity, member_graph=_walk_reader(_DEBT_NODES))

    assert result.status == "definite"
    (spec,) = result.specs
    assert spec.definition.identity == identity
    answered = [sv.ref.dcid for sv in spec.resolved.members]
    assert answered == _minted_members(identity.model_dump(mode="json", exclude_none=True))
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
    assert served_stat_vars(resolution.value, [], RoleKeys()) == [_POPULATION]


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

    assert shape_query(shape, [], ()).five_tuple == five_tuple_from_identity(identity)


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
    assert [sv.ref.dcid for sv in result.specs[0].resolved.members] == [_POPULATION_FEMALE]
    assert MEMBERSHIP_UNVERIFIED in _codes(result)


# ---------------------------------------------------------------------------
# An alternate
# ---------------------------------------------------------------------------


def _role_keys(place: PlaceContext) -> RoleKeys:
    return entity_role_keys(place.roles.resolved_entities, open_axes=place.roles.open_axes)


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

    assert served_stat_vars(answered.shape, answered.bindings, _role_keys(answered.place)) == [
        _TOTAL_GUARANTEED
    ]
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
    assert served_stat_vars(answered.shape, answered.bindings, _role_keys(answered.place)) == [
        _TOTAL_GUARANTEED
    ]


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
