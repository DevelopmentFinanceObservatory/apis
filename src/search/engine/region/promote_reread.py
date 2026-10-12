"""Re-read member arcs and verify their shared five-tuple and shape."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from code_index.shared import get_index, get_measure_index
from dfo_platform.named import named_capped
from materialise.graph import UNRESOLVED_NODE
from search.engine.envelope import now_ms
from search.engine.errors import (
    EngineInfraError,
    EngineInputError,
    FamilyNotResolvableError,
    MembershipTooBroadError,
    RoleSignatureError,
)
from search.engine.region.codes import MEMBERS_ABSENT_FROM_GRAPH
from search.engine.region.promote_results import _no_data_region, _scope_too_broad_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.resubmit_request import _identity_slots_to_wire
from search.engine.region.result import RegionResult
from search.engine.retrieval.bound_values import (
    elect_measure_family,
    identity_names_measure_family,
    identity_query,
    index_decides_membership,
    shape_member_facts,
    synthetic_shape_for_identity,
)
from search.engine.retrieval.derive import group_shape_id, member_group_class
from search.engine.retrieval.facts import _value_for, read_five_tuple
from search.engine.shape import identity_shape_for
from search.models import IdentityShapeSchema, Warning


def _family_measurement_qualifier(arcs: dict) -> str | None:
    """Return the qualifier value that the family arc reader will expose."""
    value = _value_for(arcs, "measurementQualifier")
    if value is not None:
        return value
    nodes = arcs.get("measurementQualifier", {}).get("nodes", [])
    return UNRESOLVED_NODE if nodes else None


def _elected_members(state: _PromoteState) -> list[str] | RegionResult:
    """The members of the family the posted identity resolves, as the Datasets mint elects
    them: the packaged index's election for an identity that names a measure family, the
    code index's family when it decides the identity, else the antichain-minimal members
    of the mint's graph walk (`RoleClassWalks.family` in its resubmit mode), which reads
    the whole filtered set up to the subject ceiling when its first page truncates without
    proof. A family holding a row the index lists outside its code systems, as IDS families
    do, is elected from the index with no graph read (`code_index.members.listed_group`).

    A settled walk that finds no member returns `no_matching_slice`, where the mint refuses
    `spec_not_exportable`. A family the mint refuses as too broad returns
    `scope_too_broad`. Role sets the election cannot order return `no_data`
    `role_signature_incompatible`. A member at fault, a query with no measured property and
    a walk whose reads leave a candidate undismissed return `no_data`
    `variable_not_resolved`. A walk past its deadline or a failed graph read raises,
    failing the request. A walk that runs the full read raises `state.deadline` to the
    bound it ran to, so the phases after it share that bound."""
    identity = state.inp.identity
    query = identity_query(identity)
    if identity_names_measure_family(identity, get_measure_index().axes):
        election = elect_measure_family(
            synthetic_shape_for_identity(cast(IdentityShapeSchema, identity.shape)),
            get_index(),
            get_measure_index().axes,
            query,
        )
        if election is None:
            return _no_data_region(state.variable_text)
        # Record forced axes; bindings are rebuilt from the elected members' arcs.
        state.measure_elected_properties = frozenset(prop for prop, _value in election.forced)
        return list(election.elected.members)
    if index_decides_membership(identity):
        # A synthetic shape holds no members, so it must not read as an exact elected
        # set, which `shape_member_facts` would return as its (empty) members.
        shape = replace(
            synthetic_shape_for_identity(cast(IdentityShapeSchema, identity.shape)),
            sv_arc_facts_complete=False,
        )
        members, _index_sourced = shape_member_facts(shape, query)
        return list(members)
    if state.role_class_walks is None:
        raise EngineInfraError(
            "materialise graph client not initialised; cannot elect the members of a "
            "standard identity"
        )
    try:
        walked = state.role_class_walks.family(query, (), deadline=state.deadline)
    except RoleSignatureError:
        return _no_data_region(state.variable_text, "role_signature_incompatible")
    except FamilyNotResolvableError:
        return _no_data_region(state.variable_text, "variable_not_resolved")
    except MembershipTooBroadError as exc:
        return _scope_too_broad_region(state.variable_text, state.inp.identity, str(exc))
    if walked.value is None:
        return _no_data_region(state.variable_text, "no_matching_slice")
    group = walked.value.group
    if group.full_read and walked.value.bound is not None:
        state.deadline = walked.value.bound
    # The walk's antichain-minimal members are what the mint serves; the shape phase holds
    # the whole family (`elected_role_class_shape`).
    return [member.stat_var for member in group.members]


def _members_absent_warning(absent: list[str]) -> Warning:
    """Name the elected members the graph holds no arcs for, which the answer leaves out."""
    return Warning(
        code=MEMBERS_ABSENT_FROM_GRAPH,
        severity="warn",
        message=(
            f"{len(absent)} of the series this definition holds are not in the graph, so the "
            f"answer leaves them out: {named_capped(absent)}"
        ),
    )


def _phase_reread_arcs(state: _PromoteState) -> RegionResult | None:
    """Elect the identity's members, re-read their arcs, verify the shared five-tuple, and
    regenerate the shape.

    Members come from `_elected_members`, and an election that settles none returns its
    refusal. Members whose arcs the graph does not hold drop under a
    `members_absent_from_graph` warning naming them, and `no_data` returns if none
    remain. Mixed five-tuples raise `request_invalid`. A regenerated shape mismatch
    raises `shape_mismatch` (422). The shape slug supplies `variable_text`, and errors
    propagate unchanged.

    Timing splits in two disjoint steps: `walk` times `_elected_members`, and
    `reread_arcs` times the rest of the phase (arc batch, five-tuple check, shape
    regeneration), so the two sum to the phase's wall time."""
    t0 = now_ms()
    try:
        try:
            elected = _elected_members(state)
        finally:
            state.timing["walk"] = now_ms() - t0
        if isinstance(elected, RegionResult):
            return elected
        state.stat_var_dcids = elected

        # Re-read arcs for every elected SV.
        arcs_map = state.graph.node_arcs_batch(state.stat_var_dcids)
        absent = [sv for sv in state.stat_var_dcids if arcs_map.get(sv) is None]
        if absent:
            state.warnings.append(_members_absent_warning(absent))
        state.stat_var_dcids = [sv for sv in state.stat_var_dcids if arcs_map.get(sv) is not None]
        if not state.stat_var_dcids:
            return _no_data_region(state.variable_text, warnings=tuple(state.warnings))
        # Every remaining member has confirmed arcs.
        confirmed_arcs: dict[str, dict] = {sv: a for sv, a in arcs_map.items() if a is not None}
        state.confirmed_arcs = confirmed_arcs

        # Extract a five-tuple for every SV and require them to share one.
        five_tuples = {sv: read_five_tuple(confirmed_arcs[sv]) for sv in state.stat_var_dcids}
        unique_fts = set(five_tuples.values())
        if len(unique_fts) > 1:
            detail = (
                "the identity's elected members span multiple five-tuple shapes; "
                "a candidate is one shape"
            )
            raise EngineInputError(
                detail,
                code="request_invalid",
            )

        ft = state.ft = next(iter(unique_fts))
        state.family_measurement_qualifiers = frozenset(
            _family_measurement_qualifier(arcs) for arcs in confirmed_arcs.values()
        )

        # Rebuild the identity shape from its five-tuple. Structural equality verifies the
        # identity. The shape id is the one derive gives the members' group, read from the
        # members alone (`derive.group_shape_id`), so a resubmit keeps its answer's id.
        classes = [member_group_class(dcid, ft, arcs) for dcid, arcs in confirmed_arcs.items()]
        families = {family.properties if family else None for _role_bearing, family in classes}
        state.index_family = classes[0][1] if len(families) == 1 else None
        regenerated_shape_id = state.regenerated_shape_id = group_shape_id(
            ft,
            role_bearing=any(role_bearing for role_bearing, _family in classes),
            family=state.index_family,
        )
        regenerated_shape = identity_shape_for(
            pop_type_dcid=ft.pop_type_dcid,
            meas_prop_dcid=ft.meas_prop_dcid,
            stat_type_dcid=ft.stat_type_dcid,
            meas_qual_dcid=ft.meas_qual_dcid,
            meas_denom_dcid=ft.meas_denom_dcid,
            meas_denom_unresolved=ft.meas_denom_unresolved,
        )

        if regenerated_shape != state.inp.identity.shape.model_dump(exclude_none=True):
            detail = "shape does not match the five-tuple of the identity's elected members"
            raise EngineInputError(
                detail,
                code="shape_mismatch",
                status=422,
                pointer="/input/identity/shape",
            )
        # Respell variable_text from the re-read draft's own display slug,
        # superseding the fallback rendering _phase_validate set.
        state.variable_text = regenerated_shape_id
        state.posted_slots = _identity_slots_to_wire(state.inp.identity.slots, confirmed_arcs)
        return None
    finally:
        state.timing["reread_arcs"] = now_ms() - t0 - state.timing["walk"]
