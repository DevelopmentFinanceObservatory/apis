"""Bind the posted places to the resubmit's shape and derive its place context."""

from __future__ import annotations

from code_index.flow_family import flow_family
from code_index.shared import get_index
from dfo_platform.named import named_capped
from dfo_platform.roles import FlowFamily
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError, RoleEntityUnplaced
from search.engine.place import read_default_presence
from search.engine.region.place_context import place_context
from search.engine.region.place_roles import PlaceRoles
from search.engine.region.promote_results import _no_data_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.refs import confirm_dcids_present, refuse_wrong_kind
from search.engine.region.result import RegionResult
from search.engine.region.state import identity_served_stat_vars, presence_stat_vars
from search.engine.retrieval.bound_values import RoleKeys
from search.engine.retrieval.facts import read_constraints
from search.engine.shape import ShapeDraft, entities_by_role, is_role_bearing
from search.role_kinds import graph_types


def _posted_entities(state: _PromoteState) -> dict[str, list[str]] | None:
    """Map the posted entities to the shape's role names, or `None` when one has no role.

    A directional shape reads the destination and origin refs; an observationAbout ref on it
    raises `role_incompatible` (422). A subject shape reads the observationAbout refs; a
    destination or origin ref on it raises `role_incompatible` (422). An entity on an axis the
    shape declares no role on returns `None`."""
    if is_role_bearing(state.shape_draft):
        if state.identity_entities.get("observationAbout"):
            detail = "observationAbout entity references are not valid for a directional shape"
            raise EngineInputError(
                detail,
                code="role_incompatible",
                status=422,
                pointer="/input/identity/entities/observationAbout",
            )
        try:
            return entities_by_role(
                shape=state.shape_draft,
                by_axis={
                    "destination": state.identity_entities.get("destination", []),
                    "origin": state.origin_dcids,
                },
            )
        except RoleEntityUnplaced:
            return None
    for role in ("destination", "origin"):
        role_dcids = state.identity_entities.get(role, [])
        if not role_dcids:
            continue
        detail = (
            f"{role} reference(s) {named_capped(role_dcids)} are not a role shape "
            f"{state.shape_draft.shape_id!r}'s resolved StatVars declare; its "
            "members assert no directional role"
        )
        raise EngineInputError(
            detail,
            code="role_incompatible",
            status=422,
            pointer=f"/input/identity/entities/{role}",
        )
    try:
        return entities_by_role(
            shape=state.shape_draft,
            subject=state.identity_entities.get("observationAbout", []),
        )
    except RoleEntityUnplaced:
        return None


def _posted_place_roles(
    state: _PromoteState, resolved_entities: dict[str, list[str]]
) -> PlaceRoles:
    """The posted identity's binding of its places to the shape.

    A directional shape binds its destinations as the where places and keeps its origins; a
    subject shape binds every posted place as its subject, the anchor first. An identity
    carries no direction, so no direction is dropped. A directional shape leaves open the
    axes the identity's `open_roles` name."""
    where_dcids = state.standard_where_dcids
    if is_role_bearing(state.shape_draft):
        return PlaceRoles(
            destination_dcids=where_dcids,
            origin_dcids=state.origin_dcids,
            open_axes=frozenset(state.inp.identity.open_roles or ()),
            resolved_entities=resolved_entities,
            roles=[(dcid, "destination") for dcid in where_dcids]
            + [(dcid, "origin") for dcid in state.origin_dcids],
            where_dcids=where_dcids,
            obs_destination_dcid=state.entity_dcid,
        )
    anchor = [state.entity_dcid] if state.entity_dcid else []
    return PlaceRoles(
        destination_dcids=[],
        origin_dcids=[],
        open_axes=frozenset(),
        resolved_entities=resolved_entities,
        roles=[(dcid, "observationAbout") for dcid in anchor]
        + [(dcid, "observationAbout") for dcid in where_dcids if dcid != state.entity_dcid],
        where_dcids=where_dcids,
        obs_destination_dcid=state.entity_dcid,
    )


def _shape_family(shape: ShapeDraft) -> FlowFamily | None:
    """The flow family of `shape`'s members (`code_index.flow_family.flow_family`), read from
    their recalled arcs, which a refusal's role wording reads."""
    arcs = shape.sv_arc_facts or {}
    return flow_family(
        index=get_index(),
        population_type=shape.pop_type_dcid or None,
        stat_vars=list(arcs),
        members=[read_constraints(member_arcs) for member_arcs in arcs.values()],
    )


def _phase_where_places(state: _PromoteState) -> RegionResult | None:
    """Bind the posted places to the shape and derive its place context (`place_context`).

    Every read the context drives names the posted dcids themselves. A posted continent or
    subregion is read at its own node, as the Datasets mint reads it. Confirm every entity
    ref. A subject shape with no posted place reads the default-scope presence of its
    StatVars. Missing refs raise `invalid_role_ref` (422), as does a role-bearing shape's ref
    whose kind its role does not take (`refuse_wrong_kind`), and a ref on a role the shape
    cannot take raises `role_incompatible` (422). An entity on an axis the shape declares no
    role on returns `no_data` with `role_incompatible`, carrying the place warnings."""
    t0 = now_ms()
    try:
        # entity_dcid and the complete role member list were read directly from
        # identity.entities in _phase_validate. The identity core canonicalises
        # these values, so no slot-level reconstruction is needed here.
        standard_where_dcids = state.standard_where_dcids

        # Confirm every posted entity role ref, including the full where set. Missing refs
        # raise `invalid_role_ref` (422); constraint values come from confirmed SV arcs.
        role_dcids = {
            role: dcids
            for role, dcids in state.identity_entities.items()
            if role in {"observationAbout", "destination", "origin"} and dcids
        }
        confirm_dcids_present(
            role_dcids,
            read=state.graph.node_labels_batch,
            pointer_for=lambda role, _absent: f"/input/identity/entities/{role}",
        )
        if is_role_bearing(state.shape_draft):
            refuse_wrong_kind(
                role_dcids,
                read_types=graph_types(state.graph),
                family=_shape_family(state.shape_draft),
                pointer_prefix="/input/identity",
            )

        # A subject shape with no posted place probes the countries where its StatVars,
        # the resubmit's membership, have data. The shape holds its family exactly, so the
        # read covers every member, whatever bindings unbinding leaves.
        presence = state.default_presence = (
            read_default_presence(
                graph=state.graph,
                sdmx=state.sdmx,
                stat_vars=presence_stat_vars(state.shape_draft, state.bindings, RoleKeys()),
                deadline=state.deadline,
            )
            if not standard_where_dcids and not is_role_bearing(state.shape_draft)
            else None
        )

        # An unplaced entity refuses once the place warnings are on the state, so its no-data
        # answer carries them.
        resolved_entities = _posted_entities(state)
        place = place_context(
            state.shape_draft,
            _posted_place_roles(state, resolved_entities or {}),
            stat_vars=(
                identity_served_stat_vars(state.inp.identity, state.shape_draft)
                if presence is not None
                else ()
            ),
            presence=presence,
        )
        state.warnings.extend(place.warnings)
        if resolved_entities is None:
            return _no_data_region(
                state.variable_text, "role_incompatible", warnings=tuple(state.warnings)
            )
        state.place = place
        return None
    finally:
        state.timing["where_places"] = now_ms() - t0
