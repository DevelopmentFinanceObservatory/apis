"""Resolve an options request into a server-derived target for `surface.py`. Runs no
resubmit pipeline and no materialise pipeline: the door never re-resolves a
Spec, so it runs no survivor election. Owns the identity-to-target derivation, the
anchor-ref presence check (invalid_role_ref), and the role-target check
(option_target_invalid). Does not compute an option surface itself (that is
surface.py's job, over the ResolvedOptionsTarget this module returns).

Before any of that, `resolve_options_target` runs the posted identity
through the authoritative identity core (`identity_edit.
validate_edited_identity`), the same gate `region.resubmit_promote`'s own
`_phase_validate` runs first -- `BaseIdentity`'s pydantic schema mirrors most
of the core's shape but not every cross-member invariant (two slots naming
the same property collapse silently in a naive dict build rather than
refuse, for one), so this door needs the same core pass every other door
already gets. It also runs `_check_entity_role_admission`, mirroring
`region.resubmit_promote._phase_ground_build`'s own admission rule (an
observationAbout entity refused for a role-bearing population, a
destination/origin entity refused for a subject-only one, and an entity on
a flow axis the population declares no role for) -- inline against
`role_names_by_axis` rather than through `engine.shape.entities_by_role`
itself, since this door builds no `ShapeDraft` to hand that function.

## What this module derives, and from where

- **The StatVar population**: `options.population.options_population`, the
  server-side derivation seam -- either the release's code index's full
  population for a shape it carries (CRS today), or the mint's graph walk
  through the shared enumerator (`statvar_members.shared_enumerator`,
  SDG-shaped asks). See that module's own docstring for the two-source split
  and its refusals (`population_empty`, `scope_too_broad`,
  `role_signature_incompatible`), which propagate uncaught from here.
  `options_population` reads its own `graph` parameter as a
  `materialise.GraphReader` (`node_subjects`/`node_arcs_many`) -- NOT
  Search's own `EngineGraphClient`, which does not implement that
  Protocol (see `MemberArcReader`'s own module docstring) -- so this module
  calls it with `materialise_graph`, the same materialise-package client
  pair app.py already threads through every door, required here rather
  than optional: `app.py`'s own route refuses `EngineInfraError` ahead of
  calling this module at all when it is unconfigured, since deriving a
  population is this door's one genuinely load-bearing use of it.
- **Typed anchors**: every dcid `identity.entities` binds (observationAbout,
  origin, destination alike), typed in one batched `graph.node_types_batch`
  call (`_typed_anchors`). A posted dcid the graph does not carry -- absent,
  or present but type-less, indistinguishable through this one batched read
  -- refuses `invalid_role_ref` (422), the same refusal
  `region.resubmit_promote._phase_where_places` raises for the identical
  fact (there via `node_labels_batch`; here via `node_types_batch`,
  since a typed anchor is what the surface half needs, not a label).
- **Facets**: `identity.facets`, carried through unchanged -- a sparse
  `{axis: IdentityFacetBinding}` map, `None` when no axis is bound. No
  re-derivation: the identity core already IS the source of truth for what
  was bound.
- **The requested window**: `identity.requested_window` (the identity
  core's own always-present `IdentityWindow` tagged union), translated to
  the engine's own `RequestedWindow` form via
  `region.resubmit_request._requested_window_from_identity` -- reused
  rather than re-implemented, since it is already the one place that
  translation happens.
- **Role names by axis**: `_declared_axis_roles`, computed straight off
  `population.declared_roles` -- `options_population` has already settled
  the population's single agreed role signature (both sources elect one
  role class through `statvar_members.elect_role_class`: an index-sourced
  population through `code_index.members.index_family`, a graph-sourced one
  inside `statvar_members.enumerate_members`), so this
  reads that signature rather than issuing a fresh `graph.node_arcs_batch`
  fetch.

## Role-target derivation (option_target_invalid)

`_check_role_target` is the single check for both directions of the
role-target question: a subject-only population (`role_names_by_axis ==
{}`) accepts only `observationAbout`; a role-bearing population accepts
exactly its own declared flow axes (origin and/or destination) and refuses
`observationAbout`. The client names the role the slot is bound to
(financingRecipient, say), never `observationAbout` as a stand-in for it.
Both directions raise the identical `option_target_invalid` (422, pointer
`/for/role`), naming the population's own accepted axes as `alternatives`.
Deriving `role_names_by_axis` costs no graph read: it is already implied by
the population this module derives, regardless of which role the request
names.

The population is derived from the identity with the requested flow axis open
(`identity_edit.open_requested_role`), the identity an edit binding that axis posts.
An IDS total asked for `origin` therefore reads its by-creditor twin, which declares
the creditor role, while a population with no StatVar declaring the axis drops the
open role at election and still refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from code_index.flow_family import flow_family
from code_index.shared import get_index
from dfo_platform import Alternative
from dfo_platform.named import cut, named_capped
from materialise import GraphReader
from materialise.roles import RoleAxis, roles_by_axis
from search.engine.clients.graph import EngineGraphClient
from search.engine.errors import EngineInputError
from search.engine.identity_edit import (
    check_described_arm,
    composite_table_identity,
    open_requested_role,
    refuse_over_place_bound,
    validate_edited_identity,
)
from search.engine.options.population import OptionsPopulation, options_population
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region.refs import confirm_dcids_present, refuse_wrong_kind
from search.engine.region.resubmit_request import _requested_window_from_identity
from search.engine.retrieval.bound_values import (
    canonicalise_identity,
    refuse_source_outside_family,
)
from search.models import (
    BaseIdentity,
    EntityRoleKey,
    IdentityFacetBinding,
    IdentityShapeSchema,
    OptionsRequest,
    RequestedWindow,
    ResolvedEntity,
)
from search.role_kinds import graph_types


def _typed_anchors(
    identity: BaseIdentity, *, graph: EngineGraphClient
) -> tuple[ResolvedEntity, ...]:
    """Type every bound entity with one `node_types_batch` call.

    A missing or type-less dcid raises `invalid_role_ref`/422 at its role.
    """
    role_dcids = {role: dcids for role, dcids in identity.entities.items() if dcids}
    if not role_dcids:
        return ()
    type_by_dcid = confirm_dcids_present(
        role_dcids,
        read=graph.node_types_batch,
        pointer_for=lambda role, _absent: f"/identity/entities/{role}",
    )
    return tuple(
        ResolvedEntity(ref=dcid, entity_type=type_by_dcid.get(dcid), role=cast(EntityRoleKey, role))
        for role, dcids in role_dcids.items()
        for dcid in dcids
    )


def _declared_axis_roles(population: OptionsPopulation) -> dict[RoleAxis, str]:
    """Map declared flow axes to their role names; subject-only populations yield `{}`.

    Duplicate roles on one axis raise `role_signature_incompatible`/422.
    """
    try:
        return roles_by_axis(population.declared_roles)
    except ValueError as exc:
        raise EngineInputError(
            "the resolved StatVars declare two roles on the same flow axis, "
            "so the axis has no single role to enumerate",
            code="role_signature_incompatible",
            status=422,
            pointer="/for/role",
        ) from exc


def _check_entity_role_admission(
    identity: BaseIdentity, role_names_by_axis: dict[RoleAxis, str]
) -> None:
    """Refuse entity bindings the population's role signature cannot use.

    Subject-only populations accept `observationAbout`; role-bearing
    populations accept only declared flow axes. Without this check, an
    unsupported anchor would be ignored by the surface builders.

    Raises `role_incompatible`/422 at `/identity/entities/{axis}`.
    """
    if not role_names_by_axis:
        for axis in ("origin", "destination"):
            dcids = identity.entities.get(axis, [])
            if dcids:
                detail = (
                    f"{axis} reference(s) {named_capped(dcids)} are not a role this "
                    "identity's resolved StatVars declare; its members assert "
                    "no directional role"
                )
                raise EngineInputError(
                    detail,
                    code="role_incompatible",
                    status=422,
                    pointer=f"/identity/entities/{axis}",
                )
        return
    if identity.entities.get("observationAbout"):
        detail = "observationAbout entity references are not valid for a directional shape"
        raise EngineInputError(
            detail,
            code="role_incompatible",
            status=422,
            pointer="/identity/entities/observationAbout",
        )
    for axis in ("origin", "destination"):
        dcids = identity.entities.get(axis, [])
        if dcids and axis not in role_names_by_axis:
            detail = (
                f"{axis} reference(s) {named_capped(dcids)} name a flow axis this "
                "identity's resolved StatVars declare no role for; declared "
                f"axes are {', '.join(sorted(role_names_by_axis))}"
            )
            raise EngineInputError(
                detail,
                code="role_incompatible",
                status=422,
                pointer=f"/identity/entities/{axis}",
            )


def _check_role_target(req: OptionsRequest, role_names_by_axis: dict[RoleAxis, str]) -> None:
    """Refuse `option_target_invalid` (422) for a `for.role` the population's
    declared role signature does not accept. See the module docstring's
    "Role-target derivation" section for why this is one check covering
    both directions: a subject-only population accepts only
    `observationAbout`; a role-bearing one accepts exactly its own declared
    flow axes and refuses `observationAbout`.
    """
    declared_axes: tuple[str, ...] = tuple(sorted(role_names_by_axis)) or ("observationAbout",)
    if req.for_.role in declared_axes:
        return
    raise EngineInputError(
        f"{cut(req.for_.role)!r} is not a role this definition's resolved StatVars "
        f"declare; the resolved shape declares {', '.join(declared_axes)}",
        code="option_target_invalid",
        status=422,
        pointer="/for/role",
        alternatives=[Alternative(ref=axis, label=axis) for axis in declared_axes],
    )


@dataclass(frozen=True)
class ResolvedOptionsTarget:
    """Resolved inputs required to build one role's option surface.

    `identity` is the posted identity, canonicalised, with the requested flow axis open
    (`identity_edit.open_requested_role`). The population is the one it elects.
    `population` supplies StatVars; `anchors` are typed entities from one
    `node_types_batch` call. `facets` and `requested_window` pass through the
    identity, with the window translated to engine form. `role_names_by_axis`
    maps declared flow axes to role names; it is empty for subject-only data.
    """

    identity: BaseIdentity
    population: OptionsPopulation
    anchors: tuple[ResolvedEntity, ...]
    facets: dict[str, IdentityFacetBinding] | None
    requested_window: RequestedWindow | None
    role_names_by_axis: dict[RoleAxis, str]


def resolve_options_target(
    req: OptionsRequest,
    *,
    graph: EngineGraphClient,
    materialise_graph: GraphReader,
    population_memo: OptionsPopulationMemo | None = None,
    deadline: float | None = None,
    join_inflight: bool = True,
) -> ResolvedOptionsTarget:
    """Resolve the posted request into the target used by the options surface.

    `materialise_graph` is required for graph-backed population derivation.

    Args:
        population_memo: optional per-surface population memo; `None` derives
            directly without caching.
        deadline: bounds the population's graph walk and a memo follower's wait.
        join_inflight: when false, an occupied memo key raises
            `PopulationInFlight` instead of waiting; unused without a memo.

    Raises:
        EngineInputError: `request_invalid` (400) for a pinned-ask shape
            (`check_described_arm`), a composite whose components cannot be derived
            (`composite_table_identity`) or an identity that fails the
            authoritative identity core (`validate_edited_identity`) --
            duplicate slot properties chief among what this catches that
            `BaseIdentity`'s own schema does not, or an identity binding more
            places than a Spec holds (`refuse_over_place_bound`), at the role's path
            or at `/identity/entities`; `facet_not_available` (422) at
            `/identity/facets/provenance` for an identity naming a dataset that does
            not publish its five-tuple (`refuse_source_outside_family`), before any
            read; every refusal
            `options.population.options_population` raises propagates
            unchanged (`population_empty`, `scope_too_broad`,
            `role_signature_incompatible`, 422 each); `invalid_role_ref`
            (422) for a posted entity ref the graph does not carry
            (`_typed_anchors`, run before population derivation so an
            unresolvable ref refuses before the door pays for an expensive
            enumeration or index read); `role_signature_incompatible` (422)
            for a population whose declared roles conflict on one flow axis
            (`_declared_axis_roles`); `role_incompatible` (422) for an
            entity bound to an axis the resolved population does not serve
            (`_check_entity_role_admission`); `invalid_role_ref` (422) for a
            flow-role entity whose kind the role does not take
            (`region.refs.refuse_wrong_kind`); `option_target_invalid` (422)
            when `req.for_.role` names a role the population does not
            accept (`_check_role_target`).
        options.population_memo.PopulationInFlight: only when
            `join_inflight=False` and `population_memo` is not `None` --
            the population key is already owned by another caller's
            in-flight derivation (see `OptionsPopulationMemo.get_or_derive`'s
            own docstring). Never raised at the default `join_inflight=True`.
    """
    identity = req.identity
    # This door's own field for a pinned-ask shape is /identity/shape, not
    # the resubmit path's /input/identity/shape -- this door has no entry
    # wrapper to remove a pointer prefix from, so the pointer is passed
    # directly rather than rewritten after the fact. Likewise
    # validate_edited_identity's own pointer, /identity, not
    # /input/identity -- the same door-own-field reasoning.
    check_described_arm(identity.shape, pointer="/identity/shape")
    validate_edited_identity(identity, pointer="/identity")
    # A composite's options are those of the identity its option was set on, whose one
    # table every read names.
    identity = composite_table_identity(identity, pointer="/identity")
    refuse_over_place_bound(identity.entities, pointer="/identity/entities")
    # A pin posted bound and one posted unbound denote one population.
    identity = canonicalise_identity(identity)
    # The door enumerates the requested flow axis member by member, so it reads the
    # population an edit binding that axis elects, not the posted total's.
    identity = open_requested_role(identity, role=req.for_.role)
    # A named dataset that does not publish the five-tuple refuses here as it does at the
    # mint and the resubmit.
    refuse_source_outside_family(identity, pointer="/identity/facets/provenance")

    # Anchors first: one batched node_types_batch call, cheap next to
    # options_population's own index read or graph enumeration (up to
    # enumerate_members's own cost bound). A posted ref the graph does not
    # carry refuses invalid_role_ref before the door pays for the expensive
    # read, rather than after.
    anchors = _typed_anchors(identity, graph=graph)
    # Through the caller's own memo (options.population_memo), not
    # options_population directly, whenever one is supplied: a repeat ask
    # for the same shape+slots -- across requests on the same surface, not
    # just within one -- is a hit rather than another enumeration. See that
    # module's own docstring for the staleness trade-off this accepts.
    population = (
        population_memo.get_or_derive(
            identity, materialise_graph, deadline=deadline, join_inflight=join_inflight
        )
        if population_memo is not None
        else options_population(identity, materialise_graph, deadline=deadline)
    )
    role_names_by_axis = _declared_axis_roles(population)
    _check_entity_role_admission(identity, role_names_by_axis)
    if role_names_by_axis:
        refuse_wrong_kind(
            identity.entities,
            read_types=graph_types(graph),
            family=flow_family(
                index=get_index(),
                population_type=(
                    identity.shape.population_type
                    if isinstance(identity.shape, IdentityShapeSchema)
                    else None
                ),
                stat_vars=population.stat_var_dcids,
                members=(),
            ),
            pointer_prefix="/identity",
        )
    _check_role_target(req, role_names_by_axis)

    return ResolvedOptionsTarget(
        identity=identity,
        population=population,
        anchors=anchors,
        facets=identity.facets,
        requested_window=_requested_window_from_identity(identity.requested_window),
        role_names_by_axis=role_names_by_axis,
    )
