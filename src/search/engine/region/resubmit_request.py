"""Translate a posted identity into engine request structures and guards."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager

from code_index.composites import DECLARED_COMPOSITES
from code_index.members import pinned_refs
from code_index.shared import get_index
from dfo_platform.executor import submit_with_context
from dfo_platform.roles import role_key_axis
from entity_resolution import shared_membership_policy
from entity_resolution.membership import Membership, MembershipPolicy, UnknownMembership
from entity_resolution.regions import replacing_group_id
from entity_resolution.shared import shared_memberships
from materialise import facet_axis_value
from origin_catalogue import scoped_reads
from search.engine.axis import classify_axis
from search.engine.clients.graph import EngineGraphClient, SubjectsCoverage
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import EngineInputError, GraphInfraError
from search.engine.extract import DateRequest
from search.engine.place import expand_places
from search.engine.region.facets import _facet_source_dcid, binding_admits_value
from search.engine.region.grounding import (
    reject_reversed_window,
    requested_window_from_date_request,
)
from search.engine.region.refs import provenance_labels
from search.engine.region.result import FacetAxisFilter
from search.engine.region.whole_totals import identity_years, whole_reading
from search.engine.retrieval.bound_values import identity_dataset
from search.engine.retrieval.facts import read_constraints
from search.engine.spec_build import NamedGroup
from search.engine.unit.refs import expressible as _unit_request_expressible
from search.engine.unit.refs import satisfies as _unit_request_satisfies
from search.models import (
    BaseIdentity,
    Binding,
    BindingSet,
    BindingUnbound,
    BindingValue,
    EntityGroup,
    FacetBasis,
    GraphRef,
    IdentityBinding,
    IdentityBindingSet,
    IdentityBindingUnbound,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityShape,
    IdentityShapeDenominatorRef,
    IdentityShapeDenominatorUnresolved,
    IdentityShapePin,
    IdentitySlot,
    IdentityWindow,
    IdentityWindowAll,
    IdentityWindowLatest,
    IdentityWindowRange,
    RequestedWindow,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
    ValueKind,
)
from search.scope.region import without_residuals

logger = logging.getLogger(__name__)


def _describe_shape(shape: IdentityShape) -> str:
    """Render a described shape for diagnostic text only."""
    if isinstance(shape, IdentityShapePin):
        return shape.id
    parts = [p for p in (shape.population_type, shape.measured_property, shape.stat_type) if p]
    if shape.measurement_qualifier:
        parts.append(shape.measurement_qualifier)
    denom = shape.measurement_denominator
    if isinstance(denom, IdentityShapeDenominatorRef):
        parts.append(f"per {denom.ref}")
    elif isinstance(denom, IdentityShapeDenominatorUnresolved):
        parts.append("per unresolved")
    return " ".join(parts) if parts else "shape"


def _binding_from_dcids(dcids: list[str], value_kind: ValueKind) -> Binding:
    """Build a labelled wire binding from canonical identity dcids."""
    values = [
        SlotValue(ref=GraphRef(ref=dcid, label=dcid), value_kind=value_kind) for dcid in dcids
    ]
    return BindingValue(value=values[0]) if len(values) == 1 else BindingSet(values=values)


def _binding_from_identity(binding: IdentityBinding, value_kind: ValueKind) -> Binding:
    """Adapt an identity binding to its equivalent wire binding."""
    if isinstance(binding, IdentityBindingValue):
        return _binding_from_dcids([binding.ref], value_kind)
    if isinstance(binding, IdentityBindingSet):
        return _binding_from_dcids(binding.refs, value_kind)
    if isinstance(binding, IdentityBindingUnbound):
        return BindingUnbound()
    raise ValueError(f"unknown IdentityBinding type: {type(binding)!r}")  # pragma: no cover


def _identity_slots_to_wire(
    identity_slots: list[IdentitySlot], confirmed_arcs: Mapping[str, dict]
) -> list[Slot]:
    """Rebuild slot axes from the elected members' confirmed arcs without another graph
    read."""
    observed: dict[str, list[str]] = {}
    for arcs in confirmed_arcs.values():
        for prop, value in read_constraints(arcs).items():
            observed.setdefault(prop, []).append(value)
    return [
        Slot(
            key=SlotKey(
                axis=classify_axis(slot.property, observed.get(slot.property, [])),
                property=GraphRef(ref=slot.property, label=slot.property),
                label=slot.property,
            ),
            binding=_binding_from_identity(slot.binding, "enum_value"),
        )
        for slot in identity_slots
    ]


def _requested_window_from_identity(window: IdentityWindow) -> RequestedWindow | None:
    """Translate the always-present identity window to engine request form."""
    if isinstance(window, IdentityWindowAll):
        return None
    if isinstance(window, IdentityWindowLatest):
        return "latest"
    if isinstance(window, IdentityWindowRange):
        return TimeWindow(start_year=window.start_year, end_year=window.end_year)
    raise ValueError(f"unknown IdentityWindow kind: {type(window)!r}")  # pragma: no cover


def _source_dcids_from_facets_provenance(
    facets: dict[str, IdentityFacetBinding] | None,
) -> list[str]:
    """Return the posted `facets.provenance` refs (`code_index.members.pinned_refs`), the
    one source of a resubmit's provenance pin; `[]` when provenance is unbound."""
    binding = (facets or {}).get("provenance")
    if binding is None:
        return []
    return list(pinned_refs({"provenance": binding.model_dump(mode="json")}))


def pinned_provenance_scope(
    facets: dict[str, IdentityFacetBinding] | None,
) -> AbstractContextManager[object]:
    """Bind the request's catalogue narrowed to the posted `facets.provenance` refs
    (`origin_catalogue.scoped_reads`), so a pinned source is admitted before DFO precedence
    and a base pin serves at a place where a DFO table also holds data. Binds nothing when
    provenance is unbound, or binds a declared composite, whose reads are those of the
    identity its option was set on (`code_index.composites.table_identity`), which pins no
    provenance."""
    binding = (facets or {}).get("provenance")
    if isinstance(binding, IdentityBindingValue) and binding.ref in DECLARED_COMPOSITES:
        return scoped_reads(None)
    return scoped_reads(frozenset(_source_dcids_from_facets_provenance(facets)) or None)


def _accounting_slots_with_translated_provenance(
    slots: list[Slot],
    facets: dict[str, IdentityFacetBinding] | None,
) -> list[Slot]:
    """Add a placeholder source slot for provenance-facet edit accounting.

    The request validator forbids a simultaneous source slot. This synthetic
    slot is used only to record that the client re-sent provenance.
    """
    dcids = _source_dcids_from_facets_provenance(facets)
    if not dcids:
        return slots
    values = [SlotValue(ref=GraphRef(ref=d, label=d), value_kind="source") for d in dcids]
    binding = BindingValue(value=values[0]) if len(values) == 1 else BindingSet(values=values)
    return [
        *slots,
        Slot(key=SlotKey(axis="source", property=None, label="data source"), binding=binding),
    ]


def _facet_filter_from_bindings(
    graph: EngineGraphClient,
    source_dcids: list[str],
    facets: dict[str, IdentityFacetBinding] | None,
    *,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> Mapping[str, FacetAxisFilter] | None:
    """Build the request's facet filter from the posted source, method, and unit bindings.

    Provenance uses `source_dcids`; a `provenance` key in `identity.facets` is ignored. The
    resubmit already reads under `pinned_provenance_scope`, so the provenance filter keeps
    what admission kept. Each
    other axis tests its value through `_facet_binding_admits`, the one matching rule every
    filtered election applies, so a null ref admits the facets that declare no value on the
    axis. Elected units are supplied when per-StatVar narrowing replaces the unit axis, which
    then adds no unit filter here. A unit request reads the graph's currency units within
    the absolute monotonic `deadline`. Return `None` when no axis is constrained.
    """
    result: dict[str, FacetAxisFilter] = {}
    if source_dcids:
        allowed_sources = frozenset(source_dcids)
        result["provenance"] = FacetAxisFilter(
            predicate=lambda f: _facet_source_dcid(f) in allowed_sources,
        )
    for axis in ("measurementMethod", "unit"):
        admits = _facet_binding_admits(
            graph, (facets or {}).get(axis), elected_units=elected_units, deadline=deadline
        )
        if admits is not None:
            result[axis] = FacetAxisFilter(
                predicate=lambda f, admits=admits, axis=axis: admits(facet_axis_value(f, axis))
            )
    return result or None


def _facet_binding_admits(
    graph: EngineGraphClient,
    binding: IdentityFacetBinding | None,
    *,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> Callable[[str | None], bool] | None:
    """The matching rule for one facet binding over an axis value, `None` for a facet that
    carries no value on the axis.

    Ref, set and null-ref bindings apply `binding_admits_value`. Request-form units use
    `satisfies`, and a facet without a unit fails them. With `elected_units`, unit admission
    is handled per StatVar elsewhere, so return `None`. Unbound bindings also return `None`.
    """
    if isinstance(binding, IdentityBindingValue | IdentityBindingSet | IdentityFacetValueAbsent):
        return lambda value, binding=binding: binding_admits_value(binding, value)
    if isinstance(binding, IdentityFacetValueRequest):
        if elected_units is not None:
            return None
        request = binding.request
        return lambda value, request=request: (
            value is not None and _unit_request_satisfies(graph, request, value, deadline=deadline)
        )
    return None


def _unit_request_basis_from_facets(
    facets: dict[str, IdentityFacetBinding] | None,
) -> dict[str, FacetBasis] | None:
    """Return `bound` for request-form unit members and `None` for other bindings.

    The client supplies the full currency/prices/base_year request, so each
    member is marked `bound`.
    Include `base_year` only when the request supplies it.
    """
    binding = (facets or {}).get("unit")
    if not isinstance(binding, IdentityFacetValueRequest):
        return None
    request = binding.request
    basis: dict[str, FacetBasis] = {"currency": "bound", "prices": "bound"}
    if request.base_year is not None:
        basis["base_year"] = "bound"
    return basis


def _reject_inexpressible_unit(facets: dict[str, IdentityFacetBinding] | None) -> None:
    """Reject an inexpressible request-form unit before other graph reads.

    The refusal is `facet_binding_unsupported` (501), the same as the later facet check.
    The check reads nothing from the graph. Ref and set bindings need no check.
    """
    unit_binding = (facets or {}).get("unit")
    if isinstance(unit_binding, IdentityFacetValueRequest) and not _unit_request_expressible(
        unit_binding.request
    ):
        detail = (
            "the requested currency is neither an ISO 4217 code nor a role shape, "
            "so no conversion can express it."
        )
        raise EngineInputError(
            detail,
            code="facet_binding_unsupported",
            status=501,
            pointer="/input/identity/facets/unit",
        )


def _rebind_source_slot(
    slots: list[Slot],
    source_dcids: list[str],
    graph: EngineGraphClient,
) -> list[Slot]:
    """Apply a posted source constraint after grounding, leaving facet selection unchanged.

    Empty `source_dcids` leaves the source slot unbound. Labels come from the origin
    catalogue, else the graph, using the DCID when no name is available.
    """
    if not source_dcids:
        return slots
    labels = provenance_labels(source_dcids, graph)
    values = [
        SlotValue(ref=GraphRef(ref=d, label=labels.get(d, d)), value_kind="source")
        for d in source_dcids
    ]
    binding = BindingValue(value=values[0]) if len(values) == 1 else BindingSet(values=values)
    return [
        slot.model_copy(update={"binding": binding}) if slot.key.axis == "source" else slot
        for slot in slots
    ]


def _date_request_for_resubmit(inp: ResubmitIdentityPayload) -> DateRequest | None:
    """Build the materialisation window from the identity's requested window.

    An explicit `latest` sentinel resolves against the current latest year on
    each resubmit. A concrete range is applied directly.
    """
    requested_window = _requested_window_from_identity(inp.identity.requested_window)
    if requested_window == "latest":
        return DateRequest(window=None, latest=True)
    if requested_window is not None:
        return DateRequest(window=requested_window, latest=False)
    return None


def _reject_reversed_resubmit_window(
    inp: ResubmitIdentityPayload,
    date_request: DateRequest | None,
) -> None:
    """Reject reversed windows with `request_invalid` before materialisation or graph reads."""
    reject_reversed_window(_requested_window_from_identity(inp.identity.requested_window))
    if date_request is not None:
        reject_reversed_window(date_request.window)


def _requested_window_for_id(date_request: DateRequest | None) -> RequestedWindow | None:
    """Return the identity's window for both materialisation and `spec_id` construction: the
    window `date_request` reads, the posted one (`_date_request_for_resubmit`) or the period
    a unit request narrows it to (`convertibility.unit_request_period`)."""
    return requested_window_from_date_request(date_request)


def _graph_members(
    dcids: list[str],
    *,
    graph: EngineGraphClient,
    membership: MembershipPolicy,
    deadline: float | None,
) -> dict[str, frozenset[str] | None]:
    """The members of each node in `dcids` the membership policy does not hold, `None` where
    there is no complete answer.

    A region the policy omits (`MembershipPolicy.holds_region`, such as Antarctica) has no
    members (`place.expand_places`), and graph containment is not read for it. Any other node
    has its `<-containedInPlace` subjects, read in one batch, whatever its graph type."""
    expansions = expand_places(
        [dcid for dcid in dcids if membership.holds_region(dcid)], membership=membership
    )
    members: dict[str, frozenset[str] | None] = {
        dcid: None if expansion is None else frozenset(expansion.children)
        for dcid, expansion in expansions.items()
    }
    if rest := [dcid for dcid in dcids if dcid not in members]:
        pages = graph.node_subjects_many(
            rest,
            property="containedInPlace",
            max_results=settings().max_child_places,
            deadline=deadline,
        )
        for dcid in rest:
            page = pages.get(dcid)
            complete = page is not None and page.coverage is SubjectsCoverage.complete
            members[dcid] = frozenset(page.dcids) if page is not None and complete else None
    return members


def _policy_members(
    dcids: list[str], membership: MembershipPolicy
) -> dict[str, frozenset[str] | None]:
    """The members the membership policy holds today under the default projection for each
    dcid in `dcids` that names one of its groups or regions, `None` where the authority
    records no such projection. The policy answers without a graph read, so a group whose
    node carries no member arcs, such as `GroupOfSeven`, is still confirmed, and a region
    holds the countries the Graph API's resolve and members doors give."""
    members = {dcid: membership.members(dcid).members for dcid in dcids if dcid in membership}
    return {
        dcid: frozenset(held) if isinstance(held, tuple) else None for dcid, held in members.items()
    }


def _refuse_retired_group_id(group: EntityGroup, role: str) -> None:
    """Refuse a posted group under a resolvekit id the group's graph node replaced, which
    would otherwise miss the total and members keyed by the current id.

    Raises:
        EngineInputError: `group.ref` or `group.group_id` is a retired group id.
    """
    for field, posted in (("ref", group.ref), ("group_id", group.group_id)):
        if posted is None or (current := replacing_group_id(posted)) is None:
            continue
        raise EngineInputError(
            f"group id {posted!r} is retired; post {current!r}",
            code="request_invalid",
            status=422,
            pointer=f"/input/entity_groups/{role}/{field}",
        )


def _registry_group(group: EntityGroup, role: str) -> Membership | None:
    """The membership registry's group `group.group_id` names, `None` when it posts none.

    Raises:
        EngineInputError: the registry holds no group with that id.
    """
    if group.group_id is None:
        return None
    try:
        return shared_memberships().groups.get(group.group_id)
    except UnknownMembership:
        raise EngineInputError(
            f"group id {group.group_id!r} names no group the resolver knows",
            code="request_invalid",
            status=422,
            pointer=f"/input/entity_groups/{role}/group_id",
        ) from None


def _registry_named_group(
    group: EntityGroup, registry_group: Membership, role: str, role_entities: frozenset[str]
) -> NamedGroup | None:
    """`group` read as the members of its registry group, with the total the code index
    declares for that group (`CodeIndex.totals_by_group`) kept aside, as a first answer keeps
    it (`ResolvedSpec.group_totals`). `None` when the registry's members under the default
    projection are not the role's entities."""
    members = registry_group.members()
    if not isinstance(members, tuple) or frozenset(members) != role_entities:
        logger.info(
            "resubmit dropped group %s on role %s: its registry members differ from the "
            "role's entities",
            group.group_id,
            role,
        )
        return None
    total = get_index().totals_by_group.get(registry_group.ref)
    return NamedGroup(
        label=group.label,
        dcid=None,
        member_dcids=role_entities,
        total=None if total is None else total.dcid,
        group_id=group.group_id,
    )


def resubmit_named_groups(
    inp: ResubmitIdentityPayload, *, graph: EngineGraphClient, deadline: float | None
) -> tuple[NamedGroup, ...]:
    """The groups posted beside the identity that the graph or the code index confirms.

    A group whose role binds the one party the code index declares the total of the posted
    dcid's group (`_bound_total`) is read as that total, whether or not it posts `total`, so
    a client that posts back only a group's label and dcid gets the group it was served. A
    posted `total` that is not the one so bound drops the group. Any other group is read as
    its members, and is kept only when it posts a dcid and that dcid's members are the
    identity's entities on the role: the membership policy's members for a group or region it
    holds (`_policy_members`), the graph's members of the node otherwise (`_graph_members`). A
    group posted under a retired group id refuses (`_refuse_retired_group_id`). A kept group with a dcid
    takes the graph's type and label for the node, as a first answer does
    (`place_roles.request_named_groups`). The posted type is ignored, and the posted label
    stands only where the graph gives none, or for a group that posts no dcid, as a resolver
    group does ("G7"). Such a group is kept when its total ties it to the role, or when it
    posts a `group_id` the membership registry holds and the registry's members are the
    role's entities, which restores the total the code index declares for it
    (`_registry_named_group`). A posted `group_id` the registry does not hold refuses
    (`_registry_group`). Any other group posting no dcid drops, since nothing ties its label
    to the role's entities, and so does a group on a role the identity binds no entity on. The posted dcids' types and labels are read side
    by side, one batch each, and their members after the types. Each dropped group is logged
    at info with the reason. A `GraphInfraError` on any of the reads drops the groups it
    covers, logged at warning and noted as a lost read (`degraded_reads.note_degraded_read`),
    and the role's entities still bind.

    A role whose entities the identity's dataset reads as the whole population of its side
    (`_whole_named_groups`) carries that population, whatever group the request posts on it,
    so a first answer's whole total survives a resubmit that posts no group.
    """
    kept: list[tuple[str, EntityGroup, frozenset[str], str | None]] = []
    whole = _whole_named_groups(inp.identity)
    named: list[NamedGroup] = list(whole.values())
    entities = inp.identity.entities
    for role, group in (inp.entity_groups or {}).items():
        if role in whole:
            continue
        _refuse_retired_group_id(group, role)
        registry_group = _registry_group(group, role)
        if role not in entities:
            logger.info(
                "resubmit dropped group %s on role %s: the identity binds no entity there",
                group.ref or repr(group.label),
                role,
            )
            continue
        role_entities = frozenset(entities[role])
        # A summed region binds its residual buckets beside its members; the group stands
        # for the members (`scope.region.is_residual`).
        role_members = frozenset(without_residuals(role_entities, get_index())) or role_entities
        total = _bound_total(group, role, role_entities)
        if group.total is not None and total is None:
            logger.info(
                "resubmit dropped group %r on role %s: the code index does not tie its total "
                "%s to the group on the role's flow axis, or the role binds more than it",
                group.label,
                role,
                group.total,
            )
            continue
        if group.ref is None:
            if total is not None:
                named.append(
                    NamedGroup(
                        label=group.label,
                        dcid=None,
                        member_dcids=role_entities,
                        total=total,
                        group_id=group.group_id,
                    )
                )
            elif registry_group is not None:
                if restored := _registry_named_group(group, registry_group, role, role_members):
                    named.append(restored)
            else:
                logger.info(
                    "resubmit dropped group %r on role %s: it names no graph node",
                    group.label,
                    role,
                )
            continue
        kept.append((group.ref, group, role_entities if total is not None else role_members, total))
    if not kept:
        return tuple(named)
    dcids = sorted({dcid for dcid, _, _, _ in kept})
    read_as_members = sorted({dcid for dcid, _, _, total in kept if total is None})
    membership = shared_membership_policy()
    group_members = _policy_members(read_as_members, membership)
    member_dcids = [dcid for dcid in read_as_members if dcid not in group_members]
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="group-labels") as pool:
        labels_read = submit_with_context(pool, graph.node_labels_batch, dcids)
        try:
            types = graph.node_types_batch(dcids)
            if member_dcids:
                group_members |= _graph_members(
                    member_dcids, graph=graph, membership=membership, deadline=deadline
                )
        except GraphInfraError as exc:
            logger.warning("resubmit dropped groups %s: members read failed: %s", dcids, exc)
            note_degraded_read("named group members")
            return tuple(named)
        try:
            labels = labels_read.result()
        except GraphInfraError as exc:
            logger.warning("resubmit dropped groups %s: labels read failed: %s", dcids, exc)
            note_degraded_read("named group labels")
            return tuple(named)
    for dcid, group, members, total in kept:
        held = members if total is not None else group_members[dcid]
        if held == members:
            named.append(
                NamedGroup(
                    label=labels.get(dcid, group.label),
                    dcid=dcid,
                    member_dcids=members,
                    entity_type=types.get(dcid),
                    total=total,
                )
            )
            continue
        logger.info(
            "resubmit dropped group %s: %s (role entities %d)",
            dcid,
            "the members read is incomplete"
            if held is None
            else f"members differ (held members {len(held)})",
            len(members),
        )
    return tuple(named)


def _whole_named_groups(identity: BaseIdentity) -> dict[str, NamedGroup]:
    """The group of each flow role whose entities the dataset elected for `identity`
    (`bound_values.identity_dataset`) reads as the whole population of the role's side over
    its requested window (`whole_totals.whole_reading`), by role key: one whole total ("ODA
    to Ethiopia" binds `DACOfficialDonors`) or the members of one whole total set. Empty for
    an identity whose dataset the code index does not elect."""
    dataset = identity_dataset(identity)
    if dataset is None:
        return {}
    years = identity_years(identity.requested_window)
    groups: dict[str, NamedGroup] = {}
    for role, dcids in identity.entities.items():
        reading = whole_reading(dcids, datasets=(dataset,), years=years)
        if reading is not None and reading.axis == role_key_axis(role):
            groups[role] = reading.named_group()
    return groups


def _bound_total(group: EntityGroup, role: str, role_entities: frozenset[str]) -> str | None:
    """The one party `role` binds when the code index declares it the total
    (`PartyRow.totals`) of the group the posted `group.ref` or `group.group_id` names, or of
    any group when it posts neither, published on `role`'s flow axis (`PartyRow.axes`), and
    `group` posts that party as its `total` or no total. `None` otherwise, so a subject role,
    which names no flow axis, binds no total."""
    if len(role_entities) != 1:
        return None
    (party,) = role_entities
    row = get_index().parties.get(party)
    posted = group.ref or group.group_id
    if row is None or row.totals is None or posted not in (None, row.totals):
        return None
    if role_key_axis(role) not in row.axes:
        return None
    return party if group.total in (None, party) else None
