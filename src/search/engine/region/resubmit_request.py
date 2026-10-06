"""Translate a posted identity into engine request structures and guards."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from materialise import facet_axis_value
from search.engine.axis import classify_axis
from search.engine.clients.graph import EngineGraphClient, SubjectsCoverage
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import EngineInputError, GraphInfraError
from search.engine.extract import DateRequest
from search.engine.region.facets import _facet_source_dcid, binding_admits_value
from search.engine.region.grounding import (
    reject_reversed_window,
    requested_window_from_date_request,
)
from search.engine.region.refs import provenance_labels
from search.engine.region.result import FacetAxisFilter
from search.engine.retrieval.facts import read_constraints
from search.engine.spec_build import NamedGroup
from search.engine.unit.refs import expressible as _unit_request_expressible
from search.engine.unit.refs import satisfies as _unit_request_satisfies
from search.models import (
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
        SlotValue(ref=GraphRef(dcid=dcid, label=dcid), value_kind=value_kind) for dcid in dcids
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
                property=GraphRef(dcid=slot.property, label=slot.property),
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
    """Return the posted `facets.provenance` refs, the one source of a resubmit's
    provenance pin; `[]` when provenance is unbound."""
    binding = (facets or {}).get("provenance")
    if isinstance(binding, IdentityBindingValue):
        return [binding.ref] if binding.ref else []
    if isinstance(binding, IdentityBindingSet):
        return list(dict.fromkeys(binding.refs))
    return []


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
    values = [SlotValue(ref=GraphRef(dcid=d, label=d), value_kind="source") for d in dcids]
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

    Provenance uses `source_dcids`; a `provenance` key in `identity.facets` is ignored. Each
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
            detail=detail,
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
        SlotValue(ref=GraphRef(dcid=d, label=labels.get(d, d)), value_kind="source")
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


def resubmit_named_groups(
    inp: ResubmitIdentityPayload, *, graph: EngineGraphClient, deadline: float | None
) -> tuple[NamedGroup, ...]:
    """The groups posted beside the identity that the graph confirms.

    A group is kept only when it posts a dcid and the graph's members of that node
    (`<-containedInPlace`) are the identity's entities on the role, and it takes the graph's
    label for the node, as a first answer does (`place_roles.request_named_groups`). The
    posted label stands only where the graph gives none. A group posting no dcid drops, since
    nothing ties its label to the role's entities, and so does a group on a role the identity
    binds no entity on. All posted dcids are read in one members batch, and the
    membership-confirmed ones in one labels batch. Each dropped group is logged at info with
    the reason. A `GraphInfraError` on either read drops the groups it covers, logged at
    warning and noted as a lost read (`degraded_reads.note_degraded_read`), and the role's
    entities still bind.
    """
    kept: list[tuple[str, EntityGroup, frozenset[str]]] = []
    entities = inp.identity.entities
    for role, group in (inp.entity_groups or {}).items():
        if group.dcid is None:
            logger.info(
                "resubmit dropped group %r on role %s: it names no graph node", group.label, role
            )
            continue
        if role not in entities:
            logger.info(
                "resubmit dropped group %s on role %s: the identity binds no entity there",
                group.dcid,
                role,
            )
            continue
        kept.append((group.dcid, group, frozenset(entities[role])))
    if not kept:
        return ()
    dcids = sorted({dcid for dcid, _, _ in kept})
    try:
        pages = graph.node_subjects_many(
            dcids,
            property="containedInPlace",
            max_results=settings().max_child_places,
            deadline=deadline,
        )
    except GraphInfraError as exc:
        logger.warning("resubmit dropped groups %s: members read failed: %s", dcids, exc)
        note_degraded_read("named group members")
        return ()
    verified: list[tuple[str, EntityGroup, frozenset[str]]] = []
    for dcid, group, members in kept:
        page = pages.get(dcid)
        if page is None:
            continue
        if page.coverage is SubjectsCoverage.complete and frozenset(page.dcids) == members:
            verified.append((dcid, group, members))
            continue
        logger.info(
            "resubmit dropped group %s: %s (graph members %d, role entities %d)",
            dcid,
            "members differ"
            if page.coverage is SubjectsCoverage.complete
            else f"coverage {page.coverage}",
            len(page.dcids),
            len(members),
        )
    if not verified:
        return ()
    confirmed = sorted({dcid for dcid, _, _ in verified})
    try:
        labels = graph.node_labels_batch(confirmed)
    except GraphInfraError as exc:
        logger.warning("resubmit dropped groups %s: labels read failed: %s", confirmed, exc)
        note_degraded_read("named group labels")
        return ()
    return tuple(
        NamedGroup(label=labels.get(dcid, group.label), dcid=dcid, member_dcids=members)
        for dcid, group, members in verified
    )
