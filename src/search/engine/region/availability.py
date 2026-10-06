"""Prune empty sibling shapes before election using scoped availability evidence.

Denominator-resolution pairs apply at any place scope. Rate-head pairs require a
named place and matching cohort structure.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence

from search.engine.clients.graph import EngineGraphClient
from search.engine.qualifier_match import QualifierMatch, QualifierReadings
from search.engine.quantity import DenominatorKind, QuantityRequirement
from search.engine.region.alternate_guard import inadmissible_reason, query_pins
from search.engine.region.rate_head import _rate_head, _rate_head_compatible
from search.engine.shape import ShapeDraft, is_role_bearing

logger = logging.getLogger(__name__)

type _ShapeGroupKey = tuple[str, str, str, str, str | None, str | None, bool]


def _shape_group_key(shape: ShapeDraft) -> _ShapeGroupKey:
    """Return a collision-free key for one derived shape group."""
    return (
        shape.shape_id,
        shape.pop_type_dcid,
        shape.meas_prop_dcid,
        shape.stat_type_dcid,
        shape.meas_qual_dcid,
        shape.meas_denom_dcid,
        shape.meas_denom_unresolved,
    )


def _availability_sibling_class_key(
    shape: ShapeDraft,
) -> tuple[str, str, str, str | None, frozenset[str]]:
    """Return the shared structural key, excluding denominator resolution.

    Constraint slots key on property dcid alone because axis classification
    can vary for the same property across shape groups.
    """
    constraint_property_dcids = frozenset(
        key.property_dcid for key in shape.slot_keys if key.property_dcid is not None
    )
    return (
        shape.pop_type_dcid,
        shape.meas_prop_dcid,
        shape.stat_type_dcid,
        shape.meas_qual_dcid,
        constraint_property_dcids,
    )


def _are_availability_siblings(a: ShapeDraft, b: ShapeDraft) -> bool:
    """Return whether shapes share all fields except denominator resolution.

    One side must have an unresolved denominator arc and the other a resolved
    dcid. A missing denominator arc is a different concept.
    """
    if _availability_sibling_class_key(a) != _availability_sibling_class_key(b):
        return False
    if a.meas_denom_unresolved and not b.meas_denom_unresolved and b.meas_denom_dcid is not None:
        return True
    return b.meas_denom_unresolved and not a.meas_denom_unresolved and a.meas_denom_dcid is not None


def _covers_fewer_places(shape: ShapeDraft, other: ShapeDraft) -> bool:
    """Return whether `shape` has data at a proper subset of the probed places `other`
    has data at. False when either shape's places are unknown."""
    if shape.available_places is None or other.available_places is None:
        return False
    return shape.available_places < other.available_places


def _drop_shapes(shapes: list[ShapeDraft], to_drop: set[_ShapeGroupKey]) -> list[ShapeDraft]:
    """Remove the named shape groups while keeping at least one candidate."""
    if not to_drop:
        return shapes
    survivors = [s for s in shapes if _shape_group_key(s) not in to_drop]
    return survivors or shapes


def _measure_head_constraint_value_map(shape: ShapeDraft) -> dict[str, frozenset[str]] | None:
    """Return slot values keyed by property, merging axes with the same property.

    Axis classification can vary by shape group. Missing taxonomy means
    structural compatibility is unknown. An empty map would wrongly permit a
    match.
    """
    if shape.slot_taxonomy is None:
        return None
    values_by_property: dict[str, set[str]] = {}
    for key, values in shape.slot_taxonomy.items():
        _, _, property_dcid = key.partition(":")
        values_by_property.setdefault(property_dcid, set()).update(values)
    return {
        property_dcid: frozenset(values) for property_dcid, values in values_by_property.items()
    }


def _measure_head_shapes_structurally_agree(a: ShapeDraft, b: ShapeDraft) -> bool:
    """Return whether population type and constraint values identify one cohort.

    This pairs a rate with its count but separates different cohorts. Missing
    taxonomy fails closed because the prune would otherwise discard a shape
    without evidence.
    """
    if a.pop_type_dcid != b.pop_type_dcid:
        return False
    a_constraints = _measure_head_constraint_value_map(a)
    b_constraints = _measure_head_constraint_value_map(b)
    return (
        a_constraints is not None and b_constraints is not None and a_constraints == b_constraints
    )


def _shapes_are_measure_head_siblings(a: ShapeDraft, b: ShapeDraft) -> bool:
    """Return whether labels share a rate head and structures identify one cohort.

    A head is ``<concept> rate`` or ``<concept> percentage``. The other label
    must use the same term or a close inflection. Equal population type and
    constraint values pair a rate with its
    cohort's count, while separating values such as under-five and under-14.
    Structurally identical shapes with different unrepresented cohorts remain
    indistinguishable. Callers enforce the named-place restriction.
    """
    head = _rate_head(a.label)
    if head is None or not _rate_head_compatible(b, head):
        head = _rate_head(b.label)
        if head is None or not _rate_head_compatible(a, head):
            return False
    return _measure_head_shapes_structurally_agree(a, b)


def _is_availability_pair_member(
    shape: ShapeDraft, shapes: list[ShapeDraft], *, named_place: bool
) -> bool:
    """Return whether an active pairing rule includes this shape.

    The probe and prune share this predicate. Denominator pairs apply at any
    scope. Rate-head pairs require ``named_place``.
    """
    return any(
        other is not shape
        and not is_role_bearing(other)
        and (
            _are_availability_siblings(shape, other)
            or (named_place and _shapes_are_measure_head_siblings(shape, other))
        )
        for other in shapes
    )


def _prune_availability_siblings(
    shapes: list[ShapeDraft],
    *,
    named_place: bool,
    quantity: QuantityRequirement,
    denominator_kinds: Mapping[str, DenominatorKind],
    variable: str = "",
    qualifier_matches: Sequence[QualifierMatch] = (),
    qualifier_readings: Sequence[QualifierReadings] = (),
) -> tuple[list[ShapeDraft], list[tuple[ShapeDraft, ShapeDraft]]]:
    """Drop an empty shape when a valid sibling has observations.

    Pairs share denominator resolution state, or a rate head and the same
    cohort structure. Rate-head evidence applies only to named places.
    entity-less queries use denominator pairs alone.


    A denominator-resolution pair with data on both sides drops the shape whose
    data covers a proper subset of its sibling's probed places
    (``_covers_fewer_places``): nominal GDP per ``Count_Person`` holds one of the
    six sample countries, and the ``PerCapita`` spelling holds all six.

    Unknown or equal availability leaves both shapes intact. Role-bearing
    shapes use donor entities and are excluded.

    Pairing uses structural fields alone. Before a drop, the survivor must be a reading
    of the dropped shape (`alternate_guard.inadmissible_reason`) under the query's
    wording, its quantity requirement (`quantity`, judged with the kinds of the leg's
    StatVar denominators, `denominator_kinds`, as bind and materialise judge it), its
    qualifier matches and its two-reading spans. No bind has run, so every
    member of the dropped shape is its reading. Empty query text allows structure-only pairing. Return measure-head
    drops as ``(dropped, survivor)`` pairs for the caller's
    ``ELECTED_SHAPE_NO_DATA`` warning. Denominator-resolution drops are omitted.
    If every shape qualifies for removal, restore the original list.
    """
    pins = query_pins(
        variable,
        quantity=quantity,
        denominator_kinds=denominator_kinds,
        qualifier_matches=qualifier_matches,
        qualifier_readings=qualifier_readings,
    )
    to_drop: set[_ShapeGroupKey] = set()
    measure_head_substitutions: list[tuple[ShapeDraft, ShapeDraft]] = []
    for shape in shapes:
        if is_role_bearing(shape):
            continue
        if shape.place_available is not False:
            # Two denominator spellings of one measure, both with data: the one whose
            # data covers only some of the other's probed places leaves the election.
            if any(
                not is_role_bearing(other)
                and _are_availability_siblings(shape, other)
                and _covers_fewer_places(shape, other)
                and inadmissible_reason(pins, shape, other) is None
                for other in shapes
            ):
                to_drop.add(_shape_group_key(shape))
            continue
        survivor = next(
            (
                other
                for other in shapes
                if not is_role_bearing(other)
                and other.place_available
                and (
                    _are_availability_siblings(shape, other)
                    or (named_place and _shapes_are_measure_head_siblings(shape, other))
                )
                and inadmissible_reason(pins, shape, other) is None
            ),
            None,
        )
        if survivor is None:
            continue
        to_drop.add(_shape_group_key(shape))
        if named_place and _shapes_are_measure_head_siblings(shape, survivor):
            measure_head_substitutions.append((shape, survivor))

    survivors = _drop_shapes(shapes, to_drop)
    if survivors is shapes:
        # The drop-everything guard restored the full set, so no warning is due.
        return survivors, []
    return survivors, measure_head_substitutions


# Entity-less probes sample one country per inhabited continent: USA, BRA, DEU,
# KEN, IND, and AUS. Any observation marks a shape available. This sample can
# miss coverage elsewhere, so a drop requires a confirmed-empty side and data
# on its paired shape. No positive evidence on either side prunes none. Rate-head
# pairing stays off because world-only series may have no country data.
_ENTITYLESS_AVAILABILITY_PROBE_PLACES: tuple[str, ...] = (
    "country/USA",
    "country/BRA",
    "country/DEU",
    "country/KEN",
    "country/IND",
    "country/AUS",
)

# Five members suffice because one positive observation marks a shape available.
# Probing every member up to SEARCH_MAX_CHILD_PLACES only enlarges the batch.
_PLACE_AVAILABILITY_PROBE_SAMPLE = 5


def _sibling_probe_places(
    named_place_dcids: list[str] | None,
    place_children_by_name: dict[str, list[str]],
    resolved_entity_names: dict[str, str],
) -> list[str]:
    """Build the probe scope from named places and place-set members.

    Sample each expansion up to the configured cap. Use the place-set dcid
    when expansion is empty. The map uses surface names, so a childless entry
    needs a matching resolved dcid or raises ``AssertionError``. Fall back to
    the fixed sample only when no place resolves. Deduplication keeps first-seen
    order.
    """
    places = list(named_place_dcids or [])
    for name, children in place_children_by_name.items():
        if children:
            places.extend(children[:_PLACE_AVAILABILITY_PROBE_SAMPLE])
            continue
        dcid = resolved_entity_names.get(name)
        if dcid is None:
            raise AssertionError(f"no resolved dcid for place-set name {name!r}")
        places.append(dcid)
    if not places:
        places = list(_ENTITYLESS_AVAILABILITY_PROBE_PLACES)
    return list(dict.fromkeys(places))


def _resolve_sibling_availability(
    shapes: list[ShapeDraft],
    *,
    graph: EngineGraphClient,
    probe_places: list[str],
    named_place: bool,
) -> tuple[list[ShapeDraft], bool]:
    """Refresh pair candidates against the query's full place scope.

    Only shapes in an active pair are probed. One batched call covers their
    confirmed StatVars. Any observation at any probe place marks a shape
    available, and the shape records the probe places that hold one
    (``available_places``). ``named_place`` enables rate-head pairing only for a named
    place. Return ``(shapes, probe_failed)``. On graph error, reset every pair
    candidate to unknown so the prune cannot use narrower stale evidence. Shapes
    outside a pair remain unchanged.
    """
    candidates = [
        shape
        for shape in shapes
        if not is_role_bearing(shape)
        and _is_availability_pair_member(shape, shapes, named_place=named_place)
    ]
    if not candidates or not probe_places:
        return shapes, False

    all_svs = list(dict.fromkeys(sv for shape in candidates for sv in (shape.sv_arc_facts or {})))
    if not all_svs:
        return shapes, False

    try:
        probe = graph.observation_facets_batch(all_svs, probe_places, False)
    except Exception:
        logger.warning(
            "resolve_variable: sibling-availability probe failed for %d shape(s) "
            "against probe_places=%r; resetting place_available to unknown so "
            "the prune treats every candidate as unprunable",
            len(candidates),
            probe_places,
            exc_info=True,
        )
        reset_by_group = {
            _shape_group_key(shape): dataclasses.replace(shape, place_available=None)
            for shape in candidates
        }
        return [reset_by_group.get(_shape_group_key(shape), shape) for shape in shapes], True

    places_by_sv = {
        sv: frozenset(
            place for place, facets in by_entity.items() if any(f.obs_count > 0 for f in facets)
        )
        for sv, by_entity in probe.items()
    }
    updated_by_group = {}
    for shape in candidates:
        places = frozenset().union(
            *(places_by_sv.get(sv, frozenset()) for sv in (shape.sv_arc_facts or {}))
        )
        updated_by_group[_shape_group_key(shape)] = dataclasses.replace(
            shape, place_available=bool(places), available_places=places
        )
    return [updated_by_group.get(_shape_group_key(shape), shape) for shape in shapes], False
