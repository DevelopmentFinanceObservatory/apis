"""Resolve a `dcid:` StatVar's roles, constraint bindings, measured property,
and population type for materialisation and role validation.

`observationProperties` declares entity roles. Absent and empty arcs both
read as `()` and default to `observationAbout`. `constraintProperties`
names fixed values on the StatVar itself. The two sets are independent,
and `slots` contains only constraint bindings. Other arcs, including
`statType` and `measurementQualifier`, also distinguish quantities.

The reader requires `typeOf=StatisticalVariable`. It refuses `member_not_resolvable` for a missing or
wrongly typed node or a constraint with multiple values, and refuses a declared
`observationAbout` role. Measured property and population type stay
optional because `node_arcs` cannot distinguish an absent node from an
absent arc.

`resolve_shapes` batches distinct dcids in at most two POSTs. Materialisation
and role validation use the returned shape. Minted identities carry the
caller's dcid, and those operations re-read the graph when they run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from datasets_api.codes_materialise import MemberNotResolvable
from materialise.facets import ElectedDataset
from materialise.graph import UNRESOLVED_NODE, GraphReader
from materialise.roles import DegenerateObservationProperties
from materialise.roles import declared_roles as derive_declared_roles
from materialise.trace import MemberFamily

_STAT_VAR_PREFIX = "dcid:"

_TYPE_OF = "typeOf"
_STATISTICAL_VARIABLE = "StatisticalVariable"
_OBSERVATION_PROPERTIES = "observationProperties"
_MEASURED_PROPERTY = "measuredProperty"
_POPULATION_TYPE = "populationType"
_CONSTRAINT_PROPERTIES = "constraintProperties"


@dataclass(frozen=True)
class StatVarShape:
    """`slots` pairs each declared constraint with its value, or `None` if unbound.
    Materialisation and role validation use these fields. Minted identities use the caller's dcid.
    `dataset` is the dataset elected for a family the code index resolved, and `None` for a
    shape read from the graph. `family` is the whole population the identity the shape was
    resolved for admits, and `None` when member resolution could not state it.
    """

    stat_var: str
    declared_roles: tuple[str, ...]
    slots: tuple[tuple[str, str | None], ...]
    measured_property: str | None
    population_type: str | None
    dataset: ElectedDataset | None = None
    family: MemberFamily | None = None


def resolve_shape(stat_var: str, *, graph: GraphReader) -> StatVarShape:
    # Callers supply `dcid:`-prefixed refs.
    # This assertion checks that precondition.
    assert stat_var.startswith(_STAT_VAR_PREFIX), (
        f"resolve_shape received a non-dcid-prefixed statvar {stat_var!r}"
    )
    dcid = stat_var[len(_STAT_VAR_PREFIX) :]

    arcs = graph.node_arc_values(
        dcid,
        properties=(
            _TYPE_OF,
            _OBSERVATION_PROPERTIES,
            _MEASURED_PROPERTY,
            _POPULATION_TYPE,
            _CONSTRAINT_PROPERTIES,
        ),
    )
    if _STATISTICAL_VARIABLE not in arcs.get(_TYPE_OF, ()):
        raise MemberNotResolvable(members=(stat_var,))

    try:
        roles = derive_declared_roles(arcs.get(_OBSERVATION_PROPERTIES, ()))
    except DegenerateObservationProperties as exc:
        raise MemberNotResolvable(members=(stat_var,)) from exc

    constraint_properties = sorted(arcs.get(_CONSTRAINT_PROPERTIES, ()))
    constraint_values = (
        graph.node_arc_values(dcid, properties=tuple(constraint_properties))
        if constraint_properties
        else {}
    )
    slots: list[tuple[str, str | None]] = []
    for prop in constraint_properties:
        values = constraint_values.get(prop, ())
        if len(values) > 1:
            # A first-value choice would hide data. `issue["member"]` names the StatVar.
            raise MemberNotResolvable(members=(stat_var,)) from ValueError(
                f"constraint property {prop!r} on {stat_var!r} carries {len(values)} "
                f"bound values, not one: {values!r}"
            )
        slots.append((prop, values[0] if values else None))

    measured_values = arcs.get(_MEASURED_PROPERTY, ())
    measured_property = measured_values[0] if measured_values else None

    population_values = arcs.get(_POPULATION_TYPE, ())
    population_type = population_values[0] if population_values else None

    return StatVarShape(
        stat_var=stat_var,
        declared_roles=roles,
        slots=tuple(slots),
        measured_property=measured_property,
        population_type=population_type,
    )


def _drop_unresolved(values: tuple[str, ...]) -> tuple[str, ...]:
    """Remove `node_arcs_many`'s unresolved-target marker to match `node_arcs` results."""
    return tuple(value for value in values if value != UNRESOLVED_NODE)


def resolve_shapes(stat_vars: Sequence[str], *, graph: GraphReader) -> dict[str, StatVarShape]:
    """Resolve a batch in at most two POSTs, reading each distinct dcid once.

    The first read gets the five shape arcs. A second read gets the union of
    declared constraint properties when needed. Unbatched calls cost
    one read per dcid and a second for each constrained StatVar. Single-item
    callers use `resolve_shape`.

    Returns one entry per resolvable input, preserving duplicate refs.
    Missing or non-StatVar nodes, multi-valued constraints, and degenerate
    `observationProperties` entries are omitted. Empty input returns `{}`
    without a request.
    """
    if not stat_vars:
        return {}

    dcid_by_ref: dict[str, str] = {}
    for stat_var in stat_vars:
        assert stat_var.startswith(_STAT_VAR_PREFIX), (
            f"resolve_shapes received a non-dcid-prefixed statvar {stat_var!r}"
        )
        dcid_by_ref[stat_var] = stat_var[len(_STAT_VAR_PREFIX) :]
    dcids = tuple(dict.fromkeys(dcid_by_ref.values()))

    arcs_by_dcid = graph.node_arcs_many(
        dcids,
        properties=(
            _TYPE_OF,
            _OBSERVATION_PROPERTIES,
            _MEASURED_PROPERTY,
            _POPULATION_TYPE,
            _CONSTRAINT_PROPERTIES,
        ),
    )

    candidates: dict[str, dict[str, tuple[str, ...]]] = {}
    for dcid in dcids:
        held = arcs_by_dcid.get(dcid)
        if held is None:
            continue
        arcs = {prop: _drop_unresolved(values) for prop, values in held.items()}
        if _STATISTICAL_VARIABLE not in arcs.get(_TYPE_OF, ()):
            continue
        candidates[dcid] = arcs

    constraint_properties_by_dcid = {
        dcid: tuple(sorted(arcs.get(_CONSTRAINT_PROPERTIES, ())))
        for dcid, arcs in candidates.items()
    }
    needing_constraints = tuple(
        dcid for dcid, properties in constraint_properties_by_dcid.items() if properties
    )
    constraint_values_by_dcid: dict[str, dict[str, tuple[str, ...]]] = {}
    if needing_constraints:
        union_properties = tuple(
            sorted(
                {
                    property_name
                    for properties in constraint_properties_by_dcid.values()
                    for property_name in properties
                }
            )
        )
        constraint_arcs = graph.node_arcs_many(needing_constraints, properties=union_properties)
        for dcid in needing_constraints:
            held = constraint_arcs.get(dcid)
            constraint_values_by_dcid[dcid] = (
                {prop: _drop_unresolved(values) for prop, values in held.items()}
                if held is not None
                else {}
            )

    shapes_by_dcid: dict[str, StatVarShape] = {}
    for dcid, arcs in candidates.items():
        try:
            roles = derive_declared_roles(arcs.get(_OBSERVATION_PROPERTIES, ()))
        except DegenerateObservationProperties:
            continue

        constraint_values = constraint_values_by_dcid.get(dcid, {})
        slots: list[tuple[str, str | None]] = []
        for prop in constraint_properties_by_dcid[dcid]:
            values = constraint_values.get(prop, ())
            if len(values) > 1:
                break
            slots.append((prop, values[0] if values else None))
        else:
            measured_values = arcs.get(_MEASURED_PROPERTY, ())
            population_values = arcs.get(_POPULATION_TYPE, ())
            shapes_by_dcid[dcid] = StatVarShape(
                stat_var=f"{_STAT_VAR_PREFIX}{dcid}",
                declared_roles=roles,
                slots=tuple(slots),
                measured_property=measured_values[0] if measured_values else None,
                population_type=population_values[0] if population_values else None,
            )

    return {
        ref: shapes_by_dcid[dcid] for ref, dcid in dcid_by_ref.items() if dcid in shapes_by_dcid
    }
