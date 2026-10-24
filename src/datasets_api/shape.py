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

Materialisation and role validation use the returned shape. Minted identities carry the
caller's dcid, and those operations re-read the graph when they run.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets_api.codes_materialise import MemberNotResolvable
from materialise.facets import ElectedDataset
from materialise.graph import GraphReader
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
    `stub_candidates` holds the `dcid:` refs the graph walk left out of that family because
    their complete read carries no definition arc (`MemberGroup.stub_candidates`).
    """

    stat_var: str
    declared_roles: tuple[str, ...]
    slots: tuple[tuple[str, str | None], ...]
    measured_property: str | None
    population_type: str | None
    dataset: ElectedDataset | None = None
    family: MemberFamily | None = None
    stub_candidates: tuple[str, ...] = ()


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
