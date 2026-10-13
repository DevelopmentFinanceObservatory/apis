"""Project a resolved Spec onto the graph dcids its shape, bound slots, and
selected provenance name, and match those dcids against the shared skills index.

Pure and dcid-keyed like ``skills_index.match``. Graph access belongs to the skills index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from search.models import BindingSet, BindingValue
from skills_index import SkillsIndex, match_dcids

if TYPE_CHECKING:
    from search.models import Axis, Spec

_PROJECTED_AXES: frozenset[Axis] = frozenset({"what", "how"})


def project_spec_dcids(spec: Spec) -> set[str]:
    """Return the Spec's five shape anchors, its bound constraint slots' dcids, and
    its selected provenance.

    A slot contributes when it sits on the ``what`` or ``how`` axis, its key
    names a constraint property, and its binding is a value or a set. It adds
    the property, plus each bound value whose ``value_kind`` is ``enum_value``.
    Slots on ``where``, ``when``, and ``source`` add nothing, even when a
    constraint property backs them, and neither does an unbound slot.
    The provenance facet adds its ``ref``, the one provenance every served
    series reads, which matches skills attached on ``skillAppliesToProvenance``.
    A provenance facet selected per series, and every other facet axis, add
    nothing. ``expansion.added``, entities, and the resolved StatVars add
    nothing either. Every Spec ``with_skills`` reaches is projected this way,
    candidates and no_data specs included, so each carries its selected
    provenance too. A spec bound to a code therefore reaches skills attached at
    or above that code through the index's downward reach, and none attached
    only beneath it. Equivalent readings are projected separately by calling
    this function on each twin.
    """
    dcids: set[str] = set()
    shape = spec.resolved.shape
    for ref in (
        shape.population_type,
        shape.measured_property,
        shape.stat_type,
        shape.measurement_qualifier,
        shape.measurement_denominator,
    ):
        if ref is not None:
            dcids.add(ref.ref)
    provenance = spec.resolved.facets.get("provenance")
    if provenance is not None and provenance.ref is not None:
        dcids.add(provenance.ref)
    for slot in spec.resolved.slots:
        if slot.key.axis not in _PROJECTED_AXES or slot.key.property is None:
            continue
        binding = slot.binding
        if isinstance(binding, BindingValue):
            values = [binding.value]
        elif isinstance(binding, BindingSet):
            values = binding.values
        else:
            continue
        dcids.add(slot.key.property.ref)
        dcids.update(
            value.ref.ref
            for value in values
            if value.value_kind == "enum_value" and value.ref is not None
        )
    return dcids


def match_spec_skills(spec: Spec, *, index: SkillsIndex) -> list[str]:
    """Return skill ids matching the Spec's own projected dcids, in index order."""
    return match_dcids(project_spec_dcids(spec), index=index)
