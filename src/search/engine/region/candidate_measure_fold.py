"""Fold measure-only candidate siblings into ``resolved.alternative_measures``."""

from __future__ import annotations

from collections.abc import Mapping

from code_index.labels import compose_measure_label
from code_index.shared import get_measure_index
from search.engine.clients.graph import Facet
from search.engine.retrieval.bound_values import MeasurePartition, mint_sibling_identity
from search.engine.shape import ShapeDraft
from search.engine.spec_build import measure_axis_values
from search.models import Spec, SpecAlternativeMeasure, SpecDefinition


def _facet_provenance_ids(facets: list[Facet]) -> set[str]:
    """Distinct provenanceIds across an observation-facet list, dropping any
    facet the graph reported with no provenanceId."""
    return {f.provenance_id for f in facets if f.provenance_id is not None}


def _shapes_are_measure_siblings(a: ShapeDraft, b: ShapeDraft) -> bool:
    """Match shapes with the same subject and denominator that differ only by measure.

    A denominator, declared-role, or optional-role difference keeps the shape separate. The
    predicate uses fields present on every ``ShapeDraft``.
    """
    return (
        a.meas_prop_dcid != b.meas_prop_dcid
        and a.pop_type_dcid == b.pop_type_dcid
        and a.stat_type_dcid == b.stat_type_dcid
        and a.meas_qual_dcid == b.meas_qual_dcid
        and a.meas_denom_dcid == b.meas_denom_dcid
        and a.meas_denom_unresolved == b.meas_denom_unresolved
        and a.declared_roles == b.declared_roles
        and a.optional_roles == b.optional_roles
    )


def _fold_measure_sibling_candidates(
    specs: list[Spec],
    shapes_by_spec_id: Mapping[str, ShapeDraft],
    measure_partitions_by_spec_id: Mapping[str, MeasurePartition | None],
) -> list[Spec]:
    """Fold measure-only siblings into the first surviving Spec's alternatives.

    Grouping uses ``ShapeDraft`` fields, not minted identities, because alternate bindings
    can differ. The alternative identity is the one a question electing the sibling's shape
    under the retained Spec's bindings mints (`mint_sibling_identity` with the sibling's
    shape): the retained Spec's bindings on the slots the sibling's shape declares, the
    axes the sibling's ``MeasurePartition`` asserts, and the dataset the sibling's shape
    names, in canonical form. Run before survivor-count branches, since folding can leave
    one Spec.

    Keep a sibling as a separate candidate when it lacks a partition or a matching label.
    A folded sibling shares the retained Spec's price basis, so it is a measure option
    (axis "what"), labelled without the price basis.
    """
    measure_index = get_measure_index()
    consumed: set[str] = set()
    folded_any = False
    result: list[Spec] = []
    for i, spec in enumerate(specs):
        if spec.spec_id in consumed:
            continue
        shape = shapes_by_spec_id[spec.spec_id]
        folded: list[SpecAlternativeMeasure] = []
        for other in specs[i + 1 :]:
            if other.spec_id in consumed:
                continue
            other_shape = shapes_by_spec_id[other.spec_id]
            if not _shapes_are_measure_siblings(shape, other_shape):
                continue
            other_partition = measure_partitions_by_spec_id.get(other.spec_id)
            if other_partition is None:
                continue
            sibling = mint_sibling_identity(
                spec.definition.identity,
                other_partition,
                measure_index.axes,
                measure_index.partition_labels,
                shape=other_shape,
            )
            if sibling.label is None:
                continue
            consumed.add(other.spec_id)
            folded.append(
                SpecAlternativeMeasure(
                    definition=SpecDefinition(spec_id=sibling.spec_id, identity=sibling.identity),
                    label=compose_measure_label(sibling.label, measure_index.values),
                    axis="what",
                    measure_axis_values=measure_axis_values(other_partition),
                )
            )
        if folded:
            folded_any = True
            existing = spec.resolved.alternative_measures or []
            existing_ids = {a.definition.spec_id for a in existing}
            merged = [*existing, *(a for a in folded if a.definition.spec_id not in existing_ids)]
            spec = spec.model_copy(
                update={
                    "resolved": spec.resolved.model_copy(update={"alternative_measures": merged})
                }
            )
        result.append(spec)
    return result if folded_any else specs
