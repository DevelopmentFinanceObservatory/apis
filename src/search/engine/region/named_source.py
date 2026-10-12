"""Write the dataset a shape names into the bindings a family is elected from.

Shape candidacy stamps each shape with the dataset it names (`ShapeDraft.named_dataset`,
`retrieval.bound_values.shape_for_part`): the one dataset the part's source scope
(`ResolveState.scope`) leaves it, else the one opt-in dataset that alone may answer it. The
dataset becomes one source-axis value binding (`retrieval.bound_values.source_binding`),
which `shape_query` reads into the family election and `spec_build.build_slot` binds into
the source slot. The engine writes that binding here and nowhere else on a first resolve,
for the elected shape and for each alternate alike, so an alternate's identity is the one a
question electing its shape mints.
"""

from __future__ import annotations

from collections.abc import Sequence

from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import shape_named_source, source_binding
from search.engine.shape import ShapeDraft


def without_dataset_drafts(bindings: Sequence[SlotBindingDraft]) -> list[SlotBindingDraft]:
    """Return `bindings` less every draft on the source axis.

    No model draft names a dataset: `with_named_source` writes the only binding that does.
    """
    return [binding for binding in bindings if binding.axis != "source"]


def with_named_source(
    bindings: Sequence[SlotBindingDraft], shape: ShapeDraft
) -> list[SlotBindingDraft]:
    """Return `bindings` naming the dataset a part electing `shape` binds as its source
    (`shape_named_source`).

    Every dataset draft in `bindings` is replaced, so the result holds one source binding
    or none.
    """
    kept = without_dataset_drafts(bindings)
    source = shape_named_source(shape)
    if source is None:
        return kept
    return [*kept, source_binding(source)]
