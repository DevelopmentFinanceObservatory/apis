"""Write the dataset a query asks for into the bindings a family is elected from.

A variable asks for a dataset through the mention its request's reading bound to it
(`dataset_mention.read_mentions`). The mention names the dataset as the source, through
one source-axis value binding (`retrieval.bound_values.source_binding`), which
`shape_query` reads into the family election and `spec_build.build_slot` binds into the
source slot.
The engine writes that binding here and nowhere else on a first resolve. Shape candidacy
keeps only the mentioned dataset's shapes on a part with a bound mention
(`phase_rank_bind._dataset_candidates`), so the elected shape is always one of its own.
"""

from __future__ import annotations

from collections.abc import Sequence

from search.engine.bind import SlotBindingDraft
from search.engine.dataset_mention import DatasetMention
from search.engine.retrieval.bound_values import shape_dataset, source_binding
from search.engine.shape import ShapeDraft


def without_dataset_drafts(bindings: Sequence[SlotBindingDraft]) -> list[SlotBindingDraft]:
    """Return `bindings` less every draft on the source axis.

    No model draft names a dataset: `with_mention` and `with_named_source` write the only
    bindings that do.
    """
    return [binding for binding in bindings if binding.axis != "source"]


def with_named_source(
    bindings: Sequence[SlotBindingDraft], shape: ShapeDraft, source: str | None
) -> list[SlotBindingDraft]:
    """Return `bindings` naming the dataset `source` when it publishes `shape`'s family
    (`shape_dataset`).

    Every dataset draft in `bindings` is replaced, so the result holds one source binding
    or none.
    """
    kept = without_dataset_drafts(bindings)
    if source is None or shape_dataset(shape) != source:
        return kept
    return [*kept, source_binding(source)]


def with_mention(
    bindings: Sequence[SlotBindingDraft], mention: DatasetMention
) -> list[SlotBindingDraft]:
    """Return `bindings` asking for the mentioned dataset: every dataset draft replaced by
    the source binding the mention names."""
    return [*without_dataset_drafts(bindings), source_binding(mention.provenance)]
