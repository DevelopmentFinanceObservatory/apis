"""The labels of the values a bind drafted, and which of them only the bind model read.

`draft_value_labels` names each drafted value, `value_names` lists every name the qualifier
matcher reads for a value, and `model_only_values` keeps the values no recorded term of the
request names by one of them."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from code_index.shared import get_index, get_measure_index
from code_index.tokens import tokenize
from search.engine.config import settings
from search.engine.degraded_reads import fail_open_reads
from search.engine.qualifier_match import token_occurrences
from search.render.labels import _humanize_ref

if TYPE_CHECKING:
    from search.engine.region.state import ResolveState


async def draft_value_labels(state: ResolveState, values: Sequence[str]) -> dict[str, str]:
    """The label of each of `values`, drafted values: a menu shape's
    (`ShapeDraft.value_labels`), else the code index's, else the graph's
    (`ResolveState.label_dcids`), read under the fail-open budget (`fail_open_reads`). When
    the graph's read is lost, the values it was to label are named by their dcid's tail in
    words (`_humanize_ref`) and the lost read is noted."""
    index = get_index()
    labels: dict[str, str] = {}
    for shape in state.capped_shapes:
        labels.update((shape.value_labels or {}).items())
    for value in values:
        if value not in labels and (row := index.code_by_dcid(value)) is not None:
            labels[value] = row.label
    unlabelled = list(dict.fromkeys(value for value in values if value not in labels))
    if unlabelled:
        read = [_humanize_ref(value, None) for value in unlabelled]
        with fail_open_reads(
            "drafted value labels",
            budget_s=settings().fail_open_read_budget_s,
            about={"unlabelled": unlabelled},
        ):
            read = await asyncio.to_thread(state.label_dcids, unlabelled)
        labels.update(zip(unlabelled, read, strict=True))
    return labels


def model_only_values(
    state: ResolveState, prop: str, values: Sequence[str], labels: Mapping[str, str]
) -> tuple[str, ...]:
    """The `values` bound on `prop` that no recorded term of the request supports, so only
    the bind model read them from the words. A term supports every value on `prop` when it
    is an admitted qualifier match on `prop` or a query record a binding on it covers, and
    supports one value when one of its names (`value_names`) is a run of the variable,
    a dropped qualifier phrase or the single-variable query, so "women" names Female by
    its synonym. Each source is read separately. A value with no name is the model's. An
    answer that cannot apply a supported value refuses `qualifier_not_bound` on it, and one
    that cannot apply a model-only value discloses it (`unrealisable.unbind_unrealisable`,
    `phase_rank_bind._stray_requirements`)."""
    if any(match.code_system == prop for match in state.admitted_qualifiers) or any(
        record.slot == prop and record.disposition == "bound" for record in state.query_qualifiers
    ):
        return ()
    texts = [state.variable, *state.dropped_qualifier_phrases]
    if query_text := state.query_text:
        texts.append(query_text)
    sources = [tokenize(text) for text in dict.fromkeys(texts)]
    return tuple(
        value
        for value in values
        if not any(
            token_occurrences(tokens, name)
            for tokens in sources
            for name in value_names(prop, value, labels.get(value))
        )
    )


def value_names(prop: str, value: str, label: str | None) -> tuple[str, ...]:
    """Every name the qualifier matcher reads for `value` on `prop`, and its graph `label`:
    the code index's label and synonyms of the code, and the label and synonyms of each
    matchable measure value on `prop` that spells its reading, aliases included."""
    code = get_index().code_by_dcid(value)
    measure_index = get_measure_index()
    reading = measure_index.reading_set(value)
    return (
        *(() if label is None else (label,)),
        *(() if code is None else (code.label, *code.synonyms)),
        *(
            text
            for row in measure_index.values
            if row.axis == prop and row.matchable and row.dcid in reading
            for text in (row.label, *row.synonyms)
        ),
    )
