"""Unbind bound values that match no member together, so the answer reads what holds.

The standard resolve refuses bound values that together exclude every member of the
shape (`retrieve.BoundValuesUnrealisable`), naming each member's conflicting properties.
The elected shape's door then unbinds a minimal set, one member's conflicts chosen by
`BoundValuesUnrealisable.relaxation`, and resolves again. The values the request named
weigh first, so a declared default or the elected measure partition gives way before
them; the engine-chosen values go silently and each request-named one departs
`measure_not_applied`. A resubmit binds only values its request posted, so every value
counts as request-named and each one unbound departs under its name (`unbind_posted`),
and an old share link keeps answering. The Spec, its heading and `spec_id`, a resubmit
of it and the datasets mint then all describe the bindings that were read.
"""

from __future__ import annotations

from collections.abc import Sequence

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.departures import measure_not_applied
from search.engine.qualifier_match import _normalize
from search.engine.region.phase_election import _qualifier_dropped_warning
from search.engine.region.phase_rank_bind import _resolve_query_qualifiers
from search.engine.region.promote_state import _PromoteState
from search.engine.region.state import ResolveState
from search.engine.retrieval.bound_values import without_values
from search.engine.retrieval.retrieve import BoundValuesUnrealisable


def unbind_unrealisable(state: ResolveState, unrealisable: BoundValuesUnrealisable) -> None:
    """Unbind a minimal set of the values of `unrealisable` and leave `state` to resolve again.

    The set is `unrealisable.relaxation` with the engine-chosen properties (a declared
    default, the elected measure partition) weighed after the request-named ones. Its
    engine-chosen properties go silently. Each request-named one departs
    `measure_not_applied` per term the request named for it (`binding_ref`
    `how:<property>`, `asked` the term's text). A value the request named in no recorded
    term departs under its name (`_asked_terms`). One nothing names departs nothing, so no
    dcid reaches prose. The qualifier records are read again over the remaining bindings,
    and the terms dropped here are kept out of the qualifier refusal
    (`unapplied_qualifier_phrases`).
    """
    engine_chosen = state.measure_elected_properties | state.defaulted_properties
    properties = unrealisable.relaxation(engine_chosen=engine_chosen)
    dropped_phrases: set[str] = set()
    for prop in sorted(properties - engine_chosen):
        for asked in _asked_terms(state, prop):
            dropped_phrases.add(_normalize(asked))
            state.depart(
                _qualifier_dropped_warning(asked),
                measure_not_applied(asked=asked, property_dcid=prop),
            )
    state.unapplied_qualifier_phrases |= dropped_phrases
    state.bindings = without_values(state.bindings, properties)
    state.measure_elected_properties -= properties
    state.defaulted_properties -= properties
    state.admitted_qualifiers = tuple(
        match for match in state.admitted_qualifiers if match.code_system not in properties
    )
    state.query_qualifiers = _resolve_query_qualifiers(state)


def unbind_posted(state: _PromoteState, unrealisable: BoundValuesUnrealisable) -> None:
    """Unbind a minimal set of the values of `unrealisable` from a resubmit's
    `state.bindings` (`BoundValuesUnrealisable.relaxation`, every posted property
    request-named). Each value the request posted on an unbound property departs
    `measure_not_applied` under its name (`bound_value_names`, `binding_ref`
    `how:<property>`), so the Refine action reaches its token."""
    properties = unrealisable.relaxation()
    for prop in sorted(properties):
        for asked in bound_value_names(state.bindings, prop, graph=state.graph):
            state.warnings.append(
                state.depart(
                    _qualifier_dropped_warning(asked),
                    measure_not_applied(asked=asked, property_dcid=prop),
                )
            )
    state.bindings = without_values(state.bindings, properties)


def _asked_terms(state: ResolveState, prop: str) -> list[str]:
    """The request's words for the values bound on `prop`. They are the words of each admitted
    qualifier match on it (a phrase the extractor dropped and the raw text bound), else the
    text of each term a final binding on it covers, else the bound values' names, by the
    code index (short name, else label) or the graph, leaving out a value neither names."""
    terms = [
        match.matched_text for match in state.admitted_qualifiers if match.code_system == prop
    ] or [
        qualifier.text
        for qualifier in state.query_qualifiers
        if qualifier.slot == prop and qualifier.disposition == "bound"
    ]
    if terms:
        return list(dict.fromkeys(terms))
    return bound_value_names(state.bindings, prop, graph=state.graph)


def bound_value_names(
    bindings: Sequence[SlotBindingDraft], prop: str, *, graph: EngineGraphClient
) -> list[str]:
    """The names of the values `bindings` bind on `prop`, joined in one term, by the code
    index (short name, else label) or the graph, leaving out a value neither names, so no
    dcid reaches prose. Empty when none is named."""
    values = [
        dcid
        for binding in bindings
        if binding.property_dcid == prop
        for dcid in binding.value_dcids
    ]
    index = get_index()
    indexed = {
        dcid: row.short_name or row.label
        for dcid in values
        if (row := index.code_by_dcid(dcid)) is not None
    }
    unindexed = [dcid for dcid in values if dcid not in indexed]
    graph_labels = graph.node_labels_batch(unindexed) if unindexed else {}
    names = [name for dcid in values if (name := indexed.get(dcid) or graph_labels.get(dcid))]
    return [" and ".join(dict.fromkeys(names))] if names else []
