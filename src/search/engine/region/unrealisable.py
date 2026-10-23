"""Unbind bound values that match no member together, so the answer reads what holds.

A resolve refuses bound values that together exclude every member of the shape
(`retrieve.BoundValuesUnrealisable`), naming each member's conflicting properties. The
elected shape's door then unbinds a set chosen by `BoundValuesUnrealisable.relaxation` and
resolves again. A declared default or the elected measure partition gives way first, then
every value bound as its slot's sole option that some member conflicts on, then a minimal
set of the values the request named. A single-member slot the unbinding opens takes the
value its population forces, unless it released that value
(`_with_forced_values`). The engine-chosen and sole-option values go
silently, with what the engine derived from them (`ResolveState.forget_bound`). Each
request-named one a recorded term supports is recorded as a code requirement
(`region.state.Requirement`) on a property the shape holds, so the answer read without it
refuses `qualifier_not_bound` with that answer as its flagged match, and a candidate set
departs it `measure_not_applied` unless every answer applies it
(`region.requirements.judge`). One no recorded term supports, which only the bind model
read (`phase_rank_bind.model_only_values`), is a code requirement that does not refuse
(`region.state.Requirement.refuses`), so an answer read without it stands and departs
`measure_not_applied` under its name. A resubmit binds only values its request posted, so
every value counts as request-named: each one unbound is recorded under its name
(`unbind_posted`), and the resubmit refuses on it by the same rule
(`requirements.settle_posted`). A refusal names a dropped value no name exists for by
its dcid, and its departure names its slot, so no dcid reaches prose. The Spec, its heading and
`spec_id`, a resubmit of it and the datasets mint then all describe the bindings that
were read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, slot_label
from search.engine.clients.graph import EngineGraphClient
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.materialise_step import Departed
from search.engine.region.phase_election import _settle_sole_options
from search.engine.region.phase_rank_bind import _resolve_query_qualifiers, model_only_values
from search.engine.region.promote_state import _PromoteState
from search.engine.region.requirements import record_reading
from search.engine.region.state import (
    Requirement,
    ResolveState,
    code_requirement,
    unapplied_departure,
    unjudged_requirement,
    with_requirement,
)
from search.engine.retrieval.bound_values import without_values
from search.engine.retrieval.retrieve import BoundValuesUnrealisable
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.heading.departure_fact import DepartureFact
from search.models import GraphRef, ResolvedQualifier, Warning


def unbind_unrealisable(state: ResolveState, unrealisable: BoundValuesUnrealisable) -> None:
    """Unbind values of `unrealisable` and leave `state` to resolve again.

    The set is `unrealisable.relaxation` with the engine-chosen properties (a declared
    default, the elected measure partition) and the ones bound as a slot's sole option
    (`_sole_option_properties`) giving way before the request-named ones. Its properties of
    those two classes go silently, each slot shown unbound: the request's words named
    neither. What the engine derived from every unbound property goes with it
    (`ResolveState.forget_bound`). A request-named value no recorded term supports
    (`phase_rank_bind.model_only_values`) becomes a code requirement that does not refuse
    (`_model_value_requirements`). An answer leaving it unapplied stands and departs
    `measure_not_applied` under its name, else its slot's. Each term the request named for
    the other values becomes a code requirement on the codes it names (`_asked_terms`),
    departing `measure_not_applied` (`binding_ref` `how:<property>`, `asked` the term's
    text) if the final answer leaves it unapplied. The values the words name by their names
    alone share one departure under their joined names. Each such value, and each value of
    a term naming no code, becomes a requirement of its own (`_value_requirements`), so binding one
    does not meet another. A value with no name departs under its slot's name
    (`_unnamed_departure`), so no dcid reaches prose. The qualifier records are read again
    over the remaining bindings
    (`requirements.record_reading`), and a record of a phrase unbinding now requires adds
    nothing.
    """
    engine_chosen = state.measure_elected_properties | state.defaulted_properties
    unnamed = _sole_option_properties(state.bindings)
    properties = unrealisable.relaxation(engine_chosen=engine_chosen, unnamed=unnamed)
    for prop in sorted(properties - engine_chosen - unnamed):
        bound = _bound_values(state.bindings, prop)
        labels = _value_labels(bound, graph=state.graph)
        model = model_only_values(state, prop, bound, labels)
        slot = _slot_name(state.shape_draft, prop)
        _require(state, _model_value_requirements(prop, model, labels, slot=slot))
        values = tuple(value for value in bound if value not in model)
        if not values:
            continue
        terms = _asked_terms(state, prop)
        if not terms:
            names = dict.fromkeys(labels[value] for value in values if value in labels)
            departure = _unapplied(" and ".join(names), prop) if names else None
            _require(
                state,
                _value_requirements(
                    prop,
                    values,
                    labels,
                    departure,
                    unnamed=_unnamed_departure(slot, prop),
                    named_only=True,
                ),
            )
        for asked, codes in terms:
            departure = _unapplied(asked, prop)
            named = tuple(state.codes_named(asked, prop))
            if all_codes := tuple(dict.fromkeys((*codes, *(ref.ref for ref in named)))):
                requirements = [
                    code_requirement(
                        property_dcid=prop,
                        codes=all_codes,
                        asked=asked,
                        named=named,
                        departure=departure,
                        origin="unbinding",
                    )
                ]
            else:
                requirements = _value_requirements(
                    prop,
                    values,
                    labels,
                    departure,
                    unnamed=_unnamed_departure(slot, prop),
                    named_only=False,
                )
            _require(state, requirements)
    released: dict[str | None, tuple[str, ...]] = {
        prop: _bound_values(state.bindings, prop) for prop in properties
    }
    state.bindings = _with_forced_values(
        without_values(state.bindings, properties), state.shape_draft, released
    )
    state.forget_bound(properties)
    state.admitted_qualifiers = tuple(
        match for match in state.admitted_qualifiers if match.code_system not in properties
    )
    record_reading(state, _resolve_query_qualifiers(state))


def _with_forced_values(
    bindings: list[SlotBindingDraft],
    shape: ShapeDraft,
    released: Mapping[str | None, Sequence[str]],
) -> list[SlotBindingDraft]:
    """`bindings` with each single-member slot the unbinding left open bound to the value
    its population forces (`phase_election._settle_sole_options`), so the answer states
    what the elected shape forces whatever path opened the slot. "Female unemployment rate"
    over a shape whose one member is male reads `gender=Male`, as it does when the bind
    reply leaves gender unbound. A slot keeps open when the forced value is one the
    unbinding released from it, alone or in a set, since no member holds it beside the
    rest, and each unbinding then releases a value it does not bind again."""
    forced = _settle_sole_options(bindings, shape)
    return [
        claim
        if claim.kind == "value"
        and not set(claim.value_dcids) <= set(released.get(claim.property_dcid, ()))
        else draft
        for draft, claim in zip(bindings, forced, strict=True)
    ]


def _sole_option_properties(bindings: Iterable[SlotBindingDraft]) -> frozenset[str]:
    """The properties the bind reply bound to a value only because the slot offered no
    other (`SlotBindingDraft.evidence` ``sole_option``, `drafted_by` the reply): the
    request's words name none of them, and each claims that every member asserts its
    value, which a member conflicting on it refutes. A ``sole_option`` value the engine
    wrote makes no such claim: a partition value (`standard.bind_partition_values`) sits
    on a slot the members differ on, and a series code's shared values
    (`bound_values.bind_member_values`) are shared by that code's StatVars alone. Those
    give way as values the request named."""
    return frozenset(
        b.property_dcid
        for b in bindings
        if b.kind == "value"
        and b.evidence == "sole_option"
        and b.drafted_by == "reply"
        and b.property_dcid is not None
    )


def _unapplied(asked: str, prop: str) -> Departed:
    """The `measure_not_applied` departure for `asked`, the request's words, on `prop`."""
    return unapplied_departure(asked, asked=asked, property_dcid=prop)


def _unnamed_departure(slot: str, prop: str) -> Departed:
    """The departure of a value on `prop` no name exists for, naming it by `slot`, its
    slot's name: the `query_qualifier_dropped` warning and the `measure_not_applied` fact,
    which states it among the values left out (`DepartureFact.values`), as nothing the
    request worded names it."""
    value = f"a requested {slot} value"
    return (
        Warning(
            code=QUERY_QUALIFIER_DROPPED,
            severity="degraded",
            message=f"{value[0].upper()}{value[1:]} was not applied.",
        ),
        DepartureFact(code="measure_not_applied", binding_ref=f"how:{prop}", values=(value,)),
    )


def _slot_name(shape: ShapeDraft, prop: str) -> str:
    """The label `shape`'s slot for `prop` carries (`bind.slot_label`)."""
    key = next(
        (key for key in shape.slot_keys if key.property_dcid == prop),
        SlotKeyDraft(axis="how", property_dcid=prop, label=prop),
    )
    return slot_label(key)


def _value_requirements(
    prop: str,
    values: Sequence[str],
    labels: Mapping[str, str],
    departure: Departed | None,
    *,
    unnamed: Departed,
    named_only: bool,
) -> list[Requirement]:
    """One code requirement per value in `values` on `prop`, asked under its name in
    `labels`, else nameless, so a refusal names it by dcid. `departure` discloses them all,
    or with `named_only` the named values, whose names it joins; `None` when it names none.
    A named value's own departure discloses it alone. A nameless value departs `unnamed`,
    which names its slot and no value (`Requirement.generic`)."""
    requirements = []
    for value in values:
        label = labels.get(value)
        shared = departure if label is not None or not named_only else None
        own = _unapplied(label, prop) if label is not None else None
        requirements.append(
            code_requirement(
                property_dcid=prop,
                codes=(value,),
                asked=label,
                named=(GraphRef(ref=value, label=label),) if label is not None else (),
                departure=shared or own,
                origin="unbinding",
                alone=own if shared else None,
                generic=unnamed if label is None else None,
            )
        )
    return requirements


def _require(state: ResolveState, requirements: Iterable[Requirement]) -> None:
    for requirement in requirements:
        state.requirements = with_requirement(state.requirements, requirement)


def unbind_posted(state: _PromoteState, unrealisable: BoundValuesUnrealisable) -> None:
    """Unbind a minimal set of the values of `unrealisable` from a resubmit's
    `state.bindings` (`BoundValuesUnrealisable.relaxation`, every posted property
    request-named), each dropped as `drop_posted_values` drops it."""
    properties = unrealisable.relaxation()
    for prop in sorted(properties):
        drop_posted_values(state, prop, _bound_values(state.bindings, prop))
    state.bindings = without_values(state.bindings, properties)


def drop_posted_values(state: _PromoteState, prop: str, values: Sequence[str]) -> None:
    """Record that the resubmit's answer cannot apply the `values` posted on `prop`, as a
    code requirement on them (`_PromoteState.requirements`), so it refuses on them
    (`requirements.settle_posted`). The values with a name depart `measure_not_applied`
    under it (`binding_ref` `how:<property>`) if the answer leaves them unapplied; the
    refusal names the rest by dcid."""
    labels = _value_labels(values, graph=state.graph)
    names = [name for value in values if (name := labels.get(value))]
    asked = " and ".join(dict.fromkeys(names)) if names else None
    state.requirements = with_requirement(
        state.requirements,
        code_requirement(
            property_dcid=prop,
            codes=tuple(values),
            asked=asked,
            named=tuple(GraphRef(ref=dcid, label=label) for dcid, label in labels.items()),
            departure=_unapplied(asked, prop) if asked is not None else None,
            origin="posted",
        ),
    )


def disclose_posted_values(state: _PromoteState, prop: str, values: Sequence[str]) -> None:
    """Record that no member the resubmit reads carries the `values` posted on `prop`, as
    a requirement no answer is judged on (`unjudged_requirement`) for each, so the answer
    reads what its members hold and departs `measure_not_applied` under each value's name. A value with no name departs under its slot's name
    (`_unnamed_departure`)."""
    labels = _value_labels(values, graph=state.graph)
    for value in values:
        label = labels.get(value)
        requirement = (
            unjudged_requirement(label, asked=label, property_dcid=prop, origin="posted")
            if label is not None
            else Requirement(
                term=ResolvedQualifier(text=value, named=[], disposition="unbound"),
                origin="posted",
                generic=_unnamed_departure(_slot_name(state.shape_draft, prop), prop),
            )
        )
        state.requirements = with_requirement(state.requirements, requirement)


def _model_value_requirements(
    prop: str, values: Sequence[str], labels: Mapping[str, str], *, slot: str
) -> list[Requirement]:
    """One code requirement per value in `values` on `prop`, values only the bind model
    read, that does not refuse (`Requirement.refuses`). An answer that leaves it unapplied
    departs `measure_not_applied` under its name in `labels`, else under `slot`, its
    slot's name (`_unnamed_departure`)."""
    requirements = []
    for value in values:
        label = labels.get(value)
        requirements.append(
            code_requirement(
                property_dcid=prop,
                codes=(value,),
                asked=label,
                named=(GraphRef(ref=value, label=label),) if label is not None else (),
                departure=_unapplied(label, prop) if label is not None else None,
                origin="unbinding",
                generic=_unnamed_departure(slot, prop) if label is None else None,
                refuses=False,
            )
        )
    return requirements


def _asked_terms(state: ResolveState, prop: str) -> list[tuple[str, tuple[str, ...]]]:
    """The request's words for the values bound on `prop`, each with the codes they ask
    for: the words of each admitted qualifier match on it (a phrase the extractor dropped
    and the raw text bound) with the codes it matched, else the text of each term a final
    binding on it covers with the codes it names. Empty when no recorded term names the
    values."""
    terms: dict[str, list[str]] = {}
    for match in state.admitted_qualifiers:
        if match.code_system == prop:
            terms.setdefault(match.matched_text, []).append(match.code_dcid)
    if not terms:
        for qualifier in state.query_qualifiers:
            if qualifier.slot == prop and qualifier.disposition == "bound":
                terms.setdefault(qualifier.text, []).extend(ref.ref for ref in qualifier.named)
    return [(asked, tuple(dict.fromkeys(codes))) for asked, codes in terms.items()]


def _bound_values(bindings: Sequence[SlotBindingDraft], prop: str) -> tuple[str, ...]:
    """The values `bindings` bind on `prop`, in binding order."""
    return tuple(
        dcid
        for binding in bindings
        if binding.property_dcid == prop
        for dcid in binding.value_dcids
    )


def _value_labels(values: Sequence[str], *, graph: EngineGraphClient) -> dict[str, str]:
    """The name of each of `values` that has one, by the code index (short name, else
    label) or the graph, in the order of `values`."""
    index = get_index()
    indexed = {
        dcid: row.short_name or row.label
        for dcid in values
        if (row := index.code_by_dcid(dcid)) is not None
    }
    unindexed = [dcid for dcid in values if dcid not in indexed]
    graph_labels = graph.node_labels_batch(unindexed) if unindexed else {}
    return {dcid: name for dcid in values if (name := indexed.get(dcid) or graph_labels.get(dcid))}
