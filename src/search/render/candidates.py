"""Compare candidate sentence plans and write the candidates part's sentence.

This module marks the tokens that differ between candidates and writes the part's
sentence. The sentence gives the count of readings, then each shown reading's heading
and a byline led by its source and span, then the part's departures and its assumptions
clause. Readings whose
heading and byline read alike add to the byline the graph name that sets them apart.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from search.models import (
    Assumption,
    BindingSet,
    BindingValue,
    Departure,
    SentenceSegment,
    SentenceToken,
    Slot,
    Spec,
    coverage_is_empty,
)

from .assumptions import clause_tokens
from .departures import departure_tokens
from .heading import (
    LATEST_YEAR_LABEL,
    HeadingPlan,
    heading_for,
    sibling_how_properties,
    value_signature,
)
from .tokens import _capitalize_first_token, heading_tokens
from .window import Period, _format_resolved_date

# Text bucket -> summary noun. How-axis properties contribute their declared
# nouns separately. Funder tokens use role="origin" rather than an axis.
_CANDIDATE_BUCKET_NOUNS: dict[str, str] = {
    "what": "indicator",
    "where": "place",
    "funder": "funder",
    "when": "time period",
    "source": "data source",
}
# Text-comparison order. How-axis nouns follow "what" in the header.
_TEXT_BUCKET_ORDER: tuple[str, ...] = ("what", "where", "funder", "when", "source")

# Absence is distinct from every value. It is equal when every candidate is
# silent and a difference when another candidate binds or declares a value.
_HOW_AXIS_SILENT = None

# Equal breadth markers compare as the same "every X" reading. A different
# distinct_count marks a difference between candidates.
_HOW_AXIS_VARIES_PREFIX = "__varies__:"


def _candidate_bucket(token: SentenceToken) -> str | None:
    """Return the text-comparison bucket, or None for non-text and how tokens."""
    if token.role == "origin":
        return "funder"
    if token.axis in _CANDIDATE_BUCKET_NOUNS:
        return token.axis
    return None


def _how_axis_property_dcids(specs: Sequence[Spec]) -> set[str]:
    """Return how properties declared by slots or open-axis facts."""
    dcids: set[str] = set()
    for spec in specs:
        for slot in spec.resolved.slots:
            if slot.key.axis == "how" and slot.key.property is not None:
                dcids.add(slot.key.property.ref)
        for fact in spec.resolved.open_axis_facts or ():
            dcids.add(fact.property)
    return dcids


def _how_axis_slot(spec: Spec, prop_dcid: str) -> Slot | None:
    return next(
        (
            slot
            for slot in spec.resolved.slots
            if slot.key.axis == "how"
            and slot.key.property is not None
            and slot.key.property.ref == prop_dcid
        ),
        None,
    )


def _how_axis_signature(spec: Spec, prop_dcid: str) -> frozenset[str] | None:
    """Return this Spec's how-property signature.

    Bound slots use member dcids or literals. Open-axis facts use their agreed
    value or a distinct-count marker. Absent values use ``_HOW_AXIS_SILENT``.
    """
    slot = _how_axis_slot(spec, prop_dcid)
    if slot is not None:
        binding = slot.binding
        if isinstance(binding, BindingValue):
            return frozenset({value_signature(binding.value)})
        if isinstance(binding, BindingSet):
            return frozenset(value_signature(v) for v in binding.values)
        # BindingUnbound: the query left it open, so fall through to what the
        # served members themselves agree on, if anything.
    fact = next((f for f in (spec.resolved.open_axis_facts or ()) if f.property == prop_dcid), None)
    if fact is None:
        return _HOW_AXIS_SILENT
    if fact.varies:
        return frozenset({f"{_HOW_AXIS_VARIES_PREFIX}{fact.distinct_count}"})
    if fact.value is not None:
        return frozenset({fact.value.ref})
    return _HOW_AXIS_SILENT


def _how_axis_noun(
    specs: Sequence[Spec], prop_dcid: str, axis_declarations: Mapping[str, str]
) -> str | None:
    """Return a slot label or declared plain name.

    Return None when neither exists.
    """
    for spec in specs:
        slot = _how_axis_slot(spec, prop_dcid)
        if slot is not None:
            return slot.key.label
    return axis_declarations.get(prop_dcid)


def _differing_how_property_nouns(
    specs: Sequence[Spec], axis_declarations: Mapping[str, str]
) -> list[str]:
    """Return nouns for differing how properties in declaration order.

    Mark each property's tokens in place, even when no noun exists for the
    header. Undeclared properties sort last by dcid.
    """
    order_index = {dcid: i for i, dcid in enumerate(axis_declarations)}
    differing_props: list[str] = []
    for prop_dcid in _how_axis_property_dcids(specs):
        signatures = {_how_axis_signature(spec, prop_dcid) for spec in specs}
        if len(signatures) > 1:
            differing_props.append(prop_dcid)
    differing_props.sort(key=lambda dcid: (order_index.get(dcid, len(order_index)), dcid))

    for spec in specs:
        plan = spec.resolved.sentence_plan
        if not plan:
            continue
        for i, token in enumerate(plan):
            if token.axis == "how" and token.text and token.property_dcid in differing_props:
                plan[i] = token.model_copy(update={"distinguishes": True})

    return [
        noun
        for prop_dcid in differing_props
        if (noun := _how_axis_noun(specs, prop_dcid, axis_declarations)) is not None
    ]


def candidate_differences(
    specs: Sequence[Spec],
    *,
    axis_declarations: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Mark candidate differences and return their header nouns in display order.

    Compare what, where, funder, when, and source by case-folded token text.
    A bucket differs when its per-plan signature differs. Mark a token only
    when another plan lacks that text. Compare how properties by resolved
    values, including whether an axis is silent. Marks mutate each Spec's
    sentence plan in place. Nouns come from fixed buckets and declared
    how-property labels.
    """
    declarations = axis_declarations or {}
    plans = [spec.resolved.sentence_plan or [] for spec in specs]

    bucket_texts_by_plan: list[dict[str, list[str]]] = []
    for plan in plans:
        bucket_texts: dict[str, list[str]] = defaultdict(list)
        for token in plan:
            bucket = _candidate_bucket(token)
            if bucket is not None and token.text:
                bucket_texts[bucket].append(token.text.casefold())
        bucket_texts_by_plan.append(bucket_texts)

    signatures: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for bucket_texts in bucket_texts_by_plan:
        for bucket in _TEXT_BUCKET_ORDER:
            signatures[bucket].add(tuple(bucket_texts.get(bucket, [])))
    text_differing = frozenset(
        bucket for bucket in _TEXT_BUCKET_ORDER if len(signatures[bucket]) > 1
    )

    if text_differing:
        bucket_sets_by_plan = [
            {bucket: set(texts) for bucket, texts in bucket_texts.items()}
            for bucket_texts in bucket_texts_by_plan
        ]
        for i, plan in enumerate(plans):
            for j, token in enumerate(plan):
                bucket = _candidate_bucket(token)
                if bucket not in text_differing or not token.text:
                    continue
                text = token.text.casefold()
                if any(
                    text not in bucket_sets_by_plan[k].get(bucket, set())
                    for k in range(len(plans))
                    if k != i
                ):
                    plan[j] = token.model_copy(update={"distinguishes": True})

    how_nouns = _differing_how_property_nouns(specs, declarations)

    nouns: list[str] = []
    if "what" in text_differing:
        nouns.append(_CANDIDATE_BUCKET_NOUNS["what"])
    nouns.extend(how_nouns)
    nouns.extend(
        _CANDIDATE_BUCKET_NOUNS[bucket]
        for bucket in ("where", "funder", "when", "source")
        if bucket in text_differing
    )
    return tuple(nouns)


def all_candidates_empty(specs: Sequence[Spec]) -> bool:
    """Return whether every candidate has zero observations.

    An empty set returns False.
    """
    if not specs:
        return False
    return all(coverage_is_empty(spec.resolved.coverage) for spec in specs)


def candidates_sentence_plan(
    specs: Sequence[Spec],
    *,
    truncated: bool = False,
    total: int | None = None,
    departures: Sequence[Departure] = (),
    assumptions: Sequence[Assumption] = (),
) -> list[SentenceToken]:
    """The candidates part's sentence: how many readings there are ("3 readings."), then
    each shown reading's heading and a byline led by the source and span that set it
    apart, then the part's departures, then its assumptions clause (`clause_tokens`).

    ``specs`` are the shown readings. Each reads its heading and byline (`heading_for`)
    as its own sentence plan does, keeping the how properties its siblings differ on
    (`sibling_how_properties`). A reading's span moves from its heading to its byline,
    after the source. The byline's other items (latest-year label, unit) follow, then,
    for readings that would otherwise read alike, what sets each apart
    (`_telling_names`). Every reading's tokens are text tokens, since their slots belong
    to that reading's Spec, which carries them.
    ``truncated`` and ``total`` say whether more readings exist than are shown
    (`readings_text`). When no shown reading has data, the count says so.
    """
    tokens = [SentenceToken(kind="text", text=readings_text(len(specs), truncated, total))]
    if all_candidates_empty(specs):
        tokens.append(SentenceToken(kind="text", text=f" {ALL_CANDIDATES_EMPTY_TEXT}"))
    keep = sibling_how_properties(specs)
    plans = [heading_for(spec, keep_how_properties=keep) for spec in specs]
    readings = [_reading_tokens(plan) for plan in plans]
    telling = _telling_names(specs, readings)
    for plan, reading, name in zip(plans, readings, telling, strict=True):
        if name is not None:
            reading = _reading_tokens(plan, telling=name)
        tokens.append(SentenceToken(kind="text", text=" "))
        tokens.extend(reading)
    tokens.extend(departure_tokens([d.text for d in departures], after_text=True))
    tokens.extend(clause_tokens([item.text for item in assumptions], after_text=True))
    return tokens


def _telling_names(
    specs: Sequence[Spec], readings: Sequence[Sequence[SentenceToken]]
) -> list[str | None]:
    """For each reading in `specs` that reads the same as another (`readings`, in the
    same order), the graph name that sets it apart. That name comes from the first of
    the readings' population type, measured property, stat type, measurement qualifier,
    denominator and StatVars on which the readings that read alike carry distinct graph
    names. A reading none of them sets apart has None."""
    alike: dict[str, list[int]] = defaultdict(list)
    for position, reading in enumerate(readings):
        alike["".join(token.text for token in reading)].append(position)
    names: list[str | None] = [None] * len(specs)
    for group in alike.values():
        if len(group) < 2:
            continue
        for labels in zip(
            *(_identifying_names(specs[position]) for position in group), strict=True
        ):
            if all(labels) and len(set(labels)) == len(labels):
                for position, label in zip(group, labels, strict=True):
                    names[position] = label
                break
    return names


def _identifying_names(spec: Spec) -> tuple[str | None, ...]:
    """The graph names of what identifies a reading's measure, in the order
    `_telling_names` tries them and ending with the StatVars' names joined. An entry is
    None where the graph gives no name."""
    shape = spec.resolved.shape
    refs = (
        shape.population_type,
        shape.measured_property,
        shape.stat_type,
        shape.measurement_qualifier,
        shape.measurement_denominator,
    )
    members = spec.resolved.sentence_facts.members
    stat_vars = [name for member in members if (name := _graph_name(member.ref, member.label))]
    return (
        *(_graph_name(ref.ref, ref.label) if ref is not None else None for ref in refs),
        ", ".join(stat_vars) if stat_vars and len(stat_vars) == len(members) else None,
    )


def _graph_name(dcid: str, label: str | None) -> str | None:
    """The label when the graph gives one. A label equal to its dcid is no name."""
    return label if label and label != dcid else None


def readings_text(shown: int, truncated: bool, total: int | None) -> str:
    """The count of readings: "3 readings.", "12 readings; the first 5 are shown.", or,
    with more readings than shown and no known total, "More than 5 readings; the first 5
    are shown." """
    noun = "reading" if shown == 1 else "readings"
    if not truncated:
        return f"{shown} {noun}."
    verb = "is" if shown == 1 else "are"
    first = "first" if shown == 1 else f"first {shown}"
    if total is not None and total > shown:
        return f"{total} readings; the {first} {verb} shown."
    return f"More than {shown} {noun}; the {first} {verb} shown."


ALL_CANDIDATES_EMPTY_TEXT = "None of them has data."
"""Follows the count of readings when no shown reading has any observation."""

_BYLINE_SEPARATOR = " · "


def _reading_tokens(plan: HeadingPlan, *, telling: str | None = None) -> list[SentenceToken]:
    """One reading's heading without its period (`heading_tokens`), then its byline:
    source, span, the latest-year label and unit, and the ``telling`` name that sets the
    reading apart, joined by " · ", each closed by a period. Every token is text."""
    byline = plan.byline
    period = plan.heading.period
    items = [
        *([byline.source.text] if byline.source is not None and byline.source.text else []),
        *([_span_text(period)] if period is not None else []),
        *([LATEST_YEAR_LABEL] if byline.latest else []),
        *([byline.unit_text] if byline.unit_text else []),
        *([telling] if telling else []),
    ]
    tokens = [
        _reading_text(token.text, "heading")
        for token in heading_tokens(plan.heading, period=False)
        if token.text
    ]
    for i, item in enumerate(items):
        tokens.append(_reading_text(_BYLINE_SEPARATOR if i else " ", "byline"))
        item_tokens = [_reading_text(item, "byline")]
        _capitalize_first_token(item_tokens)
        tokens.extend(item_tokens)
    if items:
        tokens.append(_reading_text(".", "byline"))
    return tokens


def _span_text(period: Period) -> str:
    """The period as a byline item: the bare date for one date ("2024"), else as the
    heading reads it ("1973 to 2024", "since 1973")."""
    return _format_resolved_date(period.end or "") if period.kind == "in" else period.text


def _reading_text(text: str, segment: SentenceSegment) -> SentenceToken:
    return SentenceToken(kind="text", text=text, segment=segment)
