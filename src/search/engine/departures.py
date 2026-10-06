"""Departure facts: each way a part's answer differs from what its request asked.

A `DepartureFact` (`search.heading.departure_fact`) is the engine's record of one departure,
without text. The part's `Departure` objects (`models.departure.Departure`) are rendered
from these facts by `search.render.departures`. Each fact's `binding_ref` equals the
`binding_ref` of the sentence token whose edit clears it (`when:` for a year,
`how:<property dcid>` for a measure value, `role_ref` for a role's entities), or is `None`
when no slot edit clears it.

Facts come from two places. Some are raised mid-pipeline beside the warning that discloses
them: `measure_not_applied` (`query_qualifier_dropped` on a measure qualifier),
`breakdown_not_applied` (`query_qualifier_dropped` on an operation over a flow axis the
answer does not carry), `direction_discarded`, `closest_match` (`elected_shape_no_data` and
`family_total_absent`), and `entity_unavailable` (`un_system_unavailable` on refused places,
`withheld_by_registry`), and `year_unavailable` for a unit request whose conversion tables
cover only part of the answer's years (`date_coverage_partial`,
`conversion_period_fact`) or whose deflators leave out some entities' rows
(`deflator_missing_omitted`, `deflator_omission_fact`). Each is recorded in a `FactsByWarning`
table under its warning's `warning_key`, and a part keeps it only while the part's own or
its envelope's warnings hold that warning (`surviving_facts`), so the fact follows
every filter, merge, and retraction the warning goes through. The other codes are read off
the Spec the part answers with and the operations it carries (`spec_departure_facts`). A
no-data answer for years its series holds no point in carries `year_unavailable`
(`window_gap_fact`), and a no-data answer whose check the engine narrowed carries
`scope_limited` (`scope_limited_fact`).

A part states one `entity_unavailable` per role whichever reads raised it
(`merged_entity_facts`), and only the facts that tell its reader something its heading and
byline do not (`stated_facts`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from code_index.shared import get_index
from search.heading.departure_fact import DepartureFact, ordered_facts
from search.heading.words import content_words
from search.models import (
    UNIT_REF,
    WHERE_REF,
    BindingProvenance,
    GraphRef,
    IdentityWindowRange,
    ObservationSeries,
    Operation,
    OperationVerb,
    RoleAxis,
    Spec,
    Warning,
    role_ref,
)

if TYPE_CHECKING:
    from search.engine.retrieval.read_plan import WindowGap

WHEN_REF = "when:"
"""The `binding_ref` of the `when` token."""

type WarningKey = tuple[str, str]
"""A warning's `(code, message)`, the key a raised fact is recorded under."""
type FactsByWarning = Mapping[WarningKey, DepartureFact]
"""Raised departure facts, each under the key of the warning that discloses it."""

NO_ENTITY_WORDS: Mapping[str, str] = MappingProxyType({})
"""No request words for any entity name, for a part whose door holds none."""

MODELLED_MEASUREMENT_METHODS: frozenset[str] = frozenset({"Modelled", "Estimated"})
"""The `measurementMethod` dcids whose values are modelled or estimated rather than
reported. They are DFO's own facet methods, `Modelled` on the ONE Imputed Multilateral
Sectors series and `Estimated` on projected series."""

OPERATIONS_NOT_RUN: frozenset[OperationVerb] = frozenset(
    {"rank", "aggregate", "growth", "share_of", "forecast"}
)
"""The operation verbs that compute a new figure from the part's data. Search runs none
of them, and the part shows the amounts the operation needs. `explain` and `report`
compute nothing, so the amounts are the answer."""


def warning_key(warning: Warning) -> WarningKey:
    """The key a fact raised beside `warning` is recorded under."""
    return warning.code, warning.message


def surviving_facts(
    raised: FactsByWarning, warnings: Iterable[Warning]
) -> tuple[DepartureFact, ...]:
    """The facts in `raised` whose warning `warnings` still holds, in `DEPARTURE_ORDER`."""
    return ordered_facts(
        fact for key in dict.fromkeys(map(warning_key, warnings)) if (fact := raised.get(key))
    )


OPERATION_WORDS: frozenset[str] = frozenset(
    {
        "trend",
        "trends",
        "compare",
        "comparing",
        "comparison",
        "versus",
        "vs",
        "rank",
        "ranked",
        "ranking",
        "top",
        "bottom",
        "most",
        "least",
        "highest",
        "lowest",
        "largest",
        "smallest",
        "biggest",
        "total",
        "change",
        "changed",
        "growing",
        "why",
        "overview",
    }
)
"""The words the extraction prompt (`engine.extract`) gives as wording for an operation
or for what Search shows anyway (a comparison, a series over time), case-folded. A
dropped phrase holding one is operation wording, which `Extraction.operations` already
records or which needs no departure."""


def names_request(phrase: str) -> bool:
    """Whether a phrase extraction dropped asks for something the answer could apply: it
    holds a content word (`heading.words.content_words`) and no operation word
    (`OPERATION_WORDS`). "fragile states" and "through regional development banks" do;
    "Trend in", "compare the" and "countries have the highest external debt stocks" do
    not. Whether the answer already states it is the part's to decide (`stated_facts`)."""
    words = content_words(phrase)
    return bool(words) and not words & OPERATION_WORDS


def stated_facts(
    facts: Iterable[DepartureFact], *, stated: Sequence[frozenset[str]]
) -> tuple[DepartureFact, ...]:
    """The facts in `facts` a part states, in `DEPARTURE_ORDER`. `stated` holds the
    content words of the heading and byline of each Spec the part shows
    (`render.tokens.plan_words`): one on a definite part, one per reading on a
    candidates part.

    Two kinds say nothing the part does not already state, and are left out:

    - a price basis the answer did not apply with no `shown`, because the part's byline
      states no price basis (a percent, an index, or a part with no one byline);
    - a measure qualifier (`measure_not_applied` cleared by a `how` slot, or by none) whose
      content words one of `stated` holds, source included: "gross" beside "Gross
      capital formation", "IMF" beside the byline "IMF World Economic Outlook (WEO)", "net"
      beside a reading of net outbound ODA.
    """
    return ordered_facts(
        fact
        for fact in facts
        if not (
            fact.code == "measure_not_applied"
            and fact.binding_ref == UNIT_REF
            and fact.shown is None
        )
        and not _qualifier_stated(fact, stated)
    )


def _qualifier_stated(fact: DepartureFact, stated: Sequence[frozenset[str]]) -> bool:
    """Whether `fact` is a measure qualifier one of `stated` already holds."""
    if fact.code != "measure_not_applied" or fact.binding_ref == UNIT_REF:
        return False
    asked = content_words(fact.asked or "")
    return any(asked <= words for words in stated)


def qualifier_property(named: Sequence[GraphRef]) -> str | None:
    """The one constraint property the code rows `named` belong to, or `None` when they
    span several or the index holds none of them."""
    index = get_index()
    properties = {row.code_system for ref in named if (row := index.code_by_dcid(ref.dcid))}
    return properties.pop() if len(properties) == 1 else None


def measure_not_applied(*, asked: str, property_dcid: str | None) -> DepartureFact:
    """A measure qualifier the request named as `asked` that the answer does not apply.
    Editing the slot of `property_dcid` clears it; `None` when no one slot does."""
    return DepartureFact(
        code="measure_not_applied",
        binding_ref=f"how:{property_dcid}" if property_dcid else None,
        asked=asked,
    )


def price_basis_not_applied(*, asked: str, shown: str | None) -> DepartureFact:
    """A price basis the request named as `asked` that the answer does not apply. `shown`
    is the price basis the byline's unit token states ("current US dollars"), `None` when
    the raise site does not know the served unit; a part fills it from its byline or leaves
    the fact out (`stated_facts`). The unit token clears it."""
    return DepartureFact(code="measure_not_applied", binding_ref=UNIT_REF, asked=asked, shown=shown)


def breakdown_not_applied(over: RoleAxis, *, asked: str | None) -> DepartureFact:
    """A breakdown over the flow axis `over`, answered with the total over it. `asked` is
    the operation's phrase, or `None` for an open role, which carries no words. Naming an
    entity in that role clears it."""
    return DepartureFact(
        code="breakdown_not_applied", binding_ref=role_ref(over), asked=asked, over=over
    )


def direction_discarded(name: str, axis: RoleAxis) -> DepartureFact:
    """A place `name` the request's own words put on the flow `axis` (`DoorRole.stated`),
    which the answer reads as its own subject."""
    return DepartureFact(code="direction_discarded", binding_ref=WHERE_REF, asked=name, over=axis)


def spec_departure_facts(
    spec: Spec | None,
    *,
    operations: Sequence[Operation],
    entity_words: Mapping[str, str] | None = NO_ENTITY_WORDS,
) -> list[DepartureFact]:
    """The departures read off `spec` and the `operations` its part carries: operations
    not run, entities the data holds no values for, years outside the served span,
    modelled values, and groups read as members.
    With no `spec`, only the operations not run.

    A forecast to a target year that a `year_unavailable` fact states departs as that
    fact alone; any other forecast departs as `operation_not_applied`. A group, read as
    members or holding members without values, is asked in the request's own words for
    it: `entity_words` maps each entity name the request gave to those words
    (`RegionResult.entity_words`), and a group absent from it is asked by its label. A door
    that holds no words for any entity (a resubmit, `entity_words` `None`) reads none, and its
    part states its departures without them (`without_request_words`)."""
    year_facts = _year_facts(spec, operations=operations) if spec is not None else []
    stated_years = {fact.requested_years for fact in year_facts}
    facts = [
        fact
        for operation in operations
        if not (
            operation.target_year is not None
            and (operation.target_year, operation.target_year) in stated_years
        )
        and (fact := _operation_fact(operation))
    ]
    if spec is None:
        return facts
    words = entity_words if entity_words is not None else NO_ENTITY_WORDS
    facts.extend(_unavailable_entity_facts(spec, entity_words=words))
    facts.extend(year_facts)
    if modelled := _modelled_fact(spec):
        facts.append(modelled)
    facts.extend(
        DepartureFact(
            code="group_read_as_members",
            binding_ref=role_ref(role),
            asked=words.get(group.label, group.label),
            group=group.label,
            group_size=sum(1 for entity in spec.resolved.entities if entity.role == role),
        )
        for role, group in (spec.resolved.entity_groups or {}).items()
    )
    return facts


def without_request_words(fact: DepartureFact) -> DepartureFact:
    """`fact` as a part states it on a door that holds no words of the request (a resubmit):
    `asked` is `None`, and a measure value it names (`measure_not_applied`) moves to
    `values`, so its text still names what the answer left out."""
    if fact.asked is None:
        return fact
    values = (fact.asked,) if fact.code == "measure_not_applied" else fact.values
    return dataclasses.replace(fact, asked=None, values=values)


def _operation_fact(operation: Operation) -> DepartureFact | None:
    """`operation_not_applied` for an operation Search does not run."""
    if operation.verb not in OPERATIONS_NOT_RUN:
        return None
    return DepartureFact(
        code="operation_not_applied",
        binding_ref=None,
        asked=operation.phrase,
        verb=operation.verb,
    )


ENGINE_CHOSEN_PROVENANCES: frozenset[BindingProvenance] = frozenset(
    {"declared_default", "measure_elected"}
)
"""The binding provenances of a value the engine supplied without the request's text:
a code system's declared default, and the elected measure partition. `binder` is the
bind call's reading of the request, and `text_recognized` a code its text named."""


def scope_limited_fact(spec: Spec) -> DepartureFact | None:
    """`scope_limited` for a no-data part whose checked series `spec` holds how values the
    engine supplied: what the check was limited to beyond what the request named, each as
    the code index names it (its short name, else its label). Editing the slot of the first
    limit clears it. `None` when the engine supplied none.

    A value counts when its provenance is in `ENGINE_CHOSEN_PROVENANCES` and its axis is not
    one the measure noun states (`measure_label_axes`). A value with neither a short name
    nor a graph label (its label is its dcid) is left out, so no dcid reaches prose. A value
    the StatVar carries on an open slot is not a slot binding, so it is not a limit."""
    index = get_index()
    consumed = set(spec.resolved.measure_label_axes)
    limits: dict[str, str] = {}
    for slot_filter in spec.resolved.trace.slot_filters:
        prop = slot_filter.key.property
        if (
            slot_filter.key.axis != "how"
            or prop is None
            or slot_filter.provenance not in ENGINE_CHOSEN_PROVENANCES
            or prop.dcid in consumed
        ):
            continue
        for ref in slot_filter.refs:
            row = index.code_by_dcid(ref.dcid)
            # A ref the graph gave no name carries its dcid as its label.
            label = ref.label if ref.label != ref.dcid else ""
            name = (row.short_name if row is not None else None) or label
            if name:
                limits.setdefault(name, prop.dcid)
    if not limits:
        return None
    return DepartureFact(
        code="scope_limited",
        binding_ref=f"how:{next(iter(limits.values()))}",
        limits=tuple(limits),
    )


def _unavailable_entity_facts(
    spec: Spec, *, entity_words: Mapping[str, str]
) -> list[DepartureFact]:
    """`entity_unavailable` for each role naming entities the answer's read plan serves no
    point at on that role (`ResolvedSpec.unserved_entities`, `unavailable_entity_facts`).
    On a `latest` request an entity whose data ends before the latest period the answer
    serves is unserved there (`read_plan.unserved_entities`)."""
    return unavailable_entity_facts(
        spec, unserved=frozenset(spec.resolved.unserved_entities), entity_words=entity_words
    )


def unavailable_entity_facts(
    spec: Spec,
    *,
    unserved: Collection[tuple[str, str]],
    entity_words: Mapping[str, str] = NO_ENTITY_WORDS,
) -> list[DepartureFact]:
    """`entity_unavailable` for each role of `spec` naming entities the data holds no
    values for, `unserved` as `(role, dcid)` pairs, in entity order, under the ref of the
    token naming the role's entities. An entity unserved on one role and served on another
    departs on the first alone, since a member of a group named on one role counts toward
    that group's members. Each names the entities by label, else by dcid. A role the
    request named as a group (`ResolvedSpec.entity_groups`) is asked as that group, in the
    request's words for it (`entity_words`, else its label), with its label and member
    count, since the request named none of its members. On a `latest` request each names
    the latest period the answer serves."""
    window = spec.resolved.coverage.window
    latest_years = (
        (window.start_year, window.end_year)
        if spec.definition.identity.requested_window.kind == "latest"
        and window is not None
        and window.start_year is not None
        and window.end_year is not None
        else None
    )
    groups = spec.resolved.entity_groups or {}
    names_by_role: dict[str, list[str]] = {}
    size_by_role: dict[str, int] = {}
    for entity in spec.resolved.entities:
        size_by_role[entity.role] = size_by_role.get(entity.role, 0) + 1
        if (entity.role, entity.ref) in unserved:
            names_by_role.setdefault(entity.role, []).append(entity.label or entity.ref)
    facts: list[DepartureFact] = []
    for role, names in names_by_role.items():
        group = groups.get(role)
        facts.append(
            DepartureFact(
                code="entity_unavailable",
                binding_ref=role_ref(role),
                asked=(
                    entity_words.get(group.label, group.label)
                    if group is not None
                    else ", ".join(names)
                ),
                entities=tuple(names),
                group=group.label if group is not None else None,
                group_size=size_by_role[role] if group is not None else None,
                latest_years=latest_years,
            )
        )
    return facts


def merged_entity_facts(facts: Iterable[DepartureFact]) -> list[DepartureFact]:
    """`facts` with every `entity_unavailable` fact on one `binding_ref` folded into the
    first, so a part states one per role whichever reads raised them: the Spec's read plan,
    a UN front door that refused places, or a registry that withholds every source holding
    values there. The first keeps its place, its group and its latest period, and gains the
    entities and sources it lacks in the order the others name them."""
    merged: list[DepartureFact] = []
    position: dict[str | None, int] = {}
    for fact in facts:
        if fact.code != "entity_unavailable":
            merged.append(fact)
            continue
        at = position.setdefault(fact.binding_ref, len(merged))
        if at == len(merged):
            merged.append(fact)
        else:
            merged[at] = _folded(merged[at], fact)
    return merged


def _folded(first: DepartureFact, other: DepartureFact) -> DepartureFact:
    """`first` naming the entities and sources of `other` too. A fact on no group asks
    for every entity it names."""
    entities = tuple(dict.fromkeys((*first.entities, *other.entities)))
    return dataclasses.replace(
        first,
        asked=first.asked if first.group is not None else ", ".join(entities),
        entities=entities,
        sources=tuple(dict.fromkeys((*first.sources, *other.sources))),
    )


def _modelled_fact(spec: Spec) -> DepartureFact | None:
    """`modelled_values` when the served method is modelled or estimated. Otherwise it fires
    with `partial` set when any series in the Spec's observations is modelled, or holds
    points converted with a projected exchange rate or deflator
    (`ObservationSeriesConversion.projected_count`, the `deflator_projected` warning), which
    are estimates too. Without observations there is nothing to read."""
    method = spec.resolved.facets.get("measurementMethod")
    if method is not None and method.ref in MODELLED_MEASUREMENT_METHODS:
        return DepartureFact(code="modelled_values", binding_ref=None, shown=method.label)
    observations = spec.resolved.observations
    series = observations.series if observations is not None else []
    if not any(_holds_estimates(one) for one in series):
        return None
    return DepartureFact(code="modelled_values", binding_ref=None, partial=True)


def _holds_estimates(series: ObservationSeries) -> bool:
    """Whether `series` selects a modelled or estimated method, or holds projected points."""
    facet = series.facet.get("measurementMethod")
    if facet is not None and facet.ref in MODELLED_MEASUREMENT_METHODS:
        return True
    return series.conversion is not None and series.conversion.projected_count > 0


def _year_facts(spec: Spec, *, operations: Sequence[Operation]) -> list[DepartureFact]:
    """`year_unavailable` for a named year or range, and for each operation's target year,
    that falls outside the years the data holds (`ResolvedSpec.data_years`, the served
    series' span over every date). `shown` is the part of the answer's served span (earliest
    start to latest end over the Spec's StatVars, clipped to the window) the request
    overlaps. None when the answer states no span or the data's years are unknown."""
    served = _served_span(spec)
    data_years = spec.resolved.data_years
    if served is None or data_years is None:
        return []
    requested: list[tuple[int | None, int | None]] = []
    window = spec.definition.identity.requested_window
    if isinstance(window, IdentityWindowRange):
        requested.append((window.start_year, window.end_year))
    requested.extend(
        (operation.target_year, operation.target_year)
        for operation in operations
        if operation.target_year is not None
    )
    return [
        DepartureFact(
            code="year_unavailable",
            binding_ref=WHEN_REF,
            asked=_range_text(start, end),
            shown=_span_text(_shown_years(start, end, span=served)),
            requested_years=(start, end),
            data_years=data_years,
        )
        for start, end in requested
        if _outside(start, end, first=data_years[0], last=data_years[1])
    ]


def window_gap_fact(gap: WindowGap | None) -> DepartureFact | None:
    """`year_unavailable` for a `no_observations` answer whose elected result holds data at
    the place but no point in the years read (`Materialised.window_gap`): the years asked,
    and the first and last years the series runs, `None` when it serves none. The answer
    shows nothing, so `shown` is `None`. None without a gap."""
    if gap is None:
        return None
    start, end = gap.years.start_year, gap.years.end_year
    data_years = (
        (int(gap.first[:4]), int(gap.last[:4]))
        if gap.first is not None and gap.last is not None
        else None
    )
    return DepartureFact(
        code="year_unavailable",
        binding_ref=WHEN_REF,
        asked=_range_text(start, end),
        requested_years=(start, end),
        data_years=data_years,
    )


def conversion_period_fact(
    *,
    served: tuple[int, int],
    converted: tuple[int, int],
    limited_by: Literal["exchange_rates", "deflators"],
    requested: tuple[int | None, int | None] | None,
    shown: tuple[int, int],
) -> DepartureFact:
    """`year_unavailable` for an answer a unit request converts over `converted` alone: the
    years its tables cover inside the `served` years the data holds. `shown` is the span
    the narrowed answer serves, and `requested` the range the request named, None for
    every year."""
    return DepartureFact(
        code="year_unavailable",
        binding_ref=WHEN_REF,
        asked=_range_text(*requested) if requested is not None else None,
        shown=_span_text(shown),
        requested_years=requested,
        data_years=served,
        converted_years=converted,
        limited_by=limited_by,
    )


def deflator_omission_fact(
    omitted: Sequence[tuple[str, tuple[int, ...], tuple[int, ...]]], *, unnamed: int = 0
) -> DepartureFact:
    """`year_unavailable` for an answer whose constant-price conversion leaves out the rows
    of `omitted` entities, each by name with the years it leaves out and keeps, and of
    `unnamed` entities no name is held for, cleared by another unit or base year on the
    unit token."""
    return DepartureFact(
        code="year_unavailable",
        binding_ref=UNIT_REF,
        omitted_years=tuple(omitted),
        omitted_unnamed=unnamed,
    )


def deflator_omission_undecided_fact() -> DepartureFact:
    """`year_unavailable` for a constant-price answer whose conversion may leave out rows a
    missing deflator lacks, when the loaded tables cannot say whose
    (`unit.conversion.UNDECIDED`), cleared by another unit or base year on the unit token."""
    return DepartureFact(code="year_unavailable", binding_ref=UNIT_REF, omission_undecided=True)


def _shown_years(start: int | None, end: int | None, *, span: tuple[int, int]) -> tuple[int, int]:
    """The years an answer over the requested range shows: the part of the data's `span`
    the range overlaps, or the whole span when they share no year."""
    first, last = span
    shown_first = first if start is None else max(start, first)
    shown_last = last if end is None else min(end, last)
    return (shown_first, shown_last) if shown_first <= shown_last else span


def _span_text(years: tuple[int, int]) -> str:
    """A span of years: one year, or `S to E`."""
    first, last = years
    return str(first) if first == last else f"{first} to {last}"


def _served_span(spec: Spec) -> tuple[int, int] | None:
    """The earliest start year and latest end year over the Spec's StatVars' served
    ranges, or None when no StatVar states both bounds."""
    starts: list[int] = []
    ends: list[int] = []
    for stat_var in spec.resolved.members:
        served = stat_var.data_date_range
        if served is not None and served.start and served.end:
            starts.append(int(served.start[:4]))
            ends.append(int(served.end[:4]))
    if not starts:
        return None
    return min(starts), max(ends)


def _outside(start: int | None, end: int | None, *, first: int, last: int) -> bool:
    """Whether a requested range, either bound open, reaches past the served span."""
    return any(year is not None and not first <= year <= last for year in (start, end))


def _range_text(start: int | None, end: int | None) -> str:
    """A requested range as the request names it: one year, `S to E`, `S or later`, or
    `E or earlier`."""
    if start is not None and end is not None:
        return str(start) if start == end else f"{start} to {end}"
    if start is not None:
        return f"{start} or later"
    return f"{end} or earlier"
