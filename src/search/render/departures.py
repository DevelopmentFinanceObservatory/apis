"""Departure text: one or two short sentences per way a part's answer differs from its
request, in the form "You asked for X. This shows Y." Every text but `modelled_values`,
which states a fact about the values, and `entity_unavailable` on a group, whose
`group_read_as_members` departure states the ask, is in the second person.

`departures_from_facts` turns a part's `DepartureFact`s into its `Departure`s. A part that
shows no data (a no-data answer) gets text that claims nothing is shown: "This checked
..." or "This check left it out." in place of "This shows ...".

What the request asked appears in its own wording (`DepartureFact.asked`), with these
exceptions. An operation is named by its verb ("a share") and a breakdown by the role noun
of the answer's flow family ("a breakdown by creditor"), since their recorded wording is a
fragment of the question ("share of ODA going to Kenya out of"). A price basis is stated
without its leading preposition ("constant 2022 dollars" for "in constant 2022 dollars").
A measure phrase the answer does not apply is the user's own words and is quoted
("You asked for "gross"."). `Departure.asked` keeps the recorded wording in every case.

A fact missing a value its code's text states raises `ValueError`. Every constructor in
`search.engine.departures` sets them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final

from code_index.labels import HOW_PHRASE_LEADS_WITH_PREPOSITION
from code_index.types import PartyKind
from dfo_platform.roles import GENERIC_ROLE_WORDING, FlowFamily, role_wording
from search.heading.departure_fact import DepartureFact, ordered_facts
from search.models import (
    UNIT_REF,
    Departure,
    DepartureCode,
    EntityGroup,
    OperationVerb,
    RoleAxis,
    SentenceToken,
)

from .labels import _join_labels, group_members_text, group_text

_OPERATION_NOUNS: Final[Mapping[OperationVerb, tuple[str, str]]] = MappingProxyType(
    {
        "rank": ("a ranking", "rankings"),
        "aggregate": ("a total", "totals"),
        "growth": ("a growth rate", "growth rates"),
        "share_of": ("a share", "shares"),
        "explain": ("an explanation", "explanations"),
        "forecast": ("a forecast", "forecasts"),
        "report": ("a report", "reports"),
    }
)
"""Each operation verb as an article noun and its plural."""

type _TextWriter = Callable[[DepartureFact, FlowFamily | None, bool], str]
"""A departure code's text writer: the fact, the flow family of the part's answer, and
whether the part shows data."""


def departures_from_facts(
    facts: Iterable[DepartureFact], *, family: FlowFamily | None, shows_data: bool
) -> list[Departure]:
    """The part's `departures`: one per distinct fact, in `DEPARTURE_ORDER`, each with its
    text. `family` is the flow family of the part's answer, which names the roles a
    breakdown is over. `shows_data` is whether the part shows data, false on a no-data
    part."""
    return [
        Departure(
            code=fact.code,
            binding_ref=fact.binding_ref,
            asked=fact.asked,
            shown=fact.shown,
            text=departure_text(fact, family=family, shows_data=shows_data),
        )
        for fact in ordered_facts(facts)
    ]


def departure_text(fact: DepartureFact, *, family: FlowFamily | None, shows_data: bool) -> str:
    """The text a person reads for `fact` on a part that shows data (`shows_data`) or on
    one that shows none."""
    return _TEXT_BY_CODE[fact.code](fact, family, shows_data)


def no_data_limit_text(limits: Sequence[str]) -> str:
    """The departure of a no-data answer whose check covered only `limits`, the values
    the engine bound without the request naming them ("the education sector", "ODA"):
    "This check covered only the education sector and ODA."
    """
    if not limits:
        raise ValueError("a no-data limit names at least one value")
    return f"This check covered only {_join_labels(list(limits))}."


def departure_tokens(texts: Sequence[str], *, after_text: bool) -> list[SentenceToken]:
    """Each departure sentence in ``texts`` as a ``departure``-segment text token, in
    order, with a space token before each one that follows text (``after_text`` for the
    first)."""
    tokens: list[SentenceToken] = []
    for position, text in enumerate(texts):
        if position or after_text:
            tokens.append(SentenceToken(kind="text", text=" ", segment="departure"))
        tokens.append(SentenceToken(kind="text", text=text, segment="departure"))
    return tokens


def _closest_match(fact: DepartureFact, _family: FlowFamily | None, shows_data: bool) -> str:
    """The closest measure shown or checked. A part offering several readings in place of
    the measure asked (`DepartureFact.readings`) says each is one the data holds. A door
    holding no words for the measure (a resubmit, `asked` is `None`) states only the
    measure shown, or that the readings are the closest measures."""
    verb = "shows" if shows_data else "checked"
    if fact.asked is None:
        if fact.readings:
            return "These readings are the closest measures the data holds."
        if fact.shown is None:
            raise ValueError("closest_match with nothing asked needs the measure shown")
        return f"This {verb} {fact.shown}, the closest measure the data holds."
    asked = _asked(fact)
    if fact.readings:
        return f"You asked for {asked}. These readings are the closest measures the data holds."
    if fact.shown is None:
        return f"You asked for {asked}. This {verb} the closest measure the data holds."
    return f"You asked for {asked}. This {verb} {fact.shown}, the closest measure the data holds."


def _measure_not_applied(fact: DepartureFact, _family: FlowFamily | None, shows_data: bool) -> str:
    """A price basis (the unit token's ref) is stated without its leading preposition. Any
    other dropped phrase is the user's own words, quoted. With no words (`asked` `None`: a
    resubmit, or a value no name exists for) it names what was left out
    (`DepartureFact.values`)."""
    if fact.binding_ref == UNIT_REF:
        if fact.shown is None:
            raise ValueError("a price basis departure needs the price basis shown")
        verb = "shows" if shows_data else "checked"
        return f"You asked for {_without_preposition(_asked(fact))}. This {verb} {fact.shown}."
    if fact.asked is None:
        if not fact.values:
            raise ValueError("measure_not_applied with nothing asked needs the values left out")
        values = _join_labels(list(fact.values))
        if shows_data:
            return f"This shows the measure without {values}."
        return f"This check left out {values}."
    asked = f'You asked for "{_asked(fact)}".'
    if shows_data:
        return f"{asked} This shows the measure without it."
    return f"{asked} This check left it out."


def _without_preposition(phrase: str) -> str:
    """`phrase` without one leading preposition ("in constant 2022 dollars")."""
    first, _, rest = phrase.partition(" ")
    return rest if rest and first.lower() in HOW_PHRASE_LEADS_WITH_PREPOSITION else phrase


def _operation_not_applied(
    fact: DepartureFact, _family: FlowFamily | None, shows_data: bool
) -> str:
    if fact.verb is None:
        raise ValueError("operation_not_applied needs the operation's verb")
    noun, plural = _OPERATION_NOUNS[fact.verb]
    if not shows_data:
        return f"You asked for {noun}. Search does not compute {plural}."
    return (
        f"You asked for {noun}. This shows the amounts behind it, "
        f"because Search does not compute {plural}."
    )


def _breakdown_not_applied(fact: DepartureFact, family: FlowFamily | None, shows_data: bool) -> str:
    if fact.over is None:
        raise ValueError("breakdown_not_applied needs the flow axis it is over")
    noun = role_wording(family).noun(fact.over).singular
    total = "This is the total" if shows_data else "This checked the total"
    return f"You asked for a breakdown by {noun}. {total}; name a {noun} to see one."


def _entity_unavailable(fact: DepartureFact, _family: FlowFamily | None, _shows_data: bool) -> str:
    """The entities the data holds no values for, by name. On a role the request named as
    a group, a count of the group as the heading names it (`labels.group_text`), since the
    request named none of its members. The group's `group_read_as_members` departure
    states the ask, so this one does not repeat it. On a role that binds the members of
    sets the request named beside other places (`DepartureFact.sets`), the places named
    one by one are named and each set's members are counted. When no place is named, the
    sentence opens with the sets asked for. On a `latest` request it names the latest
    period the answer serves (`_latest_period`). Sources that hold values for the entities
    but that Search does not read (`DepartureFact.sources`) close it."""
    # The closing sentence's pronoun counts places, and a place named one by one can also
    # be counted under its set.
    unserved = len({*fact.entities, *(e for members in fact.sets for e in members.entities)})
    if not unserved:
        raise ValueError("entity_unavailable needs the entities the data holds no values for")
    period = _latest_period(fact.latest_years)
    pronoun = "it" if unserved == 1 else "them"
    sentences: list[str] = []
    if fact.group is not None and fact.group_size is not None:
        group = EntityGroup(ref=fact.group_dcid, label=fact.group)
        members = group_text(fact.group_size, group)
        sentences.append(f"The data holds no values{period} for {len(fact.entities)} of {members}.")
    elif fact.entities:
        named = "it" if len(fact.entities) == 1 else "them"
        sentences.append(
            f"You asked for {_join_labels(list(fact.entities))}. "
            f"The data holds no values for {named}{period}."
        )
    if fact.sets:
        counts = [
            f"{len(members.entities)} of "
            + group_text(members.size, EntityGroup(ref=members.dcid, label=members.label))
            for members in fact.sets
        ]
        if not fact.entities and fact.asked is not None:
            sentences.append(f"You asked for {_join_labels([m.asked for m in fact.sets])}.")
        sentences.append(f"The data holds no values{period} for {_join_labels(counts)}.")
    text = " ".join(sentences)
    if not fact.sources:
        return text
    holds = "holds" if len(fact.sources) == 1 else "hold"
    return (
        f"{text} Search does not read {_join_labels(list(fact.sources))}, "
        f"which {holds} values for {pronoun}."
    )


def _latest_period(years: tuple[int, int] | None) -> str:
    """The phrase naming a latest period, with its leading space: "in 2024" for one year,
    "in the latest period" for one spanning several (series whose latest points fall in
    different years), and nothing without one."""
    if years is None:
        return ""
    first, last = years
    return f" in {last}" if first == last else " in the latest period"


def _year_unavailable(fact: DepartureFact, _family: FlowFamily | None, shows_data: bool) -> str:
    """The years asked against the years the data holds. An answer that shows no years
    (a no-data part, or `shown` is `None`) says only what the data holds. A door that
    holds no words for the years (a resubmit, `asked` is `None`) states only what the data
    holds and shows. An answer a unit request converts over fewer years than the data holds
    names the table that leaves the rest out (`_conversion_years`). One whose deflators
    leave out some entities' rows names them (`_omitted_years`)."""
    if fact.omitted_years or fact.omitted_unnamed or fact.omission_undecided:
        return _omitted_years(fact)
    lead = f"You asked for {_asked(fact)}. " if fact.asked is not None else ""
    if fact.limited_by is not None:
        return lead + _conversion_years(fact)
    if fact.requested_years is None:
        raise ValueError("year_unavailable needs the requested years")
    shown = fact.shown if shows_data else None
    if fact.data_years is None:
        if shown is not None:
            raise ValueError("year_unavailable showing years needs the data's years")
        return f"{lead}The data holds no values for {_those_years(fact)}."
    first, last = fact.data_years
    span = str(first) if first == last else f"{first} to {last}"
    start, end = fact.requested_years
    years = [year for year in fact.requested_years if year is not None]
    after = any(year > last for year in years)
    before = any(year < first for year in years)
    if shown is None:
        if (start is not None and start > last) or (end is not None and end < first):
            return f"{lead}The data covers only {span}."
        return f"{lead}The data covers {span} but holds no values for {_those_years(fact)}."
    if after and before:
        return f"{lead}The data covers only {span}."
    bound = f"ends in {last}" if after else f"starts in {first}"
    return f"{lead}The data {bound}, so this shows {shown}."


_CONVERSION_TABLES: Final[Mapping[str, str]] = MappingProxyType(
    {"exchange_rates": "Exchange rates", "deflators": "Deflators"}
)


def _conversion_years(fact: DepartureFact) -> str:
    """ "Exchange rates start in 1980, so this shows 1980 to 2024.": the table that leaves
    out the data's other years, and where its years stop."""
    if (
        fact.limited_by is None
        or fact.converted_years is None
        or fact.data_years is None
        or fact.shown is None
    ):
        raise ValueError("year_unavailable on a conversion needs its years and those shown")
    first, last = fact.converted_years
    data_first, data_last = fact.data_years
    table = _CONVERSION_TABLES[fact.limited_by]
    if first > data_first and last < data_last:
        bound = f"cover only {first} to {last}"
    elif first > data_first:
        bound = f"start in {first}"
    else:
        bound = f"end in {last}"
    return f"{table} {bound}, so this shows {fact.shown}."


_MAX_OMITTED_NAMED: Final = 3
"""The most entities a deflator omission names before it counts the rest."""


def _omitted_years(fact: DepartureFact) -> str:
    """ "Constant prices leave out Taiwan before 2023, Azerbaijan before 2022, and Cyprus
    before 2018, which the price deflators do not cover.": the first entities the fact
    names, each with the years it loses, then a count of the rest, those it holds no name
    for included. With no name, "some years of 2 entities"; for a conversion the tables
    cannot read, "Constant prices may leave out years the price deflators do not cover."."""
    if fact.omission_undecided:
        return "Constant prices may leave out years the price deflators do not cover."
    named = [
        _entity_years(name, lost, kept)
        for name, lost, kept in fact.omitted_years[:_MAX_OMITTED_NAMED]
    ]
    rest = len(fact.omitted_years) - len(named) + fact.omitted_unnamed
    if not named:
        entities = "entity" if rest == 1 else "entities"
        return (
            f"Constant prices leave out some years of {rest} {entities}, which the price "
            "deflators do not cover."
        )
    if rest:
        named.append(f"{rest} {'other' if rest == 1 else 'others'}")
    return (
        f"Constant prices leave out {_join_labels(named)}, which the price deflators do not cover."
    )


def _entity_years(name: str, lost: Sequence[int], kept: Sequence[int]) -> str:
    """ "Taiwan before 2023", "Taiwan after 2010", "Taiwan in 1995 to 1999 and 2003", or
    "Taiwan" when it keeps no year."""
    if not kept:
        return name
    if max(lost) < min(kept):
        return f"{name} before {min(kept)}"
    if min(lost) > max(kept):
        return f"{name} after {max(kept)}"
    runs: list[list[int]] = []
    for year in sorted(lost):
        if runs and year == runs[-1][1] + 1:
            runs[-1][1] = year
        else:
            runs.append([year, year])
    spans = [str(first) if first == last else f"{first} to {last}" for first, last in runs]
    return f"{name} in {_join_labels(spans)}"


def _those_years(fact: DepartureFact) -> str:
    """ "that year" for one requested year, else "those years". When no text names them
    first (`asked` is `None`), "the year requested" and "the years requested"."""
    assert fact.requested_years is not None
    start, end = fact.requested_years
    one = start is not None and start == end
    if fact.asked is None:
        return "the year requested" if one else "the years requested"
    return "that year" if one else "those years"


def _modelled_values(fact: DepartureFact, _family: FlowFamily | None, _shows_data: bool) -> str:
    return (
        "Some of these values are modelled estimates."
        if fact.partial
        else "These values are modelled estimates."
    )


def _group_read_as_members(
    fact: DepartureFact, _family: FlowFamily | None, shows_data: bool
) -> str:
    """The group asked for, read as its members, which a region group names as the places
    in it (`labels.group_members_text`), and the residuals its total counts that the members
    leave out (`DepartureFact.excluded`), separated by semicolons since each label holds a
    comma, or that the source publishes its own figure for the group, which can differ from
    one built from its members (`DepartureFact.sum_differs`). A door holding no words for the
    group (a resubmit, `asked` is `None`) states only the members shown, as the heading names
    them (`labels.group_text`)."""
    verb = "shows" if shows_data else "checked"
    left_out = (
        f", and {'leaves' if shows_data else 'left'} out what the data allocates to no single "
        f"country: "
        f"{'; '.join(fact.excluded)}"
        if fact.excluded
        else ""
    )
    if fact.sum_differs:
        left_out += (
            ", and the source publishes its own figure for the group, "
            "which can differ from one built from its members"
        )
    if fact.asked is not None:
        members = group_members_text(fact.group_dcid)
        return f"You asked for {_asked(fact)}. This {verb} {members}, each on its own{left_out}."
    if fact.group is None or fact.group_size is None:
        raise ValueError("group_read_as_members with nothing asked needs the group shown")
    group = EntityGroup(ref=fact.group_dcid, label=fact.group)
    members = group_text(fact.group_size, group)
    return f"This {verb} {members}, each on its own{left_out}."


WHOLE_MEMBER_NOUNS: Final[Mapping[RoleAxis, str]] = MappingProxyType(
    {"origin": "donor", "destination": "recipient"}
)
"""The `open_roles` word naming the members of each side a whole population is read on."""


def whole_population_is_official(axis: RoleAxis, whole_of: Collection[PartyKind]) -> bool:
    """Whether a whole population of the `whole_of` kinds on `axis` holds the official
    donors alone: a donor side holding no private donor."""
    return axis == "origin" and "private" not in whole_of


def whole_population_text(axis: RoleAxis, whole_of: Collection[PartyKind]) -> str:
    """The members a whole population of the `whole_of` kinds on `axis` holds: "every
    official donor" for a donor side holding no private donor, else "every donor" or
    "every recipient"."""
    noun = WHOLE_MEMBER_NOUNS.get(axis, "member")
    official = whole_population_is_official(axis, whole_of)
    return f"every official {noun}" if official else f"every {noun}"


def whole_residual_text(whole_of: Collection[PartyKind]) -> str:
    """The clause saying a whole population of the `whole_of` kinds holds aid allocated to
    no country, empty when it holds no residual."""
    return ", which includes aid not allocated to a country" if "residual" in whole_of else ""


def _whole_total_read(fact: DepartureFact, _family: FlowFamily | None, shows_data: bool) -> str:
    """The whole population the side shows (`DepartureFact.group`), what it holds, and the
    way back to the side's members. A request that named nothing on the side reads "No donor
    was named, so this shows ..."; one that named it ("all donors") is asked in its words,
    and its way back replaces those words with `open_roles`. One whose `aggregate` over the
    side the total answered (`DepartureFact.verb`) drops it on the way back. A union of totals names each
    (`DepartureFact.entities`). A donor total that holds no private donor says they are
    outside it, and a total holding residuals says it holds aid allocated to no country."""
    if fact.over is None or fact.group is None:
        raise ValueError("whole_total_read needs the side and the population it shows")
    noun = WHOLE_MEMBER_NOUNS.get(fact.over, "member")
    verb = "shows" if shows_data else "checked"
    if fact.entities:
        shown = " plus ".join(fact.entities)
    else:
        every = whole_population_text(fact.over, fact.whole_of)
        holds = whole_residual_text(fact.whole_of)
        official = whole_population_is_official(fact.over, fact.whole_of)
        outside = " Private donors are outside it." if official else ""
        shown = f"{fact.group}, the source's total for {every}{holds}.{outside}"
    if not shown.endswith("."):
        shown += "."
    if fact.asked is not None:
        lead = f"You asked for {_asked(fact)}. This {verb}"
        instead = f" in place of {_asked(fact)}"
    elif fact.inferred:
        lead = f"No {noun} was named, so this {verb}"
        instead = ""
    else:
        lead = f"This {verb}"
        instead = ""
    if fact.verb == "aggregate":
        instead += f" and no aggregate over the {noun}s"
    return f"{lead} {shown} To list each {noun}, search again with open_roles ['{noun}']{instead}."


def _whole_total_unpublished(
    fact: DepartureFact, _family: FlowFamily | None, shows_data: bool
) -> str:
    """The whole population asked for, on a side the dataset (`DepartureFact.shown`)
    publishes no total of, which the answer leaves unnamed: it covers every party the
    dataset holds there with data, the private donors and the aid allocated to no country
    among them where the dataset holds those (`DepartureFact.whole_of`). A part that shows
    no data states it for the closest match it shows, which holds data
    (`only_alternate_has_data`)."""
    if fact.over is None or fact.shown is None:
        raise ValueError("whole_total_unpublished needs the side and the dataset")
    noun = WHOLE_MEMBER_NOUNS.get(fact.over, "member")
    among = ", private donors among them" if "private" in fact.whole_of else ""
    holds = whole_residual_text(fact.whole_of)
    subject = "this" if shows_data else "the closest match"
    return (
        f"You asked for {_asked(fact)}. {fact.shown} publishes no total of every {noun}, so "
        f"{subject} covers every {noun} it holds with data{among}{holds}."
    )


def _direction_discarded(fact: DepartureFact, _family: FlowFamily | None, shows_data: bool) -> str:
    if fact.over is None:
        raise ValueError("direction_discarded needs the flow axis the place was named on")
    # The answer's measure has no flow roles, so no family names them.
    noun = GENERIC_ROLE_WORDING.noun(fact.over)
    return (
        f"You named {_asked(fact)} as a {noun.singular}. This measure has no "
        f"{noun.plural}, so this {'shows' if shows_data else 'checked'} the place itself."
    )


def _direction_inferred(fact: DepartureFact, family: FlowFamily | None, _shows_data: bool) -> str:
    if fact.over is None:
        raise ValueError("direction_inferred needs the flow axis the place was read on")
    noun = role_wording(family).noun(fact.over).singular
    return (
        f"You named {_asked(fact)} without a direction. Only the data with {_asked(fact)} as "
        f"the {noun} holds values, so this reads it as the {noun}."
    )


def _scope_limited(fact: DepartureFact, _family: FlowFamily | None, _shows_data: bool) -> str:
    return no_data_limit_text(fact.limits)


def _asked(fact: DepartureFact) -> str:
    """What the request named, as it worded it."""
    if fact.asked is None:
        raise ValueError(f"{fact.code} needs what the request asked")
    return fact.asked.strip()


_TEXT_BY_CODE: Final[Mapping[DepartureCode, _TextWriter]] = MappingProxyType(
    {
        "closest_match": _closest_match,
        "measure_not_applied": _measure_not_applied,
        "operation_not_applied": _operation_not_applied,
        "breakdown_not_applied": _breakdown_not_applied,
        "entity_unavailable": _entity_unavailable,
        "year_unavailable": _year_unavailable,
        "modelled_values": _modelled_values,
        "group_read_as_members": _group_read_as_members,
        "whole_total_read": _whole_total_read,
        "whole_total_unpublished": _whole_total_unpublished,
        "direction_discarded": _direction_discarded,
        "direction_inferred": _direction_inferred,
        "scope_limited": _scope_limited,
    }
)
"""The text writer of each departure code."""
