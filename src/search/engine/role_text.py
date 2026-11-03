"""Map role words to flow axes and check them against query wording.

The door supplies each axis. This module checks it against query text (`contradicts`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

from search.models import ParsedPart, RoleAxis

# Warning for parsed role_text values absent from the lexicon.
ROLE_TEXT_NOT_RECOGNISED = "role_text_not_recognised"


_DONOR_PREPS: frozenset[str] = frozenset({"from"})
_RECIPIENT_PREPS: frozenset[str] = frozenset({"to"})
DIRECTION_PREPS: frozenset[str] = _DONOR_PREPS | _RECIPIENT_PREPS
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]")

# English caller synonyms and declared graph role names map to flow axes.
# Unknown text stays undirected and receives ROLE_TEXT_NOT_RECOGNISED upstream.
# Avoid importing materialise.roles: materialise/__init__.py loads graph-read
# dependencies. test_role_text.py checks this lexicon against _ROLE_AXES.
_ROLE_TEXT_TO_AXIS: dict[str, RoleAxis] = {
    "from": "origin",
    "origin": "origin",
    "donor": "origin",
    "provider": "origin",
    "funder": "origin",
    "source": "origin",
    "creditor": "origin",
    "lender": "origin",
    "exporter": "origin",
    "financingprovider": "origin",
    "to": "destination",
    "recipient": "destination",
    "beneficiary": "destination",
    "destination": "destination",
    "debtor": "destination",
    "borrower": "destination",
    "importer": "destination",
    "target": "destination",
    "financingrecipient": "destination",
}


def role_axis_for_text(role_text: str | None) -> RoleAxis | None:
    """Map recognised role text to its axis; unknown text is undirected."""
    if role_text is None:
        return None
    return _ROLE_TEXT_TO_AXIS.get(role_text.casefold())


# The parties a place can be in a flow or a debt. Each is a word of the lexicon above. A
# source or creditor is the origin, a recipient or debtor the destination.
Party = Literal["source", "recipient", "creditor", "debtor"]
# "lender" and "borrower" name the debt parties too.
_PARTY_WORDS: dict[str, Party] = {party: party for party in get_args(Party)} | {
    "lender": "creditor",
    "borrower": "debtor",
}
# The parties of a debt, whose repayments run the other way (`contradicts`).
DEBT_PARTIES: frozenset[Party] = frozenset({"creditor", "debtor"})


def party_for_text(role_text: str | None) -> Party | None:
    """The party a role text names, or None. A party word ("creditor", "lender") names one.
    Other synonyms such as "donor" and prepositions such as "to" name none."""
    if role_text is None:
        return None
    return _PARTY_WORDS.get(role_text.casefold())


# Canonical echo terms round-trip through the caller lexicon.
# The assertion below catches any mapping change that breaks that contract.
_AXIS_TO_ROLE_TEXT: dict[RoleAxis, str] = {"origin": "from", "destination": "to"}

assert all(role_axis_for_text(word) == axis for axis, word in _AXIS_TO_ROLE_TEXT.items()), (
    "every _AXIS_TO_ROLE_TEXT synonym must resolve back to its own axis through "
    "_ROLE_TEXT_TO_AXIS; see _leg_restatement, whose whole echo round-trip "
    "depends on it"
)


def role_text_for_axis(axis: RoleAxis) -> str:
    """Return the canonical role text for a bound axis."""
    return _AXIS_TO_ROLE_TEXT[axis]


def role_text_for(axis: RoleAxis, *, party: Party | None) -> str:
    """The role text that restates a role on `axis`: a debt `party` as its own word, which
    the lexicon reads back to the same axis and party and so draws no contradiction
    (`contradicts`), and any other role as the axis's canonical text."""
    if party is not None and party in DEBT_PARTIES:
        return party
    return role_text_for_axis(axis)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


_OPPOSITE_PREPS: dict[RoleAxis, frozenset[str]] = {
    # Opposite-axis prepositions contradict the claimed direction.
    "origin": _RECIPIENT_PREPS,
    "destination": _DONOR_PREPS,
}


_SAME_AXIS_PREPS: dict[RoleAxis, frozenset[str]] = {
    # Same-axis prepositions support the claimed direction.
    "origin": _DONOR_PREPS,
    "destination": _RECIPIENT_PREPS,
}


def span_starts(query_tokens: Sequence[str], span_tokens: Sequence[str]) -> list[int]:
    """Return each token index where ``span_tokens`` occurs as whole words. An empty span
    occurs nowhere."""
    span_len = len(span_tokens)
    if not span_len:
        return []
    return [
        i
        for i in range(len(query_tokens) - span_len + 1)
        if tuple(query_tokens[i : i + span_len]) == tuple(span_tokens)
    ]


def contradicts(query: str, *, span: str, axis: RoleAxis, party: Party | None) -> bool:
    """Check whether query wording contradicts a door-supplied axis for ``span``.

    Only the token before each exact occurrence counts. A supporting ``from``
    or ``to`` occurrence clears all contradictions; otherwise any opposite
    preposition fires. Unmatched spans and spans at position 0 return false.

    The check uses the door's verbatim span to avoid name collisions and checks
    every occurrence. For ``aid to South Sudan from Sudan``, the inner
    ``Sudan`` is preceded by ``to`` but the standalone occurrence is supported
    by ``from``. The direction corpus covers 40 cases and 88 entity labels.

    A creditor or a debtor (``party``) draws no contradiction, since repayments run
    opposite to the debt and either preposition can name either party (``owed to
    China``, ``repays to China``, ``repayments from Zambia``).
    """
    span_tokens = _tokenize(span)
    query_tokens = _tokenize(query)
    preceding = [query_tokens[i - 1] for i in span_starts(query_tokens, span_tokens) if i > 0]
    return _preceding_contradicts(preceding, axis=axis, party=party)


def _preceding_contradicts(
    preceding: Sequence[str], *, axis: RoleAxis, party: Party | None
) -> bool:
    """Whether the words straight before a place's occurrences contradict `axis`: a
    supporting ``from`` or ``to`` clears every contradiction, else an opposite one fires.
    A debt party draws none (`contradicts`)."""
    if party in DEBT_PARTIES:
        return False
    if any(word in _SAME_AXIS_PREPS[axis] for word in preceding):
        return False
    return any(word in _OPPOSITE_PREPS[axis] for word in preceding)


def role_contradicted(role: DoorRole, *, query: str) -> bool:
    """Whether the query's wording contradicts a door-supplied `role`: the words before the
    occurrences the door tied it to (`DoorRole.preceding`), else every occurrence of its
    span in `query` (`contradicts`). The pipeline's veto and the layer-2 cache signal both
    read it."""
    if role.preceding is None:
        return contradicts(query, span=role.span, axis=role.axis, party=role.party)
    return _preceding_contradicts(role.preceding, axis=role.axis, party=role.party)


def parsed_role_query(part: ParsedPart) -> str:
    """The query the parsed door checks `part`'s role text against: its variable text and
    each entity's text, space-separated so direction cues stay local."""
    return " ".join([part.variable_text, *(entity.text for entity in part.entities or ())])


def parsed_directions_contradicted(part: ParsedPart) -> bool:
    """Whether the parsed door would find one of `part`'s stated directions contradicted
    by its own wording (`contradicts` over `parsed_role_query`), which refuses
    ``role_not_resolved``."""
    query = parsed_role_query(part)
    for entity in part.entities or ():
        axis = role_axis_for_text(entity.role_text)
        if axis is not None and contradicts(
            query, span=entity.text, axis=axis, party=party_for_text(entity.role_text)
        ):
            return True
    return False


# Words that ask for every member of what follows them: "all donors", "every provider".
_EVERY_WORDS: frozenset[str] = frozenset({"all", "every", "each"})
# The every-member words that ask for one value per member, a breakdown along the axis:
# "each recipient". "all recipients" is also met by the total over them.
_BREAKDOWN_WORDS: frozenset[str] = frozenset({"every", "each"})
# Nouns that name the places on either side of a flow without naming the side: "all
# countries". The preposition that governs them names it.
_PLACE_NOUNS: frozenset[str] = frozenset({"country", "countries"})
# The axis each role preposition names.
_ROLE_PREP_AXES: dict[str, RoleAxis] = {prep: _ROLE_TEXT_TO_AXIS[prep] for prep in DIRECTION_PREPS}


def _noun_axis(token: str) -> RoleAxis | None:
    """The axis a role noun names, singular or plural ("donor", "beneficiaries"). A
    preposition names none here, since it governs the phrase and is no noun of it."""
    if token in _ROLE_PREP_AXES:
        return None
    return role_axis_for_text(token) or role_axis_for_text(_singular(token))


def _singular(token: str) -> str:
    """The singular of a plural noun ("beneficiaries", "donors"); other tokens lose only a
    final "s"."""
    return token[:-3] + "y" if token.endswith("ies") else token.removesuffix("s")


def _governing_axes(query_tokens: list[str], phrase_tokens: list[str]) -> set[RoleAxis]:
    """The axes the role prepositions directly before each occurrence of the phrase name."""
    return {
        axis
        for start in span_starts(query_tokens, phrase_tokens)
        if start > 0 and (axis := _ROLE_PREP_AXES.get(query_tokens[start - 1])) is not None
    }


def _after_role_prep(tokens: list[str]) -> list[str]:
    """`tokens` past a leading role preposition ("to all recipients" -> "all recipients")."""
    return tokens[1:] if tokens and tokens[0] in _ROLE_PREP_AXES else tokens


def every_member_breakdown(phrase: str) -> bool:
    """Whether an every-member ``phrase`` (`every_member_axis`) asks for one value per
    member of its axis: ``each recipient`` and ``to every provider`` do, while ``all
    recipients`` is also met by the total over them."""
    rest = _after_role_prep(_tokenize(phrase))
    return bool(rest) and rest[0] in _BREAKDOWN_WORDS


def every_member_axis(phrase: str, *, query: str) -> RoleAxis | None:
    """The flow axis whose every member ``phrase`` asks for, or None.

    The phrase is an optional ``from`` or ``to``, then ``all``, ``every`` or ``each``, an
    optional ``the``, and nouns that each name a role (``all donors``, ``every provider``,
    ``to all creditors``) or a place on either side (``all countries``, ``all donor
    countries``). A role noun names the axis, so ``to all creditors`` asks for every
    origin. Place nouns alone take the axis of the preposition that governs them: the
    phrase's own, else the one before the phrase in ``query``, which must name a single
    axis. Any other word, two role nouns on opposite axes, or no axis gives None, so ``in
    all sectors`` and ``all bilateral donors`` ask for no role.
    """
    tokens = _tokenize(phrase)
    prep_axis = _ROLE_PREP_AXES.get(tokens[0]) if tokens else None
    rest = _after_role_prep(tokens)
    if not rest or rest[0] not in _EVERY_WORDS:
        return None
    named = _nouns_axes(rest[1:])
    if named is None:
        return None
    if not named:
        named = {prep_axis} if prep_axis is not None else _governing_axes(_tokenize(query), tokens)
    return next(iter(named)) if len(named) == 1 else None


def _nouns_axes(tokens: list[str]) -> set[RoleAxis] | None:
    """The axes the nouns of `tokens`, past an optional ``the``, name: empty for place
    nouns alone (``countries``), None for no noun or any word that names neither a role nor
    a place."""
    nouns = tokens[1:] if tokens[:1] == ["the"] else tokens
    if not nouns:
        return None
    named: set[RoleAxis] = set()
    for token in nouns:
        if token in _PLACE_NOUNS:
            continue
        axis = _noun_axis(token)
        if axis is None:
            return None
        named.add(axis)
    return named


# Connectives that ask for one value per member of the role nouns after them: "by donor",
# "across recipients". "for" asks it only before "each" or "every" ("for each donor").
_BREAKDOWN_CONNECTIVES: frozenset[str] = frozenset({"by", "across"})


def breakdown_axis(phrase: str, *, query: str) -> RoleAxis | None:
    """The flow axis whose members ``phrase`` asks one value each for, or None.

    The phrase is ``by`` or ``across`` and role nouns (``by donor``, ``across the
    recipients``, ``by creditor``), or an every-member phrase that asks for the breakdown
    (`every_member_axis`, `every_member_breakdown`: ``each donor``, ``from every
    provider``), after an optional ``by``, ``across`` or ``for`` (``for each donor``). Place
    nouns alone after ``by`` or ``across`` name no side (``by country``), and ``all``
    phrases ask for no breakdown, since the total over the members meets them.
    """
    tokens = _tokenize(phrase)
    if (
        tokens
        and tokens[0] in _BREAKDOWN_CONNECTIVES
        and not _BREAKDOWN_WORDS.intersection(tokens[1:2])
    ):
        named = _nouns_axes(tokens[1:])
        return next(iter(named)) if named is not None and len(named) == 1 else None
    return every_member_breakdown_axis(phrase, query=query)


def every_member_breakdown_axis(phrase: str, *, query: str) -> RoleAxis | None:
    """The flow axis whose members an each or every ``phrase`` asks one value each for, or
    None: the every-member half of `breakdown_axis` (``each donor``, ``for each donor``, ``by
    every recipient``), which asks for the breakdown whatever operation carries it. ``by``
    or ``across`` and role nouns alone (``by donor``) give None."""
    tokens = _tokenize(phrase)
    if tokens and (tokens[0] in _BREAKDOWN_CONNECTIVES or tokens[0] == "for"):
        tokens = tokens[1:]
    rest = " ".join(tokens)
    return every_member_axis(rest, query=query) if every_member_breakdown(rest) else None


@dataclass(frozen=True, slots=True)
class DoorRole:
    """One occurrence of a door-supplied flow axis and its verbatim text span.

    Raw-text spans fall back to entity text after translation; parsed text is
    preserved in its role query. A name can occur on both axes, so each
    occurrence has its own entry. ``stated`` is whether the request's own words give
    the place its direction. A parsed part's role text always does. On raw text a direction
    is stated when the extractor's cue (`ExtractedEntity.cue`) reaches beyond the place's
    own words (`raw_text._states_direction`). A possessive, "of" or a national adjective
    reads the direction off the measure, so it states none. ``members`` is
    whether the occurrence asks for the members of the set the name names
    (`ExtractedEntity.members`, `ParsedEntity.members`), on its own end alone
    (`asks_members`). ``party`` is the place's party when the door names one
    (`ExtractedEntity.role`, `party_for_text`), which `contradicts` reads. ``preceding``
    is the word straight before each occurrence of the place the raw-text door ties the
    role to (`extract.veto_words`), which `role_contradicted` reads in place of the role
    query, so another measure's wording of the same place does not veto it. None reads
    every occurrence in the role query, as the parsed door's part-local query does.
    """

    axis: RoleAxis
    span: str
    stated: bool = True
    members: bool = False
    party: Party | None = None
    # The evidence a role is checked against, so two mentions alike in every reading
    # compare equal whatever occurrences each is tied to.
    preceding: tuple[str, ...] | None = field(default=None, compare=False)


def asks_members(roles: Sequence[DoorRole], axis: RoleAxis | None, *, asked: bool) -> bool:
    """Whether a name whose directed occurrences are `roles` asks for its set's members on
    the `axis` end, where `asked` is whether any occurrence of the name asks for them
    (`ResolveState.member_entities`).

    An occurrence that asks for the members asks on its own end, so one name can read a
    place as its members on one end and whole on the other. A name no directed occurrence
    asks the members of, whose ask came with no end, asks them on every end it has.
    """
    if not any(role.members for role in roles):
        return asked
    return any(role.members and role.axis == axis for role in roles)
