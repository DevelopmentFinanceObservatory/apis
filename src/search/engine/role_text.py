"""Map parsed role text to flow axes and check it against query wording.

The door supplies each axis. This module checks it against query text, and reads
the direction of a debt (`debt_axes`) and of a receipt (`receipt_axes`) from the query's
own wording.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from search.models import ParsedPart, RoleAxis

# Warning for parsed role_text values absent from the lexicon.
ROLE_TEXT_NOT_RECOGNISED = "role_text_not_recognised"


_DONOR_PREPS: frozenset[str] = frozenset({"from"})
_RECIPIENT_PREPS: frozenset[str] = frozenset({"to"})
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


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


_OPPOSITE_AXIS: dict[RoleAxis, RoleAxis] = {"origin": "destination", "destination": "origin"}
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


# Words that govern a creditor through "to": "owed to", "debt to", "repaid to".
_CREDITOR_CUE_HEADS: frozenset[str] = frozenset(
    {
        "owe",
        "owed",
        "owes",
        "debt",
        "debts",
        "repaid",
        "repay",
        "repayment",
        "repayments",
        "liabilities",
        "borrowed",
    }
)
# Nouns that name a debt: "Kenya's debt", "Ghana's liabilities".
_DEBT_NOUNS: frozenset[str] = frozenset({"debt", "debts", "liabilities", "repayment", "repayments"})
# After a debt noun these name an action on the debt, which either party can hold:
# "China's debt relief to Kenya" and "Kenya's debt relief" both occur.
_DEBT_ACTION_WORDS: frozenset[str] = frozenset(
    {
        "relief",
        "forgiveness",
        "cancellation",
        "rescheduling",
        "restructuring",
        "refinancing",
        "swap",
        "swaps",
    }
)
# Active forms of "owe", whose subject is the debtor and whose object is the creditor:
# "how much does China owe India". The past "owed" is left out, since "China owed by
# Kenya" is passive and reverses the parties.
_OWE_VERBS: frozenset[str] = frozenset({"owe", "owes"})
# Words before a "by" that make the place after it the debtor: "owed by Kenya".
_DEBTOR_BY_HEADS: frozenset[str] = frozenset(
    {"owed", "repaid", "borrowed", "incurred", "debt", "debts"}
)
# Words that close a debt phrase before "to": "debt service to", "debt payments to".
_DEBT_PHRASE_TAILS: frozenset[str] = frozenset(
    {"service", "servicing", "payment", "payments", "stock", "stocks"}
)
# Prepositions that end a noun phrase.
_PHRASE_END_PREPS: frozenset[str] = _DONOR_PREPS | _RECIPIENT_PREPS | {"by", "of", "on"}
# A possessed debt phrase ended by one of these names a claim on another party:
# "China's debt claims on Africa", "China's holdings of Kenyan debt".
_CLAIM_PREPS: frozenset[str] = frozenset({"of", "on"})
_POSSESSIVE = "'s"
_CONJUNCTIONS: frozenset[str] = frozenset({"and", "or"})


def _ends_phrase(token: str) -> bool:
    """Whether ``token`` ends a noun phrase: a preposition or a possessive."""
    return token in _PHRASE_END_PREPS or token.endswith(_POSSESSIVE)


def _span_starts(query_tokens: list[str], span_tokens: list[str]) -> list[int]:
    """Return each token index where ``span_tokens`` occurs as whole words."""
    span_len = len(span_tokens)
    return [
        i
        for i in range(len(query_tokens) - span_len + 1)
        if query_tokens[i : i + span_len] == span_tokens
    ]


def _names_debt(tokens: list[str]) -> bool:
    """Whether ``tokens`` hold a debt noun that no action word follows."""
    return any(
        token in _DEBT_NOUNS and (i + 1 == len(tokens) or tokens[i + 1] not in _DEBT_ACTION_WORDS)
        for i, token in enumerate(tokens)
    )


def _follows_creditor_cue(
    query_tokens: list[str], start: int, debtor_ends: Collection[int] = ()
) -> bool:
    """Whether the place starting at ``start`` follows a creditor cue such as ``owed to``.

    The ``to`` follows an owe or debt word, a tail word of a debt phrase (``debt service
    payments to``), or a debtor place (``owed by Kenya to China``; ``debtor_ends`` holds the
    token index after each such place). An article between the cue and the place is
    skipped: ``repaid to the IMF``.
    """
    prep = start - 1
    if prep >= 0 and query_tokens[prep] == "the":
        prep -= 1
    if prep < 1 or query_tokens[prep] != "to":
        return False
    before = query_tokens[prep - 1]
    if before in _CREDITOR_CUE_HEADS or prep in debtor_ends:
        return True
    if before not in _DEBT_PHRASE_TAILS:
        return False
    phrase_start = prep
    while phrase_start > 0 and not _ends_phrase(query_tokens[phrase_start - 1]):
        phrase_start -= 1
    return _names_debt(query_tokens[phrase_start:prep])


def _possessed_phrase(query_tokens: list[str], after: int) -> tuple[list[str], str | None]:
    """Return the phrase a possessive governs from ``after``, and the word that ends it.

    The phrase ends at a preposition in ``_PHRASE_END_PREPS`` or a second possessive; the
    ending word is None when the phrase runs to the end of the query.
    """
    end = after
    while end < len(query_tokens):
        if _ends_phrase(query_tokens[end]):
            return query_tokens[after:end], query_tokens[end]
        end += 1
    return query_tokens[after:end], None


def _follows_debtor_by(query_tokens: list[str], start: int, creditor_ends: set[int]) -> bool:
    """Whether the place starting at ``start`` follows a debtor ``by``.

    The ``by`` follows an owe or debt word (``owed by``, ``debt incurred by``) or ends a
    creditor place (``owed to China by Kenya``). An article before the place is skipped.
    """
    prep = start - 1
    if prep >= 0 and query_tokens[prep] == "the":
        prep -= 1
    return (
        prep >= 1
        and query_tokens[prep] == "by"
        and (query_tokens[prep - 1] in _DEBTOR_BY_HEADS or prep in creditor_ends)
    )


def _owns_debt(query_tokens: list[str], after: int) -> bool | None:
    """Whether a possessive ending at ``after`` names its possessor's debt; None for a claim.

    The governed phrase names the debt when it holds a debt noun. A phrase that is empty or
    only a conjunction, before the query's end or another possessive, shares the debt the
    other possessive names: ``Kenya's debt versus Ghana's``, ``Kenya's and Ghana's debt``.
    A phrase that ``of`` or ``on`` ends names a claim on another party's debt (``Kenya's
    share of Africa's debt``), which gives None.
    """
    phrase, ended_by = _possessed_phrase(query_tokens, after)
    if ended_by in _CLAIM_PREPS:
        return None
    if _names_debt(phrase):
        return True
    shares = ended_by is None or ended_by.endswith(_POSSESSIVE)
    return shares and all(token in _CONJUNCTIONS for token in phrase)


def _follows_owe(query_tokens: list[str], start: int) -> bool:
    """Whether the place starting at ``start`` is the object of an active ``owe``, as
    ``India`` in ``China owe India``. An article before the place is skipped: ``owe the
    IMF``."""
    verb = start - 1
    if verb >= 0 and query_tokens[verb] == "the":
        verb -= 1
    return verb >= 0 and query_tokens[verb] in _OWE_VERBS


def debt_axes(
    query: str,
    *,
    measures: Sequence[str],
    places: Mapping[str, Sequence[str]],
    directed: Mapping[str, RoleAxis],
) -> dict[str, RoleAxis]:
    """Return the axis the query's debt wording sets for each place, by place name.

    A debt runs from the creditor, origin, to the debtor, destination. The rule reads one
    two-party relation: a single measure that names a debt (``Kenya's debt to China``) and
    exactly two places, each listed with its verbatim spans. Any other query gets {}.

    A place after a creditor ``to`` (``debt to``, ``owed to``, ``repaid to``, or ``to``
    after the debtor in ``owed by Kenya to China``) is the creditor. A possessive place
    over its own debt (``Kenya's debt``, ``Kenya's debt service payments``) and a place
    after a debtor ``by`` (``owed by``, or ``by`` after the creditor in ``owed to China by
    Kenya``) are the debtor. An active ``owe`` reads both parties: its subject is the debtor
    and its object the creditor (``how much does China owe India``, ``what does Zambia owe
    the IMF``). Each place needs its own evidence of being a party: a cue, or
    an extracted axis (``directed`` maps those places to their extracted axes). A cued
    place sets its axis. A place with no cue keeps an extracted axis opposite the cued one,
    so ``Kenya debt owed to China`` sets both parties. A place with no cue whose
    extracted axis equals the cued one is a second debtor or creditor, as in ``Kenya's debt
    relative to Sub-Saharan Africa``, and a place with neither, as in ``Kenya's external
    debt compared with China``: both leave the query a comparison and set nothing. So does
    a possessive claim on another party's debt
    (``Kenya's share of Africa's debt``), two cues that clash (``Kenya's debt versus
    Ghana's``), and a place both cues claim. Cues match whole words.
    """
    if len(measures) != 1 or not _names_debt(_tokenize(measures[0])) or len(places) != 2:
        return {}
    query_tokens = _tokenize(query.replace("\u2019", "'"))
    # (name, start, end) for every whole-word occurrence of every span.
    occurrences = [
        (name, start, start + len(tokens))
        for name, spans in places.items()
        for tokens in (_tokenize(span) for span in spans)
        if tokens
        for start in _span_starts(query_tokens, tokens)
    ]
    creditor_ends = {
        end for _, start, end in occurrences if _follows_creditor_cue(query_tokens, start)
    }
    debtor_by = [
        (name, end)
        for name, start, end in occurrences
        if _follows_debtor_by(query_tokens, start, creditor_ends)
    ]
    debtor_ends = {end for _, end in debtor_by}
    creditors = {
        name
        for name, start, _ in occurrences
        if _follows_creditor_cue(query_tokens, start, debtor_ends)
        or _follows_owe(query_tokens, start)
    }
    debtors = {name for name, _ in debtor_by}
    debtors.update(
        name
        for name, _, end in occurrences
        if end < len(query_tokens) and query_tokens[end] in _OWE_VERBS
    )
    for name, spans in places.items():
        for tokens in (_tokenize(span) for span in spans):
            if not tokens:
                continue
            possessive = [*tokens[:-1], tokens[-1].removesuffix(_POSSESSIVE) + _POSSESSIVE]
            for start in _span_starts(query_tokens, possessive):
                owns = _owns_debt(query_tokens, start + len(possessive))
                if owns is None:
                    return {}
                if owns:
                    debtors.add(name)
    if creditors & debtors or len(creditors) > 1 or len(debtors) > 1:
        return {}
    cued: dict[str, RoleAxis] = dict.fromkeys(creditors, "origin")
    cued.update(dict.fromkeys(debtors, "destination"))
    if not cued:
        return {}
    for name in places:
        if name not in cued:
            (other,) = cued.values()
            if directed.get(name) != _OPPOSITE_AXIS[other]:
                return {}
            cued[name] = _OPPOSITE_AXIS[other]
    return cued


# Forms of "receive" that read the place beside them as the recipient: "received by
# Ethiopia", "the ODA Ghana received".
_RECEIPT_VERBS: frozenset[str] = frozenset({"receive", "receives", "received", "receiving"})
_RECEIPT_AGENT_PREP = "by"
# A preposition before a place makes it the head of its own phrase, so a receipt verb
# after it reads another noun: "aid from Norway received by Kenya".
_PLACE_PHRASE_PREPS: frozenset[str] = _PHRASE_END_PREPS | {"in", "for"}


def receipt_axes(query: str, *, places: Mapping[str, Sequence[str]]) -> dict[str, RoleAxis]:
    """Return ``destination`` for each place the query names as a recipient, by place name.

    A place right after ``received by`` (an article skipped: ``received by the Gambia``)
    and a place right before a form of ``receive`` (``the ODA Ghana received``) is the
    recipient. A place a preposition opens is the head of its own phrase, so the verb after
    it reads another noun: in ``aid from Norway received by Kenya`` only Kenya receives.
    Spans match whole words, and every occurrence of a place's spans is read.
    """
    query_tokens = _tokenize(query.replace("\u2019", "'"))
    recipients: dict[str, RoleAxis] = {}
    for name, spans in places.items():
        for tokens in (_tokenize(span) for span in spans):
            if not tokens:
                continue
            for start in _span_starts(query_tokens, tokens):
                end = start + len(tokens)
                prep = start - 1
                if prep >= 0 and query_tokens[prep] == "the":
                    prep -= 1
                after_agent = (
                    prep >= 1
                    and query_tokens[prep] == _RECEIPT_AGENT_PREP
                    and query_tokens[prep - 1] in _RECEIPT_VERBS
                )
                before_verb = (
                    end < len(query_tokens)
                    and query_tokens[end] in _RECEIPT_VERBS
                    and not (prep >= 0 and query_tokens[prep] in _PLACE_PHRASE_PREPS)
                )
                if after_agent or before_verb:
                    recipients[name] = "destination"
    return recipients


def contradicts(query: str, *, span: str, axis: RoleAxis) -> bool:
    """Check whether query wording contradicts a door-supplied axis for ``span``.

    Only the token before each exact occurrence counts. A supporting ``from``
    or ``to`` occurrence clears all contradictions; otherwise any opposite
    preposition fires. Unmatched spans and spans at position 0 return false.

    The check uses the door's verbatim span to avoid name collisions and checks
    every occurrence. For ``aid to South Sudan from Sudan``, the inner
    ``Sudan`` is preceded by ``to`` but the standalone occurrence is supported
    by ``from``. The direction corpus covers 40 cases and 88 entity labels.

    A ``to`` that closes a creditor cue leaves an origin uncontradicted: in
    ``Kenya's debt owed to China`` the words ``owed to`` make China the creditor.
    Every other ``to`` is checked as usual, so ``aid to Kenya`` claimed as origin
    contradicts.
    """
    span_tokens = _tokenize(span)
    if not span_tokens:
        return False
    query_tokens = _tokenize(query)
    opposite_preps = _OPPOSITE_PREPS[axis]
    same_preps = _SAME_AXIS_PREPS[axis]
    saw_contradiction = False
    for i in _span_starts(query_tokens, span_tokens):
        if i == 0:
            continue
        preceding = query_tokens[i - 1]
        if preceding in same_preps:
            return False  # a supporting occurrence clears every contradiction
        if preceding not in opposite_preps:
            continue
        if axis == "origin" and _follows_creditor_cue(query_tokens, i):
            continue
        saw_contradiction = True
    return saw_contradiction


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
        if axis is not None and contradicts(query, span=entity.text, axis=axis):
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
_ROLE_PREP_AXES: dict[str, RoleAxis] = {
    prep: _ROLE_TEXT_TO_AXIS[prep] for prep in _DONOR_PREPS | _RECIPIENT_PREPS
}


def _noun_axis(token: str) -> RoleAxis | None:
    """The axis a role noun names, singular or plural ("donor", "beneficiaries"). A
    preposition names none here, since it governs the phrase and is no noun of it."""
    if token in _ROLE_PREP_AXES:
        return None
    singular = token[:-3] + "y" if token.endswith("ies") else token.removesuffix("s")
    return role_axis_for_text(token) or role_axis_for_text(singular)


def _governing_axes(query_tokens: list[str], phrase_tokens: list[str]) -> set[RoleAxis]:
    """The axes the role prepositions directly before each occurrence of the phrase name."""
    return {
        axis
        for start in _span_starts(query_tokens, phrase_tokens)
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
    lead = tokens[0] if tokens else None
    if lead in _BREAKDOWN_CONNECTIVES and not set(tokens[1:2]) & _BREAKDOWN_WORDS:
        named = _nouns_axes(tokens[1:])
        return next(iter(named)) if named is not None and len(named) == 1 else None
    if lead in _BREAKDOWN_CONNECTIVES or lead == "for":
        tokens = tokens[1:]
    rest = " ".join(tokens)
    return every_member_axis(rest, query=query) if every_member_breakdown(rest) else None


@dataclass(frozen=True, slots=True)
class DoorRole:
    """One occurrence of a door-supplied flow axis and its verbatim text span.

    Raw-text spans fall back to entity text after translation; parsed text is
    preserved in its role query. A name can occur on both axes, so each
    occurrence has its own entry. ``stated`` is whether the request's own words give
    the place its direction. A parsed part's role text always does, and so does a debt
    axis (`debt_axes`) or a receipt axis (`receipt_axes`). On raw text, any other direction is stated when the extractor's
    validated cue (`ExtractedEntity.role_span`) attaches to the place, as a cue before it
    or a verb after it (`raw_text._cue_states_direction`). A possessive, "of" or a national
    adjective reads the direction off the measure, so it states none.
    """

    axis: RoleAxis
    span: str
    stated: bool = True
