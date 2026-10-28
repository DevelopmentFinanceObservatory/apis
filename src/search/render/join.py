"""Join several parts' headings into one heading.

A qualifier every heading states once and alike (the counterpart, the subject place, the
period) is stated once after the joined phrases, a possessive lead every heading shares
opens the sentence once, and headings that differ only in their lead share one noun
("USA's and Britain's health ODA to Ethiopia"). Headings are compared by their typed
phrases: a role or subject by its axis, preposition, text, entity dcids and slot key, the
period by its `Period` and when slot key. `render.tokens.heading_pieces` lays each heading
out, so this module adds only the Oxford connectives and the closing period.

Every token is returned with its owner: the index of the part it belongs to, or None for
text the parts share. The caller stamps owners as part indices once.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from search.models import SentenceToken, Slot

from .heading import Heading, RolePhrase
from .labels import _text_token, join_labels
from .tokens import (
    _SENTENCE_ENDS,
    HeadingPiece,
    _capitalize_first_token,
    heading_pieces,
    lowered_opening,
    piece_tokens,
)

type Owned = tuple[int | None, SentenceToken]
"""A token and the index of the part it belongs to, None when the parts share it."""


@dataclass(frozen=True, slots=True)
class _Qualifier:
    """A piece after the noun phrase. ``axis`` is the qualifier axis it hoists under
    ("origin", "where", "when"), None for a how value that stays in its phrase. ``key``
    is the typed value two parts must share to state it once."""

    axis: str | None
    key: Hashable
    piece: HeadingPiece

    @property
    def has_text(self) -> bool:
        return any(token.text for token in self.piece.tokens)


@dataclass(frozen=True, slots=True)
class _Split:
    """One part's heading pieces: the possessive lead, the noun phrase up to the first
    role, subject or period, and the pieces after it."""

    owner: int
    lead: HeadingPiece | None
    noun: tuple[HeadingPiece, ...]
    tail: tuple[_Qualifier, ...]
    span_period: bool


def join_headings(headings: Sequence[tuple[int, Heading]]) -> list[Owned] | None:
    """The headings joined into one, closed by its period, or None when the join cannot
    be written. ``headings`` pairs each heading with the index of its part.

    A qualifier hoists when every heading states it once and alike (`_shared_axes`) and
    after every value it does not share (`_trailing_axes`). It is written once after the
    joined phrases, in the first heading's order. The rest stay in their own phrase, in
    its order, so a lone heading reads as itself. The join is refused when a period span
    stays inside a phrase, where its comma would read as a list separator, and when two
    phrases would state the same noun (`_joined_phrases`).
    """
    splits = [_split(owner, heading) for owner, heading in headings]
    shared = _trailing_axes(splits, _shared_axes(splits))
    inline = [
        [item for item in split.tail if item.axis is None or item.axis not in shared]
        for split in splits
    ]
    if any(
        split.span_period and any(item.axis == "when" for item in items)
        for split, items in zip(splits, inline, strict=True)
    ):
        return None

    composed = _joined_phrases(splits, inline)
    if composed is None:
        return None
    composed.extend(
        (None, token)
        for item in splits[0].tail
        if item.axis is not None and item.axis in shared
        for token in item.piece.tokens
    )
    composed = capitalised(composed)
    last = next((token.text for _owner, token in reversed(composed) if token.text), "")
    if last and not last.endswith(_SENTENCE_ENDS):
        composed.append((None, _text_token(".")))
    return composed


def join_phrases(phrases: Sequence[Sequence[Owned]]) -> list[Owned]:
    """Oxford-join the phrases with text. A phrase without text keeps its place."""
    visible = sum(1 for phrase in phrases if _has_text(phrase))
    connectives = _oxford_connectives(visible)
    composed: list[Owned] = []
    seen = 0
    for phrase in phrases:
        if _has_text(phrase):
            if seen:
                composed.append((None, _text_token(connectives[seen - 1])))
            seen += 1
        composed.extend(phrase)
    return composed


def _split(owner: int, heading: Heading) -> _Split:
    pieces = heading_pieces(heading)
    lead = pieces[0] if pieces and pieces[0].kind == "lead" else None
    rest = pieces[1:] if lead is not None else pieces
    first = next(
        (i for i, piece in enumerate(rest) if piece.kind in ("role", "subject", "period")),
        len(rest),
    )
    tail = tuple(_qualifier(piece, heading) for piece in rest[first:])
    span_period = heading.period is not None and heading.period.kind != "in"
    return _Split(owner, lead, tuple(rest[:first]), tail, span_period)


def _qualifier(piece: HeadingPiece, heading: Heading) -> _Qualifier:
    if piece.kind == "period":
        return _Qualifier("when", (heading.period, _slot_ref(heading.when_slot)), piece)
    if piece.phrase is not None:
        axis = "origin" if piece.phrase.axis == "origin" else "where"
        return _Qualifier(axis, _phrase_key(piece.phrase), piece)
    return _Qualifier(None, None, piece)


def _phrase_key(phrase: RolePhrase) -> Hashable:
    """What a role, subject or lead states: two parts state it alike only when every
    field agrees, since equal labels can stand for other entities or relations."""
    return (
        phrase.axis,
        phrase.preposition,
        phrase.text,
        phrase.entity_dcids,
        _slot_ref(phrase.slot),
    )


def _slot_ref(slot: Slot | None) -> str | None:
    return slot.key.as_ref() if slot is not None else None


def _shared_axes(splits: list[_Split]) -> frozenset[str]:
    """The qualifier axes every heading states once and alike."""
    shared: set[str] = set()
    for item in splits[0].tail:
        if item.axis is None:
            continue
        matches = [[other for other in split.tail if other.axis == item.axis] for split in splits]
        if all(len(found) == 1 for found in matches) and all(
            found[0].key == matches[0][0].key for found in matches[1:]
        ):
            shared.add(item.axis)
    return frozenset(shared)


def _trailing_axes(splits: list[_Split], axes: frozenset[str]) -> frozenset[str]:
    """The `axes` every heading states after each piece outside them, so that moving such
    a qualifier past the joined phrases keeps every phrase's own order. A piece with no
    text (an empty placeholder) holds no place in the reading order."""
    trailing = axes
    while True:
        narrowed = trailing
        for split in splits:
            suffix: set[str] = set()
            for item in reversed(split.tail):
                if item.axis is not None and item.axis in narrowed:
                    suffix.add(item.axis)
                elif item.has_text:
                    break
            narrowed &= suffix
        if narrowed == trailing:
            return trailing
        trailing = narrowed


def _joined_phrases(splits: list[_Split], inline: list[list[_Qualifier]]) -> list[Owned] | None:
    """The parts' phrases joined, a lead or noun they all share stated once, or None when
    two phrases would state the same noun.

    With two or more parts, a possessive lead every part shares opens the phrases once
    ("Brazil's GDP and population"), and parts whose phrases differ only in their lead
    join the leads before one shared noun ("USA's and Britain's health ODA").
    """
    bodies = [
        [*split.noun, *(item.piece for item in items)]
        for split, items in zip(splits, inline, strict=True)
    ]
    body_tokens = [piece_tokens(body) for body in bodies]
    leads = [split.lead for split in splits]
    lead_keys = {_phrase_key(lead.phrase) if lead and lead.phrase else None for lead in leads}
    several_leads = len(splits) > 1 and all(leads)

    if several_leads and len(lead_keys) > 1 and all(t == body_tokens[0] for t in body_tokens):
        # Each lead keeps its entity token and possessive suffix. The space before the
        # shared noun is the first lead's.
        possessives = [
            owned(split.owner, lead.tokens[:-1])
            for split, lead in zip(splits, leads, strict=True)
            if lead is not None
        ]
        first = leads[0]
        assert first is not None
        return [*join_phrases(possessives), *owned(None, [first.tokens[-1], *body_tokens[0]])]

    nouns = [
        "".join(token.text for token in piece_tokens(split.noun)).casefold() for split in splits
    ]
    if len(set(nouns)) < len(nouns):
        return None

    if several_leads and len(lead_keys) == 1:
        first = leads[0]
        assert first is not None
        phrases = [
            owned(split.owner, lowered_opening(body) if position else piece_tokens(body))
            for position, (split, body) in enumerate(zip(splits, bodies, strict=True))
        ]
        return [*owned(None, first.tokens), *join_phrases(phrases)]

    phrases = []
    for position, (split, body) in enumerate(zip(splits, bodies, strict=True)):
        pieces = [split.lead, *body] if split.lead is not None else body
        phrases.append(
            owned(split.owner, lowered_opening(pieces) if position else piece_tokens(pieces))
        )
    return join_phrases(phrases)


def _oxford_connectives(count: int) -> list[str]:
    """The separators `join_labels` puts between ``count`` items."""
    if count < 2:
        return []
    sentinels = [f"\x00{i}\x00" for i in range(count)]
    joined = join_labels(sentinels)
    return re.split("|".join(re.escape(sentinel) for sentinel in sentinels), joined)[1:-1]


def capitalised(owned: list[Owned]) -> list[Owned]:
    """The tokens with the first word of the first one with text capitalised."""
    tokens = [token for _owner, token in owned]
    _capitalize_first_token(tokens)
    return [(owner, token) for (owner, _old), token in zip(owned, tokens, strict=True)]


def owned(owner: int | None, tokens: Sequence[SentenceToken]) -> list[Owned]:
    return [(owner, token) for token in tokens]


def _has_text(phrase: Sequence[Owned]) -> bool:
    return any(token.text for _owner, token in phrase)
