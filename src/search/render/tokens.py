"""Lay out a `HeadingPlan` as sentence tokens, the one builder of heading and byline tokens.

Connectives, separators, and terminal punctuation are tokens. Empty slots keep empty
tokens. Body tokens receive a lazy space, and a connective loses its leading separators
when it opens the plan. The first non-empty token is capitalised, and the heading ends with a
period unless its final text already ends in ``.``, ``!``, or ``?``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from search.heading.words import content_words
from search.models import METHOD_REF, SOURCE_REF, UNIT_REF, SentenceToken

from .heading import LATEST_YEAR_LABEL, Byline, Heading, HeadingPlan, HowValue, RolePhrase
from .labels import _entity_token, _slot_token, _text_token, token_states
from .what import _lower_first_word

_BYLINE_SEPARATOR = " · "

_SENTENCE_ENDS = (".", "!", "?")


def plan_tokens(plan: HeadingPlan) -> list[SentenceToken]:
    """The heading's tokens (`heading_tokens`), then the byline's (`byline_tokens`)."""
    heading = heading_tokens(plan.heading)
    return [
        *heading,
        *byline_tokens(plan.byline, after_heading=any(token.text for token in heading)),
    ]


PieceKind = Literal["lead", "modifier", "noun", "extra", "unit", "role", "how", "subject", "period"]
"""The heading field a `HeadingPiece` lays out."""


@dataclass(frozen=True, slots=True)
class HeadingPiece:
    """One phrase of a heading's tokens: the connective text before it, then its token.

    ``kind`` names the heading field the phrase comes from, and ``phrase`` is the role
    or subject phrase of a lead, role or subject piece. The lead piece ends with the
    space before the noun phrase.
    """

    kind: PieceKind
    tokens: tuple[SentenceToken, ...]
    phrase: RolePhrase | None = None


def plan_words(plan: HeadingPlan) -> frozenset[str]:
    """The content words of the plan's rendered heading and byline (`content_words` of
    the `plan_tokens` text), with connectives and possessive forms as the reader sees
    them. A departure's asked words are checked against this set
    (`engine.departures.stated_facts`)."""
    return content_words(sentence_from_plan(plan_tokens(plan)))


def heading_pieces(heading: Heading) -> list[HeadingPiece]:
    """The heading's tokens as `heading_tokens` lays them out, split into one piece per
    phrase and without the terminal period."""
    return _pieces(heading, period=True, capitalise=True)


def _pieces(heading: Heading, *, period: bool, capitalise: bool) -> list[HeadingPiece]:
    """The heading's tokens split into one piece per phrase, the first word capitalised
    when ``capitalise``, and the lead ending with the space before the noun phrase."""
    tokens, marks = _layout(heading, period=period)
    if capitalise:
        _capitalize_first_token(tokens)
    ends = [start for _kind, start, _phrase in marks[1:]] + [len(tokens)]
    pieces = [
        HeadingPiece(kind=kind, tokens=tuple(tokens[start:end]), phrase=phrase)
        for (kind, start, phrase), end in zip(marks, ends, strict=True)
    ]
    if pieces and pieces[0].kind == "lead":
        _move_space_into_lead(pieces)
    return pieces


def _move_space_into_lead(pieces: list[HeadingPiece]) -> None:
    """Move the space after the possessive suffix into the lead, so a lead stated once
    before several noun phrases keeps it. The space is the first token with text after
    the lead, whichever piece holds it, since a silent modifier can come first."""
    for index, piece in enumerate(pieces[1:], start=1):
        position = next((i for i, token in enumerate(piece.tokens) if token.text), None)
        if position is None:
            continue
        if piece.tokens[position].text == " ":
            pieces[0] = replace(pieces[0], tokens=(*pieces[0].tokens, piece.tokens[position]))
            remaining = piece.tokens[:position] + piece.tokens[position + 1 :]
            pieces[index] = replace(piece, tokens=remaining)
        return


def heading_tokens(
    heading: Heading, *, period: bool = True, claim: bool = False
) -> list[SentenceToken]:
    """The heading's tokens in possessive order: lead with its possessive suffix,
    modifiers, noun, extras, unit, roles, how values, subject, period.

    After a possessive lead the first word of the noun phrase is lowered when plain Title
    Case ("Kenya's female population"). ``period`` False leaves the period and its when
    token out. ``claim`` gives the form that follows a lead-in inside a sentence: no
    terminal period, and the first word lowered as `lowered_opening` lowers it. Otherwise the
    first word is capitalised and a period closes the heading.
    """
    if claim:
        return lowered_opening(_pieces(heading, period=period, capitalise=False))
    tokens, _marks = _layout(heading, period=period)
    _capitalize_first_token(tokens)
    last_text = next((token.text for token in reversed(tokens) if token.text), "")
    if last_text and not last_text.endswith(_SENTENCE_ENDS):
        tokens.append(_text_token("."))
    return tokens


type _Mark = tuple[PieceKind, int, RolePhrase | None]


def _layout(heading: Heading, *, period: bool) -> tuple[list[SentenceToken], list[_Mark]]:
    """The heading's tokens before casing and the terminal period, and where each piece
    starts: its kind, its first token's index, and its role phrase."""
    tokens: list[SentenceToken] = []
    marks: list[_Mark] = []

    def mark(kind: PieceKind, phrase: RolePhrase | None = None) -> None:
        marks.append((kind, len(tokens), phrase))

    if heading.lead is not None:
        mark("lead", heading.lead)
        tokens.append(_role_token(heading.lead))
        tokens.append(_text_token(_possessive_suffix(heading.lead.text)))
    for value in heading.modifiers:
        mark("modifier")
        _append_body_token(tokens, _how_token(value))
    noun = heading.noun
    mark("noun")
    _append_body_token(
        tokens, _slot_token(noun.text, axis="what", slot=noun.slot, states=noun.states)
    )
    if heading.lead is not None:
        _lower_first_noun_word(tokens)
    for extra in heading.extras:
        mark("extra")
        _append_body_token(tokens, _slot_token(extra, axis="what", slot=None))
    if heading.unit:
        mark("unit")
        _append_body_token(tokens, _heading_unit_token(heading))

    # The counterpart comes before a how value: "to Nigeria for basic health care".
    for role in heading.roles:
        mark("role", role)
        _append_connective(tokens, f" {role.preposition} ")
        tokens.append(_role_token(role))
    for value in heading.how:
        mark("how")
        if value.text:
            tokens.append(_text_token(value.connective))
        tokens.append(_how_token(value))

    subject = heading.subject
    if subject is not None:
        mark("subject", subject)
        if subject.preposition is not None:
            _append_connective(tokens, f" {subject.preposition} ")
            tokens.append(_role_token(subject))
        else:
            _append_body_token(tokens, _role_token(subject))

    if period and heading.period is not None:
        mark("period")
        _append_connective(tokens, " " if heading.period.kind == "in" else ", ")
        tokens.append(_slot_token(heading.period.text, axis="when", slot=heading.when_slot))
    elif period and heading.when_slot is not None:
        mark("period")
        tokens.append(_slot_token("", axis="when", slot=heading.when_slot))
    return tokens, marks


def _heading_unit_token(heading: Heading) -> SentenceToken:
    """The heading's unit token: keyed by `UNIT_REF` when it states the unit
    (`Heading.unit_states`), so a client edits the unit from it."""
    text = heading.unit or ""
    if not heading.unit_states:
        return _slot_token(text, axis="unit", slot=None)
    return SentenceToken(
        kind="slot",
        text=text,
        axis="unit",
        binding_ref=UNIT_REF,
        states=token_states(text, UNIT_REF, heading.unit_states),
    )


def byline_tokens(byline: Byline, *, after_heading: bool) -> list[SentenceToken]:
    """The byline: the latest-year label under ``latest``, the unit token (axis ``unit``,
    `UNIT_REF`, stating `Byline.unit_states` too), the source (stating `SOURCE_REF`, its
    `binding_ref` only where a source slot backs it), and the method token (axis
    ``method``, `METHOD_REF`) when the identity binds a method (`Byline.method`), joined
    by " · ", in the byline segment.

    A space parts it from a non-empty heading (``after_heading``), each item opens with a
    capital, and a period ends it unless its last text already ends a sentence. An empty
    source token stays in place.
    """
    items: list[SentenceToken] = []
    if byline.latest:
        items.append(_text_token(LATEST_YEAR_LABEL))
    if byline.unit_text:
        items.append(
            SentenceToken(
                kind="slot",
                text=byline.unit_text,
                axis="unit",
                binding_ref=UNIT_REF,
                states=token_states(byline.unit_text, UNIT_REF, byline.unit_states),
            )
        )
    source = byline.source
    source_token = (
        _slot_token(source.text, axis="source", slot=source.slot, states=(SOURCE_REF,))
        if source is not None
        else None
    )
    if source_token is not None and source_token.text:
        items.append(source_token)
    if byline.method:
        items.append(
            SentenceToken(
                kind="slot",
                text=byline.method,
                axis="method",
                binding_ref=METHOD_REF,
                states=[METHOD_REF],
            )
        )
    tokens: list[SentenceToken] = []
    for i, item in enumerate(items):
        if i or after_heading:
            tokens.append(_text_token(_BYLINE_SEPARATOR if i else " "))
        capitalised = [item]
        _capitalize_first_token(capitalised)
        tokens.extend(capitalised)
    if source_token is not None and not source_token.text:
        tokens.append(source_token)
    if items and not items[-1].text.endswith(_SENTENCE_ENDS):
        tokens.append(_text_token("."))
    return [token.model_copy(update={"segment": "byline"}) for token in tokens]


def sentence_from_plan(tokens: list[SentenceToken]) -> str:
    """Join token text into a sentence without rebuilding the plan."""
    return "".join(tok.text for tok in tokens)


def _role_token(phrase: RolePhrase) -> SentenceToken:
    """A role's entity token (`_entity_token`), or the subject's where slot token."""
    if phrase.axis is None:
        return _slot_token(phrase.text, axis="where", slot=phrase.slot)
    return _entity_token(
        phrase.text, role=phrase.axis, entity_dcids=phrase.entity_dcids, where_slot=phrase.slot
    )


def _how_token(value: HowValue) -> SentenceToken:
    """A how value's token, backed by its slot when it has one and by its property
    otherwise."""
    if value.slot is not None:
        return _slot_token(value.text, axis="how", slot=value.slot)
    return SentenceToken(kind="slot", text=value.text, axis="how", property_dcid=value.property)


def _possessive_suffix(name: str) -> str:
    """Return the possessive ending for ``name``: "'" after a final s, else "'s"."""
    return "'" if name.endswith("s") else "'s"


def _lower_first_noun_word(tokens: list[SentenceToken]) -> None:
    """Lower the first word after a possessive lead when it is plain Title Case, since
    the noun phrase no longer starts the sentence ("Kenya's female population")."""
    for i, tok in enumerate(tokens[2:], start=2):
        if tok.kind == "slot" and tok.text:
            tokens[i] = tok.model_copy(update={"text": _lower_first_word(tok.text)})
            return


# Pieces that open on an entity's name, a place or a period, whose case is the name's.
_ENTITY_PIECES = frozenset({"lead", "role", "subject", "period"})


def lowered_opening(pieces: Sequence[HeadingPiece]) -> list[SentenceToken]:
    """The pieces' tokens with the first word lowered when plain Title Case, for a
    phrase that no longer opens the sentence. A phrase whose first piece with text is a
    lead, role, subject or period keeps its case."""
    tokens = piece_tokens(pieces)
    first = next((piece for piece in pieces if any(t.text for t in piece.tokens)), None)
    if first is None or first.kind in _ENTITY_PIECES:
        return tokens
    i = next(i for i, token in enumerate(tokens) if token.text)
    tokens[i] = tokens[i].model_copy(update={"text": _lower_first_word(tokens[i].text)})
    return tokens


def piece_tokens(pieces: Sequence[HeadingPiece]) -> list[SentenceToken]:
    """The pieces' tokens in order."""
    return [token for piece in pieces for token in piece.tokens]


def _append_body_token(tokens: list[SentenceToken], token: SentenceToken) -> None:
    """Append a body token, adding one space when both it and prior text are non-empty."""
    if token.text and any(tok.text for tok in tokens):
        tokens.append(_text_token(" "))
    tokens.append(token)


def _append_connective(tokens: list[SentenceToken], text: str) -> None:
    """Append a connective, stripping leading spaces and commas when it starts the plan."""
    tokens.append(_text_token(text if any(tok.text for tok in tokens) else text.lstrip(" ,")))


def _capitalize_first_token(tokens: list[SentenceToken]) -> None:
    """Capitalise the first word of the first non-empty token, skipping punctuation.

    Leave digit-leading words and empty plans unchanged.
    """
    for i, tok in enumerate(tokens):
        if not tok.text:
            continue
        for j, ch in enumerate(tok.text):
            if ch.isalnum():
                if ch.isalpha():
                    tokens[i] = tok.model_copy(
                        update={"text": tok.text[:j] + ch.upper() + tok.text[j + 1 :]}
                    )
                break
        return
