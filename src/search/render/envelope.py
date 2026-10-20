"""Compose the envelope's sentence plan from the response parts.

Definite parts join their headings into one sentence (`render.join.join_headings`).
Headings that cannot join, among them two that would join one noun to itself, stay one
sentence each. The byline follows once when every part's byline reads the same, and each
part's departures close the sentence in part order. Otherwise the envelope writes one
sentence per part, each followed by its own departures. A sentence two parts state alike
(a heading, a byline) is stated once, with no part index, and so is a departure two parts
state after the same sentence.

The parts' assumptions close the sentence as one clause ("Assumed: official development
assistance; every provider."). A joined sentence has one clause naming every part's items
in part order, each item under its part and an item two parts share under none. A sentence
written part by part closes each part's sentence with its own clause after its departures.
A mixed answer with a definite part opens "Answered N of M parts.", with no part index.

A part folded into an earlier part's Spec (`binding_folded`) states that part's heading and
byline, so their tokens carry no part index. Its assumptions, built from its own request,
can differ from that part's, so the clause names the items of both: an item every one of
them carries is unowned, and an item one alone carries is that part's.

A definite part's heading and byline come from its Spec (`heading_for`), with the
declared how-axis names its own sentence plan was rendered with (`declared_axis_names`).
Tokens carry their owning part while the sentence is built and receive their part index
once, at the end. The module has no I/O and adds only fixed connective text.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from search.models import (
    CandidatesPart,
    DefinitePart,
    NoDataPart,
    Part,
    SentenceSegment,
    SentenceToken,
    UnsupportedPart,
)

from .assumptions import clause_tokens
from .heading import Byline, HeadingPlan, heading_for
from .join import Owned, capitalised, join_headings, join_phrases, owned
from .labels import _text_token
from .no_data import composed_variable_outcome
from .tokens import (
    _SENTENCE_ENDS,
    _capitalize_first_token,
    byline_tokens,
    heading_tokens,
    plan_tokens,
    sentence_from_plan,
)
from .what import _lower_first_word


def compose_envelope_sentence(parts: Sequence[Part]) -> tuple[str, list[SentenceToken]] | None:
    """Return the envelope sentence and plan, or None when no part has a sentence.

    Omit parts folded into an earlier Spec, whose assumptions join that part's clause
    (`_fold_groups`). A lone part that is not definite is mirrored
    verbatim. Definite parts use general composition, even when alone, so shared tokens
    have no part index. Token indices always refer to the original ``parts`` list,
    including gaps from folded parts. A mixed answer opens with how many parts it answered
    (`_opener`).
    """
    contributing = [(index, part) for index, part in enumerate(parts) if not _is_folded(part)]
    if not contributing or not any(_has_own_sentence(part) for _, part in contributing):
        return None
    if len(contributing) == 1 and not isinstance(contributing[0][1], DefinitePart):
        index, part = contributing[0]
        composed = owned(index, _own_plan(part))
    else:
        plans = {
            index: heading_for(part.spec)
            for index, part in contributing
            if isinstance(part, DefinitePart)
        }
        groups = _fold_groups(parts, contributing)
        composed = _unowned_by_folds(
            _compose_general(contributing, plans, groups),
            frozenset(index for index, group in groups.items() if len(group) > 1),
        )
    plan = [
        token.model_copy(update={"part_index": owner})
        for owner, token in [*_opener(parts), *composed]
    ]
    return sentence_from_plan(plan), plan


def _opener(parts: Sequence[Part]) -> list[Owned]:
    """ "Answered N of M parts." and a space, unowned heading tokens, when the parts' statuses
    differ and N, the definite parts with folded ones, is at least one. M is every part.
    Otherwise none: each part's sentence states its own outcome."""
    answered = sum(isinstance(part, DefinitePart) for part in parts)
    if len({part.status for part in parts}) < 2 or not answered:
        return []
    opener = f"Answered {answered} of {len(parts)} parts."
    return owned(None, [_text_token(opener), _text_token(" ")])


def _is_folded(part: Part) -> bool:
    """Return whether a binding_folded warning marks the part as folded."""
    return any(warning.code == "binding_folded" for warning in part.warnings)


def _has_own_sentence(part: Part) -> bool:
    """Return whether the part's sentence plan is present.

    Definite parts store it under ``spec.resolved``. Other statuses store it on the part.
    """
    if isinstance(part, DefinitePart):
        return part.spec.resolved.sentence_plan is not None
    return part.sentence_plan is not None


def _fold_groups(
    parts: Sequence[Part], contributing: list[tuple[int, Part]]
) -> dict[int, list[tuple[int, Part]]]:
    """Each contributing part by index, with itself first and then, for a definite part,
    every later part folded into it, in part order. A folded part carries the Spec of the
    part it folded into, the first part of its fold group, which is never separate; a
    separate part can carry the same spec_id and takes no folds."""
    groups = {index: [(index, part)] for index, part in contributing}
    folded_into: dict[str, int] = {}
    for index, part in contributing:
        if isinstance(part, DefinitePart) and not part.separate:
            folded_into.setdefault(part.spec.spec_id, index)
    for index, part in enumerate(parts):
        if (
            isinstance(part, DefinitePart)
            and _is_folded(part)
            and (first := folded_into.get(part.spec.spec_id)) is not None
            and first < index
        ):
            groups[first].append((index, part))
    return groups


def _assumption_items(parts: Sequence[tuple[int, Part]]) -> list[tuple[int | None, str]]:
    """Each assumption text `parts` carry, in part order, under the one part that carries
    it, or under none when several do: a part and the parts folded into it state one
    sentence, so an item every one of them carries is stated once for all."""
    holders: dict[str, set[int]] = {}
    for index, part in parts:
        for item in part.assumptions:
            holders.setdefault(item.text, set()).add(index)
    return [
        (next(iter(owners)) if len(owners) == 1 else None, text) for text, owners in holders.items()
    ]


def _unowned_by_folds(composed: list[Owned], folded_into: frozenset[int]) -> list[Owned]:
    """`composed` with the heading and byline tokens of each part in ``folded_into``
    unowned, since every part folded into it states them. Departures keep their part, and
    an assumption keeps the owner `_assumption_items` gave it."""
    return [
        (None, token) if owner in folded_into and token.segment not in _TRAILING else (owner, token)
        for owner, token in composed
    ]


def _compose_general(
    contributing: list[tuple[int, Part]],
    plans: Mapping[int, HeadingPlan],
    groups: Mapping[int, list[tuple[int, Part]]],
) -> list[Owned]:
    """One joined sentence when every part is definite and it can be written, else one
    sentence per part. A lone definite part takes this path so shared-token provenance
    stays clear. ``groups`` holds each contributing part with the parts folded into it
    (`_fold_groups`), whose assumptions its sentence states."""
    if len(plans) == len(contributing):
        joined = _compose_joined(contributing, plans, groups)
        if joined is not None:
            return joined
    return _compose_per_part(contributing, plans, groups)


def _compose_joined(
    contributing: list[tuple[int, Part]],
    plans: Mapping[int, HeadingPlan],
    groups: Mapping[int, list[tuple[int, Part]]],
) -> list[Owned] | None:
    """The definite parts' headings, then the byline they share, then their departures,
    then one assumptions clause (`_clause`) naming every part's items in part order.
    Return None when the bylines differ, since one byline after the headings could not
    say which part each item is for. Bylines agree when their tokens with text are equal,
    refs included: two units that read alike but state different refs (one
    ``how:measurementQualifier``, the other ``axis:measurementQualifier``) differ, so
    the one byline stated for every part states refs that are right for each.

    The headings join into one (`join_headings`) or, when they cannot, follow one another
    as sentences of their own, a heading two parts state alike stated once. Then comes the
    byline once, unowned. An empty source placeholder every part shares is written once,
    unowned, and otherwise each stays with its own part. Each part's departures follow
    under its own part, a text two parts carry stated once, unowned. An assumption two
    parts carry alike, a folded part among them, is named once, unowned.
    """
    headings = {index: heading_tokens(plans[index].heading) for index, _part in contributing}
    after_heading = {
        index: any(token.text for token in tokens) for index, tokens in headings.items()
    }
    bylines = {
        index: byline_tokens(plans[index].byline, after_heading=after_heading[index])
        for index in headings
    }
    first_index = contributing[0][0]
    shown = [token for token in bylines[first_index] if token.text]
    if any([token for token in tokens if token.text] != shown for tokens in bylines.values()):
        return None

    composed = join_headings([(index, plans[index].heading) for index, _part in contributing])
    if composed is None:
        composed = _joined_sentences(
            _stated_once([owned(index, tokens) for index, tokens in headings.items()])
        )
    composed.extend(owned(None, shown))
    placeholders = {
        index: [token for token in tokens if not token.text] for index, tokens in bylines.items()
    }
    if len({_placeholder_key(plans[index].byline) for index in bylines}) == 1:
        composed.extend(owned(None, placeholders[first_index]))
    else:
        composed.extend(
            (index, token) for index, tokens in placeholders.items() for token in tokens
        )
    departures = _stated_once(
        [_departure_sentence(text, index) for index, part in contributing for text in _texts(part)]
    )
    clause = _clause(
        _assumption_items([member for index, _part in contributing for member in groups[index]])
    )
    return _joined_sentences([*departures, *([clause] if clause else [])], before=composed)


def _placeholder_key(byline: Byline) -> Hashable:
    """The empty source slot a byline keeps in place, keyed by its slot, or () when none."""
    source = byline.source
    if source is None or source.text:
        return ()
    return (source.slot.key.as_ref() if source.slot else None,)


def _compose_per_part(
    contributing: list[tuple[int, Part]],
    plans: Mapping[int, HeadingPlan],
    groups: Mapping[int, list[tuple[int, Part]]],
) -> list[Owned]:
    """Write one sentence per part in part order, each followed by its departures and its
    assumptions clause.

    A definite part's sentence is its heading and byline, and its clause names the
    assumptions of the parts folded into it too (`_assumption_items`). A part that is not definite
    contributes `_unresolved_sentence`, and no-data parts refused for the same reason name
    their variables in one sentence (`_shared_refusals`), written where the first of them
    stands. A sentence two parts state alike with the same clause is stated once, unowned,
    followed by each part's departures; a part whose sentence repeats another's beside a
    different clause keeps its own sentence (`_statements_stated_once`).
    """
    refusals = _shared_refusals(contributing)
    chunks: list[list[Owned]] = []
    for index, part in contributing:
        if isinstance(part, DefinitePart):
            sentence = owned(index, plan_tokens(plans[index]))
            texts = _texts(part)
            clause = _clause(_assumption_items(groups[index]))
        else:
            tokens = _unresolved_sentence(part)
            sentence = refusals.get(
                index,
                owned(index, [token for token in tokens if token.segment not in _TRAILING]),
            )
            texts = [
                token.text for token in tokens if token.segment == "departure" and token.text != " "
            ]
            clause = owned(index, _own_clause(tokens))
        if sentence:
            chunks.append(sentence)
        chunks.extend(_departure_sentence(text, index) for text in texts)
        if clause:
            chunks.append(clause)
    return _joined_sentences(_statements_stated_once(chunks))


def _statements_stated_once(chunks: list[list[Owned]]) -> list[list[Owned]]:
    """`chunks` in order as statements, each a sentence followed by its departures, then
    its clauses. A sentence equal to an earlier statement's, with the same clauses, is
    dropped: the earlier statement loses its owners and takes the dropped one's departures
    after its own. A sentence whose clauses differ keeps its own statement, so each clause
    follows the sentence of its part. A departure or clause equal to one already in its
    statement is dropped and that one loses its owner. A departure stated after another
    part's sentence is kept, so the reader sees which part each limit belongs to."""
    statements: list[_Statement] = []
    for chunk in chunks:
        if _is_trailing(chunk) and statements:
            statements[-1].add(chunk)
        else:
            statements.append(_Statement(chunk))
    kept: list[_Statement] = []
    for statement in statements:
        earlier = next((seen for seen in kept if seen.repeated_by(statement)), None)
        if earlier is None:
            kept.append(statement)
        else:
            earlier.absorb(statement)
    return [chunk for statement in kept for chunk in statement.chunks()]


class _Statement:
    """A sentence with the departures and clauses that follow it. A statement opened by a
    departure or clause has no sentence."""

    def __init__(self, chunk: list[Owned]) -> None:
        self.sentence: list[Owned] = []
        self.departures: list[list[Owned]] = []
        self.clauses: list[list[Owned]] = []
        if _is_trailing(chunk):
            self.add(chunk)
        else:
            self.sentence = chunk

    def add(self, chunk: list[Owned]) -> None:
        """Append a departure or clause, or unown an equal one already here."""
        run = self.clauses if chunk[0][1].segment == "assumption" else self.departures
        earlier = next((i for i, seen in enumerate(run) if _same_text(seen, chunk)), None)
        if earlier is None:
            run.append(chunk)
        else:
            run[earlier] = owned(None, _tokens(chunk))

    def repeated_by(self, other: _Statement) -> bool:
        """Whether `other` states this sentence again with the same clauses."""
        return (
            bool(self.sentence)
            and _same_text(self.sentence, other.sentence)
            and len(self.clauses) == len(other.clauses)
            and all(_same_text(a, b) for a, b in zip(self.clauses, other.clauses, strict=True))
        )

    def absorb(self, other: _Statement) -> None:
        """Take `other`, a repeat of this statement: the sentence and clauses lose their
        owners, and `other`'s departures follow this one's."""
        self.sentence = owned(None, _tokens(self.sentence))
        self.clauses = [owned(None, _tokens(clause)) for clause in self.clauses]
        for departure in other.departures:
            self.add(departure)

    def chunks(self) -> list[list[Owned]]:
        return [*([self.sentence] if self.sentence else []), *self.departures, *self.clauses]


def _shared_refusals(contributing: list[tuple[int, Part]]) -> dict[int, list[Owned]]:
    """The sentence of each no-data part whose refusal names its variable in a frame
    (`composed_variable_outcome`) shared with another part, keyed by part index. The
    first such part's sentence names every variable ("No data for debt share and debt as
    a flow ..."), each slot owned by its part and the frame unowned. The others have
    none. A part whose plan continues past its refusal keeps its own sentence."""
    framed: dict[tuple[str, str], list[tuple[int, NoDataPart]]] = {}
    for index, part in contributing:
        if not isinstance(part, NoDataPart) or not part.variable_text:
            continue
        frame = composed_variable_outcome(part.no_data.reason)
        rest = [t for t in (part.sentence_plan or [])[1:] if t.segment not in _TRAILING]
        if frame is not None and not rest:
            framed.setdefault(frame, []).append((index, part))
    sentences: dict[int, list[Owned]] = {}
    for (prefix, suffix), group in framed.items():
        if len(group) < 2:
            continue
        slots = [[(index, _variable_slot(part))] for index, part in group]
        tokens: list[Owned] = [(None, _text_token(prefix))] if prefix else []
        tokens.extend([*join_phrases(slots), (None, _text_token(suffix))])
        (first, _), *others = group
        sentences[first] = capitalised(tokens)
        sentences.update((index, []) for index, _part in others)
    return sentences


def _unresolved_sentence(part: Part) -> list[SentenceToken]:
    """The sentence a part that is not definite contributes beside other parts.

    A candidates part names its variable as a ``what`` slot before its own sentence
    ("GDP has 3 readings. ..."). A no-data part whose reason quotes the variable frames
    the variable as that slot (`composed_variable_outcome`). Any other no-data plan
    names what was checked itself and is mirrored. An unsupported part follows the
    variable slot with its detail.
    """
    plan = _own_plan(part)
    variable_text = part.variable_text or ""
    if isinstance(part, CandidatesPart):
        if not variable_text or not plan:
            return plan
        first, *rest = plan
        count = first.model_copy(update={"text": f" has {_lower_first_word(first.text)}"})
        tokens = [_variable_slot(part), count, *rest]
        _capitalize_first_token(tokens)
        return tokens
    if isinstance(part, NoDataPart):
        frame = composed_variable_outcome(part.no_data.reason) if variable_text else None
        if frame is None:
            return plan
        prefix, suffix = frame
        tokens = [_text_token(prefix)] if prefix else []
        tokens.extend([_variable_slot(part), _text_token(suffix), *plan[1:]])
        _capitalize_first_token(tokens)
        return tokens
    assert isinstance(part, UnsupportedPart), f"unexpected unresolved part status: {part!r}"
    text = part.unsupported.detail.strip()
    if not text.endswith(_SENTENCE_ENDS):
        text += "."
    if not variable_text:
        return [_text_token(text)]
    tokens = [_variable_slot(part), _text_token(" " + _lower_first_word(text))]
    _capitalize_first_token(tokens)
    return tokens


def _variable_slot(part: Part) -> SentenceToken:
    return SentenceToken(kind="slot", axis="what", text=part.variable_text or "")


def _own_plan(part: Part) -> list[SentenceToken]:
    """A part's own sentence plan, its tokens not yet owned by any part."""
    return [token.model_copy(update={"part_index": None}) for token in part.sentence_plan or []]


def _departure_sentence(text: str, index: int) -> list[Owned]:
    return [(index, SentenceToken(kind="text", text=text, segment="departure"))]


def _clause(items: list[tuple[int | None, str]]) -> list[Owned]:
    """The assumptions clause naming each item's text, each under its owner. Its lead,
    separators and period belong to the one owner every item shares, else to none. []
    when there are no items."""
    owners = {owner for owner, _text in items}
    frame = owners.pop() if len(owners) == 1 else None
    item_owners = iter(owner for owner, _text in items)
    tokens = clause_tokens([text for _owner, text in items], after_text=False)
    # The clause alternates lead or separator and item, so each odd token is an item.
    return [
        (next(item_owners) if position % 2 else frame, token)
        for position, token in enumerate(tokens)
    ]


def _own_clause(tokens: list[SentenceToken]) -> list[SentenceToken]:
    """The assumptions clause that closes a part's own plan, without the space before
    it."""
    clause = [token for token in tokens if token.segment == "assumption"]
    return clause[1:] if clause and clause[0].text == " " else clause


def _stated_once(chunks: list[list[Owned]]) -> list[list[Owned]]:
    """`chunks` in order, each sentence once. A chunk with the same tokens as an earlier
    one is dropped, and the earlier one loses its owner."""
    kept: list[list[Owned]] = []
    for chunk in chunks:
        earlier = next((i for i, seen in enumerate(kept) if _same_text(seen, chunk)), None)
        if earlier is None:
            kept.append(chunk)
        else:
            kept[earlier] = owned(None, _tokens(chunk))
    return kept


def _joined_sentences(
    chunks: list[list[Owned]], *, before: list[Owned] | None = None
) -> list[Owned]:
    """`before` followed by `chunks`, a space before each chunk that follows text. The space
    before a departure or a clause takes its segment and the owner of its first token. Any
    other space is unowned."""
    composed = list(before or [])
    for chunk in chunks:
        if _is_trailing(chunk):
            if any(token.text for _owner, token in composed):
                owner, token = chunk[0]
                composed.append((owner, token.model_copy(update={"text": " "})))
        elif composed:
            composed.append((None, _text_token(" ")))
        composed.extend(chunk)
    return composed


# The segments that close a sentence after its heading and byline.
_TRAILING: frozenset[SentenceSegment] = frozenset({"departure", "assumption"})


def _is_trailing(chunk: list[Owned]) -> bool:
    """Whether `chunk` is a departure or an assumptions clause."""
    return chunk[0][1].segment in _TRAILING


def _same_text(a: list[Owned], b: list[Owned]) -> bool:
    """Whether two chunks hold the same tokens, whichever parts own them."""
    return _tokens(a) == _tokens(b)


def _tokens(chunk: list[Owned]) -> list[SentenceToken]:
    return [token for _owner, token in chunk]


def _texts(part: Part) -> list[str]:
    return [departure.text for departure in part.departures]
