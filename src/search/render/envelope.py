"""Compose the envelope's sentence plan from the response parts.

Definite parts join their headings into one sentence (`render.join.join_headings`).
Headings that cannot join, among them two that would join one noun to itself, stay one
sentence each. The byline follows once when every part's byline reads the same, and each
part's departures close the sentence in part order. Otherwise the envelope writes one
sentence per part, each followed by its own departures. A sentence two parts state alike
(a heading, a byline) is stated once, with no part index, and so is a departure two parts
state after the same sentence.

A part folded into an earlier part's Spec (`binding_folded`) states that part's heading and
byline, so their tokens carry no part index.

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
    SentenceToken,
    UnsupportedPart,
)

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

    Omit parts folded into an earlier Spec. A lone part that is not definite is mirrored
    verbatim. Definite parts use general composition, even when alone, so shared tokens
    have no part index. Token indices always refer to the original ``parts`` list,
    including gaps from folded parts.
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
        composed = _unowned_by_folds(
            _compose_general(contributing, plans), _folded_into_parts(parts, contributing)
        )
    plan = [token.model_copy(update={"part_index": owner}) for owner, token in composed]
    return sentence_from_plan(plan), plan


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


def _folded_into_parts(
    parts: Sequence[Part], contributing: list[tuple[int, Part]]
) -> frozenset[int]:
    """The indices of the definite parts a later part folded into. A folded part carries
    the Spec of the part it folded into."""
    folded_into = {
        part.spec.spec_id: index for index, part in contributing if isinstance(part, DefinitePart)
    }
    return frozenset(
        first
        for index, part in enumerate(parts)
        if isinstance(part, DefinitePart)
        and _is_folded(part)
        and (first := folded_into.get(part.spec.spec_id)) is not None
        and first < index
    )


def _unowned_by_folds(composed: list[Owned], folded_into: frozenset[int]) -> list[Owned]:
    """`composed` with the heading and byline tokens of each part in ``folded_into``
    unowned, since every part folded into it states them. Departures keep their part."""
    return [
        (None, token) if owner in folded_into and token.segment != "departure" else (owner, token)
        for owner, token in composed
    ]


def _compose_general(
    contributing: list[tuple[int, Part]], plans: Mapping[int, HeadingPlan]
) -> list[Owned]:
    """One joined sentence when every part is definite and it can be written, else one
    sentence per part. A lone definite part takes this path so shared-token provenance
    stays clear."""
    if len(plans) == len(contributing):
        joined = _compose_joined(contributing, plans)
        if joined is not None:
            return joined
    return _compose_per_part(contributing, plans)


def _compose_joined(
    contributing: list[tuple[int, Part]], plans: Mapping[int, HeadingPlan]
) -> list[Owned] | None:
    """The definite parts' headings, then the byline they share, then their departures.
    Return None when the bylines differ, since one byline after the headings could not
    say which part each item is for.

    The headings join into one (`join_headings`) or, when they cannot, follow one another
    as sentences of their own, a heading two parts state alike stated once. Then comes the
    byline once, unowned. An empty source placeholder every part shares is written once,
    unowned, and otherwise each stays with its own part. Each part's departures follow
    under its own part, a text two parts carry stated once, unowned.
    """
    headings = {index: heading_tokens(plans[index].heading) for index, _part in contributing}
    after_heading = {
        index: any(token.text for token in tokens) for index, tokens in headings.items()
    }
    if len({_shown_key(plans[index].byline, after_heading[index]) for index in headings}) > 1:
        return None
    bylines = {
        index: byline_tokens(plans[index].byline, after_heading=after_heading[index])
        for index in headings
    }

    composed = join_headings([(index, plans[index].heading) for index, _part in contributing])
    if composed is None:
        composed = _joined_sentences(
            _stated_once([owned(index, tokens) for index, tokens in headings.items()])
        )
    first_index = contributing[0][0]
    composed.extend(owned(None, [token for token in bylines[first_index] if token.text]))
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
    return _joined_sentences(departures, before=composed)


def _shown_key(byline: Byline, after_heading: bool) -> Hashable:
    """What a byline states. Two parts share one byline only when the latest-year label,
    the unit, a stated source and the bound method agree, and both follow a heading with
    text."""
    source = byline.source
    shown_source = (
        (source.text, source.slot.key.as_ref() if source.slot else None)
        if source is not None and source.text
        else None
    )
    return (after_heading, byline.latest, byline.unit_text, shown_source, byline.method)


def _placeholder_key(byline: Byline) -> Hashable:
    """The empty source slot a byline keeps in place, keyed by its slot, or () when none."""
    source = byline.source
    if source is None or source.text:
        return ()
    return (source.slot.key.as_ref() if source.slot else None,)


def _compose_per_part(
    contributing: list[tuple[int, Part]], plans: Mapping[int, HeadingPlan]
) -> list[Owned]:
    """Write one sentence per part in part order, each followed by its departures.

    A definite part's sentence is its heading and byline. A part that is not definite
    contributes `_unresolved_sentence`, and no-data parts refused for the same reason name
    their variables in one sentence (`_shared_refusals`), written where the first of them
    stands. A sentence two parts state alike is stated once, unowned. A departure is
    stated once only beside the sentence it follows (`_departures_stay_with_sentences`).
    """
    refusals = _shared_refusals(contributing)
    chunks: list[list[Owned]] = []
    for index, part in contributing:
        if isinstance(part, DefinitePart):
            sentence = owned(index, plan_tokens(plans[index]))
            texts = _texts(part)
        else:
            tokens = _unresolved_sentence(part)
            sentence = refusals.get(
                index, owned(index, [token for token in tokens if token.segment != "departure"])
            )
            texts = [
                token.text for token in tokens if token.segment == "departure" and token.text != " "
            ]
        if sentence:
            chunks.append(sentence)
        chunks.extend(_departure_sentence(text, index) for text in texts)
    return _joined_sentences(_departures_stay_with_sentences(chunks))


def _departures_stay_with_sentences(chunks: list[list[Owned]]) -> list[list[Owned]]:
    """`chunks` in order with each sentence once, as `_stated_once` keeps them, and each
    departure after the sentence it follows. A departure equal to one already stated
    after the same sentence is dropped and that one loses its owner. One stated after
    another part's sentence is kept, so the reader sees which part each limit belongs
    to."""
    kept: list[list[Owned]] = []
    sentence_start = 0
    for chunk in chunks:
        is_departure = _is_departure(chunk)
        start = sentence_start if is_departure else 0
        earlier = next((i for i in range(start, len(kept)) if _same_text(kept[i], chunk)), None)
        if earlier is None:
            if not is_departure:
                sentence_start = len(kept)
            kept.append(chunk)
            continue
        kept[earlier] = owned(None, _tokens(chunk))
        if not is_departure:
            sentence_start = earlier
    return kept


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
        rest = [t for t in (part.sentence_plan or [])[1:] if t.segment != "departure"]
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
    """`before` followed by `chunks`, a space before each chunk that follows text. A
    departure's space is a departure-segment token owned by the departure's part. Any
    other space is unowned."""
    composed = list(before or [])
    for chunk in chunks:
        if _is_departure(chunk):
            if any(token.text for _owner, token in composed):
                owner, token = chunk[0]
                composed.append((owner, token.model_copy(update={"text": " "})))
        elif composed:
            composed.append((None, _text_token(" ")))
        composed.extend(chunk)
    return composed


def _is_departure(chunk: list[Owned]) -> bool:
    return chunk[0][1].segment == "departure"


def _same_text(a: list[Owned], b: list[Owned]) -> bool:
    """Whether two chunks hold the same tokens, whichever parts own them."""
    return _tokens(a) == _tokens(b)


def _tokens(chunk: list[Owned]) -> list[SentenceToken]:
    return [token for _owner, token in chunk]


def _texts(part: Part) -> list[str]:
    return [departure.text for departure in part.departures]
