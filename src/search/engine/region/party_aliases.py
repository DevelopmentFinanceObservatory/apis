"""The stretches of a phrase that name a party the part binds, by a name Search reads it by.

Intake reads them in the variable phrase once (`ResolveState.party_alias_spans`). The
coverage rule, the nomination of unread words, the candidate menu and the bind's unplaced
phrases count them as read; the phrase a model reads is unchanged. The names come from a
committed export (`pack.geo_names`), so no resolver or graph read is made."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

from code_index.shared import get_index, get_measure_index
from code_index.tokens import Token, tokenize
from entity_resolution import Group, Resolution, Resolved
from entity_resolution.pack import (
    geo_name_forms,
    geo_names,
    longest_geo_name_tokens,
    longest_stored_name_tokens,
    stored_name_forms,
)
from search.engine.extract import _span_occurrences
from search.engine.function_words import FUNCTION_WORDS
from search.engine.qualifier_match import read_qualifiers
from search.engine.unit.lexicon import (
    AMBIGUOUS_CURRENCY_NAMES,
    CURRENCY_ALIASES,
    MAX_PHRASE_WORDS,
)
from search.engine.unit.lexicon import _name_key as currency_name_key

type CharSpan = tuple[int, int]

_WORD_RE: Final = re.compile(r"\w+")
# The currency symbols the unit binder reads (`unit.lexicon.CURRENCY_ALIASES`), which a
# country's code can prefix ("US$").
_CURRENCY_SYMBOLS: Final = frozenset(key for key in CURRENCY_ALIASES if len(key) == 1)
# The ending a name may take inside its token: "UK's" holds the name "UK".
_POSSESSIVE_RE: Final = re.compile(r"['\u2019]s?", re.IGNORECASE)


def _bound_dcids(resolution: Resolution) -> tuple[str, ...]:
    """The dcids a resolution of one of the part's entities binds: a resolved entity's node,
    or a group's registry dcid and its node."""
    if isinstance(resolution, Resolved):
        return (resolution.entity.dcid,)
    if isinstance(resolution, Group):
        node = None if resolution.entity is None else resolution.entity.dcid
        return tuple(dcid for dcid in (resolution.dcid, node) if dcid is not None)
    return ()


def bound_party_spans(text: str, resolutions: Iterable[Resolution]) -> tuple[CharSpan, ...]:
    """The character range of each stretch of `text` that names a party one of `resolutions`
    binds by a name Search reads it by (`pack.geo_names`): "US" in "total US ODA" beside the
    donor United States, "G7" in "G7 ODA" beside the Group of Seven. The readers that count
    them as read (`ResolveState.party_alias_spans`) count exactly these ranges, so another
    copy of the same words ("Africa" in "Africa Finance Corporation") stays the phrase's
    own.

    A stretch is a whole-word occurrence, casing aside (`extract._span_occurrences`), that
    starts and ends where a token of `text` does (`code_index.tokens.tokenize`), so "US" in
    "US-funded" is no stretch, and that may end before a possessive ending its token holds
    ("UK's"), which the range then takes. It writes as the name does each word the name
    writes in capitals, and writes a name of two letters or fewer, a code, in capitals
    (`_written_as`), so the pronoun "us" or the words "eu" and "au" never read as the United
    States, the European Union or the African Union. Longer names take their stretch first.
    A name that is a function word is never read. A stretch inside a longer stretch that
    names something else is not read as the party: an organisation or aggregate the entity
    pack stores (`pack.stored_name_forms`: "Africa Finance Corporation") or another country,
    region or group (`pack.geo_name_forms`: "South Africa"), written as that name is, or a
    code, measure or currency phrase the indexes read (`_inside_a_read_phrase`: "constant US
    dollars", "US$")."""
    names = {
        name
        for resolution in resolutions
        for dcid in _bound_dcids(resolution)
        for name in geo_names(dcid)
        if name.casefold() not in FUNCTION_WORDS
    }
    if not names:
        return ()
    tokens = tokenize(text)
    words = list(_WORD_RE.finditer(text))
    read_phrases: list[CharSpan] | None = None
    taken: list[CharSpan] = []
    for name in sorted(names, key=lambda name: (-len(name), name)):
        for occurrence in _span_occurrences(text, name):
            if not _written_as(name, occurrence.group()):
                continue
            span = _on_token_bounds(tokens, *occurrence.span())
            if span is None:
                continue
            start, end = span
            if any(start < other_end and other_start < end for other_start, other_end in taken):
                continue
            if _inside_another_name(text, words, *occurrence.span()):
                continue
            if read_phrases is None:
                read_phrases = _read_phrases(text)
            if _inside_a_read_phrase(text, words, read_phrases, *occurrence.span()):
                continue
            taken.append(span)
    return tuple(sorted(taken))


def _on_token_bounds(tokens: Sequence[Token], start: int, end: int) -> CharSpan | None:
    """`start:end` when a token starts at `start` and one ends at `end`, the range to its
    token's end when only a possessive ending follows `end` there ("UK's"), else `None`."""
    if not any(token.start == start for token in tokens):
        return None
    for token in tokens:
        if token.end == end:
            return start, end
        if token.start < end < token.end:
            rest = token.text[end - token.start :]
            return (start, token.end) if _POSSESSIVE_RE.fullmatch(rest) else None
    return None


def _read_phrases(text: str) -> list[CharSpan]:
    """The ranges of `text` each code or measure term the indexes read in it covers
    (`qualifier_match.read_qualifiers`): "constant US dollars" in "ODA in constant US
    dollars"."""
    matches, ties = read_qualifiers(text, get_index(), get_measure_index())
    phrases = {match.matched_text for match in (*matches, *ties)}
    return [
        occurrence.span() for phrase in phrases for occurrence in _span_occurrences(text, phrase)
    ]


def _inside_a_read_phrase(
    text: str,
    words: Sequence[re.Match[str]],
    read_phrases: Sequence[CharSpan],
    start: int,
    end: int,
) -> bool:
    """Whether the stretch `start:end` of `text` lies inside a longer phrase the indexes read
    (`read_phrases`), is followed straight by a currency symbol ("US$"), or lies inside a
    longer stretch of words the unit binder reads as a currency ("US dollars",
    `unit.lexicon.CURRENCY_ALIASES`)."""
    if any(
        phrase_start <= start and end <= phrase_end and (phrase_start, phrase_end) != (start, end)
        for phrase_start, phrase_end in read_phrases
    ):
        return True
    if text[end : end + 1] in _CURRENCY_SYMBOLS:
        return True
    inside = [index for index, word in enumerate(words) if start <= word.start() < end]
    if not inside:
        return False
    first, last = inside[0], inside[-1]
    for begin in range(max(0, first - MAX_PHRASE_WORDS), first + 1):
        for stop in range(last, min(len(words), last + MAX_PHRASE_WORDS + 1)):
            if (begin, stop) == (first, last):
                continue
            key = currency_name_key(text[words[begin].start() : words[stop].end()])
            if key in CURRENCY_ALIASES or key in AMBIGUOUS_CURRENCY_NAMES:
                return True
    return False


def _written_as(name: str, stretch: str) -> bool:
    """Whether `stretch`, an occurrence of `name` casing aside, writes each word `name`
    writes in capitals with no lowercase letter ("USA", "ASEAN", "G7") exactly as `name`
    does, and writes every letter of a name of two letters or fewer, a code, in capitals:
    "EU" reads as the European Union. A stretch whose words do not pair with the name's, which the
    entity pack's normal form can make, is taken as written."""
    if sum(char.isalpha() for char in name) <= 2 and any(char.islower() for char in stretch):
        return False
    name_words, written = _WORD_RE.findall(name), _WORD_RE.findall(stretch)
    if len(name_words) != len(written):
        return True
    return all(
        word == stated
        for word, stated in zip(name_words, written, strict=True)
        if any(char.isalpha() for char in word) and not any(char.islower() for char in word)
    )


def _inside_another_name(text: str, words: Sequence[re.Match[str]], start: int, end: int) -> bool:
    """Whether the stretch `start:end` of `text` lies inside a longer stretch of its words
    that the entity pack stores as a name (`pack.stored_name_forms`) or that names a country,
    region or group (`pack.geo_name_forms`), written as that name is (`_written_as`)."""
    inside = [index for index, word in enumerate(words) if start <= word.start() < end]
    if not inside:
        return False
    first, last = inside[0], inside[-1]
    reach = max(longest_stored_name_tokens(), longest_geo_name_tokens(), last - first + 1)
    for begin in range(max(0, last - reach + 1), first + 1):
        for stop in range(last, min(len(words), begin + reach)):
            if (begin, stop) == (first, last):
                continue
            stretch = text[words[begin].start() : words[stop].end()]
            forms = (*stored_name_forms(stretch), *geo_name_forms(stretch))
            if any(_written_as(form, stretch) for form in forms):
                return True
    return False
