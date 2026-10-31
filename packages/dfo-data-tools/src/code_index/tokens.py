"""One tokeniser for code labels and query text, and the token-sequence lookup built on it.

A text is a sequence of word and separator tokens with their offsets in the text.
Whitespace is no token. A word is a run of letters and digits, joined across single
inner hyphens and apostrophes ("non-concessional", "d'Ivoire") and across a decimal point
between two digits ("2.15"). Every other visible character is a separator token of its
own: "HIV/AIDS" is the word "HIV", the separator "/" and the word "AIDS", and "$2.15" is
the separator "$" and the word "2.15". A token's text writes a typographic apostrophe as a straight
one (U+2019 as U+0027), so the two spellings of "d'Ivoire" are one word.

A label matches a query span when their words are equal, casefolded, and each gap
between two words agrees. A gap is the separators between two words. A query gap with
no separator matches any label gap, so "population policy programme" reads the label
"population policy, programme". A query gap holding separators matches only the label
gap holding the same ones, so "health. Education" never reads the label "health
education". Separators before a label's first word and after its last match nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

TokenKind = Literal["word", "sep"]

_TOKEN_RE = re.compile(
    r"(?P<word>[^\W_]+(?:(?:[-'\u2019]|(?<=\d)\.(?=\d))[^\W_]+)*)|(?P<sep>[^\w\s]|_)"
)


@dataclass(frozen=True, slots=True)
class Token:
    """One word or separator of a text, with its character offsets in that text."""

    text: str
    kind: TokenKind
    start: int
    end: int


def tokenize(text: str) -> tuple[Token, ...]:
    """Every word and separator token of `text`, in text order."""
    return tuple(
        Token(
            text=match.group().replace("\u2019", "'"),
            kind="word" if match.lastgroup == "word" else "sep",
            start=match.start(),
            end=match.end(),
        )
        for match in _TOKEN_RE.finditer(text)
    )


def word_tokens(text: str) -> tuple[Token, ...]:
    """The word tokens of `text`, in text order."""
    return tuple(token for token in tokenize(text) if token.kind == "word")


def singular(span: Sequence[Token]) -> tuple[Token, ...] | None:
    """`span` less one trailing "s" on its last token, the one plural allowance the
    matcher makes, or `None` when that token is a lone character or ends otherwise."""
    if not span:
        return None
    last = span[-1]
    if len(last.text) < 2 or not last.text.casefold().endswith("s"):
        return None
    return (
        *span[:-1],
        Token(text=last.text[:-1], kind=last.kind, start=last.start, end=last.end - 1),
    )


@dataclass(frozen=True, slots=True)
class _Key:
    """A token sequence's casefolded words and the separator texts of each gap between
    two of them; edge separators are dropped."""

    words: tuple[str, ...]
    gaps: tuple[tuple[str, ...], ...]


def _key(tokens: Sequence[Token]) -> _Key | None:
    """The key of `tokens`, or `None` when they hold no word."""
    words: list[str] = []
    gaps: list[tuple[str, ...]] = []
    pending: list[str] = []
    for token in tokens:
        if token.kind == "sep":
            pending.append(token.text)
            continue
        if words:
            gaps.append(tuple(pending))
        words.append(token.text.casefold())
        pending = []
    if not words:
        return None
    return _Key(words=tuple(words), gaps=tuple(gaps))


def _gaps_admit(
    label_gaps: tuple[tuple[str, ...], ...], query_gaps: tuple[tuple[str, ...], ...]
) -> bool:
    """Whether each query gap matches its label gap, for two keys of equal words."""
    return all(
        not query_gap or query_gap == label_gap
        for label_gap, query_gap in zip(label_gaps, query_gaps, strict=True)
    )


def text_matches(label: str, tokens: Sequence[Token]) -> bool:
    """Whether the text `label` matches the query span `tokens`."""
    label_key, query_key = _key(tokenize(label)), _key(tokens)
    return (
        label_key is not None
        and query_key is not None
        and label_key.words == query_key.words
        and _gaps_admit(label_key.gaps, query_key.gaps)
    )


class TokenLookup[T]:
    """Items found by the texts they carry, matched against query token sequences.

    `entries` pairs each text with its item. `get` returns every item one of whose texts
    matches the query span, once each, in the order of their first matching entry.
    """

    __slots__ = ("_by_words",)

    def __init__(self, entries: Iterable[tuple[str, T]]) -> None:
        by_words: dict[tuple[str, ...], list[tuple[tuple[tuple[str, ...], ...], T]]] = {}
        for text, item in entries:
            key = _key(tokenize(text))
            if key is not None:
                by_words.setdefault(key.words, []).append((key.gaps, item))
        self._by_words = by_words

    def get(self, tokens: Sequence[Token]) -> tuple[T, ...]:
        """Every item with a text matching the query span `tokens`."""
        query = _key(tokens)
        if query is None:
            return ()
        found: dict[int, T] = {}
        for gaps, item in self._by_words.get(query.words, ()):
            if id(item) not in found and _gaps_admit(gaps, query.gaps):
                found[id(item)] = item
        return tuple(found.values())
