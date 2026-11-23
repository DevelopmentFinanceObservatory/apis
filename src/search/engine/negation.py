"""Which words of a text a negation or exclusion reaches.

A negation cue is a negator ("not", "no", "never", "cannot" and the "n't" contractions:
"don't", "isn't", "won't") or an exclusion cue ("excluding", "except", "without", "other
than", "rather than", "excl"), and a spaced or hyphenated "non" before a word. A cue
reaches the words after it in its own clause, up to `REACH_WORDS` words on from it.

A clause closes at a comma, semicolon, colon, question or exclamation mark, a
parenthesis, a dash used as a break (a spaced hyphen, an en or em dash), and a full stop
that ends a sentence (one followed by a space or the end of the text, after no initial
or abbreviation such as "incl." or "e.g."). A comma inside a list an exclusion cue
governs does not close its clause: "excluding France, Germany and multilateral" is one
clause. A cue its clause closes right after reaches nothing, so a reply's "No, count it
including imputed multilateral" negates nothing after the comma.

`negation_start` reads whether a word is negated and where its cue begins
(`bind_repair.occurrence_negated`, `stated_scope._cue_start`), `negation_cues` lists the
cues that reach a word (`exclusion_read._pieces`), and `negated_stretches` the stretches
of text they reach (`qualifier_match.measure_values_named`).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

REACH_WORDS = 3

_NEGATOR = r"(?:not|no|never|cannot|\w+n['\u2019]t)"
_EXCLUDER = r"(?:excluding|exclude|excludes|except|without|other\s+than|rather\s+than|excl)"
_EXCLUSION_CUE = re.compile(rf"\b{_EXCLUDER}\b", re.IGNORECASE)
# The cues that reach words after them; "non" reaches only the word it is fused or spaced
# to (`_NON_BEFORE`).
_CUE = re.compile(rf"\b(?:{_NEGATOR}|{_EXCLUDER})\b", re.IGNORECASE)
_ANY_CUE = re.compile(rf"\b(?:non|{_NEGATOR}|{_EXCLUDER})\b", re.IGNORECASE)
_NON_BEFORE = re.compile(r"\bnon[\s-]*$", re.IGNORECASE)
_WORD = re.compile(r"[\w'\u2019]+")
_CLAUSE_MARK = re.compile(r"[,;:!?()\u2013\u2014]|\.(?=\s|$)|\s-+(?=\s)")
# A full stop after one of these, or after a single letter ("U.S."), abbreviates a word
# and ends no sentence.
_ABBREVIATION = re.compile(
    r"(?:\b(?:incl|excl|vs|etc|approx|e\.g|i\.e)|(?<![\w.])\w|\.\w)$", re.IGNORECASE
)


def _clause_marks(text: str) -> Iterator[re.Match[str]]:
    """The marks that close a clause of `text`, in order."""
    clause_start = 0
    for mark in _CLAUSE_MARK.finditer(text):
        symbol = mark.group().strip()
        if symbol == "." and _ABBREVIATION.search(text, 0, mark.start()):
            continue
        if symbol == "," and _EXCLUSION_CUE.search(text, clause_start, mark.start()):
            continue
        clause_start = mark.end()
        yield mark


def clause_start(text: str, position: int) -> int:
    """Where the clause holding `text[position]` begins."""
    start = 0
    for mark in _clause_marks(text):
        if mark.end() > position:
            break
        start = mark.end()
    return start


def clause_end(text: str, position: int) -> int:
    """Where the clause holding `text[position]` ends: its closing mark, or the text's end."""
    return next((m.start() for m in _clause_marks(text) if m.start() >= position), len(text))


def negation_start(text: str, start: int) -> int | None:
    """Where the negation cue reaching the word at `text[start:]` begins: a "non" fused or
    spaced to it, or the last cue in its clause within `REACH_WORDS` words before it. None
    when no cue reaches it."""
    if non := _NON_BEFORE.search(text, 0, start):
        return non.start()
    clause = clause_start(text, start)
    words = list(_WORD.finditer(text, clause, start))
    window = words[-REACH_WORDS].start() if len(words) >= REACH_WORDS else clause
    cues = list(_CUE.finditer(text, window, start))
    return cues[-1].start() if cues else None


def negation_cues(text: str, start: int = 0, end: int | None = None) -> list[re.Match[str]]:
    """The cues in `text[start:end]`, a "non" among them, that reach a word: each with a word
    after it in its own clause."""
    reaching = []
    for cue in _ANY_CUE.finditer(text, start, len(text) if end is None else end):
        after = _WORD.search(text, cue.end())
        if after is not None and clause_start(text, after.start()) <= cue.start():
            reaching.append(cue)
    return reaching


def negated_stretches(text: str) -> list[tuple[int, int]]:
    """The character ranges of `text` its negation cues reach, each from a cue that reaches
    a word to the end of its clause, in order and disjoint."""
    stretches: list[tuple[int, int]] = []
    for cue in negation_cues(text):
        if stretches and cue.start() < stretches[-1][1]:
            continue
        stretches.append((cue.start(), clause_end(text, cue.start())))
    return stretches
