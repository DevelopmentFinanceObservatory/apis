"""Which words of a text a negation or exclusion reaches.

A negation cue is a negator ("not", "no", "never", "cannot" and the "n't" contractions:
"don't", "isn't", "won't") or an exclusion cue ("excluding", "except", "without", "other than",
"rather than", "excl"), and a spaced or hyphenated "non" before a word. A cue reaches the words
after it in its own clause, up to `REACH_WORDS` words on from it, and an exclusion cue also
reaches each item of its list (below). A contraction in a question ("Isn't net ODA lower than
gross ODA", "Why doesn't France include it") asks and negates nothing.

A clause closes at a comma, semicolon, colon, question or exclamation mark, a parenthesis, a
dash used as a break (a spaced hyphen, an en or em dash), and a full stop that ends a sentence:
one followed by a space or the end of the text, after no dotted initialism ("U.S.") or
abbreviation ("incl.", "e.g."), and after "etc." only before a capitalised word. Three marks
close nothing:

- a mark straight after an exclusion cue, which introduces its object ("excluding:
  multilateral", "excluding (imputed) multilateral"), with the parenthesis that closes one;
- a comma inside the list an exclusion cue governs ("excluding France, Germany and
  multilateral"), while the piece after it is a list item. A piece opening on a negator, a
  scope word ("gross", "including"), a verb ("count", "show"), a question word or a preposition
  ends the list, and its comma closes the clause.

A cue its clause closes right after reaches nothing, so a reply's "No, count it including
imputed multilateral" negates nothing after the comma.

`negation_start` reads whether a word is negated and where its cue begins
(`bind_repair.occurrence_negated`, `stated_scope._cue_start`), `negation_cues` lists the cues
that reach a word (`exclusion_read._pieces`), and `negated_stretches` the stretches of text
they reach (`qualifier_match.measure_values_named`). Each text's clauses are read once
(`_clauses`).
"""

from __future__ import annotations

import functools
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

REACH_WORDS = 3
# The curly apostrophe and the en and em dashes, by code point so the source stays ASCII.
_APOSTROPHES = "'" + chr(0x2019)
_DASHES = chr(0x2013) + chr(0x2014)

_NEGATOR = rf"(?:not|no|never|cannot|\w+n[{_APOSTROPHES}]t)"
_EXCLUDER = r"(?:excluding|exclude|excludes|except|without|other\s+than|rather\s+than|excl)"
_EXCLUSION_CUE = re.compile(rf"\b{_EXCLUDER}\b", re.IGNORECASE)
_NEGATOR_WORD = re.compile(rf"{_NEGATOR}$", re.IGNORECASE)
_CONTRACTION = re.compile(rf"\w+n[{_APOSTROPHES}]t$", re.IGNORECASE)
# The cues that reach words after them; "non" reaches only the word it is fused or spaced
# to (`_NON_BEFORE`).
_CUE = re.compile(rf"\b(?:{_NEGATOR}|{_EXCLUDER})\b", re.IGNORECASE)
_ANY_CUE = re.compile(rf"\b(?:non|{_NEGATOR}|{_EXCLUDER})\b", re.IGNORECASE)
_NON_BEFORE = re.compile(r"\bnon[\s-]*$", re.IGNORECASE)
_WORD = re.compile(rf"[\w{_APOSTROPHES}]+")
_CLAUSE_MARK = re.compile(rf"[,;:!?(){_DASHES}]|\.(?=\s|$)|\s-+(?=\s)")
_SENTENCE_END = frozenset(".?!")
# A full stop after one of these abbreviates a word and ends no sentence; "etc." ends one
# before a capitalised word (`_ends_sentence`).
_ABBREVIATION = re.compile(r"(?:\b(?:incl|excl|vs|approx|e\.g|i\.e)|\.\w)$", re.IGNORECASE)
_ETC = re.compile(r"\betc$", re.IGNORECASE)
_CUE_BEFORE = re.compile(rf"\b{_EXCLUDER}\s*$", re.IGNORECASE)
# First words of a piece after a comma that end an exclusion cue's list: scope words, verbs,
# question words and prepositions. A negator ends it too (`_ends_list`).
_LIST_ENDERS = frozenset(
    {
        "net", "gross", "commitments", "commitment", "grant", "total", "totals",
        "including", "include", "includes", "incl", "inclusive", "plus", "with",
        "count", "show", "give", "list", "plot", "chart", "compare", "use", "make", "rank",
        "tell", "break", "split", "sum", "add", "display", "return", "report", "keep",
        "please", "what", "how", "which", "who", "where", "when", "why",
        "in", "for", "from", "to", "by", "during", "since", "over", "between", "as", "but",
        "then",
    }
)  # fmt: skip
_QUESTION_WORDS = frozenset({"what", "how", "which", "who", "where", "when", "why"})


@dataclass(frozen=True, slots=True)
class _Clauses:
    """The clauses of one text: where each begins and ends, the list items exclusion cues
    reach across commas (each piece's start beside its cue's start), and where each closing
    mark stands with its symbol."""

    starts: tuple[int, ...]
    ends: tuple[int, ...]
    symbols: tuple[str, ...]
    items: tuple[tuple[int, int], ...]


def _ends_sentence(text: str, mark: re.Match[str]) -> bool:
    if _ABBREVIATION.search(text, 0, mark.start()):
        return False
    if _ETC.search(text, 0, mark.start()):
        after = _WORD.search(text, mark.end())
        return after is not None and after.group()[0].isupper()
    return True


def _ends_list(text: str, position: int) -> bool:
    """Whether the piece of `text` from `position` opens on a word that ends a list."""
    first = _WORD.search(text, position)
    if first is None:
        return True
    word = first.group().casefold()
    return word in _LIST_ENDERS or _NEGATOR_WORD.match(word) is not None


@functools.lru_cache(maxsize=1024)
def _clauses(text: str) -> _Clauses:
    starts, ends, symbols, items = [0], [], [], []
    governing: int | None = None
    held_parens = 0
    for mark in _CLAUSE_MARK.finditer(text):
        symbol = mark.group().strip()
        if governing is None and (cue := _EXCLUSION_CUE.search(text, starts[-1], mark.start())):
            governing = cue.start()
        if symbol == "." and not _ends_sentence(text, mark):
            continue
        if symbol == ")" and held_parens:
            held_parens -= 1
            continue
        if _CUE_BEFORE.search(text, starts[-1], mark.start()):
            held_parens += symbol == "("
            continue
        if symbol == "," and governing is not None and not _ends_list(text, mark.end()):
            items.append((mark.end(), governing))
            continue
        ends.append(mark.start())
        symbols.append(symbol)
        starts.append(mark.end())
        governing = None
    return _Clauses(tuple(starts), tuple(ends), tuple(symbols), tuple(items))


def clause_start(text: str, position: int) -> int:
    """Where the clause holding `text[position]` begins."""
    starts = _clauses(text).starts
    return starts[bisect_right(starts, position) - 1]


def clause_end(text: str, position: int) -> int:
    """Where the clause holding `text[position]` ends: its closing mark, or the text's end."""
    ends = _clauses(text).ends
    index = bisect_left(ends, position)
    return ends[index] if index < len(ends) else len(text)


def _asks(text: str, cue: re.Match[str]) -> bool:
    """Whether `cue` is a contraction in a question: its sentence ends on "?", or it opens its
    clause, after at most a question word, as an inverted auxiliary ("Isn't", "Why doesn't").
    A "don't" opening its clause is an imperative."""
    if not _CONTRACTION.match(cue.group()):
        return False
    clauses = _clauses(text)
    index = bisect_left(clauses.ends, cue.end())
    closing = next((s for s in clauses.symbols[index:] if s in _SENTENCE_END), None)
    if closing == "?":
        return True
    before = [
        w.group().casefold()
        for w in _WORD.finditer(text, clause_start(text, cue.start()), cue.start())
    ]
    if before and (len(before) > 1 or before[0] not in _QUESTION_WORDS):
        return False
    return bool(before) or not cue.group().casefold().startswith("don")


def negation_start(text: str, start: int) -> int | None:
    """Where the negation cue reaching the word at `text[start:]` begins: a "non" fused or
    spaced to it, the last cue in its clause within `REACH_WORDS` words before it, or the
    exclusion cue whose list it is an item of, within `REACH_WORDS` words of its item's start.
    None when no cue reaches it."""
    if non := _NON_BEFORE.search(text, 0, start):
        return non.start()
    clause = clause_start(text, start)
    words = list(_WORD.finditer(text, clause, start))
    window = words[-REACH_WORDS].start() if len(words) >= REACH_WORDS else clause
    cues = [cue for cue in _CUE.finditer(text, window, start) if not _asks(text, cue)]
    if cues:
        return cues[-1].start()
    items = _clauses(text).items
    index = bisect_right(items, (start, len(text))) - 1
    if index >= 0 and items[index][0] >= clause:
        piece, cue = items[index]
        if len(_WORD.findall(text, piece, start)) < REACH_WORDS:
            return cue
    return None


def negation_cues(text: str, start: int = 0, end: int | None = None) -> list[re.Match[str]]:
    """The cues in `text[start:end]`, a "non" among them, that reach a word: each with a word
    after it in its own clause, and no contraction in a question."""
    reaching = []
    for cue in _ANY_CUE.finditer(text, start, len(text) if end is None else end):
        after = _WORD.search(text, cue.end())
        if (
            after is not None
            and clause_start(text, after.start()) <= cue.start()
            and not _asks(text, cue)
        ):
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
