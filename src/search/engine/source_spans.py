"""Where the raw-text door's extracted variables sit in the raw query, and what that says.

Extraction rewrites a variable's text: "compare aid to health and education" yields
"health aid" and "education aid". A variable is located by its content words (the words
of its text that are not `FUNCTION_WORDS`), from a word no other variable holds where it
has one, else aligned to the raw query left to right in variable order
(`locate_variables`). Located spans never overlap or cross a comparison connective.

Two located variables are compared, and marked separate (`ParsedPart.separate`), when a
comparison connective governs them by the rule a part's own codes are joined by
(`qualifier_match.governing_connectives`, `qualifier_match.joins`), reading the content
words two or more variables hold as function words: "versus" between them ("ODA to
health vs. ODA to education"), or "compare"/"compared" before both. A comparison
connective that governs a located variable also marks every located variable after it,
whatever joins them: "compare health aid with education aid in Kenya". Nothing else
marks a variable.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from code_index.tokens import word_tokens
from search.engine.function_words import FUNCTION_WORDS
from search.engine.qualifier_match import COMPARISON_CONNECTIVES, governing_connectives, joins


@dataclass(frozen=True, slots=True)
class VariableSpan:
    """A located variable's character offsets in the raw query."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SourceReading:
    """What the raw query's words say about its extracted variables.

    `separate` runs parallel to the variables. `read_connectives` holds the offsets of
    the comparison connectives that marked a variable, words the marking reads.
    """

    separate: tuple[bool, ...]
    read_connectives: frozenset[int]


def _content_words(text: str) -> set[str]:
    return {
        folded
        for token in word_tokens(text)
        if (folded := token.text.casefold()) not in FUNCTION_WORDS
    }


def locate_variables(query: str, variables: Sequence[str]) -> tuple[VariableSpan | None, ...]:
    """Each of `variables`' span in `query`, or `None` where it has no location.

    A variable with a content word no other variable holds starts at that word's first
    occurrence past the span before it, else at its first occurrence anywhere, so
    variables listed out of query order still locate: "ODA to health vs. ODA to
    education" locates "education ODA" and then "health ODA". The span widens left over
    the variable's own words and function words beside it. A variable whose every word
    another holds starts at the first of its own words past the span before it, falling
    back to an earlier unclaimed occurrence where extraction returns a later, containing
    variable first: "compare health ODA with basic health ODA" still locates both.

    From its start a span reads on, each of its words once, until it has read them all,
    meets a word it has read already, a word only other variables hold or another span.
    No span crosses a comparison connective (`COMPARISON_CONNECTIVES`). A variable whose
    every word another holds and that meets one before reading all its words starts
    again at its first own word past it, so in "For ODA to Kenya, compare health ODA with
    education ODA" "health ODA" spans "health ODA". Spans never overlap, and a variable
    with no own word outside the other spans has no location.
    """
    words = [_content_words(variable) for variable in variables]
    raw = word_tokens(query)
    folded = [token.text.casefold() for token in raw]
    claimed: set[int] = set()
    spans: list[VariableSpan | None] = []
    position = 0

    def first_of(wanted: set[str], since: int) -> int | None:
        return next(
            (at for at in range(since, len(raw)) if at not in claimed and folded[at] in wanted),
            None,
        )

    for i, own in enumerate(words):
        others = set().union(*(other for j, other in enumerate(words) if j != i))
        foreign = others - own
        distinct = own - others
        if distinct:
            first = first_of(distinct, position)
            if first is None:
                first = first_of(distinct, 0)
        else:
            first = first_of(own, position)
            if first is None:
                first = first_of(own, 0)
        if first is None:
            spans.append(None)
            continue
        read: set[str] = set()
        last = first
        at = first
        while at < len(raw):
            word = folded[at]
            if word in COMPARISON_CONNECTIVES and not distinct and read != own:
                restart = first_of(own, at + 1)
                if restart is None:
                    break
                first = last = at = restart
                read = set()
                continue
            if (
                at in claimed
                or word in COMPARISON_CONNECTIVES
                or word in foreign
                or word in read
                or read == own
            ):
                break
            if word in own:
                read.add(word)
                last = at
            at += 1
        if distinct:
            at = first - 1
            while at >= 0 and at not in claimed and read != own:
                word = folded[at]
                if word in own and word not in read:
                    read.add(word)
                    first = at
                elif word not in FUNCTION_WORDS:
                    break
                at -= 1
        claimed.update(range(first, last + 1))
        spans.append(VariableSpan(start=raw[first].start, end=raw[last].end))
        position = last + 1
    return tuple(spans)


def read_sources(query: str, variables: Sequence[str]) -> SourceReading:
    """Mark each of `variables` separate when a comparison connective of `query` governs
    it with another located variable."""
    spans = locate_variables(query, variables)
    located = sorted(
        ((span, i) for i, span in enumerate(spans) if span is not None),
        key=lambda item: item[0].start,
    )
    ordered = [span for span, _ in located]
    counts = Counter(word for variable in variables for word in _content_words(variable))
    shared = frozenset(word for word, count in counts.items() if count > 1)
    governing = governing_connectives(query, ordered, transparent=shared)
    kinds = joins(query, ordered, governing, transparent=shared)
    compared: list[Sequence[tuple[VariableSpan, int]]] = [
        pair for pair, kind in zip(pairwise(located), kinds, strict=True) if kind == "comparison"
    ]
    for connective in governing:
        run = [item for item in located if item[0].start >= connective.governed[0]]
        if connective.kind == "comparison" and len(run) > 1:
            compared.append(run)
    separate = [False] * len(variables)
    marked: set[tuple[int, int]] = set()
    for span, i in (item for group in compared for item in group):
        separate[i] = True
        marked.add((span.start, span.end))
    read = frozenset(
        connective.start
        for connective in governing
        if connective.kind == "comparison" and connective.governed in marked
    )
    return SourceReading(separate=tuple(separate), read_connectives=read)


def unclaimed_words(
    query: str, *, claimed: Iterable[str], read_connectives: frozenset[int] = frozenset()
) -> tuple[str, ...]:
    """The words of `query` no `claimed` text holds, casefolded and in query order, less
    `FUNCTION_WORDS` and the connectives at `read_connectives`."""
    held = {token.text.casefold() for text in claimed for token in word_tokens(text)}
    return tuple(
        dict.fromkeys(
            folded
            for token in word_tokens(query)
            if token.start not in read_connectives
            and (folded := token.text.casefold()) not in FUNCTION_WORDS
            and folded not in held
        )
    )
