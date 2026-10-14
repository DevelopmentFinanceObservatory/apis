"""Match code and measure text without deciding whether a query uses that vocabulary.

`match_qualifiers` returns only spans naming one row. Callers need independent
reason to admit the matched code system. `match_qualifier_ties` exposes
ambiguous all-code spans for a later election. Both read every code system;
`read_by_vocabulary` reads each text once per shape vocabulary, the code systems one
shape declares. `QualifierMatch.code_dcid` and `label` are graph text safe for
prompts; `matched_text` is raw query text and stays out of prompts. The normalised
form of `matched_text` is the phrase a match reads, which the five-tuple family
election compares across tables (`bound_values.query_phrases_read`).

Matching reads token sequences (`code_index.tokens`): a span is a run of query words,
and a separator between two of them must be the one the label holds there. Only an
exact miss may try one trailing `s`; an exact ambiguous match stays ambiguous. Bare
codes require an earlier word unique to the code system, derived from its camelCase
name; a word that also names the measure, such as "aid", counts only beside another
word of that name. The matcher does not stem or otherwise rewrite query text.
"""

from __future__ import annotations

import re
import weakref
from collections.abc import Collection, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal, Protocol

from code_index.query import CodeIndex
from code_index.tokens import (
    Token,
    TokenLookup,
    singular,
    text_matches,
    tokenize,
    word_tokens,
)
from code_index.types import CodeRow, MeasureIndex, MeasureValueRow
from search.engine.function_words import FUNCTION_WORDS

MatchSource = Literal["code", "label", "synonym"]

# A `CodeRow` and a `MeasureValueRow` are both reachable by label/synonym
# text (see `_candidate_rows`); `_row_system` is the one place this module
# reaches past that shared shape to translate each into the axis identity
# `QualifierMatch.code_system` carries -- `CodeRow.code_system` for one,
# `MeasureValueRow.axis` for the other (see that field's own docstring in
# `code_index.types`).
MatchableRow = CodeRow | MeasureValueRow


@dataclass(frozen=True, slots=True)
class QualifierMatch:
    """One unambiguous code or measure-text match.

    `code_system` is the row's constraint property; `source` identifies its
    label or synonym. `code_dcid` and `label` are graph text safe for prompts.
    `matched_text` is raw query text and stays out of prompts. Normalised, it is
    the phrase the match reads (`phrase`).

    `start` and `end` are the match's character offsets in the variable it was read
    from (`read_by_vocabulary`), and `None` for a match read from other text: the
    query text, a dropped phrase, a sidecar. They locate a match and take no part in
    its equality, so one code read from the variable and from the query text is one
    match.
    """

    code_system: str
    code_dcid: str
    label: str
    source: MatchSource
    matched_text: str
    start: int | None = field(default=None, compare=False)
    end: int | None = field(default=None, compare=False)

    @property
    def phrase(self) -> str:
        """`matched_text` normalised: two tables whose codes match one span read the
        same phrase, which `bound_values.query_phrases_read` compares across tables."""
        return _normalize(self.matched_text)


@dataclass(frozen=True, slots=True)
class QualifierSpan:
    """Every indexed row named by one query span, ambiguous or not.

    `start` and `end` are character offsets in the matched query. They keep
    repeated identical phrases distinct. `matched_text` is raw query text
    for tracing only. `rows` contains all matching code or measure rows;
    `match_qualifiers` keeps only spans with one row.
    """

    matched_text: str
    start: int
    end: int
    rows: tuple[MatchableRow, ...]


@dataclass(frozen=True, slots=True)
class QualifierTie:
    """An ambiguous span whose rows are all `CodeRow` values.

    Callers with independent grounds to use the vocabulary can present
    `rows` as closed alternatives. Spans containing measure rows are excluded;
    `_measure_query_matches` handles those conflicts. Each row is a
    `QualifierMatch` with the same `matched_text`.
    """

    matched_text: str
    rows: tuple[QualifierMatch, ...]

    @property
    def phrase(self) -> str:
        """`matched_text` normalised, as `QualifierMatch.phrase`."""
        return _normalize(self.matched_text)


@dataclass(frozen=True, slots=True)
class QualifierReadings:
    """A span naming exactly one `CodeRow` and one `MeasureValueRow`, one match with two readings.

    Text alone cannot choose, so `match_qualifiers` refuses the span. A caller holding a
    shape keeps the reading that shape carries and refuses the span when the shape carries
    both or neither. "PPP" reads `PurchasingPowerParity` on a GDP series and CRS purpose
    25010 (public-private partnership) on an aid flow. `code` and `measure` are each a
    `QualifierMatch` with the span's `matched_text`, which is trace-only.
    """

    matched_text: str
    code: QualifierMatch
    measure: QualifierMatch


@dataclass(frozen=True, slots=True)
class CodedQualifier:
    """Resolved qualifier text and matches carried between fan-out stages.

    Span offsets refer to `phrase`, not the consumer's variable text. The
    extraction stage builds this sidecar once so later stages need not rescan.
    """

    phrase: str
    spans: tuple[QualifierSpan, ...]
    matches: tuple[QualifierMatch, ...]


def recall_text(variable: str, qualifiers: tuple[CodedQualifier, ...] | None) -> str:
    """Append qualifier phrases not already present, preserving their order.

    The composed text serves free-text retrieval and restatement; structured
    consumers use `variable` unchanged. Each phrase is checked against the
    running text, so duplicates within the variable or sidecar are skipped.
    Order affects the result and its response-cache key. Empty input returns
    `variable` unchanged.
    """
    if not qualifiers:
        return variable
    composed = variable
    for qualifier in qualifiers:
        if _normalize(qualifier.phrase) in _normalize(composed):
            continue
        composed = f"{composed} {qualifier.phrase}"
    return composed


@dataclass(frozen=True, slots=True)
class _RawSpan:
    """Internal: one claimed span plus the `MatchSource` of each of its rows,
    in `rows` order. `start`/`end` are character offsets in the query text
    passed to `_recognise_raw`, from its first and last word token; `matched_text`
    is the query text between them."""

    start: int
    end: int
    matched_text: str
    rows: tuple[MatchableRow, ...]
    sources: tuple[MatchSource, ...]


# Splits a `code_system` string into its camelCase words, each an uppercase letter with the
# lowercase or digit characters after it, or a run of lowercase or digit characters.
# "crsCooperationModality" yields ["crs", "Cooperation", "Modality"].
_CAMEL_WORD_RE = re.compile(r"[A-Z][a-z0-9]*|[a-z0-9]+")

# Tokens that join two code spans of one variable, casefolded. A union connective reads
# the two codes together ("health and education", "health & education", and "health
# including malaria control", whose union is 120), an exclusion connective removes the
# second from the first ("health excluding malaria control"), and a comparison connective
# sets them side by side ("health versus education"). The qualifier coverage rule
# (`phase_rank_bind._coverage_records`) counts a connective between two code spans as read.
UNION_CONNECTIVES = frozenset({"and", "or", "/", ",", "&", "plus", "including"})
EXCLUSION_CONNECTIVES = frozenset({"excluding", "except", "without"})
COMPARISON_CONNECTIVES = frozenset({"compare", "compared", "versus", "vs"})
CONNECTIVES = UNION_CONNECTIVES | EXCLUSION_CONNECTIVES | COMPARISON_CONNECTIVES

# The comparison connectives that name what they compare at any distance after them.
_COMPARING_VERBS = frozenset({"compare", "compared"})

# How the words between two located matches on one code system join them (`joins`).
JoinKind = Literal["union", "exclusion", "comparison", "other"]


class Located(Protocol):
    """A span of a text by its character offsets, such as a `QualifierMatch` in its
    variable or an extracted variable in the raw query (`source_spans.VariableSpan`).
    `None` offsets locate nothing."""

    @property
    def start(self) -> int | None: ...

    @property
    def end(self) -> int | None: ...


@dataclass(frozen=True, slots=True)
class GoverningConnective:
    """An exclusion or comparison connective of a variable and the first match after it.

    `start` is the connective's offset and `end` the end of the governed match, so
    `variable[start:end]` is the phrase no binding expresses: "excluding malaria control",
    "versus education", "compare aid to health". `governed` is the match's offsets.
    """

    kind: Literal["exclusion", "comparison"]
    start: int
    end: int
    governed: tuple[int, int]


def governing_connectives(
    variable: str, matches: Iterable[Located], *, transparent: frozenset[str] = frozenset()
) -> tuple[GoverningConnective, ...]:
    """Each exclusion or comparison connective of `variable` followed by a located match of
    `matches`, with the first such match after it, in variable order.

    Only function words (`FUNCTION_WORDS`) and `transparent` words, casefolded, may stand
    between a connective and its match, so "Kenya versus Uganda health ODA" governs
    nothing. A comparing verb (`_COMPARING_VERBS`) names its first object at any distance,
    so "compare aid to health" governs health. A connective no match follows governs
    nothing ("ODA compared to 2010").
    """
    spans = sorted({(m.start, m.end) for m in matches if m.start is not None and m.end is not None})
    governing: list[GoverningConnective] = []
    for token in tokenize(variable):
        folded = token.text.casefold()
        if folded in EXCLUSION_CONNECTIVES:
            kind: Literal["exclusion", "comparison"] = "exclusion"
        elif folded in COMPARISON_CONNECTIVES:
            kind = "comparison"
        else:
            continue
        span = next((span for span in spans if span[0] >= token.end), None)
        if span is None:
            continue
        if folded in _COMPARING_VERBS or all(
            word.text.casefold() in FUNCTION_WORDS | transparent
            for word in word_tokens(variable[token.end : span[0]])
        ):
            governing.append(
                GoverningConnective(kind=kind, start=token.start, end=span[1], governed=span)
            )
    return tuple(governing)


def joins(
    variable: str,
    matches: Sequence[Located],
    governing: Sequence[GoverningConnective],
    *,
    transparent: frozenset[str] = frozenset(),
) -> tuple[JoinKind, ...]:
    """How each adjacent pair of `matches` (located matches on one code system, in variable
    order) is joined, one kind per pair.

    The pair is a union when every token between the two, less the `transparent` words
    (casefolded), is a union connective ("health and education", "HIV/AIDS"). A
    connective of `governing` between the two, or one governing a union's first match
    ("compare aid to health and education"), gives the pair its own kind, which carries
    along the unions after it. Any other words between the two, or none, join them as
    "other".
    """
    kinds: list[JoinKind] = []
    for first, second in pairwise(matches):
        assert first.end is not None and second.start is not None
        between = next(
            (
                connective
                for connective in governing
                if connective.governed == (second.start, second.end)
                and first.end <= connective.start
            ),
            None,
        )
        if between is not None:
            kinds.append(between.kind)
            continue
        tokens = [
            token
            for token in tokenize(variable[first.end : second.start])
            if token.text.casefold() not in transparent
        ]
        if not tokens or not all(t.text.casefold() in UNION_CONNECTIVES for t in tokens):
            kinds.append("other")
            continue
        leading = next((c.kind for c in governing if c.governed == (first.start, first.end)), None)
        if leading is None and kinds and kinds[-1] in ("exclusion", "comparison"):
            leading = kinds[-1]
        kinds.append(leading or "union")
    return tuple(kinds)


def _normalize(text: str) -> str:
    """`text` stripped and casefolded, with no other normalisation."""
    return text.strip().casefold()


def _all_normalized_text_ranges(needle: str, haystack: str) -> list[tuple[int, int]]:
    """Return every occurrence of `needle` in `haystack`, both casefolded, as a character
    range of `haystack` itself.

    Report all occurrences so repeated LLM-supplied phrases retain the right
    offsets. Casefolding can lengthen a text ("ß" folds to "ss"), so each range is
    mapped back to the characters of `haystack` it covers, and agrees with the token
    offsets of a span read from the same text. This shared helper avoids an import
    cycle between extraction and region binding. It uses plain substring matching;
    callers decide whether a range is a genuine word occurrence.
    """
    normalized_needle = _normalize(needle)
    if not normalized_needle:
        return []
    folded: list[str] = []
    origin: list[int] = []
    for position, char in enumerate(haystack):
        for folded_char in char.casefold():
            folded.append(folded_char)
            origin.append(position)
    normalized_haystack = "".join(folded)
    ranges: list[tuple[int, int]] = []
    search_from = 0
    while True:
        start = normalized_haystack.find(normalized_needle, search_from)
        if start < 0:
            break
        end = start + len(normalized_needle)
        ranges.append((origin[start], origin[end - 1] + 1))
        search_from = start + 1
    return ranges


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether character ranges `a` and `b` share any position."""
    return a[0] < b[1] and b[0] < a[1]


def names_plain_name(text: str, plain_name: str) -> bool:
    """Whether `text` names a code system by its declared plain name ("flow", "sector"),
    singular or plural, as a whole word, case-insensitively."""
    forms = {plain_name, f"{plain_name}s", f"{plain_name}es"}
    if plain_name.endswith("y"):
        forms.add(f"{plain_name[:-1]}ies")
    return any(
        re.search(rf"\b{re.escape(form)}\b", text, re.IGNORECASE) is not None for form in forms
    )


def slot_noun_named(phrase: str, plain_names: Iterable[str]) -> str | None:
    """The one plain name in `plain_names` that `phrase` names (`names_plain_name`), or
    `None` when it names none or more than one.

    An operation over a category runs over the slot its phrase names. A phrase naming two
    ("top 5 sectors by aid flows") leaves the slot unsaid, so it runs over neither. Slot
    opening and the qualifier trim both read this, so the trim credits an operation with
    a noun only where the operation opens its slot. A noun can still leave a qualifier
    as one of a variable's own words ("top 5 aid flows" beside "aid flows"); when a
    declared default then fixes its slot, the part warns `slot_defaulted`.
    """
    named = {name for name in plain_names if names_plain_name(phrase, name)}
    return named.pop() if len(named) == 1 else None


def match_unsupported_qualifier_phrases(text: str, *, terms: Collection[str]) -> tuple[str, ...]:
    """Return each word of ``text`` that is one of ``terms``, the words a code system declares
    it has no vocabulary for (`CodeSystemDeclaration.no_vocabulary_terms`), in text order.

    Each term is compared with whole word tokens, so ``non-concessionality`` and
    ``non-concessional-like`` do not match the term ``non-concessional``.
    """
    return tuple(token.text for token in word_tokens(text) if _normalize(token.text) in terms)


def _row_system(row: MatchableRow) -> str:
    """`row`'s own axis identity: `CodeRow.code_system` for a code row,
    `MeasureValueRow.axis` for a measure value -- the field either type
    plays `QualifierMatch.code_system` with."""
    return row.code_system if isinstance(row, CodeRow) else row.axis


def _match_source(row: MatchableRow, candidate: str) -> MatchSource:
    """Whether `row`'s own label matches the query text `candidate` ("label") or only a
    synonym of it does ("synonym"). `_recognise_raw` records this source on an
    unambiguous span's `QualifierMatch` and on every row of a `QualifierTie`."""
    return "label" if text_matches(row.label, tokenize(candidate)) else "synonym"


def _measure_lookup(measure_index: MeasureIndex | None) -> TokenLookup[MeasureValueRow]:
    """Every matchable `MeasureValueRow` in `measure_index`, found by its label and
    synonyms as `CodeIndex.codes_by_tokens` finds a `CodeRow`. A row with
    `matchable=False` is omitted, so its label and synonyms resolve to nothing here even
    when they collide with another row's text. Built once per matching call (`_lexicon`),
    since `MeasureIndex` carries no lookup of its own. A caller with no measure family to
    offer passes `measure_index=None` and gets an empty lookup.

    An alias row (`same_reading_as` set) is filed under its canonical row,
    so text naming either spelling of one reading resolves to one row and
    stays an unambiguous span.
    """
    if measure_index is None:
        return TokenLookup(())
    canonical_rows = {
        (row.axis, row.dcid): row for row in measure_index.values if row.same_reading_as is None
    }
    return TokenLookup(
        (
            text,
            row if row.same_reading_as is None else canonical_rows[(row.axis, row.same_reading_as)],
        )
        for row in measure_index.values
        if row.matchable
        for text in (row.label, *row.synonyms)
    )


# Each index's longest code label or synonym in words, by index object id. A weakref
# finalizer removes the entry before Python can reuse the id.
_code_phrase_words_cache: dict[int, int] = {}


def _code_phrase_words(index: CodeIndex) -> int:
    """The longest label or synonym of a code row of `index`, in words, or 1 for an index
    with no codes. Built once per index: every matching call reads it (`_lexicon`), and
    tokenising every code text takes milliseconds a call."""
    key = id(index)
    cached = _code_phrase_words_cache.get(key)
    if cached is not None:
        return cached
    cached = max(
        (len(word_tokens(text)) for row in index.codes for text in (row.label, *row.synonyms)),
        default=1,
    )
    _code_phrase_words_cache[key] = cached
    weakref.finalize(index, _code_phrase_words_cache.pop, key, None)
    return cached


def _max_phrase_words(index: CodeIndex, measure_index: MeasureIndex | None) -> int:
    """The longest label or synonym `index` or `measure_index` carries, in
    words -- bounds the span width the matcher ever tries, without
    hardcoding a width tuned to any one code system's or measure family's
    vocabulary. A measure value with `matchable=False` is excluded: its
    label is never a span the matcher looks for, so it must not widen the
    span width tried on account of a text no lookup will ever resolve to
    it."""
    longest = _code_phrase_words(index)
    if measure_index is not None:
        for row in measure_index.values:
            if not row.matchable:
                continue
            for text in (row.label, *row.synonyms):
                longest = max(longest, len(word_tokens(text)))
    return longest


def _rows_for(
    span: Sequence[Token], lexicon: _Lexicon, vocabulary: AbstractSet[str] | None
) -> tuple[MatchableRow, ...]:
    """Every code row of a system in `vocabulary` (every system when `None`) and every
    measure value whose text matches `span`. A vocabulary reads only the codes a shape can
    bind (`bindable_codes`)."""
    codes = lexicon.index.codes_by_tokens(span)
    if vocabulary is not None:
        codes = tuple(
            row for row in codes if row.code_system in vocabulary and row.dcid in lexicon.bindable
        )
    return codes + lexicon.measure_lookup.get(span)


def _candidate_rows(
    span: Sequence[Token], lexicon: _Lexicon, vocabulary: AbstractSet[str] | None
) -> tuple[bool, tuple[MatchableRow, ...]] | None:
    """Whether the query span `span` resolves only less its trailing "s", and every
    row it resolves to (a code row of a system in `vocabulary`, every system when `None`,
    or a measure value), or `None` if it resolves to nothing at all.
    A measure value is reachable the same way a code row is: neither source
    is preferred over the other, so a span both register returns both rows
    and the caller's ambiguity check refuses it exactly as it would two
    colliding `CodeRow`s.

    Exact match is tried first. Only when the span is registered nowhere in
    the vocabulary under that exact form does the bounded plural allowance
    apply: strip one trailing "s" and look that up instead. The caller
    decides ambiguity -- a returned result may still carry more than one row.
    """
    exact_rows = _rows_for(span, lexicon, vocabulary)
    if exact_rows:
        return False, exact_rows
    less_s = singular(span)
    if less_s is not None:
        singular_rows = _rows_for(less_s, lexicon, vocabulary)
        if singular_rows:
            return True, singular_rows
    return None


@dataclass(frozen=True, slots=True)
class _SystemMarkers:
    """The words of one code system's name that make a bare token a code of it.

    `words` mark the system wherever they stand before the code. `measure_words` name the
    measure as well, so they mark nothing alone. `tail` is the name's last two words, its
    head noun and the word before it: written in that order, they mark the system whether
    or not another system's name shares them. "aid type 106" marks `dac1AidType` and
    `dac2aAidType`, and the code binds the one system holding it
    (`_lookup_code_in_systems`). "aid to 106 countries" names aid, and "type of support"
    is no system's tail, so neither marks a system.
    """

    words: frozenset[str]
    measure_words: frozenset[str]
    tail: tuple[str, ...]


def _all_system_markers(index: CodeIndex) -> dict[str, _SystemMarkers]:
    """Return each system's bare-code markers, read from its camelCase name.

    A word shared by several indexed systems, such as `crs` or `of`, cannot distinguish a
    system and marks none alone. A word a declared default lists in `when_named` names the
    measure a query asks for, so it marks nothing alone either
    (`_SystemMarkers.measure_words`). The name's last two words, in its order, mark the
    system (`_SystemMarkers.tail`). New system names need no manual marker list.
    """
    names = {
        system: tuple(word.casefold() for word in _CAMEL_WORD_RE.findall(system))
        for system in index.declarations
    }
    raw = {system: frozenset(words) for system, words in names.items()}
    word_counts: dict[str, int] = {}
    for words in raw.values():
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    measure_words = frozenset(
        word.casefold()
        for declaration in index.declarations.values()
        if declaration.default is not None
        for word in declaration.default.when_named
    )
    markers: dict[str, _SystemMarkers] = {}
    for system, words in raw.items():
        unique = frozenset(word for word in words if word_counts[word] == 1)
        markers[system] = _SystemMarkers(
            words=unique - measure_words,
            measure_words=unique & measure_words,
            tail=names[system][-2:],
        )
    return markers


def _preceding_marker_systems(
    tokens: tuple[str, ...], index_in_query: int, system_markers: Mapping[str, _SystemMarkers]
) -> frozenset[str]:
    """Every code system some token before position `index_in_query` marks -- the
    vocabulary signal that makes the bare token at that position worth looking up as a
    code at all.

    Checks every earlier token, not only the immediate predecessor: "purpose" two words
    before "120" in "...purpose code 120" makes that bare number eligible, and the word
    "code" between them carries no meaning. Two adjacent tokens mark every system whose
    name ends in them (`_SystemMarkers.tail`).
    """
    preceding = [tokens[i].casefold() for i in range(index_in_query)]
    pairs = frozenset(pairwise(preceding))
    return frozenset(
        system
        for system, markers in system_markers.items()
        if markers.words.intersection(preceding) or markers.tail in pairs
    )


def _lookup_code_in_systems(
    token: str, systems: frozenset[str], index: CodeIndex
) -> CodeRow | None:
    """The code row whose `code` string casefolds to `token`, among rows
    belonging to `systems` only -- never the whole index. Returns the row
    only when there is exactly one across those systems: a code string can
    still collide between two systems the query's own vocabulary reaches
    at once (`code_system` marker words are not required to be disjoint).
    """
    folded = token.casefold()
    matches = [
        row for row in index.codes if row.code_system in systems and row.code.casefold() == folded
    ]
    if len(matches) == 1:
        return matches[0]
    return None


@dataclass(frozen=True, slots=True)
class _Lexicon:
    """The lookup tables every scan of one call reads, built once per call so a query
    read in several vocabularies builds them once."""

    index: CodeIndex
    measure_lookup: TokenLookup[MeasureValueRow]
    system_markers: Mapping[str, _SystemMarkers]
    max_width: int
    bindable: frozenset[str]


def _lexicon(index: CodeIndex, measure_index: MeasureIndex | None) -> _Lexicon:
    return _Lexicon(
        index=index,
        measure_lookup=_measure_lookup(measure_index),
        system_markers=_all_system_markers(index),
        max_width=_max_phrase_words(index, measure_index),
        bindable=bindable_codes(index),
    )


# Each index's bindable codes, by index object id. A weakref finalizer removes the entry
# before Python can reuse the id.
_bindable_codes_cache: dict[int, frozenset[str]] = {}


def bindable_codes(index: CodeIndex) -> frozenset[str]:
    """The dcids of the codes a shape can bind: each code some StatVar row of `index` holds,
    itself or through a descendant, and every code of a system no row carries, whose
    StatVars the index does not list and so cannot rule out.

    A code of a system the rows carry that no row holds binds on no shape. "population"
    (`CRSTypeOfFinance4`) is a type of finance no CRS row carries. Built once per index.
    """
    key = id(index)
    cached = _bindable_codes_cache.get(key)
    if cached is not None:
        return cached
    bindable: set[str] = set()
    carried: set[str] = set()
    for row in (*index.statvars, *index.listed_statvars):
        carried.update(row.constraint_properties)
        for value in row.constraint_properties.values():
            code = index.code_by_dcid(value)
            while code is not None and code.dcid not in bindable:
                bindable.add(code.dcid)
                code = index.code_by_dcid(code.parent) if code.parent is not None else None
    bindable.update(code.dcid for code in index.codes if code.code_system not in carried)
    cached = frozenset(bindable)
    _bindable_codes_cache[key] = cached
    weakref.finalize(index, _bindable_codes_cache.pop, key, None)
    return cached


def _recognise_raw(
    query: str, lexicon: _Lexicon, vocabulary: AbstractSet[str] | None
) -> tuple[_RawSpan, ...]:
    """Return the spans one scan claims, sorted by start offset.

    `vocabulary` holds the code systems the scan reads; `None` reads every system. A code
    row of another system neither matches nor claims a token, so a longer label only a
    foreign system carries cannot hide a shorter one this vocabulary reads. A code no shape
    can bind (`bindable_codes`) is foreign in the same way: on a CRS shape "population"
    names no type of finance and stays free for a purpose reading. The scan in every
    system keeps such codes, so its reading states what the words name, and the measure
    values every vocabulary shares stay those of that reading ("gross" in "gross national
    income" is no netting basis). Measure values are read in every vocabulary.

    First, try the widest run of words matching a label or synonym at each position
    (`code_index.tokens`). Any match claims the words, even when ambiguous; a span of
    several code rows is a tie (`_project`). Measure-row conflicts are left to
    `_measure_query_matches`. Then resolve unclaimed single words as bare codes,
    restricted to systems named by earlier query words. Bare-code ambiguity leaves the
    word unclaimed.
    """
    tokens = tokenize(query)
    word_at = tuple(position for position, token in enumerate(tokens) if token.kind == "word")
    words = tuple(tokens[position] for position in word_at)
    n = len(words)
    if not n:
        return ()

    claimed = [False] * n
    found: list[_RawSpan] = []

    for width in range(min(lexicon.max_width, n), 0, -1):
        for start in range(0, n - width + 1):
            last = start + width - 1
            if any(claimed[start : last + 1]):
                continue
            candidate = _candidate_rows(
                tokens[word_at[start] : word_at[last] + 1], lexicon, vocabulary
            )
            if candidate is None:
                continue
            # Claim these positions regardless of the ambiguity verdict --
            # a rejected candidate still consumes the span, so a shorter
            # alternative starting here is never subsequently tried.
            for i in range(start, last + 1):
                claimed[i] = True
            plural_stripped, rows = candidate
            matched_text = query[words[start].start : words[last].end]
            source_text = matched_text[:-1] if plural_stripped else matched_text
            found.append(
                _RawSpan(
                    start=words[start].start,
                    end=words[last].end,
                    matched_text=matched_text,
                    rows=rows,
                    sources=tuple(_match_source(row, source_text) for row in rows),
                )
            )

    texts = tuple(word.text for word in words)
    for i, word in enumerate(words):
        if claimed[i]:
            continue
        systems = _preceding_marker_systems(texts, i, lexicon.system_markers)
        if vocabulary is not None:
            systems = systems.intersection(vocabulary)
        if not systems:
            continue
        row = _lookup_code_in_systems(word.text, systems, lexicon.index)
        if row is None or (vocabulary is not None and row.dcid not in lexicon.bindable):
            continue
        found.append(
            _RawSpan(
                start=word.start,
                end=word.end,
                matched_text=word.text,
                rows=(row,),
                sources=("code",),
            )
        )
        claimed[i] = True

    found.sort(key=lambda item: item.start)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class VocabularyReading:
    """How one vocabulary reads a query: every claimed span, the one-row matches among
    them, and the all-code ties, each in query order.

    A vocabulary is the set of constraint properties one shape declares. Read alone, it
    gives a shape the codes its own systems name: "bilateral ODA grants" reads ODA
    (`CRSTypeOfFlow10`) and a "grants" tie of two finance types on a CRS shape, and the
    DAC1 aid type 11016 on a DAC1 shape. A tie remains only where the shape's own systems
    share a text. A measure value keeps its reading in every code system together, so a
    word another system's longer label covers names no measure value in any vocabulary:
    "gross" in "gross national income", a CRS type of finance, is not the netting basis.
    """

    spans: tuple[QualifierSpan, ...]
    matches: tuple[QualifierMatch, ...]
    ties: tuple[QualifierTie, ...]


def _is_measure_span(span: _RawSpan) -> bool:
    """Whether `span` names one measure value and nothing else."""
    return len(span.rows) == 1 and isinstance(span.rows[0], MeasureValueRow)


def _read(
    query: str,
    lexicon: _Lexicon,
    vocabulary: AbstractSet[str] | None,
    measure_spans: tuple[_RawSpan, ...] | None = None,
    *,
    located: bool = False,
) -> VocabularyReading:
    """`_recognise_raw`'s scan of `query` in `vocabulary`, projected to spans, one-row
    matches and ties (`_project`).

    `measure_spans`, the one-measure-value spans of the scan in every code system, replace
    the scan's own: a vocabulary drops a foreign code row, but the measure reading of the
    words that row's label covers stays that of every system together. One overlapping a
    span the vocabulary reads is left out.
    """
    raw = _recognise_raw(query, lexicon, vocabulary)
    if measure_spans is not None:
        kept = [r for r in raw if not _is_measure_span(r)]
        kept += [
            m
            for m in measure_spans
            if not any(_ranges_overlap((m.start, m.end), (r.start, r.end)) for r in kept)
        ]
        raw = tuple(sorted(kept, key=lambda item: item.start))
    return _project(raw, located=located)


def _row_match(raw: _RawSpan, position: int, *, located: bool) -> QualifierMatch:
    """The `QualifierMatch` of `raw`'s row at `position`, with `raw`'s offsets when
    `located`."""
    row = raw.rows[position]
    return QualifierMatch(
        code_system=_row_system(row),
        code_dcid=row.dcid,
        label=row.label,
        source=raw.sources[position],
        matched_text=raw.matched_text,
        start=raw.start if located else None,
        end=raw.end if located else None,
    )


def _project(raw: tuple[_RawSpan, ...], *, located: bool) -> VocabularyReading:
    """A scan's spans, projected to spans, one-row matches and ties: the spans naming two
    or more rows, every one a `CodeRow`. Matches carry their offsets when `located`."""
    return VocabularyReading(
        spans=tuple(
            QualifierSpan(matched_text=r.matched_text, start=r.start, end=r.end, rows=r.rows)
            for r in raw
        ),
        matches=tuple(_row_match(r, 0, located=located) for r in raw if len(r.rows) == 1),
        ties=tuple(
            QualifierTie(
                matched_text=r.matched_text,
                rows=tuple(
                    _row_match(r, position, located=located) for position in range(len(r.rows))
                ),
            )
            for r in raw
            if len(r.rows) > 1 and all(isinstance(row, CodeRow) for row in r.rows)
        ),
    )


@dataclass(frozen=True, slots=True)
class TextReading:
    """One text read in every code system together (`everywhere`) and in each vocabulary
    (`by_vocabulary`)."""

    everywhere: VocabularyReading
    by_vocabulary: dict[frozenset[str], VocabularyReading]


def read_by_vocabulary(
    texts: Iterable[str],
    index: CodeIndex,
    measure_index: MeasureIndex | None = None,
    *,
    vocabularies: Iterable[frozenset[str]],
) -> tuple[TextReading, ...]:
    """Each of `texts` read in every code system and in each of `vocabularies`, one scan per
    text and distinct vocabulary, with the lookup tables built once for the call.

    Each vocabulary's reading sees only its own code systems (`VocabularyReading`), so a
    shape reads a text in its own systems' labels and never receives two readings of one
    token from two parses. Measure values take the reading in every code system, shared by
    every vocabulary. Each match and tie row carries its offsets in its own text.
    """
    lexicon = _lexicon(index, measure_index)
    vocabularies = tuple(dict.fromkeys(vocabularies))
    readings: list[TextReading] = []
    for text in texts:
        everywhere = _recognise_raw(text, lexicon, None)
        measure_spans = tuple(span for span in everywhere if _is_measure_span(span))
        readings.append(
            TextReading(
                everywhere=_project(everywhere, located=True),
                by_vocabulary={
                    vocabulary: _read(text, lexicon, vocabulary, measure_spans, located=True)
                    for vocabulary in vocabularies
                },
            )
        )
    return tuple(readings)


def match_qualifiers(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None = None
) -> tuple[QualifierMatch, ...]:
    """Every unambiguous code-row or measure-value recognition in `query`, in every code
    system, in the order the query text names them -- sorted by each match's own start
    position, independent of the order either internal pass discovers them in. This is the
    one-row projection of the spans `read_by_vocabulary` reads in every system: ambiguous
    candidates (a text or code string resolving to more than one row) are skipped, not
    guessed. A text two systems share
    ("ODA") is a tie here; `read_by_vocabulary` reads it once per shape vocabulary.
    """
    return _read(query, _lexicon(index, measure_index), None).matches


# The words a phrase may carry beyond its code tokens and the words of the
# code systems those tokens resolved under: "and" and "or" join two codes in
# one phrase, and "code" is the bare filler the marker rule above already
# treats as carrying no meaning of its own ("purpose code 31120").
_PHRASE_FILLER_WORDS = frozenset({"and", "or", "code"})


def _token_at(phrase: str, offset: int) -> int:
    """The index, in `word_tokens(phrase)`, of the word containing character `offset`.
    The offsets a `QualifierSpan` carries always start and end inside one of the
    phrase's own words."""
    for position, token in enumerate(word_tokens(phrase)):
        if token.start <= offset < token.end:
            return position
    return -1


def _marker_tokens_before(phrase: str, code_start: int, code_system: str) -> set[int]:
    """The positions before `code_start` occupied by some word of
    `code_system`'s own name, split on camelCase boundaries
    ("crsTypeOfFinance" yields "crs", "type", "of", "finance"), casefolded --
    the words the phrase spends on naming the code system the bare token
    after them resolves under. Computed from the token positions rather than
    from raw character offsets, so a word of that name written after the
    code is never read as one.
    """
    words = {word.casefold() for word in _CAMEL_WORD_RE.findall(code_system)}
    return {
        position
        for position, token in enumerate(word_tokens(phrase)[:code_start])
        if token.text.casefold() in words
    }


def is_coded_only_phrase(
    phrase: str, index: CodeIndex, measure_index: MeasureIndex | None
) -> tuple[tuple[QualifierSpan, ...], tuple[QualifierMatch, ...]] | None:
    """Return the phrase's spans and matches if it contains only codes, system names, and fillers.

    Bare codes require an earlier system-name word. The whole phrase must be
    accounted for; ranking text around a code causes rejection. Label and
    synonym matches, including measure values, cannot pass this test. On
    success, return the recognitions so callers can build a sidecar without a
    second scan.
    """
    reading = _read(phrase, _lexicon(index, measure_index), None)
    spans, matches = reading.spans, reading.matches
    code_pairs = {(m.matched_text, m.code_dcid) for m in matches if m.source == "code"}
    resolved: list[tuple[QualifierSpan, CodeRow]] = []
    for span in spans:
        row = span.rows[0] if len(span.rows) == 1 else None
        if isinstance(row, CodeRow) and (span.matched_text, row.dcid) in code_pairs:
            resolved.append((span, row))
    if not resolved:
        return None
    tokens = [token.text for token in word_tokens(phrase)]
    accounted: set[int] = set()
    for span, row in resolved:
        code_start = _token_at(phrase, span.start)
        accounted.update(range(code_start, _token_at(phrase, span.end - 1) + 1))
        accounted.update(_marker_tokens_before(phrase, code_start, row.code_system))
    if not all(
        position in accounted or token.casefold() in _PHRASE_FILLER_WORDS
        for position, token in enumerate(tokens)
    ):
        return None
    return spans, matches


# A party's graph name ends in the name of the list that publishes it: "non-DAC countries
# (DAC)". Code labels come without it (`CodeRow.label`).
_PARENTHETICAL_SUFFIX_RE = re.compile(r"\s*\([^()]*\)\s*$")

# Each index's party lookup, by index object id. A weakref finalizer removes the entry
# before Python can reuse the id.
_party_lookup_cache: dict[int, TokenLookup[str]] = {}


def _party_lookup(index: CodeIndex) -> TokenLookup[str]:
    """Every party of `index` with a label, found by its label less a trailing
    parenthetical. Built once per index."""
    key = id(index)
    cached = _party_lookup_cache.get(key)
    if cached is not None:
        return cached
    cached = TokenLookup(
        (_PARENTHETICAL_SUFFIX_RE.sub("", party.label), party.dcid)
        for party in index.parties.values()
        if party.label
    )
    _party_lookup_cache[key] = cached
    weakref.finalize(index, _party_lookup_cache.pop, key, None)
    return cached


def names_a_label(text: str, index: CodeIndex, measure_index: MeasureIndex | None) -> bool:
    """Whether the whole of `text` is one label or synonym of a code row or a measure value,
    read as `is_coded_only_phrase` reads them (`_candidate_rows`, one trailing "s" allowed),
    or the label of a party `index` lists: "official development assistance excluding debt
    relief" (DAC2a aid type 250) and "non-DAC countries" are; "excluding humanitarian aid"
    is not."""
    tokens = tokenize(text)
    if _candidate_rows(tokens, _lexicon(index, measure_index), None) is not None:
        return True
    return bool(_party_lookup(index).get(tokens))


def match_qualifier_ties(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None = None
) -> tuple[QualifierTie, ...]:
    """Every ambiguous, all-`CodeRow` span in `query`'s own text -- the
    ambiguous counterpart to `match_qualifiers`' unambiguous one-row
    projection, sorted the same way, by each tie's own start position.

    A caller with independent grounds to believe the vocabulary is in play
    (the same grounds `match_qualifiers`' own module docstring already
    requires before admitting an ordinary match) may surface a `QualifierTie`
    to the bind model as a closed set of alternatives instead of leaving the
    span refused. "budget support" (`CRSCooperationModalityA` and
    `CRSPurposeCode510`) yields one tie of two rows; "debt relief" (three
    code systems) yields a tie of four. A span touching a `MeasureValueRow`
    -- a same-axis measure conflict -- is never a tie: `_measure_query_matches`
    (region/pipeline.py) owns that case instead. See `_recognise_raw` for
    the shared two-pass algorithm both projections read from. This reads every code
    system; `read_by_vocabulary` reads one shape vocabulary at a time.
    """
    return _read(query, _lexicon(index, measure_index), None).ties


def _row_key(row: MatchableRow) -> list[str]:
    return [_row_system(row), row.dcid]


def qualifier_occurrences(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None = None
) -> tuple[dict[str, object], ...]:
    """Every place in `query` where a row of any code system or measure names a word, with
    nothing claimed, as JSON-safe records in query order.

    A label or synonym record holds a run of words, as its tokens joined by single spaces
    with the separators between them, the rows its exact text names and the rows its text
    less one trailing "s" names. A bare-code record holds a word and every row whose code
    it is in a system an earlier word marks. Records whose word ranges overlap, directly
    or through others, form a group. Each record names its group's ordinal, its word
    offset from the group's first word and its width in words, and no position in
    `query`: a word outside every record moves no record.

    Each vocabulary's reading of `query` (`read_by_vocabulary`), and the reading in every
    system together, is a function of these records, whatever systems a shape declares.
    A scan claims and ties only within a group, and a bare code's marker is already in its
    rows. A key on them separates any two texts some vocabulary reads apart and stays equal
    for texts every vocabulary reads alike: "ODA to Kenya" and "show me ODA to Kenya".
    """
    lexicon = _lexicon(index, measure_index)
    tokens = tokenize(query)
    word_at = tuple(position for position, token in enumerate(tokens) if token.kind == "word")
    words = tuple(tokens[position].text for position in word_at)

    def rows(span: Sequence[Token] | None) -> list[list[str]]:
        if span is None:
            return []
        found = index.codes_by_tokens(span) + lexicon.measure_lookup.get(span)
        return [_row_key(row) for row in found]

    # (word start, width in words, record) for every place a row names a word.
    found: list[tuple[int, int, dict[str, object]]] = []
    for start in range(len(words)):
        for width in range(1, min(lexicon.max_width, len(words) - start) + 1):
            span = tokens[word_at[start] : word_at[start + width - 1] + 1]
            exact = rows(span)
            singular_rows = rows(singular(span))
            if exact or singular_rows:
                found.append(
                    (
                        start,
                        width,
                        {
                            "text": " ".join(token.text for token in span),
                            "rows": exact,
                            "singular_rows": singular_rows,
                        },
                    )
                )
    for i, word in enumerate(words):
        systems = _preceding_marker_systems(words, i, lexicon.system_markers)
        if not systems:
            continue
        folded = word.casefold()
        codes = [
            _row_key(row)
            for row in index.codes
            if row.code_system in systems and row.code.casefold() == folded
        ]
        if codes:
            found.append((i, 1, {"text": word, "code_rows": codes}))
    found.sort(key=lambda item: (item[0], item[1]))
    records: list[dict[str, object]] = []
    group, group_start, group_end = -1, 0, 0
    for start, width, record in found:
        if start >= group_end:
            group, group_start = group + 1, start
        group_end = max(group_end, start + width)
        records.append({"group": group, "offset": start - group_start, "width": width, **record})
    return tuple(records)


def match_qualifier_readings(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None = None
) -> tuple[QualifierReadings, ...]:
    """Every span of `query` naming one code row and one measure value, in query order.

    A span naming any other mix of rows stays refused. See `QualifierReadings` for how a
    caller chooses between the two readings.
    """
    raw = _recognise_raw(query, _lexicon(index, measure_index), None)
    readings: list[QualifierReadings] = []
    for r in raw:
        if len(r.rows) != 2:
            continue
        matches = {
            isinstance(row, CodeRow): _row_match(r, position, located=False)
            for position, row in enumerate(r.rows)
        }
        if len(matches) == 2:
            readings.append(
                QualifierReadings(
                    matched_text=r.matched_text, code=matches[True], measure=matches[False]
                )
            )
    return tuple(readings)
