"""Match code and measure text without deciding whether a query uses that vocabulary.

`match_qualifiers` returns only spans naming one row. Callers need independent
reason to admit the matched code system. `match_qualifier_ties` exposes
ambiguous all-code spans for a later election. `QualifierMatch.code_dcid` and
`label` are graph text safe for prompts; `matched_text` is raw query text and
must remain trace-only.

Matching uses normalized exact text. Only an exact miss may try one trailing
`s`; an exact ambiguous match stays ambiguous. Bare codes require an earlier
word unique to the code system, derived from its camelCase name; a word that
also names the measure, such as "aid", counts only beside another word of that
name. Multi-word spans require adjacent tokens separated only by whitespace.
The matcher does not stem or otherwise rewrite query text.

A span that names a declared dataset (`dataset_mention.dataset_mentions`) belongs to
the mention, so no word inside it is recognised as a qualifier: "imputed multilateral
ODA" names the imputed table, with no `dac2aAidType` code inside it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from code_index.query import CodeIndex
from code_index.types import CodeRow, MeasureIndex, MeasureValueRow
from search.engine.dataset_mention import _CAMEL_WORD_RE, dataset_mentions

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
    `matched_text` is raw query text and must remain trace-only.
    """

    code_system: str
    code_dcid: str
    label: str
    source: MatchSource
    matched_text: str


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
    passed to `_recognise_raw`, taken straight from the token offsets the
    word regex matched -- the same real position `QualifierSpan` carries
    onward, not re-derived later by searching for `matched_text`."""

    start: int
    end: int
    matched_text: str
    rows: tuple[MatchableRow, ...]
    sources: tuple[MatchSource, ...]


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# These terms are deliberate, bounded recognition for distinctions the
# deployed taxonomy cannot express. They are not a residual-word heuristic:
# every value is an exact lexical token checked by
# `match_unsupported_qualifier_phrases` below.
_UNSUPPORTED_QUALIFIER_TERMS = frozenset({"non-concessional"})


def _normalize(text: str) -> str:
    """Mirrors `code_index.query._normalize_text`: strip and casefold, no
    other normalization -- kept local so this module has no dependency on
    that lookup's private helper."""
    return text.strip().casefold()


def _all_normalized_text_ranges(needle: str, normalized_haystack: str) -> list[tuple[int, int]]:
    """Return every normalized substring occurrence as a character range.

    Report all occurrences so repeated LLM-supplied phrases retain the right
    offsets. This shared helper avoids an import cycle between extraction and
    region binding. It uses plain substring matching; callers decide whether
    a range is a genuine word occurrence.
    """
    normalized_needle = _normalize(needle)
    if not normalized_needle:
        return []
    ranges: list[tuple[int, int]] = []
    search_from = 0
    while True:
        start = normalized_haystack.find(normalized_needle, search_from)
        if start < 0:
            break
        end = start + len(normalized_needle)
        ranges.append((start, end))
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


def match_unsupported_qualifier_phrases(text: str) -> tuple[str, ...]:
    """Return exact unsupported qualifier terms occurring in ``text``.

    This is intentionally a closed lexical set, rather than a general
    residual-text detector. `_WORD_RE` makes the matching token-exact, so
    ``non-concessionality`` and ``non-concessional-like`` do not inherit the
    unsupported term's safety outcome.
    """
    return tuple(
        token.group()
        for token in _WORD_RE.finditer(text)
        if _normalize(token.group()) in _UNSUPPORTED_QUALIFIER_TERMS
    )


def _row_system(row: MatchableRow) -> str:
    """`row`'s own axis identity: `CodeRow.code_system` for a code row,
    `MeasureValueRow.axis` for a measure value -- the field either type
    plays `QualifierMatch.code_system` with."""
    return row.code_system if isinstance(row, CodeRow) else row.axis


def _match_source(row: MatchableRow, candidate_norm: str) -> MatchSource:
    """Whether `row`'s own label, normalized, is what the query text named
    ("label") or a synonym of it ("synonym") -- the source stamp
    `_recognise_raw` records on an unambiguous span's `QualifierMatch` and
    on every row of a `QualifierTie`."""
    return "label" if _normalize(row.label) == candidate_norm else "synonym"


def _measure_rows_by_normalized_text(
    measure_index: MeasureIndex | None,
) -> dict[str, tuple[MeasureValueRow, ...]]:
    """Every matchable `MeasureValueRow` in `measure_index`, keyed by its
    label and synonyms normalized the same way `CodeIndex.
    codes_by_normalized_text` keys a `CodeRow` -- the measure-value
    counterpart to that lookup table. A row with `matchable=False` is
    omitted, so its label and synonyms resolve to nothing here even when
    they collide with another row's text. Built fresh per
    `match_qualifiers` call, since `MeasureIndex` carries no lookup table of
    its own (unlike `CodeIndex`, which builds one once at construction).
    `measure_index=None` -- a caller with no measure family to offer --
    yields an empty mapping, not an error.

    An alias row (`same_reading_as` set) is filed under its canonical row,
    so text naming either spelling of one reading resolves to one row and
    stays an unambiguous span.
    """
    if measure_index is None:
        return {}
    canonical_rows = {
        (row.axis, row.dcid): row for row in measure_index.values if row.same_reading_as is None
    }
    by_text: dict[str, dict[tuple[str, str], MeasureValueRow]] = {}
    for row in measure_index.values:
        if not row.matchable:
            continue
        target = (
            row if row.same_reading_as is None else canonical_rows[(row.axis, row.same_reading_as)]
        )
        for text in {_normalize(t) for t in (row.label, *row.synonyms)}:
            by_text.setdefault(text, {})[(target.axis, target.dcid)] = target
    return {key: tuple(rows.values()) for key, rows in by_text.items()}


def _max_phrase_words(index: CodeIndex, measure_index: MeasureIndex | None) -> int:
    """The longest label or synonym `index` or `measure_index` carries, in
    words -- bounds the span width the matcher ever tries, without
    hardcoding a width tuned to any one code system's or measure family's
    vocabulary. A measure value with `matchable=False` is excluded: its
    label is never a span the matcher looks for, so it must not widen the
    span width tried on account of a text no lookup will ever resolve to
    it."""
    longest = 1
    for row in index.codes:
        for text in (row.label, *row.synonyms):
            longest = max(longest, len(text.split()))
    if measure_index is not None:
        for row in measure_index.values:
            if not row.matchable:
                continue
            for text in (row.label, *row.synonyms):
                longest = max(longest, len(text.split()))
    return longest


def _candidate_rows(
    span_norm: str,
    index: CodeIndex,
    measure_text_index: Mapping[str, tuple[MeasureValueRow, ...]],
) -> tuple[str, tuple[MatchableRow, ...]] | None:
    """The registered normalized text `span_norm` resolves to, and every
    row -- code or measure value -- carrying it, or `None` if it resolves to
    nothing at all. A measure value is reachable the same way a code row is:
    neither source is preferred over the other, so a span both register
    returns both rows and the caller's ambiguity check refuses it exactly as
    it would two colliding `CodeRow`s.

    Exact match is tried first. Only when the span is registered NOWHERE
    under that exact form does the bounded plural allowance apply: strip
    one trailing "s" and look that up instead (see the module docstring's
    "Plural text" section). The caller decides ambiguity -- a returned
    result may still carry more than one row.
    """
    exact_rows = index.codes_by_normalized_text(span_norm) + measure_text_index.get(span_norm, ())
    if exact_rows:
        return span_norm, exact_rows
    if len(span_norm) > 1 and span_norm.endswith("s"):
        singular = span_norm[:-1]
        singular_rows = index.codes_by_normalized_text(singular) + measure_text_index.get(
            singular, ()
        )
        if singular_rows:
            return singular, singular_rows
    return None


@dataclass(frozen=True, slots=True)
class _SystemMarkers:
    """The words of one code system's name that make a bare token a code of it.

    `words` mark the system wherever they stand before the code. `measure_words` name the
    measure as well ("aid" in `dac2aAidType`), so one marks the system only when another
    of its `name_words` follows it directly: "aid type 106" names the system, and "aid to
    106 countries" names aid.
    """

    words: frozenset[str]
    measure_words: frozenset[str]
    name_words: frozenset[str]


def _all_system_markers(index: CodeIndex) -> dict[str, _SystemMarkers]:
    """Return each system's bare-code markers, read from its camelCase name.

    A word shared by several indexed systems, such as `crs` or `of`, cannot distinguish a
    system and marks none. A word a declared default lists in `when_named` names the
    measure a query asks for, so it marks its system only beside another word of the
    system's name (`_SystemMarkers.measure_words`). New system names need no manual
    marker list.
    """
    raw = {
        system: frozenset(word.casefold() for word in _CAMEL_WORD_RE.findall(system))
        for system in index.declarations
    }
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
            name_words=words,
        )
    return markers


def _preceding_marker_systems(
    tokens: tuple[str, ...], index_in_query: int, system_markers: dict[str, _SystemMarkers]
) -> frozenset[str]:
    """Every code system some token before position `index_in_query` marks -- the
    vocabulary signal that makes the bare token at that position worth looking up as a
    code at all.

    Checks every earlier token, not only the immediate predecessor: "purpose" two words
    before "120" in "...purpose code 120" makes that bare number eligible, and the word
    "code" between them carries no meaning. A measure word marks its system only when the
    next token, itself before the code, is another word of the system's name.
    """
    preceding = [tokens[i].casefold() for i in range(index_in_query)]
    return frozenset(
        system
        for system, markers in system_markers.items()
        if markers.words.intersection(preceding)
        or any(
            word in markers.measure_words and following in markers.name_words
            for word, following in pairwise(preceding)
        )
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


def _span_is_contiguous(
    offsets: tuple[tuple[int, int], ...], start: int, width: int, query: str
) -> bool:
    """Whether the tokens at `start .. start + width` are genuinely
    adjacent in `query`'s own text -- the gap between each consecutive pair
    holds nothing but whitespace. `offsets` are the `(begin, end)` match
    spans `_WORD_RE.finditer` reported for each token, the same tokens
    `" ".join` later reassembles into a candidate span. Two tokens that
    survive `_WORD_RE.findall` as neighbours only because the
    sentence-ending period or comma between them was discarded are not
    adjacent in the query and must not be treated as one span; arbitrary
    whitespace between two words that genuinely are adjacent still counts
    as adjacent, since `" ".join` already normalizes that.
    """
    for i in range(start, start + width - 1):
        _, end = offsets[i]
        next_start, _ = offsets[i + 1]
        if query[end:next_start].strip():
            return False
    return True


@dataclass(frozen=True, slots=True)
class _RawTie:
    """Internal: one `QualifierTie` plus the `start` position `match_qualifier_ties`
    sorts by -- the tie counterpart to `_RawSpan.start`."""

    start: int
    tie: QualifierTie


def _recognise_raw(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None
) -> tuple[tuple[_RawSpan, ...], tuple[_RawTie, ...]]:
    """Return claimed spans and ambiguous all-code ties from one scan.

    Every token inside a dataset mention starts claimed. First, try the widest adjacent
    label or synonym span at each position.
    Any match claims the tokens, even when ambiguous; all-code ambiguities
    also become ties. Measure-row conflicts are left to
    `_measure_query_matches`. Then resolve unclaimed single tokens as bare
    codes, restricted to systems named by earlier query words. Bare-code
    ambiguity leaves the token unclaimed. Results are sorted by start offset.
    """
    token_matches = tuple(_WORD_RE.finditer(query))
    tokens = tuple(m.group() for m in token_matches)
    if not tokens:
        return (), ()
    offsets = tuple((m.start(), m.end()) for m in token_matches)

    n = len(tokens)
    mentions = dataset_mentions(query, index)
    claimed = [
        any(mention.start < end and start < mention.end for mention in mentions)
        for start, end in offsets
    ]
    found: list[_RawSpan] = []
    ties: list[_RawTie] = []
    system_markers = _all_system_markers(index)
    measure_text_index = _measure_rows_by_normalized_text(measure_index)

    max_width = min(_max_phrase_words(index, measure_index), n)
    for width in range(max_width, 0, -1):
        for start in range(0, n - width + 1):
            if any(claimed[start : start + width]):
                continue
            if not _span_is_contiguous(offsets, start, width, query):
                continue
            span = tokens[start : start + width]
            matched_text = " ".join(span)
            char_start = offsets[start][0]
            char_end = offsets[start + width - 1][1]
            span_norm = _normalize(matched_text)
            candidate = _candidate_rows(span_norm, index, measure_text_index)
            if candidate is None:
                continue
            # Claim these positions regardless of the ambiguity verdict --
            # a rejected candidate still consumes the span, so a shorter
            # alternative starting here is never subsequently tried.
            for i in range(start, start + width):
                claimed[i] = True
            candidate_norm, rows = candidate
            if len(rows) > 1 and all(isinstance(row, CodeRow) for row in rows):
                ties.append(
                    _RawTie(
                        start=start,
                        tie=QualifierTie(
                            matched_text=matched_text,
                            rows=tuple(
                                QualifierMatch(
                                    code_system=_row_system(row),
                                    code_dcid=row.dcid,
                                    label=row.label,
                                    source=_match_source(row, candidate_norm),
                                    matched_text=matched_text,
                                )
                                for row in rows
                            ),
                        ),
                    )
                )
            found.append(
                _RawSpan(
                    start=char_start,
                    end=char_end,
                    matched_text=matched_text,
                    rows=rows,
                    sources=tuple(_match_source(row, candidate_norm) for row in rows),
                )
            )

    for i, token in enumerate(tokens):
        if claimed[i]:
            continue
        systems = _preceding_marker_systems(tokens, i, system_markers)
        if not systems:
            continue
        row = _lookup_code_in_systems(token, systems, index)
        if row is None:
            continue
        found.append(
            _RawSpan(
                start=offsets[i][0],
                end=offsets[i][1],
                matched_text=token,
                rows=(row,),
                sources=("code",),
            )
        )
        claimed[i] = True

    found.sort(key=lambda item: item.start)
    ties.sort(key=lambda item: item.start)
    return tuple(found), tuple(ties)


def recognise_spans(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None = None
) -> tuple[QualifierSpan, ...]:
    """Every claimed span of `query`'s own text, ambiguous or not, in the
    order the query names them -- the superset `match_qualifiers` projects
    down to its unambiguous one-row subset. A caller with independent
    grounds to admit an ambiguous span (see the module docstring's own
    caution about that) reads `QualifierSpan.rows` itself; this function
    makes no ambiguity call.
    """
    raw, _ties = _recognise_raw(query, index, measure_index)
    return tuple(
        QualifierSpan(matched_text=r.matched_text, start=r.start, end=r.end, rows=r.rows)
        for r in raw
    )


def match_qualifiers(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None = None
) -> tuple[QualifierMatch, ...]:
    """Every unambiguous code-row or measure-value recognition in `query`,
    in the order the query text names them -- sorted by each match's own
    start position, independent of the order either internal pass discovers
    them in. This is `recognise_spans`' one-row projection: ambiguous
    candidates (a text or code string resolving to more than one row) are
    skipped, not guessed. See `recognise_spans` for the two-pass algorithm
    that produces the underlying spans.
    """
    raw, _ties = _recognise_raw(query, index, measure_index)
    return tuple(
        QualifierMatch(
            code_system=_row_system(r.rows[0]),
            code_dcid=r.rows[0].dcid,
            label=r.rows[0].label,
            source=r.sources[0],
            matched_text=r.matched_text,
        )
        for r in raw
        if len(r.rows) == 1
    )


# The words a phrase may carry beyond its code tokens and the words of the
# code systems those tokens resolved under: "and" and "or" join two codes in
# one phrase, and "code" is the bare filler the marker rule above already
# treats as carrying no meaning of its own ("purpose code 31120").
_PHRASE_FILLER_WORDS = frozenset({"and", "or", "code"})


def _token_at(phrase: str, offset: int) -> int:
    """The index, in `_WORD_RE.findall(phrase)`, of the token containing
    character `offset`. The offsets a `QualifierSpan` carries are always
    inside one of the phrase's own tokens, whose shape this module reads
    from the same `_WORD_RE` the matcher tokenizes with."""
    for position, token in enumerate(_WORD_RE.finditer(phrase)):
        if token.start() <= offset < token.end():
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
        for position, token in enumerate(_WORD_RE.findall(phrase)[:code_start])
        if token.casefold() in words
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
    spans = recognise_spans(phrase, index, measure_index)
    matches = match_qualifiers(phrase, index, measure_index)
    code_pairs = {(m.matched_text, m.code_dcid) for m in matches if m.source == "code"}
    resolved: list[tuple[QualifierSpan, CodeRow]] = []
    for span in spans:
        row = span.rows[0] if len(span.rows) == 1 else None
        if isinstance(row, CodeRow) and (span.matched_text, row.dcid) in code_pairs:
            resolved.append((span, row))
    if not resolved:
        return None
    tokens = _WORD_RE.findall(phrase)
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
    (region/pipeline.py) owns that case instead. See `recognise_spans` for
    the shared two-pass algorithm both projections read from.
    """
    _raw, ties = _recognise_raw(query, index, measure_index)
    return tuple(t.tie for t in ties)


def match_qualifier_readings(
    query: str, index: CodeIndex, measure_index: MeasureIndex | None = None
) -> tuple[QualifierReadings, ...]:
    """Every span of `query` naming one code row and one measure value, in query order.

    A span naming any other mix of rows stays refused. See `QualifierReadings` for how a
    caller chooses between the two readings.
    """
    raw, _ties = _recognise_raw(query, index, measure_index)
    readings: list[QualifierReadings] = []
    for r in raw:
        if len(r.rows) != 2:
            continue
        matches = {
            isinstance(row, CodeRow): QualifierMatch(
                code_system=_row_system(row),
                code_dcid=row.dcid,
                label=row.label,
                source=source,
                matched_text=r.matched_text,
            )
            for row, source in zip(r.rows, r.sources, strict=True)
        }
        if len(matches) == 2:
            readings.append(
                QualifierReadings(
                    matched_text=r.matched_text, code=matches[True], measure=matches[False]
                )
            )
    return tuple(readings)
