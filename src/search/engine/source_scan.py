"""Find the sources a raw-text query names by scanning it against the source dictionary.

The scan reads the query's words as `normalise_source_text` does: casefolded, with
punctuation read as spaces. At each word it takes the longest dictionary key that starts
there and runs over whole words, then moves past it. A one-word key of at most five
characters is an abbreviation (IDS, CRS, WHO, UN) and matches only where the query writes
it in capitals, so "who" and "un" in running text name nothing.

The extraction's entities and measures decide which hits are sources:

- A hit inside a longer entity name belongs to that name: "CRS" in "Catholic Relief
  Services (CRS)", "World Bank" in "World Bank IDA".
- A hit whose words are a whole measure is that measure: "adolescent birth rate", also a
  dataset's label.
- A hit inside a longer code label the query writes is that code's words: "UN" in "UN
  peacekeeping", a CRS channel.
- A hit that names an organisation (a Source, or a collection such as the DAC) and is an
  entity, or lies inside a measure's words, is a party to the flow: "ODA from the World
  Bank", "World Bank core contributions". A data word just after it ("OECD data on aid"),
  or "according to" or a negation cue just before it ("GDP of Kenya excluding the IMF"),
  makes it the source of the numbers.
- A hit that names a table (a provenance or Dataset: CRS, WEO, IDS, DAC2a, imputed
  multilateral sectors) is a source wherever it stands. A table gives and receives nothing.

A hit read as a source is no entity: an entity every occurrence of whose words a source
covers ("OECD" in "OECD data on education aid") leaves the part's entities
(`SourceScan.source_entities`), so one span gets one reading. An entity whose words also
stand elsewhere as a party stays ("World Bank data on aid from the World Bank").

A source is excluded when a negation cue ("excluding", "not", "neither", "apart from")
comes just before it, with only articles or "from" between, and included when an inclusion
cue ("including") does. A source with no cue of its own takes the polarity of the source
before it when both qualify the same measures and only "and", "or", "nor", articles,
punctuation and the words of an entity stand between them: "excluding CRS and DAC2a" and
"excluding the CRS, Norway and DAC2a" leave out both, "neither the CRS nor DAC2a" leaves
out both, and "without CRS, from DAC2a" asks for DAC2a. Any other source is included.

An occurrence of an entity's words inside an occurrence of a longer entity's words belongs
to the longer entity: in "World Bank data on World Bank IDA disbursements" the entity
"World Bank" occurs once, as the source.

Each source qualifies the measures whose words hold its words ("CRS education aid"), else
the measure whose words it directly precedes, else the nearest measure before it ("GDP
from the WEO"), else the nearest after it. A question with one measure gives it every
source, and a source no rule places qualifies every measure.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from code_index.query import CodeIndex
from search.engine.qualifier_match import read_by_vocabulary
from search.engine.source_scope import (
    SourceCollection,
    SourceDictionary,
    match_source_words,
    normalise_source_text,
)
from search.models import _PARSED_MAX_PART_SOURCES, _PARSED_MAX_SOURCE_MENTION_CHARS, SourceMention

_ABBREVIATION_MAX_CHARS = 5
"""The longest one-word key read as an abbreviation, which only capitals match."""

_NEGATION_CUES: tuple[tuple[str, ...], ...] = (
    ("excluding",),
    ("exclude",),
    ("excludes",),
    ("except",),
    ("except", "for"),
    ("not",),
    ("neither",),
    ("without",),
    ("minus",),
    ("other", "than"),
    ("apart", "from"),
    ("aside", "from"),
    ("not", "counting"),
    ("leaving", "out"),
)
"""The word runs that, just before a source, leave it out."""

_INCLUSION_CUES: tuple[tuple[str, ...], ...] = (("including",), ("include",), ("includes",))
"""The word runs that, just before a source, ask for it."""

_COORDINATORS = frozenset({"and", "or", "nor"})
"""Words that join a source to the one before it, so it shares that one's polarity."""

_ARTICLES = frozenset({"the", "a", "an", "le", "la", "el"})

_FILLER_BEFORE_SOURCE = _ARTICLES | {"from"}
"""Words that may stand between a cue and the source it introduces."""

_SOURCE_CUES: tuple[tuple[str, ...], ...] = (("according", "to"), ("selon",), ("según",))
"""The word runs that, just before an organisation, make it the source of the numbers."""

_DATA_WORDS = frozenset(
    {"data", "dataset", "database", "estimates", "figures", "numbers", "series", "statistics"}
)
"""Words that, just after an organisation, make it the source of the numbers."""


_SOURCE_PHRASE_WORDS = frozenset(
    {word for cue in (*_NEGATION_CUES, *_SOURCE_CUES) for word in cue}
    | _FILLER_BEFORE_SOURCE
    | _DATA_WORDS
    | {word for cue in _INCLUSION_CUES for word in cue}
)
"""The words a phrase may carry beside a source's own and still only name that source."""


@dataclass(frozen=True, slots=True)
class _Word:
    start: int
    end: int
    text: str
    folded: str


@dataclass(frozen=True, slots=True)
class SourceHit:
    """A source the query names: the query's characters `start` to `end`, its own words for
    the source, whether it asks for the source's numbers or leaves them out, and the
    positions of the measures it qualifies (`targets`)."""

    start: int
    end: int
    text: str
    polarity: Literal["include", "exclude"]
    targets: tuple[int, ...]


def _words(text: str) -> list[_Word]:
    """The words of `text` with their offsets, punctuation read as a space."""
    words: list[_Word] = []
    start: int | None = None
    for index, char in enumerate(f"{text} "):
        separator = char.isspace() or unicodedata.category(char).startswith("P")
        if separator and start is not None:
            piece = text[start:index]
            words.append(_Word(start, index, piece, piece.casefold()))
            start = None
        elif not separator and start is None:
            start = index
    return words


def _folded(text: str) -> tuple[str, ...]:
    return tuple(word.folded for word in _words(text))


def _written_in_capitals(word: str) -> bool:
    """Whether `word`'s letters before any digit are capitals, at least two of them, as in
    "IDS", "UN" and "DAC2a"."""
    head = word
    for index, char in enumerate(word):
        if char.isdigit():
            head = word[:index]
            break
    letters = [char for char in head if char.isalpha()]
    return len(letters) >= 2 and all(char.isupper() for char in letters)


def _key_matches(key: tuple[str, ...], words: Sequence[_Word]) -> bool:
    if tuple(word.folded for word in words) != key:
        return False
    if len(key) == 1 and len(key[0]) <= _ABBREVIATION_MAX_CHARS:
        return _written_in_capitals(words[0].text)
    return True


def _holds(container: Sequence[str], run: Sequence[str]) -> bool:
    """Whether the word run `run` occurs in `container` as whole words."""
    width = len(run)
    return any(
        tuple(container[i : i + width]) == tuple(run) for i in range(len(container) - width + 1)
    )


def _occurrences(phrase: str, words: Sequence[_Word]) -> list[tuple[int, int]]:
    """The character spans where `phrase`'s words occur as whole words in `words`."""
    target = _folded(phrase)
    if not target:
        return []
    width = len(target)
    return [
        (words[i].start, words[i + width - 1].end)
        for i in range(len(words) - width + 1)
        if tuple(word.folded for word in words[i : i + width]) == target
    ]


def _entity_occurrences(
    entity_spans: Sequence[str], words: Sequence[_Word]
) -> dict[str, list[tuple[int, int]]]:
    """Each of `entity_spans`' own occurrences in `words`: those not lying inside an
    occurrence of a longer one, which belong to the longer entity."""
    found = {phrase: _occurrences(phrase, words) for phrase in entity_spans}
    every = [span for spans in found.values() for span in spans]
    return {
        phrase: [
            (s, e)
            for s, e in spans
            if not any(os <= s and e <= oe and oe - os > e - s for os, oe in every)
        ]
        for phrase, spans in found.items()
    }


def _cued(words: Sequence[_Word], index: int, cues: Sequence[tuple[str, ...]]) -> bool:
    """Whether one of `cues` ends just before `words[index]`, or before a run of filler
    words that reaches it. A cue may end in a filler word itself ("apart from")."""
    before = [word.folded for word in words[:index]]
    end = len(before)
    while True:
        if any(len(cue) <= end and tuple(before[end - len(cue) : end]) == cue for cue in cues):
            return True
        if not end or before[end - 1] not in _FILLER_BEFORE_SOURCE:
            return False
        end -= 1


def _coordinated(between: Sequence[_Word], entities: Sequence[tuple[int, int]]) -> bool:
    """Whether `between`, the words separating two sources, only joins them: coordinators,
    articles and the words of an entity occurrence (`entities`), as in "the CRS, Norway and
    DAC2a"."""
    return all(
        word.folded in _COORDINATORS | _ARTICLES
        or any(s <= word.start and word.end <= e for s, e in entities)
        for word in between
    )


def _names_a_table(dictionary: SourceDictionary, text: str) -> bool:
    """Whether `text` names only provenances and Datasets, which publish numbers and are
    never a party to a flow."""
    named = match_source_words(dictionary, text)
    if isinstance(named, SourceCollection):
        return False
    return bool(named) and all(reference.kind != "source" for reference in named)


def _keys_by_first_word(dictionary: SourceDictionary) -> dict[str, list[tuple[str, ...]]]:
    keys: dict[str, list[tuple[str, ...]]] = {}
    for key in dictionary.keys:
        words = tuple(key.split())
        keys.setdefault(words[0], []).append(words)
    for candidates in keys.values():
        candidates.sort(key=len, reverse=True)
    return keys


def find_source_hits(
    query: str,
    *,
    dictionary: SourceDictionary,
    entity_spans: Sequence[str],
    measures: Sequence[str],
    index: CodeIndex,
) -> list[SourceHit]:
    """The sources `query` names, in query order, read against the extraction's entities
    (`entity_spans`, the query's words for each), its `measures` and the code labels
    `index` reads in it, each with its polarity and the measures it qualifies, as the
    module describes."""
    words = _words(query)
    (reading,) = read_by_vocabulary([query], index, vocabularies=())
    codes = [(span.start, span.end) for span in reading.everywhere.spans]
    keys = _keys_by_first_word(dictionary)
    entities = [
        (span, normalise_source_text(phrase))
        for phrase, spans in _entity_occurrences(entity_spans, words).items()
        for span in spans
    ]
    entity_occurrences = [span for span, _ in entities]
    measure_words = [_folded(measure) for measure in measures]
    measure_spans = [next(iter(_occurrences(measure, words)), None) for measure in measures]

    hits: list[SourceHit] = []
    previous_end = 0
    cursor = 0
    while cursor < len(words):
        first = cursor
        key = next(
            (
                key
                for key in keys.get(words[first].folded, ())
                if _key_matches(key, words[first : first + len(key)])
            ),
            None,
        )
        if key is None:
            cursor += 1
            continue
        cursor = first + len(key)
        start, end = words[first].start, words[cursor - 1].end
        text = query[start:end]

        overlapping = [name for (s, e), name in entities if s < end and start < e]
        if any(name != normalise_source_text(text) for name in overlapping):
            continue
        if key in measure_words:
            continue
        if any(s <= start and end <= e and e - s > end - start for s, e in codes):
            continue
        negated = _cued(words, first, _NEGATION_CUES)
        in_measure = any(_holds(held, key) for held in measure_words)
        if (overlapping or in_measure) and not _names_a_table(dictionary, text):
            data_word = cursor < len(words) and words[cursor].folded in _DATA_WORDS
            if not negated and not data_word and not _cued(words, first, _SOURCE_CUES):
                continue
        targets = _attach(key, start, end, measure_words, measure_spans)
        polarity: Literal["include", "exclude"] = "include"
        if negated:
            polarity = "exclude"
        elif (
            hits
            and hits[-1].targets == targets
            and not _cued(words, first, _INCLUSION_CUES)
            and _coordinated(words[previous_end:first], entity_occurrences)
        ):
            polarity = hits[-1].polarity
        hits.append(SourceHit(start, end, text, polarity, targets))
        previous_end = cursor
    return hits


def _attach(
    key: tuple[str, ...],
    start: int,
    end: int,
    measure_words: Sequence[tuple[str, ...]],
    spans: Sequence[tuple[int, int] | None],
) -> tuple[int, ...]:
    """The positions of the measures a source written as `key` at the query's characters
    `start` to `end` qualifies, given each measure's words and its span in the query
    (`None` where the query does not write it)."""
    holding = tuple(position for position, held in enumerate(measure_words) if _holds(held, key))
    if holding:
        return holding
    located = [(position, span) for position, span in enumerate(spans) if span is not None]
    after = sorted((s, position) for position, (s, _) in located if s >= end)
    before = sorted((e, position) for position, (_, e) in located if e <= start)
    if after and after[0][0] - end <= 1:
        return (after[0][1],)
    if before:
        return (before[-1][1],)
    if after:
        return (after[0][1],)
    return tuple(range(len(spans)))


@dataclass(frozen=True, slots=True)
class SourceScan:
    """What `scan_sources` reads in a query: each measure's source mentions
    (`per_measure`, parallel to the measures) and the entity spans a source took
    (`source_entities`, a subset of the spans passed in), which are no entity. A span is
    taken when it occurs in the query and a source covers every occurrence of it that lies
    inside no longer entity span's."""

    per_measure: list[tuple[SourceMention, ...]]
    source_entities: frozenset[str]


def scan_sources(
    query: str,
    *,
    dictionary: SourceDictionary,
    measures: Sequence[str],
    entity_spans: Sequence[str],
    index: CodeIndex,
) -> SourceScan:
    """The sources `find_source_hits` finds in `query`, attached to the measures they
    qualify, and the entity spans they cover. A measure keeps one mention per wording and
    polarity, at most `_PARSED_MAX_PART_SOURCES`, and an entity span leaves the entities only
    when every occurrence of it that lies inside no longer entity span's lies under a
    source."""
    hits = find_source_hits(
        query, dictionary=dictionary, entity_spans=entity_spans, measures=measures, index=index
    )
    attached: list[dict[tuple[str, str], SourceMention]] = [{} for _ in measures]
    read: list[SourceHit] = []
    for hit in hits:
        if len(hit.text) > _PARSED_MAX_SOURCE_MENTION_CHARS:
            continue
        read.append(hit)
        key = (normalise_source_text(hit.text), hit.polarity)
        for position in hit.targets:
            bucket = attached[position]
            if key not in bucket and len(bucket) < _PARSED_MAX_PART_SOURCES:
                bucket[key] = SourceMention(text=hit.text, polarity=hit.polarity)
    occurrences = _entity_occurrences(entity_spans, _words(query))
    source_entities = frozenset(
        phrase
        for phrase, spans in occurrences.items()
        if spans and all(any(s < hit.end and hit.start < e for hit in read) for s, e in spans)
    )
    return SourceScan(
        per_measure=[tuple(bucket.values()) for bucket in attached],
        source_entities=source_entities,
    )


def source_phrases(phrases: Sequence[str], mentions: Sequence[SourceMention]) -> set[str]:
    """The `phrases` that only name one of `mentions`: their words, less cues, articles and
    data words, lie within its words ("excluding the CRS", "selon le SNPC", "IMF"). The
    source mention applies them, so they are no qualifier left unapplied."""
    named = [_folded(mention.text or "") for mention in mentions]
    covered = set()
    for phrase in phrases:
        rest = tuple(word for word in _folded(phrase) if word not in _SOURCE_PHRASE_WORDS)
        if rest and any(_holds(words, rest) for words in named):
            covered.add(phrase)
    return covered
