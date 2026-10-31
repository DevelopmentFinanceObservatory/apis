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
  Bank", "World Bank core contributions". A data word just after it ("OECD data on aid")
  or "according to" just before it makes it the source of the numbers.
- A hit that names a table (a provenance or Dataset: CRS, WEO, IDS, DAC2a, imputed
  multilateral sectors) is a source wherever it stands. A table gives and receives nothing.

A hit read as a source is no entity: an entity every occurrence of whose words a source
covers ("OECD" in "OECD data on education aid") leaves the part's entities
(`SourceScan.source_entities`), so one span gets one reading. An entity whose words also
stand elsewhere as a party stays ("World Bank data on aid from the World Bank").

The scan reads the text with each stretch that asks to leave something out blanked
(`exclusion_read.admit_exclusions`), so every source it finds there is asked for. A stretch
is read on its own for the sources it leaves out (`_left_out_sources`): each item of its list
whose left-out words a source covers, so "excluding the CRS, Norway and DAC2a" leaves out the
CRS and DAC2a, and "without IMF credit", whose "credit" no source covers, leaves out no IMF.

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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from code_index.query import CodeIndex
from search.engine.qualifier_match import read_by_vocabulary
from search.engine.source_scope import (
    SourceCollection,
    SourceDictionary,
    match_source_words,
    normalise_source_text,
)
from search.models import _PARSED_MAX_PART_SOURCES, _PARSED_MAX_SOURCE_MENTION_CHARS, SourceMention

if TYPE_CHECKING:
    from search.engine.exclusion_read import LeftOut

_ABBREVIATION_MAX_CHARS = 5
"""The longest one-word key read as an abbreviation, which only capitals match."""

_INCLUSION_CUES: tuple[tuple[str, ...], ...] = (("including",), ("include",), ("includes",))
"""The word runs that, just before a source, ask for it."""

_COORDINATORS = frozenset({"and", "or", "nor"})
"""Words that end one item of a stretch's list and start the next."""

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
    {word for cue in _SOURCE_CUES for word in cue}
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
    """A source the query names: the text's characters `start` to `end`, its own words for
    the source, whether it asks for the source's numbers or leaves them out, and the
    positions of the measures it qualifies (`targets`). A source a stretch leaves out sits at
    the stretch's blank."""

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
    `index` reads in it, each with the measures it qualifies, as the module describes. Each
    asks for its source."""
    words = _words(query)
    (reading,) = read_by_vocabulary([query], index, vocabularies=())
    codes = [(span.start, span.end) for span in reading.everywhere.spans]
    keys = _keys_by_first_word(dictionary)
    entities = [
        (span, normalise_source_text(phrase))
        for phrase, spans in _entity_occurrences(entity_spans, words).items()
        for span in spans
    ]
    measure_words = [_folded(measure) for measure in measures]
    measure_spans = [next(iter(_occurrences(measure, words)), None) for measure in measures]

    hits: list[SourceHit] = []
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
        in_measure = any(_holds(held, key) for held in measure_words)
        if (overlapping or in_measure) and not _names_a_table(dictionary, text):
            data_word = cursor < len(words) and words[cursor].folded in _DATA_WORDS
            if not data_word and not _cued(words, first, _SOURCE_CUES):
                continue
        targets = _attach(key, start, end, measure_words, measure_spans)
        hits.append(SourceHit(start, end, text, "include", targets))
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
    (`per_measure`, parallel to the measures), the entity spans a source took
    (`source_entities`, a subset of the spans passed in), which are no entity, and the
    stretches that leave out sources and nothing else (`source_stretches`), which their
    mentions apply. A span is taken when it occurs in the query and a source covers every
    occurrence of it that lies inside no longer entity span's."""

    per_measure: list[tuple[SourceMention, ...]]
    source_entities: frozenset[str]
    source_stretches: frozenset[str] = frozenset()


def _left_out_sources(
    stretch: LeftOut, *, dictionary: SourceDictionary, index: CodeIndex
) -> tuple[list[SourceHit], bool]:
    """The sources `stretch` leaves out, and whether it leaves out nothing else.

    The stretch's words split into the items of its list, at commas and at "and", "or" and
    "nor". An item leaves out a source when each of its left-out words (`LeftOut.things`),
    less articles and "from", lies in a source the item names."""
    named = find_source_hits(
        stretch.phrase, dictionary=dictionary, entity_spans=(), measures=(), index=index
    )
    asked = {word for thing in stretch.things for word in _folded(thing)} - _FILLER_BEFORE_SOURCE
    if not asked:
        return [], False
    items: list[list[_Word]] = [[]]
    previous_end = 0
    for word in _words(stretch.phrase):
        if word.folded in _COORDINATORS:
            items.append([])
        else:
            if "," in stretch.phrase[previous_end : word.start]:
                items.append([])
            items[-1].append(word)
        previous_end = word.end
    hits: list[SourceHit] = []
    whole = bool(named)
    for item in items:
        held = [hit for hit in named if any(hit.start <= w.start < hit.end for w in item)]
        rest = [
            word
            for word in item
            if word.folded in asked and not any(h.start <= word.start < h.end for h in held)
        ]
        if rest:
            whole = False
        else:
            hits.extend(held)
    return hits, whole


def scan_sources(
    query: str,
    *,
    dictionary: SourceDictionary,
    measures: Sequence[str],
    entity_spans: Sequence[str],
    index: CodeIndex,
    left_out: Sequence[LeftOut] = (),
) -> SourceScan:
    """The sources `find_source_hits` finds in `query`, the text with each of the `left_out`
    stretches blanked, and those each stretch leaves out (`_left_out_sources`), attached to the
    measures they qualify, and the entity spans they cover. A stretch's sources attach from
    its blank. A measure keeps one mention per wording and polarity, at most
    `_PARSED_MAX_PART_SOURCES`, and an entity span leaves the entities only when every
    occurrence of it that lies inside no longer entity span's lies under a source."""
    hits = find_source_hits(
        query, dictionary=dictionary, entity_spans=entity_spans, measures=measures, index=index
    )
    words = _words(query)
    measure_words = [_folded(measure) for measure in measures]
    measure_spans = [next(iter(_occurrences(measure, words)), None) for measure in measures]
    source_stretches: set[str] = set()
    for stretch in left_out:
        found, whole = _left_out_sources(stretch, dictionary=dictionary, index=index)
        hits += [
            replace(
                hit,
                start=stretch.at,
                end=stretch.at + 1,
                polarity="exclude",
                targets=_attach(
                    _folded(hit.text), stretch.at, stretch.at + 1, measure_words, measure_spans
                ),
            )
            for hit in found
        ]
        if whole:
            source_stretches.add(stretch.phrase)
    hits.sort(key=lambda hit: hit.start)
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
        source_stretches=frozenset(source_stretches),
    )


def source_phrases(phrases: Sequence[str], mentions: Sequence[SourceMention]) -> set[str]:
    """The `phrases` that only name one of `mentions`: their words, less cues, articles and
    data words, lie within its words ("selon le SNPC", "IMF"). The source mention applies
    them, so they are no qualifier left unapplied."""
    named = [_folded(mention.text or "") for mention in mentions]
    covered = set()
    for phrase in phrases:
        rest = tuple(word for word in _folded(phrase) if word not in _SOURCE_PHRASE_WORDS)
        if rest and any(_holds(words, rest) for words in named):
            covered.add(phrase)
    return covered
