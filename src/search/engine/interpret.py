"""Recall stage: graph-backed SV detection and entity resolution.

Calls graph.detect_svs() for candidate SV dcids on the bare variable text, or takes the
StatVars a part's series code names in its place, and
entity_resolution.EntityResolver.resolve() for every entity name recall() was
not handed a cached resolution for -- one call for the whole miss set,
concurrent with detect_svs in the same gather.

All graph calls run in asyncio.to_thread (graph client is sync); resolver.resolve
is itself async and awaited directly alongside the to_thread coroutine.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from concurrent.futures import Future
from dataclasses import dataclass, replace

from pydantic import BaseModel, ConfigDict, Field

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.tokens import word_tokens
from entity_resolution import EntityResolver, Resolution, Resolved
from origin_catalogue import current_catalogue
from search.engine.clients.graph import (
    DetectedSV,
    Detection,
    EngineGraphClient,
    cut_hits,
    settled_un,
    stand_candidates,
    stand_in,
)
from search.engine.config import settings
from search.engine.retrieval.topics import is_topic_candidate
from search.engine.surface_resolution import resolve_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PinnedResolved(Resolved):
    """A caller-pinned dcid (the parsed door's `ParsedEntity.dcid`), graph-
    confirmed by type only: `entity.name` is the graph's own label for the
    dcid, and the caller's text is only the key it was pinned under. Every
    reader treats it as a `Resolved`; grounding alone tells it apart, so
    the caller's text never overrides the graph label the way a resolver-
    accepted alias does (region/state.py's `entity_label_text_by_dcid`).
    """


@functools.cache
def _alias_pattern(aliases: tuple[str, ...]) -> re.Pattern[str] | None:
    """The whole-word, case-insensitive pattern matching any of `aliases`."""
    if not aliases:
        return None
    return re.compile(r"\b(" + "|".join(map(re.escape, aliases)) + r")\b", re.IGNORECASE)


def _canonicalize_acronyms(text: str) -> str:
    """Expand the aliases the code index declares (`CodeIndex.variable_aliases`) at word
    boundaries, preserving surrounding text.

    Case-insensitive and controlled by `settings().acronym_canonicalize`.
    Applied only to recall's variable query, not entities or displayed fields.
    """
    if not settings().acronym_canonicalize:
        return text
    aliases = get_index().variable_aliases
    pattern = _alias_pattern(tuple(aliases))
    if pattern is None:
        return text
    return pattern.sub(lambda m: aliases[m.group(0).lower()], text)


class Recall(BaseModel):
    """Graph-backed recall: candidate SVs (dcid + score + type_of + sentence) and
    entity resolution.

    candidates is the single source of truth; candidate_svs and candidate_sv_scores
    are thin derived views kept so existing call sites (region/__init__.py, shape.py) that
    only need the dcid/score parallel lists stay unchanged.

    resolutions carries every entity name's own entity_resolution.Resolution
    (Resolved, Group, Ambiguous, NotHeld, or NotFound) -- a hit or a miss,
    always present for every name in the entities recall() was called with,
    plus any pre_resolved name still carried over.
    region/phase_intake.py's _phase_resolve_places reads this
    result into the request's place sets (`ResolveState.place_sets`), which
    every later phase reads for the places a name binds.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidates: list[DetectedSV]  # descending by score (detect_svs's contract)
    resolutions: dict[str, Resolution]  # surface name → Resolution (hit or miss)
    # The UN System Data Commons' admitted candidates, None when its resolve failed. The
    # resolve may still be running: the UN leg beside our shape derivation waits for it
    # (`settle_un`, region/phase_intake.py), and `un_candidates` and `un_unavailable` read
    # it once settled.
    un_leg: Future[tuple[DetectedSV, ...] | None] = Field(default_factory=lambda: settled_un(()))
    # Our resolve's hits under their own dcids, above the recall floor (`Detection.hits`):
    # our pool's derivation confirms these. `None` reads as `candidates`.
    hits: list[DetectedSV] | None = None

    async def settle_un(self) -> None:
        """Wait for the UN System Data Commons resolve to finish."""
        await asyncio.wrap_future(self.un_leg)

    @property
    def un_candidates(self) -> list[DetectedSV]:
        """The settled UN leg's candidates above the recall floor, descending by score.
        Raises `TimeoutError` while the resolve is running."""
        un = self.un_leg.result(timeout=0)
        return [] if un is None else _above_recall_floor(un)

    @property
    def un_unavailable(self) -> bool:
        """The settled UN resolve failed, so `un_candidates` is empty. Raises `TimeoutError`
        while the resolve is running."""
        return self.un_leg.result(timeout=0) is None

    @property
    def candidate_svs(self) -> list[str]:
        """Derived view: candidate dcids, in detect_svs's descending-score order."""
        return [c.dcid for c in self.candidates]

    @property
    def candidate_sv_scores(self) -> list[float]:
        """Derived view: candidate scores, parallel to candidate_svs."""
        return [c.score for c in self.candidates]


def _above_recall_floor(candidates: Sequence[DetectedSV]) -> list[DetectedSV]:
    """`candidates` without those under `SEARCH_RECALL_SCORE_FLOOR`, Topics exempt."""
    floor = settings().recall_score_floor
    if floor <= 0:
        return list(candidates)
    # Topic candidates are exempt: the floor runs here, strictly before
    # derive.derive_shapes(), where Topic expansion (expand_topics) picks
    # BFS roots from this same recalled pool. A low-scoring-but-legitimate
    # Topic root dropped here would never reach derive_shapes at all, silently
    # losing its whole expansion subtree with no stamped score to debug from.
    # Non-Topic candidates are still subject to the floor.
    return [c for c in candidates if c.score >= floor or is_topic_candidate(c.dcid, c.type_of)]


def _without_stated_words(
    text: str, index: CodeIndex, party_words: AbstractSet[str] = frozenset()
) -> tuple[str, frozenset[str]] | None:
    """`text` less each word some declared dataset states of its rows
    (`DatasetDeclaration.stated_words`) and each of `party_words`, with its whitespace
    collapsed, and the words removed, casefolded. A word is a whole token (`word_tokens`),
    so "non-bilateral" holds no stated word. `None` when `text` holds none, or nothing else.

    The graph's text search matches a variable's name. "Bilateral ODA" returns the DAC1
    series whose names hold "bilateral" and none of the CRS or DAC2a series, which state
    the word of every row without naming it, so those tables never become candidates.
    "Multilateral ODA" returns no DAC2a series either, though DAC2a's provider
    `DACMultilaterals` is what the word names (`DatasetDeclaration.party_words`)."""
    stated = {
        word.word for dataset in index.datasets.values() for word in dataset.stated_words
    } | party_words
    tokens = word_tokens(text)
    removed = [token for token in tokens if token.text.casefold() in stated]
    if not removed or len(removed) == len(tokens):
        return None
    kept: list[str] = []
    position = 0
    for token in removed:
        kept.append(text[position : token.start])
        position = token.end
    kept.append(text[position:])
    return " ".join("".join(kept).split()), frozenset(token.text.casefold() for token in removed)


def _states_a_word(
    dcid: str,
    words: AbstractSet[str],
    index: CodeIndex,
    party_words: AbstractSet[str] = frozenset(),
) -> bool:
    """Whether the index row of `dcid` (`CodeIndex.statvar_by_dcid`) belongs to a dataset
    that states one of `words` of the row's measured property, or names a party for one of
    them that is also among `party_words` (`DatasetDeclaration.party_words`). A series the
    index holds no row for states nothing."""
    row = index.statvar_by_dcid(dcid)
    return row is not None and any(
        declared.word in words and declared.measured_property == row.measured_property
        for provenance in row.provenances
        if provenance in index.datasets
        for declared in (
            *index.datasets[provenance].stated_words,
            *(
                named
                for named in index.datasets[provenance].party_words
                if named.word in party_words
            ),
        )
    )


def _pooled(first: Detection, second: Detection, joins: Callable[[str], bool]) -> Detection:
    """`first` with the series of `second` that `joins` admits by the dcid each stands as
    (`stand_in`). `candidates` and `hits` each hold a series once, a recalled row before a
    rival-family row (`DetectedSV.rival`) and then at its higher score, in descending
    score. A rival-family row of `second` does not join. A `hits` of `None` reads as that
    detection's `candidates` and stays `None` only when both are.

    The pool is not cut again, so it holds at most twice `SEARCH_DETECT_K` hits: the two
    scores rate different texts, and a cut would let one search displace the other. The UN
    leg is `first`'s, the resolve of the text as asked."""
    catalogue = current_catalogue()

    def joining(svs: Sequence[DetectedSV]) -> list[DetectedSV]:
        return [
            sv
            for sv in svs
            if not sv.rival
            and (standing := stand_in(sv.dcid, catalogue=catalogue)) is not None
            and joins(standing)
        ]

    def union(ours: Sequence[DetectedSV], theirs: Sequence[DetectedSV]) -> tuple[DetectedSV, ...]:
        best: dict[str, DetectedSV] = {}
        for sv in (*ours, *joining(theirs)):
            held = best.get(sv.dcid)
            if held is None or (sv.rival, -sv.score) < (held.rival, -held.score):
                best[sv.dcid] = sv
        return tuple(sorted(best.values(), key=lambda sv: sv.score, reverse=True))

    hits = None
    if first.hits is not None or second.hits is not None:
        hits = union(
            first.candidates if first.hits is None else first.hits,
            second.candidates if second.hits is None else second.hits,
        )
    return replace(first, candidates=union(first.candidates, second.candidates), hits=hits)


async def _detect(
    graph: EngineGraphClient, text: str, party_words: AbstractSet[str] = frozenset()
) -> Detection:
    """The graph's detection of `text` (`EngineGraphClient.detect_svs`). When `text` holds a
    word some declared dataset states, or one of `party_words` (`_without_stated_words`),
    the graph is searched concurrently on the text without it, and the series of the
    datasets that state or name a party for a removed word join the pool (`_states_a_word`,
    `_pooled`). A failure of either search raises."""
    index = get_index()
    unstated = _without_stated_words(text, index, party_words)
    if unstated is None:
        return await asyncio.to_thread(graph.detect_svs, text)
    rest, removed = unstated
    first, second = await asyncio.gather(
        asyncio.to_thread(graph.detect_svs, text),
        asyncio.to_thread(graph.detect_svs, rest),
    )
    return _pooled(first, second, lambda dcid: _states_a_word(dcid, removed, index, party_words))


def _member_detection(members: Sequence[str]) -> Detection:
    """`members` as a detection whose hits the request's catalogue admits (`cut_hits`),
    standing as `detect_svs` stands its own (`stand_candidates`), with no UN candidate."""
    catalogue = current_catalogue()
    pool = [
        DetectedSV(dcid=dcid, score=1.0, type_of="StatisticalVariable", sentence=None)
        for dcid in members
    ]
    hits = cut_hits(pool, catalogue=catalogue, detect_k=len(pool))
    return Detection(
        candidates=stand_candidates(hits, catalogue=catalogue), un=settled_un(()), hits=hits
    )


async def recall(
    variable: str,
    entities: list[str],
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    raw_query: str | None = None,
    pre_resolved: dict[str, Resolution] | None = None,
    entity_surfaces: Mapping[str, str] | None = None,
    members: Sequence[str] | None = None,
    party_words: AbstractSet[str] = frozenset(),
) -> Recall:
    """Recall candidate StatVars and resolve entity names concurrently.

    `raw_query` overrides `variable` for graph detection; callers use it to
    add coded-qualifier retrieval hints. Those hints do not constrain the
    population. `pre_resolved` supplies entity results shared across variables;
    only missing names go to one resolver call alongside `detect_svs`. `entity_surfaces`
    maps a name to the query words that name it, which that call resolves it through
    (`surface_resolution.resolve_names`).

    Candidates stay in descending score order. The recall floor drops low
    scores except Topic candidates. Results remain unconfirmed until the
    materialise stage reads the graph.

    A text holding a word some declared dataset states is also searched without it, for
    the series of the tables that state the word (`_detect`), and so is one holding a word
    of `party_words`, the words that name a party on an axis the request leaves unnamed
    (`region.place_roles.stated_party_words`), for the series of the tables that declare it.

    `members` are the StatVars a part's series code names (`search.engine.source_code`).
    They are the candidates, each at score 1.0 and admitted by the request's catalogue as a
    resolve's hits are, and neither resolver is asked, so the part has no UN candidate.
    """
    detect_query = raw_query or variable
    _pre = pre_resolved or {}
    cache_hits = {name: _pre[name] for name in entities if name in _pre}
    misses = [name for name in entities if name not in _pre]
    surfaces = entity_surfaces or {}

    if members is not None:
        detection = _member_detection(members)
        miss_resolutions = await resolve_names(misses, resolver=resolver, surfaces=surfaces)
    else:
        detection, miss_resolutions = await asyncio.gather(
            _detect(graph, _canonicalize_acronyms(detect_query), party_words),
            resolve_names(misses, resolver=resolver, surfaces=surfaces),
        )
    candidates = _above_recall_floor(detection.candidates)
    resolutions: dict[str, Resolution] = dict(cache_hits)
    resolutions.update(zip(misses, miss_resolutions, strict=True))
    for c in candidates:
        logger.debug(
            "recall candidate dcid=%s score=%.4f sentence=%r",
            c.dcid,
            c.score,
            c.sentence,
        )

    return Recall(
        candidates=candidates,
        resolutions=resolutions,
        un_leg=detection.un,
        hits=None if detection.hits is None else _above_recall_floor(detection.hits),
    )
