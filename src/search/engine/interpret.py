"""Recall stage: graph-backed SV detection and entity resolution.

Calls graph.detect_svs() for candidate SV dcids on the bare variable text, and
entity_resolution.EntityResolver.resolve() for every entity name recall() was
not handed a cached resolution for -- one call for the whole miss set,
concurrent with detect_svs in the same gather.

All graph calls run in asyncio.to_thread (graph client is sync); resolver.resolve
is itself async and awaited directly alongside the to_thread coroutine.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from entity_resolution import EntityResolver, Group, Resolution, Resolved
from search.engine.clients.graph import DetectedSV, EngineGraphClient, settled_un
from search.engine.config import settings
from search.engine.retrieval.topics import is_topic_candidate

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


# Expand acronyms with demonstrated retrieval confusion before bare-variable recall.
# GNI scores 0.6342 for GiniIndex_EconomicActivity; its full term finds the aggregate.
# U5MR and IMR hit unrelated SVs, while their full terms find the mortality variables.
# CO2's full term scores higher than the acronym. GDP recalls correctly without expansion.
_ACRONYM_EXPANSIONS: dict[str, str] = {
    "gni": "gross national income",
    "co2": "carbon dioxide emissions",
    "u5mr": "under five mortality rate",
    "imr": "infant mortality rate",
}
_ACRONYM_RE = re.compile(r"\b(" + "|".join(_ACRONYM_EXPANSIONS) + r")\b", re.IGNORECASE)


def _canonicalize_acronyms(text: str) -> str:
    """Expand known acronyms at word boundaries, preserving surrounding text.

    Case-insensitive and controlled by `settings().acronym_canonicalize`.
    Applied only to recall's variable query, not entities or displayed fields.
    """
    if not settings().acronym_canonicalize:
        return text
    return _ACRONYM_RE.sub(lambda m: _ACRONYM_EXPANSIONS[m.group(0).lower()], text)


class Recall(BaseModel):
    """Graph-backed recall: candidate SVs (dcid + score + type_of + sentence) and
    entity resolution.

    candidates is the single source of truth; candidate_svs and candidate_sv_scores
    are thin derived views kept so existing call sites (region/__init__.py, shape.py) that
    only need the dcid/score parallel lists stay unchanged.

    resolutions carries every entity name's own entity_resolution.Resolution
    (Resolved, Group, Ambiguous, NotHeld, or NotFound) -- a hit or a miss,
    always present for every name in the entities recall() was called with,
    plus any pre_resolved name still carried over. resolved_entity_names
    narrows that same map to the Resolved AND Group names, surface name ->
    dcid, for every existing call site that only ever wanted a successfully
    resolved dcid. A Group's own dcid here is provisional -- its own entity
    reading when one exists, else its first member (entity_resolution's own
    "at least one of entity/members is always populated" contract for
    Group) -- and exists only to keep a Group name out of entity_not_resolved
    before region/phase_intake.py's _phase_resolve_places, the one place that
    decides a Group's entity vs. member reading for the role in hand, ever
    runs; that phase always revisits this entry, either confirming it,
    replacing it with the Group's own entity, or (a too-large Group)
    removing it again.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidates: list[DetectedSV]  # descending by score (detect_svs's contract)
    resolutions: dict[str, Resolution]  # surface name → Resolution (hit or miss)
    resolved_entity_names: dict[str, str]  # surface name → dcid, Resolved and Group names
    # The UN System Data Commons' admitted candidates, None when its resolve failed. The
    # resolve may still be running: only the gap-fill pass waits for it (`settle_un`,
    # region/pipeline.py), and `un_candidates` and `un_unavailable` read it once settled.
    un_leg: Future[tuple[DetectedSV, ...] | None] = Field(default_factory=lambda: settled_un(()))

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


async def recall(
    variable: str,
    entities: list[str],
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    raw_query: str | None = None,
    pre_resolved: dict[str, Resolution] | None = None,
) -> Recall:
    """Recall candidate StatVars and resolve entity names concurrently.

    `raw_query` overrides `variable` for graph detection; callers use it to
    add coded-qualifier retrieval hints. Those hints do not constrain the
    population. `pre_resolved` supplies entity results shared across variables;
    only missing names go to one resolver call alongside `detect_svs`.

    Candidates stay in descending score order. The recall floor drops low
    scores except Topic candidates. Results remain unconfirmed until the
    materialise stage reads the graph.
    """
    detect_query = raw_query or variable
    _pre = pre_resolved or {}
    cache_hits = {name: _pre[name] for name in entities if name in _pre}
    misses = [name for name in entities if name not in _pre]

    detection, miss_resolutions = await asyncio.gather(
        asyncio.to_thread(graph.detect_svs, _canonicalize_acronyms(detect_query)),
        resolver.resolve(misses),
    )
    candidates = _above_recall_floor(detection.candidates)
    resolutions: dict[str, Resolution] = dict(cache_hits)
    resolutions.update(zip(misses, miss_resolutions, strict=True))
    resolved_entity_names: dict[str, str] = {}
    for name, resolution in resolutions.items():
        if isinstance(resolution, Resolved):
            resolved_entity_names[name] = resolution.entity.dcid
        elif isinstance(resolution, Group):
            # Provisional only -- see this class's own docstring. Prefers
            # the entity reading when one exists, since members can
            # legitimately be empty exactly when entity is not None
            # (entity_resolution.Group's own "at least one of entity/
            # members is always populated" contract); members is
            # otherwise guaranteed non-empty by entity_resolution's own
            # _assemble_group (its NotHeld contract for an empty member
            # set with no entity reading), but that guarantee is
            # _assemble_group's own, not Resolution's -- a hand-built
            # pre_resolved Group can still carry both entity=None and
            # empty members. Guarded the same way region/phase_intake.py's
            # _phase_resolve_places guards the identical case one phase
            # later: an empty member set with no entity reading leaves
            # `name` out of resolved_entity_names, the ordinary
            # still-unresolved path, rather than raising.
            if resolution.entity is not None:
                resolved_entity_names[name] = resolution.entity.dcid
            elif resolution.members:
                resolved_entity_names[name] = resolution.members[0].dcid

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
        resolved_entity_names=resolved_entity_names,
        un_leg=detection.un,
    )
