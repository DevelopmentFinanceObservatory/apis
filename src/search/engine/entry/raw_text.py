"""Entry point for ResolveInput.kind == "raw_text": the free-text query
door. Reads what the query asks to leave out beside extract(), which pulls
variables/entities/dates out of the query before fanning out, or answers a
catalogue or declined question straight after it.

Every reader of the query's words past that read takes the blanked text
(`exclusion_read.QueryReading.text`). Only the echo and the layer 1 key take the
query as written.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import re
from collections.abc import AsyncGenerator, Collection, Sequence
from collections.abc import Set as AbstractSet
from contextlib import aclosing

from code_index.shared import get_index
from code_index.tokens import word_tokens
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.build import build_commit
from dfo_platform.trace import get_trace_id
from entity_resolution import Admission, EntityResolver, Resolved
from entity_resolution.pack import organisations_named
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import Catalogue, current_catalogue
from search.engine.assemble import (
    _surviving_qualifier_warnings,
    build_no_data_part,
    skills_index_incomplete_warning,
    skills_index_unavailable_warning,
    unapplied_qualifier_warnings,
)
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.conjoin import VARIABLES_CLAMPED, assemble_region
from search.engine.deadline import _await_bounded
from search.engine.departures import (
    WarningKey,
    exclusion_not_applied,
    measure_not_applied,
    names_request,
    price_basis_not_applied,
    warning_key,
)
from search.engine.enrichment import _stream_slot_options
from search.engine.entry.multi import _stream_combined_regions
from search.engine.entry.params import EntryParams
from search.engine.envelope import (
    _sum_usage,
    _terminal_events,
    make_diagnostics,
    make_query_echo,
    make_response,
    now_ms,
)
from search.engine.events import (
    DoneEvent,
    DoneTelemetry,
    Event,
    InterpretationEvent,
    LateDepartures,
    StepEvent,
)
from search.engine.exclusion_read import (
    LeftOut,
    QueryReading,
    admit_exclusions,
    read_exclusions,
    stored_exclusion_read,
)
from search.engine.extract import (
    DateRequest,
    ExtractedEntity,
    Extraction,
    ExtractionResult,
    _dedupe_stripped,
    _ground_roles,
    _grounding_form,
    cue_grounded_in_form,
    cue_is_its_span,
    dates_to_request,
    extract,
    scoped_measures,
    veto_words,
)
from search.engine.fanout import (
    LegReading,
    PartyReading,
    _FanOutResult,
    _resolve_entities_bounded,
    _run_variables_streaming,
    leg_runs,
)
from search.engine.function_words import FUNCTION_WORDS
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.qualifier_match import (
    CodedQualifier,
    _all_normalized_text_ranges,
    _normalize,
    _ranges_overlap,
    recall_text,
)
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.composite_answer import ComponentReads
from search.engine.region.phase_rank_bind import holds_words, names_value
from search.engine.region.place_options import PlaceOptionsPool, replayed_place_options
from search.engine.region.result import (
    RegionResult,
    any_degraded,
    with_departure_facts,
    with_entity_words,
)
from search.engine.region.sentence_events import answer_events
from search.engine.region.source_choice import ContestedOrganisation
from search.engine.region.state import (
    PartExclusions,
    exclusion_requirements,
    qualifier_dropped_warning,
)
from search.engine.response_cache_flow import CacheScope
from search.engine.role_text import (
    DoorRole,
    breakdown_axis,
    every_member_breakdown_axis,
    role_axis_for_text,
)
from search.engine.source_code import PartCode, compile_part_code, scan_source_codes
from search.engine.source_scan import (
    SourceHit,
    SourceScan,
    code_spans,
    direction_word,
    find_organisations,
    left_out_sources,
    scan_sources,
    source_phrases,
)
from search.engine.source_scope import (
    SourceDictionary,
    SourceRefusal,
    SourceScope,
    compile_sources,
    normalise_source_text,
    source_dictionary,
)
from search.engine.source_spans import read_sources, unclaimed_stretches, unclaimed_words
from search.engine.stated_scope import part_scope_intakes, without_scope_phrases
from search.engine.surface_resolution import corrected_surfaces, is_surface, named_dcids
from search.engine.unit.bind import names_price_basis, names_unit_request
from search.heading.departure_fact import DepartureFact, DirectionWord
from search.models import (
    CatalogueAnswer,
    DeclinedAnswer,
    GraphRef,
    Operation,
    RawTextInput,
    RoleAxis,
    Scope,
    SourceMention,
    Spec,
    Warning,
)
from search.render.assumptions import NameCorrection
from search.scope.state import ScopeIntake

logger = logging.getLogger(__name__)


# What a cue that reads the direction off the measure holds once the place's own words and
# "the" are removed: nothing for a national adjective ("French aid"), a possessive marker
# for "Ghana's economy" and "of" for "debtors of China".
_MEASURE_READING_CUES: frozenset[str] = frozenset({"", "'s", "'", "of"})

# Words that narrow a measure to what one kind of provider gives or to one kind of terms.
# Each one a part's variable holds is judged on that part alone (`_narrowing_words`).
_NARROWING_WORDS: tuple[str, ...] = ("bilateral", "multilateral", "concessional")


def _entity_key(entity: ExtractedEntity) -> str:
    """Return the canonical entity identity used by bind, roles, and response data."""
    return entity.text.strip()


def _grounded(span: str, query: str) -> bool:
    """Whether `span` occurs in `query`, both casefolded."""
    return bool(_all_normalized_text_ranges(span, _normalize(query)))


def _balanced_parenthetical_extensions(query: str, span: str) -> list[str]:
    """Return source surfaces made by one immediately adjacent balanced parenthetical."""
    if not _grounded(span, query):
        return []

    extensions: list[str] = []
    for match in re.finditer(re.escape(span), query, flags=re.IGNORECASE):
        cursor = match.end()
        while cursor < len(query) and query[cursor].isspace():
            cursor += 1
        if cursor == len(query) or query[cursor] != "(":
            continue
        depth = 0
        end = cursor
        while end < len(query):
            if query[end] == "(":
                depth += 1
            elif query[end] == ")":
                depth -= 1
                if depth == 0:
                    extensions.append(query[match.start() : end + 1])
                    break
            end += 1
    return extensions


def _whole_word_ranges(words: str, text: str) -> list[tuple[int, int]]:
    """The character ranges of `text` where `words` occur, casefolded, as whole words
    (`_all_normalized_text_ranges`), so "DAC2a" holds no "DAC"."""
    return [
        (start, end)
        for start, end in _all_normalized_text_ranges(words, text)
        if not (start > 0 and text[start - 1].isalnum())
        and not (end < len(text) and text[end].isalnum())
    ]


def _folded_words(text: str) -> list[str]:
    return [token.text.casefold() for token in word_tokens(text)]


def _holds_run(words: Sequence[str], run: Sequence[str]) -> bool:
    """Whether `words` hold `run` as consecutive whole words."""
    return any(
        tuple(words[i : i + len(run)]) == tuple(run) for i in range(len(words) - len(run) + 1)
    )


async def _query_words_of(
    name: str,
    query: str,
    *,
    owned: Sequence[tuple[int, int]],
    rivals: Collection[str],
    resolver: EntityResolver,
    deadline: float,
) -> str | None:
    """The query's own words for a `name` the query does not write, as written in `query`.

    They are the longest run of whole query words that `name` holds as consecutive whole
    words, holding a word other than a function word: "Congo" in "aid to Congo" for the name
    "Democratic Republic of the Congo". The run grows to the widest stretch of query words
    around it that resolves to the one entity `name` resolves to, so the user's mention
    keeps the words that choose its place: "DR Congo" and "South Korea" rather than "Congo"
    and "Korea". A stretch opens and closes on a word other than a function word, beyond the
    run's own, and the run and every stretch resolve with `name` in one resolver call. A
    stretch naming another place ("South Korea" for "Democratic People's Republic of
    Korea") is no such stretch, and two of one width leave the run as it is. Neither takes a
    word inside `owned`, the character ranges of `query` another entity's mention holds.
    `None` when no run holds one, when two runs of different words tie for longest, or
    when a run left as it is is contested: one of the `rivals`, the other entities' names,
    holds it too, so "Congo" in "ODA from DRC to Congo" chooses neither Congo and each
    resolves by its name, or it names entities none of which is the one `name` resolves to,
    so "Korea" for "Democratic People's Republic of Korea" resolves by the name. A name in
    other words ("Germany" for "German", "United Kingdom" for "UK") shares no word with the
    query and has none."""
    tokens = word_tokens(query)
    free = [
        not any(_ranges_overlap((token.start, token.end), span) for span in owned)
        for token in tokens
    ]
    folded = [token.text.casefold() for token in tokens]
    name_words = _folded_words(name)
    runs: dict[tuple[str, ...], tuple[int, int]] = {}
    for start in range(len(folded)):
        for end in range(start + 1, len(folded) + 1):
            run = tuple(folded[start:end])
            if not free[end - 1] or not _holds_run(name_words, run):
                break
            if any(word not in FUNCTION_WORDS for word in run):
                runs.setdefault(run, (start, end))
    if not runs:
        return None
    longest = max(len(run) for run in runs)
    ((run, (start, end)), *others) = [item for item in runs.items() if len(item[0]) == longest]
    if others:
        return None

    def text(lo: int, hi: int) -> str:
        return query[tokens[lo].start : tokens[hi - 1].end]

    lo, hi = start, end
    while lo > 0 and free[lo - 1]:
        lo -= 1
    while hi < len(tokens) and free[hi]:
        hi += 1
    stretches = [
        (i, j)
        for i in range(lo, start + 1)
        for j in range(end, hi + 1)
        if (i, j) != (start, end)
        and (i == start or folded[i] not in FUNCTION_WORDS)
        and (j == end or folded[j - 1] not in FUNCTION_WORDS)
    ]
    texts = list(dict.fromkeys([name, text(start, end), *(text(i, j) for i, j in stretches)]))
    answers = dict(
        zip(texts, await _await_bounded(resolver.resolve(texts), deadline=deadline), strict=True)
    )
    named = answers[name]
    if isinstance(named, Resolved):
        dcid = named.entity.dcid
        choosing = [(i, j) for i, j in stretches if named_dcids(answers[text(i, j)]) == {dcid}]
        width = max((j - i for i, j in choosing), default=0)
        widest = [(i, j) for i, j in choosing if j - i == width]
        if len(widest) == 1:
            return text(*widest[0])
    if any(_holds_run(_folded_words(rival), run) for rival in rivals):
        return None
    run_names = named_dcids(answers[text(start, end)])
    if run_names and not run_names & named_dcids(named):
        return None
    return text(start, end)


def _mention_ranges(others: Sequence[ExtractedEntity], *, query: str) -> list[tuple[int, int]]:
    """The character ranges of `query` the `others` name in whole words: each one's grounded
    span, else its name where the query writes it."""
    ranges: list[tuple[int, int]] = []
    for other in others:
        span = (other.span or "").strip()
        words = span if span and _grounded(span, query) else other.text.strip()
        ranges.extend(_whole_word_ranges(words, query))
    return ranges


async def _with_query_span(
    entity: ExtractedEntity,
    *,
    others: Sequence[ExtractedEntity],
    query: str,
    resolver: EntityResolver,
    deadline: float,
) -> ExtractedEntity:
    """`entity` whose span is the query's own words for it (`_query_words_of`), where the
    query writes neither its name nor its span: entity resolution then reads those words,
    and an extraction that rewrote "Congo" as "Democratic Republic of the Congo" leaves the
    query's ambiguity in place (`surface_resolution`). Words one of the `others` names are
    that entity's, so "Republic of the Congo" in "ODA from DRC to Republic of the Congo" is
    no span of the Democratic Republic of the Congo, and words another's name holds as well
    are contested (`_query_words_of`). With no words of its own, the entity resolves by its
    name."""
    span = (entity.span or "").strip()
    if (span and _grounded(span, query)) or _grounded(entity.text.strip(), query):
        return entity
    own = await _query_words_of(
        entity.text,
        query,
        owned=_mention_ranges(others, query=query),
        rivals={
            other.text.strip()
            for other in others
            if _normalize(other.text) != _normalize(entity.text)
        },
        resolver=resolver,
        deadline=deadline,
    )
    return entity if own is None else entity.model_copy(update={"span": own})


async def _recover_entity_surfaces(
    entities: list[ExtractedEntity], *, query: str, resolver: EntityResolver, deadline: float
) -> list[ExtractedEntity]:
    """Recover each entity's surface before scans or cache keys read it: the query's own
    words for a name the query writes neither as the name nor as its span
    (`_with_query_span`), and a source-qualified surface. Only a span that occurs in
    `query` (`_grounded`) is read for the qualifier.
    """
    # Each entity reads the spans the ones before it recovered, so two never take one word.
    spanned = list(entities)
    for i, entity in enumerate(entities):
        spanned[i] = await _with_query_span(
            entity,
            others=[*spanned[:i], *spanned[i + 1 :]],
            query=query,
            resolver=resolver,
            deadline=deadline,
        )
    recovered: list[ExtractedEntity] = []
    for entity in spanned:
        span = (entity.span or "").strip()
        if not span or not _grounded(span, query):
            recovered.append(entity)
            continue
        if "(" in span:
            # The extraction can carry the complete source span while `text` remains its
            # bare English head.
            surface = span
        else:
            surface = next(iter(_balanced_parenthetical_extensions(query, span)), None)
            if surface is None or any(
                other is not entity and other.span and _normalize(other.span) == _normalize(surface)
                for other in entities
            ):
                # A query can name bare Asia and Asia (DAC). The existing full span owns the
                # qualified occurrence; a truncated duplicate has no source position to claim it.
                recovered.append(entity)
                continue
        # The English name with the source's qualifier, for a translated span: "Asia (DAC)"
        # for "Asie (DAC)".
        qualified_text = f"{entity.text} {surface[surface.index('(') :]}"
        if resolver.is_exact_registered_name(surface):
            # The registry, rather than a bare parse surface, confirms a distinct entity.
            recovered.append(entity.model_copy(update={"text": surface, "span": surface}))
        elif "(" not in entity.text and resolver.is_exact_registered_name(qualified_text):
            recovered.append(entity.model_copy(update={"text": qualified_text, "span": surface}))
        elif _normalize(entity.text) == _normalize(surface):
            recovered.append(entity.model_copy(update={"span": surface}))
        else:
            recovered.append(entity)
    return _without_absorbed(recovered, query=query)


async def _name_corrections(
    entities: list[ExtractedEntity], *, query: str, resolver: EntityResolver, deadline: float
) -> tuple[NameCorrection, ...]:
    """The names `entities` read in place of the query's own words for them, in first-seen
    order: each name whose surface (`_entity_surfaces`) lists no name or alias of the entity
    the name settles on (`surface_resolution.corrected_surfaces`), as "Germany" for
    "Germny". A translated name the query writes no words of has no surface, and so no
    correction, as "Germany" for "Deutschland" with a null span. A national adjective,
    whose cue is its own span ("French" in "French aid"), names its country by its
    adjective and corrects nothing. Run on the recovered entities
    (`_recover_entity_surfaces`), whose surfaces that call resolved already."""
    adjectives = {_entity_key(entity) for entity in entities if cue_is_its_span(entity)}
    surfaces = {
        name: surface
        for name, surface in _entity_surfaces(entities, query=query).items()
        if name not in adjectives
    }
    corrected = await _await_bounded(
        corrected_surfaces(surfaces, resolver=resolver), deadline=deadline
    )
    return tuple(NameCorrection(asked=surfaces[name], ref=ref) for name, ref in corrected.items())


def _without_absorbed(entities: list[ExtractedEntity], *, query: str) -> list[ExtractedEntity]:
    """`entities` without one the query names only inside another's parenthetical span.

    "DAC" in "ODA to Africa (DAC)" is part of the name "Africa (DAC)", never a party of its
    own; "the DAC" named elsewhere in the query stays. A mention is whole words, so "DAC2a"
    names no DAC.
    """
    normalized = _normalize(query)
    qualified = [
        qualified_range
        for entity in entities
        if entity.span and "(" in entity.span
        for qualified_range in _whole_word_ranges(entity.span, normalized)
    ]

    def absorbed(entity: ExtractedEntity) -> bool:
        if not entity.span or "(" in entity.span:
            return False
        ranges = _whole_word_ranges(entity.span, normalized)
        return bool(ranges) and all(
            any(lo <= start and end <= hi for lo, hi in qualified) for start, end in ranges
        )

    return [entity for entity in entities if not absorbed(entity)]


def _entity_surfaces(entities: list[ExtractedEntity], *, query: str) -> dict[str, str]:
    """Map each entity name to the query words its first mention cites, where they differ.

    A span counts only when it occurs in `query` (`_grounded`); recall resolves the name
    through it (`surface_resolution.resolve_names`). A name with an ungrounded span, or
    none, resolves as itself.
    """
    surfaces: dict[str, str] = {}
    seen: set[str] = set()
    for entity in entities:
        key = _entity_key(entity)
        if key in seen:
            continue
        seen.add(key)
        span = (entity.span or "").strip()
        if span and is_surface(span, key) and _grounded(span, query):
            surfaces[key] = span
    return surfaces


def _states_direction(entity: ExtractedEntity, *, folded_query: str) -> bool:
    """Whether `entity`'s cue (`ExtractedEntity.cue`) states its direction, which it does
    when the query holds it (`extract.cue_grounded`) and it reaches beyond the place's own
    words (its span and its name, each as whole words), the article "the" and a possessive
    marker or "of" (`_MEASURE_READING_CUES`). A cue the query does not hold is one the
    model misquoted, and its role is read off the measure. `folded_query` is the query
    folded once by `extract._grounding_form`, which a request that reads many entities
    shares.

    "to", "funded by" and the case ending in "Norjasta" with the span "Norja" state it. A
    national adjective or an inflected name, whose cue is its own span ("French aid",
    "Norjasta" as span and cue), a possessive ("Ghana's economy", cue "Ghana's" or "'s";
    "the US's aid", cue "the US's") and "of" ("debtors of China", "of the Gambia") read
    the direction off the measure, so none states it.
    """
    if not cue_grounded_in_form(entity, folded_query):
        return False
    rest = (entity.cue or "").replace("\u2019", "'").casefold()
    for words in (entity.span, entity.text):
        if words and words.strip():
            place = re.escape(words.strip().casefold())
            rest = re.sub(rf"(?<!\w){place}(?!\w)", " ", rest)
    residue = " ".join(token for token in rest.split() if token != "the")
    return residue not in _MEASURE_READING_CUES


def _part_positions(
    entities: Sequence[ExtractedEntity], *, measures: Sequence[str], variables: Sequence[str]
) -> list[list[int]]:
    """The positions in `entities` of each part's entities, parallel to `measures`. A part
    holds the entities whose linked measures (`extract.scoped_measures`) include its own.
    `variables` are the extraction's own, whose positions the entities' links name, and
    `measures` the parts' stripped distinct measures (`_ExtractionPrep.unique`). An entity
    that names no measure scopes every part.
    """
    scopes = [scoped_measures(entity, variables) for entity in entities]
    return [[i for i, scope in enumerate(scopes) if measure in scope] for measure in measures]


def _scoping_parts(
    entities: Sequence[ExtractedEntity], *, variables: Sequence[str], parts: Sequence[str]
) -> list[ExtractedEntity]:
    """`entities` with each one's measure positions moved from the extraction's `variables`
    onto `parts`, the stripped distinct measures (`_ExtractionPrep.unique`), which the
    layer-2 key holds. `variables` can repeat a measure, so the same position there can
    name a different part in two readings. A position in `parts` names one part."""
    return [
        entity.model_copy(
            update={
                "variables": sorted({parts.index(variables[i].strip()) for i in entity.variables})
            }
        )
        for entity in entities
    ]


def _entity_roles(
    part: Sequence[int],
    *,
    entities: Sequence[ExtractedEntity],
    stated: Sequence[bool],
    preceding: Sequence[tuple[str, ...]],
) -> dict[str, list[DoorRole]]:
    """Map the stripped names of one part's entities to their roles, as the parsed door
    maps a part's (`parsed._entity_roles_for_part`). `part` holds positions in `entities`
    (`_part_positions`), and `stated` and `preceding` hold, for each of `entities`, whether
    its cue states its direction (`_states_direction`) and the words its role's veto reads
    (`extract.veto_words`). The request reads both once for all its parts.

    An entity's party word (`ExtractedEntity.role`) gives its axis through the parsed
    door's lexicon (`role_text.role_axis_for_text`), so a creditor is the origin and a
    debtor the destination. `extract` keeps a role only when the model cites a cue for it
    (`extract._ground_roles`), so every role here has one, and `stated` records whether
    `query` holds the cue and it states the direction, or the role is read off the
    measure. `preceding` holds the words before the occurrences of the place its cue ties
    it to, read beside the other roles of `entities`. An entity with no role is undirected.
    Preserve distinct mentions in order, including self-flows on opposite axes, and
    collapse duplicate mentions with the same axis, span and reading, which then read the
    occurrences each is tied to. The span is the query's own words for the place, else its
    name.
    """
    roles: dict[str, list[DoorRole]] = {}
    for position in part:
        entity = entities[position]
        axis = role_axis_for_text(entity.role)
        if axis is None:
            continue
        occurrences = roles.setdefault(_entity_key(entity), [])
        role = DoorRole(
            axis=axis,
            span=entity.span or entity.text,
            stated=stated[position],
            members=entity.members,
            party=entity.role,
            preceding=preceding[position],
        )
        if role not in occurrences:
            occurrences.append(role)
            continue
        i = occurrences.index(role)
        words = (*(occurrences[i].preceding or ()), *(role.preceding or ()))
        occurrences[i] = dataclasses.replace(role, preceding=tuple(dict.fromkeys(words)))
    return roles


async def _empty_query_response(
    query: str, *, start_ms: int, params: EntryParams
) -> AsyncGenerator[Event]:
    """Return `variable_not_resolved` directly for a blank query."""
    echo = make_query_echo(query, extract_skipped=True)
    skills_warnings = [
        w
        for w in (
            skills_index_unavailable_warning(
                include_skills=params.include_skills, skills_index=params.skills_index
            ),
            skills_index_incomplete_warning(
                include_skills=params.include_skills, skills_index=params.skills_index
            ),
        )
        if w is not None
    ]
    diag = make_diagnostics(build_commit(), skills_warnings, {}, now_ms() - start_ms)
    part = build_no_data_part(
        "variable_not_resolved",
        variable_text=None,
        warnings=[],
        include_skills=params.include_skills,
        skills_index=params.skills_index,
    )
    response = make_response(
        echo,
        diag,
        [part],
        include_skills=params.include_skills,
        skills_index=params.skills_index,
    )
    for ev in _terminal_events(response, start_ms=start_ms, degraded_leg=False):
        yield ev


async def _run_extract_phase(
    query: str, *, llm: SupportsLLM, deadline: float
) -> tuple[ExtractionResult, dict | None, int]:
    """Run extraction under the request deadline and return its timing."""
    t0 = now_ms()
    extract_result, extract_usage = await _await_bounded(extract(query, llm=llm), deadline=deadline)
    timing_extract = now_ms() - t0
    return extract_result, extract_usage, timing_extract


@dataclasses.dataclass(frozen=True)
class _DoorRead:
    """What the door reads of a query before the parts: its extraction with the usage of both
    calls, what the query leaves out (`reading`), and the time each call took, by step."""

    extract_result: ExtractionResult
    usage: dict | None
    reading: QueryReading
    timing: dict[str, int]


# The `llm_usage` key counting the model calls a request sent whose tokens it cannot count.
UNCOUNTED_CALLS = "uncounted_calls"


async def _discarded_usage(
    task: asyncio.Task[tuple[ExtractionResult, dict | None, int]],
) -> dict | None:
    """The usage of a speculative parse the door discards: its token counts when it finished
    with an answer, else one uncounted call (`UNCOUNTED_CALLS`), since a parse cancelled in
    flight or failed had sent its request and reports no counts. One still running is
    cancelled."""
    if not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if task.cancelled() or task.exception() is not None:
        return {UNCOUNTED_CALLS: 1}
    return task.result()[1]


async def _read_and_extract(query: str, *, llm: SupportsLLM, deadline: float) -> _DoorRead:
    """Read what `query` asks to leave out and extract the query without it.

    A read the extraction store holds decides first, so a repeated query waits on neither
    call. Otherwise the read and a speculative extraction of `query` start together and the
    read is awaited first: with nothing admitted the parse stands, and an admitted stretch
    cancels it and extraction takes the blanked text, so a hit costs the read plus one parse.

    A hit cancels the speculative parse when the read is admitted, before the parse of the
    blanked text starts, so no second extraction runs to its end. Its tokens count in the
    usage when it had already finished, else it counts as one uncounted call
    (`_discarded_usage`): its request was sent, and no response reports its tokens. A read
    whose every stretch is kept unblanked (`LeftOut.at` `None`) leaves the text as it was,
    so the speculative parse stands.

    A failed read raises as a failed extraction does and fails the query, and the speculative
    parse goes with it: a query the read could not check for what it leaves out gets no
    answer that may state the opposite of what it asked.
    """
    if (stored := stored_exclusion_read(query, llm=llm)) is not None:
        reading = admit_exclusions(query, stored)
        result, usage, timing_extract = await _run_extract_phase(
            reading.text, llm=llm, deadline=deadline
        )
        timing = {"extract": timing_extract, "exclusion_read": 0}
        return _DoorRead(result, usage, reading, timing)

    t0 = now_ms()
    speculative = asyncio.create_task(_run_extract_phase(query, llm=llm, deadline=deadline))
    try:
        read, read_usage = await _await_bounded(read_exclusions(query, llm=llm), deadline=deadline)
        timing_read = now_ms() - t0
        reading = admit_exclusions(query, read)
        if reading.text == query:
            result, usage, timing_extract = await speculative
        else:
            discarded = await _discarded_usage(speculative)
            result, usage, timing_extract = await _run_extract_phase(
                reading.text, llm=llm, deadline=deadline
            )
            usage = _sum_usage(usage, discarded)
    finally:
        speculative.cancel()
        await asyncio.gather(speculative, return_exceptions=True)
    timing = {"extract": timing_extract, "exclusion_read": timing_read}
    return _DoorRead(result, _sum_usage(read_usage, usage), reading, timing)


async def _no_variables_response(
    query: str,
    *,
    start_ms: int,
    params: EntryParams,
    extract_usage: dict | None,
    timing: dict[str, int],
    warnings: Sequence[Warning] = (),
) -> AsyncGenerator[Event]:
    """Terminal response when extract() ran but found no variables at all:
    reports variable_not_resolved with the query's own stripped text, and `warnings`.
    """
    echo = make_query_echo(query, extract_skipped=False)
    skills_warnings = [
        w
        for w in (
            skills_index_unavailable_warning(
                include_skills=params.include_skills, skills_index=params.skills_index
            ),
            skills_index_incomplete_warning(
                include_skills=params.include_skills, skills_index=params.skills_index
            ),
        )
        if w is not None
    ]
    diag = make_diagnostics(
        build_commit(),
        skills_warnings,
        timing,
        now_ms() - start_ms,
        llm_usage=extract_usage,
    )
    part = build_no_data_part(
        "variable_not_resolved",
        variable_text=query.strip(),
        warnings=list(warnings),
        include_skills=params.include_skills,
        skills_index=params.skills_index,
    )
    response = make_response(
        echo,
        diag,
        [part],
        include_skills=params.include_skills,
        skills_index=params.skills_index,
    )
    for ev in _terminal_events(
        response, start_ms=start_ms, degraded_leg=False, llm_usage=extract_usage
    ):
        yield ev


async def _confirmed_country(
    entities: list[ExtractedEntity], *, resolver: EntityResolver, deadline: float
) -> GraphRef | None:
    """Return the one country the named places resolve to, or `None`.

    Every distinct entity name goes to the resolver in one call. The answer is
    the country when exactly one country is confirmed among them; no country, or
    two different ones, leaves the place unset. Only a country qualifies; a city,
    region or organisation is skipped, even when it lies in a country. `extract`
    clears the entities of a question naming more places than a resolution may, so
    the call names at most `max_entities`.
    """
    names = list(dict.fromkeys(name for e in entities if (name := e.text.strip())))
    if not names:
        return None
    resolutions = await _await_bounded(resolver.resolve(names), deadline=deadline)
    countries = {
        r.entity.dcid: r.entity
        for r in resolutions
        if isinstance(r, Resolved) and r.entity.kind == "country"
    }
    if len(countries) != 1:
        return None
    (country,) = countries.values()
    return GraphRef(ref=country.dcid, label=country.name)


async def _disposition_response(
    query: str,
    extraction: Extraction,
    *,
    resolver: EntityResolver,
    start_ms: int,
    params: EntryParams,
    extract_usage: dict | None,
    timing: dict[str, int],
    scope: CacheScope | None,
    warnings: Sequence[Warning] = (),
) -> AsyncGenerator[Event]:
    """Answer a catalogue, out-of-scope or requires-analysis question straight
    after extract, with no recall, bind or second cache lookup, carrying `warnings`.

    `scope.place` is best effort: a resolver failure or timeout leaves it unset,
    still answers, and keeps the answer out of the response cache.
    """
    timed_out = False
    degraded = False
    check_t0 = now_ms()
    try:
        place = await _confirmed_country(
            extraction.entities, resolver=resolver, deadline=params.deadline
        )
    except TimeoutError:
        place, timed_out = None, True
    except Exception:
        # The place only fills the website's examples; the answer stands without it.
        logger.warning("scope.place: resolver failed; answering without a place", exc_info=True)
        place, degraded = None, True
    timing_entity_check = now_ms() - check_t0
    if scope is not None:
        scope.note_health(timed_out=timed_out, degraded_leg=degraded)

    echo = make_query_echo(query, extract_skipped=False)
    diag = make_diagnostics(
        build_commit(),
        list(warnings),
        {**timing, "entity_check": timing_entity_check},
        now_ms() - start_ms,
        llm_usage=extract_usage,
    )
    answer_scope = Scope(place=place)
    match extraction.kind:
        case "catalogue":
            response = CatalogueAnswer(query_echo=echo, diagnostics=diag, scope=answer_scope)
        case "out_of_scope" | "requires_analysis" as reason:
            response = DeclinedAnswer(
                query_echo=echo, diagnostics=diag, reason=reason, scope=answer_scope
            )
        case _:
            raise AssertionError(f"not a disposition kind: {extraction.kind!r}")
    for ev in _terminal_events(
        response,
        start_ms=start_ms,
        timed_out=timed_out,
        degraded_leg=degraded,
        llm_usage=extract_usage,
    ):
        yield ev


@dataclasses.dataclass(frozen=True)
class _ExtractionPrep:
    """One query's prepared extraction, shared by events, cache lookup, and fan-out.

    `unique` and `entities` preserve first-seen order. `entities` leaves out an entity
    every occurrence of whose words a source mention covers
    (`source_scan.SourceScan.source_entities`), as do `entity_words` and the per-part
    entity fields. `per_variable_entities` runs parallel to `unique` and holds the names of
    the entities whose measures (`ExtractedEntity.variables`) include that part's
    (`_part_positions`). `per_variable_entity_roles` holds each part's roles, read from its
    own entities (`_entity_roles`), and `per_variable_member_entities` the names
    each part asks for as their members (`ResolveState.member_entities`).
    `interpretation_variables` adds coded-qualifier sidecars, while resolution and cache
    keys use bare text.
    `per_variable_operations` runs parallel to `unique`: each variable's operations
    ride on its part, and their phrases reach only that variable's bind, beside
    `dropped_qualifier_phrases`. They raise no warning.
    `per_variable_open_axes` runs parallel to `unique` too: the flow roles each part asks
    one value per member of (`role_text.breakdown_axis`), read off its own operations'
    phrases (`every_member_breakdown_axis` for a verb other than `rank`), the request's
    dropped phrases and the entities scoping its measure (`extract.scoped_measures`) that
    are such a breakdown, which leave those lists.
    `per_variable_date_requests` runs parallel to `unique` too: `date_request`, or a
    forecasting variable's own window without its target years.
    `unit_text` is the dropped phrases that bind as one unit request
    (`_split_unit_phrases`); they stay out of `dropped_qualifier_phrases`.
    `text` is the query with each stretch it asks to leave out blanked, which every reader of
    its words takes. `unclaimed` are the stretches of `text` that hold a word no part,
    entity, date, unit, operation or source mention claims
    (`source_spans.unclaimed_stretches`). In "Germany attributed climate finance
    commitments 2022", where extraction reads the variable "climate finance commitments",
    "attributed" is one. A blanked word is in none. `exclusions` are the blanked stretches,
    less each that leaves out sources alone, which its mentions apply
    (`SourceScan.source_stretches`), `exclusion_warnings` one
    `query_qualifier_dropped` warning for each and `exclusion_targets` the positions in
    `unique` of the measures each qualifies (`SourceScan.stretch_targets`), both parallel to
    it. `per_variable_exclusions` runs parallel to `unique`: the stretches qualifying each
    part and the requirements judging its answers on them (`state.exclusion_requirements`),
    read once for the request. A warning stands on a part whose answer does not leave its
    phrase out, and on the envelope while one such part stands (`surviving`). Its departure
    stands on that part alone (`onto`).
    `per_variable_unit_texts` runs parallel to `unique`: each part's unit request, as the
    parsed door's `ParsedPart.unit_text` is (`_part_unit_text`).
    `per_variable_measure_qualifiers` runs parallel to `unique` too: the words that narrow
    each variable's measure (`_narrowing_words`), less any the request's dropped phrases
    still hold. The part warns for each its answer leaves unapplied (`onto`). They stay out
    of its bind, which reads them in the variable already. `per_variable_dropped_phrases`
    runs parallel to `unique` too: each dropped phrase that is such a word and that one
    variable alone holds, which leaves the request's phrases for that variable's bind alone
    (`dropped_phrases_of`).
    `qualifier_facts` holds the departure of each `qualifier_warnings` entry whose phrase
    asks for something (`_dropped_phrase_fact`), under its warning's key. `entity_words`
    maps each entity name to the query's own words for it (`ExtractedEntity.span`), first
    mention first. `entity_surfaces` holds those words where they occur in the query and
    differ from the name, which recall resolves the name through (`_entity_surfaces`).
    `separate` runs parallel to `unique` and
    marks each variable the raw query compares with another (`source_spans.read_sources`),
    as `ParsedPart.separate` does. `read_connectives` are the offsets of the comparison
    words that marked one. `per_variable_sources` runs parallel to `unique` too: the
    source mentions the query writes for each variable (`source_scan.scan_sources`), and
    `per_variable_source_scopes` what each compiles to, `None` for a variable with none.
    `per_variable_source_codes` runs parallel to `unique` too: the series code the query
    writes for each variable (`source_code.scan_source_codes`), and
    `per_variable_compiled_codes` what each compiles to within its variable's scope, `None`
    for a variable with none. `per_variable_scope_intakes` runs parallel to `unique` too:
    the scope entries and total intent each part's words state
    (`stated_scope.part_scope_intakes`), read from the query when it asks for one variable
    and from each variable's own text when it asks for several.
    `per_variable_party_readings` runs parallel to `unique` too: for a part whose sources
    hold contested organisations (`source_scan.SourceHit.contested`), its readings with some
    or all of them as entities in place of sources (`_party_readings`), one of which the part
    runs in place of its own where the data read one of them as a party
    (`source_choice.read_organisations`), and `None` for any other part.
    `extraction` is the extraction with the organisations the query names joined to its
    entities (`_with_organisations`), the one every reader after `_prepare_extraction` takes:
    the name corrections, the layer-2 key and `unclaimed`.
    """

    date_request: DateRequest | None
    unique: list[str]
    entities: list[str]
    per_variable_entities: list[list[str]]
    per_variable_entity_roles: list[dict[str, list[DoorRole]]]
    per_variable_member_entities: list[frozenset[str]]
    per_variable_qualifiers: list[tuple[CodedQualifier, ...]]
    per_variable_operations: list[tuple[Operation, ...]]
    per_variable_open_axes: list[frozenset[RoleAxis]]
    per_variable_date_requests: list[DateRequest | None]
    interpretation_variables: list[str]
    dropped_qualifier_phrases: list[str]
    per_variable_measure_qualifiers: list[tuple[str, ...]]
    per_variable_dropped_phrases: list[tuple[str, ...]]
    clamp_warnings: list[Warning]
    qualifier_warnings: list[Warning]
    qualifier_facts: dict[WarningKey, DepartureFact]
    entity_words: dict[str, str]
    entity_surfaces: dict[str, str]
    unit_text: str | None
    per_variable_unit_texts: list[str | None]
    separate: list[bool]
    read_connectives: frozenset[int]
    per_variable_sources: list[tuple[SourceMention, ...]]
    per_variable_source_scopes: list[SourceScope | SourceRefusal | None]
    per_variable_source_codes: list[str | None]
    per_variable_compiled_codes: list[PartCode | None]
    per_variable_scope_intakes: list[ScopeIntake]
    text: str
    extraction: Extraction
    unclaimed: tuple[str, ...] = ()
    exclusions: tuple[LeftOut, ...] = ()
    exclusion_warnings: tuple[Warning, ...] = ()
    exclusion_targets: tuple[tuple[int, ...], ...] = ()
    per_variable_exclusions: list[PartExclusions] | None = None
    name_corrections: tuple[NameCorrection, ...] = ()
    per_variable_party_readings: list[PartyReading | None] | None = None

    def _unmet_exclusions(
        self, regions: Sequence[RegionResult], *, parts: Sequence[int]
    ) -> list[tuple[Warning, LeftOut]]:
        """Each of `exclusions` with its warning whose phrase qualifies one of `parts`, the
        positions in `unique` of `regions`, whose region does not leave it out. One that
        refuses shows nothing that leaves it out, whatever it judged before refusing."""
        return [
            (warning, exclusion)
            for warning, exclusion, targets in zip(
                self.exclusion_warnings, self.exclusions, self.exclusion_targets, strict=True
            )
            if any(
                not (region.specs and exclusion.phrase in region.met_exclusions)
                for i, region in zip(parts, regions, strict=True)
                if i in targets
            )
        ]

    def surviving(self, regions: Sequence[RegionResult]) -> list[Warning]:
        """The request-wide warnings that stay true across `regions`, one per part of
        `unique` in order: a dropped phrase no region admitted
        (`_surviving_qualifier_warnings`), and a phrase the query asked to leave out that the
        region of some part it qualifies does not leave out."""
        return _surviving_qualifier_warnings(
            self.qualifier_warnings, self.dropped_qualifier_phrases, regions
        ) + [warning for warning, _ in self._unmet_exclusions(regions, parts=range(len(regions)))]

    def dropped_phrases_of(self, i: int) -> list[str]:
        """The dropped phrases the part of `unique[i]` reads: the request's, then its own."""
        return [*self.dropped_qualifier_phrases, *self.per_variable_dropped_phrases[i]]

    def onto(self, region: RegionResult, i: int) -> RegionResult:
        """`region`, the part of `unique[i]`, holding what this extraction knows beyond it:
        the departures raised beside the request-wide qualifier warnings, a
        `query_qualifier_dropped` warning and its departure for each word narrowing its own
        measure that its answer leaves unapplied, the request-wide qualifier warnings whose
        phrase the region leaves unapplied (`RegionResult.unapplied_qualifier_warnings`), the
        query's words for each entity and those its resolution read
        (`RegionResult.entity_surfaces`), the names read in place of the query's words
        (`RegionResult.name_corrections`), and that its door asks for members in the
        question's words (`RegionResult.members_in_words`). A stretch the query asks to leave
        out raises its departure only on a part it qualifies whose answer does not leave it
        out, so the envelope's warning shows it on no other part.

        The answer applies a narrowing word when the region admitted it as a qualifier
        phrase, or when every Spec it answers with binds a value or a party named by it
        (`_binds_value_named`), such as "concessional" on a part that binds IDS
        Concessional, or "multilateral" on one whose provider is `DACMultilaterals`. A part
        with no answer warns only for the words the model dropped
        (`per_variable_dropped_phrases`): its refusal states why nothing is shown, and the
        words of its variable stay unjudged."""
        judged = (
            self.per_variable_measure_qualifiers[i]
            if region.specs
            else self.per_variable_dropped_phrases[i]
        )
        unapplied = [
            phrase
            for phrase in judged
            if _normalize(phrase) not in region.admitted_qualifier_phrases
            and not (region.specs and all(_binds_value_named(phrase, s) for s in region.specs))
        ]
        warnings = [qualifier_dropped_warning(phrase) for phrase in unapplied]
        unmet = self._unmet_exclusions([region], parts=[i])
        facts = self.qualifier_facts | {
            warning_key(warning): _exclusion_fact(exclusion) for warning, exclusion in unmet
        }
        for warning, phrase in zip(warnings, unapplied, strict=True):
            if (fact := _dropped_phrase_fact(phrase)) is not None:
                facts[warning_key(warning)] = fact
        if warnings:
            region = dataclasses.replace(region, warnings=(*region.warnings, *warnings))
        request_unapplied = unapplied_qualifier_warnings(
            self.qualifier_warnings,
            self.dropped_qualifier_phrases,
            admitted=region.admitted_qualifier_phrases,
        )
        return with_entity_words(
            dataclasses.replace(
                with_departure_facts(region, facts),
                unapplied_qualifier_warnings=(
                    *request_unapplied,
                    *(warning for warning, _ in unmet),
                ),
                name_corrections=self.name_corrections,
                members_in_words=True,
            ),
            self.entity_words,
            surfaces=self.entity_surfaces,
        )


def _exclusion_fact(stretch: LeftOut) -> DepartureFact:
    """The `measure_not_applied` departure of a stretch a part's answer does not leave out."""
    return exclusion_not_applied(asked=stretch.phrase[: settings().max_dropped_qualifier_length])


def _binds_value_named(phrase: str, spec: Spec) -> bool:
    """Whether `spec` binds a value `phrase` names (`phase_rank_bind.names_value`):
    "bilateral" names DAC1 "bilateral ODA" and IDS creditor type "Bilateral", and no name
    of DAC2a "official development assistance"; "concessional" names no "official
    non-concessional flow"."""
    return any(
        names_value(phrase, prop=slot.key.property.ref, value=ref.ref, label=ref.label)
        for slot in spec.resolved.trace.slot_filters
        if slot.key.property is not None
        for ref in slot.refs
    ) or _binds_party_named(phrase, spec)


def _binds_party_named(phrase: str, spec: Spec) -> bool:
    """Whether `spec` binds, on its axis, a party a dataset declares `phrase` names
    (`DatasetDeclaration.party_words`): "multilateral" names `DACMultilaterals` as the
    provider of a DAC2a amount."""
    entities = spec.definition.identity.entities
    return any(
        declared.party in (entities.get(declared.axis) or ())
        for dataset in get_index().datasets.values()
        for declared in dataset.party_words
        if declared.word == phrase.casefold()
    )


def _narrowing_words(variable: str) -> tuple[str, ...]:
    """The narrowing words (`_NARROWING_WORDS`) `variable` holds as whole words
    (`phase_rank_bind.holds_words`): "bilateral" in "bilateral aid", "multilateral" in
    "imputed multilateral aid", and none in "non-concessional flows"."""
    return tuple(word for word in _NARROWING_WORDS if holds_words(variable, word))


def _held_by_one(phrase: str, narrowing: Sequence[tuple[str, ...]]) -> bool:
    """Whether dropped `phrase` is a narrowing word that exactly one variable holds
    (`narrowing`, each variable's `_narrowing_words`), so it narrows that part alone."""
    folded = phrase.casefold()
    return sum(folded in words for words in narrowing) == 1


def _split_unit_phrases(phrases: Sequence[str]) -> tuple[str | None, list[str]]:
    """The unit text the dropped `phrases` bind as, and the phrases left over.

    A phrase joins the unit text when it binds on its own and the text so far plus the
    phrase still binds (`unit.bind.names_unit_request`), so a second phrase that conflicts
    with the first ("in euros", "in yen") stays dropped, and a bare year ("in 2020") never
    becomes the base year of a currency another phrase names. Phrases are taken in query
    order."""
    claimed: list[str] = []
    rest: list[str] = []
    for phrase in phrases:
        if names_unit_request(phrase) and names_unit_request(" ".join([*claimed, phrase])):
            claimed.append(phrase)
        else:
            rest.append(phrase)
    return (" ".join(claimed) or None), rest


def _part_unit_text(own: str | None, shared: str | None) -> str | None:
    """A part's unit request: the unit text extraction peeled from its own variable
    (`ExtractionResult.unit_texts`) with the request-wide `shared` one, or its own alone
    when the two bind as no one request ("in euros" beside "in yen"). A part whose variable
    held none carries `shared`, so a variable's own phrase reaches no other part."""
    if own is None or shared is None:
        return own or shared
    joined = f"{own} {shared}"
    return joined if names_unit_request(joined) else own


def _dropped_phrase_fact(phrase: str) -> DepartureFact | None:
    """The departure of a phrase extraction dropped as naming no code and binding as no unit
    request (`_split_unit_phrases`): a price basis the binder refuses ("constant 2022
    dollars" beside a second currency), which the unit token clears; else a request
    the answer could apply (`departures.names_request`: "fragile states"), which no one
    slot clears. Operation wording ("Trend in", "compare the") writes none, and its
    `query_qualifier_dropped` warning alone discloses it."""
    if names_price_basis(phrase):
        return price_basis_not_applied(asked=phrase, shown=None)
    return measure_not_applied(asked=phrase, property_dcid=None) if names_request(phrase) else None


def _exclusion_warning(phrase: str) -> Warning:
    """The `query_qualifier_dropped` warning for a stretch the query asked to leave out, which
    quotes it as a dropped phrase is shown."""
    return qualifier_dropped_warning(phrase[: settings().max_dropped_qualifier_length])


def _breakdown_under(phrase: str, *, verbs: Collection[str], query: str) -> RoleAxis | None:
    """The flow axis `phrase` asks one value per member of, under a part's operation
    `verbs`. An each or every phrase ("for each donor") asks it under any verb
    (`role_text.every_member_breakdown_axis`). "By" or "across" a role ("by donor") asks it
    only where no verb but a rank claims the phrase (`role_text.breakdown_axis`), since
    "across donors" on a sum asks for the total."""
    if all(verb == "rank" for verb in verbs):
        return breakdown_axis(phrase, query=query)
    return every_member_breakdown_axis(phrase, query=query)


def _prepare_extraction(
    extraction: Extraction,
    extract_result: ExtractionResult,
    *,
    query: str,
    left_out: Sequence[LeftOut] = (),
) -> _ExtractionPrep:
    """Prepare deduped extraction fields, qualifiers, and warnings for `handle`.
    ``query`` is the text with each of the ``left_out`` stretches blanked. The prep holds no
    name corrections; `handle` adds them, read from `_ExtractionPrep.extraction`.

    An organisation the query names that is no source and that no entity covers
    (`source_scan.find_organisations`) joins the extraction's entities first, as if the
    model had returned it (`_with_organisations`). A contested source
    (`source_scan.SourceHit.contested`) gives its parts a reading with it as a party
    (`_party_readings`)."""
    date_request = dates_to_request(extraction.dates)
    unique: list[str] = _dedupe_stripped(extraction.variables)

    # The sources the query names, each attached to the variables it qualifies. An entity
    # whose every occurrence a source covers is that source, and no entity of the part.
    catalogue = current_catalogue()
    dictionary = source_dictionary(catalogue)
    codes = code_spans(query, get_index())
    extraction = _with_organisations(
        extraction,
        [
            (mention.text, mention.targets)
            for mention in find_organisations(
                query,
                dictionary=dictionary,
                entity_spans=[entity.span or entity.text for entity in extraction.entities],
                measures=unique,
                index=get_index(),
                codes=codes,
            )
        ],
        query=query,
        unique=unique,
    )
    scan = _scan(
        extraction,
        query=query,
        unique=unique,
        left_out=left_out,
        codes=codes,
        dictionary=dictionary,
    )
    per_variable_sources = scan.per_measure
    # Each stretch qualifies the measures of the clause its blank lies in, else every one.
    disclosed = [
        (stretch, targets)
        for stretch, targets in zip(left_out, scan.stretch_targets, strict=True)
        if stretch.phrase not in scan.source_stretches
    ]
    exclusions = tuple(stretch for stretch, _ in disclosed)
    exclusion_targets = tuple(targets for _, targets in disclosed)
    exclusion_warnings = tuple(_exclusion_warning(stretch.phrase) for stretch in exclusions)
    per_part_stretches = [
        tuple(stretch for stretch, targets in disclosed if i in targets) for i in range(len(unique))
    ]
    per_variable_exclusions = [
        PartExclusions(stretches=stretches, requirements=requirements)
        for stretches, requirements in zip(
            per_part_stretches,
            exclusion_requirements(per_part_stretches, variables=unique),
            strict=True,
        )
    ]
    # The extraction names no series code. A token of the query that writes a held one does.
    per_variable_source_codes = scan_source_codes(query, measures=unique, index=get_index())
    # An entity that only asks for one value per member of a flow role ("every donor",
    # "across donors") names no place (`role_text.breakdown_axis`): it is a breakdown
    # phrase, which opens that role as a dropped one does (`_breakdown_under`), on the
    # parts it scopes.
    extracted_entities, entity_breakdowns = _split_entities(extraction, scan, query=query)
    entities: list[str] = list(dict.fromkeys(_entity_key(e) for e in extracted_entities))
    legs = _leg_readings(
        extracted_entities,
        scan,
        query=query,
        unique=unique,
        variables=extraction.variables,
        catalogue=catalogue,
    )

    # A relative date the extraction could not read is gone from `dates`; its warning says so.
    clamp_warnings: list[Warning] = list(extract_result.warnings)
    # Compare distinct model measures with the cap, before emitting a warning.
    if extract_result.raw_variable_count > settings().max_variables:
        clamp_warnings.append(
            Warning(
                code=VARIABLES_CLAMPED,
                severity="warn",
                message=(
                    f"Query named {extract_result.raw_variable_count} variables; "
                    f"resolved the first {settings().max_variables}."
                ),
            )
        )
    # Keep the pre-cap count so this warning reports phrases omitted from the list.
    if extract_result.raw_dropped_qualifier_count > settings().max_dropped_qualifiers:
        overflow = extract_result.raw_dropped_qualifier_count - settings().max_dropped_qualifiers
        clamp_warnings.append(
            Warning(
                code=QUERY_QUALIFIER_DROPPED,
                severity="degraded",
                message=(
                    f"{overflow} additional unsupported qualifier"
                    f"{'s were' if overflow != 1 else ' was'} not applied."
                ),
            )
        )

    # Qualifier sidecars use stripped `unique` keys. Missing entries default to ().
    per_variable_qualifiers = [extract_result.qualifiers.get(v, ()) for v in unique]
    per_variable_date_requests = [
        dates_to_request(extract_result.dates_by_variable[v])
        if v in extract_result.dates_by_variable
        else date_request
        for v in unique
    ]
    interpretation_variables = [
        recall_text(v, q) for v, q in zip(unique, per_variable_qualifiers, strict=True)
    ]

    # A dropped phrase that only names a source is applied by its mention and raises no
    # warning. The words of a phrase stating a provider scope are its scope entry's
    # (`stated_scope.without_scope_phrases`), so where a part's words state that entry they
    # leave the dropped phrases and narrow no measure. A dropped phrase is in no part's
    # text, so only a request of one part, whose intake read the whole message, states the
    # entry it reads; on a request of several the phrase stays and warns.
    intakes = part_scope_intakes(message=query, part_texts=unique)
    reading = intakes[0] if len(intakes) == 1 else None
    dropped = list(
        dict.fromkeys(
            phrase
            for qualifier in extraction.dropped_qualifiers
            if (phrase := without_scope_phrases(qualifier.strip(), reading))
        )
    )
    narrowing = [
        _narrowing_words(without_scope_phrases(v, intake))
        for v, intake in zip(unique, intakes, strict=True)
    ]
    per_variable_dropped_phrases = [
        tuple(
            phrase
            for phrase in dropped
            if _held_by_one(phrase, narrowing) and phrase.casefold() in words
        )
        for words in narrowing
    ]
    dropped = [phrase for phrase in dropped if not _held_by_one(phrase, narrowing)]
    naming_sources = source_phrases(dropped, query=query, hits=scan.hits)
    unit_text, dropped_qualifier_phrases = _split_unit_phrases(
        [phrase for phrase in dropped if phrase not in naming_sources]
    )
    # An operation whose phrase only asks for one value per member of a flow role is a
    # breakdown (`_breakdown_under`): it opens that role, as `open_roles` does on the parsed
    # door, and runs no operation. A dropped phrase, or an entity asking for it on the parts
    # it scopes, opens that role on each such part whose operations read it so, by the same
    # rule. Beside a sum it opens no part, and the total meets it (`role_text.breakdown_axis`),
    # so it raises no warning either way.
    per_variable_operations: list[tuple[Operation, ...]] = []
    per_variable_open_axes: list[frozenset[RoleAxis]] = []
    for v in unique:
        operations = extract_result.operations.get(v, ())
        verbs = {op.verb for op in operations}
        read = [
            (op, _breakdown_under(op.phrase, verbs=(op.verb,), query=query)) for op in operations
        ]
        breakdown_phrases = [
            *(words for words, scope in entity_breakdowns if v in scope),
            *dropped_qualifier_phrases,
        ]
        phrase_axes = (
            _breakdown_under(phrase, verbs=verbs, query=query) for phrase in breakdown_phrases
        )
        per_variable_operations.append(tuple(op for op, axis in read if axis is None))
        per_variable_open_axes.append(
            frozenset(axis for _, axis in read if axis is not None)
            | {axis for axis in phrase_axes if axis is not None}
        )
    dropped_qualifier_phrases = [
        phrase
        for phrase in dropped_qualifier_phrases
        if breakdown_axis(phrase, query=query) is None
    ]
    qualifier_warnings = [
        Warning(
            code=QUERY_QUALIFIER_DROPPED,
            severity="degraded",
            message=f'The requested qualifier "{phrase}" was not applied.',
        )
        for phrase in dropped_qualifier_phrases
    ]
    qualifier_facts = {
        warning_key(warning): fact
        for warning, phrase in zip(qualifier_warnings, dropped_qualifier_phrases, strict=True)
        if (fact := _dropped_phrase_fact(phrase)) is not None
    }
    request_phrases = {_normalize(phrase) for phrase in dropped}
    per_variable_measure_qualifiers = [
        tuple(word for word in words if _normalize(word) not in request_phrases)
        for words in narrowing
    ]

    entity_words: dict[str, str] = {}
    for entity in extracted_entities:
        entity_words.setdefault(_entity_key(entity), (entity.span or entity.text).strip())
    entity_surfaces = _entity_surfaces(extracted_entities, query=query)

    reading = read_sources(query, interpretation_variables)

    per_variable_source_scopes = [leg.source_scope for leg in legs]
    per_variable_compiled_codes = [
        _compiled_code(code, scope)
        for code, scope in zip(per_variable_source_codes, per_variable_source_scopes, strict=True)
    ]
    per_variable_unit_texts = [
        _part_unit_text(extract_result.unit_texts.get(v), unit_text) for v in unique
    ]
    claimed = [
        *interpretation_variables,
        *(operation.phrase for operations in per_variable_operations for operation in operations),
        *(
            text
            for entity in extraction.entities
            for text in (entity.text, entity.span, entity.cue)
            if text
        ),
        *(bound for date in extraction.dates for bound in (date.start, date.end) if bound),
        *([unit_text] if unit_text else []),
        *(text for text in per_variable_unit_texts if text),
        *(m.text for mentions in per_variable_sources for m in mentions if m.text),
        *(code for code in per_variable_source_codes if code),
    ]

    return _ExtractionPrep(
        date_request=date_request,
        unique=unique,
        entities=entities,
        per_variable_entities=[leg.entities for leg in legs],
        per_variable_entity_roles=[leg.entity_roles for leg in legs],
        per_variable_member_entities=[leg.member_entities for leg in legs],
        per_variable_qualifiers=per_variable_qualifiers,
        per_variable_operations=per_variable_operations,
        per_variable_open_axes=per_variable_open_axes,
        per_variable_date_requests=per_variable_date_requests,
        interpretation_variables=interpretation_variables,
        dropped_qualifier_phrases=dropped_qualifier_phrases,
        per_variable_measure_qualifiers=per_variable_measure_qualifiers,
        per_variable_dropped_phrases=per_variable_dropped_phrases,
        clamp_warnings=clamp_warnings,
        qualifier_warnings=qualifier_warnings,
        qualifier_facts=qualifier_facts,
        entity_words=entity_words,
        entity_surfaces=entity_surfaces,
        unit_text=unit_text,
        per_variable_unit_texts=per_variable_unit_texts,
        separate=list(reading.separate),
        read_connectives=reading.read_connectives,
        per_variable_sources=per_variable_sources,
        per_variable_source_scopes=per_variable_source_scopes,
        per_variable_source_codes=per_variable_source_codes,
        per_variable_compiled_codes=per_variable_compiled_codes,
        per_variable_scope_intakes=intakes,
        text=query,
        extraction=extraction,
        unclaimed=unclaimed_stretches(
            query, claimed=claimed, read_connectives=reading.read_connectives
        ),
        exclusions=exclusions,
        exclusion_warnings=exclusion_warnings,
        exclusion_targets=exclusion_targets,
        per_variable_exclusions=per_variable_exclusions,
        per_variable_party_readings=_party_readings(
            scan,
            extraction,
            query=query,
            unique=unique,
            left_out=left_out,
            codes=codes,
            source_codes=per_variable_source_codes,
            catalogue=catalogue,
            dictionary=dictionary,
        ),
    )


def _compiled_code(code: str | None, scope: SourceScope | SourceRefusal | None) -> PartCode | None:
    """What the series `code` a part names compiles to within `scope`
    (`source_code.compile_part_code`), `None` where the part names none."""
    if code is None:
        return None
    return compile_part_code(
        code, scope=scope if isinstance(scope, SourceScope) else None, index=get_index()
    )


def _with_organisations(
    extraction: Extraction,
    named: Sequence[tuple[str, tuple[int, ...]]],
    *,
    query: str,
    unique: Sequence[str],
) -> Extraction:
    """`extraction` with an entity for each organisation of `named` it does not already hold,
    its words and the positions in `unique` of the measures it qualifies, its role grounded
    as a returned entity's is (`extract._ground_roles`): "Global Fund" in "Global Fund
    disbursements to Kenya" stands straight before a measure word, so it is the provider
    (origin). Each scopes the extraction's variables that are those measures, every one when
    it qualifies them all. An organisation whose words, normalised
    (`normalise_source_text`), are an extracted entity's keeps that entity and its role
    (`_held_entities`)."""
    held = _held_entities(extraction)
    named = [
        (words, targets) for words, targets in named if normalise_source_text(words) not in held
    ]
    if not named:
        return extraction
    added = [
        ExtractedEntity(
            text=words,
            span=words,
            variables=(
                []
                if len(targets) == len(unique)
                else [
                    position
                    for position, variable in enumerate(extraction.variables)
                    if variable.strip() in {unique[target] for target in targets}
                ]
            ),
        )
        for words, targets in dict.fromkeys(named)
    ]
    grounded = _ground_roles(query, added, extraction.variables)
    return extraction.model_copy(update={"entities": [*extraction.entities, *grounded]})


def _held_entities(extraction: Extraction) -> dict[str, str]:
    """The key (`_entity_key`) of each of `extraction`'s entities by its words, normalised
    (`normalise_source_text`), the first entity's where several share them."""
    held: dict[str, str] = {}
    for entity in extraction.entities:
        held.setdefault(normalise_source_text(entity.span or entity.text), _entity_key(entity))
    return held


def _party_readings(
    scan: SourceScan,
    extraction: Extraction,
    *,
    query: str,
    unique: Sequence[str],
    left_out: Sequence[LeftOut],
    codes: Sequence[tuple[int, int]],
    source_codes: Sequence[str | None],
    catalogue: Catalogue,
    dictionary: SourceDictionary,
) -> list[PartyReading | None] | None:
    """Each part's readings with the contested sources of `scan` (`SourceHit.contested`) it
    holds as entities, parallel to `unique`, or `None` when no source is contested. A part
    none of whose measures a contested source qualifies has none.

    A part's contest names each organisation it holds by its words, with the dcids the entity
    pack names it (`entity_resolution.pack.organisations_named`), the word naming a party
    to a flow before a mention of it (`source_scan.direction_word`), "to" over "from", the
    position of its mention among the part's sources (`SourceScan.hit_mentions`), and the
    entity a reading with it as a party names it by: the extracted entity of its words
    where one holds them, else its own words (`_with_organisations`). Its
    readings read a set of the organisations as parties (`_readings_with`): every one of
    them (`PartyReading.reading`), or the set the part's data choose
    (`PartyReading.reading_with`)."""
    contested = [hit for hit in scan.hits if hit.contested]
    if not contested:
        return None
    mentions = {
        hit: mentions
        for hit, mentions in zip(scan.hits, scan.hit_mentions, strict=True)
        if hit.contested
    }
    held = _held_entities(extraction)

    @functools.cache
    def legs(parties: frozenset[str]) -> list[LegReading]:
        return _readings_with(
            [hit for hit in contested if hit.text in parties],
            extraction,
            query=query,
            unique=unique,
            left_out=left_out,
            codes=codes,
            source_codes=source_codes,
            catalogue=catalogue,
            dictionary=dictionary,
        )

    spans = [entity.span or entity.text for entity in extraction.entities]
    readings: list[PartyReading | None] = []
    for i in range(len(unique)):
        hits = [hit for hit in contested if i in hit.targets]
        if not hits:
            readings.append(None)
            continue
        organisations = tuple(
            ContestedOrganisation(
                words=words,
                parties=organisations_named(words),
                direction=_direction(
                    {
                        direction_word(query, hit, entity_spans=spans)
                        for hit in hits
                        if hit.text == words
                    }
                ),
                mention=next(
                    (mentions[hit][i] for hit in hits if hit.text == words and i in mentions[hit]),
                    None,
                ),
                entity=held.get(normalise_source_text(words), words.strip()),
            )
            for words in dict.fromkeys(hit.text for hit in hits)
        )
        every = frozenset(organisation.words for organisation in organisations)
        readings.append(
            PartyReading(
                contested=organisations,
                reading=legs(every)[i],
                reading_with=lambda parties, i=i: legs(parties)[i],
            )
        )
    return readings


def _direction(words: AbstractSet[DirectionWord | None]) -> DirectionWord | None:
    """The word naming a party to a flow among `words`, those before each mention of one
    organisation: "to" over "from"."""
    return "to" if "to" in words else "from" if "from" in words else None


def _readings_with(
    parties: Sequence[SourceHit],
    extraction: Extraction,
    *,
    query: str,
    unique: Sequence[str],
    left_out: Sequence[LeftOut],
    codes: Sequence[tuple[int, int]],
    source_codes: Sequence[str | None],
    catalogue: Catalogue,
    dictionary: SourceDictionary,
) -> list[LegReading]:
    """Each part's reading with the contested sources `parties` read as parties, parallel to
    `unique`.

    The reading adds an entity for each of `parties` the extraction did not return
    (`_with_organisations`) and scans the query again with them among the scan's `parties`,
    so the scan reads each as a party and every other source as a source; its entities,
    their roles and its source scope follow from that scan as the source reading's do
    (`_leg_readings`), and the part's series code, of `source_codes`, compiles within that
    scope."""
    party = _with_organisations(
        extraction, [(hit.text, hit.targets) for hit in parties], query=query, unique=unique
    )
    scan = _scan(
        party,
        query=query,
        unique=unique,
        left_out=left_out,
        codes=codes,
        dictionary=dictionary,
        parties=frozenset(normalise_source_text(hit.text) for hit in parties),
    )
    extracted, _ = _split_entities(party, scan, query=query)
    legs = _leg_readings(
        extracted,
        scan,
        query=query,
        unique=unique,
        variables=party.variables,
        catalogue=catalogue,
    )
    return [
        dataclasses.replace(leg, source_code=_compiled_code(code, leg.source_scope))
        for leg, code in zip(legs, source_codes, strict=True)
    ]


def _scan(
    extraction: Extraction,
    *,
    query: str,
    unique: Sequence[str],
    left_out: Sequence[LeftOut],
    codes: Sequence[tuple[int, int]],
    dictionary: SourceDictionary,
    parties: AbstractSet[str] = frozenset(),
) -> SourceScan:
    """The sources `query` names, read against `extraction`'s entities, with the contested
    organisations whose normalised words are among `parties` read as parties
    (`scan_sources`)."""
    return scan_sources(
        query,
        dictionary=dictionary,
        measures=unique,
        entity_spans=[entity.span or entity.text for entity in extraction.entities],
        index=get_index(),
        left_out=left_out,
        codes=codes,
        parties=parties,
    )


def _split_entities(
    extraction: Extraction, scan: SourceScan, *, query: str
) -> tuple[list[ExtractedEntity], list[tuple[str, frozenset[str]]]]:
    """`extraction`'s entities that name places or parties, and those that only ask for one
    value per member of a flow role ("every donor", "across donors"), each with the measures
    it scopes. An entity whose every occurrence a source covers is that source
    (`SourceScan.source_entities`), and neither."""
    extracted: list[ExtractedEntity] = []
    breakdowns: list[tuple[str, frozenset[str]]] = []
    for entity in extraction.entities:
        words = entity.span or entity.text
        if words in scan.source_entities:
            continue
        if breakdown_axis(words, query=query) is None:
            extracted.append(entity)
        else:
            breakdowns.append((words, scoped_measures(entity, extraction.variables)))
    return extracted, breakdowns


def _leg_readings(
    extracted: Sequence[ExtractedEntity],
    scan: SourceScan,
    *,
    query: str,
    unique: Sequence[str],
    variables: Sequence[str],
    catalogue: Catalogue,
) -> list[LegReading]:
    """Each part's entities of `extracted`, their roles, those asked for as their members,
    and its source scope compiled from `scan`, parallel to `unique`."""
    extracted = list(extracted)
    part_positions = _part_positions(extracted, measures=unique, variables=variables)
    # Every part reads the same query, so the readings of the entities that carry a role are
    # made once for the request, on the query folded once.
    folded_query = _grounding_form(query)
    stated = [
        bool(entity.role) and _states_direction(entity, folded_query=folded_query)
        for entity in extracted
    ]
    preceding = veto_words(extracted, query=query)
    legs: list[LegReading] = []
    for positions, mentions in zip(part_positions, scan.per_measure, strict=True):
        part = [extracted[i] for i in positions]
        legs.append(
            LegReading(
                entities=list(dict.fromkeys(_entity_key(e) for e in part)),
                entity_roles=_entity_roles(
                    positions, entities=extracted, stated=stated, preceding=preceding
                ),
                # A text any mention of the part asks the members of. A directed mention
                # asks on its own end (`DoorRole.members`).
                member_entities=frozenset(_entity_key(e) for e in part if e.members),
                # A raw-text mention has no field of its own, so a refusal points at the
                # query, and mentions that leave a part no source refuse that part alone.
                source_scope=compile_sources(
                    list(mentions),
                    catalogue=catalogue,
                    pointers=["/input/query"] * len(mentions),
                    refuse_empty=False,
                )
                if mentions
                else None,
            )
        )
    return legs


def _disclosed_exclusions(left_out: Sequence[LeftOut]) -> tuple[LeftOut, ...]:
    """The `left_out` stretches an answer with no part reports: each but those that leave out
    sources alone (`source_scan.left_out_sources`), which leave nothing a warning should
    name."""
    dictionary, index = source_dictionary(current_catalogue()), get_index()
    return tuple(
        stretch
        for stretch in left_out
        if not left_out_sources(stretch, dictionary=dictionary, index=index)[1]
    )


def _log_unclaimed_words(prep: _ExtractionPrep) -> None:
    """Log, once, the words of `prep.unclaimed` that are not function words."""
    words = unclaimed_words(" ".join(prep.unclaimed), claimed=())
    if words:
        logger.info("raw_query_coverage unclaimed=%s", ",".join(words))


async def _handle_single_variable(
    prep: _ExtractionPrep,
    *,
    query: str,
    timing: dict[str, int],
    extract_usage: dict | None,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    population_memo: OptionsPopulationMemo | None,
    place_options_pool: PlaceOptionsPool | None,
    params: EntryParams,
    start_ms: int,
    scope: CacheScope | None,
) -> AsyncGenerator[Event]:
    """Resolve one variable directly, bypassing the same-shape combiner."""
    out = _FanOutResult()
    async with aclosing(
        _run_variables_streaming(
            prep.unique,
            entities=prep.entities,
            per_variable_entities=prep.per_variable_entities,
            per_variable_entity_roles=prep.per_variable_entity_roles,
            per_variable_member_entities=prep.per_variable_member_entities,
            date_request=prep.date_request,
            query=prep.text,
            pac=params.pac,
            graph=graph,
            resolver=resolver,
            admission=admission,
            llm=llm,
            binder=params.binder,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            population_memo=population_memo,
            place_options_pool=place_options_pool,
            base_timing=timing,
            pre_resolved=None,
            entity_surfaces=prep.entity_surfaces,
            entity_words=prep.entity_words,
            deadline=params.deadline,
            degrade_on_error=False,
            use_semaphore=False,
            out=out,
            per_variable_dropped_qualifier_phrases=[
                prep.dropped_phrases_of(i) for i in range(len(prep.unique))
            ],
            per_variable_exclusions=prep.per_variable_exclusions,
            query_text=prep.text,
            unclaimed=prep.unclaimed,
            per_variable_qualifiers=prep.per_variable_qualifiers,
            per_variable_operations=prep.per_variable_operations,
            per_variable_open_axes=prep.per_variable_open_axes,
            per_variable_date_requests=prep.per_variable_date_requests,
            per_variable_unit_texts=prep.per_variable_unit_texts,
            per_variable_source_scopes=prep.per_variable_source_scopes,
            per_variable_source_codes=prep.per_variable_compiled_codes,
            per_variable_scope_intakes=prep.per_variable_scope_intakes,
            per_variable_party_readings=prep.per_variable_party_readings,
            unit_text_best_effort=True,
            door="raw_text",
            include_option_coverage=params.include_option_coverage,
            stream_slot_options=params.stream_option_coverage,
            include_observations=params.include_observations,
            stream_observations=params.stream_observations,
            coverage=params.coverage,
        )
    ) as events:
        async for ev in events:
            yield ev
    region = out.regions[0]
    # Record run health. `is_storable` decides cache eligibility.
    if scope is not None:
        scope.note_health(timed_out=out.timed_out, degraded_leg=any_degraded(out.regions))
    total_usage = _sum_usage(extract_usage, region.llm_usage)
    # The part and its late departures read the same facts and entity words.
    answered = prep.onto(region, 0)
    response = assemble_region(
        answered,
        query=query,
        extra_warnings=prep.clamp_warnings + prep.surviving([region]),
        start_ms=start_ms,
        engine_build=build_commit(),
        max_candidates=params.max_candidates,
        include_observations=params.include_observations,
        include_skills=params.include_skills,
        skills_index=params.skills_index,
        llm_usage=total_usage,
    )
    for event in answer_events(response):
        yield event
    slot_options_warnings: list[Warning] = []
    late_departures: list[LateDepartures] = []
    incomplete: list[str] = []
    async with aclosing(
        _stream_slot_options(
            [answered],
            deadline=params.deadline,
            warnings_out=slot_options_warnings,
            late_departures_out=late_departures,
            incomplete_out=incomplete,
            response=response,
            answered=[answered],
        )
    ) as events:
        async for ev in events:
            yield ev
    if scope is not None:
        scope.note_health(trailing_incomplete=bool(incomplete))
    yield DoneEvent(
        elapsed_ms=now_ms() - start_ms,
        timed_out=out.timed_out,
        degraded_leg=any_degraded(out.regions),
        trace_id=get_trace_id(),
        telemetry=DoneTelemetry(llm_usage=total_usage),
        warnings=slot_options_warnings,
        late_departures=late_departures,
    )


async def _handle_multi_variable(
    prep: _ExtractionPrep,
    *,
    query: str,
    timing: dict[str, int],
    extract_usage: dict | None,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    rate_cache: RateTableCache | None,
    population_memo: OptionsPopulationMemo | None,
    place_options_pool: PlaceOptionsPool | None,
    params: EntryParams,
    start_ms: int,
    scope: CacheScope | None,
) -> AsyncGenerator[Event]:
    """Resolve shared entities once, fan out variables concurrently, then combine.

    Non-infrastructure leg errors become no-data parts. Infrastructure failures
    fail the request. The shared resolver call runs only when some leg runs its pipeline
    (`leg_runs`), so a request whose every leg a source settles calls no resolver.
    """
    runs = any(
        leg_runs(source_scope, source_code)
        for source_scope, source_code in zip(
            prep.per_variable_source_scopes, prep.per_variable_compiled_codes, strict=True
        )
    )
    pre_resolved = (
        await _resolve_entities_bounded(
            prep.entities,
            resolver=resolver,
            deadline=params.deadline,
            surfaces=prep.entity_surfaces,
        )
        if runs
        else {}
    )
    out = _FanOutResult()
    async with aclosing(
        _run_variables_streaming(
            prep.unique,
            entities=prep.entities,
            per_variable_entities=prep.per_variable_entities,
            per_variable_entity_roles=prep.per_variable_entity_roles,
            per_variable_member_entities=prep.per_variable_member_entities,
            date_request=prep.date_request,
            query=prep.text,
            pac=params.pac,
            graph=graph,
            resolver=resolver,
            admission=admission,
            llm=llm,
            binder=params.binder,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            population_memo=population_memo,
            place_options_pool=place_options_pool,
            base_timing=timing,
            pre_resolved=pre_resolved,
            entity_words=prep.entity_words,
            deadline=params.deadline,
            degrade_on_error=True,
            use_semaphore=True,
            out=out,
            per_variable_dropped_qualifier_phrases=[
                prep.dropped_phrases_of(i) for i in range(len(prep.unique))
            ],
            per_variable_exclusions=prep.per_variable_exclusions,
            per_variable_qualifiers=prep.per_variable_qualifiers,
            per_variable_operations=prep.per_variable_operations,
            per_variable_open_axes=prep.per_variable_open_axes,
            per_variable_date_requests=prep.per_variable_date_requests,
            per_variable_unit_texts=prep.per_variable_unit_texts,
            per_variable_source_scopes=prep.per_variable_source_scopes,
            per_variable_source_codes=prep.per_variable_compiled_codes,
            per_variable_scope_intakes=prep.per_variable_scope_intakes,
            per_variable_party_readings=prep.per_variable_party_readings,
            unit_text_best_effort=True,
            per_variable_separate=prep.separate,
            door="raw_text",
            include_option_coverage=params.include_option_coverage,
            stream_slot_options=params.stream_option_coverage,
            include_observations=params.include_observations,
            stream_observations=params.stream_observations,
            coverage=params.coverage,
        )
    ) as events:
        async for ev in events:
            yield ev

    async with aclosing(
        _stream_combined_regions(
            [prep.onto(region, i) for i, region in enumerate(out.regions)],
            initial_llm_usage=extract_usage,
            query=query,
            # Filter before merging. same_shape unions admitted phrases per group.
            extra_warnings=prep.clamp_warnings + prep.surviving(out.regions),
            echo=None,
            timed_out=out.timed_out,
            graph=graph,
            start_ms=start_ms,
            params=params,
            # Built once every part's own source scope has exited, so the reads run
            # through the request's own catalogue.
            preview_reads=ComponentReads(
                graph=graph,
                sdmx=sdmx,
                coverage=params.coverage,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
                deadline=params.deadline,
                catalogue=current_catalogue(),
            ),
            scope=scope,
        )
    ) as events:
        async for event in events:
            yield event


async def handle(
    inp: RawTextInput,
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    admission: Admission,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    start_ms: int,
    params: EntryParams,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    scope: CacheScope | None = None,
) -> AsyncGenerator[Event]:
    query: str = inp.query

    if not query.strip():
        async with aclosing(
            _empty_query_response(query, start_ms=start_ms, params=params)
        ) as events:
            async for ev in events:
                yield ev
        return

    door_read = await _read_and_extract(query, llm=llm, deadline=params.deadline)
    extract_result, extract_usage, timing = (
        door_read.extract_result,
        door_read.usage,
        door_read.timing,
    )
    text, left_out = door_read.reading.text, door_read.reading.left_out
    # Record extraction once before the per-variable fan-out.
    yield StepEvent(variable_index=None, step="extract", ran=True, ms=timing["extract"])
    extraction = extract_result.extraction

    # An answer with no part says what the query asked to leave out and it did not apply,
    # beside the disposition the rest of the query reads as: text left without a measure once
    # a stretch is blanked refuses as any query naming none does.
    unapplied = (
        [_exclusion_warning(stretch.phrase) for stretch in _disclosed_exclusions(left_out)]
        if left_out and (extraction.kind != "resolution" or not extraction.variables)
        else []
    )

    # `extract` turns a catalogue or requires-analysis question that names a
    # measure into a resolution.
    if extraction.kind != "resolution":
        async with aclosing(
            _disposition_response(
                query,
                extraction,
                resolver=resolver,
                start_ms=start_ms,
                params=params,
                extract_usage=extract_usage,
                timing=timing,
                scope=scope,
                warnings=unapplied,
            )
        ) as events:
            async for ev in events:
                yield ev
        return

    if not extraction.variables:
        async with aclosing(
            _no_variables_response(
                query,
                start_ms=start_ms,
                params=params,
                extract_usage=extract_usage,
                timing=timing,
                warnings=unapplied,
            )
        ) as events:
            async for ev in events:
                yield ev
        return

    extraction = extraction.model_copy(
        update={
            "entities": await _recover_entity_surfaces(
                extraction.entities,
                query=text,
                resolver=resolver,
                deadline=params.deadline,
            )
        }
    )
    prep = _prepare_extraction(extraction, extract_result, query=text, left_out=left_out)
    # The prep's extraction holds the organisations the query names; nothing reads the
    # extraction it was prepared from.
    extraction = prep.extraction
    prep = dataclasses.replace(
        prep,
        name_corrections=await _name_corrections(
            extraction.entities, query=text, resolver=resolver, deadline=params.deadline
        ),
    )

    yield InterpretationEvent(
        variables=prep.interpretation_variables,
        entities=prep.entities,
        dates=extraction.dates,
        expected_results=len(prep.unique),
    )

    # Include entities, roles, dates, qualifiers, operations, sources, the unit text peeled
    # from each variable, the scope each part's words state, raw-text signals and the
    # extraction's warnings in the cache key so semantically distinct queries retain
    # separate entries.
    # `query_text` applies only to N=1, matching the single-variable cache readers.
    # A cache hit returns before binding. Core always supplies `scope`.
    # Direct tests may omit it.
    cached = None
    if scope is not None:
        cached = await scope.lookup_parsed(
            query=query,
            text=text,
            exclusions=tuple(zip(prep.exclusions, prep.exclusion_targets, strict=True)),
            variables=prep.unique,
            entities=_scoping_parts(
                extraction.entities, variables=extraction.variables, parts=prep.unique
            ),
            dates=extraction.dates,
            dropped_qualifiers=extraction.dropped_qualifiers,
            unit_texts=[extract_result.unit_texts.get(v) for v in prep.unique],
            variable_qualifiers=prep.per_variable_qualifiers,
            operations=prep.per_variable_operations,
            entity_roles=prep.per_variable_entity_roles,
            sources=prep.per_variable_sources,
            source_codes=prep.per_variable_source_codes,
            query_text=text if len(prep.unique) == 1 else None,
            separate=prep.separate,
            extraction_warnings=prep.clamp_warnings,
            measure_qualifiers=prep.per_variable_measure_qualifiers,
            open_axes=prep.per_variable_open_axes,
            stated_scopes=prep.per_variable_scope_intakes,
            llm_usage=extract_usage,
        )
    _log_unclaimed_words(prep)
    if scope is not None and cached is not None:
        async with aclosing(
            scope.hit_events(
                cached,
                start_ms=start_ms,
                llm_usage=extract_usage,
                place_options=replayed_place_options(
                    stream=params.stream_option_coverage,
                    graph=graph,
                    sdmx=sdmx,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    pool=place_options_pool,
                    coverage=params.coverage,
                    population_memo=population_memo,
                ),
            )
        ) as hit:
            async for ev in hit:
                yield ev
        return

    if len(prep.unique) == 1:
        async with aclosing(
            _handle_single_variable(
                prep,
                query=query,
                timing=timing,
                extract_usage=extract_usage,
                graph=graph,
                resolver=resolver,
                admission=admission,
                llm=llm,
                sdmx=sdmx,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
                population_memo=population_memo,
                place_options_pool=place_options_pool,
                params=params,
                start_ms=start_ms,
                scope=scope,
            )
        ) as events:
            async for ev in events:
                yield ev
        return

    async with aclosing(
        _handle_multi_variable(
            prep,
            query=query,
            timing=timing,
            extract_usage=extract_usage,
            graph=graph,
            resolver=resolver,
            admission=admission,
            llm=llm,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            population_memo=population_memo,
            place_options_pool=place_options_pool,
            params=params,
            start_ms=start_ms,
            scope=scope,
        )
    ) as events:
        async for ev in events:
            yield ev
