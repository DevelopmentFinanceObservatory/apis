"""Resolve pre-parsed inputs with per-part variable, entity, time, unit, and source data.

Parsed inputs bypass extraction.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing

from pydantic import ValidationError

from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.build import build_commit
from dfo_platform.trace import get_trace_id
from entity_resolution import Admission, Entity, EntityResolver, Resolution
from entity_resolution.types import entity_kind
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.conjoin import assemble_region
from search.engine.dataset_mention import apply_readings, read_mentions
from search.engine.deadline import _await_bounded
from search.engine.departures import WarningKey, dataset_not_applied, warning_key
from search.engine.enrichment import _stream_slot_options
from search.engine.entity_text import ENTITY_LIKE_PHRASING, detect_entity_like_phrasing
from search.engine.entry.multi import _stream_combined_regions
from search.engine.entry.params import EntryParams
from search.engine.envelope import _sum_usage, make_pipeline_step, now_ms
from search.engine.errors import EngineInputError
from search.engine.events import (
    DoneEvent,
    DoneTelemetry,
    Event,
    InterpretationEvent,
    LateDepartures,
    ResponseEvent,
)
from search.engine.extract import DateRequest, ExtractedDate, dates_to_request
from search.engine.fanout import _FanOutResult, _resolve_entities_bounded, _run_variables_streaming
from search.engine.interpret import PinnedResolved
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.qualifier_match import CodedQualifier
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.refs import confirm_dcids_present
from search.engine.region.result import RegionResult, with_departure_facts
from search.engine.retrieval.read_plan import current_year
from search.engine.role_text import (
    ROLE_TEXT_NOT_RECOGNISED,
    DoorRole,
    role_axis_for_text,
)
from search.engine.time_bind import parse_time_text, time_window_defaulted_warning
from search.heading.departure_fact import DepartureFact
from search.models import Operation, ParsedInput, ParsedPart, QueryEcho, Warning

logger = logging.getLogger(__name__)


def _time_for_part(part: ParsedPart) -> tuple[list[ExtractedDate], Warning | None]:
    """Parse one part's time text, warning when it is absent.

    Unparsed text gets its own parser warning. `current_year()` keeps date parsing
    pure and testable.
    """
    if part.time_text:
        return parse_time_text(part.time_text, current_year=current_year())
    return [], time_window_defaulted_warning()


def _role_query_for_part(part: ParsedPart) -> tuple[str, list[Warning]]:
    """Build the role query from variable text and bare entity names.

    Spaces separate entity names, keeping direction cues local. Axes come from
    `_entity_roles_for_part`. Return warnings for unrecognised role text. Those
    entities remain undirected.
    """
    words = [part.variable_text]
    warnings: list[Warning] = []
    for entity in part.entities or ():
        axis = role_axis_for_text(entity.role_text)
        if entity.role_text is not None and axis is None:
            warnings.append(
                Warning(
                    code=ROLE_TEXT_NOT_RECOGNISED,
                    severity="warn",
                    message=(
                        f"role_text {entity.role_text!r} did not match any recognised "
                        "role word ('from'/'to', or a synonym -- donor/creditor/lender/"
                        "exporter for 'from', recipient/debtor/borrower/importer for "
                        "'to'); entity treated as undirected."
                    ),
                )
            )
        words.append(entity.text)
    return " ".join(words), warnings


def _entity_roles_for_part(part: ParsedPart) -> dict[str, list[DoorRole]]:
    """Map entity text to declared axes and verbatim spans for one part.

    Preserve distinct mentions in order, including a repeated name on opposite
    axes. Missing or unrecognised role text leaves the entity undirected.
    """
    roles: dict[str, list[DoorRole]] = {}
    for entity in part.entities or ():
        axis = role_axis_for_text(entity.role_text)
        if axis is None:
            continue
        occurrences = roles.setdefault(entity.text, [])
        role = DoorRole(axis=axis, span=entity.text)
        if role not in occurrences:
            occurrences.append(role)
    return roles


def _entity_text_warning(part: ParsedPart) -> Warning | None:
    """Warn when entity-like text appears without a structured entities list.

    Parts with entities skip this heuristic. Parsed input does not run extraction.
    """
    if part.entities:
        return None
    token = detect_entity_like_phrasing(part.variable_text)
    if token is None:
        return None
    return Warning(
        code=ENTITY_LIKE_PHRASING,
        severity="warn",
        message=(
            f"variable_text {part.variable_text!r} appears to name an entity "
            f"({token!r}) in free text; passing it in this part's entities array "
            "instead resolves more reliably."
        ),
    )


def _per_part_warnings(
    parts: list[ParsedPart],
    dates_and_warnings: list[tuple[list[ExtractedDate], Warning | None]],
    role_queries_and_warnings: list[tuple[str, list[Warning]]],
) -> list[list[Warning]]:
    """Build one ordered warning list per part.

    Raw-text query warnings stay at the envelope level.

    Warnings cover time, role text, and entity-like variable text. Helper results
    are passed in to avoid recomputation.
    """
    per_part: list[list[Warning]] = [[w] if w is not None else [] for _, w in dates_and_warnings]
    for i, (_, warnings) in enumerate(role_queries_and_warnings):
        per_part[i].extend(warnings)
    for i, part in enumerate(parts):
        entity_text_warning = _entity_text_warning(part)
        if entity_text_warning is not None:
            per_part[i].append(entity_text_warning)
    return per_part


@dataclasses.dataclass(frozen=True, slots=True)
class _PartMentions:
    """What reading the parts' dataset mentions came to, per part in posted order."""

    qualifiers: list[tuple[CodedQualifier, ...]] | None
    warnings: list[list[Warning]]
    departure_facts: list[dict[WarningKey, DepartureFact]]
    usage: dict | None


async def _read_part_mentions(variable_texts: list[str], *, llm: SupportsLLM) -> _PartMentions:
    """Read the dataset mentions the parts' variable texts hold, as raw-text extraction
    reads a query's, and return each part's mention sidecar, warnings and departures with
    the token usage of the reading call.

    `dataset_mention.read_mentions` reads each mention and `apply_readings` decides what
    took effect, with the variables held as posted: a part is what the caller sent, so a
    reading binds where every part keeps its text, and one that would add, rename or
    remove a part is reported. Each reading that took no effect warns
    `query_qualifier_dropped` on the part its mention stands in, beside a
    `measure_not_applied` departure the source token clears (`dataset_not_applied`). With
    no mention no model is asked, the sidecar is `None` and so is the usage.
    """
    index = get_index()
    warnings: list[list[Warning]] = [[] for _ in variable_texts]
    departure_facts: list[dict[WarningKey, DepartureFact]] = [{} for _ in variable_texts]
    result = await read_mentions(
        "; ".join(variable_texts),
        variable_texts,
        (),
        index=index,
        llm=llm,
    )
    if not result.readings:
        return _PartMentions(None, warnings, departure_facts, result.usage)
    applied = apply_readings(
        variable_texts,
        result.readings,
        index=index,
        fixed_variables=True,
    )
    sidecar: list[tuple[CodedQualifier, ...]] = [
        (CodedQualifier(phrase=applied.phrases[position], spans=(), matches=()),)
        if position in applied.phrases
        else ()
        for position in range(len(variable_texts))
    ]
    for read in applied.reported:
        if (position := read.gated.variable) is None:
            continue
        warning = Warning(
            code=QUERY_QUALIFIER_DROPPED,
            severity="degraded",
            message=f'The requested qualifier "{read.mention.text}" was not applied.',
        )
        warnings[position].append(warning)
        departure_facts[position][warning_key(warning)] = dataset_not_applied(read.mention.text)
    return _PartMentions(sidecar, warnings, departure_facts, result.usage)


@dataclasses.dataclass(frozen=True, slots=True)
class PartsInterpretation:
    """What a fan-out leg is handed for each posted part, in posted order, with the token
    usage of the request's one mention-reading call."""

    variable_texts: list[str]
    entities: list[list[str]]
    dates: list[list[ExtractedDate]]
    date_requests: list[DateRequest | None]
    role_queries: list[str]
    entity_roles: list[dict[str, list[DoorRole]]]
    unit_texts: list[str | None]
    source_texts: list[str | None]
    operations: list[tuple[Operation, ...]]
    qualifiers: list[tuple[CodedQualifier, ...]] | None
    warnings: list[list[Warning]]
    departure_facts: list[dict[WarningKey, DepartureFact]]
    restatements: list[ParsedPart | None]
    mention_usage: dict | None

    def interpreted(self, region: RegionResult, position: int) -> RegionResult:
        """`region`, the fan-out leg of part `position`, carrying that part's warnings
        after its own, the departures they disclose, and its restatement."""
        restatement = self.restatements[position]
        return dataclasses.replace(
            with_departure_facts(region, self.departure_facts[position]),
            warnings=tuple(region.warnings) + tuple(self.warnings[position]),
            restatement=restatement,
            parsed_form=restatement,
        )


async def interpret_parts(
    parts: list[ParsedPart], *, llm: SupportsLLM, deadline: float
) -> PartsInterpretation:
    """Interpret posted parts for the parsed door and for `dataset_resubmit`'s parsed
    entries, so identical parts read the same mentions, carry the same warnings and
    count the same token usage on either door.

    Only absent time text gets `time_window_defaulted`; unrecognised text gets the
    parser's own warning. The mention reading is bounded by `deadline`.
    """
    variable_texts = [p.variable_text.strip() for p in parts]
    dates_and_warnings = [_time_for_part(p) for p in parts]
    role_queries_and_warnings = [_role_query_for_part(p) for p in parts]
    warnings = _per_part_warnings(parts, dates_and_warnings, role_queries_and_warnings)
    mentions = await _await_bounded(_read_part_mentions(variable_texts, llm=llm), deadline=deadline)
    for part_warnings, part_mention_warnings in zip(warnings, mentions.warnings, strict=True):
        part_warnings.extend(part_mention_warnings)
    return PartsInterpretation(
        variable_texts=variable_texts,
        entities=[[e.text for e in (p.entities or [])] for p in parts],
        dates=[dates for dates, _ in dates_and_warnings],
        date_requests=[dates_to_request(dates) for dates, _ in dates_and_warnings],
        role_queries=[rq for rq, _ in role_queries_and_warnings],
        entity_roles=[_entity_roles_for_part(p) for p in parts],
        unit_texts=[p.unit_text for p in parts],
        source_texts=[p.source_text for p in parts],
        operations=[tuple(p.operations) for p in parts],
        qualifiers=mentions.qualifiers,
        warnings=warnings,
        departure_facts=mentions.departure_facts,
        restatements=[
            _part_restatement(p, variable_text=v)
            for p, v in zip(parts, variable_texts, strict=True)
        ],
        mention_usage=mentions.usage,
    )


def _part_restatement(part: ParsedPart, *, variable_text: str) -> ParsedPart | None:
    """Rebuild a part from the inputs used by its resolution leg.

    Use the stripped text sent to resolution. Return `None` when it is empty,
    since a posted whitespace-only part cannot pass `ParsedPart` validation.
    """
    try:
        return ParsedPart(
            variable_text=variable_text,
            entities=part.entities,
            time_text=part.time_text,
            unit_text=part.unit_text,
            source_text=part.source_text,
            operations=part.operations,
        )
    except ValidationError:
        return None


def _dedupe_parts_with_indices(parts: list[ParsedPart]) -> list[tuple[int, ParsedPart]]:
    """Deduplicate identical parts in first-seen order, keeping each posted index.

    Equality includes variable text, entities with role text and dcid pins, time,
    unit, source, and operations. Parts sharing only variable text remain distinct. Original
    indices keep refusal pointers aligned with the client's posted `/input/parts/{i}`.
    """
    seen: dict[tuple, tuple[int, ParsedPart]] = {}
    for i, part in enumerate(parts):
        key = (
            part.variable_text.strip(),
            tuple((e.text, e.role_text, e.dcid) for e in (part.entities or ())),
            part.time_text,
            part.unit_text,
            part.source_text,
            tuple((o.verb, o.phrase, o.over, o.target_year) for o in part.operations),
        )
        seen.setdefault(key, (i, part))
    return list(seen.values())


async def confirm_part_pins(
    parts: list[ParsedPart],
    *,
    graph: EngineGraphClient,
    pointer_prefixes: Sequence[str],
) -> list[dict[str, Resolution]]:
    """Confirm pinned dcids and build per-part pre-resolved maps.

    The same text may pin differently across parts. Conflicting pins within one
    part raise `request_invalid`, and an unconfirmed dcid raises
    `invalid_role_ref` (422). Refusals point to the posted field.

    Type confirmation decides validity. Labels load concurrently. `Entity.name` uses
    the graph label or dcid fallback. Label warnings name the exception and pinned
    dcids. Type confirmation remains authoritative. Caller text keys the pin. Graph
    data supplies its label.
    """
    pins_by_part: list[dict[str, str]] = []
    for prefix, part in zip(pointer_prefixes, parts, strict=True):
        part_pins: dict[str, str] = {}
        for entity in part.entities or ():
            if entity.dcid is None:
                continue
            existing = part_pins.get(entity.text)
            if existing is not None and existing != entity.dcid:
                detail = (
                    f"part pins entity {entity.text!r} to two different dcids "
                    f"({existing!r} and {entity.dcid!r}); one text can only name one "
                    "place within a single part"
                )
                raise EngineInputError(
                    detail,
                    code="request_invalid",
                    pointer=f"{prefix}/entities",
                    detail=detail,
                )
            part_pins[entity.text] = entity.dcid
        pins_by_part.append(part_pins)

    all_dcids = list(
        dict.fromkeys(dcid for part_pins in pins_by_part for dcid in part_pins.values())
    )
    if not all_dcids:
        return [{} for _ in parts]

    def _pointer_for(_group: str, absent: list[str]) -> str:
        absent_set = set(absent)
        return next(
            f"{prefix}/entities/{j}/dcid"
            for prefix, part in zip(pointer_prefixes, parts, strict=True)
            for j, entity in enumerate(part.entities or ())
            if entity.dcid in absent_set
        )

    type_result, label_result = await asyncio.gather(
        asyncio.to_thread(
            confirm_dcids_present,
            {"pins": all_dcids},
            read=graph.node_types_batch,
            pointer_for=_pointer_for,
            noun="dcid",
        ),
        asyncio.to_thread(graph.node_labels_batch, all_dcids),
        return_exceptions=True,
    )
    if isinstance(type_result, BaseException):
        raise type_result
    type_by_dcid = type_result
    if isinstance(label_result, BaseException):
        label_by_dcid: Mapping[str, str] = {}
        logger.warning(
            "parsed: label read failed (%s), confirm_part_pins falling back to dcid for %r",
            type(label_result).__name__,
            all_dcids,
        )
    else:
        label_by_dcid = label_result
    return [
        {
            text: PinnedResolved(
                entity=Entity(
                    dcid=dcid,
                    name=label_by_dcid.get(dcid, dcid),
                    kind=entity_kind(type_by_dcid[dcid]),
                    graph_type=type_by_dcid[dcid],
                )
            )
            for text, dcid in part_pins.items()
        }
        for part_pins in pins_by_part
    ]


async def per_part_pre_resolved(
    parts: list[ParsedPart],
    pins: list[dict[str, Resolution]],
    *,
    resolver: EntityResolver,
    deadline: float,
) -> list[dict[str, Resolution]]:
    """Batch-resolve texts unpinned in any part, then overlay each part's pins.

    Text pinned everywhere is excluded. Both parsed-entry doors use this map.
    """
    unpinned_texts = list(
        dict.fromkeys(
            entity.text
            for part, part_pins in zip(parts, pins, strict=True)
            for entity in (part.entities or ())
            if entity.text not in part_pins
        )
    )
    shared = await _resolve_entities_bounded(unpinned_texts, resolver=resolver, deadline=deadline)
    return [{**shared, **part_pins} for part_pins in pins]


async def handle(
    inp: ParsedInput,
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
) -> AsyncIterator[Event]:
    parts_with_indices = _dedupe_parts_with_indices(inp.parts)
    original_indices = [i for i, _ in parts_with_indices]
    parts = [p for _, p in parts_with_indices]

    # Confirm pins before fan-out, preserving original indices for refusal pointers.
    pins = await confirm_part_pins(
        parts, graph=graph, pointer_prefixes=[f"/input/parts/{i}" for i in original_indices]
    )

    interpretation = await interpret_parts(parts, llm=llm, deadline=params.deadline)
    variable_texts = interpretation.variable_texts
    all_dates = [d for dates in interpretation.dates for d in dates]

    entities_flat = list(dict.fromkeys(e for lst in interpretation.entities for e in lst))
    echo = QueryEcho(
        entry_path="parsed",
        raw_query=inp.raw_query,
        normalized_query=None,
        extract_skipped=True,
    )
    yield InterpretationEvent(
        variables=variable_texts,
        entities=entities_flat,
        dates=all_dates,
        expected_results=len(variable_texts),
    )

    extract_step = make_pipeline_step("extract", ran=False)

    if len(variable_texts) == 1:
        out = _FanOutResult()
        async for ev in _run_variables_streaming(
            variable_texts,
            entities=interpretation.entities[0],
            date_request=interpretation.date_requests[0],
            query="",
            pac=params.pac,
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
            base_steps=[extract_step],
            base_timing={},
            pre_resolved=pins[0] or None,
            deadline=params.deadline,
            degrade_on_error=False,
            use_semaphore=False,
            out=out,
            include_option_coverage=params.include_option_coverage,
            stream_slot_options=params.stream_option_coverage,
            include_observations=params.include_observations,
            stream_observations=params.stream_observations,
            per_variable_role_queries=interpretation.role_queries,
            per_variable_entity_roles=interpretation.entity_roles,
            per_variable_unit_texts=interpretation.unit_texts,
            per_variable_source_texts=interpretation.source_texts,
            per_variable_operations=interpretation.operations,
            per_variable_qualifiers=interpretation.qualifiers,
            coverage=params.coverage,
        ):
            yield ev
        region = interpretation.interpreted(out.regions[0], 0)
        total_usage = _sum_usage(interpretation.mention_usage, region.llm_usage)
        response = assemble_region(
            region,
            query="",
            extra_warnings=[],
            start_ms=start_ms,
            engine_build=build_commit(),
            include_sentence=params.include_sentence,
            max_candidates=params.max_candidates,
            include_observations=params.include_observations,
            include_skills=params.include_skills,
            skills_index=params.skills_index,
            llm_usage=total_usage,
            echo=echo,
        )
        yield ResponseEvent(response=response)
        slot_options_warnings: list[Warning] = []
        late_departures: list[LateDepartures] = []
        async for ev in _stream_slot_options(
            [region],
            deadline=params.deadline,
            warnings_out=slot_options_warnings,
            late_departures_out=late_departures,
            response=response,
            answered=[region],
        ):
            yield ev
        yield DoneEvent(
            elapsed_ms=now_ms() - start_ms,
            timed_out=out.timed_out,
            trace_id=get_trace_id(),
            telemetry=DoneTelemetry(llm_usage=total_usage),
            warnings=slot_options_warnings,
            late_departures=late_departures,
        )
        return

    # Each leg receives its own overrides. Unpinned names share one batch,
    # then each part's pins overlay the shared results.
    per_variable_pre_resolved = await per_part_pre_resolved(
        parts, pins, resolver=resolver, deadline=params.deadline
    )

    out = _FanOutResult()
    async for ev in _run_variables_streaming(
        variable_texts,
        entities=[],
        date_request=None,
        query="",
        pac=params.pac,
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
        base_steps=[extract_step],
        base_timing={},
        pre_resolved=None,
        deadline=params.deadline,
        degrade_on_error=True,
        use_semaphore=True,
        out=out,
        include_option_coverage=params.include_option_coverage,
        stream_slot_options=params.stream_option_coverage,
        include_observations=params.include_observations,
        stream_observations=params.stream_observations,
        per_variable_entities=interpretation.entities,
        per_variable_date_requests=interpretation.date_requests,
        per_variable_role_queries=interpretation.role_queries,
        per_variable_entity_roles=interpretation.entity_roles,
        per_variable_unit_texts=interpretation.unit_texts,
        per_variable_source_texts=interpretation.source_texts,
        per_variable_pre_resolved=per_variable_pre_resolved,
        per_variable_operations=interpretation.operations,
        per_variable_qualifiers=interpretation.qualifiers,
        coverage=params.coverage,
    ):
        yield ev

    regions = [interpretation.interpreted(region, i) for i, region in enumerate(out.regions)]

    async with aclosing(
        _stream_combined_regions(
            regions,
            initial_llm_usage=interpretation.mention_usage,
            query="",
            extra_warnings=[],
            echo=echo,
            timed_out=out.timed_out,
            graph=graph,
            start_ms=start_ms,
            params=params,
        )
    ) as events:
        async for event in events:
            yield event
    return
