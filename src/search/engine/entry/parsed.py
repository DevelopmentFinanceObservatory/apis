"""Resolve pre-parsed inputs with per-part variable, entity, time, unit, and source data.

Parsed inputs bypass extraction. A part names or excludes the sources it answers from in
`sources`, which `interpret_parts` compiles into each part's source scope. The scope words of
the user's message (`ParsedInput.user_message`) on a request of one part, or of each part's
own text on a request of several, become the part's explicit scope entries beside the
conversation's carried entries (`ParsedInput.carried_entries`,
`stated_scope.part_scope_intakes`).
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
from origin_catalogue import current_catalogue
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.conjoin import assemble_region
from search.engine.enrichment import _stream_slot_options
from search.engine.entity_checks import part_entities
from search.engine.entity_text import ENTITY_LIKE_PHRASING, detect_entity_like_phrasing
from search.engine.entry.multi import _stream_combined_regions
from search.engine.entry.params import EntryParams
from search.engine.envelope import now_ms
from search.engine.errors import EngineInputError
from search.engine.events import (
    DoneEvent,
    DoneTelemetry,
    Event,
    InterpretationEvent,
    LateDepartures,
)
from search.engine.extract import DateRequest, ExtractedDate, dates_to_request
from search.engine.fanout import (
    _FanOutResult,
    _resolve_entities_bounded,
    _run_variables_streaming,
    answered_provenance,
    restated_sources,
)
from search.engine.interpret import PinnedResolved
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.region.place_options import PlaceOptionsPool, replayed_place_options
from search.engine.region.refs import confirm_dcids_present
from search.engine.region.result import RegionResult, with_entity_words
from search.engine.region.sentence_events import answer_events
from search.engine.response_cache_flow import CacheScope
from search.engine.retrieval.read_plan import current_year
from search.engine.role_text import (
    ROLE_TEXT_NOT_RECOGNISED,
    DoorRole,
    parsed_role_query,
    role_axis_for_text,
)
from search.engine.source_code import PartCode, compile_part_code, without_code_words
from search.engine.source_scope import (
    SourceRefusal,
    SourceScope,
    compile_part_sources,
    without_source_words,
)
from search.engine.stated_scope import part_scope_intakes
from search.engine.time_bind import parse_time_text, time_window_defaulted_warning
from search.models import Operation, ParsedInput, ParsedPart, QueryEcho, RoleAxis, Warning

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
    return parsed_role_query(part), warnings


def _entity_roles_for_part(part: ParsedPart) -> dict[str, list[DoorRole]]:
    """Map each entity's identity (`ParsedEntity.identity`) to its declared axes, each with
    the entity's text as its verbatim span, for one part.

    Preserve distinct mentions in order, including a repeated name on opposite
    axes. Missing or unrecognised role text leaves the entity undirected.
    """
    roles: dict[str, list[DoorRole]] = {}
    for entity in part.entities or ():
        axis = role_axis_for_text(entity.role_text)
        if axis is None:
            continue
        occurrences = roles.setdefault(entity.identity, [])
        role = DoorRole(axis=axis, span=entity.text)
        if role not in occurrences:
            occurrences.append(role)
    return roles


def _open_axes_for_part(part: ParsedPart, *, pointer_prefix: str) -> frozenset[RoleAxis]:
    """Map the part's open role words to flow axes, as `role_text` maps an entity's.

    An unrecognised word, or an axis an entity of the part binds through its `role_text`,
    raises `request_invalid` at the word's pointer. An axis a role-text-less entity comes to
    bind is settled at place binding, where the bound role wins (`PlaceRoles.open_axes`).
    """
    bound = {role_axis_for_text(entity.role_text) for entity in part.entities or ()}
    axes: set[RoleAxis] = set()
    for j, word in enumerate(part.open_roles or ()):
        axis = role_axis_for_text(word)
        if axis is None:
            detail = (
                f"open_roles word {word!r} names no recognised role; use a role_text word "
                "such as creditor, donor, recipient, debtor, origin or destination"
            )
        elif axis in bound:
            detail = (
                f"open_roles word {word!r} asks for every {axis} member, but an entity of "
                "this part binds that role through role_text"
            )
        else:
            axes.add(axis)
            continue
        raise EngineInputError(
            detail,
            code="request_invalid",
            pointer=f"{pointer_prefix}/open_roles/{j}",
        )
    return frozenset(axes)


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
class PartsInterpretation:
    """What a fan-out leg is handed for each posted part, in posted order.

    A part's entities are their identities (`ParsedEntity.identity`). Its
    `entity_surfaces` map each unpinned entity's identity to its text where the text names
    it in other words (`entity_checks.part_entities`), the words its resolution reads beside
    the name (`surface_resolution.resolve_names`). Its `entity_words` map every identity
    that is not its row's text to that text (`entity_checks.part_entities`), the words bind masks
    and its refusals quote (`RegionResult.entity_words`).
    """

    variable_texts: list[str]
    entities: list[list[str]]
    entity_surfaces: list[dict[str, str]]
    entity_words: list[dict[str, str]]
    dates: list[list[ExtractedDate]]
    date_requests: list[DateRequest | None]
    role_queries: list[str]
    entity_roles: list[dict[str, list[DoorRole]]]
    member_entities: list[frozenset[str]]
    unit_texts: list[str | None]
    operations: list[tuple[Operation, ...]]
    open_axes: list[frozenset[RoleAxis]]
    separate: list[bool]
    source_scopes: list[SourceScope | SourceRefusal | None]
    source_codes: list[PartCode | None]
    warnings: list[list[Warning]]
    restatements: list[ParsedPart | None]

    def interpreted(self, region: RegionResult, position: int) -> RegionResult:
        """`region`, the fan-out leg of part `position`, carrying that part's warnings
        after its own, the words of each entity (`with_entity_words`), and its
        restatement carrying the part's source mentions with their resolved references, or
        for a part naming none the dataset the leg's answer binds
        (`fanout.restated_sources`)."""
        restatement = self.restatements[position]
        if restatement is not None:
            restatement = restatement.model_copy(
                update={
                    "sources": restated_sources(
                        self.source_scopes[position], answered_provenance(region)
                    )
                }
            )
        return dataclasses.replace(
            with_entity_words(
                region, self.entity_words[position], surfaces=self.entity_surfaces[position]
            ),
            warnings=tuple(region.warnings) + tuple(self.warnings[position]),
            restatement=restatement,
            parsed_form=restatement,
        )


def interpret_parts(
    parts: list[ParsedPart], *, pointer_prefixes: Sequence[str]
) -> PartsInterpretation:
    """Interpret posted parts for the parsed door and for `dataset_resubmit`'s parsed
    entries, so identical parts carry the same inputs and warnings on either door.

    Only absent time text gets `time_window_defaulted`; unrecognised text gets the
    parser's own warning. An open role word `_open_axes_for_part` refuses, an entity
    `part_entities` refuses, and a source mention `compile_part_sources` refuses, point
    into its part's entry of `pointer_prefixes`. Each part's
    `source_scopes` entry is its compiled sources against the request's catalogue:
    `None` when it names none, a `SourceRefusal` when its words resolve to no source or
    to several. Each part's `source_codes` entry is its series code compiled against its
    scope (`source_code.compile_part_code`): `None` when it names none, the StatVars the code
    names, or a `SourceCodeRefusal`. A part's variable text leaves out the words of each
    source its scope resolved (`source_scope.without_source_words`) and of its code
    (`source_code.without_code_words`), and its restatement states the rest.
    """
    source_scopes = [
        compile_part_sources(p, catalogue=current_catalogue(), pointer_prefix=prefix)
        if p.sources
        else None
        for p, prefix in zip(parts, pointer_prefixes, strict=True)
    ]
    open_axes = [
        _open_axes_for_part(p, pointer_prefix=prefix)
        for p, prefix in zip(parts, pointer_prefixes, strict=True)
    ]
    entities = [
        part_entities(p, pointer_prefix=prefix)
        for p, prefix in zip(parts, pointer_prefixes, strict=True)
    ]
    # A code compiles within the sources the part allows. A part whose sources refuse
    # refuses on them first.
    source_codes: list[PartCode | None] = [
        compile_part_code(
            p.source_code,
            scope=scope if isinstance(scope, SourceScope) else None,
            index=get_index(),
        )
        if p.source_code is not None
        else None
        for p, scope in zip(parts, source_scopes, strict=True)
    ]
    # Words a mention reads as a source, or that write the part's code, are not the
    # measure's too.
    variable_texts = []
    for p, scope in zip(parts, source_scopes, strict=True):
        text = p.variable_text.strip()
        if isinstance(scope, SourceScope):
            text = without_source_words(text, scope)
        if p.source_code is not None:
            text = without_code_words(text, p.source_code)
        variable_texts.append(text)
    dates_and_warnings = [_time_for_part(p) for p in parts]
    role_queries_and_warnings = [_role_query_for_part(p) for p in parts]
    return PartsInterpretation(
        variable_texts=variable_texts,
        entities=[[e.identity for e in (p.entities or [])] for p in parts],
        entity_surfaces=[e.surfaces for e in entities],
        entity_words=[e.words for e in entities],
        dates=[dates for dates, _ in dates_and_warnings],
        date_requests=[dates_to_request(dates) for dates, _ in dates_and_warnings],
        role_queries=[rq for rq, _ in role_queries_and_warnings],
        entity_roles=[_entity_roles_for_part(p) for p in parts],
        member_entities=[e.member_entities for e in entities],
        unit_texts=[p.unit_text for p in parts],
        operations=[tuple(p.operations) for p in parts],
        open_axes=open_axes,
        separate=[p.separate for p in parts],
        source_scopes=source_scopes,
        source_codes=source_codes,
        warnings=_per_part_warnings(parts, dates_and_warnings, role_queries_and_warnings),
        restatements=[
            _part_restatement(p, variable_text=v)
            for p, v in zip(parts, variable_texts, strict=True)
        ],
    )


def _part_restatement(part: ParsedPart, *, variable_text: str) -> ParsedPart | None:
    """Rebuild a part from the inputs used by its resolution leg. Its source mentions are
    the posted ones, the deprecated fields folded in; `PartsInterpretation.interpreted`
    restates them with their resolved references once the leg has run.

    Use the stripped text sent to resolution. Return `None` when it is empty,
    since a posted whitespace-only part cannot pass `ParsedPart` validation.
    """
    try:
        return ParsedPart(
            variable_text=variable_text,
            entities=part.entities,
            time_text=part.time_text,
            unit_text=part.unit_text,
            sources=part.sources,
            source_code=part.source_code,
            operations=part.operations,
            open_roles=part.open_roles,
            separate=part.separate,
        )
    except ValidationError:
        return None


def _dedupe_parts_with_indices(parts: list[ParsedPart]) -> list[tuple[int, ParsedPart]]:
    """Deduplicate identical parts in first-seen order, keeping each posted index.

    Equality includes variable text, entities with name, role text, dcid pins and the
    members flag, time, unit, source mentions (the deprecated `provenance` and `source_text`
    folded in), series code, operations, open roles, and `separate`. Parts sharing only variable text
    remain distinct. Original indices keep refusal pointers aligned with the client's
    posted `/input/parts/{i}`.
    """
    seen: dict[tuple, tuple[int, ParsedPart]] = {}
    for i, part in enumerate(parts):
        key = (
            part.variable_text.strip(),
            tuple((e.text, e.name, e.role_text, e.dcid, e.members) for e in (part.entities or ())),
            part.time_text,
            part.unit_text,
            tuple((m.text, m.ref, m.polarity) for m in part.sources),
            part.source_code,
            tuple((o.verb, o.phrase, o.over, o.target_year) for o in part.operations),
            tuple(part.open_roles or ()),
            part.separate,
        )
        seen.setdefault(key, (i, part))
    return list(seen.values())


async def confirm_part_pins(
    parts: list[ParsedPart],
    *,
    graph: EngineGraphClient,
    resolver: EntityResolver,
    pointer_prefixes: Sequence[str],
) -> list[dict[str, Resolution]]:
    """Confirm pinned dcids and build per-part pre-resolved maps.

    Pins key by the entity's identity (`ParsedEntity.identity`), which for a pinned row is
    its dcid, so rows pinning one dcid are one entity. An unconfirmed dcid raises
    `invalid_role_ref` (422). A pin that asks for the members of a group's own node
    (`EntityResolver.group_node_dcids`) raises `request_invalid` at the entity: a group's
    members come from resolving its name, which a pin skips. Refusals point to the posted
    field.

    Type confirmation decides validity. Labels load concurrently. `Entity.name` uses
    the graph label or dcid fallback. Label warnings name the exception and pinned
    dcids. Type confirmation remains authoritative. Graph data supplies the pin's label.
    """
    pins_by_part = [
        {entity.identity: entity.dcid for entity in part.entities or () if entity.dcid is not None}
        for part in parts
    ]

    members_of_pins = [
        (f"{prefix}/entities/{j}", entity.dcid)
        for prefix, part in zip(pointer_prefixes, parts, strict=True)
        for j, entity in enumerate(part.entities or ())
        if entity.dcid is not None and entity.members
    ]
    if members_of_pins:
        group_nodes = await resolver.group_node_dcids()
        for pointer, dcid in members_of_pins:
            if dcid in group_nodes:
                detail = (
                    f"members cannot be asked of the pinned dcid {dcid!r}, a group's own "
                    "node; post the group's name without a dcid to get its members"
                )
                raise EngineInputError(detail, code="request_invalid", pointer=pointer)

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
            identity: PinnedResolved(
                entity=Entity(
                    dcid=dcid,
                    name=label_by_dcid.get(dcid, dcid),
                    kind=entity_kind((type_by_dcid[dcid],)),
                    graph_type=type_by_dcid[dcid],
                )
            )
            for identity, dcid in part_pins.items()
        }
        for part_pins in pins_by_part
    ]


async def per_part_pre_resolved(
    parts: list[ParsedPart],
    pins: list[dict[str, Resolution]],
    *,
    surfaces: list[dict[str, str]],
    resolver: EntityResolver,
    deadline: float,
) -> list[dict[str, Resolution]]:
    """Batch-resolve the entities unpinned in any part, then give each part its own
    entities' resolutions with its pins.

    Each map keys by identity (`ParsedEntity.identity`). An entity resolves through its
    part's surface for it (`PartsInterpretation.entity_surfaces`,
    `surface_resolution.resolve_names`), so an identity two parts post under two surfaces
    resolves once per surface, each in its own resolver call; every other request makes
    one.
    An entity pinned everywhere is excluded. Both parsed-entry doors use this map.
    """
    unpinned = list(
        dict.fromkeys(
            (entity.identity, part_surfaces.get(entity.identity))
            for part, part_pins, part_surfaces in zip(parts, pins, surfaces, strict=True)
            for entity in (part.entities or ())
            if entity.identity not in part_pins
        )
    )
    # `resolve_names` reads one surface per identity, so each batch holds an identity once.
    batches: list[dict[str, str | None]] = []
    for identity, surface in unpinned:
        batch = next((b for b in batches if identity not in b), None)
        if batch is None:
            batch = {}
            batches.append(batch)
        batch[identity] = surface
    resolved = await asyncio.gather(
        *(
            _resolve_entities_bounded(
                list(batch),
                resolver=resolver,
                deadline=deadline,
                surfaces={
                    identity: surface for identity, surface in batch.items() if surface is not None
                },
            )
            for batch in batches
        )
    )
    by_pair = {
        (identity, batch[identity]): resolution
        for batch, resolutions in zip(batches, resolved, strict=True)
        for identity, resolution in resolutions.items()
    }
    return [
        {
            **{
                entity.identity: by_pair[entity.identity, part_surfaces.get(entity.identity)]
                for entity in part.entities or ()
                if entity.identity not in part_pins
            },
            **part_pins,
        }
        for part, part_pins, part_surfaces in zip(parts, pins, surfaces, strict=True)
    ]


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
    scope: CacheScope | None = None,
) -> AsyncIterator[Event]:
    """Answer a parsed request, from `scope`'s layer 2 when it holds the same input.

    A fresh run reports its health to `scope` so the caller stores only a complete
    answer. Direct callers may omit `scope`."""
    if scope is not None:
        cached = await scope.lookup_parsed_input(inp)
        if cached is not None:
            async with aclosing(
                scope.hit_events(
                    cached,
                    start_ms=start_ms,
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

    parts_with_indices = _dedupe_parts_with_indices(inp.parts)
    original_indices = [i for i, _ in parts_with_indices]
    parts = [p for _, p in parts_with_indices]

    # Confirm pins before fan-out, preserving original indices for refusal pointers.
    pointer_prefixes = [f"/input/parts/{i}" for i in original_indices]
    pins = await confirm_part_pins(
        parts, graph=graph, resolver=resolver, pointer_prefixes=pointer_prefixes
    )

    interpretation = interpret_parts(parts, pointer_prefixes=pointer_prefixes)
    variable_texts = interpretation.variable_texts
    scope_intakes = part_scope_intakes(
        message=inp.user_message,
        message_pointer="/input/user_message",
        part_texts=[part.variable_text for part in parts],
        part_pointers=[f"{prefix}/variable_text" for prefix in pointer_prefixes],
        carried=inp.carried_entries or (),
    )
    all_dates = [d for dates in interpretation.dates for d in dates]

    entities_flat = list(
        dict.fromkeys(
            words.get(identity, identity)
            for identities, words in zip(
                interpretation.entities, interpretation.entity_words, strict=True
            )
            for identity in identities
        )
    )
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
            binder=params.binder,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            population_memo=population_memo,
            place_options_pool=place_options_pool,
            base_timing={},
            pre_resolved=pins[0] or None,
            entity_surfaces=interpretation.entity_surfaces[0] or None,
            entity_words=interpretation.entity_words[0] or None,
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
            per_variable_member_entities=interpretation.member_entities,
            per_variable_unit_texts=interpretation.unit_texts,
            per_variable_source_scopes=interpretation.source_scopes,
            per_variable_source_codes=interpretation.source_codes,
            per_variable_operations=interpretation.operations,
            per_variable_open_axes=interpretation.open_axes,
            per_variable_separate=interpretation.separate,
            per_variable_scope_intakes=scope_intakes,
            coverage=params.coverage,
        ):
            yield ev
        region = interpretation.interpreted(out.regions[0], 0)
        if scope is not None:
            scope.note_health(timed_out=out.timed_out, degraded_leg=out.degraded_leg)
        total_usage = region.llm_usage
        response = assemble_region(
            region,
            query="",
            extra_warnings=[],
            start_ms=start_ms,
            engine_build=build_commit(),
            max_candidates=params.max_candidates,
            include_observations=params.include_observations,
            include_skills=params.include_skills,
            skills_index=params.skills_index,
            llm_usage=total_usage,
            echo=echo,
        )
        for event in answer_events(response):
            yield event
        slot_options_warnings: list[Warning] = []
        late_departures: list[LateDepartures] = []
        incomplete: list[str] = []
        async for ev in _stream_slot_options(
            [region],
            deadline=params.deadline,
            warnings_out=slot_options_warnings,
            late_departures_out=late_departures,
            incomplete_out=incomplete,
            response=response,
            answered=[region],
        ):
            yield ev
        if scope is not None:
            scope.note_health(trailing_incomplete=bool(incomplete))
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
        parts,
        pins,
        surfaces=interpretation.entity_surfaces,
        resolver=resolver,
        deadline=params.deadline,
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
        binder=params.binder,
        sdmx=sdmx,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
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
        per_variable_member_entities=interpretation.member_entities,
        per_variable_entity_surfaces=interpretation.entity_surfaces,
        per_variable_entity_words=interpretation.entity_words,
        per_variable_unit_texts=interpretation.unit_texts,
        per_variable_source_scopes=interpretation.source_scopes,
        per_variable_source_codes=interpretation.source_codes,
        per_variable_pre_resolved=per_variable_pre_resolved,
        per_variable_operations=interpretation.operations,
        per_variable_open_axes=interpretation.open_axes,
        per_variable_separate=interpretation.separate,
        per_variable_scope_intakes=scope_intakes,
        coverage=params.coverage,
    ):
        yield ev

    regions = [interpretation.interpreted(region, i) for i, region in enumerate(out.regions)]

    async with aclosing(
        _stream_combined_regions(
            regions,
            initial_llm_usage=None,
            query="",
            extra_warnings=[],
            echo=echo,
            timed_out=out.timed_out,
            degraded_leg=out.degraded_leg,
            graph=graph,
            start_ms=start_ms,
            params=params,
            scope=scope,
        )
    ) as events:
        async for event in events:
            yield event
    return
