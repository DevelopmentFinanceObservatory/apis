"""Entry point for ResolveInput.kind == "raw_text": the free-text query
door. Runs extract() to pull variables/entities/dates out of the query
before fanning out, or answers a catalogue or declined question straight
after it.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing

from code_index.shared import get_index
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.build import build_commit
from dfo_platform.trace import get_trace_id
from entity_resolution import Admission, EntityResolver, Resolved
from materialise import GraphReader
from materialise.rate_tables import RateTableCache
from origin_catalogue import current_catalogue
from search.engine.assemble import (
    _surviving_qualifier_warnings,
    build_no_data_part,
    skills_index_incomplete_warning,
    skills_index_unavailable_warning,
)
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.conjoin import VARIABLES_CLAMPED, assemble_region
from search.engine.deadline import _await_bounded
from search.engine.departures import (
    WarningKey,
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
from search.engine.extract import (
    DateRequest,
    ExtractedEntity,
    Extraction,
    ExtractionResult,
    _dedupe_stripped,
    dates_to_request,
    extract,
    role_cue_pairing,
)
from search.engine.fanout import (
    _FanOutResult,
    _resolve_entities_bounded,
    _run_variables_streaming,
    leg_runs,
)
from search.engine.options.population_memo import OptionsPopulationMemo
from search.engine.qualifier_match import (
    CodedQualifier,
    _all_normalized_text_ranges,
    _normalize,
    recall_text,
)
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.engine.region.phase_rank_bind import names_value
from search.engine.region.place_options import PlaceOptionsPool, replayed_place_options
from search.engine.region.result import RegionResult, with_departure_facts, with_entity_words
from search.engine.region.sentence_events import answer_events
from search.engine.region.state import qualifier_dropped_warning
from search.engine.response_cache_flow import CacheScope
from search.engine.role_text import DoorRole, debt_axes, receipt_axes
from search.engine.source_code import PartCode, compile_part_code, scan_source_codes
from search.engine.source_scan import scan_sources, source_phrases
from search.engine.source_scope import (
    SourceRefusal,
    SourceScope,
    compile_sources,
    source_dictionary,
)
from search.engine.source_spans import read_sources, unclaimed_words
from search.engine.unit.bind import names_price_basis, names_unit_request
from search.heading.departure_fact import DepartureFact
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

logger = logging.getLogger(__name__)


# Cues that read the direction off the measure: possessive markers ("Ghana's economy")
# and "of" ("debtors of China"). The debt relation in "debt of Ghana" comes from
# `role_text.debt_axes`, which states its axes by construction.
_POSSESSIVE_CUES: frozenset[str] = frozenset({"'s", "'", "\u2019s", "\u2019", "of"})

# Prepositions govern the noun after them, so one after a span belongs to the next phrase
# ("to" in "Ghana to China" directs China, not Ghana).
_PREPOSITIONS: frozenset[str] = frozenset(
    {"to", "from", "by", "of", "on", "for", "in", "into", "through", "towards", "via", "with"}
)


def _entity_key(entity: ExtractedEntity) -> str:
    """Return the canonical entity identity used by bind, roles, and response data."""
    return entity.text.strip()


def _grounded(span: str, query: str) -> bool:
    """Whether `span` occurs in `query`, casefolded as role-span validation reads it."""
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


def _recover_entity_surfaces(
    entities: list[ExtractedEntity], *, query: str, resolver: EntityResolver
) -> list[ExtractedEntity]:
    """Recover a source-qualified entity surface before scans or cache keys read it.

    Only a span that occurs in `query` (`_grounded`) is read; an ungrounded span leaves the
    entity unchanged.
    """
    recovered: list[ExtractedEntity] = []
    for entity in entities:
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


def _without_absorbed(entities: list[ExtractedEntity], *, query: str) -> list[ExtractedEntity]:
    """`entities` without one the query names only inside another's parenthetical span.

    "DAC" in "ODA to Africa (DAC)" is part of the name "Africa (DAC)", never a party of its
    own; "the DAC" named elsewhere in the query stays. A mention is whole words, so "DAC2a"
    names no DAC.
    """
    normalized = _normalize(query)

    def mentions(span: str) -> list[tuple[int, int]]:
        return [
            (start, end)
            for start, end in _all_normalized_text_ranges(span, normalized)
            if not (start > 0 and normalized[start - 1].isalnum())
            and not (end < len(normalized) and normalized[end].isalnum())
        ]

    qualified = [
        qualified_range
        for entity in entities
        if entity.span and "(" in entity.span
        for qualified_range in mentions(entity.span)
    ]

    def absorbed(entity: ExtractedEntity) -> bool:
        if not entity.span or "(" in entity.span:
            return False
        ranges = mentions(entity.span)
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
        if span and _normalize(span) != _normalize(key) and _grounded(span, query):
            surfaces[key] = span
    return surfaces


def _cue_states_direction(
    entity: ExtractedEntity, *, entities: list[ExtractedEntity], query: str
) -> bool:
    """Whether `entity`'s validated cue (`ExtractedEntity.role_span`) states its direction.

    Attachment is the pairing role-span validation accepted (`extract.role_cue_pairing`).
    The cue stands next to the entity, with only an article, list separators or
    coordinated entities between them. A cue before the span states the direction
    ("from the United States", "from Norway and Sweden", "funded by Germany"). A cue
    after it does unless it is a preposition. "funds" in "Norway funds Kenya" states
    Norway's, while "to" in "Ghana to China" and "Zambia debt owed to bondholders"
    belongs to the next phrase. A possessive cue or "of" ("Ghana's economy", "debtors of
    China"), and a national adjective, whose cue is its own span ("French aid"), read the
    direction off the measure, so none states it. An entity with no cue states nothing.
    """
    cue = (entity.role_span or "").strip().casefold()
    if cue in _POSSESSIVE_CUES:
        return False
    pairing = role_cue_pairing(query, entity, entities)
    if pairing is None or not pairing.adjacent:
        return False
    return pairing.cue_first or cue.split()[-1] not in _PREPOSITIONS


def _entity_roles(
    entities: list[ExtractedEntity], *, query: str, measures: Sequence[str]
) -> dict[str, list[DoorRole]]:
    """Map stripped entity names to their flow axes and spans.

    The query's debt wording sets the axes of a two-party debt relation
    (`role_text.debt_axes`), and its receipt wording sets each recipient's
    (`role_text.receipt_axes`); every other place keeps its extracted axis. Preserve
    distinct mentions in order, including self-flows on opposite axes. Collapse duplicate
    mentions with the same axis and span. Store directional roles only, each recording
    whether the query's words give its direction. A debt or receipt axis does by
    construction, and an extracted axis does when its cue does (`_cue_states_direction`). Fall back to entity text for
    the overlap anchor when translation leaves no span.
    """
    places: dict[str, list[str]] = {}
    for entity in entities:
        places.setdefault(_entity_key(entity), []).append(entity.span or entity.text)
    # A place extraction put on both axes has no single extracted axis.
    axes_by_place: dict[str, set[RoleAxis]] = {}
    for entity in entities:
        if entity.flow_role != "unspecified":
            axes_by_place.setdefault(_entity_key(entity), set()).add(entity.flow_role)
    directed = {name: next(iter(axes)) for name, axes in axes_by_place.items() if len(axes) == 1}
    debt = debt_axes(query, measures=measures, places=places, directed=directed)
    worded = receipt_axes(query, places=places) | debt
    roles: dict[str, list[DoorRole]] = {}
    for entity in entities:
        name = _entity_key(entity)
        axis = worded.get(name)
        if axis is None:
            if entity.flow_role == "unspecified":
                continue
            axis = entity.flow_role
        occurrences = roles.setdefault(name, [])
        span = entity.span or entity.text
        stated = name in worded or _cue_states_direction(entity, entities=entities, query=query)
        role = DoorRole(axis=axis, span=span, stated=stated)
        if role not in occurrences:
            occurrences.append(role)
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
    for ev in _terminal_events(response, start_ms=start_ms):
        yield ev


async def _run_extract_phase(
    query: str, *, llm: SupportsLLM, deadline: float
) -> tuple[ExtractionResult, dict | None, int]:
    """Run extraction under the request deadline and return its timing."""
    t0 = now_ms()
    extract_result, extract_usage = await _await_bounded(extract(query, llm=llm), deadline=deadline)
    timing_extract = now_ms() - t0
    return extract_result, extract_usage, timing_extract


async def _no_variables_response(
    query: str,
    *,
    start_ms: int,
    params: EntryParams,
    extract_usage: dict | None,
    timing_extract: int,
) -> AsyncGenerator[Event]:
    """Terminal response when extract() ran but found no variables at all:
    reports variable_not_resolved with the query's own stripped text.
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
        {"extract": timing_extract},
        now_ms() - start_ms,
        llm_usage=extract_usage,
    )
    part = build_no_data_part(
        "variable_not_resolved",
        variable_text=query.strip(),
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
    for ev in _terminal_events(response, start_ms=start_ms, llm_usage=extract_usage):
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
    timing_extract: int,
    scope: CacheScope | None,
) -> AsyncGenerator[Event]:
    """Answer a catalogue, out-of-scope or requires-analysis question straight
    after extract, with no recall, bind or second cache lookup.

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
        [],
        {"extract": timing_extract, "entity_check": timing_entity_check},
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
        response, start_ms=start_ms, timed_out=timed_out, llm_usage=extract_usage
    ):
        yield ev


@dataclasses.dataclass(frozen=True)
class _ExtractionPrep:
    """One query's prepared extraction, shared by events, cache lookup, and fan-out.

    `unique` and `entities` preserve first-seen order. `entities` leaves out an entity
    every occurrence of whose words a source mention covers
    (`source_scan.SourceScan.source_entities`), as do
    `entity_roles`, `entity_words` and `member_entities`. `interpretation_variables`
    adds coded-qualifier sidecars, while resolution and cache keys use bare text.
    `per_variable_operations` runs parallel to `unique`: each variable's operations
    ride on its part, and their phrases reach only that variable's bind, beside
    `dropped_qualifier_phrases`. They raise no warning.
    `per_variable_date_requests` runs parallel to `unique` too: `date_request`, or a
    forecasting variable's own window without its target years.
    `unit_text` is the dropped phrases that bind as one unit request
    (`_split_unit_phrases`); they stay out of `dropped_qualifier_phrases`.
    `per_variable_unit_texts` runs parallel to `unique`: each part's unit request, as the
    parsed door's `ParsedPart.unit_text` is (`_part_unit_text`).
    `per_variable_measure_qualifiers` runs parallel to `unique` too: the words that narrow
    each variable's measure (`ExtractionResult.measure_qualifiers`), less any the request's
    dropped phrases already hold. They reach that variable's part alone
    (`dropped_phrases_of`), which warns for each its answer leaves unapplied (`onto`).
    `qualifier_facts` holds the departure of each `qualifier_warnings` entry whose phrase
    asks for something (`_dropped_phrase_fact`), under its warning's key. `entity_words`
    maps each entity name to the query's own words for it (`ExtractedEntity.span`), first
    mention first. `entity_surfaces` holds those words where they occur in the query and
    differ from the name, which recall resolves the name through (`_entity_surfaces`).
    `member_entities` are the entity names the query asks for as their
    members (`ResolveState.member_entities`). `separate` runs parallel to `unique` and
    marks each variable the raw query compares with another (`source_spans.read_sources`),
    as `ParsedPart.separate` does. `read_connectives` are the offsets of the comparison
    words that marked one. `per_variable_sources` runs parallel to `unique` too: the
    source mentions the query writes for each variable (`source_scan.scan_sources`), and
    `per_variable_source_scopes` what each compiles to, `None` for a variable with none.
    `per_variable_source_codes` runs parallel to `unique` too: the series code the query
    writes for each variable (`source_code.scan_source_codes`), and
    `per_variable_compiled_codes` what each compiles to within its variable's scope, `None`
    for a variable with none.
    """

    date_request: DateRequest | None
    unique: list[str]
    entities: list[str]
    entity_roles: dict[str, list[DoorRole]]
    member_entities: frozenset[str]
    per_variable_qualifiers: list[tuple[CodedQualifier, ...]]
    per_variable_operations: list[tuple[Operation, ...]]
    per_variable_date_requests: list[DateRequest | None]
    interpretation_variables: list[str]
    dropped_qualifier_phrases: list[str]
    per_variable_measure_qualifiers: list[tuple[str, ...]]
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

    def dropped_phrases_of(self, i: int) -> list[str]:
        """The dropped phrases the part of `unique[i]` reads: the request's, then the words
        that narrow its own measure."""
        return [*self.dropped_qualifier_phrases, *self.per_variable_measure_qualifiers[i]]

    def onto(self, region: RegionResult, i: int) -> RegionResult:
        """`region`, the part of `unique[i]`, holding what this extraction knows beyond it:
        the departures raised beside the request-wide qualifier warnings, a
        `query_qualifier_dropped` warning and its departure for each word narrowing its own
        measure that its answer leaves unapplied, and the query's words for each entity.

        The answer applies a narrowing word when the region admitted it as a qualifier
        phrase, or when every Spec it answers with binds a value named by it
        (`_binds_value_named`), such as "concessional" on a part that binds IDS
        Concessional."""
        unapplied = [
            phrase
            for phrase in self.per_variable_measure_qualifiers[i]
            if _normalize(phrase) not in region.admitted_qualifier_phrases
            and not (region.specs and all(_binds_value_named(phrase, s) for s in region.specs))
        ]
        warnings = [qualifier_dropped_warning(phrase) for phrase in unapplied]
        facts = dict(self.qualifier_facts)
        for warning, phrase in zip(warnings, unapplied, strict=True):
            if (fact := _dropped_phrase_fact(phrase)) is not None:
                facts[warning_key(warning)] = fact
        if warnings:
            region = dataclasses.replace(region, warnings=(*region.warnings, *warnings))
        return with_entity_words(with_departure_facts(region, facts), self.entity_words)


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
    )


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


def _prepare_extraction(
    extraction: Extraction, extract_result: ExtractionResult, *, query: str
) -> _ExtractionPrep:
    """Prepare deduped extraction fields, qualifiers, and warnings for `handle`."""
    date_request = dates_to_request(extraction.dates)
    unique: list[str] = _dedupe_stripped(extraction.variables)

    # The sources the query names, each attached to the variables it qualifies. An entity
    # whose every occurrence a source covers is that source, and no entity of the part.
    catalogue = current_catalogue()
    scan = scan_sources(
        query,
        dictionary=source_dictionary(catalogue),
        measures=unique,
        entity_spans=[entity.span or entity.text for entity in extraction.entities],
        index=get_index(),
    )
    per_variable_sources = scan.per_measure
    # The extraction names no series code. A token of the query that writes a held one does.
    per_variable_source_codes = scan_source_codes(query, measures=unique, index=get_index())
    extracted_entities = [
        entity
        for entity in extraction.entities
        if (entity.span or entity.text) not in scan.source_entities
    ]
    entities: list[str] = list(dict.fromkeys(_entity_key(e) for e in extracted_entities))
    entity_roles = _entity_roles(extracted_entities, query=query, measures=unique)

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
    per_variable_operations = [extract_result.operations.get(v, ()) for v in unique]
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
    # warning.
    dropped = list(
        dict.fromkeys(
            qualifier.strip() for qualifier in extraction.dropped_qualifiers if qualifier.strip()
        )
    )
    naming_sources = source_phrases(dropped, [m for ms in per_variable_sources for m in ms])
    unit_text, dropped_qualifier_phrases = _split_unit_phrases(
        [phrase for phrase in dropped if phrase not in naming_sources]
    )
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
        tuple(
            phrase
            for phrase in extract_result.measure_qualifiers.get(v, ())
            if _normalize(phrase) not in request_phrases
        )
        for v in unique
    ]

    entity_words: dict[str, str] = {}
    for entity in extracted_entities:
        entity_words.setdefault(_entity_key(entity), (entity.span or entity.text).strip())
    entity_surfaces = _entity_surfaces(extracted_entities, query=query)

    reading = read_sources(query, interpretation_variables)

    # A raw-text mention has no field of its own, so a refusal points at the query, and
    # mentions that leave a part no source refuse that part alone.
    per_variable_source_scopes = [
        compile_sources(
            list(mentions),
            catalogue=catalogue,
            pointers=["/input/query"] * len(mentions),
            refuse_empty=False,
        )
        if mentions
        else None
        for mentions in per_variable_sources
    ]
    per_variable_compiled_codes: list[PartCode | None] = [
        compile_part_code(
            code, scope=scope if isinstance(scope, SourceScope) else None, index=get_index()
        )
        if code is not None
        else None
        for code, scope in zip(per_variable_source_codes, per_variable_source_scopes, strict=True)
    ]

    return _ExtractionPrep(
        date_request=date_request,
        unique=unique,
        entities=entities,
        entity_roles=entity_roles,
        # One text holds one reading: a mention that asks for the members decides it.
        member_entities=frozenset(_entity_key(e) for e in extracted_entities if e.members),
        per_variable_qualifiers=per_variable_qualifiers,
        per_variable_operations=per_variable_operations,
        per_variable_date_requests=per_variable_date_requests,
        interpretation_variables=interpretation_variables,
        dropped_qualifier_phrases=dropped_qualifier_phrases,
        per_variable_measure_qualifiers=per_variable_measure_qualifiers,
        clamp_warnings=clamp_warnings,
        qualifier_warnings=qualifier_warnings,
        qualifier_facts=qualifier_facts,
        entity_words=entity_words,
        entity_surfaces=entity_surfaces,
        unit_text=unit_text,
        per_variable_unit_texts=[
            _part_unit_text(extract_result.unit_texts.get(v), unit_text) for v in unique
        ],
        separate=list(reading.separate),
        read_connectives=reading.read_connectives,
        per_variable_sources=per_variable_sources,
        per_variable_source_scopes=per_variable_source_scopes,
        per_variable_source_codes=per_variable_source_codes,
        per_variable_compiled_codes=per_variable_compiled_codes,
    )


def _log_unclaimed_words(prep: _ExtractionPrep, extraction: Extraction, *, query: str) -> None:
    """Log, once, the words of `query` that no part, entity, time or unit text claims and
    that are not function words (`source_spans.unclaimed_words`)."""
    claimed = [
        *prep.interpretation_variables,
        *(
            operation.phrase
            for operations in prep.per_variable_operations
            for operation in operations
        ),
        *(
            text
            for entity in extraction.entities
            for text in (entity.text, entity.span, entity.role_span)
            if text
        ),
        *(bound for date in extraction.dates for bound in (date.start, date.end) if bound),
        *([prep.unit_text] if prep.unit_text else []),
        *(text for text in prep.per_variable_unit_texts if text),
        *(m.text for mentions in prep.per_variable_sources for m in mentions if m.text),
        *(code for code in prep.per_variable_source_codes if code),
    ]
    words = unclaimed_words(query, claimed=claimed, read_connectives=prep.read_connectives)
    if words:
        logger.info("raw_query_coverage unclaimed=%s", ",".join(words))


async def _handle_single_variable(
    prep: _ExtractionPrep,
    *,
    query: str,
    timing_extract: int,
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
    async for ev in _run_variables_streaming(
        prep.unique,
        entities=prep.entities,
        entity_roles=prep.entity_roles,
        member_entities=prep.member_entities,
        date_request=prep.date_request,
        query=query,
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
        base_timing={"extract": timing_extract},
        pre_resolved=None,
        entity_surfaces=prep.entity_surfaces,
        deadline=params.deadline,
        degrade_on_error=False,
        use_semaphore=False,
        out=out,
        per_variable_dropped_qualifier_phrases=[
            prep.dropped_phrases_of(i) for i in range(len(prep.unique))
        ],
        query_text=query,
        per_variable_qualifiers=prep.per_variable_qualifiers,
        per_variable_operations=prep.per_variable_operations,
        per_variable_date_requests=prep.per_variable_date_requests,
        per_variable_unit_texts=prep.per_variable_unit_texts,
        per_variable_source_scopes=prep.per_variable_source_scopes,
        per_variable_source_codes=prep.per_variable_compiled_codes,
        unit_text_best_effort=True,
        door="raw_text",
        include_option_coverage=params.include_option_coverage,
        stream_slot_options=params.stream_option_coverage,
        include_observations=params.include_observations,
        stream_observations=params.stream_observations,
        coverage=params.coverage,
    ):
        yield ev
    region = out.regions[0]
    # Record run health. `is_storable` decides cache eligibility.
    if scope is not None:
        scope.note_health(timed_out=out.timed_out, degraded_leg=out.degraded_leg)
    total_usage = _sum_usage(extract_usage, region.llm_usage)
    # The part and its late departures read the same facts and entity words.
    answered = prep.onto(region, 0)
    response = assemble_region(
        answered,
        query=query,
        extra_warnings=prep.clamp_warnings
        + _surviving_qualifier_warnings(
            prep.qualifier_warnings, prep.dropped_qualifier_phrases, [region]
        ),
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
    async for ev in _stream_slot_options(
        [answered],
        deadline=params.deadline,
        warnings_out=slot_options_warnings,
        late_departures_out=late_departures,
        incomplete_out=incomplete,
        response=response,
        answered=[answered],
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


async def _handle_multi_variable(
    prep: _ExtractionPrep,
    *,
    query: str,
    timing_extract: int,
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
    async for ev in _run_variables_streaming(
        prep.unique,
        entities=prep.entities,
        entity_roles=prep.entity_roles,
        member_entities=prep.member_entities,
        date_request=prep.date_request,
        query=query,
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
        base_timing={"extract": timing_extract},
        pre_resolved=pre_resolved,
        deadline=params.deadline,
        degrade_on_error=True,
        use_semaphore=True,
        out=out,
        per_variable_dropped_qualifier_phrases=[
            prep.dropped_phrases_of(i) for i in range(len(prep.unique))
        ],
        per_variable_qualifiers=prep.per_variable_qualifiers,
        per_variable_operations=prep.per_variable_operations,
        per_variable_date_requests=prep.per_variable_date_requests,
        per_variable_unit_texts=prep.per_variable_unit_texts,
        per_variable_source_scopes=prep.per_variable_source_scopes,
        per_variable_source_codes=prep.per_variable_compiled_codes,
        unit_text_best_effort=True,
        per_variable_separate=prep.separate,
        door="raw_text",
        include_option_coverage=params.include_option_coverage,
        stream_slot_options=params.stream_option_coverage,
        include_observations=params.include_observations,
        stream_observations=params.stream_observations,
        coverage=params.coverage,
    ):
        yield ev

    async with aclosing(
        _stream_combined_regions(
            [prep.onto(region, i) for i, region in enumerate(out.regions)],
            initial_llm_usage=extract_usage,
            query=query,
            # Filter before merging. same_shape unions admitted phrases per group.
            extra_warnings=prep.clamp_warnings
            + _surviving_qualifier_warnings(
                prep.qualifier_warnings, prep.dropped_qualifier_phrases, out.regions
            ),
            echo=None,
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

    extract_result, extract_usage, timing_extract = await _run_extract_phase(
        query, llm=llm, deadline=params.deadline
    )
    # Record extraction once before the per-variable fan-out.
    yield StepEvent(variable_index=None, step="extract", ran=True, ms=timing_extract)
    extraction = extract_result.extraction

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
                timing_extract=timing_extract,
                scope=scope,
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
                timing_extract=timing_extract,
            )
        ) as events:
            async for ev in events:
                yield ev
        return

    extraction = extraction.model_copy(
        update={
            "entities": _recover_entity_surfaces(
                extraction.entities,
                query=query,
                resolver=resolver,
            )
        }
    )
    prep = _prepare_extraction(extraction, extract_result, query=query)

    yield InterpretationEvent(
        variables=prep.interpretation_variables,
        entities=prep.entities,
        dates=extraction.dates,
        expected_results=len(prep.unique),
    )

    # Include entities, roles, dates, qualifiers, operations, sources, the unit text peeled
    # from each variable, raw-text signals and the extraction's warnings in the cache key so
    # semantically distinct queries retain separate entries.
    # `query_text` applies only to N=1, matching the single-variable cache readers.
    # A cache hit returns before binding. Core always supplies `scope`.
    # Direct tests may omit it.
    cached = None
    if scope is not None:
        cached = await scope.lookup_parsed(
            query=query,
            variables=prep.unique,
            entities=extraction.entities,
            dates=extraction.dates,
            dropped_qualifiers=extraction.dropped_qualifiers,
            unit_texts=[extract_result.unit_texts.get(v) for v in prep.unique],
            variable_qualifiers=prep.per_variable_qualifiers,
            operations=prep.per_variable_operations,
            entity_roles=prep.entity_roles,
            sources=prep.per_variable_sources,
            source_codes=prep.per_variable_source_codes,
            query_text=query if len(prep.unique) == 1 else None,
            separate=prep.separate,
            extraction_warnings=prep.clamp_warnings,
            measure_qualifiers=prep.per_variable_measure_qualifiers,
            llm_usage=extract_usage,
        )
    _log_unclaimed_words(prep, extraction, query=query)
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
                timing_extract=timing_extract,
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
            timing_extract=timing_extract,
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
