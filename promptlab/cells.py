"""Capture production LLM calls, then replay extract and bind prompts as diagnostic cells.

`run_capture` stores every call once per case. Bind cells score shape election
and over-nomination from captured data. Graph-based alternate recall belongs to rescore.
Extract cells use a graph client and score fresh extraction against the capture.
`GraphCache` serves captured reads and bounds live misses. E2E cells exercise the
composed pipeline.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import itertools
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from typing import Any, get_args

from promptlab.bindmenu import MenuEntry, parse_menu, truncate_menu
from promptlab.compose import compose
from promptlab.expectations import (
    LegExpectation,
    ShapeAssertion,
    case_entities,
    case_stat_vars,
    leg_expectation,
)
from promptlab.llm import CallRecord, LadderLLM
from promptlab.manifest import Rung
from promptlab.scoring import UNRESOLVED_DENOMINATOR, menu_five_tuple
from promptlab.store import BindRecord, CaptureStore, CellStore, ExtractRecord
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import LLM, SupportsLLM, for_stage
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.core import resolve_async
from search.engine.errors import EngineInputError
from search.engine.extract import Extraction, dates_to_request, extract
from search.eval.cases import Case, ShapeKey
from search.eval.runner import build_request
from search.models import ResolveRequest


async def _run_concurrently(coros: Iterable[Coroutine[Any, Any, None]]) -> None:
    """Run tasks in a `TaskGroup`. A failure cancels and awaits unfinished siblings.
    Unwrap a single exception. Preserve a `BaseExceptionGroup` when multiple tasks
    fail."""
    try:
        async with asyncio.TaskGroup() as tg:
            for coro in coros:
                tg.create_task(coro)
    except* BaseException as eg:
        if len(eg.exceptions) == 1:
            raise eg.exceptions[0] from None
        raise


# Resubmitted facet refusals return before LLM calls. Propagate every other input error.
_FACET_REFUSAL_CODES = frozenset({"facet_not_available", "facet_binding_unsupported"})


def _build_request(case: Case) -> ResolveRequest:
    """Build the request the eval runner posts for a case."""
    return ResolveRequest.model_validate(build_request(case))


def _with_seed(llm: SupportsLLM, seed: int) -> SupportsLLM:
    """Clone a real `LLM` with `_seed` set. Unseeded test doubles pass through unchanged."""
    if not isinstance(llm, LLM):
        return llm
    seeded = copy.copy(llm)
    seeded._seed = seed
    return seeded


def _call_payload(call: CallRecord) -> dict[str, Any]:
    """Convert a call record to a payload and rename its actual LLM seed to `llm_seed`.
    `CellStore.append` uses `seed` as its key, so preserving the record's value there
    would overwrite the cell seed, especially for unseeded test doubles."""
    payload = dataclasses.asdict(call)
    payload["llm_seed"] = payload.pop("seed")
    return payload


def _assign_legs(
    bind_calls: Sequence[CallRecord], variable_text: Sequence[str]
) -> list[tuple[int, CallRecord]]:
    """Match each call's `Variable phrase:` to its exact text in `variable_text`.
    Concurrent bind calls finish out of variable order, so sink position cannot
    identify a leg. Raises `ValueError` for an unmatched or duplicate phrase."""
    remaining = list(enumerate(variable_text))
    pairs: list[tuple[int, CallRecord]] = []
    for call in bind_calls:
        phrase, _entries = parse_menu(call.prompt)
        match = next((entry for entry in remaining if entry[1] == phrase), None)
        if match is None:
            raise ValueError(
                f"bind call's variable phrase {phrase!r} does not match any "
                f"unclaimed entry in parts[*].variable_text={list(variable_text)!r}"
            )
        remaining.remove(match)
        pairs.append((match[0], call))
    return pairs


async def run_capture(
    cases: Sequence[Case],
    *,
    out: CaptureStore,
    llm: SupportsLLM,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    concurrency: int,
) -> None:
    """Run the production pipeline once per case and persist extract and bind calls.

    Pass `graph` through to `resolve_async`. It must be non-`None`. A live capture
    can use `GraphCache`, while tests can use `FakeGraph`. `LadderLLM` records every
    call while preserving the shipped prompts.

    Resume skips cases with durable extract records. Otherwise it reruns the full
    case and appends only missing bind records. `CaptureStore.open()` drops
    orphaned binds from interrupted attempts. Only the two resubmitted-facet
    refusal codes are skipped. Other input errors propagate.
    """
    done_extract = out.done_extract_keys()
    done_bind = out.done_bind_keys()
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(case: Case) -> None:
        case_id = case.id
        if case_id in done_extract:
            return
        async with semaphore:
            sink: list[CallRecord] = []
            wrapped = LadderLLM(llm, systems={}, sink=sink)
            request = _build_request(case)
            try:
                resp = await resolve_async(request, graph=graph, llm=wrapped, sdmx=sdmx)
            except EngineInputError as exc:
                if exc.code not in _FACET_REFUSAL_CODES:
                    raise
                return

            # A resolution's parts keep the fan-out's variable order. A catalogue or declined
            # answer has no parts and makes no bind call.
            variable_text: list[str] = (
                [part["variable_text"] for part in resp.model_dump(mode="json")["parts"]]
                if resp.kind == "resolution"
                else []
            )

            extract_calls = [r for r in sink if r.schema == "Extraction"]
            bind_calls = [r for r in sink if r.schema == "_UniversalBindOutput"]
            if len(extract_calls) > 1:
                raise ValueError(
                    f"{case_id}: expected at most one extract call, got {len(extract_calls)}"
                )

            for leg, call in sorted(_assign_legs(bind_calls, variable_text)):
                key = (case_id, leg)
                if key in done_bind:
                    continue
                out.append_bind(
                    BindRecord(
                        case_id=case_id,
                        leg=leg,
                        variable=variable_text[leg],
                        prompt=call.prompt,
                        prompt_sha1=call.prompt_sha1,
                        system_sha1=call.system_sha1,
                        output=call.output,
                        usage=call.usage,
                        elapsed_ms=call.elapsed_ms,
                    )
                )

            if extract_calls:
                call = extract_calls[0]
                out.append_extract(
                    ExtractRecord(
                        case_id=case_id,
                        query=case.query,
                        system_sha1=call.system_sha1,
                        output=call.output,
                        usage=call.usage,
                        elapsed_ms=call.elapsed_ms,
                    )
                )

    await _run_concurrently(_one(case) for case in cases)


def _shape_matches(entry: MenuEntry, shape: ShapeAssertion) -> bool:
    """Whether the menu entry carries every dimension the shape asserts; `None` asserts absence.

    The menu shows an unresolved denominator as a sentinel without its dcid, so any asserted
    denominator matches the sentinel.
    """
    entry_tuple = dict(zip(get_args(ShapeKey), menu_five_tuple(entry), strict=True))
    for key, want in shape.items():
        got = entry_tuple[key]
        if key == "measurement_denominator" and want is not None and got == UNRESOLVED_DENOMINATOR:
            continue
        if got != want:
            return False
    return True


def _shape_election_score(
    output: Mapping[str, Any], menu: Sequence[MenuEntry], expected: LegExpectation | None
) -> float | None:
    """Return 1.0 for a choice matching an expected shape and 0.0 for a wrong or invalid
    choice. Return `None` when the leg has no confirmed shape to choose."""
    if expected is None or not expected.shapes:
        return None
    if output.get("ask"):
        return 0.0
    index = output.get("chosen_shape_index")
    if not isinstance(index, int) or not (0 <= index < len(menu)):
        return 0.0
    return 1.0 if any(_shape_matches(menu[index], shape) for shape in expected.shapes) else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_bind_scores(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elections = [
        r["scores"]["shape_election_match"]
        for r in records
        if "shape_election_match" in r.get("scores", {})
    ]
    over_nominations = [
        r["scores"]["over_nomination_count"]
        for r in records
        if "over_nomination_count" in r.get("scores", {})
    ]
    return {
        "n_calls": len(records),
        "shape_election_match_rate": _mean(elections),
        "shape_election_scored": len(elections),
        "over_nomination_count_mean": _mean(over_nominations),
        "over_nomination_scored": len(over_nominations),
    }


async def run_bind_cell(
    rung: Rung,
    cases: Sequence[Case],
    *,
    capture: CaptureStore,
    out: CellStore,
    llm: SupportsLLM,
    concurrency: int,
) -> None:
    """Replay captured bind records with the rung's prompt, top-k, and seeds. Score
    each against its case. Missing cases raise `ValueError`."""
    capture.require_complete()
    case_by_id = {c.id: c for c in cases}
    bind_system = compose(rung.bind)
    done = out.done_keys()
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(
        record: BindRecord, menu: Sequence[MenuEntry], truncated: str, seed: int
    ) -> None:
        key = (record.case_id, record.leg, seed)
        if key in done:
            return
        async with semaphore:
            seeded = _with_seed(llm, seed)
            sink: list[CallRecord] = []
            wrapped = LadderLLM(seeded, systems={"bind": bind_system}, sink=sink)
            await for_stage(wrapped, "bind").generate_structured(
                prompt=truncated,
                system=bind_system,
                schema=_UniversalBindOutput,
            )
            call = sink[0]

            case = case_by_id.get(record.case_id)
            if case is None:
                raise ValueError(f"no case supplied for captured case {record.case_id!r}")
            leg_expected = leg_expectation(case, record.leg)
            scores: dict[str, Any] = {}
            election = _shape_election_score(call.output, menu, leg_expected)
            if election is not None:
                scores["shape_election_match"] = election
            if leg_expected is not None and leg_expected.sole_status == "definite":
                scores["over_nomination_count"] = len(call.output.get("also_plausible") or [])

            out.append(
                case_id=record.case_id,
                leg=record.leg,
                seed=seed,
                call={**_call_payload(call), "scores": scores},
            )

    coros = []
    for record in capture.bind_records():
        truncated = truncate_menu(record.prompt, rung.run.top_k)
        _variable, menu = parse_menu(truncated)
        coros.extend(_one(record, menu, truncated, seed) for seed in rung.run.seeds)
    await _run_concurrently(coros)

    out.finish(_aggregate_bind_scores(out.call_records()))


_PLACE_DCID_ARC_LADDER: tuple[str, ...] = (
    "<-description{typeOf:Country}->dcid",
    "<-description{typeOf:Continent}->dcid",
    "<-description{typeOf:UNGeoRegion}->dcid",
    "<-description->dcid",
)


def _resolve_place_dcid(name: str, graph: EngineGraphClient) -> str | None:
    """Resolve a place name through Country, Continent, UNGeoRegion, then bare
    description arcs. This diagnostic fallback approximates the engine's resolution;
    `interpretation_match` owns axis-aware precedence."""
    for arc in _PLACE_DCID_ARC_LADDER:
        dcids = graph.resolve_by_arc([name], arc=arc).get(name, [])
        if dcids:
            return dcids[0]
    return None


def _variable_split_matches(fresh: Sequence[str], reference: Sequence[str]) -> bool:
    """Pair phrases by maximum total shared-token overlap, independent of list order.
    Every pair must also have Jaccard similarity above 0.5. Exact wording may
    change across rungs. The score measures whether the same variables remain
    split across wording changes. Exhaustive permutations are
    cheap because each query has only a handful of variables."""
    if len(fresh) != len(reference):
        return False
    if not fresh:
        return True
    fresh_tokens = [set(phrase.lower().split()) for phrase in fresh]
    ref_tokens = [set(phrase.lower().split()) for phrase in reference]

    def overlap(fresh_index: int, ref_index: int) -> int:
        return len(fresh_tokens[fresh_index] & ref_tokens[ref_index])

    best_total = -1
    best_perm: tuple[int, ...] = ()
    for perm in itertools.permutations(range(len(fresh_tokens))):
        total = sum(overlap(fresh_index, ref_index) for ref_index, fresh_index in enumerate(perm))
        if total > best_total:
            best_total = total
            best_perm = perm

    for ref_index, fresh_index in enumerate(best_perm):
        union = fresh_tokens[fresh_index] | ref_tokens[ref_index]
        jaccard = overlap(fresh_index, ref_index) / len(union) if union else 1.0
        if jaccard <= 0.5:
            return False
    return True


def _score_extract(
    extraction: Extraction,
    *,
    reference: Extraction,
    case: Case,
    graph: EngineGraphClient,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}

    scores["date_structure_match"] = dates_to_request(extraction.dates) == dates_to_request(
        reference.dates
    )

    expected_entities = case_entities(case)
    resolved_entities = {
        dcid
        for name in (e.text for e in extraction.entities)
        for dcid in [_resolve_place_dcid(name, graph)]
        if dcid is not None
    }
    if expected_entities or extraction.entities:
        scores["entity_dcid_match"] = resolved_entities == expected_entities

    expected_stat_vars = case_stat_vars(case)
    if expected_stat_vars:
        detected: set[str] = set()
        for phrase in extraction.variables:
            detected.update(c.dcid for c in graph.detect_svs(phrase).candidates)
        scores["shape_recall"] = len(expected_stat_vars & detected) / len(expected_stat_vars)

    scores["variable_split_match"] = _variable_split_matches(
        extraction.variables, reference.variables
    )

    return scores


def _rate(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [r["scores"][key] for r in records if key in r.get("scores", {})]
    return _mean([1.0 if v else 0.0 for v in values])


def _aggregate_extract_scores(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shape_recalls = [
        r["scores"]["shape_recall"] for r in records if "shape_recall" in r.get("scores", {})
    ]
    return {
        "n_calls": len(records),
        "date_structure_match_rate": _rate(records, "date_structure_match"),
        "entity_dcid_match_rate": _rate(records, "entity_dcid_match"),
        "shape_recall_mean": _mean(shape_recalls),
        "shape_recall_scored": len(shape_recalls),
        "variable_split_match_rate": _rate(records, "variable_split_match"),
    }


async def run_extract_cell(
    rung: Rung,
    cases: Sequence[Case],
    *,
    capture: CaptureStore,
    graph: EngineGraphClient,
    out: CellStore,
    llm: SupportsLLM,
    concurrency: int,
) -> None:
    """Re-extract captured queries with the rung's prompt and seeds. Score shape
    recall, resolved entities, dates, and the variable split against the capture.
    Replace `[[TODAY]]` with the capture's date so extraction matches its bind menus.
    Store each single extract call under `leg=0`."""
    capture.require_complete()
    case_by_id = {c.id: c for c in cases}
    extract_system = compose(rung.extract).replace("[[TODAY]]", capture.manifest.substituted_date)
    done = out.done_keys()
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(record: ExtractRecord, reference: Extraction, seed: int) -> None:
        key = (record.case_id, 0, seed)
        if key in done:
            return
        case = case_by_id.get(record.case_id)
        if case is None:
            raise ValueError(f"no case supplied for captured case {record.case_id!r}")
        async with semaphore:
            seeded = _with_seed(llm, seed)
            sink: list[CallRecord] = []
            wrapped = LadderLLM(seeded, systems={"extract": extract_system}, sink=sink)
            extract_result, _usage = await extract(record.query, llm=wrapped)
            call = sink[0]

            scores = await asyncio.to_thread(
                _score_extract,
                extract_result.extraction,
                reference=reference,
                case=case,
                graph=graph,
            )
            out.append(
                case_id=record.case_id,
                leg=0,
                seed=seed,
                call={**_call_payload(call), "scores": scores},
            )

    coros = []
    for record in capture.extract_records():
        reference = Extraction.model_validate(record.output)
        coros.extend(_one(record, reference, seed) for seed in rung.run.seeds)
    await _run_concurrently(coros)

    out.finish(_aggregate_extract_scores(out.call_records()))
