"""Pure response-envelope assembly, with no LLM or graph calls.

The module has no dependency on the pipeline, so core and conjoin can import it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from dfo_platform.trace import get_trace_id
from search.engine.events import (
    DoneEvent,
    DoneTelemetry,
    Event,
    EventSink,
    ResponseEvent,
    StepEvent,
)
from search.models import (
    Diagnostics,
    EntryPath,
    Part,
    PipelineStep,
    PipelineStepName,
    QueryEcho,
    Resolution,
    ResolveResponse,
    Skill,
    Spec,
    Timing,
    Warning,
)
from search.render.envelope import compose_envelope_sentence
from skills_index import SkillsIndex


def now_ms() -> int:
    """Current monotonic time in milliseconds."""
    return int(time.monotonic() * 1000)


def make_pipeline_step(step: PipelineStepName, ran: bool, ms: int | None = None) -> PipelineStep:
    """Build a PipelineStep record."""
    return PipelineStep(step=step, ran=ran, ms=ms)


async def _record_step(
    pipeline_steps: list[PipelineStep],
    emit: EventSink | None,
    variable_index: int,
    step: PipelineStepName,
    *,
    ran: bool,
    ms: int | None = None,
) -> None:
    """Record a step and emit it when streaming if it ran."""
    ps = make_pipeline_step(step, ran=ran, ms=ms)
    pipeline_steps.append(ps)
    if emit is not None and ran:
        await emit(StepEvent(variable_index=variable_index, step=ps.step, ran=ps.ran, ms=ps.ms))


def make_query_echo(
    query: str,
    extract_skipped: bool,
    *,
    entry_path: EntryPath = "raw_text",
) -> QueryEcho:
    """Build a query echo with its entry path and normalised query text."""
    return QueryEcho(
        entry_path=entry_path,
        raw_query=query,
        normalized_query=query.strip() or None,
        extract_skipped=extract_skipped,
    )


def make_diagnostics(
    engine_build: str,
    warnings: list[Warning],
    timing_by_step: dict[str, int],
    total_ms: int,
    *,
    llm_usage: dict[str, int] | None = None,
) -> Diagnostics:
    """Build diagnostics with request timing, warnings, and optional token usage."""
    return Diagnostics(
        engine_build=engine_build,
        trace_id=get_trace_id(),
        warnings=warnings,
        timing_ms=Timing(total=total_ms, by_step=timing_by_step or None),
        llm_usage=llm_usage,
    )


def iter_response_specs(parts: list[Part]) -> Iterator[Spec]:
    """Yield every response Spec and its equivalent readings, including candidates
    and all matched Specs on no-data parts. Called after region collapse.
    """
    for part in parts:
        if part.status == "definite":
            yield from _spec_and_twins(part.spec)
        elif part.status == "candidates":
            for spec in part.candidates.specs:
                yield from _spec_and_twins(spec)
        elif part.status == "no_data":
            for spec in part.no_data.nearest_real or []:
                yield from _spec_and_twins(spec)
            if part.no_data.flagged_match is not None:
                yield from _spec_and_twins(part.no_data.flagged_match)
            if part.no_data.alternate_match is not None:
                yield from _spec_and_twins(part.no_data.alternate_match)


def _spec_and_twins(spec: Spec) -> Iterator[Spec]:
    yield spec
    yield from spec.resolved.equivalent_readings or []


def _skills_map(
    parts: list[Part], *, include_skills: bool, index: SkillsIndex | None
) -> dict[str, Skill] | None:
    """Return `None` when skills were not requested, `{}` when none are attached,
    or the index entries for all skill ids on response Specs. Missing ids violate
    the shared `SkillsIndex` snapshot invariant and raise `KeyError`.
    """
    if not include_skills:
        return None
    ids: set[str] = set()
    for spec in iter_response_specs(parts):
        if spec.resolved.skills is not None:
            ids.update(spec.resolved.skills)
    if not ids:
        return {}
    assert index is not None, (
        "a Spec carrying resolved.skills requires the same index snapshot to be here"
    )
    skills: dict[str, Skill] = {}
    for skill_id in ids:
        front_matter = index.skills[skill_id].front_matter
        skills[skill_id] = Skill(
            id=front_matter.id,
            name=front_matter.name,
            description=front_matter.description,
            content_version=front_matter.content_version,
        )
    return skills


def make_response(
    echo: QueryEcho,
    diag: Diagnostics,
    parts: list[Part],
    *,
    include_skills: bool = False,
    skills_index: SkillsIndex | None = None,
) -> Resolution:
    """Build a resolution and its composed sentence pair.

    This is the sole engine constructor, keeping the sentence and plan in sync.
    `include_skills` preserves the request option when parts contain no Specs;
    `skills_index` shares the generation used by part builders.
    """
    composed = compose_envelope_sentence(parts)
    rendered_sentence, sentence_plan = composed if composed is not None else (None, None)
    return Resolution(
        query_echo=echo,
        diagnostics=diag,
        parts=parts,
        rendered_sentence=rendered_sentence,
        sentence_plan=sentence_plan,
        skills=_skills_map(parts, include_skills=include_skills, index=skills_index),
    )


def _sum_usage(a: dict | None, b: dict | None) -> dict | None:
    """Sum two LLM usage dicts. Return `None` when both are `None`."""
    if a is None:
        return b
    if b is None:
        return a
    return {
        "input_tokens": a["input_tokens"] + b["input_tokens"],
        "output_tokens": a["output_tokens"] + b["output_tokens"],
        "cached_tokens": a["cached_tokens"] + b["cached_tokens"],
    }


def _terminal_events(
    response: ResolveResponse,
    *,
    start_ms: int,
    timed_out: bool = False,
    llm_usage: dict | None = None,
) -> tuple[Event, Event]:
    """Build terminal response and done events for early-exit paths."""
    return (
        ResponseEvent(response=response),
        DoneEvent(
            elapsed_ms=now_ms() - start_ms,
            timed_out=timed_out,
            trace_id=get_trace_id(),
            telemetry=DoneTelemetry(llm_usage=llm_usage),
        ),
    )
