from __future__ import annotations

from pydantic import Field

from dfo_platform import WarningSeverity

from .base import StrictModel
from .vocab import EntryPath

# ---------------------------------------------------------------------------
# QueryEcho, Warning, Timing, Diagnostics
# ---------------------------------------------------------------------------


class QueryEcho(StrictModel):
    """What the response was understood to be answering."""

    entry_path: EntryPath = Field(description="which input was used")
    raw_query: str | None = Field(
        default=None, description="the original query text, when one was supplied"
    )
    normalized_query: str | None = Field(
        default=None,
        description="normalised query text, when extraction ran",
    )
    extract_skipped: bool = Field(
        description="true when the extraction step was skipped for this request"
    )


def _is_none(value: object) -> bool:
    return value is None


class Warning(StrictModel):
    """A run-health signal. Note: this name shadows the Python builtin Warning.

    Only `observations_series_truncated`, `elected_shape_no_data`, `un_system_unavailable`
    and `whole_total_available` set `params`. A `None` `params` is left out of every dump, so other codes' warnings stay
    exactly `{code, severity, message}` plus `spec_id` where one applies.
    """

    code: str = Field(
        description=(
            "stable machine code from the platform registry, lower snake_case, "
            "e.g. date_coverage_partial, binding_folded"
        )
    )
    severity: WarningSeverity = Field(description="the severity level")
    message: str = Field(description="human-readable explanation of the warning")
    spec_id: str | None = Field(
        default=None,
        description=(
            "the spec_id this warning attaches to, when it concerns one returned "
            "spec specifically; omitted otherwise"
        ),
    )
    params: dict[str, int | str | list[str]] | None = Field(
        default=None,
        exclude_if=_is_none,
        description=(
            "the counts and labels the message states, for a client to compose its own "
            "text. observations_series_truncated: {shown, total, order}, the series shown, "
            "the confirmed series behind them, and how the shown series were chosen: "
            "'ranked' (the most recent and largest) or 'fixed' (a selection made before "
            "values were read, not ranked by value). elected_shape_no_data: {no_data, "
            "elected}, the labels of the readings with no data and of the readings "
            "answered instead. un_system_unavailable: {places, variables, variable_labels}, "
            "the place dcids the UN front door refused, the variables of the UN reads that "
            "failed, each empty when none, and each variable's graph name, in variables "
            "order, '' where the graph holds none; a warning without params says the UN "
            "search itself was "
            "unavailable. whole_total_available: {total, total_label, whole_of}, the dcid "
            "and label of the total the source publishes for the side the answer lists "
            "member by member, and the kinds of party it holds, comma-joined (party, "
            "private, residual). Omitted for every other code"
        ),
    )


class Timing(StrictModel):
    """Latency breakdown, reused at two scopes: the envelope's own
    diagnostics.timing_ms, where total is the whole request's wall-clock
    elapsed time and by_step names the steps that run once per request,
    shared across every part (e.g. extract), and a part's own timing_ms,
    where total sums that part's own by_step values and by_step names only
    the phases that part ran, excluding the shared steps already reported
    on the envelope.
    """

    total: int = Field(description="total latency in milliseconds, at whichever scope this sits")
    by_step: dict[str, int] | None = Field(default=None, description="per-step latency in ms")


class Diagnostics(StrictModel):
    """Run health, warnings, timing, and build stamp."""

    engine_build: str = Field(
        description="the commit sha the image was built from, the same value the "
        "x-dfo-commit response header carries; distinct from schema_version"
    )
    trace_id: str = Field(
        description=(
            "this request's trace_id, shared verbatim with every RFC 9457 "
            "problem-details body and the SSE stream's `done` event"
        )
    )
    warnings: list[Warning] = Field(
        description="degraded-mode, applied-signal, partial-graph signals"
    )
    timing_ms: Timing | None = Field(default=None, description="latency breakdown")
    llm_usage: dict[str, int] | None = Field(
        default=None,
        description="per-request LLM token counts {input_tokens, output_tokens, cached_tokens}",
    )
    timed_out: bool | None = Field(
        default=None,
        description=(
            "True when the pipeline's soft deadline truncated this result to a "
            "partial response; False when it ran to completion. None on the SSE "
            "response event (whose `done` event carries the authoritative flag "
            "instead) -- only resolve_async's buffered ResolveResponse populates "
            "this, since a buffered caller has no `done` event of its own to read."
        ),
    )
