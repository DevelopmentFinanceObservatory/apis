"""Shared `no_data` result builder for the standard-promote phases."""

from __future__ import annotations

from dfo_platform import Cause
from search.engine.region.result import RegionResult
from search.heading.departure_fact import DepartureFact
from search.models import GraphRef, NoDataGate, Spec, Warning


def _no_data_region(
    variable_text: str,
    reason: str = "variable_not_resolved",
    *,
    warnings: tuple[Warning, ...] = (),
    flagged_match: tuple[Spec, ...] | None = None,
    gate: NoDataGate | None = None,
    alternate_match: Spec | None = None,
    elected_stat_vars: list[GraphRef] | None = None,
    window_gap_fact: DepartureFact | None = None,
    cause: Cause | None = None,
) -> RegionResult:
    """Build a `no_data` result for an unresolved identity or slice.

    The default reason is `variable_not_resolved`; refine failures use `no_matching_slice`.
    Pass accumulated warnings through unchanged. `no_observations` may carry a flagged spec
    and gate so the client sees the requested identity marked unavailable. Pass elected
    StatVars after materialisation; `_elected_stat_vars` returns `None` for other outcomes.
    `window_gap_fact` is the years asked that hold no point (`RegionResult.window_gap_fact`),
    and `cause` says why the reason fired (`RegionResult.cause`).

    Resubmits omit `nearest_real` because their definitions come from the client."""
    return RegionResult(
        variable_text=variable_text,
        status="no_data",
        specs=(),
        no_data_reason=reason,
        warnings=warnings,
        timing_by_step={},
        flagged_match=flagged_match,
        gate=gate,
        alternate_match=alternate_match,
        elected_stat_vars=elected_stat_vars,
        window_gap_fact=window_gap_fact,
        cause=cause,
    )
