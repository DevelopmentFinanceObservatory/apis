"""Admission checks run before materialisation or network calls.

`bounded_materialise_slot` limits concurrent work and caps waiting at both
the configured wait and client deadline. `sync_estimate` models dispatch
from one coverage snapshot: it groups by request-batch key and window years,
then applies the same lane and StatVar chunk rules as materialisation.

Uncovered StatVars contribute zero priced rows or requests. The estimate counts them separately
and refuses more than 700 with a 422 member-cardinality error. Sync budgets assess covered costs,
so an uncovered request always runs inline by design. A subject-only request that binds no entity
is uncovered and is priced as the same request naming each country would be: zero, with the
materialiser's row ceiling as its bound. Over-budget requests face separate async row and request
ceilings. `ContainedIn` bindings are unbound to avoid a graph read. A range window clips the row
estimate only: requests are priced across all years, as the fetch sends them. `latest` uses all
years for row estimates and preserves `c[TIME_PERIOD]=LATEST` for SDMX request-size estimates. A
member that binds no role prices `CoverageIndex.exhaustive_single_axis_filter`, the entity list
its fetch sends.

Excel estimates include the header row. The writer enforces column and cell
limits after fetch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.timings import phase
from dc_sdmx import ContainedIn
from materialise.errors import OverCeilingError
from materialise.formats import EXCEL_MAX_ROWS, ExcelLimitError
from materialise.generation import (
    _time_period_for_window,
    dispatch_lanes,
    request_batch_key,
    stat_var_chunk_positions,
)
from materialise.pipeline import MaterialisationRequest
from series_coverage.index import BoundRoles, CoverageIndex, Estimate, Role

_STAT_VAR_PREFIX = "dcid:"

# Coverage cannot price uncovered members, so this separate cap bounds their count.
UNCOVERED_MEMBER_CAP = 700


@contextmanager
def bounded_materialise_slot(
    materialise_semaphore: threading.BoundedSemaphore | None,
    *,
    wait_seconds: float,
    deadline: float | None = None,
) -> Iterator[None]:
    """Acquire a slot, waiting up to the smaller of `wait_seconds` and the remaining deadline.

    `deadline` is an absolute monotonic timestamp. A past deadline makes one
    zero-wait acquire attempt. A missing semaphore skips the limit. Failure
    to acquire raises retryable `MintBackpressure` (503). The wait records as
    the `materialise_wait` phase.
    """
    if materialise_semaphore is None:
        yield
        return
    _wait_seconds = float(wait_seconds)
    if deadline is not None:
        _wait_seconds = max(0.0, min(_wait_seconds, deadline - time.monotonic()))
    with phase("materialise_wait"):
        acquired = materialise_semaphore.acquire(timeout=_wait_seconds)
    if not acquired:
        raise MintBackpressure()
    try:
        yield
    finally:
        materialise_semaphore.release()


def _bare_dcid(stat_var: str) -> str:
    return stat_var.removeprefix(_STAT_VAR_PREFIX)


def _bound_roles(request: MaterialisationRequest, *, index: CoverageIndex) -> BoundRoles:
    """Map declared roles to coverage's donor/recipient labels.

    Omit unbound roles and `ContainedIn` filters.
    """
    names = index.role_names.get(_bare_dcid(request.stat_var))
    if names is None:
        return {}
    bound: dict[Role, frozenset[str]] = {}
    for role, declared_role in (("donor", names.donor), ("recipient", names.recipient)):
        refs = request.role_bindings.get(declared_role)
        if refs is None or isinstance(refs, ContainedIn):
            continue
        bound[role] = frozenset(refs)
    return bound


def _window_years(request: MaterialisationRequest) -> tuple[int | None, int | None]:
    window = request.window
    if window.kind == "range":
        return window.start_year, window.end_year
    return None, None


def sync_estimate(
    requests: Sequence[MaterialisationRequest],
    *,
    index: CoverageIndex,
    member_concurrency: int,
) -> Estimate:
    """Estimate covered rows and SDMX requests, plus uncovered members, before network access.

    `uncovered_requests` counts uncovered materialisation requests separately from SDMX calls.
    More than 700 raises `OverCeilingError` (`member_cardinality`, 422)
    before covered costs are computed. `index` must come from the caller's
    coverage snapshot so its estimate and any coverage stamp share one artifact.

    Group by dispatch key and window years, then use the same lanes and
    StatVar chunks as materialisation. Estimate each chunk with `lanes=1`,
    using the first member's roles and window. The `latest` window keeps
    `LATEST` in request-size estimates while coverage prices all years.
    """
    lanes = dispatch_lanes(member_concurrency, len(requests))
    groups: dict[
        tuple[tuple[object, ...], int | None, int | None], list[MaterialisationRequest]
    ] = {}
    uncovered_requests = 0
    for request in requests:
        if index.role_names.get(_bare_dcid(request.stat_var)) is None:
            uncovered_requests += 1
        first_year, last_year = _window_years(request)
        key = (request_batch_key(request), first_year, last_year)
        groups.setdefault(key, []).append(request)
    if uncovered_requests > UNCOVERED_MEMBER_CAP:
        raise OverCeilingError(
            f"{uncovered_requests} member(s) carry no coverage row at all, exceeding "
            f"member_cap={UNCOVERED_MEMBER_CAP}. Coverage has no price to put on an "
            "uncovered member, so this ceiling counts how many of them one request "
            "names, independent of the covered part's own rows/requests estimate. "
            "Narrow the identity (another slot binding, a smaller statvar_set) to "
            "bring the uncovered member count under the cap.",
            ceiling_kind="member_cardinality",
            count=uncovered_requests,
            cap=UNCOVERED_MEMBER_CAP,
        )
    total_rows = 0
    total_requests = 0
    for (_batch_key, first_year, last_year), members in groups.items():
        bound = _bound_roles(members[0], index=index)
        variables = [_bare_dcid(member.stat_var) for member in members]
        time_period = _time_period_for_window(members[0].window)
        for position_group in stat_var_chunk_positions(variables, lanes=lanes):
            chunk_variables = [variables[position] for position in position_group]
            chunk_estimate = index.estimate(
                variables=chunk_variables,
                bound=bound,
                first_year=first_year,
                last_year=last_year,
                lanes=1,
                time_period=time_period,
            )
            total_rows += chunk_estimate.rows
            total_requests += chunk_estimate.requests
    return Estimate(rows=total_rows, requests=total_requests, uncovered_requests=uncovered_requests)


def within_sync_budget(
    estimate: Estimate, *, sync_budget_rows: int, sync_budget_requests: int
) -> bool:
    """Return whether covered rows and requests both fit their limits.

    Only covered costs enter this test. `sync_estimate` caps uncovered
    members separately. Callers submit over-budget work asynchronously and
    run in-budget work inline.
    """
    return estimate.rows <= sync_budget_rows and estimate.requests <= sync_budget_requests


def refuse_excel_over_estimate(estimate: Estimate, *, format: str) -> None:
    """For `xlsx`, raise `ExcelLimitError` (`excel_limit`, 422) before fetch when rows plus
    the header exceed `EXCEL_MAX_ROWS`.

    The writer checks columns and cell lengths after fetch.
    """
    if format != "xlsx" or estimate.rows + 1 <= EXCEL_MAX_ROWS:
        return
    raise ExcelLimitError(
        detail=(
            f"estimated {estimate.rows + 1} row(s) (including the header) exceeds Excel's "
            f"{EXCEL_MAX_ROWS}-row limit; request csv or parquet instead"
        )
    )


def refuse_inline_over_ceiling(estimate: Estimate, *, ceiling: int) -> None:
    """Raise `OverCeilingError` (`rows`, 422) before fetch when the estimate exceeds `ceiling`,
    the row ceiling the inline materialisation runs under.

    A door that runs inline and cannot submit an operation calls this, so a request the
    mid-fetch guard would stop refuses without an SDMX call.
    """
    if estimate.rows <= ceiling:
        return
    raise OverCeilingError(
        f"estimated {estimate.rows} row(s) exceeds the inline materialisation ceiling "
        f"cap={ceiling}. Narrow the identity (another slot binding, a shorter window) to "
        "bring the estimate under the cap.",
        ceiling_kind="rows",
        count=estimate.rows,
        cap=ceiling,
    )


def refuse_preview_over_budget(
    estimate: Estimate, *, sync_budget_rows: int, sync_budget_requests: int
) -> None:
    """Raise `OverCeilingError` (422) when a preview's estimate is over the sync budget.

    A preview reads every row a mint reads and has no asynchronous path, so it runs only where
    a mint would run inline. Rows are checked first, so a request over both reports `sync_rows`.
    """
    if estimate.rows > sync_budget_rows:
        raise OverCeilingError(
            f"estimated {estimate.rows} row(s) exceeds the sync budget cap={sync_budget_rows}. "
            "A preview reads every row a mint reads and runs only inline; mint the request "
            "instead, or narrow the identity to bring the estimate under the cap.",
            ceiling_kind="sync_rows",
            count=estimate.rows,
            cap=sync_budget_rows,
        )
    if estimate.requests > sync_budget_requests:
        raise OverCeilingError(
            f"estimated {estimate.requests} SDMX request(s) exceeds the sync budget "
            f"cap={sync_budget_requests}. A preview reads every row a mint reads and runs only "
            "inline; mint the request instead, or narrow the identity to bring the estimate "
            "under the cap.",
            ceiling_kind="sync_requests",
            count=estimate.requests,
            cap=sync_budget_requests,
        )


def refuse_async_over_ceiling(
    estimate: Estimate, *, async_max_rows: int, async_max_requests: int
) -> None:
    """Raise `OverCeilingError` (422) when async row or request limits are exceeded.

    Rows are checked first, so a request over both reports `operation_rows`.
    """
    if estimate.rows > async_max_rows:
        raise OverCeilingError(
            f"estimated {estimate.rows} row(s) exceeds the async admission ceiling "
            f"cap={async_max_rows}. Narrow the identity (another slot "
            "binding, a shorter window) to bring the estimate under the cap.",
            ceiling_kind="operation_rows",
            count=estimate.rows,
            cap=async_max_rows,
        )
    if estimate.requests > async_max_requests:
        raise OverCeilingError(
            f"estimated {estimate.requests} SDMX request(s) exceeds the async admission "
            f"ceiling cap={async_max_requests}. Narrow the identity (another "
            "slot binding, a shorter window) to bring the estimate under the cap.",
            ceiling_kind="operation_requests",
            count=estimate.requests,
            cap=async_max_requests,
        )


__all__ = [
    "UNCOVERED_MEMBER_CAP",
    "bounded_materialise_slot",
    "refuse_async_over_ceiling",
    "refuse_excel_over_estimate",
    "refuse_inline_over_ceiling",
    "refuse_preview_over_budget",
    "sync_estimate",
    "within_sync_budget",
]
