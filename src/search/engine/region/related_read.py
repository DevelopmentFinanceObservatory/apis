"""Start a definite answer's related-measures reads (`related.read_related`) beside its own
reads, and state their result on its Spec.

A read starts once materialisation names the elected shape's members. On the single-shape
path it starts as grounding does (`phase_answer._ground_elected_answer`), on the candidates
path before candidates are grounded (`candidate_collapse.assemble_candidates`), and on the
resubmit door as grounding starts (`promote_ground._phase_ground_build`). On the first two
it runs in a task, which the definite answer takes when it reads the same StatVar, starting
its own otherwise (`related_read_for`). The resubmit door runs synchronously, so its read
runs on a thread of its own and the answer joins it as it finalises (`submit_related_read`).
`read_runner` starts both kinds off the calling thread.

The pass that starts a read owns it: however the pass ends, it waits for every read it
started before it returns (`settle_related_reads` in `pipeline._run_phases`,
`settle_pending_related` in `resubmit_promote._resolve_standard_promote`), so no read outlives
the answer. A cancelled pass cancels its tasks instead, since waiting would hold the
cancellation.

The read runs concurrently with the answer's remaining reads, so it can delay the answer by
at most its deadline.

A read qualifies when the answer serves exactly one StatVar, has no measure family and names
places with no roles. It ends at `SEARCH_RELATED_MEASURES_TIMEOUT_S` or the request
deadline, whichever comes first. Past it, or on a failed read, the Spec states none, the
reason is logged (`_log_unavailable`), and the answer that joins the read notes it lost
(`degraded_reads.note_lost_read`), so no cache stores the answer. A read that ran past
`SEARCH_RELATED_MEASURES_TIMEOUT_S` while the request had time left is noted as cut. It was
healthy but slow, so the answer carries no warning, and no cache stores it.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from typing import Any

from code_index.shared import get_index, get_measure_index
from dfo_platform.executor import submit_with_context
from search.engine.clients.graph import EngineGraphClient
from search.engine.config import settings
from search.engine.degraded_reads import ReadBudget, note_lost_read
from search.engine.errors import GraphInfraError
from search.engine.region.grounding import requested_window_from_date_request
from search.engine.region.state import ResolveState
from search.engine.related import Related, read_related, related_measures
from search.engine.retrieval.bound_values import (
    MeasurePartition,
    entity_role_keys,
    observed_measure_family,
    shape_query,
)
from search.engine.retrieval.retrieve import Materialised
from search.models import EntityRoleKey, RequestedWindow, Spec

logger = logging.getLogger(__name__)

# Tests join the threads of `ReadRunner.submit`'s pools by this name prefix.
RELATED_THREAD_PREFIX = "related"


class ReadRunner:
    """Starts related reads off the calling thread."""

    def start[R](self, read: Callable[[], R]) -> Awaitable[R]:
        """`read` on the event loop's default executor, for a task to await."""
        return asyncio.to_thread(read)

    def submit[R](self, read: Callable[[], R]) -> Future[R]:
        """Start `read` on a one-worker pool of its own, under a copy of the caller's context.
        The thread ends with the read, which its deadline bounds, and the pass that started
        it waits for the future (`settle_pending_related`), so nothing waits for the pool."""
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=RELATED_THREAD_PREFIX)
        future = submit_with_context(pool, read)
        pool.shutdown(wait=False)
        return future


# Starts every related read. The offline engine tests replace it with one that runs each read
# before returning (`tests/engine/conftest.py`).
read_runner = ReadRunner()


@dataclass(frozen=True, slots=True)
class LostRelated:
    """The exception that ended a related read, and whether the read's own budget cut it,
    decided as the read ended (`ReadBudget.ran_out`), since the answer may join the read
    after the budget's deadline has passed."""

    exc: Exception
    cut: bool


@dataclass(frozen=True, slots=True)
class RelatedRead:
    """The related-measures reads started in a task for `stat_var`. `task` settles to the
    loss that ended them when a read failed or ran past its budget."""

    stat_var: str
    task: asyncio.Task[Related | LostRelated]


@dataclass(frozen=True, slots=True)
class PendingRelated:
    """The related-measures reads running on a thread for `stat_var`, ending by the deadline
    of `budget`. `future` settles to the loss that ended a failed read."""

    stat_var: str
    future: Future[Related | LostRelated]
    budget: ReadBudget


def start_elected_related_read(state: ResolveState, mat_result: Materialised) -> None:
    """Read the measure family the elected shape's served members exhibit into
    `state.measure_family`, and start their related-measures reads into `state.related_read`
    (`start_related_read`), replacing one the pass started before."""
    served_member_facts = mat_result.member_facts or {}
    role_keys = entity_role_keys(
        state.place.roles.resolved_entities,
        open_axes=state.place.roles.open_axes,
    )
    state.measure_family = observed_measure_family(
        state.shape_draft,
        get_index(),
        get_measure_index().axes,
        shape_query(
            state.shape_draft,
            state.bindings,
            role_keys.bound,
            open_role_keys=role_keys.open,
        ),
        # Every real resolve supplies member facts for each StatVar. A missing entry
        # violates the contract and raises.
        [served_member_facts[dcid] for dcid in mat_result.sv_dcids],
    )
    state.related_read = start_related_read(
        state,
        stat_vars=mat_result.sv_dcids,
        measure_partition=state.measure_family[0],
        places=state.place.roles.roles,
        arcs=state.shape_draft.sv_arc_facts,
    )


def start_related_read(
    state: ResolveState,
    *,
    stat_vars: Sequence[str],
    measure_partition: MeasurePartition | None,
    places: Sequence[tuple[str, EntityRoleKey]],
    arcs: Mapping[str, Mapping] | None,
) -> RelatedRead | None:
    """Start the reads in a task for an answer that qualifies, None for any other answer,
    which reads nothing.

    Args:
        state: The pass, whose graph, date request and deadline the reads use.
        stat_vars: The StatVars the answer serves.
        measure_partition: The measure partition they exhibit, None outside a family.
        places: The answer's places with their roles.
        arcs: Full arcs by StatVar, as recall read them (`ShapeDraft.sv_arc_facts`).
    """
    if not _qualifies(stat_vars, measure_partition, places):
        return None
    stat_var = stat_vars[0]
    budget = _read_budget(state.deadline)
    read = read_runner.start(
        functools.partial(
            read_related,
            stat_var=stat_var,
            arcs=(arcs or {}).get(stat_var),
            places=[dcid for dcid, _role in places],
            window=requested_window_from_date_request(state.date_request),
            graph=state.graph,
            deadline=budget.deadline,
        )
    )
    started = RelatedRead(
        stat_var=stat_var, task=asyncio.create_task(_within(read, stat_var=stat_var, budget=budget))
    )
    state.related_reads.append(started)
    return started


async def settle_related_reads(reads: Sequence[RelatedRead]) -> None:
    """Wait for every read in `reads` to settle, each by its deadline. A read cancelled
    on its own, with the pass still running, counts as settled, so its cancellation does
    not replace the pass's result."""
    await asyncio.gather(*(read.task for read in reads), return_exceptions=True)


def cancel_related_reads(reads: Sequence[RelatedRead]) -> None:
    """Cancel every read in `reads`. A running thread ends by its deadline."""
    for read in reads:
        read.task.cancel()


def related_read_for(
    state: ResolveState,
    *,
    stat_vars: Sequence[str],
    measure_partition: MeasurePartition | None,
    places: Sequence[tuple[str, EntityRoleKey]],
    arcs: Mapping[str, Mapping] | None,
) -> RelatedRead | None:
    """The pass's read when it reads the answer's one StatVar, else a read started for the
    answer (`start_related_read`), replacing the pass's."""
    held = state.related_read
    if held is not None and list(stat_vars) == [held.stat_var] and measure_partition is None:
        return held
    state.related_read = start_related_read(
        state,
        stat_vars=stat_vars,
        measure_partition=measure_partition,
        places=places,
        arcs=arcs,
    )
    return state.related_read


async def with_related(spec: Spec, read: RelatedRead | None) -> Spec:
    """`spec` stating `read`'s group and measures, unchanged when there is no read or it
    failed. A failed read is noted as a lost read, or as cut when its own budget cut it
    (`note_lost_read`, `LostRelated.cut`)."""
    if read is None:
        return spec
    related = await read.task
    if isinstance(related, LostRelated):
        note_lost_read("related measures", related.exc, cut=related.cut)
        return spec
    return stating_related(spec, related)


def submit_related_read(
    *,
    stat_vars: Sequence[str],
    measure_partition: MeasurePartition | None,
    places: Sequence[tuple[str, EntityRoleKey]],
    arcs: Mapping[str, Mapping] | None,
    window: RequestedWindow | None,
    graph: EngineGraphClient,
    request_deadline: float | None,
) -> PendingRelated | None:
    """Start the reads on a thread of their own for an answer that qualifies, None for any
    other answer, which reads nothing. The arguments are `start_related_read`'s, with the
    answer's requested `window` and the request's deadline given apart."""
    if not _qualifies(stat_vars, measure_partition, places):
        return None
    stat_var = stat_vars[0]
    budget = _read_budget(request_deadline)
    future = read_runner.submit(
        functools.partial(
            _logged_read,
            stat_var=stat_var,
            arcs=(arcs or {}).get(stat_var),
            places=[dcid for dcid, _role in places],
            window=window,
            graph=graph,
            budget=budget,
        )
    )
    return PendingRelated(stat_var=stat_var, future=future, budget=budget)


def joined_related(pending: PendingRelated | None) -> Related | None:
    """`pending`'s result once it ends, or None when there is none, a read failed or its
    deadline passed first. The thread logs a failed read (`_logged_read`), and a deadline
    that passes first is logged here. Either is noted here as a lost read, or as cut when
    the read's own budget cut it (`note_lost_read`, `LostRelated.cut`), in the caller's
    record."""
    if pending is None:
        return None
    try:
        related = pending.future.result(
            timeout=max(0.0, pending.budget.deadline - time.monotonic())
        )
    except TimeoutError as exc:
        _log_unavailable(pending.stat_var, exc)
        related = LostRelated(exc, cut=pending.budget.ran_out(exc))
    if isinstance(related, LostRelated):
        note_lost_read("related measures", related.exc, cut=related.cut)
        return None
    return related


def settle_pending_related(pending: PendingRelated | None) -> None:
    """Wait for `pending`'s thread to end, by its deadline."""
    if pending is not None:
        wait_futures([pending.future], timeout=max(0.0, pending.budget.deadline - time.monotonic()))


def stating_related(spec: Spec, related: Related | None) -> Spec:
    """`spec` stating `related`'s group and measures, unchanged when `related` is None."""
    if related is None:
        return spec
    resolved = spec.resolved.model_copy(
        update={
            "related_group": related.group,
            "related_measures": related_measures(related, spec.definition.identity),
        }
    )
    return spec.model_copy(update={"resolved": resolved})


def _qualifies(
    stat_vars: Sequence[str],
    measure_partition: MeasurePartition | None,
    places: Sequence[tuple[str, EntityRoleKey]],
) -> bool:
    return (
        len(stat_vars) == 1
        and measure_partition is None
        and bool(places)
        and all(role == "observationAbout" for _dcid, role in places)
    )


def _read_budget(request_deadline: float | None) -> ReadBudget:
    return ReadBudget.from_now(
        settings().related_measures_timeout_s, request_deadline=request_deadline
    )


def _logged_read(*, stat_var: str, budget: ReadBudget, **kwargs: Any) -> Related | LostRelated:
    """`read_related` on the resubmit door's thread, bound by `budget`'s deadline, or the
    loss that ended it. The failure is logged here, so it surfaces however the pass ends, as
    `_within` does for a task."""
    try:
        return read_related(stat_var=stat_var, deadline=budget.deadline, **kwargs)
    except Exception as exc:
        _log_unavailable(stat_var, exc)
        return LostRelated(exc, cut=budget.ran_out(exc))


async def _within(
    read: Awaitable[Related], *, stat_var: str, budget: ReadBudget
) -> Related | LostRelated:
    """`read`'s result, or the loss that ended it when a read failed or `budget`'s deadline
    passed first."""
    try:
        return await asyncio.wait_for(read, timeout=max(0.0, budget.deadline - time.monotonic()))
    except Exception as exc:
        _log_unavailable(stat_var, exc)
        return LostRelated(exc, cut=budget.ran_out(exc))


def _log_unavailable(stat_var: str, exc: Exception) -> None:
    """Log why `stat_var`'s related reads state none. A graph failure or a timeout is a
    warning. Any other exception is a defect in the read, logged at error with its
    traceback, and the answer stands, since related measures are optional."""
    if isinstance(exc, GraphInfraError | TimeoutError):
        logger.warning("related measures unavailable for %s: %r", stat_var, exc)
    else:
        logger.error("related measures read failed for %s", stat_var, exc_info=exc)
