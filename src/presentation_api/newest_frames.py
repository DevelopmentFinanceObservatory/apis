"""The newest frame a live share slot's card can draw, asked of the Datasets surface in process.

A live slot's data read (`presentation_api.doors.share_data`) asks `NewestFrames.newest` for the
frame of the newest data in place of the frame its card revision is stored on, once the slot's
last check is due (`presentation_api.share_slots`):

- a dataset frame (`spec_`) is answered as the frame refresh answers it
  (`datasets_api.doors.refresh.answer_frame_refresh`): the current frame of the same rendering,
  rebuilding the dataset when its data moved or its head is too old. The rendering is the one
  the card revision recorded (`RevisionRecord.frame_rendering`) when it holds one, so a file of
  another rendering linked later under the stored version changes nothing, as on the render
  routes; a revision that recorded none is refreshed under the stored frame's records;
- a derived frame (`ops_`) is answered as `POST /api/datasets/analysis` answers a bare derived
  root with `latest=true`: the run is keyed (`datasets_api.analysis.prepare_analysis`), a key
  holding a stored answer whose frame file is still served answers that frame
  (`datasets_api.analysis.stored_answer`), and only a new key computes its stored recipe again
  on its datasets' newest versions (`datasets_api.analysis.run_prepared`). So an unchanged
  recipe on unchanged roots computes nothing, and a key that another check or a
  `POST /api/datasets/analysis` in this process is computing answers `NewestPending` after the
  analysis door's join wait (both hold one `RunningAnalyses`).

Either answers `NewestPending` while a rebuild runs over its inline budget, with the seconds
the refresh or the runner asks the caller to wait. Any refusal or failure raises, and the read
states the slot `stale`.

`NewestFrames.newest_of_spec` answers a bare `spec_` id, which the frames door and a component
create accept in place of a pair, with the dataset version the analysis runner reads
for a Spec root `{spec_id}` (`datasets_api.analysis_roots.newest_spec_version`): the served
definition's newest version in the root's default rendering, rebuilt when its head is stale.
`resolve_bare_spec` turns its refusals into the `frame_unavailable` the presentation doors serve.

`DatasetsNewestFrames` reads the Datasets surface's `DepsCell`, so it answers only in an app
that mounts that surface (`dfo_apis.app` passes the cell and the `RunningAnalyses` to both).
The work runs on the share door's executor thread under the Presentation request's bound catalogue and release, holding
the materialise permits Datasets' own reads hold.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from assembler.ports import FrameRef, FrameUnavailable
from datasets_api.analysis import (
    AnalysisPending,
    AnalysisRequest,
    DerivedRoot,
    prepare_analysis,
    run_prepared,
    stored_answer,
)
from datasets_api.analysis_roots import LiveRootSource, newest_spec_version
from datasets_api.app import classify_datasets_exception
from datasets_api.current import DefinitionDeps
from datasets_api.deps import Deps, DepsCell
from datasets_api.doors.analysis import (
    RunningAnalyses,
    analysis_bounds,
    analysis_permit,
    analysis_stores,
)
from datasets_api.doors.definition import definition_deps
from datasets_api.doors.refresh import FramePending, StoredRendering, answer_frame_refresh
from dfo_platform.codes import bound_registry_endpoint
from dfo_platform.trace import get_trace_id

__all__ = [
    "FRESHNESS_WINDOW",
    "DatasetsNewestFrames",
    "NewestFrames",
    "NewestPending",
    "resolve_bare_spec",
]

FRESHNESS_WINDOW = timedelta(minutes=15)
"""How long a live slot serves a `current` or `stale` check's answer before it asks for newer
data again, and the longest a `pending` check waits."""

_DERIVED_SPEC_PREFIX = "ops_"


@dataclass(frozen=True)
class NewestPending:
    """Newer data is building as a queued operation, and the refresh or the run asks for a retry
    after `retry_after_s` seconds."""

    retry_after_s: float


class NewestFrames(Protocol):
    def newest(
        self, frame: FrameRef, *, rendering: Mapping[str, object] | None
    ) -> FrameRef | NewestPending:
        """The frame of the newest data that stands for `frame`, which is `frame` itself while
        its data has not moved, or `NewestPending` while newer data builds. `rendering` is the
        frame's rendering as the card revision recorded it (`RevisionRecord.frame_rendering`),
        `None` when it recorded none.

        Raises:
            Exception: Any refusal or failure of the refresh or the run.
        """
        ...

    def newest_of_spec(self, spec_id: str) -> FrameRef | NewestPending:
        """The dataset version a Spec root `{spec_id}` reads, named by the served id it is
        recorded under, or `NewestPending` while it builds.

        Raises:
            Exception: Any refusal or failure of the serve.
        """
        ...


class DatasetsNewestFrames:
    """`NewestFrames` over the Datasets surface's dependencies. Each call gets `budget_seconds`
    before its SDMX reads and materialise permit give up. `running` is the analysis door's set
    of keys computing, so a check and `POST /api/datasets/analysis` of one key compute once."""

    def __init__(self, cell: DepsCell, *, running: RunningAnalyses, budget_seconds: float) -> None:
        self._cell = cell
        self._running = running
        self._budget_seconds = budget_seconds

    def newest(
        self, frame: FrameRef, *, rendering: Mapping[str, object] | None
    ) -> FrameRef | NewestPending:
        current = self._cell.get()
        deps = definition_deps(current, deadline=time.monotonic() + self._budget_seconds)
        trace_id = get_trace_id()
        if frame.spec_id.startswith(_DERIVED_SPEC_PREFIX):
            try:
                answer = self._derived_answer(frame, current=current, deps=deps, trace_id=trace_id)
            except AnalysisPending as pending:
                return NewestPending(retry_after_s=pending.retry_after_s)
            member = answer["ref"]["members"][0]
            return FrameRef(
                spec_id=member["spec"]["spec_id"], content_version=member["content_version"]
            )
        refreshed = answer_frame_refresh(
            spec_id=frame.spec_id,
            content_version=frame.content_version,
            deps=deps,
            trace_id=trace_id,
            recorded=StoredRendering.from_record(rendering) if rendering is not None else None,
        )
        if isinstance(refreshed, FramePending):
            return NewestPending(retry_after_s=refreshed.retry_after_s)
        return FrameRef(
            spec_id=refreshed.frame.spec_id, content_version=refreshed.frame.content_version
        )

    def newest_of_spec(self, spec_id: str) -> FrameRef | NewestPending:
        current = self._cell.get()
        deps = definition_deps(current, deadline=time.monotonic() + self._budget_seconds)
        try:
            dataset = newest_spec_version(
                spec_id, deps=deps, recipes=current.recipe_store, trace_id=get_trace_id()
            )
        except AnalysisPending as pending:
            return NewestPending(retry_after_s=pending.retry_after_s)
        return FrameRef(spec_id=dataset.spec_id, content_version=dataset.content_version)

    def _derived_answer(
        self, frame: FrameRef, *, current: Deps, deps: DefinitionDeps, trace_id: str | None
    ) -> dict[str, Any]:
        """The analysis answer body of `frame`'s recipe on its datasets' newest versions: the
        one stored under the run's key, else the one a run computes and stores.

        Raises:
            AnalysisPending: A root is still building, or another check computes the key.
        """
        request = AnalysisRequest(
            tree=DerivedRoot(spec_id=frame.spec_id, content_version=frame.content_version),
            latest=True,
        )
        source = LiveRootSource(deps=deps, recipes=current.recipe_store)
        stores = analysis_stores(current)
        bounds = analysis_bounds(current)
        prepared = prepare_analysis(request, source=source, bounds=bounds, trace_id=trace_id)
        answer = stored_answer(prepared, stores=stores)
        if answer is not None:
            return answer
        with self._running.claim(prepared.key):
            return run_prepared(
                prepared,
                source=source,
                stores=stores,
                bounds=bounds,
                permit=analysis_permit(current, deps),
                trace_id=trace_id,
            ).body()


def resolve_bare_spec(spec_id: str, *, newest: NewestFrames | None) -> FrameRef:
    """The frame a bare `spec_id` names (`NewestFrames.newest_of_spec`).

    Raises:
        FrameUnavailable: `io` (503, retryable) while the version builds or for a refusal a
            retry may clear; `quarantined` for a quarantined identity; `unknown` (404) for any
            other refusal below status 500, the Datasets code and detail in its detail; and
            `unknown` when no Datasets surface is mounted beside this one, so no bare id
            resolves.
        Exception: Any other Datasets refusal the Datasets surface answers at 500 or above,
            or a failure it does not classify, unchanged.
    """
    if newest is None:
        raise FrameUnavailable(
            reason="unknown",
            detail=f"this deployment resolves no bare spec_id; name {spec_id}'s content_version",
        )
    try:
        resolved = newest.newest_of_spec(spec_id)
    except Exception as exc:
        # The resolve is the analysis runner's, so its refusals classify as that door's do.
        with bound_registry_endpoint("datasets.analysis"):
            problem = classify_datasets_exception(exc)
        if problem is None:
            raise
        if problem.retryable:
            reason = "io"
        elif problem.code == "identity_quarantined":
            reason = "quarantined"
        elif problem.status >= 500:
            # A server fault stays one; the surface answers it `internal_error`.
            raise
        else:
            reason = "unknown"
        raise FrameUnavailable(
            reason=reason,
            detail=f"the newest version of {spec_id} cannot be served: {problem.code}: "
            f"{problem.detail}",
        ) from exc
    if isinstance(resolved, NewestPending):
        raise FrameUnavailable(
            reason="io",
            detail=f"the newest version of {spec_id} is building; retry in "
            f"{math.ceil(resolved.retry_after_s)} s",
        )
    return resolved
