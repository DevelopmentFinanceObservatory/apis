"""The newest frame a live share slot's card can draw, asked of the Datasets surface in process.

A live slot's data read (`presentation_api.doors.share_data`) asks `NewestFrames.newest` for the
frame of the newest data in place of the frame its card revision is stored on, once the slot's
last check is due (`presentation_api.share_slots`):

- a dataset frame (`spec_`) is answered as the frame refresh answers it
  (`datasets_api.doors.refresh.answer_frame_refresh`): the current frame of the same rendering,
  rebuilding the dataset when its data moved or its head is too old;
- a derived frame (`ops_`) is answered as the analysis runner answers a bare derived root with
  `latest=true` (`datasets_api.analysis.run_analysis`): its stored recipe computed again on its
  datasets' newest versions and stored, the frame a client's `fetch(ref, latest=True)` reads.

Either answers `NewestPending` while a rebuild runs over its inline budget, with the seconds
the refresh or the runner asks the caller to wait. Any refusal or failure raises, and the read
states the slot `stale`.

`DatasetsNewestFrames` reads the Datasets surface's `DepsCell`, so it answers only in an app
that mounts that surface (`dfo_apis.app` passes the cell to both). The work runs on the share
door's executor thread under the Presentation request's bound catalogue and release, holding
the materialise permits Datasets' own reads hold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from assembler.ports import FrameRef
from datasets_api.analysis import AnalysisPending, AnalysisRequest, DerivedRoot, run_analysis
from datasets_api.analysis_roots import LiveRootSource
from datasets_api.deps import DepsCell
from datasets_api.doors.analysis import analysis_bounds, analysis_permit, analysis_stores
from datasets_api.doors.definition import definition_deps
from datasets_api.doors.refresh import FramePending, answer_frame_refresh
from dfo_platform.trace import get_trace_id

__all__ = ["FRESHNESS_WINDOW", "DatasetsNewestFrames", "NewestFrames", "NewestPending"]

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
    def newest(self, frame: FrameRef) -> FrameRef | NewestPending:
        """The frame of the newest data that stands for `frame`, which is `frame` itself while
        its data has not moved, or `NewestPending` while newer data builds.

        Raises:
            Exception: Any refusal or failure of the refresh or the run.
        """
        ...


class DatasetsNewestFrames:
    """`NewestFrames` over the Datasets surface's dependencies. Each call gets `budget_seconds`
    before its SDMX reads and materialise permit give up."""

    def __init__(self, cell: DepsCell, *, budget_seconds: float) -> None:
        self._cell = cell
        self._budget_seconds = budget_seconds

    def newest(self, frame: FrameRef) -> FrameRef | NewestPending:
        current = self._cell.get()
        deps = definition_deps(current, deadline=time.monotonic() + self._budget_seconds)
        trace_id = get_trace_id()
        if frame.spec_id.startswith(_DERIVED_SPEC_PREFIX):
            request = AnalysisRequest(
                tree=DerivedRoot(spec_id=frame.spec_id, content_version=frame.content_version),
                latest=True,
            )
            try:
                result = run_analysis(
                    request,
                    source=LiveRootSource(deps=deps, recipes=current.recipe_store),
                    stores=analysis_stores(current),
                    bounds=analysis_bounds(current),
                    permit=analysis_permit(current, deps),
                    trace_id=trace_id,
                )
            except AnalysisPending as pending:
                return NewestPending(retry_after_s=pending.retry_after_s)
            ref = result.evaluation.ref
            return FrameRef(spec_id=ref.spec_id, content_version=ref.content_version)
        answer = answer_frame_refresh(
            spec_id=frame.spec_id,
            content_version=frame.content_version,
            deps=deps,
            trace_id=trace_id,
        )
        if isinstance(answer, FramePending):
            return NewestPending(retry_after_s=answer.retry_after_s)
        return FrameRef(spec_id=answer.frame.spec_id, content_version=answer.frame.content_version)
