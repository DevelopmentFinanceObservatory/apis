"""Checks `DatasetsNewestFrames` sends a dataset frame through the frame refresh and a derived
frame through the analysis runner as a bare derived root with `latest=true`, and reads both
pending answers as `NewestPending`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import presentation_api.newest_frames as newest_frames
from assembler.ports import FrameRef
from datasets_api.analysis import AnalysisPending, AnalysisRequest, DerivedRoot
from datasets_api.deps import DepsCell
from datasets_api.doors.refresh import FrameMoved, FramePair, FramePending
from dfo_analysis_engine.frame_ref import FrameRef as EngineFrameRef
from presentation_api.newest_frames import DatasetsNewestFrames, NewestPending

_DATASET = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)
_DATASET_NOW = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "c" * 32)
_DERIVED = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "e" * 32)
_DERIVED_NOW = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "f" * 32)


class _Calls:
    """What the replaced Datasets functions answer, and the arguments they were called with."""

    def __init__(self) -> None:
        self.answers: dict[str, Any] = {}
        self.refresh: list[tuple[str, str]] = []
        self.run: list[AnalysisRequest] = []


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """Replace the Datasets functions the class calls, recording their arguments."""
    recorded = _Calls()

    def refresh(*, spec_id: str, content_version: str, deps: object, trace_id: object):
        recorded.refresh.append((spec_id, content_version))
        return recorded.answers["refresh"]

    def run(request: AnalysisRequest, **_stores: object):
        recorded.run.append(request)
        answer = recorded.answers["run"]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(evaluation=SimpleNamespace(ref=answer))

    monkeypatch.setattr(newest_frames, "definition_deps", lambda current, deadline: current)
    monkeypatch.setattr(newest_frames, "answer_frame_refresh", refresh)
    monkeypatch.setattr(newest_frames, "run_analysis", run)
    monkeypatch.setattr(newest_frames, "LiveRootSource", lambda **_: None)
    for helper in ("analysis_stores", "analysis_bounds", "analysis_permit"):
        monkeypatch.setattr(newest_frames, helper, lambda *_: None)
    return recorded


def _newest() -> DatasetsNewestFrames:
    deps = SimpleNamespace(recipe_store=None)
    return DatasetsNewestFrames(DepsCell(deps), budget_seconds=5)


def test_a_dataset_frame_moves_as_the_frame_refresh_answers(calls: _Calls):
    calls.answers["refresh"] = FrameMoved(
        frame=FramePair(spec_id=_DATASET_NOW.spec_id, content_version=_DATASET_NOW.content_version)
    )

    assert _newest().newest(_DATASET) == _DATASET_NOW
    assert calls.refresh == [(_DATASET.spec_id, _DATASET.content_version)]
    assert calls.run == []


def test_a_derived_frame_reruns_its_recipe_on_the_newest_data(calls: _Calls):
    calls.answers["run"] = EngineFrameRef(
        spec_id=_DERIVED_NOW.spec_id, content_version=_DERIVED_NOW.content_version
    )

    assert _newest().newest(_DERIVED) == _DERIVED_NOW
    assert calls.run == [
        AnalysisRequest(
            tree=DerivedRoot(spec_id=_DERIVED.spec_id, content_version=_DERIVED.content_version),
            latest=True,
        )
    ]
    assert calls.refresh == []


def test_a_rebuild_over_budget_is_pending_on_either_path(calls: _Calls):
    calls.answers["refresh"] = FramePending(operation_id="op_1", retry_after_s=2.0)
    calls.answers["run"] = AnalysisPending(operations=["op_2"], retry_after_s=2.0)

    newest = _newest()

    assert newest.newest(_DATASET) == NewestPending()
    assert newest.newest(_DERIVED) == NewestPending()
