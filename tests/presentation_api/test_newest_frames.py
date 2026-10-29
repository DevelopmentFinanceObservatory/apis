"""Checks `DatasetsNewestFrames` sends a dataset frame through the frame refresh under the
rendering its card recorded, and a derived frame through the analysis door's keyed path as a
bare derived root with `latest=true`: a stored answer computes nothing, and a key another check
is computing joins it. Both pending answers read as `NewestPending`. A bare dataset id reads the
dataset version the runner's Spec root of it reads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import presentation_api.newest_frames as newest_frames
from assembler.ports import FrameRef
from datasets_api.analysis import AnalysisPending, AnalysisRequest, DerivedRoot, SpecRoot
from datasets_api.deps import DepsCell
from datasets_api.doors.analysis import RunningAnalyses
from datasets_api.doors.refresh import FrameMoved, FramePair, FramePending, StoredRendering
from datasets_api.recorded_rendering import MintingOptions
from presentation_api.newest_frames import DatasetsNewestFrames, NewestPending

_DATASET = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)
_DATASET_NOW = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "c" * 32)
_DERIVED = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "e" * 32)
_DERIVED_NOW = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "f" * 32)


def _answer_body(frame: FrameRef) -> dict[str, Any]:
    """The part of an analysis answer body naming its stored frame."""
    return {
        "ref": {
            "members": [
                {"spec": {"spec_id": frame.spec_id}, "content_version": frame.content_version}
            ]
        }
    }


class _Calls:
    """What the replaced Datasets functions answer, and the arguments they were called with."""

    def __init__(self) -> None:
        self.answers: dict[str, Any] = {}
        self.refresh: list[tuple[str, str, StoredRendering | None]] = []
        self.prepared: list[AnalysisRequest] = []
        self.run: int = 0
        # Called inside a run, as another check arriving while it computes.
        self.during_run: Any = None


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """Replace the Datasets functions the class calls, recording their arguments."""
    seen = _Calls()

    def refresh(
        *, spec_id: str, content_version: str, deps: object, trace_id: object, recorded: object
    ):
        seen.refresh.append((spec_id, content_version, recorded))
        return seen.answers["refresh"]

    def prepare(request: AnalysisRequest, **_: object):
        seen.prepared.append(request)
        answer = seen.answers.get("prepare")
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(key="key_1")

    def stored(prepared: object, **_: object):
        frame = seen.answers.get("stored")
        return None if frame is None else _answer_body(frame)

    def run(prepared: object, **_: object):
        seen.run += 1
        if (during := seen.during_run) is not None:
            seen.during_run = None
            during()
        return SimpleNamespace(body=lambda: _answer_body(seen.answers["run"]))

    monkeypatch.setattr(newest_frames, "definition_deps", lambda current, deadline: current)
    monkeypatch.setattr(newest_frames, "answer_frame_refresh", refresh)
    monkeypatch.setattr(newest_frames, "prepare_analysis", prepare)
    monkeypatch.setattr(newest_frames, "stored_answer", stored)
    monkeypatch.setattr(newest_frames, "run_prepared", run)
    monkeypatch.setattr(newest_frames, "LiveRootSource", lambda **_: None)
    for helper in ("analysis_stores", "analysis_bounds", "analysis_permit"):
        monkeypatch.setattr(newest_frames, helper, lambda *_: None)
    return seen


def _newest() -> DatasetsNewestFrames:
    deps = SimpleNamespace(recipe_store=None)
    return DatasetsNewestFrames(DepsCell(deps), running=RunningAnalyses(), budget_seconds=5)


_CUT = StoredRendering(
    base_content_version=_DATASET.content_version,
    minting=MintingOptions(without_party_totals=True),
    mintings=(MintingOptions(without_party_totals=True),),
)


@pytest.mark.parametrize(
    "recorded", [pytest.param(_CUT, id="recorded"), pytest.param(None, id="unrecorded")]
)
def test_a_dataset_frame_moves_as_the_frame_refresh_answers_under_its_recorded_rendering(
    calls: _Calls, recorded: StoredRendering | None
):
    calls.answers["refresh"] = FrameMoved(
        frame=FramePair(spec_id=_DATASET_NOW.spec_id, content_version=_DATASET_NOW.content_version)
    )
    rendering = recorded.to_record() if recorded is not None else None

    assert _newest().newest(_DATASET, rendering=rendering) == _DATASET_NOW
    assert calls.refresh == [(_DATASET.spec_id, _DATASET.content_version, recorded)]
    assert calls.prepared == []


def test_a_derived_frame_reruns_its_recipe_on_the_newest_data(calls: _Calls):
    calls.answers["run"] = _DERIVED_NOW

    assert _newest().newest(_DERIVED, rendering=None) == _DERIVED_NOW
    assert calls.prepared == [
        AnalysisRequest(
            tree=DerivedRoot(spec_id=_DERIVED.spec_id, content_version=_DERIVED.content_version),
            latest=True,
        )
    ]
    assert calls.run == 1
    assert calls.refresh == []


def test_a_derived_frame_whose_key_holds_a_stored_answer_computes_nothing(calls: _Calls):
    calls.answers["stored"] = _DERIVED_NOW

    assert _newest().newest(_DERIVED, rendering=None) == _DERIVED_NOW
    assert calls.run == 0


def test_a_check_of_a_key_another_check_computes_joins_it_as_pending(calls: _Calls):
    calls.answers["run"] = _DERIVED_NOW
    newest = _newest()
    joined: list[object] = []
    calls.during_run = lambda: joined.append(newest.newest(_DERIVED, rendering=None))

    assert newest.newest(_DERIVED, rendering=None) == _DERIVED_NOW
    assert joined == [NewestPending(retry_after_s=5.0)]
    assert calls.run == 1


def test_a_rebuild_over_budget_is_pending_with_its_retry_time_on_either_path(calls: _Calls):
    calls.answers["refresh"] = FramePending(operation_id="op_1", retry_after_s=2.0)
    calls.answers["prepare"] = AnalysisPending(operations=["op_2"], retry_after_s=3.5)

    newest = _newest()

    assert newest.newest(_DATASET, rendering=None) == NewestPending(retry_after_s=2.0)
    assert newest.newest(_DERIVED, rendering=None) == NewestPending(retry_after_s=3.5)


class _SpecRoots:
    """`LiveRootSource` resolving every root to `answer`'s dataset, or raising `answer`."""

    def __init__(self, answer: FrameRef | Exception) -> None:
        self.answer = answer
        self.resolved: list[tuple[object, bool]] = []

    def resolve(self, root: object, *, latest: bool, trace_id: object) -> Any:
        self.resolved.append((root, latest))
        if isinstance(self.answer, Exception):
            raise self.answer
        return SimpleNamespace(dataset=self.answer)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param(_DATASET_NOW, _DATASET_NOW, id="served"),
        pytest.param(
            AnalysisPending(operations=["op_3"], retry_after_s=2.0),
            NewestPending(retry_after_s=2.0),
            id="building",
        ),
    ],
)
def test_a_bare_spec_id_reads_the_dataset_the_runner_reads_for_its_spec_root(
    calls: _Calls,
    monkeypatch: pytest.MonkeyPatch,
    answer: FrameRef | Exception,
    expected: FrameRef | NewestPending,
):
    """The runner's Spec root of the id in its default rendering, its dataset version named."""
    roots = _SpecRoots(answer)
    monkeypatch.setattr(newest_frames, "LiveRootSource", lambda **_: roots)

    assert _newest().newest_of_spec(_DATASET.spec_id) == expected
    assert roots.resolved == [(SpecRoot(spec_id=_DATASET.spec_id), False)]
