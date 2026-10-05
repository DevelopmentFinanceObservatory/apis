"""Checks capture and cell persistence, recovery, and completion state."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

import promptlab.store as store_module
from promptlab.store import (
    BindRecord,
    CaptureManifest,
    CaptureStore,
    CellManifest,
    CellStore,
    ExtractRecord,
    NotComplete,
    StoreError,
)

_BASE_CAPTURE_MANIFEST = CaptureManifest(
    capture_id="k8-2026-08-31",
    git_sha="abc123",
    engine_build_id="build-1",
    model="gemini-3.5-flash-lite",
    top_k=8,
    substituted_date="2026-08-31",
    graph_base="https://graph.example",
    corpus_sha256="deadbeef",
    seeds=(1, 2, 3),
    extract_system_sha1="e" * 40,
    bind_system_sha1="b" * 40,
)

_BASE_CELL_MANIFEST = CellManifest(
    cell_id="ceiling",
    rung={
        "name": "production",
        "extract": {"stage": "extract", "blocks": [{"id": "00-task-and-schema"}]},
        "bind": {"stage": "bind", "blocks": [{"id": "00-role", "core": True}]},
        "run": {"capture": "k8-2026-08-31", "top_k": 8, "seeds": [1, 2, 3]},
    },
    capture_id="k8-2026-08-31",
    git_sha="abc123",
    model="gemini-3.5-flash-lite",
    seeds=(1, 2, 3),
    extract_system="line one\nline two\n",
    bind_system="role\n\nshape definition\n",
)


def _capture_manifest(**overrides: object) -> CaptureManifest:
    return dataclasses.replace(_BASE_CAPTURE_MANIFEST, **overrides)


def _cell_manifest(**overrides: object) -> CellManifest:
    return dataclasses.replace(_BASE_CELL_MANIFEST, **overrides)


# -- fresh-create layout --------------------------------------------------


def test_capture_create_writes_manifest_and_empty_logs(tmp_path: Path):
    root = tmp_path / "capture" / "k8-2026-08-31"
    manifest = _capture_manifest()

    store = CaptureStore.create(root, manifest=manifest)

    assert (root / "manifest.json").is_file()
    assert (root / "extract.jsonl").is_file()
    assert (root / "bind.jsonl").is_file()
    assert not (root / "status.json").exists()
    assert store.done_extract_keys() == set()
    assert store.done_bind_keys() == set()
    assert store.graph_db_path == root / "graph.sqlite"
    store.close()


def test_cell_create_writes_manifest_and_empty_log(tmp_path: Path):
    root = tmp_path / "cells" / "ceiling"
    manifest = _cell_manifest()

    store = CellStore.create(root, manifest=manifest)

    assert (root / "manifest.json").is_file()
    assert (root / "calls.jsonl").is_file()
    assert not (root / "scores.json").exists()
    assert not (root / "status.json").exists()
    assert store.done_keys() == set()
    store.close()


def test_create_over_existing_manifest_raises(tmp_path: Path):
    root = tmp_path / "capture" / "dup"
    CaptureStore.create(root, manifest=_capture_manifest()).close()

    with pytest.raises(StoreError, match="already exists"):
        CaptureStore.create(root, manifest=_capture_manifest())


def test_open_without_manifest_raises(tmp_path: Path):
    root = tmp_path / "nothing-here"
    root.mkdir()

    with pytest.raises(StoreError, match="nothing to resume"):
        CaptureStore.open(root)
    with pytest.raises(StoreError, match="nothing to resume"):
        CellStore.open(root)


# -- append + reload round-trip --------------------------------------------------


def test_capture_append_reload_roundtrip(tmp_path: Path):
    root = tmp_path / "capture" / "roundtrip"
    store = CaptureStore.create(root, manifest=_capture_manifest())

    extract = ExtractRecord(
        case_id="g1",
        query="health ODA grants from USA to Ethiopia",
        system_sha1="e" * 40,
        output={"variables": []},
        usage={"input_tokens": 10},
        elapsed_ms=120,
    )
    bind = BindRecord(
        case_id="g1",
        leg=0,
        variable="health ODA",
        prompt="Variable phrase: health ODA\n",
        prompt_sha1="p" * 40,
        system_sha1="b" * 40,
        output={"chosen_index": 0},
        usage=None,
        elapsed_ms=340,
    )
    store.append_extract(extract)
    store.append_bind(bind)
    store.close()

    reopened = CaptureStore.open(root)
    assert reopened.extract_records() == (extract,)
    assert reopened.bind_records() == (bind,)
    assert reopened.done_extract_keys() == {"g1"}
    assert reopened.done_bind_keys() == {("g1", 0)}
    reopened.close()


def test_cell_append_reload_roundtrip(tmp_path: Path):
    root = tmp_path / "cells" / "roundtrip"
    store = CellStore.create(root, manifest=_cell_manifest())

    store.append(case_id="g1", leg=0, seed=1, call={"schema": "Extraction", "output": {"a": 1}})
    store.append(case_id="g1", leg=1, seed=1, call={"schema": "Extraction", "output": {"a": 2}})
    store.close()

    reopened = CellStore.open(root)
    records = reopened.call_records()
    assert records == (
        {"case_id": "g1", "leg": 0, "seed": 1, "schema": "Extraction", "output": {"a": 1}},
        {"case_id": "g1", "leg": 1, "seed": 1, "schema": "Extraction", "output": {"a": 2}},
    )
    assert reopened.done_keys() == {("g1", 0, 1), ("g1", 1, 1)}
    reopened.close()


# -- resume after a simulated kill --------------------------------------------------


def test_cell_resume_after_kill_has_no_duplicates_and_correct_done_set(tmp_path: Path):
    root = tmp_path / "cells" / "killed"
    store = CellStore.create(root, manifest=_cell_manifest())
    store.append(case_id="g1", leg=0, seed=1, call={"v": 1})
    store.append(case_id="g1", leg=0, seed=2, call={"v": 2})
    # Reopen after appends without closing or finishing the writer.
    del store

    resumed = CellStore.open(root)
    assert resumed.done_keys() == {("g1", 0, 1), ("g1", 0, 2)}
    assert len(resumed.call_records()) == 2

    # Reject duplicate keys.
    with pytest.raises(StoreError, match="already written"):
        resumed.append(case_id="g1", leg=0, seed=1, call={"v": 1})

    # The remaining key can be appended after resume.
    resumed.append(case_id="g1", leg=0, seed=3, call={"v": 3})
    resumed.close()

    final = CellStore.open(root)
    assert final.done_keys() == {("g1", 0, 1), ("g1", 0, 2), ("g1", 0, 3)}
    assert len(final.call_records()) == 3
    final.close()


def test_capture_resume_after_kill_has_no_duplicates(tmp_path: Path):
    root = tmp_path / "capture" / "killed"
    store = CaptureStore.create(root, manifest=_capture_manifest())
    store.append_extract(
        ExtractRecord(
            case_id="g1", query="q", system_sha1="e" * 40, output={}, usage=None, elapsed_ms=1
        )
    )
    del store  # simulated kill, no close()

    resumed = CaptureStore.open(root)
    assert resumed.done_extract_keys() == {"g1"}
    with pytest.raises(StoreError, match="already written"):
        resumed.append_extract(
            ExtractRecord(
                case_id="g1", query="q", system_sha1="e" * 40, output={}, usage=None, elapsed_ms=1
            )
        )
    resumed.close()


def test_replay_tolerates_one_malformed_trailing_line(tmp_path: Path):
    """Reopen recovers the complete record and truncates the torn trailing line."""
    root = tmp_path / "cells" / "torn"
    store = CellStore.create(root, manifest=_cell_manifest())
    store.append(case_id="g1", leg=0, seed=1, call={"v": 1})
    store.close()

    calls_path = root / "calls.jsonl"
    with calls_path.open("a") as f:
        f.write('{"case_id": "g1", "leg": 0, "seed": 2, "call": {"v"')  # torn mid-write, no newline

    resumed = CellStore.open(root)
    assert resumed.done_keys() == {("g1", 0, 1)}
    resumed.append(case_id="g1", leg=0, seed=2, call={"v": 2})
    resumed.close()

    lines = calls_path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["seed"] for line in lines] == [1, 2]


def test_replay_raises_on_malformed_non_trailing_line(tmp_path: Path):
    root = tmp_path / "cells" / "corrupt"
    store = CellStore.create(root, manifest=_cell_manifest())
    store.close()
    (root / "calls.jsonl").write_text(
        '{"case_id": "g1", "leg": 0, "seed": 1, "call": {}}\nnot json\n'
        '{"case_id": "g1", "leg": 0, "seed": 2, "call": {}}\n'
    )

    with pytest.raises(StoreError, match="malformed record"):
        CellStore.open(root)


# -- status.json / finish() --------------------------------------------------


def test_cell_finish_writes_scores_then_status_and_flips_to_complete(tmp_path: Path):
    root = tmp_path / "cells" / "finish"
    store = CellStore.create(root, manifest=_cell_manifest())
    store.append(case_id="g1", leg=0, seed=1, call={"v": 1})

    assert store.is_complete() is False
    assert store.status()["state"] == "partial"
    assert not (root / "scores.json").exists()
    assert not (root / "status.json").exists()

    store.finish({"interpretation_match_rate": 0.9})

    assert store.is_complete() is True
    assert (root / "scores.json").is_file()
    assert (root / "status.json").is_file()
    status = json.loads((root / "status.json").read_text())
    assert status == {"state": "complete", "done": [["g1", 0, 1]]}
    assert store.read_scores() == {"interpretation_match_rate": 0.9}

    # append() and finish() are both refused once a store is finished.
    with pytest.raises(StoreError, match="finished"):
        store.append(case_id="g1", leg=0, seed=2, call={"v": 2})
    with pytest.raises(StoreError, match="already finished"):
        store.finish({})


def test_capture_finish_flips_to_complete(tmp_path: Path):
    root = tmp_path / "capture" / "finish"
    store = CaptureStore.create(root, manifest=_capture_manifest())
    store.append_extract(
        ExtractRecord(
            case_id="g1", query="q", system_sha1="e" * 40, output={}, usage=None, elapsed_ms=1
        )
    )

    assert store.is_complete() is False
    store.finish()
    assert store.is_complete() is True
    assert (root / "status.json").is_file()

    with pytest.raises(StoreError, match="finished"):
        store.append_extract(
            ExtractRecord(
                case_id="g2", query="q", system_sha1="e" * 40, output={}, usage=None, elapsed_ms=1
            )
        )


# -- close() idempotency and finish()-failure recovery ---------------------


def test_capture_close_is_idempotent(tmp_path: Path):
    root = tmp_path / "capture" / "double-close"
    store = CaptureStore.create(root, manifest=_capture_manifest())
    store.append_extract(
        ExtractRecord(
            case_id="g1", query="q", system_sha1="e" * 40, output={}, usage=None, elapsed_ms=1
        )
    )
    store.close()

    # Closing an already-closed store is safe.
    store.close()


def test_cell_finish_failure_leaves_the_store_cleanly_closeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed score write leaves the store closable and `finish()` retryable."""
    root = tmp_path / "cells" / "finish-write-fails"
    store = CellStore.create(root, manifest=_cell_manifest())
    store.append(case_id="g1", leg=0, seed=1, call={"v": 1})

    def _boom(path, data):
        raise OSError("simulated disk-full writing scores.json")

    monkeypatch.setattr(store_module, "_write_json_durable", _boom)

    with pytest.raises(OSError, match="simulated disk-full"):
        store.finish({"interpretation_match_rate": 0.9})

    assert store._finished is False
    assert not (root / "status.json").exists()

    # `close()` remains safe after the failed finish.
    store.close()

    # Retry finish after restoring the JSON writer.
    monkeypatch.undo()
    store.finish({"interpretation_match_rate": 0.9})
    assert store.is_complete()
    assert store.read_scores() == {"interpretation_match_rate": 0.9}


def test_cell_read_scores_before_finish_raises(tmp_path: Path):
    root = tmp_path / "cells" / "no-scores-yet"
    store = CellStore.create(root, manifest=_cell_manifest())

    with pytest.raises(StoreError, match=re.escape("no scores.json yet")):
        store.read_scores()
    store.close()


# -- require_complete --------------------------------------------------


def test_cell_require_complete_raises_on_partial(tmp_path: Path):
    root = tmp_path / "cells" / "partial"
    store = CellStore.create(root, manifest=_cell_manifest())
    store.append(case_id="g1", leg=0, seed=1, call={"v": 1})
    store.close()

    reopened = CellStore.open(root)
    with pytest.raises(NotComplete):
        reopened.require_complete()
    reopened.close()


def test_cell_require_complete_passes_once_finished(tmp_path: Path):
    root = tmp_path / "cells" / "complete"
    store = CellStore.create(root, manifest=_cell_manifest())
    store.finish({})

    CellStore.open(root).require_complete()  # does not raise


def test_capture_require_complete_raises_on_partial(tmp_path: Path):
    root = tmp_path / "capture" / "partial"
    store = CaptureStore.create(root, manifest=_capture_manifest())
    store.close()

    with pytest.raises(NotComplete):
        CaptureStore.open(root).require_complete()


# -- manifest round-trip --------------------------------------------------


def test_capture_manifest_roundtrip(tmp_path: Path):
    root = tmp_path / "capture" / "manifest-roundtrip"
    manifest = _capture_manifest(seeds=(1, 2, 3), top_k=8)
    CaptureStore.create(root, manifest=manifest).close()

    reopened = CaptureStore.open(root)
    assert reopened.manifest == manifest


def test_cell_manifest_roundtrip_carries_inlined_rung_and_verbatim_systems(tmp_path: Path):
    root = tmp_path / "cells" / "manifest-roundtrip"
    rung = {
        "name": "abl-bind-also-plausible",
        "description": "Production minus the also_plausible unit.",
        "extract": {
            "stage": "extract",
            "blocks": [{"id": "00-task-and-schema", "sep": "blank", "core": True}],
        },
        "bind": {
            "stage": "bind",
            "blocks": [
                {"id": "00-role", "core": True},
                {"id": "06a-header", "sep": "blank", "unit": "binding-rules"},
            ],
        },
        "run": {
            "capture": "k8-2026-08-31",
            "top_k": 8,
            "seeds": [1, 2, 3],
            "model": "gemini-3.5-flash-lite",
            "thinking": {"extract": None, "bind": "minimal"},
        },
    }
    extract_system = 'Line one.\n\nLine two with ünicode and "quotes".\n'
    bind_system = "Role.\n\nShape selection rule.\n\nBINDING RULES (follow exactly):\n- a\n- b\n"
    manifest = _cell_manifest(
        cell_id="abl-bind-also-plausible",
        rung=rung,
        extract_system=extract_system,
        bind_system=bind_system,
        seeds=(1, 2, 3),
    )
    CellStore.create(root, manifest=manifest).close()

    reopened = CellStore.open(root)
    assert reopened.manifest == manifest
    assert reopened.manifest.rung == rung
    assert reopened.manifest.extract_system == extract_system
    assert reopened.manifest.bind_system == bind_system
