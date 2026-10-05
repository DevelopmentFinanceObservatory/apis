"""Checks the ladder CLI parser, dry-run paths, reports, and rung generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ladder.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))

import ladder  # noqa: E402

from promptlab.store import (  # noqa: E402
    BindRecord,
    CaptureManifest,
    CaptureStore,
    CellManifest,
    CellStore,
    ExtractRecord,
)
from search.engine.bind import _build_universal_bind_prompt  # noqa: E402
from search.engine.shape import ShapeDraft  # noqa: E402

_FIXTURES = Path(__file__).parent / "fixtures"


def _case_dict(case_id: str, *, why: str = "A stub case.") -> dict:
    return {"id": case_id, "query": f"query {case_id}", "why": why, "expect": {"definite": {}}}


def _write_corpus(corpus_dir: Path, *case_ids: str, why: str = "A stub case.") -> Path:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    cases = [_case_dict(case_id, why=why) for case_id in case_ids]
    (corpus_dir / "standard.yaml").write_text(yaml.safe_dump(cases))
    return corpus_dir


def _make_capture(
    runs_root: Path,
    *,
    capture_id: str,
    model: str = "gemini-3.5-flash-lite",
    top_k: int = 8,
    corpus_sha256: str | None = None,
) -> CaptureStore:
    # Default to the hash of the corpus the ladder selects.
    if corpus_sha256 is None:
        corpus_sha256 = ladder._corpus_sha256()
    manifest = CaptureManifest(
        capture_id=capture_id,
        git_sha="abc123",
        engine_build_id="test-build",
        model=model,
        top_k=top_k,
        substituted_date="2026-08-30",
        graph_base="https://graph.example",
        corpus_sha256=corpus_sha256,
        seeds=(1, 2, 3),
        extract_system_sha1="e" * 40,
        bind_system_sha1="b" * 40,
    )
    return CaptureStore.create(runs_root / "capture" / capture_id, manifest=manifest)


def _make_cell_manifest(
    cell_id: str, *, capture_id: str = "cap", rung_name: str = "r"
) -> CellManifest:
    return CellManifest(
        cell_id=cell_id,
        rung={"name": rung_name},
        capture_id=capture_id,
        git_sha="abc123",
        model="test-model",
        seeds=(1,),
        extract_system="",
        bind_system="",
    )


def _make_complete_capture(runs_root: Path, **kwargs: object) -> CaptureStore:
    store = _make_capture(runs_root, **kwargs)
    store.append_extract(
        ExtractRecord(
            case_id="g1", query="a query", system_sha1="s", output={}, usage=None, elapsed_ms=0
        )
    )
    store.finish()
    return store


def _one_shape_bind_prompt(variable: str = "gdp per capita") -> str:
    # Build a one-shape prompt; its count is below the capture's `top_k`.
    shape = ShapeDraft(
        shape_id="s0",
        label="GDP per capita",
        pop_type_dcid="Country",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        slot_taxonomy={},
        representative_score=0.9,
    )
    return _build_universal_bind_prompt(variable, [shape])


def _make_complete_capture_with_bind_prompt(
    runs_root: Path, *, capture_id: str, prompt: str, **kwargs: object
) -> CaptureStore:
    store = _make_capture(runs_root, capture_id=capture_id, **kwargs)
    store.append_extract(
        ExtractRecord(
            case_id="g1", query="a query", system_sha1="s", output={}, usage=None, elapsed_ms=0
        )
    )
    store.append_bind(
        BindRecord(
            case_id="g1",
            leg=0,
            variable="gdp per capita",
            prompt=prompt,
            prompt_sha1="p" * 40,
            system_sha1="b" * 40,
            output={},
            usage=None,
            elapsed_ms=0,
        )
    )
    store.finish()
    return store


# ---------------------------------------------------------------------------
# Argument parsing -- no network, no execution
# ---------------------------------------------------------------------------


def test_parser_capture_defaults():
    args = ladder.build_parser().parse_args(["capture", "--id", "k8-test"])
    assert args.command == "capture"
    assert args.id == "k8-test"
    assert args.k == ladder.settings().bind_top_k
    assert args.file is None
    assert args.ids is None
    assert args.model == "gemini-3.5-flash-lite"
    assert args.concurrency_llm == 16
    assert args.resume is False
    assert args.runs_root is None


def test_parser_capture_overrides():
    args = ladder.build_parser().parse_args(
        [
            "capture",
            "--id",
            "k3-test",
            "--k",
            "3",
            "--file",
            "sdg",
            "--ids",
            "a,b,c",
            "--model",
            "test-model",
            "--concurrency-llm",
            "8",
            "--resume",
        ]
    )
    assert args.k == 3
    assert args.file == "sdg"
    assert args.ids == "a,b,c"
    assert args.model == "test-model"
    assert args.concurrency_llm == 8
    assert args.resume is True


def test_parser_capture_requires_id():
    with pytest.raises(SystemExit):
        ladder.build_parser().parse_args(["capture"])


def test_parser_run_defaults():
    args = ladder.build_parser().parse_args(["run", "production"])
    assert args.command == "run"
    assert args.rung == "production"
    assert args.mode == "e2e"
    assert args.capture is None
    assert args.seeds is None
    assert args.cell_id is None
    assert args.concurrency == 16
    assert args.resume is False
    assert args.dry_run is False


def test_parser_run_overrides_and_dry_run():
    args = ladder.build_parser().parse_args(
        [
            "run",
            "abl-bind-x",
            "--mode",
            "bind",
            "--capture",
            "k3-test",
            "--seeds",
            "1,2,3",
            "--cell-id",
            "my-cell",
            "--concurrency",
            "4",
            "--resume",
            "--dry-run",
        ]
    )
    assert args.rung == "abl-bind-x"
    assert args.mode == "bind"
    assert args.capture == "k3-test"
    assert args.seeds == "1,2,3"
    assert args.cell_id == "my-cell"
    assert args.concurrency == 4
    assert args.resume is True
    assert args.dry_run is True


def test_parser_run_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        ladder.build_parser().parse_args(["run", "production", "--mode", "bogus"])


def test_report_ablate_build_are_registered():
    """Report, ablate, and build reject missing required arguments cleanly."""
    for name in ("report", "ablate", "build"):
        with pytest.raises(SystemExit):
            ladder.build_parser().parse_args([name])
    args = ladder.build_parser().parse_args(["report", "cell-a", "cell-b"])
    assert args.command == "report"
    assert args.cell_ids == ["cell-a", "cell-b"]
    assert args.against is None
    assert args.format == "md"
    assert args.rescore is False
    args = ladder.build_parser().parse_args(
        ["ablate", "base-extract.json", "base-bind.json", "--capture", "k8-test"]
    )
    assert args.command == "ablate"
    assert args.granularity == "both"
    args = ladder.build_parser().parse_args(
        [
            "build",
            "base-extract.json",
            "base-bind.json",
            "--ranking",
            "ranking.json",
            "--capture",
            "k8-test",
        ]
    )
    assert args.command == "build"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_structured_door_ids_finds_only_the_other_doors():
    cases = [
        SimpleNamespace(id="a", door="raw_text"),
        SimpleNamespace(id="b", door="dataset_resubmit"),
        SimpleNamespace(id="c", door="parsed"),
    ]
    assert ladder._structured_door_ids(cases) == ["b", "c"]


@pytest.mark.parametrize(
    ("model", "expected_conflict_terms"),
    [
        pytest.param("gemini-3.5-flash-lite", None, id="models-match"),
        pytest.param(
            "some-other-model",
            ("gemini-3.5-flash-lite", "some-other-model"),
            id="models-mismatch",
        ),
    ],
)
def test_capture_conflicts_checks_models(tmp_path, model, expected_conflict_terms):
    capture = _make_complete_capture(tmp_path / "ladder", capture_id="cap", model=model)
    rung = _load_production_rung()
    conflict = ladder._capture_conflicts(rung, capture)
    if expected_conflict_terms is None:
        assert conflict is None
    else:
        assert conflict is not None
        assert expected_conflict_terms[0] in conflict
        assert expected_conflict_terms[1] in conflict


def _load_production_rung():
    return ladder.load_rung(
        ladder.PROMPTLAB_ROOT / "rungs" / "production-extract.json",
        ladder.PROMPTLAB_ROOT / "rungs" / "production-bind.json",
    )


# ---------------------------------------------------------------------------
# dataset_resubmit exclusion, printed by name
# ---------------------------------------------------------------------------


def test_print_structured_door_exclusion_names_the_excluded_case(monkeypatch, capsys):
    monkeypatch.setenv("SEARCH_CORPUS_DIR", str(_FIXTURES / "structured_door_corpus"))

    ladder._print_structured_door_exclusion()

    out = capsys.readouterr().out
    assert "spec-01" in out
    assert "excluding 1 non-raw_text case" in out


def test_load_scoring_cases_excludes_other_doors(monkeypatch):
    monkeypatch.setenv("SEARCH_CORPUS_DIR", str(_FIXTURES / "structured_door_corpus"))

    cases = ladder._load_scoring_cases()

    assert {c.id for c in cases} == {"raw-01"}


def test_load_scoring_cases_filters_by_file_stem_and_refuses_an_unknown_one(monkeypatch):
    monkeypatch.setenv("SEARCH_CORPUS_DIR", str(_FIXTURES / "structured_door_corpus"))

    assert {c.id for c in ladder._load_scoring_cases(corpus_file="standard")} == {"raw-01"}
    with pytest.raises(ladder._CliError, match="no corpus case file 'sdg'"):
        ladder._load_scoring_cases(corpus_file="sdg")


# ---------------------------------------------------------------------------
# --dry-run, end to end against the landed production and rung0 manifests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rung_name", ["production", "rung0"])
def test_dry_run_against_landed_rungs(tmp_path, capsys, rung_name):
    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id="synthetic-capture")

    exit_code = ladder.main(
        [
            "run",
            rung_name,
            "--mode",
            "e2e",
            "--capture",
            "synthetic-capture",
            "--seeds",
            "5,6",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "extract system (" in out
    assert "bind system (" in out
    assert "seeds=[5, 6]" in out
    assert "capture 'synthetic-capture': complete" in out
    assert "1 case(s)" in out


def test_dry_run_defaults_capture_to_the_rungs_own_run_capture(tmp_path, capsys):
    rung = _load_production_rung()
    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id=rung.run.capture)

    exit_code = ladder.main(["run", "production", "--dry-run", "--runs-root", str(runs_root)])

    assert exit_code == 0
    assert f"capture {rung.run.capture!r}: complete" in capsys.readouterr().out


def test_dry_run_refuses_unknown_rung(tmp_path, capsys):
    exit_code = ladder.main(
        ["run", "no-such-rung", "--dry-run", "--runs-root", str(tmp_path / "ladder")]
    )
    assert exit_code == 1
    assert "no rung named 'no-such-rung'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Refusal paths
# ---------------------------------------------------------------------------


def test_dry_run_refuses_incomplete_capture(tmp_path, capsys):
    runs_root = tmp_path / "ladder"
    _make_capture(runs_root, capture_id="partial-capture")  # no .finish()

    exit_code = ladder.main(
        [
            "run",
            "production",
            "--capture",
            "partial-capture",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    assert exit_code == 1
    assert "not complete" in capsys.readouterr().err


def test_dry_run_refuses_missing_capture(tmp_path, capsys):
    exit_code = ladder.main(
        [
            "run",
            "production",
            "--capture",
            "never-captured",
            "--dry-run",
            "--runs-root",
            str(tmp_path / "ladder"),
        ]
    )
    assert exit_code == 1
    assert "never-captured" in capsys.readouterr().err


def test_dry_run_refuses_a_model_conflict(tmp_path, capsys):
    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id="wrong-model-capture", model="some-other-model")

    exit_code = ladder.main(
        [
            "run",
            "production",
            "--capture",
            "wrong-model-capture",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "some-other-model" in err
    assert "gemini-3.5-flash-lite" in err


def test_bind_dry_run_refuses_top_k_above_the_captures_own(tmp_path, capsys):
    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id="k3-capture", top_k=3)

    exit_code = ladder.main(
        [
            "run",
            "production",  # top_k=8
            "--mode",
            "bind",
            "--capture",
            "k3-capture",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "top_k=8" in err
    assert "top_k=3" in err


def test_bind_dry_run_succeeds_when_a_menu_is_shorter_than_top_k(tmp_path, capsys):
    # A one-shape menu remains valid when production `top_k` is 8.
    runs_root = tmp_path / "ladder"
    _make_complete_capture_with_bind_prompt(
        runs_root,
        capture_id="short-menu-capture",
        prompt=_one_shape_bind_prompt(),
        top_k=8,
    )

    exit_code = ladder.main(
        [
            "run",
            "production",
            "--mode",
            "bind",
            "--capture",
            "short-menu-capture",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "checked 1 bind record(s) against top_k=8: all parse" in out


def test_bind_dry_run_fails_cleanly_on_a_malformed_menu(tmp_path, capsys):
    runs_root = tmp_path / "ladder"
    _make_complete_capture_with_bind_prompt(
        runs_root,
        capture_id="malformed-menu-capture",
        prompt="not a bind prompt at all\n",
        top_k=8,
    )

    exit_code = ladder.main(
        [
            "run",
            "production",
            "--mode",
            "bind",
            "--capture",
            "malformed-menu-capture",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err
    assert "g1" in err


def test_capture_refuses_to_overwrite_an_existing_capture_without_resume(tmp_path):
    runs_root = tmp_path / "ladder"
    _make_capture(runs_root, capture_id="dup").close()

    with pytest.raises(ladder._CliError) as excinfo:
        ladder.cmd_capture(
            ladder.build_parser().parse_args(
                ["capture", "--id", "dup", "--runs-root", str(runs_root)]
            )
        )
    assert "already exists" in str(excinfo.value)
    assert "--resume" in str(excinfo.value)


def test_dry_run_refuses_a_corpus_hash_mismatch(tmp_path, capsys, monkeypatch):
    corpus_dir = _write_corpus(tmp_path / "corpus", "g1")
    monkeypatch.setenv("SEARCH_CORPUS_DIR", str(corpus_dir))

    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id="corpus-drift-capture")

    # Adding a case changes the capture's corpus hash.
    _write_corpus(corpus_dir, "g1", "g2")

    exit_code = ladder.main(
        [
            "run",
            "production",
            "--capture",
            "corpus-drift-capture",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "corpus-drift-capture" in err
    assert "corpus" in err


def test_dry_run_passes_when_only_a_cases_why_changes(tmp_path, capsys, monkeypatch):
    corpus_dir = _write_corpus(tmp_path / "corpus", "g1")
    monkeypatch.setenv("SEARCH_CORPUS_DIR", str(corpus_dir))

    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id="corpus-stable-capture")
    _write_corpus(corpus_dir, "g1", why="Rewritten prose, same expectation.")

    exit_code = ladder.main(
        [
            "run",
            "production",
            "--capture",
            "corpus-stable-capture",
            "--dry-run",
            "--runs-root",
            str(runs_root),
        ]
    )

    assert exit_code == 0


def test_run_rejects_a_malformed_seeds_value(tmp_path, capsys):
    exit_code = ladder.main(
        [
            "run",
            "production",
            "--seeds",
            "abc",
            "--dry-run",
            "--runs-root",
            str(tmp_path / "ladder"),
        ]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "--seeds" in err
    assert "abc" in err


def test_capture_wraps_a_store_error_into_a_cli_error(tmp_path, monkeypatch):
    runs_root = tmp_path / "ladder"

    def _boom(cls, root, *, manifest):
        raise ladder.StoreError("simulated store race")

    monkeypatch.setattr(ladder.CaptureStore, "create", classmethod(_boom))

    with pytest.raises(ladder._CliError, match="simulated store race"):
        ladder.cmd_capture(
            ladder.build_parser().parse_args(
                ["capture", "--id", "boom", "--runs-root", str(runs_root)]
            )
        )


def test_run_wraps_a_cell_store_error_into_a_cli_error(tmp_path, capsys, monkeypatch):
    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id="cap-for-boom")

    def _boom(cls, root, *, manifest):
        raise ladder.StoreError("simulated cell store race")

    monkeypatch.setattr(ladder.CellStore, "create", classmethod(_boom))

    exit_code = ladder.main(
        ["run", "production", "--capture", "cap-for-boom", "--runs-root", str(runs_root)]
    )

    assert exit_code == 1
    assert "simulated cell store race" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# report --rescore refusal paths
# ---------------------------------------------------------------------------


def test_rescore_help_discloses_persistence(capsys):
    with pytest.raises(SystemExit):
        ladder.build_parser().parse_args(["report", "--help"])
    out = capsys.readouterr().out
    assert "scores.json" in out
    assert "status.json" in out


@pytest.mark.parametrize(
    ("partial_cell_id", "reported_cell_id", "against_cell_id"),
    [
        pytest.param("partial-cell", "partial-cell", None, id="reported-partial"),
        pytest.param("ceiling-cell", "complete-cell", "ceiling-cell", id="against-ceiling-partial"),
    ],
)
def test_report_rescore_refuses_partial_cells(
    tmp_path, capsys, partial_cell_id, reported_cell_id, against_cell_id
):
    runs_root = tmp_path / "ladder"
    cells_root = runs_root / "cells"
    partial = CellStore.create(
        cells_root / partial_cell_id, manifest=_make_cell_manifest(partial_cell_id)
    )
    partial.append(case_id="g1", leg=0, seed=1, call={"stage": "bind", "scores": {}})
    partial.close()  # no finish() -- stays partial

    arguments = ["report", reported_cell_id]
    if against_cell_id is not None:
        complete = CellStore.create(
            cells_root / reported_cell_id, manifest=_make_cell_manifest(reported_cell_id)
        )
        complete.append(case_id="g1", leg=0, seed=1, call={"stage": "bind", "scores": {}})
        complete.finish({})
        arguments.extend(["--against", against_cell_id])
    arguments.extend(["--allow-partial", "--rescore", "--runs-root", str(runs_root)])

    exit_code = ladder.main(arguments)

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "cannot --rescore" in err
    assert partial_cell_id in err
    with CellStore.open(cells_root / partial_cell_id) as refused_target:
        assert not refused_target.is_complete()
        assert refused_target.status()["state"] == "partial"


def test_report_rescore_refuses_a_corpus_hash_mismatch(tmp_path, capsys, monkeypatch):
    corpus_dir = _write_corpus(tmp_path / "corpus", "g1")
    monkeypatch.setenv("SEARCH_CORPUS_DIR", str(corpus_dir))

    runs_root = tmp_path / "ladder"
    _make_complete_capture(runs_root, capture_id="corpus-drift-cap-2")

    cell_store = CellStore.create(
        runs_root / "cells" / "drift-cell",
        manifest=_make_cell_manifest("drift-cell", capture_id="corpus-drift-cap-2"),
    )
    cell_store.append(case_id="g1", leg=0, seed=1, call={"stage": "bind", "scores": {}})
    cell_store.finish({})

    # Add a case after capturing the original corpus.
    _write_corpus(corpus_dir, "g1", "g2")

    exit_code = ladder.main(["report", "drift-cell", "--rescore", "--runs-root", str(runs_root)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "corpus-drift-cap-2" in err
    assert "corpus" in err


# ---------------------------------------------------------------------------
# build --ranking (repeatable, one artifact per ceiling family)
# ---------------------------------------------------------------------------


def _ranking_report_json(rung_name: str) -> dict:
    """Build a minimal report with one banded metric for ranking."""
    return {
        "cells": [
            {
                "cell_id": "c1",
                "rung_name": rung_name,
                "n_seeds": 3,
                "metrics": {
                    "shape_election_match_rate": {"value": 0.5, "ceiling_band": [0.9, 0.95]}
                },
            }
        ]
    }


def _production_base_paths() -> tuple[str, str]:
    return (
        str(ladder.PROMPTLAB_ROOT / "rungs" / "production-extract.json"),
        str(ladder.PROMPTLAB_ROOT / "rungs" / "production-bind.json"),
    )


def test_build_merges_units_from_two_ranking_files(tmp_path, capsys):
    runs_root = tmp_path / "ladder"
    base_extract, base_bind = _production_base_paths()

    bind_ranking = tmp_path / "bind-ranking.json"
    bind_ranking.write_text(json.dumps(_ranking_report_json("abl-bind-also-plausible")))
    extract_ranking = tmp_path / "extract-ranking.json"
    extract_ranking.write_text(json.dumps(_ranking_report_json("abl-extract-04-scope-words")))

    exit_code = ladder.main(
        [
            "build",
            base_extract,
            base_bind,
            "--ranking",
            str(bind_ranking),
            "--ranking",
            str(extract_ranking),
            "--capture",
            "synthetic-capture",
            "--runs-root",
            str(runs_root),
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    # One unit from each ranking file: build-0 (rung0's own floor), plus one
    # step per merged unit -- both files' units landed in the same sequence.
    assert [line.strip() for line in out.splitlines() if line.strip().startswith("build-")] == [
        "build-0",
        "build-1",
        "build-2",
    ]
    written = sorted(p.name for p in (runs_root / "rungs").glob("build-*"))
    assert written == [
        "build-0-bind.json",
        "build-0-extract.json",
        "build-1-bind.json",
        "build-1-extract.json",
        "build-2-bind.json",
        "build-2-extract.json",
    ]


def test_build_two_rankings_sharing_a_unit_id_exits_1_with_a_clean_error(tmp_path, capsys):
    runs_root = tmp_path / "ladder"
    base_extract, base_bind = _production_base_paths()

    ranking_a = tmp_path / "ranking-a.json"
    ranking_a.write_text(json.dumps(_ranking_report_json("abl-bind-also-plausible")))
    ranking_b = tmp_path / "ranking-b.json"
    ranking_b.write_text(json.dumps(_ranking_report_json("abl-bind-also-plausible")))

    exit_code = ladder.main(
        [
            "build",
            base_extract,
            base_bind,
            "--ranking",
            str(ranking_a),
            "--ranking",
            str(ranking_b),
            "--capture",
            "synthetic-capture",
            "--runs-root",
            str(runs_root),
        ]
    )

    err = capsys.readouterr().err
    assert exit_code == 1
    assert err.startswith("error: ")
    assert "Traceback" not in err
    assert "also-plausible" in err
