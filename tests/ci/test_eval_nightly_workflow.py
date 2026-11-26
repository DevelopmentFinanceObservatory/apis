"""Check the eval-nightly run record steps without running an evaluation.

The nightly writes its run record under ``$RUNNER_TEMP/eval-records`` and two steps after
it copy the record to the eval runs bucket as the eval runs writer, whatever the run's exit
code. The upload step's shell runs here under bash, with ``uv`` stubbed; the upload itself
is ``search.eval.record_upload``, tested in ``tests/eval/test_record_upload.py``.
"""

import os
import subprocess
from pathlib import Path

import yaml

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_WORKFLOW = yaml.safe_load((_WORKFLOWS / "eval-nightly.yml").read_text())
_SYNC_WORKFLOW = yaml.safe_load((_WORKFLOWS / "eval-sync.yml").read_text())
_STEPS = _WORKFLOW["jobs"]["nightly"]["steps"]
_RECORDS = '--records "$RUNNER_TEMP/eval-records"'


def _step(name: str) -> dict:
    """The nightly job's step whose name holds ``name``."""
    return next(step for step in _STEPS if name in step.get("name", ""))


def test_the_nightly_writes_its_record_and_names_its_langfuse_project():
    run = _step("Run the nightly eval")

    assert _RECORDS in run["run"]
    assert run["env"]["LANGFUSE_PROJECT_ID"] == "${{ vars.LANGFUSE_PROJECT_ID }}"


def test_the_record_is_uploaded_as_the_writer_after_the_run_whatever_its_code():
    names = [step.get("name", "") for step in _STEPS]
    auth = _step("eval runs writer")
    upload = _step("Upload the run record")

    assert names.index(auth["name"]) == names.index("Run the nightly eval") + 1
    assert names.index(upload["name"]) == names.index(auth["name"]) + 1
    assert auth["if"] == upload["if"] == "${{ !cancelled() }}"
    assert auth["uses"] == "google-github-actions/auth@v3"
    assert auth["with"] == {
        "workload_identity_provider": "${{ secrets.EVAL_WIF_PROVIDER }}",
        "service_account": "${{ secrets.EVAL_RUNS_WRITER_SA }}",
    }
    assert upload["env"]["BUCKET"] == "${{ vars.EVAL_RUNS_BUCKET }}"


def _upload(tmp_path: Path, run_ids: list[str]) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the upload step on a records root holding ``run_ids``; return the result and
    each ``uv`` call's arguments."""
    runner_temp = tmp_path / "runner"
    for run_id in run_ids:
        run_dir = runner_temp / "eval-records" / "apis-search" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "results.jsonl").write_text("{}\n")
        (run_dir / "manifest.json").write_text("{}\n")
    runner_temp.mkdir(exist_ok=True)
    calls = tmp_path / "calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(f'#!/bin/sh\necho "$*" >> "{calls}"\n')
    uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
        "BUCKET": "eval-runs",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }
    done = subprocess.run(
        ["bash", "-c", _step("Upload the run record")["run"]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done, calls.read_text().splitlines() if calls.exists() else []


def test_the_upload_hands_the_records_root_to_the_create_only_uploader(tmp_path):
    done, calls = _upload(tmp_path, ["nightly-20261125T041700Z"])
    root = tmp_path / "runner" / "eval-records"

    assert done.returncode == 0, done.stderr
    assert calls == [f"run python -m search.eval.record_upload --bucket eval-runs {root}"]
    assert "gcloud" not in _step("Upload the run record")["run"]


def test_the_upload_skips_with_a_notice_when_no_record_was_written(tmp_path):
    done, calls = _upload(tmp_path, [])

    assert done.returncode == 0, done.stderr
    assert calls == []
    assert "::notice::The nightly wrote no run record" in done.stdout


def test_the_nightly_and_the_corpus_sync_never_run_at_once():
    """Both write the Langfuse datasets, and a sync during the nightly would change the
    item inputs its push checks."""
    assert _WORKFLOW["concurrency"] == _SYNC_WORKFLOW["concurrency"]
    assert _WORKFLOW["concurrency"]["cancel-in-progress"] is False
    # Without a queue a group keeps one pending run, and a newer one cancels it.
    assert _WORKFLOW["concurrency"]["queue"] == "max"
