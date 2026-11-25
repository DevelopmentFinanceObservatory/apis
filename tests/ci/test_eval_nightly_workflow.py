"""Check the eval-nightly run record steps without running an evaluation.

The nightly writes its run record under ``$RUNNER_TEMP/eval-records`` and two steps after
it copy the record to the eval runs bucket as the eval runs writer, whatever the run's exit
code. The upload step's shell runs here under bash, with ``gcloud`` stubbed.
"""

import os
import subprocess
from pathlib import Path

import yaml

_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "eval-nightly.yml"
_WORKFLOW = yaml.safe_load(_WORKFLOW_PATH.read_text())
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
    each ``gcloud`` call's arguments."""
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
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(f'#!/bin/sh\necho "$*" >> "{calls}"\n')
    gcloud.chmod(0o755)
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


def test_the_upload_copies_results_before_the_manifest_without_overwriting(tmp_path):
    done, calls = _upload(tmp_path, ["nightly-20261125T041700Z"])
    source = tmp_path / "runner" / "eval-records" / "apis-search" / "nightly-20261125T041700Z"
    dest = "gs://eval-runs/apis-search/nightly-20261125T041700Z"

    assert done.returncode == 0, done.stderr
    assert calls == [
        f"storage cp --no-clobber {source}/results.jsonl {dest}/results.jsonl",
        f"storage cp --no-clobber {source}/manifest.json {dest}/manifest.json",
    ]


def test_the_upload_skips_with_a_notice_when_no_record_was_written(tmp_path):
    done, calls = _upload(tmp_path, [])

    assert done.returncode == 0, done.stderr
    assert calls == []
    assert "::notice::The nightly wrote no run record" in done.stdout
