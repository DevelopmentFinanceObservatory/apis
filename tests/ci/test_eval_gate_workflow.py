"""Check the eval-gate triggers, jobs and verdict without running an evaluation.

The gate runs on opened, synchronize and reopened only. An `edited` run that skipped
its jobs would replace the failed check on a commit with a skipped one, which GitHub
counts as satisfied; ci.yml records the same hazard. The verdict job's shell step that
combines the shards' exit codes runs here under bash, with `gh` stubbed.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "eval-gate.yml"
_WORKFLOW = yaml.safe_load(_WORKFLOW_PATH.read_text())
# PyYAML reads the bare `on` key as boolean true.
_TRIGGERS = _WORKFLOW[True]
# The eval reads dev's deployed builds, so every job runs in ONEcampaign only.
_ONECAMPAIGN_ONLY = "github.repository_owner == 'ONEcampaign'"


def _group(pr_number: int) -> str:
    """Render the concurrency group for a pull request event."""
    context = {"github.event.pull_request.number": str(pr_number)}
    return re.sub(
        r"\$\{\{\s*([\w.]+)\s*\}\}",
        lambda match: context[match.group(1)],
        _WORKFLOW["concurrency"]["group"],
    )


def test_only_pull_requests_open_synchronize_and_reopen_trigger_the_gate():
    assert set(_TRIGGERS) == {"pull_request"}
    assert set(_TRIGGERS["pull_request"]["types"]) == {"opened", "synchronize", "reopened"}


def test_runs_for_one_pr_share_one_group_and_cancel_in_progress():
    assert _group(7) == _group(7)
    assert _group(7) != _group(8)
    assert _WORKFLOW["concurrency"]["cancel-in-progress"] is True


def test_no_job_is_skipped_by_an_event_condition():
    """A job held back by an `if` on the event reports as skipped on the commit."""
    conditions = [str(job.get("if", "")) for job in _WORKFLOW["jobs"].values()]
    assert not any("github.event" in condition for condition in conditions)


def test_a_head_app_that_died_during_the_run_fails_the_run_job():
    """After the driver, a dead head app exits 1, which no label releases; a dead base
    app only warns, so the driver's code stands."""
    run = next(
        step["run"] for step in _WORKFLOW["jobs"]["run"]["steps"] if step.get("id") == "eval"
    )
    after_driver = run[run.index("scripts/eval_paired.py") :]
    head_branch = after_driver.split('kill -0 "$head_pid"')[1].split("\nfi\n")[0]
    base_branch = after_driver.split('kill -0 "$base_pid"')[1].split("\nfi\n")[0]

    assert re.search(r"^\s*exit 1$", head_branch, re.M)
    assert not re.search(r"^\s*exit\b", base_branch, re.M)
    assert after_driver.index('kill -0 "$base_pid"') < after_driver.index('case "$rc"')


def _steps(job: str) -> list[dict]:
    return _WORKFLOW["jobs"][job]["steps"]


def test_the_reset_job_clears_the_override_without_running_pr_code():
    reset = _WORKFLOW["jobs"]["reset"]

    assert reset["permissions"] == {"pull-requests": "write"}
    assert not any("uses" in step for step in reset["steps"])
    assert reset["if"] == _ONECAMPAIGN_ONLY
    assert "reset" in _WORKFLOW["jobs"]["verdict"]["needs"]


def test_each_side_has_its_own_resolvekit_cache_with_no_fallback_key():
    caches = {
        step["with"]["path"]: step["with"]
        for step in _steps("run")
        if str(step.get("uses", "")).startswith("actions/cache")
    }

    assert len(caches) == 2
    assert all("restore-keys" not in spec for spec in caches.values())
    assert len({spec["key"] for spec in caches.values()}) == 2
    run = next(step["run"] for step in _steps("run") if step.get("id") == "eval")
    launches = re.findall(r'RESOLVEKIT_CACHE_DIR="(\$\w+)"', run)
    assert len(launches) == 2 and launches[0] != launches[1]


def test_no_eval_workflow_persists_checkout_credentials():
    workflows = sorted(_WORKFLOW_PATH.parent.glob("eval-*.yml"))
    checkouts = [
        step
        for path in workflows
        for job in yaml.safe_load(path.read_text())["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout")
    ]

    assert len(workflows) == 3 and len(checkouts) == 4
    assert all(step["with"]["persist-credentials"] is False for step in checkouts)


def test_a_pr_that_changes_the_eval_says_so_in_the_summary():
    note = next(step for step in _steps("changes") if "eval itself" in step.get("run", ""))

    assert "GITHUB_STEP_SUMMARY" in note["run"]
    assert "src/search/eval/" in note["run"]


def test_eval_workflows_authenticate_with_repository_secrets():
    """A pull request job cannot read the `dev` environment's WIF_PROVIDER and WIF_SA."""
    for name in ("eval-gate.yml", "eval-nightly.yml"):
        text = (_WORKFLOW_PATH.parent / name).read_text()
        assert "secrets.EVAL_WIF_PROVIDER" in text and "secrets.EVAL_WIF_SA" in text
        assert not re.search(r"secrets\.WIF_(PROVIDER|SA)", text)


def test_one_changes_job_gates_a_matrix_of_shards_that_do_not_cancel_each_other():
    jobs = _WORKFLOW["jobs"]
    run = jobs["run"]

    assert jobs["changes"]["if"] == _ONECAMPAIGN_ONLY
    assert set(jobs["changes"]["outputs"]) == {"eval", "shards"}
    assert run["needs"] == "changes"
    assert run["if"] == "needs.changes.outputs.eval == 'true'"
    assert run["strategy"]["fail-fast"] is False
    assert run["strategy"]["matrix"]["shard"] == "${{ fromJSON(needs.changes.outputs.shards) }}"
    assert "matrix.shard" in run["name"]
    assert not any("steps.filter" in str(step.get("if", "")) for step in _steps("run"))


def test_the_matrix_numbers_the_shards_from_1_and_labels_them_by_the_same_number(tmp_path):
    step = next(step for step in _steps("changes") if step.get("id") == "shards")
    output = tmp_path / "output"
    env = {**os.environ, "EVAL_SHARDS": "4", "GITHUB_OUTPUT": str(output)}

    subprocess.run(["bash", "-c", step["run"]], env=env, check=True)

    shards = json.loads(output.read_text().removeprefix("shards="))
    assert [(s["index"], s["label"]) for s in shards] == [(k, f"{k}/4") for k in range(1, 5)]


def test_each_shard_runs_its_own_slice_and_uploads_its_code_under_its_index():
    eval_step = next(step for step in _steps("run") if step.get("id") == "eval")
    upload = next(
        step for step in _steps("run") if str(step.get("uses", "")).startswith("actions/upload")
    )

    assert eval_step["env"]["SHARD"] == "${{ matrix.shard.index }}"
    assert '--shard "$SHARD/$EVAL_SHARDS"' in eval_step["run"]
    assert '--concurrency "$EVAL_CONCURRENCY"' in eval_step["run"]
    assert upload["with"]["name"] == "eval-gate-${{ matrix.shard.index }}"
    assert upload["if"] == "always()"
    # No deployed build, a base app that did not serve, and the driver's own code.
    assert eval_step["run"].count('> "$out/code"') == 3
    assert "GITHUB_OUTPUT" not in eval_step["run"]


def test_the_verdict_job_keeps_its_name_and_waits_for_every_job():
    verdict = _WORKFLOW["jobs"]["verdict"]

    assert verdict["name"] == "Eval gate"
    assert set(verdict["needs"]) == {"reset", "changes", "run"}
    assert verdict["if"] == f"always() && {_ONECAMPAIGN_ONLY}"
    assert verdict["permissions"] == {"pull-requests": "read"}
    download = next(
        step
        for step in verdict["steps"]
        if str(step.get("uses", "")).startswith("actions/download")
    )
    assert download["with"]["pattern"] == "eval-gate-*"


def _combine(tmp_path: Path, codes: list[str | None], *, labels: str = "") -> tuple[int, str]:
    """Run the verdict job's combining step on shard directories holding ``codes``; None
    is a shard that wrote no code. ``gh`` prints ``labels``."""
    step = next(
        step for step in _WORKFLOW["jobs"]["verdict"]["steps"] if "combined" in step["name"]
    )
    shards = tmp_path / "shards"
    for index, code in enumerate(codes, start=1):
        (shards / f"eval-gate-{index}").mkdir(parents=True)
        if code is not None:
            (shards / f"eval-gate-{index}" / "code").write_text(f"{code}\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!/bin/sh\nprintf '%s' '{labels}'\n")
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "EVAL_SHARDS": str(len(codes)),
        "SHARDS_DIR": str(shards),
        "LABELS": "repos/o/r/issues/1/labels",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }
    done = subprocess.run(
        ["bash", "-c", step["run"]], env=env, capture_output=True, text=True, check=False
    )
    return done.returncode, done.stdout


@pytest.mark.parametrize(
    ("codes", "message"),
    [
        (["0", "0", "0", "0"], None),
        (["0", "3", "0", "0"], "Eval exited 3 (an infrastructure failure)"),
        (["3", "1", "0", "3"], "Eval exited 1 (a blocking verdict)"),
        (["0", None, "0", "0"], "Shard 2 wrote no exit code"),
        (["0", "2", "0", "0"], "Shard 2 wrote the code '2'"),
    ],
    ids=["all pass", "any infra", "a block outranks infra", "missing code", "crash code"],
)
def test_the_verdict_combines_the_shard_codes(tmp_path, codes, message):
    returncode, out = _combine(tmp_path, codes)

    assert returncode == (0 if message is None else 1)
    if message is not None:
        assert f"::error::{message}" in out


def test_the_override_label_releases_the_combined_code(tmp_path):
    returncode, out = _combine(tmp_path, ["0", "1"], labels="eval-override")

    assert returncode == 0
    assert "::warning::Eval exited 1" in out
