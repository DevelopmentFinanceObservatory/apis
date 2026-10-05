"""Check the eval-gate triggers, jobs and verdict without running an evaluation.

The gate runs when a non-draft PR is opened or reopened, when a draft is marked ready,
and when eval-request.yml dispatches it for the `run-eval` label. A push or a label
event never starts it: a run that skipped its jobs would replace the failed check on a
commit with a skipped one, which GitHub counts as satisfied; ci.yml records the same
hazard. The verdict job's shell step that combines the shards' exit codes runs here
under bash, with `gh` stubbed.
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
_REQUEST = yaml.safe_load((_WORKFLOW_PATH.parent / "eval-request.yml").read_text())
# The eval reads dev's deployed builds, so every job runs in ONEcampaign only.
_ONECAMPAIGN_ONLY = "github.repository_owner == 'ONEcampaign'"
# Every job skips on a draft; a dispatched run has no pull_request payload.
_NOT_DRAFT = "github.event.pull_request.draft != true"
_PR_NUMBER = "github.event.pull_request.number || inputs.pr"


def _clauses(condition: str) -> set[str]:
    """Split a job's ``if`` into its top-level ``&&`` clauses."""
    return {clause.strip() for clause in condition.split("&&")}


def test_neither_a_push_nor_a_label_starts_the_gate():
    pull_request = set(_TRIGGERS["pull_request"]["types"])

    assert set(_TRIGGERS) == {"pull_request", "workflow_dispatch"}
    assert {"synchronize", "labeled", "edited"}.isdisjoint(pull_request)
    assert "ready_for_review" in pull_request
    pr_input = _TRIGGERS["workflow_dispatch"]["inputs"]["pr"]
    assert pr_input["required"] is True and pr_input["type"] == "number"


def test_runs_for_one_pr_share_one_group_on_either_trigger():
    assert _WORKFLOW["env"]["PR_NUMBER"] == f"${{{{ {_PR_NUMBER} }}}}"
    assert _WORKFLOW["concurrency"]["group"] == f"eval-gate-${{{{ {_PR_NUMBER} }}}}"
    assert _WORKFLOW["concurrency"]["cancel-in-progress"] is True
    assert "github.event.pull_request.number" not in yaml.safe_dump(_WORKFLOW["jobs"])


def test_every_job_skips_on_a_draft():
    """A pull_request run on a draft skips every job, the verdict included."""
    for name, job in _WORKFLOW["jobs"].items():
        assert _NOT_DRAFT in _clauses(job["if"]), name


def test_a_dispatched_run_evaluates_the_merge_of_the_head_it_reports_on():
    changes_checkout = _steps("changes")[0]
    run_checkout = _steps("run")[0]
    check = next(step for step in _steps("changes") if step.get("id") == "merge")

    assert "refs/pull/{0}/merge" in changes_checkout["with"]["ref"]
    assert "workflow_dispatch" in changes_checkout["with"]["ref"]
    assert run_checkout["with"]["ref"] == "${{ needs.changes.outputs.merge }}"
    assert _WORKFLOW["jobs"]["changes"]["outputs"]["merge"] == "${{ steps.merge.outputs.sha }}"
    assert check["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    assert all(step["with"]["fetch-depth"] == 2 for step in (changes_checkout, run_checkout))


def _merge_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a repo whose HEAD merges a branch into its base; return it, the merged
    branch tip, and the base tip."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    git("init", "-q", "-b", "base")
    git("commit", "-q", "--allow-empty", "-m", "base")
    # Not "head": on a case-insensitive filesystem that branch shadows HEAD.
    git("checkout", "-q", "-b", "topic")
    git("commit", "-q", "--allow-empty", "-m", "head")
    head = git("rev-parse", "HEAD")
    git("checkout", "-q", "base")
    git("commit", "-q", "--allow-empty", "-m", "base 2")
    base = git("rev-parse", "HEAD")
    git("merge", "-q", "--no-ff", "-m", "merge", "topic")
    return repo, head, base


def _check_merge(repo: Path, head_sha: str, script: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "HEAD_SHA": head_sha,
        "PR_NUMBER": "7",
        "GITHUB_OUTPUT": str(repo.parent / "output"),
    }
    return subprocess.run(
        ["bash", "-c", script], cwd=repo, env=env, capture_output=True, text=True, check=False
    )


def test_the_merge_check_refuses_a_merge_ref_that_does_not_merge_the_head(tmp_path):
    script = next(step for step in _steps("changes") if step.get("id") == "merge")["run"]
    repo, head, base = _merge_repo(tmp_path)

    assert _check_merge(repo, head, script).returncode == 0
    assert (tmp_path / "output").read_text().startswith("sha=")
    stale = _check_merge(repo, base, script)
    assert stale.returncode == 1
    assert "::error::refs/pull/7/merge merges" in stale.stdout


def test_eval_request_dispatches_only_for_run_eval_on_an_open_same_repo_pr():
    triggers = _REQUEST[True]
    jobs = _REQUEST["jobs"]
    (job,) = jobs.values()

    assert triggers["pull_request"]["types"] == ["labeled"]
    assert _clauses(job["if"]) == {
        _ONECAMPAIGN_ONLY,
        "github.event.label.name == 'run-eval'",
        "github.event.pull_request.state == 'open'",
        "github.event.pull_request.head.repo.full_name == github.repository",
    }
    assert job["name"] != _WORKFLOW["jobs"]["verdict"]["name"]
    assert job["permissions"] == {"actions": "write", "pull-requests": "write"}
    (step,) = job["steps"]
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert step["env"]["HEAD_REF"] == "${{ github.event.pull_request.head.ref }}"
    assert (
        'gh workflow run eval-gate.yml --repo "$GITHUB_REPOSITORY" --ref "$HEAD_REF" -f "pr=$PR_NUMBER"'
        in step["run"]
    )
    assert '"repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/labels/run-eval"' in step["run"]


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
    assert _clauses(reset["if"]) == {_ONECAMPAIGN_ONLY, _NOT_DRAFT}
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

    assert len(workflows) == 4 and len(checkouts) == 4
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

    assert _clauses(jobs["changes"]["if"]) == {_ONECAMPAIGN_ONLY, _NOT_DRAFT}
    assert set(jobs["changes"]["outputs"]) == {"eval", "merge", "shards"}
    assert run["needs"] == "changes"
    assert _clauses(run["if"]) == {"needs.changes.outputs.eval == 'true'", _NOT_DRAFT}
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
    assert _clauses(verdict["if"]) == {"always()", _ONECAMPAIGN_ONLY, _NOT_DRAFT}
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
