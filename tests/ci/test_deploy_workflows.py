"""Check deploy triggers, guards, permissions, target selection, and step gates without a deploy.

deploy.yml holds the steps. deploy-dev.yml and deploy-prod.yml call it and choose only the
triggers and the GitHub Environment.
"""

import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "deploy_decide.sh"

_CALLER_PERMISSIONS = {"contents": "read", "id-token": "write", "actions": "read"}


def _workflow(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text())


def _triggers(workflow: dict) -> dict:
    """The `on:` mapping. PyYAML resolves the unquoted `on` key to True."""
    return workflow.get("on") or workflow[True]


def _steps() -> list[dict]:
    return _workflow("deploy.yml")["jobs"]["deploy"]["steps"]


def _step(step_id: str) -> dict:
    return next(step for step in _steps() if step.get("id") == step_id)


def _named(name: str) -> dict:
    return next(step for step in _steps() if step.get("name") == name)


def test_dev_deploys_after_a_successful_ci_run_on_dev():
    """Wait for successful CI so an older run cannot deploy after a newer commit."""
    workflow = _workflow("deploy-dev.yml")
    triggers = _triggers(workflow)

    assert "push" not in triggers, "a push trigger races CI for the same commit"
    workflow_run = triggers["workflow_run"]
    assert workflow_run["workflows"] == ["CI"]
    assert workflow_run["types"] == ["completed"]
    assert workflow_run["branches"] == ["dev"]
    assert "workflow_dispatch" in triggers

    job = workflow["jobs"]["deploy"]
    assert "success" in job["if"]
    assert "workflow_dispatch" in job["if"]
    assert "workflow_run.event == 'push'" in job["if"], "a fork PR from dev must not deploy"


def test_prod_deploys_on_a_push_to_main_or_a_dispatch():
    triggers = _triggers(_workflow("deploy-prod.yml"))

    assert triggers["push"] == {"branches": ["main"]}
    assert "workflow_dispatch" in triggers
    assert set(triggers) == {"push", "workflow_dispatch"}


@pytest.mark.parametrize(
    ("caller", "environment"),
    [("deploy-dev.yml", "dev"), ("deploy-prod.yml", "production")],
)
def test_each_caller_names_its_environment(caller: str, environment: str):
    """The branch and the trigger pick the environment."""
    job = _workflow(caller)["jobs"]["deploy"]

    assert job["uses"] == "./.github/workflows/deploy.yml"
    assert job["with"] == {"environment": environment}
    assert job["permissions"] == _CALLER_PERMISSIONS
    assert job["secrets"] == "inherit"


def test_prod_deploy_has_no_condition_beyond_its_triggers():
    """A push to main or a dispatch deploys, and no other condition gates the job."""
    assert "if" not in _workflow("deploy-prod.yml")["jobs"]["deploy"]


@pytest.mark.parametrize("path", sorted(_WORKFLOWS.glob("*.yml")), ids=lambda path: path.name)
def test_no_workflow_chooses_its_behaviour_by_repository_owner(path: Path):
    """The repository lives in one organisation, so events and branches decide what runs."""
    assert "repository_owner" not in path.read_text()


@pytest.mark.parametrize(
    ("caller", "group"),
    [("deploy-dev.yml", "deploy-dev"), ("deploy-prod.yml", "deploy-production")],
)
def test_deploys_serialise_and_every_pending_run_executes_in_order(caller: str, group: str):
    """A branch-wide group prevents overlapping migrations and service updates. `queue: max`
    keeps pending runs, so a queued dispatch is never cancelled by the next one.
    """
    concurrency = _workflow(caller)["concurrency"]

    assert concurrency["group"] == group
    assert concurrency["cancel-in-progress"] is False
    assert concurrency["queue"] == "max"


def test_reusable_job_runs_in_the_callers_environment_with_the_permissions_it_needs():
    """Tip CI queries need `actions: read`; WIF uses `id-token: write` for Cloud Run access."""
    workflow = _workflow("deploy.yml")
    call = _triggers(workflow)["workflow_call"]
    job = workflow["jobs"]["deploy"]

    assert call["inputs"]["environment"]["required"] is True
    assert job["environment"] == "${{ inputs.environment }}"
    assert job["permissions"] == _CALLER_PERMISSIONS


def test_target_resolution_checks_the_tip_when_it_differs_from_the_trigger():
    """A workflow_run targets the CI-tested commit only when it matches the branch's queried tip."""
    target = _step("target")
    run = target["run"]

    assert target["env"]["BRANCH"] == "${{ github.event.workflow_run.head_branch }}"
    assert 'gh api "repos/${{ github.repository }}/git/ref/heads/$BRANCH"' in run
    assert "--jq '.object.sha // empty'" in run
    assert "github.event.workflow_run.head_sha" in run
    assert "ci.yml/runs" in run
    assert "head_sha=$tip" in run
    assert "branch=$BRANCH" in run
    assert "event=push" in run
    assert "completed" in run
    assert "success" in run
    assert "::notice::" in run


def test_a_push_or_dispatch_deploys_the_commit_it_ran_for():
    """Anything but a workflow_run targets github.sha, skipping the tip/CI query."""
    run = _step("target")["run"]

    assert 'if [ "${{ github.event_name }}" != "workflow_run" ]; then' in run
    assert 'echo "sha=${{ github.sha }}"' in run


def test_every_step_names_the_resolved_target():
    """Checkout and build steps use the resolved target commit."""
    checkout_refs = [
        step["with"]["ref"]
        for step in _steps()
        if step.get("uses", "").startswith("actions/checkout")
    ]
    assert checkout_refs == ["${{ steps.target.outputs.sha }}"]

    assert "steps.target.outputs.sha" in _step("image")["run"]

    build = next(
        step for step in _steps() if step.get("uses", "").startswith("docker/build-push-action")
    )
    assert "DFO_BUILD_COMMIT=${{ steps.target.outputs.sha }}" in build["with"]["build-args"]
    assert build["with"]["tags"] == "${{ steps.image.outputs.image }}"
    assert "secrets" not in build["with"]


def test_every_name_comes_from_the_environment_variables():
    """No literal project, service, or job name survives, so prod never touches dev's resources."""
    text = (_WORKFLOWS / "deploy.yml").read_text()
    assert "dfo-dev" not in text
    assert "dev-fin-observatory" not in text

    image = _step("image")
    for name in (
        "PROJECT_ID",
        "REGION",
        "AR_REPO",
        "SERVICE_APIS",
        "SERVICE_WORKER",
        "JOB_DATASETS_MIGRATE",
        "JOB_PRESENTATION_MIGRATE",
        "JOB_ORIGIN_CATALOGUE",
        "JOB_SEARCH_CATALOGUE",
        "JOB_DATASETS_MAINTENANCE",
        "JOB_GRAPH_RELEASE",
    ):
        assert image["env"][name] == f"${{{{ vars.{name} }}}}"
        assert name in image["run"]

    targets = [
        target
        for step in _steps()
        for target in re.findall(
            r"gcloud run (?:jobs|services) [\w-]+ (\S+ \S+ \S+)", step.get("run", "")
        )
    ]
    # Six job updates, twelve job executions, two service updates, two reads of the apis
    # service, and four traffic updates.
    assert Counter(targets) == {
        f'"${{{{ vars.{name} }}}}"': count
        for name, count in {
            "JOB_DATASETS_MIGRATE": 2,
            "JOB_PRESENTATION_MIGRATE": 2,
            "JOB_DATASETS_MAINTENANCE": 5,
            "JOB_ORIGIN_CATALOGUE": 3,
            "JOB_GRAPH_RELEASE": 3,
            "JOB_SEARCH_CATALOGUE": 3,
            "SERVICE_WORKER": 1,
            "SERVICE_APIS": 7,
        }.items()
    }


@pytest.mark.parametrize(
    ("update", "execute"),
    [
        (
            "Point the datasets migration job at the new image",
            "Run the datasets migration job",
        ),
        (
            "Point the presentation migration job at the new image",
            "Run the presentation migration job",
        ),
        (
            "Point the origin catalogue job at the new image",
            "Ensure a compatible origin catalogue is published",
        ),
        (
            "Point the graph release job at the new image",
            "Ensure a compatible graph release is published",
        ),
        (
            "Point the search catalogue job at the new image",
            "Build the search catalogue on the new image",
        ),
    ],
)
def test_each_job_moves_to_the_new_image_before_it_executes(update: str, execute: str):
    """A fresh project's jobs start on a placeholder image, so an execute before the update
    would run the placeholder and report success."""
    names = [step.get("name") for step in _steps()]
    update_step = _named(update)
    match = re.search(r"gcloud run jobs update (\S+ \S+ \S+)", update_step["run"])
    assert match is not None
    job = match.group(1)

    assert "--image=" in update_step["run"]
    assert f"gcloud run jobs execute {job}" in _named(execute)["run"]
    assert names.index(update) < names.index(execute)


def test_the_build_fetches_the_public_spec_kit_without_a_credential():
    """The deploy mints no App token and passes the build no secret."""
    workflow = _workflow("deploy.yml")
    assert set(_triggers(workflow)["workflow_call"]) == {"inputs"}
    assert not any(step.get("uses", "").startswith("./.github/actions") for step in _steps())
    assert "secrets" not in _step("image").get("env", {})


def test_search_catalogue_builds_on_the_new_image_before_traffic_moves():
    """The build key includes the image's commit and a format bump starts an empty format
    root, so the deploy runs the job once on the new image and waits for it, after the
    origin catalogue and graph release it reads are ensured and before either service moves.
    The job exits 0 on some runs that publish nothing, so a verify execution after it decides
    whether a build is published; either failure stops the deploy."""
    names = [step.get("name") for step in _steps()]
    update = "Point the search catalogue job at the new image"
    build = "Build the search catalogue on the new image"
    executions = [
        step.get("name")
        for step in _steps()
        if 'jobs execute "${{ vars.JOB_SEARCH_CATALOGUE }}"' in step.get("run", "")
    ]
    update_run = _named(update)["run"]
    build_run = _named(build)["run"]

    assert 'gcloud run jobs update "${{ vars.JOB_SEARCH_CATALOGUE }}"' in update_run
    assert "--image=" in update_run
    assert executions == [build]
    assert "--async" not in build_run
    run_job, verify_job = build_run.split("gcloud run jobs execute ")[1:]
    assert "--wait" in run_job
    assert "--args" not in run_job
    assert "--wait" in verify_job
    assert "--args=-m,search.catalogue.verify" in verify_job
    assert build_run.lstrip().startswith("set -euo pipefail")
    assert (
        names.index("Ensure a compatible origin catalogue is published")
        < names.index("Ensure a compatible graph release is published")
        < names.index(update)
        < names.index(build)
        < names.index("Point the datasets worker at the new image")
        < names.index("Stage the new revision with no traffic")
    )


_FAKE_GCLOUD = """#!/bin/bash
echo "$*" >> "$GCLOUD_LOG"
case "$*" in
  *graph_release.verify*)
    statuses=($VERIFY_STATUSES)
    exit "${statuses[$(($(grep -c graph_release.verify "$GCLOUD_LOG") - 1))]}" ;;
esac
"""


@pytest.mark.parametrize(
    ("verify_statuses", "status", "executions"),
    [
        ("0", 0, ["verify"]),
        ("3 0", 0, ["verify", "build", "verify"]),
        ("1 1", 1, ["verify", "build", "verify"]),
    ],
    ids=["opens", "builds-then-opens", "still-refused"],
)
def test_the_graph_release_step_verifies_builds_once_and_verifies_again(
    tmp_path: Path, verify_statuses: str, status: int, executions: list[str]
):
    """A failed verify runs one build, and a second failed verify stops the deploy."""
    gcloud = tmp_path / "bin" / "gcloud"
    gcloud.parent.mkdir()
    gcloud.write_text(_FAKE_GCLOUD)
    gcloud.chmod(0o755)
    log = tmp_path / "gcloud.log"
    log.touch()
    # The step, with workflow expressions filled in as Actions would.
    step = re.sub(
        r"\$\{\{[^}]*\}\}", "dev", _named("Ensure a compatible graph release is published")["run"]
    )

    ran = subprocess.run(
        ["bash", "-c", step],
        env={
            "PATH": f"{gcloud.parent}:/usr/bin:/bin",
            "GCLOUD_LOG": str(log),
            "VERIFY_STATUSES": verify_statuses,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert ran.returncode == status, ran.stderr
    calls = log.read_text().splitlines()
    assert [
        "verify" if "--args=-m,graph_release.verify" in call else "build" for call in calls
    ] == executions
    assert all(call.startswith("run jobs execute") and "--wait" in call for call in calls)


def test_maintenance_job_moves_to_the_new_image_after_both_migrations():
    """Its next run, scheduled or the settle below, finds the schema the new image expects."""
    names = [step.get("name") for step in _steps()]
    update = "Point the datasets maintenance job at the new image"
    run = _named(update)["run"]

    assert 'gcloud run jobs update "${{ vars.JOB_DATASETS_MAINTENANCE }}"' in run
    assert "--image=" in run
    assert names.index("Run the datasets migration job") < names.index(update)
    assert names.index("Run the presentation migration job") < names.index(update)
    assert names.index(update) < names.index("Point the datasets worker at the new image")


def test_the_maintenance_job_settles_then_backfills_once_both_services_run_the_new_image():
    """The previous image writes rows in the old shape, and versions without a description
    record, until both services move, so the settle, the warning rewrite, the warning backfill
    and then the description backfill run after the apis service moves, and they are the
    deploy's only maintenance executions. The settle runs first, so its pin repair waits on
    none of the others."""
    names = [step.get("name") for step in _steps()]
    settle = "Settle rows the previous image wrote"
    rewrite = "Rewrite warnings in an old format"
    warnings = "Backfill warning counts"
    backfill = "Backfill description records"
    executions = [
        step.get("name")
        for step in _steps()
        if 'jobs execute "${{ vars.JOB_DATASETS_MAINTENANCE }}"' in step.get("run", "")
    ]

    assert executions == [settle, rewrite, warnings, backfill]
    assert "--args=settle" in _named(settle)["run"]
    assert "--args=rewrite-warnings" in _named(rewrite)["run"]
    assert "--args=backfill-warnings" in _named(warnings)["run"]
    assert "--args=backfill-descriptions" in _named(backfill)["run"]
    for name in executions:
        assert "--wait" in _named(name)["run"]
    assert names.index("Move traffic to the candidate revision") < names.index(settle)


def test_the_upfront_check_refuses_a_deploy_without_the_maintenance_job():
    """The settle step needs the maintenance job, so an environment that does not name it fails
    the check before the build, and the maintenance steps carry no guard of their own."""
    image = _step("image")
    # The check, with workflow expressions filled in as Actions would.
    check = re.sub(r"\$\{\{[^}]*\}\}", "dev", image["run"])
    env = dict.fromkeys(image["env"], "set") | {"JOB_DATASETS_MAINTENANCE": ""}
    unset = subprocess.run(
        ["bash", "-c", check],
        env=env | {"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": "/dev/null"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert unset.returncode == 1
    assert unset.stdout.splitlines() == [
        "::error::Missing vars.JOB_DATASETS_MAINTENANCE in the 'dev' GitHub Environment."
    ]
    for name in (
        "Point the datasets maintenance job at the new image",
        "Settle rows the previous image wrote",
        "Rewrite warnings in an old format",
        "Backfill warning counts",
        "Backfill description records",
    ):
        assert _named(name)["run"].lstrip().startswith("set -euo pipefail\ngcloud run jobs")


_REVISION = "${{ steps.stage.outputs.revision }}"
_PUSHED = "host/repo/dfo-apis@sha256:aaa"
_INDEX = json.dumps(
    {
        "manifests": [
            {"digest": "sha256:aaa", "platform": {"architecture": "amd64", "os": "linux"}},
            {"digest": "sha256:bbb", "platform": {"architecture": "unknown", "os": "unknown"}},
        ]
    }
)


def test_one_named_revision_is_staged_probed_given_traffic_and_checked():
    """Every deploy runs one order, and each step after the stage acts on the revision it
    named: probe the tag URL, move traffic to it by name, settle and backfill, follow the
    latest revision again without the tag, then check the released image serves."""
    names = [step.get("name") for step in _steps()]

    assert names[-13:] == [
        "Point the datasets worker at the new image",
        "Stage the new revision with no traffic",
        "Warm the candidate revision",
        "Fail on errors the candidate revision answered while warming",
        "Move traffic to the candidate revision",
        "Settle rows the previous image wrote",
        "Rewrite warnings in an old format",
        "Backfill warning counts",
        "Backfill description records",
        "Follow the latest revision again",
        "Check the released image serves",
        "Fail on a failed warning step",
        "Follow the latest revision after a failure",
    ]
    for name in ("Warm the candidate revision", "Move traffic to the candidate revision"):
        assert _named(name)["env"]["REVISION"] == _REVISION
    assert _named("Check the released image serves")["env"]["IMAGE"] == (
        "${{ steps.stage.outputs.image }}"
    )


_RECORDING_GCLOUD = """#!/bin/bash
echo "$*" >> "$GCLOUD_LOG"
case "$*" in
  *"revisions describe"*) echo "$REVISION_IMAGE" ;;
  *--format=json*) echo "$DESCRIBED" ;;
  *latestCreatedRevisionName*) echo "$LATEST" ;;
esac
"""

# `docker buildx imagetools inspect --raw` prints the pushed image index.
_DOCKER = """#!/bin/bash
echo "$INDEX"
"""


def _run_step(
    name: str, tmp_path: Path, expressions: dict[str, str] | None = None, **env: str
) -> tuple[subprocess.CompletedProcess, list]:
    """Runs a step's script with workflow expressions filled in as Actions would, from
    `expressions` or else as `svc`, against a gcloud that records each call and a docker that
    prints `INDEX`, and returns the gcloud calls without their region and project."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool, body in (("gcloud", _RECORDING_GCLOUD), ("docker", _DOCKER)):
        (bin_dir / tool).write_text(body)
        (bin_dir / tool).chmod(0o755)
    log = tmp_path / "gcloud.log"
    log.touch()
    script = _named(name)["run"]
    for expression, value in (expressions or {}).items():
        script = script.replace(f"${{{{ {expression} }}}}", value)
    script = re.sub(r"\$\{\{[^}]*\}\}", "svc", script)
    ran = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "GCLOUD_LOG": str(log), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    return ran, [call.split(" --region")[0] for call in log.read_text().splitlines()]


def test_the_digest_step_reads_the_index_and_its_linux_amd64_manifest(tmp_path: Path):
    """It runs right after the build, while the registry login is fresh."""
    output_file = tmp_path / "github_output"
    output_file.touch()
    names = [step.get("name") for step in _steps()]

    ran, _ = _run_step(
        "Read the pushed image's digests",
        tmp_path,
        {"steps.build.outputs.digest": "sha256:index"},
        INDEX=_INDEX,
        GITHUB_OUTPUT=str(output_file),
    )

    assert ran.returncode == 0, ran.stderr
    assert output_file.read_text() == "digests=sha256:index sha256:aaa\n"
    build = next(i for i, step in enumerate(_steps()) if step.get("id") == "build")
    assert names[build + 1] == "Read the pushed image's digests"


@pytest.mark.parametrize(
    ("latest", "image", "exit_code"),
    [
        ("svc-00002-bbb", _PUSHED, 0),
        # An `infra` apply during the update created the latest revision from the previous
        # image, so it is not probed.
        ("svc-00002-bbb", "host/repo/dfo-apis@sha256:old", 1),
        ("", _PUSHED, 1),
    ],
    ids=["pushed-image", "previous-image", "unreadable"],
)
def test_the_stage_step_tags_and_outputs_the_created_revision_only_when_it_runs_the_pushed_image(
    tmp_path: Path, latest: str, image: str, exit_code: int
):
    """The update leaves the template revision unnamed, since `--tag` would name it, and the
    tag goes on the revision the update created once its digest is one the digest step read
    after the build. The stage step reads no registry itself."""
    output_file = tmp_path / "github_output"
    output_file.touch()
    assert "imagetools" not in _named("Stage the new revision with no traffic")["run"]

    ran, calls = _run_step(
        "Stage the new revision with no traffic",
        tmp_path,
        {"steps.pushed.outputs.digests": "sha256:index sha256:aaa"},
        LATEST=latest,
        REVISION_IMAGE=image,
        GITHUB_OUTPUT=str(output_file),
    )

    assert ran.returncode == exit_code, ran.stderr
    released = exit_code == 0
    assert output_file.read_text() == (
        f"revision=svc-00002-bbb\nimage={image}\n" if released else ""
    )
    assert calls == [
        "run services update svc --image=svc --no-traffic",
        "run services describe svc",
        *(["run revisions describe svc-00002-bbb"] if latest else []),
        *(
            ["run services update-traffic svc --update-tags=candidate=svc-00002-bbb"]
            if released
            else []
        ),
    ]


@pytest.mark.parametrize(
    ("step", "call"),
    [
        (
            "Move traffic to the candidate revision",
            "run services update-traffic svc --to-revisions=svc-00002-bbb=100",
        ),
        (
            "Follow the latest revision again",
            "run services update-traffic svc --to-latest --remove-tags=candidate",
        ),
        # The tag stays, so the next run deploys.
        (
            "Follow the latest revision after a failure",
            "run services update-traffic svc --to-latest",
        ),
    ],
)
def test_traffic_moves_by_name_then_follows_the_latest_revision(
    tmp_path: Path, step: str, call: str
):
    ran, calls = _run_step(step, tmp_path, REVISION="svc-00002-bbb")

    assert ran.returncode == 0, ran.stderr
    assert calls == [call]


def test_a_failure_after_the_move_makes_traffic_follow_the_latest_revision():
    """Settle or backfill can fail after traffic is pinned by name, and an `infra` revision
    must still take traffic until the next deploy."""
    step = _named("Follow the latest revision after a failure")

    assert step["if"] == (
        "failure() && steps.move.outcome == 'success' && steps.follow.outcome != 'success'"
    )
    assert _named("Move traffic to the candidate revision")["id"] == "move"
    assert _named("Follow the latest revision again")["id"] == "follow"


@pytest.mark.parametrize(
    ("rewrite", "backfill", "named"),
    [
        ("success", "success", []),
        ("failure", "success", ["Rewrite warnings in an old format"]),
        ("success", "failure", ["Backfill warning counts"]),
        ("failure", "failure", ["Rewrite warnings in an old format", "Backfill warning counts"]),
    ],
)
def test_a_failed_warning_step_fails_the_run_once_the_release_is_checked(
    rewrite: str, backfill: str, named: list[str]
):
    """A warning step that fails leaves the description backfill, the follow step and the
    release check to run, and the step after the release check fails the run naming it. The
    follow step has succeeded by then, so the recovery step after a failure stays idle."""
    rewrite_step = _step("rewrite_warnings")
    backfill_step = _step("backfill_warnings")
    assert rewrite_step["name"] == "Rewrite warnings in an old format"
    assert backfill_step["name"] == "Backfill warning counts"
    for step in (rewrite_step, backfill_step):
        assert step["continue-on-error"] is True
    names = [step.get("name") for step in _steps()]
    last = _named("Fail on a failed warning step")
    assert names.index("Check the released image serves") + 1 == names.index(last["name"])
    assert last["env"] == {
        "REWRITE_WARNINGS": "${{ steps.rewrite_warnings.outcome }}",
        "BACKFILL_WARNINGS": "${{ steps.backfill_warnings.outcome }}",
    }

    check = subprocess.run(
        ["bash", "-c", last["run"]],
        env={"PATH": "/usr/bin:/bin", "REWRITE_WARNINGS": rewrite, "BACKFILL_WARNINGS": backfill},
        capture_output=True,
        text=True,
        check=False,
    )

    assert check.returncode == (1 if named else 0)
    errors = [line for line in check.stdout.splitlines() if line.startswith("::error::")]
    assert [name for name in named if any(f"'{name}'" in line for line in errors)] == named
    assert len(errors) == len(named)


@pytest.mark.parametrize(
    ("entry", "image", "exit_code"),
    [
        ({"percent": 100, "latestRevision": True, "revisionName": "svc-00002-bbb"}, _PUSHED, 0),
        # Traffic follows the latest revision, which `infra` created after the move from the
        # released image.
        ({"percent": 100, "latestRevision": True, "revisionName": "svc-00003-ccc"}, _PUSHED, 0),
        # An `infra` apply that overlapped the deploy put the previous image back.
        (
            {"percent": 100, "latestRevision": True, "revisionName": "svc-00001-aaa"},
            "host/repo/dfo-apis@sha256:old",
            1,
        ),
        # An `infra` apply that refreshed during settle pinned traffic by name again.
        ({"percent": 100, "revisionName": "svc-00002-bbb"}, _PUSHED, 1),
        ({"percent": 50, "latestRevision": True, "revisionName": "svc-00002-bbb"}, _PUSHED, 1),
    ],
    ids=["released", "infra-revision", "previous-image", "pinned", "split"],
)
def test_the_run_fails_unless_the_released_image_serves(
    tmp_path: Path, entry: dict, image: str, exit_code: int
):
    ran, calls = _run_step(
        "Check the released image serves",
        tmp_path,
        IMAGE=_PUSHED,
        REVISION_IMAGE=image,
        DESCRIBED=json.dumps({"status": {"traffic": [entry]}}),
    )

    assert ran.returncode == exit_code, ran.stderr
    assert calls[0] == "run services describe svc"


@pytest.mark.parametrize(
    ("status", "required", "exit_code"),
    [
        ("error", "false", 1),
        ("warm", "true", 0),
        ("forbidden", "false", 0),
        ("slow", "false", 0),
        # Production moves traffic only on `warm`.
        ("forbidden", "true", 1),
        # A missing status is not a pass.
        ("", "false", 1),
    ],
)
def test_the_deploy_fails_on_a_warm_error_before_traffic_moves(
    status: str, required: str, exit_code: int
):
    """The warm step calls the candidate revision as the deployer and records its outcome
    without failing; the next step fails the deploy unless the outcome lets traffic move,
    ahead of the traffic step."""
    warm = _step("warm")
    assert warm["run"].strip() == ".github/scripts/deploy_warm.sh"
    assert warm["env"]["SERVICE"] == "${{ vars.SERVICE_APIS }}"
    assert warm["env"]["DEPLOYER"] == "${{ secrets.WIF_SA }}"
    assert warm["env"]["PROBE_TAG"] == "candidate"
    assert warm["env"]["REQUIRE_PROBE"] == "${{ inputs.environment == 'production' }}"
    assert "continue-on-error" not in warm

    last = _named("Fail on errors the candidate revision answered while warming")
    assert last["env"]["WARM_STATUS"] == "${{ steps.warm.outputs.status }}"
    assert last["env"]["REQUIRE_PROBE"] == warm["env"]["REQUIRE_PROBE"]
    check = subprocess.run(
        ["bash", "-c", last["run"]],
        env={"PATH": "/usr/bin:/bin", "WARM_STATUS": status, "REQUIRE_PROBE": required},
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == exit_code


def test_decide_step_runs_gcloud_describe_and_git_diff_and_gates_every_later_step():
    """The decide step reads the service and gates every later step on `deploy=true`, so a
    run that deploys runs every step in order and a run that stopped part-way is recovered by
    the next one from the start. Only the recovery step after a failure runs on another
    condition."""
    decide = _step("decide")
    assert decide["run"].strip() == ".github/scripts/deploy_decide.sh"
    assert decide["env"]["TARGET"] == "${{ steps.target.outputs.sha }}"

    script = _SCRIPT.read_text()
    assert "gcloud run services describe" in script
    assert "git diff --quiet" in script
    assert "exit 1" in script, "an empty or failed describe call is a hard failure"

    steps = _steps()
    later = steps[steps.index(decide) + 1 :]
    assert len(later) == 28
    for step in later:
        if step.get("name") != "Follow the latest revision after a failure":
            assert step["if"] == "steps.decide.outputs.deploy == 'true'", step


def test_decide_step_forces_deploy_on_workflow_dispatch():
    """Without FORCE, the decide step's live-vs-target comparison can decide a dispatched commit
    needs no deploy, which defeats dispatch's purpose of forcing a redeploy."""
    assert _step("decide")["env"]["FORCE"] == "${{ github.event_name == 'workflow_dispatch' }}"


def test_no_paths_filter_or_parent_resolution_remain():
    """Deployment decisions compare the target with the commit Cloud Run serves."""
    for step in _steps():
        assert not step.get("uses", "").startswith("dorny/paths-filter")

    step_ids = {step.get("id") for step in _steps()}
    assert not step_ids & {"parent", "tip", "filter", "commit"}
