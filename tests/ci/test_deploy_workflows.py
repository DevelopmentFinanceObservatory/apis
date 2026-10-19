"""Check deploy triggers, guards, permissions, target selection, and step gates without a deploy.

deploy.yml holds the steps. deploy-dev.yml and deploy-prod.yml call it and choose only the
triggers and the GitHub Environment.
"""

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
        "JOB_COVERAGE",
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
            r"gcloud run (?:jobs|services) \w+ (\S+ \S+ \S+)", step.get("run", "")
        )
    ]
    # Seven job updates, ten job executions, and two service updates.
    assert Counter(targets) == {
        f'"${{{{ vars.{name} }}}}"': count
        for name, count in {
            "JOB_DATASETS_MIGRATE": 2,
            "JOB_PRESENTATION_MIGRATE": 2,
            "JOB_DATASETS_MAINTENANCE": 3,
            "JOB_COVERAGE": 5,
            "JOB_ORIGIN_CATALOGUE": 3,
            "JOB_GRAPH_RELEASE": 1,
            "JOB_SEARCH_CATALOGUE": 1,
            "SERVICE_WORKER": 1,
            "SERVICE_APIS": 1,
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
            "Point the coverage job at the new image",
            "Ensure a compatible coverage artifact is published",
        ),
        (
            "Point the origin catalogue job at the new image",
            "Ensure a compatible origin catalogue is published",
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


def test_search_catalogue_job_moves_to_the_new_image_without_executing():
    """Its Scheduler rebuilds on the new image within 15 minutes, and the service starts
    without a catalogue, so the deploy only moves the image."""
    names = [step.get("name") for step in _steps()]
    update = "Point the search catalogue job at the new image"
    run = _named(update)["run"]

    assert 'gcloud run jobs update "${{ vars.JOB_SEARCH_CATALOGUE }}"' in run
    assert "--image=" in run
    for step in _steps():
        assert 'jobs execute "${{ vars.JOB_SEARCH_CATALOGUE }}"' not in step.get("run", "")
    assert names.index("Ensure a compatible origin catalogue is published") < names.index(update)
    assert names.index(update) < names.index("Point the datasets worker at the new image")


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
    record, until both services move, so the settle and then the description backfill run after
    the apis service moves, and they are the deploy's only maintenance executions. The backfill
    runs after the code index record, so its failure never holds that record back."""
    names = [step.get("name") for step in _steps()]
    settle = "Settle rows the previous image wrote"
    backfill = "Backfill description records"
    executions = [
        step.get("name")
        for step in _steps()
        if 'jobs execute "${{ vars.JOB_DATASETS_MAINTENANCE }}"' in step.get("run", "")
    ]

    assert executions == [settle, backfill]
    assert "--args=settle" in _named(settle)["run"]
    assert "--args=backfill-descriptions" in _named(backfill)["run"]
    for name in (settle, backfill):
        assert "--wait" in _named(name)["run"]
    assert names.index("Point Cloud Run at the new image") < names.index(settle)
    assert names.index("Record the code index the services run") < names.index(backfill)


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
        "Backfill description records",
    ):
        assert _named(name)["run"].lstrip().startswith("set -euo pipefail\ngcloud run jobs")


def test_worker_then_service_move_then_the_code_index_is_recorded():
    names = [step.get("name") for step in _steps()]

    assert names[-5:] == [
        "Point the datasets worker at the new image",
        "Point Cloud Run at the new image",
        "Settle rows the previous image wrote",
        "Record the code index the services run",
        "Backfill description records",
    ]


def test_decide_step_runs_gcloud_describe_and_git_diff_and_gates_every_later_step():
    """The decide step compares Cloud Run's live image with the target and gates every build,
    migration, and service update on `deploy=true`."""
    decide = _step("decide")
    assert decide["run"].strip() == ".github/scripts/deploy_decide.sh"
    assert decide["env"]["TARGET"] == "${{ steps.target.outputs.sha }}"

    script = _SCRIPT.read_text()
    assert "gcloud run services describe" in script
    assert "git diff --quiet" in script
    assert "exit 1" in script, "an empty or failed describe call is a hard failure"

    steps = _steps()
    decide_index = steps.index(decide)
    later = steps[decide_index + 1 :]
    assert len(later) == 20
    for step in later:
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
