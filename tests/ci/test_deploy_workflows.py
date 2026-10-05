"""Check deploy triggers, guards, permissions, target selection, and step gates without a deploy.

deploy.yml holds the steps. deploy-dev.yml and deploy-prod.yml call it and choose only the
triggers, the GitHub Environment, and the owner guard.
"""

import re
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


def test_prod_deploys_on_a_push_to_main_or_a_dispatch():
    triggers = _triggers(_workflow("deploy-prod.yml"))

    assert triggers["push"] == {"branches": ["main"]}
    assert "workflow_dispatch" in triggers
    assert set(triggers) == {"push", "workflow_dispatch"}


@pytest.mark.parametrize(
    ("caller", "owner", "environment"),
    [
        ("deploy-dev.yml", "ONEcampaign", "dev"),
        ("deploy-prod.yml", "DevelopmentFinanceObservatory", "production"),
    ],
)
def test_each_caller_guards_its_owner_and_names_its_environment(
    caller: str, owner: str, environment: str
):
    """The same files run in both organisations, so the guard picks which one deploys."""
    job = _workflow(caller)["jobs"]["deploy"]

    assert f"github.repository_owner == '{owner}'" in job["if"]
    assert job["uses"] == "./.github/workflows/deploy.yml"
    assert job["with"] == {"environment": environment}
    assert job["permissions"] == _CALLER_PERMISSIONS
    assert job["secrets"] == "inherit"


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
    assert build["with"]["secrets"].strip() == (
        "spec_kit_token=${{ steps.spec-kit-auth.outputs.token }}"
    )


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
    # Five job updates, eight job executions, and two service updates.
    assert len(targets) == 15
    for target in targets:
        assert re.fullmatch(r'"\$\{\{ vars\.(JOB|SERVICE)_[A-Z_]+ \}\}"', target), target


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


def test_spec_kit_token_comes_from_the_reader_app():
    """The build reads dfo-spec-kit with the read-only reader App, never the release App."""
    workflow = _workflow("deploy.yml")
    assert set(_triggers(workflow)["workflow_call"]) == {"inputs"}
    assert _workflow("deploy.yml")["jobs"]["deploy"]["environment"] == "${{ inputs.environment }}"
    assert _step("spec-kit-auth")["with"] == {
        "client-id": "${{ vars.READER_APP_CLIENT_ID }}",
        "private-key": "${{ secrets.READER_APP_PRIVATE_KEY }}",
    }


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


def test_worker_then_service_move_then_the_code_index_is_recorded():
    names = [step.get("name") for step in _steps()]

    assert names[-3:] == [
        "Point the datasets worker at the new image",
        "Point Cloud Run at the new image",
        "Record the code index the services run",
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
    assert len(later) == 17
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
