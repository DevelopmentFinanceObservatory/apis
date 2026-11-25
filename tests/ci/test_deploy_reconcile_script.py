"""Test `deploy_reconcile.sh`, alone and after `deploy_decide.sh` on a same-commit retry."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.ci import cloud_run_fake
from tests.ci.test_deploy_decide_script import (
    _JOB,
    _SERVING,
    _SERVING_DIGEST,
    _STAGED,
    _STAGED_DIGEST,
    _commit,
    _git,
    _image_for,
    _run_decide,
)

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "deploy_reconcile.sh"


def _service(*, spec_traffic: list, status_traffic: list, template: str = "") -> dict:
    return {
        "spec": {
            "template": {"spec": {"containers": [{"image": template}]}},
            "traffic": spec_traffic,
        },
        "status": {
            "latestCreatedRevisionName": _STAGED,
            "latestReadyRevisionName": _STAGED,
            "traffic": status_traffic,
        },
    }


def _install(bin_dir: Path, service: dict, *, job_image: str, revisions: dict | None = None):
    return cloud_run_fake.install(
        bin_dir,
        service=service,
        revisions=revisions or {_SERVING: _SERVING_DIGEST, _STAGED: _STAGED_DIGEST},
        jobs={_JOB: job_image},
    )


def _run_reconcile(bin_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SERVICE": "dfo-apis-dev",
            "JOB": _JOB,
            "REGION": "northamerica-northeast1",
            "PROJECT_ID": "dev-fin-observatory",
        },
    )


def _updates(state: Path) -> list[str]:
    return [call for call in cloud_run_fake.read(state)["calls"] if " update " in f" {call} "]


@pytest.mark.parametrize(
    ("job_image", "updated"),
    [(_SERVING_DIGEST, False), ("host/repo/dfo-apis@sha256:old", True)],
    ids=["matching", "differing"],
)
def test_the_job_moves_to_the_serving_image_only_when_it_runs_another(
    tmp_path: Path, job_image: str, updated: bool
):
    """A revision with the `candidate` tag and no traffic does not serve."""
    service = _service(
        spec_traffic=[
            {"percent": 100, "revisionName": _SERVING},
            {"revisionName": _STAGED, "tag": "candidate"},
        ],
        status_traffic=[
            {"percent": 100, "revisionName": _SERVING},
            {"revisionName": _STAGED, "tag": "candidate"},
        ],
    )
    state = _install(tmp_path / "bin", service, job_image=job_image)

    ran = _run_reconcile(tmp_path / "bin")

    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert cloud_run_fake.read(state)["jobs"][_JOB] == _SERVING_DIGEST
    assert _updates(state) == (
        [f"run jobs update {_JOB} --image={_SERVING_DIGEST} --quiet"] if updated else []
    )


def test_traffic_split_across_images_refuses_and_moves_nothing(tmp_path: Path):
    split = [
        {"percent": 50, "revisionName": _SERVING},
        {"percent": 50, "revisionName": _STAGED},
    ]
    state = _install(
        tmp_path / "bin",
        _service(spec_traffic=split, status_traffic=split),
        job_image=_SERVING_DIGEST,
    )

    ran = _run_reconcile(tmp_path / "bin")

    assert ran.returncode != 0
    assert "::error::" in ran.stderr
    assert _updates(state) == []
    assert cloud_run_fake.read(state)["jobs"][_JOB] == _SERVING_DIGEST


_PINNED_PREVIOUS = [
    {"percent": 100, "revisionName": _SERVING},
    {"revisionName": _STAGED, "tag": "candidate"},
]
_PINNED_CANDIDATE = [
    {"percent": 100, "revisionName": _STAGED},
    {"revisionName": _STAGED, "tag": "candidate"},
]
_RELEASED = [{"percent": 100, "latestRevision": True, "revisionName": _STAGED}]


@pytest.mark.parametrize(
    ("spec_traffic", "status_traffic", "serving"),
    [
        # Cancelled before Cloud Run received the traffic move: the stage pinned traffic to the
        # previous revision, and nothing makes traffic follow the candidate.
        (_PINNED_PREVIOUS, _PINNED_PREVIOUS, _SERVING_DIGEST),
        # Cancelled after the traffic move, before the scheduled job's update.
        (_PINNED_CANDIDATE, _PINNED_CANDIDATE, _STAGED_DIGEST),
        # The scheduled job's update failed and the run went on to release the service.
        ([{"percent": 100, "latestRevision": True}], _RELEASED, _STAGED_DIGEST),
    ],
    ids=["cancelled-before-move", "cancelled-after-move", "failed-job-update"],
)
def test_a_same_commit_retry_deploys_and_points_the_job_at_the_serving_image(
    tmp_path: Path, spec_traffic: list, status_traffic: list, serving: str
):
    """The job stays on the previous image in each state, and the retry's decide step deploys
    whatever the traffic, so its reconcile step leaves the job on the image that serves."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    sha = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    state = _install(
        bin_dir,
        _service(
            spec_traffic=spec_traffic, status_traffic=status_traffic, template=_image_for(sha)
        ),
        job_image=_SERVING_DIGEST,
    )

    decided, output = _run_decide(repo, bin_dir, tmp_path, target=sha)
    reconciled = _run_reconcile(bin_dir)

    assert decided.returncode == 0, decided.stdout + decided.stderr
    assert output == "deploy=true\n"
    assert reconciled.returncode == 0, reconciled.stdout + reconciled.stderr
    assert cloud_run_fake.read(state)["jobs"][_JOB] == serving
