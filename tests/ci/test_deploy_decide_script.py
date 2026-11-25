"""Test `deploy_decide.sh` against a throwaway Git repository and stub `gcloud`."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.ci import cloud_run_fake

_SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "deploy_decide.sh"

_WATCHED_PATHS = (
    "src",
    "packages",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "Dockerfile",
    ".github/workflows/deploy.yml",
    ".github/workflows/deploy-dev.yml",
    ".github/workflows/deploy-prod.yml",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


def _commit(repo: Path, message: str, files: dict[str, str]) -> str:
    for relpath, content in files.items():
        path = repo / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    return repo


_SERVING = "dfo-dev-apis-00001-aaa"
_STAGED = "dfo-dev-apis-00002-bbb"
_JOB = "dfo-dev-origin-catalogue"
# Revisions record their image by digest, and a finished deploy points the scheduled origin
# catalogue job at the serving revision's digest.
_SERVING_DIGEST = "host/repo/dfo-apis@sha256:aaa"
_STAGED_DIGEST = "host/repo/dfo-apis@sha256:bbb"

# Traffic following the latest revision, as a finished deploy leaves it.
_FOLLOWING = [{"percent": 100, "latestRevision": True}]


def _service(commit: str, *, traffic: list | None = None, ready: str = _STAGED) -> dict:
    """A `gcloud run services describe --format=json` document, trimmed to the fields the
    scripts read: the template's image, the traffic asked for and the traffic serving, and the
    latest created (`_STAGED`) and latest ready revisions."""
    return {
        "spec": {
            "template": {"spec": {"containers": [{"image": _image_for(commit)}]}},
            "traffic": _FOLLOWING if traffic is None else traffic,
        },
        "status": {
            "latestCreatedRevisionName": _STAGED,
            "latestReadyRevisionName": ready,
            "traffic": [{"percent": 100, "latestRevision": True, "revisionName": ready}],
        },
    }


def _fake_gcloud(
    bin_dir: Path,
    *,
    service: dict | None = None,
    exit_code: int = 0,
    job_image: str | None = _STAGED_DIGEST,
) -> None:
    """A PATH shim standing in for `gcloud run`, serving `service`, the two revisions' digests
    and the scheduled origin catalogue job on `job_image`, or no such job when it is None."""
    cloud_run_fake.install(
        bin_dir,
        service=service or {},
        revisions={_SERVING: _SERVING_DIGEST, _STAGED: _STAGED_DIGEST},
        jobs={} if job_image is None else {_JOB: job_image},
        fail=exit_code != 0,
    )


def _run_decide(
    repo: Path, bin_dir: Path, tmp_path: Path, *, target: str, force: str = "false"
) -> tuple[subprocess.CompletedProcess, str]:
    output_file = tmp_path / "github_output"
    output_file.write_text("")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SERVICE": "dfo-apis-dev",
            "REGION": "northamerica-northeast1",
            "PROJECT_ID": "dev-fin-observatory",
            "TARGET": target,
            "JOB": _JOB,
            "FORCE": force,
            "GITHUB_OUTPUT": str(output_file),
        },
    )
    return result, output_file.read_text()


def _image_for(sha: str) -> str:
    return f"northamerica-northeast1-docker.pkg.dev/dev-fin-observatory/datacommons/dfo-apis:{sha}"


def test_live_equal_to_target_does_not_deploy(repo: Path, tmp_path: Path):
    sha = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(sha))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha)

    assert result.returncode == 0, result.stderr
    assert output == "deploy=false\n"


def test_docs_only_diff_since_live_does_not_deploy(repo: Path, tmp_path: Path):
    live = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    target = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(live))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert output == "deploy=false\n"


def test_src_change_several_commits_back_still_deploys(repo: Path, tmp_path: Path):
    """A source change two commits before a docs-only tip still triggers deployment."""
    live = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    target = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(live))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert output == "deploy=true\n"


def test_unknown_live_tag_deploys_with_a_warning(repo: Path, tmp_path: Path):
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service("not-a-real-commit-placeholder"))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert output == "deploy=true\n"
    assert "::warning::" in result.stdout


@pytest.mark.parametrize(
    ("traffic", "ready"),
    [
        # A probe failed: the stage pinned traffic to the previous revision and tagged its own.
        (
            [
                {"percent": 100, "revisionName": _SERVING},
                {"revisionName": _STAGED, "tag": "candidate"},
            ],
            _STAGED,
        ),
        # A run stopped after the move, before traffic followed the latest revision again.
        (
            [
                {"percent": 100, "revisionName": _STAGED},
                {"revisionName": _STAGED, "tag": "candidate"},
            ],
            _STAGED,
        ),
        # A run stopped after traffic followed the latest revision, before the tag went.
        ([*_FOLLOWING, {"revisionName": _STAGED, "tag": "candidate"}], _STAGED),
        # Traffic moved by hand.
        ([{"percent": 100, "revisionName": _SERVING}], _STAGED),
        (
            [{"percent": 50, "revisionName": _SERVING}, {"percent": 50, "revisionName": _STAGED}],
            _STAGED,
        ),
        # The latest revision never started, so an older one serves.
        (_FOLLOWING, _SERVING),
    ],
    ids=[
        "probe-failed",
        "stopped-pinned",
        "stopped-tagged",
        "hand-pin",
        "split",
        "latest-not-ready",
    ],
)
def test_a_service_not_in_the_state_a_finished_deploy_leaves_deploys(
    repo: Path, tmp_path: Path, traffic, ready
):
    """The template's commit need not serve, so even the template's own commit deploys, and
    the next run recovers a stopped one from the start."""
    sha = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(sha, traffic=traffic, ready=ready))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha)

    assert result.returncode == 0, result.stderr
    assert output == "deploy=true\n"


@pytest.mark.parametrize("force", ["false", "true"])
@pytest.mark.parametrize(
    "service",
    [{"spec": {"template": {"spec": {"containers": [{}]}}}, "status": {}}, {}],
    ids=["no-image", "empty"],
)
def test_an_unreadable_service_exits_nonzero(repo: Path, tmp_path: Path, force, service):
    """FORCE still requires a readable service."""
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=service)

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target, force=force)

    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert "deploy=" not in output


def test_failed_describe_call_exits_nonzero(repo: Path, tmp_path: Path):
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, exit_code=1)

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert "deploy=" not in output


def test_force_with_live_equal_to_target_still_deploys(repo: Path, tmp_path: Path):
    """FORCE redeploys a commit already live, including to rerun migrations."""
    sha = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(sha))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha, force="true")

    assert result.returncode == 0, result.stderr
    assert output == "deploy=true\n"


@pytest.mark.parametrize(
    "job_image",
    [
        _SERVING_DIGEST,
        # The tag a deploy before the digest pin set, which names no serving revision's digest.
        "host/repo/dfo-apis:abc",
    ],
    ids=["previous-image", "tag"],
)
def test_a_released_service_with_the_job_behind_deploys(repo: Path, tmp_path: Path, job_image):
    """A run whose scheduled job update failed or never ran leaves the service released, so the
    same commit deploys again and the workflow's reconcile step moves the job."""
    sha = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(sha), job_image=job_image)

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha)

    assert result.returncode == 0, result.stderr
    assert output == "deploy=true\n"


def test_an_unreadable_job_exits_nonzero(repo: Path, tmp_path: Path):
    sha = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(sha), job_image=None)

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha)

    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert "deploy=" not in output


def test_watched_paths_match_the_workflows_declared_image_inputs():
    """The watched path list must match the script."""
    script = _SCRIPT.read_text()
    for path in _WATCHED_PATHS:
        assert path in script
