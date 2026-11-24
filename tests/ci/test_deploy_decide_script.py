"""Test `deploy_decide.sh` against a throwaway Git repository and stub `gcloud`."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

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
_NEWER = "dfo-dev-apis-00003-ccc"


def _service(
    commit: str,
    *,
    serving: str | None = _STAGED,
    candidate: str | None = None,
    latest: str = _STAGED,
    ready: str | None = None,
) -> dict:
    """A `gcloud run services describe --format=json` document, trimmed to the fields Cloud Run
    returns for the apis service: the template's image, the latest created revision (built
    from that image), the latest ready one (`latest` unless given), and the traffic entries.
    `serving` names the revision with 100 percent of traffic, or None for a split."""
    traffic = (
        [{"percent": 100, "revisionName": serving, "latestRevision": serving == _STAGED}]
        if serving
        else [
            {"percent": 50, "revisionName": _SERVING},
            {"percent": 50, "revisionName": _STAGED},
        ]
    )
    if candidate:
        traffic.append({"revisionName": candidate, "tag": "candidate", "url": "https://tagged"})
    return {
        "spec": {"template": {"spec": {"containers": [{"image": _image_for(commit)}]}}},
        "status": {
            "latestCreatedRevisionName": latest,
            "latestReadyRevisionName": ready or latest,
            "traffic": traffic,
        },
    }


def _fake_gcloud(bin_dir: Path, *, service: dict | None = None, exit_code: int = 0) -> None:
    """A PATH shim standing in for `gcloud run services describe`: prints the service as JSON
    under `--format=json`, and the template's image under the value format that reads it."""
    bin_dir.mkdir(exist_ok=True)
    image = service["spec"]["template"]["spec"]["containers"][0].get("image", "") if service else ""
    script = bin_dir / "gcloud"
    script.write_text(
        f"""#!/usr/bin/env bash
if [ "{exit_code}" != 0 ]; then
  exit {exit_code}
fi
for arg in "$@"; do
  case "$arg" in
    --format=json) echo '{json.dumps(service or {})}'; exit 0 ;;
    --format=value*) echo "{image}"; exit 0 ;;
  esac
done
exit 2
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _run_decide(
    repo: Path,
    bin_dir: Path,
    tmp_path: Path,
    *,
    target: str,
    force: str = "false",
    override: str = "false",
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
            "FORCE": force,
            "OVERRIDE": override,
            "GITHUB_OUTPUT": str(output_file),
        },
    )
    return result, output_file.read_text()


def _image_for(sha: str) -> str:
    return f"northamerica-northeast1-docker.pkg.dev/dev-fin-observatory/datacommons/dfo-apis:{sha}"


def _outputs(deploy: bool, release: bool, revision: str = "") -> str:
    return f"deploy={str(deploy).lower()}\nrelease={str(release).lower()}\nrevision={revision}\n"


_NOTHING = _outputs(False, False)
_DEPLOY = _outputs(True, True)


def test_live_equal_to_target_does_not_deploy(repo: Path, tmp_path: Path):
    sha = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(sha))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha)

    assert result.returncode == 0, result.stderr
    assert output == _NOTHING


def test_docs_only_diff_since_live_does_not_deploy(repo: Path, tmp_path: Path):
    live = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    target = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(live))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert output == _NOTHING


def test_src_change_several_commits_back_still_deploys(repo: Path, tmp_path: Path):
    """A source change two commits before a docs-only tip still triggers deployment."""
    live = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    target = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(live))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert output == _DEPLOY


def test_unknown_live_tag_deploys_with_a_warning(repo: Path, tmp_path: Path):
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service("not-a-real-commit-placeholder"))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert output == _DEPLOY
    assert "::warning::" in result.stdout


@pytest.mark.parametrize("force", ["false", "true"])
@pytest.mark.parametrize(
    "service",
    [
        {"spec": {"template": {"spec": {"containers": [{}]}}}, "status": {}},
        {},
    ],
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
    assert output == _DEPLOY


@pytest.mark.parametrize(
    ("target_of", "latest", "ready", "force", "expected"),
    [
        # A rerun releases the revision an earlier run staged for this commit, without building.
        ("staged", _STAGED, _STAGED, "false", _outputs(False, True, _STAGED)),
        # A staged revision that never became ready is built again.
        ("staged", _STAGED, _SERVING, "false", _DEPLOY),
        # A dispatch rebuilds even a ready staged revision.
        ("staged", _STAGED, _STAGED, "true", _DEPLOY),
        # The commit after a staged one deploys even with a docs-only diff from it, since the
        # previous revision still serves.
        ("docs", _STAGED, _STAGED, "false", _DEPLOY),
        # A revision created after the stop (an `infra` apply) gets no traffic while the pin
        # holds, so the next deploy builds and moves traffic, whatever the target.
        ("staged", _NEWER, _NEWER, "false", _DEPLOY),
        ("docs", _NEWER, _NEWER, "false", _DEPLOY),
    ],
    ids=[
        "rerun",
        "rerun-not-ready",
        "dispatch",
        "docs-only-after-staged",
        "rerun-after-newer-revision",
        "docs-only-after-newer-revision",
    ],
)
def test_a_revision_staged_by_a_stopped_deploy(
    repo: Path, tmp_path: Path, target_of, latest, ready, force, expected
):
    """An earlier deploy staged revision `_STAGED` with the `candidate` tag and stopped before
    traffic moved, so `_SERVING` still serves, pinned by name."""
    _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    staged = _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    docs = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(
        bin_dir,
        service=_service(staged, serving=_SERVING, candidate=_STAGED, latest=latest, ready=ready),
    )
    target = staged if target_of == "staged" else docs

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target, force=force)

    assert result.returncode == 0, result.stderr
    assert output == expected


@pytest.mark.parametrize(
    ("target_of", "serving", "force", "override", "refused"),
    [
        # A rerun of the commit someone moved traffic off refuses.
        ("held", _SERVING, "false", "false", True),
        # So does a dispatch of it, and a commit with the same image inputs.
        ("held", _SERVING, "true", "false", True),
        ("docs", _SERVING, "false", "false", True),
        # A split counts as traffic moved by hand.
        ("held", None, "false", "false", True),
        # The override deploys.
        ("held", _SERVING, "true", "true", False),
        ("held", None, "true", "true", False),
        # A commit with other image inputs deploys over the hold.
        ("fix", _SERVING, "false", "false", False),
    ],
    ids=["rerun", "dispatch", "docs-only", "split", "override", "split-override", "new-code"],
)
def test_traffic_moved_by_hand_refuses_a_target_that_would_undo_it(
    repo: Path, tmp_path: Path, target_of, serving, force, override, refused
):
    """Someone moved traffic off revision `_STAGED`, which carries no `candidate` tag since its
    deploy released it."""
    _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    held = _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    docs = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    fix = _commit(repo, "fix: change app back", {"src/app.py": "3\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(held, serving=serving))
    target = {"held": held, "docs": docs, "fix": fix}[target_of]

    result, output = _run_decide(
        repo, bin_dir, tmp_path, target=target, force=force, override=override
    )

    if refused:
        assert result.returncode == 1
        assert "::error::" in result.stdout
        assert output == ""
    else:
        assert result.returncode == 0, result.stderr
        assert output == _DEPLOY


def test_watched_paths_match_the_workflows_declared_image_inputs():
    """The watched path list must match the script."""
    script = _SCRIPT.read_text()
    for path in _WATCHED_PATHS:
        assert path in script
