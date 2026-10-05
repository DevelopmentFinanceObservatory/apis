"""Test `deploy_decide.sh` against a throwaway Git repository and stub `gcloud`."""

from __future__ import annotations

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


def _fake_gcloud(bin_dir: Path, *, image: str = "", exit_code: int = 0) -> None:
    """A PATH shim standing in for `gcloud run services describe`. The real
    command's exact flags don't matter here: the script only reads stdout and
    the exit code."""
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "gcloud"
    body = f"""#!/usr/bin/env bash
exit_code={exit_code}
if [ "$exit_code" != "0" ]; then
  exit "$exit_code"
fi
echo "{image}"
"""
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _run_decide(
    repo: Path,
    bin_dir: Path,
    tmp_path: Path,
    *,
    target: str,
    force: str = "false",
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
            "GITHUB_OUTPUT": str(output_file),
        },
    )
    return result, output_file.read_text()


def _image_for(sha: str) -> str:
    return f"northamerica-northeast1-docker.pkg.dev/dev-fin-observatory/datacommons/dfo-apis:{sha}"


def test_live_equal_to_target_does_not_deploy(repo: Path, tmp_path: Path):
    sha = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, image=_image_for(sha))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha)

    assert result.returncode == 0, result.stderr
    assert "deploy=false" in output


def test_docs_only_diff_since_live_does_not_deploy(repo: Path, tmp_path: Path):
    live = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    target = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, image=_image_for(live))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert "deploy=false" in output


def test_src_change_several_commits_back_still_deploys(repo: Path, tmp_path: Path):
    """A source change two commits before a docs-only tip still triggers deployment."""
    live = _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    target = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, image=_image_for(live))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert "deploy=true" in output


def test_unknown_live_tag_deploys_with_a_warning(repo: Path, tmp_path: Path):
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, image=_image_for("not-a-real-commit-placeholder"))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert "deploy=true" in output
    assert "::warning::" in result.stdout


def test_empty_image_output_exits_nonzero(repo: Path, tmp_path: Path):
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, image="")

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

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
    _fake_gcloud(bin_dir, image=_image_for(sha))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=sha, force="true")

    assert result.returncode == 0, result.stderr
    assert "deploy=true" in output


def test_force_with_empty_image_still_exits_nonzero(repo: Path, tmp_path: Path):
    """FORCE still requires a readable live image."""
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, image="")

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target, force="true")

    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert "deploy=" not in output


def test_watched_paths_match_the_workflows_declared_image_inputs():
    """The watched path list must match the script."""
    script = _SCRIPT.read_text()
    for path in _WATCHED_PATHS:
        assert path in script
