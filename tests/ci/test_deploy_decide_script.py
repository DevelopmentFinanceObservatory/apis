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
_HAND = "dfo-dev-apis-00000-zzz"


def _service(
    commit: str,
    *,
    pinned: str | None = None,
    split: bool = False,
    candidate: str | None = None,
    previous: str | None = None,
    latest: str = _STAGED,
    ready: str | None = None,
) -> dict:
    """A `gcloud run services describe --format=json` document, trimmed to the fields Cloud Run
    returns for the apis service: the template's image, the latest created revision (built from
    that image), the latest ready one (`latest` unless given), and the traffic asked for in
    `spec.traffic` and resolved in `status.traffic`. Traffic follows the latest revision unless
    `pinned` names a revision or `split` is set. A `candidate` tag on a revision that never
    became ready appears in `spec.traffic` only. `previous` tags the revision a stage step
    pinned traffic to."""
    ready = ready or latest
    if split:
        asked = [
            {"percent": 50, "revisionName": _SERVING},
            {"percent": 50, "revisionName": _STAGED},
        ]
        resolved = asked
    elif pinned:
        asked = [{"percent": 100, "revisionName": pinned}]
        resolved = asked
    else:
        asked = [{"percent": 100, "latestRevision": True}]
        resolved = [{"percent": 100, "latestRevision": True, "revisionName": ready}]
    if candidate:
        tag = {"revisionName": candidate, "tag": "candidate"}
        asked = [*asked, tag]
        if not (candidate == latest and ready != latest):
            resolved = [*resolved, {**tag, "url": "https://tagged"}]
    if previous:
        tag = {"revisionName": previous, "tag": "previous"}
        asked = [*asked, tag]
        resolved = [*resolved, {**tag, "url": "https://previous"}]
    return {
        "spec": {
            "template": {"spec": {"containers": [{"image": _image_for(commit)}]}},
            "traffic": asked,
        },
        "status": {
            "latestCreatedRevisionName": latest,
            "latestReadyRevisionName": ready,
            "traffic": resolved,
        },
    }


def _fake_gcloud(
    bin_dir: Path, *, service: dict | None = None, job_commit: str = "", exit_code: int = 0
) -> None:
    """A PATH shim standing in for `gcloud run services describe`, which prints the service as
    JSON under `--format=json` and the template's image under the value format that reads it,
    and for `gcloud run jobs describe`, which prints the job's image tagged `job_commit`, or the
    template's commit when unset."""
    bin_dir.mkdir(exist_ok=True)
    image = service["spec"]["template"]["spec"]["containers"][0].get("image", "") if service else ""
    job_image = _image_for(job_commit) if job_commit else image
    script = bin_dir / "gcloud"
    script.write_text(
        f"""#!/usr/bin/env bash
if [ "{exit_code}" != 0 ]; then
  exit {exit_code}
fi
if [ "$2" = jobs ]; then
  echo "{job_image}"
  exit 0
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
            "MAINTENANCE_JOB": "dfo-dev-datasets-maintenance",
            "FORCE": force,
            "OVERRIDE": override,
            "GITHUB_OUTPUT": str(output_file),
        },
    )
    return result, output_file.read_text()


def _image_for(sha: str) -> str:
    return f"northamerica-northeast1-docker.pkg.dev/dev-fin-observatory/datacommons/dfo-apis:{sha}"


def _outputs(deploy: bool, probe: bool, release: bool, revision: str = "") -> str:
    flags = {"deploy": deploy, "probe": probe, "release": release}
    return "".join(f"{key}={str(value).lower()}\n" for key, value in flags.items()) + (
        f"revision={revision}\n"
    )


_NOTHING = _outputs(False, False, False)
_DEPLOY = _outputs(True, True, True)
_RELEASE_STAGED = _outputs(False, True, True, _STAGED)
_FINISH_STAGED = _outputs(False, False, True, _STAGED)


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
    ("target_of", "latest", "ready", "job_of", "force", "expected"),
    [
        # A rerun releases the revision an earlier run staged for this commit, without building.
        ("staged", _STAGED, _STAGED, "staged", "false", _RELEASE_STAGED),
        # A staged revision that never became ready is built again.
        ("staged", _STAGED, _SERVING, "staged", "false", _DEPLOY),
        # A dispatch rebuilds even a ready staged revision.
        ("staged", _STAGED, _STAGED, "staged", "true", _DEPLOY),
        # A later run moved the jobs and the worker to another image, so the release rebuilds.
        ("staged", _STAGED, _STAGED, "docs", "false", _DEPLOY),
        # The commit after a staged one deploys even with a docs-only diff from it, since the
        # previous revision still serves.
        ("docs", _STAGED, _STAGED, "staged", "false", _DEPLOY),
        # A revision created after the stop (an `infra` apply) gets no traffic while the pin
        # holds, so the next deploy builds and moves traffic, whatever the target.
        ("staged", _NEWER, _NEWER, "staged", "false", _DEPLOY),
        ("docs", _NEWER, _NEWER, "staged", "false", _DEPLOY),
    ],
    ids=[
        "rerun",
        "rerun-not-ready",
        "dispatch",
        "jobs-on-another-image",
        "docs-only-after-staged",
        "rerun-after-newer-revision",
        "docs-only-after-newer-revision",
    ],
)
def test_a_revision_staged_by_a_stopped_deploy(
    repo: Path, tmp_path: Path, target_of, latest, ready, job_of, force, expected
):
    """An earlier deploy tagged `_SERVING` `previous`, staged revision `_STAGED` with the
    `candidate` tag, and stopped before traffic moved, so `_SERVING` still serves, pinned by
    name."""
    _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    staged = _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    docs = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    commits = {"staged": staged, "docs": docs}
    _fake_gcloud(
        bin_dir,
        service=_service(
            staged,
            pinned=_SERVING,
            candidate=_STAGED,
            previous=_SERVING,
            latest=latest,
            ready=ready,
        ),
        job_commit=commits[job_of],
    )

    result, output = _run_decide(repo, bin_dir, tmp_path, target=commits[target_of], force=force)

    assert result.returncode == 0, result.stderr
    assert output == expected


@pytest.mark.parametrize(
    ("traffic", "target_of", "expected"),
    [
        # The release moved traffic and stopped before settle, backfill or the tag removal: a
        # rerun finishes it without probing.
        ({"candidate": _STAGED}, "released", _FINISH_STAGED),
        ({"pinned": _STAGED, "candidate": _STAGED}, "released", _FINISH_STAGED),
        # A pin by name on the latest revision is what a move that stopped before restoring
        # `--to-latest` leaves.
        ({"pinned": _STAGED}, "released", _FINISH_STAGED),
        ({"pinned": _STAGED}, "docs", _FINISH_STAGED),
        # A dispatch rebuilds.
        ({"candidate": _STAGED}, "new", _DEPLOY),
    ],
    ids=["tag-on-serving", "pinned-and-tagged", "pinned-latest", "pinned-latest-docs", "new"],
)
def test_a_release_that_moved_traffic_and_stopped_is_finished(
    repo: Path, tmp_path: Path, traffic, target_of, expected
):
    """Revision `_STAGED` serves; its release did not remove the tag or restore following the
    latest revision."""
    _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    released = _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    docs = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    new = _commit(repo, "feat: change app again", {"src/app.py": "3\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(released, **traffic))
    target = {"released": released, "docs": docs, "new": new}[target_of]

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stderr
    assert output == expected


@pytest.mark.parametrize(
    ("service", "target_of"),
    [
        # The latest revision never started, so traffic still follows the latest ready one.
        ({"latest": _NEWER, "ready": _STAGED}, "latest"),
        ({"latest": _NEWER, "ready": _STAGED}, "docs"),
        # The stage step's revision never started, so its tag appears in `spec.traffic` only.
        (
            {"pinned": _SERVING, "candidate": _STAGED, "previous": _SERVING, "ready": _SERVING},
            "latest",
        ),
        # The stage step pinned traffic and stopped before it tagged its revision.
        ({"pinned": _SERVING, "previous": _SERVING}, "latest"),
    ],
    ids=["latest-not-ready", "latest-not-ready-docs", "staged-not-ready", "stage-untagged"],
)
def test_a_revision_that_never_started_is_not_a_move_by_hand(
    repo: Path, tmp_path: Path, service, target_of
):
    """Nobody moved traffic, so a rerun or a docs-only commit deploys."""
    _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    latest = _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    docs = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(latest, **service))
    target = {"latest": latest, "docs": docs}[target_of]

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stdout
    assert output == _DEPLOY


@pytest.mark.parametrize(
    ("target_of", "traffic", "force", "override", "refused"),
    [
        # A rerun of the commit someone moved traffic off refuses.
        ("held", {"pinned": _SERVING}, "false", "false", True),
        # So does a dispatch of it, and a commit with the same image inputs.
        ("held", {"pinned": _SERVING}, "true", "false", True),
        ("docs", {"pinned": _SERVING}, "false", "false", True),
        # A split counts as traffic moved by hand.
        ("held", {"split": True}, "false", "false", True),
        # The override deploys.
        ("held", {"pinned": _SERVING}, "true", "true", False),
        ("held", {"split": True}, "true", "true", False),
        # A commit with other image inputs deploys over the hold.
        ("fix", {"pinned": _SERVING}, "false", "false", False),
        # A hand pin made while a stopped deploy's tags linger: traffic serves a revision other
        # than the one the stage step tagged `previous`.
        (
            "held",
            {"pinned": _SERVING, "candidate": _NEWER, "previous": _HAND},
            "false",
            "false",
            True,
        ),
        (
            "docs",
            {"pinned": _SERVING, "candidate": _NEWER, "previous": _HAND},
            "false",
            "false",
            True,
        ),
        (
            "fix",
            {"pinned": _SERVING, "candidate": _NEWER, "previous": _HAND},
            "false",
            "false",
            False,
        ),
    ],
    ids=[
        "rerun",
        "dispatch",
        "docs-only",
        "split",
        "override",
        "split-override",
        "new-code",
        "stale-tags-rerun",
        "stale-tags-docs-only",
        "stale-tags-new-code",
    ],
)
def test_traffic_moved_by_hand_refuses_a_target_that_would_undo_it(
    repo: Path, tmp_path: Path, target_of, traffic, force, override, refused
):
    """Someone moved traffic off revision `_STAGED`, which carries no `candidate` tag since its
    release finished."""
    _commit(repo, "init", {"src/app.py": "1\n", "README.md": "hello\n"})
    held = _commit(repo, "feat: change app", {"src/app.py": "2\n"})
    docs = _commit(repo, "docs: tweak readme", {"README.md": "hello again\n"})
    fix = _commit(repo, "fix: change app back", {"src/app.py": "3\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service(held, **traffic))
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


def test_a_hold_off_a_commit_the_checkout_does_not_know_deploys(repo: Path, tmp_path: Path):
    """Inputs that can't be compared are not the same inputs, so the target deploys."""
    target = _commit(repo, "init", {"src/app.py": "1\n"})
    bin_dir = tmp_path / "bin"
    _fake_gcloud(bin_dir, service=_service("not-a-real-commit-placeholder", pinned=_SERVING))

    result, output = _run_decide(repo, bin_dir, tmp_path, target=target)

    assert result.returncode == 0, result.stdout
    assert output == _DEPLOY


def test_watched_paths_match_the_workflows_declared_image_inputs():
    """The watched path list must match the script."""
    script = _SCRIPT.read_text()
    for path in _WATCHED_PATHS:
        assert path in script
