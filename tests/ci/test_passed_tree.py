"""Check that ci.yml skips its suite only for a tree with a verified record of passing.

`passed_tree.py` is loaded from `.github/scripts` and runs here against a throwaway Git
repository, with tokens signed by a test key standing in for GitHub's and its `gh` and
HTTP reads replaced. The workflow's wiring is read from the YAML, and the `test` job's
shell step runs under bash.
"""

import base64
import importlib.util
import io
import json
import os
import subprocess
import time
import urllib.parse
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / ".github" / "scripts" / "passed_tree.py"
_WORKFLOW = yaml.safe_load((_ROOT / ".github" / "workflows" / "ci.yml").read_text())
_JOBS = _WORKFLOW["jobs"]
_LOOKUP = "passed-tree"
_RECORD = "record-passed-tree"
# The jobs a record vouches for: every job but the lookup and the record itself.
_SUITE = sorted(set(_JOBS) - {_LOOKUP, _RECORD})
_REPO = "ONEcampaign/dfo-apis"
_REPO_ID = "1311485891"
_BRANCH = "dev"
_KID = "test-key"

_spec = importlib.util.spec_from_file_location("passed_tree", _SCRIPT)
passed_tree: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(passed_tree)

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwks(key: rsa.RSAPrivateKey = _KEY) -> dict:
    numbers = key.public_key().public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": _KID,
                "n": _b64url(numbers.n.to_bytes(256, "big")),
                "e": _b64url(numbers.e.to_bytes(3, "big")),
            }
        ]
    }


def _token(claims: dict, *, key: rsa.RSAPrivateKey = _KEY, alg: str = "RS256", kid: str = _KID):
    signed = f"{_b64url(json.dumps({'alg': alg, 'kid': kid}).encode())}.{_b64url(json.dumps(claims).encode())}"
    signature = key.sign(signed.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signed}.{_b64url(signature)}"


def _mispadded(token: str) -> str:
    """Re-sign `token` with one padding byte changed and its digest left whole."""
    signed, _, signature = token.rpartition(".")
    private = _KEY.private_numbers()
    public = private.public_numbers
    genuine = int.from_bytes(base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)))
    encoded = bytearray(pow(genuine, public.e, public.n).to_bytes(256, "big"))
    encoded[2] ^= 0x01
    forged = pow(int.from_bytes(encoded), private.d, public.n)
    return f"{signed}.{_b64url(forged.to_bytes(256, 'big'))}"


def _claims(sha: str, **overrides: object) -> dict:
    return {
        "iss": passed_tree.ISSUER,
        "aud": passed_tree.AUDIENCE,
        "repository_id": _REPO_ID,
        "workflow_ref": f"{_REPO}/.github/workflows/ci.yml@refs/pull/7/merge",
        "event_name": "pull_request",
        "ref": "refs/pull/7/merge",
        "base_ref": _BRANCH,
        "run_attempt": "1",
        "sha": sha,
        "workflow_sha": sha,
        "iat": int(time.time()),
        "run_id": "4242",
        **overrides,
    }


class _Commit:
    """A commit in a throwaway repository: its id, its tree and its raw object."""

    def __init__(self, repo: Path, content: str, message: str = "change") -> None:
        (repo / "app.py").write_text(content)
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", message)
        self.sha = self._git(repo, "rev-parse", "HEAD").decode().strip()
        self.tree = self._git(repo, "rev-parse", "HEAD^{tree}").decode().strip()
        self.raw = self._git(repo, "cat-file", "commit", "HEAD")

    @staticmethod
    def _git(repo: Path, *args: str) -> bytes:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=repo,
            capture_output=True,
            check=True,
        ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "dev"], cwd=repo, check=True)
    return repo


def _recorded_run(*, token: str, commit: _Commit, tree: str, jwks: dict | None = None) -> str:
    return passed_tree.recorded_run(
        token=token,
        commit=commit.raw,
        tree=tree,
        repo=_REPO,
        repo_id=_REPO_ID,
        branch=_BRANCH,
        jwks=_jwks() if jwks is None else jwks,
        now=time.time(),
    )


@pytest.mark.parametrize(
    "event",
    [{}, {"event_name": "push", "ref": f"refs/heads/{_BRANCH}", "base_ref": ""}],
    ids=["pull request into dev", "push to dev"],
)
def test_a_record_vouches_for_the_tree_its_own_workflow_ran_on(repo: Path, event: dict):
    commit = _Commit(repo, "1")
    token = _token(_claims(commit.sha, **event))

    assert _recorded_run(token=token, commit=commit, tree=commit.tree) == "4242"


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"iss": "https://token.example"}, "issuer"),
        ({"aud": "https://github.com/ONEcampaign"}, "audience"),
        ({"repository_id": "7"}, "repository"),
        ({"workflow_ref": f"{_REPO}/.github/workflows/other.yml@refs/heads/x"}, "workflow"),
        ({"workflow_ref": f"{_REPO}/.github/workflows/ci.yml.yml@refs/heads/x"}, "workflow"),
        ({"workflow_sha": "0" * 40}, "workflow commit"),
        ({"base_ref": "ci/x"}, "branch"),
        ({"event_name": "push", "ref": f"refs/heads/ci/{_BRANCH}", "base_ref": ""}, "branch"),
        ({"event_name": "pull_request_target"}, "branch"),
        ({"event_name": "workflow_dispatch", "ref": f"refs/heads/{_BRANCH}", "base_ref": ""}, "branch"),
        ({"run_attempt": "2"}, "attempt"),
        ({"iat": int(time.time()) - 8 * 24 * 3600}, "age"),
        ({"iat": int(time.time()) + 3600}, "age"),
    ],
    ids=["issuer", "audience", "another repository", "another workflow", "a workflow named like ci.yml", "workflow from another commit", "pull request into another branch", "push to another branch", "pull_request_target", "dispatch on dev", "re-run", "older than a week", "from the future"],
)  # fmt: skip
def test_a_token_from_anything_but_this_repositorys_ci_is_refused(
    repo: Path, overrides: dict, refusal: str
):
    commit = _Commit(repo, "1")

    with pytest.raises(passed_tree.Refused, match=f"^{refusal}$"):
        _recorded_run(
            token=_token(_claims(commit.sha, **overrides)), commit=commit, tree=commit.tree
        )


@pytest.mark.parametrize(
    "message", ["change", "tree {tree}\n\ntree {tree}"], ids=["plain", "message names this tree"]
)
def test_a_run_on_another_tree_cannot_vouch_for_this_one(repo: Path, message: str):
    """A workflow elsewhere uploads its own valid token and commit under this tree's name."""
    wanted = _Commit(repo, "1")
    elsewhere = _Commit(repo, "2", message.format(tree=wanted.tree))

    with pytest.raises(passed_tree.Refused, match=r"^tree$"):
        _recorded_run(token=_token(_claims(elsewhere.sha)), commit=elsewhere, tree=wanted.tree)


def test_a_token_cannot_be_paired_with_another_commits_object(repo: Path):
    wanted = _Commit(repo, "1")
    elsewhere = _Commit(repo, "2")

    with pytest.raises(passed_tree.Refused, match=r"^commit$"):
        _recorded_run(token=_token(_claims(elsewhere.sha)), commit=wanted, tree=wanted.tree)


def _tampered(token: str, sha: str) -> str:
    header, _, signature = token.split(".")
    return f"{header}.{_b64url(json.dumps(_claims(sha, run_id='1')).encode())}.{signature}"


@pytest.mark.parametrize(
    ("forge", "refusal"),
    [
        (lambda sha: _tampered(_token(_claims(sha)), sha), "signature"),
        (lambda sha: _token(_claims(sha), key=_OTHER_KEY), "signature"),
        (lambda sha: _mispadded(_token(_claims(sha))), "signature"),
        (lambda sha: _token(_claims(sha), kid="unpublished"), "signing key"),
        (lambda sha: _token(_claims(sha), alg="none"), "algorithm"),
        (lambda sha: "not-a-token", "token format"),
    ],
    ids=["claims edited after signing", "another key", "padding changed", "unpublished key id", "alg none", "junk"],
)  # fmt: skip
def test_only_a_token_a_published_key_signed_is_read(repo: Path, forge, refusal: str):
    commit = _Commit(repo, "1")

    with pytest.raises(passed_tree.Refused, match=f"^{refusal}$"):
        _recorded_run(token=forge(commit.sha), commit=commit, tree=commit.tree)


class _GitHub:
    """Stands in for the token service, the key set and `gh api`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path) -> None:
        self.artifacts: list[dict] = []
        self.archives: dict[int, bytes] = {}
        self.fail: set[str] = set()
        self.requests: list[str] = []
        self.sha = ""
        self.output = tmp_path / "github_output"
        self.output.write_text("")
        monkeypatch.chdir(repo)
        monkeypatch.setattr(passed_tree, "_gh", self._gh)
        monkeypatch.setattr(passed_tree, "_fetch_json", self._fetch_json)
        for name, value in {
            "REPO": _REPO,
            "REPO_ID": _REPO_ID,
            "DEFAULT_BRANCH": _BRANCH,
            "RUN_ATTEMPT": "1",
            "GITHUB_OUTPUT": str(self.output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
            "ACTIONS_ID_TOKEN_REQUEST_URL": "https://token.example/mint?api-version=2",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "bearer-token",
        }.items():
            monkeypatch.setenv(name, value)

    def _fetch_json(self, request) -> dict:
        if request == passed_tree.JWKS_URL:
            if "keys" in self.fail:
                raise OSError("key set unreachable")
            return _jwks()
        # The token service mints for the audience the request names.
        audience = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)["audience"]
        return {"value": _token(_claims(self.sha, aud=audience[0]))}

    def _gh(self, path: str) -> bytes:
        self.requests.append(path)
        if path.endswith("/zip"):
            return self.archives[int(path.split("/")[-2])]
        if "listing" in self.fail:
            raise subprocess.CalledProcessError(1, "gh")
        wanted = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["name"][0]
        return json.dumps(
            {"artifacts": [item for item in self.artifacts if item["name"] == wanted]}
        ).encode()

    def record(self, tmp_path: Path, commit: _Commit, *, expired: bool = False) -> None:
        """Run `record` for `commit` and publish what it wrote as an artifact."""
        self.sha = commit.sha
        directory = tmp_path / f"record-{len(self.artifacts)}"
        assert passed_tree.main(["record", str(directory)]) == 0
        name = self.outputs().pop("name")
        self.output.write_text("")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for path in directory.iterdir():
                archive.write(path, path.name)
        number = len(self.artifacts) + 1
        self.artifacts.append({"id": number, "name": name, "expired": expired})
        self.archives[number] = buffer.getvalue()

    def outputs(self) -> dict[str, str]:
        return dict(line.split("=", 1) for line in self.output.read_text().splitlines())


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repo: Path) -> _GitHub:
    return _GitHub(monkeypatch, tmp_path, repo)


def test_lookup_finds_the_record_that_record_wrote_for_the_same_tree(
    github: _GitHub, tmp_path: Path, repo: Path
):
    commit = _Commit(repo, "1")
    github.record(tmp_path, commit)

    assert passed_tree.main(["lookup"]) == 0

    assert github.outputs() == {"tree": commit.tree, "passed": "true", "run": "4242"}


@pytest.mark.parametrize(
    ("arrange", "warns", "downloads"),
    [
        (lambda github, tmp_path, commit: None, False, 0),
        (lambda github, tmp_path, commit: github.record(tmp_path, commit, expired=True), False, 0),
        (lambda github, tmp_path, commit: github.fail.add("listing"), True, 0),
        (
            lambda github, tmp_path, commit: (
                github.record(tmp_path, commit),
                github.fail.add("keys"),
            ),
            True,
            0,
        ),
        (
            lambda github, tmp_path, commit: (
                github.record(tmp_path, commit),
                github.archives.update({1: b"not a zip"}),
            ),
            True,
            1,
        ),
    ],
    ids=["no record", "expired", "listing fails", "key set unreachable", "unreadable archive"],
)
def test_anything_short_of_a_verified_record_runs_the_suite(
    github: _GitHub, tmp_path: Path, repo: Path, capsys, arrange, warns: bool, downloads: int
):
    commit = _Commit(repo, "1")
    arrange(github, tmp_path, commit)

    assert passed_tree.main(["lookup"]) == 0

    assert github.outputs() == {"tree": commit.tree, "passed": "false"}
    assert ("::warning::" in capsys.readouterr().out) is warns
    assert sum(path.endswith("/zip") for path in github.requests) == downloads


def test_a_record_another_tree_uploaded_under_this_trees_name_runs_the_suite(
    github: _GitHub, tmp_path: Path, repo: Path, capsys
):
    elsewhere = _Commit(repo, "1")
    github.record(tmp_path, elsewhere)
    wanted = _Commit(repo, "2")
    github.artifacts[0]["name"] = f"{passed_tree.AUDIENCE}-{wanted.tree}"

    assert passed_tree.main(["lookup"]) == 0

    assert github.outputs() == {"tree": wanted.tree, "passed": "false"}
    assert "is refused: tree" in capsys.readouterr().out


def _encrypted(archive: bytes) -> bytes:
    """Flag every member as encrypted, which `zipfile` raises RuntimeError on reading."""
    marked = bytearray(archive)
    at = marked.find(b"PK\x01\x02")
    while at != -1:
        marked[at + 8] |= 0x01
        at = marked.find(b"PK\x01\x02", at + 4)
    return bytes(marked)


def test_lookup_reads_past_the_records_it_cannot_use(
    github: _GitHub, tmp_path: Path, repo: Path, capsys
):
    """A refused record and an unreadable one sit ahead of the genuine one."""
    elsewhere = _Commit(repo, "1")
    github.record(tmp_path, elsewhere)
    wanted = _Commit(repo, "2")
    github.artifacts[0]["name"] = f"{passed_tree.AUDIENCE}-{wanted.tree}"
    github.record(tmp_path, wanted)
    github.archives[2] = _encrypted(github.archives[2])
    github.record(tmp_path, wanted)

    assert passed_tree.main(["lookup"]) == 0

    assert github.outputs() == {"tree": wanted.tree, "passed": "true", "run": "4242"}
    assert capsys.readouterr().out.count("::warning::") == 2
    assert sum(path.endswith("/zip") for path in github.requests) == 3


def test_a_rerun_runs_the_suite_without_looking_for_a_record(
    github: _GitHub, tmp_path: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
):
    commit = _Commit(repo, "1")
    github.record(tmp_path, commit)
    monkeypatch.setenv("RUN_ATTEMPT", "2")

    assert passed_tree.main(["lookup"]) == 0

    assert github.outputs() == {"tree": commit.tree, "passed": "false"}
    assert github.requests == []


def test_the_lookup_job_passes_the_script_its_inputs_and_exposes_its_outputs():
    job = _JOBS[_LOOKUP]
    step = next(step for step in job["steps"] if "run" in step)

    assert step["run"] == "python3 .github/scripts/passed_tree.py lookup"
    assert step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "REPO": "${{ github.repository }}",
        "REPO_ID": "${{ github.repository_id }}",
        "DEFAULT_BRANCH": "${{ github.event.repository.default_branch }}",
        "RUN_ATTEMPT": "${{ github.run_attempt }}",
    }
    assert job["outputs"] == {
        "tree": f"${{{{ steps.{step['id']}.outputs.tree }}}}",
        "passed": f"${{{{ steps.{step['id']}.outputs.passed }}}}",
    }
    assert job["permissions"]["actions"] == "read"


def test_the_record_job_uploads_what_the_script_wrote_under_the_name_it_chose():
    job = _JOBS[_RECORD]
    write = next(step for step in job["steps"] if "run" in step)
    upload = next(step for step in job["steps"] if "upload-artifact" in step.get("uses", ""))

    assert write["run"] == 'python3 .github/scripts/passed_tree.py record "$RUNNER_TEMP/record"'
    assert upload["with"]["path"] == "${{ runner.temp }}/record/"
    assert upload["with"]["name"] == f"${{{{ steps.{write['id']}.outputs.name }}}}"


def test_only_the_record_job_can_request_a_token():
    """A token names a workflow and a commit and no job, so it means "the suite passed"
    only while the job that follows the suite is the one job that can mint it."""
    assert "id-token" not in _WORKFLOW["permissions"]
    assert [name for name, job in _JOBS.items() if "id-token" in job.get("permissions", {})] == [
        _RECORD
    ]
    assert _JOBS[_RECORD]["permissions"]["id-token"] == "write"


def test_neither_the_lookup_nor_the_record_can_fail_a_run():
    """Both only save work, so a `dev` push whose suite passed still deploys."""
    assert _JOBS[_LOOKUP]["continue-on-error"] is True
    assert _JOBS[_RECORD]["continue-on-error"] is True


def _clauses(condition: str) -> set[str]:
    """Split a job's ``if`` into its top-level ``&&`` clauses."""
    return {clause.strip() for clause in condition.split("&&")}


def test_only_a_record_skips_a_suite_job():
    """A suite job's `if` adds the lookup's `passed` output to the lookup's own guards and
    nothing else, and `!cancelled()` runs it whatever result a failed lookup reports."""
    guards = _clauses(_JOBS[_LOOKUP]["if"])
    for name in _SUITE:
        job = _JOBS[name]
        needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
        skip = (
            {"always()"}
            if name == "test"
            else {
                "!cancelled()",
                "needs.passed-tree.outputs.passed != 'true'",
            }
        )
        assert _LOOKUP in needs, name
        assert _clauses(job["if"]) == guards | skip, name


def test_the_record_names_the_success_of_every_suite_job():
    """A job that skipped tested nothing, so the record job's `if` reads each result. A
    re-run and a run on another base record nothing."""
    record = _JOBS[_RECORD]
    clauses = _clauses(record["if"])

    assert sorted(record["needs"]) == sorted([_LOOKUP, *_SUITE])
    assert clauses == {
        "github.run_attempt == 1",
        "(github.event_name == 'push' || github.base_ref == github.event.repository.default_branch)",
        "needs.passed-tree.outputs.passed == 'false'",
        *(f"needs.{name}.result == 'success'" for name in _SUITE),
    }


@pytest.mark.parametrize(
    ("passed", "shards", "code"),
    [
        ("true", "skipped", 0),
        ("false", "success", 0),
        ("false", "failure", 1),
        ("false", "cancelled", 1),
        ("", "skipped", 1),
        ("", "success", 0),
        ("true", "failure", 1),
    ],
)
def test_the_test_check_passes_on_passing_shards_or_a_record(passed: str, shards: str, code: int):
    step = _JOBS["test"]["steps"][0]
    assert step["env"] == {
        "PASSED": "${{ needs.passed-tree.outputs.passed }}",
        "SHARDS": "${{ needs.test-shard.result }}",
    }

    done = subprocess.run(
        ["bash", "-c", step["run"]],
        env={**os.environ, "PASSED": passed, "SHARDS": shards},
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == code, done.stdout


def test_durations_are_saved_only_when_the_shards_ran():
    """A push that passed on a record has no shard durations to merge."""
    later = _JOBS["test"]["steps"][1:]

    assert len(later) == 3
    for step in later:
        assert "needs.test-shard.result == 'success'" in _clauses(step["if"]), step["name"]
