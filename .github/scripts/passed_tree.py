#!/usr/bin/env python3
"""Record that a tree passed CI, and look such a record up.

ci.yml's `record-passed-tree` job runs `record` once every suite job has succeeded, and
its `passed-tree` job runs `lookup` ahead of every suite job. Both run inside a checkout of the
commit under test, on the runner's own Python with the standard library alone.

A tree hash names the content of every tracked file, the workflow and the lock file
included, so two commits with one tree give CI the same tracked inputs. A squash merge of
a PR that was level with its base has the tree the PR's own run tested.

A record is an artifact named `ci-passed-tree-<tree sha>` holding two files:

- `token`, the recording job's GitHub OIDC token for the audience `ci-passed-tree`.
  GitHub signs its claims, so no workflow can choose them. `sha` is the commit the run
  checked out and `workflow_sha` the commit its workflow file came from.
- `commit`, the raw Git commit object of the checked-out commit. Its SHA-1 is the commit
  id, and its first line names the tree.

`lookup` accepts a record for a tree only when GitHub's published keys verify the token,
the token belongs to this repository's ci.yml, `sha` and `workflow_sha` are one commit,
that commit is the `commit` file, and its tree is the tree under test. The run that
recorded a tree therefore ran the workflow file that tree holds, on that tree. A run on
another branch can upload an artifact of any name, and its token names its own commit.

A run also restores Actions caches, which a tree hash leaves unnamed. The token must
therefore come from a push to the default branch or a pull request into it. Those runs
restore caches saved on the default branch and, for a pull request, by that pull request's
own earlier runs. The token must also come from a run's first attempt, so a job re-run
weeks later mints no record for an old tree.

The token expires minutes after issue and names the audience `ci-passed-tree` alone,
which nothing else accepts. `lookup` reads its signature, ignores its expiry, and bounds
its age by `iat`.

`lookup` reads REPO, REPO_ID, DEFAULT_BRANCH and RUN_ATTEMPT (github.repository,
github.repository_id, github.event.repository.default_branch, github.run_attempt) and
needs GH_TOKEN with `actions: read`. It writes `tree=<sha>` and
`passed=true|false` to $GITHUB_OUTPUT, and `run=<id>` of the recording run when
passed=true. Any attempt after the first reports passed=false without looking, so "Re-run
all jobs" runs the suite on a tree that has a record. A listing, key set or record that
cannot be read or does not verify reports passed=false with a ::warning::, and the suite
runs.

`record <dir>` needs `id-token: write`. It writes the two files into <dir> and the
artifact name to $GITHUB_OUTPUT as `name=`.
"""

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

AUDIENCE = "ci-passed-tree"
ISSUER = "https://token.actions.githubusercontent.com"
JWKS_URL = f"{ISSUER}/.well-known/jwks"
WORKFLOW = ".github/workflows/ci.yml"
MAX_AGE_SECONDS = 7 * 24 * 3600
# A recording runner's clock may run ahead of this one's.
CLOCK_SKEW_SECONDS = 300
# DER prefix of a SHA-256 DigestInfo (RFC 8017, section 9.2).
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


class Refused(Exception):
    """A record that does not vouch for the tree. The message names what failed."""


def _b64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def verified_claims(token: str, jwks: dict) -> dict:
    """Return the claims of an RS256 token one of `jwks` signed, or raise Refused.

    The check rebuilds the whole PKCS #1 v1.5 encoding of the token's digest and compares
    it with the signature raised to the public exponent, so no part of the padding goes
    unread.
    """
    try:
        header_part, claims_part, signature_part = token.strip().split(".")
        header = json.loads(_b64url(header_part))
        signature = _b64url(signature_part)
        if header["alg"] != "RS256":
            raise Refused("algorithm")
        key = next(
            key for key in jwks["keys"] if key["kid"] == header["kid"] and key["kty"] == "RSA"
        )
        modulus = int.from_bytes(_b64url(key["n"]), "big")
        exponent = int.from_bytes(_b64url(key["e"]), "big")
    except StopIteration:
        raise Refused("signing key") from None
    except (ValueError, KeyError, TypeError, AttributeError):
        raise Refused("token format") from None
    size = (modulus.bit_length() + 7) // 8
    digest = hashlib.sha256(f"{header_part}.{claims_part}".encode()).digest()
    tail = b"\x00" + _SHA256_DIGEST_INFO + digest
    if len(signature) != size or size < len(tail) + 10:
        raise Refused("signature")
    expected = b"\x00\x01" + b"\xff" * (size - 2 - len(tail)) + tail
    if pow(int.from_bytes(signature, "big"), exponent, modulus).to_bytes(size, "big") != expected:
        raise Refused("signature")
    try:
        claims = json.loads(_b64url(claims_part))
    except ValueError:
        raise Refused("token format") from None
    if not isinstance(claims, dict):
        raise Refused("token format")
    return claims


def recorded_run(
    *,
    token: str,
    commit: bytes,
    tree: str,
    repo: str,
    repo_id: str,
    branch: str,
    jwks: dict,
    now: float,
) -> str:
    """Return the id of the run whose record vouches for `tree`, or raise Refused.

    `branch` is the repository's default branch.
    """
    claims = verified_claims(token, jwks)
    sha = claims.get("sha")
    issued = claims.get("iat")
    event = claims.get("event_name")
    checks = {
        "issuer": claims.get("iss") == ISSUER,
        "audience": claims.get("aud") == AUDIENCE,
        "repository": str(claims.get("repository_id")) == str(repo_id),
        "workflow": str(claims.get("workflow_ref")).startswith(f"{repo}/{WORKFLOW}@"),
        "workflow commit": isinstance(sha, str) and claims.get("workflow_sha") == sha,
        "branch": (event, claims.get("ref")) == ("push", f"refs/heads/{branch}")
        or (event, claims.get("base_ref")) == ("pull_request", branch),
        "attempt": str(claims.get("run_attempt")) == "1",
        "age": isinstance(issued, int) and -CLOCK_SKEW_SECONDS <= now - issued <= MAX_AGE_SECONDS,
        "commit": hashlib.sha1(b"commit %d\x00" % len(commit) + commit).hexdigest() == sha,
        "tree": commit.split(b"\n", 1)[0] == b"tree " + tree.encode(),
    }
    failed = [name for name, held in checks.items() if not held]
    if failed:
        raise Refused(", ".join(failed))
    return str(claims.get("run_id"))


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def _gh(path: str) -> bytes:
    return subprocess.run(["gh", "api", path], check=True, capture_output=True).stdout


def _fetch_json(request: str | urllib.request.Request) -> dict:
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _emit(line: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as handle:
            handle.write(line + "\n")
    else:
        print(line)


def _head_tree() -> str:
    return _git("rev-parse", "--verify", "HEAD^{tree}").decode().strip()


def lookup() -> int:
    repo, repo_id = os.environ["REPO"], os.environ["REPO_ID"]
    branch = os.environ["DEFAULT_BRANCH"]
    attempt = os.environ["RUN_ATTEMPT"]
    tree = _head_tree()
    _emit(f"tree={tree}")
    if attempt != "1":
        print(f"::notice::run attempt {attempt}: running the suite on tree {tree}.")
        _emit("passed=false")
        return 0

    name = f"{AUDIENCE}-{tree}"
    try:
        listing = json.loads(_gh(f"repos/{repo}/actions/artifacts?name={name}&per_page=100"))
        artifacts = [item for item in listing["artifacts"] if item["expired"] is False]
        jwks = _fetch_json(JWKS_URL) if artifacts else {}
    except Exception as error:
        print(f"::warning::could not look for {name} ({type(error).__name__}); running the suite.")
        _emit("passed=false")
        return 0

    for artifact in artifacts:
        try:
            archive = zipfile.ZipFile(
                io.BytesIO(_gh(f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"))
            )
            run = recorded_run(
                token=archive.read("token").decode(),
                commit=archive.read("commit"),
                tree=tree,
                repo=repo,
                repo_id=repo_id,
                branch=branch,
                jwks=jwks,
                now=time.time(),
            )
        except Refused as refusal:
            print(f"::warning::artifact {artifact['id']} named {name} is refused: {refusal}.")
        # An artifact holds whatever the run that uploaded it chose to put there.
        except Exception as error:
            print(f"::warning::could not read artifact {artifact['id']} ({type(error).__name__}).")
        else:
            print(f"tree={tree} passed in run {run}; skipping the suite.")
            _emit(f"run={run}")
            _emit("passed=true")
            summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary:
                url = f"https://github.com/{repo}/actions/runs/{run}"
                with open(summary, "a") as handle:
                    handle.write(
                        f"The tree `{tree}` passed lint, test, docker and the install jobs in "
                        f"[run {run}]({url}), so this run skips them.\n\n"
                        '"Re-run all jobs" runs them on this commit.\n'
                    )
            return 0

    print(f"tree={tree} has no record; running the suite.")
    _emit("passed=false")
    return 0


def record(directory: Path) -> int:
    request = urllib.request.Request(
        f"{os.environ['ACTIONS_ID_TOKEN_REQUEST_URL']}&audience={AUDIENCE}",
        headers={"Authorization": f"bearer {os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']}"},
    )
    token = _fetch_json(request)["value"]
    directory.mkdir(parents=True)
    (directory / "token").write_text(token)
    (directory / "commit").write_bytes(_git("cat-file", "commit", "HEAD"))
    _emit(f"name={AUDIENCE}-{_head_tree()}")
    return 0


def main(argv: list[str]) -> int:
    match argv:
        case ["lookup"]:
            return lookup()
        case ["record", directory]:
            return record(Path(directory))
    print("usage: passed_tree.py lookup | record <dir>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
