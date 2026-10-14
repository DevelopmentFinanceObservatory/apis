"""Tests for `tests/file_shards.py`, which runs one shard of the suite split by test file."""

from __future__ import annotations

import json
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

from tests.file_shards import recorded_groups

pytest_plugins = ["pytester"]

_JOBS = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml").read_text()
)["jobs"]
_COUNT = 3
# Six test files; the durations name four of them, and one of those twice.
_FILES = [f"test_{name}.py" for name in "abcdef"]
_DURATIONS = {
    "test_a.py::test_one": 6.0,
    "test_a.py::test_two[x::y]": 3.0,
    "test_b.py::test_one": 5.0,
    "test_c.py::test_one": 4.0,
    "test_d.py::test_one": 1.0,
}


def test_recorded_files_go_heaviest_first_into_the_lightest_group():
    assert recorded_groups(_DURATIONS, _COUNT) == {
        "test_a.py": 1,
        "test_b.py": 2,
        "test_c.py": 3,
        "test_d.py": 3,
    }


def _collected(pytester: pytest.Pytester, *args: str) -> list[str]:
    result = pytester.runpytest("--collect-only", "-q", "-p", "tests.file_shards", *args)
    assert result.ret == 0, result.outlines
    return [line for line in result.outlines if "::" in line]


@pytest.mark.parametrize("recorded", [_DURATIONS, None], ids=["durations", "no durations file"])
def test_the_shards_of_a_run_hold_every_test_once(pytester: pytest.Pytester, recorded):
    for name in _FILES:
        pytester.makepyfile(**{name: "def test_one(): pass\ndef test_two(): pass\n"})
    if recorded is not None:
        pytester.makefile(".json", durations=json.dumps(recorded))
    path = ("--durations-path", "durations.json")

    shards = [_collected(pytester, "--shard", f"{k}/{_COUNT}", *path) for k in (1, 2, 3)]

    assert sorted(test for shard in shards for test in shard) == sorted(_collected(pytester, *path))
    if recorded is not None:
        # The recorded files sit where their weights put them, whatever joins them.
        for name, group in recorded_groups(recorded, _COUNT).items():
            assert f"{name}::test_one" in shards[group - 1]


def test_a_shard_never_imports_another_shards_file(pytester: pytest.Pytester):
    pytester.makepyfile(test_a="def test_one(): pass\n", test_b="raise RuntimeError('imported')\n")
    pytester.makefile(
        ".json", durations=json.dumps({"test_a.py::test_one": 2.0, "test_b.py::test_one": 1.0})
    )

    collected = _collected(pytester, "--shard", "1/2", "--durations-path", "durations.json")

    assert collected == ["test_a.py::test_one"]


def test_ci_runs_every_group_of_the_matrix_it_declares():
    job = _JOBS["test-shard"]
    groups = job["strategy"]["matrix"]["group"]
    run = next(step["run"] for step in job["steps"] if step.get("name") == "Run tests")

    assert groups == list(range(1, len(groups) + 1))
    assert "--shard ${{ matrix.group }}/${{ strategy.job-total }} " in run


def test_each_shard_downloads_the_durations_the_lookup_job_uploaded_and_reads_no_other():
    """A shard that read the newest cache entry for itself could split by another file
    than its siblings did, and leave out a file none of them ran. The lookup job uploads
    one file whether or not its lookup failed, and a shard without it fails."""
    after_lookup = "${{ !cancelled() && steps.lookup.outputs.passed != 'true' }}"
    lookup = _JOBS["passed-tree"]["steps"]
    restore = next(step for step in lookup if "cache/restore" in step.get("uses", ""))
    upload = next(step for step in lookup if "upload-artifact" in step.get("uses", ""))
    between = lookup[lookup.index(restore) + 1 : lookup.index(upload)]
    steps = _JOBS["test-shard"]["steps"]
    download = next(step for step in steps if "download-artifact" in step.get("uses", ""))
    run = next(step for step in steps if step.get("name") == "Run tests")
    gathered = next(
        step for step in _JOBS["test"]["steps"] if "download-artifact" in step.get("uses", "")
    )

    assert restore["with"]["path"] == upload["with"]["path"] == ".test_durations"
    assert [step["if"] for step in (restore, *between, upload)] == [after_lookup] * 3
    assert [step["run"] for step in between] == [
        "test -f .test_durations || echo '{}' > .test_durations"
    ]
    assert upload["with"]["if-no-files-found"] == "error"
    assert "overwrite" not in upload["with"]
    assert "if" not in download
    assert "continue-on-error" not in download
    assert download["with"] == {"name": upload["with"]["name"]}
    assert steps.index(download) < steps.index(run)
    assert not any("actions/cache" in step.get("uses", "") for step in steps)
    # The `test` job gathers each shard's own durations by pattern.
    assert not fnmatch(upload["with"]["name"], gathered["with"]["pattern"])


@pytest.mark.parametrize(
    "args",
    [["--shard", "3"], ["--shard", "0/3"], ["--shard", "4/3"], ["--shard", "a/b"], ["--shard", "1/1", "test_a.py"], ["--shard", "1/1", "test_a.py::test_one"]],
    ids=["no total", "group zero", "group past the total", "words", "a file", "a test"],
)  # fmt: skip
def test_a_malformed_shard_is_a_usage_error(pytester: pytest.Pytester, args: list[str]):
    pytester.makepyfile(test_a="def test_one(): pass\n")

    result = pytester.runpytest("-p", "tests.file_shards", *args)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
