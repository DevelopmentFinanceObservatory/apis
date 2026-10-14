"""Run one shard of the suite, split by test file.

`--shard K/N` keeps the test files of group K of N and drops the others before pytest
imports them, so a shard collects its own files alone. ci.yml's `test-shard` job passes
it, and `tests/conftest.py` loads these hooks for the whole suite.

A file's weight is the summed duration of its tests in the file that pytest-split's
`--store-durations` writes (`--durations-path`, `.test_durations` by default). Recorded
files go heaviest first, each into the group that is lightest so far. A file with no
recorded test goes by a checksum of its path.

A file's group depends on its path and the durations file alone, and pytest decides which
files hold tests, so the N shards of one checkout and one durations file hold every test
file once. With no durations file every file goes by checksum. `--shard` takes directory
arguments alone, because pytest fails a named file that another shard holds.

This module imports only pytest and the standard library, so the conftest still loads in
installs without the engine extras.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest

_GROUPS = pytest.StashKey[dict[str, int]]()


def recorded_groups(durations: dict[str, float], count: int) -> dict[str, int]:
    """Map each test file named in `durations` to its group, numbered from 1."""
    weights: dict[str, float] = {}
    for nodeid, seconds in durations.items():
        path = nodeid.split("::", 1)[0]
        weights[path] = weights.get(path, 0.0) + seconds
    loads = [0.0] * count
    groups: dict[str, int] = {}
    for path, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        lightest = loads.index(min(loads))
        loads[lightest] += weight
        groups[path] = lightest + 1
    return groups


def _shard(value: str) -> tuple[int, int]:
    try:
        group, count = (int(part) for part in value.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard takes K/N, got {value!r}") from None
    if not 1 <= group <= count:
        raise pytest.UsageError(f"--shard K/N needs 1 <= K <= N, got {value!r}")
    return group, count


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="K/N",
        type=_shard,
        default=None,
        help="Run the test files of group K of N, split by recorded duration.",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    config = session.config
    if config.getoption("shard") is None:
        return
    named = [
        arg
        for arg in config.args
        if (config.invocation_params.dir / arg.split("::", 1)[0]).is_file()
    ]
    if named:
        raise pytest.UsageError(f"--shard splits directories, got the file {named[0]!r}")


@pytest.hookimpl(wrapper=True)
def pytest_collect_file(file_path: Path, parent: pytest.Collector):
    collectors = yield
    config = parent.config
    shard = config.getoption("shard")
    if shard is None or not collectors:
        return collectors
    group, count = shard
    if _GROUPS not in config.stash:
        durations = Path(config.getoption("durations_path", ".test_durations"))
        config.stash[_GROUPS] = recorded_groups(
            json.loads(durations.read_text()) if durations.is_file() else {}, count
        )
    path = file_path.relative_to(config.rootpath).as_posix()
    own = config.stash[_GROUPS].get(path) or zlib.crc32(path.encode()) % count + 1
    return collectors if own == group else []
