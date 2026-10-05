"""Record the graph and SDMX requests a test makes and compare them with a stored list.

`checked_requests`, behind the `request_log` fixture, wraps the public methods of the offline
graph and SDMX doubles (`FakeGraph`, `FakeSdmxClient` and the materialise `FakeGraphReader`) for
one test and records each outermost call with its bound arguments, in call order. Collections
inside the arguments are sorted, because several callers build their dcid lists from sets and
string hashing varies their order between processes. At teardown it compares the calls with the
list stored under the test's node id in `tests/fixtures/requests/<test module>.json`, in order
unless the test is marked `unordered_requests`, and a test with no stored list fails. With
`SEARCH_RECORD_REQUESTS=1` it stores the list instead. Recording refuses to run under xdist,
so one process writes each file.

A resubmit starts its related-measures reads on a thread of its own and returns without
joining it when it answers no data (`related_read.submit_related_read`). Teardown joins the
threads of that pool the test started before it reads the list, so those reads are recorded
whatever the load on the machine, and none lands in a later test's list.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import inspect
import json
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from search.engine.region.related_read import RELATED_THREAD_PREFIX
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

_RECORD_ENV = "SEARCH_RECORD_REQUESTS"
_REQUESTS_DIR = Path(__file__).parent.parent / "fixtures" / "requests"
_RECORDED_CLASSES: tuple[type, ...] = (FakeGraph, FakeSdmxClient, FakeGraphReader)
# Each related read ends by its deadline, at most `SEARCH_RELATED_MEASURES_TIMEOUT_S` (1 s).
_RELATED_JOIN_TIMEOUT_S = 10.0


def _plain(value: object) -> Any:
    """A JSON value for a request argument, with every collection in sorted order."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, BaseModel):
        return _plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__qualname__,
            **{f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=_key_text)}
    if isinstance(value, set | frozenset | list | tuple):
        return sorted((_plain(item) for item in value), key=json.dumps)
    return f"<{type(value).__qualname__}>"


def _key_text(item: tuple[object, object]) -> str:
    return str(item[0])


class _RequestLog:
    """The outermost calls made on the recorded doubles, in call order."""

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []
        self._lock = threading.Lock()
        self._depth = threading.local()

    def wrap(self, owner: type, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(method)

        @functools.wraps(method)
        def recorded(double: object, *args: Any, **kwargs: Any) -> Any:
            depth = getattr(self._depth, "value", 0)
            if depth == 0:
                bound = signature.bind(double, *args, **kwargs).arguments
                # A deadline is a `time.monotonic()` instant, so only whether one is set is kept.
                arguments = {
                    key: ("set" if value is not None else None)
                    if key == "deadline"
                    else _plain(value)
                    for key, value in list(bound.items())[1:]
                }
                with self._lock:
                    self.calls.append([f"{owner.__name__}.{name}", arguments])
            self._depth.value = depth + 1
            try:
                return method(double, *args, **kwargs)
            finally:
                self._depth.value = depth

        return recorded


def _dump(stored: Mapping[str, list[Any]]) -> str:
    """The stored lists by node id, sorted, one compact call per line."""
    tests = [
        f"  {json.dumps(node_id)}: [\n"
        + ",\n".join(f"    {json.dumps(call, separators=(',', ':'))}" for call in calls)
        + "\n  ]"
        if calls
        else f"  {json.dumps(node_id)}: []"
        for node_id, calls in sorted(stored.items())
    ]
    return "{\n" + ",\n".join(tests) + "\n}\n"


def _first_difference(expected: list[Any], actual: list[Any]) -> str:
    for index, (want, got) in enumerate(zip(expected, actual, strict=False)):
        if want != got:
            return f"call {index}: expected {want!r}, got {got!r}"
    return f"expected {len(expected)} calls, got {len(actual)}"


def _set_difference(expected: list[Any], actual: list[Any]) -> str:
    want = {json.dumps(call) for call in expected}
    got = {json.dumps(call) for call in actual}
    return f"missing {sorted(want - got)[:3]}, unexpected {sorted(got - want)[:3]}"


def checked_requests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Fail the test unless it makes the graph and SDMX requests stored for it.

    The calls must match in order. A test marked `unordered_requests` resolves parts
    concurrently, so their interleaving and how often concurrent parts repeat a read vary
    between runs; its distinct calls must match. A test marked `unrecorded_requests` races a
    cancellation against its reads, so which reads run depends on timing, and it is not
    checked.
    """
    if request.node.get_closest_marker("unrecorded_requests") is not None:
        yield
        return
    log = _RequestLog()
    for owner in _RECORDED_CLASSES:
        for name, method in list(vars(owner).items()):
            if not name.startswith("_") and inspect.isfunction(method):
                monkeypatch.setattr(owner, name, log.wrap(owner, name, method))
    before = set(threading.enumerate())
    yield
    _join_related_reads(before)

    path = _REQUESTS_DIR / f"{Path(str(request.node.fspath)).stem}.json"
    stored: dict[str, list[Any]] = json.loads(path.read_text()) if path.exists() else {}
    node_id = request.node.nodeid.split("::", 1)[1]
    actual = json.loads(json.dumps(log.calls))
    if os.environ.get(_RECORD_ENV) == "1":
        if os.environ.get("PYTEST_XDIST_WORKER"):
            pytest.fail(f"{_RECORD_ENV}=1 records requests only without xdist (-p no:xdist)")
        stored[node_id] = actual
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(stored))
        return
    if node_id not in stored:
        pytest.fail(f"no stored requests for {node_id} in {path}; record with {_RECORD_ENV}=1")
    expected = stored[node_id]
    if request.node.get_closest_marker("unordered_requests") is not None:
        if {json.dumps(call) for call in expected} != {json.dumps(call) for call in actual}:
            pytest.fail(
                f"distinct requests differ from {path.name}: {_set_difference(expected, actual)}"
            )
    elif expected != actual:
        pytest.fail(f"requests differ from {path.name}: {_first_difference(expected, actual)}")


def _join_related_reads(before: set[threading.Thread]) -> None:
    """Wait for the related-measures threads started since `before` to end."""
    for thread in set(threading.enumerate()) - before:
        if thread.name.startswith(RELATED_THREAD_PREFIX):
            thread.join(_RELATED_JOIN_TIMEOUT_S)
            if thread.is_alive():
                pytest.fail(f"{thread.name} still runs {_RELATED_JOIN_TIMEOUT_S}s after the test")
