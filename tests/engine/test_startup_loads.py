"""Tests the search surface's startup loads of the origin catalogue and the graph release."""

from __future__ import annotations

import contextvars
import logging
import threading

import pytest

import search.engine.app as app_module
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR

_BOUND: contextvars.ContextVar[str | None] = contextvars.ContextVar("_BOUND", default=None)


def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[tuple[int, dict]]:
    return [
        (record.levelno, fields)
        for record in caplog.records
        if (fields := getattr(record, JSON_FIELDS_ATTR, {})).get("event") == event
    ]


def test_the_release_loads_beside_the_catalogue_under_the_callers_context(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Each load waits at a barrier the two share, so the call returns only when they run
    at once."""
    barrier = threading.Barrier(2, timeout=5)
    seen: list[str | None] = []
    catalogue, release = object(), object()

    def load_catalogue() -> object:
        barrier.wait()
        return catalogue

    def load_release() -> object:
        seen.append(_BOUND.get())
        barrier.wait()
        return release

    monkeypatch.setattr(app_module, "get_catalogue_service", load_catalogue)
    monkeypatch.setattr(app_module, "get_release_service", load_release)
    context = contextvars.copy_context()
    context.run(_BOUND.set, "bound")

    with caplog.at_level(logging.INFO, logger=app_module.__name__):
        loaded = context.run(app_module._load_catalogue_and_release)

    assert loaded == (catalogue, release)
    assert seen == ["bound"]
    stages = _events(caplog, "search_startup_stage")
    assert sorted((level, fields["stage"]) for level, fields in stages) == [
        (logging.INFO, "graph_release"),
        (logging.INFO, "origin_catalogue"),
    ]
    assert all(isinstance(fields["duration_ms"], int) for _, fields in stages)


@pytest.mark.parametrize("release_fails", [True, False])
def test_a_failing_catalogue_raises_after_logging_the_release_outcome(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, release_fails: bool
) -> None:
    release_error = FileNotFoundError("no release")

    def load_catalogue() -> object:
        raise ValueError("no catalogue")

    def load_release() -> object:
        if release_fails:
            raise release_error
        return object()

    monkeypatch.setattr(app_module, "get_catalogue_service", load_catalogue)
    monkeypatch.setattr(app_module, "get_release_service", load_release)

    with caplog.at_level(logging.INFO, logger=app_module.__name__), pytest.raises(ValueError):
        app_module._load_catalogue_and_release()

    failed = _events(caplog, "search_startup_stage_failed")
    loaded = [fields["stage"] for _, fields in _events(caplog, "search_startup_stage")]
    if release_fails:
        assert [(level, fields["stage"], fields["error"]) for level, fields in failed] == [
            (logging.ERROR, "graph_release", "FileNotFoundError")
        ]
        record = next(r for r in caplog.records if r.getMessage() == "search_startup_stage_failed")
        assert record.exc_info is not None
        assert record.exc_info[1] is release_error
        assert loaded == []
    else:
        assert failed == []
        assert loaded == ["graph_release"]
