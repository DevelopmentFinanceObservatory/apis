"""Tests the search surface's startup loads of the origin catalogue and the graph release."""

from __future__ import annotations

import contextvars
import gc
import logging
import threading
import weakref
from pathlib import Path

import pytest

import graph_release.service as release_service
import origin_catalogue.service as catalogue_service
import search.engine.app as app_module
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from graph_release.service import ReleaseService
from origin_catalogue.service import CatalogueUriMissingError
from statvar_members import shared_enumerator
from tests.graph_release.support import build_view_release

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


def _join_release_load() -> None:
    for thread in threading.enumerate():
        if thread.name == "graph-release-load":
            thread.join(timeout=5)


@pytest.mark.parametrize("release_fails", [True, False])
def test_an_unset_catalogue_uri_raises_before_the_release_loads(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, release_fails: bool
) -> None:
    """The release load waits until the catalogue's error has reached the caller. Once it
    finishes, the abandoned load gives back the service it acquired or logs its error."""
    raised = threading.Event()
    waited: list[bool] = []
    released: list[object] = []
    release_error = FileNotFoundError("no release")
    service = object()

    def load_release() -> object:
        waited.append(raised.wait(timeout=5))
        if release_fails:
            raise release_error
        return service

    monkeypatch.delenv(catalogue_service._ARTIFACT_URI_ENV_VAR, raising=False)
    monkeypatch.setattr(app_module, "get_release_service", load_release)
    monkeypatch.setattr(app_module, "release_release_service", released.append)

    with (
        catalogue_service._singleton.isolated(),
        caplog.at_level(logging.INFO, logger=app_module.__name__),
    ):
        with pytest.raises(CatalogueUriMissingError):
            app_module._load_catalogue_and_release()
        raised.set()
        _join_release_load()

    assert waited == [True]
    failed = _events(caplog, "search_startup_stage_failed")
    loaded = [fields["stage"] for _, fields in _events(caplog, "search_startup_stage")]
    if release_fails:
        assert [(level, fields["stage"], fields["error"]) for level, fields in failed] == [
            (logging.ERROR, "graph_release", "FileNotFoundError")
        ]
        record = next(r for r in caplog.records if r.getMessage() == "search_startup_stage_failed")
        assert record.exc_info is not None
        assert record.exc_info[1] is release_error
        assert (loaded, released) == ([], [])
    else:
        assert failed == []
        assert (loaded, released) == (["graph_release"], [service])


def test_an_interrupted_catalogue_load_gives_back_the_release_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `KeyboardInterrupt` in the catalogue load abandons the release load as an error
    does, so a process that stays alive and retries startup holds no stray release lease."""
    released: list[object] = []
    service = object()

    def interrupted() -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(app_module, "get_catalogue_service", interrupted)
    monkeypatch.setattr(app_module, "get_release_service", lambda: service)
    monkeypatch.setattr(app_module, "release_release_service", released.append)

    with pytest.raises(KeyboardInterrupt):
        app_module._load_catalogue_and_release()
    _join_release_load()

    assert released == [service]


@pytest.mark.unbound_release
def test_an_abandoned_release_load_frees_the_release_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release service a failed start abandoned dies once its load finishes: its last
    lease gives back the shared member enumerator's swap subscription with it."""
    root = (tmp_path / "releases").as_uri()
    build_view_release(root)
    loaded: list[weakref.ref[ReleaseService]] = []

    def load_release() -> ReleaseService:
        service = release_service.get_release_service()
        loaded.append(weakref.ref(service))
        return service

    monkeypatch.setenv("GRAPH_RELEASE_URI", root)
    monkeypatch.delenv(catalogue_service._ARTIFACT_URI_ENV_VAR, raising=False)
    monkeypatch.setattr(app_module, "get_release_service", load_release)
    try:
        with catalogue_service._singleton.isolated(), release_service._singleton.isolated():
            with pytest.raises(CatalogueUriMissingError):
                app_module._load_catalogue_and_release()
            _join_release_load()
        gc.collect()
        assert len(loaded) == 1
        assert loaded[0]() is None
    finally:
        shared_enumerator().close()


def test_a_failed_release_load_gives_back_the_catalogue_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release's error reaches the caller, and the catalogue service the start acquired
    is given back, so the next start builds a fresh one."""
    built: list[object] = []

    def build() -> object:
        built.append(object())
        return built[-1]

    def load_release() -> object:
        raise FileNotFoundError("no release")

    monkeypatch.setattr(catalogue_service._singleton, "_build", build)
    monkeypatch.setattr(app_module, "get_release_service", load_release)
    with catalogue_service._singleton.isolated():
        for _ in range(2):
            with pytest.raises(FileNotFoundError):
                app_module._load_catalogue_and_release()

    assert len(built) == 2
