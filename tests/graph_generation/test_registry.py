"""`CacheRegistry` registration, removal, clearing, and stats behaviour."""

from __future__ import annotations

import logging

import pytest

from graph_generation.registry import CacheRegistry


def test_register_then_registered_names_then_close_removes_it() -> None:
    registry = CacheRegistry()
    handle = registry.register("alpha", stats=lambda: {"entries": 1}, clear=lambda: None)
    assert registry.registered_names() == ("alpha",)

    handle.close()
    assert registry.registered_names() == ()

    handle.close()  # idempotent


def test_duplicate_name_is_refused() -> None:
    registry = CacheRegistry()
    registry.register("alpha", stats=lambda: {}, clear=lambda: None)
    with pytest.raises(ValueError, match="alpha"):
        registry.register("alpha", stats=lambda: {}, clear=lambda: None)


def test_clear_all_runs_every_registered_clear() -> None:
    registry = CacheRegistry()
    calls: list[str] = []
    registry.register("alpha", stats=lambda: {}, clear=lambda: calls.append("alpha"))
    registry.register("beta", stats=lambda: {}, clear=lambda: calls.append("beta"))

    registry.clear_all()
    assert sorted(calls) == ["alpha", "beta"]


def test_stats_summary_reflects_every_registered_cache() -> None:
    registry = CacheRegistry()
    registry.register("alpha", stats=lambda: {"hits": 1}, clear=lambda: None)
    registry.register("beta", stats=lambda: {"hits": 2}, clear=lambda: None)

    assert registry.stats_summary() == {"alpha": {"hits": 1}, "beta": {"hits": 2}}


def test_clear_all_contains_a_raising_clear_and_runs_the_rest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = CacheRegistry()
    calls: list[str] = []
    registry.register("alpha", stats=lambda: {}, clear=lambda: calls.append("alpha"))

    def raising_clear() -> None:
        raise RuntimeError("boom")

    registry.register("beta", stats=lambda: {}, clear=raising_clear)
    registry.register("gamma", stats=lambda: {}, clear=lambda: calls.append("gamma"))

    with caplog.at_level(logging.ERROR, logger="graph_generation.registry"):
        registry.clear_all()  # must not propagate beta's failure

    assert sorted(calls) == ["alpha", "gamma"]
    failure_lines = [r.message for r in caplog.records if "clearing cache" in r.message]
    assert len(failure_lines) == 1
    assert "'beta'" in failure_lines[0]
