"""Pins ``skills_index_snapshot``: the one skills index build an eval run serves."""

from __future__ import annotations

from typing import Any

import search.eval.live_clients as live_clients
from search.eval.live_clients import skills_index_snapshot

# Stand-in for the graph client: the patched build reads nothing from it.
_GRAPH: Any = "graph-stub"


def test_a_successful_build_serves_the_same_index_for_every_read(monkeypatch):
    built = object()
    build_graphs: list[object] = []

    def _fake_build_index(*, graph):
        build_graphs.append(graph)
        return built

    monkeypatch.setattr(live_clients, "build_index", _fake_build_index)

    snapshot = skills_index_snapshot(_GRAPH)

    assert [snapshot.index for _ in range(3)] == [built, built, built]
    assert build_graphs == ["graph-stub"]


def test_a_failed_build_leaves_the_index_unset_and_warns(monkeypatch, capsys):
    def _failing_build_index(*, graph):
        raise RuntimeError("graph down")

    monkeypatch.setattr(live_clients, "build_index", _failing_build_index)

    snapshot = skills_index_snapshot(_GRAPH)

    assert snapshot.index is None
    out = capsys.readouterr().out
    assert "WARNING: the skills index build failed" in out
    assert "graph down" in out
