"""Shared doubles for the measurement scripts' offline tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

SKILLS_INDEX_SENTINEL: Any = object()


@pytest.fixture
def fake_live_clients(monkeypatch):
    """Replace the live graph and SDMX clients and the skills index snapshot the scripts import.

    Returns the list of graphs the skills index was built on.
    """
    import search.eval.live_clients as live_clients

    index_graphs: list[object] = []

    @contextmanager
    def _clients():
        yield ("graph-stub", "sdmx-stub")

    def _skills_index_snapshot(graph):
        index_graphs.append(graph)
        return SKILLS_INDEX_SENTINEL

    monkeypatch.setattr(live_clients, "live_engine_clients", _clients)
    monkeypatch.setattr(live_clients, "skills_index_snapshot", _skills_index_snapshot)
    return index_graphs
