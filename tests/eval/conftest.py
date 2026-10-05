"""Shared fixtures and stubs for the Search eval test suite."""

from __future__ import annotations

import pytest

from dc_graph import Facet
from search.engine.errors import GraphReadFailure


class StubGraphClient:
    """In-memory graph with known dcids and facets keyed by `(stat_var, entity)`."""

    def __init__(
        self,
        known_dcids: set[str] | None = None,
        facets: dict[tuple[str, str], list[Facet]] | None = None,
    ):
        self.known_dcids: set[str] = known_dcids or set()
        self.facets: dict[tuple[str, str], list[Facet]] = facets or {}
        self.exists_batch_calls: list[list[str]] = []

    def exists_batch(self, dcids: list[str]) -> set[str]:
        self.exists_batch_calls.append(dcids)
        return {dcid for dcid in dcids if dcid in self.known_dcids}

    def observation_facets_batch(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> dict[str, dict[str, list[Facet]]]:
        return {
            sv: {e: self.facets[sv, e] for e in entities if (sv, e) in self.facets}
            for sv in stat_vars
        }


class RaisingGraphClient:
    """Graph client whose reads raise ``error``, by default the read failure the live client
    raises."""

    def __init__(self, error: Exception | None = None):
        self.error = error or GraphReadFailure("graph error: network unavailable")

    def exists_batch(self, dcids: list[str]) -> set[str]:
        raise self.error

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False):
        raise self.error


@pytest.fixture
def stub_graph():
    """A StubGraphClient with no pre-loaded dcids (configure in the test)."""
    return StubGraphClient()


@pytest.fixture
def raising_graph():
    return RaisingGraphClient()


def make_worked_example_response() -> dict:
    """Build the worked example response."""
    from tests.conftest import _build_worked_example_response

    return _build_worked_example_response()


@pytest.fixture
def worked_example_response() -> dict:
    """Worked example response."""
    return make_worked_example_response()
