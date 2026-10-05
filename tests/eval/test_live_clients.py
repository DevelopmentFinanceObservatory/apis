"""score_clients yields the checks' live reads and closes what it opened."""

import pytest

from search.eval import live_clients


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def opened(monkeypatch) -> tuple[_Closable, _Closable]:
    graph, sdmx = _Closable(), _Closable()
    monkeypatch.setattr(
        live_clients.LiveGraphClient, "from_search_settings", staticmethod(lambda _s: graph)
    )
    monkeypatch.setattr(live_clients, "LiveSdmxClient", lambda: sdmx)
    return graph, sdmx


def test_score_clients_yields_the_reads_and_closes_them_when_the_block_raises(monkeypatch, opened):
    graph, sdmx = opened
    coverage = object()
    monkeypatch.setattr(live_clients, "get_coverage_service", lambda: coverage)

    with (
        pytest.raises(RuntimeError, match="scoring failed"),
        live_clients.score_clients() as clients,
    ):
        assert (clients.graph, clients.sdmx, clients.coverage) == (graph, sdmx, coverage)
        assert not graph.closed
        raise RuntimeError("scoring failed")

    assert graph.closed
    assert sdmx.closed


def test_score_clients_closes_the_clients_when_the_coverage_service_fails(monkeypatch, opened):
    def _missing_uri():
        raise RuntimeError("COVERAGE_ARTIFACT_URI is unset")

    monkeypatch.setattr(live_clients, "get_coverage_service", _missing_uri)

    with (
        pytest.raises(RuntimeError, match="COVERAGE_ARTIFACT_URI"),
        live_clients.score_clients(),
    ):
        pass

    assert all(client.closed for client in opened)
