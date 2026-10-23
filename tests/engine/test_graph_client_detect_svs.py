"""Tests `detect_svs` caching, error propagation, and validation of upstream candidates on our
resolve. The UN origin is off here; test_detect_two_resolves.py covers the UN resolve."""

import pytest

import dc_graph
from code_index.query import CodeIndex
from code_index.types import IndexStamp
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.errors import GraphInfraError
from tests.graph_release.support import serving_index
from tests.origin_catalogue.fake_catalogue import FakeCatalogue


@pytest.fixture(autouse=True)
def _un_origin_off(bind_catalogue) -> None:
    bind_catalogue(FakeCatalogue(un_system_enabled=False))


def test_detect_svs_cache_hit_skips_network(monkeypatch) -> None:
    """A second detect_svs call for the same query returns the cached result."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        # /v2/resolve "indicator" candidate shape: entities[*].candidates[*]
        # with dcid/typeOf/metadata.score/metadata.sentence.
        return {
            "entities": [
                {
                    "candidates": [
                        {
                            "dcid": "sv/Z",
                            "typeOf": ["StatisticalVariable"],
                            "metadata": {"score": "0.9", "sentence": "Z"},
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    r1 = client.detect_svs("test query")
    r2 = client.detect_svs("test query")

    assert call_count[0] == 1, "Second detect_svs for same query must hit cache"
    assert r1 == r2


def _index(content_hash: str) -> CodeIndex:
    stamp = IndexStamp(
        graph_endpoint="test",
        exported_at="2026-10-11T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash=content_hash,
    )
    return CodeIndex(stamp=stamp, declarations=(), codes=(), statvars=())


def test_the_served_code_index_keys_a_detection(monkeypatch) -> None:
    """The rival rows come from the bound release's code index, so a release serving another
    index detects afresh and one serving the first hits."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return {"entities": []}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")
    first, second = _index("index-1"), _index("index-2")

    for index in (first, second, first):
        with serving_index(index):
            client.detect_svs("test query")

    assert call_count[0] == 2


def test_detect_svs_cache_error_does_not_cache(monkeypatch) -> None:
    """A failed detect_svs call does not populate the cache."""

    def fake_post(self, path, payload, *, deadline=None):
        raise GraphInfraError("simulated 503", upstream_status=503)

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.detect_svs("failing query")

    assert len(client._detect_cache) == 0


# ---------------------------------------------------------------------------
# upstream_status propagation
# ---------------------------------------------------------------------------


def test_detect_svs_non2xx_sets_upstream_status(monkeypatch) -> None:
    """A non-2xx detect response carries upstream_status on GraphInfraError."""

    def fake_post(self, path, payload, *, deadline=None):
        raise GraphInfraError("simulated 502", upstream_status=502)

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError) as exc_info:
        client.detect_svs("any query")

    assert exc_info.value.upstream_status == 502


# ---------------------------------------------------------------------------
# Defensive parsing of /v2/resolve candidates
# ---------------------------------------------------------------------------


def test_detect_svs_missing_score_raises_graph_infra_error(monkeypatch) -> None:
    """A candidate without `metadata.score` raises `GraphInfraError` before sorting."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "candidates": [
                        {
                            "dcid": "sv/Z",
                            "typeOf": ["StatisticalVariable"],
                            "metadata": {"sentence": "no score here"},
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.detect_svs("bad score query")


def test_detect_svs_non_numeric_score_raises_graph_infra_error(monkeypatch) -> None:
    """A non-numeric metadata.score raises GraphInfraError."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "candidates": [
                        {
                            "dcid": "sv/Z",
                            "typeOf": ["StatisticalVariable"],
                            "metadata": {"score": "not-a-number"},
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.detect_svs("non-numeric score query")


def test_detect_svs_malformed_candidates_raises_graph_infra_error(monkeypatch) -> None:
    """candidates that is not a list re-raises as GraphInfraError (structural walk guard)."""

    def fake_post(self, path, payload, *, deadline=None):
        return {"entities": [{"candidates": "not-a-list"}]}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.detect_svs("malformed candidates query")


def test_detect_svs_infinite_score_raises_graph_infra_error(monkeypatch) -> None:
    """A metadata.score of Infinity (a valid json.loads literal) must not win
    dominance/sort by fiat and must not defeat the unknown-gate's "< threshold"
    check (inf < threshold is always False)."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "candidates": [
                        {
                            "dcid": "sv/Z",
                            "typeOf": ["StatisticalVariable"],
                            "metadata": {"score": float("inf")},
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.detect_svs("infinite score query")


def test_detect_svs_nan_score_raises_graph_infra_error(monkeypatch) -> None:
    """A metadata.score of NaN (a valid json.loads literal) must not silently
    defeat every "< threshold" comparison (nan < threshold is always False)."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "candidates": [
                        {
                            "dcid": "sv/Z",
                            "typeOf": ["StatisticalVariable"],
                            "metadata": {"score": float("nan")},
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.detect_svs("nan score query")


def test_detect_svs_non_string_dcid_raises_graph_infra_error(monkeypatch) -> None:
    """A truthy non-string dcid (e.g. upstream returns an int) must raise
    GraphInfraError via the DetectedSV pydantic ValidationError, not escape the
    module's uniform fail-loud contract as an uncaught ValidationError."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "candidates": [
                        {
                            "dcid": 12345,
                            "typeOf": ["StatisticalVariable"],
                            "metadata": {"score": 0.9},
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    with pytest.raises(GraphInfraError):
        client.detect_svs("non-string dcid query")
