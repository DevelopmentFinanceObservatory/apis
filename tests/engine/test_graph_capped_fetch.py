"""Tests the streamed byte cap and its interaction with the shared facet cache.

The cap applies to uncached response bytes; cached pairs bypass the fetch. The client has no row cap.
"""

from __future__ import annotations

import json

import pytest

from graph_generation.stamp import UNKNOWN_KEY
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.errors import GraphCapExceededError, GraphInfraError


class _FakeStreamResponse:
    """A fake httpx streaming response context manager: exposes status_code,
    iter_bytes() (chunked), and a no-op read() for the non-2xx path."""

    def __init__(self, chunks: list[bytes], *, status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code
        self.pulled: list[bytes] = []

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def iter_bytes(self):
        for chunk in self._chunks:
            self.pulled.append(chunk)
            yield chunk

    def read(self) -> None:
        pass


def test_observation_facets_batch_byte_cap_aborts_the_stream_before_full_read(monkeypatch):
    """The 800-byte cap aborts after two 500-byte chunks, before the third is read."""
    fake_response = _FakeStreamResponse([b"a" * 500, b"b" * 500, b"c" * 500])

    def fake_stream(method, url, *, json=None, headers=None, timeout=None):
        assert method == "POST"
        return fake_response

    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(client._client._ours._client, "stream", fake_stream)

    with pytest.raises(GraphCapExceededError) as exc_info:
        client.observation_facets_batch(["sv1"], ["e1"], needs_dates=True, max_bytes=800)

    assert exc_info.value.cap_kind == "bytes"
    assert exc_info.value.cap == 800
    assert exc_info.value.observed == 1000
    assert len(fake_response.pulled) == 2, "the third chunk must never be pulled off the wire"
    # A cap trip must never populate the per-pair cache with a partial read.
    assert client._client._ours._obs_cache == {}


def test_observation_facets_batch_byte_cap_not_tripped_returns_normally(monkeypatch):
    """A response under the byte cap parses and caches normally."""
    body = {
        "facets": {},
        "byVariable": {"sv1": {"byEntity": {"e1": {"orderedFacets": []}}}},
    }
    fake_response = _FakeStreamResponse([json.dumps(body).encode()])

    def fake_stream(method, url, *, json=None, headers=None, timeout=None):
        return fake_response

    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(client._client._ours._client, "stream", fake_stream)

    result = client.observation_facets_batch(["sv1"], ["e1"], max_bytes=1_000_000)

    assert result == {"sv1": {"e1": []}}


def test_observation_facets_batch_uncapped_never_streams(monkeypatch):
    """max_bytes=None (the default) must take the identical unbounded path
    _post itself would: client._client.stream must never even be called."""

    def fail_if_streamed(*args, **kwargs):
        raise AssertionError("stream() must not be called when max_bytes is None")

    def fake_post(url, json=None, headers=None):
        class _Resp:
            status_code = 200
            is_success = True

            def json(self_) -> dict:
                return {"facets": {}, "byVariable": {"sv1": {"byEntity": {"e1": {}}}}}

        return _Resp()

    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(client._client._ours._client, "stream", fail_if_streamed)
    monkeypatch.setattr(client._client._ours._client, "post", fake_post)

    result = client.observation_facets_batch(["sv1"], ["e1"])

    assert result == {"sv1": {"e1": []}}


def test_post_capped_bytes_non_200_drains_only_a_bounded_snippet(monkeypatch):
    """A 502 response drains only a diagnostic snippet, even below the configured byte cap."""
    fake_response = _FakeStreamResponse([b"e" * 4096 for _ in range(50)], status_code=502)

    def fake_stream(method, url, *, json=None, headers=None, timeout=None):
        return fake_response

    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(client._client._ours._client, "stream", fake_stream)

    with pytest.raises(GraphInfraError) as exc_info:
        client.observation_facets_batch(["sv1"], ["e1"], max_bytes=1_000_000)

    assert exc_info.value.upstream_status == 502
    assert len(fake_response.pulled) < 50, (
        "the full oversized error body must never be drained off the wire"
    )


def test_warm_cache_hit_never_reaches_the_byte_cap(monkeypatch):
    """An uncapped call warms the (SV, entity, needs_dates) cache; a later capped call bypasses the fetch."""
    body = {
        "facets": {"f1": {"provenanceId": "dc/base/TestProvenance"}},
        "byVariable": {
            "sv1": {
                "byEntity": {
                    "e1": {
                        "orderedFacets": [
                            {
                                "facetId": "f1",
                                "earliestDate": "P00000",
                                "latestDate": "P00004",
                                "obsCount": 5,
                                "observations": [
                                    {"date": f"P{i:05d}", "value": float(i)} for i in range(5)
                                ],
                            }
                        ]
                    }
                }
            }
        },
    }

    def fake_post(url, json=None, headers=None):
        class _Resp:
            status_code = 200
            is_success = True

            def json(self_) -> dict:
                return body

        return _Resp()

    client = LiveGraphClient(base="http://fake")
    monkeypatch.setattr(client._client._ours._client, "post", fake_post)

    # An earlier, uncapped caller (mirrors the family resolver's own coverage
    # probe) warms the cache with the full, unbounded facet.
    client.observation_facets_batch(["sv1"], ["e1"], needs_dates=True)
    assert (UNKNOWN_KEY, "sv1", "e1", True, None) in client._client._ours._obs_cache

    def fail_if_streamed(*args, **kwargs):
        raise AssertionError("a cache hit must never stream over the wire")

    monkeypatch.setattr(client._client._ours._client, "stream", fail_if_streamed)

    # A later, capped call for the same pair is a guaranteed cache hit:
    # max_bytes never engages (no stream() call), yet the cached facet is
    # still returned in full.
    result = client.observation_facets_batch(["sv1"], ["e1"], needs_dates=True, max_bytes=1)
    assert len(result["sv1"]["e1"]) == 1
    assert len(result["sv1"]["e1"][0].dates) == 5
