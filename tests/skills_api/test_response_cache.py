"""Pins generation-scoped body caching, admission limits, and conditional responses."""

from __future__ import annotations

from fastapi import Request

import skills_api.doors._response_cache as response_cache
from skills_api.doors._response_cache import GenerationBodyCache


def _request(if_none_match: str | None = None) -> Request:
    headers = [(b"if-none-match", if_none_match.encode())] if if_none_match else []
    return Request({"type": "http", "headers": headers})


def test_build_payload_runs_once_per_generation_and_key():
    cache = GenerationBodyCache()
    generation = object()
    calls = []

    def build_payload():
        calls.append(1)
        return {"n": len(calls)}

    first = cache.respond(
        generation=generation,
        key="a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=build_payload,
    )
    second = cache.respond(
        generation=generation,
        key="a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=build_payload,
    )

    assert len(calls) == 1
    assert first.body == second.body
    assert first.headers["etag"] == second.headers["etag"]


def test_different_keys_within_one_generation_cache_independently():
    cache = GenerationBodyCache()
    generation = object()

    first = cache.respond(
        generation=generation,
        key="skill-a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=lambda: {"id": "skill-a"},
    )
    second = cache.respond(
        generation=generation,
        key="skill-b",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=lambda: {"id": "skill-b"},
    )

    assert first.body != second.body
    assert first.headers["etag"] != second.headers["etag"]
    assert len(cache._bodies) == 2


def test_generation_change_replaces_the_whole_cache_rather_than_growing_it():
    """After several generations, each carrying its own key, the cache
    holds only the current generation's own entries -- proof the cache
    cannot grow without bound as generations succeed one another.
    """
    cache = GenerationBodyCache()

    for n in range(5):
        generation = object()
        cache.respond(
            generation=generation,
            key=f"key-{n}",
            request=_request(),
            cache_control="public, max-age=900",
            build_payload=lambda n=n: {"n": n},
        )
        assert len(cache._bodies) == 1


def test_new_generation_never_serves_the_previous_generations_body():
    cache = GenerationBodyCache()
    first_generation = object()
    second_generation = object()

    first = cache.respond(
        generation=first_generation,
        key="skill-a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=lambda: {"revision": 1},
    )
    second = cache.respond(
        generation=second_generation,
        key="skill-a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=lambda: {"revision": 2},
    )

    assert first.body != second.body
    assert first.headers["etag"] != second.headers["etag"]
    assert b'"revision":2' in second.body


def test_matching_if_none_match_answers_304_off_the_cached_etag():
    cache = GenerationBodyCache()
    generation = object()

    first = cache.respond(
        generation=generation,
        key=None,
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=lambda: {"skills": []},
    )
    etag = first.headers["etag"]

    conditional = cache.respond(
        generation=generation,
        key=None,
        request=_request(if_none_match=etag),
        cache_control="public, max-age=900",
        build_payload=lambda: (_ for _ in ()).throw(
            AssertionError("build_payload must not run again for a 304")
        ),
    )

    assert conditional.status_code == 304
    assert conditional.body == b""


def test_cache_module_uses_the_public_static_response_seam():
    """This module reaches `dfo_platform.static_response` through its two
    public functions only -- `encode_json_body` and `serve_encoded_json`,
    the same names `dynamic_public_json` itself is built from -- never
    through `_conditional_response`/`_strong_etag`, so a refactor inside
    `static_response` that keeps the public seam's contract cannot break
    this module silently.
    """
    import dfo_platform.static_response as static_response

    assert response_cache.encode_json_body is static_response.encode_json_body
    assert response_cache.serve_encoded_json is static_response.serve_encoded_json
    assert not hasattr(response_cache, "_conditional_response")
    assert not hasattr(response_cache, "_strong_etag")


def test_past_budget_the_cache_stops_admitting_but_still_answers_correctly(monkeypatch):
    """A budget of 1 byte means no real JSON body is ever admitted --
    `respond` still serves the right body and ETag on every call, still
    answers a matching `If-None-Match` with a 304, and re-runs
    `build_payload` every time rather than ever storing the declined
    entry.
    """
    monkeypatch.setattr(response_cache, "_MAX_CACHED_BYTES", 1)

    cache = GenerationBodyCache()
    generation = object()
    calls = []

    def build_payload():
        calls.append(1)
        return {"skills": []}

    first = cache.respond(
        generation=generation,
        key="a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=build_payload,
    )
    second = cache.respond(
        generation=generation,
        key="a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=build_payload,
    )

    assert len(calls) == 2
    assert first.body == second.body
    assert first.headers["etag"] == second.headers["etag"]
    assert cache._bodies == {}

    conditional = cache.respond(
        generation=generation,
        key="a",
        request=_request(if_none_match=first.headers["etag"]),
        cache_control="public, max-age=900",
        build_payload=build_payload,
    )

    assert len(calls) == 3
    assert conditional.status_code == 304
    assert conditional.body == b""


def test_declining_a_new_entry_never_evicts_one_already_admitted(monkeypatch):
    cache = GenerationBodyCache()
    generation = object()

    first = cache.respond(
        generation=generation,
        key="a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=lambda: {"id": "a"},
    )
    # No budget left for anything past the entry already admitted for "a".
    monkeypatch.setattr(response_cache, "_MAX_CACHED_BYTES", len(first.body))

    cache.respond(
        generation=generation,
        key="b",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=lambda: {"id": "b, a larger payload than a"},
    )

    assert "b" not in cache._bodies

    calls = []

    def build_payload_a_again():
        calls.append(1)
        return {"id": "a"}

    again = cache.respond(
        generation=generation,
        key="a",
        request=_request(),
        cache_control="public, max-age=900",
        build_payload=build_payload_a_again,
    )

    assert calls == []
    assert again.body == first.body
    assert again.headers["etag"] == first.headers["etag"]
