"""Cache encoded response bodies by index generation and door-specific key.
A generation is immutable, so each body and strong ETag can be computed once. Generation
identity uses `is`, since `SkillsIndex` is unhashable. The platform helpers preserve
conditional-request handling and create a fresh `Response` per call.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

from fastapi import Request, Response

from dfo_platform.static_response import encode_json_body, serve_encoded_json

# Each door retains 8 MiB per generation. Over-budget bodies are encoded per request.
_MAX_CACHED_BYTES = 8 * 1024 * 1024


class GenerationBodyCache:
    """Cache one door's encoded bodies for one index generation.

    A changed generation clears every entry by identity. Keys distinguish payloads within
    one door, such as a skill id or `(skill_id, path)`. Each generation retains up to 8 MiB.
    Over-budget bodies are encoded per request. Admitted entries stay until the next generation.
    """

    def __init__(self) -> None:
        self._generation: object | None = None
        self._bodies: dict[Hashable, tuple[bytes, str]] = {}
        self._retained_bytes = 0

    def respond(
        self,
        *,
        generation: object,
        key: Hashable,
        request: Request,
        cache_control: str,
        build_payload: Callable[[], Any],
    ) -> Response:
        """Serve the cached `(generation, key)` body, or build it and cache it within budget."""
        if generation is not self._generation:
            self._generation = generation
            self._bodies = {}
            self._retained_bytes = 0

        cached = self._bodies.get(key)
        if cached is None:
            cached = encode_json_body(build_payload())
            body, _etag = cached
            if self._retained_bytes + len(body) <= _MAX_CACHED_BYTES:
                self._bodies[key] = cached
                self._retained_bytes += len(body)

        body, etag = cached
        return serve_encoded_json(
            request,
            body=body,
            etag=etag,
            cache_control=cache_control,
        )
