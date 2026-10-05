"""Mint `cmp_` and `cps_` ULIDs without a dependency. Each ID encodes a 48-bit millisecond
timestamp and 80 random bits as 26 Crockford Base32 characters. The big-endian timestamp keeps
lexical order chronological. Randomness removes the need for monotonic increments. Primary keys
guard against collisions.
"""

from __future__ import annotations

import os
import time

__all__ = ["mint_component_id", "mint_composition_id", "new_ulid"]

_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TIMESTAMP_BYTES = 6
_RANDOMNESS_BYTES = 10
_ULID_CHARS = 26  # ceil(128 bits / 5 bits-per-char)


def _encode_crockford32(data: bytes) -> str:
    """Encode big-endian bytes as zero-padded Crockford Base32."""
    value = int.from_bytes(data, "big")
    bit_length = len(data) * 8
    char_count = -(-bit_length // 5)  # ceil division, no float rounding
    return "".join(
        _CROCKFORD_ALPHABET[(value >> shift) & 0x1F]
        for shift in range((char_count - 1) * 5, -1, -5)
    )


def new_ulid(*, now_ms: int | None = None) -> str:
    """Return a ULID, using `now_ms` for deterministic tests."""
    timestamp_ms = int(time.time() * 1000) if now_ms is None else now_ms
    timestamp_bytes = timestamp_ms.to_bytes(_TIMESTAMP_BYTES, "big")
    randomness = os.urandom(_RANDOMNESS_BYTES)
    return _encode_crockford32(timestamp_bytes + randomness)


def mint_component_id() -> str:
    return f"cmp_{new_ulid()}"


def mint_composition_id() -> str:
    return f"cps_{new_ulid()}"
