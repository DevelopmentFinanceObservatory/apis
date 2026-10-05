"""Tests trace-id format, context caching, and reset behaviour."""

from __future__ import annotations

import re

from dfo_platform.trace import get_trace_id, new_trace_id, reset_trace_id, set_trace_id

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def test_new_trace_id_is_32_lowercase_hex_chars():
    trace_id = new_trace_id()
    assert _HEX32.match(trace_id)


def test_new_trace_id_is_unique_per_call():
    assert new_trace_id() != new_trace_id()


def test_get_trace_id_mints_and_caches_within_a_context():
    # No trace_id bound yet in this test's context: get_trace_id mints one...
    first = get_trace_id()
    assert _HEX32.match(first)
    # ...and every subsequent call in the SAME context returns the SAME value.
    second = get_trace_id()
    assert second == first


def test_set_and_reset_round_trip():
    token = set_trace_id("a" * 32)
    try:
        assert get_trace_id() == "a" * 32
    finally:
        reset_trace_id(token)
    # After reset, the contextvar is back to unbound: get_trace_id mints a
    # fresh one rather than returning the just-reset value.
    assert get_trace_id() != "a" * 32
