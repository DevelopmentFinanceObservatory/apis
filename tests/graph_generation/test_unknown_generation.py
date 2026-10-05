"""`UNKNOWN_GENERATION`: the default every cache constructor falls back to,
always unknown, `register`/`refresh_if_stale` both no-ops.
"""

from __future__ import annotations

import asyncio

from graph_generation import UNKNOWN_GENERATION


def test_stamp_is_always_unknown() -> None:
    assert UNKNOWN_GENERATION.stamp().token is None


def test_register_is_a_noop_whose_handle_closes_cleanly() -> None:
    handle = UNKNOWN_GENERATION.register("cache", stats=lambda: {}, clear=lambda: None)
    handle.close()
    handle.close()  # idempotent


def test_refresh_if_stale_is_an_immediate_noop() -> None:
    asyncio.run(UNKNOWN_GENERATION.refresh_if_stale(timeout_s=1.0))
    assert UNKNOWN_GENERATION.stamp().token is None
