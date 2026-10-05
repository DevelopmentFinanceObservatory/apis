"""`FakeGeneration`: directly constructible with no fixture, `advance()`'s
clear-on-change rule mirroring `GraphGeneration`'s own, and `set_unknown()`/
`refresh_if_stale()` as no-ops.
"""

from __future__ import annotations

import asyncio

from graph_generation import FakeGeneration


def _register_counting_cache(fake: FakeGeneration, *, name: str = "cache") -> list[int]:
    clear_calls = [0]

    def clear() -> None:
        clear_calls[0] += 1

    fake.register(name, stats=lambda: {"entries": 0}, clear=clear)
    return clear_calls


def test_starts_unknown_by_default() -> None:
    fake = FakeGeneration()
    assert fake.stamp().token is None


def test_constructor_token_is_served_immediately() -> None:
    fake = FakeGeneration(token="gen-a")
    assert fake.stamp().token == "gen-a"


def test_advance_to_a_new_token_clears_every_registered_cache() -> None:
    fake = FakeGeneration(token="gen-a")
    clear_calls = _register_counting_cache(fake)

    fake.advance("gen-b")
    assert fake.stamp().token == "gen-b"
    assert clear_calls[0] == 1


def test_advance_to_the_same_token_clears_nothing() -> None:
    fake = FakeGeneration(token="gen-a")
    clear_calls = _register_counting_cache(fake)

    fake.advance("gen-a")
    assert clear_calls[0] == 0


def test_advance_from_unknown_clears_nothing() -> None:
    fake = FakeGeneration()
    clear_calls = _register_counting_cache(fake)

    fake.advance("gen-a")
    assert fake.stamp().token == "gen-a"
    assert clear_calls[0] == 0


def test_set_unknown_then_advance_to_last_known_token_still_clears_nothing() -> None:
    fake = FakeGeneration(token="gen-a")
    clear_calls = _register_counting_cache(fake)

    fake.set_unknown()
    assert fake.stamp().token is None
    assert clear_calls[0] == 0

    fake.advance("gen-a")  # recovering the same token the fake last knew about
    assert clear_calls[0] == 0


def test_refresh_if_stale_is_an_immediate_no_op() -> None:
    fake = FakeGeneration(token="gen-a")
    asyncio.run(fake.refresh_if_stale(timeout_s=1.0))
    assert fake.stamp().token == "gen-a"


def test_registered_names_and_close_unregisters() -> None:
    fake = FakeGeneration()
    handle = fake.register("alpha", stats=lambda: {}, clear=lambda: None)
    assert fake.registered_names() == ("alpha",)
    handle.close()
    assert fake.registered_names() == ()
