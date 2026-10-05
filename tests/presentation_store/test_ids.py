"""Checks ULID shape, ordering, uniqueness, and ID prefixes."""

from __future__ import annotations

from presentation_store.ids import mint_component_id, mint_composition_id, new_ulid

_CROCKFORD_CHARS = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def test_new_ulid_is_26_crockford_base32_characters():
    ulid = new_ulid()
    assert len(ulid) == 26
    assert set(ulid) <= _CROCKFORD_CHARS


def test_a_later_timestamp_sorts_after_an_earlier_one():
    earlier = new_ulid(now_ms=1_700_000_000_000)
    later = new_ulid(now_ms=1_700_000_000_001)
    assert earlier < later


def test_the_same_millisecond_still_mints_distinct_ids():
    first = new_ulid(now_ms=1_700_000_000_000)
    second = new_ulid(now_ms=1_700_000_000_000)
    assert first != second


def test_mint_component_id_carries_the_cmp_prefix():
    component_id = mint_component_id()
    assert component_id.startswith("cmp_")
    assert len(component_id) == len("cmp_") + 26


def test_mint_composition_id_carries_the_cps_prefix():
    composition_id = mint_composition_id()
    assert composition_id.startswith("cps_")
    assert len(composition_id) == len("cps_") + 26
