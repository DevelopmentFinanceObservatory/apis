"""Series ids depend on coordinates; batch collision resolution is deterministic and order-independent."""

from __future__ import annotations

import hashlib
import itertools
import re

import assembler.series_id as series_id_module
from assembler.series_id import mint_series_id, mint_series_ids


def test_same_coordinates_mint_the_same_id():
    coordinates = (("recipient", "country/ETH"), ("stat_var", "dcid:X"))
    assert mint_series_id(coordinates=coordinates) == mint_series_id(coordinates=coordinates)


def test_coordinate_order_does_not_affect_the_id():
    a = (("recipient", "country/ETH"), ("stat_var", "dcid:X"))
    b = (("stat_var", "dcid:X"), ("recipient", "country/ETH"))
    assert mint_series_id(coordinates=a) == mint_series_id(coordinates=b)


def test_any_coordinate_change_mints_a_different_id():
    base = (("recipient", "country/ETH"), ("stat_var", "dcid:X"), ("unit", None))
    variants = [
        (("recipient", "country/GHA"), ("stat_var", "dcid:X"), ("unit", None)),
        (("recipient", "country/ETH"), ("stat_var", "dcid:Y"), ("unit", None)),
        (("recipient", "country/ETH"), ("stat_var", "dcid:X"), ("unit", "unit/USD")),
    ]
    base_id = mint_series_id(coordinates=base)
    for variant in variants:
        assert mint_series_id(coordinates=variant) != base_id


def test_null_value_does_not_collide_with_the_string_null():
    with_none = (("unit", None),)
    with_string = (("unit", "null"),)
    assert mint_series_id(coordinates=with_none) != mint_series_id(coordinates=with_string)


def test_mint_series_ids_is_order_independent_for_a_fixed_set():
    coordinates_list = [(("recipient", f"country/{i}"), ("stat_var", "dcid:X")) for i in range(5)]
    forward = mint_series_ids(coordinates_list=coordinates_list)
    backward = mint_series_ids(coordinates_list=list(reversed(coordinates_list)))
    assert set(forward) == set(backward)


def test_mint_series_ids_resolves_collisions_deterministically(monkeypatch):
    # A one-hex-char tail (16 possible values) makes a collision inside a
    # 30-item batch a near-certainty by pigeonhole; the batch mint must
    # still return 30 unique ids, and the same set of coordinate lists must
    # resolve to the same ids regardless of the order they arrive in.
    monkeypatch.setattr(series_id_module, "_BASE_TAIL_LENGTH", 1)
    coordinates_list = [
        (("recipient", f"country/{i:03d}"), ("stat_var", "dcid:X")) for i in range(30)
    ]
    forward = mint_series_ids(coordinates_list=coordinates_list)
    assert len(set(forward)) == len(coordinates_list)

    shuffled = list(reversed(coordinates_list))
    backward = mint_series_ids(coordinates_list=shuffled)
    assert set(forward) == set(backward)


SERIES_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


def test_long_coordinate_values_mint_an_opaque_id_within_the_schema_pattern():
    # Three values of 20+ characters would overrun the 64-character pattern if
    # the id spelled them out.
    coordinates = (
        ("observationAbout", "european_union_institutions_and_bodies"),
        ("provenance", "provenance_oecd_credit_reporting_system"),
        ("stat_var", "dcid_spec_1c81924611_official_development_assistance"),
    )
    series_id = mint_series_id(coordinates=coordinates)
    assert re.fullmatch(r"s_[0-9a-f]{16}", series_id)
    assert SERIES_ID_PATTERN.fullmatch(series_id)


def test_all_null_coordinates_mint_an_opaque_id():
    coordinates = (("unit", None), ("measurementMethod", None))
    assert re.fullmatch(r"s_[0-9a-f]{16}", mint_series_id(coordinates=coordinates))


def test_extended_collision_ids_stay_within_the_schema_pattern(monkeypatch):
    # Every digest shares its first 16 hex characters, so all 30 series collide
    # and the extension for rank 24 and above would pass 64 characters uncapped.
    def colliding_digest(coordinates):
        tail = hashlib.sha256(repr(coordinates).encode()).hexdigest()[:48]
        return "ab" * 8 + tail

    monkeypatch.setattr(series_id_module, "_digest", colliding_digest)
    coordinates_list = [(("recipient", f"country/{i:03d}"),) for i in range(30)]
    ids = mint_series_ids(coordinates_list=coordinates_list)
    assert len(set(ids)) == len(coordinates_list)
    assert max(len(i) for i in ids) == 64
    assert all(SERIES_ID_PATTERN.fullmatch(i) for i in ids)


def test_all_pairs_of_small_coordinate_space_mint_distinct_ids():
    # A denser property check than the two hand-picked variants above:
    # every pair drawn from a small combinatorial space of coordinate sets
    # mints a distinct id.
    recipients = ["country/A", "country/B", "country/C"]
    stat_vars = ["dcid:X", "dcid:Y"]
    coordinate_sets = [(("recipient", r), ("stat_var", s)) for r in recipients for s in stat_vars]
    ids = [mint_series_id(coordinates=c) for c in coordinate_sets]
    for a, b in itertools.combinations(ids, 2):
        assert a != b
