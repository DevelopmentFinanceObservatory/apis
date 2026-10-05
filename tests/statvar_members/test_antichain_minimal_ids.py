"""Pins `antichain_minimal_ids` ordering, domination, and grouped reduction."""

from __future__ import annotations

import random
import time

from statvar_members import AbsentReading, antichain_minimal_ids, dominates_constraint_keys


def test_a_dominating_empty_key_set_drops_its_more_constrained_sibling():
    kept = antichain_minimal_ids(
        {"Total": frozenset(), "BySectorHealth": frozenset({"purpose"})},
        ["Total", "BySectorHealth"],
        readings={},
    )
    assert kept == ("Total",)


def test_disjoint_key_sets_have_no_domination_relation_and_both_survive():
    kept = antichain_minimal_ids(
        {"Male": frozenset({"gender"}), "Female": frozenset({"gender"})},
        ["Male", "Female"],
        readings={},
    )
    assert kept == ("Male", "Female")


def test_not_declared_reading_refuses_domination_across_the_differing_property():
    kept = antichain_minimal_ids(
        {"Purpose": frozenset({"purpose"}), "PurposeModality": frozenset({"purpose", "modality"})},
        ["Purpose", "PurposeModality"],
        readings={"modality": "not_declared"},
    )
    assert kept == ("Purpose", "PurposeModality")


def test_an_id_with_no_known_key_set_is_never_judged_and_is_always_kept():
    kept = antichain_minimal_ids({"Total": frozenset()}, ["Total", "Unknown"], readings={})
    assert kept == ("Total", "Unknown")


def test_order_is_preserved_and_a_dominated_id_is_dropped_not_reordered():
    kept = antichain_minimal_ids(
        {
            "Male": frozenset({"gender"}),
            "MaleAdult": frozenset({"gender", "age"}),
            "Female": frozenset({"maritalStatus"}),
        },
        ["Male", "MaleAdult", "Female"],
        readings={},
    )
    # "Male" dominates "MaleAdult" ({gender} ⊂ {gender, age}, "age" reading
    # aggregate by default); "Female" shares no subset relation with either
    # and is unaffected. The dropped id sat in the middle of `order`, and
    # the two survivors come back in their own original order, not sorted
    # or reversed.
    assert kept == ("Male", "Female")


def test_a_key_set_for_an_id_absent_from_order_is_never_consulted():
    # "Extra" carries a key set that would dominate "Total" if it were
    # allowed to participate, but it is not named in `order`.
    kept = antichain_minimal_ids(
        {"Total": frozenset({"purpose"}), "Extra": frozenset()},
        ["Total"],
        readings={},
    )
    assert kept == ("Total",)


# ---------------------------------------------------------------------------
# Grouping by key set must match a direct all-pairs reduction.
# ---------------------------------------------------------------------------


def _reference_antichain_minimal_ids(
    keys_by_id: dict[str, frozenset[str]],
    order: list[str],
    *,
    readings: dict[str, AbsentReading],
) -> tuple[str, ...]:
    """All-pairs reference that drops an id when another known key set dominates it."""
    ids_with_keys = frozenset(keys_by_id)
    return tuple(
        item_id
        for item_id in order
        if item_id not in ids_with_keys
        or not any(
            other != item_id
            and other in ids_with_keys
            and dominates_constraint_keys(keys_by_id[other], keys_by_id[item_id], readings=readings)
            for other in order
        )
    )


def _random_key_set_family(rng: random.Random) -> tuple[dict[str, frozenset[str]], list[str]]:
    properties = ["purpose", "modality", "gender", "age", "recipient"]
    distinct_key_sets = [
        frozenset(rng.sample(properties, k=rng.randint(0, 3))) for _ in range(rng.randint(1, 5))
    ]
    order = [f"id{i}" for i in range(rng.randint(1, 12))]
    keys_by_id = {}
    for item_id in order:
        # Some ids carry no key set at all, exercising the always-kept path.
        if rng.random() < 0.8:
            keys_by_id[item_id] = rng.choice(distinct_key_sets)
    return keys_by_id, order


def _random_readings(rng: random.Random) -> dict[str, AbsentReading]:
    properties = ["purpose", "modality", "gender", "age", "recipient"]
    readings: dict[str, AbsentReading] = {}
    for property in properties:
        if rng.random() < 0.5:
            readings[property] = "not_declared" if rng.random() < 0.5 else "aggregate"
    return readings


def test_grouped_reduction_matches_the_all_pairs_reference_over_random_families():
    rng = random.Random(20260922)
    for _ in range(500):
        keys_by_id, order = _random_key_set_family(rng)
        readings = _random_readings(rng)
        grouped = antichain_minimal_ids(keys_by_id, order, readings=readings)
        reference = _reference_antichain_minimal_ids(keys_by_id, order, readings=readings)
        assert grouped == reference


# ---------------------------------------------------------------------------
# Timing guard: a CRS-shaped family (thousands of ids, a few dozen distinct
# key sets) reduces in well under a second, since the grouped reduction is
# k^2 over distinct key sets rather than over ids.
# ---------------------------------------------------------------------------


def test_a_crs_shaped_family_reduces_in_well_under_a_second():
    distinct_key_sets = [
        frozenset({"crsPurpose"} | ({"crsModality"} if i % 2 else set())) for i in range(20)
    ]
    order = [f"CRSStatVar_{i}" for i in range(13_090)]
    keys_by_id = {
        item_id: distinct_key_sets[i % len(distinct_key_sets)] for i, item_id in enumerate(order)
    }
    readings: dict[str, AbsentReading] = {"crsModality": "not_declared"}

    started = time.monotonic()
    result = antichain_minimal_ids(keys_by_id, order, readings=readings)
    elapsed = time.monotonic() - started

    assert result == tuple(order)
    assert elapsed < 1.0
