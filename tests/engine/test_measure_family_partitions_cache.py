"""Tests measure-family partition caching against the fixture CRS index."""

from __future__ import annotations

import dataclasses

from code_index.shared import get_index, get_measure_index
from search.engine.retrieval.bound_values import (
    measure_family_cache_key,
    measure_family_partitions,
    observed_measure_family,
)
from search.engine.shape import ShapeDraft
from tests.engine._index_slots import crs_flow_slot_keys, query_for
from tests.engine.test_measure_observed import _members_for


def _crs_shape() -> ShapeDraft:
    index = get_index()
    flow_row = next(row for row in index.statvars if row.population_type == "FinancialFlow")
    return ShapeDraft(
        shape_id="crs-amount",
        label="crs-amount",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid=flow_row.stat_type,
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=crs_flow_slot_keys(),
    )


def test_a_cached_call_returns_the_same_partitions_as_the_uncached_one():
    index = get_index()
    axes = get_measure_index().axes
    shape = _crs_shape()

    uncached = measure_family_partitions(shape, index, axes, query_for(shape, {}))
    cache: dict = {}
    cached_first = measure_family_partitions(shape, index, axes, query_for(shape, {}), cache=cache)
    cached_second = measure_family_partitions(shape, index, axes, query_for(shape, {}), cache=cache)

    assert cached_first == uncached
    assert cached_second == uncached


def test_a_second_call_for_the_same_key_is_served_from_the_cache_dict():
    index = get_index()
    axes = get_measure_index().axes
    shape = _crs_shape()
    cache: dict = {}

    result = measure_family_partitions(shape, index, axes, query_for(shape, {}), cache=cache)

    key = measure_family_cache_key(shape, axes, query_for(shape, {}))
    assert key in cache
    assert cache[key] == result
    # A distinct MeasurePartition tuple written into the cache under the
    # same key proves the second call reads it back rather than
    # recomputing -- the evidence this is a cache and not a no-op.
    sentinel = ()
    cache[key] = sentinel
    assert (
        measure_family_partitions(shape, index, axes, query_for(shape, {}), cache=cache) is sentinel
    )


def test_a_bound_values_entry_on_a_declared_axis_does_not_change_the_key():
    """A declared axis's bound value does not alter candidate rows or the cache key."""
    axes = get_measure_index().axes
    shape = _crs_shape()

    bare = measure_family_cache_key(shape, axes, query_for(shape, {}))
    with_measure_axis_bound = measure_family_cache_key(
        shape, axes, query_for(shape, {"nettingBasis": ["Net"]})
    )

    assert bare == with_measure_axis_bound


def test_a_bound_values_entry_on_a_non_measure_axis_changes_the_key():
    axes = get_measure_index().axes
    shape = _crs_shape()

    bare = measure_family_cache_key(shape, axes, query_for(shape, {}))
    with_other_bound = measure_family_cache_key(
        shape, axes, query_for(shape, {"crsPurpose": ["11110"]})
    )

    assert bare != with_other_bound


def test_the_key_follows_the_dataset_the_election_reads():
    """A query that names a dataset other than the five-tuple's filters the same rows as one
    that names none and reads no member, so the two keep separate cache entries."""
    index = get_index()
    axes = get_measure_index().axes
    shape = _crs_shape()
    bare = query_for(shape, {})
    (opt_in,) = (d for d in index.datasets.values() if d.opt_in)

    named = dataclasses.replace(bare, source=opt_in.provenance)
    keys = {
        measure_family_cache_key(shape, axes, dataclasses.replace(bare, source=source))
        for source in (None, *index.datasets)
    }
    assert len(keys) == 1 + len(index.datasets)

    cache: dict = {}
    default_members = {
        dcid
        for partition in measure_family_partitions(shape, index, axes, bare, cache=cache)
        for dcid in partition.members
    }
    named_members = {
        dcid
        for partition in measure_family_partitions(shape, index, axes, named, cache=cache)
        for dcid in partition.members
    }
    assert default_members
    assert not named_members


def test_observed_measure_family_forwards_its_cache_and_still_finds_siblings():
    # All partitions share one cache entry. Only `amount` partitions have siblings; the
    # `grantEquivalent` basis value has no match in the fixture index.
    index = get_index()
    axes = get_measure_index().axes
    shape = _crs_shape()
    partitions = measure_family_partitions(shape, index, axes, query_for(shape, {}))

    cache: dict = {}
    for partition in partitions:
        members = _members_for(partition.members)
        observed, siblings = observed_measure_family(
            shape, index, axes, query_for(shape, {}), members, cache=cache
        )
        assert observed is not None
        assert dict(observed.values) == dict(partition.values)
        if dict(partition.values)["measuredProperty"] == "amount":
            assert siblings
        else:
            assert siblings == ()

    # Every partition shares the same (five-tuple, declared_roles,
    # non-measure bound values) key, so the whole loop above filled exactly
    # one cache entry.
    assert len(cache) == 1


def test_disagreeing_members_publish_none_without_ever_populating_the_cache():
    index = get_index()
    axes = get_measure_index().axes
    shape = _crs_shape()
    partitions = measure_family_partitions(shape, index, axes, query_for(shape, {}))
    a, b = partitions[0], partitions[1]
    members = _members_for((a.members[0], b.members[0]))
    cache: dict = {}

    assert observed_measure_family(
        shape, index, axes, query_for(shape, {}), members, cache=cache
    ) == (None, ())
    # observed_measure_values already returns None for disagreeing members,
    # ahead of ever calling measure_family_partitions -- the cache stays
    # untouched.
    assert cache == {}
