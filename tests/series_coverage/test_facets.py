"""Tests candidate ranking and metadata joins in `series_coverage.facets`."""

from __future__ import annotations

import itertools
import random
from dataclasses import replace

from series_coverage.facets import (
    FacetCandidate,
    FacetMetadata,
    facet_candidates,
    rank_candidates,
)
from series_coverage.index import FacetCoverage

_BASE = FacetCandidate(facet_id="fA", n_obs=1, first_year=2020, last_year=2020)


def test_rank_candidates_orders_by_obs_count_then_recency_then_lexicographic_then_facet_id() -> (
    None
):
    most_observed = FacetCandidate(
        facet_id="fZ", n_obs=100, first_year=2010, last_year=2010, provenance_id="Z"
    )
    recent_tiebreak_winner = FacetCandidate(
        facet_id="fB", n_obs=10, first_year=2023, last_year=2023, provenance_id="B"
    )
    recent_tiebreak_loser = FacetCandidate(
        facet_id="fA", n_obs=10, first_year=2020, last_year=2020, provenance_id="A"
    )
    facet_id_tiebreak_winner = FacetCandidate(
        facet_id="fA",
        n_obs=1,
        first_year=2020,
        last_year=2020,
        provenance_id="same",
        measurement_method="same",
        unit="same",
    )
    facet_id_tiebreak_loser = FacetCandidate(
        facet_id="fB",
        n_obs=1,
        first_year=2020,
        last_year=2020,
        provenance_id="same",
        measurement_method="same",
        unit="same",
    )

    ranked = rank_candidates(
        [
            recent_tiebreak_loser,
            most_observed,
            facet_id_tiebreak_loser,
            recent_tiebreak_winner,
            facet_id_tiebreak_winner,
        ]
    )

    assert ranked[0] is most_observed
    assert ranked[1] is recent_tiebreak_winner
    assert ranked[2] is recent_tiebreak_loser
    assert ranked[3] is facet_id_tiebreak_winner
    assert ranked[4] is facet_id_tiebreak_loser


def test_rank_candidates_sorts_a_missing_value_last_on_every_tiebreak_key() -> None:
    with_provenance = replace(_BASE, facet_id="fA", provenance_id="P")
    without_provenance = replace(_BASE, facet_id="fB", provenance_id=None)

    ranked = rank_candidates([without_provenance, with_provenance])

    assert ranked == (with_provenance, without_provenance)


def test_rank_candidates_sorts_a_missing_unit_and_method_last_too() -> None:
    with_unit = replace(_BASE, facet_id="fA", unit="USD")
    without_unit = replace(_BASE, facet_id="fB", unit=None)
    with_method = replace(_BASE, facet_id="fA", measurement_method="m")
    without_method = replace(_BASE, facet_id="fB", measurement_method=None)

    assert rank_candidates([without_unit, with_unit]) == (with_unit, without_unit)
    assert rank_candidates([without_method, with_method]) == (with_method, without_method)


def test_rank_candidates_prefers_higher_n_obs_over_lower_at_any_magnitude() -> None:
    # Coverage candidates always have an integer `n_obs`, so only magnitude matters.
    low = replace(_BASE, facet_id="fA", n_obs=1)
    high = replace(_BASE, facet_id="fB", n_obs=2)

    ranked = rank_candidates([low, high])

    assert ranked == (high, low)


def test_rank_candidates_is_a_total_order_independent_of_input_order() -> None:
    candidates = [
        FacetCandidate(facet_id="fA", n_obs=5, first_year=2020, last_year=2020, provenance_id="P1"),
        FacetCandidate(facet_id="fB", n_obs=5, first_year=2021, last_year=2021, provenance_id="P2"),
        FacetCandidate(facet_id="fC", n_obs=9, first_year=2019, last_year=2019, provenance_id="P3"),
        FacetCandidate(facet_id="fD", n_obs=1, first_year=2022, last_year=2022, provenance_id=None),
    ]
    expected = rank_candidates(candidates)

    rng = random.Random(0)
    for _ in range(20):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        assert rank_candidates(shuffled) == expected

    for permutation in itertools.permutations(candidates):
        assert rank_candidates(list(permutation)) == expected

    assert rank_candidates(list(expected)) == expected
    original_ids = [c.facet_id for c in candidates]
    rank_candidates(candidates)
    assert [c.facet_id for c in candidates] == original_ids


def test_facet_candidates_joins_coverage_with_caller_supplied_metadata() -> None:
    coverage = [
        FacetCoverage(facet_id="f1", n_obs=8, first_year=2018, last_year=2022),
        FacetCoverage(facet_id="f2", n_obs=1, first_year=2023, last_year=2023),
    ]
    metadata = {
        "f1": FacetMetadata(provenance_id="P1", measurement_method="m", unit="USD"),
    }

    joined = facet_candidates(coverage, metadata=metadata)

    assert joined == [
        FacetCandidate(
            facet_id="f1",
            n_obs=8,
            first_year=2018,
            last_year=2022,
            provenance_id="P1",
            measurement_method="m",
            unit="USD",
        ),
        FacetCandidate(facet_id="f2", n_obs=1, first_year=2023, last_year=2023),
    ]


def test_facet_candidates_leaves_metadata_all_none_for_an_unresolved_facet_id() -> None:
    coverage = [FacetCoverage(facet_id="unmapped", n_obs=3, first_year=2000, last_year=2005)]

    [joined] = facet_candidates(coverage, metadata={})

    assert joined.provenance_id is None
    assert joined.measurement_method is None
    assert joined.unit is None
    assert joined.scaling_factor is None


def test_facet_candidates_then_rank_candidates_ranks_a_coverage_join() -> None:
    coverage = [
        FacetCoverage(facet_id="low", n_obs=1, first_year=2020, last_year=2020),
        FacetCoverage(facet_id="high", n_obs=9, first_year=2015, last_year=2018),
    ]
    metadata = {
        "low": FacetMetadata(provenance_id="P-low"),
        "high": FacetMetadata(provenance_id="P-high"),
    }

    ranked = rank_candidates(facet_candidates(coverage, metadata=metadata))

    assert [c.facet_id for c in ranked] == ["high", "low"]
