"""Pins `CoverageIndex` lookups against the shared test rows and the recorded slice."""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import series_coverage.index as coverage_index_module
from origin_catalogue import SeriesAdmission, bound_catalogue
from series_coverage.coverage_types import ServedExtent
from series_coverage.index import (
    BoundRoles,
    CoverageIndex,
    Estimate,
    KeptFacets,
    PairRows,
    ProvenanceShape,
    elected_year,
)
from series_coverage.schema import FacetMetadata, RoleNames, YearHorizon
from tests.origin_catalogue.fake_catalogue import FakeCatalogue
from tests.series_coverage.support import (
    _ROWS,
    BUILD_EPOCH_YEAR,
    COVERAGE_SLICE_BUILD,
    DEU,
    ETH,
    FACET_ACTUALS,
    FACET_PROJECTED,
    FRA,
    KEN,
    ROLE_NAMES,
    UNICEF,
    V1,
    V2,
    V4,
    V5,
    build_test_index,
    coverage_table,
    slice_index,
    with_repeated_dictionaries,
)

_CRS_PROVENANCE = "provenance/OECDCreditorReportingSystem"


@pytest.fixture
def index() -> CoverageIndex:
    return build_test_index()


# A minimal CoverageIndex built straight from rows, unchecked, for the
# windowing tests below, which need row shapes support.py's own shared
# fixture doesn't carry (a facet with data confined to an arbitrary narrow
# span; a window that reaches past the horizon's epoch or its last year).
_WINDOW_SV = "FixtureStatVar_WindowTestFacet"
_WINDOW_FACET = "window-test-facet"
_WINDOW_ROLE_NAMES = {
    _WINDOW_SV: RoleNames(donor="financingProvider", recipient="financingRecipient")
}


def _window_index(rows: list[tuple[str, str, str, str, list[int]]]) -> CoverageIndex:
    table, horizon = coverage_table(rows)
    return CoverageIndex(table, _WINDOW_ROLE_NAMES, horizon=horizon)


def test_filter_candidates_windowed_to_a_year_with_no_data_reports_nothing() -> None:
    # A facet with data only 2020-2024 (no row's year_mask carries a 2015 bit)
    # must not survive a 2015 window at all: no candidate with a zero count,
    # no candidate whose dates a caller could later mistake for confirmed data.
    idx = _window_index([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020, 2021, 2022, 2023, 2024])])
    result = idx.filter_candidates(
        variables=[_WINDOW_SV],
        bound={"recipient": frozenset({KEN})},
        first_year=2015,
        last_year=2015,
    )
    assert result == []


def test_filter_candidates_windowed_to_a_year_with_data_reports_the_exact_count() -> None:
    idx = _window_index([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020, 2021, 2022, 2023, 2024])])
    [coverage] = idx.filter_candidates(
        variables=[_WINDOW_SV],
        bound={"recipient": frozenset({KEN})},
        first_year=2022,
        last_year=2022,
    )
    [facet] = coverage.facets
    assert facet.n_obs == 1
    assert facet.year_counts == {2022: 1}


def test_filter_candidates_windowed_count_sums_across_rows_sharing_a_facet() -> None:
    # Two (donor, recipient) rows carry the same facet_id; the windowed count
    # is their summed per-year observations, not either row's own count alone.
    idx = _window_index(
        [
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020, 2021, 2022]),
            (_WINDOW_SV, _WINDOW_FACET, DEU, KEN, [2021, 2022, 2023]),
        ]
    )
    [coverage] = idx.filter_candidates(
        variables=[_WINDOW_SV],
        bound={"recipient": frozenset({KEN})},
        first_year=2021,
        last_year=2022,
    )
    [facet] = coverage.facets
    assert facet.year_counts == {2021: 2, 2022: 2}
    assert facet.n_obs == 4


def test_filter_candidates_clamps_a_window_starting_before_epoch_year() -> None:
    # The horizon's epoch is 1960; a caller's requested window (a query's own year
    # span) can start well before it. The row's real years (2020-2022) still
    # overlap the clamped window, so this must not raise and must not lose
    # the overlapping years.
    idx = _window_index([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020, 2021, 2022])])
    [coverage] = idx.filter_candidates(
        variables=[_WINDOW_SV],
        bound={"recipient": frozenset({KEN})},
        first_year=1900,
        last_year=2021,
    )
    [facet] = coverage.facets
    assert facet.year_counts == {2020: 1, 2021: 1}


def test_filter_candidates_clamps_a_window_ending_after_the_representable_range() -> None:
    # A one-word horizon ends at 2023; a requested window can end well past
    # it without raising, and the overlapping years still survive.
    idx = _window_index([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020, 2021, 2022])])
    [coverage] = idx.filter_candidates(
        variables=[_WINDOW_SV],
        bound={"recipient": frozenset({KEN})},
        first_year=2021,
        last_year=3000,
    )
    [facet] = coverage.facets
    assert facet.year_counts == {2021: 1, 2022: 1}


def test_filter_candidates_a_window_entirely_outside_the_representable_range_is_empty() -> None:
    # first_year/last_year both fall before the epoch: clamping leaves
    # first_year > last_year, which must report no candidates, never raise.
    idx = _window_index([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020, 2021, 2022])])
    result = idx.filter_candidates(
        variables=[_WINDOW_SV],
        bound={"recipient": frozenset({KEN})},
        first_year=1900,
        last_year=1950,
    )
    assert result == []


def test_filter_candidates_omits_a_variable_with_no_surviving_rows(index) -> None:
    result = index.filter_candidates(variables=[V1, V4], bound={"recipient": frozenset({KEN})})
    variables = {v.variable for v in result}
    assert variables == {V1}


def test_filter_candidates_rolls_up_facets_per_variable(index) -> None:
    # V1/KEN rows: (FRA,KEN,FACET_ACTUALS,2018-22 => 5 obs), (DEU,KEN,FACET_ACTUALS,2020-22 => 3
    # obs), (FRA,KEN,FACET_PROJECTED,2023 => 1 obs). facet_actuals rolls up to n_obs=8,
    # first=2018, last=2022; facet_projected is n_obs=1, first=last=2023.
    [coverage] = index.filter_candidates(variables=[V1], bound={"recipient": frozenset({KEN})})
    assert coverage.variable == V1
    by_facet = {f.facet_id: f for f in coverage.facets}
    assert by_facet[FACET_ACTUALS].n_obs == 8
    assert by_facet[FACET_ACTUALS].first_year == 2018
    assert by_facet[FACET_ACTUALS].last_year == 2022
    assert by_facet[FACET_PROJECTED].n_obs == 1
    assert by_facet[FACET_PROJECTED].first_year == 2023
    assert coverage.n_obs == 9
    assert coverage.first_year == 2018
    assert coverage.last_year == 2023


def test_filter_candidates_donor_binding_narrows_rows(index) -> None:
    result = index.filter_candidates(
        variables=[V1], bound={"donor": frozenset({DEU}), "recipient": frozenset({KEN})}
    )
    [coverage] = result
    assert coverage.n_obs == 3  # DEU/KEN/FACET_ACTUALS 2020-2022 only
    assert coverage.first_year == 2020
    assert coverage.last_year == 2022


def test_filter_candidates_empty_bound_set_matches_nothing(index) -> None:
    # A role bound to an explicit empty set is not the same as unbound: it matches no row.
    result = index.filter_candidates(variables=[V1], bound={"donor": frozenset()})
    assert result == []


def test_filter_candidates_windowed_excludes_variables_with_no_data_in_window(index) -> None:
    # V2's only KEN-adjacent row (well, V2/FRA/KEN) spans 2018-2022; UNICEF/ZWE spans 2005.
    result = index.filter_candidates(
        variables=[V1, V2], bound={"recipient": frozenset({KEN})}, first_year=2025, last_year=2026
    )
    assert result == []


def test_filter_candidates_windowed_n_obs_is_exact(index) -> None:
    # V1/FRA/KEN/FACET_ACTUALS covers 2018-2022; windowing to 2020-2021 leaves exactly those
    # two years for that row, plus V1/DEU/KEN/FACET_ACTUALS's 2020-2021 (of its 2020-2022 span).
    [coverage] = index.filter_candidates(
        variables=[V1], bound={"recipient": frozenset({KEN})}, first_year=2020, last_year=2021
    )
    by_facet = {f.facet_id: f for f in coverage.facets}
    assert by_facet[FACET_ACTUALS].n_obs == 4  # 2 years x 2 donor rows
    assert by_facet[FACET_ACTUALS].first_year == 2020
    assert by_facet[FACET_ACTUALS].last_year == 2021
    assert FACET_PROJECTED not in by_facet  # its only year, 2023, is outside the window
    # Both years are carried by both FRA/KEN and DEU/KEN rows: the sum across
    # this facet's own (donor, recipient) rows, not either row's count alone.
    assert by_facet[FACET_ACTUALS].year_counts == {2020: 2, 2021: 2}


def _pair_map(rows: PairRows, **window: int) -> dict[tuple[str, str, str, str], int]:
    """{(variable, facet_id, donor, recipient): n_obs} for every row, read
    back through `groups` and `pairs`."""
    return {
        (group.variable, group.facet_id, donor, recipient): n_obs
        for group in rows.groups()
        for donor, recipient, n_obs in rows.pairs(
            variable=group.variable, facet_id=group.facet_id, **window
        )
    }


def test_pair_rows_returns_real_donor_recipient_pairs(index) -> None:
    # Every V1 pair the coverage table actually carries a row for, no
    # fabricated cross product: (FRA,KEN) carries both facets, (DEU,KEN)
    # and (FRA,ETH) carry only FACET_ACTUALS. (DEU,ETH) and (FRA,ZWE) never
    # appear: no row exists for them.
    assert _pair_map(index.pair_rows(variables=[V1])) == {
        (V1, FACET_ACTUALS, FRA, KEN): 5,
        (V1, FACET_PROJECTED, FRA, KEN): 1,
        (V1, FACET_ACTUALS, DEU, KEN): 3,
        (V1, FACET_ACTUALS, FRA, ETH): 2,
    }


def test_pair_rows_groups_summarise_each_variable_and_facet(index) -> None:
    groups = {(g.variable, g.facet_id): g for g in index.pair_rows(variables=[V1]).groups()}
    assert groups.keys() == {(V1, FACET_ACTUALS), (V1, FACET_PROJECTED)}
    actuals = groups[(V1, FACET_ACTUALS)]
    assert actuals.pair_count == 3
    assert actuals.last_year == 2022
    assert actuals.donors == {FRA, DEU}
    assert actuals.recipients == {KEN, ETH}
    projected = groups[(V1, FACET_PROJECTED)]
    assert (projected.pair_count, projected.donors, projected.recipients) == (1, {FRA}, {KEN})


def test_pair_rows_windowed_n_obs_is_exact(index) -> None:
    # Same window as test_filter_candidates_windowed_n_obs_is_exact: (FRA,KEN)'s
    # FACET_ACTUALS narrows to 2020-2021 (2 of 5 years), its FACET_PROJECTED
    # (2023 only) drops out entirely; (DEU,KEN) keeps both its years; (FRA,ETH)'s
    # 2019-2020 span narrows to 2020 alone (1 of its 2 years).
    rows = index.pair_rows(variables=[V1], first_year=2020, last_year=2021)
    assert _pair_map(rows) == {
        (V1, FACET_ACTUALS, FRA, KEN): 2,
        (V1, FACET_ACTUALS, DEU, KEN): 2,
        (V1, FACET_ACTUALS, FRA, ETH): 1,
    }
    assert rows.n_obs(variable=V1, facet_id=FACET_ACTUALS) == 5


def test_pair_rows_counts_a_narrower_window_without_dropping_rows(index) -> None:
    # 2021 alone: (FRA,ETH) has no 2021 observation but is still listed.
    rows = index.pair_rows(variables=[V1])
    pairs = rows.pairs(variable=V1, facet_id=FACET_ACTUALS, first_year=2021, last_year=2021)
    assert sorted(pairs) == [(DEU, KEN, 1), (FRA, ETH, 0), (FRA, KEN, 1)]
    assert rows.n_obs(variable=V1, facet_id=FACET_ACTUALS, first_year=2021, last_year=2021) == 2


def test_pair_rows_empty_when_nothing_survives(index) -> None:
    rows = index.pair_rows(variables=[V4])
    assert rows.num_rows == 0
    assert rows.groups() == []
    assert rows.pairs(variable=V4, facet_id=FACET_ACTUALS) == []
    assert rows.n_obs(variable=V4, facet_id=FACET_ACTUALS) == 0


def test_pair_rows_omits_variables_with_no_rows(index) -> None:
    groups = index.pair_rows(variables=[V1, V4]).groups()
    assert {g.variable for g in groups} == {V1}


def test_pair_rows_bound_keeps_the_open_axis_real_dcids(index) -> None:
    # Bound to recipient KEN: the V1 rows naming KEN are (FRA,KEN) under both
    # facets and (DEU,KEN). The open donor axis keeps both real donors.
    # (FRA,ETH) is dropped by the bound.
    rows = index.pair_rows(variables=[V1], bound={"recipient": frozenset({KEN})})
    assert set(_pair_map(rows)) == {
        (V1, FACET_ACTUALS, FRA, KEN),
        (V1, FACET_PROJECTED, FRA, KEN),
        (V1, FACET_ACTUALS, DEU, KEN),
    }


def test_pair_rows_role_bound_to_empty_set_matches_nothing(index) -> None:
    assert index.pair_rows(variables=[V1], bound={"donor": frozenset()}).num_rows == 0


# -- bound pushdown parity over the recorded slice --------------------------

_GBR = "country/GBR"


def _grouped_then_narrowed(
    table: pa.Table, *, variables: Sequence[str], bound: BoundRoles
) -> dict[tuple[str, str, str, str], int]:
    """A reference read over the plain-string on-disk table: a four-key
    group_by over every row of `variables`, then narrowed in Python to
    `bound`'s dcids per axis."""
    filtered = table.filter(
        pc.is_in(table["variable"], value_set=pa.array(variables, type=pa.string()))
    )
    grouped = filtered.group_by(["donor", "recipient", "variable", "facet_id"]).aggregate(
        (("n_obs", "sum"),)
    )
    return {
        (row["variable"], row["facet_id"], row["donor"], row["recipient"]): row["n_obs_sum"]
        for row in grouped.to_pylist()
        if ("donor" not in bound or row["donor"] in bound["donor"])
        and ("recipient" not in bound or row["recipient"] in bound["recipient"])
    }


_PARITY_BOUNDS: list[BoundRoles] = [
    {},
    {"donor": frozenset({FRA})},
    {"recipient": frozenset({KEN})},
    {"donor": frozenset({FRA, _GBR}), "recipient": frozenset({KEN, ETH})},
]


@pytest.mark.parametrize("bound", _PARITY_BOUNDS, ids=["none", "donor", "recipient", "both"])
def test_pair_rows_bound_matches_grouped_then_narrowed_read(bound: BoundRoles) -> None:
    index = slice_index()
    on_disk = pq.read_table(COVERAGE_SLICE_BUILD / "coverage.parquet").cast(index.horizon.schema)
    variables = sorted(index.role_names)

    expected = _grouped_then_narrowed(on_disk, variables=variables, bound=bound)
    actual = _pair_map(index.pair_rows(variables=variables, bound=bound))

    assert expected, "the slice carries no row under this bound; the parity check is vacuous"
    assert actual == expected


@pytest.mark.parametrize(
    "window", [(None, None), (2015, 2020), (2022, None), (None, 1990)], ids=str
)
@pytest.mark.parametrize("bound", _PARITY_BOUNDS, ids=["none", "donor", "recipient", "both"])
def test_series_counts_count_each_facets_pairs_with_data_in_the_window(
    bound: BoundRoles, window: tuple[int | None, int | None]
) -> None:
    """For every variable of the slice, under every bound and window, a facet's series count
    is the number of its donor and recipient pairs holding an observation in the window, and
    a facet with none is absent."""
    index = slice_index()
    first, last = window
    counted = 0
    for variable in sorted(index.role_names):
        pairs: dict[str, int] = {}
        rows = index.pair_rows(variables=[variable], bound=bound, first_year=first, last_year=last)
        for (_variable, facet_id, _donor, _recipient), n_obs in _pair_map(rows).items():
            if n_obs:
                pairs[facet_id] = pairs.get(facet_id, 0) + 1
        counts = index.series_counts(
            variable=variable, bound=bound, first_year=first, last_year=last
        )
        assert counts == pairs
        counted += len(counts)
    if window == (None, None):
        assert counted, "the slice carries no row under this bound; the check is vacuous"


@pytest.mark.parametrize("bound", _PARITY_BOUNDS, ids=["none", "donor", "recipient", "both"])
def test_facet_spans_state_what_filter_candidates_states_over_every_year(
    bound: BoundRoles,
) -> None:
    """For every variable of the slice under every bound, a facet's span is the count and
    the first and last years `filter_candidates` states for it with no window."""
    index = slice_index()
    variables = sorted(index.role_names)
    expected = {
        (coverage.variable, facet.facet_id): (facet.n_obs, facet.first_year, facet.last_year)
        for coverage in index.filter_candidates(variables=variables, bound=bound)
        for facet in coverage.facets
    }

    spans = index.facet_spans(variables=variables, bound=bound)

    assert expected, "the slice carries no row under this bound; the parity check is vacuous"
    assert {
        key: (span.n_obs, span.first_year, span.last_year) for key, span in spans.items()
    } == expected
    assert index.facet_spans(variables=["TEST/NoSuchVariable"], bound=bound) == {}


@pytest.mark.parametrize("bound", _PARITY_BOUNDS[1:], ids=["donor", "recipient", "both"])
def test_pair_rows_windowed_bound_matches_the_narrowed_unbound_read(bound: BoundRoles) -> None:
    index = slice_index()
    variables = sorted(index.role_names)
    unbound = _pair_map(index.pair_rows(variables=variables, first_year=2015, last_year=2020))
    expected = {
        key: n_obs
        for key, n_obs in unbound.items()
        if all(key[2 if role == "donor" else 3] in dcids for role, dcids in bound.items())
    }
    actual = _pair_map(
        index.pair_rows(variables=variables, bound=bound, first_year=2015, last_year=2020)
    )

    assert expected
    assert actual == expected


# -- a horizon past 2023: two mask words ------------------------------------

# FRA->KEN spans the word boundary (2022-2025 plus 2027); DEU->KEN has data
# only in word 1 (2024); FRA->ETH only in word 0 (2017).
_BOUNDARY_ROWS = [
    (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2022, 2023, 2024, 2025, 2027]),
    (_WINDOW_SV, _WINDOW_FACET, DEU, KEN, [2024]),
    (_WINDOW_SV, _WINDOW_FACET, FRA, ETH, [2017]),
]


@pytest.fixture
def boundary_index() -> CoverageIndex:
    idx = _window_index(_BOUNDARY_ROWS)
    assert idx.horizon == YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=2)
    return idx


def test_filter_candidates_windowed_across_the_word_boundary(boundary_index) -> None:
    [coverage] = boundary_index.filter_candidates(
        variables=[_WINDOW_SV], first_year=2023, last_year=2024
    )
    [facet] = coverage.facets
    assert facet.year_counts == {2023: 1, 2024: 2}
    assert (facet.n_obs, facet.first_year, facet.last_year) == (3, 2023, 2024)


def test_filter_candidates_unwindowed_reads_both_words(boundary_index) -> None:
    [coverage] = boundary_index.filter_candidates(variables=[_WINDOW_SV])
    assert coverage.n_obs == 7
    assert (coverage.first_year, coverage.last_year) == (2017, 2027)


def test_pair_rows_windowed_across_the_word_boundary(boundary_index) -> None:
    rows = boundary_index.pair_rows(variables=[_WINDOW_SV], first_year=2023, last_year=2024)
    assert _pair_map(rows) == {
        (_WINDOW_SV, _WINDOW_FACET, FRA, KEN): 2,
        (_WINDOW_SV, _WINDOW_FACET, DEU, KEN): 1,
    }
    [group] = rows.groups()
    assert (group.pair_count, group.last_year) == (2, 2024)
    # A narrower read over the same rows, entirely inside word 1.
    assert rows.n_obs(variable=_WINDOW_SV, facet_id=_WINDOW_FACET, first_year=2024) == 2


def test_pair_rows_first_and_last_year_cross_words(boundary_index) -> None:
    rows = boundary_index.pair_rows(variables=[_WINDOW_SV], first_year=2023, last_year=2027)
    table = rows._table.to_pylist()
    fra_ken = next(r for r in table if r["donor"] == FRA and r["recipient"] == KEN)
    assert (fra_ken["first_year"], fra_ken["last_year"], fra_ken["n_obs"]) == (2023, 2027, 4)


def test_open_axis_windowed_to_word_one_only(boundary_index) -> None:
    assert boundary_index.open_axis(
        variables=[_WINDOW_SV], axis="donor", first_year=2024, last_year=2087
    ) == {FRA, DEU}
    assert boundary_index.open_axis(
        variables=[_WINDOW_SV], axis="recipient", first_year=1960, last_year=2023
    ) == {KEN, ETH}


def _grouped(table: pa.Table) -> dict[tuple[str, ...], tuple[int, int]]:
    """A `grouped` table as key tuple -> (first_year, last_year)."""
    *keys, first, last = (table[name].cast(pa.string()).to_pylist() for name in table.column_names)
    return {
        tuple(key): (int(low), int(high))
        for *key, low, high in zip(*keys, first, last, strict=True)
    }


def test_grouped_spans_each_combination_across_both_mask_words(boundary_index) -> None:
    # KEN's first year sits in word 0 (FRA, 2022) and its last in word 1 (FRA, 2027).
    assert _grouped(boundary_index.grouped(by=("recipient",))) == {
        (KEN,): (2022, 2027),
        (ETH,): (2017, 2017),
    }
    assert _grouped(boundary_index.grouped(by=("donor", "recipient"))) == {
        (FRA, KEN): (2022, 2027),
        (DEU, KEN): (2024, 2024),
        (FRA, ETH): (2017, 2017),
    }


def test_grouped_agrees_with_filter_candidates_per_recipient(index) -> None:
    variables = list(index.role_names)
    spans = _grouped(index.grouped(by=("recipient", "variable", "facet_id")))
    for recipient in index.open_axis(variables=variables, axis="recipient"):
        held = index.filter_candidates(
            variables=variables, bound={"recipient": frozenset({recipient})}
        )
        assert {
            (recipient, entry.variable, facet.facet_id): (facet.first_year, facet.last_year)
            for entry in held
            for facet in entry.facets
        } == {key: span for key, span in spans.items() if key[0] == recipient}
    assert {key[0] for key in spans} == index.open_axis(variables=variables, axis="recipient")


def test_grouped_restricts_to_the_named_variables(index) -> None:
    assert set(_grouped(index.grouped(by=("donor",), variables=[V2]))) == {(FRA,), (UNICEF,)}
    assert index.grouped(by=("donor",), variables=["Unknown"]).num_rows == 0


def test_headline_finds_the_latest_year_in_the_second_word(boundary_index) -> None:
    headline = boundary_index.headline(variables=[_WINDOW_SV])
    assert headline is not None
    assert (headline.year, headline.observations, headline.series) == (2027, 1, 1)


def _recent(headline) -> list[tuple[int, int, dict[str, int]]]:
    return [(c.year, c.series, dict(c.role_counts)) for c in headline.recent_years]


def test_headline_counts_the_two_years_before_the_latest_and_keeps_an_empty_one(
    boundary_index,
) -> None:
    headline = boundary_index.headline(variables=[_WINDOW_SV])
    assert headline is not None
    assert _recent(headline) == [
        (2027, 1, {"donor": 1, "recipient": 1}),
        (2026, 0, {}),
        (2025, 1, {"donor": 1, "recipient": 1}),
    ]
    assert all(c.observations == c.series for c in headline.recent_years)


def test_headline_window_picks_the_latest_year_but_not_the_years_counted(boundary_index) -> None:
    """2024 holds two series (France and Germany into Kenya). 2023 and 2022 lie outside the
    window and still count what the rows hold."""
    headline = boundary_index.headline(variables=[_WINDOW_SV], first_year=2024, last_year=2024)
    assert headline is not None
    assert headline.year == 2024
    assert _recent(headline) == [
        (2024, 2, {"donor": 2, "recipient": 1}),
        (2023, 1, {"donor": 1, "recipient": 1}),
        (2022, 1, {"donor": 1, "recipient": 1}),
    ]
    latest = headline.recent_years[0]
    assert (headline.observations, headline.series, headline.role_counts) == (
        latest.observations,
        latest.series,
        latest.role_counts,
    )


@pytest.mark.parametrize(
    ("series_by_year", "expected"),
    [
        pytest.param({2025: 9, 2024: 85, 2023: 91}, 2024, id="thin-newest-year"),
        pytest.param({2025: 89, 2024: 94, 2023: 120}, 2025, id="newest-over-half"),
        pytest.param({2025: 45, 2024: 90, 2023: 60}, 2025, id="exactly-half"),
        pytest.param({2025: 44, 2024: 85, 2023: 91}, 2024, id="just-under-half"),
        pytest.param({2021: 1, 2019: 1, 2006: 5, 2003: 10}, 2021, id="sparse"),
        pytest.param({2026: 1, 2025: 1, 2024: 100}, 2024, id="full-year-in-reach"),
        pytest.param({2027: 1, 2026: 1, 2025: 1, 2024: 100}, 2027, id="full-year-out-of-reach"),
        pytest.param({2025: 0, 2024: 3}, 2024, id="empty-year-holds-nothing"),
        pytest.param({}, None, id="no-series"),
    ],
)
def test_elected_year_is_the_newest_recent_year_half_as_full_as_the_fullest(
    series_by_year: dict[int, int], expected: int | None
) -> None:
    assert elected_year(series_by_year) == expected


_SECOND_SV = "FixtureStatVar_SecondElection"
_ELECTION_ROLE_NAMES = {
    sv: RoleNames(donor="financingProvider", recipient="financingRecipient")
    for sv in (_WINDOW_SV, _SECOND_SV)
}


def _donor(i: int) -> str:
    return f"country/D{i:03d}"


def _donor_rows(
    variable: str, series_by_year: Mapping[int, int], *, facet_id: str = _WINDOW_FACET
) -> list[tuple[str, str, str, str, list[int]]]:
    """One series per donor into Kenya, the first `series_by_year[year]` donors holding
    `year`."""
    return [
        (variable, facet_id, _donor(i), KEN, [y for y, held in series_by_year.items() if i < held])
        for i in range(max(series_by_year.values()))
    ]


def _election_index(
    rows: list[tuple[str, str, str, str, list[int]]],
    facet_metadata: Mapping[str, FacetMetadata] | None = None,
    *,
    build_year: int | None = None,
) -> CoverageIndex:
    table, horizon = coverage_table(rows)
    return CoverageIndex(
        table,
        _ELECTION_ROLE_NAMES,
        facet_metadata or {},
        horizon=horizon,
        build_year=build_year,
    )


def test_a_latest_read_serves_and_heads_the_year_elected_over_the_series_it_holds() -> None:
    """Five donors hold 2025, nine 2024 and eleven 2023, so the variable elects 2024. The
    read serves the nine holding it and the headline counts them, beside the raw counts of
    all three years. The nine alone would elect 2025, which five of them hold, so every
    count takes the year elected over all eleven."""
    idx = _election_index(_donor_rows(_WINDOW_SV, {2025: 5, 2024: 9, 2023: 11}))

    elected = idx.elected_years(variables=[_WINDOW_SV])
    served = idx.series_years(variables=[_WINDOW_SV], latest=elected)
    headline = idx.headline(variables=[_WINDOW_SV], elected=elected)

    assert elected == {_WINDOW_SV: 2024}
    assert served is not None
    epoch = idx.horizon.epoch_year
    assert dict(zip(served.keys(), served.year_bits(), strict=True)) == {
        (_WINDOW_SV, KEN, _donor(i)): 1 << (2024 - epoch) for i in range(9)
    }
    assert headline is not None
    assert (headline.year, headline.series, dict(headline.role_counts)) == (
        2024,
        9,
        {"donor": 9, "recipient": 1},
    )
    assert _recent(headline) == [
        (2025, 5, {"donor": 5, "recipient": 1}),
        (2024, 9, {"donor": 9, "recipient": 1}),
        (2023, 11, {"donor": 11, "recipient": 1}),
    ]
    survivors = [(_WINDOW_SV, _WINDOW_FACET, KEN, _donor(i)) for i in range(9)]
    assert idx.elected_years(variables=[_WINDOW_SV], served=survivors) == {_WINDOW_SV: 2025}
    by_year = idx.series_by_year(variables=[_WINDOW_SV], first_year=2022, last_year=2025)
    assert by_year == {2022: 0, **{c.year: c.series for c in headline.recent_years}}
    assert idx.series_by_year(
        variables=[_WINDOW_SV], served=survivors, first_year=2025, last_year=2025
    ) == {2025: 5}
    assert idx.series_by_year(variables=["Unknown"], first_year=2024, last_year=2025) == {
        2024: 0,
        2025: 0,
    }


def test_a_latest_read_elects_no_year_after_the_one_the_index_was_built_in() -> None:
    """An index built in 2026 holds one variable at three donors in 2024, one in 2025 and
    2026, and all six in projections for 2027 and 2030, and another in projections alone.
    The first elects 2024 and the headline's recent years end at 2026. The second elects
    nothing and serves its own latest year, which heads it alone and sets no year beside
    the first. A window from 2025 to 2030 heads 2030 with six series. An index
    stating no build year elects the projected 2030."""
    rows = [
        *_donor_rows(_WINDOW_SV, {2030: 6, 2027: 6, 2026: 1, 2025: 1, 2024: 3}),
        *_donor_rows(_SECOND_SV, {2030: 3, 2029: 3}),
    ]
    idx = _election_index(rows, build_year=2026)
    variables = [_WINDOW_SV, _SECOND_SV]

    elected = idx.elected_years(variables=variables)
    headline = idx.headline(variables=[_WINDOW_SV], elected=elected)
    both = idx.headline(variables=variables, elected=elected)
    alone = idx.headline(variables=[_SECOND_SV], elected=elected)
    projected = idx.series_years(variables=[_SECOND_SV], latest=elected)
    named = idx.headline(variables=[_WINDOW_SV], first_year=2025, last_year=2030)

    assert elected == {_WINDOW_SV: 2024}
    assert headline is not None
    assert (headline.year, headline.series) == (2024, 3)
    assert [(c.year, c.series) for c in headline.recent_years] == [(2026, 1), (2025, 1), (2024, 3)]
    assert both is not None and alone is not None
    assert (both.year, both.series, both.recent_years) == (2024, 3, headline.recent_years)
    assert (alone.year, alone.series) == (2030, 3)
    assert projected is not None
    epoch = idx.horizon.epoch_year
    assert set(projected.year_bits()) == {1 << (2030 - epoch)}
    assert named is not None
    assert (named.year, named.series, [c.year for c in named.recent_years]) == (
        2030,
        6,
        [2030, 2029, 2028],
    )
    assert _election_index(rows).elected_years(variables=variables) == {
        _WINDOW_SV: 2030,
        _SECOND_SV: 2030,
    }


def test_elected_years_count_the_series_the_read_selects() -> None:
    """Two donors hold 2025 and eight 2024 in one facet, and two more hold 2025 in
    another. Both facets elect 2025, four of eight, and the first alone 2024, as do both
    without one of the 2025 donors and both at four donors of whom one holds 2025."""
    idx = _election_index(
        [
            *_donor_rows(_WINDOW_SV, {2025: 2, 2024: 8}),
            (_WINDOW_SV, _OTHER_FACET, _donor(2), KEN, [2025]),
            (_WINDOW_SV, _OTHER_FACET, _donor(3), KEN, [2025]),
        ]
    )

    assert idx.elected_years(variables=[_WINDOW_SV]) == {_WINDOW_SV: 2025}
    assert idx.elected_years(
        variables=[_WINDOW_SV], admits=lambda _variable, facet: facet == _WINDOW_FACET
    ) == {_WINDOW_SV: 2024}
    assert idx.elected_years(
        variables=[_WINDOW_SV], served=[(_WINDOW_SV, _WINDOW_FACET, None, None)]
    ) == {_WINDOW_SV: 2024}
    assert idx.elected_years(
        variables=[_WINDOW_SV], excluded=[(_WINDOW_SV, "donor", _donor(0))]
    ) == {_WINDOW_SV: 2024}
    assert idx.series_by_year(
        variables=[_WINDOW_SV],
        excluded=[(_WINDOW_SV, "donor", _donor(0))],
        first_year=2024,
        last_year=2025,
    ) == {2024: 7, 2025: 3}
    assert idx.elected_years(
        variables=[_WINDOW_SV], bound={"donor": frozenset(map(_donor, (0, 5, 6, 7)))}
    ) == {_WINDOW_SV: 2024}


def test_a_variable_the_index_cannot_elect_for_is_absent_and_serves_its_own_latest() -> None:
    """A variable with a monthly facet elects no year, as the index cannot find a latest
    month, and neither does a variable the index holds no row of. The monthly variable
    serves and heads its own latest year, 2025, where two of its six donors hold data."""
    idx = _election_index(
        [
            *_donor_rows(_WINDOW_SV, {2025: 2, 2024: 6}, facet_id=_MONTHLY_FACET),
            *_donor_rows(_SECOND_SV, {2025: 2, 2024: 6}),
        ],
        {
            _MONTHLY_FACET: FacetMetadata(provenance_id="p", observation_period="P1M"),
            _WINDOW_FACET: FacetMetadata(provenance_id="p", observation_period="P1Y"),
        },
    )

    elected = idx.elected_years(variables=[_WINDOW_SV, _SECOND_SV, "Unknown"])
    served = idx.series_years(variables=[_WINDOW_SV], latest=elected)
    headline = idx.headline(variables=[_WINDOW_SV], elected=elected)

    assert elected == {_SECOND_SV: 2024}
    assert served is not None and served.series == 2
    assert headline is not None
    assert (headline.year, headline.series) == (2025, 2)


@pytest.mark.parametrize("build_year", [2026, None])
def test_a_monthly_variable_heads_its_recent_year_beside_older_elected_ones(
    build_year: int | None,
) -> None:
    """A monthly variable holds 2025 at two donors and elects nothing. An annual one ends
    in 2021 and elects it. The headline names 2025, one of its `recent_years`, with the
    monthly variable's two series, whether or not the index states a build year."""
    idx = _election_index(
        [
            *_donor_rows(_WINDOW_SV, {2025: 2, 2024: 6}, facet_id=_MONTHLY_FACET),
            *_donor_rows(_SECOND_SV, {2021: 4, 2020: 4}),
        ],
        {
            _MONTHLY_FACET: FacetMetadata(provenance_id="p", observation_period="P1M"),
            _WINDOW_FACET: FacetMetadata(provenance_id="p", observation_period="P1Y"),
        },
        build_year=build_year,
    )
    variables = [_WINDOW_SV, _SECOND_SV]

    elected = idx.elected_years(variables=variables)
    headline = idx.headline(variables=variables, elected=elected)

    assert elected == {_SECOND_SV: 2021}
    assert headline is not None
    assert (headline.year, headline.series) == (2025, 2)
    assert [c.year for c in headline.recent_years] == [2025, 2024, 2023]


def test_a_two_variable_headline_names_the_newest_elected_year_and_counts_its_variable() -> None:
    """One variable holds 2025 at four of five donors and elects it. The other holds 2025
    at one of six and elects 2024. The headline names 2025 with the four series serving it,
    where `recent_years` counts all five that hold it."""
    idx = _election_index(
        [
            *_donor_rows(_WINDOW_SV, {2025: 4, 2024: 5}),
            *_donor_rows(_SECOND_SV, {2025: 1, 2024: 6}),
        ]
    )
    variables = [_WINDOW_SV, _SECOND_SV]

    elected = idx.elected_years(variables=variables)
    headline = idx.headline(variables=variables, elected=elected)
    extent = idx.served_extent(variables=variables, latest=elected)

    assert elected == {_WINDOW_SV: 2025, _SECOND_SV: 2024}
    assert headline is not None
    assert (headline.year, headline.series, dict(headline.role_counts)) == (
        2025,
        4,
        {"donor": 4, "recipient": 1},
    )
    assert _recent(headline)[:2] == [
        (2025, 5, {"donor": 4, "recipient": 1}),
        (2024, 11, {"donor": 6, "recipient": 1}),
    ]
    assert extent == ServedExtent(rows=10, series=10, first_year=2024, last_year=2025)


@pytest.mark.parametrize("offset", [0, 1])
def test_headline_recent_years_stop_at_the_horizon(offset) -> None:
    """The artifact holds no year before its epoch, so `recent_years` lists none."""
    latest = BUILD_EPOCH_YEAR + offset
    idx = _window_index([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [latest])])
    headline = idx.headline(variables=[_WINDOW_SV])
    assert headline is not None
    assert [c.year for c in headline.recent_years] == list(range(latest, BUILD_EPOCH_YEAR - 1, -1))


def test_estimate_counts_rows_across_the_word_boundary(boundary_index) -> None:
    estimate = boundary_index.estimate(
        variables=[_WINDOW_SV],
        bound={"recipient": frozenset({KEN})},
        first_year=2023,
        last_year=2024,
    )
    assert estimate.rows == 3


def test_a_table_whose_mask_columns_disagree_with_the_horizon_is_refused() -> None:
    table, _horizon = coverage_table(_BOUNDARY_ROWS)
    with pytest.raises(ValueError, match="year-mask columns"):
        CoverageIndex(
            table,
            _WINDOW_ROLE_NAMES,
            horizon=YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=1),
        )


# -- dictionary-encoded key columns -----------------------------------------

_KEY_COLUMNS = ("variable", "facet_id", "donor", "recipient")


def test_loaded_index_holds_key_columns_dictionary_encoded(index) -> None:
    for name in _KEY_COLUMNS:
        assert pa.types.is_dictionary(index._table.schema.field(name).type), name
        assert index._table[name].num_chunks == 1, name


def test_plain_string_table_is_dictionary_encoded_and_answers_in_str() -> None:
    idx = _window_index(
        [
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020, 2021]),
            (_WINDOW_SV, _WINDOW_FACET, DEU, KEN, [2021]),
        ]
    )
    for name in _KEY_COLUMNS:
        assert pa.types.is_dictionary(idx._table.schema.field(name).type), name
    donors = idx.open_axis(variables=[_WINDOW_SV], axis="donor")
    assert donors == {FRA, DEU}
    assert all(type(donor) is str for donor in donors)
    pairs = idx.pair_rows(variables=[_WINDOW_SV]).pairs(variable=_WINDOW_SV, facet_id=_WINDOW_FACET)
    assert all(type(d) is str and type(r) is str for d, r, _n_obs in pairs)
    [coverage] = idx.filter_candidates(variables=[_WINDOW_SV])
    assert type(coverage.variable) is str
    assert type(coverage.facets[0].facet_id) is str


def test_multi_chunk_table_is_unified_into_one_chunk() -> None:
    first, horizon = coverage_table([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020])])
    second, _horizon = coverage_table([(_WINDOW_SV, _WINDOW_FACET, DEU, ETH, [2021])])
    chunked = pa.concat_tables([first, second])
    assert chunked["donor"].num_chunks == 2
    idx = CoverageIndex(chunked, _WINDOW_ROLE_NAMES, horizon=horizon)
    assert idx._table["donor"].num_chunks == 1
    pairs = idx.pair_rows(variables=[_WINDOW_SV]).pairs(variable=_WINDOW_SV, facet_id=_WINDOW_FACET)
    assert {(d, r) for d, r, _n_obs in pairs} == {(FRA, KEN), (DEU, ETH)}


def test_empty_table_builds_an_index_that_answers_nothing() -> None:
    horizon = YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=1)
    idx = CoverageIndex(horizon.schema.empty_table(), {}, horizon=horizon)
    assert idx.num_rows == 0
    assert idx.pair_rows(variables=[_WINDOW_SV]).groups() == []
    assert idx.filter_candidates(variables=[_WINDOW_SV]) == []
    assert idx.facet_summary.variable(_WINDOW_SV) is None


_FLOW_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")


def test_provenance_currencies_give_each_provenance_its_iso_codes() -> None:
    table, horizon = coverage_table(
        [
            ("V-a", "f-usd", FRA, KEN, [2020]),
            ("V-a", "f-eur", FRA, KEN, [2020]),
            ("V-b", "f-usd-again", FRA, KEN, [2020]),
            ("V-b", "f-b", FRA, KEN, [2020]),
            ("V-a", "f-people", FRA, KEN, [2020]),
            ("V-a", "f-lower", FRA, KEN, [2020]),
            ("V-a", "f-long", FRA, KEN, [2020]),
            ("V-a", "f-no-unit", FRA, KEN, [2020]),
            ("V-a", "f-no-provenance", FRA, KEN, [2020]),
            ("V-unnamed", "f-unnamed", FRA, KEN, [2020]),
        ]
    )
    idx = CoverageIndex(
        table,
        {"V-a": _FLOW_ROLES, "V-b": _FLOW_ROLES},
        {
            "f-usd": FacetMetadata(provenance_id="provenance/A", unit="CurrencyUSD"),
            "f-eur": FacetMetadata(provenance_id="provenance/A", unit="CurrencyEUR"),
            "f-usd-again": FacetMetadata(provenance_id="provenance/A", unit="CurrencyUSD"),
            "f-b": FacetMetadata(provenance_id="provenance/B", unit="CurrencyGBP"),
            # A non-currency unit, a lower-case or over-long code, a missing
            # unit and a missing provenance each contribute nothing.
            "f-people": FacetMetadata(provenance_id="provenance/C", unit="Person"),
            "f-lower": FacetMetadata(provenance_id="provenance/C", unit="Currencyusd"),
            "f-long": FacetMetadata(provenance_id="provenance/C", unit="CurrencyUSDX"),
            "f-no-unit": FacetMetadata(provenance_id="provenance/C"),
            "f-no-provenance": FacetMetadata(unit="CurrencyJPY"),
            # A variable without role names is outside the population
            # `provenance_shapes` reads, so its facet contributes nothing.
            "f-unnamed": FacetMetadata(provenance_id="provenance/E", unit="CurrencyCHF"),
        },
        horizon=horizon,
    )

    currencies = idx.provenance_currencies

    assert dict(currencies) == {
        "provenance/A": frozenset({"USD", "EUR"}),
        "provenance/B": frozenset({"GBP"}),
    }
    # Built once when the index loads, so the second read returns the same mapping.
    assert idx.provenance_currencies is currencies


def test_provenance_currencies_ignore_facets_without_coverage_rows() -> None:
    """A currency facet that no coverage row uses adds no code to its provenance."""
    table, horizon = coverage_table([("V-a", "f-people", FRA, KEN, [2020])])
    idx = CoverageIndex(
        table,
        {"V-a": _FLOW_ROLES},
        {
            "f-people": FacetMetadata(provenance_id="provenance/A", unit="Person"),
            "f-unused-usd": FacetMetadata(provenance_id="provenance/A", unit="CurrencyUSD"),
            "f-unused-eur": FacetMetadata(provenance_id="provenance/Z", unit="CurrencyEUR"),
        },
        horizon=horizon,
    )

    assert dict(idx.provenance_currencies) == {}
    # The same pairs give provenance/A a shape, so every currency entry has one too.
    assert set(idx.provenance_shapes) == {"provenance/A"}


def test_provenance_shapes_join_facet_periods_and_variable_axes_per_provenance() -> None:
    table, horizon = coverage_table(
        [
            ("V-flow", "f-annual", FRA, KEN, [2020]),
            ("V-flow", "f-annual", DEU, ETH, [2021]),
            ("V-flow", "f-monthly", FRA, KEN, [2020]),
            ("V-debt", "f-annual", FRA, KEN, [2020]),
            ("V-odd", "f-other", FRA, KEN, [2020]),
            ("V-flow", "f-no-provenance", FRA, KEN, [2020]),
            ("V-unnamed", "f-other", FRA, KEN, [2020]),
        ]
    )
    idx = CoverageIndex(
        table,
        {
            "V-flow": RoleNames(donor="financingProvider", recipient="financingRecipient"),
            "V-debt": RoleNames(donor="creditor", recipient="debtor"),
            # A role outside the vocabulary maps to no axis.
            "V-odd": RoleNames(donor="financingProvider", recipient="beneficiaryOrganisation"),
        },
        {
            "f-annual": FacetMetadata(provenance_id="provenance/A", observation_period="P1Y"),
            "f-monthly": FacetMetadata(provenance_id="provenance/B", observation_period="P1M"),
            "f-other": FacetMetadata(provenance_id="provenance/C"),
            "f-no-provenance": FacetMetadata(observation_period="P1Y"),
            # Metadata for a facet with no coverage rows contributes nothing.
            "f-unused": FacetMetadata(provenance_id="provenance/D", observation_period="P1Y"),
        },
        horizon=horizon,
    )

    shapes = idx.provenance_shapes

    both = frozenset({"origin", "destination"})
    assert dict(shapes) == {
        "provenance/A": ProvenanceShape(
            observation_periods=frozenset({"P1Y"}), role_axes=frozenset({both})
        ),
        "provenance/B": ProvenanceShape(
            observation_periods=frozenset({"P1M"}), role_axes=frozenset({both})
        ),
        # `V-unnamed` has no role names, so only `V-odd` counts for C.
        "provenance/C": ProvenanceShape(
            observation_periods=frozenset({None}),
            role_axes=frozenset({frozenset({"origin", None})}),
        ),
    }
    # Built once when the index loads, so the second read returns the same mapping.
    assert idx.provenance_shapes is shapes


def test_provenance_shapes_of_an_empty_table_is_empty() -> None:
    horizon = YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=1)
    idx = CoverageIndex(
        horizon.schema.empty_table(),
        {},
        {"f": FacetMetadata(provenance_id="provenance/A", observation_period="P1Y")},
        horizon=horizon,
    )
    assert dict(idx.provenance_shapes) == {}


def test_variable_provenances_maps_each_variable_to_its_sorted_facet_provenances() -> None:
    table, horizon = coverage_table(
        [
            ("V-two", "f-b", FRA, KEN, [2020]),
            ("V-two", "f-a", FRA, KEN, [2020]),
            ("V-two", "f-a", DEU, ETH, [2021]),
            ("V-one", "f-a", FRA, KEN, [2020]),
            ("V-none", "f-no-provenance", FRA, KEN, [2020]),
            ("V-mixed", "f-no-provenance", FRA, KEN, [2020]),
            ("V-mixed", "f-b", FRA, KEN, [2020]),
            ("V-unknown-facet", "f-not-in-metadata", FRA, KEN, [2020]),
        ]
    )
    idx = CoverageIndex(
        table,
        {},
        {
            "f-a": FacetMetadata(provenance_id="provenance/A"),
            "f-b": FacetMetadata(provenance_id="provenance/B"),
            "f-no-provenance": FacetMetadata(observation_period="P1Y"),
            # Metadata for a facet with no coverage rows contributes nothing.
            "f-unused": FacetMetadata(provenance_id="provenance/D"),
        },
        horizon=horizon,
    )

    assert idx.variable_provenances() == {
        "V-two": ("provenance/A", "provenance/B"),
        "V-one": ("provenance/A",),
        "V-none": (),
        "V-mixed": ("provenance/B",),
        "V-unknown-facet": (),
    }


def test_variable_provenances_of_an_empty_table_is_empty() -> None:
    horizon = YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=1)
    idx = CoverageIndex(
        horizon.schema.empty_table(),
        {},
        {"f": FacetMetadata(provenance_id="provenance/A")},
        horizon=horizon,
    )
    assert idx.variable_provenances() == {}


def test_repeated_series_key_is_refused() -> None:
    with pytest.raises(ValueError, match="repeating a"):
        _window_index(
            [
                (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020]),
                (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2021]),
            ]
        )


def test_open_axis_recipient_under_donor_binding(index) -> None:
    assert index.open_axis(variables=[V1], axis="recipient", bound={"donor": frozenset({FRA})}) == {
        KEN,
        ETH,
    }


def test_open_axis_donor_under_recipient_binding(index) -> None:
    assert index.open_axis(variables=[V1], axis="donor", bound={"recipient": frozenset({KEN})}) == {
        FRA,
        DEU,
    }


def test_open_axis_unbound_returns_every_dcid_across_variables(index) -> None:
    assert index.open_axis(variables=[V1, V2], axis="donor") == {FRA, DEU, UNICEF}


def test_open_axis_empty_for_a_variable_with_no_rows(index) -> None:
    assert index.open_axis(variables=[V4], axis="donor") == frozenset()


def test_headline_picks_the_latest_year_across_every_surviving_row(index) -> None:
    # Rows touching V1/V2 under recipient=KEN: V1 rows through 2023 (FACET_PROJECTED), V2/FRA/KEN
    # through 2022. Latest is 2023, from the single V1/FRA/KEN/FACET_PROJECTED row.
    headline = index.headline(variables=[V1, V2], bound={"recipient": frozenset({KEN})})
    assert headline is not None
    assert headline.year == 2023
    assert headline.observations == 1
    assert headline.series == 1
    assert headline.role_counts == {"donor": 1, "recipient": 1}


def test_headline_role_counts_at_the_latest_year(index) -> None:
    # At V1's own latest year across every donor/recipient (2023, only FRA/KEN/PROJECTED), vs.
    # 2022, which both FRA/KEN and DEU/KEN reach.
    headline = index.headline(variables=[V1])
    assert headline is not None
    assert headline.year == 2023
    assert headline.series == 1


def test_headline_inside_a_window_stops_at_its_last_year() -> None:
    """A facet holding 2015 to 2030 heads a 2020 to 2026 window in 2026."""
    idx = _window_index([(_WINDOW_SV, _WINDOW_FACET, FRA, KEN, list(range(2015, 2031)))])

    windowed = idx.headline(variables=[_WINDOW_SV], first_year=2020, last_year=2026)
    lifetime = idx.headline(variables=[_WINDOW_SV])

    assert windowed is not None and windowed.year == 2026
    assert lifetime is not None and lifetime.year == 2030


def test_pair_rows_report_the_largest_row_and_its_members_with_data() -> None:
    """Kenya's row holds 2020 to 2024 and Ethiopia's 2024 alone: inside 2020 to 2022 the
    largest row counts three and only Kenya has data."""
    idx = _window_index(
        [
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, list(range(2020, 2025))),
            (_WINDOW_SV, _WINDOW_FACET, FRA, ETH, [2024]),
        ]
    )
    rows = idx.pair_rows(variables=[_WINDOW_SV])
    lookup = {"variable": _WINDOW_SV, "facet_id": _WINDOW_FACET}

    assert rows.max_n_obs(**lookup) == 5
    assert rows.max_n_obs(**lookup, first_year=2020, last_year=2022) == 3
    assert rows.members_with_data(**lookup, member_is_donor=False) == {KEN, ETH}
    assert rows.members_with_data(
        **lookup, member_is_donor=False, first_year=2020, last_year=2022
    ) == {KEN}
    assert rows.members_with_data(**lookup, member_is_donor=True) == {FRA}


_OTHER_FACET = "window-test-other-facet"


@pytest.mark.parametrize(
    ("read", "expected"),
    [
        ({}, ServedExtent(rows=6, series=2, first_year=2018, last_year=2024)),
        ({"latest": {}}, ServedExtent(rows=1, series=1, first_year=2024, last_year=2024)),
        (
            {"latest": {_WINDOW_SV: 2020}},
            ServedExtent(rows=1, series=1, first_year=2020, last_year=2020),
        ),
        (
            {"first_year": 2019, "last_year": 2021},
            ServedExtent(rows=3, series=1, first_year=2019, last_year=2021),
        ),
        (
            {"served": [(_WINDOW_SV, _WINDOW_FACET, None, None)]},
            ServedExtent(rows=4, series=2, first_year=2018, last_year=2024),
        ),
        (
            {"excluded": [(_WINDOW_SV, "recipient", ETH)]},
            ServedExtent(rows=5, series=1, first_year=2018, last_year=2022),
        ),
    ],
    ids=["every_year", "own_latest", "elected_latest", "range", "served", "excluded"],
)
def test_served_extent_counts_each_series_years_once_across_its_facets(
    read: dict[str, Any], expected: ServedExtent
) -> None:
    """Kenya's series holds 2018 to 2020 in one facet and 2020 to 2022 in another, five
    distinct years where the facets hold six observations; Ethiopia's holds 2024, past
    the first mask word. A `latest` read serves the variable's elected year from the
    series holding it, and with none elected the variable's own latest year."""
    idx = _window_index(
        [
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2018, 2019, 2020]),
            (_WINDOW_SV, _OTHER_FACET, FRA, KEN, [2020, 2021, 2022]),
            (_WINDOW_SV, _WINDOW_FACET, FRA, ETH, [2024]),
        ]
    )

    assert idx.served_extent(variables=[_WINDOW_SV], **read) == expected


def test_headline_none_when_nothing_survives(index) -> None:
    assert index.headline(variables=[V4]) is None
    assert index.headline(variables=[V1], bound={"donor": frozenset({"country/ZZZ"})}) is None


def test_estimate_rows_and_requests(index) -> None:
    estimate = index.estimate(variables=[V1], bound={"recipient": frozenset({KEN})})
    assert estimate.rows == 9  # matches test_filter_candidates_rolls_up_facets_per_variable
    assert estimate.requests == 1  # small variable list, well under the URL budget


def test_estimate_groups_by_role_names_across_heterogeneous_variables(index) -> None:
    # V1 (financingProvider/financingRecipient) and V5 (creditor/debtor) share no role-name
    # pair, so estimate() must build two separate entities filters/chunk plans. Since both
    # groups fit in one request each, requests is still small and does not raise for
    # mixing role-name shapes.
    estimate = index.estimate(variables=[V1, V5], bound={"donor": frozenset({FRA})})
    assert estimate.requests >= 1
    assert estimate.rows > 0


def test_estimate_fully_unbound_is_admitted_with_an_exact_estimate(index) -> None:
    # SDMX requires an entity filter, so an unbound request uses the smaller axis. Both
    # axes have 2 dcids; donor wins the tie. The 4 V1 rows total 11 observations.
    estimate = index.estimate(variables=[V1], bound={})
    assert estimate.rows == 11
    assert estimate.requests == 1


def test_estimate_empty_bound_set_admits_zero_rather_than_raising(index) -> None:
    # A role explicitly bound to an empty set matches no row: zero surviving rows, not the
    # fully-unbound case above, and not a refusal either.
    estimate = index.estimate(variables=[V1], bound={"donor": frozenset()})
    assert estimate == Estimate(rows=0, requests=0)


def test_exhaustive_single_axis_filter_matches_estimate_fully_unbound(index) -> None:
    # Same fully-unbound V1 case as test_estimate_fully_unbound_is_admitted_with_an_exact_
    # estimate: donor {FRA, DEU} ties recipient {KEN, ETH} at 2, donor wins the tie, sorted.
    entities = index.exhaustive_single_axis_filter(
        variables=[V1], donor_role="financingProvider", recipient_role="financingRecipient"
    )
    assert entities == {"financingProvider": (DEU, FRA)}


def test_exhaustive_single_axis_filter_empty_for_no_surviving_coverage(index) -> None:
    assert (
        index.exhaustive_single_axis_filter(
            variables=[V4], donor_role="financingProvider", recipient_role="financingRecipient"
        )
        == {}
    )


def test_role_names_are_carried_through(index) -> None:
    assert index.role_names[V1].donor == "financingProvider"
    assert index.role_names[V5].donor == "creditor"


def test_year_mask_invariant_holds_for_every_row_in_the_test_artifact() -> None:
    # Sanity check on the fixture itself: the mask words' own bit count agrees
    # with the n_obs support.py wrote for every row it built.
    _table, horizon = coverage_table(_ROWS)
    for _variable, _facet_id, _donor, _recipient, years in _ROWS:
        assert sum(word.bit_count() for word in horizon.masks_from_years(years)) == len(years)


# -- group-scoped pair reads ------------------------------------------------

# Rows interleaved across groups, so no group arrives contiguous, with names
# out of code-point order (including one outside ASCII) on every key column.
_GROUPED_VARIABLES = ("sv/b", "sv/a", "sv/c")
_GROUPED_FACETS = ("facet-2", "facet-1")
_GROUPED_DONORS = ("donor/Z", "donor/b", "donor/É", "donor/a")
_GROUPED_RECIPIENTS = ("recipient/y", "recipient/X", "recipient/ñ")
_GROUPED_ROWS = [
    (variable, facet_id, donor, recipient, [2000 + (i % 7), 2010 + (i % 3)])
    for i, (donor, recipient, variable, facet_id) in enumerate(
        (donor, recipient, variable, facet_id)
        for donor in _GROUPED_DONORS
        for recipient in _GROUPED_RECIPIENTS
        for variable in _GROUPED_VARIABLES
        for facet_id in _GROUPED_FACETS
    )
]
_GROUPED_ROLE_NAMES = {
    variable: RoleNames(donor="financingProvider", recipient="financingRecipient")
    for variable in _GROUPED_VARIABLES
}


def _grouped_index() -> CoverageIndex:
    table, horizon = coverage_table(_GROUPED_ROWS)
    return CoverageIndex(table, _GROUPED_ROLE_NAMES, horizon=horizon)


def test_index_rows_are_grouped_by_sorted_variable_and_facet() -> None:
    idx = _grouped_index()
    for name in _KEY_COLUMNS:
        dictionary = cast(
            "list[str]",
            cast("pa.DictionaryArray", idx._table[name].chunk(0)).dictionary.to_pylist(),
        )
        assert dictionary == sorted(dictionary), name
    keys = list(
        zip(*(idx._table[name].to_pylist() for name in ("variable", "facet_id")), strict=True)
    )
    assert keys == sorted(keys)
    groups = idx.pair_rows(variables=list(_GROUPED_VARIABLES)).groups()
    assert [(g.variable, g.facet_id) for g in groups] == sorted(
        (v, f) for v in _GROUPED_VARIABLES for f in _GROUPED_FACETS
    )


@pytest.mark.parametrize("chunks", [1, 2])
def test_dictionaries_repeating_a_value_answer_like_plain_strings(chunks: int) -> None:
    """A key column whose dictionary holds each value at two codes gets one
    code per value, so every lookup reads the same rows as a plain-string
    build of the same data."""
    table, horizon = coverage_table(_GROUPED_ROWS)
    repeated = with_repeated_dictionaries(table, chunks=chunks)
    assert repeated["variable"].num_chunks == chunks
    plain = CoverageIndex(table, _GROUPED_ROLE_NAMES, horizon=horizon)
    idx = CoverageIndex(repeated, _GROUPED_ROLE_NAMES, horizon=horizon)
    variables = ["sv/a", "sv/c"]
    bounds: list[BoundRoles] = [
        {},
        {"donor": frozenset({"donor/b", "donor/É"})},
        {"recipient": frozenset({"recipient/ñ"})},
    ]
    for bound in bounds:
        for first_year, last_year in ((None, None), (2003, 2010)):
            window = {"first_year": first_year, "last_year": last_year}
            assert idx.filter_candidates(
                variables=variables, bound=bound, **window
            ) == plain.filter_candidates(variables=variables, bound=bound, **window)
            rows = idx.pair_rows(variables=variables, bound=bound, **window)
            plain_rows = plain.pair_rows(variables=variables, bound=bound, **window)
            assert rows.groups() == plain_rows.groups()
            for group in plain_rows.groups():
                key = {"variable": group.variable, "facet_id": group.facet_id}
                assert rows.pairs(**key) == plain_rows.pairs(**key)
            for axis in ("donor", "recipient"):
                assert idx.open_axis(
                    variables=variables, axis=axis, bound=bound, **window
                ) == plain.open_axis(variables=variables, axis=axis, bound=bound, **window)
            assert idx.estimate(variables=variables, bound=bound, **window) == plain.estimate(
                variables=variables, bound=bound, **window
            )
        assert idx.exhaustive_single_axis_filter(
            variables=variables, donor_role="d", recipient_role="r"
        ) == plain.exhaustive_single_axis_filter(
            variables=variables, donor_role="d", recipient_role="r"
        )
        assert idx.headline(variables=variables, bound=bound) == plain.headline(
            variables=variables, bound=bound
        )
    assert idx._table.equals(plain._table)


def test_repeated_series_key_across_repeated_dictionary_codes_is_refused() -> None:
    table, horizon = coverage_table(
        [
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2020]),
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2021]),
        ]
    )
    repeated = with_repeated_dictionaries(table, chunks=1)
    assert repeated["donor"].chunk(0).indices.to_pylist() == [0, 1]
    with pytest.raises(ValueError, match="repeating a"):
        CoverageIndex(repeated, _WINDOW_ROLE_NAMES, horizon=horizon)


@pytest.mark.parametrize(("first_year", "last_year"), [(None, None), (2003, 2010)])
@pytest.mark.parametrize("member_is_donor", [True, False])
@pytest.mark.parametrize("member_sorts_first", [True, False])
def test_head_pairs_ranks_like_sorting_the_decoded_pairs(
    first_year: int | None, last_year: int | None, member_is_donor: bool, member_sorts_first: bool
) -> None:
    rows = _grouped_index().pair_rows(variables=list(_GROUPED_VARIABLES))
    for variable in _GROUPED_VARIABLES:
        for facet_id in _GROUPED_FACETS:
            decoded = [
                ((d, r) if member_is_donor else (r, d), n)
                for d, r, n in rows.pairs(
                    variable=variable, facet_id=facet_id, first_year=first_year, last_year=last_year
                )
            ]
            ranked = sorted(
                decoded, key=lambda p: p[0] if member_sorts_first else (p[0][1], p[0][0])
            )
            expected = [(member, opposite, n) for (member, opposite), n in ranked[:5]]
            assert (
                rows.head_pairs(
                    variable=variable,
                    facet_id=facet_id,
                    limit=5,
                    member_is_donor=member_is_donor,
                    has_opposite=True,
                    member_sorts_first=member_sorts_first,
                    first_year=first_year,
                    last_year=last_year,
                )
                == expected
            )
            totals: dict[str, int] = {}
            for (member, _opposite), n in decoded:
                totals[member] = totals.get(member, 0) + n
            assert rows.head_pairs(
                variable=variable,
                facet_id=facet_id,
                limit=2,
                member_is_donor=member_is_donor,
                has_opposite=False,
                first_year=first_year,
                last_year=last_year,
            ) == [(member, None, n) for member, n in sorted(totals.items())[:2]]
            assert rows.n_obs(
                variable=variable, facet_id=facet_id, first_year=first_year, last_year=last_year
            ) == sum(totals.values())


@pytest.mark.parametrize(("first_year", "last_year"), [(None, None), (2003, 2040)])
@pytest.mark.parametrize("member_is_donor", [True, False])
@pytest.mark.parametrize("has_opposite", [True, False])
def test_pair_year_counts_hold_what_head_pairs_counts_one_year_at_a_time(
    first_year: int | None, last_year: int | None, member_is_donor: bool, has_opposite: bool
) -> None:
    """Each row's years and observations, read across both mask words, are what a one-year
    `head_pairs` window counts for it, so rows a one-role read sums add their observations
    in each year."""
    rows_with_late_years = [
        (variable, facet_id, donor, recipient, [*years, 2035 + i % 2])
        for i, (variable, facet_id, donor, recipient, years) in enumerate(_GROUPED_ROWS)
    ]
    table, horizon = coverage_table(rows_with_late_years)
    assert horizon.mask_words == 2
    rows = CoverageIndex(table, _GROUPED_ROLE_NAMES, horizon=horizon).pair_rows(
        variables=list(_GROUPED_VARIABLES)
    )
    key = {"member_is_donor": member_is_donor, "has_opposite": has_opposite}
    for variable in _GROUPED_VARIABLES:
        for facet_id in _GROUPED_FACETS:
            group = {"variable": variable, "facet_id": facet_id}
            expected: dict[tuple[str, str | None], list[tuple[int, int]]] = {}
            for year in range(2000, 2040):
                if (first_year is not None and year < first_year) or (
                    last_year is not None and year > last_year
                ):
                    continue
                for member, opposite, n in rows.head_pairs(
                    **group, **key, limit=1000, first_year=year, last_year=year, with_data=True
                ):
                    expected.setdefault((member, opposite), []).append((year, n))
            found = rows.pair_year_counts(
                **group, **key, first_year=first_year, last_year=last_year
            )
            assert {(m, o): list(counts) for m, o, counts in found} == expected
            if not has_opposite:
                assert any(n > 1 for _m, _o, counts in found for _year, n in counts)
            assert len(found) == len(expected)


def test_group_reads_miss_cleanly() -> None:
    rows = _grouped_index().pair_rows(variables=["sv/a"], bound={"donor": frozenset({"donor/a"})})
    for variable, facet_id in (("sv/b", "facet-1"), ("sv/a", "no-such-facet"), ("nope", "facet-1")):
        assert rows.pairs(variable=variable, facet_id=facet_id) == []
        assert rows.n_obs(variable=variable, facet_id=facet_id) == 0
        assert (
            rows.head_pairs(
                variable=variable,
                facet_id=facet_id,
                limit=5,
                member_is_donor=True,
                has_opposite=True,
            )
            == []
        )
    assert {(d, r) for d, r, _n in rows.pairs(variable="sv/a", facet_id="facet-1")} == {
        ("donor/a", recipient) for recipient in _GROUPED_RECIPIENTS
    }


class _ArraySizeSpy:
    """Stands in for `np`, `pa` or `pc` inside `series_coverage.index`,
    recording per function the length of the largest array (or sequence
    of arrays) any call passed it. Types and ufuncs pass through unwrapped."""

    def __init__(self, module: object, largest: dict[str, int]) -> None:
        self._module = module
        self._largest = largest

    @staticmethod
    def _size(value: object) -> int:
        if isinstance(value, (np.ndarray, pa.Array, pa.ChunkedArray)):
            return len(value)
        if isinstance(value, (list, tuple)):
            items = (_ArraySizeSpy._size(item) for item in value)
            return max(len(value) if isinstance(value, list) else 0, *items, 0)
        if isinstance(value, dict):
            return max((_ArraySizeSpy._size(item) for item in value.values()), default=0)
        return 0

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._module, name)
        if not callable(attr) or isinstance(attr, (type, np.ufunc)):
            return attr

        def spied(*args: object, **kwargs: object) -> object:
            size = max((self._size(a) for a in (*args, *kwargs.values())), default=0)
            self._largest[name] = max(self._largest.get(name, 0), size)
            return attr(*args, **kwargs)

        return spied


_MANY_VARIABLES = [f"sv/{v:04d}" for v in range(4000)]
_ONE_GROUP_SIZE = 24  # 4 donors x 6 recipients


@pytest.fixture(scope="module")
def many_groups_index() -> CoverageIndex:
    """4,000 (variable, facet) groups of 24 rows each, rows interleaved
    across groups."""
    rows = [
        (variable, "facet", f"donor/{d}", f"recipient/{r}", [2000 + d, 2010 + r])
        for d in range(4)
        for r in range(6)
        for variable in _MANY_VARIABLES
    ]
    table, horizon = coverage_table(rows)
    idx = CoverageIndex(
        table,
        {variable: RoleNames(donor="d", recipient="r") for variable in _MANY_VARIABLES},
        horizon=horizon,
    )
    assert idx.num_rows == len(_MANY_VARIABLES) * _ONE_GROUP_SIZE
    return idx


@pytest.fixture
def many_groups(many_groups_index: CoverageIndex) -> PairRows:
    """A fresh `PairRows` over every row of `many_groups_index`, with
    `groups()` already read (as `role_scope` reads it before any preview)
    and no group read yet."""
    pair_rows = many_groups_index.pair_rows(variables=_MANY_VARIABLES)
    pair_rows.groups()
    return pair_rows


def _read_one_group(pair_rows: PairRows) -> None:
    """Every preview read of one group: `head_pairs` in both shapes, `n_obs`
    and `pairs`, with and without a narrower window."""
    for first_year, last_year in ((None, None), (2001, 2012)):
        for has_opposite in (True, False):
            head = pair_rows.head_pairs(
                variable="sv/0042",
                facet_id="facet",
                limit=5,
                member_is_donor=False,
                has_opposite=has_opposite,
                first_year=first_year,
                last_year=last_year,
            )
            assert len(head) == 5
        assert (
            pair_rows.n_obs(
                variable="sv/0042", facet_id="facet", first_year=first_year, last_year=last_year
            )
            > 0
        )
        pairs = pair_rows.pairs(
            variable="sv/0042", facet_id="facet", first_year=first_year, last_year=last_year
        )
        assert len(pairs) == _ONE_GROUP_SIZE


def test_one_group_read_passes_only_that_groups_rows_to_any_kernel(
    many_groups: PairRows, monkeypatch
) -> None:
    """Group reads pass only group-sized arrays to kernels.
    Binary searches may read the full code column."""
    largest: dict[str, int] = {}
    for name in ("np", "pa", "pc"):
        module = getattr(coverage_index_module, name)
        monkeypatch.setattr(coverage_index_module, name, _ArraySizeSpy(module, largest))
    _read_one_group(many_groups)
    monkeypatch.undo()

    # The locating binary searches take the whole code column and read
    # O(log rows) of it; every other call sees the group's rows alone.
    probed = largest.pop("searchsorted", 0)
    assert max(largest.values()) <= _ONE_GROUP_SIZE, largest
    assert largest["lexsort"] == _ONE_GROUP_SIZE
    assert probed == many_groups.num_rows


def test_one_group_read_allocates_nothing_sized_by_the_population(
    many_groups_index: CoverageIndex,
) -> None:
    """Tracemalloc includes numpy buffers. At 96,000 rows, a bool array alone needs 96 KB,
    above the 48 KB bound. The minimum peak across 3 fresh `PairRows` limits thread noise."""
    peaks = []
    for _ in range(3):
        pair_rows = many_groups_index.pair_rows(variables=_MANY_VARIABLES)
        pair_rows.groups()
        tracemalloc.start()
        try:
            _read_one_group(pair_rows)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert min(peaks) < 48 * 1024, peaks


def test_lookups_pass_only_the_requested_variables_rows_to_any_kernel(
    many_groups_index: CoverageIndex, monkeypatch
) -> None:
    """Every `CoverageIndex` lookup over three of 4,000 variables passes no
    array longer than those variables' rows to any numpy, pyarrow or
    pyarrow.compute call, other than the binary searches that locate each
    variable's rows and the scan of `_key_values`' dictionary-length
    presence mask: no mask over the whole table, and `_decode` converts
    only the variables the rows use, never the whole dictionary."""
    idx = many_groups_index
    variables = ["sv/0007", "sv/1234", "sv/3999"]
    bound: BoundRoles = {"donor": frozenset({"donor/1", "donor/3"})}
    largest: dict[str, int] = {}
    for name in ("np", "pa", "pc"):
        module = getattr(coverage_index_module, name)
        monkeypatch.setattr(coverage_index_module, name, _ArraySizeSpy(module, largest))
    decoded: list[int] = []
    decode = coverage_index_module._decode

    def spied_decode(dictionary: pa.Array, codes: np.ndarray) -> list[str]:
        decoded.append(len(codes))
        return decode(dictionary, codes)

    monkeypatch.setattr(coverage_index_module, "_decode", spied_decode)
    for first_year, last_year in ((None, None), (2001, 2012)):
        window = {"first_year": first_year, "last_year": last_year}
        assert len(idx.filter_candidates(variables=variables, bound=bound, **window)) == 3
        assert idx.pair_rows(variables=variables, bound=bound, **window).groups()
        assert idx.open_axis(variables=variables, axis="recipient", bound=bound, **window)
        assert idx.estimate(variables=variables, bound=bound, **window).rows > 0
        assert idx.estimate(variables=variables, bound={}, **window).rows > 0
    assert idx.exhaustive_single_axis_filter(
        variables=variables, donor_role="d", recipient_role="r"
    )
    assert idx.headline(variables=variables, bound=bound) is not None
    monkeypatch.undo()

    probed = largest.pop("searchsorted", 0)
    marked = largest.pop("flatnonzero", 0)
    assert max(largest.values()) <= len(variables) * _ONE_GROUP_SIZE, largest
    assert probed == idx.num_rows
    assert marked == len(idx._variable_codes) < idx.num_rows
    assert decoded
    assert max(decoded) <= len(variables), decoded


@pytest.mark.parametrize(
    "pair_roles",
    [("recipient",), ("recipient", "donor")],
    ids=["recipient_pairs", "donor_recipient_pairs"],
)
def test_pair_facet_candidates_match_a_filter_on_each_pair(index, pair_roles) -> None:
    """One grouped read returns what `filter_candidates` returns for each pair bound alone,
    with rows on a role outside the pair pooled into it."""
    variables = sorted({row[0] for row in _ROWS})

    by_pair = index.pair_facet_candidates(variables=variables, bound={}, pair_roles=pair_roles)

    assert by_pair
    for (variable, dcids), facets in by_pair.items():
        bound = {role: frozenset({dcid}) for role, dcid in zip(pair_roles, dcids, strict=True)}
        (expected,) = index.filter_candidates(variables=[variable], bound=bound)
        assert facets == expected.facets


def test_pair_facet_candidates_respect_the_bound_roles(index) -> None:
    by_pair = index.pair_facet_candidates(
        variables=[V1], bound={"recipient": frozenset({KEN})}, pair_roles=("recipient",)
    )

    assert {dcids for _variable, dcids in by_pair} == {(KEN,)}


@pytest.mark.parametrize("pair_roles", [("recipient",), ("recipient", "donor")])
@pytest.mark.parametrize(
    ("bound", "years"),
    [({}, (None, None)), ({}, (2015, 2020)), ({"recipient": frozenset({KEN})}, (None, None))],
    ids=["every_year", "a_range", "a_bound_recipient"],
)
def test_pair_count_counts_the_pairs_pair_facet_candidates_returns(
    index, pair_roles, bound, years
) -> None:
    """The count a read decision takes before building any pair equals the pairs it would
    build, without building them."""
    variables = sorted({row[0] for row in _ROWS})
    first_year, last_year = years
    arguments = {
        "variables": variables,
        "bound": bound,
        "pair_roles": pair_roles,
        "first_year": first_year,
        "last_year": last_year,
    }

    assert index.pair_count(**arguments) == len(index.pair_facet_candidates(**arguments))


def test_pair_count_is_zero_without_a_row(index) -> None:
    assert index.pair_count(variables=["unknown"], bound={}, pair_roles=("recipient",)) == 0


@pytest.mark.parametrize("pair_roles", [("recipient",), ("recipient", "donor")])
def test_a_range_selects_pairs_holding_data_in_it_and_keeps_their_facets_from_every_year(
    pair_roles,
) -> None:
    """KEN holds the facet from FRA in 2005 and 2020 to 2022, and ETH from DEU in 2000 alone.
    A 2020 to 2022 range selects KEN's pair alone, with both its facets' years, 2005
    included; its count equals the pairs it returns. Both bounds open select every pair."""
    other = "other-facet"
    idx = _window_index(
        [
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2005, 2020, 2021, 2022]),
            (_WINDOW_SV, other, FRA, KEN, [2005]),
            (_WINDOW_SV, _WINDOW_FACET, DEU, ETH, [2000]),
        ]
    )
    kenya = (KEN,) if pair_roles == ("recipient",) else (KEN, FRA)

    def pairs(first_year: int | None, last_year: int | None):
        return idx.pair_facet_candidates(
            variables=[_WINDOW_SV],
            bound={},
            pair_roles=pair_roles,
            first_year=first_year,
            last_year=last_year,
        )

    def count(first_year: int | None, last_year: int | None) -> int:
        return idx.pair_count(
            variables=[_WINDOW_SV],
            bound={},
            pair_roles=pair_roles,
            first_year=first_year,
            last_year=last_year,
        )

    by_pair = pairs(2020, 2022)

    assert list(by_pair) == [(_WINDOW_SV, kenya)]
    assert [
        (f.facet_id, f.first_year, f.last_year, f.n_obs) for f in by_pair[_WINDOW_SV, kenya]
    ] == [
        (other, 2005, 2005, 1),
        (_WINDOW_FACET, 2005, 2022, 4),
    ]
    assert count(2020, 2022) == 1
    assert len(pairs(None, None)) == count(None, None) == 2


_MONTHLY_FACET = "monthly-facet"


@pytest.mark.parametrize(
    ("read", "expected", "kept"),
    [
        pytest.param(
            {},
            {(_WINDOW_SV, KEN, FRA): [2005, 2020, 2021, 2022], (_WINDOW_SV, ETH, DEU): [2000]},
            KeptFacets(frozenset({_WINDOW_FACET, _MONTHLY_FACET}), groups=2, sub_annual=True),
            id="every-year",
        ),
        pytest.param(
            {"first_year": 2020, "last_year": 2022},
            {(_WINDOW_SV, KEN, FRA): [2020, 2021, 2022]},
            KeptFacets(frozenset({_WINDOW_FACET}), groups=1, sub_annual=False),
            id="a-range",
        ),
        pytest.param(
            {"latest": {}},
            {(_WINDOW_SV, KEN, FRA): [2022]},
            KeptFacets(frozenset({_WINDOW_FACET, _MONTHLY_FACET}), groups=2, sub_annual=True),
            id="latest-keeps-the-variables-latest-year",
        ),
        pytest.param(
            {"latest": {_WINDOW_SV: 2005}},
            {(_WINDOW_SV, KEN, FRA): [2005]},
            KeptFacets(frozenset({_WINDOW_FACET, _MONTHLY_FACET}), groups=2, sub_annual=True),
            id="latest-keeps-the-elected-year",
        ),
        pytest.param(
            {"admits": lambda _variable, facet: facet != _WINDOW_FACET},
            {(_WINDOW_SV, KEN, FRA): [2005]},
            KeptFacets(frozenset({_MONTHLY_FACET}), groups=1, sub_annual=True),
            id="an-admitted-facet",
        ),
    ],
)
def test_series_years_or_each_series_facets(read, expected, kept) -> None:
    """KEN holds the annual facet from FRA in 2005 and 2020 to 2022 and a monthly facet
    with another method in 2005 alone. ETH holds the annual facet from DEU in 2000. Each
    series ORs the years of the facets it keeps. Under `latest` the variable serves 2022
    alone, so ETH, whose latest year is 2000, serves nothing, and elected at 2005 it
    serves that year from KEN, which holds it."""
    table, horizon = coverage_table(
        [
            (_WINDOW_SV, _WINDOW_FACET, FRA, KEN, [2005, 2020, 2021, 2022]),
            (_WINDOW_SV, _MONTHLY_FACET, FRA, KEN, [2005]),
            (_WINDOW_SV, _WINDOW_FACET, DEU, ETH, [2000]),
        ]
    )
    idx = CoverageIndex(
        table,
        _WINDOW_ROLE_NAMES,
        {
            _WINDOW_FACET: FacetMetadata(provenance_id="p", observation_period="P1Y"),
            _MONTHLY_FACET: FacetMetadata(
                provenance_id="p", measurement_method="Monthly", observation_period="P1M"
            ),
        },
        horizon=horizon,
    )

    found = idx.series_years(variables=[_WINDOW_SV], **read)

    assert found is not None
    epoch = idx.horizon.epoch_year
    assert dict(zip(found.keys(), found.year_bits(), strict=True)) == {
        key: sum(1 << (year - epoch) for year in years) for key, years in expected.items()
    }
    assert found.variables == {_WINDOW_SV: kept}


def test_a_scoped_read_keeps_the_admitted_provenances_facets_judging_each_provenance_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission keeps each provenance of a series that the scope admits, so a scoped read
    keeps every row whose facet's provenance is admitted and asks the catalogue once,
    whatever the number of series."""
    other = "provenance/Other"
    table, horizon = coverage_table(_ROWS)
    idx = CoverageIndex(
        table,
        ROLE_NAMES,
        {
            FACET_ACTUALS: FacetMetadata(provenance_id=_CRS_PROVENANCE, observation_period="P1Y"),
            FACET_PROJECTED: FacetMetadata(provenance_id=other, observation_period="P1Y"),
        },
        horizon=horizon,
    )
    asked: list[tuple[str, ...]] = []
    admit = FakeCatalogue.admit_series

    def counted(self: FakeCatalogue, provenances: Iterable[str]) -> SeriesAdmission:
        asked.append(tuple(provenances))
        return admit(self, asked[-1])

    monkeypatch.setattr(FakeCatalogue, "admit_series", counted)

    with bound_catalogue(FakeCatalogue().scoped(frozenset({_CRS_PROVENANCE}))):
        crs = idx.admitted_pairs(variable=V1)
    with bound_catalogue(FakeCatalogue().scoped(frozenset({other}))):
        projected = idx.admitted_pairs(variable=V1)

    assert crs == {FACET_ACTUALS: frozenset({(FRA, KEN), (DEU, KEN), (FRA, ETH)})}
    assert projected == {FACET_PROJECTED: frozenset({(FRA, KEN)})}
    assert len(asked) == 2
