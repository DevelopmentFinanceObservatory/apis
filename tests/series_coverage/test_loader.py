"""Pins the checks `checked_index` runs on coverage rows before they become an index."""

from __future__ import annotations

import pyarrow as pa
import pytest

from series_coverage.loader import CoverageArtifactError
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames, YearHorizon
from tests.series_coverage.support import (
    BUILD_EPOCH_YEAR,
    FACET_ACTUALS,
    FACET_METADATA,
    KEN,
    V1,
    build_test_index,
    checked_test_index,
    coverage_table,
)

_HORIZON = YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=1)
_TWO_ROLE = RoleNames(donor="financingProvider", recipient="financingRecipient")
_FACETS = {"f1": FacetMetadata(provenance_id="provenance/Test")}


def _table(*, n_obs: list[int], years: list[int]) -> pa.Table:
    """Rows of variable `V1` under facet `f1` from `country/FRA` to `country/KEN`, one per
    year in `years`, each stating its `n_obs` beside a mask holding that one year."""
    return pa.table(
        {
            "variable": ["V1"] * len(years),
            "facet_id": ["f1"] * len(years),
            "donor": ["country/FRA"] * len(years),
            "recipient": ["country/KEN"] * len(years),
            "n_obs": pa.array(n_obs, type=pa.uint32()),
            "first_year": pa.array(years, type=pa.uint16()),
            "last_year": pa.array(years, type=pa.uint16()),
            "year_mask_0": pa.array(
                [1 << (year - BUILD_EPOCH_YEAR) for year in years], type=pa.uint64()
            ),
        },
        schema=_HORIZON.schema,
    )


def test_checked_index_builds_the_index_of_rows_that_pass_every_check() -> None:
    index = build_test_index()
    assert index.num_rows > 0
    assert index.build_year == 2026
    assert index.facet_metadata == FACET_METADATA
    assert index.facet_metadata[FACET_ACTUALS].observation_period == "P1Y"
    assert index.filter_candidates(variables=[V1], bound={"recipient": frozenset({KEN})})


def test_checked_index_raises_on_n_obs_year_mask_mismatch() -> None:
    table = _table(n_obs=[5], years=[2007])
    with pytest.raises(CoverageArtifactError, match="bit count"):
        checked_test_index(table, {"V1": _TWO_ROLE}, _FACETS, horizon=_HORIZON)


def test_checked_index_raises_on_a_repeated_series_key() -> None:
    # Two rows share (V1, f1, FRA, KEN); each row alone satisfies the popcount invariant.
    table = _table(n_obs=[1, 1], years=[2007, 2008])
    with pytest.raises(CoverageArtifactError, match=r"1 row\(s\) repeating"):
        checked_test_index(table, {"V1": _TWO_ROLE}, _FACETS, horizon=_HORIZON)


@pytest.mark.parametrize(
    ("names", "rows"),
    [
        (
            RoleNames(donor="financingProvider", recipient=None),
            [("country/FRA", ABSENT_ENTITY), ("country/DEU", KEN)],
        ),
        (
            RoleNames(donor=None, recipient="financingRecipient"),
            [(ABSENT_ENTITY, KEN), ("country/FRA", "country/UGA")],
        ),
    ],
    ids=["provider-only-with-a-recipient-row", "recipient-only-with-a-donor-row"],
)
def test_checked_index_refuses_a_one_role_variable_whose_rows_mix_an_empty_and_a_set_entity(
    names: RoleNames, rows: list[tuple[str, str]]
) -> None:
    # Every row of a variable lacking a role must leave that role's column empty; one
    # empty row beside a populated one would let a bound lookup on the lacking role match.
    table, horizon = coverage_table(
        [("V1", "f1", donor, recipient, [2020]) for donor, recipient in rows]
    )
    with pytest.raises(CoverageArtifactError, match="V1"):
        checked_test_index(table, {"V1": names}, _FACETS, horizon=horizon)


def test_checked_index_raises_when_a_coverage_facet_id_has_no_metadata_entry() -> None:
    table = _table(n_obs=[1], years=[2007])
    with pytest.raises(CoverageArtifactError, match="f1"):
        checked_test_index(table, {"V1": _TWO_ROLE}, {}, horizon=_HORIZON)
