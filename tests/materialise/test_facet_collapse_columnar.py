"""Table-path facet collapse tests compare against the independent row oracle."""

from __future__ import annotations

import random
import unicodedata

import pyarrow as pa

from dc_sdmx import SdmxObservation
from materialise.facet_collapse import collapse_duplicate_identity_facets_table

from . import _row_oracle
from ._table_support import table_from_observations

_STAT_VAR = "dcid:Count_Person_Female"
_ROLES = ("observationAbout",)


def _row(
    *,
    facet_id: str | None,
    observation_period: str | None,
    time_period: str,
    value: str,
    unit: str | None = None,
    measurement_method: str | None = None,
    provenance: str | None = "dc/base/Kenya_Census",
    entity: str = "country/KEN",
) -> SdmxObservation:
    return SdmxObservation(
        variable=_STAT_VAR,
        role_values={"observationAbout": entity},
        unit=unit,
        measurement_method=measurement_method,
        observation_period=observation_period,
        provenance=provenance,
        time_period=time_period,
        value=value,
        scaling_factor=None,
        facet_id=facet_id,
    )


def _table(rows: tuple[SdmxObservation, ...]) -> pa.Table:
    return table_from_observations(rows, roles=_ROLES)


def _collapse_table(rows: tuple[SdmxObservation, ...]) -> pa.Table:
    return collapse_duplicate_identity_facets_table(_table(rows), declared_roles=_ROLES)


def _assert_table_matches_row_oracle(rows: tuple[SdmxObservation, ...]) -> None:
    """Runs both paths over the same `rows` and asserts they keep the same
    rows: the row oracle's result, converted through the same table
    builder the table path's input goes through, must equal the table
    path's own result.
    """
    row_result = _row_oracle.collapse_duplicate_identity_facets(rows, declared_roles=_ROLES)
    assert _collapse_table(rows).equals(_table(row_result))


# ---------------------------------------------------------------------------
# A genuine grain collision resolves to one row on the table path too.
# ---------------------------------------------------------------------------


def test_same_facet_id_role_dropped_pair_leaves_both_rows_for_the_grammar_to_refuse():
    present_row = _row(facet_id="f-shared", observation_period="P1Y", time_period="2020", value="1")
    omitted_row = SdmxObservation(
        variable=_STAT_VAR,
        role_values={},  # the declared role's cell was the NA fallback
        unit=None,
        measurement_method=None,
        observation_period="P1Y",
        provenance="dc/base/Kenya_Census",
        time_period="2020",
        value="2",
        scaling_factor=None,
        facet_id="f-shared",
    )

    _assert_table_matches_row_oracle((present_row, omitted_row))


# ---------------------------------------------------------------------------
# Facets on the same identity survive together when their grain keys diverge.
# ---------------------------------------------------------------------------


def test_facets_on_the_same_identity_with_disjoint_years_both_survive():
    facet_2015 = _row(facet_id="f-2015", observation_period=None, time_period="2015", value="1000")
    facet_2016 = _row(facet_id="f-2016", observation_period=None, time_period="2016", value="2000")

    _assert_table_matches_row_oracle((facet_2015, facet_2016))


def test_facets_on_the_same_identity_with_disjoint_entities_both_survive():
    facet_ken = _row(
        facet_id="f-ken",
        observation_period=None,
        time_period="2015",
        value="1000",
        entity="country/KEN",
    )
    facet_tgo = _row(
        facet_id="f-tgo",
        observation_period=None,
        time_period="2015",
        value="2000",
        entity="country/TGO",
    )

    _assert_table_matches_row_oracle((facet_ken, facet_tgo))


def test_identity_equal_facets_with_partial_period_overlap_lose_only_the_contested_row():
    wide_rows = tuple(
        _row(facet_id="f-wide", observation_period=None, time_period=year, value="10")
        for year in ("2015", "2018", "2020")
    )
    narrow_2015 = _row(facet_id="f-narrow", observation_period=None, time_period="2015", value="99")
    narrow_2016 = _row(facet_id="f-narrow", observation_period=None, time_period="2016", value="98")

    _assert_table_matches_row_oracle((*wide_rows, narrow_2015, narrow_2016))


# ---------------------------------------------------------------------------
# Contest resolution is per grain key, not per facet or per component.
# ---------------------------------------------------------------------------


def test_two_independent_contested_keys_resolve_independently():
    a_row = _row(facet_id="A", observation_period="P1Y", time_period="2015", value="1")
    b_row = _row(facet_id="B", observation_period=None, time_period="2015", value="2")
    c_row = _row(facet_id="C", observation_period="P1Y", time_period="2020", value="3")
    d_row = _row(facet_id="D", observation_period=None, time_period="2020", value="4")

    _assert_table_matches_row_oracle((a_row, b_row, c_row, d_row))


# ---------------------------------------------------------------------------
# Each ranking tier isolated, same as the row-path tests.
# ---------------------------------------------------------------------------


def test_only_the_observation_period_tier_decides():
    period_row = _row(facet_id="B-period", observation_period="P1Y", time_period="2020", value="1")
    periodless_row = _row(
        facet_id="A-periodless", observation_period=None, time_period="2020", value="2"
    )

    _assert_table_matches_row_oracle((period_row, periodless_row))


def test_only_the_row_count_tier_decides():
    many_rows = tuple(
        _row(facet_id="z-many", observation_period=None, time_period=year, value="1")
        for year in ("2018", "2019", "2020")
    )
    few_rows = tuple(
        _row(facet_id="a-few", observation_period=None, time_period=year, value="2")
        for year in ("2018", "2020")
    )

    _assert_table_matches_row_oracle(many_rows + few_rows)


def test_only_the_span_tier_decides():
    wide_2010 = _row(facet_id="z-wide", observation_period=None, time_period="2010", value="1")
    wide_2020 = _row(facet_id="z-wide", observation_period=None, time_period="2020", value="1")
    narrow_2020 = _row(facet_id="a-narrow", observation_period=None, time_period="2020", value="2")
    narrow_2021 = _row(facet_id="a-narrow", observation_period=None, time_period="2021", value="2")

    _assert_table_matches_row_oracle((wide_2010, wide_2020, narrow_2020, narrow_2021))


def test_real_facet_id_always_beats_the_none_pseudo_group():
    real_row = _row(facet_id="42", observation_period=None, time_period="2020", value="1")
    unkeyed_row = _row(facet_id=None, observation_period="P1Y", time_period="2020", value="2")

    _assert_table_matches_row_oracle((real_row, unkeyed_row))


def test_tiebreak_by_smallest_facet_id_is_order_independent():
    row_a = _row(
        facet_id="200000000000000000", observation_period=None, time_period="2020", value="1"
    )
    row_b = _row(
        facet_id="1000000000000000000", observation_period=None, time_period="2020", value="2"
    )
    row_c = _row(
        facet_id="30000000000000000000", observation_period=None, time_period="2020", value="3"
    )

    _assert_table_matches_row_oracle((row_c, row_b, row_a))
    _assert_table_matches_row_oracle((row_a, row_b, row_c))


# ---------------------------------------------------------------------------
# `facet_id` tiebreak edge cases.
# ---------------------------------------------------------------------------


def test_facet_id_tiebreak_is_deterministic_for_numerically_equal_ids():
    row_1 = _row(facet_id="1", observation_period=None, time_period="2020", value="1")
    row_01 = _row(facet_id="01", observation_period=None, time_period="2020", value="2")

    _assert_table_matches_row_oracle((row_1, row_01))
    _assert_table_matches_row_oracle((row_01, row_1))


def test_non_decimal_facet_id_does_not_raise():
    weird_row = _row(facet_id="²", observation_period=None, time_period="2020", value="1")
    other_row = _row(facet_id="9", observation_period=None, time_period="2020", value="2")

    _assert_table_matches_row_oracle((weird_row, other_row))


def test_huge_facet_id_does_not_raise():
    huge_id = "9" * 4301
    huge_row = _row(facet_id=huge_id, observation_period=None, time_period="2020", value="1")
    other_row = _row(facet_id="9", observation_period=None, time_period="2020", value="2")

    _assert_table_matches_row_oracle((huge_row, other_row))


def test_bare_and_dcid_prefixed_variable_collapse_via_the_canonicalised_grain():
    bare_row = SdmxObservation(
        variable="Count_Person_Female",  # the mixer's bare on-the-wire spelling
        role_values={"observationAbout": "country/KEN"},
        unit=None,
        measurement_method=None,
        observation_period="P1Y",
        provenance="dc/base/Kenya_Census",
        time_period="2020",
        value="1",
        scaling_factor=None,
        facet_id="f-bare",
    )
    prefixed_row = SdmxObservation(
        variable=_STAT_VAR,  # already dcid:-prefixed
        role_values={"observationAbout": "country/KEN"},
        unit=None,
        measurement_method=None,
        observation_period=None,
        provenance="dc/base/Kenya_Census",
        time_period="2020",
        value="1",
        scaling_factor=None,
        facet_id="f-prefixed",
    )

    _assert_table_matches_row_oracle((bare_row, prefixed_row))


def test_facets_differing_only_on_an_undeclared_role_collapse():
    row_a = SdmxObservation(
        variable=_STAT_VAR,
        role_values={"observationAbout": "country/KEN", "donor": "country/USA"},
        unit=None,
        measurement_method=None,
        observation_period="P1Y",
        provenance="dc/base/Kenya_Census",
        time_period="2020",
        value="1",
        scaling_factor=None,
        facet_id="f-a",
    )
    row_b = SdmxObservation(
        variable=_STAT_VAR,
        role_values={"observationAbout": "country/KEN", "donor": "country/GBR"},
        unit=None,
        measurement_method=None,
        observation_period=None,
        provenance="dc/base/Kenya_Census",
        time_period="2020",
        value="1",
        scaling_factor=None,
        facet_id="f-b",
    )

    _assert_table_matches_row_oracle((row_a, row_b))


# ---------------------------------------------------------------------------
# Early exit: the table function returns the identical table object when at
# most one distinct facetId is present (zero rows, one real facet, or one
# null pseudo-facet), matching the row function's own identity return.
# ---------------------------------------------------------------------------


def test_table_early_exit_returns_the_identical_object_for_zero_rows():
    table = _table(())

    assert collapse_duplicate_identity_facets_table(table, declared_roles=_ROLES) is table


def test_table_early_exit_returns_the_identical_object_for_one_real_facet():
    rows = (
        _row(facet_id="f-1", observation_period="P1Y", time_period="2020", value="1"),
        _row(facet_id="f-1", observation_period=None, time_period="2021", value="2"),
    )
    table = _table(rows)

    assert collapse_duplicate_identity_facets_table(table, declared_roles=_ROLES) is table


def test_table_early_exit_returns_the_identical_object_for_one_null_facet():
    rows = (
        _row(facet_id=None, observation_period="P1Y", time_period="2020", value="1"),
        _row(facet_id=None, observation_period=None, time_period="2021", value="2"),
    )
    table = _table(rows)

    assert collapse_duplicate_identity_facets_table(table, declared_roles=_ROLES) is table


# ---------------------------------------------------------------------------
# Seeded differential test: the row and table paths agree over many
# randomly generated inputs designed to force facet contests, NFC variants,
# variable spelling variants, and undeclared roles.
# ---------------------------------------------------------------------------

_FACET_ID_POOL: tuple[str | None, ...] = (
    None,
    "1",
    "2",
    "42",
    "300000000000000000",
    "7825127397789950237",
)
_ENTITIES = ("country/KEN", "country/TGO", "country/USA")
_PERIODS = ("2015", "2016", "2017", "2018")
_CAFE_NFC = unicodedata.normalize("NFC", "café")
_CAFE_NFD = unicodedata.normalize("NFD", "café")
_UNITS: tuple[str | None, ...] = (None, "USD", "KES", _CAFE_NFC, _CAFE_NFD)
_METHODS: tuple[str | None, ...] = (None, "CensusPWID", "WorldBankEstimate")
_PROVENANCES: tuple[str | None, ...] = (None, "dc/base/Kenya_Census", "WorldBank")
_VARIABLE_SPELLINGS = ("Count_Person_Female", "dcid:Count_Person_Female")


def _random_row(rng: random.Random, facet_ids: tuple[str | None, ...]) -> SdmxObservation:
    role_values = {"observationAbout": rng.choice(_ENTITIES)}
    if rng.random() < 0.3:
        role_values["donor"] = rng.choice(_ENTITIES)  # undeclared: plays no part in the grain
    return SdmxObservation(
        variable=rng.choice(_VARIABLE_SPELLINGS),
        role_values=role_values,
        unit=rng.choice(_UNITS),
        measurement_method=rng.choice(_METHODS),
        observation_period=rng.choice((None, "P1Y")),
        provenance=rng.choice(_PROVENANCES),
        time_period=rng.choice(_PERIODS),
        value=str(rng.random()),
        scaling_factor=None,
        facet_id=rng.choice(facet_ids),
    )


def test_differential_against_the_row_path():
    rng = random.Random(20260904)
    for _ in range(200):
        facet_ids = tuple(rng.sample(_FACET_ID_POOL, k=rng.randint(2, 4)))
        rows = tuple(_random_row(rng, facet_ids) for _ in range(rng.randint(4, 20)))

        _assert_table_matches_row_oracle(rows)
