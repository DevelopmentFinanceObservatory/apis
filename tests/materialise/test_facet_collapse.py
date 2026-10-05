"""Facet collapse resolves only duplicate `Frame.build` grain keys.
Disjoint periods, entities, or facet identities remain separate.
"""

from __future__ import annotations

import unicodedata

import pytest
from dfo_frame_grammar import FrameError

from dc_sdmx import SdmxObservation
from materialise import columnar
from materialise.facet_collapse import _facet_id_sort_key, collapse_duplicate_identity_facets_table
from materialise.frame_build import build_frame_from_table

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


def _table(rows):
    return table_from_observations(rows, roles=_ROLES)


def _collapse(rows):
    return collapse_duplicate_identity_facets_table(_table(rows), declared_roles=_ROLES)


def _build(table):
    return build_frame_from_table(table, role_keys=_ROLES, stat_var=_STAT_VAR)


# ---------------------------------------------------------------------------
# A genuine grain collision resolves to one row, and Frame.build accepts the
# result.
# ---------------------------------------------------------------------------


def test_kenya_shape_collapses_to_the_period_annotated_facet_and_frame_builds():
    # `8600...` sorts after `7825...`. observationPeriod outranks facet id.
    period_row = _row(
        facet_id="8600821667449073091",
        observation_period="P1Y",
        time_period="2019",
        value="24014716.0",
    )
    periodless_row = _row(
        facet_id="7825127397789950237",
        observation_period=None,
        time_period="2019",
        value="24014716.0",
    )
    rows = (period_row, periodless_row)

    # Uncollapsed, the two facets are downstream-identical. Frame.build
    # refuses the grain collision this resolution works around.
    with pytest.raises(FrameError):
        _build(_table(rows))

    collapsed = _collapse(rows)

    assert collapsed.equals(_table((period_row,)))
    frame = _build(collapsed)
    assert len(frame.rows) == 1
    coverage = columnar.coverage(collapsed, declared_roles=_ROLES, applied_window=None)
    assert coverage.rows == 1


def test_genuine_duplicate_within_one_facet_raises_frame_error():
    # Duplicate wire rows share one facet. `Frame.build` rejects them.
    row = _row(
        facet_id="7825127397789950237",
        observation_period="P1Y",
        time_period="2019",
        value="24014716.0",
    )
    duplicate = _row(
        facet_id="7825127397789950237",
        observation_period="P1Y",
        time_period="2019",
        value="24014716.0",
    )
    rows = (row, duplicate)

    collapsed = _collapse(rows)

    assert collapsed.equals(_table(rows))
    with pytest.raises(FrameError):
        _build(collapsed)


def test_same_facet_id_role_dropped_pair_still_raises_frame_error():
    # Rows from one facet pass through unchanged. `Frame.build` rejects the
    # omitted required role.
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
    rows = (present_row, omitted_row)

    collapsed = _collapse(rows)

    assert collapsed.equals(_table(rows))
    with pytest.raises(FrameError):
        _build(collapsed)


# ---------------------------------------------------------------------------
# Facets with one identity survive when their grain keys differ.
# ---------------------------------------------------------------------------


def test_facets_on_the_same_identity_with_disjoint_years_both_survive():
    facet_2015 = _row(facet_id="f-2015", observation_period=None, time_period="2015", value="1000")
    facet_2016 = _row(facet_id="f-2016", observation_period=None, time_period="2016", value="2000")

    assert _collapse((facet_2015, facet_2016)).equals(_table((facet_2015, facet_2016)))


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

    assert _collapse((facet_ken, facet_tgo)).equals(_table((facet_ken, facet_tgo)))


def test_facets_differing_in_unit_are_both_kept():
    usd_row = _row(
        facet_id="f-usd", observation_period="P1Y", time_period="2020", value="10", unit="USD"
    )
    kes_row = _row(
        facet_id="f-kes", observation_period="P1Y", time_period="2020", value="10", unit="KES"
    )

    assert _collapse((usd_row, kes_row)).equals(_table((usd_row, kes_row)))


def test_facets_differing_in_measurement_method_are_both_kept():
    census_row = _row(
        facet_id="f-census",
        observation_period="P1Y",
        time_period="2020",
        value="10",
        measurement_method="CensusPWID",
    )
    estimate_row = _row(
        facet_id="f-estimate",
        observation_period="P1Y",
        time_period="2020",
        value="12",
        measurement_method="WorldBankEstimate",
    )

    assert _collapse((census_row, estimate_row)).equals(_table((census_row, estimate_row)))


def test_facets_differing_in_provenance_are_both_kept():
    kenya_row = _row(
        facet_id="f-kenya-census",
        observation_period="P1Y",
        time_period="2020",
        value="10",
        provenance="dc/base/Kenya_Census",
    )
    worldbank_row = _row(
        facet_id="f-worldbank",
        observation_period="P1Y",
        time_period="2020",
        value="11",
        provenance="WorldBank",
    )

    assert _collapse((kenya_row, worldbank_row)).equals(_table((kenya_row, worldbank_row)))


def test_identity_equal_facets_with_partial_period_overlap_lose_only_the_contested_row():
    # They contest 2015. f-wide wins by row count (3 vs 2), and
    # f-narrow's uncontested 2016 row survives.
    wide_rows = tuple(
        _row(facet_id="f-wide", observation_period=None, time_period=year, value="10")
        for year in ("2015", "2018", "2020")
    )
    narrow_2015 = _row(facet_id="f-narrow", observation_period=None, time_period="2015", value="99")
    narrow_2016 = _row(facet_id="f-narrow", observation_period=None, time_period="2016", value="98")

    collapsed = _collapse((*wide_rows, narrow_2015, narrow_2016))

    assert collapsed.equals(_table((*wide_rows, narrow_2016)))


# ---------------------------------------------------------------------------
# Contests resolve per grain key. Uncontested rows survive.
# ---------------------------------------------------------------------------


def test_partial_overlap_drops_only_the_losing_row_at_the_contested_key():
    # P and Q contest 2015. Q wins by row count (11 vs 6). P's 2001-2005
    # rows survive because their grain keys are uncontested.
    p_rows = tuple(
        _row(facet_id="P", observation_period=None, time_period=year, value="1")
        for year in ("2015", "2001", "2002", "2003", "2004", "2005")
    )
    q_rows = tuple(
        _row(facet_id="Q", observation_period=None, time_period=year, value="2")
        for year in ["2015", *[str(y) for y in range(2020, 2030)]]
    )

    collapsed = _collapse(p_rows + q_rows)

    assert collapsed.num_rows == len(p_rows) + len(q_rows) - 1
    # p_rows[0] is P's 2015 row, the loser.
    assert collapsed.equals(_table(p_rows[1:] + q_rows))


def test_two_independent_contested_keys_resolve_independently():
    # A/B contest 2015 and C/D contest 2020. Each key resolves independently.
    a_row = _row(facet_id="A", observation_period="P1Y", time_period="2015", value="1")
    b_row = _row(facet_id="B", observation_period=None, time_period="2015", value="2")
    c_row = _row(facet_id="C", observation_period="P1Y", time_period="2020", value="3")
    d_row = _row(facet_id="D", observation_period=None, time_period="2020", value="4")

    collapsed = _collapse((a_row, b_row, c_row, d_row))

    assert collapsed.equals(_table((a_row, c_row)))


def test_transitive_chain_does_not_merge_facets_that_never_collided():
    # B bridges A's 2015 row and C's 2020 row. A and C share no period, so
    # B loses both contested rows while A and C keep theirs.
    a_rows = tuple(
        _row(facet_id="A", observation_period=None, time_period=year, value="1")
        for year in ("2001", "2002", "2003", "2004", "2005", "2015")
    )
    b_rows = (
        _row(facet_id="B", observation_period=None, time_period="2015", value="2"),
        _row(facet_id="B", observation_period=None, time_period="2020", value="3"),
    )
    c_rows = tuple(
        _row(facet_id="C", observation_period=None, time_period=year, value="4")
        for year in ("2020", "2030", "2031", "2032", "2033", "2034")
    )

    forward = _collapse(a_rows + b_rows + c_rows)
    backward = _collapse(c_rows + b_rows + a_rows)

    assert forward.equals(_table(a_rows + c_rows))
    assert backward.equals(_table(c_rows + a_rows))
    assert set(forward["facetId"].to_pylist()) == {"A", "C"}


# ---------------------------------------------------------------------------
# Each ranking tier isolated: only the named tier decides the winner, so
# deleting that tier flips the result and fails the test.
# ---------------------------------------------------------------------------


def test_only_the_observation_period_tier_decides():
    # Row count and span tie. Facet ids rank against the expected winner.
    period_row = _row(facet_id="B-period", observation_period="P1Y", time_period="2020", value="1")
    periodless_row = _row(
        facet_id="A-periodless", observation_period=None, time_period="2020", value="2"
    )

    assert _collapse((period_row, periodless_row)).equals(_table((period_row,)))


def test_only_the_row_count_tier_decides():
    # Period presence and span tie. Both contest 2018 and 2020, so the
    # three-row facet wins by count (3 vs 2).
    many_rows = tuple(
        _row(facet_id="z-many", observation_period=None, time_period=year, value="1")
        for year in ("2018", "2019", "2020")
    )
    few_rows = tuple(
        _row(facet_id="a-few", observation_period=None, time_period=year, value="2")
        for year in ("2018", "2020")
    )

    assert _collapse(many_rows + few_rows).equals(_table(many_rows))


def test_only_the_span_tier_decides():
    # Both facets have 2 rows and `observation_period=None`. They contest 2020.
    # Span selects wide. 2010 and 2021 are uncontested.
    wide_2010 = _row(facet_id="z-wide", observation_period=None, time_period="2010", value="1")
    wide_2020 = _row(facet_id="z-wide", observation_period=None, time_period="2020", value="1")
    narrow_2020 = _row(facet_id="a-narrow", observation_period=None, time_period="2020", value="2")
    narrow_2021 = _row(facet_id="a-narrow", observation_period=None, time_period="2021", value="2")

    collapsed = _collapse((wide_2010, wide_2020, narrow_2020, narrow_2021))

    assert collapsed.equals(_table((wide_2010, wide_2020, narrow_2021)))


def test_real_facet_id_always_beats_the_none_pseudo_group():
    # The unkeyed row has an observation period, so real facet ids rank first.
    real_row = _row(facet_id="42", observation_period=None, time_period="2020", value="1")
    unkeyed_row = _row(facet_id=None, observation_period="P1Y", time_period="2020", value="2")

    assert _collapse((real_row, unkeyed_row)).equals(_table((real_row,)))


def test_tiebreak_by_smallest_facet_id_is_order_independent():
    # Scores tie, leaving only facet id. The 18-, 19-, and 20-digit ids
    # sort lexically opposite to numeric order.
    row_a = _row(
        facet_id="200000000000000000",  # 18 digits, numerically smallest
        observation_period=None,
        time_period="2020",
        value="1",
    )
    row_b = _row(
        facet_id="1000000000000000000",  # 19 digits
        observation_period=None,
        time_period="2020",
        value="2",
    )
    row_c = _row(
        facet_id="30000000000000000000",  # 20 digits, numerically largest
        observation_period=None,
        time_period="2020",
        value="3",
    )

    forward = _collapse((row_c, row_b, row_a))
    backward = _collapse((row_a, row_b, row_c))

    assert forward.equals(_table((row_a,)))
    assert backward.equals(_table((row_a,)))


# ---------------------------------------------------------------------------
# Facet-id ordering handles equal numeric values and unusual strings.
# ---------------------------------------------------------------------------


def test_facet_id_tiebreak_is_deterministic_for_numerically_equal_ids():
    # "1" and "01" have equal numeric value, so lexical order breaks the tie.
    row_1 = _row(facet_id="1", observation_period=None, time_period="2020", value="1")
    row_01 = _row(facet_id="01", observation_period=None, time_period="2020", value="2")

    forward = _collapse((row_1, row_01))
    backward = _collapse((row_01, row_1))

    assert forward.equals(_table((row_01,)))
    assert backward.equals(_table((row_01,)))


def test_non_decimal_facet_id_does_not_raise():
    # Superscript ² passes `isdigit` but fails `isdecimal` and `int`. 9 wins.
    weird_row = _row(facet_id="²", observation_period=None, time_period="2020", value="1")
    other_row = _row(facet_id="9", observation_period=None, time_period="2020", value="2")

    assert _collapse((weird_row, other_row)).equals(_table((other_row,)))


def test_huge_facet_id_does_not_raise():
    # CPython 3.11 rejects `int()` inputs over 4,300 digits. This 4,301-digit
    # id falls back to string ordering.
    huge_id = "9" * 4301
    huge_row = _row(facet_id=huge_id, observation_period=None, time_period="2020", value="1")
    other_row = _row(facet_id="9", observation_period=None, time_period="2020", value="2")

    assert _collapse((huge_row, other_row)).equals(_table((other_row,)))


def test_facet_id_sort_key_handles_none_and_oversized_ids_without_raising():
    assert _facet_id_sort_key(None) == (1, "")
    assert _facet_id_sort_key("²") == (1, "²")
    assert _facet_id_sort_key("9" * 4301) == (1, "9" * 4301)
    assert _facet_id_sort_key("42") == (0, 42, "42")
    # A real 19-digit Data Commons facet id sorts numerically.
    assert _facet_id_sort_key("7825127397789950237") == (
        0,
        7825127397789950237,
        "7825127397789950237",
    )


# ---------------------------------------------------------------------------
# Grain-key normalisation matches Frame.build's own view: NFC form, the
# canonical `dcid:`-prefixed stat_var spelling, and declared roles only.
# ---------------------------------------------------------------------------


def test_nfc_variant_entity_dcids_collapse_via_the_normalised_grain():
    nfc_form = unicodedata.normalize("NFC", "café")
    nfd_form = unicodedata.normalize("NFD", "café")
    assert nfc_form != nfd_form  # sanity: genuinely different raw strings

    nfc_row = _row(
        facet_id="f-nfc",
        observation_period="P1Y",
        time_period="2020",
        value="1",
        entity=f"country/{nfc_form}",
    )
    nfd_row = _row(
        facet_id="f-nfd",
        observation_period=None,
        time_period="2020",
        value="2",
        entity=f"country/{nfd_form}",
    )
    rows = (nfc_row, nfd_row)

    # Frame normalises both raw dcids to NFC, creating a grain collision.
    with pytest.raises(FrameError):
        _build(_table(rows))

    collapsed = _collapse(rows)

    assert collapsed.equals(_table((nfc_row,)))
    _build(collapsed)


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
    rows = (bare_row, prefixed_row)

    # Both variable spellings canonicalise to the same `dcid:` form before
    # `Frame.build`, creating a grain collision.
    with pytest.raises(FrameError):
        _build(_table(rows))

    collapsed = _collapse(rows)

    assert collapsed.equals(_table((bare_row,)))
    _build(collapsed)


def test_facets_differing_only_on_an_undeclared_role_collapse():
    # `donor` is outside `_ROLES`, so it is absent from the frame grain.
    # Real generation rejects undeclared SDMX roles before row parsing.
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

    assert _collapse((row_a, row_b)).equals(_table((row_a,)))
