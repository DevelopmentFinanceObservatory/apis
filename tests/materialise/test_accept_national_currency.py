"""Acceptance cases for the unit binding that serves each place its national currency (#1273).

`scope.facets.unit` binds `{"national_currency": {"role": "observationAbout"}}`, which
`materialise` holds as `FacetBinding(state="national_currency",
national_currency=NationalCurrency(role="observationAbout"))`. At each place the role binds,
the read serves the facet whose unit is the one currency the graph's `nationalCurrencyOf`
names for that place, within one provenance and one measurement method. Nothing is converted.

A place the graph maps to no currency or to several, and a mapped place holding no facet in
its own currency, is a coverage gap (P34): it serves no row, it is never converted, and the
mint's trace names it in `ManifestTrace.coverage_gaps`, each entry carrying its `role` and its
`entities`.

The data is WEO nominal GDP as the dev graph holds it (`_weo_recorded`): France in euros,
Kenya in shillings, Zimbabwe in Zimbabwe gold and Ecuador, whose national currency is the US
dollar, in US dollars. The three gap places are constructed, since every place holding WEO GDP
on dev maps to exactly one currency and holds a facet in it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.pipeline import MaterialisationResult, PreviewResult
from materialise.reshape import RollupNotSummableError, classify_columns, groupby_sum
from materialise.window import RequestedWindow

from ._weo_recorded import (
    ECUADOR,
    FRANCE,
    KENYA,
    WEO,
    ZIMBABWE,
    facet,
    facet_id,
    mint,
    preview,
    recorded_currencies,
    recorded_facets,
    recorded_values,
    rows,
)
from .conftest import only_chunk

_REPORTED = RequestedWindow(kind="range", start_year=2016, end_year=2024)
_LOCAL = {
    FRANCE: "CurrencyEUR",
    KENYA: "CurrencyKES",
    ZIMBABWE: "CurrencyZWG",
    ECUADOR: "CurrencyUSD",
}
"""Each recorded place's national currency, as `nationalCurrencyOf` names it on dev."""

_UNNAMED = "country/XNA"
"""A constructed place no currency names."""

_DOUBLY_NAMED = "country/XNB"
"""A constructed place two currencies name."""

_UNHELD = "country/XNC"
"""A constructed place the shilling names, holding Kenya's facets but none in shillings."""


def _national_currency() -> dict[str, FacetBinding]:
    from materialise.facets import NationalCurrency

    return {
        "unit": FacetBinding(
            state="national_currency",
            national_currency=NationalCurrency(role="observationAbout"),
        )
    }


def _served_units(result: PreviewResult | MaterialisationResult) -> dict[str, set[str]]:
    units: dict[str, set[str]] = {}
    for row in rows(result):
        units.setdefault(str(row["observationAbout"]), set()).add(str(row["unit"]))
    return units


def _with_a_second_source_in_euros() -> tuple[dict[str, list[Facet]], dict]:
    """A second source holding France's euro series alone, with five more observations than
    the IMF's: its one facet outranks any one of the IMF's, each of which holds one place."""
    facets_by_place = recorded_facets()
    values = recorded_values()
    imf = facet(FRANCE, unit="CurrencyEUR", method=None)
    other = replace(
        imf,
        facet_id="facetSecondSourceEUR",
        provenance_id="provenance/SecondSource",
        obs_count=imf.obs_count + 5,
        earliest_date="2000",
    )
    facets_by_place[FRANCE] = [*facets_by_place[FRANCE], other]
    values[(FRANCE, "facetSecondSourceEUR")] = {str(y): 1.0 for y in range(2000, 2025)}
    return facets_by_place, values


@pytest.mark.parametrize(
    ("places", "second_source"),
    [
        pytest.param((FRANCE, KENYA), False, id="france-and-kenya"),
        pytest.param((FRANCE, KENYA, ZIMBABWE, ECUADOR), False, id="a-place-whose-currency-is-usd"),
        # The IMF's facets together serve all four places, so the provenance holding the
        # most places leads, though the second source's one facet outranks each of them.
        pytest.param(
            (FRANCE, KENYA, ZIMBABWE, ECUADOR), True, id="a-second-source-in-one-currency"
        ),
    ],
)
def test_each_place_is_served_in_its_own_currency_from_one_provenance(places, second_source):
    """France in euros and Kenya in shillings from the IMF, each value the place's own
    recorded series, with one provenance and one method across the places."""
    facets_by_place, values = _with_a_second_source_in_euros() if second_source else (None, None)
    result = preview(
        places,
        window=_REPORTED,
        facet_bindings=_national_currency(),
        facets_by_place=facets_by_place,
        values=values,
    )

    assert _served_units(result) == {place: {_LOCAL[place]} for place in places}
    assert {row["provenance"] for row in rows(result)} == {WEO}
    assert {row["measurementMethod"] for row in rows(result)} == {None}
    values = recorded_values()
    for row in rows(result):
        place, year = str(row["observationAbout"]), str(row["time_period"])
        native = facet_id(place, unit=_LOCAL[place], method=None)
        assert row["value"] == values[(place, native)][year]
    assert result.coverage.entity_refs["observationAbout"] == frozenset(places)


def _with_constructed_places() -> tuple[dict[str, list[Facet]], dict, dict[str, tuple[str, ...]]]:
    """Two constructed places holding France's facets and values, one no currency names and
    one both the euro and the shilling name; and Kenya without its shilling facets, so that
    its one currency has no facet."""
    facets_by_place = recorded_facets()
    values = recorded_values()
    for place in (_UNNAMED, _DOUBLY_NAMED):
        facets_by_place[place] = list(facets_by_place[FRANCE])
        for candidate in facets_by_place[FRANCE]:
            assert candidate.facet_id is not None
            values[(place, candidate.facet_id)] = dict(values[(FRANCE, candidate.facet_id)])
    facets_by_place[_UNHELD] = [
        candidate for candidate in facets_by_place[KENYA] if candidate.unit != "CurrencyKES"
    ]
    for candidate in facets_by_place[_UNHELD]:
        assert candidate.facet_id is not None
        values[(_UNHELD, candidate.facet_id)] = dict(values[(KENYA, candidate.facet_id)])
    currencies = {
        **recorded_currencies(),
        _DOUBLY_NAMED: ("CurrencyEUR", "CurrencyKES"),
        _UNHELD: ("CurrencyKES",),
    }
    return facets_by_place, values, currencies


@pytest.mark.parametrize(
    ("places", "left_out"),
    [
        pytest.param(
            (FRANCE, KENYA, _UNNAMED),
            {_UNNAMED: "no_national_currency"},
            id="no-currency-names-it",
        ),
        pytest.param(
            (FRANCE, KENYA, _DOUBLY_NAMED),
            {_DOUBLY_NAMED: "several_national_currencies"},
            id="two-currencies-name-it",
        ),
        pytest.param(
            (FRANCE, KENYA, _UNHELD),
            {_UNHELD: "no_facet_in_national_currency"},
            id="no-facet-in-its-own-currency",
        ),
        pytest.param(
            (FRANCE, KENYA, _UNNAMED, _UNHELD),
            {_UNNAMED: "no_national_currency", _UNHELD: "no_facet_in_national_currency"},
            id="two-gaps-of-two-kinds",
        ),
        pytest.param((FRANCE, KENYA), {}, id="control-every-place-mapped"),
    ],
)
def test_a_place_without_a_facet_in_one_national_currency_is_a_coverage_gap(places, left_out):
    """The other places are served in their own currencies. A place left out serves no row,
    is never converted, and the trace names it with the reason it was left out."""
    facets_by_place, values, currencies = _with_constructed_places()

    result = mint(
        places,
        window=_REPORTED,
        facet_bindings=_national_currency(),
        facets_by_place=facets_by_place,
        values=values,
        currencies=currencies,
    )

    gaps = result.trace.coverage_gaps
    named = {
        entity: gap["reason"]
        for gap in gaps
        if gap["role"] == "observationAbout"
        for entity in gap["entities"]
    }
    assert named == left_out
    served = set(places) - set(left_out)
    assert _served_units(result) == {place: {_LOCAL[place]} for place in served}
    assert result.coverage.entity_refs["observationAbout"] == frozenset(served)
    assert result.conversion_record is None
    assert result.trace.conversion is None


def test_each_series_names_its_unit_and_no_total_is_offered_across_them():
    """The mint's figures hold one series per currency, which the description reads to name
    each series' unit, and a sum across places refuses, since the unit differs within the
    group it would form."""
    places = (FRANCE, KENYA, ZIMBABWE, ECUADOR)
    result = mint(places, window=_REPORTED, facet_bindings=_national_currency())

    by_unit = {
        (figure.source, figure.unit): figure.series
        for figure in result.series_statistics.figures.by_source_unit
    }
    assert by_unit == {(WEO, _LOCAL[place]): 1 for place in places}

    frame = only_chunk(result)
    with pytest.raises(RollupNotSummableError) as refused:
        groupby_sum(
            frame,
            axis="observationAbout",
            classification=classify_columns(frame.columns),
            summable_axes=frozenset({"observationAbout"}),
        )
    assert (refused.value.code, refused.value.precondition) == (
        "rollup_not_summable",
        "facet_homogeneity",
    )
