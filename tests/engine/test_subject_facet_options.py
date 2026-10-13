"""Facet options for subject StatVars offer a value only when pinning it is not refused.

`facet_refusal` needs a pinned value on every resolved StatVar, so a value only some of them
carry is withheld from the options, and every offered value passes the refusal when pinned.
"""

from __future__ import annotations

from search.engine.facet_refusals import facet_refusal
from search.engine.region.facet_probe import facet_pins
from search.engine.region.option_coverage import _facet_options
from search.models import (
    FacetOptions,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetValueAbsent,
    TimeWindow,
)
from tests.engine._fake_coverage import build_coverage_index
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_SV_A = "TESTFIXTURE/SubjectFacetOptions/A"
_SV_B = "TESTFIXTURE/SubjectFacetOptions/B"
_PROVENANCE = "dc/base/TestSurvey"
_OTHER_PROVENANCE = "dc/base/TestCensus"
_USD = "USDollar"
_EMPTY_COVERAGE = build_coverage_index([], role_names={})


def _facet(
    facet_id: str,
    *,
    years: range,
    provenance: str = _PROVENANCE,
    method: str | None = None,
    unit: str | None = _USD,
) -> dict:
    return {
        "facetId": facet_id,
        "earliestDate": str(years[0]),
        "latestDate": str(years[-1]),
        "obsCount": len(years),
        "dates": [str(y) for y in years],
        "provenanceId": provenance,
        "measurementMethod": method,
        "unit": unit,
    }


# Both StatVars carry a surveyed series with no method. Only A carries an Estimated series,
# and only B a census series under its own provenance.
_GRAPH = FakeGraph(
    obs={
        f"{_SV_A}|{_KEN}": [
            _facet("a-survey", years=range(2010, 2021)),
            _facet("a-est", years=range(2015, 2025), method="Estimated"),
        ],
        f"{_SV_B}|{_KEN}": [
            _facet("b-survey", years=range(2012, 2021)),
            _facet("b-census", years=range(2010, 2021), provenance=_OTHER_PROVENANCE),
        ],
    }
)


def _refs(options: list[FacetOptions]) -> dict[str, list[str | None]]:
    return {
        fo.axis: [o.value.ref.ref if o.value.ref is not None else None for o in fo.options]
        for fo in options
    }


def _options(svs: list[str]) -> dict[str, list[str | None]]:
    return _refs(
        _facet_options(
            svs,
            {"observationAbout": [_KEN]},
            years=TimeWindow(start_year=2010, end_year=2024),
            graph=_GRAPH,
            coverage=_EMPTY_COVERAGE,
            deadline=None,
        )
    )


def test_one_statvar_offers_every_value_it_carries():
    assert _options([_SV_A]) == {
        "provenance": [_PROVENANCE],
        "measurementMethod": [None, "Estimated"],
        "unit": [_USD],
    }


def test_values_only_one_statvar_carries_are_withheld():
    """Estimated is only on A and the census provenance only on B, so pinning either is
    refused. The method axis would hold only the ref-less option, which is no choice."""
    assert _options([_SV_A, _SV_B]) == {"provenance": [_PROVENANCE], "unit": [_USD]}


def test_an_option_is_offered_exactly_when_its_pin_is_not_refused():
    offered = {(axis, ref) for axis, refs in _options([_SV_A, _SV_B]).items() for ref in refs}
    carried = {
        ("provenance", _PROVENANCE),
        ("provenance", _OTHER_PROVENANCE),
        ("measurementMethod", "Estimated"),
        ("measurementMethod", None),
        ("unit", _USD),
    }

    refused = {
        (axis, ref)
        for axis, ref in carried
        if facet_refusal(
            _GRAPH,
            [_SV_A, _SV_B],
            [_KEN],
            {
                axis: IdentityFacetValueAbsent(ref=None)
                if ref is None
                else IdentityBindingValue(ref=ref)
            },
            deadline=None,
        )
        is not None
    }

    assert refused == {("provenance", _OTHER_PROVENANCE), ("measurementMethod", "Estimated")}
    assert offered == carried - refused - {("measurementMethod", None)}


_SV_C = "TESTFIXTURE/SubjectFacetOptions/C"
_OTHER_METHOD_UNIT = "Euro"

# One StatVar: a surveyed series and a census under the survey provenance, and an Estimated
# series under the census provenance. A unit-set pin needs each unit under the pinned
# method: method M reports in both units, method N in US dollars only.
_PINNED_GRAPH = FakeGraph(
    obs={
        f"{_SV_C}|{_KEN}": [
            _facet("c-survey", years=range(2010, 2021)),
            _facet("c-census", years=range(2012, 2021), method="Census"),
            _facet(
                "c-est", years=range(2015, 2025), provenance=_OTHER_PROVENANCE, method="Estimated"
            ),
            _facet("c-m-usd", years=range(2010, 2012), provenance=_OTHER_PROVENANCE, method="M"),
            _facet(
                "c-m-eur",
                years=range(2010, 2012),
                provenance=_OTHER_PROVENANCE,
                method="M",
                unit=_OTHER_METHOD_UNIT,
            ),
            _facet("c-n-usd", years=range(2010, 2012), provenance=_OTHER_PROVENANCE, method="N"),
        ]
    }
)


def _pinned_options(bindings: dict) -> dict[str, list[str | None]]:
    return _refs(
        _facet_options(
            [_SV_C],
            {"observationAbout": [_KEN]},
            years=TimeWindow(start_year=2010, end_year=2024),
            graph=_PINNED_GRAPH,
            coverage=_EMPTY_COVERAGE,
            pins=facet_pins(bindings),
            deadline=None,
        )
    )


def test_options_beside_a_provenance_pin_are_the_values_that_provenance_serves():
    """With the survey provenance pinned, Estimated, M and N are withheld: each is served only
    under the census provenance. The provenance axis still lists both sources, since a
    provenance option replaces the pin."""
    assert _pinned_options({"provenance": IdentityBindingValue(ref=_PROVENANCE)}) == {
        "provenance": [_PROVENANCE, _OTHER_PROVENANCE],
        "measurementMethod": [None, "Census"],
        "unit": [_USD],
    }


def test_a_method_beside_a_unit_set_pin_needs_every_unit_in_the_set():
    """A unit set pin needs a facet for each member under the pinned method. M reports in
    both units, N only in US dollars, so N is withheld though a single N facet carries a unit
    in the set."""
    options = _pinned_options({"unit": IdentityBindingSet(refs=[_USD, _OTHER_METHOD_UNIT])})

    assert "M" in options["measurementMethod"]
    assert "N" not in options["measurementMethod"]


def test_every_option_beside_a_pin_passes_the_joint_refusal():
    """Each value the single StatVar carries, pinned together with the provenance pin, is
    refused exactly when the options withhold it, apart from a lone ref-less option."""
    pin = {"provenance": IdentityBindingValue(ref=_PROVENANCE)}
    offered = {(axis, ref) for axis, refs in _pinned_options(pin).items() for ref in refs}
    carried = {
        ("provenance", _PROVENANCE),
        ("provenance", _OTHER_PROVENANCE),
        ("measurementMethod", None),
        ("measurementMethod", "Census"),
        ("measurementMethod", "Estimated"),
        ("measurementMethod", "M"),
        ("measurementMethod", "N"),
        ("unit", _USD),
        ("unit", _OTHER_METHOD_UNIT),
    }

    admitted = {
        (axis, ref)
        for axis, ref in carried
        if facet_refusal(
            _PINNED_GRAPH,
            [_SV_C],
            [_KEN],
            {
                **pin,
                axis: IdentityFacetValueAbsent(ref=None)
                if ref is None
                else IdentityBindingValue(ref=ref),
            },
            deadline=None,
        )
        is None
    }

    assert offered == admitted


_SV_D = "TESTFIXTURE/SubjectFacetOptions/D"
_WINDOW = TimeWindow(start_year=2020, end_year=2024)


def test_a_value_beside_a_pin_needs_an_in_window_point_under_that_pin():
    """The pinned survey provenance serves US dollars only before the window, and the census
    provenance serves them inside it. Pinning survey and US dollars passes the unwindowed
    refusal but serves no point in the window, so US dollars is withheld beside the pin."""
    graph = FakeGraph(
        obs={
            f"{_SV_D}|{_KEN}": [
                _facet("d-survey", years=range(2010, 2013)),
                _facet("d-census", years=range(2020, 2023), provenance=_OTHER_PROVENANCE),
            ]
        }
    )
    pin = {"provenance": IdentityBindingValue(ref=_PROVENANCE)}

    options = _refs(
        _facet_options(
            [_SV_D],
            {"observationAbout": [_KEN]},
            years=_WINDOW,
            graph=graph,
            coverage=_EMPTY_COVERAGE,
            pins=facet_pins(pin),
            deadline=None,
        )
    )

    assert options == {"provenance": [_OTHER_PROVENANCE]}
    assert (
        facet_refusal(
            graph, [_SV_D], [_KEN], {**pin, "unit": IdentityBindingValue(ref=_USD)}, deadline=None
        )
        is None
    )


def test_a_pin_is_judged_on_every_facet_in_or_out_of_the_window():
    """A carries US dollars inside the window; B carries them only before it, and euros inside
    it. The refusal judges every facet, so a US dollar pin is admitted and offered."""
    graph = FakeGraph(
        obs={
            f"{_SV_A}|{_KEN}": [_facet("a-usd", years=range(2020, 2023))],
            f"{_SV_B}|{_KEN}": [
                _facet("b-usd", years=range(2010, 2013)),
                _facet("b-eur", years=range(2020, 2023), unit="Euro"),
            ],
        }
    )

    options = _refs(
        _facet_options(
            [_SV_A, _SV_B],
            {"observationAbout": [_KEN]},
            years=_WINDOW,
            graph=graph,
            coverage=_EMPTY_COVERAGE,
            deadline=None,
        )
    )

    assert options["unit"] == [_USD]
    assert (
        facet_refusal(
            graph, [_SV_A, _SV_B], [_KEN], {"unit": IdentityBindingValue(ref=_USD)}, deadline=None
        )
        is None
    )
