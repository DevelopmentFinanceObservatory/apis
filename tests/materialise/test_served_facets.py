"""`served_facets` names the facets materialise dispatches for an unbound member: the ids
`joint_dispatch_facet_ids` takes over the selections `_select_member_facets` makes, so
search coverage dates and counts the same group the preview reads."""

from __future__ import annotations

import pytest

from materialise import MaterialisationRequest
from materialise.errors import FacetNotAvailableError
from materialise.facets import (
    FacetBinding,
    admitted_candidates,
    joint_dispatch_facet_ids,
    served_facets,
)
from materialise.generation import _select_member_facets
from materialise.graph import Facet
from materialise.window import RequestedWindow

from .conftest import FakeGraphReader

_VARIABLE = "Count_Person"


def _facet(
    facet_id: str,
    *,
    provenance: str = "dc/base/Import",
    method: str | None = None,
    unit: str | None = None,
    obs_count: int,
    years: tuple[str, str],
) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        measurement_method=method,
        unit=unit,
        obs_count=obs_count,
        earliest_date=years[0],
        latest_date=years[1],
    )


# A long recorded facet the ranking elects, and a short facet reaching 2026.
_LONG = _facet("f-long", obs_count=55, years=("1970", "2024"))
_SAME_IMPORT = _facet("f-short", obs_count=7, years=("2020", "2026"))
_OTHER_IMPORT = _facet("f-other", provenance="dc/base/Other", obs_count=7, years=("2020", "2026"))
_OTHER_METHOD = _facet("f-est", method="Estimated", obs_count=7, years=("2020", "2026"))
_OTHER_UNIT = _facet("f-eur", unit="EUR", obs_count=7, years=("2020", "2026"))

_CASES = {
    "one import": [_LONG, _SAME_IMPORT],
    "another import": [_LONG, _OTHER_IMPORT],
    "another method": [_LONG, _OTHER_METHOD],
    "another unit": [_LONG, _OTHER_UNIT],
    "every kind": [_LONG, _SAME_IMPORT, _OTHER_IMPORT, _OTHER_METHOD, _OTHER_UNIT],
}
_WINDOWS = {
    "all": RequestedWindow(kind="all"),
    "latest": RequestedWindow(kind="latest"),
    "2025 on": RequestedWindow(kind="range", start_year=2025, end_year=2030),
}


@pytest.mark.parametrize("window_name", list(_WINDOWS))
@pytest.mark.parametrize("case", list(_CASES))
def test_served_facets_are_the_joint_dispatch_group(case: str, window_name: str) -> None:
    candidates = _CASES[case]
    window = _WINDOWS[window_name]
    request = MaterialisationRequest(
        stat_var=f"dcid:{_VARIABLE}",
        role_bindings={"observationAbout": ["country/KEN"]},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=window,
    )
    election = _select_member_facets(request, candidates, graph=FakeGraphReader())

    served = served_facets(candidates, stat_vars=[f"dcid:{_VARIABLE}"], window=window)

    assert tuple(sorted(f.facet_id for f in served if f.facet_id)) == joint_dispatch_facet_ids(
        candidates, election, member=0
    )


def test_one_import_serves_both_facets() -> None:
    served = served_facets(
        [_LONG, _SAME_IMPORT], stat_vars=[_VARIABLE], window=RequestedWindow(kind="latest")
    )
    assert [f.facet_id for f in served] == ["f-long", "f-short"]


def test_a_candidate_filter_narrows_the_pool_before_election() -> None:
    """A caller's filter admitting the other method leaves `f-est` to elect, where the
    unfiltered ranking elects `f-long`."""
    request = MaterialisationRequest(
        stat_var=f"dcid:{_VARIABLE}",
        role_bindings={"observationAbout": ["country/KEN"]},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=RequestedWindow(kind="all"),
        candidate_filter=lambda facet: facet.measurement_method == "Estimated",
    )
    candidates = admitted_candidates([_LONG, _OTHER_METHOD], request.candidate_filter)

    election = _select_member_facets(request, candidates, graph=FakeGraphReader())

    assert joint_dispatch_facet_ids(candidates, election, member=0) == ("f-est",)


def test_a_candidate_filter_that_admits_nothing_is_a_facet_refusal() -> None:
    with pytest.raises(FacetNotAvailableError):
        admitted_candidates([_LONG], lambda facet: False)
    assert admitted_candidates([], lambda facet: False) == []


_RANGE = RequestedWindow(kind="range", start_year=2016, end_year=2030)


def _estimated(facet_id: str, **overrides: object) -> Facet:
    """An estimated facet of `_LONG`'s import reaching 2030, with `overrides` applied."""
    fields: dict = {"method": "Estimated", "obs_count": 7, "years": ("2025", "2030")} | overrides
    return _facet(facet_id, **fields)


@pytest.mark.parametrize(
    ("candidates", "window", "bindings", "continued_by", "dispatched"),
    [
        # The window runs past the elected series, and its import holds estimates in its unit.
        pytest.param(
            [_LONG, _estimated("f-est")],
            _RANGE,
            {},
            "Estimated",
            ("f-est", "f-long"),
            id="estimates-continue-the-series",
        ),
        # The estimates cover more of the window than the series they continue, and still
        # rank behind it as one candidate.
        pytest.param(
            [_estimated("f-est"), _LONG],
            RequestedWindow(kind="range", start_year=2022, end_year=2030),
            {},
            "Estimated",
            ("f-est", "f-long"),
            id="estimates-covering-more-years-follow-the-series",
        ),
        # A method the classification names estimated continues it too.
        pytest.param(
            [_LONG, _estimated("f-mod", method="Modelled")],
            _RANGE,
            {},
            "Modelled",
            ("f-long", "f-mod"),
            id="any-estimated-class-continues",
        ),
        pytest.param([_LONG], _RANGE, {}, None, ("f-long",), id="control-no-estimated-facet"),
        pytest.param(
            [_LONG, _estimated("f-est")],
            RequestedWindow(kind="range", start_year=2020, end_year=2024),
            {},
            None,
            ("f-long",),
            id="control-window-ends-within-the-series",
        ),
        pytest.param(
            [_LONG, _estimated("f-est")],
            RequestedWindow(kind="all"),
            {},
            None,
            ("f-long",),
            id="control-no-window",
        ),
        pytest.param(
            [_LONG, _estimated("f-est", provenance="dc/base/Other")],
            _RANGE,
            {},
            None,
            ("f-long",),
            id="control-another-import",
        ),
        pytest.param(
            [_LONG, _estimated("f-est", unit="EUR")],
            _RANGE,
            {},
            None,
            ("f-long",),
            id="control-another-unit",
        ),
        pytest.param(
            [_LONG, _estimated("f-proj", method="Projected")],
            _RANGE,
            {},
            None,
            ("f-long",),
            id="control-unclassed-method",
        ),
        pytest.param(
            [_LONG, _estimated("f-est", years=("2020", "2024"))],
            _RANGE,
            {},
            None,
            ("f-long",),
            id="control-estimates-end-with-the-series",
        ),
        pytest.param(
            [_LONG, _estimated("f-est")],
            _RANGE,
            {"measurementMethod": FacetBinding(state="bound", ref=None)},
            None,
            ("f-long",),
            id="control-bound-method",
        ),
    ],
)
def test_a_window_past_the_elected_series_takes_its_imports_estimates(
    candidates, window, bindings, continued_by, dispatched
) -> None:
    """A mint continues the elected series with the same import's estimated facets in its
    unit, and Search's group (`served_facets`) names the same facets."""
    request = MaterialisationRequest(
        stat_var=f"dcid:{_VARIABLE}",
        role_bindings={"observationAbout": ["country/KEN"]},
        declared_roles=("observationAbout",),
        facet_bindings=bindings,
        window=window,
    )

    election = _select_member_facets(request, candidates, graph=FakeGraphReader())

    assert election.selections["measurementMethod"].continued_by == continued_by
    assert joint_dispatch_facet_ids(candidates, election, member=0) == dispatched
    if not bindings:
        served = served_facets(candidates, stat_vars=[f"dcid:{_VARIABLE}"], window=window)
        assert tuple(sorted(f.facet_id for f in served if f.facet_id)) == dispatched
