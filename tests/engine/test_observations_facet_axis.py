"""Tests for per-series and resolved facet maps, plus facet-option coverage.
Multi-axis selection tests use the 18-candidate fixture in
`tests.engine._observations_doubles`.
"""

from __future__ import annotations

import pytest

from search.engine.clients.graph import Facet
from search.engine.errors import GraphInfraError
from search.engine.region.codes import (
    OPTION_COVERAGE_PROBE_FAILED,
)
from search.engine.region.facet_probe import facet_probe
from search.engine.region.facets import (
    _resolved_facets,
)
from search.engine.region.option_coverage import (
    _facet_options,
    _maybe_build_facet_options,
)
from search.engine.region.resubmit_request import _facet_filter_from_bindings
from search.engine.region.result import FacetAxisFilter
from search.models import (
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    SeriesFacetRef,
    TimeWindow,
)
from search.observations import series_facet_map
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import build_observations_planned
from tests.engine._observations_doubles import (
    _FACET_AXIS_NODES,
    _FACET_AXIS_OBS,
    _MATERIALISE_SINGLE_FACET_OBS,
    _MULTI_AXIS_SV,
    _PER_SERIES_SV_A,
    _PER_SERIES_SV_B,
    _SINGLE_FACET_SV,
    _TWO_ENTITY_SV,
    _facet_axis_graph,
    _facet_axis_materialise_double,
)
from tests.fixtures import FakeGraph

_NO_COVERAGE = empty_coverage_snapshot().index


def test_observation_series_facet_map_omits_an_axis_with_no_data():
    """With no unit value, the series omits the unit axis."""
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double(
        _MATERIALISE_SINGLE_FACET_OBS
    )
    observations, warn = build_observations_planned(
        [_SINGLE_FACET_SV],
        {"observationAbout": ["country/USA"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    series = observations.series[0]
    assert set(series.facet) == {"measurementMethod", "provenance"}
    assert series.facet["measurementMethod"].ref == "Commitment"


def test_observation_series_facet_map_provenance_agrees_with_source_refs():
    """The facet map and `source_refs` use the same provenance precedence, and both carry
    the route of the provenance's document; other axes carry none."""
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    observations, warn = build_observations_planned(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        graph=FakeGraph(nodes=_FACET_AXIS_NODES),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    series = observations.series[0]
    assert series.source_refs
    assert series.facet["provenance"].ref == series.source_refs[0].ref
    assert series.facet["provenance"].documentation == series.source_refs[0].documentation
    assert series.facet["provenance"].documentation == (
        f"/api/graph/documentation/{series.facet['provenance'].ref}"
    )
    assert series.facet["measurementMethod"].documentation is None


def test_open_axis_with_differing_data_serves_one_group():
    """Each StatVar has one open-axis candidate, under its own source and method. The two
    serve one group, the one holding more points, so the other StatVar serves nothing."""
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    observations, warn = build_observations_planned(
        [_PER_SERIES_SV_A, _PER_SERIES_SV_B],
        {"observationAbout": ["country/USA"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    winners = {s.stat_var.ref: s.facet["measurementMethod"].ref for s in observations.series}
    assert winners == {_PER_SERIES_SV_A: "GrossDisbursement"}


def test_resolved_facets_single_ref_shape():
    """One distinct ref yields `ref` and `label`, without `selection`."""
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    observations, warn = build_observations_planned(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    facets = _resolved_facets(
        [s.facet for s in observations.series], {"measurementMethod": "bound"}
    )
    assert facets["measurementMethod"].basis == "bound"
    assert facets["measurementMethod"].ref == "GrossDisbursement"
    assert facets["measurementMethod"].selection is None
    assert facets["measurementMethod"].request_basis is None


def test_resolved_facets_per_series_shape():
    """Different method values yield `per_series`; an unbound axis has
    `fallback` basis. Each StatVar is read alone, since one read's members serve one
    group.
    """
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    series = []
    for stat_var in (_PER_SERIES_SV_A, _PER_SERIES_SV_B):
        observations, warn = build_observations_planned(
            [stat_var],
            {"observationAbout": ["country/USA"]},
            graph=FakeGraph(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            coverage=empty_coverage_snapshot(),
        )
        assert warn is None
        assert observations is not None
        series += observations.series
    facets = _resolved_facets([s.facet for s in series], {})
    entry = facets["measurementMethod"]
    assert entry.selection == "per_series"
    assert entry.ref is None
    assert entry.label is None
    assert entry.basis == "fallback"


def test_resolved_facets_request_basis_carrying_shape():
    """A USD/current request carries `request_basis` on the unit facet. The
    unconstrained winner is USD, so the request matches its native unit.
    """
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    graph = FakeGraph(currency_units={"USD": "USD", "EUR": "EUR", "GBP": "GBP"})
    request = IdentityUnitRequest(currency="USD", prices="current")
    filter_ = _facet_filter_from_bindings(
        graph, [], {"unit": IdentityFacetValueRequest(request=request)}, deadline=None
    )
    observations, warn = build_observations_planned(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=filter_,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    facets = _resolved_facets(
        [s.facet for s in observations.series],
        {"unit": "bound"},
        request_basis={"currency": "bound", "prices": "bound"},
    )
    assert facets["unit"].ref == "USD"
    assert facets["unit"].basis == "bound"
    assert facets["unit"].request_basis == {"currency": "bound", "prices": "bound"}
    # request_basis is scoped to the unit axis only.
    assert facets["measurementMethod"].request_basis is None


def test_resolved_facets_absent_axis_shape():
    """The series carry method and provenance but no unit, so `unit` is
    omitted from the resolved facet map.
    """
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double(
        _MATERIALISE_SINGLE_FACET_OBS
    )
    observations, warn = build_observations_planned(
        [_SINGLE_FACET_SV],
        {"observationAbout": ["country/USA"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    facets = _resolved_facets([s.facet for s in observations.series], {})
    assert set(facets) == {"measurementMethod", "provenance"}
    assert "unit" not in facets


def test_resolved_facets_series_missing_axis_forces_per_series():
    """A missing unit on one series forces `per_series`, even when the other
    series agree on USD.
    """
    usd_series = {"unit": SeriesFacetRef(ref="USD", label="US dollars")}
    axisless_series = {"measurementMethod": SeriesFacetRef(ref="GrossDisbursement", label=None)}
    facets = _resolved_facets([usd_series, axisless_series], {})
    entry = facets["unit"]
    assert entry.selection == "per_series"
    assert entry.ref is None
    assert entry.label is None


def test_resolved_facets_series_all_carrying_the_same_value_stays_single_ref():
    """All series carry the same unit, so one `ref` represents them."""
    s1 = {"unit": SeriesFacetRef(ref="USD", label="US dollars")}
    s2 = {"unit": SeriesFacetRef(ref="USD", label="US dollars")}
    facets = _resolved_facets([s1, s2], {})
    entry = facets["unit"]
    assert entry.selection is None
    assert entry.ref == "USD"
    assert entry.label == "US dollars"


def test_resolved_facets_no_series_carries_axis_stays_omitted():
    """An axis absent from every series is omitted."""
    s1 = {"measurementMethod": SeriesFacetRef(ref="GrossDisbursement", label=None)}
    s2 = {"measurementMethod": SeriesFacetRef(ref="GrossDisbursement", label=None)}
    facets = _resolved_facets([s1, s2], {})
    assert "unit" not in facets


def test_resolved_facets_key_order_follows_facet_axis_order():
    """Facet keys follow the declared axis order, regardless of input order."""
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    observations, warn = build_observations_planned(
        [_TWO_ENTITY_SV],
        {"observationAbout": ["country/USA", "country/GBR"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    facets = _resolved_facets([s.facet for s in observations.series], {})
    assert list(facets) == [a for a in ("provenance", "measurementMethod", "unit") if a in facets]


def test_facet_options_ranks_by_in_window_count_desc_then_ref_asc():
    """Options sort by summed in-window count, then ref. Expected counts come
    directly from the graph facets.
    """
    graph = _facet_axis_graph()
    facets_batch = graph.observation_facets_batch(
        [_MULTI_AXIS_SV], ["country/USA"], needs_dates=True
    )
    raw_facets = facets_batch[_MULTI_AXIS_SV]["country/USA"]

    expected_unit_counts: dict[str, int] = {}
    for f in raw_facets:
        if f.unit is not None:
            expected_unit_counts[f.unit] = expected_unit_counts.get(f.unit, 0) + len(f.dates)
    expected_unit_order = sorted(expected_unit_counts, key=lambda d: (-expected_unit_counts[d], d))

    options = _facet_options(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    by_axis = {fo.axis: fo for fo in options}
    unit_refs = [opt.value.ref.ref for opt in by_axis["unit"].options]
    assert unit_refs == expected_unit_order


# The graph names some methods as sentences; units keep their label as the graph has it.
_SENTENCE_LABELS = {"Commitment": "Commitment reporting.", "USD": "US dollars."}


def test_a_series_method_label_drops_its_trailing_full_stop():
    facet_map = series_facet_map(
        {"measurementMethod": "Commitment", "unit": "USD"}, _SENTENCE_LABELS
    )
    assert {axis: ref.label for axis, ref in facet_map.items()} == {
        "measurementMethod": "Commitment reporting",
        "unit": "US dollars.",
    }


def test_a_method_option_label_drops_its_trailing_full_stop():
    graph = FakeGraph(
        nodes={dcid: {"label": label} for dcid, label in _SENTENCE_LABELS.items()},
        obs=_FACET_AXIS_OBS,
        detect={},
        resolve={},
    )
    options = _facet_options(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    labels = {
        (fo.axis, opt.value.ref.ref): opt.value.ref.label
        for fo in options
        for opt in fo.options
        if opt.value.ref is not None
    }
    assert labels[("measurementMethod", "Commitment")] == "Commitment reporting"
    assert labels[("unit", "USD")] == "US dollars."


def test_facet_options_deterministic_across_repeated_calls():
    """Repeated calls with the same graph return identical option order."""
    graph = _facet_axis_graph()
    first = _facet_options(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    second = _facet_options(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    assert [fo.model_dump() for fo in first] == [fo.model_dump() for fo in second]


def test_facet_options_omits_axis_with_no_data():
    """A facet with only `measurementMethod` yields no unit or provenance options."""
    graph = _facet_axis_graph()
    options = _facet_options(
        [_SINGLE_FACET_SV],
        {"observationAbout": ["country/USA"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    assert {fo.axis for fo in options} == {"measurementMethod"}


def test_facet_options_source_is_always_observed():
    graph = _facet_axis_graph()
    options = _facet_options(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    assert options
    assert all(fo.source == "observed" for fo in options)


def test_facet_options_unaffected_by_a_narrowing_binding():
    """The preview serves the facet the binding admits, while facet options list every
    method value from the raw candidates.
    """
    sv, entity = [_MULTI_AXIS_SV], ["country/USA"]
    graph = _facet_axis_graph()
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()

    narrowed, warn = build_observations_planned(
        sv,
        {"observationAbout": entity},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter={
            "measurementMethod": FacetAxisFilter(
                predicate=lambda f: f.measurement_method in frozenset({"GrantEquivalent"}),
            )
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert {s.facet["measurementMethod"].ref for s in narrowed.series} == {"GrantEquivalent"}

    options = _facet_options(
        sv,
        {"observationAbout": entity},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    by_axis = {fo.axis: fo for fo in options}
    method_refs = {opt.value.ref.ref for opt in by_axis["measurementMethod"].options}
    assert method_refs == {"GrossDisbursement", "Commitment", "GrantEquivalent"}


def test_facet_options_caps_at_search_max_options(search_settings):
    search_settings(max_options=2)
    graph = _facet_axis_graph()
    options = _facet_options(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    by_axis = {fo.axis: fo for fo in options}
    assert len(by_axis["unit"].options) == 2
    assert by_axis["unit"].truncated is True
    assert by_axis["measurementMethod"].truncated is True


def test_facet_options_excludes_zero_obs_facets():
    """Live graph reads contain observed facets. This synthetic zero-count
    facet checks that axis refs alone do not create options.
    """
    graph = FakeGraph()
    zero_obs_facet = Facet(
        earliest_date=None,
        latest_date=None,
        obs_count=0,
        dates=[],
        values=[],
        provenance_id="dc/base/ZeroObsSource",
        measurement_method="ZeroObsMethod",
        unit="ZeroObsUnit",
    )
    graph.observation_facets_batch = lambda *a, **k: {  # type: ignore[method-assign]
        "sv": {"entity": [zero_obs_facet]}
    }
    options = _facet_options(
        ["sv"],
        {"observationAbout": ["entity"]},
        years=None,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    assert options == []


def _ids_like_facet_graph(monkeypatch: pytest.MonkeyPatch) -> FakeGraph:
    recorded_years = range(1970, 2025)
    estimated_years = range(2025, 2033)
    recorded = Facet(
        earliest_date="1970",
        latest_date="2024",
        obs_count=len(recorded_years),
        dates=[str(year) for year in recorded_years],
        values=[1.0 for _ in recorded_years],
        provenance_id="dc/base/WorldBank_IDS",
        unit="USDollar",
    )
    estimated = Facet(
        earliest_date="2025",
        latest_date="2032",
        obs_count=len(estimated_years),
        dates=[str(year) for year in estimated_years],
        values=[1.0 for _ in estimated_years],
        provenance_id="dc/base/WorldBank_IDS",
        measurement_method="Estimated",
        unit="USDollar",
    )
    graph = FakeGraph()
    monkeypatch.setattr(
        graph,
        "observation_facets_batch",
        lambda *a, **k: {"sv": {"country/KEN": [recorded, estimated]}},
    )
    return graph


@pytest.mark.parametrize(
    ("window", "expected_methods"),
    [
        (TimeWindow(start_year=2015, end_year=2024), None),
        (TimeWindow(start_year=2027, end_year=2027), ["Estimated"]),
        (None, [None, "Estimated"]),
        ("latest", [None, "Estimated"]),
    ],
)
def test_facet_options_list_only_values_with_observations_in_the_window(
    monkeypatch, window, expected_methods
):
    """A facet value with no observation inside the window is not offered. The recorded
    facet declares no method, so it is the ref-less option (`None` here), listed only beside
    a declared method."""
    graph = _ids_like_facet_graph(monkeypatch)
    options = _facet_options(
        ["sv"],
        {"observationAbout": ["country/KEN"]},
        years=window,
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    by_axis = {
        fo.axis: [opt.value.ref.ref if opt.value.ref else None for opt in fo.options]
        for fo in options
    }

    assert by_axis.get("measurementMethod") == expected_methods
    assert by_axis["provenance"] == ["dc/base/WorldBank_IDS"]
    assert by_axis["unit"] == ["USDollar"]


def test_facet_options_offer_a_facet_whose_dates_came_back_short(monkeypatch):
    """A dated read that returned fewer dates than `obs_count` may omit in-window points, so
    the facet is offered though none of its returned dates falls in the window."""
    short = Facet(
        earliest_date="1970",
        latest_date="2024",
        obs_count=55,
        dates=["1970", "1971"],
        values=[1.0, 1.0],
        dates_complete=False,
        provenance_id="dc/base/WorldBank_IDS",
        measurement_method="Recorded",
    )
    complete = Facet(
        earliest_date="1970",
        latest_date="1971",
        obs_count=2,
        dates=["1970", "1971"],
        values=[1.0, 1.0],
        provenance_id="dc/base/WorldBank_IDS",
        measurement_method="Archived",
    )
    graph = FakeGraph()
    monkeypatch.setattr(
        graph,
        "observation_facets_batch",
        lambda *a, **k: {"sv": {"country/KEN": [short, complete]}},
    )

    options = _facet_options(
        ["sv"],
        {"observationAbout": ["country/KEN"]},
        years=TimeWindow(start_year=2020, end_year=2024),
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    by_axis = {fo.axis: [opt.value.ref.ref for opt in fo.options] for fo in options}

    assert by_axis["measurementMethod"] == ["Recorded"]


def test_maybe_build_facet_options_builds():
    graph = _facet_axis_graph()
    options, warn = _maybe_build_facet_options(
        facet_probe([_MULTI_AXIS_SV], {"observationAbout": ["country/USA"]}, years=None),
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    assert warn is None
    assert options


def test_maybe_build_facet_options_fails_open_on_graph_error():
    """A graph error returns no options with `OPTION_COVERAGE_PROBE_FAILED`."""
    graph = _facet_axis_graph()

    def _boom(*a, **k):
        raise GraphInfraError("graph down")

    graph.observation_facets_batch = _boom  # type: ignore[method-assign]
    options, warn = _maybe_build_facet_options(
        facet_probe([_MULTI_AXIS_SV], {"observationAbout": ["country/USA"]}, years=None),
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )
    assert options == []
    assert warn is not None
    assert warn.code == OPTION_COVERAGE_PROBE_FAILED


def test_maybe_build_facet_options_non_infra_error_propagates():
    """A programming error propagates without an option-coverage warning."""
    graph = _facet_axis_graph()

    def _boom(*a, **k):
        raise TypeError("bug in the probe, not a graph error")

    graph.observation_facets_batch = _boom  # type: ignore[method-assign]
    with pytest.raises(TypeError):
        _maybe_build_facet_options(
            [_MULTI_AXIS_SV],
            {"observationAbout": ["country/USA"]},
            years=None,
            graph=graph,
            coverage=_NO_COVERAGE,
            deadline=None,
        )


def test_facet_options_skip_a_short_read_whose_span_ends_before_the_window(monkeypatch):
    """A short read spanning 1970 to 2012 holds no point from 2020 to 2024 by its span, the
    point rule every plan reads, so its method is not offered: picking it would answer
    `no_observations`."""
    short = Facet(
        earliest_date="1970",
        latest_date="2012",
        obs_count=43,
        dates=["1970", "1971"],
        values=[1.0, 1.0],
        dates_complete=False,
        provenance_id="dc/base/WorldBank_IDS",
        measurement_method="Recorded",
    )
    graph = FakeGraph()
    monkeypatch.setattr(
        graph, "observation_facets_batch", lambda *a, **k: {"sv": {"country/KEN": [short]}}
    )

    options = _facet_options(
        ["sv"],
        {"observationAbout": ["country/KEN"]},
        years=TimeWindow(start_year=2020, end_year=2024),
        graph=graph,
        coverage=_NO_COVERAGE,
        deadline=None,
    )

    assert not [option for option in options if option.axis == "measurementMethod"]
