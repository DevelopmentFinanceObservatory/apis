"""A request's facet filter applies once, to the candidates each series or pair elects from,
before any election. Coverage, the headline, discovery and the preview all describe the
facets the filter admits, and a pair whose best facet fails the filter serves its best
facet that passes."""

from __future__ import annotations

import pytest

from search.engine.extract import DateRequest
from search.engine.region import _build_observations, resolve_spec_resubmit
from search.engine.region.facets import FacetAdmission, filter_answer
from search.engine.region.result import FacetAxisFilter
from search.engine.region.state import EngineOptions
from search.engine.retrieval.coverage import CoverageForm, answer_coverage
from search.engine.retrieval.read_plan import plan_standard
from search.engine.retrieval.retrieve import Materialised
from search.models import (
    BaseIdentity,
    CoverageExact,
    IdentityBindingValue,
    ResubmitIdentityPayload,
    TimeWindow,
)
from series_coverage.schema import FacetMetadata
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    materialise_double_from_obs,
    pair_plan,
    walk_options,
    with_member_arcs,
)
from tests.engine.test_role_scope_resubmit_narrowing_e2e import (
    _KEN,
    _NOR,
    _ROLE_NAMES,
    _SV_DCID,
    _UNBOUND_DONOR,
    _facet,
    _fresh_discover,
    _nodes,
)
from tests.fixtures import FakeGraph

_LATEST = DateRequest(window=None, latest=True)
_RESOLVE = {"Kenya": _KEN, "Norway": _NOR}
_SOURCES = ["source/A", "source/B"]


def _materialise_clients(observations: dict):
    """Build role-bearing materialise doubles for a resubmit preview fixture."""
    pairs = {
        (stat_var, role_tuple): [
            ObsFacetFixture(
                facet_id=facet.facet_id or "fixture",
                points=tuple(zip(facet.dates or (), facet.values or (), strict=True)),
                provenance_id=facet.provenance_id,
                measurement_method=facet.measurement_method,
                unit=facet.unit,
            )
            for facet in facets
        ]
        for stat_var, by_role_tuple in observations.items()
        for role_tuple, facets in by_role_tuple.items()
    }
    graph, sdmx, _requests = build_materialise_double(pairs)
    return graph, sdmx


def _discover_and_resubmit(
    monkeypatch,
    search_settings,
    *,
    bulk: bool,
    rows: list,
    facet_metadata: dict[str, FacetMetadata],
    observations: dict,
    facets: dict,
    date_request: DateRequest = _LATEST,
):
    """Discover Kenya and Norway on the open destination axis, then resubmit that answer
    with `facets` bound, reading its preview."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)
    index = patch_coverage(monkeypatch, rows, role_names=_ROLE_NAMES, facet_metadata=facet_metadata)
    coverage = FakeCoverageService(index).snapshot()
    fresh = _fresh_discover(
        resolve=_RESOLVE,
        label_dcids=_SOURCES,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        date_request=date_request,
        coverage=coverage,
    )
    assert fresh.status == "definite", fresh
    spec = fresh.specs[0]
    materialise_graph, materialise_sdmx = _materialise_clients(observations)
    resubmit_graph = FakeGraph(
        nodes=_nodes(resolve=_RESOLVE, label_dcids=_SOURCES), detect={}, resolve={}, obs={}
    )
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=BaseIdentity.model_validate(
                {**spec.definition.identity.model_dump(mode="python"), "facets": facets}
            ),
        ),
        graph=resubmit_graph,
        sdmx=FakeSdmxClient(observations=observations, availability={}),
        coverage=coverage,
        options=walk_options(
            resubmit_graph,
            EngineOptions(
                include_observations=True,
                materialise_graph=with_member_arcs(materialise_graph, resubmit_graph),
                materialise_sdmx=materialise_sdmx,
            ),
        ),
    )
    assert region.status == "definite", region
    return region.specs[0].resolved


def _dims(resolved) -> dict[str, int]:
    return {d.label: d.count for d in resolved.coverage.dimensions or []}


# Kenya holds `fA` from source A with a 2023 point, Norway `fB` from source B with 2020.
_SOURCE_ROWS = [
    (_SV_DCID, "fA", _UNBOUND_DONOR, _KEN, [2023]),
    (_SV_DCID, "fB", _UNBOUND_DONOR, _NOR, [2020]),
]
_SOURCE_METADATA = {
    "fA": FacetMetadata(provenance_id="source/A"),
    "fB": FacetMetadata(provenance_id="source/B"),
}
_SOURCE_OBSERVATIONS = {
    _SV_DCID: {
        (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [
            _facet(facet_id="fA", provenance_id="source/A", latest_date="2023")
        ],
        (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [
            _facet(facet_id="fB", provenance_id="source/B", latest_date="2020")
        ],
    }
}


@pytest.mark.parametrize("bulk", [True, False], ids=["bulk", "per-pair"])
def test_a_source_bound_answer_dates_and_counts_the_bound_source(
    monkeypatch, search_settings, bulk: bool
) -> None:
    """Bound to source B, the answer covers Norway's 2020 point alone: one recipient, one
    source, 2020 in coverage and the headline, never Kenya's 2023 from source A."""
    resolved = _discover_and_resubmit(
        monkeypatch,
        search_settings,
        bulk=bulk,
        rows=_SOURCE_ROWS,
        facet_metadata=_SOURCE_METADATA,
        observations=_SOURCE_OBSERVATIONS,
        facets={"provenance": IdentityBindingValue(ref="source/B")},
    )

    assert resolved.coverage.window == TimeWindow(start_year=2020, end_year=2020)
    assert (_dims(resolved)["recipients"], _dims(resolved)["sources"]) == (1, 1)
    assert resolved.headline is not None
    assert (resolved.headline.year, resolved.headline.series) == (2020, 1)
    if bulk:
        assert resolved.observations is not None
        assert [
            (next(e.ref for e in s.entities if e.role == "destination"), [p.date for p in s.points])
            for s in resolved.observations.series
        ] == [(_NOR, ["2020"])]


@pytest.mark.parametrize("bulk", [True, False], ids=["bulk", "per-pair"])
def test_a_unit_bound_answer_heads_the_admitted_rows(
    monkeypatch, search_settings, bulk: bool
) -> None:
    """Kenya's `fA` is in USD to 2021 and Norway's `fB` in EUR to 2023. Bound to USD, the
    headline names Kenya's 2021 row, as coverage does, never Norway's EUR 2023."""
    resolved = _discover_and_resubmit(
        monkeypatch,
        search_settings,
        bulk=bulk,
        rows=[
            (_SV_DCID, "fA", _UNBOUND_DONOR, _KEN, [2021]),
            (_SV_DCID, "fB", _UNBOUND_DONOR, _NOR, [2023]),
        ],
        facet_metadata={
            "fA": FacetMetadata(unit="USD", provenance_id="source/A"),
            "fB": FacetMetadata(unit="EUR", provenance_id="source/B"),
        },
        observations={
            _SV_DCID: {
                (("creditor", _UNBOUND_DONOR), ("debtor", _KEN)): [
                    _facet(facet_id="fA", unit="USD", provenance_id="source/A", latest_date="2021")
                ],
                (("creditor", _UNBOUND_DONOR), ("debtor", _NOR)): [
                    _facet(facet_id="fB", unit="EUR", provenance_id="source/B", latest_date="2023")
                ],
            }
        },
        facets={"unit": IdentityBindingValue(ref="USD")},
    )

    assert resolved.coverage.window == TimeWindow(start_year=2021, end_year=2021)
    assert resolved.headline is not None
    assert (resolved.headline.year, resolved.headline.series) == (2021, 1)
    assert resolved.headline.role_counts.get("debtor") == 1


def test_a_discovered_range_counts_the_sources_of_pairs_inside_it(monkeypatch) -> None:
    """Kenya's source A has only 2010; Norway's source B has 2020. A 2019 to 2021 request
    discovers Norway alone, so one source counts, the one the preview reads."""
    index = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "fA", _UNBOUND_DONOR, _KEN, [2010]),
            (_SV_DCID, "fB", _UNBOUND_DONOR, _NOR, [2020]),
        ],
        role_names=_ROLE_NAMES,
        facet_metadata=_SOURCE_METADATA,
    )
    fresh = _fresh_discover(
        resolve=_RESOLVE,
        label_dcids=_SOURCES,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        date_request=DateRequest(window=TimeWindow(start_year=2019, end_year=2021), latest=False),
        coverage=FakeCoverageService(index).snapshot(),
    )

    resolved = fresh.specs[0].resolved
    assert _dims(resolved)["sources"] == 1
    assert [m.ref for m in resolved.scope.members] == [_NOR]


# One Kenya series: `f1` (method M1, 10 points to 2023) outranks `f2` (M2, 5 points to 2020).
_SET_OBS = {
    "Count_Person|country/KEN": [
        {
            "facetId": "f1",
            "earliestDate": "2014",
            "latestDate": "2023",
            "obsCount": 10,
            "provenanceId": "dc/base/P1",
            "measurementMethod": "M1",
            "observations": [{"date": str(y), "value": float(y)} for y in range(2014, 2024)],
        },
        {
            "facetId": "f2",
            "earliestDate": "2016",
            "latestDate": "2020",
            "obsCount": 5,
            "provenanceId": "dc/base/P2",
            "measurementMethod": "M2",
            "observations": [{"date": str(y), "value": float(y)} for y in range(2016, 2021)],
        },
    ]
}
_METHOD_SET = {
    "measurementMethod": FacetAxisFilter(predicate=lambda f: f.measurement_method in {"M2", "M3"})
}


def test_a_set_filter_serves_the_best_admitted_facet_as_coverage_dates_it() -> None:
    """A method set {M2, M3} admits `f2` alone. The per-pair preview elects among the
    admitted facets and serves `f2`'s five points, and coverage under the same filter counts
    those five, where electing `f1` first would reject the pair and serve nothing."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_SET_OBS)
    entities = {"observationAbout": ["country/KEN"]}
    observations, warn = _build_observations(
        pair_plan(["Count_Person"], entities, reader=materialise_graph).narrowed(
            FacetAdmission(axes=_METHOD_SET).admits
        ),
        ["Count_Person"],
        entities,
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        facet_filter=_METHOD_SET,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    (served,) = observations.series
    assert served.facet["measurementMethod"].ref == "M2"
    assert [p.date for p in served.points] == ["2016", "2017", "2018", "2019", "2020"]

    facets = FakeGraph(obs=_SET_OBS).observation_facets(
        stat_var="Count_Person", entity="country/KEN"
    )
    plan = plan_standard(
        {("Count_Person", "country/KEN"): facets}, members=["Count_Person"], years=None
    )
    form = CoverageForm(variables=True)
    mat = Materialised(
        sv_dcids=["Count_Person"],
        has_data=True,
        coverage=answer_coverage(plan, form),
        coverage_form=form,
        facets_by_sv={"Count_Person": facets},
        read_plan=plan,
    )
    filtered = filter_answer(
        mat,
        FacetAdmission(axes=_METHOD_SET),
        date_request=None,
        coverage_index=empty_coverage_snapshot().index,
    )
    assert isinstance(filtered.coverage, CoverageExact)
    assert filtered.coverage.observation_count == 5
