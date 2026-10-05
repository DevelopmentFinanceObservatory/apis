"""The per-series preview reads its read plan: materialise receives each series' planned
candidates and a filter admitting the planned group alone, so it elects nothing the plan did
not, and a result that dispatched a facet outside its group is never served, whether or not
a unit request converts it."""

from __future__ import annotations

import logging
from dataclasses import replace
from unittest.mock import patch

import pytest

import search.engine.region.observations.fetch as region_fetch
from materialise.convert import ConversionRecord, merged_record
from materialise.facets import FacetSelection
from materialise.frame_build import Frame
from materialise.graph import Facet
from search.engine.extract import DateRequest
from search.engine.region.codes import OBSERVATIONS_PROBE_FAILED
from search.engine.region.observations.fetch import _build_observations
from search.engine.retrieval.read_plan import PairPlan, PlanSeries, plan_pairs, plan_standard
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_scope import discover_role_scope
from search.models import IdentityUnitRequest, TimeWindow
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    sdmx_double,
    stub_preview_result,
)
from tests.fixtures import FakeGraph
from tests.materialise.conftest import FakeGraphReader

_SV = "Count_Person"
_KEN = "country/KEN"


def _facet(facet_id: str, provenance: str, years: range) -> Facet:
    dates = tuple(str(y) for y in years)
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        obs_count=len(dates),
        earliest_date=dates[0],
        latest_date=dates[-1],
        dates=dates,
        dates_complete=True,
    )


def test_the_preview_reads_the_plans_candidates_and_probes_nothing() -> None:
    """Kenya's plan holds `f-long` (2000-2020) and `f-short` (2018-2020) from two sources;
    the group is `f-long`. Materialise reads the planned candidates, admits `f-long` alone,
    dispatches it, and the graph serves no facet read."""
    long_fixture = ObsFacetFixture(
        facet_id="f-long",
        points=tuple((str(y), float(y)) for y in range(2000, 2021)),
        provenance_id="dc/base/Long",
    )
    short_fixture = ObsFacetFixture(
        facet_id="f-short",
        points=tuple((str(y), 1.0) for y in range(2018, 2021)),
        provenance_id="dc/base/Short",
    )
    reader, sdmx, requests = build_materialise_double(
        {(_SV, (("observationAbout", _KEN),)): [long_fixture, short_fixture]}
    )
    plan = plan_standard(
        {
            (_SV, _KEN): [
                _facet("f-long", "dc/base/Long", range(2000, 2021)),
                _facet("f-short", "dc/base/Short", range(2018, 2021)),
            ]
        },
        members=[_SV],
        years=None,
    )

    observations, warn = _build_observations(
        plan,
        [_SV],
        {"observationAbout": [_KEN]},
        graph=FakeGraph(),
        materialise_graph=reader,
        materialise_sdmx=sdmx,
        coverage=FakeCoverageService(build_coverage_index([], role_names={})).snapshot(),
    )

    assert warn is None
    (served,) = observations.series
    assert served.facet["provenance"].ref == "dc/base/Long"
    assert reader.calls["variable_facets_batch"] == 0
    assert {r.url.params.get("c[facetId]") for r in requests} == {"f-long"}


def test_a_windowed_discovered_pair_serves_the_planned_facet(search_settings) -> None:
    """Kenya holds `f-long` 1970-2021 and `f-short` 2020-2022; inside 2020 to 2022 the plan
    elects `f-short`, which holds more points in those years, as a named pair does, and the
    preview serves it."""
    search_settings(obs_preview_max_pairs=120)
    index = build_coverage_index(
        [
            ("V", "f-long", ABSENT_ENTITY, _KEN, list(range(1970, 2022))),
            ("V", "f-short", ABSENT_ENTITY, _KEN, [2020, 2021, 2022]),
        ],
        role_names={"V": RoleNames(donor=None, recipient="debtor")},
        facet_metadata={
            "f-long": FacetMetadata(provenance_id="dc/base/Long"),
            "f-short": FacetMetadata(provenance_id="dc/base/Short"),
        },
    )
    snapshot = FakeCoverageService(index).snapshot()
    window = DateRequest(window=TimeWindow(start_year=2020, end_year=2022), latest=False)
    mat = discover_role_scope(
        surviving_svs=["V"],
        member_facts={},
        member_population_exact=True,
        open_flow_axis="destination",
        role_name="debtor",
        opposite_role_name=None,
        opposite_dcids=(),
        facet_filters={},
        date_request=window,
        coverage_snapshot=snapshot,
    )
    assert isinstance(mat, Materialised) and isinstance(mat.read_plan, PairPlan)
    dispatched: list[str | None] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        admitted = [f for f in precomputed if request.candidate_filter(f)]
        dispatched.extend(f.facet_id for f in admitted)
        result = stub_preview_result(role_key="debtor", entity=_KEN, provenance="dc/base/Short")
        selection = replace(result.selections["provenance"], dispatch_facet_id=admitted[0].facet_id)
        return replace(result, selections={"provenance": selection})

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = _build_observations(
            mat.read_plan,
            ["V"],
            {"debtor": [_KEN]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            coverage=snapshot,
        )

    assert warn is None
    assert dispatched == ["f-short"]
    assert [s.facet["provenance"].ref for s in observations.series] == ["dc/base/Short"]


@pytest.mark.parametrize("converting", [False, True], ids=["native", "unit_request"])
def test_a_result_dispatching_outside_its_planned_group_is_never_served(
    caplog, converting: bool
) -> None:
    """Materialise dispatching `f1`, a facet outside Kenya's planned group, is a plan
    mismatch: the pair is logged at WARNING, counted as failed, and not served, with or
    without a unit request converting the read."""
    plan = plan_standard(
        {(_SV, _KEN): [replace(_facet("f-a", "prov1", range(2015, 2021)), unit="CurrencyUSD")]},
        members=[_SV],
        years=None,
    )
    unit = FacetSelection(
        axis="unit",
        ref="CurrencyUSD",
        basis="deployment_fallback",
        default_policy=None,
        default_policy_version=None,
        selection_reason="native_currency_preferred",
        dispatch_facet_id="f-a",
    )
    conversion = (
        {
            "unit_request": IdentityUnitRequest(currency="USD", prices="current"),
            "elected": {_SV: ({"unit": unit},)},
        }
        if converting
        else {}
    )

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        result = stub_preview_result(entity=_KEN)
        assert result.selections["provenance"].dispatch_facet_id == "f1"
        return result

    with (
        patch.object(region_fetch, "materialise_preview", _spy),
        caplog.at_level(logging.WARNING, logger=region_fetch.__name__),
    ):
        observations, warn = _build_observations(
            plan,
            [_SV],
            {"observationAbout": [_KEN]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            coverage=FakeCoverageService(build_coverage_index([], role_names={})).snapshot(),
            **conversion,
        )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert observations.omitted.hard_failed == 1
    assert warn is not None
    assert "outside the preview's read plan" in warn.message
    assert any("outside its planned group" in r.getMessage() for r in caplog.records)


_NOR, _FRA = "country/NOR", "country/FRA"


def test_an_open_role_read_gives_each_series_its_own_rows_and_conversion_record() -> None:
    """Kenya is bound and the creditor open, with a series planned for Norway and France.
    One read serves both: each series takes the rows carrying its creditor and the
    conversion record materialise kept for it, and rows of a creditor the plan does not
    serve are dropped."""
    facet = replace(_facet("f1", "prov1", range(2019, 2021)), unit="CurrencyUSD")
    plan = plan_pairs(
        [
            PlanSeries(
                stat_var="V",
                role_tuple=(("creditor", creditor), ("debtor", _KEN)),
                candidates=(facet,),
            )
            for creditor in (_NOR, _FRA)
        ],
        members=["V"],
        years=None,
    )
    held = {_NOR: 2, _FRA: 1, "country/DEU": 3}
    stub = stub_preview_result(sv="V", role_key="debtor", entity=_KEN)
    record = ConversionRecord(
        recipe_id="recipe/Test",
        exchange_rate_table_version="v1",
        deflator_table_version=None,
        native_currency="CurrencyUSD",
        native_unit_ref="CurrencyUSD",
        table_provenance_id="prov/table",
        table_refresh_date="2026-01-01",
        fallback_count=0,
        fallback_examples=(),
        projected_count=0,
        projected_examples=(),
    )
    records = {
        (creditor,): replace(record, projected_count=count) for creditor, count in held.items()
    }
    requests: list[tuple[dict, tuple[str, ...]]] = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        requests.append((dict(request.role_bindings), tuple(series_roles)))
        frame = Frame.build(
            role_keys=("creditor", "debtor"),
            rows=[
                {
                    "stat_var": "dcid:V",
                    "creditor": creditor,
                    "debtor": _KEN,
                    "time_period": str(2020 - offset),
                    "value": 1.0,
                    "unit": "CurrencyUSD",
                    "measurementMethod": None,
                    "provenance": "prov1",
                }
                for creditor, count in held.items()
                for offset in range(count)
            ],
        )
        selection = replace(stub.selections["provenance"], dispatch_facet_id="f1")
        return replace(
            stub,
            frame=frame,
            total_rows=len(frame.rows),
            selections={"provenance": selection},
            conversion_record=merged_record(list(records.values())),
            series_conversion_records=records,
            series_total_rows={(creditor,): count for creditor, count in held.items()},
        )

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = _build_observations(
            plan,
            ["V"],
            {"debtor": [_KEN]},
            graph=FakeGraph(),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=sdmx_double(),
            coverage=FakeCoverageService(build_coverage_index([], role_names={})).snapshot(),
        )

    assert warn is None
    assert requests == [({"debtor": [_KEN]}, ("creditor",))]
    served = {
        next(e.ref for e in s.entities if e.role == "origin"): (
            len(s.points),
            s.total_obs,
            s.conversion.projected_count if s.conversion is not None else None,
        )
        for s in observations.series
    }
    assert served == {_NOR: (2, 2, 2), _FRA: (1, 1, 1)}
    assert observations.total_obs == 3
