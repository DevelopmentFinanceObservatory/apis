"""Tests that observation previews use each StatVar's elected facet combination
and perform the requested unit conversion. Real `materialise_preview` calls
cover buffered dispatch and the streaming relaunch path.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import patch

import httpx
import pytest

import search.engine.region.observations.fetch as region_fetch
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import MaterialisationRequest
from materialise.facets import FacetSelection
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.concurrency import scoped_concurrency
from search.engine.region.codes import OBSERVATIONS_NO_MATCHING_SOURCE
from search.engine.region.observations.dispatch import PreviewRead, PreviewReader
from search.engine.region.result import FacetAxisFilter
from search.engine.region.un_system import NO_LABELS
from search.models import (
    IdentityCurrencyRole,
    IdentityUnitRequest,
    SeriesConversionFallbackExample,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
    build_observations_planned,
    harness_reader,
    pair_plan,
    sdmx_double,
    stub_preview_result,
)
from tests.fixtures import FakeGraph
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_BARE_STAT_VAR = "Test_Amount_USD"
_PROVENANCE = "provenance/TestGrantSource"
_RECIPE = "ConversionRecipeDAC"
_TABLE_PROVENANCE = RECIPE_BODIES[_RECIPE].table_provenance
_FALLBACK_ENTITY = "DACCountriesTotal"
_YEAR = "2022"
_MEASUREMENT_METHOD = "Actual"
_KEN = "country/KEN"


def _recipe_arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    """Build the recipe provenance, version, and currency nodes used by the tests."""
    arcs = conversion_arcs(
        recipe_id=_RECIPE, provenance_id=_PROVENANCE, currencies=["CurrencyUSD", "CurrencyEUR"]
    )
    arcs[_PROVENANCE]["isPartOf"] = ()
    arcs[_BARE_STAT_VAR] = dict(CONVERTIBLE_STAT_VAR_ARCS)
    return arcs


def _elected_combo(
    *, provenance: str = _PROVENANCE, unit: str = "CurrencyUSD"
) -> dict[str, FacetSelection]:
    """Build elected provenance, method, and native-unit selections."""

    def _selection(axis: str, ref: str) -> FacetSelection:
        return FacetSelection(
            axis=axis,
            ref=ref,
            basis="bound",
            default_policy=None,
            default_policy_version=None,
            selection_reason=None,
            dispatch_facet_id="f1",
        )

    return {
        "provenance": _selection("provenance", provenance),
        "measurementMethod": _selection("measurementMethod", _MEASUREMENT_METHOD),
        "unit": _selection("unit", unit),
    }


def _usd_fixture(*, provenance: str = _PROVENANCE) -> ObsFacetFixture:
    return ObsFacetFixture(
        facet_id="f1",
        points=(),
        provenance_id=provenance,
        measurement_method=_MEASUREMENT_METHOD,
        unit="CurrencyUSD",
        # The facet's one observation, the row each fixture serves.
        obs_count=1,
    )


_BASE_YEAR = 2020

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _main_body(*, entity: str, fixture: ObsFacetFixture, value: float) -> str:
    return _main_body_for([(entity, fixture, value)])


def _main_body_for(rows: list[tuple[str, ObsFacetFixture, float]]) -> str:
    return _csv_header(["observationAbout"]) + "".join(
        _csv_row(
            stat_var=_BARE_STAT_VAR,
            role_dcids=[entity],
            fixture=fixture,
            date_str=_YEAR,
            value=value,
        )
        for entity, fixture, value in rows
    )


def _sdmx_router(
    *,
    rates: dict[str, dict[int, float]],
    deflators: dict[str, dict[int, float]] | None = None,
    observations: dict[str, str] | None = None,
) -> MaterialiseSdmxClient:
    """Serves the DAC tables built from `rates` and `deflators`, and each
    variable's body in `observations`, or a header-only body for a variable
    this test never expects to be fetched.
    """
    bodies = observations or {}

    def fetch(request: httpx.Request) -> httpx.Response:
        params = dict(httpx.QueryParams(request.url.query.decode()))
        var = params.get("c[variableMeasured]", "")
        body = bodies.get(var, _csv_header(["observationAbout"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(
        recipe_table_bodies(_RECIPE, rates=rates, deflators=deflators), fallback=fetch
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


_USD_AND_EUR_RATES = {"CurrencyUSD": {int(_YEAR): 1.0}, "CurrencyEUR": {int(_YEAR): 0.9}}


def test_elected_dispatch_converts_native_value_by_the_fake_rate():
    """A real preview converts a native USD value of 100 using the fake EUR
    rate of 0.9. The conversion output changes with the rate.
    """
    entity = "country/KEN"
    usd_fixture = _usd_fixture()
    graph = FakeGraphReader(
        arcs=_recipe_arcs(),
        facets={(_BARE_STAT_VAR, entity): [_facet_from_fixture(usd_fixture)]},
    )
    sdmx = _sdmx_router(
        rates=_USD_AND_EUR_RATES,
        observations={_BARE_STAT_VAR: _main_body(entity=entity, fixture=usd_fixture, value=100.0)},
    )
    elected = {_BARE_STAT_VAR: (_elected_combo(),)}
    unit_request = IdentityUnitRequest(currency="EUR", prices="current")

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [entity]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=unit_request,
        elected=elected,
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == 1
    series = observations.series[0]
    assert [p.date for p in series.points] == [_YEAR]
    # The native USD value (100) times the fake rate, 0.9 EUR per USD
    # against CurrencyUSD's own 1.0 anchor row.
    assert [p.value for p in series.points] == [pytest.approx(100.0 * 0.9 / 1.0)]
    # A converted series carries what produced it: the recipe id, the
    # table provenance, and the exchange-rate table's own version (a SHA-256 of
    # its rows). No deflator table loads for a current-prices conversion.
    assert series.conversion is not None
    assert series.conversion.recipe_id == _RECIPE
    assert series.conversion.table_provenance.ref == _TABLE_PROVENANCE
    assert _SHA256_HEX.fullmatch(series.conversion.exchange_rate_table_version)
    assert series.conversion.deflator_table_version is None
    assert series.conversion.fallback_count == 0
    assert series.conversion.projected_count == 0
    # facet.unit names the requested currency the points are actually in
    # (EUR), not the native facet's own USD -- FacetSelection.ref for a
    # request-form unit binding is the resolved NATIVE unit the conversion
    # started from.
    assert series.facet["unit"].ref == "CurrencyEUR"


def test_elected_dispatch_keeps_a_bound_null_method_ref(monkeypatch):
    """A method pinned to the facets that declare none narrows the plan to them, and the
    converting read dispatches the plan's group, so the preview serves the recorded series
    and never the Estimated projection beside it."""
    entity = "country/KEN"
    recorded = ObsFacetFixture(
        facet_id="f-rec",
        provenance_id=_PROVENANCE,
        measurement_method=None,
        unit="CurrencyUSD",
        obs_count=1,
    )
    estimated = ObsFacetFixture(
        facet_id="f-est",
        provenance_id=_PROVENANCE,
        measurement_method="Estimated",
        unit="CurrencyUSD",
        obs_count=1,
    )
    graph = FakeGraphReader(
        arcs=_recipe_arcs(),
        facets={
            (_BARE_STAT_VAR, entity): [
                _facet_from_fixture(recorded),
                _facet_from_fixture(estimated),
            ]
        },
    )
    main = _main_body(entity=entity, fixture=recorded, value=100.0) + _csv_row(
        stat_var=_BARE_STAT_VAR,
        role_dcids=[entity],
        fixture=estimated,
        date_str=_YEAR,
        value=200.0,
    )
    sdmx = _sdmx_router(rates=_USD_AND_EUR_RATES, observations={_BARE_STAT_VAR: main})
    combo = _elected_combo()
    combo["measurementMethod"] = FacetSelection(
        axis="measurementMethod",
        ref=None,
        basis="bound",
        default_policy=None,
        default_policy_version=None,
        selection_reason=None,
        dispatch_facet_id="f-rec",
    )
    dispatched: list[MaterialisationRequest] = []
    real_preview = region_fetch.materialise_preview

    def _spy(request, **kwargs):
        dispatched.append(request)
        return real_preview(request, **kwargs)

    monkeypatch.setattr(region_fetch, "materialise_preview", _spy)

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [entity]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        elected={_BARE_STAT_VAR: (combo,)},
        facet_filter={
            "measurementMethod": FacetAxisFilter(
                predicate=lambda facet: facet.measurement_method is None
            )
        },
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    candidates = [_facet_from_fixture(recorded), _facet_from_fixture(estimated)]
    assert [
        [c.facet_id for c in candidates if r.candidate_filter is not None and r.candidate_filter(c)]
        for r in dispatched
    ] == [["f-rec"]]
    assert warn is None
    assert observations is not None
    assert [p.value for s in observations.series for p in s.points] == [pytest.approx(100.0 * 0.9)]


def test_elected_dispatch_pair_missing_the_elected_facet_yields_one_series():
    """The pair with the wrong elected provenance is rejected; the matching
    pair converts and supplies the only series.
    """
    entity_with = "country/KEN"
    entity_without = "country/NGA"
    usd_fixture = _usd_fixture()
    other_source_fixture = _usd_fixture(provenance="provenance/SomeOtherSource")
    graph = FakeGraphReader(
        arcs=_recipe_arcs(),
        facets={
            (_BARE_STAT_VAR, entity_with): [_facet_from_fixture(usd_fixture)],
            (_BARE_STAT_VAR, entity_without): [_facet_from_fixture(other_source_fixture)],
        },
    )
    sdmx = _sdmx_router(
        rates=_USD_AND_EUR_RATES,
        observations={
            _BARE_STAT_VAR: _main_body(entity=entity_with, fixture=usd_fixture, value=100.0)
        },
    )
    elected = {_BARE_STAT_VAR: (_elected_combo(),)}
    unit_request = IdentityUnitRequest(currency="EUR", prices="current")

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [entity_with, entity_without]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=unit_request,
        elected=elected,
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    assert observations is not None
    assert len(observations.series) == 1
    assert observations.series[0].entities[0].ref == entity_with
    # One pair (entity_without) was rejected and one (entity_with)
    # contributed, which is the partial-rejection warning.
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE


def test_a_pair_outside_the_elected_unit_is_a_rejected_series_and_the_other_converts():
    """One variable serves one group at every place. A place holding only a facet in another
    unit is outside the elected unit's group: it is a rejected series with no read, and the
    other pair converts.
    """
    entity_with = "country/KEN"
    entity_unmapped = "country/NGA"
    usd_fixture = _usd_fixture()
    unmapped_fixture = ObsFacetFixture(
        facet_id="f2",
        points=(),
        provenance_id=_PROVENANCE,
        measurement_method=_MEASUREMENT_METHOD,
        unit="CurrencyXYZ",
        obs_count=1,
    )
    arcs = _recipe_arcs()
    arcs["CurrencyXYZ"] = {"typeOf": ("CurrencyUnit",)}  # no row in the rate table.
    graph = FakeGraphReader(
        arcs=arcs,
        facets={
            (_BARE_STAT_VAR, entity_with): [_facet_from_fixture(usd_fixture)],
            (_BARE_STAT_VAR, entity_unmapped): [_facet_from_fixture(unmapped_fixture)],
        },
    )
    sdmx = _sdmx_router(
        rates=_USD_AND_EUR_RATES,
        observations={_BARE_STAT_VAR: _main_body_for([(entity_with, usd_fixture, 100.0)])},
    )
    elected = {_BARE_STAT_VAR: (_elected_combo(),)}
    unit_request = IdentityUnitRequest(currency="EUR", prices="current")

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [entity_with, entity_unmapped]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=unit_request,
        elected=elected,
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    assert observations is not None
    assert len(observations.series) == 1
    assert observations.series[0].entities[0].ref == entity_with
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE
    assert warn.message.startswith("1 of 2 requested series")
    assert "currency conversion" not in warn.message


def test_elected_dispatch_pair_refusing_conversion_yields_one_series_and_warning():
    """Both places serve the elected US dollar facet, but one holds its point in a year the
    rate table lacks, so that pair's conversion refuses with `rate_missing` and the other
    pair converts.
    """
    entity_with = "country/KEN"
    entity_gap = "country/NGA"
    usd_fixture = _usd_fixture()
    graph = FakeGraphReader(
        arcs=_recipe_arcs(),
        facets={
            (_BARE_STAT_VAR, entity_with): [_facet_from_fixture(usd_fixture)],
            (_BARE_STAT_VAR, entity_gap): [_facet_from_fixture(usd_fixture)],
        },
    )
    body = _main_body_for([(entity_with, usd_fixture, 100.0)]) + _csv_row(
        stat_var=_BARE_STAT_VAR,
        role_dcids=[entity_gap],
        fixture=usd_fixture,
        date_str=str(int(_YEAR) - 1),
        value=50.0,
    )
    sdmx = _sdmx_router(rates=_USD_AND_EUR_RATES, observations={_BARE_STAT_VAR: body})

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [entity_with, entity_gap]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=IdentityUnitRequest(currency="EUR", prices="current"),
        elected={_BARE_STAT_VAR: (_elected_combo(),)},
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    assert observations is not None
    assert len(observations.series) == 1
    assert observations.series[0].entities[0].ref == entity_with
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE
    assert "rate_missing" in warn.message


def test_a_place_no_currency_names_is_left_out_of_a_destination_currency_preview():
    """Kenya and a region each read alone, converted into the place's own currency. The
    region's rows are omitted, which the answer's `currency_unmapped_omitted` warning reports;
    the preview serves Kenya's series and counts the region as neither refused nor failed."""
    region = "africaRegion"
    usd_fixture = _usd_fixture()
    arcs = conversion_arcs(
        recipe_id=_RECIPE,
        provenance_id=_PROVENANCE,
        currencies={"CurrencyUSD": (), "CurrencyKES": (_KEN,)},
    )
    arcs[_PROVENANCE]["isPartOf"] = ()
    arcs[_BARE_STAT_VAR] = dict(CONVERTIBLE_STAT_VAR_ARCS)
    graph = FakeGraphReader(
        arcs=arcs,
        facets={(_BARE_STAT_VAR, e): [_facet_from_fixture(usd_fixture)] for e in (_KEN, region)},
    )
    body = _main_body_for([(_KEN, usd_fixture, 100.0), (region, usd_fixture, 50.0)])
    sdmx = _sdmx_router(
        rates={"CurrencyUSD": {int(_YEAR): 1.0}, "CurrencyKES": {int(_YEAR): 112.0}},
        observations={_BARE_STAT_VAR: body},
    )

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [_KEN, region]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=IdentityUnitRequest(
            currency=IdentityCurrencyRole(role="observationAbout"), prices="current"
        ),
        elected={_BARE_STAT_VAR: (_elected_combo(),)},
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    assert [s.entities[0].ref for s in observations.series] == [_KEN]
    assert [p.value for p in observations.series[0].points] == [pytest.approx(11200.0)]
    assert warn is None


def test_elected_dispatch_every_pair_refusing_conversion_matches_facet_rejection_shape():
    """When both pairs carry a withdrawn currency, no series is returned and
    the warning names `rate_missing`.
    """
    entity_a = "country/KEN"
    entity_b = "country/NGA"
    withdrawn_fixture = ObsFacetFixture(
        facet_id="f1",
        points=(),
        provenance_id=_PROVENANCE,
        measurement_method=_MEASUREMENT_METHOD,
        unit="CurrencyXYZ",
        obs_count=1,
    )
    arcs = _recipe_arcs()
    arcs["CurrencyXYZ"] = {"typeOf": ("CurrencyUnit",)}  # no row in the rate table.
    graph = FakeGraphReader(
        arcs=arcs,
        facets={
            (_BARE_STAT_VAR, entity_a): [_facet_from_fixture(withdrawn_fixture)],
            (_BARE_STAT_VAR, entity_b): [_facet_from_fixture(withdrawn_fixture)],
        },
    )
    sdmx = _sdmx_router(
        rates=_USD_AND_EUR_RATES,
        observations={
            _BARE_STAT_VAR: _main_body_for(
                [(entity_a, withdrawn_fixture, 50.0), (entity_b, withdrawn_fixture, 50.0)]
            )
        },
    )
    elected = {_BARE_STAT_VAR: (_elected_combo(unit="CurrencyXYZ"),)}
    unit_request = IdentityUnitRequest(currency="EUR", prices="current")

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [entity_a, entity_b]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=unit_request,
        elected=elected,
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_NO_MATCHING_SOURCE
    assert warn is not None
    assert warn.code == OBSERVATIONS_NO_MATCHING_SOURCE
    assert warn.severity == "warn"
    assert "rate_missing" in warn.message


def test_no_unit_request_leaves_existing_fixture_unchanged():
    """Without unit selections, the preview leaves the unit axis unbound."""
    captured: list[MaterialisationRequest] = []

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
        captured.append(request)
        return stub_preview_result(sv=_BARE_STAT_VAR, entity="country/IND")

    with patch.object(region_fetch, "materialise_preview", _spy):
        observations, warn = build_observations_planned(
            [_BARE_STAT_VAR],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(),
            materialise_graph=harness_reader(["country/IND"], stat_var=_BARE_STAT_VAR),
            materialise_sdmx=sdmx_double(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert observations is not None
    assert len(captured) == 1
    assert captured[0].facet_bindings == {}
    # A natively served series (nothing converted) carries no `conversion`.
    assert observations.series[0].conversion is None


def test_elected_dispatch_fallback_entity_reports_count_and_example():
    """Without a deflator row for Kenya, the recipe substitutes
    `DACCountriesTotal` and records the fallback.
    """
    entity = "country/KEN"
    usd_fixture = _usd_fixture()
    graph = FakeGraphReader(
        arcs=_recipe_arcs(),
        facets={(_BARE_STAT_VAR, entity): [_facet_from_fixture(usd_fixture)]},
    )
    # `country/KEN` has no deflator row, so the `DACCountriesTotal` row stands in.
    sdmx = _sdmx_router(
        rates={
            "CurrencyUSD": {_BASE_YEAR: 1.0, int(_YEAR): 1.0},
            "CurrencyEUR": {_BASE_YEAR: 0.9},
        },
        deflators={_FALLBACK_ENTITY: {_BASE_YEAR: 90.0, int(_YEAR): 110.0}},
        observations={_BARE_STAT_VAR: _main_body(entity=entity, fixture=usd_fixture, value=100.0)},
    )
    elected = {_BARE_STAT_VAR: (_elected_combo(),)}
    unit_request = IdentityUnitRequest(currency="EUR", prices="constant", base_year=_BASE_YEAR)

    observations, warn = build_observations_planned(
        [_BARE_STAT_VAR],
        {"observationAbout": [entity]},
        graph=FakeGraph(),
        materialise_graph=graph,
        materialise_sdmx=sdmx,
        unit_request=unit_request,
        elected=elected,
        rate_cache=RateTableCache(),
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert observations is not None
    assert len(observations.series) == 1
    series = observations.series[0]
    assert series.conversion is not None
    assert series.conversion.fallback_count == 1
    assert series.conversion.fallback_examples == [
        SeriesConversionFallbackExample(entity=entity, fallback_entity=_FALLBACK_ENTITY)
    ]
    assert _SHA256_HEX.fullmatch(series.conversion.deflator_table_version)


def test_relaunch_closure_forwards_unit_request_and_elected_to_conversion():
    """The streaming relaunch applies the elected conversion. Changing the
    fake rate changes the value; omitting selections returns native points.
    """
    entity = "country/KEN"
    usd_fixture = _usd_fixture()
    elected = {_BARE_STAT_VAR: (_elected_combo(),)}
    unit_request = IdentityUnitRequest(currency="EUR", prices="current")

    async def _relaunch(*, rate: float, unit_request, elected):
        graph = FakeGraphReader(
            arcs=_recipe_arcs(),
            facets={(_BARE_STAT_VAR, entity): [_facet_from_fixture(usd_fixture)]},
        )
        sdmx = _sdmx_router(
            rates={"CurrencyUSD": {int(_YEAR): 1.0}, "CurrencyEUR": {int(_YEAR): rate}},
            observations={
                _BARE_STAT_VAR: _main_body(entity=entity, fixture=usd_fixture, value=100.0)
            },
        )
        async with scoped_concurrency():
            relaunch = PreviewReader(
                read=PreviewRead(
                    sv_dcids=(_BARE_STAT_VAR,),
                    obs_entities={"observationAbout": [entity]},
                    plan=pair_plan([_BARE_STAT_VAR], {"observationAbout": [entity]}, reader=graph),
                    facet_filter=None,
                    bulk_places=None,
                    member_facts=None,
                    unit_request=unit_request,
                    elected=elected,
                ),
                graph=FakeGraph(),
                sdmx=None,
                materialise_graph=graph,
                materialise_sdmx=sdmx,
                rate_cache=RateTableCache(),
                deadline=None,
                coverage=empty_coverage_snapshot(),
            ).relaunch(NO_LABELS)
            events, warns = await relaunch("spec-1")
        assert not warns
        assert len(events) == 1
        return events[0].observations.series[0]

    series = asyncio.run(_relaunch(rate=0.9, unit_request=unit_request, elected=elected))
    assert series.conversion is not None
    assert [p.value for p in series.points] == [pytest.approx(100.0 * 0.9 / 1.0)]

    # Perturb the fake rate: the converted value moves with it.
    series = asyncio.run(_relaunch(rate=0.5, unit_request=unit_request, elected=elected))
    assert series.conversion is not None
    assert [p.value for p in series.points] == [pytest.approx(100.0 * 0.5 / 1.0)]

    # Restore the original rate: the value returns.
    series = asyncio.run(_relaunch(rate=0.9, unit_request=unit_request, elected=elected))
    assert series.conversion is not None
    assert [p.value for p in series.points] == [pytest.approx(100.0 * 0.9 / 1.0)]

    # With both None, the relaunch yields the native points, unconverted.
    series = asyncio.run(_relaunch(rate=0.9, unit_request=None, elected=None))
    assert series.conversion is None
    assert [p.value for p in series.points] == [pytest.approx(100.0)]
