"""A role-bearing per-series preview with more reads than `obs_preview_concurrency` reads its
rows from SDMX in one bulk request per time period (`bulk_pair_reads.BulkPairSdmx`) and serves
the preview the per-series reads serve: the same series, values, order, facets, conversions,
omissions and warnings. Below the threshold, and for every request the bulk rows cannot serve,
each read requests its own rows.

The SDMX double answers as the mixer does: a request names any number of variables, filters
each role and the facets as a product, and `LATEST` keeps each series' newest row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import httpx
import pyarrow as pa
import pytest

from dc_sdmx import SdmxClient
from dc_sdmx.results import observation_schema
from materialise.facets import FacetSelection
from materialise.graph import Facet
from materialise.rate_tables import RateTableCache
from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.region.codes import OBSERVATIONS_PROBE_FAILED
from search.engine.region.observations.bulk_pair_reads import BulkPairSdmx, PlannedRead
from search.engine.region.observations.fetch import _build_observations
from search.engine.retrieval.read_plan import PairPlan, PlanRoles, PlanSeries, plan_pairs
from search.models import IdentityUnitRequest, SpecObservations, TimeWindow, Warning
from search.models.identity import IdentityCurrencyRole
from series_coverage.index import CoverageIndex
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
)
from tests.fixtures import FakeGraph
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_KEN, _UGA = "country/KEN", "country/UGA"
_CREDITORS = ("country/FRA", "country/NOR")
_ROLES = ("creditor", "debtor")
_PROVENANCE = "dc/base/ClimateTest"
_USD, _EUR, _KES = "CurrencyUSD", "CurrencyEUR", "CurrencyKES"
_RECIPE = RECIPE_BODIES["ConversionRecipeDAC"]
_RATES = _RECIPE.exchange_rate_variable
_YEARS = range(2010, 2025)
_RATE_BODIES = recipe_table_bodies(
    _RECIPE.recipe_id,
    rates={
        _USD: dict.fromkeys(_YEARS, 1.0),
        _EUR: dict.fromkeys(_YEARS, 0.9),
        _KES: dict.fromkeys(_YEARS, 130.0),
    },
)
_FAILING = "TESTFIXTURE/BulkPair/V03"
_MEMBERS = 12
"""Members, one read each: above a threshold of 2 and below one of 100."""


@dataclass(frozen=True)
class _Holding:
    """`years` of one facet at one series, each valued from the series and the year.
    `planned` is whether the read plan lists the facet among the series' candidates."""

    stat_var: str
    creditor: str
    debtor: str
    facet: ObsFacetFixture
    years: tuple[int, ...]
    planned: bool = True

    def value(self, year: int) -> float:
        key = self.stat_var + self.creditor + self.debtor + (self.facet.facet_id or "")
        seed = sum(map(ord, key))
        return float(seed % 97 * 10 + year % 7)


@dataclass
class _Mixer:
    """The SDMX `/data` rows of `holdings`, every request it served and the variables
    whose requests fail with a 500."""

    holdings: Sequence[_Holding]
    failing: frozenset[str] = frozenset()
    # A response to a request naming several variables states an extra role column
    # (`"extra_role"`) or no facet ids (`"no_facet_id"`).
    bulk_quirk: str | None = None
    data_requests: list[httpx.QueryParams] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        self.data_requests.append(params)
        variables = [v.removeprefix("dcid:") for v in params["c[variableMeasured]"].split(",")]
        if self.failing & set(variables):
            return httpx.Response(500, text="boom")
        role_filters = {
            role: params[f"c[{role}]"].split(",") for role in _ROLES if f"c[{role}]" in params
        }
        facets = params["c[facetId]"].split(",") if "c[facetId]" in params else None
        period = params.get("c[TIME_PERIOD]")
        rows: list[tuple[_Holding, int]] = []
        for holding in self.holdings:
            dcids = {"creditor": holding.creditor, "debtor": holding.debtor}
            if holding.stat_var not in variables or (
                facets is not None and holding.facet.facet_id not in facets
            ):
                continue
            if any(dcids[role] not in wanted for role, wanted in role_filters.items()):
                continue
            years = holding.years
            if period == "LATEST":
                years = years[-1:]
            elif period is not None:
                years = tuple(y for y in years if str(y) in period.split(","))
            rows.extend((holding, year) for year in years)
        quirk = self.bulk_quirk if len(variables) > 1 else None
        extra = ("observationAbout",) if quirk == "extra_role" else ()
        body = _csv_header((*_ROLES, *extra)) + "".join(
            _csv_row(
                stat_var=holding.stat_var,
                role_dcids=[holding.creditor, holding.debtor, *("NotApplicable" for _ in extra)],
                fixture=replace(holding.facet, facet_id=None)
                if quirk == "no_facet_id"
                else holding.facet,
                date_str=str(year),
                value=holding.value(year),
            )
            for holding, year in sorted(
                rows,
                key=lambda r: (
                    r[0].stat_var,
                    r[0].creditor,
                    r[0].debtor,
                    r[0].facet.facet_id or "",
                    r[1],
                ),
            )
        )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})


def _facet(fixture: ObsFacetFixture, years: Sequence[int]) -> Facet:
    dates = tuple(str(y) for y in years)
    return Facet(
        facet_id=fixture.facet_id,
        provenance_id=fixture.provenance_id,
        measurement_method=fixture.measurement_method,
        unit=fixture.unit,
        obs_count=len(dates),
        earliest_date=dates[0],
        latest_date=dates[-1],
        dates=dates,
        dates_complete=True,
    )


_DEFAULT = ObsFacetFixture(facet_id="f-default", provenance_id=_PROVENANCE, unit=_USD)
_SECOND_METHOD = ObsFacetFixture(
    facet_id="f-method", provenance_id=_PROVENANCE, unit=_USD, measurement_method="Estimated"
)


@dataclass(frozen=True)
class _Case:
    years: Any = None
    two_facets: bool = False
    failing: frozenset[str] = frozenset()
    unit_request: IdentityUnitRequest | None = None
    elect: Mapping[str, int] | None = None
    # The debtors the answer binds. Odd members hold no series at the second.
    debtors: tuple[str, ...] = (_KEN,)
    # Whether the first member's second-method rows are held by SDMX alone, outside the plan.
    unplanned_second_facet: bool = False


def _holdings(case: _Case) -> list[_Holding]:
    holdings: list[_Holding] = []
    for i in range(_MEMBERS):
        sv = f"TESTFIXTURE/BulkPair/V{i:02d}"
        for k, debtor in enumerate(case.debtors):
            if k and i % 2:
                continue
            for j, creditor in enumerate(_CREDITORS):
                end = 2020 + (i + j + k) % 4
                years = tuple(range(2012, end + 1))
                holdings.append(_Holding(sv, creditor, debtor, _DEFAULT, years))
                if case.two_facets:
                    holdings.append(
                        _Holding(
                            sv,
                            creditor,
                            debtor,
                            _SECOND_METHOD,
                            tuple(range(2016, end + 3)),
                            planned=not (case.unplanned_second_facet and i == 0),
                        )
                    )
    return holdings


def _plan(holdings: Sequence[_Holding], case: _Case) -> PairPlan:
    by_series: dict[tuple[str, str, str], list[Facet]] = {}
    for h in holdings:
        facets = by_series.setdefault((h.stat_var, h.creditor, h.debtor), [])
        if h.planned:
            facets.append(_facet(h.facet, h.years))
    members = list(dict.fromkeys(sv for sv, _c, _d in by_series))
    elect = case.elect
    return plan_pairs(
        [
            PlanSeries(
                stat_var=sv,
                role_tuple=(("creditor", creditor), ("debtor", debtor)),
                candidates=tuple(facets),
            )
            for (sv, creditor, debtor), facets in by_series.items()
        ],
        members=members,
        years=case.years,
        roles=PlanRoles(
            bound={"recipient": frozenset(case.debtors)},
            role_names={"donor": "creditor", "recipient": "debtor"},
        ),
        elect=None if elect is None else (lambda _plan: elect),
    )


def _materialise_graph(members: Sequence[str]) -> FakeGraphReader:
    return FakeGraphReader(
        arcs={
            **{sv: dict(CONVERTIBLE_STAT_VAR_ARCS) for sv in members},
            **conversion_arcs(
                recipe_id=_RECIPE.recipe_id,
                provenance_id=_PROVENANCE,
                currencies={_USD: ("country/USA",), _EUR: (), _KES: (_KEN,)},
            ),
        }
    )


def _elected(members: Sequence[str]) -> dict[str, tuple[Mapping[str, FacetSelection], ...]]:
    unit = FacetSelection(
        axis="unit",
        ref=_USD,
        basis="deployment_fallback",
        default_policy=None,
        default_policy_version=None,
        selection_reason="native_currency_preferred",
        dispatch_facet_id=_DEFAULT.facet_id,
    )
    return {sv: ({"unit": unit},) for sv in members}


def _preview(
    case: _Case, *, concurrency: int, search_settings
) -> tuple[SpecObservations, Warning | None, list[httpx.QueryParams]]:
    """The preview of `case` under a pool of `concurrency` workers, and the `/data`
    requests it made for observations."""
    search_settings(obs_preview_concurrency=concurrency)
    holdings = _holdings(case)
    plan = _plan(holdings, case)
    members = list(plan.members)
    mixer = _Mixer(holdings, failing=case.failing)
    server = RateTableServer(_RATE_BODIES, fallback=mixer.handle)
    index: CoverageIndex = build_coverage_index([], role_names={})
    conversion: dict[str, Any] = (
        {
            "unit_request": case.unit_request,
            "elected": _elected(members),
            "rate_cache": RateTableCache(),
        }
        if case.unit_request is not None
        else {}
    )
    observations, warn = _build_observations(
        plan,
        members,
        {"debtor": list(case.debtors)},
        graph=FakeGraph(),
        materialise_graph=_materialise_graph(members),
        materialise_sdmx=SdmxClient(base_url=MATERIALISE_BASE_URL, transport=server.transport()),
        coverage=FakeCoverageService(index).snapshot(),
        **conversion,
    )
    return observations, warn, mixer.data_requests


_CASES = {
    "dateless": _Case(),
    "window": _Case(years=TimeWindow(start_year=2015, end_year=2020)),
    "latest": _Case(years="latest"),
    "latest_elected": _Case(
        years="latest",
        elect={f"TESTFIXTURE/BulkPair/V{i:02d}": 2020 + i % 2 for i in range(_MEMBERS)},
    ),
    "two_facets": _Case(two_facets=True),
    "two_facets_latest": _Case(two_facets=True, years="latest"),
    "two_facets_continued": _Case(
        two_facets=True, years=TimeWindow(start_year=2015, end_year=2024)
    ),
    "second_facet_held_by_sdmx_alone": _Case(
        two_facets=True,
        years=TimeWindow(start_year=2015, end_year=2024),
        unplanned_second_facet=True,
    ),
    "two_debtors": _Case(debtors=(_KEN, _UGA)),
    "one_failing_member": _Case(failing=frozenset({_FAILING})),
    "unit_request": _Case(unit_request=IdentityUnitRequest(currency="EUR", prices="current")),
    "role_currency": _Case(
        unit_request=IdentityUnitRequest(
            currency=IdentityCurrencyRole(role="destination"), prices="current"
        )
    ),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_the_bulk_read_serves_the_preview_the_per_series_reads_serve(
    name: str, search_settings
) -> None:
    """Twelve members, each read with its creditor open and two series, preview alike from
    twelve per-series requests and from bulk rows, under every window, a member holding a
    second method's facet, a failing member, a currency conversion and a conversion into
    the debtor's own currency."""
    case = _CASES[name]
    per_series, per_series_warn, per_series_requests = _preview(
        case, concurrency=100, search_settings=search_settings
    )
    bulk, bulk_warn, bulk_requests = _preview(case, concurrency=2, search_settings=search_settings)

    assert per_series.series, per_series
    assert bulk.model_dump() == per_series.model_dump()
    assert bulk_warn == per_series_warn
    assert len(per_series_requests) >= _MEMBERS
    if case.failing:
        # The bulk request fails, and each read then requests its own rows.
        assert len(bulk_requests) == len(per_series_requests) + 1
        assert per_series_warn is not None and per_series_warn.code == OBSERVATIONS_PROBE_FAILED
    else:
        periods = {params.get("c[TIME_PERIOD]") for params in per_series_requests}
        assert len(bulk_requests) == len(periods)
        assert all(len(params["c[variableMeasured]"].split(",")) > 1 for params in bulk_requests)


def test_reads_at_the_threshold_request_their_own_rows(search_settings) -> None:
    """With as many reads as the pool has workers, each read requests its own rows."""
    _observations, _warn, requests = _preview(
        _Case(), concurrency=_MEMBERS, search_settings=search_settings
    )

    assert len(requests) == _MEMBERS
    assert all(len(params["c[variableMeasured]"].split(",")) == 1 for params in requests)


def _rows(sdmx: SdmxClient | BulkPairSdmx, read: PlannedRead, facet_ids: list[str], period):
    with sdmx.stream_data_batches(
        variables=[read.variable],
        entities={role: list(dcids) for role, dcids in read.bound},
        facet_ids=facet_ids,
        time_period=period,
    ) as (roles, batches):
        return set(roles), pa.Table.from_batches(list(batches), schema=observation_schema(roles))


@pytest.mark.parametrize("period", [None, "LATEST"], ids=["every_date", "latest"])
@pytest.mark.parametrize("quirk", [None, "extra_role", "no_facet_id"])
def test_each_read_gets_the_rows_its_own_request_returns(period, quirk) -> None:
    """Each planned read, the first member's dispatching its default facet alone and every
    other member's both facets, gets from the bulk rows the rows its own request returns,
    under one bulk request: the first member's second-method rows, which the bulk request
    returns for the other members' sake, are not its own. A bulk response naming a role
    the reads do not declare, or holding a row with no facet id, serves no read, and each
    read requests its own rows."""
    case = _Case(two_facets=True, debtors=(_KEN, _UGA))
    mixer = _Mixer(_holdings(case), bulk_quirk=quirk)
    inner = SdmxClient(base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(mixer.handle))
    reads = [
        PlannedRead(
            variable=f"TESTFIXTURE/BulkPair/V{i:02d}",
            bound=(("debtor", (debtor,)),),
            facet_ids=frozenset({"f-default"} if i == 0 else {"f-default", "f-method"}),
            declared_roles=frozenset(_ROLES),
            latest=period == "LATEST",
        )
        for i in range(_MEMBERS)
        for debtor in case.debtors
    ]
    bulk = BulkPairSdmx(inner, reads, deadline=None)

    for read in reads:
        facet_ids = sorted(read.facet_ids)
        served = _rows(bulk, read, facet_ids, period)
        before = len(mixer.data_requests)
        assert served == _rows(inner, read, facet_ids, period)
        del mixer.data_requests[before:]

    assert len(mixer.data_requests) == (1 if quirk is None else 1 + len(reads))
