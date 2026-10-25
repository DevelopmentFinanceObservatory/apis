"""The bound-unit gate's conversion coverage (`convertibility.unit_coverage`): every
row an answer's read plan serves must convert under its elected unit, found from the plan and
the rate tables before any observation is read, and from a series' dates when its facet states
only a span the rates do not cover."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType

import httpx
import pytest

from materialise import materialise_preview, member_conversion
from materialise.facets import FacetBinding, FacetSelection, UnitRequest
from materialise.member_conversion import (
    apply_member_conversion,
    omit_unmapped_rows,
    plan_member_conversion,
    refuse_if_every_row_omitted,
)
from materialise.pipeline import MaterialisationRequest, PreviewResult
from materialise.rate_tables import RateTableCache, RateTableInvalidError
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.recipes import RateCoverageGapError
from materialise.window import RequestedWindow
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import RoleFacetCandidate
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphDeadlineFailure, GraphInfraError
from search.engine.extract import DateRequest
from search.engine.region import convertibility
from search.engine.region.bound_unit_gate import bound_unit_gate_reason, plan_unit_evidence
from search.engine.region.convertibility import UnitCoverage, convertible_answer, unit_coverage
from search.engine.region.facets import FilteredAnswer
from search.engine.retrieval import read_plan
from search.engine.retrieval.coverage import CoverageForm, answer_coverage, answer_headline
from search.engine.retrieval.facet_election import RoleBulkCandidate
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import BulkPlan, PairPlan, PlanSeries, PlanYears, ReadPlan
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_resolve import _role_plan
from search.engine.unit.conversion_catalogue import ConversionCatalogue
from search.models import CoverageBare, CoverageExact, TimeWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise._table_support import table_from_cells
from tests.materialise.conftest import FakeGraphReader

_RECIPE = "ConversionRecipeDAC"
_PROVENANCE = "dc/base/Src"
_SV = "TEST/UnitCoverage_Flow"
_ROLES = ("financingProvider", "financingRecipient")
_DONOR = "country/FRA"
_KEN = "country/KEN"
_REGIONS = ("africaRegion", "southOfSahara")


def _graph() -> FakeGraphReader:
    """Conversion arcs for `_SV` under the DAC recipe; only Kenya has a national currency."""
    return FakeGraphReader(
        arcs={
            _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROVENANCE,
                currencies={
                    "CurrencyUSD": (),
                    "CurrencyEUR": (),
                    "CurrencyKES": (_KEN,),
                },
            ),
        }
    )


def _server(rates: Mapping[str, Mapping[int, float]]) -> RateTableServer:
    return RateTableServer(recipe_table_bodies(_RECIPE, rates=rates))


def _elected(currency: str) -> dict[str, tuple[Mapping[str, FacetSelection], ...]]:
    """`_SV`'s election: its euro facet under a current-price request for `currency`."""

    def selection(axis: str, ref: str, request: UnitRequest | None = None) -> FacetSelection:
        return FacetSelection(
            axis=axis,
            ref=ref,
            basis="bound",
            default_policy=None,
            default_policy_version=None,
            selection_reason=None,
            dispatch_facet_id="f1",
            request=request,
        )

    request = UnitRequest(currency=currency, prices="current")
    return {
        _SV: (
            {
                "unit": selection("unit", "CurrencyEUR", request),
                "provenance": selection("provenance", _PROVENANCE),
            },
        )
    }


def _facet(*years: int) -> Facet:
    return Facet(
        obs_count=len(years),
        earliest_date=str(min(years)),
        latest_date=str(max(years)),
        facet_id="f1",
        provenance_id=_PROVENANCE,
        unit="CurrencyEUR",
        dates=tuple(str(y) for y in years),
    )


def _pair_plan(
    *recipients: str,
    years: tuple[int, ...] = (2020, 2021),
    recipient_years: Mapping[str, tuple[int, ...]] | None = None,
    plan_years: PlanYears = None,
) -> PairPlan:
    """France's series to each recipient, holding `years`, or the recipient's own years in
    `recipient_years`, read under `plan_years`."""
    by_recipient = recipient_years or {}
    series = tuple(
        PlanSeries(
            stat_var=_SV,
            role_tuple=(("financingProvider", _DONOR), ("financingRecipient", r)),
            candidates=(_facet(*by_recipient.get(r, years)),),
            group=(_facet(*by_recipient.get(r, years)),),
        )
        for r in recipients
    )
    return PairPlan(members=(_SV,), years=plan_years, series=series)


def _refusal(plan, *, currency: str, rates) -> RateCoverageGapError | None:
    """The rate-gap refusal `plan` meets for `_SV`, or None when every row converts."""
    gap = unit_coverage(
        plan,
        _elected(currency),
        declared_roles=_ROLES,
        graph=_graph(),
        sdmx=_server(rates).client(),
        rate_cache=RateTableCache(),
    ).refusal
    if gap is None:
        return None
    sv, exc = gap
    assert sv == _SV
    assert isinstance(exc, RateCoverageGapError)
    return exc


_RATES = {"CurrencyUSD": {2020: 1.0, 2021: 1.0}, "CurrencyEUR": {2020: 0.9, 2021: 0.95}}


def test_a_covered_per_series_ask_converts_and_reads_no_observation():
    server = _server(_RATES)
    gap = unit_coverage(
        _pair_plan(_KEN),
        _elected("USD"),
        declared_roles=_ROLES,
        graph=_graph(),
        sdmx=server.client(),
        rate_cache=RateTableCache(),
    ).refusal
    assert gap is None
    # Only the rate table is read: its availability and data, nothing of `_SV`.
    assert {r.url.params.get("c[variableMeasured]") for r in server.requests} == {
        RECIPE_BODIES[_RECIPE].exchange_rate_variable
    }


def test_a_per_series_year_outside_the_rate_table_refuses_rate_missing():
    """The series serves 2020 and 2021; the table holds no euro rate for 2021."""
    rates = {"CurrencyUSD": {2020: 1.0, 2021: 1.0}, "CurrencyEUR": {2020: 0.9}}
    exc = _refusal(_pair_plan(_KEN), currency="USD", rates=rates)
    assert exc is not None
    assert exc.cause == "rate_missing"
    assert exc.entities == ("CurrencyEUR@2021",)


def _destination_coverage(plan) -> UnitCoverage:
    return unit_coverage(
        plan,
        _elected("destination"),
        declared_roles=_ROLES,
        graph=_graph(),
        sdmx=_server(_KES_RATES).client(),
        rate_cache=RateTableCache(),
    )


_KES_RATES = {**_RATES, "CurrencyKES": {2020: 100.0, 2021: 110.0}}
_TWELVE_REGIONS = tuple(f"region/R{i:02d}" for i in range(12))
# A catalogue holding a name for one omitted region only.
_NAMING_CATALOGUE = ConversionCatalogue(
    recipes=MappingProxyType({}),
    tables=MappingProxyType({}),
    national_currencies=MappingProxyType({}),
    default_base_year=2024,
    entity_names=MappingProxyType({"africaRegion": "Africa"}),
)


@pytest.mark.parametrize(
    ("regions", "named", "params"),
    [
        pytest.param(
            _REGIONS,
            "It left out 2 entities.",
            {
                "count": 2,
                "entities": list(_REGIONS),
                "entity_labels": ["Africa", ""],
            },
            id="two",
        ),
        pytest.param(
            _TWELVE_REGIONS,
            "It left out 12 entities.",
            {
                "count": 12,
                "entities": list(_TWELVE_REGIONS[:10]),
                "entity_labels": [""] * 10,
            },
            id="twelve",
        ),
    ],
)
def test_regional_recipients_under_a_destination_currency_are_omitted_and_counted(
    regions: tuple[str, ...], named: str, params: dict[str, object]
):
    """Regional recipients beside Kenya: no region has a national currency, so their rows are
    omitted, Kenya's convert, and the answer's warning counts every region. Its params carry
    the first ten with the names the conversion catalogue holds."""
    coverage = _destination_coverage(_pair_plan(_KEN, *regions))
    assert coverage.refusal is None
    assert coverage.omitted_entities == tuple(sorted(regions))
    warning = coverage.omitted_warning(catalogue=_NAMING_CATALOGUE, spec_id="spec-1")
    assert warning is not None
    assert (warning.code, warning.severity, warning.spec_id, warning.params) == (
        "currency_unmapped_omitted",
        "warn",
        "spec-1",
        params,
    )
    assert warning.message.endswith(named)


def test_only_regional_recipients_under_a_destination_currency_refuse_naming_each_one():
    coverage = _destination_coverage(_pair_plan(*_REGIONS))
    assert coverage.refusal is not None
    sv, exc = coverage.refusal
    assert sv == _SV
    assert isinstance(exc, RateCoverageGapError)
    assert (exc.cause, exc.entities, exc.entities_truncated) == (
        "currency_unmapped",
        _REGIONS,
        False,
    )
    assert coverage.omitted_warning(catalogue=None) is None


def test_regional_recipients_beside_rows_no_conversion_reads_are_omitted_without_refusing():
    """Kenya's series is served under a dollar facet the euro election does not convert, so
    its rows survive as served and the regions' omission warns rather than refuses."""
    kenya = PlanSeries(
        stat_var=_SV,
        role_tuple=(("financingProvider", _DONOR), ("financingRecipient", _KEN)),
        candidates=(replace(_facet(2020), unit="CurrencyUSD"),),
        group=(replace(_facet(2020), unit="CurrencyUSD"),),
    )
    regions = _pair_plan(*_REGIONS)
    coverage = _destination_coverage(replace(regions, series=(*regions.series, kenya)))
    assert coverage.refusal is None
    assert coverage.omitted_entities == _REGIONS


_CURRENT_DESTINATION = UnitRequest(currency="destination", prices="current")
_EVERY_YEAR = RequestedWindow(kind="all")


def _preview_of(
    pairs: tuple[tuple[str, str], ...],
    *,
    unit: UnitRequest = _CURRENT_DESTINATION,
    rates: Mapping[str, Mapping[int, float]] = _KES_RATES,
    deflators: Mapping[str, Mapping[int, float]] | None = None,
    recipient_years: Mapping[str, tuple[int, ...]] | None = None,
    window: RequestedWindow = _EVERY_YEAR,
) -> PreviewResult:
    """Datasets' read under `window` of one euro row per `(donor, recipient)` pair and year,
    2020 and 2021 or the recipient's own years in `recipient_years`, converted as `unit`
    asks."""
    by_recipient = recipient_years or {}

    def years_of(recipient: str) -> tuple[int, ...]:
        return by_recipient.get(recipient, (2020, 2021))

    donors = sorted({donor for donor, _r in pairs})
    recipients = sorted({recipient for _d, recipient in pairs})

    def role_rows(request: httpx.Request) -> httpx.Response:
        rows = "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{_SV},{donor},{r},CurrencyEUR,GrantEquivalent,"
            f"NotApplicable,{_PROVENANCE},{year},100,,f1\r\n"
            for donor, r in pairs
            for year in years_of(r)
        )
        return httpx.Response(
            200,
            text=_ROLE_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    server = RateTableServer(
        recipe_table_bodies(_RECIPE, rates=rates, deflators=deflators), fallback=role_rows
    )
    coverage = build_coverage_index(
        [(_SV, "f1", donor, r, list(years_of(r))) for donor, r in pairs],
        role_names={_SV: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        facet_metadata={
            "f1": FacetMetadata(
                provenance_id=_PROVENANCE, measurement_method="GrantEquivalent", unit="CurrencyEUR"
            )
        },
    )
    request = MaterialisationRequest(
        stat_var=f"dcid:{_SV}",
        role_bindings={"financingProvider": donors, "financingRecipient": recipients},
        declared_roles=_ROLES,
        facet_bindings={
            "provenance": FacetBinding(state="bound", ref=_PROVENANCE),
            "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
            "unit": FacetBinding(state="request", request=unit),
        },
        window=window,
    )
    return materialise_preview(
        request,
        sdmx=server.client(),
        graph=_ssp_graph(),
        coverage=coverage,
        limit=100,
        rate_cache=RateTableCache(),
    )


_ROLE_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


_LATEST = RequestedWindow(kind="latest")


@pytest.mark.parametrize(
    ("window", "plan_years", "years", "bulk"),
    [
        pytest.param(_EVERY_YEAR, None, {}, False, id="all"),
        pytest.param(_LATEST, "latest", {_KEN: (2020,)}, False, id="latest-regions-newest"),
        pytest.param(
            _LATEST, "latest", dict.fromkeys(_REGIONS, (2020,)), False, id="latest-kenya-newest"
        ),
        pytest.param(_LATEST, "latest", {_KEN: (2020,)}, True, id="bulk-latest-regions-newest"),
    ],
)
def test_datasets_and_search_agree_on_the_omitted_recipients(
    window: RequestedWindow,
    plan_years: PlanYears,
    years: Mapping[str, tuple[int, ...]],
    bulk: bool,
    search_settings,
):
    """One fixture through both surfaces: Datasets serves Kenya's rows with the warning
    Search's gate attaches, and both refuse `currency_unmapped` when no recipient maps.
    Under `latest` Datasets omits the regions' rows before it picks the member's latest
    period, so it serves Kenya's latest row and names the regions however their years
    fall; a bulk plan, past the pair cap, omits before it takes its latest year too."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)

    def plan_of(*of: str) -> ReadPlan:
        if not bulk:
            return _pair_plan(*of, recipient_years=years, plan_years=plan_years)
        plan, _index = _recipient_plan(of, recipient_years=years, latest=True)
        assert isinstance(plan, BulkPlan)
        return plan

    recipients = (_KEN, *_REGIONS)
    preview = _preview_of(
        tuple((_DONOR, r) for r in recipients), recipient_years=years, window=window
    )
    search = _destination_coverage(plan_of(*recipients)).omitted_warning(catalogue=None)
    assert search is not None
    (datasets,) = [w for w in preview.warnings if w["code"] == "currency_unmapped_omitted"]
    assert (datasets["severity"], datasets["message"]) == (search.severity, search.message)
    assert set(preview.frame.table["financingRecipient"].to_pylist()) == {_KEN}

    with pytest.raises(RateCoverageGapError) as datasets_refusal:
        _preview_of(tuple((_DONOR, r) for r in _REGIONS), window=window)
    search_refusal = _destination_coverage(plan_of(*_REGIONS)).refusal
    assert search_refusal is not None
    assert isinstance(search_refusal[1], RateCoverageGapError)
    assert (datasets_refusal.value.cause, datasets_refusal.value.entities) == (
        search_refusal[1].cause,
        search_refusal[1].entities,
    )


def test_a_latest_per_series_plan_omits_before_it_takes_the_latest_period():
    """Kenya's series ends in 2020 and a region's, which no currency names, in 2021. The read
    omits the region before it takes the StatVar's latest period, so the answer and the gate
    both serve Kenya's 2020 series and name the region, where taking the period first would
    serve only the region and refuse."""
    plan = _pair_plan(
        _KEN,
        _REGIONS[0],
        recipient_years={_KEN: (2020,), _REGIONS[0]: (2021,)},
        plan_years="latest",
    )
    kenya = tuple(s for s in plan.series if ("financingRecipient", _KEN) in s.role_tuple)
    omitted = frozenset({(_SV, "financingRecipient", _REGIONS[0])})

    coverage = _destination_coverage(plan)
    gate = convertibility.unit_gate(
        Materialised(
            sv_dcids=[_SV],
            has_data=True,
            coverage=CoverageBare(has_data=True),
            read_plan=plan,
            member_facts={_SV: ArcsMemberFacts(arcs=_DECLARED_ROLE_ARCS)},
        ),
        plan,
        _elected("destination"),
        {"financingRecipient": [_KEN], "financingProvider": [_DONOR]},
        graph=_graph(),
        deadline=None,
    )

    assert coverage.refusal is None
    assert coverage.omitted == omitted
    assert isinstance(gate.plan, PairPlan)
    assert gate.plan.served == kenya
    assert gate.plan.omitted == omitted
    assert gate.reason is None


def _recipient_plan(
    recipients: tuple[str, ...],
    *,
    recipient_years: Mapping[str, tuple[int, ...]] | None = None,
    latest: bool = False,
) -> tuple[ReadPlan, CoverageIndex]:
    """France's euro flows to each of `recipients` in 2020 and 2021, or the recipient's own
    years in `recipient_years`, planned as resolve plans them from coverage, for every date
    or with `latest` the latest: per series, or in bulk past the pair cap."""
    by_recipient = recipient_years or {}
    index = build_coverage_index(
        [(_SV, "f1", _DONOR, r, list(by_recipient.get(r, (2020, 2021)))) for r in recipients],
        role_names={_SV: RoleNames(donor="financingProvider", recipient="financingRecipient")},
        facet_metadata={
            "f1": FacetMetadata(
                provenance_id=_PROVENANCE, measurement_method="GrantEquivalent", unit="CurrencyEUR"
            )
        },
    )
    plan = _role_plan(
        [_SV],
        member_facts={},
        names={"destination": "financingRecipient", "origin": "financingProvider"},
        subject_dcids=list(recipients),
        origin_dcids=[_DONOR],
        bound_roles={"recipient": frozenset(recipients), "donor": frozenset({_DONOR})},
        years=read_plan.role_years(
            DateRequest(window=None, latest=True) if latest else None, index.horizon
        ),
        coverage_index=index,
    )
    return plan, index


@pytest.mark.parametrize("bulk", [False, True], ids=["per-series", "bulk"])
def test_an_answer_states_only_the_rows_left_once_unmapped_recipients_are_omitted(
    bulk: bool, search_settings
):
    """Kenya and two regions under a destination currency: Datasets serves Kenya's two rows.
    The answer's plan, coverage and headline state those rows alone, and name each region
    as unserved on the destination role, so the answer's `entity_unavailable` departure
    names them."""
    if bulk:
        search_settings(obs_preview_max_pairs=1)
    recipients = (_KEN, *_REGIONS)
    plan, index = _recipient_plan(recipients)
    assert isinstance(plan, BulkPlan if bulk else PairPlan)
    form = CoverageForm()
    mat = Materialised(
        sv_dcids=[_SV],
        has_data=True,
        coverage=answer_coverage(plan, form),
        coverage_form=form,
        read_plan=plan,
        headline=answer_headline(plan, index, date_request=None),
        member_facts={_SV: ArcsMemberFacts(arcs=_DECLARED_ROLE_ARCS)},
    )

    answer, coverage = convertible_answer(
        mat,
        FilteredAnswer(mat.coverage, mat.headline, plan),
        _elected("destination"),
        graph=_graph(),
        sdmx=_server(_KES_RATES).client(),
        rate_cache=RateTableCache(),
        dated_facets=None,
        deadline=None,
        date_request=None,
        coverage_index=index,
    )

    preview = _preview_of(tuple((_DONOR, r) for r in recipients))
    served = set(preview.frame.table["financingRecipient"].to_pylist())
    assert coverage.omitted_entities == _REGIONS
    assert isinstance(answer.coverage, CoverageExact)
    assert answer.coverage.observation_count == preview.total_rows == 2
    assert answer.headline is not None
    assert answer.headline.role_counts["financingRecipient"] == len(served) == 1
    assert answer.headline.observations == 1
    assert answer.plan is not None
    assert read_plan.unserved_entities(answer.plan) == {("destination", r) for r in _REGIONS}
    assert plan_unit_evidence(answer.plan).covers([_SV])


_DECLARED_ROLE_ARCS = {"observationProperties": {"nodes": [{"dcid": role} for role in _ROLES]}}

SECOND_FLOW = "TEST/UnitCoverage_Flow2"


def two_flow_graph() -> FakeGraphReader:
    """`_graph` with `SECOND_FLOW` beside `_SV` under the same recipe."""
    return FakeGraphReader(
        arcs={
            _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            SECOND_FLOW: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROVENANCE,
                currencies={"CurrencyUSD": (), "CurrencyEUR": (), "CurrencyKES": (_KEN,)},
            ),
        }
    )


def two_flow_elected() -> dict[str, tuple[Mapping[str, FacetSelection], ...]]:
    """Each flow's euro facet under a destination-currency request."""
    (combo,) = _elected("destination")[_SV]
    return {_SV: (combo,), SECOND_FLOW: (combo,)}


def two_flow_bulk_plan(
    years: Mapping[str, Mapping[str, tuple[int, ...]]],
) -> tuple[ReadPlan, Materialised]:
    """France's euro flows by StatVar and recipient in `years`, planned under `latest` as
    resolve plans them, with the carrier a gate reads."""
    rows = [
        (sv, "f1", _DONOR, recipient, list(held))
        for sv, by_recipient in years.items()
        for recipient, held in by_recipient.items()
    ]
    index = build_coverage_index(
        rows,
        role_names={
            sv: RoleNames(donor="financingProvider", recipient="financingRecipient") for sv in years
        },
        facet_metadata={
            "f1": FacetMetadata(
                provenance_id=_PROVENANCE, measurement_method="GrantEquivalent", unit="CurrencyEUR"
            )
        },
    )
    recipients = sorted({r for by_recipient in years.values() for r in by_recipient})
    plan = _role_plan(
        list(years),
        member_facts={},
        names={"destination": "financingRecipient", "origin": "financingProvider"},
        subject_dcids=recipients,
        origin_dcids=[_DONOR],
        bound_roles={"recipient": frozenset(recipients), "donor": frozenset({_DONOR})},
        years=read_plan.role_years(DateRequest(window=None, latest=True), index.horizon),
        coverage_index=index,
    )
    mat = Materialised(
        sv_dcids=list(years),
        has_data=True,
        coverage=CoverageBare(has_data=True),
        read_plan=plan,
        member_facts={sv: ArcsMemberFacts(arcs=_DECLARED_ROLE_ARCS) for sv in years},
    )
    return plan, mat


def test_a_bulk_latest_plan_confirms_each_member_at_its_own_year(search_settings):
    """The first flow gives Kenya 2020 and a region 2019, the second Kenya 2021. A bulk
    `latest` plan serves each flow its own latest year, 2020 and 2021, as a mint of each
    does, so the gate confirms both with or without the region. The region's omitted rows
    fall outside the first flow's year, so they confirm nothing."""
    search_settings(obs_preview_max_pairs=1)
    obs = {"financingRecipient": [_KEN], "financingProvider": [_DONOR]}
    for years in (
        {_SV: {_KEN: (2020,)}, SECOND_FLOW: {_KEN: (2021,)}},
        {_SV: {_KEN: (2020,), _REGIONS[0]: (2019,)}, SECOND_FLOW: {_KEN: (2021,)}},
    ):
        plan, _mat = two_flow_bulk_plan(years)
        assert isinstance(plan, BulkPlan)
        coverage = unit_coverage(
            plan,
            two_flow_elected(),
            declared_roles=_ROLES,
            graph=two_flow_graph(),
            sdmx=_server(_KES_RATES).client(),
            rate_cache=RateTableCache(),
        )
        narrowed = plan.omitting(coverage.omitted)
        assert narrowed.member_years(_SV) == (2020, 2020)
        assert narrowed.member_years(SECOND_FLOW) == (2021, 2021)
        assert narrowed.omitted_served == frozenset()
        assert plan_unit_evidence(narrowed).confirmed == {_SV, SECOND_FLOW}
        assert bound_unit_gate_reason(narrowed, obs, [_SV, SECOND_FLOW]) is None


_PairRow = tuple[str, str, int, int]
"""A `(donor, recipient)` tuple with the first and last year it holds data in."""
_ROLE_COLUMN = {"financingProvider": 0, "financingRecipient": 1}


@dataclass(frozen=True)
class _Pairs:
    """A bulk facet's pairs as the coverage index answers for them: one row per
    `(donor, recipient)` tuple, holding data in every year from its first to its last."""

    rows: tuple[_PairRow, ...]

    def head(self, limit, *, first_year, last_year, with_data=False):
        raise AssertionError("the coverage check lists no pair")

    def n_obs(self, *, first_year, last_year) -> int:
        return 0

    def n_with_data(self, *, first_year, last_year) -> int:
        return len(list(self._clipped(first_year, last_year)))

    def max_n_obs(self, *, first_year, last_year) -> int:
        return 0

    def _clipped(self, first_year, last_year):
        for donor, recipient, first, last in self.rows:
            lo = max(first, first_year) if first_year is not None else first
            hi = min(last, last_year) if last_year is not None else last
            if lo <= hi:
                yield donor, recipient, lo, hi

    def year_span(self, *, first_year, last_year) -> tuple[int, int] | None:
        clipped = list(self._clipped(first_year, last_year))
        if not clipped:
            return None
        return min(lo for *_, lo, _hi in clipped), max(hi for *_, hi in clipped)

    def members_with_data(
        self, *, first_year, last_year, role: str | None = None
    ) -> frozenset[str]:
        assert role is not None, "the coverage check names the role"
        column = _ROLE_COLUMN[role]
        return frozenset(row[column] for row in self._clipped(first_year, last_year))

    def excluding(self, *, role, dcids) -> _Pairs:
        column = _ROLE_COLUMN[role]
        return _Pairs(tuple(row for row in self.rows if row[column] not in dcids))


def _bulk_candidate(rows: tuple[_PairRow, ...]) -> RoleFacetCandidate:
    return RoleFacetCandidate(
        facet_id="f1",
        members_by_role={
            "financingProvider": frozenset(donor for donor, *_rest in rows),
            "financingRecipient": frozenset(recipient for _donor, recipient, *_years in rows),
        },
        pair_count=len(rows),
        latest_date=str(max(last for *_, last in rows)),
        pairs=_Pairs(rows),
        provenance_id=_PROVENANCE,
        unit="CurrencyEUR",
    )


def _bulk_plan_of(
    by_sv: Mapping[str, tuple[_PairRow, ...]], *, years: PlanYears = None
) -> BulkPlan:
    """A bulk plan over `years` electing facet `f1`, each StatVar's pairs given by row."""
    candidates = {sv: [_bulk_candidate(rows)] for sv, rows in by_sv.items()}
    bulk = RoleBulkCandidate(
        facet_id="f1",
        by_sv=candidates,
        pair_count=len(candidates),
        latest_date=max(c[0].latest_date or "" for c in candidates.values()),
        provenance_id=_PROVENANCE,
        unit="CurrencyEUR",
    )
    return BulkPlan(members=tuple(by_sv), years=years, facet_id="f1", candidates={"f1": bulk})


def _bulk_plan(span: tuple[int, int]) -> BulkPlan:
    return _bulk_plan_of({_SV: ((_DONOR, _KEN, *span),)})


def test_a_bulk_year_outside_the_rate_table_refuses_rate_missing():
    """The bulk facet's pairs run 2020 to 2021; the table holds no euro rate for 2021."""
    rates = {"CurrencyUSD": {2020: 1.0, 2021: 1.0}, "CurrencyEUR": {2020: 0.9}}
    exc = _refusal(_bulk_plan((2020, 2021)), currency="USD", rates=rates)
    assert exc is not None
    assert exc.cause == "rate_missing"
    assert exc.entities == ("CurrencyEUR@2021",)


def test_a_covered_bulk_ask_converts():
    assert _refusal(_bulk_plan((2020, 2021)), currency="USD", rates=_RATES) is None


@pytest.mark.parametrize("plan", [_pair_plan(_KEN), _bulk_plan((2020, 2021))], ids=["pair", "bulk"])
def test_a_natively_satisfied_unit_expands_no_row_and_loads_no_table(plan, monkeypatch):
    expanded: list[object] = []

    def counted(*args):
        expanded.append(args)
        return set()

    monkeypatch.setattr(convertibility, "_pair_rows", counted)
    monkeypatch.setattr(convertibility, "_bulk_member_rows", counted)
    server = _server(_RATES)
    gap = unit_coverage(
        plan,
        _elected("EUR"),
        declared_roles=_ROLES,
        graph=_graph(),
        sdmx=server.client(),
        rate_cache=RateTableCache(),
    ).refusal
    assert gap is None
    assert expanded == []
    assert server.requests == []


_SSD = "country/SSD"
_TWO_RANGES = ((_DONOR, _KEN, 2018, 2021), (_DONOR, _SSD, 2020, 2021))
"""Kenya's rows run 2018 to 2021; South Sudan's start in 2020."""


def _ssp_graph() -> FakeGraphReader:
    """`_graph` with South Sudan's pound as its national currency."""
    return FakeGraphReader(
        arcs={
            _SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROVENANCE,
                currencies={
                    "CurrencyUSD": (),
                    "CurrencyEUR": (),
                    "CurrencyKES": (_KEN,),
                    "CurrencySSP": (_SSD,),
                },
            ),
        }
    )


def _ssp_rates(ssp_years: range) -> dict[str, dict[int, float]]:
    span = range(2018, 2022)
    return {
        "CurrencyUSD": dict.fromkeys(span, 1.0),
        "CurrencyEUR": dict.fromkeys(span, 0.9),
        "CurrencyKES": dict.fromkeys(span, 100.0),
        "CurrencySSP": dict.fromkeys(ssp_years, 300.0),
    }


def _ssp_refusal(plan, *, unit_request: UnitRequest, rates, deflators=None):
    return unit_coverage(
        plan,
        {
            _SV: (
                {
                    "unit": _selection("unit", "CurrencyEUR", unit_request),
                    "provenance": _selection("provenance", _PROVENANCE),
                },
            )
        },
        declared_roles=_ROLES,
        graph=_ssp_graph(),
        sdmx=RateTableServer(
            recipe_table_bodies(_RECIPE, rates=rates, deflators=deflators)
        ).client(),
        rate_cache=RateTableCache(),
    ).refusal


def _selection(axis: str, ref: str, request: UnitRequest | None = None) -> FacetSelection:
    return FacetSelection(
        axis=axis,
        ref=ref,
        basis="bound",
        default_policy=None,
        default_policy_version=None,
        selection_reason=None,
        dispatch_facet_id="f1",
        request=request,
    )


def test_a_bulk_member_is_checked_only_in_the_years_its_own_rows_reach():
    """South Sudan's rows start in 2020 and its pound's rates do too. The read never serves
    South Sudan before 2020, so the destination-currency ask converts."""
    gap = _ssp_refusal(
        _bulk_plan_of({_SV: _TWO_RANGES}),
        unit_request=UnitRequest(currency="destination", prices="current"),
        rates=_ssp_rates(range(2020, 2022)),
    )
    assert gap is None


def test_a_gap_inside_a_bulk_members_own_years_still_refuses():
    """The pound's 2020 rate is missing, a year South Sudan's rows reach."""
    gap = _ssp_refusal(
        _bulk_plan_of({_SV: _TWO_RANGES}),
        unit_request=UnitRequest(currency="destination", prices="current"),
        rates=_ssp_rates(range(2021, 2022)),
    )
    assert gap is not None
    exc = gap[1]
    assert isinstance(exc, RateCoverageGapError)
    assert (exc.cause, exc.entities) == ("rate_missing", ("CurrencySSP@2020",))


_SHORT_DONOR = "country/NOR"
_DONOR_RANGES = ((_DONOR, _KEN, 2018, 2021), (_SHORT_DONOR, _KEN, 2020, 2021))
"""France gives from 2018; Norway's rows start in 2020."""


def _deflators(short_years: range) -> dict[str, dict[int, float]]:
    return {
        _DONOR: dict.fromkeys(range(2018, 2022), 100.0),
        _SHORT_DONOR: dict.fromkeys(short_years, 100.0),
    }


def test_a_constant_price_bulk_ask_deflates_each_donor_only_in_its_own_years():
    """The DAC deflator role is the donor. Norway's deflator starts in 2020, as its rows do,
    so the constant-price ask converts."""
    gap = _ssp_refusal(
        _bulk_plan_of({_SV: _DONOR_RANGES}),
        unit_request=UnitRequest(currency="USD", prices="constant", base_year=2021),
        rates=_ssp_rates(range(2020, 2022)),
        deflators=_deflators(range(2020, 2022)),
    )
    assert gap is None


@pytest.mark.parametrize(
    ("ranges", "refused"),
    [
        pytest.param(_DONOR_RANGES, None, id="other-rows-survive"),
        pytest.param(
            ((_SHORT_DONOR, _KEN, 2020, 2020),),
            ("deflator_missing", (f"{_SHORT_DONOR}@2020",)),
            id="no-row-survives",
        ),
    ],
)
def test_a_deflator_gap_inside_a_donors_own_years_leaves_out_its_rows(ranges, refused):
    """Norway's deflator lacks 2020, inside its own years. The read leaves its 2020 row out
    and serves the rest; only a plan left with no row refuses."""
    gap = _ssp_refusal(
        _bulk_plan_of({_SV: ranges}),
        unit_request=UnitRequest(currency="USD", prices="constant", base_year=2021),
        rates=_ssp_rates(range(2020, 2022)),
        deflators=_deflators(range(2021, 2022)),
    )
    if refused is None:
        assert gap is None
        return
    assert gap is not None
    exc = gap[1]
    assert isinstance(exc, RateCoverageGapError)
    assert (exc.cause, exc.entities) == refused


_NOR_UNDEFLATED = {_DONOR: {2020: 100.0, 2021: 100.0}}
"""France's deflator for 2020 and 2021; Norway has none."""
_CONSTANT_DESTINATION = UnitRequest(currency="destination", prices="constant", base_year=2021)


def _flows_plan(pairs: tuple[tuple[str, str], ...]) -> PairPlan:
    """One series per `(donor, recipient)` pair, observed in 2020 and 2021."""
    series = tuple(
        PlanSeries(
            stat_var=_SV,
            role_tuple=(("financingProvider", donor), ("financingRecipient", recipient)),
            candidates=(_facet(2020, 2021),),
            group=(_facet(2020, 2021),),
        )
        for donor, recipient in pairs
    )
    return PairPlan(members=(_SV,), years=None, series=series)


def test_a_donor_whose_rows_all_went_to_omitted_recipients_needs_no_deflator():
    """France gives to Kenya, Norway only to a region no currency names. The read omits the
    region's rows, so it never deflates Norway: it serves Kenya's rows with the warning, and
    the gate answers the same way however the plan's other entities are pooled."""
    pairs = ((_DONOR, _KEN), (_SHORT_DONOR, _REGIONS[0]))
    coverage = unit_coverage(
        _flows_plan(pairs),
        {
            _SV: (
                {
                    "unit": _selection("unit", "CurrencyEUR", _CONSTANT_DESTINATION),
                    "provenance": _selection("provenance", _PROVENANCE),
                },
            )
        },
        declared_roles=_ROLES,
        graph=_ssp_graph(),
        sdmx=RateTableServer(
            recipe_table_bodies(
                _RECIPE, rates=_ssp_rates(range(2018, 2022)), deflators=_NOR_UNDEFLATED
            )
        ).client(),
        rate_cache=RateTableCache(),
    )
    assert coverage == UnitCoverage(omitted=frozenset({(_SV, "financingRecipient", _REGIONS[0])}))

    preview = _preview_of(
        pairs,
        unit=_CONSTANT_DESTINATION,
        rates=_ssp_rates(range(2018, 2022)),
        deflators=_NOR_UNDEFLATED,
    )
    assert set(preview.frame.table["financingRecipient"].to_pylist()) == {_KEN}
    assert {w["code"] for w in preview.warnings} >= {"currency_unmapped_omitted"}


def _constant_destination_coverage(plan: BulkPlan) -> UnitCoverage:
    """`plan`'s coverage under the constant-price destination-currency ask, with every rate
    and France's deflator alone."""
    return unit_coverage(
        plan,
        {
            _SV: (
                {
                    "unit": _selection("unit", "CurrencyEUR", _CONSTANT_DESTINATION),
                    "provenance": _selection("provenance", _PROVENANCE),
                },
            )
        },
        declared_roles=_ROLES,
        graph=_ssp_graph(),
        sdmx=RateTableServer(
            recipe_table_bodies(
                _RECIPE, rates=_ssp_rates(range(2018, 2022)), deflators=_NOR_UNDEFLATED
            )
        ).client(),
        rate_cache=RateTableCache(),
    )


def test_a_bulk_read_needs_no_deflator_for_a_donor_whose_rows_all_went_to_omitted_recipients():
    """France gives to Kenya, Norway only to a region no currency names. The bulk plan omits
    the region's tuples before it checks, so no year's donors include Norway, and the gate
    serves Kenya's rows with the warning, as Datasets does."""
    pairs = ((_DONOR, _KEN), (_SHORT_DONOR, _REGIONS[0]))
    coverage = _constant_destination_coverage(
        _bulk_plan_of({_SV: tuple((*pair, 2020, 2021) for pair in pairs)})
    )
    assert coverage == UnitCoverage(omitted=frozenset({(_SV, "financingRecipient", _REGIONS[0])}))

    preview = _preview_of(
        pairs,
        unit=_CONSTANT_DESTINATION,
        rates=_ssp_rates(range(2018, 2022)),
        deflators=_NOR_UNDEFLATED,
    )
    assert set(preview.frame.table["financingRecipient"].to_pylist()) == {_KEN}
    assert {w["code"] for w in preview.warnings} >= {"currency_unmapped_omitted"}


def test_a_windowed_bulk_read_omits_only_the_recipients_its_window_serves():
    """Under 2020 to 2021, Norway gives only to one region and France gives to another only
    in 2018. The read omits the first region's rows and never fetches the second's, so the
    warning names the first alone and Norway needs no deflator."""
    window = TimeWindow(start_year=2020, end_year=2021)
    rows = (
        (_DONOR, _KEN, 2020, 2021),
        (_SHORT_DONOR, _REGIONS[0], 2020, 2021),
        (_DONOR, _REGIONS[1], 2018, 2018),
    )
    coverage = _constant_destination_coverage(_bulk_plan_of({_SV: rows}, years=window))
    assert coverage == UnitCoverage(omitted=frozenset({(_SV, "financingRecipient", _REGIONS[0])}))

    preview = _preview_of(
        tuple((donor, recipient) for donor, recipient, *_years in rows),
        unit=_CONSTANT_DESTINATION,
        rates=_ssp_rates(range(2018, 2022)),
        deflators=_NOR_UNDEFLATED,
        recipient_years={_REGIONS[1]: (2018,)},
        window=RequestedWindow(kind="range", start_year=2020, end_year=2021),
    )
    assert set(preview.frame.table["financingRecipient"].to_pylist()) == {_KEN}
    (warning,) = (w for w in preview.warnings if w["code"] == "currency_unmapped_omitted")
    assert _REGIONS[0] in str(warning)
    assert _REGIONS[1] not in str(warning)


def test_a_bulk_donor_with_a_surviving_row_loses_it_to_its_missing_deflator():
    """Norway gives to the omitted region and to Kenya. Its Kenya rows survive the omission,
    so the read deflates Norway in 2020, finds no deflator and leaves those rows out. France's
    rows convert, so the answer stands; with Norway alone it refuses."""
    norway = ((_SHORT_DONOR, _KEN, 2020, 2020), (_SHORT_DONOR, _REGIONS[0], 2020, 2020))
    served = _constant_destination_coverage(
        _bulk_plan_of({_SV: ((_DONOR, _KEN, 2020, 2020), *norway)})
    )
    assert served.refusal is None
    gap = _constant_destination_coverage(_bulk_plan_of({_SV: norway})).refusal
    assert gap is not None
    exc = gap[1]
    assert isinstance(exc, RateCoverageGapError)
    assert exc.cause == "deflator_missing"


def _latest_series(*facets: Facet) -> PairPlan:
    series = PlanSeries(
        stat_var=_SV,
        role_tuple=(("financingProvider", _DONOR), ("financingRecipient", _KEN)),
        candidates=facets,
        group=facets,
    )
    return PairPlan(members=(_SV,), years="latest", series=(series,))


def test_a_latest_read_checks_only_the_facets_holding_the_groups_latest_year():
    """A latest read serves the group's latest year, 2021. The second facet, under another
    provenance with no recipe, ends in 2019 and serves no row, so it needs no check."""
    other = Facet(
        obs_count=1,
        earliest_date="2019",
        latest_date="2019",
        facet_id="f2",
        provenance_id="dc/base/NoRecipe",
        unit="CurrencyEUR",
        dates=("2019",),
    )
    plan = _latest_series(_facet(2020, 2021), other)
    gap = unit_coverage(
        plan,
        _elected("USD"),
        declared_roles=_ROLES,
        graph=_graph(),
        sdmx=_server(_RATES).client(),
        rate_cache=RateTableCache(),
    ).refusal
    assert gap is None


def test_a_wide_bulk_ask_over_many_statvars_runs_one_check(monkeypatch):
    """150 recipients over 30 years under 40 StatVars that share one facet and recipe make
    one coverage check over their distinct rows, one block per year, not one per StatVar."""
    rows = tuple((_DONOR, f"country/R{i:03d}", 1990, 2019) for i in range(150))
    by_sv = {f"{_SV}_{n}": rows for n in range(40)}
    checks: list[int] = []
    real = convertibility.require_decision_coverage

    def counted(decision, *, served, **kwargs):
        checks.append(len(served))
        return real(decision, served=served, **kwargs)

    monkeypatch.setattr(convertibility, "require_decision_coverage", counted)
    graph = FakeGraphReader(
        arcs={
            **{sv: dict(CONVERTIBLE_STAT_VAR_ARCS) for sv in by_sv},
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROVENANCE,
                currencies={"CurrencyUSD": (), "CurrencyEUR": ()},
            ),
        }
    )
    elected = {sv: _elected("USD")[_SV] for sv in by_sv}
    rates = {
        "CurrencyUSD": dict.fromkeys(range(1990, 2020), 1.0),
        "CurrencyEUR": dict.fromkeys(range(1990, 2020), 0.9),
    }
    gap = unit_coverage(
        _bulk_plan_of(by_sv),
        elected,
        declared_roles=_ROLES,
        graph=graph,
        sdmx=_server(rates).client(),
        rate_cache=RateTableCache(),
    ).refusal
    assert gap is None
    assert checks == [30]


_PLACE_ROLES = ("observationAbout",)
_SPARSE_YEARS = (2020, 2022)
_EUR_ALL_YEARS = {"CurrencyUSD": dict.fromkeys(range(2020, 2023), 1.0)}
_SPARSE_RATES = {**_EUR_ALL_YEARS, "CurrencyEUR": {2020: 0.9, 2022: 0.95}}
"""Every dollar rate, and euro rates in the two years the series serves alone."""
_SHORT_RATES = {**_EUR_ALL_YEARS, "CurrencyEUR": {2020: 0.9}}
"""No euro rate for 2022, a year the series serves."""


def _sparse_plan() -> PairPlan:
    """Kenya's series, observed in 2020 and 2022 alone, whose facet states only its span
    and count, as a dateless read's facet probe returns it."""
    summary = Facet(
        obs_count=len(_SPARSE_YEARS),
        earliest_date=str(min(_SPARSE_YEARS)),
        latest_date=str(max(_SPARSE_YEARS)),
        facet_id="f1",
        provenance_id=_PROVENANCE,
        unit="CurrencyEUR",
    )
    series = PlanSeries(
        stat_var=_SV,
        role_tuple=(("observationAbout", _KEN),),
        candidates=(summary,),
        group=(summary,),
    )
    return PairPlan(members=(_SV,), years=None, series=(series,))


class _DatedFacets:
    """The Search graph's facet read: `_SV` at each place with `_SPARSE_YEARS` dated, or
    `fails` raised. Each read records whether it asked for dates and its deadline."""

    def __init__(self, *, fails: GraphInfraError | None = None) -> None:
        self.fails = fails
        self.reads: list[tuple[bool, float | None]] = []

    def observation_facets_batch(
        self, stat_vars, entities, needs_dates=False, *, max_bytes=None, deadline=None
    ):
        self.reads.append((needs_dates, deadline))
        if self.fails is not None:
            raise self.fails
        return {sv: {entity: [_facet(*_SPARSE_YEARS)] for entity in entities} for sv in stat_vars}


def _mint_refusal(rates) -> RateCoverageGapError | None:
    """The refusal a Datasets mint of the sparse series' rows raises, or None when it
    converts them."""
    graph, sdmx = _graph(), _server(rates).client()
    plan = plan_member_conversion(
        request=MaterialisationRequest(
            stat_var=f"dcid:{_SV}",
            role_bindings={"observationAbout": [_KEN]},
            declared_roles=_PLACE_ROLES,
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
        selections=_elected("USD")[_SV][0],
        graph=graph,
        sdmx=sdmx,
        rate_cache=RateTableCache(),
    )
    rows = table_from_cells(
        [
            {
                "variableMeasured": _SV,
                "observationAbout": _KEN,
                "unit": "CurrencyEUR",
                "TIME_PERIOD": str(year),
                "OBS_VALUE": "100",
            }
            for year in _SPARSE_YEARS
        ],
        roles=_PLACE_ROLES,
    )
    try:
        apply_member_conversion(plan, rows, declared_roles=_PLACE_ROLES)
    except RateCoverageGapError as exc:
        return exc
    return None


@pytest.mark.parametrize(
    ("rates", "missing"),
    [(_SPARSE_RATES, None), (_SHORT_RATES, ("CurrencyEUR@2022",))],
    ids=["no-rate-in-an-unserved-year", "no-rate-in-a-served-year"],
)
def test_a_sparse_series_refuses_exactly_when_its_mint_would(rates, missing):
    """The series serves 2020 and 2022. Its facet states the span alone, so the gate reads
    its dates before refusing on 2021 and then needs rates only where its rows are."""
    dated = _DatedFacets()
    deadline = time.monotonic() + 60
    gap = unit_coverage(
        _sparse_plan(),
        _elected("USD"),
        declared_roles=_PLACE_ROLES,
        graph=_graph(),
        sdmx=_server(rates).client(),
        rate_cache=RateTableCache(),
        dated_facets=dated,
        deadline=deadline,
    ).refusal
    mint = _mint_refusal(rates)
    exc = gap[1] if gap is not None else None
    assert exc is None or isinstance(exc, RateCoverageGapError)
    assert dated.reads == [(True, deadline)]
    assert (exc.entities if exc is not None else None) == missing
    assert (mint.entities if mint is not None else None) == missing


@pytest.mark.parametrize(
    ("dated", "deadline", "logged"),
    [
        (None, None, None),
        (_DatedFacets(fails=GraphInfraError("graph unavailable")), None, "failed"),
        (_DatedFacets(fails=GraphDeadlineFailure("deadline")), None, "failed"),
        (_DatedFacets(), 0.0, "skipped past the deadline"),
    ],
    ids=["no-dated-read", "dated-read-fails", "dated-read-times-out", "deadline-passed"],
)
def test_a_sparse_series_without_its_dates_refuses_on_its_span(dated, deadline, logged, caplog):
    """Without the series' dates the gate assumes a row in every year of its span, so the
    missing 2021 euro rate refuses, and a dated read that failed or never ran is logged
    with the series it was for. The rate table is loaded beforehand, so a passed deadline
    stops the dated read alone."""
    graph, sdmx, rate_cache = _graph(), _server(_SPARSE_RATES).client(), RateTableCache()
    check = partial(
        unit_coverage,
        _sparse_plan(),
        _elected("USD"),
        declared_roles=_PLACE_ROLES,
        graph=graph,
        sdmx=sdmx,
        rate_cache=rate_cache,
    )
    assert check().refusal is not None
    with (
        caplog.at_level(logging.WARNING, logger=convertibility.__name__),
        recording_degraded_reads() as lost,
    ):
        gap = check(dated_facets=dated, deadline=deadline).refusal
    # A dated read that failed or never ran is lost to the answer, so no cache stores it.
    assert lost.degraded is (logged is not None)
    assert gap is not None
    exc = gap[1]
    assert isinstance(exc, RateCoverageGapError)
    assert (exc.cause, exc.entities) == ("rate_missing", ("CurrencyEUR@2021",))
    warnings = [r.getMessage() for r in caplog.records if r.name == convertibility.__name__]
    if logged is None:
        assert warnings == []
    else:
        (warning,) = warnings
        assert logged in warning
        assert repr([_SV]) in warning and repr([_KEN]) in warning
    if deadline is not None:
        assert dated is not None and dated.reads == []


def test_a_span_only_series_with_no_point_in_the_window_leaves_no_row_to_survive():
    """Kenya's facet states a 2020 to 2022 span, but its points are in 2020 and 2022, and a
    region no currency names has a 2021 point. Read for 2021, Kenya serves no row and the
    region's is omitted, so the read refuses `currency_unmapped`. The gate reads Kenya's dates
    and refuses the same way, naming the region."""
    sparse = _sparse_plan()
    region = PlanSeries(
        stat_var=_SV,
        role_tuple=(("observationAbout", _REGIONS[0]),),
        candidates=(_facet(2021),),
        group=(_facet(2021),),
    )
    plan = replace(
        sparse,
        years=TimeWindow(start_year=2021, end_year=2021),
        series=(*sparse.series, region),
    )
    dated = _DatedFacets()
    coverage = unit_coverage(
        plan,
        _elected("observationAbout"),
        declared_roles=_PLACE_ROLES,
        graph=_graph(),
        sdmx=_server(_KES_RATES).client(),
        rate_cache=RateTableCache(),
        dated_facets=dated,
    )
    assert dated.reads == [(True, None)]
    assert coverage.refusal is not None
    exc = coverage.refusal[1]
    assert isinstance(exc, RateCoverageGapError)
    assert (exc.cause, exc.entities) == ("currency_unmapped", (_REGIONS[0],))

    # The read of the window's one row: the region's, which conversion omits.
    conversion = plan_member_conversion(
        request=MaterialisationRequest(
            stat_var=f"dcid:{_SV}",
            role_bindings={"observationAbout": [_KEN, _REGIONS[0]]},
            declared_roles=_PLACE_ROLES,
            facet_bindings={},
            window=RequestedWindow(kind="all"),
        ),
        selections=_elected("observationAbout")[_SV][0],
        graph=_graph(),
        sdmx=_server(_KES_RATES).client(),
        rate_cache=RateTableCache(),
    )
    rows = table_from_cells(
        [
            {
                "variableMeasured": _SV,
                "observationAbout": _REGIONS[0],
                "unit": "CurrencyEUR",
                "TIME_PERIOD": "2021",
                "OBS_VALUE": "100",
            }
        ],
        roles=_PLACE_ROLES,
    )
    kept, currencies = omit_unmapped_rows(conversion, rows, served=rows)
    table, record = apply_member_conversion(
        conversion, kept, declared_roles=_PLACE_ROLES, currencies=currencies
    )
    with pytest.raises(RateCoverageGapError) as read:
        refuse_if_every_row_omitted(served_rows=table.num_rows, records=(record,))
    assert (read.value.cause, read.value.entities) == (exc.cause, exc.entities)


def test_a_rate_table_that_fails_to_load_refuses_an_all_omitted_ask_on_both_surfaces(
    monkeypatch,
):
    """Only regions are asked for, so every row is omitted. The read loads the recipe's
    tables before it learns that, so a table that fails to load refuses it; the gate loads
    them too and refuses the same way, before deciding `currency_unmapped`."""
    failure = RateTableInvalidError("the exchange-rate table holds no annual period")

    def failing_load(**_kwargs):
        raise failure

    monkeypatch.setattr(member_conversion, "load_table", failing_load)
    with pytest.raises(RateTableInvalidError) as gate:
        _destination_coverage(_pair_plan(*_REGIONS))
    with pytest.raises(RateTableInvalidError) as read:
        _preview_of(tuple((_DONOR, r) for r in _REGIONS))
    assert gate.value is read.value is failure
