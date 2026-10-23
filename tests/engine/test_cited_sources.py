"""An answer states one list of sources: the facets its read plan serves.

`trace.resolved_sources`, the sentence's source and coverage's source count all read
`read_plan.facets_served` of the answer's plan, on the standard route and the four role
routes (named or discovered destinations, read per pair or in bulk), on a first answer, an
unchanged resubmit of it, an also-plausible candidate and under a unit request or filter.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass

import httpx
import pytest

import search.engine.region.candidate_grounding as candidate_grounding
import search.engine.region.phase_answer as phase_answer
import search.engine.region.promote_ground as promote_ground
from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.rate_tables import RateTableCache
from search.engine.clients.graph import Facet
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_spec_resubmit
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.retrieval.coverage import citable_source_ids
from search.engine.retrieval.read_plan import BulkPlan, PairPlan, ReadPlan, facets_served
from search.models import (
    BaseIdentity,
    IdentityBindingValue,
    ResubmitIdentityPayload,
    Spec,
    TimeWindow,
)
from search.render.plan import render_sentence, render_sentence_plan
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine import test_parsed_pipeline as parsed
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import (
    CoverageRow,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    _SDMX_CSV_CONTENT_TYPE,
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    door_roles_for_test_query,
    materialise_double_from_obs,
    walk_options,
    with_member_arcs,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

_FLOW = "TESTFIXTURE/CitedSources/Flow"
_ALT_FLOW = "TESTFIXTURE/CitedSources/FlowAlt"
_ALAND, _BLAND, _CREDITOR = "country/AAA", "country/BBB", "country/CRA"
_PLACES = {"Aland": _ALAND, "Bland": _BLAND}
_IDS, _ALT, _PROJ = "dc/base/Ids", "dc/base/Alt", "dc/base/Proj"
_LABELS = {_IDS: "IDS", _ALT: "Alternative", _PROJ: "Projections"}
_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
_CLOSED = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)

_ROUTES = pytest.mark.parametrize(
    ("discovered", "bulk"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["named-per_pair", "named-bulk", "discovered-per_pair", "discovered-bulk"],
)


def _flow_arcs(measured_property: str = "amount") -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": measured_property}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        # The qualifier a unit request converts, as materialise reads it (`_UNIT_ARCS`).
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
    }


def _nodes() -> dict:
    return {
        **{dcid: {"label": name, "type": "Country"} for name, dcid in _PLACES.items()},
        _FLOW: {"label": "Test flow", "arcs": _flow_arcs()},
        _ALT_FLOW: {"label": "Alternate flow", "arcs": _flow_arcs("amountAlt")},
        **{dcid: {"label": label} for dcid, label in _LABELS.items()},
    }


def _two_source_rows(flow: str = _FLOW, unserved_years: range = range(2016, 2019)) -> list:
    """Aland and Bland hold `f1` for 2010 to 2020, and Aland a second facet `f9` the
    2015 to 2020 plan serves no point from: a shorter series the election ranks below `f1`."""
    return [
        (flow, "f1", _CREDITOR, _ALAND, list(range(2010, 2021))),
        (flow, "f1", _CREDITOR, _BLAND, list(range(2010, 2021))),
        (flow, "f9", _CREDITOR, _ALAND, list(unserved_years)),
    ]


_TWO_SOURCES = {"f1": FacetMetadata(provenance_id=_IDS), "f9": FacetMetadata(provenance_id=_ALT)}


class _BindLLM:
    """Elects shape index 0, with `also_plausible` shapes kept as candidates."""

    def __init__(self, also_plausible: tuple[int, ...] = ()) -> None:
        self._also_plausible = list(also_plausible)

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            elected_reply(0, bindings=[], also_plausible=self._also_plausible),
            None,
        )


def _spy_plans(monkeypatch, module) -> list[ReadPlan | None]:
    """Record the plan of every answer `module` filters (`facets.filter_answer`)."""
    plans: list[ReadPlan | None] = []
    real = module.filter_answer

    def _filter_answer(*args, **kwargs):
        filtered = real(*args, **kwargs)
        plans.append(filtered.plan)
        return filtered

    monkeypatch.setattr(module, "filter_answer", _filter_answer)
    return plans


def _role_answer(
    monkeypatch,
    search_settings,
    *,
    discovered: bool,
    bulk: bool,
    rows: list[CoverageRow],
    metadata: dict[str, FacetMetadata],
    flows: tuple[str, ...] = (_FLOW,),
    date_request: DateRequest | None = _CLOSED,
    llm: _BindLLM | None = None,
    sdmx: FakeSdmxClient | None = None,
    unit_text: str | None = None,
    options: EngineOptions | None = None,
):
    """Debt flows to Aland and Bland, named as destinations or discovered on the open
    axis, read per pair or, with `bulk`, in one read. Returns the result and its index."""
    search_settings(obs_preview_max_pairs=1 if bulk else 120)
    index = patch_coverage(
        monkeypatch,
        rows,
        role_names=dict.fromkeys(flows, _ROLE_NAMES),
        facet_metadata=metadata,
    )
    entities = [] if discovered else list(_PLACES)
    query = "debt flows" if discovered else "debt flows to Aland and Bland"
    graph = FakeGraph(
        nodes=_nodes(),
        detect={
            query: {
                "svs": list(flows),
                "cosine_scores": [0.9 - 0.05 * i for i in range(len(flows))],
            }
        },
        resolve=_PLACES,
        obs={},
    )
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=entities,
            date_request=date_request,
            detect_query=query,
            role_query=query,
            entity_roles=door_roles_for_test_query(query, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm or _BindLLM(),
            sdmx=sdmx or FakeSdmxClient(observations={}, availability={}),
            base_timing={},
            unit_text=unit_text,
            coverage=FakeCoverageService(index).snapshot(),
            options=options or EngineOptions(),
        )
    )
    return result, index


def _resubmit(
    spec: Spec,
    index,
    *,
    facets: dict | None = None,
    sdmx: FakeSdmxClient | None = None,
    options: EngineOptions | None = None,
):
    """Post `spec`'s own identity back, with `facets` added as a new binding."""
    identity = spec.definition.identity
    if facets is not None:
        identity = BaseIdentity.model_validate(
            {**identity.model_dump(mode="python"), "facets": facets}
        )
    graph = FakeGraph(nodes=_nodes(), detect={}, resolve={}, obs={})
    options = options or EngineOptions()
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(identity=identity),
        graph=graph,
        sdmx=sdmx or FakeSdmxClient(observations={}, availability={}),
        coverage=FakeCoverageService(index).snapshot(),
        options=dataclasses.replace(
            options, materialise_graph=with_member_arcs(options.materialise_graph, graph)
        ),
    )


def _stated_sources(spec: Spec) -> tuple[list[str], int | None, str]:
    """What `spec` states about its sources: `resolved_sources`, coverage's source count
    and the sentence."""
    resolved = spec.resolved
    dims = {d.label: d.count for d in resolved.coverage.dimensions or []}
    return (
        [ref.ref for ref in resolved.trace.resolved_sources],
        dims.get("sources"),
        render_sentence(spec),
    )


def _assert_cites(spec: Spec, plan: ReadPlan | None, expected: list[str]) -> str:
    """`spec` states exactly `expected`, the sources `plan` serves. Returns the sentence."""
    assert plan is not None
    assert citable_source_ids(facets_served(plan)) == expected
    sources, count, sentence = _stated_sources(spec)
    assert sources == expected
    assert count == len(expected)
    for dcid, label in _LABELS.items():
        assert (label in sentence) == (dcid in expected), sentence
    return sentence


# ---------------------------------------------------------------------------
# A first answer and an unchanged resubmit of it state the same sources.
# ---------------------------------------------------------------------------


@_ROUTES
def test_an_unchanged_role_resubmit_states_the_sources_its_first_answer_did(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_two_source_rows(),
        metadata=_TWO_SOURCES,
    )
    assert first.status == "definite", (first.status, first.no_data_reason)
    plans = _spy_plans(monkeypatch, promote_ground)

    again = _resubmit(first.specs[0], index)

    assert again.status == "definite", (again.status, again.no_data_reason)
    assert again.specs[0].spec_id == first.specs[0].spec_id
    (plan,) = plans
    assert isinstance(plan, BulkPlan if bulk else PairPlan)
    _assert_cites(again.specs[0], plan, [_IDS])
    assert _stated_sources(again.specs[0]) == _stated_sources(first.specs[0])


# ---------------------------------------------------------------------------
# A unit request and a unit filter cite what the plan serves under that unit.
# ---------------------------------------------------------------------------

_USD = "CurrencyUSD"
_UNIT_ARCS = {
    _FLOW: {"measurementQualifier": ("Nominal",)},
    _ALT_FLOW: {"measurementQualifier": ("Nominal",)},
    _USD: {"typeOf": ("CurrencyUnit",), "rateEntity": ("dc/base/UsRate",)},
}


@dataclass(frozen=True)
class _DollarCase:
    """Which of the two sources report US dollars, and the facet a "USD" request serves:
    its id, source, places and years inside 2015 to 2020."""

    metadata: dict[str, FacetMetadata]
    facet_id: str
    source: str
    places: tuple[str, ...]
    years: range


_DOLLAR_CASES = {
    # Both report dollars, and the plan elects the longer IDS series.
    "both_sources": _DollarCase(
        metadata={
            "f1": FacetMetadata(provenance_id=_IDS, unit=_USD),
            "f9": FacetMetadata(provenance_id=_ALT, unit=_USD),
        },
        facet_id="f1",
        source=_IDS,
        places=(_ALAND, _BLAND),
        years=range(2015, 2021),
    ),
    # IDS reports no unit, so the plan under the request serves Alternative at Aland.
    "runner_up_alone": _DollarCase(
        metadata={
            "f1": FacetMetadata(provenance_id=_IDS),
            "f9": FacetMetadata(provenance_id=_ALT, unit=_USD),
        },
        facet_id="f9",
        source=_ALT,
        places=(_ALAND,),
        years=range(2016, 2019),
    ),
}
_DOLLARS = pytest.mark.parametrize("dollars", list(_DOLLAR_CASES))


def _dollar_options(case: _DollarCase, flows: tuple[str, ...] = (_FLOW,)) -> EngineOptions:
    """Materialise doubles serving each flow's dollar facet at the places it covers."""
    fixture = ObsFacetFixture(facet_id=case.facet_id, provenance_id=case.source, unit=_USD)
    bodies = {
        flow: _csv_header(["creditor", "debtor"])
        + "".join(
            _csv_row(
                stat_var=flow,
                role_dcids=[_CREDITOR, place],
                fixture=fixture,
                date_str=str(case.years[-1]),
                value=1.0,
            )
            for place in case.places
        )
        for flow in flows
    }

    def handler(request: httpx.Request) -> httpx.Response:
        flow = request.url.params.get("c[variableMeasured]", "")
        return httpx.Response(
            200, text=bodies[flow], headers={"content-type": _SDMX_CSV_CONTENT_TYPE}
        )

    return EngineOptions(
        materialise_graph=FakeGraphReader(arcs=_UNIT_ARCS),
        materialise_sdmx=MaterialiseSdmxClient(
            base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(handler)
        ),
        rate_cache=RateTableCache(),
    )


def _dollar_series(case: _DollarCase, flows: tuple[str, ...] = (_FLOW,)) -> FakeSdmxClient:
    """The bulk read's rows: each flow's dollar facet at the places it covers."""
    years = [str(y) for y in case.years]
    facet = Facet(
        facet_id=case.facet_id,
        earliest_date=years[0],
        latest_date=years[-1],
        obs_count=len(years),
        dates=years,
        values=[1.0] * len(years),
        provenance_id=case.source,
        unit=_USD,
    )
    return FakeSdmxClient(
        observations={
            flow: {(("creditor", _CREDITOR), ("debtor", place)): [facet] for place in case.places}
            for flow in flows
        },
        availability={},
    )


@_ROUTES
@_DOLLARS
def test_a_unit_request_cites_the_sources_its_plan_serves(
    monkeypatch, search_settings, discovered: bool, bulk: bool, dollars: str
) -> None:
    """A "USD" request states the one source its plan serves in dollars: IDS when both
    report dollars, although Alternative holds dollars in the same years, and Alternative
    when it alone does, although the unfiltered plan serves IDS."""
    case = _DOLLAR_CASES[dollars]
    plans = _spy_plans(monkeypatch, phase_answer)

    result, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_two_source_rows(),
        metadata=case.metadata,
        unit_text="USD",
        sdmx=_dollar_series(case),
        options=_dollar_options(case),
    )

    assert result.status == "definite", (result.status, result.no_data_reason)
    (plan,) = plans
    assert isinstance(plan, BulkPlan if bulk else PairPlan)
    _assert_cites(result.specs[0], plan, [case.source])


@_ROUTES
def test_a_unit_filter_cites_the_sources_the_narrowed_plan_serves(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    """IDS reports US dollars and Alternative euros. The first answer serves and cites IDS;
    a resubmit posting the euro unit serves Alternative at Aland, and cites it alone."""
    first, index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_two_source_rows(),
        metadata={
            "f1": FacetMetadata(provenance_id=_IDS, unit="USD"),
            "f9": FacetMetadata(provenance_id=_ALT, unit="EUR"),
        },
    )
    assert first.status == "definite", (first.status, first.no_data_reason)
    assert _stated_sources(first.specs[0])[0] == [_IDS]
    plans = _spy_plans(monkeypatch, promote_ground)

    again = _resubmit(first.specs[0], index, facets={"unit": IdentityBindingValue(ref="EUR")})

    assert again.status == "definite", (again.status, again.no_data_reason)
    (plan,) = plans
    _assert_cites(again.specs[0], plan, [_ALT])


# ---------------------------------------------------------------------------
# An also-plausible candidate cites its own plan's sources.
# ---------------------------------------------------------------------------


def _candidate_rows() -> list[CoverageRow]:
    return [*_two_source_rows(_FLOW), *_two_source_rows(_ALT_FLOW)]


@_ROUTES
def test_an_also_plausible_candidate_cites_the_sources_its_plan_serves(
    monkeypatch, search_settings, discovered: bool, bulk: bool
) -> None:
    flows = (_FLOW, _ALT_FLOW)
    plans = _spy_plans(monkeypatch, candidate_grounding)

    result, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_candidate_rows(),
        metadata=_TWO_SOURCES,
        flows=flows,
        llm=_BindLLM(also_plausible=(1,)),
    )

    assert result.status == "candidates", (result.status, result.no_data_reason)
    assert len(result.specs) == len(plans) == 2
    for spec, plan in zip(result.specs, plans, strict=True):
        assert isinstance(plan, BulkPlan if bulk else PairPlan)
        _assert_cites(spec, plan, [_IDS])


@_ROUTES
@_DOLLARS
def test_an_also_plausible_candidate_under_a_unit_request_cites_its_plan(
    monkeypatch, search_settings, discovered: bool, bulk: bool, dollars: str
) -> None:
    case = _DOLLAR_CASES[dollars]
    flows = (_FLOW, _ALT_FLOW)
    plans = _spy_plans(monkeypatch, candidate_grounding)

    result, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_candidate_rows(),
        metadata=case.metadata,
        flows=flows,
        llm=_BindLLM(also_plausible=(1,)),
        unit_text="USD",
        sdmx=_dollar_series(case, flows),
        options=_dollar_options(case, flows),
    )

    assert result.status == "candidates", (result.status, result.no_data_reason)
    assert len(result.specs) == len(plans) == 2
    for spec, plan in zip(result.specs, plans, strict=True):
        _assert_cites(spec, plan, [case.source])


# ---------------------------------------------------------------------------
# A plan whose facets carry no provenance cites nothing.
# ---------------------------------------------------------------------------


@_ROUTES
@pytest.mark.parametrize("second_source", [False, True], ids=["one_facet", "sourced_runner_up"])
def test_a_plan_serving_no_provenance_cites_nothing(
    monkeypatch, search_settings, discovered: bool, bulk: bool, second_source: bool
) -> None:
    """`f1` carries no provenance at either place. With `second_source`, Aland also holds
    the shorter `f9` from Alternative, which no plan elects: the role index lists that
    source and the answer does not cite it."""
    plans = _spy_plans(monkeypatch, phase_answer)

    result, _index = _role_answer(
        monkeypatch,
        search_settings,
        discovered=discovered,
        bulk=bulk,
        rows=_two_source_rows() if second_source else _two_source_rows()[:2],
        metadata={
            "f1": FacetMetadata(provenance_id=None),
            "f9": FacetMetadata(provenance_id=_ALT),
        },
    )

    assert result.status == "definite", (result.status, result.no_data_reason)
    (plan,) = plans
    _assert_cites(result.specs[0], plan, [])
    assert not any(
        token.axis == "source" and token.text for token in render_sentence_plan(result.specs[0])
    )


# ---------------------------------------------------------------------------
# The standard route.
# ---------------------------------------------------------------------------

_USD_SOURCE, _EUR_SOURCE = "dc/base/UsdSource", "dc/base/EurSource"
_STANDARD_LABELS = {_USD_SOURCE: "USD Source", _EUR_SOURCE: "EUR Source"}


def _fertility(monkeypatch, *, time_text: str | None = None) -> tuple[Spec, ReadPlan | None]:
    """Kenya's fertility rate, which holds USD Source for 2018 to 2020 and EUR Source for
    2015 to 2016. Returns the answer and its plan."""
    plans = _spy_plans(monkeypatch, phase_answer)
    result = parsed._resolve_parsed(
        [parsed._fertility_kenya_part(time_text=time_text)],
        graph=parsed._graph_with_two_unit_facets_on_fertility_kenya(),
    )
    assert result.status == "definite", result
    (plan,) = plans
    return result.parts[0].spec, plan


def _assert_cites_standard(spec: Spec, plan: ReadPlan | None, expected: list[str]) -> None:
    assert plan is not None
    assert citable_source_ids(facets_served(plan)) == expected
    sources, count, sentence = _stated_sources(spec)
    assert sources == expected
    assert count == len(expected)
    for dcid, label in _STANDARD_LABELS.items():
        assert (label in sentence) == (dcid in expected), sentence


@pytest.mark.parametrize("time_text", ["latest", "2018 to 2020", None], ids=str)
def test_a_standard_answer_cites_the_sources_its_plan_serves(monkeypatch, time_text) -> None:
    """EUR Source holds no point the plan serves under a latest, a 2018 to 2020 or a
    dateless request, so the sentence and `resolved_sources` name USD Source alone."""
    spec, plan = _fertility(monkeypatch, time_text=time_text)

    _assert_cites_standard(spec, plan, [_USD_SOURCE])


def _resubmit_fertility(monkeypatch, spec: Spec, *, facets: dict | None = None):
    plans = _spy_plans(monkeypatch, promote_ground)
    identity = spec.definition.identity
    if facets is not None:
        identity = BaseIdentity.model_validate(
            {**identity.model_dump(mode="python"), "facets": facets}
        )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(parsed._two_unit_facets_obs())
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=identity,
        ),
        graph=parsed._graph_with_two_unit_facets_on_fertility_kenya(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            parsed._graph_with_two_unit_facets_on_fertility_kenya(),
            EngineOptions(
                materialise_graph=with_member_arcs(
                    materialise_graph, parsed._graph_with_two_unit_facets_on_fertility_kenya()
                ),
                materialise_sdmx=materialise_sdmx,
            ),
        ),
    )
    assert region.status == "definite", (region.status, region.no_data_reason)
    (plan,) = plans
    return region.specs[0], plan


def test_an_unchanged_standard_resubmit_states_the_sources_its_first_answer_did(
    monkeypatch,
) -> None:
    first, _plan = _fertility(monkeypatch)

    again, plan = _resubmit_fertility(monkeypatch, first)

    assert again.spec_id == first.spec_id
    _assert_cites_standard(again, plan, [_USD_SOURCE])
    assert _stated_sources(again) == _stated_sources(first)


def test_a_standard_unit_filter_cites_the_sources_the_narrowed_plan_serves(monkeypatch) -> None:
    """A resubmit posting the euro unit serves EUR Source's two points and cites it alone."""
    first, _plan = _fertility(monkeypatch)

    again, plan = _resubmit_fertility(
        monkeypatch, first, facets={"unit": IdentityBindingValue(ref="CurrencyEUR")}
    )

    _assert_cites_standard(again, plan, [_EUR_SOURCE])


# ---------------------------------------------------------------------------
# Two named places, each served by its own source.
# ---------------------------------------------------------------------------

_POPULATION = "Count_Person"
_ZETA, _YPSILON = "country/ZZZ", "country/YYY"
_WDI, _UN_DATA, _PROJECTIONS = "dc/base/Wdi", "dc/base/UnData", "dc/base/Projections"
_PLACE_SOURCE_LABELS = {_WDI: "WDI", _UN_DATA: "UN data", _PROJECTIONS: "Projections"}


def _facet(facet_id: str, provenance: str, years: list[int]) -> dict:
    dates = [str(y) for y in years]
    return {
        "facetId": facet_id,
        "provenanceId": provenance,
        "earliestDate": dates[0],
        "latestDate": dates[-1],
        "obsCount": len(dates),
        "dates": dates,
        "values": [1.0] * len(dates),
    }


def _two_place_graph(obs: dict) -> FakeGraph:
    return FakeGraph(
        nodes={
            _POPULATION: {
                "label": "Population",
                "type": "StatisticalVariable",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            },
            _ZETA: {"label": "Zeta", "type": "Country"},
            _YPSILON: {"label": "Ypsilon", "type": "Country"},
            **{dcid: {"label": label} for dcid, label in _PLACE_SOURCE_LABELS.items()},
        },
        obs=obs,
        detect={"population": {"svs": [_POPULATION]}},
        resolve={"Zeta": _ZETA, "Ypsilon": _YPSILON},
    )


def _two_place_answer(obs: dict, places: tuple[str, ...], date_request: DateRequest | None):
    graph = _two_place_graph(obs)
    query = "population of " + " and ".join(places)
    return asyncio.run(
        resolve_variable(
            query,
            entities=list(places),
            date_request=date_request,
            detect_query="population",
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(),
        )
    )


def _two_place_resubmit(spec: Spec, obs: dict, *, facets: dict | None = None):
    payload = spec.definition.identity.model_dump(mode="python")
    if facets is not None:
        payload["facets"] = facets
    return resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=BaseIdentity.model_validate(payload),
        ),
        graph=_two_place_graph(obs),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(_two_place_graph(obs), EngineOptions()),
    )


# Zeta's WDI series has a gap and its Projections run 2021 to 2030; Ypsilon holds WDI alone.
# WDI covers both places, so it is the variable's one source, and Projections serves nothing.
_PROJECTION_AT_ZETA = {
    f"{_POPULATION}|{_ZETA}": [
        _facet("fw", _WDI, [*range(2010, 2015), *range(2018, 2021)]),
        _facet("fp", _PROJECTIONS, list(range(2021, 2031))),
    ],
    f"{_POPULATION}|{_YPSILON}": [_facet("fw", _WDI, list(range(2010, 2021)))],
}


@pytest.mark.parametrize("places", [("Zeta", "Ypsilon"), ("Ypsilon", "Zeta")], ids="-".join)
def test_a_two_place_resubmit_cites_the_source_its_first_answer_cites(monkeypatch, places) -> None:
    """However the places were named, a first answer and its resubmit cite WDI alone, the
    source covering both places, and neither cites the Projections only Zeta holds, so the
    sentence and `resolved_sources` agree on both."""
    plans = _spy_plans(monkeypatch, phase_answer)
    first = _two_place_answer(_PROJECTION_AT_ZETA, places, None)
    assert first.status == "definite", (first.status, first.no_data_reason)
    assert plans and plans[0] is not None
    assert citable_source_ids(facets_served(plans[0])) == [_WDI]

    again = _two_place_resubmit(first.specs[0], _PROJECTION_AT_ZETA)

    assert again.status == "definite", (again.status, again.no_data_reason)
    for result in (first, again):
        spec = result.specs[0]
        assert [ref.ref for ref in spec.resolved.trace.resolved_sources] == [_WDI]
        assert render_sentence(spec).endswith(", 2010 to 2020. WDI.")


# Zeta holds WDI for 2000 to 2018 and UN data for 2019 to 2022; Ypsilon WDI alone.
_UN_RECENT_AT_ZETA = {
    f"{_POPULATION}|{_ZETA}": [
        _facet("fw", _WDI, list(range(2000, 2019))),
        _facet("fu", _UN_DATA, list(range(2019, 2023))),
    ],
    f"{_POPULATION}|{_YPSILON}": [_facet("fw", _WDI, list(range(2005, 2021)))],
}


@pytest.mark.parametrize("places", [("Zeta",), ("Zeta", "Ypsilon")], ids="-".join)
def test_a_pinned_source_with_no_point_in_the_years_names_the_span_it_runs(places) -> None:
    """Up to 2012 the answer serves WDI. Pinning UN data, which runs 2019 to 2022, leaves no
    point in those years: the resubmit answers `no_observations` with the note an unpinned
    window answer gives, naming the years asked and the span the pinned series runs."""
    until_2012 = DateRequest(window=TimeWindow(start_year=None, end_year=2012), latest=False)
    first = _two_place_answer(_UN_RECENT_AT_ZETA, places, until_2012)
    assert first.status == "definite", (first.status, first.no_data_reason)

    pinned = _two_place_resubmit(
        first.specs[0],
        _UN_RECENT_AT_ZETA,
        facets={"provenance": IdentityBindingValue(ref=_UN_DATA)},
    )

    assert (pinned.status, pinned.no_data_reason) == ("no_data", "no_observations")
    assert pinned.window_gap_fact is not None
    assert (pinned.window_gap_fact.asked, pinned.window_gap_fact.data_years) == (
        "2012 or earlier",
        (2019, 2022),
    )


@pytest.mark.parametrize("places", [("Zeta", "Ypsilon"), ("Ypsilon", "Zeta")], ids="-".join)
def test_a_two_place_answer_and_its_resubmit_name_their_places_in_one_order(places) -> None:
    """However the places were named, a first answer and its resubmit state them in the one
    place order, by dcid: the where slot, the entities and the sentence agree."""
    first = _two_place_answer(_PROJECTION_AT_ZETA, places, None)
    again = _two_place_resubmit(first.specs[0], _PROJECTION_AT_ZETA)

    for result in (first, again):
        assert result.status == "definite", (result.status, result.no_data_reason)
        spec = result.specs[0]
        (where,) = [s for s in spec.resolved.slots if s.key.axis == "where"]
        assert [v.ref.ref for v in where.binding.values] == [_YPSILON, _ZETA]
        assert [e.ref for e in spec.resolved.entities] == [_YPSILON, _ZETA]
        assert render_sentence(spec).startswith("Population in Ypsilon and Zeta, ")
    assert render_sentence(first.specs[0]) == render_sentence(again.specs[0])
