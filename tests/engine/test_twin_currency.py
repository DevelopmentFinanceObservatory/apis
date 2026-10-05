"""A registered twin is equivalent only in its cluster's currency.

IMF WEO GDP reports each country's national currency beside US dollars, with equal counts
and spans, so an unbound unit elects the national currency on the unit tiebreak. The
registry declares the GDP cluster in US dollars, so the twin serves and lists its US dollar
facets and binds them in its identity. A twin that reads another currency in the answer is
dropped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import httpx
import pytest

from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft
from search.engine.extract import DateRequest
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.region.state import EngineOptions
from search.models import IdentityBindingValue, ResubmitIdentityPayload, TimeWindow
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
)
from tests.engine.test_admissibility_table import (
    _GDP_SOLE_OPTION,
    _IMF_BINDINGS,
    _IMF_GDP_NODE,
    _IMF_GDP_SV,
    _KEN,
    _KENYA,
    _NOMINAL_GDP_SV,
    _SOMALIA,
    _TIED,
    _WB_NOMINAL_GDP_NODES,
    _SoleOptionLLM,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_WB_PROVENANCE = "dc/base/WorldDevelopmentIndicators"
_IMF_PROVENANCE = "provenance/IMFWorldEconomicOutlook"
_SPAN = {"earliestDate": "1980", "latestDate": "2024", "obsCount": 45}
_WB = {
    **_SPAN,
    "provenanceId": _WB_PROVENANCE,
    "unit": "USDollar",
    "facetId": "wb-usd",
}
_NATIONAL = {"country/KEN": "KES", "country/SOM": "SOS", "country/IND": "INR"}
_INDIA = {"India": "country/IND"}


def _imf(place: str, *, usd: bool) -> list[dict]:
    """The IMF facets at `place`, as dev serves them: one national-currency facet per
    country, and one US dollar facet shared by every country."""
    national = _NATIONAL[place]
    facets = [
        {
            **_SPAN,
            "provenanceId": _IMF_PROVENANCE,
            "unit": f"Currency{national}",
            "facetId": f"imf-{national}",
        }
    ]
    if usd:
        facets.append(
            {
                **_SPAN,
                "provenanceId": _IMF_PROVENANCE,
                "unit": "CurrencyUSD",
                "facetId": "imf-usd",
            }
        )
    return facets


def _wb_national(place: str) -> dict:
    national = _NATIONAL[place]
    return {**_WB, "unit": f"Currency{national}", "facetId": f"wb-{national}"}


_RECIPE = "ConversionRecipeDAC"
_SDMX_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"


def _converting_options() -> EngineOptions:
    """Materialise clients that convert World Bank GDP at Kenya into Kenyan shillings and
    serve the IMF series in its native shilling facet."""
    wb = ObsFacetFixture(
        facet_id="wb-usd",
        points=(("2024", 100.0),),
        provenance_id=_WB_PROVENANCE,
        unit="USDollar",
    )
    imf = [
        ObsFacetFixture(
            facet_id=f"imf-{currency}",
            points=(("2024", value),),
            provenance_id=_IMF_PROVENANCE,
            unit=f"Currency{currency}",
        )
        for currency, value in (("KES", 13000.0), ("USD", 100.0))
    ]
    rows = {
        sv: _csv_header(["observationAbout"])
        + "".join(
            _csv_row(stat_var=sv, role_dcids=[_KEN], fixture=f, date_str="2024", value=v)
            for f in fixtures
            for _, v in f.points
        )
        for sv, fixtures in ((_NOMINAL_GDP_SV, [wb]), (_IMF_GDP_SV, imf))
    }

    def observations(request: httpx.Request) -> httpx.Response:
        body = rows.get(request.url.params.get("c[variableMeasured]", ""), "")
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV})

    server = RateTableServer(
        recipe_table_bodies(
            _RECIPE,
            rates={
                currency: dict.fromkeys(range(1980, 2025), rate)
                for currency, rate in (("CurrencyUSD", 1.0), ("CurrencyKES", 130.0))
            },
        ),
        fallback=observations,
    )
    graph = FakeGraphReader(
        arcs={
            _NOMINAL_GDP_SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            _IMF_GDP_SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_WB_PROVENANCE,
                currencies={"CurrencyUSD": (), "CurrencyKES": (_KEN,)},
            ),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_IMF_PROVENANCE,
                currencies={"CurrencyUSD": (), "CurrencyKES": (_KEN,)},
            ),
        },
        facets={
            (_NOMINAL_GDP_SV, _KEN): [_facet_from_fixture(wb)],
            (_IMF_GDP_SV, _KEN): [_facet_from_fixture(f) for f in imf],
        },
    )
    return EngineOptions(
        materialise_graph=graph,
        materialise_sdmx=server.client(base_url=MATERIALISE_BASE_URL),
        rate_cache=RateTableCache(),
    )


def _resolve(
    svs: dict[str, dict],
    bindings: Sequence[SlotBindingDraft],
    places: Mapping[str, str],
    *,
    wb_absent: frozenset[str] = frozenset(),
    imf_absent: frozenset[str] = frozenset(),
    wb_national: bool = False,
    imf_usd: bool = True,
    unit_text: str | None = None,
    date_request: DateRequest | None = None,
    imf_facets=None,
    wb_facets: Sequence[dict] = (_WB,),
):
    """`wb_national` adds a World Bank facet in each place's national currency, as WDI
    reports at India. `imf_facets` stands in for `_imf` and `wb_facets` for the World Bank
    US dollar facet."""
    imf_facets = imf_facets or _imf
    query = "GDP of " + " and ".join(places)
    obs = {
        f"{_IMF_GDP_SV}|{dcid}": imf_facets(dcid, usd=imf_usd)
        for dcid in places.values()
        if dcid not in imf_absent
    }
    obs |= {
        f"{_NOMINAL_GDP_SV}|{dcid}": [*wb_facets, *([_wb_national(dcid)] if wb_national else [])]
        for dcid in places.values()
        if dcid not in wb_absent
    }
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in places.items()},
            **svs,
        },
        detect={query: {"svs": list(svs), "cosine_scores": _TIED}},
        resolve=dict(places),
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            query,
            entities=list(places),
            date_request=date_request,
            detect_query=query,
            role_query=query,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_SoleOptionLLM([1], bindings),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            coverage=empty_coverage_snapshot(),
            unit_text=unit_text,
            options=_converting_options() if unit_text is not None else None,
        )
    )


def _reading(spec) -> tuple[str, str | None, object]:
    """A spec's StatVar, served unit and identity unit binding."""
    (stat_var,) = [sv.ref for sv in spec.resolved.stat_vars.items]
    unit = spec.resolved.facets.get("unit")
    return (
        stat_var,
        unit.ref if unit is not None else None,
        (spec.definition.identity.facets or {}).get("unit"),
    )


_IMF_IN_USD = (_IMF_GDP_SV, "CurrencyUSD", IdentityBindingValue(ref="CurrencyUSD"))
_WB_IN_USD = (_NOMINAL_GDP_SV, "USDollar", None)


@pytest.mark.parametrize(
    ("svs", "bindings", "places", "wb_absent", "served", "listed"),
    [
        pytest.param(
            {**_WB_NOMINAL_GDP_NODES, _IMF_GDP_SV: _IMF_GDP_NODE},
            _GDP_SOLE_OPTION,
            _KENYA,
            frozenset(),
            _WB_IN_USD,
            _IMF_IN_USD,
            id="imf-listed",
        ),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _IMF_BINDINGS,
            _SOMALIA,
            frozenset({"country/SOM"}),
            _IMF_IN_USD,
            _WB_IN_USD,
            id="imf-served-first",
        ),
    ],
)
def test_the_imf_twin_reads_us_dollars(svs, bindings, places, wb_absent, served, listed):
    result = _resolve(svs, bindings, places, wb_absent=wb_absent)

    assert result.status == "definite", result
    (spec,) = result.specs
    assert _reading(spec) == served
    assert [_reading(twin) for twin in spec.resolved.equivalent_readings] == [listed]


@pytest.mark.parametrize(
    ("svs", "places", "imf_absent", "served"),
    [
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _INDIA,
            frozenset({"country/IND"}),
            (_NOMINAL_GDP_SV, "USDollar", IdentityBindingValue(ref="USDollar")),
            id="no-data-election-answered-in-usd",
        ),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE},
            _KENYA,
            frozenset(),
            _IMF_IN_USD,
            id="lone-imf-in-usd",
        ),
    ],
)
def test_a_cluster_member_serves_the_cluster_currency_with_no_twin_beside_it(
    svs, places, imf_absent, served
):
    """A member of the GDP cluster serves US dollars whatever else is on the menu. World
    Bank GDP answering for an IMF election with no data at India serves its US dollar facet
    beside the rupee one, and a lone IMF candidate serves its US dollar facet beside the
    shilling one."""
    result = _resolve(svs, _IMF_BINDINGS, places, imf_absent=imf_absent, wb_national=True)

    assert result.status == "definite", result
    (spec,) = result.specs
    assert _reading(spec) == served
    assert not spec.resolved.equivalent_readings


@pytest.mark.parametrize(
    ("imf_usd", "unit_text", "unit"),
    [
        pytest.param(False, None, "USDollar", id="no-usd-facet"),
        pytest.param(True, "Kenyan shillings", "CurrencyKES", id="unit-request-kes"),
    ],
)
def test_a_twin_read_outside_the_cluster_currency_is_dropped(imf_usd, unit_text, unit):
    """Where the IMF series serves Kenyan shillings, because it has no US dollar facet or
    the query asks for shillings, it is not World Bank GDP's reading. It was admitted as the
    same concept, so it is dropped rather than offered, and World Bank GDP answers in the
    served unit."""
    result = _resolve(
        {**_WB_NOMINAL_GDP_NODES, _IMF_GDP_SV: _IMF_GDP_NODE},
        _GDP_SOLE_OPTION,
        _KENYA,
        imf_usd=imf_usd,
        unit_text=unit_text,
    )

    assert result.status == "definite", result
    (spec,) = result.specs
    assert _reading(spec)[:2] == (_NOMINAL_GDP_SV, unit)
    assert not spec.resolved.equivalent_readings


def _dated(facet: dict, first: int, last: int) -> dict:
    """`facet` holding a point in every year from `first` to `last`, its dates listed."""
    years = [str(year) for year in range(first, last + 1)]
    return {
        **facet,
        "earliestDate": years[0],
        "latestDate": years[-1],
        "obsCount": len(years),
        "dates": years,
        "values": [1.0] * len(years),
    }


@pytest.mark.parametrize(
    ("wb_last", "served", "listed"),
    [
        pytest.param(2023, _IMF_IN_USD, _WB_IN_USD, id="world-bank-lacks-two-years"),
        pytest.param(2025, _WB_IN_USD, _IMF_IN_USD, id="world-bank-holds-every-year"),
    ],
)
def test_the_preferred_twin_answers_only_where_it_holds_every_requested_year(
    wb_last, served, listed
):
    """ "GDP of Kenya from 2010 to 2025" elects IMF GDP, which holds all 16 years. World Bank
    GDP is the registry's preferred member and answers in its place only when it holds
    them all; holding 2010 to 2023 it is listed beside the IMF answer, which keeps its 16
    observations."""
    result = _resolve(
        {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
        _IMF_BINDINGS,
        _KENYA,
        date_request=DateRequest(window=TimeWindow(start_year=2010, end_year=2025), latest=False),
        imf_facets=lambda place, *, usd: [_dated(f, 2010, 2025) for f in _imf(place, usd=usd)],
        wb_facets=[_dated(_WB, 2010, wb_last)],
    )

    assert result.status == "definite", result
    (spec,) = result.specs
    assert _reading(spec) == served
    assert [_reading(twin) for twin in spec.resolved.equivalent_readings] == [listed]
    assert spec.resolved.coverage.window == TimeWindow(
        start_year=2010, end_year=2025 if served == _IMF_IN_USD else wb_last
    )


def _imf_with_unitless(place: str, *, usd: bool) -> list[dict]:
    """The IMF facets at `place`, and one stating no unit that holds more observations than
    either."""
    return [
        *_imf(place, usd=usd),
        {
            **_SPAN,
            "obsCount": 60,
            "latestDate": "2025",
            "provenanceId": _IMF_PROVENANCE,
            "facetId": "imf-unitless",
        },
    ]


@pytest.mark.parametrize(
    ("svs", "bindings", "wb_absent"),
    [
        pytest.param({_IMF_GDP_SV: _IMF_GDP_NODE}, _IMF_BINDINGS, frozenset(), id="definite"),
        pytest.param(
            {_IMF_GDP_SV: _IMF_GDP_NODE, **_WB_NOMINAL_GDP_NODES},
            _IMF_BINDINGS,
            frozenset({_KEN}),
            id="candidates",
        ),
    ],
)
def test_a_cluster_currency_answer_serves_what_its_resubmit_serves(svs, bindings, wb_absent):
    """IMF GDP at Kenya reports shillings, US dollars and a facet stating no unit. The
    answer binds US dollars in its identity, so it serves the US dollar facet alone, and an
    unchanged resubmit of that identity serves the same facets and observations."""
    result = _resolve(svs, bindings, _KENYA, wb_absent=wb_absent, imf_facets=_imf_with_unitless)

    assert result.status == "definite", result
    spec = result.specs[0]
    assert _reading(spec) == _IMF_IN_USD
    graph = FakeGraph(
        nodes={_KEN: {"label": "Kenya", "type": "Country"}, _IMF_GDP_SV: _IMF_GDP_NODE},
        obs={f"{_IMF_GDP_SV}|{_KEN}": _imf_with_unitless(_KEN, usd=True)},
    )
    posted = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=spec.definition.identity,
            base_identity=spec.definition.identity,
            base_spec_id=spec.spec_id,
            resolved_stat_var_dcids=[_IMF_GDP_SV],
        ),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
    )

    assert posted.status == "definite", posted
    (resubmitted,) = posted.specs
    assert _reading(resubmitted) == _reading(spec)
    assert {axis: f.ref for axis, f in resubmitted.resolved.facets.items()} == {
        axis: f.ref for axis, f in spec.resolved.facets.items()
    }
    assert resubmitted.resolved.coverage == spec.resolved.coverage
    assert spec.resolved.coverage.observation_count == 45
