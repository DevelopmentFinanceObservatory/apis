"""End-to-end guard that unit narrowing uses each StatVar's own elected currency.

Two role-bearing StatVars share FRA and DEU. SV_A elects native USD at FRA; SV_B elects native GBP
at DEU. Their 2020 and 2023 dates expose a pooled-scope regression.
"""

from __future__ import annotations

import asyncio

import httpx

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.rate_tables import RateTableCache
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.region import convertibility as convertibility_module
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_SV_A = "TESTFIXTURE/UnitNarrowPerSv/FlowA"
_SV_B = "TESTFIXTURE/UnitNarrowPerSv/FlowB"
_FRA = "country/FRA"
_DEU = "country/DEU"
_QUERY = "aid disbursed by source"

_PROV_A = "dc/base/UnitNarrowPerSvSrcA"
_PROV_B = "dc/base/UnitNarrowPerSvSrcB"
_RECIPE = "ConversionRecipeDAC"

# An origin-only role-bearing signature (declares only the "creditor" role,
# no destination at all) -- the same shape retrieval/role_scope.py's
# discover_role_scope opens for discovery when no entity is named, mirroring
# test_role_scope_narrowing_e2e.py's own origin-only fixture.
_FIVE_TUPLE_ARCS = {
    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "observationProperties": {"nodes": [{"dcid": "creditor"}]},
}


def _sv_arcs(*, label: str, source: str) -> dict:
    return {
        **_FIVE_TUPLE_ARCS,
        "name": {"nodes": [{"value": label}]},
        "constraintProperties": {"nodes": [{"dcid": "financingSource"}]},
        "financingSource": {"nodes": [{"dcid": source}]},
    }


_NODES = {
    _FRA: {"label": "France", "type": "Country"},
    _DEU: {"label": "Germany", "type": "Country"},
    _SV_A: {"label": "Flow A", "arcs": _sv_arcs(label="Flow A", source="SourceA")},
    _SV_B: {"label": "Flow B", "arcs": _sv_arcs(label="Flow B", source="SourceB")},
}


def _facet(*, facet_id: str, unit: str, latest_date: str) -> Facet:
    return Facet(
        earliest_date="2015",
        latest_date=latest_date,
        obs_count=1,
        dates=[latest_date],
        values=[0.0],
        facet_id=facet_id,
        unit=unit,
        provenance_id=f"source/{facet_id}",
    )


class _FixedBindLLM:
    """Elect shape 0 without narrowing either StatVar."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


_USD_FIXTURE_A = ObsFacetFixture(
    facet_id="pv_a", points=(("2020", 100.0),), provenance_id=_PROV_A, unit="CurrencyUSD"
)
_GBP_FIXTURE_B = ObsFacetFixture(
    facet_id="pv_b", points=(("2023", 50.0),), provenance_id=_PROV_B, unit="CurrencyGBP"
)
# SV_A's GBP facet at DEU, which serves only when a pooled election admits GBP for SV_A.
_PROV_A_OTHER = "dc/base/UnitNarrowPerSvSrcAOther"
_GBP_FIXTURE_A = ObsFacetFixture(
    facet_id="a_deu", points=(("2023", 80.0),), provenance_id=_PROV_A_OTHER, unit="CurrencyGBP"
)


def _decision_and_fetch_sdmx() -> MaterialiseSdmxClient:
    """Routes combined currency probes and per-entity fetches through one SDMX transport.

    Single-entity reads return that pair's data; combined probes use the same fixture per StatVar.
    Exchange-rate reads return the full table, including its `CurrencyUSD` anchor rows.
    """
    fetch_bodies = {
        (_SV_A, _FRA): _csv_header(["creditor"])
        + _csv_row(
            stat_var=_SV_A, role_dcids=[_FRA], fixture=_USD_FIXTURE_A, date_str="2020", value=100.0
        ),
        (_SV_B, _DEU): _csv_header(["creditor"])
        + _csv_row(
            stat_var=_SV_B, role_dcids=[_DEU], fixture=_GBP_FIXTURE_B, date_str="2023", value=50.0
        ),
        (_SV_A, _DEU): _csv_header(["creditor"])
        + _csv_row(
            stat_var=_SV_A, role_dcids=[_DEU], fixture=_GBP_FIXTURE_A, date_str="2023", value=80.0
        ),
    }
    decision_bodies = {_SV_A: fetch_bodies[(_SV_A, _FRA)], _SV_B: fetch_bodies[(_SV_B, _DEU)]}

    def observations(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        sv = params.get("c[variableMeasured]", "")
        creditor_param = params.get("c[creditor]", "")
        creditor_dcids = creditor_param.split(",") if creditor_param else []
        if len(creditor_dcids) == 1:
            body = fetch_bodies.get((sv, creditor_dcids[0]), _csv_header(["creditor"]))
        else:
            body = decision_bodies.get(sv, _csv_header(["creditor"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    # load_table (materialise.rate_tables) reads the exchange-rate variable
    # unfiltered by entity, where the main stat_vars' own fetch makes one
    # call per pair. Both currencies' own rows always
    # ship together here.
    server = RateTableServer(
        recipe_table_bodies(
            _RECIPE,
            rates={
                "CurrencyUSD": {2020: 1.0, 2023: 1.0},
                "CurrencyGBP": {2020: 0.8, 2023: 0.8},
            },
        ),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _run(monkeypatch):
    _unbound_recipient = "TESTFIXTURE/UnitNarrowPerSv/UnboundRecipient"
    # Only SV_A has a USD coverage candidate; SV_B's native GBP candidate is at DEU.
    # SV_A's USD facet at FRA is its native election. Its GBP facet at DEU holds more years,
    # so it leads SV_A's one group only when another StatVar's elected unit admits it.
    rows = [
        (_SV_A, "a_fra", _FRA, _unbound_recipient, [2018, 2019, 2020]),
        (_SV_A, "a_deu", _DEU, _unbound_recipient, [2020, 2021, 2022, 2023]),
        (_SV_B, "b_deu", _DEU, _unbound_recipient, [2023]),
    ]
    role_names = {sv: RoleNames(donor="creditor", recipient="debtor") for sv in (_SV_A, _SV_B)}
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        rows,
        role_names=role_names,
        facet_metadata={
            "a_fra": FacetMetadata(provenance_id=_PROV_A, unit="CurrencyUSD"),
            "a_deu": FacetMetadata(provenance_id=_PROV_A_OTHER, unit="CurrencyGBP"),
            "b_deu": FacetMetadata(provenance_id=_PROV_B, unit="CurrencyGBP"),
        },
    )
    # Both StatVars see both currencies at both members, so pooling would retain both places.
    observations = {
        _SV_A: {
            (("creditor", _FRA),): [
                _facet(facet_id="a_fra", unit="CurrencyUSD", latest_date="2020")
            ],
            (("creditor", _DEU),): [
                _facet(facet_id="a_deu", unit="CurrencyGBP", latest_date="2023")
            ],
        },
        _SV_B: {
            (("creditor", _FRA),): [
                _facet(facet_id="b_fra", unit="CurrencyUSD", latest_date="2020")
            ],
            (("creditor", _DEU),): [
                _facet(facet_id="b_deu", unit="CurrencyGBP", latest_date="2023")
            ],
        },
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": [_SV_A, _SV_B], "cosine_scores": [0.9, 0.85]}},
        resolve={},
        obs={},
    )
    materialise_graph = FakeGraphReader(
        facets={
            (_SV_A, _FRA): [_facet_from_fixture(_USD_FIXTURE_A)],
            (_SV_B, _DEU): [_facet_from_fixture(_GBP_FIXTURE_B)],
        }
    )
    materialise_graph._arcs = {
        _SV_A: dict(CONVERTIBLE_STAT_VAR_ARCS),
        _SV_B: dict(CONVERTIBLE_STAT_VAR_ARCS),
        **conversion_arcs(
            recipe_id=_RECIPE, provenance_id=_PROV_B, currencies=["CurrencyUSD", "CurrencyGBP"]
        ),
        _PROV_A_OTHER: {"conversionRecipe": (_RECIPE,)},
    }
    materialise_sdmx = _decision_and_fetch_sdmx()
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=[],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_steps=[],
            base_timing={},
            unit_text="USD",
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=EngineOptions(
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=RateTableCache(),
            ),
        )
    )


def _date_range_ends(result) -> dict[str, str | None]:
    spec = result.specs[0]
    return {
        sv.ref: (sv.data_date_range.end if sv.data_date_range is not None else None)
        for sv in spec.resolved.stat_vars.items
    }


def test_per_statvar_election_narrows_each_to_its_own_surviving_member(monkeypatch):
    """Each StatVar's end date comes from its own elected member: 2020 for SV_A, 2023 for SV_B."""
    result = _run(monkeypatch)
    assert result.status == "definite", result
    ends = _date_range_ends(result)
    assert ends[_SV_A] == "2020", ends
    assert ends[_SV_B] == "2023", ends


def test_pooled_union_would_have_widened_both_statvars_the_same_way(monkeypatch):
    """Pooling elected units widens both StatVars' end dates to 2023."""
    real_elected_unit_refs = convertibility_module.elected_unit_refs

    def _pooled(elected):
        real = real_elected_unit_refs(elected)
        union = frozenset().union(*real.values()) if real else frozenset()
        return dict.fromkeys(real, union)

    monkeypatch.setattr("search.engine.region.phase_answer.elected_unit_refs", _pooled)
    result = _run(monkeypatch)
    assert result.status == "definite", result
    ends = _date_range_ends(result)
    assert ends[_SV_A] == "2023", ends
    assert ends[_SV_B] == "2023", ends
