"""Tests role-currency phrases through the pipeline to the wire request.

A role-relative phrase binds to a role shape, which the engine converts per member: the
bound entity's currency comes from `nationalCurrencyOf`, and a role the shape does not declare
refuses `recipe_not_declared`.
"""

from __future__ import annotations

import asyncio

import httpx

from materialise.rate_tables import RateTableCache
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.models import IdentityCurrencyRole
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
    door_roles_for_test_query,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_STANDARD_SV = "TESTFIXTURE/UnitBindRolePipeline/Standard"
_ROLE_SV = "TESTFIXTURE/UnitBindRolePipeline/RoleBearing"
_KEN = "country/KEN"
_NOR = "country/NOR"

_STANDARD_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "name": {"nodes": [{"value": "Unit bind role pipeline standard"}]},
    "populationType": {"nodes": [{"dcid": "TestUnitBindRoleActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testUnitBindRoleAmount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}

_ROLE_BEARING_ARCS = {
    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
}


_SDMX_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"
_PROVENANCE = "dc/base/UnitBindRoleSrc"
_RECIPE = "ConversionRecipeDAC"


_USD_FACET = ObsFacetFixture(
    facet_id="native", points=(("2022", 100.0),), provenance_id=_PROVENANCE, unit="CurrencyUSD"
)


def _materialise_options() -> EngineOptions:
    graph = FakeGraphReader(
        facets={(_STANDARD_SV, _KEN): [_facet_from_fixture(_USD_FACET)]},
        arcs={
            _STANDARD_SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            _ROLE_SV: dict(CONVERTIBLE_STAT_VAR_ARCS),
            **conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROVENANCE,
                currencies={"CurrencyUSD": (), "CurrencyKES": (_KEN,)},
            ),
        },
    )
    native_body = _csv_header(["observationAbout"]) + _csv_row(
        stat_var=_STANDARD_SV, role_dcids=[_KEN], fixture=_USD_FACET, date_str="2022", value=100.0
    )

    def native_rows(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=native_body, headers={"content-type": _SDMX_CSV})

    server = RateTableServer(
        recipe_table_bodies(
            _RECIPE,
            rates={"CurrencyUSD": {2022: 1.0}, "CurrencyKES": {2022: 130.0}},
        ),
        fallback=native_rows,
    )
    return EngineOptions(
        materialise_graph=graph,
        materialise_sdmx=server.client(base_url=MATERIALISE_BASE_URL),
        rate_cache=RateTableCache(),
    )


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _run_standard(*, unit_text: str, sdmx: EngineSdmxClient | None = None):
    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            _STANDARD_SV: {"label": "Unit bind role pipeline standard", "arcs": _STANDARD_ARCS},
        },
        detect={"unit bind role pipeline query": {"svs": [_STANDARD_SV], "cosine_scores": [0.9]}},
        resolve={"Kenya": _KEN},
        obs={
            f"{_STANDARD_SV}|{_KEN}": [
                {
                    # The same facet the SDMX rows carry: the answer's read plan elects it,
                    # and the preview dispatches its id.
                    "facetId": _USD_FACET.facet_id,
                    "earliestDate": "2022",
                    "latestDate": "2022",
                    "obsCount": 1,
                    "provenanceId": _PROVENANCE,
                    "unit": "CurrencyUSD",
                    "dates": ["2022"],
                    "values": [100.0],
                }
            ]
        },
    )
    return asyncio.run(
        resolve_variable(
            "unit bind role pipeline query",
            entities=["Kenya"],
            date_request=None,
            detect_query="unit bind role pipeline query",
            role_query="unit bind role pipeline query",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx if sdmx is not None else FakeSdmxClient(),
            base_timing={},
            unit_text=unit_text,
            coverage=empty_coverage_snapshot(),
            options=_materialise_options(),
        )
    )


def _run_role_bearing(*, unit_text: str):
    role_query = "aid from Norway to Kenya"
    sdmx = FakeSdmxClient(observations={_ROLE_SV: {(("provider", _NOR), ("recipient", _KEN)): []}})
    graph = FakeGraph(
        nodes={
            _NOR: {"label": "Norway", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _ROLE_SV: {"label": "Unit bind role pipeline flow", "arcs": _ROLE_BEARING_ARCS},
        },
        detect={role_query: {"svs": [_ROLE_SV], "cosine_scores": [0.9]}},
        resolve={"Norway": _NOR, "Kenya": _KEN},
        obs={},
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=["Norway", "Kenya"],
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["Norway", "Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            unit_text=unit_text,
            coverage=empty_coverage_snapshot(),
            options=_materialise_options(),
        )
    )


class TestFixedRolePhraseReachesTheWireAsARoleObject:
    def test_donor_currency_on_a_standard_query_binds_to_origin(self) -> None:
        # A fixed-role phrase binds to its role regardless of shape/bound
        # roles. A standard query declares only observationAbout, so the
        # origin-shaped request refuses at conversion, naming the role.
        result = _run_standard(unit_text="donor currency")
        assert result.status == "unsupported"
        assert result.unsupported.code == "recipe_not_declared"
        assert result.unsupported.cause.code == "role_undeclared"

    def test_local_currency_on_a_standard_query_binds_to_observation_about(self) -> None:
        # A "beside" phrase resolves to the query's one bound role,
        # observationAbout for a standard (non-role-bearing) query, and
        # converts to the currency Kenya uses.
        result = _run_standard(unit_text="local currency")
        assert result.status == "definite"
        unit_binding = result.specs[0].definition.identity.facets["unit"]
        assert unit_binding.request.currency == IdentityCurrencyRole(role="observationAbout")

    def test_role_phrase_combined_with_an_iso_currency_refuses_no_data(self) -> None:
        result = _run_standard(unit_text="donor currency USD")
        assert result.status == "no_data"
        assert result.no_data_reason == "unit_not_resolved"


class TestBesidePhraseOnTwoBoundRolesRefuses:
    def test_local_currency_beside_both_directional_roles_refuses_with_both_role_shapes(
        self,
    ) -> None:
        # A role-bearing query binding BOTH origin and destination leaves no
        # single role for a "beside" phrase to mean. The refusal offers both.
        result = _run_role_bearing(unit_text="local currency")
        assert result.status == "no_data"
        assert result.no_data_reason == "unit_not_resolved"
        assert [a.currency for a in result.unit_alternatives] == [
            IdentityCurrencyRole(role="destination"),
            IdentityCurrencyRole(role="origin"),
        ]

    def test_donor_currency_beside_both_directional_roles_still_binds(self) -> None:
        # A fixed-role phrase is unaffected by how many roles are bound. It
        # passes binding and stops at the data, which this fixture lacks.
        result = _run_role_bearing(unit_text="donor currency")
        assert result.status == "no_data"
        assert result.no_data_reason == "facet_not_available"
