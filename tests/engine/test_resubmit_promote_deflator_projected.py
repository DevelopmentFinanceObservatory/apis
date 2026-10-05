"""A resubmit preview raises `deflator_projected` when its exchange rate comes
from an IMFProjection row. Changing the rate row to Actual removes the warning.
"""

from __future__ import annotations

import httpx

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise.rate_tables import RateTableCache
from search.engine.region import resolve_spec_resubmit
from search.engine.region.codes import DEFLATOR_PROJECTED
from search.engine.region.state import EngineOptions
from search.models import IdentityFacetValueRequest, IdentityUnitRequest, ResubmitIdentityPayload
from tests.engine._base_identity import make_base, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    _facet_from_fixture,
)
from tests.fixtures import FakeGraph
from tests.materialise._rate_table_support import (
    CONVERTIBLE_STAT_VAR_ARCS,
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise.conftest import FakeGraphReader

_ENTITY = "country/KEN"
_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "TestPromoteDeflatorActivity",
    "measured_property": "testPromoteDeflatorExpenditure",
    "stat_type": "measuredValue",
    "measurement_qualifier": "Nominal",
}
_SV = "Test_PromoteDeflatorProjected_Amount"
_PROVENANCE = "dc/base/PromoteDeflatorSrc"
_RECIPE = "ConversionRecipeDAC"
_YEAR = "2022"
_EUR = "CurrencyEUR"
_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_FIVE_TUPLE_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "TestPromoteDeflatorActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testPromoteDeflatorExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
}

_NODES: dict = {
    _SV: {
        "label": "Promote deflator projected amount",
        "type": "StatisticalVariable",
        "arcs": _FIVE_TUPLE_ARCS,
    },
    _ENTITY: {"label": "Kenya", "type": "Country"},
}

# Search coverage reads are separate from materialisation's conversion reads.
_OBS = {
    f"{_SV}|{_ENTITY}": [
        {
            "facetId": "f1",
            "earliestDate": _YEAR,
            "latestDate": _YEAR,
            "obsCount": 1,
            "provenanceId": _PROVENANCE,
            "unit": "CurrencyEUR",
        }
    ],
}

_EUR_FIXTURE = ObsFacetFixture(
    facet_id="f1", points=(), provenance_id=_PROVENANCE, unit="CurrencyEUR"
)


def _sdmx_router(*, eur_projected: bool, main_body: str) -> MaterialiseSdmxClient:
    """Price USD at 1.0 and EUR at 0.9, the EUR row an `IMFProjection` when asked."""

    def observations(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=main_body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    server = RateTableServer(
        recipe_table_bodies(
            _RECIPE,
            rates={"CurrencyUSD": {int(_YEAR): 1.0}, _EUR: {int(_YEAR): 0.9}},
            projected={(_EUR, int(_YEAR))} if eur_projected else (),
        ),
        fallback=observations,
    )
    return server.client(base_url=MATERIALISE_BASE_URL)


def _run(*, eur_projected: bool):
    graph = FakeGraph(nodes=_NODES, obs=_OBS)
    arcs = conversion_arcs(
        recipe_id=_RECIPE, provenance_id=_PROVENANCE, currencies=["CurrencyUSD", _EUR]
    )
    arcs[_PROVENANCE]["isPartOf"] = ()
    materialise_graph = FakeGraphReader(
        arcs={_SV: dict(CONVERTIBLE_STAT_VAR_ARCS), **arcs},
        facets={(_SV, _ENTITY): [_facet_from_fixture(_EUR_FIXTURE)]},
    )
    materialise_sdmx = _sdmx_router(
        eur_projected=eur_projected,
        main_body=_csv_header(["observationAbout"])
        + _csv_row(
            stat_var=_SV,
            role_dcids=[_ENTITY],
            fixture=_EUR_FIXTURE,
            date_str=_YEAR,
            value=100.0,
        ),
    )
    entities = [subject_entity(_ENTITY)]
    # Preview native EUR data in USD using its sole elected facet.
    request = IdentityUnitRequest(currency="USD", prices="current")
    base_spec_id, base_identity = make_base(_SHAPE, slots=[], entities=entities)
    inp = ResubmitIdentityPayload(
        identity=base_identity.model_copy(
            update={"facets": {"unit": IdentityFacetValueRequest(request=request)}}
        ),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_SV],
    )
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            include_observations=True,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        ),
    )


def test_promote_door_raises_deflator_projected_through_the_shared_attach():
    """An `IMFProjection` exchange-rate row raises `deflator_projected`. An actual row clears it."""
    region = _run(eur_projected=True)
    assert region.status == "definite", f"expected definite, got {region.status!r}"
    matches = [w for w in region.warnings if w.code == DEFLATOR_PROJECTED]
    assert len(matches) == 1, (
        f"expected exactly one deflator_projected warning, got {region.warnings}"
    )
    assert "1 row(s)" in matches[0].message
    assert f"{_EUR}@{_YEAR}" in matches[0].message
    assert matches[0].spec_id == region.specs[0].spec_id

    # An actual rate for the same entity and year removes the warning.
    region_actual = _run(eur_projected=False)
    assert region_actual.status == "definite"
    assert not any(w.code == DEFLATOR_PROJECTED for w in region_actual.warnings)
