"""End-to-end tests of a bound-unit request whose per-series preview cannot be shown.

The fixture groups SV_A and SV_B into one spec with native USD facets for Kenya. The bound-unit
gate decides from the read plan (`bound_unit_gate`), so an over-cap preview or a failed preview
read leaves the answer definite, with the preview's own omission or warning.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import UpstreamResponseError
from materialise.graph import SubjectPage
from materialise.rate_tables import RateTableCache
from search.engine.bind import SlotBindingDraft
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import (
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_TIMED_OUT,
)
from search.engine.region.observations import fetch as fetch_module
from search.engine.region.state import EngineOptions
from search.observations import OBSERVATIONS_PREVIEW_TOO_LARGE
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    MATERIALISE_BASE_URL,
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    materialise_double_from_obs,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_PROP = "financingSource"
_VAL_A = "SourceA"
_VAL_B = "SourceB"

_FIVE_TUPLE_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "TestBoundUnitScopeActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testBoundUnitScopeExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    # Declare Nominal on both graphs so their five-tuples agree during the USD unit check.
    "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
}


def _sv_arcs(label: str, value: str) -> dict:
    return {
        **_FIVE_TUPLE_ARCS,
        "name": {"nodes": [{"value": label}]},
        "constraintProperties": {"nodes": [{"dcid": _PROP}]},
        _PROP: {"nodes": [{"dcid": value}]},
    }


_SV_A = "Test_BoundUnitScope_SourceA"
_SV_B = "Test_BoundUnitScope_SourceB"
_QUERY = "bound unit scope too broad test variable in Kenya"

_NODES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _SV_A: {
        "label": "Bound unit scope SourceA",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("Bound unit scope too broad test variable, source A", _VAL_A),
    },
    _SV_B: {
        "label": "Bound unit scope SourceB",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("Bound unit scope too broad test variable, source B", _VAL_B),
    },
}


def _member_arcs(node_arcs: dict) -> dict[str, tuple[str, ...]]:
    return {
        prop: tuple(n.get("dcid") or n.get("value") for n in entry["nodes"])
        for prop, entry in node_arcs.items()
    }


def _set_bind_llm() -> object:
    class _SetBindLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema is QuantityOutput:
                return NO_QUANTITY
            assert schema.__name__ == "_UniversalBindOutput"
            return elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid=_PROP,
                        kind="set",
                        value_dcids=[_VAL_A, _VAL_B],
                    )
                ],
                also_plausible=None,
            ), None

    return _SetBindLLM()


def _obs_facet(*, unit: str = "CurrencyUSD", value: float = 10.0) -> dict:
    return {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": "dc/base/Src",
        "unit": unit,
        "dates": ["2020"],
        "values": [value],
    }


def _materialise_sdmx(obs: dict) -> MaterialiseSdmxClient:
    """Serve one CSV body per StatVar. Native USD facets need no rate lookup."""
    bodies: dict[str, str] = {}
    for key, facet_dicts in obs.items():
        sv, entity = key.split("|", 1)
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        for i, facet_dict in enumerate(facet_dicts):
            fixture = ObsFacetFixture(
                facet_id=facet_dict.get("facetId") or f"f{i}",
                unit=facet_dict.get("unit"),
                provenance_id=facet_dict.get("provenanceId"),
            )
            body += _csv_row(
                stat_var=sv,
                role_dcids=[entity],
                fixture=fixture,
                date_str=facet_dict["dates"][0],
                value=facet_dict["values"][0],
            )
        bodies[sv] = body

    def handler(request: httpx.Request) -> httpx.Response:
        sv = request.url.params.get("c[variableMeasured]", "")
        body = bodies.get(sv, _csv_header(["observationAbout"]))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    return MaterialiseSdmxClient(
        base_url=MATERIALISE_BASE_URL, transport=httpx.MockTransport(handler)
    )


def _run():
    """SV_A and SV_B each have a native USD facet for Kenya, giving two preview pairs."""
    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet()],
        f"{_SV_B}|country/KEN": [_obs_facet()],
    }
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": [_SV_A, _SV_B], "cosine_scores": [0.95, 0.9]}},
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    materialise_graph, _unused_sdmx = materialise_double_from_obs(obs)
    materialise_sdmx = _materialise_sdmx(obs)
    materialise_graph._arcs = {
        _SV_A: {**_member_arcs(_NODES[_SV_A]["arcs"]), "measurementQualifier": ("Nominal",)},
        _SV_B: {**_member_arcs(_NODES[_SV_B]["arcs"]), "measurementQualifier": ("Nominal",)},
        "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
    }
    materialise_graph._subjects = {
        ("testBoundUnitScopeExpenditure", "measuredProperty"): SubjectPage(
            dcids=(_SV_A, _SV_B), truncated=False
        ),
    }
    return resolve_variable(
        _QUERY,
        entities=["Kenya"],
        date_request=None,
        detect_query=_QUERY,
        role_query=_QUERY,
        pac=True,
        graph=graph,
        resolver=make_resolver(graph),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        llm=_set_bind_llm(),
        base_timing={},
        unit_text="US dollars",
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            include_observations=True,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            rate_cache=RateTableCache(),
        ),
    )


def test_over_cap_spec_under_bound_unit_answers_definite_with_the_preview_omitted(
    search_settings,
):
    """The read plan serves both StatVars under the USD unit, so the gate passes without a
    fetch. The preview's two series exceed the cap of 1, so it omits them as it does without
    a unit request; the answer neither refuses `scope_too_broad` nor escalates."""
    search_settings(obs_preview_max_pairs=1)

    result = asyncio.run(_run())

    assert result.status == "definite", (
        f"expected definite, got status={result.status!r} "
        f"(no_data_reason={result.no_data_reason!r})"
    )
    assert result.unsupported is None
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PREVIEW_TOO_LARGE
    assert (observations.omitted.pairs, observations.omitted.cap) == (2, 1)
    assert observations.series == []


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (UpstreamResponseError("role divergence"), OBSERVATIONS_PROBE_FAILED),
        (TimeoutError("simulated shared-deadline crossing"), OBSERVATIONS_TIMED_OUT),
    ],
)
def test_failed_preview_under_bound_unit_answers_definite_with_its_warning(
    monkeypatch, raised, code
):
    """The plan decides the unit, so a failed preview read is a display failure: the answer
    is definite and carries the preview's warning, as it does without a unit request."""

    def _fails(*args, **kwargs):
        raise raised

    monkeypatch.setattr(fetch_module, "materialise_preview", _fails)

    result = asyncio.run(_run())

    assert result.status == "definite", result
    assert code in {w.code for w in result.warnings}
