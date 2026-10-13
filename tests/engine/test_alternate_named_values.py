"""An alternate reading keeps the concept values of the elected reading.

The model can list a shape of another measure as also plausible: GNI at purchasing power
parity for a question that names GDP. A shape whose every member holds another value on a
concept property the elected reading holds is not offered as a candidate, and a lower
price basis of the elected measure gives way to it once the elected shape has data.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.quantity import quantity_requirement
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.alternate_guard import offered_alternates, query_pins
from search.engine.retrieval.derive import derive_shapes
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "GDP in Kenya"
_NOMINAL_GDP = "NominalGdpNamedValueGuard"
_REAL_GDP = "RealGdpNamedValueGuard"
_PPP_GNI = "PppGniNamedValueGuard"
_ESTIMATED_GDP = "EstimatedGdpNamedValueGuard"
_OBS = [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]


def _node(label: str, *, qualifier: str, activity: str) -> dict:
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementQualifier": {"nodes": [{"dcid": qualifier}]},
            "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
            "activitySource": {"nodes": [{"dcid": activity}]},
        },
    }


_NODES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _NOMINAL_GDP: _node(
        "Nominal gross domestic product", qualifier="Nominal", activity="GrossDomesticProduction"
    ),
    _REAL_GDP: _node(
        "Real gross domestic product", qualifier="RealValue", activity="GrossDomesticProduction"
    ),
    _PPP_GNI: _node(
        "Gross national income at purchasing power parity",
        qualifier="PurchasingPowerParity",
        activity="GrossNationalIncome",
    ),
    _ESTIMATED_GDP: _node(
        "Estimated gross domestic product",
        qualifier="Estimated",
        activity="GrossDomesticProduction",
    ),
}


class _NamesGdpLLM:
    """Elects shape 0, binds the activity the query names, and lists every other shape as
    also plausible."""

    def __init__(self, *, evidence: str = "mentioned") -> None:
        self._evidence = evidence

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="what",
                    property_dcid="activitySource",
                    kind="value",
                    value_dcids=["GrossDomesticProduction"],
                    evidence=self._evidence,  # ty: ignore[invalid-argument-type]
                )
            ],
            also_plausible=[1, 2],
            ask=None,
        ), None


def _run(svs: list[str], *, llm=None):
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": svs, "cosine_scores": [0.9, 0.7, 0.5][: len(svs)]}},
        resolve={"Kenya": "country/KEN"},
        obs={f"{sv}|country/KEN": _OBS for sv in svs},
    )
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm or _NamesGdpLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def _served(result) -> list[list[str]]:
    return [[sv.ref.ref for sv in spec.resolved.members] for spec in result.specs]


def test_an_alternate_of_another_measure_is_not_a_candidate():
    result = _run([_NOMINAL_GDP, _PPP_GNI])

    assert result.status == "definite"
    assert _served(result) == [[_NOMINAL_GDP]]


def test_an_alternate_holding_the_named_value_stays_a_candidate():
    """An estimate of GDP holds the named activity and states no price basis, so it stays a
    reading beside nominal GDP."""
    result = _run([_NOMINAL_GDP, _ESTIMATED_GDP, _PPP_GNI])

    assert result.status == "candidates"
    assert sorted(_served(result)) == [[_ESTIMATED_GDP], [_NOMINAL_GDP]]


def test_a_lower_price_basis_of_the_elected_measure_is_not_a_candidate():
    """Real GDP ranks below nominal GDP, which has data, and the query names no price
    basis, so the default reading answers alone."""
    result = _run([_NOMINAL_GDP, _REAL_GDP, _PPP_GNI])

    assert result.status == "definite"
    assert _served(result) == [[_NOMINAL_GDP]]


def test_a_value_bound_as_the_slots_only_option_rejects_another_measure():
    """The elected reading holds GDP whatever the binding's evidence, so GNI is another
    concept."""
    result = _run([_NOMINAL_GDP, _PPP_GNI], llm=_NamesGdpLLM(evidence="sole_option"))

    assert result.status == "definite"
    assert _served(result) == [[_NOMINAL_GDP]]


def test_an_alternate_is_offered_through_the_members_holding_the_elected_values():
    graph = FakeGraph(nodes=_NODES, obs={}, detect={}, resolve={})
    (gdp,) = derive_shapes(confirmed_svs=[_NOMINAL_GDP], graph=graph)
    (gni,) = derive_shapes(confirmed_svs=[_PPP_GNI], graph=graph)
    (gdp_other_basis,) = derive_shapes(confirmed_svs=[_REAL_GDP], graph=graph)

    offered = offered_alternates(
        query_pins(_QUERY, quantity=quantity_requirement([_QUERY])),
        gdp,
        {1: gni, 2: gdp_other_basis},
    )

    assert list(offered[1].sv_arc_facts or {}) == []
    assert list(offered[2].sv_arc_facts or {}) == [_REAL_GDP]
