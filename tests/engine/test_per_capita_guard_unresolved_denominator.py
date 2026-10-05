"""The per-capita guard distinguishes an unresolved denominator arc from no arc.

An unresolved target preserves a valid per-capita result; a missing arc rejects
it. Both cases run through derive_shapes, bind, and the guard.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.region import resolve_variable
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.retrieve import Materialised
from search.models import CoverageBare
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "GDP per capita of Vietnam"
_SV = "Amount_EconomicActivity_GDP_Test_PerCapita"


def _gdp_arcs(*, denominator: dict | None) -> dict:
    # Modelled on the real dfo-dev arc for
    # Amount_EconomicActivity_GrossDomesticProduction_Nominal_PerCapita: its
    # measurementDenominator is a dcid-less ProvisionalNode.
    arcs: dict = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Nominal gross domestic product per capita"}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if denominator is not None:
        arcs["measurementDenominator"] = denominator
    return arcs


class _ElectSingleShapeLLM:
    """Elects the sole candidate shape (index 0) with no constraint bindings
    and no also_plausible alternates -- the shape has no constraint slots, so
    an empty bindings list is a complete, valid bind() output for it."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[],
                also_plausible=[],
                ask=None,
            ),
            None,
        )


def _run(*, denominator: dict | None, monkeypatch):
    nodes = {
        "country/VNM": {"label": "Vietnam", "type": "Country"},
        _SV: {
            "label": "Nominal gross domestic product per capita",
            "type": "StatisticalVariable",
            "arcs": _gdp_arcs(denominator=denominator),
        },
    }
    detect = {_QUERY: {"svs": [_SV], "cosine_scores": [0.9]}}
    graph = FakeGraph(nodes=nodes, detect=detect, resolve={"Vietnam": "country/VNM"}, obs={})

    # materialise itself is monkeypatched to a guaranteed-successful result --
    # these tests isolate the guard's own precedence decision from graph/sdmx
    # observation resolution, mirroring test_denominator_order.py's pattern.
    monkeypatch.setattr(
        "search.engine.region.materialise_step.materialise",
        lambda *args, **kwargs: Materialised(
            sv_dcids=[_SV],
            has_data=True,
            coverage=CoverageBare(has_data=True, window=None),
            # A real resolve path always populates this (Materialised's own
            # docstring); _phase_assemble_definite reads it unconditionally
            # to observe this Spec's own served measure partition.
            member_facts={_SV: ArcsMemberFacts(arcs=_gdp_arcs(denominator=denominator))},
        ),
    )

    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Vietnam"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectSingleShapeLLM(),
            base_steps=[],
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
    )


def test_unresolved_denominator_guard_does_not_fire(monkeypatch):
    """An unresolved denominator arc preserves the per-capita result."""
    result = _run(
        denominator={"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]},
        monkeypatch=monkeypatch,
    )
    assert result.status == "definite", (
        f"expected the guard to stand down for an unresolved-but-present denominator, "
        f"got status={result.status!r} reason={result.no_data_reason!r}"
    )
    assert len(result.specs) == 1
    assert result.specs[0].resolved.stat_vars.items[0].ref == _SV


def test_genuinely_bare_denominator_guard_fires(monkeypatch):
    """A missing denominator rejects the base total for a per-capita query."""
    result = _run(denominator=None, monkeypatch=monkeypatch)
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
