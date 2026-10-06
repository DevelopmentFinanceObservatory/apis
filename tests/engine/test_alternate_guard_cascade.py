"""Test alternate-guard decisions for shapes added by the near-neighbour cascade.

The fixtures put a rejected cross-population shape and an admitted role-bearing
shape in the cascade, then check their menu and warning outcomes end to end.
The gate fires on a data-less flow scored 0.846, so the 0.840 shape under test reaches
the menu through the cascade alone. None of these five-tuples is in the packaged code
index, so each shape resolves from its own arcs.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.region import (
    NEAR_NEIGHBOR_AMBIGUOUS,
    NEAR_NEIGHBOR_POOL_WIDENED,
    RegionResult,
    resolve_variable,
)
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.fixtures import FakeGraph, make_resolver

_SV_AID_FLOW = "TESTFIXTURE/AlternateGuardCascade/AidFlow"
_SV_TIED_FLOW = "TESTFIXTURE/AlternateGuardCascade/TiedFlow"
_SV_SDG_CREDIT_ACCESS = "TESTFIXTURE/AlternateGuardCascade/SdgCreditAccess"
_SV_FLOW_NET = "TESTFIXTURE/AlternateGuardCascade/FlowNet"

_QUERY = "concessional loans to Kenya"
_KEN = "country/KEN"


def _flow_arcs(*, meas_prop: str, qualifier: str) -> dict:
    """Build role-bearing shapes that differ by measure and qualifier."""
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": qualifier}]},
        "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
    }


_NODES: dict = {
    _SV_AID_FLOW: {
        "label": "Concessional loans received (grant equivalent)",
        "arcs": _flow_arcs(meas_prop="amount", qualifier="Nominal"),
    },
    _SV_TIED_FLOW: {
        # The shared population and roles leave this measuredProperty variant admissible.
        "label": "Concessional loans received as a share of gross domestic product",
        "arcs": _flow_arcs(meas_prop="shareOfGdp", qualifier="Nominal"),
    },
    _SV_SDG_CREDIT_ACCESS: {
        "label": "Proportion of small-scale industries with a loan or line of credit",
        "arcs": {
            "populationType": {"nodes": [{"dcid": "SDG_Series"}]},
            "measuredProperty": {"nodes": [{"dcid": "value"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _SV_FLOW_NET: {
        # R5 admits this distinct measure because population type and roles match.
        "label": "Concessional loans received (net disbursements)",
        "arcs": _flow_arcs(meas_prop="netDisbursement", qualifier="Nominal"),
    },
    _KEN: {"label": "Kenya", "type": "Country"},
}

# Scores 0.850, 0.846, and 0.840 put the tested shape inside the margin and below
# the fired neighbour. Only the 0.846 shape enters also_plausible.
_DETECT_A = {
    _QUERY: {
        "svs": [_SV_AID_FLOW, _SV_TIED_FLOW, _SV_SDG_CREDIT_ACCESS],
        "cosine_scores": [0.850, 0.846, 0.840],
    }
}
_DETECT_B = {
    _QUERY: {
        "svs": [_SV_AID_FLOW, _SV_TIED_FLOW, _SV_FLOW_NET],
        "cosine_scores": [0.850, 0.846, 0.840],
    }
}

# Here the gate's fired SDG index is rejected, so both warning and ordering clear.
_DETECT_FIRED_IS_REJECTED = {
    _QUERY: {
        "svs": [_SV_AID_FLOW, _SV_SDG_CREDIT_ACCESS, _SV_FLOW_NET],
        "cosine_scores": [0.850, 0.846, 0.840],
    }
}

# The elected role flow has Kenya data. The tied flow's empty probe triggers the cascade.
_ROLE_OBS: dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]] = {
    _SV_AID_FLOW: {
        (("recipient", _KEN),): [
            Facet(
                earliest_date="2015",
                latest_date="2020",
                obs_count=4,
                dates=["2020"],
                values=[1.0],
                facet_id="f1",
            )
        ]
    },
    _SV_FLOW_NET: {
        (("recipient", _KEN),): [
            Facet(
                earliest_date="2015",
                latest_date="2020",
                obs_count=4,
                dates=["2020"],
                values=[2.0],
                facet_id="f1",
            )
        ]
    },
}
# The SDG shape is subject-only and resolves at Kenya through the graph's plain
# per-entity facets, so it carries real data without the guard.
_GRAPH_OBS = {
    f"{_SV_SDG_CREDIT_ACCESS}|{_KEN}": [
        {"earliestDate": "2015", "latestDate": "2020", "obsCount": 4}
    ],
}

# Role flows bind Kenya as recipient and leave provider open. The tied flow has no coverage row.
_ROLE_NAMES = RoleNames(donor="provider", recipient="recipient")
_COVERAGE_ROWS = [
    (_SV_AID_FLOW, "f1", "country/USA", _KEN, [2017, 2018, 2019, 2020]),
    (_SV_FLOW_NET, "f1", "country/USA", _KEN, [2017, 2018, 2019, 2020]),
]


class _OneAlsoPlausibleBindLLM:
    """Elect the top-ranked shape and nominate index 2 as an alternate."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=[2],
            ask=None,
        ), None


class _NoAlsoPlausibleBindLLM:
    """Elect the top-ranked shape without an alternate nomination."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[],
            also_plausible=None,
            ask=None,
        ), None


def _run(monkeypatch, detect: dict, llm=None) -> RegionResult:
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        _COVERAGE_ROWS,
        role_names={_SV_AID_FLOW: _ROLE_NAMES, _SV_FLOW_NET: _ROLE_NAMES},
    )
    graph = FakeGraph(nodes=_NODES, detect=detect, resolve={"Kenya": _KEN}, obs=_GRAPH_OBS)
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            entity_roles=door_roles_for_test_query(_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations=_ROLE_OBS, availability={}),
            llm=llm or _NoAlsoPlausibleBindLLM(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def _sv_dcids(result: RegionResult) -> set[str]:
    return {sv.ref.dcid for spec in result.specs for sv in spec.resolved.members}


def test_guard_rejecting_the_fired_index_retracts_the_ordering_flag(monkeypatch):
    """Rejecting the gate's fired shape clears the ambiguity warning and ordering metadata."""
    result = _run(monkeypatch, _DETECT_FIRED_IS_REJECTED, llm=_OneAlsoPlausibleBindLLM())
    assert result.status == "candidates"
    assert len(result.specs) == 2
    assert NEAR_NEIGHBOR_AMBIGUOUS not in [warning.code for warning in result.warnings]
    assert result.near_neighbor_fired is False
    assert result.candidate_near_neighbor_spec_ids == frozenset()


def test_rejected_shape_never_reaches_the_menu_through_the_cascade(monkeypatch):
    """The cascade must exclude the in-margin SDG shape rejected by the guard."""
    result = _run(monkeypatch, _DETECT_A)
    assert _SV_SDG_CREDIT_ACCESS not in _sv_dcids(result)


def test_rejected_shape_as_the_only_cascade_candidate_does_not_widen_the_pool(monkeypatch):
    """With the cascade's only candidate rejected, the elected reading stands alone."""
    result = _run(monkeypatch, _DETECT_A)
    assert result.status == "definite"
    assert NEAR_NEIGHBOR_POOL_WIDENED not in [warning.code for warning in result.warnings]
    assert _sv_dcids(result) == {_SV_AID_FLOW}


def test_admitted_shape_enters_through_the_widened_cascade(monkeypatch):
    """An admissible role-bearing variant enters through the cascade with both warnings."""
    result = _run(monkeypatch, _DETECT_B)
    assert result.status == "candidates"
    codes = [warning.code for warning in result.warnings]
    assert NEAR_NEIGHBOR_AMBIGUOUS in codes
    assert NEAR_NEIGHBOR_POOL_WIDENED in codes
    sv_dcids = _sv_dcids(result)
    assert _SV_FLOW_NET in sv_dcids
    assert _SV_AID_FLOW in sv_dcids
