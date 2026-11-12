"""A StatVar's `defaultMeasurementMethod` arc decides the method `select_facet` serves.

The graph is a real `GraphClient` with its transport monkeypatched, so the policy comes
from the arc read rather than a fake reader. The StatVar, provenance and methods are ONE Data
Climate Finance's: its pipeline declares `ONEClimateFinanceWeighting` on a received StatVar,
which also carries a full-value facet.
"""

from __future__ import annotations

import logging

import pytest

from dc_graph import GraphClient
from materialise.facets import (
    DEPLOYMENT_FACET_POLICY,
    FacetBinding,
    declared_default_policies,
    select_facet,
)
from materialise.graph import Facet

_STAT_VAR = "dcid:Amount_FinancialObligation_Commitment_ClimateChange_Nominal"
_AXIS = "measurementMethod"
_UNBOUND = {_AXIS: FacetBinding(state="unbound")}
_ONE_WEIGHTING = "ONEClimateFinanceWeighting"

_BROAD = Facet(
    facet_id="fullValue",
    provenance_id="provenance/ONEClimateFinance",
    measurement_method="OECDClimateFinanceFullValue",
    unit="USDollar",
    obs_count=900,
    earliest_date="2012",
    latest_date="2023",
)
_DECLARED = Facet(
    facet_id="weighting",
    provenance_id="provenance/ONEClimateFinance",
    measurement_method=_ONE_WEIGHTING,
    unit="USDollar",
    obs_count=120,
    earliest_date="2018",
    latest_date="2023",
)


def _graph(monkeypatch: pytest.MonkeyPatch, declared: str | None) -> GraphClient:
    arcs: dict = {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}}
    if declared is not None:
        arcs["defaultMeasurementMethod"] = {"nodes": [{"dcid": declared}]}

    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {node: {"arcs": arcs} for node in payload["nodes"]}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    return GraphClient(base_url="http://fake/core/api")


def _select(graph: GraphClient, candidates: list[Facet]):
    return select_facet(
        _AXIS,
        FacetBinding(state="unbound"),
        candidates=candidates,
        graph=graph,
        stat_var=_STAT_VAR,
        window=None,
        policies=declared_default_policies(_UNBOUND, graph=graph, stat_var=_STAT_VAR),
    )


@pytest.mark.parametrize(
    ("declared", "ref", "basis", "policy"),
    [
        pytest.param(
            _ONE_WEIGHTING,
            _ONE_WEIGHTING,
            "declared_default",
            "defaultMeasurementMethod",
            id="declared_over_more_observations",
        ),
        pytest.param(
            None,
            "OECDClimateFinanceFullValue",
            "deployment_fallback",
            DEPLOYMENT_FACET_POLICY,
            id="no_arc_ranks",
        ),
    ],
)
def test_declared_method_decides_the_unbound_axis(monkeypatch, declared, ref, basis, policy):
    selection = _select(_graph(monkeypatch, declared), [_BROAD, _DECLARED])

    assert (selection.ref, selection.basis, selection.default_policy) == (ref, basis, policy)


def test_a_declared_method_no_candidate_carries_serves_the_ranking_and_logs_it(monkeypatch, caplog):
    """A declared method the probe did not find serves the ranking's one method with a
    fallback basis, and one log line names the StatVar and the declared method."""
    with caplog.at_level(logging.INFO, logger="materialise.facets"):
        selection = _select(_graph(monkeypatch, _ONE_WEIGHTING), [_BROAD])

    assert (selection.ref, selection.basis, selection.dispatch_facet_id) == (
        "OECDClimateFinanceFullValue",
        "deployment_fallback",
        "fullValue",
    )
    (record,) = caplog.records
    assert _STAT_VAR in record.getMessage()
    assert _ONE_WEIGHTING in record.getMessage()
