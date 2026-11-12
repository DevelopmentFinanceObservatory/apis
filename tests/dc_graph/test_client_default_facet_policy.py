"""`GraphClient.default_facet_policy` reads a StatVar's `defaultMeasurementMethod` arc.

Offline only: `GraphClient.post` is monkeypatched. The StatVar and method are a ONE Data
Climate Finance received StatVar and the method its pipeline declares on it.
"""

from __future__ import annotations

import logging

import pytest

from dc_graph import FacetDefaultPolicy, GraphClient
from dc_graph import client as client_module
from dfo_platform.lru_memo import LruMemo

_STAT_VAR = "Amount_FinancialObligation_Commitment_ClimateChange_Nominal"
_ONE_WEIGHTING = "ONEClimateFinanceWeighting"


def _client(monkeypatch: pytest.MonkeyPatch, arcs: dict | None, posted: list[dict]) -> GraphClient:
    """A client whose `/v2/node` answers `arcs` for `_STAT_VAR` (`None` omits the node's
    arcs), recording each payload in `posted`."""

    def fake_post(self, path, payload, *, deadline=None):
        posted.append(payload)
        return {"data": {_STAT_VAR: {} if arcs is None else {"arcs": arcs}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    return GraphClient(base_url="http://fake/core/api", arcs_cache_size=100)


def _method_arc(*methods: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "defaultMeasurementMethod": {"nodes": [{"dcid": method} for method in methods]},
    }


@pytest.mark.parametrize("stat_var", [_STAT_VAR, f"dcid:{_STAT_VAR}"])
def test_declared_method_resolves_from_the_arc(monkeypatch, stat_var) -> None:
    """The arc's method is the policy's ref, read for the bare dcid whether or not the
    caller passes materialise's `dcid:` prefix."""
    posted: list[dict] = []
    client = _client(monkeypatch, _method_arc(_ONE_WEIGHTING), posted)

    policy = client.default_facet_policy(stat_var=stat_var, axis="measurementMethod")

    assert policy == FacetDefaultPolicy(
        resolved_ref=_ONE_WEIGHTING,
        policy_node="defaultMeasurementMethod",
        policy_version="1",
        selection_reason="graph_declared",
    )
    assert [p["nodes"] for p in posted] == [[_STAT_VAR]]


@pytest.mark.parametrize(
    "arcs",
    [
        pytest.param({"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}}, id="no_arc"),
        pytest.param(None, id="absent_node"),
    ],
)
def test_no_declared_method_is_none(monkeypatch, arcs) -> None:
    client = _client(monkeypatch, arcs, [])

    assert client.default_facet_policy(stat_var=_STAT_VAR, axis="measurementMethod") is None


@pytest.mark.parametrize("axis", ["provenance", "unit"])
def test_axis_without_a_default_arc_reads_nothing(monkeypatch, axis) -> None:
    posted: list[dict] = []
    client = _client(monkeypatch, _method_arc(_ONE_WEIGHTING), posted)

    assert client.default_facet_policy(stat_var=_STAT_VAR, axis=axis) is None
    assert posted == []


def test_repeat_reads_share_the_arcs_cache(monkeypatch) -> None:
    """Every member of a request asks for the same StatVar's policy, and the StatVar's
    arcs are read once."""
    posted: list[dict] = []
    client = _client(monkeypatch, _method_arc(_ONE_WEIGHTING), posted)

    first = client.default_facet_policy(stat_var=_STAT_VAR, axis="measurementMethod")
    second = client.default_facet_policy(stat_var=f"dcid:{_STAT_VAR}", axis="measurementMethod")

    assert first == second
    assert first is not None
    assert len(posted) == 1


def test_two_declared_methods_declare_nothing_and_log_an_error_once(monkeypatch, caplog) -> None:
    """Two defaults decide nothing, so the read returns no policy, leaving the axis to the
    deployment ranking, and logs the StatVar once however often it is read."""
    monkeypatch.setattr(client_module, "_CONFLICTING_DECLARATIONS_LOGGED", LruMemo(16))
    client = _client(monkeypatch, _method_arc(_ONE_WEIGHTING, "OECDClimateFinanceFullValue"), [])

    with caplog.at_level(logging.ERROR, logger="dc_graph.client"):
        policy = client.default_facet_policy(stat_var=_STAT_VAR, axis="measurementMethod")
        client.default_facet_policy(stat_var=_STAT_VAR, axis="measurementMethod")

    assert policy is None
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert _STAT_VAR in caplog.records[0].getMessage()
