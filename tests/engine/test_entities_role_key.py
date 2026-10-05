"""Tests that the bulk role builder names each role-tuple entity by its wire role key."""

from __future__ import annotations

from search.engine.clients.graph import Facet
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import bulk_role_series_planned
from tests.fixtures import FakeGraph

_SV = "TEST/RoleKeyByAxis_Flow"
_DONOR = "country/RKDonor"
_RECIPIENT = "country/RKRecipient"
_ROLE_TUPLE = (("financingProvider", _DONOR), ("financingRecipient", _RECIPIENT))
_GRAPH = FakeGraph(
    nodes={
        _SV: {
            "label": "Role Key By Axis Flow",
            "arcs": {
                "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "observationProperties": {
                    "nodes": [
                        {"dcid": "financingProvider"},
                        {"dcid": "financingRecipient"},
                    ]
                },
            },
        },
        _DONOR: {"label": "Donor"},
        _RECIPIENT: {"label": "Recipient"},
    },
    obs={},
)


def test_financing_provider_recipient_role_tuple_names_both_entities(monkeypatch) -> None:
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV, "F1", _DONOR, _RECIPIENT, [2020])],
        role_names={_SV: RoleNames(donor="financingProvider", recipient="financingRecipient")},
    )
    sdmx = FakeSdmxClient(
        observations={
            _SV: {
                _ROLE_TUPLE: [
                    Facet(
                        earliest_date="2020",
                        latest_date="2020",
                        obs_count=1,
                        dates=["2020"],
                        values=[1.0],
                        facet_id="F1",
                    )
                ]
            }
        }
    )
    observations, warn = bulk_role_series_planned(
        [_SV],
        {"financingProvider": [_DONOR], "financingRecipient": [_RECIPIENT]},
        sdmx=sdmx,
        graph=_GRAPH,
        facet_filter=None,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert len(observations.series) == 1
    entities = {(e.role, e.ref) for e in observations.series[0].entities}
    assert entities == {("origin", _DONOR), ("destination", _RECIPIENT)}
