"""Regression tests for duplicate graph arcs in ``LiveGraphClient.node_arcs_batch`` (#698).

Raw ``/v2/node`` responses exercise the real client method."""

from __future__ import annotations

import pytest

import dc_graph
from dc_graph import UNRESOLVED_NODE
from materialise.roles import roles_by_axis
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.region import detect_set_ref
from search.engine.region.promote_reread import _family_measurement_qualifier
from search.engine.retrieval.facts import read_declared_roles
from tests.materialise._dup_arcs_fixtures import doubled_wire_arcs


def test_dual_provenance_ispartof_still_yields_one_parent_for_set_ref(monkeypatch) -> None:
    # (#698) Duplicate provenance arcs must leave one parent for set-ref detection.
    def fake_post(self, path, payload, *, deadline=None):
        prop = payload["property"]
        if prop == "->*":
            return {
                "data": {
                    dcid: {"arcs": doubled_wire_arcs({"isPartOf": ("Agg",)})}
                    for dcid in payload["nodes"]
                }
            }
        if prop == "<-isPartOf":
            return {
                "data": {"Agg": {"arcs": {"isPartOf": {"nodes": [{"dcid": "c1"}, {"dcid": "c2"}]}}}}
            }
        assert prop == "->name"
        return {"data": {"Agg": {"arcs": {"name": {"nodes": [{"value": "Aggregate Label"}]}}}}}

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    result = detect_set_ref(value_dcids=["c1", "c2"], graph=client)

    assert result is not None
    assert result.dcid == "Agg"
    assert result.label == "Aggregate Label"


def test_dual_provenance_observation_properties_resolves_role_axes_instead_of_raising(
    monkeypatch,
) -> None:
    # (#698) Duplicate role arcs must collapse before axis validation.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/FLOW": {
                    "arcs": doubled_wire_arcs(
                        {"observationProperties": ("financingProvider", "financingRecipient")}
                    )
                }
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    arcs = client.node_arcs("sv/FLOW")
    roles = read_declared_roles(arcs)

    assert roles == ("financingProvider", "financingRecipient")
    assert roles_by_axis(roles) == {
        "origin": "financingProvider",
        "destination": "financingRecipient",
    }


def test_dual_provenance_genuinely_conflicting_roles_still_raise(monkeypatch) -> None:
    # Distinct role names remain a conflict after duplicate arcs collapse.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/AMBIGUOUS": {
                    "arcs": {
                        "observationProperties": {
                            "nodes": [
                                {"dcid": "financingProvider", "provenanceId": "dc/base/A"},
                                {"dcid": "provider", "provenanceId": "dc/base/B"},
                                {"dcid": "financingRecipient", "provenanceId": "dc/base/A"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    arcs = client.node_arcs("sv/AMBIGUOUS")
    roles = read_declared_roles(arcs)

    with pytest.raises(ValueError, match="origin"):
        roles_by_axis(roles)


def test_dual_provenance_unresolvable_qualifier_still_reports_unresolved_node(monkeypatch) -> None:
    # A dcid-less qualifier target stays present after duplicate arcs collapse.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "sv/UNRESOLVABLE_QUAL": {
                    "arcs": {
                        "measurementQualifier": {
                            "nodes": [
                                {"provenanceId": "dc/base/India_LifeExpectancy"},
                                {"provenanceId": "dc/base/Schema"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(dc_graph.GraphClient, "post", fake_post)
    client = LiveGraphClient(base="http://fake")

    arcs = client.node_arcs("sv/UNRESOLVABLE_QUAL")

    assert _family_measurement_qualifier(arcs) == UNRESOLVED_NODE
