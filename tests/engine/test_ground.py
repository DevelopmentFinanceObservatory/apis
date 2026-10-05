"""Tests for graphref and graphrefs grounding functions."""

from __future__ import annotations

import pytest

from search.engine.errors import EngineInfraError, GroundingMiss
from search.engine.ground import graphref, graphrefs
from search.models import GraphRef
from tests.fixtures import FakeGraph


class TestGraphref:
    def test_existing_node_returns_graphref(self):
        graph = FakeGraph()
        ref = graphref("country/ETH", graph=graph)
        assert isinstance(ref, GraphRef)
        assert ref.dcid == "country/ETH"
        assert ref.label == "Ethiopia"

    def test_absent_node_raises_grounding_miss(self):
        graph = FakeGraph()
        with pytest.raises(GroundingMiss) as exc_info:
            graphref("country/DOESNOTEXIST", graph=graph)
        assert exc_info.value.dcid == "country/DOESNOTEXIST"

    def test_transport_error_raises_engine_infra_error(self):
        graph = FakeGraph(raise_on_call=True)
        with pytest.raises(EngineInfraError):
            graphref("country/ETH", graph=graph)

    def test_well_known_nodes(self):
        graph = FakeGraph()
        for dcid, expected_label in [
            ("Person", "Person"),
            ("GrossDomesticProduction", "GrossDomesticProduction"),
            (
                "Amount_EconomicActivity_GrossDomesticProduction_Nominal",
                "Nominal gross domestic product",
            ),
        ]:
            ref = graphref(dcid, graph=graph)
            assert ref.dcid == dcid
            assert ref.label == expected_label


class TestGraphrefs:
    def test_all_confirmed(self):
        graph = FakeGraph()
        refs = graphrefs(["country/ETH", "country/KEN"], graph=graph)
        assert len(refs) == 2
        dcids = {r.dcid for r in refs}
        assert dcids == {"country/ETH", "country/KEN"}

    def test_absent_node_dropped(self):
        graph = FakeGraph()
        refs = graphrefs(["country/ETH", "country/MISSING", "country/KEN"], graph=graph)
        assert len(refs) == 2
        assert all(r.dcid != "country/MISSING" for r in refs)

    def test_all_absent(self):
        graph = FakeGraph()
        refs = graphrefs(["MISSING_1", "MISSING_2"], graph=graph)
        assert refs == []

    def test_transport_error_propagates(self):
        graph = FakeGraph(raise_on_call=True)
        with pytest.raises(EngineInfraError):
            graphrefs(["country/ETH"], graph=graph)
