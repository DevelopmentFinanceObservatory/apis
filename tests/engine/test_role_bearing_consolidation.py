"""Test how declared roles control availability, resolution, and graph access.

Changing `declared_roles` flips every gate together, so all gates share one predicate.
"""

from __future__ import annotations

import dataclasses

from search.engine.clients.graph import Facet
from search.engine.quantity import NO_REQUIREMENT
from search.engine.region.availability import (
    _prune_availability_siblings,
    _resolve_sibling_availability,
)
from search.engine.retrieval.retrieve import materialise
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.shape import ShapeDraft, is_role_bearing
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

# TESTFIXTURE/ is reserved for tests and absent from production StatVar ids.
_SV_DCID = "TESTFIXTURE/RoleConsolidation/Flow"


def _role_arcs(declared_roles: tuple[str, ...]) -> dict:
    """Build member arcs for the shape's declared roles."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if declared_roles == ("observationAbout",):
        return arcs
    arcs["observationProperties"] = {"nodes": [{"dcid": r} for r in declared_roles]}
    return arcs


def _shape(*, declared_roles: tuple[str, ...]) -> ShapeDraft:
    """Build a fixture shape whose role status comes from declared_roles."""
    return ShapeDraft(
        shape_id="test_role_consolidation_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={_SV_DCID: _role_arcs(declared_roles)},
        slot_taxonomy={},
        declared_roles=declared_roles,
    )


class TestIsRoleBearingDeclarationPerturbation:
    def test_non_declaring_shape_flips_every_consolidated_gate(self):
        """Clearing the role declaration changes the consolidated gate to non-role-bearing."""
        shape = _shape(declared_roles=("observationAbout",))
        assert is_role_bearing(shape) is False

        to_survive, _ = _prune_availability_siblings(
            [shape], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
        )
        assert to_survive == [shape]


class TestResolveSiblingAvailabilityDeclarationPerturbation:
    """Declared pairs skip availability probes. Undeclared pairs reach them."""

    def _sibling_pair(self, shape: ShapeDraft) -> tuple[ShapeDraft, ShapeDraft]:
        # Keep the class key fixed and vary denominator resolution only.
        unresolved = dataclasses.replace(
            shape,
            shape_id="sib_unresolved",
            meas_denom_dcid=None,
            meas_denom_unresolved=True,
            sv_arc_facts={"sv_unresolved": {}},
        )
        resolved = dataclasses.replace(
            shape,
            shape_id="sib_resolved",
            meas_denom_dcid="Count_Person",
            meas_denom_unresolved=False,
            sv_arc_facts={"sv_resolved": {}},
        )
        return unresolved, resolved

    def test_declaring_pair_never_reaches_the_probe(self):
        shape = _shape(declared_roles=("creditor", "debtor"))
        unresolved, resolved = self._sibling_pair(shape)
        graph = FakeGraph(obs={"sv_unresolved|country/KEN": [{"obsCount": 5}]}, nodes={})

        result, probe_failed = _resolve_sibling_availability(
            [unresolved, resolved], graph=graph, named_place=True, probe_places=["country/KEN"]
        )

        assert probe_failed is False
        # A role-bearing pair leaves place_available unset.
        assert all(s.place_available is None for s in result)


class TestGraphConfirmResolveDeclarationPerturbation:
    """Declared roles select the role path. Clearing them selects the standard path's graph
    facets (`retrieve.materialise`)."""

    def test_declaring_shape_uses_the_sdmx_branch_only(self):
        shape = _shape(declared_roles=("creditor", "debtor"))
        assert is_role_bearing(shape) is True

        role_tuple = (("creditor", "country/USA"), ("debtor", "country/ETH"))
        sdmx = FakeSdmxClient(
            observations={
                _SV_DCID: {
                    role_tuple: [
                        Facet(
                            earliest_date="2010",
                            latest_date="2020",
                            obs_count=5,
                            dates=["2020"],
                            values=[1.0],
                            facet_id="f1",
                        )
                    ]
                }
            }
        )
        # Role-bearing resolution uses the SDMX facet. This graph raises on access.
        result = graph_confirm_resolve(
            shape=shape,
            bindings=[],
            subject_dcids=["country/ETH"],
            origin_dcids=["country/USA"],
            graph=FakeGraph(raise_on_call=True),
            sdmx=sdmx,
            coverage_snapshot=empty_coverage_snapshot(),
        )

        from search.engine.retrieval.retrieve import Materialised

        assert isinstance(result, Materialised)

    def test_non_declaring_shape_routes_through_the_generic_branch(self):
        shape = _shape(declared_roles=("observationAbout",))
        assert is_role_bearing(shape) is False

        graph = FakeGraph(
            nodes={}, obs={f"{_SV_DCID}|country/ETH": [{"obsCount": 5}]}, detect={}, resolve={}
        )
        # The graph holds the matching facet. The SDMX client has no observations.
        result = materialise(
            shape,
            [],
            {"observationAbout": ["country/ETH"]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        from search.engine.retrieval.retrieve import Materialised

        assert isinstance(result, Materialised)
