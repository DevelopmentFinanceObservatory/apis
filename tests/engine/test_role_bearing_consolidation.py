"""Test how declared roles control availability, resolution, and graph access.

Changing `declared_roles` flips every gate together, so all gates share one predicate.
"""

from __future__ import annotations

from search.engine.clients.graph import Facet
from search.engine.region.availability import read_menu_availability
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


_KENYA_OBS = {f"{_SV_DCID}|country/KEN": [{"obsCount": 5}]}


class TestMenuAvailabilityDeclarationPerturbation:
    """The availability read over the menu reads a subject-only shape's StatVars and leaves
    a role-bearing one as it is: its observations are not subject facets."""

    def test_declaring_shape_is_not_read(self):
        shape = _shape(declared_roles=("creditor", "debtor"))
        graph = FakeGraph(obs=_KENYA_OBS, nodes={})

        (read,), failed = read_menu_availability(
            [shape], graph=graph, places=["country/KEN"], deadline=None
        )

        assert failed is False
        assert read is shape

    def test_non_declaring_shape_is_read_at_the_places(self):
        shape = _shape(declared_roles=("observationAbout",))
        assert is_role_bearing(shape) is False
        graph = FakeGraph(obs=_KENYA_OBS, nodes={})

        (read,), failed = read_menu_availability(
            [shape], graph=graph, places=["country/KEN"], deadline=None
        )

        assert failed is False
        assert set(read.named_place_coverage[_SV_DCID]) == {"country/KEN"}


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
