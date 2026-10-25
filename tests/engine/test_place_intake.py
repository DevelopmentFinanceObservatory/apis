"""`phase_intake._phase_resolve_places` reads a resolved name as a region by the membership
policy (`MembershipPolicy.holds_region`), never by the node's graph types."""

from __future__ import annotations

import asyncio
from typing import cast

from entity_resolution import Admission, Entity, Resolved
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.door_names import own_texts
from search.engine.interpret import Recall
from search.engine.region.phase_intake import _phase_resolve_places
from search.engine.region.state import ResolveState
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.fixtures import FakeGraph, make_resolver

_AMERICAS = "undata-geo/G00134000"
# A region-typed node the policy neither holds nor omits.
_UNLISTED = "undata-geo/G00099999"

# The types dev's graph gives each node: the Americas is a `GeoRegion`, a type no graph-type
# rule reads as a region.
_NODES = {
    _AMERICAS: {"label": "Americas", "type": "GeoRegion"},
    "antarctica": {"label": "Antarctica", "type": "Continent"},
    _UNLISTED: {"label": "Unlisted region", "type": "UNGeoRegion"},
}


def _state(graph: FakeGraph, names: list[str]) -> ResolveState:
    return ResolveState(
        variable="population",
        entities=names,
        entity_texts=own_texts(names),
        date_request=None,
        detect_query="population",
        role_query="population",
        pac=True,
        graph=cast(EngineGraphClient, graph),
        resolver=make_resolver(graph),
        admission=cast(Admission, None),
        llm=cast(SupportsLLM, None),
        sdmx=cast(EngineSdmxClient, None),
        deadline=None,
        pre_resolved=None,
        variable_index=0,
        emit=None,
        include_option_coverage=False,
        stream_slot_options=False,
        include_observations=False,
        stream_observations=False,
        unit_text=None,
        materialise_graph=None,
        materialise_sdmx=None,
        timing={},
        warnings=[],
        coverage=empty_coverage_snapshot(),
    )


def test_a_region_is_what_the_policy_holds_or_omits_whatever_its_graph_type():
    """The Americas, a `GeoRegion` node, resolves and expands to the policy's countries.
    Antarctica, which the policy omits, reads as its node alone with no degraded note. A
    region-typed node the policy neither holds nor omits is an ordinary place."""
    graph = FakeGraph(nodes=_NODES)
    state = _state(graph, ["Americas", "Antarctica", "Unlisted"])
    resolver = state.resolver
    (americas, antarctica) = asyncio.run(resolver.resolve(["Americas", "Antarctica"]))
    unlisted = Resolved(entity=Entity(_UNLISTED, "Unlisted region", "subregion", "UNGeoRegion"))
    state.rcl = Recall(
        candidates=[],
        resolutions={"Americas": americas, "Antarctica": antarctica, "Unlisted": unlisted},
    )

    with recording_degraded_reads() as lost:
        asyncio.run(_phase_resolve_places(state))

    sets = state.place_sets
    countries = resolver.membership.members(_AMERICAS).members
    assert isinstance(countries, tuple)
    assert isinstance(americas, Resolved) and americas.entity.dcid == _AMERICAS
    assert (sets["Americas"].kind, sets["Americas"].members) == ("region", countries)
    assert len(countries) == 35
    assert (sets["Antarctica"].kind, sets["Antarctica"].members) == ("region", None)
    assert (sets["Unlisted"].kind, sets["Unlisted"].members) == ("place", None)
    assert lost.reads == []
