"""Regression for #698. The resolve door emits one member for duplicate containment arcs.

The test runs the real `GraphClient` against doubled `/v2/node` wire data because the door's
`TestClient` substitutes `FakeGraphNodeReader`.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from dc_graph import GraphClient
from entity_resolution import Entity, Resolved
from graph_api.deps import GraphDeps
from graph_api.doors.resolve import RegionMember, _assemble, _region_members
from graph_api.doors.sources import new_provenance_doc_cache
from graph_generation import UNKNOWN_GENERATION
from tests.graph_api.conftest import make_resolver
from tests.materialise._dup_arcs_fixtures import DUAL_PROVENANCES


def test_assemble_emits_one_region_member_when_containment_read_is_doubled(shared_resolver):
    # Two provenances return the same containment arc for `country/KEN` (#698).
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        prop = payload["property"]
        if prop == "<-containedInPlace+{typeOf:Country}":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "Africa": {
                            "arcs": {
                                "containedInPlace+": {
                                    "nodes": [
                                        {"dcid": "country/KEN", "provenanceId": prov}
                                        for prov in DUAL_PROVENANCES
                                    ]
                                }
                            }
                        }
                    }
                },
            )
        assert prop == "->name"
        return httpx.Response(
            200,
            json={"data": {"country/KEN": {"arcs": {"name": {"nodes": [{"value": "Kenya"}]}}}}},
        )

    reader = GraphClient(
        base_url="https://graph.example.org/core/api", transport=httpx.MockTransport(handler)
    )
    # GraphDeps requires valid fields for the direct `_region_members` call.
    executor = ThreadPoolExecutor(max_workers=1)
    deps = GraphDeps(
        graph=reader,
        resolver=make_resolver(shared_resolver, graph=reader),
        dcp_executor=executor,
        generation=UNKNOWN_GENERATION,
        provenance_docs=new_provenance_doc_cache(executor=executor, generation=UNKNOWN_GENERATION),
    )
    resolution = Resolved(
        entity=Entity(dcid="Africa", name="Africa", kind="continent", graph_type="Continent")
    )

    members_by_dcid, names_by_dcid = asyncio.run(
        _region_members(deps, [resolution], deadline=time.monotonic() + 30.0)
    )
    result = _assemble(resolution, members_by_dcid, names_by_dcid)

    assert result.members == [RegionMember(dcid="country/KEN", name="Kenya")]
