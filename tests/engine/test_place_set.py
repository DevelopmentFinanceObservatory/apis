"""`PlaceSet.dcids`: the places a set stands for under each reading."""

from __future__ import annotations

import pytest

from search.engine.region.place_set import PlaceNode, PlaceSet

_NODE = PlaceNode(dcid="africa", graph_types=("Continent",))
_MEMBERS = ("country/GHA", "country/KEN")


@pytest.mark.parametrize(
    ("node", "members", "default", "as_members"),
    [
        pytest.param(_NODE, None, ("africa",), ("africa",), id="place"),
        pytest.param(_NODE, _MEMBERS, ("africa",), _MEMBERS, id="region"),
        pytest.param(_NODE, (), ("africa",), (), id="memberless-region"),
        pytest.param(None, _MEMBERS, _MEMBERS, _MEMBERS, id="group-without-a-node"),
    ],
)
def test_a_set_reads_as_its_node_or_its_members(node, members, default, as_members) -> None:
    place_set = PlaceSet(name="Africa", kind="region", node=node, members=members)

    assert place_set.expands is (members is not None)
    assert place_set.dcids() == place_set.dcids("default") == default
    assert place_set.dcids("members") == as_members
    assert (
        PlaceSet(
            name="Africa", kind="region", node=node, members=members, reading="members"
        ).dcids()
        == as_members
    )
