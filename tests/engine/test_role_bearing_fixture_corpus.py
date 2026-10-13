"""Fixture-corpus StatVars produce role-bearing shapes."""

from __future__ import annotations

from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import _dcids_for
from search.engine.shape import is_role_bearing
from tests.fixtures import FakeGraph, _load_json


def _role_declaring_dcids(nodes: dict) -> dict[str, list[str]]:
    """dcid -> declared role property dcids, for every node whose own
    observationProperties out-arc is non-empty."""
    declared = {}
    for dcid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        arcs = node.get("arcs") or {}
        roles = _dcids_for(arcs, "observationProperties")
        if roles:
            declared[dcid] = roles
    return declared


def _stat_variable_dcids(nodes: dict) -> list[str]:
    """Return corpus StatVars identified by their type or typeOf arc."""
    dcids = []
    for dcid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if node.get("type") == "StatisticalVariable":
            dcids.append(dcid)
            continue
        if "StatisticalVariable" in _dcids_for(node.get("arcs") or {}, "typeOf"):
            dcids.append(dcid)
    return dcids


def test_derive_shapes_yields_a_role_bearing_shape():
    """Derivation yields a role-bearing shape using declared fixture roles."""
    nodes = _load_json("graph_nodes.json")
    declared_role_names = {
        role for roles in _role_declaring_dcids(nodes).values() for role in roles
    }
    sv_dcids = _stat_variable_dcids(nodes)
    assert sv_dcids, "fixture corpus carries no StatisticalVariable nodes to derive shapes from"

    graph = FakeGraph()
    shapes = derive_shapes(confirmed_svs=sv_dcids, graph=graph)
    role_bearing_shapes = [shape for shape in shapes if is_role_bearing(shape)]

    assert role_bearing_shapes, "derive_shapes built no role-bearing shape from the fixture corpus"
    for shape in role_bearing_shapes:
        undeclared = set(shape.declared_roles or ()) - declared_role_names
        assert not undeclared, (
            f"shape {shape.shape_id!r} declares roles {shape.declared_roles!r} that no "
            f"fixture node's own observationProperties arc names: {undeclared!r}"
        )
