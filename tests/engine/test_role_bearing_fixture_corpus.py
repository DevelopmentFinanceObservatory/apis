"""Fixture-corpus StatVars produce role-bearing shapes and role-keyed SDMX rows.
The SDMX check requires row presence, including role_facet_index stubs."""

from __future__ import annotations

from search.engine.config import settings
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import _dcids_for
from search.engine.shape import is_role_bearing
from search.eval.cases import load_cases
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
    """Chunked derivation yields a role-bearing shape using declared fixture roles."""
    nodes = _load_json("graph_nodes.json")
    declared_role_names = {
        role for roles in _role_declaring_dcids(nodes).values() for role in roles
    }
    sv_dcids = _stat_variable_dcids(nodes)
    assert sv_dcids, "fixture corpus carries no StatisticalVariable nodes to derive shapes from"

    graph = FakeGraph()
    role_bearing_shapes = []
    for start in range(0, len(sv_dcids), settings().max_confirm_candidates):
        chunk = sv_dcids[start : start + settings().max_confirm_candidates]
        shapes = derive_shapes(confirmed_svs=chunk, graph=graph)
        role_bearing_shapes.extend(shape for shape in shapes if is_role_bearing(shape))

    assert role_bearing_shapes, "derive_shapes built no role-bearing shape from the fixture corpus"
    for shape in role_bearing_shapes:
        undeclared = set(shape.declared_roles or ()) - declared_role_names
        assert not undeclared, (
            f"shape {shape.shape_id!r} declares roles {shape.declared_roles!r} that no "
            f"fixture node's own observationProperties arc names: {undeclared!r}"
        )


def _case_stat_var_dcids() -> set[str]:
    """Every dcid any dataset_resubmit case's own stat_var_dcids names, across the corpus."""
    return {
        dcid
        for case in load_cases()
        if case.resubmit is not None
        for dcid in case.resubmit.stat_var_dcids or ()
    }


def _sdmx_obs_role_names_by_variable(obs: dict) -> dict[str, set[str]]:
    """variable -> the set of role names appearing in any sdmx_obs.json key
    for that variable, parsed off each key's own
    "variable|role=dcid,role=dcid,..." shape (see tests/fixtures/record.py's
    _role_tuple_key)."""
    by_variable: dict[str, set[str]] = {}
    for key in obs:
        if key == "_comment":
            continue
        variable, role_tuple_text = key.split("|", 1)
        role_names = {segment.split("=", 1)[0] for segment in role_tuple_text.split(",")}
        by_variable.setdefault(variable, set()).update(role_names)
    return by_variable


def test_every_case_role_bearing_statvar_has_a_role_keyed_sdmx_row():
    """Each case role-bearing StatVar has an SDMX row for a declared role."""
    nodes = _load_json("graph_nodes.json")
    declared_roles = _role_declaring_dcids(nodes)
    obs = _load_json("sdmx_obs.json")
    role_names_by_variable = _sdmx_obs_role_names_by_variable(obs)

    role_bearing_case_svs = _case_stat_var_dcids() & set(declared_roles)
    assert role_bearing_case_svs, (
        "no case's stat_var_dcids names a StatVar this fixture corpus declares role-bearing"
    )
    for sv_dcid in role_bearing_case_svs:
        covered_roles = role_names_by_variable.get(sv_dcid, set())
        assert covered_roles & set(declared_roles[sv_dcid]), (
            f"{sv_dcid!r} declares role properties {declared_roles[sv_dcid]!r} but "
            f"sdmx_obs.json carries no role-tuple row for it naming any of them "
            f"(rows found for roles: {covered_roles!r})"
        )
