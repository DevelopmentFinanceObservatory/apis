"""Pin the 23-node `crsPurpose` health closure and 9-code flat `crsTypeOfFlow` no-op."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from code_index import CodeIndex


def _bounded(fn, /, *args, timeout: float = 3.0, **kwargs):
    """Run in a worker thread and raise `TimeoutError` after `timeout` seconds."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout)


def test_health_closure_matches_the_live_probe(code_index: CodeIndex):
    result = code_index.descendants("CRSPurposeCode120")
    assert result[0] == "CRSPurposeCode120"
    assert len(result) == 23
    assert len(set(result)) == 23  # no duplicate node reached twice


def test_health_closure_includes_its_documented_direct_children(code_index: CodeIndex):
    result = set(code_index.descendants("CRSPurposeCode120"))
    assert {"CRSPurposeCode121", "CRSPurposeCode122", "CRSPurposeCode123"} <= result


def test_malaria_control_is_a_leaf(code_index: CodeIndex):
    # CRSPurposeCode12262 sits under 122 with no children of its own.
    assert code_index.descendants("CRSPurposeCode12262") == ("CRSPurposeCode12262",)


def test_expansion_over_a_flat_system_is_a_genuine_no_op(code_index: CodeIndex):
    declaration = code_index.declarations["crsTypeOfFlow"]
    assert declaration.hierarchical is False

    seed = "CRSTypeOfFlow10"
    result = code_index.descendants(seed)

    assert result == (seed,)
    # The rows have no parent arcs, so expansion stays at the seed.
    has_any_parent = any(
        row.parent is not None for row in code_index.codes if row.code_system == "crsTypeOfFlow"
    )
    assert has_any_parent is False


def test_a_dcid_this_index_does_not_carry_as_a_code_expands_to_only_itself(
    code_index: CodeIndex,
):
    assert code_index.descendants("NotACode") == ("NotACode",)


# The constructor rejects parent cycles. These tests inject cycles to check that traversal ends.


def test_a_self_parent_terminates_and_returns_the_node_once(code_index: CodeIndex):
    code_index._children_by_parent = {
        **code_index._children_by_parent,
        "CRSPurposeCode120": ("CRSPurposeCode120",),
    }
    result = _bounded(code_index.descendants, "CRSPurposeCode120")
    assert result == ("CRSPurposeCode120",)


def test_a_two_node_cycle_terminates_and_returns_each_node_once(code_index: CodeIndex):
    code_index._children_by_parent = {
        **code_index._children_by_parent,
        "CRSPurposeCode120": ("CRSPurposeCode121",),
        "CRSPurposeCode121": ("CRSPurposeCode120",),
    }
    result = _bounded(code_index.descendants, "CRSPurposeCode120")
    assert set(result) == {"CRSPurposeCode120", "CRSPurposeCode121"}
    assert len(result) == len(set(result))  # each node reached once, not re-queued


def test_an_acyclic_walk_still_terminates_and_is_unaffected_by_the_visited_set(
    code_index: CodeIndex,
):
    result = _bounded(code_index.descendants, "CRSPurposeCode120")
    assert result[0] == "CRSPurposeCode120"
    assert len(result) == 23
    assert len(set(result)) == 23
