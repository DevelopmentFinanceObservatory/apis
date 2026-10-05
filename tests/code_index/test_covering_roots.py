"""Pin root coverage and depth ancestors for hierarchical codes and passthrough for flat codes."""

from __future__ import annotations

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _row(dcid: str, code_system: str, parent: str | None) -> CodeRow:
    return CodeRow(
        dcid=dcid,
        code_system=code_system,
        code=dcid,
        label=dcid,
        synonyms=(),
        parent=parent,
    )


def _synthetic_index() -> CodeIndex:
    """Build a hierarchical tree and a flat system with two roots."""
    codes = [
        _row("HierRootA", "synthHier", None),
        _row("HierA1", "synthHier", "HierRootA"),
        _row("HierA1a", "synthHier", "HierA1"),
        _row("HierRootB", "synthHier", None),
        _row("HierB1", "synthHier", "HierRootB"),
        _row("FlatX", "synthFlat", None),
        _row("FlatY", "synthFlat", None),
    ]
    declarations = [
        CodeSystemDeclaration(
            code_system="synthHier",
            hierarchical=True,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        ),
        CodeSystemDeclaration(
            code_system="synthFlat",
            hierarchical=False,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        ),
    ]
    return CodeIndex(stamp=_STAMP, declarations=declarations, codes=codes, statvars=())


def test_a_leaf_two_levels_deep_rolls_up_to_its_system_root():
    index = _synthetic_index()
    assert index.covering_roots(["HierA1a"]) == ("HierRootA",)


def test_two_values_sharing_a_root_dedupe_to_one_entry():
    index = _synthetic_index()
    assert index.covering_roots(["HierA1a", "HierA1"]) == ("HierRootA",)


def test_values_under_different_roots_each_keep_their_own_root_in_first_seen_order():
    index = _synthetic_index()
    assert index.covering_roots(["HierB1", "HierA1a"]) == ("HierRootB", "HierRootA")
    # Reversing the input reverses the output: no sorting, first-appearance order.
    assert index.covering_roots(["HierA1a", "HierB1"]) == ("HierRootA", "HierRootB")


def test_a_root_itself_rolls_up_to_itself():
    index = _synthetic_index()
    assert index.covering_roots(["HierRootA"]) == ("HierRootA",)


def test_a_flat_systems_own_values_are_never_rolled_up():
    """A flat-system declaration preserves each value as its own root."""
    index = _synthetic_index()
    declaration = index.declarations["synthFlat"]
    assert declaration.hierarchical is False
    assert index.covering_roots(["FlatX", "FlatY"]) == ("FlatX", "FlatY")


def test_a_value_with_no_code_row_rolls_up_to_itself():
    index = _synthetic_index()
    assert index.covering_roots(["NoSuchValue"]) == ("NoSuchValue",)


def test_a_mix_of_known_and_unknown_values_preserves_first_seen_order():
    index = _synthetic_index()
    assert index.covering_roots(["NoSuchValue", "HierA1a", "HierB1"]) == (
        "NoSuchValue",
        "HierRootA",
        "HierRootB",
    )


def test_an_empty_value_list_covers_to_nothing():
    index = _synthetic_index()
    assert index.covering_roots([]) == ()


def test_a_dangling_parent_reference_stops_at_the_missing_dcid():
    """A dangling parent becomes the root. The test inserts this invalid row after index creation."""
    index = _synthetic_index()
    index._codes_by_dcid = {
        **index._codes_by_dcid,
        "Orphan": _row("Orphan", "synthHier", "MissingParent"),
    }
    assert index.covering_roots(["Orphan"]) == ("MissingParent",)


def test_real_crs_purpose_health_leaf_rolls_up_to_its_root(code_index: CodeIndex):
    # Malaria-control code `CRSPurposeCode12262` rolls up to the social-infrastructure root.
    assert code_index.covering_roots(["CRSPurposeCode12262"]) == ("CRSPurposeCode100",)


def test_real_crs_type_of_flow_a_flat_system_is_never_rolled_up(code_index: CodeIndex):
    declaration = code_index.declarations["crsTypeOfFlow"]
    assert declaration.hierarchical is False
    assert code_index.covering_roots(["CRSTypeOfFlow10", "CRSTypeOfFlow21"]) == (
        "CRSTypeOfFlow10",
        "CRSTypeOfFlow21",
    )


# The synthetic tree is ragged: branch A is three codes deep, branch B two.
@pytest.mark.parametrize(
    ("dcid", "depth", "expected"),
    [
        ("HierA1a", 0, "HierRootA"),
        ("HierA1a", 1, "HierA1"),
        ("HierA1a", 2, "HierA1a"),
        ("HierA1a", 3, None),
        ("HierB1", 0, "HierRootB"),
        ("HierB1", 1, "HierB1"),
        ("HierB1", 2, None),
        ("HierRootA", 0, "HierRootA"),
        ("HierRootA", 1, None),
        ("FlatX", 0, "FlatX"),
        ("FlatX", 1, None),
        ("NoSuchValue", 0, "NoSuchValue"),
    ],
)
def test_ancestor_at_depth_walks_the_parent_chain_from_the_root(
    dcid: str, depth: int, expected: str | None
):
    assert _synthetic_index().ancestor_at_depth(dcid, depth) == expected


def test_real_crs_purpose_codes_of_one_length_sit_at_different_depths(code_index: CodeIndex):
    # 60010 (depth 1) and 12262 (depth 3) are both five-digit purposes.
    assert code_index.ancestor_at_depth("CRSPurposeCode60010", 0) == "CRSPurposeCode600"
    assert code_index.ancestor_at_depth("CRSPurposeCode60010", 2) is None
    assert code_index.ancestor_at_depth("CRSPurposeCode12262", 0) == "CRSPurposeCode100"
    assert code_index.ancestor_at_depth("CRSPurposeCode12262", 1) == "CRSPurposeCode120"
