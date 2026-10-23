"""Tests level-scoped code options and single-pass member counts.

Real CRS data and a synthetic flat system with parent links cover declared hierarchy behaviour.
"""

from __future__ import annotations

import pytest

from code_index import (
    Bound,
    CodeIndex,
    CodeRow,
    CodeSystemDeclaration,
    CrossSystemParent,
    IndexStamp,
    StatVarRow,
)
from tests.code_index.support import INDEX_FIXTURE, load_index

_STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def _row(dcid: str, code_system: str, parent: str | None) -> CodeRow:
    return CodeRow(
        dcid=dcid, code_system=code_system, code=dcid, label=dcid, synonyms=(), parent=parent
    )


def _sv(dcid: str, code_system: str, value: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="TestPopulation",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=(),
        constraint_properties={code_system: value},
    )


def _synthetic_index() -> CodeIndex:
    """Build a two-root hierarchy and a flat declaration with three-level parent links."""
    codes = [
        _row("RootA", "hier", None),
        _row("A1", "hier", "RootA"),
        _row("A1a", "hier", "A1"),
        _row("A1b", "hier", "A1"),
        _row("A2", "hier", "RootA"),
        _row("RootB", "hier", None),
        _row("B1", "hier", "RootB"),
        _row("FlatX", "flatButNested", None),
        _row("FlatXChild", "flatButNested", "FlatX"),
        _row("FlatXGrandchild", "flatButNested", "FlatXChild"),
        _row("FlatY", "flatButNested", None),
    ]
    declarations = [
        CodeSystemDeclaration(
            code_system="hier",
            hierarchical=True,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        ),
        CodeSystemDeclaration(
            code_system="flatButNested",
            hierarchical=False,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        ),
    ]
    statvars = [
        _sv("sv1", "hier", "A1a"),
        _sv("sv2", "hier", "A1a"),
        _sv("sv3", "hier", "A1b"),
        _sv("sv4", "hier", "RootA"),
        _sv("sv5", "hier", "RootB"),
        _sv("sv6", "hier", "RootB"),
        _sv("sv7", "hier", "RootB"),
        _sv("sv8", "flatButNested", "FlatXChild"),
    ]
    return CodeIndex(stamp=_STAMP, declarations=declarations, codes=codes, statvars=statvars)


# --- CodeIndex construction: a parent's code system ---


def test_constructing_an_index_with_a_cross_system_parent_raises():
    """#852: `parent` names a `specializationOf` target, so it is a row of
    the child's own `code_system`. Pointing `hier`'s A1 at `FlatX` -- a
    `flatButNested` row, the nearest real vocabulary in reach -- is refused
    at construction, before `_children_by_parent` or the acyclic walk is
    built. The rows are acyclic and their dcids unique, so only the
    code-system check can reject this index."""
    codes = [
        _row("RootA", "hier", None),
        _row("A1", "hier", "FlatX"),
        _row("FlatX", "flatButNested", None),
    ]
    declarations = [
        CodeSystemDeclaration(
            code_system="hier",
            hierarchical=True,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        ),
        CodeSystemDeclaration(
            code_system="flatButNested",
            hierarchical=False,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="axis",
        ),
    ]
    with pytest.raises(CrossSystemParent) as excinfo:
        CodeIndex(stamp=_STAMP, declarations=declarations, codes=codes, statvars=())
    assert "A1" in str(excinfo.value)
    assert "FlatX" in str(excinfo.value)


# --- value_counts: the single aggregating pass ---


def test_value_counts_reproduces_every_per_value_filter_count_in_one_call():
    """A `filter` call bound to one value reproduces that value's entry in
    `value_counts`' result. Asserted for every value in `hier` from a
    single `value_counts` call, which shows one axis scan is enough for the
    whole vocabulary."""
    index = _synthetic_index()
    counts = index.value_counts("hier")
    for value in ("RootA", "A1", "A1a", "A1b", "A2", "RootB", "B1"):
        expected = len(index.filter({"hier": Bound(value)}))
        assert counts.get(value, 0) == expected


def test_value_counts_omits_a_value_no_statvar_declares():
    index = _synthetic_index()
    counts = index.value_counts("hier")
    assert "A2" not in counts  # a real code, zero direct StatVars


# --- level_scoped_options: bound hierarchical ---


@pytest.mark.parametrize(
    ("bound", "expected"),
    [
        (
            "A1",
            [("A1a", "child", 2, "A1"), ("A1b", "child", 1, "A1"), ("RootA", "parent", 1, None)],
        ),
        (
            "RootA",
            [("A1", "child", 0, "RootA"), ("A2", "child", 0, "RootA"), ("RootB", "root", 3, None)],
        ),
        ("RootB", [("B1", "child", 0, "RootB"), ("RootA", "root", 1, None)]),
        ("A1a", [("A1", "parent", 0, "RootA")]),
    ],
    ids=["node", "root", "other-root", "leaf"],
)
def test_a_bound_hierarchical_node_publishes_its_children_then_its_parent_or_the_other_roots(
    bound: str, expected: list[tuple[str, str, int, str | None]]
):
    """A bound root has no parent, so the other roots follow its children, without their
    own children."""
    index = _synthetic_index()
    options = index.level_scoped_options("hier", bound)
    assert [(o.dcid, o.relation, o.member_count, o.parent) for o in options] == expected


# --- level_scoped_options: unbound ---


def test_an_unbound_hierarchical_slot_publishes_each_root_then_its_children():
    index = _synthetic_index()
    options = index.level_scoped_options("hier", None)
    assert [(o.dcid, o.relation, o.member_count, o.parent) for o in options] == [
        ("RootB", "root", 3, None),
        ("B1", "child", 0, "RootB"),
        ("RootA", "root", 1, None),
        ("A1", "child", 0, "RootA"),
        ("A2", "child", 0, "RootA"),
    ]


def test_a_bound_dcid_from_a_different_code_system_is_treated_as_unbound():
    index = _synthetic_index()
    assert index.level_scoped_options("hier", "FlatX") == index.level_scoped_options("hier", None)


def test_an_unrecognised_bound_dcid_is_treated_as_unbound():
    index = _synthetic_index()
    assert index.level_scoped_options("hier", "NotACode") == index.level_scoped_options(
        "hier", None
    )


# --- level_scoped_options: a declared-flat system ---


def test_a_non_hierarchical_declaration_with_real_parent_links_is_never_read():
    """A flat declaration publishes every code as a root despite real parent links."""
    index = _synthetic_index()
    declaration = index.declarations["flatButNested"]
    assert declaration.hierarchical is False

    bound = index.level_scoped_options("flatButNested", "FlatXChild")
    unbound = index.level_scoped_options("flatButNested", None)
    assert bound == unbound

    dcids = {o.dcid for o in bound}
    assert dcids == {"FlatX", "FlatXChild", "FlatXGrandchild", "FlatY"}
    assert all(o.relation == "root" for o in bound)
    by_dcid = {o.dcid: o.member_count for o in bound}
    assert by_dcid["FlatXChild"] == 1
    assert by_dcid["FlatX"] == 0


# --- level_scoped_options: real crsPurpose/crsTypeOfFlow data ---


def test_real_health_bound_children_and_parent_match_filter_counts(code_index: CodeIndex):
    # CRSPurposeCode120 (health) has three real children (121/122/123) and one
    # real parent (100) -- see test_descendants.py's own health-closure tests.
    options = code_index.level_scoped_options("crsPurpose", "CRSPurposeCode120")
    by_dcid = {o.dcid: o for o in options}
    assert set(by_dcid) == {
        "CRSPurposeCode121",
        "CRSPurposeCode122",
        "CRSPurposeCode123",
        "CRSPurposeCode100",
    }
    for dcid, opt in by_dcid.items():
        expected = len(code_index.filter({"crsPurpose": Bound(dcid)}))
        assert opt.member_count == expected
    assert by_dcid["CRSPurposeCode100"].relation == "parent"
    for child in ("CRSPurposeCode121", "CRSPurposeCode122", "CRSPurposeCode123"):
        assert by_dcid[child].relation == "child"


def test_real_flat_type_of_flow_is_never_invented_into_a_hierarchy(code_index: CodeIndex):
    declaration = code_index.declarations["crsTypeOfFlow"]
    assert declaration.hierarchical is False
    all_flow_dcids = {r.dcid for r in code_index.codes if r.code_system == "crsTypeOfFlow"}
    bound = code_index.level_scoped_options("crsTypeOfFlow", "CRSTypeOfFlow10")
    unbound = code_index.level_scoped_options("crsTypeOfFlow", None)
    assert {o.dcid for o in bound} == all_flow_dcids
    assert bound == unbound
    assert all(o.relation == "root" for o in bound)


@pytest.mark.parametrize(
    ("code_system", "bound"),
    [
        ("crsPurpose", "CRSPurposeCode100"),
        ("dac2aAidType", "DAC2aAidType106"),
        ("dac2aAidType", "DAC2aAidType216"),
        ("crsPurpose", "CRSPurposeCode120"),
    ],
    ids=["crs-purpose-root", "dac2a-childless-root", "dac2a-other-childless-root", "non-root"],
)
def test_a_real_bound_root_offers_the_other_roots_an_unbound_slot_lists(
    code_system: str, bound: str
):
    """On the fixture index, a bound root's children are followed by the roots an unbound
    slot lists, in its order and less the bound root. A bound code with a parent is followed
    by that parent alone."""
    index = load_index(INDEX_FIXTURE)
    row = index.code_by_dcid(bound)
    assert row is not None

    options = index.level_scoped_options(code_system, bound)

    children = [o for o in options if o.relation == "child"]
    assert {o.dcid for o in children} == {
        r.dcid for r in index.codes if r.code_system == code_system and r.parent == bound
    }
    assert options[: len(children)] == tuple(children)
    if row.parent is None:
        other_roots = tuple(
            o
            for o in index.level_scoped_options(code_system, None)
            if o.relation == "root" and o.dcid != bound
        )
        assert other_roots
        assert options[len(children) :] == other_roots
    else:
        assert [(o.dcid, o.relation) for o in options[len(children) :]] == [(row.parent, "parent")]
