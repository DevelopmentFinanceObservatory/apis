"""Derive groups an index row under the family its provenances assign it, and another
StatVar under the family its constraint properties select. The shape built from a group
reads that group's family from its members, and a shape with no recalled member reads the
family its slot properties select."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from code_index import shared as shared_module
from code_index.query import CodeIndex
from search.engine.retrieval.bound_values import (
    RoleKeys,
    five_tuple_member_facts,
    index_carries_shape,
    index_member_rows,
    member_family,
    part_datasets,
    part_source,
    shape_family,
    shape_member_facts,
    shape_query,
    shape_slot_properties,
)
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import FiveTuple
from tests.code_index.test_elect_dataset import (
    _PLAIN,
    _TABLES,
    _dataset,
    _index,
    _not_declared,
    _row,
    _split_index,
)
from tests.engine._source_scopes import part_scope
from tests.fixtures import FakeGraph

_FIVE_TUPLE = FiveTuple("TestFlow", "amount", "measuredValue", None, None)


def test_a_datasets_rows_share_its_family_when_another_family_holds_fewer_properties():
    index = _split_index(sparse=True)

    plain = member_family("plain_health", _FIVE_TUPLE, ("sector",), index)
    tables = member_family("tables_health", _FIVE_TUPLE, ("sector",), index)
    unindexed = member_family("Unindexed_Health", _FIVE_TUPLE, ("sector",), index)

    assert plain is not None
    assert tables is not None
    assert plain.datasets == (_PLAIN,)
    assert tables.datasets == (_TABLES,)
    assert unindexed is tables


def _no_place() -> RoleKeys:
    """The role keys of a part bound to no place."""
    return RoleKeys()


def _use_index(monkeypatch: pytest.MonkeyPatch, index: CodeIndex) -> None:
    """Serve `index` wherever a loaded module reads the packaged index."""
    packaged = shared_module.get_index
    for module in list(sys.modules.values()):
        if vars(module).get("get_index") is packaged:
            monkeypatch.setattr(module, "get_index", lambda: index)


def _overlapping_index(*, tables_sector_only: bool = True) -> CodeIndex:
    """`_PLAIN` publishes a `sector` row and one that also carries `channel`, and `_TABLES`
    one that carries `sector` and `basis`: two families neither of whose properties hold
    the other's. With `tables_sector_only`, `_TABLES` publishes a `sector` row too, so the
    keys `{sector}` cover a row of each family and select neither."""
    rows = [
        _row("plain_health", {"sector": "Health"}, _PLAIN),
        _row("plain_health_ngo", {"sector": "Health", "channel": "NGO"}, _PLAIN),
        _row("tables_health_gross", {"sector": "Health", "basis": "Gross"}, _TABLES),
    ]
    if tables_sector_only:
        rows.append(_row("tables_health", {"sector": "Health"}, _TABLES))
    return _index(
        *rows,
        datasets=(_dataset(_PLAIN, rank=0), _dataset(_TABLES, rank=1)),
        declarations=tuple(_not_declared(prop) for prop in ("sector", "channel", "basis")),
    )


_INDEXES = {
    "nested": lambda: _split_index(sparse=True),
    "overlapping": _overlapping_index,
    "overlapping-one-sector-only": lambda: _overlapping_index(tables_sector_only=False),
}

_NESTED_PLAIN = ({"sector", "channel"}, {"plain_health", "plain_education", "plain_health_ngo"})
_OVERLAPPING_PLAIN = ({"sector", "channel"}, {"plain_health", "plain_health_ngo"})
_OVERLAPPING_TABLES = ({"sector", "basis"}, {"tables_health", "tables_health_gross"})


def _member(dcid: str, index: CodeIndex) -> dict:
    """The graph node of index row `dcid`, a subject-only StatVar."""
    row = next(row for row in index.statvars if row.dcid == dcid)
    return {
        "label": dcid,
        "arcs": {
            "name": {"nodes": [{"value": dcid}]},
            "populationType": {"nodes": [{"dcid": _FIVE_TUPLE.pop_type_dcid}]},
            "measuredProperty": {"nodes": [{"dcid": _FIVE_TUPLE.meas_prop_dcid}]},
            "statType": {"nodes": [{"dcid": _FIVE_TUPLE.stat_type_dcid}]},
            "constraintProperties": {
                "nodes": [{"dcid": prop} for prop in row.constraint_properties]
            },
            **{
                prop: {"nodes": [{"dcid": value}]}
                for prop, value in row.constraint_properties.items()
            },
        },
    }


def _derive(recalled: tuple[str, ...], index: CodeIndex) -> list:
    members = {dcid: _member(dcid, index) for dcid in recalled}
    return derive_shapes(
        confirmed_svs=list(members),
        graph=FakeGraph(nodes=members, obs={}, detect={}, resolve={}),
    )


@pytest.mark.parametrize(
    ("index_name", "recalled", "expected"),
    [
        # `_PLAIN`'s members' keys alone select `_TABLES`' family, which holds fewer
        # properties.
        (
            "nested",
            ("plain_health", "tables_health"),
            {
                _PLAIN: ({"plain_health"}, *_NESTED_PLAIN),
                _TABLES: ({"tables_health"}, {"sector"}, {"tables_health"}),
            },
        ),
        # The keys `{sector}` cover a row of each family and select neither.
        (
            "overlapping",
            ("plain_health",),
            {_PLAIN: ({"plain_health"}, *_OVERLAPPING_PLAIN)},
        ),
        (
            "overlapping",
            ("plain_health", "tables_health"),
            {
                _PLAIN: ({"plain_health"}, *_OVERLAPPING_PLAIN),
                _TABLES: ({"tables_health"}, *_OVERLAPPING_TABLES),
            },
        ),
        # A row carrying its family's sparse property selects that family by its keys too.
        (
            "overlapping",
            ("plain_health_ngo", "tables_health_gross"),
            {
                _PLAIN: ({"plain_health_ngo"}, *_OVERLAPPING_PLAIN),
                _TABLES: ({"tables_health_gross"}, *_OVERLAPPING_TABLES),
            },
        ),
        # Only `_PLAIN` has a `sector`-only row, so the keys `{sector}` select its family.
        (
            "overlapping-one-sector-only",
            ("plain_health", "tables_health_gross"),
            {
                _PLAIN: ({"plain_health"}, *_OVERLAPPING_PLAIN),
                _TABLES: ({"tables_health_gross"}, {"sector", "basis"}, {"tables_health_gross"}),
            },
        ),
    ],
    ids=[
        "nested",
        "overlapping-one-recalled",
        "overlapping-both-recalled",
        "overlapping-sparse-rows",
        "one-sector-only-row",
    ],
)
def test_a_derived_shape_reads_the_family_its_members_are_recorded_in(
    monkeypatch: pytest.MonkeyPatch,
    index_name: str,
    recalled: tuple[str, ...],
    expected: dict[str, tuple[set[str], set[str], set[str]]],
):
    """Each recalled row confirms a shape of the family the index records it in, whatever
    its keys select: the shape carries that family's sparse slot, and every reader of its
    family reads that family's rows."""
    index = _INDEXES[index_name]()
    _use_index(monkeypatch, index)

    by_dataset = {}
    for shape in _derive(recalled, index):
        family = shape_family(shape, index)
        assert family is not None
        assert all(family is index.family_by_dcid[dcid] for dcid in shape.sv_arc_facts or ())
        assert index_carries_shape(shape, index)
        (dataset,) = family.datasets
        rows = {row.dcid for row in index_member_rows(shape, index)}
        pre_bind, pre_bind_index_sourced = five_tuple_member_facts(shape)
        post_bind, post_bind_index_sourced = shape_member_facts(shape, shape_query(shape, (), ()))
        assert (set(pre_bind), pre_bind_index_sourced) == (rows, True)
        assert (set(post_bind), post_bind_index_sourced) == (rows, True)
        by_dataset[dataset] = (set(shape.sv_arc_facts or ()), shape_slot_properties(shape), rows)

    assert by_dataset == expected


@pytest.mark.parametrize(
    ("index_name", "selected"),
    [("overlapping", None), ("overlapping-one-sector-only", _PLAIN)],
)
def test_a_shape_with_no_recalled_member_reads_the_family_its_slot_properties_select(
    monkeypatch: pytest.MonkeyPatch, index_name: str, selected: str | None
):
    """Slot keys `{sector}` with no member behind them select a family only where one
    family alone has a `sector`-only row. Where both have one, the index decides nothing
    and the graph walk resolves the members."""
    index = _INDEXES[index_name]()
    _use_index(monkeypatch, index)
    (derived,) = _derive(("plain_health",), index)
    shape = dataclasses.replace(
        derived,
        sv_arc_facts=None,
        slot_keys=tuple(slot for slot in derived.slot_keys if slot.property_dcid == "sector"),
    )

    family = shape_family(shape, index)
    members, index_sourced = shape_member_facts(shape, shape_query(shape, (), ()))

    assert shape_slot_properties(shape) == {"sector"}
    assert (family.datasets if family is not None else None) == (selected and (selected,))
    assert index_sourced is (selected is not None)
    assert set(members) == ({"plain_health"} if selected else set())
    assert index_carries_shape(shape, index)


@pytest.mark.parametrize("index_name", ["nested", "overlapping"])
def test_a_shape_whose_members_belong_to_two_families_reads_neither(
    monkeypatch: pytest.MonkeyPatch, index_name: str
):
    """Derive never builds such a shape. One built by hand is not carried by the index, so
    it reads its own members and the rows of neither family, whichever family its slot
    properties select."""
    index = _INDEXES[index_name]()
    _use_index(monkeypatch, index)
    plain, tables = _derive(("plain_health", "tables_health"), index)
    mixed = dataclasses.replace(
        plain, sv_arc_facts={**(plain.sv_arc_facts or {}), **(tables.sv_arc_facts or {})}
    )

    assert shape_family(mixed, index) is None
    assert not index_carries_shape(mixed, index)
    assert index_member_rows(mixed, index) == ()
    members, index_sourced = five_tuple_member_facts(mixed)
    assert (set(members), index_sourced) == ({"plain_health", "tables_health"}, False)


def test_a_part_reads_its_shapes_own_tables_when_its_slot_keys_select_another_family(
    monkeypatch: pytest.MonkeyPatch,
):
    """`_PLAIN`'s shape with the `channel` slot left out keeps `_PLAIN`'s family, the one
    its member is recorded in, while its `sector` key alone selects `_TABLES`' family, which
    is what its unnamed identity resolves to at the mint. The part answers from `_PLAIN`
    and its identity names it. `_TABLES`' shape selects its own family and names none."""
    index = _split_index(sparse=True)
    _use_index(monkeypatch, index)
    shapes = _derive(("plain_health", "tables_health"), index)
    by_member = {next(iter(shape.sv_arc_facts or ())): shape for shape in shapes}
    plain = by_member["plain_health"]
    plain = dataclasses.replace(
        plain, slot_keys=tuple(key for key in plain.slot_keys if key.property_dcid != "channel")
    )
    tables = by_member["tables_health"]

    assert part_datasets(plain, None, index) == (_PLAIN,)
    assert part_datasets(plain, part_scope(_TABLES), index) == ()
    assert part_source(plain, [], _no_place) == _PLAIN
    assert part_datasets(tables, None, index) == (_TABLES,)
    assert part_source(tables, [], _no_place) is None
