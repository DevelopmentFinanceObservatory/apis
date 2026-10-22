"""`elect_role_class` and the graph walk decide every cell of the role grid as the rule in
`role_grid` does.

The grid crosses populations of rows, slot bindings, bound role keys and open role keys.
The fixture populations are each dataset's rows of each family of the fixture code
index, and the synthetic ones cover the structures the fixture index lacks. Each test
runs once per grid and label, so a difference names the case it falls in.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection

import pytest

from statvar_members import (
    MemberRoleDisagreement,
    elect_role_class,
    enumerate_members,
)
from statvar_members.enumerate import RoleClassElection

from .conftest import FakeMemberArcReader
from .role_grid import (
    LABELS,
    MOVED_LABELS,
    NESTED,
    ONE_ROLE_SET,
    OPEN_HONOURED,
    OPEN_UNPLACED,
    OPEN_UNTWINNED,
    SUBJECT,
    Cell,
    Row,
    antichain,
    count_by_label,
    elect,
    elect_any_row,
    fixture_cells,
    synthetic_cells,
)

_DAC1 = "provenance/OECDDAC1"

_GRIDS: dict[str, Callable[[], tuple[Cell, ...]]] = {
    "fixture": fixture_cells,
    "synthetic": synthetic_cells,
}
_GRID_LABELS = {
    "fixture": (ONE_ROLE_SET, NESTED, OPEN_HONOURED, OPEN_UNPLACED, OPEN_UNTWINNED),
    "synthetic": LABELS,
}
"""The labels each grid holds. The fixture index has no role twin, no role sets that do
not nest, and no nested role sets whose rows hold other constraint keys under one slot
binding."""


def _cases(*, grids: tuple[str, ...] = tuple(_GRIDS)):
    """One case per grid and label it holds."""
    return [
        pytest.param(grid, label, id=f"{grid}: {label}")
        for grid in grids
        for label in _GRID_LABELS[grid]
    ]


def _labelled(grid: str, label: str) -> list[Cell]:
    found = [cell for cell in _GRIDS[grid]() if cell.expected.label == label]
    assert found, f"the {grid} grid holds no cell labelled {label!r}"
    return found


@functools.cache
def _elections(grid: str) -> dict[int, RoleClassElection]:
    """`elect_role_class` over each cell of `grid` under the cell's keys, by the cell's
    position in the grid."""
    return {
        position: _run(cell, cell.rows, bound=cell.bound, open_keys=cell.open_keys)
        for position, cell in enumerate(_GRIDS[grid]())
    }


def _elected(grid: str, label: str) -> list[tuple[Cell, RoleClassElection]]:
    """The cells of `grid` labelled `label`, each with its election."""
    elections = _elections(grid)
    found = [
        (cell, elections[position])
        for position, cell in enumerate(_GRIDS[grid]())
        if cell.expected.label == label
    ]
    assert found, f"the {grid} grid holds no cell labelled {label!r}"
    return found


def _run(cell: Cell, rows: tuple[Row, ...], *, bound, open_keys=()) -> RoleClassElection:
    return elect_role_class(
        {row.dcid: row.constraints for row in rows},
        {row.dcid: row.roles for row in rows},
        [row.dcid for row in rows],
        role_keys=bound,
        open_role_keys=open_keys,
        readings=cell.population.readings,
    )


def _outcome(election: RoleClassElection):
    return (
        frozenset(election.family),
        frozenset(election.members),
        election.tied,
        election.role_keys,
    )


def _antichain_of(cell: Cell, family: Collection[str]) -> frozenset[str]:
    """The dcids of the antichain of `cell`'s rows named in `family`."""
    named = frozenset(family)
    rows = [row for row in cell.rows if row.dcid in named]
    return frozenset(row.dcid for row in antichain(rows, cell.population.readings))


def _report(differences: list[str], checked: int) -> str:
    return f"{len(differences)} of {checked} cells differ; first: {differences[:1]}"


@pytest.mark.parametrize("grid", _GRIDS)
def test_each_grid_holds_the_labels_its_cases_name(grid: str):
    """A label a grid stops holding, and one it starts to hold, both fail here: every
    other test reads the cells of the labels `_GRID_LABELS` names."""
    counts = count_by_label(_GRIDS[grid]())
    print(f"{grid}: {counts}")

    assert {label for label, count in counts.items() if count} == set(_GRID_LABELS[grid])


@pytest.mark.parametrize(("grid", "label"), _cases())
def test_the_election_decides_each_cell_as_the_rule_does(grid: str, label: str):
    """The family, the members, the tie and the elected keys of `elect_role_class` over a
    cell's rows are the rule's."""
    found = _elected(grid, label)

    differences = [
        str(cell) for cell, election in found if _outcome(election) != cell.expected.outcome
    ]

    assert not differences, _report(differences, len(found))


def _arc(value: str | None) -> tuple[str, ...]:
    return () if value is None else (value,)


def _walked(cell: Cell):
    """What `enumerate_members` decides for `cell` over a graph holding its population:
    the outcome an election reports, with a role disagreement read as the tie."""
    population = cell.population
    five_tuple = population.five_tuple
    arcs = {
        row.dcid: {
            "typeOf": ("StatisticalVariable",),
            "populationType": _arc(five_tuple.population_type),
            "measuredProperty": _arc(five_tuple.measured_property),
            "statType": _arc(five_tuple.stat_type),
            "constraintProperties": tuple(row.constraints),
            "observationProperties": () if row.roles == (SUBJECT,) else row.roles,
            **{prop: (value,) for prop, value in row.constraints.items()},
        }
        for row in population.rows
    }
    try:
        group = enumerate_members(
            five_tuple=five_tuple,
            slots=cell.slots,
            graph=FakeMemberArcReader(arcs=arcs),
            readings=population.readings,
            role_keys=cell.bound,
            open_role_keys=cell.open_keys,
        )
    except MemberRoleDisagreement as disagreement:
        return ("tied", frozenset(disagreement.roles_by_stat_var))
    return (
        frozenset(group.family),
        frozenset(member.stat_var for member in group.members),
        False,
        cell.bound | group.elected_open_role_keys,
    )


@pytest.mark.parametrize(("grid", "label"), _cases(grids=("synthetic",)))
def test_the_walk_decides_each_cell_as_the_rule_does(grid: str, label: str):
    """`enumerate_members` over a graph holding a synthetic population returns the rule's
    family, members and elected keys, and raises `MemberRoleDisagreement` naming the
    rule's members where the rule ties. The members it returns are the antichain of its
    family."""
    found = _labelled(grid, label)
    differences = []

    for cell in found:
        walked = _walked(cell)
        if cell.expected.tied:
            agrees = walked == ("tied", cell.expected.members)
        else:
            agrees = walked == cell.expected.outcome and walked[1] == _antichain_of(cell, walked[0])
        if not agrees:
            differences.append(str(cell))

    assert not differences, _report(differences, len(found))


@pytest.mark.parametrize(("grid", "label"), _cases())
def test_electing_over_an_elected_family_returns_it_and_its_members(grid: str, label: str):
    """A reader handed an election's family elects among it again under the same keys.
    Both `elect_role_class` and the rule return the family they were given. Where the rule
    does not tie, the members are the antichain of the family, and the second election
    returns them."""
    found = _elected(grid, label)
    differences = []

    for cell, election in found:
        elected = frozenset(election.family)
        family = tuple(row for row in cell.rows if row.dcid in elected)
        again = _run(cell, family, bound=cell.bound, open_keys=cell.open_keys)
        if again.family != election.family:
            differences.append(f"elect_role_class family: {cell}")
        elif not cell.expected.tied and not (
            frozenset(election.members) == _antichain_of(cell, elected) == frozenset(again.members)
        ):
            differences.append(f"elect_role_class members: {cell}")
        expected_family = tuple(row for row in cell.rows if row.dcid in cell.expected.family)
        ruled = elect(
            expected_family,
            bound=cell.bound,
            open_keys=cell.open_keys,
            readings=cell.population.readings,
        )
        if ruled.family != cell.expected.family:
            differences.append(f"rule family: {cell}")
        elif not cell.expected.tied and not (
            cell.expected.members == _antichain_of(cell, cell.expected.family) == ruled.members
        ):
            differences.append(f"rule members: {cell}")

    assert not differences, _report(differences, len(found))


@pytest.mark.parametrize(("grid", "label"), _cases())
def test_electing_under_the_elected_keys_alone_returns_the_same_members(grid: str, label: str):
    """An identity carries the keys an election ran under, bound, with nothing open. An
    election under those keys reads the members the first one read, for
    `elect_role_class` and for the rule."""
    found = _elected(grid, label)
    differences = []

    for cell, election in found:
        again = _run(cell, cell.rows, bound=election.role_keys)
        if (again.members, again.tied) != (election.members, election.tied):
            differences.append(f"elect_role_class: {cell}")
        ruled = elect(cell.rows, bound=cell.expected.keys, readings=cell.population.readings)
        if (ruled.members, ruled.tied, ruled.keys) != cell.expected.outcome[1:]:
            differences.append(f"rule: {cell}")

    assert not differences, _report(differences, len(found))


@pytest.mark.parametrize("grid", _GRIDS)
def test_the_rule_departs_from_the_any_row_rule_on_the_moved_labels_alone(grid: str):
    """The rule decides a cell otherwise than the any-row rule exactly when the cell's
    label is one of `MOVED_LABELS`."""
    found = _GRIDS[grid]()
    moved = [cell for cell in found if cell.expected.label in MOVED_LABELS]

    mislabelled = [
        str(cell)
        for cell in found
        if (
            elect_any_row(
                cell.rows,
                bound=cell.bound,
                open_keys=cell.open_keys,
                readings=cell.population.readings,
            ).outcome
            != cell.expected.outcome
        )
        != (cell.expected.label in MOVED_LABELS)
    ]

    assert moved
    assert len(moved) < len(found)
    assert not mislabelled, _report(mislabelled, len(found))


def test_the_moved_labels_occur_in_the_fixture_index_only_in_dac1():
    """Of the fixture datasets, only OECD DAC1 publishes a family whose rows declare
    nested role sets, so the rule reads every other dataset as the any-row rule does."""
    moved = {
        cell.population.dataset for cell in fixture_cells() if cell.expected.label in MOVED_LABELS
    }

    assert moved == {_DAC1}
