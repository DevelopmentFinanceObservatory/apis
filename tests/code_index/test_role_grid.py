"""`index_family` and `dataset_declares_roles` decide the role grid as the rule in
`tests.statvar_members.role_grid` does.

Each population is one dataset's rows of one family. The packaged populations are read
from the packaged index, and each synthetic one from an index that holds it alone. A
cell asks the index twice, naming the dataset and naming none. Each family of
`role_grid.SHARED` is published by two datasets, and a cell asks its index naming none
and naming each.

The packaged grid keeps every population under every pair of role keys with no slot
bound, every slot binding that holds a cell of a moved label, and `_SAMPLED_SLOTS`
single-value bindings per population drawn with a fixed seed
(`role_grid.sampled_packaged_cells`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable, Iterator

import pytest

from code_index.members import (
    FamilyQuery,
    antichain_minimal_rows,
    dataset_declares_roles,
    index_family,
)
from code_index.query import CodeIndex
from tests.statvar_members.role_grid import (
    BOUND_KEYS,
    DATASET_CASES,
    DESTINATION,
    LABELS,
    NESTED,
    ONE_ROLE_SET,
    OPEN_HONOURED,
    OPEN_UNPLACED,
    OPEN_UNTWINNED,
    SYNTHETIC,
    Cell,
    Election,
    Population,
    SharedCell,
    count_by_case,
    count_by_label,
    declares,
    packaged_index,
    packaged_populations,
    sampled_packaged_cells,
    shared_cells,
    shared_index,
    synthetic_cells,
    synthetic_index,
)

_DAC1 = "provenance/OECDDAC1"
_SAMPLED_SLOTS = 12


def _sampled_packaged_cells() -> tuple[Cell, ...]:
    return sampled_packaged_cells(_SAMPLED_SLOTS)


_GRIDS = {"packaged": _sampled_packaged_cells, "synthetic": synthetic_cells}
_GRID_LABELS = {
    "packaged": (ONE_ROLE_SET, NESTED, OPEN_HONOURED, OPEN_UNPLACED, OPEN_UNTWINNED),
    "synthetic": LABELS,
}


def _index_of(grid: str, population: Population) -> CodeIndex:
    return packaged_index() if grid == "packaged" else synthetic_index(population.name)


def _query(cell: Cell | SharedCell, source: str | None) -> FamilyQuery:
    """The query `cell` asks, naming `source`."""
    five_tuple = cell.population.five_tuple if isinstance(cell, Cell) else cell.family.five_tuple
    return FamilyQuery(
        five_tuple=five_tuple,
        slots=cell.slots,
        source=source,
        role_keys=cell.bound,
        open_role_keys=cell.open_keys,
    )


def _resolved(index: CodeIndex, query: FamilyQuery):
    """`index_family`'s outcome for `query`: the elected dataset, its rows' dcids, the
    dcids of their antichain (`antichain_minimal_rows`), the tie and the elected keys, or
    `None` when the index leaves the query to the graph walk."""
    family = index_family(index, query)
    if family is None:
        return None
    return (
        family.provenance,
        frozenset(row.dcid for row in family.rows),
        frozenset(row.dcid for row in antichain_minimal_rows(index, family.rows)),
        family.tied,
        family.role_keys,
    )


def _agrees(resolved, dataset: str, expected: Election) -> bool:
    """Whether `resolved` names `dataset`, the rule's family, tie and keys, and, where the
    rule does not tie, an antichain that is the rule's members."""
    if resolved is None:
        return False
    provenance, family, members, tied, keys = resolved
    if (provenance, family, tied, keys) != (dataset, expected.family, expected.tied, expected.keys):
        return False
    return expected.tied or members == expected.members


@pytest.mark.parametrize("grid", _GRIDS)
def test_each_grid_holds_the_labels_its_cases_name(grid: str):
    """A label a grid stops holding, and one it starts to hold, both fail here."""
    counts = count_by_label(_GRIDS[grid]())
    print(f"{grid}: {counts}")

    assert {label for label, count in counts.items() if count} == set(_GRID_LABELS[grid])


@pytest.mark.parametrize(
    ("grid", "label"),
    [
        pytest.param(grid, label, id=f"{grid}: {label}")
        for grid, labels in _GRID_LABELS.items()
        for label in labels
    ],
)
def test_index_family_returns_the_family_the_rule_elects(grid: str, label: str):
    """Naming the dataset and naming none, `index_family` elects the cell's dataset and
    returns the rows of the rule's family: the rows of the elected role set, or every row
    placing the elected keys where the rule ties. Where it does not tie, the antichain of
    those rows is the rule's members."""
    found = [cell for cell in _GRIDS[grid]() if cell.expected.label == label]
    assert found, f"the {grid} grid holds no cell labelled {label!r}"

    differences = [
        f"{'named' if named else 'unnamed'}: {cell}"
        for cell in found
        for named in (True, False)
        if not _agrees(
            _resolved(
                _index_of(grid, cell.population),
                _query(cell, cell.population.dataset if named else None),
            ),
            cell.population.dataset,
            cell.expected,
        )
    ]

    assert not differences, (
        f"{len(differences)} of {2 * len(found)} queries differ; first: {differences[:1]}"
    )


def test_the_shared_grid_holds_every_case_of_the_dataset_rule():
    """Each step of the dataset rule decides at least one cell."""
    counts = count_by_case(shared_cells())
    print(f"shared: {counts}")

    assert all(counts[case] for case in DATASET_CASES), counts


@pytest.mark.parametrize("case", DATASET_CASES)
def test_index_family_elects_the_dataset_the_rule_elects_among_two(case: str):
    """Naming no dataset, `index_family` elects the dataset the rule elects among the two
    publishing the family, with the family, tie and keys of that dataset's own reading."""
    found = [cell for cell in shared_cells() if cell.expected.case == case]

    differences = [
        str(cell)
        for cell in found
        if not _agrees(
            _resolved(shared_index(cell.family.name), _query(cell, None)),
            cell.expected.dataset,
            cell.expected.election,
        )
    ]

    assert not differences, f"{len(differences)} of {len(found)} differ; first: {differences[:1]}"


def test_index_family_reads_a_named_dataset_of_two_on_its_own_rows():
    """Naming one of two publishers, `index_family` reads that dataset's rows alone by
    the rule, and returns no row where the slot binding admits none of them."""
    differences = []
    for cell in shared_cells():
        index = shared_index(cell.family.name)
        for dataset in cell.family.datasets:
            resolved = _resolved(index, _query(cell, dataset))
            expected = cell.named(dataset)
            if expected is None:
                agrees = resolved is not None and resolved[:2] == (None, frozenset())
            else:
                agrees = _agrees(resolved, dataset, expected)
            if not agrees:
                differences.append(f"{dataset}: {cell}")

    assert not differences, f"{len(differences)} differ; first: {differences[:1]}"


def _queries(grid: str) -> Iterator[tuple[CodeIndex, FamilyQuery, str]]:
    """Every query the cells of `grid` ask, naming a dataset and naming none."""
    if grid == "shared":
        for shared in shared_cells():
            index = shared_index(shared.family.name)
            for source in (None, *shared.family.datasets):
                yield index, _query(shared, source), f"{source}: {shared}"
        return
    for cell in _GRIDS[grid]():
        index = _index_of(grid, cell.population)
        for source in (None, cell.population.dataset):
            yield index, _query(cell, source), f"{source}: {cell}"


@pytest.mark.parametrize("grid", [*_GRIDS, "shared"])
def test_asking_again_with_only_the_joined_open_keys_elects_the_same_family(grid: str):
    """A query whose open keys are the open ones among the keys `index_family` returned
    elects the same dataset, rows, keys and tie. Search mints an identity with those open
    keys, so the mint reads the members Search answered from."""
    asked = 0
    differences = []
    for index, query, name in _queries(grid):
        family = index_family(index, query)
        if family is None:
            continue
        asked += 1
        joined = dataclasses.replace(query, open_role_keys=family.role_keys - query.role_keys)
        if index_family(index, joined) != family:
            differences.append(name)

    assert asked
    assert not differences, f"{len(differences)} of {asked} differ; first: {differences[:1]}"


def _declared(
    grid: str, population: Population, key_sets: Iterable[Collection[str]] = BOUND_KEYS
) -> list[tuple[bool, bool, str]]:
    """For each set of `key_sets`: whether `dataset_declares_roles` and the rule say
    the population's dataset declares it in the population's family."""
    index = _index_of(grid, population)
    family = next(
        family
        for family in index.families_by_five_tuple[population.five_tuple]
        if population.dataset in family.datasets
    )
    return [
        (
            dataset_declares_roles(index, family, population.dataset, keys),
            declares(population.rows, keys, readings=population.readings),
            f"{population.name} {sorted(keys)}",
        )
        for keys in map(frozenset, key_sets)
    ]


def test_each_packaged_dataset_declares_the_keys_the_rule_says():
    """Some packaged datasets declare a set of keys and others do not, each as the
    rule says."""
    answers = [
        answer
        for population in packaged_populations(packaged_index())
        for answer in _declared("packaged", population)
    ]

    assert {ruled for _declared_, ruled, _keys in answers} == {True, False}
    assert not [keys for declared, ruled, keys in answers if declared != ruled]


def test_no_packaged_dac1_family_declares_a_recipient():
    """OECD DAC1 publishes provider totals, two of them flows to one recipient. Both
    `dataset_declares_roles` and the rule say it declares no recipient in any family."""
    answers = [
        (declared, ruled)
        for population in packaged_populations(packaged_index())
        if population.dataset == _DAC1
        for declared, ruled, _keys in _declared("packaged", population, ({DESTINATION},))
    ]

    assert answers
    assert set(answers) == {(False, False)}


@pytest.mark.parametrize("name", [population.name for population in SYNTHETIC])
def test_each_synthetic_dataset_declares_the_keys_the_rule_says(name: str):
    """A dataset declares a set of keys when a row its rows read with nothing bound, or a
    twin of one, places them."""
    population = next(found for found in SYNTHETIC if found.name == name)

    answers = _declared("synthetic", population)

    assert len(answers) == len(BOUND_KEYS)
    assert not [keys for declared, ruled, keys in answers if declared != ruled]
