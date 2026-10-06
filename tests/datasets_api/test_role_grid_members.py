"""The members a mint serves for an identity are the ones the rule in
`tests.statvar_members.role_grid` elects.

`POST /api/datasets/{id}/members` serves what `resolve_definition_members` resolves for the
identity and `check_reproducible` admits. Each cell of the role grid becomes the identity
Search mints for it: its bound keys as entities and, as open roles, the open keys the rule
admits. Three paths resolve it: the packaged code index, a synthetic index that holds one
population, and the graph walk over a graph that holds one. A family several datasets
publish (`role_grid.SHARED`) is asked of an index holding it, naming each dataset and
naming none.

The packaged grid is `role_grid.sampled_packaged_cells` with `_SAMPLED_SLOTS` single-value
bindings per population. The index paths ask each cell naming the dataset and naming none.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import pytest

from code_index.query import CodeIndex
from datasets_api.adapter import check_reproducible, resolve_definition_members
from datasets_api.codes_materialise import DefinitionNotReproducible, RoleSignatureIncompatible
from statvar_members import MemberEnumerator, SlotBinding, member_digest
from statvar_members.types import FiveTuple
from tests.statvar_members.role_grid import (
    DATASET_CASES,
    LABELS,
    NESTED,
    ONE_ROLE_SET,
    OPEN_HONOURED,
    OPEN_UNPLACED,
    OPEN_UNTWINNED,
    SUBJECT,
    Cell,
    Election,
    Row,
    SharedCell,
    packaged_index,
    places,
    sampled_packaged_cells,
    shared_cells,
    shared_index,
    synthetic_cells,
    synthetic_index,
)

from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds

if TYPE_CHECKING:
    from materialise.graph import GraphReader

_SPEC_ID = "spec_role_grid"
_SAMPLED_SLOTS = 4
TIED = ("role_signature_incompatible",)
NOT_REPRODUCIBLE = ("definition_not_reproducible",)

PACKAGED = "packaged index"
SYNTHETIC_INDEX = "synthetic index"
WALK = "walk"
_PATH_LABELS = {
    PACKAGED: (ONE_ROLE_SET, NESTED, OPEN_HONOURED, OPEN_UNPLACED, OPEN_UNTWINNED),
    SYNTHETIC_INDEX: LABELS,
    WALK: LABELS,
}
"""The labels each path's grid holds."""


def _cells(path: str) -> tuple[Cell, ...]:
    return sampled_packaged_cells(_SAMPLED_SLOTS) if path == PACKAGED else synthetic_cells()


def minted_identity(cell: Cell, *, named: bool, open_roles: Collection[str] | None = None) -> dict:
    """The identity minted for `cell`: an entity under each bound key, `open_roles` as its
    open roles, by default the open keys the rule admits, and the dataset in the provenance
    facet when `named`. The bound slot is `exact`, so a parent code names its own rows and
    not its descendants'."""
    if open_roles is None:
        open_roles = cell.expected.keys - cell.bound
    return _identity(
        cell.population.five_tuple,
        cell.slots,
        bound=cell.bound,
        open_roles=open_roles,
        dataset=cell.population.dataset if named else None,
    )


def shared_identity(
    cell: SharedCell, *, dataset: str | None, open_roles: Collection[str] | None = None
) -> dict:
    """The identity minted for `cell` of a family several datasets publish, naming
    `dataset` or none. `open_roles` default to the open keys the rule admits for it."""
    if open_roles is None:
        open_roles = _shared_election(cell, dataset).keys - cell.bound
    return _identity(
        cell.family.five_tuple, cell.slots, bound=cell.bound, open_roles=open_roles, dataset=dataset
    )


def _shared_election(cell: SharedCell, dataset: str | None) -> Election:
    """What the rule elects for `cell` naming `dataset`, which then holds a row, or none."""
    election = cell.expected.election if dataset is None else cell.named(dataset)
    assert election is not None
    return election


def _identity(
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
    *,
    bound: Collection[str],
    open_roles: Collection[str],
    dataset: str | None,
) -> dict:
    shape: dict = {
        "kind": "statvar_schema",
        **{
            field: value
            for field in (
                "population_type",
                "measured_property",
                "stat_type",
                "measurement_qualifier",
            )
            if (value := getattr(five_tuple, field)) is not None
        },
    }
    if five_tuple.measurement_denominator is not None:
        shape["measurement_denominator"] = {"ref": five_tuple.measurement_denominator}
    identity: dict = {
        "identity_version": "spec1",
        "shape": shape,
        "slots": [
            {
                "property": prop,
                "binding": {"state": "exact", "ref": binding[1]}
                if binding[0] == "value"
                else {"state": "unbound"},
            }
            for prop, binding in sorted(slots.items())
        ],
        "entities": {key: ["country/KEN"] for key in sorted(bound)},
        "requested_window": {"kind": "all"},
    }
    if open_roles:
        identity["open_roles"] = sorted(open_roles)
    if dataset is not None:
        identity["facets"] = {"provenance": {"state": "value", "ref": dataset}}
    return identity


def population_graph(cell: Cell) -> FakeMintGraphReader:
    """A graph holding `cell`'s population."""
    five_tuple = cell.population.five_tuple
    return FakeMintGraphReader(
        arcs={
            row.dcid: {
                "typeOf": ("StatisticalVariable",),
                "populationType": (five_tuple.population_type or "",),
                "measuredProperty": (five_tuple.measured_property or "",),
                "statType": (five_tuple.stat_type or "",),
                "constraintProperties": tuple(row.constraints),
                "observationProperties": () if row.roles == (SUBJECT,) else row.roles,
                **{prop: (value,) for prop, value in row.constraints.items()},
            }
            for row in cell.population.rows
        }
    )


def served(cell: Cell, path: str, *, named: bool, open_roles: Collection[str] | None = None):
    """What the mint serves for `cell`'s identity on `path` (`minted_identity`): the member
    dcids with the family's count and digest, or the code of its refusal."""
    identity = minted_identity(cell, named=named, open_roles=open_roles)
    index: CodeIndex = (
        synthetic_index(cell.population.name) if path == SYNTHETIC_INDEX else packaged_index()
    )
    graph = population_graph(cell) if path == WALK else FakeMintGraphReader(arcs={})
    return _served(identity, index, graph)[:3]


def shared_served(
    cell: SharedCell, *, dataset: str | None, open_roles: Collection[str] | None = None
):
    """What the mint serves for `cell`'s identity (`shared_identity`) from an index holding
    its family: the member dcids, the family's count and digest and the elected dataset, or
    the code of its refusal."""
    return _served(
        shared_identity(cell, dataset=dataset, open_roles=open_roles),
        shared_index(cell.family.name),
        FakeMintGraphReader(arcs={}),
    )


def _served(identity: dict, index: CodeIndex, graph: FakeMintGraphReader):
    try:
        resolved = resolve_definition_members(
            identity,
            spec_id=_SPEC_ID,
            graph=cast("GraphReader", graph),
            enumerator=MemberEnumerator(),
            code_index=index,
            bounds=mint_bounds(),
        )
        check_reproducible(identity, spec_id=_SPEC_ID, shape=resolved.shapes[0])
    except RoleSignatureIncompatible:
        return TIED
    except DefinitionNotReproducible:
        return NOT_REPRODUCIBLE
    family = resolved.shapes[0].family
    assert family is not None
    datasets = {shape.dataset.provenance for shape in resolved.shapes if shape.dataset}
    return (
        frozenset(shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes),
        family.count,
        family.digest,
        next(iter(datasets)) if len(datasets) == 1 else None,
    )


def expected_served(cell: Cell):
    """What the rule says the mint serves for `cell`'s identity. A tie refuses
    `role_signature_incompatible`. Members that do not place a key the identity names
    refuse `definition_not_reproducible`: the rule reads every row when none places the
    bound keys. Else the rule's members, with the count and digest of its family."""
    return _expected_served(cell.expected, cell.rows)


def shared_expected_served(cell: SharedCell, *, dataset: str | None):
    """What the rule says the mint serves for `cell`'s identity naming `dataset`, which
    then holds a row, or none (`expected_served`), with the dataset it elects."""
    elected = cell.expected.dataset if dataset is None else dataset
    outcome = _expected_served(_shared_election(cell, dataset), cell.rows[elected])
    return outcome if isinstance(outcome[0], str) else (*outcome, elected)


def _expected_served(expected: Election, rows: Sequence[Row]):
    if expected.tied:
        return TIED
    roles = next(row.roles for row in rows if row.dcid in expected.members)
    if not all(places(roles, frozenset({key})) for key in expected.keys):
        return NOT_REPRODUCIBLE
    return (expected.members, len(expected.family), member_digest(expected.family))


@pytest.mark.parametrize(
    ("path", "label"),
    [
        pytest.param(path, label, id=f"{path}: {label}")
        for path, labels in _PATH_LABELS.items()
        for label in labels
    ],
)
def test_the_mint_serves_the_members_the_rule_elects(path: str, label: str):
    """The mint serves the rule's members and states its family, refuses
    `role_signature_incompatible` only where the rule ties, and refuses
    `definition_not_reproducible` only where no row places a key the identity names."""
    found = [cell for cell in _cells(path) if cell.expected.label == label]
    assert found, f"the {path} grid holds no cell labelled {label!r}"
    asked = (True,) if path == WALK else (True, False)

    differences = [
        f"{'named' if named else 'unnamed'}: {cell}"
        for cell in found
        for named in asked
        if served(cell, path, named=named) != expected_served(cell)
    ]

    assert not differences, (
        f"{len(differences)} of {len(asked) * len(found)} identities differ; "
        f"first: {differences[:1]}"
    )


@pytest.mark.parametrize("path", _PATH_LABELS)
def test_each_path_refuses_and_serves(path: str):
    """Each path's grid holds identities the rule serves and ones it refuses as not
    reproducible, and the synthetic ones hold ties, so no outcome goes unread."""
    outcomes = {
        outcome if isinstance(outcome[0], str) else ("served",)
        for outcome in map(expected_served, _cells(path))
    }

    assert outcomes == {("served",), NOT_REPRODUCIBLE} | (set() if path == PACKAGED else {TIED})


@pytest.mark.parametrize("case", DATASET_CASES)
def test_the_mint_serves_a_shared_family_from_the_dataset_the_rule_elects(case: str):
    """For a family several datasets publish, an identity naming no dataset is served from
    the dataset the rule elects, with the members and family of that dataset's own
    reading, and one naming a dataset from that dataset's."""
    found = [cell for cell in shared_cells() if cell.expected.case == case]
    assert found, f"the shared grid holds no cell of the case {case!r}"
    asked = [
        (cell, dataset)
        for cell in found
        for dataset in (None, *cell.family.datasets)
        if dataset is None or cell.rows[dataset]
    ]

    differences = [
        f"{dataset}: {cell}"
        for cell, dataset in asked
        if shared_served(cell, dataset=dataset) != shared_expected_served(cell, dataset=dataset)
    ]

    assert not differences, (
        f"{len(differences)} of {len(asked)} identities differ; first: {differences[:1]}"
    )
