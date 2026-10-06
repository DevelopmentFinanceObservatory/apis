"""A grid of identities over the five-tuples several datasets publish, and the outcome the
index rows decide for each.

`grid_identities` crosses slot-key sets, bindings and named datasets. `asked_family` reads
the outcome of one identity's query from the rows of its five-tuple alone: it reads no
`IndexFamily`, and so states what `code_index.members.index_family` must return.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import defaultdict
from collections.abc import Iterator, Sequence
from collections.abc import Set as AbstractSet

from code_index.members import FamilyQuery
from code_index.query import CodeIndex, row_five_tuple
from code_index.types import StatVarRow
from statvar_members import FiveTuple, elect_role_class

STRAY_PROPERTY = "propertyNoRowCarries"
OUTSIDE_PROVENANCE = "provenance/NotInTheIndex"
_UNBOUND = {"state": "unbound"}
_ENTITIES = {"destination": ["country/KEN"]}


def shared_five_tuples(index: CodeIndex) -> list[FiveTuple]:
    """The five-tuples of `index` that more than one dataset publishes."""
    return [ft for ft, datasets in index.datasets_by_five_tuple.items() if len(datasets) > 1]


@dataclasses.dataclass(frozen=True)
class FiveTupleRows:
    """The rows of one five-tuple, as the index lists them."""

    rows: tuple[StatVarRow, ...]
    carried: dict[str, frozenset[str]]
    """The constraint properties each dataset's rows carry."""
    by_keys: dict[frozenset[str], tuple[StatVarRow, ...]]
    """The rows declaring each set of constraint properties."""


def five_tuple_rows(index: CodeIndex, five_tuple: FiveTuple) -> FiveTupleRows:
    rows = tuple(row for row in index.statvars if row_five_tuple(row) == five_tuple)
    carried: dict[str, set[str]] = defaultdict(set)
    by_keys: dict[frozenset[str], list[StatVarRow]] = defaultdict(list)
    for row in rows:
        by_keys[frozenset(row.constraint_properties)].append(row)
        for dataset in row.provenances:
            carried[dataset].update(row.constraint_properties)
    return FiveTupleRows(
        rows=rows,
        carried={dataset: frozenset(props) for dataset, props in carried.items()},
        by_keys={keys: tuple(found) for keys, found in by_keys.items()},
    )


def _key_sets(published: FiveTupleRows) -> list[frozenset[str]]:
    """The slot-key sets of the grid: each dataset's row key sets and carried properties,
    the union and the intersection of those, no key, and each of them with a property no row
    carries, with one more property a row carries, and with one property less."""
    carried = published.carried.values()
    every = frozenset().union(*carried)
    seeds = {*published.by_keys, *carried, every, every.intersection(*carried), frozenset()}
    found = set(seeds)
    for keys in seeds:
        found.add(keys | {STRAY_PROPERTY})
        found.update(keys | {prop} for prop in every - keys)
        found.update(keys - {prop} for prop in keys)
    return sorted(found, key=sorted)


def _bindings(published: FiveTupleRows, keys: frozenset[str]) -> Iterator[dict[str, str]]:
    """The value bindings of the grid for `keys`: none; the shared properties bound to the
    values each dataset's first row holds, and to a value no row holds; and each other
    property bound alone to the first value a row holds."""
    yield {}
    rows, carried = published.rows, published.carried
    shared = sorted(keys.intersection(*carried.values()))
    seen: set[tuple[str, ...]] = set()
    for dataset in sorted(carried):
        first = next(row for row in rows if dataset in row.provenances)
        values = tuple(first.constraint_properties.get(prop, "") for prop in shared)
        if shared and all(values) and values not in seen:
            seen.add(values)
            yield dict(zip(shared, values, strict=True))
    if shared:
        yield {shared[0]: "ValueNoRowHolds"}
    for prop in sorted(keys - set(shared) - {STRAY_PROPERTY}):
        held = (row.constraint_properties.get(prop) for row in rows)
        yield {prop: min(value for value in held if value is not None)}


def _identity(
    five_tuple: FiveTuple, keys: AbstractSet[str], bound: dict[str, str], source: str | None
) -> dict:
    shape = {
        field: value
        for field in ("population_type", "measured_property", "stat_type", "measurement_qualifier")
        if (value := getattr(five_tuple, field)) is not None
    }
    identity: dict = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", **shape},
        "slots": [
            {
                "property": prop,
                "binding": {"state": "value", "ref": bound[prop]} if prop in bound else _UNBOUND,
            }
            for prop in sorted(keys)
        ],
        "entities": _ENTITIES,
        "requested_window": {"kind": "all"},
    }
    if source is not None:
        identity["facets"] = {"provenance": {"state": "value", "ref": source}}
    return identity


def grid_identities(index: CodeIndex) -> Iterator[dict]:
    """Every identity of the grid over `index`: on each five-tuple several datasets publish
    (none of which has a denominator), each slot-key set (`_key_sets`) under each binding
    (`_bindings`), naming no dataset, each declared dataset, each dataset publishing the
    five-tuple, and a provenance the index does not hold."""
    for five_tuple in shared_five_tuples(index):
        assert five_tuple.measurement_denominator is None, five_tuple
        published = five_tuple_rows(index, five_tuple)
        declared = {*index.datasets, *index.datasets_by_five_tuple[five_tuple]}
        sources = (None, *sorted(declared), OUTSIDE_PROVENANCE)
        for keys in _key_sets(published):
            for bound, source in itertools.product(_bindings(published, keys), sources):
                yield _identity(five_tuple, keys, bound, source)


@dataclasses.dataclass(frozen=True)
class Asked:
    """What the index rows decide for one query."""

    refused: frozenset[str] | None = None
    """The datasets a `SourceNotInFamily` refusal names, or `None` when the query is not
    refused."""
    decided: bool = False
    """Whether the index decides the query's membership."""
    datasets: frozenset[str] = frozenset()
    """The datasets that may serve a decided query."""
    rows: dict[str, tuple[StatVarRow, ...]] = dataclasses.field(default_factory=dict)
    """For each of `datasets`, the rows it publishes that the query describes, before the
    role-class election."""


def _selected(
    carried: dict[str, frozenset[str]], fits: dict[str, list[StatVarRow]], keys: AbstractSet[str]
) -> frozenset[str] | None:
    """The datasets slot keys `keys` select with no dataset named: those of one property
    set. It is the one property set a fitting dataset carries. With several or none, it is
    the one among them, or among all, that holds every key and extends no other that does."""
    groups: dict[frozenset[str], set[str]] = defaultdict(set)
    for dataset, props in carried.items():
        groups[props].add(dataset)
    fitting = [props for props, datasets in groups.items() if any(fits[d] for d in datasets)]
    if len(fitting) != 1:
        holding = [props for props in fitting or groups if keys <= props]
        fitting = [props for props in holding if not any(other < props for other in holding)]
    return frozenset(groups[fitting[0]]) if len(fitting) == 1 else None


def asked_family(published: FiveTupleRows, query: FamilyQuery) -> Asked:
    """What `published`, the rows of `query`'s five-tuple, decide for `query`.

    A dataset's rows fit the query when their constraint properties lie within its slot
    keys, less the unbound keys no row carries. The named dataset serves when its rows fit
    and carry every bound property: the index then decides the query, to its fitting rows
    that hold the bound values, unless the keys select no dataset at all. A named dataset
    that does not serve is refused for the datasets the keys select, when they select
    others. It is not refused when no dataset's rows fit and its own carry every key: the
    index does not decide those keys. A query naming no dataset is served by the datasets
    its keys select."""
    carried = published.carried
    every = frozenset().union(*carried.values())
    keys = {prop for prop, bound in query.slots.items() if bound != ("unbound",) or prop in every}
    fitting = [
        row for row_keys, rows in published.by_keys.items() if row_keys <= keys for row in rows
    ]
    fits = {dataset: [row for row in fitting if dataset in row.provenances] for dataset in carried}
    selected = _selected(carried, fits, keys)
    decided = selected is not None and any(fits[dataset] for dataset in selected)
    source = query.source
    serving = selected or frozenset()
    if source is not None:
        own = carried.get(source)
        fitted = any(fits.values())
        if own is not None and query.bound <= own and (fits[source] if fitted else keys <= own):
            serving = frozenset({source})
        else:
            others = selected if selected is not None else frozenset(carried)
            if source not in others:
                return Asked(refused=others)
            serving = frozenset()
    allowed: dict[str, set[str]] = {}
    for prop, binding in query.slots.items():
        match binding:
            case ("value", ref):
                allowed[prop] = {ref}
            case ("set", refs):
                allowed[prop] = set(refs)
    return Asked(
        decided=decided,
        datasets=serving,
        rows={
            dataset: tuple(
                row
                for row in fits[dataset]
                if all(row.constraint_properties.get(p) in refs for p, refs in allowed.items())
            )
            for dataset in serving
        },
    )


def role_class_rows(
    index: CodeIndex, rows: Sequence[StatVarRow], role_keys: AbstractSet[str]
) -> frozenset[str]:
    """The dcids of the role class `role_keys` elect among `rows`."""
    election = elect_role_class(
        {row.dcid: row.constraint_properties for row in rows},
        {row.dcid: row.observation_properties or ("observationAbout",) for row in rows},
        [row.dcid for row in rows],
        role_keys=role_keys,
        readings={
            system: declared.absent_reading for system, declared in index.declarations.items()
        },
    )
    return frozenset(election.family)
