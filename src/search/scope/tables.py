"""The inputs compile reads: one table per axis, the conflicts between options of different
axes, and the engine facts the axis rules build the tables from."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from code_index.query import CodeIndex, IndexFamily
from search.models import EntityGroup, Operation, Option, ScopeAxis
from search.scope.edits import AXIS_BINDING, ScopeTableError
from series_coverage.shared import CoverageSnapshot


@dataclass(frozen=True, kw_only=True)
class AxisTable:
    """One axis of one part, as its rule reads it.

    `options` are the axis's options in declared order, each with its binding and its
    intrinsic availability (what the part's own data and constraints allow, before any
    other entry is weighed). `preference` is the order a default is derived in: the first
    option no entry blocks. `held` is the option the base identity holds. `rule` names the
    declared rule, and `concept` the concept the axis was asked for, None when it has none.

    Raises `ScopeTableError` for an empty table, a repeated option, an option of another
    axis or binding kind, or a `preference` or `held` naming no option."""

    axis: ScopeAxis
    rule: str
    options: tuple[Option, ...]
    preference: tuple[str, ...]
    held: str
    concept: str | None = None

    def __post_init__(self) -> None:
        ids = [option.option_id for option in self.options]
        if not ids or len(set(ids)) != len(ids):
            raise ScopeTableError(f"{self.axis!r} needs distinct options, got {ids}")
        for option in self.options:
            if option.option_id.partition(":")[0] != self.axis:
                raise ScopeTableError(f"{option.option_id!r} is not an option of {self.axis!r}")
            if option.binding.kind != AXIS_BINDING[self.axis]:
                raise ScopeTableError(f"{option.option_id!r} carries a {option.binding.kind!r}")
        if not self.preference or not set(self.preference) <= set(ids):
            raise ScopeTableError(f"{self.axis!r} preference {self.preference} names no option")
        if self.held not in ids:
            raise ScopeTableError(f"{self.axis!r} holds {self.held!r}, which it does not offer")

    def option(self, option_id: str) -> Option | None:
        """The option `option_id`, None when the axis offers none."""
        return next((o for o in self.options if o.option_id == option_id), None)


@dataclass(frozen=True)
class Conflict:
    """Two options of different axes that cannot hold together, and why."""

    first: str
    second: str
    cause: str


@dataclass(frozen=True, kw_only=True)
class CompileFacts:
    """Everything compile reads beside the base identity and the entries: the table of
    every axis the part settles, in default-derivation order, and the conflicts."""

    axes: tuple[AxisTable, ...]
    conflicts: tuple[Conflict, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PartFacts:
    """What the part asks beside its identity, which a rule reads on both doors alike: its
    operations, the groups its roles stand for by role key (`ResolvedSpec.entity_groups`,
    or those a dataset_resubmit posts), and the datasets its identity is served from,
    empty when unknown."""

    operations: tuple[Operation, ...] = ()
    entity_groups: Mapping[str, EntityGroup] = field(default_factory=dict)
    datasets: tuple[str, ...] = ()


@dataclass(frozen=True)
class FamilyValues:
    """The value tuples a family's rows hold on some properties, in their order: `published`
    those of every row, `covered` those of the rows with a series in the coverage index."""

    published: frozenset[tuple[str | None, ...]]
    covered: frozenset[tuple[str | None, ...]]


@dataclass(frozen=True, kw_only=True)
class ScopeFacts:
    """The engine state the axis rules read: the release's code index, the request's
    coverage snapshot and the part's own facts (`PartFacts`), all in memory. Passed
    explicitly, so a rule reads no global. One `ScopeFacts` serves one build of one part's
    scope, and memoises the family scans its rules share (`family_values`) and the years a
    composite's components hold (`remembered`)."""

    index: CodeIndex
    coverage: CoverageSnapshot
    part: PartFacts = field(default_factory=PartFacts)
    _scans: dict[tuple[int, tuple[str, ...]], tuple[IndexFamily, FamilyValues]] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )
    _years: dict[str, object] = field(default_factory=dict, init=False, compare=False, repr=False)

    def remembered[T](self, key: str, read: Callable[[], T]) -> T:
        """`read()`, read once per build for `key`, a canonical document of what it reads
        (`composite.component_years` keys the years an identity's family holds on it)."""
        if key not in self._years:
            self._years[key] = read()
        return cast(T, self._years[key])

    def family_values(self, family: IndexFamily, properties: tuple[str, ...]) -> FamilyValues:
        """What `family`'s rows hold on `properties` (`FamilyValues`), read in one scan of
        its rows per build however many rules and options ask. The memo holds `family`, so
        its id names no other family while the memo lives."""
        key = (id(family), properties)
        if (held := self._scans.get(key)) is not None:
            return held[1]
        coverage = self.coverage.index
        published: set[tuple[str | None, ...]] = set()
        covered: set[tuple[str | None, ...]] = set()
        for row in family.rows:
            values = tuple(row.constraint_properties.get(p) for p in properties)
            published.add(values)
            if values not in covered and coverage.covers(row.dcid):
                covered.add(values)
        scanned = FamilyValues(published=frozenset(published), covered=frozenset(covered))
        self._scans[key] = (family, scanned)
        return scanned


def allowed_values(
    slots: Mapping[str, tuple], *, skip: str | None = None
) -> dict[str, frozenset[str]]:
    """The values each value or set binding of a family query's `slots` allows, by
    property, leaving out `skip`'s."""
    allowed: dict[str, frozenset[str]] = {}
    for prop, binding in slots.items():
        if prop == skip:
            continue
        match binding:
            case ("value", ref):
                allowed[prop] = frozenset((ref,))
            case ("set", refs):
                allowed[prop] = frozenset(refs)
    return allowed
