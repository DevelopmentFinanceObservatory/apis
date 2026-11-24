"""The inputs compile reads: one table per axis, the conflicts between options of different
axes, and the engine facts the axis rules build the tables from."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from code_index.query import CodeIndex, IndexFamily
from origin_catalogue.context import current_source_scope
from search.models import BaseIdentity, EntityGroup, Operation, Option, ScopeAxis
from search.scope.edits import AXIS_BINDING, PUBLISHED_BINDING, ScopeTableError
from series_coverage.shared import CoverageSnapshot


@dataclass(frozen=True, kw_only=True)
class AxisTable:
    """One axis of one part, as its rule reads it.

    `options` are the axis's options in declared order, each with its binding and its
    intrinsic availability (what the part's own data and constraints allow, before any
    other entry is weighed). `preference` is the order a default is derived in: the first
    option no entry blocks. `held` is the option the base identity holds. `rules` names the
    declared rule an option names when the part holds it, `held`'s among them (`rule`,
    `rule_of`). `concept` is the concept the axis was asked for, None when it has none.
    `published` says the part counts a published total, whose options may also bind the code
    of another total (`PUBLISHED_BINDING`). `conflicts` are the pairs of one of the axis's
    options and an option of another axis that cannot hold together on this part, beside the
    ones every part declares (`rules.CONFLICTS`). `whole` is the option that holds the
    axis's whole, the one an entry asking for `<axis>:whole` applies (`build.WHOLE`); None
    where the axis declares none, with `whole_cause` why. `whole_alternatives` are the options
    a client may ask for in place of the whole: beside a declared whole, the other options
    that also hold the part (DAC2a's inclusive scope beside total ODA); without one, or where
    the whole is unavailable, the options that can stand for it. `whole_base` is the identity
    the whole is compiled over where the part's identity cannot serve the whole's option:
    DAC2a's total ODA to all recipients, the whole of a composite over the whole recipient
    population, whose own identity would read total ODA over the open side and count its
    regional totals beside their members.

    Raises `ScopeTableError` for an empty table, a repeated option, an option of another
    axis or binding kind, a `preference`, `held` or `rules` naming no option, `rules`
    naming no rule for `held`, or a `whole` or `whole_alternatives` naming no option."""

    axis: ScopeAxis
    options: tuple[Option, ...]
    preference: tuple[str, ...]
    held: str
    rules: Mapping[str, str]
    concept: str | None = None
    published: bool = False
    conflicts: tuple[Conflict, ...] = ()
    whole: str | None = None
    whole_alternatives: tuple[str, ...] = ()
    whole_cause: str = "whole_undeclared"
    whole_base: BaseIdentity | None = None

    def __post_init__(self) -> None:
        ids = [option.option_id for option in self.options]
        kinds = {AXIS_BINDING[self.axis]}
        if self.published and self.axis in PUBLISHED_BINDING:
            kinds.add(PUBLISHED_BINDING[self.axis])
        if not ids or len(set(ids)) != len(ids):
            raise ScopeTableError(f"{self.axis!r} needs distinct options, got {ids}")
        for option in self.options:
            if option.option_id.partition(":")[0] != self.axis:
                raise ScopeTableError(f"{option.option_id!r} is not an option of {self.axis!r}")
            if option.binding.kind not in kinds:
                raise ScopeTableError(f"{option.option_id!r} carries a {option.binding.kind!r}")
        if not self.preference or not set(self.preference) <= set(ids):
            raise ScopeTableError(f"{self.axis!r} preference {self.preference} names no option")
        if self.held not in ids:
            raise ScopeTableError(f"{self.axis!r} holds {self.held!r}, which it does not offer")
        if not set(self.rules) <= set(ids):
            raise ScopeTableError(f"{self.axis!r} rules {sorted(self.rules)} name no option")
        if self.held not in self.rules:
            raise ScopeTableError(f"{self.axis!r} names no rule for {self.held!r}")
        if self.whole is not None and self.whole not in ids:
            raise ScopeTableError(f"{self.axis!r} whole {self.whole!r} names no option")
        if not set(self.whole_alternatives) <= set(ids):
            raise ScopeTableError(
                f"{self.axis!r} whole alternatives {self.whole_alternatives} name no option"
            )

    @property
    def rule(self) -> str:
        """The declared rule of the option the part holds."""
        return self.rules[self.held]

    def rule_of(self, option_id: str) -> str:
        """The rule the axis names when the part holds `option_id`: its entry in `rules`,
        else `rule`."""
        return self.rules.get(option_id, self.rule)

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
    or those a dataset_resubmit posts), and the datasets its identity is served from, empty
    when unknown."""

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
    explicitly, so a rule reads no global. One `ScopeFacts` serves the scope builds of one
    part over these facts, the elections of its reads included (`reading.ElectionFacts`),
    and memoises the family scans its rules share (`family_values`) and the years a
    composite's components hold (`remembered`), each keyed by what it reads."""

    index: CodeIndex
    coverage: CoverageSnapshot
    part: PartFacts = field(default_factory=PartFacts)
    _scans: dict[tuple[int, tuple[str, ...]], tuple[IndexFamily, FamilyValues]] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )
    _years: dict[tuple[tuple[str, ...] | None, str], object] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def remembered[T](self, key: str, read: Callable[[], T]) -> T:
        """`read()`, read once per build for `key`, a canonical document of what it reads
        (`composite.component_years` keys the years an identity's family holds on it), and
        the source scope bound to the caller's context
        (`origin_catalogue.current_source_scope`), which admits the coverage rows it reads.
        So a read under a part's source scope and one under none keep apart."""
        scope = current_source_scope()
        scoped = (None if scope is None else tuple(sorted(scope)), key)
        if scoped not in self._years:
            self._years[scoped] = read()
        return cast(T, self._years[scoped])

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
