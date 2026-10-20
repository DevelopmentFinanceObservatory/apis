"""One summary row per five-tuple family of the variables a build holds.

Search groups recalled variables by five-tuple and role class
(`search.engine.retrieval.derive`). A group whose membership the code index
does not decide reads its slot keys, slot taxonomy and label from every member
of its family the catalogue holds, and its members are that family's, where it would
otherwise read all of them from the recalled sample alone. The catalogue job computes that
summary once per family (`family_tables`) from the variables of the `dfo` and `base` arc
tables and, apart, from the admitted UN variables, so serving reads one row
(`FamilyTables.family`) and reads a member's arcs from the arc or UN tables.

A variable's family key is its five-tuple as search reads it from the
variable's `->*` arcs (`search.engine.retrieval.facts.read_five_tuple`) and
whether it declares a role (`derive.member_group_class`): a property's first
literal, else its first node; a missing population type, measured property or
stat type reads as the empty string; a `measurementQualifier` whose every
target is typed only `ProvisionalNode` reads as none; a `measurementDenominator`
with targets but no value, or whose every target is a `ProvisionalNode`, is
unresolved. A variable declares a role when its `observationProperties` arc
names any node (`materialise.roles.declared_roles` reads an arc naming only
`observationAbout` as degenerate, which search also files as role-bearing). A variable's
constraints are its `constraintProperties` targets less the `import_added` ones, which no
served read shows. A constraint counts as declared whether or not the variable's arcs
resolve its value, so a member with an unresolved constraint is never the family total.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, NamedTuple

from origin_catalogue.un_mirror import UnArc, UnMirror, import_added

_PROVISIONAL_TYPES: Final[tuple[str, ...]] = ("ProvisionalNode",)


class FamilyKey(NamedTuple):
    """A five-tuple family and role class, in the field order of search's `FiveTuple`
    followed by the role class, so `FamilyKey(*five_tuple, role_bearing)` builds one."""

    population_type: str
    measured_property: str
    stat_type: str
    measurement_qualifier: str | None
    measurement_denominator: str | None
    denominator_unresolved: bool
    role_bearing: bool


@dataclass(frozen=True, slots=True)
class FamilySummary:
    """What a shape reads from the members of one family a build holds.

    `members` names them in table order. `values` maps each constraint property some member
    asserts to the distinct values the members assert, properties and values in first-seen
    order over the members in table order. `shared` maps the properties every member asserts
    at one value to that value. `total` is the first member declaring no constraint that has
    a name, the family total, and `total_name` its name (the last of its names), both None
    when no such member exists. `un_only` says every member is a UN variable."""

    members: tuple[str, ...]
    values: Mapping[str, tuple[str, ...]]
    shared: Mapping[str, str]
    total: str | None
    total_name: str | None
    un_only: bool


@dataclass(frozen=True, slots=True)
class FamilyTables:
    """A build's family summaries: `arcs` over the variables of the `dfo` and `base` arc
    tables and `un` over the admitted UN variables, each keyed by `FamilyKey`."""

    arcs: Mapping[FamilyKey, FamilySummary] = field(default_factory=dict)
    un: Mapping[FamilyKey, FamilySummary] = field(default_factory=dict)

    def family(
        self,
        key: FamilyKey,
        *,
        un_system_enabled: bool,
        release: ReleaseFamilies | None = None,
    ) -> FamilySummary | None:
        """The summary of every member of `key` the build holds, the arc tables' first,
        leaving the UN variables out with the UN origin off. With `release`, the arc tables'
        part is `release`'s in place of the build's own. None when they hold none."""
        held = self.arcs.get(key) if release is None else release.family(key)
        return _joined(held, self.un.get(key) if un_system_enabled else None)


EMPTY_FAMILY_TABLES: Final = FamilyTables()
"""The family tables of a build holding no variable."""


@dataclass(frozen=True, slots=True)
class ReleaseFamilies:
    """The arc tables' part of each family with a graph release answering the DFO half:
    `dfo`, the release's summaries of its DFO variables
    (`graph_release.origin.ReleaseOrigin.families`), joined to `base`, a build's summaries of
    its `base` arc table alone (`summarise_table`). `family_tables` joins the build's own
    DFO and base summaries the same way, so for one import the two agree.

    `base_without_dfo` answers in place of `base` for each family a base variable the
    release holds as DFO belongs to (`base_without`), so that variable is a member once,
    with the release's arcs. None there means no other base variable is left."""

    dfo: Mapping[FamilyKey, FamilySummary]
    base: Mapping[FamilyKey, FamilySummary]
    base_without_dfo: Mapping[FamilyKey, FamilySummary | None] = field(default_factory=dict)

    def family(self, key: FamilyKey) -> FamilySummary | None:
        if key in self.base_without_dfo:
            return _joined(self.dfo.get(key), self.base_without_dfo[key])
        return _joined(self.dfo.get(key), self.base.get(key))


def base_without(
    variables: Mapping[str, tuple[UnArc, ...]],
    summaries: Mapping[FamilyKey, FamilySummary],
    *,
    dropped: Iterable[str],
) -> dict[FamilyKey, FamilySummary | None]:
    """The summary of each family of `summaries`, `summarise_table(variables)`, that a
    `dropped` variable belongs to, summarised again from its other members in table order,
    or None when it has none. Families no dropped variable belongs to are left out."""
    gone = {dcid for dcid in dropped if variables.get(dcid)}
    keys = {_VariableFacts.read(variables[dcid]).key for dcid in gone}
    resummarised: dict[FamilyKey, FamilySummary | None] = {}
    for key in keys:
        kept = [dcid for dcid in summaries[key].members if dcid not in gone]
        resummarised[key] = summarise(
            ((dcid, variables[dcid]) for dcid in kept), un_only=False
        ).get(key)
    return resummarised


def merge_summaries(first: FamilySummary, second: FamilySummary) -> FamilySummary:
    """The summary of `first`'s members followed by `second`'s."""
    values = {prop: list(found) for prop, found in first.values.items()}
    for prop, found in second.values.items():
        seen = values.setdefault(prop, [])
        seen.extend(value for value in found if value not in seen)
    return FamilySummary(
        members=first.members + second.members,
        values={prop: tuple(found) for prop, found in values.items()},
        shared={
            prop: value for prop, value in first.shared.items() if second.shared.get(prop) == value
        },
        total=first.total if first.total is not None else second.total,
        total_name=first.total_name if first.total is not None else second.total_name,
        un_only=first.un_only and second.un_only,
    )


def _joined(first: FamilySummary | None, second: FamilySummary | None) -> FamilySummary | None:
    if first is None or second is None:
        return first or second
    return merge_summaries(first, second)


def family_tables(
    *,
    dfo: Mapping[str, tuple[UnArc, ...]],
    base: Mapping[str, tuple[UnArc, ...]],
    un: UnMirror,
) -> FamilyTables:
    """The summaries of the `dfo` and `base` variables, each table in dcid order, and of
    `un`'s admitted variables in dcid order. A variable held empty has no family. A family's
    `arcs` summary is its `dfo` summary joined to its `base` one, which is the summary of
    both tables read in turn, since a summary's members, values and total keep the order
    its members are read in."""
    return FamilyTables(
        arcs=_merge_families(summarise_table(dfo), summarise_table(base)),
        un=summarise(
            ((dcid, un.variables[dcid].arcs) for dcid in sorted(un.variables)), un_only=True
        ),
    )


def _merge_families(
    first: Mapping[FamilyKey, FamilySummary], second: Mapping[FamilyKey, FamilySummary]
) -> dict[FamilyKey, FamilySummary]:
    """Each family of `first` or `second`, its `first` summary joined to its `second` one,
    keyed in the order `first` then `second` name them."""
    merged = dict(first)
    for key, summary in second.items():
        found = merged.get(key)
        merged[key] = summary if found is None else merge_summaries(found, summary)
    return merged


def summarise_table(variables: Mapping[str, tuple[UnArc, ...]]) -> dict[FamilyKey, FamilySummary]:
    """`summarise` over an arc table's variables in dcid order."""
    return summarise(((dcid, variables[dcid]) for dcid in sorted(variables)), un_only=False)


def summarise(
    variables: Iterable[tuple[str, tuple[UnArc, ...]]], *, un_only: bool
) -> dict[FamilyKey, FamilySummary]:
    """One summary per family key of `variables`, pairs of a dcid and its arc rows, read
    in the order given. A variable with no rows is left out."""
    builders: dict[FamilyKey, _Builder] = {}
    for dcid, arcs in variables:
        if not arcs:
            continue
        facts = _VariableFacts.read(arcs)
        builder = builders.get(facts.key)
        if builder is None:
            builder = builders[facts.key] = _Builder()
        builder.add(dcid, facts)
    return {key: builder.summary(un_only=un_only) for key, builder in builders.items()}


@dataclass(frozen=True, slots=True)
class _VariableFacts:
    """What a family summary reads from one variable's arcs: `constrained` says it declares
    a constraint, and `constraints` maps each declared one whose value its arcs resolve to
    that value."""

    key: FamilyKey
    constrained: bool
    constraints: Mapping[str, str]
    name: str | None

    @staticmethod
    def read(arcs: tuple[UnArc, ...]) -> _VariableFacts:
        by_property: dict[str, list[UnArc]] = {}
        for arc in arcs:
            by_property.setdefault(arc.property, []).append(arc)

        def target(property: str) -> UnArc | None:
            rows = by_property.get(property, ())
            return next((arc for arc in rows if arc.value is not None), None) or next(
                (arc for arc in rows if arc.dcid is not None), None
            )

        def value(property: str) -> str | None:
            found = target(property)
            if found is None:
                return None
            return found.value if found.value is not None else found.dcid

        def provisional(property: str) -> bool:
            rows = by_property.get(property, ())
            return bool(rows) and all(arc.types == _PROVISIONAL_TYPES for arc in rows)

        denominator = value("measurementDenominator")
        key = FamilyKey(
            population_type=value("populationType") or "",
            measured_property=value("measuredProperty") or "",
            stat_type=value("statType") or "",
            measurement_qualifier=(
                None if provisional("measurementQualifier") else value("measurementQualifier")
            ),
            measurement_denominator=denominator,
            denominator_unresolved=(
                bool(by_property.get("measurementDenominator")) and denominator is None
            )
            or provisional("measurementDenominator"),
            role_bearing=any(
                arc.dcid is not None for arc in by_property.get("observationProperties", ())
            ),
        )
        declared = [
            arc.dcid
            for arc in by_property.get("constraintProperties", ())
            if arc.dcid is not None and not import_added(arc)
        ]
        constraints: dict[str, str] = {}
        for prop in declared:
            found = value(prop)
            if found is not None:
                constraints[prop] = found
        names = [arc.value for arc in by_property.get("name", ()) if arc.value]
        return _VariableFacts(
            key=key,
            constrained=bool(declared),
            constraints=constraints,
            name=names[-1] if names else None,
        )


@dataclass(slots=True)
class _Builder:
    """One family's summary as its members are read."""

    members: list[str] = field(default_factory=list)
    values: dict[str, dict[str, None]] = field(default_factory=dict)
    asserting: dict[str, int] = field(default_factory=dict)
    total: str | None = None
    total_name: str | None = None

    def add(self, dcid: str, facts: _VariableFacts) -> None:
        self.members.append(dcid)
        for prop, value in facts.constraints.items():
            self.values.setdefault(prop, {})[value] = None
            self.asserting[prop] = self.asserting.get(prop, 0) + 1
        if self.total is None and not facts.constrained and facts.name is not None:
            self.total, self.total_name = dcid, facts.name

    def summary(self, *, un_only: bool) -> FamilySummary:
        return FamilySummary(
            members=tuple(self.members),
            values={prop: tuple(found) for prop, found in self.values.items()},
            shared={
                prop: next(iter(found))
                for prop, found in self.values.items()
                if len(found) == 1 and self.asserting[prop] == len(self.members)
            },
            total=self.total,
            total_name=self.total_name,
            un_only=un_only,
        )


__all__ = [
    "EMPTY_FAMILY_TABLES",
    "FamilyKey",
    "FamilySummary",
    "FamilyTables",
    "ReleaseFamilies",
    "base_without",
    "family_tables",
    "merge_summaries",
    "summarise",
    "summarise_table",
]
