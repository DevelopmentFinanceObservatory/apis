"""Code-index rows, declarations, stamps, and constraint-query types.

This module has no graph or file-I/O imports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CodeRow:
    """One classification code.

    `code_system` is the constraint-property key. The graph enum dcid is separate.
    `label` comes from `name` without its parenthetical system suffix.
    `synonyms` comes from `alternateName` without the label. `parent` is the
    same-system `specializationOf` dcid, or `None` for roots and flat systems.
    `short_name` is the graph `shortName` arc when the code has one, else the
    name the export authors, else `None`.
    """

    dcid: str
    code_system: str
    code: str
    label: str
    synonyms: tuple[str, ...]
    parent: str | None
    short_name: str | None = None


@dataclass(frozen=True, slots=True)
class StatVarRow:
    """One StatVar constrained by one or more code rows.

    `constraint_properties` holds this row's declared properties and values.
    An absent key differs from an `Open` query binding. Shape fields sit
    outside that mapping. `observation_properties` holds declared roles.
    CRS StatVars carry no `unit` arc, so this row has no unit field.

    `provenances` holds the provenance of every dataset that publishes
    observations of this StatVar, sorted. The export joins it from a series
    coverage build. A StatVar with no coverage row carries `()` and belongs to
    no dataset.
    """

    dcid: str
    population_type: str | None
    measured_property: str
    stat_type: str | None
    measurement_qualifier: str | None
    measurement_denominator: str | None
    observation_properties: tuple[str, ...]
    constraint_properties: Mapping[str, str]
    provenances: tuple[str, ...] = ()


AbsentReading = Literal["not_declared", "aggregate"]
"""How a code system's absent constraint-property cell reads.

`"not_declared"` is the ruling for `crsCooperationModality`: a StatVar row
that does not declare the property comes from a vintage when the axis was
not collected, a population distinct from any row that does declare a
value. No "unspecified" code exists in the vocabulary.

`"aggregate"` reads a row that does not declare the property as a summary
over every value the property could take, the reading that keeps a
demographic group like `{Person, Person_Male, Person_Female}` reducing to
`{Person}` alone.

Kept as a two-valued `Literal` so a reduction keyed on this field has an
opposite value to be exercised against.
"""


@dataclass(frozen=True, slots=True)
class DeclaredDefault:
    """Default for an open slot, triggered by a binding on another axis or a phrase.

    `code` names a code in this system. `when_bound_under` lists dcids in
    other systems whose subtrees trigger the default. `when_named` lists
    lowercase whole words that trigger it case-insensitively. A default fires only for an open slot, and the scoping
    code must belong to a different system than `code`. All other
    `apply_declared_defaults` conditions apply. Both triggers default to empty
    tuples where unspecified.
    """

    code: str
    when_bound_under: tuple[str, ...]
    when_named: tuple[str, ...] = ()


def invalid_when_named_entry(when_named: Sequence[str]) -> int | None:
    """Return the first index that fails the non-empty lowercase string rule.

    Return `None` when all entries pass. The index distinguishes an invalid
    `None` entry from no invalid entry.
    The loader and exporter share this check.
    """
    return next(
        (
            index
            for index, name in enumerate(when_named)
            if not isinstance(name, str) or not name or name != name.lower()
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class CodeSystemDeclaration:
    """Rules declared by one code system.

    `hierarchical` controls `specializationOf` traversal. `disjoint_from_children`
    says parent-coded records differ from child-coded records. CRS assigns
    one code per transaction per axis, so those populations are disjoint.
    `summable` says sibling values can reconstruct a parent total. `absent_reading`
    defines how missing constraints read, `plain_name` supplies the axis
    noun, and `default` defines an open-slot default or is `None`.
    """

    code_system: str
    hierarchical: bool
    disjoint_from_children: bool
    summable: bool
    absent_reading: AbsentReading
    plain_name: str
    default: DeclaredDefault | None = None


def summable_axes(declarations: Mapping[str, CodeSystemDeclaration]) -> frozenset[str]:
    """Return declared code systems with `summable=True`.

    This controls whether an axis can be summed, separate from child overlap or hierarchy. An
    undeclared axis and one declared `summable=False` are both absent from the result. The
    datasets door's `groupby` and capabilities, and the search response's `Slot.summable`, all
    read summability through this function, keyed by the declaration's `code_system` string.
    """
    return frozenset(
        code_system for code_system, declaration in declarations.items() if declaration.summable
    )


AbsentObservation = Literal["zero", "unknown"]
"""Whether an observation absent from a provenance's loaded data, for a
(roles, time, member) cell that provenance constrains, reads as zero.

`"zero"` is the register reading: a source whose loaded data carries
explicit zero rows for other cells in the same (roles, time) shape is read
as reporting comprehensively, so a cell it carries no row for at all means
nothing happened, not that reporting stopped. `"unknown"` is the refusing
default -- both for a provenance this index carries no declaration for, and
for a source whose reporting practice does not support the register
reading. Absence never defaults to zero without a positive declaration.

Kept as a two-valued `Literal`, the same reasoning as `AbsentReading`: a
reduction keyed on this field needs an opposite value to be exercised
against.
"""


@dataclass(frozen=True, slots=True)
class DatasetDeclaration:
    """One published table, keyed by the provenance its observations carry.

    `label` and `synonyms` are the names a query or a sentence uses for the
    dataset. `opt_in` marks a table a question answers from only when it
    names the table (the imputed multilateral sectors table). Any other table
    (CRS) answers any question its five-tuples fit. Each five-tuple
    the index carries belongs to one dataset (`CodeIndex.dataset_by_five_tuple`).

    `absent_observation` says whether an observation the table carries no row
    for reads as zero. This per-dataset reporting fact differs from
    `CodeSystemDeclaration.absent_reading`, which governs an undeclared
    constraint property on a StatVar row.

    A StatVar row belongs to the dataset when `StatVarRow.provenances` names
    its `provenance`.
    """

    provenance: str
    label: str
    synonyms: tuple[str, ...]
    opt_in: bool
    absent_observation: AbsentObservation


@dataclass(frozen=True, slots=True)
class IndexStamp:
    """Export provenance and counts. Both count mappings use `code_system` keys.
    `statvar_shape_counts` counts StatVar rows. `code_counts` counts codes.
    `content_hash`
    gives the same stamp for identical exported rows.

    `coverage_build` is the build id of the series coverage build the export
    joined `StatVarRow.provenances` from, and `statvars_without_coverage`
    counts the rows that build holds no series for. An index built in memory
    names no build.
    """

    graph_endpoint: str
    exported_at: str
    code_counts: Mapping[str, int]
    statvar_shape_counts: Mapping[str, int]
    content_hash: str
    coverage_build: str | None = None
    statvars_without_coverage: int = 0


@dataclass(frozen=True, slots=True)
class Bound:
    """The queried property must be declared on the row, with exactly this
    value."""

    value: str


@dataclass(frozen=True, slots=True)
class BoundSet:
    """The queried property must be declared on the row, with a value in
    this set."""

    values: frozenset[str]


@dataclass(frozen=True, slots=True)
class Open:
    """Match rows regardless of whether they declare this property or which value they hold.

    An "any modality" query must preserve rows without
    `crsCooperationModality`, including pre-2010 observations. Expanding the
    scope to known codes would drop those rows.
    """


PropertyBinding = Bound | BoundSet | Open
ConstraintQuery = Mapping[str, PropertyBinding]
"""A resolved constraint tuple, keyed by constraint-property name (the same
key as `StatVarRow.constraint_properties` and `CodeSystemDeclaration.code_system`).
A property named in the query as `Bound`/`BoundSet` narrows the match; a
property named as `Open`, or simply absent from this mapping entirely, is
not examined, so `CodeIndex.filter` accepts a query that names only some of
the properties a row might declare.
"""


MeasureAxisKind = Literal["shape", "constraint"]
"""Which space a declared measure axis's value lives in. `"shape"` names a
field on `StatVarRow` itself (`population_type`, `measured_property`,
`measurement_qualifier`); `"constraint"` names a key in
`StatVarRow.constraint_properties`. A measure spans both spaces, which is
why the kind is recorded per axis. `CodeSystemDeclaration` answers a
different question, what one code system's own structure permits."""

MeasureAxisRole = Literal["basis", "free"]
"""Whether a sibling of an elected measure partition must agree with it on
this axis (`"basis"`) or may differ from it (`"free"`)."""


@dataclass(frozen=True, slots=True)
class MeasureAxis:
    """One measure-family dimension. `axis` names a shape field or constraint property.
    `prefer` orders its values, with axis declarations composed
    lexicographically to choose the default partition when all axes are open.
    """

    axis: str
    kind: MeasureAxisKind
    role: MeasureAxisRole
    prefer: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeasureValueRow:
    """One measure-axis value, shaped like `CodeRow` for qualifier matching but
    separate from code-system members. `label`, `synonyms`, and `short_name`
    come from the graph's `name`, `alternateName`, and `shortName` arcs.
    `matchable` says whether query text can identify the value. The axis role
    separately determines what a match steers.

    `same_reading_as` names the canonical value on the same axis when the graph
    spells one reading with two dcids (`InflationAdjusted` reads as `RealValue`),
    or is `None` for a canonical value. Query text naming an alias resolves to
    its canonical value, and an alias ranks inside its canonical value's reading.
    """

    dcid: str
    axis: str
    label: str
    synonyms: tuple[str, ...]
    short_name: str | None
    matchable: bool
    same_reading_as: str | None = None


@dataclass(frozen=True, slots=True)
class MeasurePartitionLabel:
    """A label authored for a full measure partition, represented by its `(axis, value)` pairs.

    Per-partition labels preserve renderings that omit axes or change number,
    such as "gross disbursements" from an `amount` axis and plural "commitments".

    `named_values` are the `(constraint property, code)` pairs outside the key that the
    label's text names: "imputed multilateral ODA" names `dac2aAidType = DAC2aAidType106`
    and `crsTypeOfFlow = CRSTypeOfFlow10`. A series carrying such a value reads it in the
    label alone, so no other phrase repeats it.
    """

    axis_values: tuple[tuple[str, str], ...]
    label: str
    named_values: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PartitionLabelMatch:
    """A matched authored label, the input pairs its key left uncovered, in caller order,
    and the code pairs its text names (`MeasurePartitionLabel.named_values`)."""

    label: str
    uncovered: tuple[tuple[str, str], ...]
    named_values: tuple[tuple[str, str], ...] = ()


def label_for_pairs(
    pairs: Sequence[tuple[str, str | None]], labels: Sequence[MeasurePartitionLabel]
) -> PartitionLabelMatch | None:
    """Match the most specific authored label whose pairs are contained in `pairs`.

    `None` values in `pairs` are ignored. A tie between equally specific
    labels raises `ValueError`. No match returns `None`.
    """
    known = tuple((axis, value) for axis, value in pairs if value is not None)
    known_set = frozenset(known)

    winner: MeasurePartitionLabel | None = None
    winner_size = -1
    tied = False
    for candidate in labels:
        candidate_pairs = frozenset(candidate.axis_values)
        if not candidate_pairs <= known_set:
            continue
        size = len(candidate_pairs)
        if size > winner_size:
            winner, winner_size, tied = candidate, size, False
        elif size == winner_size:
            tied = True

    if winner is None:
        return None
    if tied:
        raise ValueError(
            f"two authored partition labels match {pairs!r} with the same number of "
            "axis_values -- the authored table is ambiguous"
        )

    covered = frozenset(winner.axis_values)
    uncovered = tuple(pair for pair in known if pair not in covered)
    return PartitionLabelMatch(
        label=winner.label, uncovered=uncovered, named_values=winner.named_values
    )


@dataclass(frozen=True, slots=True)
class MeasureIndex:
    """The measure-axis declarations, their values, their partition labels, and
    the measure nouns parsed from one export -- everything `code_index.loader.
    load_measure_axes` reads beyond the `CodeIndex` itself. `axes` is empty
    for a deployment that declares no measure family at all, which is a
    supported state.

    `nouns` name measures outside the measure family (IDS and WEO series), each keyed
    on five-tuple and constraint pairs (`populationType`, `measuredProperty`,
    `statType`, or a constraint property) and matched with `label_for_pairs`.
    """

    axes: tuple[MeasureAxis, ...]
    values: tuple[MeasureValueRow, ...]
    partition_labels: tuple[MeasurePartitionLabel, ...]
    nouns: tuple[MeasurePartitionLabel, ...] = ()

    def canonical_reading(self, dcid: str) -> str:
        """The canonical value of `dcid`'s reading: its `same_reading_as`, else `dcid` itself."""
        for row in self.values:
            if row.dcid == dcid and row.same_reading_as is not None:
                return row.same_reading_as
        return dcid

    def reading_set(self, dcid: str) -> frozenset[str]:
        """Every value spelling `dcid`'s reading: the canonical value and its aliases."""
        canonical = self.canonical_reading(dcid)
        return frozenset(
            {canonical} | {row.dcid for row in self.values if row.same_reading_as == canonical}
        )

    def reading_order(self, axis: str) -> tuple[str, ...]:
        """`axis`'s declared `prefer` with each value's aliases ranked directly after it.

        With `prefer` = (Nominal, RealValue, PurchasingPowerParity) and `InflationAdjusted`
        an alias of `RealValue`, the order is Nominal, RealValue, InflationAdjusted,
        PurchasingPowerParity. An undeclared axis has no order.
        """
        prefer = next((declared.prefer for declared in self.axes if declared.axis == axis), ())
        return tuple(
            value
            for canonical in prefer
            for value in (
                canonical,
                *sorted(row.dcid for row in self.values if row.same_reading_as == canonical),
            )
        )


@dataclass(frozen=True, slots=True)
class LevelOption:
    """A menu value, its relation to the bound node, and its directly coded StatVar count.

    `relation` is `"child"` or `"parent"` for a bound hierarchical system. An unbound
    hierarchical system lists each root as `"root"` and its children as `"child"`. A flat
    system lists every value as `"root"`. `parent` is the value's parent dcid in a
    hierarchical system, or `None` for a root and for every value of a flat system.
    """

    dcid: str
    relation: Literal["child", "parent", "root"]
    member_count: int
    parent: str | None = None
