"""Code-index rows, declarations, stamps, and constraint-query types.

This module has no graph or file-I/O imports.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

from dfo_platform.roles import FlowAccount, RoleAxis


@dataclass(frozen=True, slots=True)
class CodeRow:
    """One classification code.

    `code_system` is the constraint-property key. The graph enum dcid is separate.
    `label` comes from `name` without its parenthetical system suffix.
    `synonyms` comes from `alternateName` without the label. `parent` is the
    same-system `specializationOf` dcid, or `None` for roots and flat systems.
    `short_name` is the graph `shortName` arc when the code has one, else the
    name the export authors, else `None`. `definition` is the graph `description`
    arc, the sentence that says what the code means, or `None` when the graph
    holds none.
    """

    dcid: str
    code_system: str
    code: str
    label: str
    synonyms: tuple[str, ...]
    parent: str | None
    short_name: str | None = None
    definition: str | None = None


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
    """Default for an open slot, triggered by a binding on another axis, a phrase, or the
    slot being open.

    `code` names a code in this system. `when_bound_under` lists dcids in
    other systems whose subtrees trigger the default. `when_named` lists
    lowercase whole words that trigger it case-insensitively. `when_open` fires it on every
    open slot, with no other trigger. A default fires only for an open slot, and the
    scoping code must belong to a different system than `code`. All other
    `apply_declared_defaults` conditions apply. Both trigger lists default to empty
    tuples where unspecified.

    `note` is the sentence an answer states whenever the default fires. A default that
    fires on an open slot carries one, since the query holds no word that names the slot;
    one without a note warns only where the query names the slot (`defaulted_slots_named`).
    """

    code: str
    when_bound_under: tuple[str, ...]
    when_named: tuple[str, ...] = ()
    when_open: bool = False
    note: str | None = None


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

    `enum_dcid` is the system's graph id, the enumeration its codes are `typeOf`
    (`CRSPurposeCodeEnum` for `crsPurpose`). Every exported declaration carries
    one; a declaration built in memory may name none.

    `no_vocabulary_terms` lists lowercase words naming a distinction of this axis that no
    code of the system expresses. A question naming one on a shape declaring the system
    reads it as a qualifier with no vocabulary, and the answer refuses it.
    """

    code_system: str
    hierarchical: bool
    disjoint_from_children: bool
    summable: bool
    absent_reading: AbsentReading
    plain_name: str
    default: DeclaredDefault | None = None
    enum_dcid: str | None = None
    no_vocabulary_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ShapeSlotDefault:
    """The value an open constraint slot takes on every shape of one `populationType`.

    A code system declares defaults for its own slots (`CodeSystemDeclaration.default`).
    This declares one for a graph property no code system of the index carries.
    """

    population_type: str
    property_dcid: str
    value_dcid: str


@dataclass(frozen=True, slots=True)
class PartlyStatedValue:
    """A constraint value a dataset states on only some of the series that hold it.

    The series that state the value and the series that leave it unsaid hold the same kind
    of thing, so the value separates no reading a question can name: the debt statistics
    state ``financingResource=DebtInstrument`` on the series that name a lender and on no
    debt total. The value is stated partly on a five-tuple where one of the dataset's
    rows states the value and another states nothing on the property
    (`CodeIndex.partly_stated_on`).
    """

    provenance: str
    property_dcid: str
    value_dcid: str


@dataclass(frozen=True, slots=True)
class MeasureClassDeclaration:
    """One pattern naming a measure class (prevalence, incidence, mortality) in a phrase or
    label.

    `pattern` searches lowercase text. A `fallback` pattern names its class only where no
    other pattern names any class: "people living with HIV" is a prevalence, while "deaths
    among people living with HIV" is a mortality.
    """

    measure_class: str
    pattern: re.Pattern[str]
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class StatTypeWordDeclaration:
    """One pattern naming a statType other than `measuredValue` in a variable phrase, as
    "growth" names `growthRate`. `pattern` searches lowercase text."""

    pattern: re.Pattern[str]
    stat_type: str


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


def level_axes(declarations: Mapping[str, CodeSystemDeclaration]) -> frozenset[str]:
    """Return declared code systems that are summable, hierarchical, and disjoint from children.

    These are the axes a datasets `groupby` can roll up to one depth of their parent chain
    (`groupby.levels`), since only a hierarchy has depths, only a summable axis can be summed
    into its ancestors, and only a parent-coded record disjoint from its children's can be
    added to them without counting them twice.
    """
    return frozenset(
        code_system
        for code_system, declaration in declarations.items()
        if declaration.summable and declaration.hierarchical and declaration.disjoint_from_children
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
class StatedWord:
    """One word a table's rows state (`DatasetDeclaration.stated_words`).

    `word` is one lowercase word and `measured_property` the measured-property dcid of
    the rows that state it, one the table has a row for. `where` maps a flow axis to the
    entity kind (`entity_resolution.types.Entity.kind`, such as "country") the row's
    party on that axis must have; empty, every row with the measured property states the
    word.
    """

    word: str
    measured_property: str
    where: Mapping[RoleAxis, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PartyWord:
    """One word that names a party of a table's rows (`DatasetDeclaration.party_words`).

    `word` is one lowercase word, `measured_property` the measured-property dcid of the
    rows it selects, one the table has a row for, `axis` the flow axis the party stands
    on, and `party` the dcid of a `PartyRow` that lists the dataset and the axis. The word
    selects the rows whose party on `axis` is `party`. `where` maps a flow axis to the
    entity kind the row's party on that axis must have, as `StatedWord.where` does; empty,
    every row with the measured property is selected.
    """

    word: str
    measured_property: str
    axis: RoleAxis
    party: str
    where: Mapping[RoleAxis, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetDeclaration:
    """One published table, keyed by the provenance its observations carry.

    `label` is the name a sentence uses for the dataset. Its words must describe
    every row the table serves: a phrase the bind model leaves unplaced reads as
    applied when each of its words occurs in the answering table's label
    (`search.engine.region.phase_rank_bind._names_answering_dataset`), so a label
    word true of only some rows would drop that qualifier with no refusal.
    `description` says in one plain sentence what the table measures; the bind
    menu's source slot shows it beside the label (`search.engine.bind`). `opt_in`
    marks a table a question answers from only when it names the table, or on a
    five-tuple no other table publishes (the imputed multilateral sectors table).
    Any other table (CRS) answers any question its five-tuples fit. Several
    datasets may publish one five-tuple, and one StatVar row may belong to several.
    `rank` orders them: of the datasets that can serve a family, the lowest
    rank serves it (`code_index.members.elect_dataset`). Ranks are unique
    across an index's declarations.

    `absent_observation` says whether an observation the table carries no row
    for reads as zero. This per-dataset reporting fact differs from
    `CodeSystemDeclaration.absent_reading`, which governs an undeclared
    constraint property on a StatVar row.

    `account` says whose account of a flow the table gives
    (`dfo_platform.roles.FlowAccount`). It names the two parties of a
    role-bearing series and says which one leads a heading.

    A StatVar row belongs to the dataset when `StatVarRow.provenances` names
    its `provenance`.

    `invariants` maps a constraint property to the code every row the dataset
    publishes carries explicitly. The loader refuses an index with a row of the
    dataset that does not (`code_index.loader._check_datasets`), whatever the
    property's `absent_reading`. A canonical identity that names the dataset binds
    each invariant (`code_index.members.invariant_bindings`), so an invariant is a
    contract: dropping one changes what stored identities mean, and a table whose
    invariant stops holding takes a new provenance.

    `stated_words` lists the words the table's rows state (`StatedWord`): a word every
    row with the measured property states, and where `StatedWord.where` is non-empty
    only the rows whose party on each named axis is of the named kind. Every row of the
    CRS states "bilateral", since the import leaves out core contributions to
    multilateral organisations; a DAC2a or DAC2b `amount` row states it only where its
    recipient is a country, since a flow to a multilateral organisation is a core
    contribution. A query word the table states restricts nothing on those rows, so
    search reads it there without a code. A word belongs here only when the table's
    import guarantees it of every such row.

    `party_words` lists the words that name a party of the table's rows (`PartyWord`).
    "Multilateral" names DAC2a's and DAC2b's provider `DACMultilaterals`, the DAC's
    published total of the multilateral organisations, on `amount` rows to a country. Such
    a word restricts the rows, so a query that states it and leaves the axis unnamed binds
    the party there.
    """

    provenance: str
    label: str
    description: str
    opt_in: bool
    absent_observation: AbsentObservation
    account: FlowAccount
    rank: int
    invariants: Mapping[str, str] = field(default_factory=dict)
    stated_words: tuple[StatedWord, ...] = ()
    party_words: tuple[PartyWord, ...] = ()


PartyKind = Literal["party", "private", "total", "residual"]
"""What one party of a flow is beside the other parties on its side.

A `party` is a country, a territory or an organisation. A `private` party is a private
provider of development finance, such as a foundation; it sums like a `party` and is
counted and labelled apart. A `total` holds other parties' figures: DAC2A publishes
"Africa, Total" (`DACAfrica`) and DAC1 the G7 (`GroupOfSeven`) beside the countries they
hold, so a sum across a flow role that counts a total counts its members twice. A `residual`
holds figures no other party does ("Africa, unallocated by country", `DACAfricaUnallocated`).
"""

PARTY_KINDS: frozenset[str] = frozenset(get_args(PartyKind))

WHOLE_OF_KINDS: frozenset[str] = frozenset({"party", "private", "residual"})
"""The kinds a whole total can hold (`WholeMarker.holds`). A `total` is never one of them, so
no whole total holds another."""

POPULATION_KIND: PartyKind = "party"
"""The kind every whole population holds: a side read whole holds its ordinary parties, so a
total whose markers hold it stands for a population (`CodeIndex.population_totals_by_axis`)."""


@dataclass(frozen=True, slots=True)
class WholeMarker:
    """What a total holds in one dataset: every party of the kinds in `holds` on its axis.

    `dataset` is a provenance among `PartyRow.datasets`, and `holds` a non-empty, sorted
    subset of `party`, `private` and `residual`. `years` is the first and last year,
    inclusive, in which the total holds that population in `dataset`, or `None` for every
    year the dataset publishes it. "All Recipients, Total" (`DACAllRecipients`) holds every
    `party` and `residual` recipient in DAC2A from the year its composition check starts, and
    in DAC2B it holds nothing of the kind, so it has no marker there.
    """

    dataset: str
    holds: tuple[PartyKind, ...]
    years: tuple[int, int] | None = None


def party_kind(
    types: Iterable[str], type_kinds: Mapping[str, PartyKind | None]
) -> PartyKind | None:
    """The kind of a party whose graph types are `types`, by the declared `type_kinds`.

    A type declared `None` says nothing on its own: `Place` sits beside `Country` on
    India, and `ProvisionalNode` beside `Organization` on IDA. The party takes the one kind
    its other types share. `None` when a type is undeclared, when no type carries a kind,
    or when two types carry different kinds, so a reader that needs the kind refuses rather
    than guesses.
    """
    kinds: set[PartyKind] = set()
    for type_ in types:
        if type_ not in type_kinds:
            return None
        if (kind := type_kinds[type_]) is not None:
            kinds.add(kind)
    return kinds.pop() if len(kinds) == 1 else None


@dataclass(frozen=True, slots=True)
class PartyRow:
    """One party a declared dataset publishes, as donor or recipient, and its kind.

    `totals` is the group a total holds, as search names the group: a graph place
    (`africa`, `Caribbean`) or a resolver group (`GroupOfSeven`). A total holds its group in
    one of two ways. When `sums_members` is true it equals the sum of every member of its
    group plus `residuals`, the residuals it counts beside them ("Africa, unallocated by
    country"), sorted. When `sums_members` is false it is the figure the source publishes
    for the group, which can differ from the sum of the members ("DAC countries, total",
    where members report late), and `residuals` is empty. `totals` is `None`, `residuals`
    empty and `sums_members` true for a party or a residual, and for a total no group
    matches (`DACAllRecipients`, or "lower middle-income countries", whose DAC members
    differ from the World Bank list). `label` is the party's
    graph name ("Africa, unallocated by country (DAC)"), `None` when the graph gives it none.
    `datasets` are the provenances of the declared and listed datasets that publish the
    party, and `axes` the flow axes they publish it on (`origin` as a donor, `destination` as a
    recipient), each sorted.

    `whole` is empty for every row but a whole total: a `total` that holds the whole
    population of a kind on its one axis in the datasets it carries a marker for
    (`WholeMarker`), sorted by dataset, one per dataset. "Official Donors, Total"
    (`DACOfficialDonors`) holds every `party` donor in DAC2A and DAC2B. A dataset with no
    marker reads the party as an ordinary `total`. The kind stays `total`, so a reader of
    `kind == "total"` stays correct.
    """

    dcid: str
    kind: PartyKind
    totals: str | None = None
    residuals: tuple[str, ...] = ()
    sums_members: bool = True
    label: str | None = None
    datasets: tuple[str, ...] = ()
    axes: tuple[RoleAxis, ...] = ()
    whole: tuple[WholeMarker, ...] = ()

    def marker(self, dataset: str) -> WholeMarker | None:
        """The marker for `dataset`, or `None` where the party is an ordinary total."""
        return next((marker for marker in self.whole if marker.dataset == dataset), None)


def marked_datasets(parties: Iterable[PartyRow]) -> frozenset[str]:
    """The datasets in which every one of `parties` carries a whole marker. Empty for no party."""
    sets = [frozenset(marker.dataset for marker in party.whole) for party in parties]
    return frozenset.intersection(*sets) if sets else frozenset()


@dataclass(frozen=True, slots=True)
class WholeTotalSet:
    """Whole totals read as one population under a phrase.

    `phrase` is the words that name the union ("all donors including private"), `axis` the
    flow axis it stands on, and `members` the dcids of the whole totals it joins, which
    are each marked in at least one dataset where all are marked
    (`marked_datasets`), and hold disjoint kinds there, so their sum counts no
    party twice (`code_index.loader._check_whole_totals`).
    """

    phrase: str
    axis: RoleAxis
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceCodeRow:
    """The StatVars a publisher's series code names under one provenance.

    `code` is the publisher's series code as its `sourceIndicatorCode` arcs carry it,
    upper-cased ("DT.TDS.DEGG.CD"). `dcids` are the StatVar rows carrying the code under
    `provenance`, sorted. An IDS code names the aggregate series and, where the World Bank
    publishes one, its by-creditor variant. One StatVar can carry several codes."""

    provenance: str
    code: str
    dcids: tuple[str, ...]


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
    `label` is the lowercase noun phrase a sentence names the axis by ("flow type"
    for `financialFlowType`), or `None` where the export authors none.
    """

    axis: str
    kind: MeasureAxisKind
    role: MeasureAxisRole
    prefer: tuple[str, ...]
    label: str | None = None


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

    `definition` is the graph `description` arc, or `None` when the graph holds none.
    """

    dcid: str
    axis: str
    label: str
    synonyms: tuple[str, ...]
    short_name: str | None
    matchable: bool
    same_reading_as: str | None = None
    definition: str | None = None


@dataclass(frozen=True, slots=True)
class MeasurePartitionLabel:
    """A label authored for a full measure partition, represented by its `(axis, value)` pairs.

    Per-partition labels preserve renderings that omit axes or change number,
    such as "gross disbursements" from an `amount` axis and plural "commitments".

    `named_values` are the `(constraint property, code)` pairs outside the key that the
    label's text names: "imputed multilateral ODA" names `dac2aAidType = DAC2aAidType106`
    and `crsTypeOfFlow = CRSTypeOfFlow10`. A series carrying such a value reads it in the
    label alone, so no other phrase repeats it.

    `definition` is the authored sentence that says what the measure counts, taken from the
    publisher's glossary or methodology, or `None` where the export authors none.
    """

    axis_values: tuple[tuple[str, str], ...]
    label: str
    named_values: tuple[tuple[str, str], ...] = ()
    definition: str | None = None


@dataclass(frozen=True, slots=True)
class PartitionLabelMatch:
    """A matched authored label, the input pairs its key left uncovered, in caller order,
    the code pairs its text names (`MeasurePartitionLabel.named_values`), and its authored
    definition (`MeasurePartitionLabel.definition`)."""

    label: str
    uncovered: tuple[tuple[str, str], ...]
    named_values: tuple[tuple[str, str], ...] = ()
    definition: str | None = None


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
        label=winner.label,
        uncovered=uncovered,
        named_values=winner.named_values,
        definition=winner.definition,
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

    `relation` is `"child"` or `"parent"` for a bound hierarchical system, and `"root"`
    for each other root beside a bound root. An unbound
    hierarchical system lists each root as `"root"` and its children as `"child"`. A flat
    system lists every value as `"root"`. `parent` is the value's parent dcid in a
    hierarchical system, or `None` for a root and for every value of a flat system.
    """

    dcid: str
    relation: Literal["child", "parent", "root"]
    member_count: int
    parent: str | None = None
