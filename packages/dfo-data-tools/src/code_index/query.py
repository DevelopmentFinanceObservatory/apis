"""`CodeIndex`: the in-memory, queryable catalogue built from parsed rows and
declarations. No graph access and no file I/O here -- `loader.parse_index`
constructs one of these from a code index document.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

from code_index.tokens import Token, TokenLookup, word_tokens
from code_index.types import (
    POPULATION_CLASS,
    AbsentObservation,
    Bound,
    BoundSet,
    BroadReading,
    CodeRow,
    CodeSystemDeclaration,
    ConstraintQuery,
    DatasetDeclaration,
    HouseDefaultCase,
    ImpliedValue,
    IndexStamp,
    LevelOption,
    MeasureClassDeclaration,
    PartlyStatedValue,
    PartyClass,
    PartyKind,
    PartyRow,
    ShapeSlotDefault,
    SourceCodeRow,
    StatTypeWordDeclaration,
    StatVarRow,
    WholeTotalSet,
    party_kind,
)
from dfo_platform.roles import RoleAxis
from statvar_members import FiveTuple

_NO_IMPLIED: Mapping[str, str] = MappingProxyType({})


class DuplicateDcid(ValueError):
    """Two rows -- code or StatVar -- in the same index share one dcid, or
    two code rows in the same code system share one code string."""


class CyclicParent(ValueError):
    """A code row's `parent` chain, or a `value_parents` chain, revisits a
    dcid already in that chain, as a self-parent or a longer cycle does.
    `descendants`'s breadth-first walk and `value_ancestors` follow these
    relations, so a cycle here would otherwise grow their result without
    bound."""


class CrossSystemParent(ValueError):
    """A code row's `parent` names a row in a different `code_system`.
    `specializationOf` orders one code system's own codes, so a parent from
    another system would walk `descendants`/`covering_roots` outside the
    vocabulary and roll that other system's members up as this one's
    siblings."""


class InvalidDeclaredDefault(ValueError):
    """A default code belongs outside its declaring system, or a `when_bound_under` code is
    missing or belongs to that system. Defaults scope over another system's slot."""


def row_five_tuple(row: StatVarRow) -> FiveTuple:
    """Returns the row's five fields as a resolved `FiveTuple`.

    Exported rows set `denominator_unresolved=False`."""
    return FiveTuple(
        population_type=row.population_type,
        measured_property=row.measured_property,
        stat_type=row.stat_type,
        measurement_qualifier=row.measurement_qualifier,
        measurement_denominator=row.measurement_denominator,
    )


def _datasets_by_five_tuple(
    statvars: Iterable[StatVarRow], datasets: Mapping[str, DatasetDeclaration]
) -> dict[FiveTuple, tuple[str, ...]]:
    """Maps each five-tuple to the datasets publishing any of its rows, lowest
    `DatasetDeclaration.rank` first. A provenance no dataset declares follows the declared
    ones, and provenance breaks a tie. A row with no provenance belongs to no dataset."""
    publishers: dict[FiveTuple, set[str]] = defaultdict(set)
    for row in statvars:
        publishers[row_five_tuple(row)].update(row.provenances)

    def serving_order(provenance: str) -> tuple[bool, int, str]:
        declaration = datasets.get(provenance)
        return (declaration is None, declaration.rank if declaration else 0, provenance)

    return {
        five_tuple: tuple(sorted(provenances, key=serving_order))
        for five_tuple, provenances in publishers.items()
        if provenances
    }


def partly_stated_five_tuples(
    statvars: Iterable[StatVarRow], declared: PartlyStatedValue
) -> frozenset[FiveTuple]:
    """The five-tuples `declared` is stated partly on: those where a row of its dataset states
    the property with the declared value and another row of the dataset states nothing on the
    property. A five-tuple whose dataset rows all state the property, or none state the value,
    is not among them. The loader refuses a declaration with no such five-tuple."""
    stating: set[FiveTuple] = set()
    silent: set[FiveTuple] = set()
    for row in statvars:
        if declared.provenance not in row.provenances:
            continue
        stated = row.constraint_properties.get(declared.property_dcid)
        if stated == declared.value_dcid:
            stating.add(row_five_tuple(row))
        elif stated is None:
            silent.add(row_five_tuple(row))
    return frozenset(stating & silent)


def _partly_stated_by_five_tuple(
    statvars: Iterable[StatVarRow], declared: Iterable[PartlyStatedValue]
) -> dict[FiveTuple, dict[str, frozenset[tuple[str, str]]]]:
    """Maps each five-tuple to the declared (property, value) pairs each dataset states
    partly on it (`partly_stated_five_tuples`)."""
    statvars = tuple(statvars)
    partly: dict[FiveTuple, dict[str, set[tuple[str, str]]]] = defaultdict(lambda: defaultdict(set))
    for value in declared:
        for five_tuple in partly_stated_five_tuples(statvars, value):
            partly[five_tuple][value.provenance].add((value.property_dcid, value.value_dcid))
    return {
        five_tuple: {provenance: frozenset(pairs) for provenance, pairs in by_provenance.items()}
        for five_tuple, by_provenance in partly.items()
    }


@dataclass(frozen=True, slots=True)
class IndexFamily:
    """The rows of one five-tuple that one property set describes, and the datasets
    publishing them.

    The datasets publishing a five-tuple form one family when their rows there carry the same
    constraint properties (`properties`, the union over each dataset's rows), and separate
    families when they do not: the CRS and DAC1 both publish ODA disbursements, under
    `crs*` and `dac1AidType` codes, and so are two families. A family is what an election
    elects among (`code_index.members.elect_dataset`), what a Search shape is built from,
    and what an identity's named dataset or slot keys select
    (`code_index.members.query_family`). A row no dataset publishes joins the one family
    whose properties hold its own, else a family of its own with no dataset."""

    five_tuple: FiveTuple
    properties: frozenset[str]
    """The constraint properties the family's rows carry."""
    datasets: tuple[str, ...]
    """The datasets publishing the family's rows, in serving order
    (`DatasetDeclaration.rank`)."""
    rows: tuple[StatVarRow, ...]
    """The family's rows, in index order."""
    key_sets: frozenset[frozenset[str]]
    """The distinct constraint-property sets of `rows`."""

    def covers(self, keys: AbstractSet[str]) -> bool:
        """Whether `keys` holds every constraint property of one of the family's rows."""
        return any(row_keys <= keys for row_keys in self.key_sets)


def _families_by_five_tuple(
    statvars: Iterable[StatVarRow], datasets: Mapping[str, DatasetDeclaration]
) -> dict[FiveTuple, tuple[IndexFamily, ...]]:
    """Groups each five-tuple's rows into its families (`IndexFamily`), in the serving order
    of each family's first dataset, a family with no dataset last."""
    rows_by_five_tuple: dict[FiveTuple, list[StatVarRow]] = defaultdict(list)
    for row in statvars:
        rows_by_five_tuple[row_five_tuple(row)].append(row)

    def serving_order(provenance: str) -> tuple[bool, int, str]:
        declaration = datasets.get(provenance)
        return (declaration is None, declaration.rank if declaration else 0, provenance)

    families: dict[FiveTuple, tuple[IndexFamily, ...]] = {}
    for five_tuple, rows in rows_by_five_tuple.items():
        carried: dict[str, set[str]] = defaultdict(set)
        unpublished: set[str] = set()
        for row in rows:
            for provenance in row.provenances:
                carried[provenance].update(row.constraint_properties)
            if not row.provenances:
                unpublished.update(row.constraint_properties)
        groups: dict[frozenset[str], list[str]] = defaultdict(list)
        for provenance, properties in carried.items():
            groups[frozenset(properties)].append(provenance)
        # Rows no dataset publishes join the one family whose properties hold theirs.
        holding = [properties for properties in groups if unpublished <= properties]
        home = holding[0] if len(holding) == 1 else frozenset(unpublished)
        if any(not row.provenances for row in rows):
            groups.setdefault(home, [])
        built: list[IndexFamily] = []
        for properties, provenances in groups.items():
            members = tuple(
                row
                for row in rows
                if set(row.provenances) & set(provenances)
                or (not row.provenances and properties == home)
            )
            built.append(
                IndexFamily(
                    five_tuple=five_tuple,
                    properties=properties,
                    datasets=tuple(sorted(provenances, key=serving_order)),
                    rows=members,
                    key_sets=frozenset(frozenset(row.constraint_properties) for row in members),
                )
            )
        built.sort(
            key=lambda family: (
                serving_order(family.datasets[0]) if family.datasets else (True, 0, "~")
            )
        )
        families[five_tuple] = tuple(built)
    return families


def _dataset_pins(
    statvars: Iterable[StatVarRow], code_systems: Iterable[str]
) -> dict[tuple[str, FiveTuple], dict[str, str]]:
    """Maps each dataset and five-tuple it publishes to the code-system properties that
    every one of the dataset's rows of the five-tuple carries with one value, and that
    value. Measure axes are not declared code systems, so no measure axis is pinned."""
    rows_by_key: dict[tuple[str, FiveTuple], list[StatVarRow]] = defaultdict(list)
    for row in statvars:
        five_tuple = row_five_tuple(row)
        for provenance in row.provenances:
            rows_by_key[provenance, five_tuple].append(row)
    pins: dict[tuple[str, FiveTuple], dict[str, str]] = {}
    for key, rows in rows_by_key.items():
        pinned: dict[str, str] = {}
        for code_system in sorted(code_systems):
            held = {row.constraint_properties.get(code_system) for row in rows}
            if len(held) == 1 and (value := held.pop()) is not None:
                pinned[code_system] = value
        pins[key] = pinned
    return pins


def _enrichment_digest(parties: Iterable[PartyRow]) -> str:
    """The hex SHA-256 of `parties`' `(dcid, kind, party_class, whole markers, dataset kinds)`
    rows, sorted by dcid, each marker as `(dataset, covers, years)` and each dataset kind as
    `(dataset, kind)`. Neither row order nor a label, group or dataset-list change moves it.
    The markers are in it, windows included, because a whole total's frame cell depends on the
    row's dataset and on whether its year falls inside the window, and the dataset kinds
    because a party's kind and class cells depend on the row's dataset. A row without dataset
    kinds hashes as it did before they existed."""
    rows = sorted(
        (
            party.dcid,
            party.kind,
            party.party_class,
            [(marker.dataset, sorted(marker.covers), marker.years) for marker in party.whole],
            *([list(party.dataset_kinds)] if party.dataset_kinds else []),
        )
        for party in parties
    )
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _check_consistent_parents(codes_by_dcid: dict[str, CodeRow]) -> None:
    """Raises `CrossSystemParent` for a parent in another code system.

    Missing parent rows are left for `_root_of` to treat as chain roots."""
    for row in codes_by_dcid.values():
        if row.parent is None:
            continue
        parent_row = codes_by_dcid.get(row.parent)
        if parent_row is not None and parent_row.code_system != row.code_system:
            raise CrossSystemParent(
                f"code row {row.dcid!r} in code system {row.code_system!r} has parent "
                f"{row.parent!r} in code system {parent_row.code_system!r}"
            )


def _check_acyclic_parents(parent_of: Mapping[str, str | None], *, relation: str) -> None:
    """Raises `CyclicParent` when a chain of `parent_of` revisits a dcid. Each dcid has at
    most one parent. `relation` names the relation in the error.

    Chains reaching a resolved dcid or one `parent_of` does not hold terminate."""
    resolved: set[str] = set()
    for start_dcid in parent_of:
        if start_dcid in resolved:
            continue
        chain: list[str] = []
        node: str | None = start_dcid
        while node is not None and node not in resolved:
            if node in chain:
                cycle = [*chain[chain.index(node) :], node]
                raise CyclicParent(f"{relation} relation forms a cycle: {' -> '.join(cycle)}")
            chain.append(node)
            node = parent_of.get(node)
        resolved.update(chain)


def _check_declared_defaults(
    declarations: Mapping[str, CodeSystemDeclaration], codes_by_dcid: dict[str, CodeRow]
) -> None:
    """Raises `InvalidDeclaredDefault` for a default code outside its system or an invalid
    `when_bound_under` dcid."""
    for declaration in declarations.values():
        default = declaration.default
        if default is None:
            continue
        code_row = codes_by_dcid.get(default.code)
        if code_row is None or code_row.code_system != declaration.code_system:
            raise InvalidDeclaredDefault(
                f"code system {declaration.code_system!r} declares default code "
                f"{default.code!r}, which is not a row of that code system"
            )
        for scoping_dcid in default.when_bound_under:
            scoping_row = codes_by_dcid.get(scoping_dcid)
            if scoping_row is None:
                raise InvalidDeclaredDefault(
                    f"code system {declaration.code_system!r} declares a default "
                    f"when_bound_under code {scoping_dcid!r}, which this index holds no "
                    "row for"
                )
            if scoping_row.code_system == declaration.code_system:
                raise InvalidDeclaredDefault(
                    f"code system {declaration.code_system!r} declares a default "
                    f"when_bound_under code {scoping_dcid!r} in its own code system -- a "
                    "default cannot condition on the slot it defaults"
                )


CANDIDATE_HITS = 8
"""The most lexical hits one phrase's candidate menu keeps (`CodeIndex.candidate_codes`)."""

CANDIDATE_LIMIT = 24
"""The most codes one phrase's candidate menu holds (`CodeIndex.candidate_codes`)."""


@dataclass(frozen=True, slots=True)
class CandidateCodes:
    """A phrase's candidate menu (`CodeIndex.candidate_codes`): its lexical `hits`, the
    codes sharing a word of the phrase, in rank order, and every code of the menu in menu
    order, the hits first. Both are empty when no code shares a word of the phrase."""

    hits: tuple[str, ...]
    codes: tuple[str, ...]


def fold_word(word: str) -> str:
    """`word` casefolded, less one trailing "s" when it is longer than two letters.
    Candidate retrieval applies this one plural allowance to query words and code text
    alike."""
    folded = word.casefold()
    return folded[:-1] if len(folded) > 2 and folded.endswith("s") else folded


def fold_text(text: str) -> str:
    """`text` casefolded, with its whitespace runs collapsed to one space and stripped: the
    fold `CodeIndex.codes_by_text` compares a query and a code's texts under."""
    return " ".join(text.casefold().split())


def dedupe_first_seen(groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    """Flattens resolved groups, keeping each dcid's first occurrence.

    Descendant expansion can return the same dcid for multiple input refs."""
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


class CodeIndex:
    """Read-only index. Lookup tables are built once in `__init__`."""

    __slots__ = (
        "__weakref__",
        "_children_by_parent",
        "_codes_by_dcid",
        "_codes_by_system_code",
        "_codes_by_text",
        "_codes_by_tokens",
        "_codes_by_word",
        "_implied_by_measure",
        "_members_by_source_code",
        "_partly_stated_by_five_tuple",
        "_source_codes_by_dcid",
        "_statvars_by_dcid",
        "_whole_sets_by_phrase",
        "_words_by_code",
        "broad_readings",
        "codes",
        "codes_by_system",
        "dataset_pins",
        "datasets",
        "datasets_by_five_tuple",
        "declarations",
        "enrichment_digest",
        "families_by_five_tuple",
        "family_by_dcid",
        "implied_values",
        "listed_by_five_tuple",
        "listed_statvars",
        "measure_classes",
        "parties",
        "partly_stated_values",
        "party_types",
        "population_totals_by_axis",
        "shape_slot_defaults",
        "side_classes",
        "slot_plain_names",
        "source_code_provenances",
        "source_codes",
        "stamp",
        "stat_type_words",
        "statvars",
        "totals_by_group",
        "value_parents",
        "variable_aliases",
        "variables_by_provenance",
        "whole_total_sets",
    )

    def __init__(
        self,
        *,
        stamp: IndexStamp,
        declarations: Iterable[CodeSystemDeclaration],
        codes: Iterable[CodeRow],
        statvars: Iterable[StatVarRow],
        datasets: Iterable[DatasetDeclaration] = (),
        party_types: Mapping[str, PartyKind | None] | None = None,
        parties: Iterable[PartyRow] = (),
        listed_statvars: Iterable[StatVarRow] = (),
        shape_slot_defaults: Iterable[ShapeSlotDefault] = (),
        broad_readings: Iterable[BroadReading] = (),
        partly_stated_values: Iterable[PartlyStatedValue] = (),
        implied_values: Iterable[ImpliedValue] = (),
        measure_classes: Iterable[MeasureClassDeclaration] = (),
        variable_aliases: Mapping[str, str] | None = None,
        slot_plain_names: Mapping[str, str] | None = None,
        stat_type_words: Iterable[StatTypeWordDeclaration] = (),
        source_codes: Iterable[SourceCodeRow] = (),
        value_parents: Mapping[str, str] | None = None,
        whole_total_sets: Iterable[WholeTotalSet] = (),
    ) -> None:
        self.stamp = stamp
        self.shape_slot_defaults = tuple(shape_slot_defaults)
        """The open-slot defaults declared per `populationType` for properties outside the
        code systems, in declaration order."""
        self.broad_readings = tuple(broad_readings)
        """The constraint values a question's words read as a broader value unless it names
        a value on another property (`BroadReading`), in declaration order."""
        self.partly_stated_values = tuple(partly_stated_values)
        """The constraint values a dataset states on only some of its series
        (`PartlyStatedValue`), in declaration order."""
        self.implied_values = tuple(implied_values)
        """The constraint values each measure's headline series already means
        (`ImpliedValue`), in declaration order."""
        implied_by_measure: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for implied in self.implied_values:
            implied_by_measure[(implied.population_type, implied.measured_property)][
                implied.property_dcid
            ] = implied.value_dcid
        self._implied_by_measure = {
            measure: MappingProxyType(values) for measure, values in implied_by_measure.items()
        }
        self.measure_classes = tuple(measure_classes)
        """The patterns naming each measure class, in declaration order."""
        self.variable_aliases: dict[str, str] = dict(variable_aliases or {})
        """Lowercase words recall reads as the full term each stands for, applied to the
        variable query alone."""
        self.slot_plain_names: dict[str, str] = dict(slot_plain_names or {})
        """The noun naming each constraint property no code system declares, as a question
        names its slot (`code_index.declarations.SLOT_PLAIN_NAMES`)."""
        self.stat_type_words = tuple(stat_type_words)
        """The patterns naming a statType other than `measuredValue`, in declaration order;
        the frame reads the first that matches (`search.engine.frame`)."""
        self.codes = tuple(codes)
        self.statvars = tuple(statvars)
        self.listed_statvars = tuple(listed_statvars)
        """The StatVars of the datasets no code system enumerates, World Bank IDS among
        them, listed from the coverage build with the provenances it gives them. They are
        member rows alone: no family, dataset election, measure axis or shape reads them,
        and `code_index.members.listed_group` elects members over them in place of the
        graph walk."""
        self.source_codes = tuple(source_codes)
        """The StatVars each publisher's series code names under each provenance holding it
        (`SourceCodeRow`). The export reads them from the `sourceIndicatorCode` arcs of
        `listed_statvars`."""
        self.source_code_provenances: frozenset[str] = frozenset(
            row.provenance for row in self.source_codes
        )
        """The provenances under which this index holds a series code."""
        self.value_parents: dict[str, str] = dict(value_parents or {})
        """The `specializationOf` parent of each constraint value a listed StatVar holds, for
        the values with one (`IDSDebtorTypeGG` to `IDSDebtorTypePS`)."""
        _check_acyclic_parents(self.value_parents, relation="value parent")
        self.declarations: dict[str, CodeSystemDeclaration] = {
            d.code_system: d for d in declarations
        }
        self.datasets: dict[str, DatasetDeclaration] = {d.provenance: d for d in datasets}
        self.party_types: dict[str, PartyKind | None] = dict(party_types or {})
        """The kind each graph type gives a party (`code_index.types.party_kind`), for every
        type a party of a declared dataset carries. A type mapped to `None` gives no kind of
        its own."""
        self.parties: dict[str, PartyRow] = {}
        """Every party a declared dataset publishes, by dcid, with its kind and the group a
        total holds."""
        self.totals_by_group: dict[str, PartyRow] = {}
        """The total that holds each group (`PartyRow.totals`), by the group as search names
        it, whether or not it equals the sum of the group's members (`PartyRow.sums_members`)."""
        for party in parties:
            if party.dcid in self.parties:
                raise DuplicateDcid(f"party dcid {party.dcid!r} appears more than once")
            self.parties[party.dcid] = party
            if party.totals is not None:
                if party.totals in self.totals_by_group:
                    raise DuplicateDcid(f"two totals hold the group {party.totals!r}")
                self.totals_by_group[party.totals] = party
        side_classes: dict[tuple[RoleAxis, str], set[PartyClass]] = defaultdict(set)
        population_totals_by_axis: dict[RoleAxis, list[str]] = defaultdict(list)
        for party in self.parties.values():
            for dataset in party.datasets:
                kind = party.kind_in(dataset)
                party_class = party.class_of(kind)
                if kind != "total" and party_class is not None:
                    for axis in party.axes:
                        side_classes[axis, dataset].add(party_class)
            if any(POPULATION_CLASS in marker.covers for marker in party.whole):
                population_totals_by_axis[party.axes[0]].append(party.dcid)
        self.side_classes: dict[tuple[RoleAxis, str], frozenset[PartyClass]] = {
            side: frozenset(classes) for side, classes in side_classes.items()
        }
        """The classes of party, totals aside, each dataset publishes on each
        flow axis, by `(axis, dataset)`. A `PartyRow` keeps its axes and datasets apart, so a
        party published on both axes counts on each in every dataset that publishes it."""
        self.population_totals_by_axis: dict[RoleAxis, tuple[str, ...]] = {
            axis: tuple(sorted(dcids)) for axis, dcids in population_totals_by_axis.items()
        }
        """The totals some dataset marks as covering the `official` class (`PartyRow.whole`),
        by the one axis each stands on, sorted by dcid."""
        self.enrichment_digest = _enrichment_digest(self.parties.values())
        """SHA-256 over the sorted `(dcid, kind, party_class, whole markers)` rows of `parties`,
        the part of the index the `{role}_party_kind`, `{role}_party_class` and
        `{role}_party_whole` cells are read from. A file holding such cells is linked under a key salted with it
        (`datasets_api.version_store.enriched_rendering_key`), so an export that changes no
        kind, class or whole marker leaves every link in place."""
        self.whole_total_sets = tuple(whole_total_sets)
        """The unions of whole totals read as one population (`WholeTotalSet`), in declaration
        order. The loader refuses two that share a phrase (`loader._check_whole_totals`)."""
        self._whole_sets_by_phrase = {
            fold_text(whole_set.phrase): whole_set for whole_set in self.whole_total_sets
        }

        codes_by_dcid: dict[str, CodeRow] = {}
        codes_by_system_code: dict[tuple[str, str], CodeRow] = {}
        children_by_parent: dict[str, list[str]] = defaultdict(list)
        text_index: dict[str, dict[str, CodeRow]] = defaultdict(dict)
        for row in self.codes:
            if row.dcid in codes_by_dcid:
                raise DuplicateDcid(f"code dcid {row.dcid!r} appears more than once")
            codes_by_dcid[row.dcid] = row
            system_code = (row.code_system, row.code)
            if system_code in codes_by_system_code:
                raise DuplicateDcid(
                    f"code system {row.code_system!r} code {row.code!r} appears more than once"
                )
            codes_by_system_code[system_code] = row
            if row.parent is not None:
                children_by_parent[row.parent].append(row.dcid)
            for text in (row.label, *row.synonyms):
                text_index[fold_text(text)][row.dcid] = row

        _check_consistent_parents(codes_by_dcid)
        _check_acyclic_parents(
            {dcid: row.parent for dcid, row in codes_by_dcid.items()}, relation="code parent"
        )
        _check_declared_defaults(self.declarations, codes_by_dcid)

        statvars_by_dcid: dict[str, StatVarRow] = {}
        variables_by_provenance: dict[str, list[str]] = defaultdict(list)
        listed_by_five_tuple: dict[FiveTuple, list[StatVarRow]] = defaultdict(list)
        for row in self.statvars + self.listed_statvars:
            if row.dcid in statvars_by_dcid:
                raise DuplicateDcid(f"StatVar dcid {row.dcid!r} appears more than once")
            statvars_by_dcid[row.dcid] = row
            for provenance in row.provenances:
                variables_by_provenance[provenance].append(row.dcid)
        self.variables_by_provenance: dict[str, tuple[str, ...]] = {
            provenance: tuple(sorted(dcids))
            for provenance, dcids in sorted(variables_by_provenance.items())
        }
        """The dcids of the StatVars, `statvars` and `listed_statvars` alike, whose
        `StatVarRow.provenances` name each provenance, sorted. A provenance no row names is
        absent."""
        for row in self.listed_statvars:
            listed_by_five_tuple[row_five_tuple(row)].append(row)
        self.listed_by_five_tuple: dict[FiveTuple, tuple[StatVarRow, ...]] = {
            five_tuple: tuple(rows) for five_tuple, rows in listed_by_five_tuple.items()
        }
        """Each five-tuple's `listed_statvars`, in index order."""

        self.datasets_by_five_tuple: dict[FiveTuple, tuple[str, ...]] = _datasets_by_five_tuple(
            self.statvars, self.datasets
        )
        """The provenances of the datasets publishing each five-tuple's rows, in serving
        order (`DatasetDeclaration.rank`), across the five-tuple's families."""
        self.families_by_five_tuple: dict[FiveTuple, tuple[IndexFamily, ...]] = (
            _families_by_five_tuple(self.statvars, self.datasets)
        )
        """Each five-tuple's families (`IndexFamily`), in serving order."""
        self.family_by_dcid: dict[str, IndexFamily] = {
            row.dcid: family
            for families in self.families_by_five_tuple.values()
            for family in reversed(families)
            for row in family.rows
        }
        """The family each row belongs to, by StatVar dcid, as the row's provenances
        assign it. A row two datasets of different families publish belongs to the first
        of them in serving order. Search groups an index row under this family whatever
        its constraint properties would select (`family_of`), so a dataset's rows stay in
        its own family when another dataset's family holds fewer properties."""
        self.dataset_pins: dict[tuple[str, FiveTuple], dict[str, str]] = _dataset_pins(
            self.statvars, self.declarations
        )
        """For each dataset and five-tuple it publishes, every code-system property the
        dataset's rows of the five-tuple all carry with one value, mapped to that value. An
        identity whose family that dataset serves binds it."""

        self.codes_by_system: dict[str, tuple[CodeRow, ...]] = {
            code_system: tuple(
                sorted(
                    (row for row in self.codes if row.code_system == code_system),
                    key=lambda row: row.code,
                )
            )
            for code_system in sorted({row.code_system for row in self.codes})
        }
        """Each code system's rows, ordered by `CodeRow.code` as a string, so a hierarchy
        such as `crsPurpose` reads parent before child ("100", "110", "111", "11110")."""

        self._partly_stated_by_five_tuple = _partly_stated_by_five_tuple(
            self.statvars + self.listed_statvars, self.partly_stated_values
        )
        self._statvars_by_dcid = statvars_by_dcid
        self._codes_by_dcid = codes_by_dcid
        self._codes_by_system_code = codes_by_system_code
        self._children_by_parent = {k: tuple(v) for k, v in children_by_parent.items()}
        self._codes_by_text = {k: tuple(v.values()) for k, v in text_index.items()}
        self._codes_by_tokens = TokenLookup(
            (text, row) for row in self.codes for text in (row.label, *row.synonyms)
        )
        codes_by_word: dict[str, dict[str, None]] = defaultdict(dict)
        words_by_code: dict[str, set[str]] = defaultdict(set)
        for row in self.codes:
            for text in (row.label, *row.synonyms):
                for word in word_tokens(text):
                    codes_by_word[fold_word(word.text)][row.dcid] = None
                    words_by_code[row.dcid].add(fold_word(word.text))
        self._codes_by_word = {word: tuple(dcids) for word, dcids in codes_by_word.items()}
        self._words_by_code = {dcid: frozenset(words) for dcid, words in words_by_code.items()}

        members_by_source_code: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
        source_codes_by_dcid: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in self.source_codes:
            members_by_source_code[row.code][row.provenance] = row.dcids
            for dcid in row.dcids:
                source_codes_by_dcid[dcid].append((row.provenance, row.code))
        self._members_by_source_code = {
            code: MappingProxyType(dict(sorted(members.items())))
            for code, members in members_by_source_code.items()
        }
        self._source_codes_by_dcid = {
            dcid: tuple(sorted(pairs)) for dcid, pairs in source_codes_by_dcid.items()
        }

    # --- Lookup ---

    def implied_values_of(self, population_type: str, measured_property: str) -> Mapping[str, str]:
        """Each property mapped to the value the headline series of the measure
        (`population_type`, `measured_property`) already means (`implied_values`). Empty
        for a measure that declares none."""
        return self._implied_by_measure.get((population_type, measured_property), _NO_IMPLIED)

    def partly_stated_on(self, five_tuple: FiveTuple) -> Mapping[str, frozenset[tuple[str, str]]]:
        """Each dataset mapped to the (property, value) pairs it states partly on `five_tuple`:
        the ones it declares (`partly_stated_values`) where one of its rows on the five-tuple
        states the value and another states nothing on the property. A five-tuple where every
        row of the dataset states the property, or none states the value, holds none. Empty
        for a five-tuple no declaring dataset states a value partly on."""
        return MappingProxyType(self._partly_stated_by_five_tuple.get(five_tuple, {}))

    def family_of(self, five_tuple: FiveTuple, keys: AbstractSet[str]) -> IndexFamily | None:
        """The family of `five_tuple` that constraint properties `keys` select: the one
        family one of whose rows declares only properties among `keys`. When several or
        none have such a row, it is the one of those, or of all families, whose properties
        hold every key and are fewest: a family whose properties extend another's is told
        apart from it by the keys the extension adds. The constraint properties of a StatVar
        the index does not hold and a shape's slot properties select their family through
        this method; an index row's family is `family_by_dcid`, and an identity's is
        `code_index.members.query_family`, which reads the dataset it names and which of
        its slots it binds. Whether `keys` cover a row of the family, so that the index
        decides membership, is `IndexFamily.covers`.

        `None` when no family of `five_tuple` is selected, as for a debt series beside CRS
        flows, whose keys neither cover a CRS row nor lie among the CRS properties, and
        when `keys` fit two families neither of whose properties hold the other's: no
        StatVar carries both families' codes, so such keys name no population the index
        describes, and the graph walk resolves them."""
        families = self.families_by_five_tuple.get(five_tuple, ())
        covered = [family for family in families if family.covers(keys)]
        if len(covered) == 1:
            return covered[0]
        holding = [family for family in covered or families if keys <= family.properties]
        fewest = [
            family
            for family in holding
            if not any(other.properties < family.properties for other in holding)
        ]
        return fewest[0] if len(fewest) == 1 else None

    def code_by_dcid(self, dcid: str) -> CodeRow | None:
        """The code row for `dcid`, or `None` if this index holds no such
        row."""
        return self._codes_by_dcid.get(dcid)

    def is_code_system(self, dcid: str) -> bool:
        """Whether `dcid` is a code system's graph id (`CodeSystemDeclaration.enum_dcid`)."""
        return any(d.enum_dcid == dcid for d in self.declarations.values())

    def system_of_code(self, dcid: str) -> str | None:
        """The graph id of the system the code `dcid` belongs to (its declaration's
        `enum_dcid`), or `None` when this index holds no such code or its system declares no
        graph id."""
        row = self._codes_by_dcid.get(dcid)
        declaration = self.declarations.get(row.code_system) if row is not None else None
        return declaration.enum_dcid if declaration is not None else None

    def party_kind_of(self, dcid: str, types: Iterable[str]) -> PartyKind | None:
        """The kind of the party `dcid` whose graph types are `types`, read for no dataset: its
        row's kind in every dataset that publishes it when `parties` lists it
        (`PartyRow.kind_over`), `None` when that kind differs between datasets, else the kind
        its types give (`code_index.types.party_kind`). A reader with a dataset in hand reads
        `PartyRow.kind_in` instead."""
        party = self.parties.get(dcid)
        return party.kind_over() if party is not None else party_kind(types, self.party_types)

    def whole_set(self, phrase: str) -> WholeTotalSet | None:
        """The whole total set a `phrase` names, compared under `fold_text`, or `None`."""
        return self._whole_sets_by_phrase.get(fold_text(phrase))

    def statvar_by_dcid(self, dcid: str) -> StatVarRow | None:
        """The StatVar row for `dcid`, from `statvars` or `listed_statvars`, or `None` if
        this index holds no such row."""
        return self._statvars_by_dcid.get(dcid)

    def statvar_provenances(self, dcids: Iterable[str]) -> frozenset[str]:
        """The provenances that publish the StatVars `dcids` (`StatVarRow.provenances`), the
        datasets a reader of them serves. A StatVar this index does not list adds none."""
        return frozenset(
            provenance
            for dcid in dcids
            if (row := self.statvar_by_dcid(dcid)) is not None
            for provenance in row.provenances
        )

    def code_by_code(self, code_system: str, code: str) -> CodeRow | None:
        """Returns the row for `code` in `code_system`. Code strings may recur across systems,
        so pass both fields to identify the row."""
        return self._codes_by_system_code.get((code_system, code))

    def codes_by_text(self, text: str) -> tuple[CodeRow, ...]:
        """Every code row whose label or synonym equals `text` under `fold_text`, across
        every code system this index carries, in index order. A row appears once, however
        many of its texts match, and the same text naming codes in different systems returns
        more than one row. The caller disambiguates. This method performs lookup only."""
        return self._codes_by_text.get(fold_text(text), ())

    def codes_by_tokens(self, tokens: Sequence[Token]) -> tuple[CodeRow, ...]:
        """Every code row whose label or synonym matches the query span `tokens`
        (`code_index.tokens`). The words are equal casefolded, and each separator the span
        holds sits where the text holds it. A row appears once, whichever of its texts
        match, and rows of several systems are all returned."""
        return self._codes_by_tokens.get(tokens)

    def candidate_codes(
        self,
        words: Iterable[str],
        *,
        code_systems: AbstractSet[str],
        context: Iterable[str] = (),
        seeds: Sequence[str] = (),
    ) -> CandidateCodes:
        """The candidate menu (`CandidateCodes`) of a phrase whose `words` no code reads. It
        holds the codes of `code_systems` whose label or synonyms share one of `words`, its
        hits, and the relatives of each.

        A word of a code's texts matches a query word when the two are equal casefolded,
        once one trailing "s" is dropped from each word longer than two letters. A hit is
        a code sharing at least one of `words`. Hits rank by how many of `words` and
        `context` (the words of the code terms beside the phrase) their texts share, then
        by how rare the shared words are (the sum, over the shared words, of one over the
        number of codes whose texts hold the word), then by dcid, and the first
        `CANDIDATE_HITS` stay. The menu lists those hits, then `seeds` (the codes the terms
        beside the phrase name), then the ancestors of each hit and seed, nearest first,
        and its children, in index order, each code once and at most `CANDIDATE_LIMIT` of
        them. Empty when no code of `code_systems` shares one of `words`. "maternal"
        shares no CRS purpose's text, while "contraceptive" shares 13030's synonym
        "contraceptive delivery".
        """
        asked = {fold_word(word) for word in words}
        shared: dict[str, set[str]] = defaultdict(set)
        for word in asked | {fold_word(word) for word in context}:
            for dcid in self._codes_by_word.get(word, ()):
                if self._codes_by_dcid[dcid].code_system in code_systems:
                    shared[dcid].add(word)

        def rank(dcid: str) -> tuple[int, Fraction, str]:
            rarity = sum(
                (Fraction(1, len(self._codes_by_word[word])) for word in shared[dcid]),
                Fraction(0),
            )
            return -len(shared[dcid]), -rarity, dcid

        hits = sorted((dcid for dcid, found in shared.items() if found & asked), key=rank)[
            :CANDIDATE_HITS
        ]
        if not hits:
            return CandidateCodes(hits=(), codes=())
        roots = [*hits, *seeds]
        menu = dict.fromkeys(roots)
        for dcid in roots:
            row = self._codes_by_dcid.get(dcid)
            while row is not None and row.parent is not None:
                menu.setdefault(row.parent)
                row = self._codes_by_dcid.get(row.parent)
            for child in self._children_by_parent.get(dcid, ()):
                menu.setdefault(child)
        return CandidateCodes(hits=tuple(hits), codes=tuple(menu)[:CANDIDATE_LIMIT])

    def words_shared(self, dcid: str, words: Iterable[str]) -> frozenset[str]:
        """The words of `words` the label or a synonym of the code `dcid` holds, compared
        as `candidate_codes` compares them: casefolded, less one trailing "s" on a word
        longer than two letters, and returned in that form. Empty for an unknown code."""
        held = self._words_by_code.get(dcid, frozenset())
        return frozenset(folded for word in words if (folded := fold_word(word)) in held)

    # --- Source codes ---

    def source_code_members(self, code: str) -> Mapping[str, tuple[str, ...]]:
        """The StatVars the series code `code` names, by the provenance holding it, each
        sorted, and provenances in order. `code` compares upper-cased. Empty when no
        provenance holds the code."""
        return self._members_by_source_code.get(code.upper(), MappingProxyType({}))

    def source_codes_of(self, dcid: str) -> tuple[tuple[str, str], ...]:
        """The `(provenance, code)` pairs naming the StatVar `dcid`, sorted. Empty for a
        StatVar no code names."""
        return self._source_codes_by_dcid.get(dcid, ())

    def shared_codes(self, provenance: str, code: str) -> tuple[str, ...]:
        """The other codes `provenance` gives a StatVar the code `code` names there, sorted.
        Such a StatVar answers either code, so it cannot stand for `code` alone. IDS gives
        one net-flow StatVar both `DT.NFL.IMFC.CD` and `DT.NFL.MIDA.CD`. `code` compares
        upper-cased. Empty for a code no StatVar shares, and for one `provenance` does not
        hold."""
        code = code.upper()
        return tuple(
            sorted(
                {
                    other
                    for dcid in self.source_code_members(code).get(provenance, ())
                    for held_by, other in self._source_codes_by_dcid[dcid]
                    if held_by == provenance and other != code
                }
            )
        )

    # --- Value parents ---

    def value_ancestors(self, dcid: str) -> tuple[str, ...]:
        """The `value_parents` chain above the constraint value `dcid`, nearest first. Empty
        for a value with no parent and for one the index does not hold."""
        ancestors: list[str] = []
        node = self.value_parents.get(dcid)
        while node is not None:
            ancestors.append(node)
            node = self.value_parents.get(node)
        return tuple(ancestors)

    # --- Dataset declarations ---

    def absent_observation(self, provenance: str) -> AbsentObservation:
        """Returns the dataset's declared absent-observation rule, or `unknown` when no dataset
        declares `provenance`."""
        dataset = self.datasets.get(provenance)
        return dataset.absent_observation if dataset is not None else "unknown"

    def house_case_systems(self) -> dict[str, HouseDefaultCase]:
        """Each code system that declares a kind of question (`CodeSystemDeclaration.house_case`)
        mapped to that kind. A system declaring none is absent."""
        return {
            system: declaration.house_case
            for system, declaration in self.declarations.items()
            if declaration.house_case is not None
        }

    def house_defaults(self) -> dict[HouseDefaultCase, str]:
        """Each kind of question mapped to the provenance of its house default dataset
        (`DatasetDeclaration.house_default_for`). A kind no dataset claims is absent."""
        return {
            case: dataset.provenance
            for dataset in self.datasets.values()
            for case in dataset.house_default_for
        }

    # --- Descendants ---

    def has_children(self, dcid: str) -> bool:
        """Whether any row has `dcid` as its `parent`."""
        return bool(self._children_by_parent.get(dcid))

    def children(self, dcid: str) -> tuple[CodeRow, ...]:
        """The rows whose `parent` is `dcid`, ordered by `CodeRow.code` as a string. Empty
        for a leaf, a code of a flat system, and an unknown dcid."""
        return tuple(
            sorted(
                (self._codes_by_dcid[child] for child in self._children_by_parent.get(dcid, ())),
                key=lambda row: row.code,
            )
        )

    def expansion(self, dcid: str) -> tuple[str, ...]:
        """The codes a binding of `dcid` reads: `dcid` and its descendants (`descendants`)
        in a system declared `disjoint_from_children`, whose parent-coded records are apart
        from its children's, so the parent's total spans them all; `dcid` alone in a system
        whose parent row is already the total of its children (DAC1 aid types), so a bound
        parent names its own StatVar."""
        row = self._codes_by_dcid.get(dcid)
        declaration = self.declarations.get(row.code_system) if row is not None else None
        if declaration is not None and not declaration.disjoint_from_children:
            return (dcid,)
        return self.descendants(dcid)

    def descendants(self, dcid: str) -> tuple[str, ...]:
        """Returns `dcid` and its descendants breadth-first.

        Flat or unknown codes return `(dcid,)`. The walk tracks visited nodes to
        terminate on cyclic parent data."""
        row = self._codes_by_dcid.get(dcid)
        if row is None:
            return (dcid,)
        declaration = self.declarations.get(row.code_system)
        if declaration is None or not declaration.hierarchical:
            return (dcid,)
        result = [dcid]
        visited = {dcid}
        frontier = [dcid]
        while frontier:
            next_frontier = [
                child
                for node in frontier
                for child in self._children_by_parent.get(node, ())
                if child not in visited
            ]
            visited.update(next_frontier)
            result.extend(next_frontier)
            frontier = next_frontier
        return tuple(result)

    # --- Rollup ---

    def covering_roots(self, values: Iterable[str]) -> tuple[str, ...]:
        """Returns each value's root ancestor, deduplicated in input order.

        Flat-system and unknown values roll up to themselves."""
        result: list[str] = []
        for value in values:
            root = self._root_of(value)
            if root not in result:
                result.append(root)
        return tuple(result)

    def ancestor_at_depth(self, dcid: str, depth: int) -> str | None:
        """Returns the ancestor of `dcid` at `depth` on its parent chain, or `None` when `dcid`
        sits shallower than `depth`.

        Depth 0 is the chain's root, and `dcid`'s own depth returns `dcid`. Depth counts parent
        steps, so codes of one length can sit at different depths. Flat-system and unknown codes
        are roots. A dangling parent is the root, as in `covering_roots`."""
        chain = [dcid]
        row = self._codes_by_dcid.get(dcid)
        while row is not None and row.parent is not None:
            chain.append(row.parent)
            row = self._codes_by_dcid.get(row.parent)
        if depth >= len(chain):
            return None
        return chain[-1 - depth]

    def _root_of(self, dcid: str) -> str:
        """Returns the topmost ancestor, or `dcid` when unknown. A dangling parent returns
        that missing dcid as the root."""
        row = self._codes_by_dcid.get(dcid)
        if row is None:
            return dcid
        while row.parent is not None:
            parent_row = self._codes_by_dcid.get(row.parent)
            if parent_row is None:
                return row.parent
            row = parent_row
        return row.dcid

    # --- Filter ---

    def filter(self, query: ConstraintQuery) -> tuple[StatVarRow, ...]:
        """Returns matching StatVar rows in index order.

        `Bound` and `BoundSet` require a declared matching property. `Open`
        and omitted properties are ignored."""
        return tuple(row for row in self.statvars if self._matches(row, query))

    @staticmethod
    def _matches(row: StatVarRow, query: ConstraintQuery) -> bool:
        for prop, binding in query.items():
            if isinstance(binding, Bound):
                if row.constraint_properties.get(prop) != binding.value:
                    return False
            elif (
                isinstance(binding, BoundSet)
                and row.constraint_properties.get(prop) not in binding.values
            ):
                return False
        return True

    # --- Level-scoped menus ---

    def value_counts(self, code_system: str) -> dict[str, int]:
        """Counts StatVar rows by declared value in one linear scan."""
        counts: dict[str, int] = defaultdict(int)
        for row in self.statvars:
            value = row.constraint_properties.get(code_system)
            if value is not None:
                counts[value] += 1
        return dict(counts)

    def level_scoped_options(
        self, code_system: str, bound_dcid: str | None
    ) -> tuple[LevelOption, ...]:
        """Returns the menu of values for one code-system slot.

        A flat system returns every value as a root. A valid hierarchical binding returns
        its immediate children, then its parent. A bound root has no parent, so its
        children are followed by every other root of the system, without those roots'
        children. An unbound hierarchical slot returns each root followed by that root's
        children. `hierarchical` selects the branch, so a flat system's parent fields are
        ignored. Every hierarchical option with a parent names it in `parent`.

        Unknown or foreign `bound_dcid` values act as unbound. Set bindings pass `None`.
        Siblings sort by member count descending, then dcid ascending."""
        counts = self.value_counts(code_system)
        declaration = self.declarations.get(code_system)
        hierarchical = declaration is not None and declaration.hierarchical

        def _ordered(dcids: Iterable[str]) -> list[str]:
            return sorted(dcids, key=lambda d: (-counts.get(d, 0), d))

        def _children(parent: str) -> list[LevelOption]:
            return [
                LevelOption(dcid=d, relation="child", member_count=counts.get(d, 0), parent=parent)
                for d in _ordered(self._children_by_parent.get(parent, ()))
            ]

        if not hierarchical:
            all_dcids = (row.dcid for row in self.codes if row.code_system == code_system)
            return tuple(
                LevelOption(dcid=d, relation="root", member_count=counts.get(d, 0))
                for d in _ordered(all_dcids)
            )

        def _root(dcid: str) -> LevelOption:
            return LevelOption(dcid=dcid, relation="root", member_count=counts.get(dcid, 0))

        roots = _ordered(
            r.dcid for r in self.codes if r.code_system == code_system and r.parent is None
        )
        row = self._codes_by_dcid.get(bound_dcid) if bound_dcid is not None else None
        if row is not None and row.code_system == code_system:
            options = _children(row.dcid)
            if row.parent is None:
                options.extend(_root(root) for root in roots if root != row.dcid)
            else:
                parent_row = self._codes_by_dcid.get(row.parent)
                options.append(
                    LevelOption(
                        dcid=row.parent,
                        relation="parent",
                        member_count=counts.get(row.parent, 0),
                        parent=parent_row.parent if parent_row is not None else None,
                    )
                )
            return tuple(options)

        options = []
        for root in roots:
            options.append(_root(root))
            options.extend(_children(root))
        return tuple(options)
