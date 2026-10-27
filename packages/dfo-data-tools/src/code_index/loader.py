"""Parses a code index document into a `CodeIndex` (`parse_index`) and its measure-axis
blocks into a separate `MeasureIndex` (`parse_measure_axes`)."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import cast, get_args

from code_index.declarations import COMPOSITES
from code_index.query import CodeIndex, DuplicateDcid, fold_text, partly_stated_five_tuples
from code_index.types import (
    PARTY_CLASSES,
    PARTY_KINDS,
    AbsentObservation,
    BroadReading,
    CodeRow,
    CodeSystemDeclaration,
    CompositeDeclaration,
    DatasetDeclaration,
    DeclaredDefault,
    HouseDefaultCase,
    ImpliedValue,
    IndexStamp,
    MeasureAxis,
    MeasureClassDeclaration,
    MeasureIndex,
    MeasurePartitionLabel,
    MeasureValueRow,
    PartlyStatedValue,
    PartyKind,
    PartyRow,
    PartyWord,
    ShapeSlotDefault,
    SourceCodeRow,
    StatedWord,
    StatTypeWordDeclaration,
    StatVarRow,
    WholeMarker,
    WholeTotalSet,
    YearPeriod,
    covered_class,
    invalid_when_named_entry,
    marked_datasets,
)
from dfo_platform.roles import AXIS_LABELS, FlowAccount

_ABSENT_OBSERVATION_VALUES: frozenset[AbsentObservation] = frozenset({"zero", "unknown"})
_FLOW_ACCOUNTS: frozenset[FlowAccount] = frozenset(get_args(FlowAccount.__value__))
_HOUSE_DEFAULT_CASES: frozenset[HouseDefaultCase] = frozenset(get_args(HouseDefaultCase))


class IndexFormatError(ValueError):
    """A `CodeIndex` export has a missing required key or a value of the wrong type."""


def parse_index(
    payload: Mapping,
    *,
    source: object = "<document>",
    composites: Sequence[CompositeDeclaration] = COMPOSITES,
) -> CodeIndex:
    """Parses and validates a `CodeIndex` export document. `source` names it in errors.

    Raises `IndexFormatError` for missing or malformed fields, invalid
    dataset declarations, dataset declarations the StatVar rows contradict
    (`_check_datasets`), a party word naming a party the `parties` block does not
    list for its dataset and axis (`_check_party_words`), or an invalid
    `default.when_named`. Duplicate
    dcids or code pairs raise `DuplicateDcid`. Parent cycles raise
    `CyclicParent`. Cross-system parents raise `CrossSystemParent`.
    Invalid defaults raise `InvalidDeclaredDefault` when their code is
    outside the declaring system or `when_bound_under` names a missing or
    same-system code.

    The `party_types`, `parties`, `listed_statvars`, `source_codes` and `value_parents` blocks
    are optional: an index without the first two declares no party kind, one without the third
    lists no StatVar outside its code systems (`CodeIndex.listed_statvars`), one without the
    fourth holds no series code, and one without the fifth no value parent. So are
    `shape_slot_defaults`, `broad_readings`, `partly_stated_values`, `implied_values`,
    `measure_classes`, `variable_aliases`, `slot_plain_names`, `stat_type_words` and
    `whole_total_sets`, each reading as empty where absent. A measure class or stat-type
    pattern that does not compile, a variable alias or a `no_vocabulary_terms` entry that is not
    a non-empty lowercase string, or a slot plain name that is not a non-empty string raises
    `IndexFormatError`. A value that two `value_parents` rows name raises `DuplicateDcid`, and a
    parent chain that cycles
    `CyclicParent`. A `source_codes` row the StatVar rows contradict raises `IndexFormatError`
    (`_check_source_codes`), as does a `partly_stated_values` row they contradict
    (`_check_partly_stated_values`), and an `implied_values` row the code or StatVar rows
    contradict (`_check_implied_values`). A kind outside `PartyKind`, a type declared twice, a
    `totals` on a party that is not a total, or a false `sums_members` on a party holding no
    group raises `IndexFormatError`, as does a dataset kind the row cannot carry
    (`_check_dataset_kinds`), a class the row cannot carry (`_check_party_class`), a
    whole marker (`PartyRow.whole`) the row cannot carry (`_check_whole_markers`), two whole
    totals covering the same class on one dataset and axis, and a whole total set its members
    contradict (`_check_whole_totals`). A party without
    a `whole` key is no whole total. The index serves each of `composites` whose component
    tables it declares, and refuses one `_check_composites` refuses. A code row without a
    `definition` carries none, as does a measure value or measure label in `parse_measure_axes`."""
    try:
        stamp = _parse_stamp(payload["stamp"])
        declarations = [_parse_declaration(row) for row in payload["declarations"]]
        codes = [_parse_code(row) for row in payload["codes"]]
        statvars = [_parse_statvar(row) for row in payload["statvars"]]
        datasets = _parse_datasets(payload["datasets"])
        party_types = _parse_party_types(payload.get("party_types", ()))
        parties = [_parse_party(row) for row in payload.get("parties", ())]
        listed_statvars = [_parse_statvar(row) for row in payload.get("listed_statvars", ())]
        shape_slot_defaults = [
            _parse_shape_slot_default(row) for row in payload.get("shape_slot_defaults", ())
        ]
        broad_readings = [_parse_broad_reading(row) for row in payload.get("broad_readings", ())]
        partly_stated_values = [
            _parse_partly_stated_value(row) for row in payload.get("partly_stated_values", ())
        ]
        implied_values = [_parse_implied_value(row) for row in payload.get("implied_values", ())]
        measure_classes = [_parse_measure_class(row) for row in payload.get("measure_classes", ())]
        variable_aliases = _parse_variable_aliases(payload.get("variable_aliases", {}))
        slot_plain_names = _parse_slot_plain_names(payload.get("slot_plain_names", {}))
        stat_type_words = [_parse_stat_type_word(row) for row in payload.get("stat_type_words", ())]
        source_codes = [_parse_source_code(row) for row in payload.get("source_codes", ())]
        value_parents = _parse_value_parents(payload.get("value_parents", ()))
        whole_total_sets = [
            _parse_whole_total_set(row) for row in payload.get("whole_total_sets", ())
        ]
    except (KeyError, TypeError) as exc:
        raise IndexFormatError(f"malformed code_index export at {source!r}: {exc}") from exc
    _check_datasets(datasets, statvars)
    _check_party_words(datasets, parties)
    _check_whole_totals(parties, whole_total_sets)
    _check_source_codes(source_codes, [*statvars, *listed_statvars])
    _check_partly_stated_values(partly_stated_values, [*statvars, *listed_statvars])
    _check_implied_values(implied_values, declarations, codes, [*statvars, *listed_statvars])
    served_composites = _check_composites(composites, datasets, statvars)
    return CodeIndex(
        stamp=stamp,
        declarations=declarations,
        codes=codes,
        statvars=statvars,
        datasets=datasets,
        party_types=party_types,
        parties=parties,
        listed_statvars=listed_statvars,
        shape_slot_defaults=shape_slot_defaults,
        broad_readings=broad_readings,
        partly_stated_values=partly_stated_values,
        implied_values=implied_values,
        measure_classes=measure_classes,
        variable_aliases=variable_aliases,
        slot_plain_names=slot_plain_names,
        stat_type_words=stat_type_words,
        source_codes=source_codes,
        value_parents=value_parents,
        whole_total_sets=whole_total_sets,
        composites=served_composites,
    )


def _check_composites(
    composites: Sequence[CompositeDeclaration],
    datasets: Sequence[DatasetDeclaration],
    statvars: Sequence[StatVarRow],
) -> tuple[CompositeDeclaration, ...]:
    """The `composites` whose component tables `datasets` all declare, each checked.

    Raises `IndexFormatError` for a composite whose provenance a dataset declares, with no
    component or two of one role, a recipe setting both a measured property and a basis, a
    stacked component limited to periods, a spliced component with none, or spliced periods
    that overlap or leave a year between them. A served stacked composite whose component
    tables' rows declare different role sets is refused too, since the composed frame holds
    one set of role columns. A composite naming a table `datasets` does not declare is not served."""
    declared = {dataset.provenance for dataset in datasets}
    role_sets: dict[str, set[tuple[str, ...]]] = {}
    for row in statvars:
        for provenance in row.provenances:
            role_sets.setdefault(provenance, set()).add(row.observation_properties)
    served: list[CompositeDeclaration] = []
    for composite in composites:
        _check_composite_declaration(composite, declared)
        tables = {recipe.dataset for recipe in composite.components}
        if not tables <= declared:
            continue
        if composite.kind == "stacked":
            signatures = {frozenset(role_sets.get(table, set())) for table in sorted(tables)}
            if len(signatures) > 1:
                raise IndexFormatError(
                    f"stacked composite {composite.provenance!r} reads tables {sorted(tables)!r} "
                    "whose rows declare different role sets"
                )
        served.append(composite)
    return tuple(served)


def _check_composite_declaration(composite: CompositeDeclaration, declared: set[str]) -> None:
    """Refuse a malformed composite declaration (`_check_composites`)."""
    name = composite.provenance
    if name in declared:
        raise IndexFormatError(f"composite {name!r} is also declared as a dataset")
    roles = [recipe.role for recipe in composite.components]
    if not roles or len(set(roles)) != len(roles):
        raise IndexFormatError(f"composite {name!r} declares components {roles!r}")
    for recipe in composite.components:
        if recipe.measured_property is not None and recipe.basis is not None:
            raise IndexFormatError(
                f"composite {name!r} component {recipe.role!r} sets both a measured property "
                "and a basis"
            )
        if (recipe.periods is None) != (composite.kind == "stacked"):
            raise IndexFormatError(
                f"composite {name!r} is {composite.kind} and its component {recipe.role!r} "
                f"declares periods {recipe.periods!r}"
            )
        # A spliced component's zeros would fill the other components' years, which it
        # does not cover.
        if recipe.zero_absent and composite.kind == "spliced":
            raise IndexFormatError(
                f"spliced composite {name!r} component {recipe.role!r} reads absence as zero"
            )
    if composite.kind == "spliced":
        _check_spliced_periods(composite)


def _check_spliced_periods(composite: CompositeDeclaration) -> None:
    """Refuse spliced periods that overlap or leave a year out between them. Ordered by first
    year, each period must begin the year after the one before it ends."""
    periods = sorted(
        (cast("YearPeriod", recipe.periods) for recipe in composite.components),
        key=lambda period: -1 if period.first is None else period.first,
    )
    for earlier, later in itertools.pairwise(periods):
        if earlier.last is None or later.first is None or later.first != earlier.last + 1:
            raise IndexFormatError(
                f"spliced composite {composite.provenance!r} declares periods {earlier!r} and "
                f"{later!r}, which overlap or leave a year out"
            )


def _parse_shape_slot_default(row: Mapping) -> ShapeSlotDefault:
    return ShapeSlotDefault(
        population_type=row["population_type"],
        property_dcid=row["property"],
        value_dcid=row["value"],
    )


def _parse_broad_reading(row: Mapping) -> BroadReading:
    return BroadReading(
        property_dcid=row["property"],
        value_dcid=row["value"],
        reads_as=row["reads_as"],
        unless_bound=row["unless_bound"],
        held_with=row["held_with"],
        note=row["note"],
    )


def _parse_partly_stated_value(row: Mapping) -> PartlyStatedValue:
    return PartlyStatedValue(
        provenance=row["provenance"],
        property_dcid=row["property"],
        value_dcid=row["value"],
    )


def _parse_implied_value(row: Mapping) -> ImpliedValue:
    return ImpliedValue(
        population_type=row["population_type"],
        measured_property=row["measured_property"],
        property_dcid=row["property"],
        value_dcid=row["value"],
    )


def _parse_measure_class(row: Mapping) -> MeasureClassDeclaration:
    """One `measure_classes` row, its pattern compiled, refusing a pattern that does not
    compile and a `fallback` that is not a boolean."""
    try:
        pattern = re.compile(row["pattern"])
    except re.error as exc:
        raise IndexFormatError(
            f"measure class {row['class']!r} pattern {row['pattern']!r} does not compile: {exc}"
        ) from exc
    fallback = row.get("fallback", False)
    if not isinstance(fallback, bool):
        raise IndexFormatError(f"measure class fallback {fallback!r} is not a boolean")
    return MeasureClassDeclaration(measure_class=row["class"], pattern=pattern, fallback=fallback)


def _parse_stat_type_word(row: Mapping) -> StatTypeWordDeclaration:
    """One `stat_type_words` row, its pattern compiled, refusing a pattern that does not
    compile."""
    try:
        pattern = re.compile(row["pattern"])
    except re.error as exc:
        raise IndexFormatError(
            f"stat type {row['stat_type']!r} pattern {row['pattern']!r} does not compile: {exc}"
        ) from exc
    return StatTypeWordDeclaration(pattern=pattern, stat_type=row["stat_type"])


def _parse_variable_aliases(payload: Mapping) -> dict[str, str]:
    """The `variable_aliases` block, refusing an alias that is not a non-empty lowercase
    string and an empty expansion."""
    aliases = dict(payload)
    invalid = invalid_when_named_entry(list(aliases))
    if invalid is not None:
        raise IndexFormatError(
            f"variable alias {list(aliases)[invalid]!r} must be a non-empty lowercase string"
        )
    for alias, expansion in aliases.items():
        if not isinstance(expansion, str) or not expansion.strip():
            raise IndexFormatError(f"variable alias {alias!r} expands to {expansion!r}")
    return aliases


def _parse_slot_plain_names(payload: Mapping) -> dict[str, str]:
    """The `slot_plain_names` block, refusing a name that is not a non-empty string without
    surrounding space."""
    names = dict(payload)
    for prop, name in names.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise IndexFormatError(f"slot {prop!r} has plain name {name!r}")
    return names


def _parse_party_types(rows: Sequence[Mapping]) -> dict[str, PartyKind | None]:
    """Each `party_types` row's type and kind, refusing a kind outside `PartyKind` other than
    `null` and a type declared twice."""
    kinds: dict[str, PartyKind | None] = {}
    for row in rows:
        type_, kind = row["type"], row["kind"]
        if kind is not None and kind not in PARTY_KINDS:
            raise IndexFormatError(
                f"party type {type_!r} carries kind {kind!r}, outside {sorted(PARTY_KINDS)!r}"
            )
        if type_ in kinds:
            raise IndexFormatError(f"party type {type_!r} is declared more than once")
        kinds[type_] = kind
    return kinds


def _parse_party(row: Mapping) -> PartyRow:
    """One `parties` row, refusing a kind outside `PartyKind`, an axis outside the flow axes,
    a `totals` or `residuals` on a party that is not a total, and a false `sums_members` on a
    party that holds no group or that counts residuals. `whole` markers are parsed by
    `_parse_whole_marker` and checked by `_check_whole_markers`, the dataset kinds by
    `_check_dataset_kinds`, and the class by `_check_party_class`. `residuals`,
    `sums_members`, `label`, `datasets`, `axes`, `whole` and `dataset_kinds` are optional, and
    a row without `sums_members` reads it as true. `party_class` is required on a `party` and a
    residual and may be absent on a total."""
    party = PartyRow(
        dcid=row["dcid"],
        kind=row["kind"],
        party_class=row.get("party_class"),
        totals=row["totals"],
        residuals=tuple(row.get("residuals", ())),
        sums_members=row.get("sums_members", True),
        label=row.get("label"),
        datasets=tuple(row.get("datasets", ())),
        axes=tuple(row.get("axes", ())),
        whole=tuple(
            sorted(
                (_parse_whole_marker(marker) for marker in row.get("whole", ())),
                key=lambda marker: marker.dataset,
            )
        ),
        dataset_kinds=tuple(
            sorted((entry["dataset"], entry["kind"]) for entry in row.get("dataset_kinds", ()))
        ),
    )
    if party.kind not in PARTY_KINDS:
        raise IndexFormatError(
            f"party {party.dcid!r} carries kind {party.kind!r}, outside {sorted(PARTY_KINDS)!r}"
        )
    if unknown := sorted(set(party.axes) - AXIS_LABELS):
        raise IndexFormatError(
            f"party {party.dcid!r} names axes {unknown!r}, outside {sorted(AXIS_LABELS)!r}"
        )
    if (party.totals is not None or party.residuals) and party.kind != "total":
        raise IndexFormatError(
            f"party {party.dcid!r} of kind {party.kind!r} names a group or residuals; "
            "only a total holds them"
        )
    if not party.sums_members and (party.totals is None or party.residuals):
        raise IndexFormatError(
            f"party {party.dcid!r} claims no sum of members but holds no group or counts "
            "residuals; only a total that holds a group and counts no residual makes no claim"
        )
    _check_dataset_kinds(party)
    _check_whole_markers(party)
    _check_party_class(party)
    return party


def _check_dataset_kinds(party: PartyRow) -> None:
    """Raises `IndexFormatError` for a `dataset_kinds` list the row cannot carry: a kind outside
    `PartyKind` or equal to the row's `kind`, a dataset the party is not published in or named
    twice, entries naming every dataset that publishes the party, whose kind is then one kind
    and belongs in `kind`, or any entry on a total that holds a group or counts residuals, whose
    group holds in every dataset."""
    named: set[str] = set()
    for dataset, kind in party.dataset_kinds:
        if kind not in PARTY_KINDS or kind == party.kind:
            raise IndexFormatError(
                f"party {party.dcid!r} takes kind {kind!r} in {dataset!r}; a dataset kind is one "
                f"of {sorted(PARTY_KINDS)!r} other than the row's kind {party.kind!r}"
            )
        if dataset not in party.datasets:
            raise IndexFormatError(
                f"party {party.dcid!r} takes a kind in {dataset!r}, which does not publish it"
            )
        if dataset in named:
            raise IndexFormatError(f"party {party.dcid!r} takes two kinds in {dataset!r}")
        named.add(dataset)
    if named and named >= set(party.datasets):
        raise IndexFormatError(
            f"party {party.dcid!r} takes a dataset kind in every dataset that publishes it; "
            "one kind in every dataset is its kind"
        )
    if party.dataset_kinds and (party.totals is not None or party.residuals):
        raise IndexFormatError(
            f"party {party.dcid!r} holds a group or residuals and takes a kind per dataset; a "
            "total that holds a group is a total in every dataset"
        )


def _parse_whole_marker(row: Mapping) -> WholeMarker:
    """One `whole` entry: its dataset, the classes it covers (sorted) and its optional window,
    which must be two years in order. `_check_whole_markers` checks the rest."""
    years = row.get("years")
    if years is not None:
        years = tuple(years)
        if len(years) != 2 or not all(isinstance(year, int) for year in years):
            raise IndexFormatError(f"whole marker years {years!r} is not a first and last year")
        if years[0] > years[1]:
            raise IndexFormatError(f"whole marker years {years!r} are out of order")
    return WholeMarker(dataset=row["dataset"], covers=tuple(sorted(row["covers"])), years=years)


def _check_whole_markers(party: PartyRow) -> None:
    """Raises `IndexFormatError` for a `whole` marker list the row cannot carry: on a party
    that is not a total, on other than exactly one published axis, for a dataset the party is
    not published in, takes another kind in (`PartyRow.kind_in`) or is marked twice for, or
    covering no class, a class outside `PartyClass` or one class twice."""
    if not party.whole:
        return
    if party.kind != "total":
        raise IndexFormatError(
            f"party {party.dcid!r} of kind {party.kind!r} carries a whole marker; only a total "
            "holds a population"
        )
    if len(party.axes) != 1:
        raise IndexFormatError(
            f"party {party.dcid!r} carries a whole marker but is published on "
            f"{list(party.axes)!r}; a whole total stands on exactly one published axis"
        )
    marked: set[str] = set()
    for marker in party.whole:
        if marker.dataset not in party.datasets:
            raise IndexFormatError(
                f"party {party.dcid!r} is marked whole in {marker.dataset!r}, which does not "
                "publish it"
            )
        if (kind := party.kind_in(marker.dataset)) != "total":
            raise IndexFormatError(
                f"party {party.dcid!r} is marked whole in {marker.dataset!r}, where it takes the "
                f"kind {kind!r}; only a total holds a population"
            )
        if marker.dataset in marked:
            raise IndexFormatError(
                f"party {party.dcid!r} carries two whole markers for {marker.dataset!r}"
            )
        marked.add(marker.dataset)
        if not marker.covers:
            raise IndexFormatError(f"party {party.dcid!r} covers no class in {marker.dataset!r}")
        if unknown := sorted(set(marker.covers) - PARTY_CLASSES):
            raise IndexFormatError(
                f"party {party.dcid!r} covers classes {unknown!r} in {marker.dataset!r}, outside "
                f"{sorted(PARTY_CLASSES)!r}"
            )
        if len(set(marker.covers)) != len(marker.covers):
            raise IndexFormatError(
                f"party {party.dcid!r} names a covered class more than once in {marker.dataset!r}"
            )


def _check_party_class(party: PartyRow) -> None:
    """Raises `IndexFormatError` for a class the row cannot carry: a `party` reading other
    than `official` or `private`, a residual reading other than `residual`, or a total whose
    markers name different classes or whose class is not the one they name (`covered_class`),
    `None` for a total with no marker. An index exported before parties carried a class
    therefore refuses to load."""
    if party.kind == "party":
        if party.party_class not in ("official", "private"):
            raise IndexFormatError(
                f"party {party.dcid!r} carries class {party.party_class!r}; a party is "
                "official or private"
            )
        return
    if party.kind == "residual":
        if party.party_class != "residual":
            raise IndexFormatError(
                f"residual {party.dcid!r} carries class {party.party_class!r}; a residual is "
                "residual"
            )
        return
    classes = {covered_class(marker.covers) for marker in party.whole}
    if len(classes) > 1:
        raise IndexFormatError(
            f"total {party.dcid!r} has whole markers that name different classes "
            f"{sorted(map(str, classes))!r}"
        )
    expected = classes.pop() if classes else None
    if party.party_class != expected:
        raise IndexFormatError(
            f"total {party.dcid!r} carries class {party.party_class!r}, but its whole markers "
            f"give {expected!r}"
        )


def _parse_stamp(payload: Mapping) -> IndexStamp:
    return IndexStamp(
        graph_endpoint=payload["graph_endpoint"],
        exported_at=payload["exported_at"],
        code_counts=dict(payload["code_counts"]),
        statvar_shape_counts=dict(payload["statvar_shape_counts"]),
        content_hash=payload["content_hash"],
        coverage_build=payload["coverage_build"],
        statvars_without_coverage=payload["statvars_without_coverage"],
    )


def _parse_declared_default(payload: Mapping | None) -> DeclaredDefault | None:
    """Parses a default block, requiring non-empty lowercase `when_named` terms, a boolean
    `when_open` and a non-empty `note`, and a note on a default that fires on an open slot.
    `when_open` and `note` read as false and none where absent.

    Matching casefolds terms and checks whole words."""
    if payload is None:
        return None
    when_named = tuple(payload["when_named"])
    invalid = invalid_when_named_entry(when_named)
    if invalid is not None:
        raise IndexFormatError(
            f"declared default when_named entry {when_named[invalid]!r} must be a "
            "non-empty lowercase string"
        )
    when_open = payload.get("when_open", False)
    note = payload.get("note")
    if not isinstance(when_open, bool):
        raise IndexFormatError(f"declared default when_open {when_open!r} is not a boolean")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise IndexFormatError(f"declared default note {note!r} is not a non-empty string")
    if when_open and note is None:
        raise IndexFormatError(
            f"declared default {payload['code']!r} fires on an open slot and carries no note"
        )
    return DeclaredDefault(
        code=payload["code"],
        when_bound_under=tuple(payload["when_bound_under"]),
        when_named=when_named,
        when_open=when_open,
        note=note,
    )


def _parse_declaration(payload: Mapping) -> CodeSystemDeclaration:
    # `default` is required. Use `null` when the code system declares none.
    return CodeSystemDeclaration(
        code_system=payload["code_system"],
        hierarchical=payload["hierarchical"],
        disjoint_from_children=payload["disjoint_from_children"],
        summable=payload["summable"],
        absent_reading=payload["absent_reading"],
        plain_name=payload["plain_name"],
        default=_parse_declared_default(payload["default"]),
        enum_dcid=payload["enum_dcid"],
        no_vocabulary_terms=_parse_no_vocabulary_terms(payload.get("no_vocabulary_terms", ())),
        house_case=_parse_house_case(payload["code_system"], payload.get("house_case")),
    )


def _parse_house_case(code_system: str, case: object) -> HouseDefaultCase | None:
    """A declaration's `house_case`, refusing a value outside `HouseDefaultCase`. Absent
    reads as none."""
    if case is None:
        return None
    if case not in _HOUSE_DEFAULT_CASES:
        raise IndexFormatError(
            f"declaration for {code_system!r} carries house_case={case!r}, outside the "
            f"vocabulary {sorted(_HOUSE_DEFAULT_CASES)!r}"
        )
    return cast(HouseDefaultCase, case)


def _parse_no_vocabulary_terms(terms: Sequence[str]) -> tuple[str, ...]:
    """A declaration's `no_vocabulary_terms`, refusing an entry that is not a non-empty
    lowercase string. Absent reads as none."""
    terms = tuple(terms)
    invalid = invalid_when_named_entry(terms)
    if invalid is not None:
        raise IndexFormatError(
            f"no_vocabulary_terms entry {terms[invalid]!r} must be a non-empty lowercase string"
        )
    return terms


def _parse_stated_words(provenance: str, rows: Sequence[Mapping]) -> tuple[StatedWord, ...]:
    """A datasets row's `stated_words`, refusing a word that is not a single lowercase
    word, an empty measured property, a `where` axis outside `AXIS_LABELS`, and a `where`
    kind that is not a non-empty lowercase string. Absent reads as empty. `_check_datasets`
    checks each measured property against the dataset's rows."""
    stated: list[StatedWord] = []
    for row in rows:
        word, prop = row["word"], row["measured_property"]
        _check_declared_word(provenance, word, prop)
        where = _parse_where(provenance, word, row.get("where", {}))
        stated.append(StatedWord(word=word, measured_property=prop, where=where))
    return tuple(stated)


def _parse_whole_total_set(row: Mapping) -> WholeTotalSet:
    """One `whole_total_sets` row, refusing a phrase that is not a non-empty string, an axis
    outside the flow axes and fewer than two members. `_check_whole_totals` checks the members
    against the `parties` block."""
    phrase, axis, members = row["phrase"], row["axis"], tuple(row["members"])
    if not isinstance(phrase, str) or not fold_text(phrase):
        raise IndexFormatError(f"whole total set {phrase!r} is not a non-empty phrase")
    if axis not in AXIS_LABELS:
        raise IndexFormatError(
            f"whole total set {phrase!r} names axis {axis!r}, outside {sorted(AXIS_LABELS)!r}"
        )
    if len(members) < 2 or len(set(members)) != len(members):
        raise IndexFormatError(f"whole total set {phrase!r} needs two or more distinct members")
    return WholeTotalSet(phrase=phrase, axis=axis, members=members)


def _parse_party_words(provenance: str, rows: Sequence[Mapping]) -> tuple[PartyWord, ...]:
    """A datasets row's `party_words`, refusing what `_parse_stated_words` refuses of a
    stated word, an axis outside `AXIS_LABELS`, a party that is not a non-empty string, and
    a word declared twice for one measured property and axis, which would name two parties
    there. Absent reads as empty. `_check_datasets` checks each measured property against the
    dataset's rows, and `_check_party_words` each party against the `parties` block."""
    named: list[PartyWord] = []
    for row in rows:
        word, prop, axis, party = row["word"], row["measured_property"], row["axis"], row["party"]
        _check_declared_word(provenance, word, prop)
        if axis not in AXIS_LABELS:
            raise IndexFormatError(
                f"datasets row for {provenance!r} names a party for {word!r} on {axis!r}, "
                f"outside {sorted(AXIS_LABELS)!r}"
            )
        if not isinstance(party, str) or not party:
            raise IndexFormatError(
                f"datasets row for {provenance!r} names {party!r} for {word!r}, "
                "which is not a party dcid"
            )
        if any(
            (word, prop, axis) == (other.word, other.measured_property, other.axis)
            for other in named
        ):
            raise IndexFormatError(
                f"datasets row for {provenance!r} names two parties for {word!r} on {axis!r} "
                f"of {prop!r}"
            )
        where = _parse_where(provenance, word, row.get("where", {}))
        named.append(
            PartyWord(word=word, measured_property=prop, axis=axis, party=party, where=where)
        )
    return tuple(named)


def _check_declared_word(provenance: str, word: str, prop: str) -> None:
    """Refuses a declared word that is not a single lowercase word, and a measured
    property that is not a non-empty string."""
    if invalid_when_named_entry([word]) is not None or len(word.split()) != 1:
        raise IndexFormatError(
            f"datasets row for {provenance!r} states {word!r}, which is not a single lowercase word"
        )
    if not isinstance(prop, str) or not prop:
        raise IndexFormatError(
            f"datasets row for {provenance!r} states {word!r} for {prop!r}, "
            "which is not a measured property"
        )


def _parse_where(provenance: str, word: str, where: Mapping) -> dict:
    """A declared word's `where`, refusing an axis outside `AXIS_LABELS` and a kind that is
    not a non-empty lowercase string."""
    where = dict(where)
    if unknown := sorted(set(where) - AXIS_LABELS):
        raise IndexFormatError(
            f"datasets row for {provenance!r} states {word!r} where {unknown!r}, "
            f"outside {sorted(AXIS_LABELS)!r}"
        )
    if (invalid := invalid_when_named_entry(list(where.values()))) is not None:
        raise IndexFormatError(
            f"datasets row for {provenance!r} states {word!r} where a party is "
            f"{list(where.values())[invalid]!r}, which is not a non-empty lowercase string"
        )
    return where


def _parse_datasets(rows: Sequence[Mapping]) -> list[DatasetDeclaration]:
    """Parses `datasets` rows, failing closed on a row missing a field, an
    `absent_observation` value outside `{"zero", "unknown"}`, an `account`
    outside `dfo_platform.roles.FlowAccount`, an `opt_in` that is not a boolean,
    a `rank` that is not an integer, an empty `label`, an empty `description`,
    a `stated_words` row `_parse_stated_words` refuses or a `party_words` row
    `_parse_party_words` refuses (both keys are optional), a `house_default_for`
    (optional) that is not a list of `HouseDefaultCase` values, a `ratio_base_axis`
    (optional, absent reads as `None`) outside `AXIS_LABELS`, or a
    `provenance` or `rank` repeated across rows. Each raises
    `IndexFormatError` naming the offending row (or the repeated value)."""
    datasets: list[DatasetDeclaration] = []
    seen: set[str] = set()
    ranks: dict[int, str] = {}
    for row in rows:
        try:
            dataset = DatasetDeclaration(
                provenance=row["provenance"],
                label=row["label"],
                description=row["description"],
                opt_in=row["opt_in"],
                absent_observation=row["absent_observation"],
                account=row["account"],
                rank=row["rank"],
                invariants=dict(row["invariants"]),
                stated_words=_parse_stated_words(row["provenance"], row.get("stated_words", ())),
                party_words=_parse_party_words(row["provenance"], row.get("party_words", ())),
                house_default_for=tuple(row.get("house_default_for", ())),
                ratio_base_axis=row.get("ratio_base_axis"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexFormatError(f"malformed datasets row {row!r}: {exc}") from exc
        if dataset.absent_observation not in _ABSENT_OBSERVATION_VALUES:
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries "
                f"absent_observation={dataset.absent_observation!r}, outside the vocabulary "
                f"{sorted(_ABSENT_OBSERVATION_VALUES)!r}"
            )
        if dataset.account not in _FLOW_ACCOUNTS:
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries "
                f"account={dataset.account!r}, outside the vocabulary "
                f"{sorted(_FLOW_ACCOUNTS)!r}"
            )
        if not isinstance(dataset.opt_in, bool):
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries opt_in={dataset.opt_in!r}, "
                "which is not a boolean"
            )
        if not isinstance(dataset.rank, int) or isinstance(dataset.rank, bool):
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries rank={dataset.rank!r}, "
                "which is not an integer"
            )
        if not set(dataset.house_default_for) <= _HOUSE_DEFAULT_CASES:
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries "
                f"house_default_for={list(dataset.house_default_for)!r}, outside the "
                f"vocabulary {sorted(_HOUSE_DEFAULT_CASES)!r}"
            )
        if dataset.ratio_base_axis is not None and dataset.ratio_base_axis not in AXIS_LABELS:
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries "
                f"ratio_base_axis={dataset.ratio_base_axis!r}, outside {sorted(AXIS_LABELS)!r}"
            )
        if not dataset.label:
            raise IndexFormatError(f"datasets row for {dataset.provenance!r} carries no label")
        if not dataset.description:
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries no description"
            )
        if dataset.provenance in seen:
            raise IndexFormatError(
                f"provenance {dataset.provenance!r} is declared more than once in datasets"
            )
        seen.add(dataset.provenance)
        if (holder := ranks.setdefault(dataset.rank, dataset.provenance)) != dataset.provenance:
            raise IndexFormatError(
                f"datasets rows for {holder!r} and {dataset.provenance!r} share "
                f"rank {dataset.rank!r}"
            )
        datasets.append(dataset)
    return datasets


def _check_datasets(datasets: Sequence[DatasetDeclaration], statvars: Sequence[StatVarRow]) -> None:
    """Checks the dataset declarations against the StatVar rows.

    Raises `IndexFormatError` when a row names a provenance no dataset
    declares, a row of a dataset does not carry one of its declared
    `invariants` explicitly (an absent property fails whatever its
    `absent_reading`), a declared dataset has no row, or a dataset states a
    word (`DatasetDeclaration.stated_words`) or names a party for one
    (`DatasetDeclaration.party_words`) of a measured property none of its rows
    has, or two datasets claim one kind of question as its house default
    (`DatasetDeclaration.house_default_for`). A row with no provenance belongs to no
    dataset and passes."""
    claimed: dict[str, str] = {}
    for dataset in datasets:
        for case in dataset.house_default_for:
            if (holder := claimed.setdefault(case, dataset.provenance)) != dataset.provenance:
                raise IndexFormatError(
                    f"datasets {holder!r} and {dataset.provenance!r} both claim "
                    f"house_default_for {case!r}"
                )
    invariants = {dataset.provenance: dataset.invariants for dataset in datasets}
    populated: set[str] = set()
    measured: dict[str, set[str]] = {}
    for row in statvars:
        for provenance in row.provenances:
            if provenance not in invariants:
                raise IndexFormatError(
                    f"StatVar {row.dcid!r} carries provenance {provenance!r}, which no "
                    f"dataset declares (declared: {sorted(invariants)!r})"
                )
            for prop, code in invariants[provenance].items():
                if row.constraint_properties.get(prop) != code:
                    raise IndexFormatError(
                        f"StatVar {row.dcid!r} of dataset {provenance!r} carries "
                        f"{prop}={row.constraint_properties.get(prop)!r}, but the dataset "
                        f"declares {prop}={code!r} as an invariant"
                    )
            populated.add(provenance)
            measured.setdefault(provenance, set()).add(row.measured_property)
    empty = sorted(invariants.keys() - populated)
    if empty:
        raise IndexFormatError(f"declared dataset(s) {empty!r} have no StatVar row")
    for dataset in datasets:
        for stated in dataset.stated_words:
            if stated.measured_property not in measured[dataset.provenance]:
                raise IndexFormatError(
                    f"dataset {dataset.provenance!r} states {stated.word!r} of "
                    f"{stated.measured_property!r}, a measured property none of its rows has "
                    f"(has: {sorted(measured[dataset.provenance])!r})"
                )
        for named in dataset.party_words:
            if named.measured_property not in measured[dataset.provenance]:
                raise IndexFormatError(
                    f"dataset {dataset.provenance!r} names a party for {named.word!r} of "
                    f"{named.measured_property!r}, a measured property none of its rows has "
                    f"(has: {sorted(measured[dataset.provenance])!r})"
                )


def _check_party_words(datasets: Sequence[DatasetDeclaration], parties: Sequence[PartyRow]) -> None:
    """Raises `IndexFormatError` when a dataset names a party for a word
    (`DatasetDeclaration.party_words`) that the `parties` block does not list for that
    dataset on that axis (`PartyRow.datasets`, `PartyRow.axes`)."""
    by_dcid = {party.dcid: party for party in parties}
    for dataset in datasets:
        for named in dataset.party_words:
            party = by_dcid.get(named.party)
            if (
                party is None
                or dataset.provenance not in party.datasets
                or named.axis not in party.axes
            ):
                raise IndexFormatError(
                    f"dataset {dataset.provenance!r} names {named.party!r} for {named.word!r} "
                    f"on {named.axis!r}, which the parties block does not list for that "
                    "dataset and axis"
                )


def _check_whole_totals(parties: Sequence[PartyRow], sets: Sequence[WholeTotalSet]) -> None:
    """Raises `IndexFormatError` for two whole totals whose markers cover a class in common on one
    dataset and axis, so every population a dataset's axis counts is counted once, and for a
    `WholeTotalSet` that names a phrase twice, a member that is no whole total on the set's
    axis, or members no dataset marks together. Members marked together in one dataset cover
    disjoint classes there by the first rule, so their sum counts no party twice."""
    by_dcid = {party.dcid: party for party in parties}
    held: dict[tuple[str, str], list[tuple[PartyRow, WholeMarker]]] = {}
    for party in parties:
        for marker in party.whole:
            held.setdefault((marker.dataset, party.axes[0]), []).append((party, marker))
    for (dataset, axis), rows in held.items():
        for index, (first, first_marker) in enumerate(rows):
            for second, second_marker in rows[index + 1 :]:
                if shared := sorted(set(first_marker.covers) & set(second_marker.covers)):
                    raise IndexFormatError(
                        f"whole totals {first.dcid!r} and {second.dcid!r} both cover {shared!r} "
                        f"on {axis!r} in {dataset!r}"
                    )
    phrases: set[str] = set()
    for whole_set in sets:
        if (phrase := fold_text(whole_set.phrase)) in phrases:
            raise IndexFormatError(f"whole total set {whole_set.phrase!r} is declared twice")
        phrases.add(phrase)
        members = [by_dcid.get(dcid) for dcid in whole_set.members]
        for dcid, member in zip(whole_set.members, members, strict=True):
            if member is None or not member.whole or member.axes != (whole_set.axis,):
                raise IndexFormatError(
                    f"whole total set {whole_set.phrase!r} names {dcid!r}, which is no whole "
                    f"total on {whole_set.axis!r}"
                )
        if not marked_datasets(member for member in members if member is not None):
            raise IndexFormatError(
                f"whole total set {whole_set.phrase!r} has members no dataset marks together"
            )


def _parse_code(payload: Mapping) -> CodeRow:
    return CodeRow(
        dcid=payload["dcid"],
        code_system=payload["code_system"],
        code=payload["code"],
        label=payload["label"],
        synonyms=tuple(payload["synonyms"]),
        short_name=payload["short_name"],
        parent=payload["parent"],
        definition=payload.get("definition"),
    )


def _parse_statvar(payload: Mapping) -> StatVarRow:
    return StatVarRow(
        dcid=payload["dcid"],
        population_type=payload["population_type"],
        measured_property=payload["measured_property"],
        stat_type=payload["stat_type"],
        measurement_qualifier=payload["measurement_qualifier"],
        measurement_denominator=payload["measurement_denominator"],
        observation_properties=tuple(payload["observation_properties"]),
        constraint_properties=dict(payload["constraint_properties"]),
        provenances=tuple(payload["provenances"]),
    )


def _parse_source_code(payload: Mapping) -> SourceCodeRow:
    return SourceCodeRow(
        provenance=payload["provenance"],
        code=payload["code"],
        dcids=tuple(payload["dcids"]),
    )


def _parse_value_parents(rows: Iterable[Mapping]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for row in rows:
        if row["dcid"] in parents:
            raise DuplicateDcid(f"value parent dcid {row['dcid']!r} appears more than once")
        parents[row["dcid"]] = row["parent"]
    return parents


def _check_source_codes(rows: Sequence[SourceCodeRow], statvars: Sequence[StatVarRow]) -> None:
    """Checks the `source_codes` rows against the StatVar rows.

    Raises `IndexFormatError` for a code that is empty or not upper-cased and stripped, a
    `(provenance, code)` pair on two rows, a row naming no StatVar, and a dcid that is no
    StatVar row of the index or whose row does not carry the code's provenance.
    Search answers a code with these rows, so each must be a StatVar it serves from that source."""
    provenances = {row.dcid: row.provenances for row in statvars}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not row.code or row.code != row.code.strip().upper():
            raise IndexFormatError(
                f"source code {row.code!r} under {row.provenance!r} is not a non-empty, "
                "stripped, upper-case code"
            )
        if (row.provenance, row.code) in seen:
            raise IndexFormatError(
                f"source code {row.code!r} under {row.provenance!r} appears more than once"
            )
        seen.add((row.provenance, row.code))
        if not row.dcids:
            raise IndexFormatError(
                f"source code {row.code!r} under {row.provenance!r} names no StatVar"
            )
        for dcid in row.dcids:
            if dcid not in provenances:
                raise IndexFormatError(
                    f"source code {row.code!r} under {row.provenance!r} names {dcid!r}, which "
                    "is no StatVar row of the index"
                )
            if row.provenance not in provenances[dcid]:
                raise IndexFormatError(
                    f"source code {row.code!r} under {row.provenance!r} names {dcid!r}, whose "
                    f"row carries provenances {list(provenances[dcid])!r}"
                )


def _check_partly_stated_values(
    rows: Sequence[PartlyStatedValue], statvars: Sequence[StatVarRow]
) -> None:
    """Checks the `partly_stated_values` rows against the StatVar rows.

    Raises `IndexFormatError` for a `(provenance, property, value)` triple on two rows, and
    for a row with no five-tuple where the dataset has both a row stating the value and a row
    stating nothing on the property (`partly_stated_five_tuples`). A dataset that states the
    value on every row of a five-tuple or on none holds no partly stated value there."""
    seen: set[tuple[str, str, str]] = set()
    for declared in rows:
        key = (declared.provenance, declared.property_dcid, declared.value_dcid)
        if key in seen:
            raise IndexFormatError(
                f"partly stated value {declared.property_dcid}={declared.value_dcid} under "
                f"{declared.provenance!r} appears more than once"
            )
        seen.add(key)
        if not partly_stated_five_tuples(statvars, declared):
            raise IndexFormatError(
                f"{declared.provenance!r} states {declared.property_dcid}="
                f"{declared.value_dcid} on every row or on none of each five-tuple, so it is "
                "not partly stated; delete the declaration"
            )


def _check_implied_values(
    rows: Sequence[ImpliedValue],
    declarations: Sequence[CodeSystemDeclaration],
    codes: Sequence[CodeRow],
    statvars: Sequence[StatVarRow],
) -> None:
    """Checks the `implied_values` rows against what the index holds of each.

    Raises `IndexFormatError` for two rows on one measure's property, for a value that is
    not a code of the property where a code system of the index is the property, and for a
    measure the index holds rows of where no row states the value. A measure the index
    holds no row of, such as a catalogue family, is checked against the live graph by
    `tests/code_index/test_implied_values_live.py`."""
    systems = {declaration.code_system for declaration in declarations}
    seen: set[tuple[str, str, str]] = set()
    for declared in rows:
        prop, value = declared.property_dcid, declared.value_dcid
        measure = f"{declared.population_type}/{declared.measured_property}"
        key = (declared.population_type, declared.measured_property, prop)
        if key in seen:
            raise IndexFormatError(f"{measure} declares an implied {prop} more than once")
        seen.add(key)
        if prop in systems and not any(
            code.code_system == prop and code.dcid == value for code in codes
        ):
            raise IndexFormatError(
                f"{measure} declares {prop}={value} implied, but {value!r} is not a code of {prop}"
            )
        measure_rows = [
            row
            for row in statvars
            if (row.population_type, row.measured_property)
            == (declared.population_type, declared.measured_property)
        ]
        if measure_rows and not any(
            row.constraint_properties.get(prop) == value for row in measure_rows
        ):
            raise IndexFormatError(
                f"{measure} declares {prop}={value} implied, but no row of it states the value"
            )


def _parse_measure_axis(payload: Mapping) -> MeasureAxis:
    return MeasureAxis(
        axis=payload["axis"],
        kind=payload["kind"],
        role=payload["role"],
        prefer=tuple(payload["prefer"]),
        label=payload["label"],
    )


def _parse_measure_value(payload: Mapping) -> MeasureValueRow:
    return MeasureValueRow(
        dcid=payload["dcid"],
        axis=payload["axis"],
        label=payload["label"],
        synonyms=tuple(payload["synonyms"]),
        short_name=payload["short_name"],
        matchable=payload["matchable"],
        same_reading_as=payload["same_reading_as"],
        definition=payload.get("definition"),
    )


def _parse_measure_partition_label(payload: Mapping) -> MeasurePartitionLabel:
    return MeasurePartitionLabel(
        axis_values=tuple((axis, value) for axis, value in payload["axis_values"]),
        label=payload["label"],
        named_values=tuple((prop, value) for prop, value in payload.get("named_values", ())),
        definition=payload.get("definition"),
    )


def _check_same_readings(values: Sequence[MeasureValueRow]) -> None:
    """Checks that every `same_reading_as` names a canonical value on the alias's own axis.

    Raises `IndexFormatError` for a target the block does not hold on that axis, or a target
    that is itself an alias, since a chain would leave the canonical reading ambiguous."""
    by_key = {(row.axis, row.dcid): row for row in values}
    for row in values:
        if row.same_reading_as is None:
            continue
        target = by_key.get((row.axis, row.same_reading_as))
        if target is None:
            raise IndexFormatError(
                f"measure value {row.dcid!r} reads as {row.same_reading_as!r}, which is not a "
                f"{row.axis!r} measure value"
            )
        if target.same_reading_as is not None:
            raise IndexFormatError(
                f"measure value {row.dcid!r} reads as {target.dcid!r}, which itself reads as "
                f"{target.same_reading_as!r}"
            )


def parse_measure_axes(payload: Mapping, *, source: object = "<document>") -> MeasureIndex:
    """Parses the measure-axis blocks of an export document into a `MeasureIndex`.

    Missing blocks yield an empty index. Malformed data, or a `same_reading_as` that names no
    canonical value on its axis, raises `IndexFormatError`."""
    try:
        axes = tuple(_parse_measure_axis(row) for row in payload.get("measure_axes", ()))
        values = tuple(_parse_measure_value(row) for row in payload.get("measure_values", ()))
        partition_labels = tuple(
            _parse_measure_partition_label(row)
            for row in payload.get("measure_partition_labels", ())
        )
        nouns = tuple(
            _parse_measure_partition_label(row) for row in payload.get("measure_nouns", ())
        )
    except (KeyError, TypeError) as exc:
        raise IndexFormatError(f"malformed code_index export at {source!r}: {exc}") from exc
    _check_same_readings(values)
    return MeasureIndex(axes=axes, values=values, partition_labels=partition_labels, nouns=nouns)


__all__ = [
    "IndexFormatError",
    "parse_index",
    "parse_measure_axes",
]
