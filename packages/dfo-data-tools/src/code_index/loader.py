"""Parses a code index document into a `CodeIndex` (`parse_index`) and its measure-axis
blocks into a separate `MeasureIndex` (`parse_measure_axes`)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import get_args

from code_index.query import CodeIndex, DuplicateDcid, partly_stated_five_tuples
from code_index.types import (
    PARTY_KINDS,
    AbsentObservation,
    CodeRow,
    CodeSystemDeclaration,
    DatasetDeclaration,
    DeclaredDefault,
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
    invalid_when_named_entry,
)
from dfo_platform.roles import AXIS_LABELS, FlowAccount

_ABSENT_OBSERVATION_VALUES: frozenset[AbsentObservation] = frozenset({"zero", "unknown"})
_FLOW_ACCOUNTS: frozenset[FlowAccount] = frozenset(get_args(FlowAccount.__value__))


class IndexFormatError(ValueError):
    """A `CodeIndex` export has a missing required key or a value of the wrong type."""


def parse_index(payload: Mapping, *, source: object = "<document>") -> CodeIndex:
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
    `shape_slot_defaults`, `partly_stated_values`, `measure_classes`, `variable_aliases`
    and `stat_type_words`, each reading as empty where absent; a measure class or stat-type
    pattern that does not compile, or a variable alias or a `no_vocabulary_terms` entry that
    is not a non-empty lowercase string, raises `IndexFormatError`. A value that two
    `value_parents` rows name raises `DuplicateDcid`, and a parent chain that cycles
    `CyclicParent`. A `source_codes` row the StatVar rows contradict raises `IndexFormatError`
    (`_check_source_codes`), as does a `partly_stated_values` row they contradict
    (`_check_partly_stated_values`). A kind outside `PartyKind`, a type declared twice, a
    `totals` on a party that is not a total, or a false `sums_members` on a party holding no
    group raises `IndexFormatError`. A code row without a `definition` carries none, as does
    a measure value or measure label in `parse_measure_axes`."""
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
        partly_stated_values = [
            _parse_partly_stated_value(row) for row in payload.get("partly_stated_values", ())
        ]
        measure_classes = [_parse_measure_class(row) for row in payload.get("measure_classes", ())]
        variable_aliases = _parse_variable_aliases(payload.get("variable_aliases", {}))
        stat_type_words = [_parse_stat_type_word(row) for row in payload.get("stat_type_words", ())]
        source_codes = [_parse_source_code(row) for row in payload.get("source_codes", ())]
        value_parents = _parse_value_parents(payload.get("value_parents", ()))
    except (KeyError, TypeError) as exc:
        raise IndexFormatError(f"malformed code_index export at {source!r}: {exc}") from exc
    _check_datasets(datasets, statvars)
    _check_party_words(datasets, parties)
    _check_source_codes(source_codes, [*statvars, *listed_statvars])
    _check_partly_stated_values(partly_stated_values, [*statvars, *listed_statvars])
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
        partly_stated_values=partly_stated_values,
        measure_classes=measure_classes,
        variable_aliases=variable_aliases,
        stat_type_words=stat_type_words,
        source_codes=source_codes,
        value_parents=value_parents,
    )


def _parse_shape_slot_default(row: Mapping) -> ShapeSlotDefault:
    return ShapeSlotDefault(
        population_type=row["population_type"],
        property_dcid=row["property"],
        value_dcid=row["value"],
    )


def _parse_partly_stated_value(row: Mapping) -> PartlyStatedValue:
    return PartlyStatedValue(
        provenance=row["provenance"],
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
    party that holds no group or that counts residuals. `residuals`, `sums_members`, `label`,
    `datasets` and `axes` are optional, and a row without `sums_members` reads it as true."""
    party = PartyRow(
        dcid=row["dcid"],
        kind=row["kind"],
        totals=row["totals"],
        residuals=tuple(row.get("residuals", ())),
        sums_members=row.get("sums_members", True),
        label=row.get("label"),
        datasets=tuple(row.get("datasets", ())),
        axes=tuple(row.get("axes", ())),
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
    return party


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
    )


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
    `_parse_party_words` refuses (both keys are optional), or a
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
    has. A row with no provenance belongs to no dataset and passes."""
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
