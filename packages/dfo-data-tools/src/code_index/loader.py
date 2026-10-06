"""Loads `CodeIndex` and measure-axis data from JSON exports.

The index loads through `importlib.resources`, including from an installed wheel.
Its repetitive fields compress roughly ninety-fold. `load_index` accepts plain or gzip
JSON, detected by the magic number. `load_measure_axes` returns a separate `MeasureIndex`."""

from __future__ import annotations

import gzip
import importlib.resources
import json
from collections.abc import Mapping, Sequence
from importlib.resources.abc import Traversable
from pathlib import Path

from code_index.query import CodeIndex
from code_index.types import (
    AbsentObservation,
    CodeRow,
    CodeSystemDeclaration,
    DatasetDeclaration,
    DeclaredDefault,
    IndexStamp,
    MeasureAxis,
    MeasureIndex,
    MeasurePartitionLabel,
    MeasureValueRow,
    StatVarRow,
    invalid_when_named_entry,
)

_PACKAGE = __package__
_DATA_DIR = "data"
_GZIP_MAGIC = b"\x1f\x8b"
_INDEX_FILENAME = "index.json.gz"
_ABSENT_OBSERVATION_VALUES: frozenset[AbsentObservation] = frozenset({"zero", "unknown"})


def default_index_path() -> Traversable:
    """The packaged `data/index.json.gz` path."""
    return importlib.resources.files(_PACKAGE) / _DATA_DIR / _INDEX_FILENAME


class IndexFormatError(ValueError):
    """A `CodeIndex` export has a missing required key or a value of the wrong type."""


def load_index(source: Traversable | Path | str) -> CodeIndex:
    """Reads a `CodeIndex` export and parses it with `parse_index`."""
    return parse_index(_read_json(source), source=source)


def parse_index(payload: Mapping, *, source: object = "<document>") -> CodeIndex:
    """Parses and validates a `CodeIndex` export document. `source` names it in errors.

    Raises `IndexFormatError` for missing or malformed fields, invalid
    dataset declarations, dataset declarations the StatVar rows contradict
    (`_check_datasets`), or an invalid `default.when_named`. Duplicate
    dcids or code pairs raise `DuplicateDcid`. Parent cycles raise
    `CyclicParent`. Cross-system parents raise `CrossSystemParent`.
    Invalid defaults raise `InvalidDeclaredDefault` when their code is
    outside the declaring system or `when_bound_under` names a missing or
    same-system code."""
    try:
        stamp = _parse_stamp(payload["stamp"])
        declarations = [_parse_declaration(row) for row in payload["declarations"]]
        codes = [_parse_code(row) for row in payload["codes"]]
        statvars = [_parse_statvar(row) for row in payload["statvars"]]
        datasets = _parse_datasets(payload["datasets"])
    except (KeyError, TypeError) as exc:
        raise IndexFormatError(f"malformed code_index export at {source!r}: {exc}") from exc
    _check_datasets(datasets, statvars)
    return CodeIndex(
        stamp=stamp,
        declarations=declarations,
        codes=codes,
        statvars=statvars,
        datasets=datasets,
    )


def _read_json(source: Traversable | Path | str) -> dict:
    """Reads `source` as JSON, decompressing it first when it carries the
    gzip magic number. Reading the bytes and sniffing them keeps a plain
    `.json` fixture and the packaged `.json.gz` on one code path.
    """
    raw = Path(source).read_bytes() if isinstance(source, Path | str) else source.read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


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
    )


def _parse_datasets(rows: Sequence[Mapping]) -> list[DatasetDeclaration]:
    """Parses `datasets` rows, failing closed on a row missing a field, an
    `absent_observation` value outside `{"zero", "unknown"}`, an `opt_in`
    that is not a boolean, a `rank` that is not an integer, an empty `label`,
    an empty `description`, or a `provenance` or `rank` repeated across rows. Each raises
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
                rank=row["rank"],
                invariants=dict(row["invariants"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexFormatError(f"malformed datasets row {row!r}: {exc}") from exc
        if dataset.absent_observation not in _ABSENT_OBSERVATION_VALUES:
            raise IndexFormatError(
                f"datasets row for {dataset.provenance!r} carries "
                f"absent_observation={dataset.absent_observation!r}, outside the vocabulary "
                f"{sorted(_ABSENT_OBSERVATION_VALUES)!r}"
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
    `absent_reading`), or a declared dataset has no row. A row with no
    provenance belongs to no dataset and passes."""
    invariants = {dataset.provenance: dataset.invariants for dataset in datasets}
    populated: set[str] = set()
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
    empty = sorted(invariants.keys() - populated)
    if empty:
        raise IndexFormatError(f"declared dataset(s) {empty!r} have no StatVar row")


def _parse_code(payload: Mapping) -> CodeRow:
    return CodeRow(
        dcid=payload["dcid"],
        code_system=payload["code_system"],
        code=payload["code"],
        label=payload["label"],
        synonyms=tuple(payload["synonyms"]),
        short_name=payload["short_name"],
        parent=payload["parent"],
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


def _parse_measure_axis(payload: Mapping) -> MeasureAxis:
    return MeasureAxis(
        axis=payload["axis"],
        kind=payload["kind"],
        role=payload["role"],
        prefer=tuple(payload["prefer"]),
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
    )


def _parse_measure_partition_label(payload: Mapping) -> MeasurePartitionLabel:
    return MeasurePartitionLabel(
        axis_values=tuple((axis, value) for axis, value in payload["axis_values"]),
        label=payload["label"],
        named_values=tuple((prop, value) for prop, value in payload.get("named_values", ())),
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


def load_measure_axes(source: Traversable | Path | str) -> MeasureIndex:
    """Reads an export and parses its measure-axis blocks with `parse_measure_axes`."""
    return parse_measure_axes(_read_json(source), source=source)


__all__ = [
    "IndexFormatError",
    "default_index_path",
    "load_index",
    "load_measure_axes",
    "parse_index",
    "parse_measure_axes",
]
