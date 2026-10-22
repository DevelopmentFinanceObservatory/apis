"""DFO's hand-written declarations, the one input of the DFO layer beside the platform parts.

`Declarations` holds what `graph_release.dfo` and `graph_release.platform` pass on: the
code-index declarations `code_index.export` takes as arguments, `ROLE_AXES` and the authored
rows. `declarations_json` serialises them with the tables `code_index.export` and
`dc_graph.visible_arcs` read from their own modules (the synonym tables, short names, concept
groups, measure value texts, slot plain names and `IMPORT_ADDED_CONSTRAINT_PROPERTIES`), so the declarations hash
moves when any input of the DFO layer does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from code_index import declarations as code_index_declarations
from code_index.declarations import CodeSystemConfig, KindOverride, TotalGroup
from code_index.types import (
    BroadReading,
    DatasetDeclaration,
    ImpliedValue,
    MeasureAxis,
    MeasureClassDeclaration,
    MeasurePartitionLabel,
    PartlyStatedValue,
    PartyKind,
    ShapeSlotDefault,
    StatTypeWordDeclaration,
)
from dc_graph.visible_arcs import IMPORT_ADDED_CONSTRAINT_PROPERTIES
from dfo_platform.roles import ROLE_AXES, RoleAxis
from graph_release.authored import AUTHORED, Authored

_MODULE_TABLES: Final[tuple[str, ...]] = (
    "CODE_SHORT_NAMES",
    "CODE_SYNONYMS",
    "CONCEPT_GROUPS",
    "DECLARED_MEASURE_VALUES",
    "MEASURE_VALUE_TEXT",
    "MEASURE_WORD_SYNONYMS_KEPT",
    "SLOT_PLAIN_NAMES",
)
"""The `code_index.declarations` tables `code_index.export` reads from the module rather than
from its arguments."""


@dataclass(frozen=True)
class Declarations:
    """The declarations one release is built with. `DECLARATIONS` holds the committed ones. A test
    may replace fields to build from a smaller graph."""

    code_systems: Sequence[CodeSystemConfig]
    datasets: Sequence[DatasetDeclaration]
    listed_datasets: Sequence[str]
    party_type_kinds: Mapping[str, PartyKind | None]
    party_kind_overrides: Mapping[str, KindOverride]
    total_groups: Mapping[str, TotalGroup]
    measure_axes: Sequence[MeasureAxis]
    measure_partition_labels: Sequence[MeasurePartitionLabel]
    measure_nouns: Sequence[MeasurePartitionLabel]
    shape_slot_defaults: Sequence[ShapeSlotDefault]
    broad_readings: Sequence[BroadReading]
    partly_stated_values: Sequence[PartlyStatedValue]
    implied_values: Sequence[ImpliedValue]
    measure_classes: Sequence[MeasureClassDeclaration]
    variable_aliases: Mapping[str, str]
    stat_type_words: Sequence[StatTypeWordDeclaration]
    role_axes: Mapping[str, RoleAxis]
    authored: Authored

    def declarations_json(self) -> bytes:
        """Every declaration as JSON with sorted keys: the code-index declarations under
        `code_index`, keyed by their constant names in lower case, then `role_axes`,
        `import_added_constraint_properties` and `authored`."""
        code_index = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in {"role_axes", "authored"}
        }
        code_index.update(
            {name.lower(): getattr(code_index_declarations, name) for name in _MODULE_TABLES}
        )
        document = {
            "code_index": code_index,
            "role_axes": self.role_axes,
            "import_added_constraint_properties": IMPORT_ADDED_CONSTRAINT_PROPERTIES,
            "authored": self.authored,
        }
        text = json.dumps(_plain(document), sort_keys=True, ensure_ascii=False, indent=1)
        return (text + "\n").encode("utf-8")

    def declarations_hash(self) -> str:
        """The SHA-256 hex digest of `declarations_json`."""
        return hashlib.sha256(self.declarations_json()).hexdigest()


def _plain(value: object) -> object:
    """`value` as JSON values: a dataclass as an object of its fields, a mapping as an object, a
    set as a sorted array, a compiled pattern as its pattern and flags. Raises `TypeError` for
    anything else that JSON cannot hold, and for a mapping key that is not a string."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": int(value.flags)}
    if isinstance(value, Mapping):
        keys = [key for key in value if not isinstance(key, str)]
        if keys:
            raise TypeError(f"declaration mapping keys must be strings; got {keys[:5]!r}")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, frozenset | set):
        return sorted(cast("list[str]", [_plain(item) for item in value]))
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, str | int | float):
        return value
    raise TypeError(f"cannot serialise a declaration of type {type(value).__name__}")


DECLARATIONS: Final[Declarations] = Declarations(
    code_systems=code_index_declarations.CODE_SYSTEMS,
    datasets=code_index_declarations.DATASETS,
    listed_datasets=code_index_declarations.LISTED_DATASETS,
    party_type_kinds=code_index_declarations.PARTY_TYPE_KINDS,
    party_kind_overrides=code_index_declarations.PARTY_KIND_OVERRIDES,
    total_groups=code_index_declarations.TOTAL_GROUPS,
    measure_axes=code_index_declarations.MEASURE_AXES,
    measure_partition_labels=code_index_declarations.MEASURE_PARTITION_LABELS,
    measure_nouns=code_index_declarations.MEASURE_NOUNS,
    shape_slot_defaults=code_index_declarations.SHAPE_SLOT_DEFAULTS,
    broad_readings=code_index_declarations.BROAD_READINGS,
    partly_stated_values=code_index_declarations.PARTLY_STATED_VALUES,
    implied_values=code_index_declarations.IMPLIED_VALUES,
    measure_classes=code_index_declarations.MEASURE_CLASSES,
    variable_aliases=code_index_declarations.VARIABLE_ALIASES,
    stat_type_words=code_index_declarations.STAT_TYPE_WORDS,
    role_axes=ROLE_AXES,
    authored=AUTHORED,
)


__all__ = ["DECLARATIONS", "Declarations"]
