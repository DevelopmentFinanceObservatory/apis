"""Read-only in-memory catalogue of classification codes and constrained StatVars.

`load_index` reads a JSON export. CRS behaviour comes from its declarations."""

from __future__ import annotations

from code_index.loader import IndexFormatError, default_index_path, load_index, parse_index
from code_index.members import (
    DatasetFamily,
    FamilyQuery,
    SourceNotInFamily,
    absent_readings,
    antichain_minimal_rows,
    elect_dataset,
    identity_family_query,
    index_covers_slot_keys,
)
from code_index.query import (
    CodeIndex,
    CrossSystemParent,
    CyclicParent,
    DatasetsShareFiveTuple,
    DuplicateDcid,
    InvalidDeclaredDefault,
    dedupe_first_seen,
)
from code_index.types import (
    AbsentObservation,
    AbsentReading,
    Bound,
    BoundSet,
    CodeRow,
    CodeSystemDeclaration,
    ConstraintQuery,
    DatasetDeclaration,
    DeclaredDefault,
    IndexStamp,
    LevelOption,
    Open,
    PropertyBinding,
    StatVarRow,
    summable_axes,
)

__all__ = [
    "AbsentObservation",
    "AbsentReading",
    "Bound",
    "BoundSet",
    "CodeIndex",
    "CodeRow",
    "CodeSystemDeclaration",
    "ConstraintQuery",
    "CrossSystemParent",
    "CyclicParent",
    "DatasetDeclaration",
    "DatasetFamily",
    "DatasetsShareFiveTuple",
    "DeclaredDefault",
    "DuplicateDcid",
    "FamilyQuery",
    "IndexFormatError",
    "IndexStamp",
    "InvalidDeclaredDefault",
    "LevelOption",
    "Open",
    "PropertyBinding",
    "SourceNotInFamily",
    "StatVarRow",
    "absent_readings",
    "antichain_minimal_rows",
    "dedupe_first_seen",
    "default_index_path",
    "elect_dataset",
    "identity_family_query",
    "index_covers_slot_keys",
    "load_index",
    "parse_index",
    "summable_axes",
]
