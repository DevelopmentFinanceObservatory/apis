"""Loads the versioned presentation catalogue from `dfo_spec_kit_components`.

Schemas, intents, fixtures, and the digest manifest ship as package data.
`CatalogueRelease` lives here because `assembler.ports` imports it for the
`ReleaseRegistry` protocol's return type.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from assembler.catalogue.constraints import CONSTRAINT_REGISTRY, ConstraintSpec

_MEMBERS = ("line_chart", "stat_card", "bar_chart", "table", "note_narration", "source_card")
_NON_CLEARABLE_FIELDS = frozenset({"type", "frame_ref", "select"})
"""Tolerant schemas keep `type`, `frame_ref`, and `select` non-null.

`type` and `frame_ref` are create-only. A revise may omit or replace `select`."""


@dataclass(frozen=True)
class CatalogueRelease:
    """Immutable, digest-verified release. Its fields define the `ReleaseRegistry` contract."""

    catalog_version: str
    schema_version: int
    envelope_schema: Mapping[str, object]
    component_schemas: Mapping[str, Mapping[str, object]]
    intent_schemas: Mapping[str, Mapping[str, object]]
    intent_schemas_tolerant: Mapping[str, Mapping[str, object]]
    minimal_intents: Mapping[str, Mapping[str, object]]
    grammar_members: tuple[str, ...]
    encoding_classes: Mapping[str, str]
    constraints: tuple[ConstraintSpec, ...]
    formatter_profile: Mapping[str, object]
    retry_modes: Mapping[str, str]


class CatalogueIntegrityError(Exception):
    """A bundle artifact's digest differs from its `manifest.json` entry."""

    def __init__(self, *, path: str, expected: str, actual: str) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"catalogue artifact {path!r} digest mismatch: manifest says {expected!r}, "
            f"bytes on disk hash to {actual!r}"
        )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _verify_digests(*, directory: Path, manifest: Mapping[str, object]) -> None:
    artifacts = cast("Mapping[str, str]", manifest["artifacts"])
    for relative_path, expected in artifacts.items():
        artifact_path = directory / relative_path
        actual = "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual != expected:
            raise CatalogueIntegrityError(path=relative_path, expected=expected, actual=actual)


def _default_catalogue_dir() -> Path:
    """Returns the installed bundle path, resolved lazily so the package imports without it."""
    from importlib import resources

    return cast("Path", resources.files("dfo_spec_kit_components") / "catalogue")


def _grammar_member_names(*, grammar: Mapping[str, object]) -> tuple[str, ...]:
    """Returns component names from the grammar's top-level `oneOf` refs."""
    one_of = cast("Sequence[Mapping[str, object]]", grammar["oneOf"])
    return tuple(cast("str", entry["$ref"]).rsplit("/", 1)[-1] for entry in one_of)


def _member_strict_schema(*, grammar: Mapping[str, object], member: str) -> dict[str, object]:
    """Extracts a member's grammar branch with the full `$defs` map so shared refs resolve."""
    defs = cast("Mapping[str, object]", grammar["$defs"])
    branch = cast("Mapping[str, object]", defs[member])
    return {
        "$schema": grammar.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$id": f"{grammar['$id']}/{member}",
        **branch,
        "$defs": defs,
    }


def _derive_tolerant_schema(*, strict: Mapping[str, object]) -> dict[str, object]:
    """Derives the pre-merge schema: fields are optional, and all but
    `_NON_CLEARABLE_FIELDS` admit `null` for RFC 7386 clears."""
    properties = cast("Mapping[str, object]", strict["properties"])
    tolerant_properties = {
        name: (schema if name in _NON_CLEARABLE_FIELDS else {"anyOf": [schema, {"type": "null"}]})
        for name, schema in properties.items()
    }
    return {**strict, "properties": tolerant_properties, "required": []}


def load_release(*, catalogue_dir: Path | None = None) -> CatalogueRelease:
    """Loads and digest-verifies a catalogue bundle into `CatalogueRelease`.

    Loads once during surface or door construction. Imports stay side-effect-free."""
    directory = catalogue_dir or _default_catalogue_dir()
    manifest = _load_json(directory / "manifest.json")
    _verify_digests(directory=directory, manifest=manifest)

    envelope_schema = _load_json(directory / "schemas" / "envelope.schema.json")
    component_schemas = {
        member: _load_json(directory / "schemas" / "component" / f"{member}.schema.json")
        for member in _MEMBERS
    }
    grammar = _load_json(directory / "schemas" / "intent.v0.json")
    grammar_members = _grammar_member_names(grammar=grammar)
    intent_schemas = {
        member: _member_strict_schema(grammar=grammar, member=member) for member in _MEMBERS
    }
    intent_schemas_tolerant = {
        member: _derive_tolerant_schema(strict=schema) for member, schema in intent_schemas.items()
    }
    minimal_intents = {
        member: _load_json(directory / "intents" / f"{member}.minimal.json") for member in _MEMBERS
    }

    return CatalogueRelease(
        catalog_version=cast("str", manifest["catalog_version"]),
        schema_version=cast("int", manifest["schema_version"]),
        envelope_schema=envelope_schema,
        component_schemas=component_schemas,
        intent_schemas=intent_schemas,
        intent_schemas_tolerant=intent_schemas_tolerant,
        minimal_intents=minimal_intents,
        grammar_members=grammar_members,
        encoding_classes=cast("Mapping[str, str]", manifest["encoding_classes"]),
        constraints=CONSTRAINT_REGISTRY,
        formatter_profile=cast("Mapping[str, object]", manifest["formatter_profile"]),
        retry_modes=cast("Mapping[str, str]", manifest["retry_modes"]),
    )


class BakedReleaseRegistry:
    """JSON-backed `ReleaseRegistry` implementation.

    It satisfies the protocol structurally to avoid an import cycle, and
    loads and verifies its release at construction."""

    def __init__(self, *, catalogue_dir: Path | None = None) -> None:
        self._release = load_release(catalogue_dir=catalogue_dir)

    def pinned(self) -> CatalogueRelease:
        return self._release
