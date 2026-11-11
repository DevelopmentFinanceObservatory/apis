"""Loads the versioned presentation catalogue from `dfo_spec_kit_components`.

Schemas, intents, fixtures, and the digest manifest ship as package data. The envelope
references the description and step schemas of `dfo_descriptions` by `$id`; those load from
that package, and the manifest's `referenced_schemas` digests pin their bytes. `CatalogueRelease` lives here because `assembler.ports` imports it for the
`ReleaseRegistry` protocol's return type.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from assembler.catalogue.constraints import CONSTRAINT_REGISTRY, ConstraintSpec

_MEMBERS = ("line_chart", "stat_card", "bar_chart", "table", "note_narration", "source_card")
_DESCRIPTION_SCHEMA_FILES = ("description.schema.json", "steps.schema.json")
"""The `dfo_descriptions` schemas the envelope's `description` and `source_ref` reference."""
_NON_CLEARABLE_FIELDS = frozenset({"type", "frame_ref", "select"})
"""Tolerant schemas keep `type`, `frame_ref`, and `select` non-null.

`type` and `frame_ref` are create-only. A revise may omit or replace `select`."""


@dataclass(frozen=True)
class CatalogueRelease:
    """Immutable, digest-verified release. Its fields define the `ReleaseRegistry` contract.

    `referenced_schemas` holds the schemas of other kit packages the envelope references by
    `$id`, keyed by `$id`."""

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
    line_chart_max_series: int
    """Series a `line_chart` draws, the `maxItems` of its select's one array."""
    referenced_schemas: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    write_schemas: Mapping[str, Mapping[str, object]] = field(init=False)
    """`intent_schemas` as a write validates against them: `line_chart`'s without its select
    array's `maxItems` (`without_select_list_cap`), the series cap a write checks only where
    it applies. Derived from `intent_schemas` on construction, so `dataclasses.replace`
    derives it again."""
    write_schemas_tolerant: Mapping[str, Mapping[str, object]] = field(init=False)
    """`intent_schemas_tolerant` as a write validates against them, as `write_schemas`."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "write_schemas", _write_schemas(self.intent_schemas))
        object.__setattr__(
            self, "write_schemas_tolerant", _write_schemas(self.intent_schemas_tolerant)
        )


class CatalogueIntegrityError(Exception):
    """A bundle artifact's or referenced schema's digest differs from its `manifest.json` entry.
    `path` is the artifact's path in the bundle, or the referenced schema's `$id`."""

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


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _verify_digests(
    *, directory: Path, manifest: Mapping[str, object], referenced: Mapping[str, bytes]
) -> None:
    """Checks each bundle artifact's bytes, then each schema the envelope references by `$id`
    (`referenced`, the installed bytes by `$id`), against the manifest's digests. A schema the
    manifest names that is not installed, or one installed that it does not name, refuses too."""
    artifacts = cast("Mapping[str, str]", manifest["artifacts"])
    for relative_path, expected in artifacts.items():
        actual = _digest((directory / relative_path).read_bytes())
        if actual != expected:
            raise CatalogueIntegrityError(path=relative_path, expected=expected, actual=actual)
    digests = cast("Mapping[str, str]", manifest["referenced_schemas"])
    for schema_id in sorted(digests.keys() | referenced.keys()):
        expected = digests.get(schema_id, "absent")
        actual = _digest(referenced[schema_id]) if schema_id in referenced else "absent"
        if actual != expected:
            raise CatalogueIntegrityError(path=schema_id, expected=expected, actual=actual)


def _default_catalogue_dir() -> Path:
    """Returns the installed bundle path, resolved lazily so the package imports without it."""
    from importlib import resources

    return cast("Path", resources.files("dfo_spec_kit_components") / "catalogue")


def _installed_description_schemas() -> dict[str, bytes]:
    """The bytes of the installed `dfo_descriptions` description and step schemas, by `$id`."""
    from importlib import resources

    directory = resources.files("dfo_descriptions") / "schemas"
    contents = [(directory / name).read_bytes() for name in _DESCRIPTION_SCHEMA_FILES]
    return {cast("str", json.loads(content)["$id"]): content for content in contents}


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


class CatalogueShapeError(Exception):
    """A schema in the bundle lacks a value the assembler reads from it."""


def _select_value_branches(schema: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    """The `anyOf` branches a select value of a member's intent schema may take. A tolerant
    schema holds `select` as the strict one does (`_NON_CLEARABLE_FIELDS`)."""
    properties = cast("Mapping[str, Mapping[str, object]]", schema.get("properties", {}))
    select = properties.get("select", {})
    values = cast("Mapping[str, object]", select.get("additionalProperties", {}))
    return cast("Sequence[Mapping[str, object]]", values.get("anyOf", ()))


def _line_chart_max_series(*, tolerant: Mapping[str, object]) -> int:
    """The `maxItems` of the array a `line_chart` select may bind one dimension to."""
    caps = [
        branch["maxItems"]
        for branch in _select_value_branches(tolerant)
        if branch.get("type") == "array" and isinstance(branch.get("maxItems"), int)
    ]
    if len(caps) != 1:
        msg = "the line_chart select schema names no single array maxItems"
        raise CatalogueShapeError(msg)
    return cast(int, caps[0])


def without_select_list_cap(schema: Mapping[str, object]) -> dict[str, object]:
    """`schema`, a `line_chart` intent schema, without the `maxItems` of its select array.

    The array's `maxItems` is the line chart's series cap
    (`CatalogueRelease.line_chart_max_series`), which a write holds a select to only where it
    creates a line chart or changes its select, so the assembler validates against this form
    and checks the cap itself. Every other rule of the schema is kept."""
    properties = cast("Mapping[str, Mapping[str, object]]", schema["properties"])
    select = properties["select"]
    values = cast("Mapping[str, object]", select["additionalProperties"])
    branches = [
        {key: value for key, value in branch.items() if key != "maxItems"}
        if branch.get("type") == "array"
        else branch
        for branch in _select_value_branches(schema)
    ]
    return {
        **schema,
        "properties": {
            **properties,
            "select": {**select, "additionalProperties": {**values, "anyOf": branches}},
        },
    }


def _write_schemas(
    schemas: Mapping[str, Mapping[str, object]],
) -> Mapping[str, Mapping[str, object]]:
    """`schemas` with `line_chart`'s select list cap dropped (`without_select_list_cap`)."""
    return {
        member: without_select_list_cap(schema) if member == "line_chart" else schema
        for member, schema in schemas.items()
    }


def load_release(*, catalogue_dir: Path | None = None) -> CatalogueRelease:
    """Loads and digest-verifies a catalogue bundle, and the schemas it references, into
    `CatalogueRelease`.

    Loads once during surface or door construction. Imports stay side-effect-free."""
    directory = catalogue_dir or _default_catalogue_dir()
    manifest = _load_json(directory / "manifest.json")
    referenced = _installed_description_schemas()
    _verify_digests(directory=directory, manifest=manifest, referenced=referenced)

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
        line_chart_max_series=_line_chart_max_series(
            tolerant=intent_schemas_tolerant["line_chart"]
        ),
        referenced_schemas={
            schema_id: json.loads(content) for schema_id, content in referenced.items()
        },
    )


class BakedReleaseRegistry:
    """JSON-backed `ReleaseRegistry` implementation.

    It satisfies the protocol structurally to avoid an import cycle, and
    loads and verifies its release at construction."""

    def __init__(self, *, catalogue_dir: Path | None = None) -> None:
        self._release = load_release(catalogue_dir=catalogue_dir)

    def pinned(self) -> CatalogueRelease:
        return self._release
