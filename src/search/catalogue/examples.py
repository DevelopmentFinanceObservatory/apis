"""Loads `examples.toml`, the example sentences a catalogue answer offers.

Loading checks each line's shape: a template with exactly one `{place}` slot
and no other braces, a StatVar, a `country/` default place, and an optional
coverage role. Whether the StatVar has data anywhere is `presence`'s check,
since it needs the origin catalogue and the coverage index.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from importlib import resources
from typing import Final, cast

from series_coverage.coverage_types import Role

PLACE_SLOT: Final[str] = "{place}"
"""The slot a template fills with a place's name."""

_EXAMPLES_PACKAGE = "search.catalogue"
_FILENAME = "examples.toml"
_KEYS: Final[frozenset[str]] = frozenset({"template", "stat_var", "place", "role"})
_ROLES: Final[frozenset[str]] = frozenset({"donor", "recipient"})


class ExamplesError(ValueError):
    """`examples.toml` does not parse or a line breaks its shape."""


@dataclass(frozen=True, slots=True)
class Example:
    """One line of `examples.toml`. `id` is its position in the file. `role`
    is the coverage role its place fills, set only for a StatVar the
    coverage index holds."""

    id: int
    template: str
    stat_var: str
    place: str
    role: Role | None = None

    def text(self, place_name: str) -> str:
        """The template filled with `place_name`."""
        return self.template.replace(PLACE_SLOT, place_name)


@dataclass(frozen=True, slots=True)
class ExampleSet:
    """The loaded lines and a digest of the file's bytes, which the job's
    build key carries so an edited file rebuilds."""

    examples: tuple[Example, ...]
    digest: str


def load_examples(raw: bytes | None = None) -> ExampleSet:
    """Parses `raw`, or the packaged `examples.toml` when `raw` is `None`.

    Raises `ExamplesError` naming the first line that breaks its shape or
    repeats an earlier template.
    """
    if raw is None:
        raw = (resources.files(_EXAMPLES_PACKAGE) / _FILENAME).read_bytes()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ExamplesError(f"{_FILENAME} does not parse: {exc}") from exc
    if set(document) != {"example"} or not isinstance(document["example"], list):
        raise ExamplesError(f"{_FILENAME} must hold only an `[[example]]` array")
    lines = cast("list[object]", document["example"])
    if not lines:
        raise ExamplesError(f"{_FILENAME} holds no examples")
    examples = tuple(_parse(index, line) for index, line in enumerate(lines))
    seen: set[str] = set()
    for example in examples:
        if example.template in seen:
            raise ExamplesError(f"example {example.id} repeats template {example.template!r}")
        seen.add(example.template)
    return ExampleSet(examples=examples, digest=hashlib.sha256(raw).hexdigest())


def _parse(index: int, line: object) -> Example:
    where = f"example {index}"
    if not isinstance(line, dict):
        raise ExamplesError(f"{where} is not a table")
    fields = cast("dict[str, object]", line)
    unknown = set(fields) - _KEYS
    if unknown:
        raise ExamplesError(f"{where} has unknown keys {sorted(unknown)}")
    template = _text(fields, "template", where=where)
    stat_var = _text(fields, "stat_var", where=where)
    place = _text(fields, "place", where=where)
    rest = template.replace(PLACE_SLOT, "", 1)
    if PLACE_SLOT not in template or "{" in rest or "}" in rest:
        raise ExamplesError(f"{where} template {template!r} needs exactly one {PLACE_SLOT} slot")
    if not place.startswith("country/"):
        raise ExamplesError(f"{where} place {place!r} is not a country/ dcid")
    role = fields.get("role")
    if role is not None and role not in _ROLES:
        raise ExamplesError(f"{where} role {role!r} is not one of {sorted(_ROLES)}")
    return Example(
        id=index,
        template=template,
        stat_var=stat_var,
        place=place,
        role=cast("Role | None", role),
    )


def _text(fields: dict[str, object], key: str, *, where: str) -> str:
    value = fields.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExamplesError(f"{where} needs a non-empty string {key!r}")
    return value


__all__ = ["PLACE_SLOT", "Example", "ExampleSet", "ExamplesError", "load_examples"]
