"""Reads a skill's YAML front matter from the decoded text of its `SKILL.md`.

The block opens on the text's first line with `---` and closes at the next
line that is `---`. It parses with YAML's base loader, which reads every
scalar as a string, so `version: 1.10` stays `"1.10"`. A block with an anchor
or alias, or nesting deeper than `_MAX_NESTING_DEPTH` collections, is refused
before any value is built: aliases let a few bytes expand into an unbounded
value, and composing deep nesting recurses once per level. Any other failure
to parse is also `FrontMatterInvalid`, so a malformed block costs only its own
skill. The parsed mapping must hold a string `name` equal to the skill id and a
string `description`, and must serialise to JSON as the Skills API wire model's
`frontmatter` field, `dict[str, FrontMatterValue]`, so every admitted mapping
can be served.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml
from pydantic import TypeAdapter

from skills_index.types import FrontMatterValue

_FRONT_MATTER_BLOCK = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*\r?$", re.DOTALL | re.MULTILINE)


# Agent Skills front matter is a flat mapping with an occasional list or
# `metadata` map, two or three levels deep. 32 leaves wide room for that and
# stays far below the depth at which PyYAML's recursive composer exhausts
# CPython's default 1,000-frame limit, a few frames per level.
_MAX_NESTING_DEPTH = 32

# The wire model's `frontmatter` field has this type. Validating and dumping
# through it at build is the check the list door's JSON response repeats.
_WIRE_FRONT_MATTER: TypeAdapter[dict[str, FrontMatterValue]] = TypeAdapter(
    dict[str, FrontMatterValue]
)


class FrontMatterInvalid(ValueError):
    """The text carries no usable front matter. The message says why."""


@dataclass(frozen=True, slots=True)
class ParsedFrontMatter:
    """The whole parsed `mapping`, with its string `name` and `description`."""

    name: str
    description: str
    mapping: dict[str, FrontMatterValue]


def _refuse_unbounded_events(source: str) -> None:
    """Raises `FrontMatterInvalid` at the first anchor or alias, or at nesting past
    `_MAX_NESTING_DEPTH`. Reads parser events, which build no values."""
    depth = 0
    for event in yaml.parse(source, Loader=yaml.BaseLoader):
        # An alias event carries the name it refers to in `anchor`.
        if isinstance(event, yaml.NodeEvent) and event.anchor is not None:
            raise FrontMatterInvalid("front matter uses a YAML anchor or alias")
        if isinstance(event, yaml.CollectionStartEvent):
            depth += 1
            if depth > _MAX_NESTING_DEPTH:
                raise FrontMatterInvalid(
                    f"front matter nests deeper than {_MAX_NESTING_DEPTH} collections"
                )
        elif isinstance(event, yaml.CollectionEndEvent):
            depth -= 1


def read_front_matter(text: str, *, skill_id: str) -> ParsedFrontMatter:
    """Returns the front matter at the head of `text`.

    Raises:
        FrontMatterInvalid: the block is missing, does not parse, uses a YAML
            anchor or alias, or nests too deep, the parsed value is not a mapping,
            `name` or `description` is not a string, `name` differs from
            `skill_id`, or the mapping does not serialise as the wire model.
    """
    if (block := _FRONT_MATTER_BLOCK.match(text)) is None:
        raise FrontMatterInvalid("does not open with a '---' delimited front matter block")
    source = block.group(1)
    try:
        _refuse_unbounded_events(source)
        parsed = yaml.load(source, Loader=yaml.BaseLoader)
    except FrontMatterInvalid:
        raise
    except Exception as exc:
        # The source is untrusted author text, so any parser failure, a
        # `RecursionError` included, is a fault of this skill's block alone.
        raise FrontMatterInvalid(f"front matter does not parse as YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise FrontMatterInvalid("front matter is not a mapping")
    name, description = parsed.get("name"), parsed.get("description")
    if not isinstance(name, str):
        raise FrontMatterInvalid("front matter carries no string 'name'")
    if not isinstance(description, str):
        raise FrontMatterInvalid("front matter carries no string 'description'")
    if name != skill_id:
        raise FrontMatterInvalid(
            f"front matter name {name[:80]!r} differs from the skill id {skill_id!r}"
        )
    try:
        _WIRE_FRONT_MATTER.dump_json(_WIRE_FRONT_MATTER.validate_python(parsed))
    except ValueError as exc:
        raise FrontMatterInvalid(
            f"front matter does not serialise as the Skills API wire model: {exc}"
        ) from exc
    return ParsedFrontMatter(name=name, description=description, mapping=parsed)
