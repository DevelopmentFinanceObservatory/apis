"""Parses the closed dotted-query namespace for `GET /api/datasets/{id}` into `MintOptions`.
The encoder uses the same namespace in reverse. Unknown or repeated keys fail here. `MintOptions`
validates recognised values as body requests. `groupby.by` and `columns` use comma-separated lists
whose grammars exclude commas from entries. `groupby.levels` is a comma-separated list of
`<code_system>:<depth>` pairs, such as `crsPurpose:0`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import cast

from pydantic import ValidationError

from datasets_api.mint_models import MAX_GROUPBY_LEVEL_DEPTH, MintOptions
from dfo_platform.validation import pointer_segment

# Each closed-namespace query key's equivalent (top-level field, nested
# key) in the mint body's `options` object. A top-level scalar field
# (`format`, `bundle`, `columns`) carries `None` as its nested key.
# Extending the namespace -- a new option, or a dotted key for a field
# that gains its own sub-structure -- means adding one row here.
_NAMESPACE: dict[str, tuple[str, str | None]] = {
    "format": ("format", None),
    "bundle": ("bundle", None),
    "groupby.by": ("groupby", "by"),
    "groupby.op": ("groupby", "op"),
    "groupby.levels": ("groupby", "levels"),
    "pivot.index": ("pivot", "index"),
    "pivot.columns": ("pivot", "columns"),
    "columns": ("columns", None),
}
# `groupby.by` is a comma-separated list like `columns`. The other nested keys are strings.
_COMMA_JOINED_NESTED_KEYS: frozenset[str] = frozenset({"groupby.by"})


class QueryOptionsInvalid(Exception):
    """A dotted-query key outside the closed namespace, a key supplied more
    than once, a value `MintOptions` itself refuses once assembled into
    the equivalent body shape, or an option this deployment's code index
    cannot serve (`datasets_api.render.check_options_supported`). Wire
    mapping: `request_invalid`, 400, `pointer` naming the equivalent body path.
    """

    def __init__(self, *, pointer: str, detail: str) -> None:
        self.pointer = pointer
        super().__init__(detail)


def pointer_for_key(key: str) -> str:
    """Maps a dotted query key to its body path, escaping each segment per RFC 6901. Both GET
    routes use this mapping for query-key refusals."""
    return "/options/" + "/".join(pointer_segment(segment) for segment in key.split("."))


def parse_query_options(items: Sequence[tuple[str, str]]) -> MintOptions:
    """Builds `MintOptions` from query pairs. Raises `QueryOptionsInvalid` for unknown or repeated
    keys and values rejected by the model."""
    seen: set[str] = set()
    payload: dict[str, object] = {}
    nested: dict[str, dict[str, object]] = {}

    for key, value in items:
        if key not in _NAMESPACE:
            raise QueryOptionsInvalid(
                pointer=pointer_for_key(key), detail=f"unrecognised query key {key!r}"
            )
        if key in seen:
            raise QueryOptionsInvalid(
                pointer=pointer_for_key(key),
                detail=f"query key {key!r} was supplied more than once",
            )
        seen.add(key)

        field, subfield = _NAMESPACE[key]
        if key == "groupby.levels":
            nested.setdefault(field, {})["levels"] = _parse_levels(value)
        elif subfield is not None:
            nested.setdefault(field, {})[subfield] = (
                value.split(",") if key in _COMMA_JOINED_NESTED_KEYS else value
            )
        elif field == "columns":
            payload[field] = value.split(",")
        else:
            payload[field] = value

    payload.update(nested)

    try:
        return MintOptions.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        pointer = "/options/" + "/".join(pointer_segment(segment) for segment in first["loc"])
        raise QueryOptionsInvalid(pointer=pointer, detail=first["msg"]) from exc


def _parse_levels(value: str) -> dict[str, int]:
    """Parses `groupby.levels` pairs. A pair without a `:`, a depth that is not a JSON integer
    (so none with a leading zero), or a repeated code system raises `QueryOptionsInvalid`. A
    depth with more digits than `MAX_GROUPBY_LEVEL_DEPTH` refuses here, before integer
    conversion, at the code system's pointer. `MintOptions` refuses any other depth outside
    0..`MAX_GROUPBY_LEVEL_DEPTH` and an axis absent from `groupby.by`, at the same pointers as
    the body."""
    levels: dict[str, int] = {}
    for pair in value.split(","):
        code_system, separator, depth = pair.partition(":")
        if not separator or re.fullmatch(r"-?(0|[1-9][0-9]*)", depth) is None:
            raise QueryOptionsInvalid(
                pointer=pointer_for_key("groupby.levels"),
                detail=f"groupby.levels entry {pair!r} is not <code_system>:<depth>",
            )
        if code_system in levels:
            raise QueryOptionsInvalid(
                pointer=pointer_for_key("groupby.levels"),
                detail=f"groupby.levels names {code_system!r} more than once",
            )
        if len(depth.removeprefix("-")) > len(str(MAX_GROUPBY_LEVEL_DEPTH)):
            raise QueryOptionsInvalid(
                pointer=f"{pointer_for_key('groupby.levels')}/{pointer_segment(code_system)}",
                detail=(
                    f"groupby.levels depth for {code_system!r} is outside "
                    f"0..{MAX_GROUPBY_LEVEL_DEPTH}"
                ),
            )
        levels[code_system] = int(depth)
    return levels


def encode_query_options(options: MintOptions) -> dict[str, str]:
    """Encodes applied options as the flat query map accepted by `GET /api/datasets/{id}`.
    `format` and `bundle` are always present. Other keys appear only when set. `columns` and
    `groupby.by` are comma-joined, and their closed grammars exclude commas from entries.
    `groupby.levels` joins its `<code_system>:<depth>` pairs in code-system order."""
    encoded: dict[str, str] = {}
    for key, (field, subfield) in _NAMESPACE.items():
        if field == "format":
            encoded[key] = options.format
        elif field == "bundle":
            encoded[key] = "true" if options.bundle else "false"
        elif field == "groupby":
            if options.groupby is not None:
                value = getattr(options.groupby, cast(str, subfield))
                if key == "groupby.levels":
                    if value is not None:
                        encoded[key] = ",".join(
                            f"{code_system}:{depth}"
                            for code_system, depth in sorted(cast(dict[str, int], value).items())
                        )
                    continue
                encoded[key] = (
                    ",".join(cast(list[str], value))
                    if key in _COMMA_JOINED_NESTED_KEYS
                    else cast(str, value)
                )
        elif field == "pivot":
            if options.pivot is not None:
                encoded[key] = str(getattr(options.pivot, cast(str, subfield)))
        elif field == "columns" and options.columns is not None:
            encoded[key] = ",".join(options.columns)
    return encoded
