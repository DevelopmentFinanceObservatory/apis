"""Parses the closed dotted-query namespace for `GET /api/datasets/{id}` into `MintOptions`.
The encoder uses the same namespace in reverse. Unknown or repeated keys fail here. `MintOptions`
validates recognised values as body requests. `groupby.by` and `columns` use comma-separated lists
whose grammars exclude commas from entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pydantic import ValidationError

from datasets_api.mint_models import MintOptions
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
    "pivot.index": ("pivot", "index"),
    "pivot.columns": ("pivot", "columns"),
    "columns": ("columns", None),
}
# `groupby.by` is a comma-separated list like `columns`. The other nested keys are strings.
_COMMA_JOINED_NESTED_KEYS: frozenset[str] = frozenset({"groupby.by"})


class QueryOptionsInvalid(Exception):
    """A dotted-query key outside the closed namespace, a key supplied more
    than once, or a value `MintOptions` itself refuses once assembled into
    the equivalent body shape. Wire mapping: `request_invalid`, 400,
    `pointer` naming the equivalent body path.
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
        if subfield is not None:
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


def encode_query_options(options: MintOptions) -> dict[str, str]:
    """Encodes applied options as the flat query map accepted by `GET /api/datasets/{id}`.
    `format` and `bundle` are always present. Other keys appear only when set. `columns` and
    `groupby.by` are comma-joined, and their closed grammars exclude commas from entries."""
    encoded: dict[str, str] = {}
    for key, (field, subfield) in _NAMESPACE.items():
        if field == "format":
            encoded[key] = options.format
        elif field == "bundle":
            encoded[key] = "true" if options.bundle else "false"
        elif field == "groupby":
            if options.groupby is not None:
                value = getattr(options.groupby, cast(str, subfield))
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
