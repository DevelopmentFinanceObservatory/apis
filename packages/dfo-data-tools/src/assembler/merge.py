"""RFC 7386 JSON Merge Patch for revise intents."""

from __future__ import annotations

from collections.abc import Mapping

from assembler.errors import SchemaInvalid

__all__ = ["merge_intent"]

_CREATE_ONLY_FIELDS = ("type", "frame_ref")


def merge_intent(
    *, current: Mapping[str, object], patch: Mapping[str, object]
) -> dict[str, object]:
    """Applies RFC 7386: absent keys survive, `null` deletes, and other values
    replace whole, including arrays and objects. The write set is the patch's
    key set. Raises `SchemaInvalid` if the patch names create-only `type` or
    `frame_ref`."""
    named = [field for field in _CREATE_ONLY_FIELDS if field in patch]
    if named:
        field = named[0]
        raise SchemaInvalid(
            pointer=f"/{field}",
            detail=f"{field!r} is create-only and cannot be revised",
            legal_fields=tuple(key for key in current if key not in _CREATE_ONLY_FIELDS),
        )

    merged = dict(current)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged
