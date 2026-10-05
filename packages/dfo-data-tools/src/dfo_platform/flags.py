"""Builds a Pydantic before-validator for environment-backed boolean flags.

Pydantic's own parsing rejects an empty string and accepts spellings such as "yes" and "on".
A string is true only when its lowercased, optionally stripped value is in `truthy`. Every other
string is false. Non-strings pass through for Pydantic coercion. Search flags preserve whitespace.
Surface flags strip it before lookup.
"""

from __future__ import annotations

from collections.abc import Callable


def lenient_bool(*, truthy: frozenset[str], strip: bool = True) -> Callable[[object], object]:
    """A `mode="before"` field-validator function: parses a string `value`
    against `truthy` (lowercased, and stripped first when `strip`), and
    returns any non-string `value` unchanged for pydantic's own bool
    coercion to handle.
    """

    def _parse(value: object) -> object:
        if isinstance(value, str):
            raw = value.strip().lower() if strip else value.lower()
            return raw in truthy
        return value

    return _parse
