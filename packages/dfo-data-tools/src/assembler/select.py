"""Normalizes flat `select` maps into the selector bindings used by resolution.

A dimension's list of values is a set: `canonical_select` sorts it and drops repeats, so an
intent states it in one order whatever order its caller sent. Scalar values become
single-value bindings. This module also enforces
non-empty selections and the one-arrayed-dimension limit beyond the open
map's JSON Schema constraints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from assembler.errors import BindingInvalid, escape_pointer_segment

__all__ = [
    "canonical_select",
    "check_select_nonempty",
    "check_single_arrayed_dimension",
    "has_arrayed_dimension",
    "normalize_select",
]


def canonical_select(*, select: Mapping[str, object]) -> dict[str, object]:
    """`select` with each list of values sorted and its repeats dropped. A scalar value stays
    as it is."""
    return {
        # The intent schema holds a dimension's values as strings.
        dimension: sorted(set(cast("list[str]", value))) if isinstance(value, list) else value
        for dimension, value in select.items()
    }


def normalize_select(*, select: Mapping[str, object]) -> list[dict[str, object]]:
    """Converts each dimension into a binding, promoting scalar values to one-item lists."""
    bindings = [
        {"dimension": dimension, "values": value if isinstance(value, list) else [value]}
        for dimension, value in select.items()
    ]
    return [{"bindings": bindings}]


def check_select_nonempty(*, select: Mapping[str, object]) -> None:
    """Requires at least one dimension. An empty map is schema-valid and would resolve to
    every frame series."""
    if not select:
        raise BindingInvalid(
            pointer="/select",
            dimension="",
            legal_values=(),
            truncated=False,
            detail="select must name at least one dimension",
        )


def has_arrayed_dimension(*, select: Mapping[str, object]) -> bool:
    """Whether `select` binds an arrayed dimension.

    Bar charts use this to choose their mode."""
    return any(isinstance(value, list) for value in select.values())


def check_single_arrayed_dimension(*, select: Mapping[str, object]) -> None:
    """Allows one arrayed dimension. A second raises `BindingInvalid` at the
    lexicographically second offending name."""
    arrayed = sorted(name for name, value in select.items() if isinstance(value, list))
    if len(arrayed) <= 1:
        return
    offending = arrayed[1]
    raise BindingInvalid(
        pointer=f"/select/{escape_pointer_segment(offending)}",
        dimension=offending,
        legal_values=tuple(arrayed),
        truncated=False,
        detail=(
            f"at most one dimension in select may bind an array of values; "
            f"{arrayed!r} are all arrays"
        ),
    )
