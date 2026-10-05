"""Checks that no earlier classifier entry shadows a later entry's exception type."""

from __future__ import annotations

from datasets_api.doors.mint_errors import _MINT_CLASSIFIERS


def _flatten(
    exc_types: type[BaseException] | tuple[type[BaseException], ...],
) -> tuple[type[BaseException], ...]:
    return exc_types if isinstance(exc_types, tuple) else (exc_types,)


def test_no_entry_is_shadowed_by_an_earlier_base_class():
    rows = [
        (index, _flatten(exc_types))
        for index, (exc_types, _handler) in enumerate(_MINT_CLASSIFIERS)
    ]
    shadowed = [
        (earlier_index, earlier_type, later_index, later_type)
        for earlier_index, earlier_types in rows
        for later_index, later_types in rows
        if earlier_index < later_index
        for earlier_type in earlier_types
        for later_type in later_types
        if earlier_type is not later_type and issubclass(later_type, earlier_type)
    ]
    assert shadowed == []
