"""Name values in a refusal or warning message without printing a list or a whole string.

This standard-library module is shared by `dfo_platform.validation`, `search.models`,
`search.engine` and `datasets_api`, so a message that names dcids, keys or labels stays
short whatever the request carried.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

MAX_NAMED: Final = 5
"""How many values `named_capped` lists before it counts the rest."""

MAX_VALUE_CHARS: Final = 256
"""How many characters of one value a message carries before `cut` shortens it."""


def cut(value: str, *, max_chars: int = MAX_VALUE_CHARS) -> str:
    """`value`, or its first `max_chars` characters marked with `...` when it is longer.

    A message names a string a caller posted through this, so it echoes none whole. A lone
    surrogate, which JSON allows and UTF-8 cannot encode, prints as its `\\udXXX` escape, so the
    result always encodes as UTF-8 and counts against `max_chars` as written."""
    text = value.encode("utf-8", "backslashreplace").decode("utf-8")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def named_capped(values: Iterable[str], *, max_chars: int = MAX_VALUE_CHARS) -> str:
    """The first `MAX_NAMED` of `values` joined by commas, then `and N more` for the rest.

    Each listed value is cut to `max_chars` characters (`cut`). The order is the order
    given."""
    listed = list(values)
    named = ", ".join(cut(value, max_chars=max_chars) for value in listed[:MAX_NAMED])
    if len(listed) > MAX_NAMED:
        named = f"{named} and {len(listed) - MAX_NAMED} more"
    return named
