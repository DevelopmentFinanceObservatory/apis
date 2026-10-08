"""Check that a refusal's wire `detail` names its values plainly and within the cap."""

from __future__ import annotations

import re
from collections.abc import Sequence

from dfo_platform.named import MAX_NAMED, MAX_VALUE_CHARS

_WRAPPERS = frozenset("'\"[]()")


def assert_names_plainly(detail: str, values: Sequence[str], *, always: Sequence[str] = ()) -> None:
    """`detail` names the first `MAX_NAMED` of `values`, none of the rest, and the count
    of the rest. `always` are names the message names in full.

    A value is plain when no quote, bracket or parenthesis touches it, the shape of a list
    or tuple repr, and the detail carries no square bracket anywhere."""
    listed, withheld = values[:MAX_NAMED], values[MAX_NAMED:]
    for value in (*listed, *always):
        found = list(re.finditer(re.escape(value), detail))
        assert found, f"{value!r} is not named in {detail!r}"
        for match in found:
            touching = (
                detail[max(match.start() - 1, 0) : match.start()]
                + detail[match.end() : match.end() + 1]
            )
            assert not _WRAPPERS & set(touching), f"{value!r} is wrapped in {detail!r}"
    for value in withheld:
        assert value not in detail
    if withheld:
        assert re.search(rf"\b{len(withheld)}\b", detail), f"no count of {len(withheld)}"
    assert "[" not in detail
    assert "]" not in detail


def assert_cut(response, oversized: str) -> None:
    """The body holds at most `MAX_VALUE_CHARS` characters of `oversized` in a row, and no
    pointer segment runs past `MAX_VALUE_CHARS`."""
    assert oversized[: MAX_VALUE_CHARS + 1] not in response.text
    segments = response.json()["pointer"].split("/")
    assert max(map(len, segments)) <= MAX_VALUE_CHARS
