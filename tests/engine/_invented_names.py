"""The test-wide check that no place name a door invented reaches the wire.

A door names the two places of one text posted on both ends with invented names
(`search.engine.door_names`), joined by a control character. Every boundary reads a
name's text, so the character never appears in a response or a streamed event.
"""

from __future__ import annotations

import json

from search.engine.door_names import NAME_SEPARATOR

# The separator as it is written raw and as JSON escapes it.
_MARKS = (NAME_SEPARATOR, json.dumps(NAME_SEPARATOR)[1:-1])


def assert_no_invented_names(body: str, *, where: str) -> None:
    """Fail when `body`, a serialised response or SSE stream, holds an invented name."""
    for mark in _MARKS:
        at = body.find(mark)
        assert at < 0, (
            f"{where} holds an invented place name: ...{body[max(at - 80, 0) : at + 40]!r}..."
        )
