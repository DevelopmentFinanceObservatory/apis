"""Pin RFC 6901 escaping for JSON Pointer segments."""

from __future__ import annotations

import pytest

from assembler import errors


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        ("recipient", "recipient"),
        ("foo~bar", "foo~0bar"),
        ("foo/bar", "foo~1bar"),
        ("foo/bar~baz", "foo~1bar~0baz"),
    ],
)
def test_escape_pointer_segment(segment: str, expected: str):
    assert errors.escape_pointer_segment(segment) == expected
