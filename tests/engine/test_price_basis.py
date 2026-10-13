"""Tests which dropped phrases name a price basis."""

from __future__ import annotations

import pytest

from search.engine.unit.bind import names_price_basis


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("constant 2022 dollars", True),
        ("in real terms", True),
        ("at current prices", True),
        ("in euros", False),
        ("by sector", False),
    ],
)
def test_names_price_basis(text: str, expected: bool) -> None:
    assert names_price_basis(text) is expected
