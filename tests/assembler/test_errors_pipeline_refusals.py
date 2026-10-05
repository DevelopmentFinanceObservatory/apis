"""Pin required `SchemaInvalid` disclosure fields."""

from __future__ import annotations

import pytest

from assembler import errors


def test_schema_invalid_requires_exactly_one_disclosure_payload():
    with pytest.raises(TypeError):
        errors.SchemaInvalid(pointer="/x", detail="d")
    with pytest.raises(TypeError):
        errors.SchemaInvalid(
            pointer="/x", detail="d", minimal_intent={"type": "line_chart"}, legal_fields=("type",)
        )
