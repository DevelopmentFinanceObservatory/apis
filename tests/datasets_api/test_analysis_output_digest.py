"""The output digest against the client contract's `output-digest` vectors, each frame built
through the frame grammar as the runner holds it."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

import pytest

from datasets_api.output_digest import digest_document, output_digest
from materialise.frame_build import build_frame

_VECTORS: list[dict[str, Any]] = json.loads(
    (files("dfo_client_contract") / "rules" / "output-digest.json").read_text(encoding="utf-8")
)["vectors"]
_FIXED = frozenset(
    {"stat_var", "time_period", "value", "unit", "measurementMethod", "provenance", "status"}
)


@pytest.mark.parametrize("vector", _VECTORS, ids=[vector["name"] for vector in _VECTORS])
def test_each_vector_frame_digests_to_its_document_and_digest(vector: dict[str, Any]) -> None:
    columns = vector["frame"]["columns"]
    roles = [column for column in columns if column not in _FIXED]
    frame = build_frame(
        role_keys=roles,
        rows=[
            {column: cell for column, cell in zip(columns, row, strict=True) if column != "status"}
            for row in vector["frame"]["rows"]
        ],
    )

    assert output_digest(frame) == vector["digest"]
    picked = ["time_period", "value", "unit"]
    rows = [
        [row[columns.index(name)] for name in (*roles, *picked)] for row in vector["frame"]["rows"]
    ]
    assert digest_document(role_columns=roles, rows=rows).decode() == vector["document"]
