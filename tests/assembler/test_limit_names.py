"""The `limit_name`s the assembler raises are the spec-kit limit registry's, all of them."""

from __future__ import annotations

import ast
from pathlib import Path

from dfo_spec_kit.registry import LIMIT_NAMES

import assembler

_SOURCES = Path(assembler.__file__).parent


def _raised_limit_names() -> set[str]:
    """Every string literal the assembler's source passes as `ResourceExhausted`'s
    `limit_name` or as `SeriesLimit`'s `name`."""
    names: set[str] = set()
    for path in _SOURCES.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.id if isinstance(node.func, ast.Name) else None
            keyword = {"ResourceExhausted": "limit_name", "SeriesLimit": "name"}.get(callee or "")
            for argument in node.keywords:
                if argument.arg == keyword and isinstance(argument.value, ast.Constant):
                    names.add(argument.value.value)
    return names


def test_the_assembler_raises_every_registered_limit_name_and_no_other():
    assert _raised_limit_names() == set(LIMIT_NAMES)
