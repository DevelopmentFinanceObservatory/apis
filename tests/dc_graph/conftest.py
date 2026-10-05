"""Shared fixtures for the `dc_graph` test suite."""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def read_fixture_text(name: str) -> str:
    """Read a fixture file as text, without normalising its line endings."""
    return (FIXTURES_DIR / name).read_bytes().decode("utf-8")
