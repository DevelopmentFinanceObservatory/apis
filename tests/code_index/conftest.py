"""Shared fixture with 398 CRS rows captured from dev on 9 September 2026.

Synthetic StatVar rows model graph shapes; their dcids are fixture-only identifiers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_index import CodeIndex
from tests.code_index.support import load_index

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "index.json"


@pytest.fixture
def code_index() -> CodeIndex:
    return load_index(FIXTURE_PATH)
