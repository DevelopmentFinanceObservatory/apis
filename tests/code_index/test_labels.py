"""Test `normalize_label` over the fixture index and its import isolation."""

from __future__ import annotations

import subprocess
import sys

from code_index import CodeIndex
from code_index.labels import normalize_label


def test_indexed_dcid_takes_the_index_label(code_index: CodeIndex):
    row = code_index.code_by_dcid("CRSPurposeCode100")
    assert row is not None
    raw = f"{row.label} (CRS Purpose)"

    assert normalize_label(row.dcid, raw, index=code_index) == row.label


def test_unindexed_dcid_keeps_the_raw_label(code_index: CodeIndex):
    assert normalize_label("country/KEN", "Kenya", index=code_index) == "Kenya"


def test_importing_labels_pulls_in_no_search_module():
    probe = (
        "import sys, code_index.labels; "
        "sys.exit(any(m == 'search' or m.startswith('search.') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0
