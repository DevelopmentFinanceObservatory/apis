"""Prove `dc_sdmx` imports without Search modules in a fresh subprocess."""

from __future__ import annotations

import subprocess
import sys


def test_import_does_not_pull_in_search():
    script = (
        "import sys\n"
        "import dc_sdmx\n"
        "bad = [m for m in sys.modules if m == 'search' or m.startswith('search.')]\n"
        "assert not bad, f'dc_sdmx import pulled in search modules: {bad}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
