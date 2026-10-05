"""Require `dataset_store` imports to avoid `search`, `dfo_apis`, and `fastapi` dependencies.
A fresh subprocess prevents earlier imports from masking a leak.
"""

from __future__ import annotations

import subprocess
import sys


def test_import_does_not_pull_in_search_dfo_apis_or_fastapi():
    script = (
        "import sys\n"
        "import dataset_store\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules if m.startswith(('search', 'dfo_apis', 'fastapi'))\n"
        ")\n"
        "assert not leaked, f'dataset_store import pulled in: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
