"""Importing `search` does not import `search.eval`, even with the eval extra installed."""

import subprocess
import sys


def test_core_import_does_not_pull_in_eval():
    code = (
        "import sys, search; "
        "assert 'search.eval' not in sys.modules, "
        "'search.eval was imported transitively from search'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Core import isolation check failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
