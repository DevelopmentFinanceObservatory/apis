"""Checks that `dfo_platform` imports without Search, Dataset Store, or FastAPI."""

from __future__ import annotations

import subprocess
import sys


def test_import_does_not_pull_in_search_or_datasets():
    script = (
        "import sys\n"
        "import dfo_platform\n"
        "bad = [\n"
        "    m for m in sys.modules\n"
        "    if m in ('search', 'dataset_store')\n"
        "    or m.startswith(('search.', 'dataset_store.'))\n"
        "]\n"
        "assert not bad, f'dfo_platform import pulled in: {bad}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_pure_core_is_importable_without_fastapi():
    script = (
        "import sys\n"
        "from dfo_platform.problem import Alternative, ProblemIssue\n"
        "from dfo_platform.diagnostics import WarningSeverity\n"
        "from dfo_platform.codes import CodeCatalogue, merge_catalogues\n"
        "from dfo_platform.validation import validation_problem\n"
        "from dfo_platform.config import parse_cors_origins\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('fastapi'))\n"
        "assert not leaked, f'pure-core import pulled in: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
