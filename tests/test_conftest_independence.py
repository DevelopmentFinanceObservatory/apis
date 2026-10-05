"""Checks that `tests/conftest.py` imports without resolvekit.

migrations.yml and sdmx-live.yml install only the `datasets` or `sdmx` extra, which
lack resolvekit, and still load this conftest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_conftest_imports_when_resolvekit_is_absent():
    # A None entry in sys.modules makes `import resolvekit` raise ModuleNotFoundError.
    script = "import sys\nsys.modules['resolvekit'] = None\nimport tests.conftest\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
