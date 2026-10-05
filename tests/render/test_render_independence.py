"""Checks that the renderer imports in a fresh interpreter without the Search engine."""

from __future__ import annotations

import subprocess
import sys


def test_render_plan_imports_without_the_engine():
    """`search.render.plan` and the heading facts it reads load first and alone: none of
    them imports `search.engine`, whose package import loads the engine that imports the
    renderer back."""
    script = (
        "import sys\n"
        "import search.render.plan\n"
        "leaked = sorted(m for m in sys.modules if m == 'search.engine' "
        "or m.startswith('search.engine.'))\n"
        "assert not leaked, f'render import pulled in: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
