"""Checks that `datasets_api` imports without the `engine` extra's own packages.

A fresh subprocess blocks the packages only `engine` installs (resolvekit, google-genai), so
the check holds in a test environment that carries every extra. CI's datasets-only install
also syncs `graph`, which brings resolvekit back.
"""

from __future__ import annotations

import subprocess
import sys

_ENGINE_ONLY_PACKAGES = ("resolvekit", "google.genai")


def test_import_does_not_need_engine_only_packages():
    script = (
        "import sys\n"
        "from importlib.abc import MetaPathFinder\n"
        "class Block(MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        f"        if name.startswith({_ENGINE_ONLY_PACKAGES!r}):\n"
        "            raise ImportError(f'{name} is installed only by the engine extra')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import datasets_api\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('search.engine'))\n"
        "assert not leaked, f'datasets_api import pulled in: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
