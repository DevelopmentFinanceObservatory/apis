"""Checks that `datasets_api` runs without the `engine` extra's own packages.

A fresh subprocess blocks the packages only `engine` installs (google-genai), so the check holds
in a test environment that carries every extra. Resolvekit is not one of them: a role-bearing
mint classes its entities by the membership policy (`search.role_kinds`), which is built from
resolvekit's stores, so the `datasets` extra installs dfo-data-tools's `resolve` extra. CI's
`datasets_only_install` job runs this file on the `datasets` extra alone, where the role check
fails if `datasets` stops declaring a package it needs. Importing `datasets_api` still loads no
resolvekit.
"""

from __future__ import annotations

import subprocess
import sys

_ENGINE_ONLY_PACKAGES = ("google.genai",)

_BLOCK = (
    "import sys\n"
    "from importlib.abc import MetaPathFinder\n"
    "class Block(MetaPathFinder):\n"
    "    def find_spec(self, name, path, target=None):\n"
    f"        if name.startswith({_ENGINE_ONLY_PACKAGES!r}):\n"
    "            raise ImportError(f'{name} is installed only by the engine extra')\n"
    "sys.meta_path.insert(0, Block())\n"
)


def _run_blocked(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK + script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_import_does_not_need_engine_only_packages():
    _run_blocked(
        "import datasets_api\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('search.engine'))\n"
        "assert not leaked, f'datasets_api import pulled in: {leaked}'\n"
        "assert 'resolvekit' not in sys.modules, 'datasets_api import loaded resolvekit'\n"
    )


def test_a_role_bearing_mint_checks_its_entities_without_engine_only_packages():
    """The role check a mint binding Kenya as recipient runs, in the session release the
    subprocess inherits (`tests/conftest.py`)."""
    _run_blocked(
        "from code_index.shared import get_index\n"
        "from graph_release.context import bound_served_release\n"
        "from graph_release.service import get_release_service\n"
        "from search.role_kinds import check_role_kinds\n"
        "with bound_served_release(get_release_service()):\n"
        "    check_role_kinds(\n"
        "        {'destination': ['country/KEN']},\n"
        "        read_types=lambda dcids: {dcid: ('Country',) for dcid in dcids},\n"
        "        code_index=get_index(),\n"
        "        family=None,\n"
        "    )\n"
    )
