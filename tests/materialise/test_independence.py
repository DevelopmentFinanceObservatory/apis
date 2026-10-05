"""Checks that importing `materialise` avoids application, FastAPI, and optional-format modules.

A fresh subprocess checks imports; CI's clean install checks which packages are installed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("forbidden_prefixes", "expects_pure_problem_vocabulary"),
    [
        pytest.param(
            ("search", "dfo_apis", "fastapi"),
            True,
            id="search-api-fastapi",
        ),
        pytest.param(
            ("datasets_api", "dataset_store", "spec_registry", "artifact_store", "openpyxl"),
            False,
            id="application-store-format-extras",
        ),
    ],
)
def test_import_does_not_pull_in_application_or_optional_dependencies(
    forbidden_prefixes: tuple[str, ...],
    expects_pure_problem_vocabulary: bool,
):
    required_problem_assertion = (
        "assert 'dfo_platform.problem' in sys.modules, "
        "'expected materialise to import dfo_platform.problem for its pure wire vocabulary'\n"
        if expects_pure_problem_vocabulary
        else ""
    )
    script = (
        "import sys\n"
        "import materialise\n"
        "import materialise.roles\n"
        f"leaked = sorted(m for m in sys.modules if m.startswith({forbidden_prefixes!r}))\n"
        "assert not leaked, f'materialise import pulled in: {leaked}'\n"
        f"{required_problem_assertion}"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
