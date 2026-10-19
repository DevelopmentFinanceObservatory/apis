"""Checks that a surface outside the Datasets one imports without the release reader."""

from __future__ import annotations

import subprocess
import sys

import pytest

# `graph_api.app` and `search.engine.app` import no Datasets surface module. A fresh
# interpreter is needed because the test process has already imported `graph_release.service`.
_SCRIPT = """
import importlib, sys
importlib.import_module({module!r})
assert 'graph_release.service' not in sys.modules, 'graph_release.service was imported'
"""


@pytest.mark.parametrize("uri", [None, "", "  "], ids=["unset", "empty", "blank"])
@pytest.mark.parametrize("module", ["graph_api.app", "search.engine.app"])
def test_surface_imports_without_the_release_service_when_no_release_is_configured(
    module: str, uri: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `search.engine.config` builds its settings at import and requires a graph base.
    monkeypatch.setenv("SEARCH_GRAPH_BASE", "https://graph.offline.invalid")
    if uri is None:
        monkeypatch.delenv("GRAPH_RELEASE_URI", raising=False)
    else:
        monkeypatch.setenv("GRAPH_RELEASE_URI", uri)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(module=module)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
