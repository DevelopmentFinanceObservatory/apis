"""Checks that the production bind rung matches the shipped prompt constant."""

from __future__ import annotations

import json

from promptlab.compose import compose
from promptlab.manifest import PROMPTLAB_ROOT, load_rung
from search.engine.bind import _UNIVERSAL_BIND_SYSTEM_PROMPT

EXTRACT_MANIFEST = PROMPTLAB_ROOT / "rungs" / "production-extract.json"
BIND_MANIFEST = PROMPTLAB_ROOT / "rungs" / "production-bind.json"


def test_production_bind_composes_byte_for_byte_to_the_shipped_constant():
    rung = load_rung(EXTRACT_MANIFEST, BIND_MANIFEST)
    assert compose(rung.bind) == _UNIVERSAL_BIND_SYSTEM_PROMPT


def test_production_bind_run_config():
    run = json.loads(BIND_MANIFEST.read_text())["run"]
    assert run["capture"] == "k8-2026-08-31"
    assert run["top_k"] == 8
    assert run["seeds"] == [1, 2, 3]
    assert run["model"] == "gemini-3.5-flash-lite"
    assert run["thinking"] == {"extract": None, "bind": "minimal"}
