"""Checks rung 0 composition and approximate prompt lengths."""

from __future__ import annotations

from promptlab.compose import compose
from promptlab.manifest import PROMPTLAB_ROOT, load_rung

RUNGS = PROMPTLAB_ROOT / "rungs"
EXTRACT_MANIFEST = RUNGS / "rung0-extract.json"
BIND_MANIFEST = RUNGS / "rung0-bind.json"


def _load_rung():
    return load_rung(EXTRACT_MANIFEST, BIND_MANIFEST)


def test_rung0_composes_both_stages():
    rung = _load_rung()

    extract_text = compose(rung.extract)
    bind_text = compose(rung.bind)

    assert extract_text.startswith("You are the extraction stage")
    assert bind_text.startswith("You are the binding stage")


def test_rung0_extract_lands_near_350_characters():
    rung = _load_rung()
    extract_text = compose(rung.extract)
    # The 304-character floor catches any missing extract block.
    assert 304 <= len(extract_text) <= 384


def test_rung0_bind_lands_near_650_characters():
    rung = _load_rung()
    bind_text = compose(rung.bind)
    # The 590-character floor catches the 46-character security block being omitted.
    assert 590 <= len(bind_text) <= 670


def test_rung0_extract_carries_the_literal_today_placeholder():
    rung = _load_rung()
    extract_text = compose(rung.extract)
    assert "[[TODAY]]" in extract_text


def test_rung0_run_config_mirrors_production_bind():
    rung = _load_rung()
    assert rung.run.top_k == 8
    assert rung.run.seeds == (1, 2, 3)
    assert rung.run.model == "gemini-3.5-flash-lite"
    assert rung.run.thinking == {"extract": None, "bind": "minimal"}
