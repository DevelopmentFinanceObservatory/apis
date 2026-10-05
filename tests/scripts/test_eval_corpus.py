"""`scripts/eval_corpus.py`: `check` against a temporary corpus, and `schema`."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import scripts.eval_corpus as m
from search.eval.cases import CHECK_NAMES

CASES = """
- id: population-kenya
  query: population of Kenya
  why: Count_Person is the bare population count.
  expect:
    entities: [{dcid: country/KEN, role: observationAbout}]
    definite: {measure: {family: base.population}}
- id: population-holdout
  query: how many people live in Kenya
  slice: holdout
  why: Count_Person is the bare population count.
  expect:
    definite: {measure: {family: base.population}}
"""

FAMILIES = """
base.population:
  origin: base
  dataset: {gl: the base Data Commons population provenance}
  stat_vars: [Count_Person]
"""


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "families").mkdir()
    (tmp_path / "families" / "base.yaml").write_text(FAMILIES)
    (tmp_path / "standard.yaml").write_text(CASES)
    return tmp_path


@pytest.fixture
def all_checks(monkeypatch):
    monkeypatch.setattr(m, "implemented_checks", lambda: frozenset(CHECK_NAMES))


def test_check_passes_and_lists_ineligible_cases(corpus_dir, all_checks, capsys):
    assert m.main(["check", "--corpus-dir", str(corpus_dir)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "ineligible population-kenya: unconfirmed value at "
        "expect.definite.measure.forms[0].provenance.dataset",
        "ineligible population-holdout: slice holdout; unconfirmed value at "
        "expect.definite.measure.forms[0].provenance.dataset",
        "CORPUS CHECK PASSED: 2 cases, 0 eligible, 2 ineligible",
    ]


def test_check_fails_on_corpus_errors(corpus_dir, all_checks, capsys):
    (corpus_dir / "zz.yaml").write_text(
        "- id: population-kenya\n  query: q\n  why: w\n  expect: {definite: {}}\n"
    )
    assert m.main(["check", "--corpus-dir", str(corpus_dir)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "CORPUS CHECK FAILED (1 error(s)):"
    assert out[1].startswith("  - zz.yaml: population-kenya: duplicate id")


def test_check_fails_on_a_warning_outside_the_doors_allowlist(corpus_dir, all_checks, capsys):
    """raw_text extraction decides role_scope_defaulted, so a case cannot require it there."""
    (corpus_dir / "zz.yaml").write_text(
        "- id: aid-kenya\n  query: aid to Kenya\n  why: w\n"
        "  expect: {definite: {warnings: [role_scope_defaulted]}}\n"
    )
    assert m.main(["check", "--corpus-dir", str(corpus_dir)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "CORPUS CHECK FAILED (1 error(s)):"
    assert out[1].startswith("  - aid-kenya:")
    assert len(out) == 2


def test_implemented_checks_reads_the_checks_registry(monkeypatch):
    registry = types.SimpleNamespace(CHECKS={"measure": object(), "entities": object()})
    monkeypatch.setitem(sys.modules, "search.eval.checks", registry)
    assert m.implemented_checks() == {"measure", "entities"}


def test_implemented_checks_is_empty_before_the_registry_exists(monkeypatch):
    monkeypatch.setitem(sys.modules, "search.eval.checks", None)
    assert m.implemented_checks() == frozenset()


def test_schema_prints_the_case_file_schema(capsys):
    assert m.main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["items"] == {"$ref": "#/$defs/Case"}
