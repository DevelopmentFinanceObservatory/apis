"""No module that elects or serves a dataset names one.

Datasets enter Search, the Datasets API and the code index only through the index's
declarations. A string literal equal to a declared dataset's provenance (with or without
its `provenance/` prefix), or to a code value only one declared dataset's rows carry (with
or without a `dcid:` prefix), would let a code path decide for one dataset by name.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from code_index.query import CodeIndex
from code_index.shared import get_index
from scripts.export_code_index import DATASETS

_REPO = Path(__file__).resolve().parents[2]
_ROOTS = (
    _REPO / "src" / "search" / "engine",
    _REPO / "src" / "search" / "catalogue",
    _REPO / "src" / "datasets_api",
    _REPO / "packages" / "dfo-data-tools" / "src" / "code_index",
)


def _dataset_names(index: CodeIndex) -> set[str]:
    provenances = {dataset.provenance for dataset in DATASETS} | set(index.datasets)
    return provenances | {provenance.removeprefix("provenance/") for provenance in provenances}


def _single_dataset_codes(index: CodeIndex) -> set[str]:
    """The constraint values whose rows all belong to one declared dataset."""
    carriers: dict[str, set[str]] = defaultdict(set)
    for row in index.statvars:
        for value in row.constraint_properties.values():
            carriers[value].update(row.provenances)
    codes = {value for value, provenances in carriers.items() if len(provenances) == 1}
    return codes | {f"dcid:{code}" for code in codes}


def _literal_hits(source: str, banned: set[str]) -> list[tuple[int, str]]:
    return [
        (node.lineno, node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in banned
    ]


def test_the_scan_finds_a_dataset_named_in_a_literal():
    """The scan reads string literals anywhere, an f-string's text included."""
    banned = {"provenance/Ledger", "CodeOnlyLedgerCarries"}
    source = 'X = "provenance/Ledger"\ndef f(y):\n    return f"CodeOnlyLedgerCarries{y}" or y\n'

    assert sorted(_literal_hits(source, banned)) == [
        (1, "provenance/Ledger"),
        (3, "CodeOnlyLedgerCarries"),
    ]


def test_no_election_module_names_a_declared_dataset_or_its_own_codes():
    index = get_index()
    banned = _dataset_names(index) | _single_dataset_codes(index)
    files = sorted(path for root in _ROOTS for path in root.rglob("*.py"))

    hits = [
        f"{path.relative_to(_REPO)}:{line}: {value!r}"
        for path in files
        for line, value in _literal_hits(path.read_text(), banned)
    ]

    assert files
    assert len(_dataset_names(index)) >= 2 * len(DATASETS)
    assert hits == []
