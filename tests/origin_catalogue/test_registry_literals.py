"""No Python source names a provenance, series or variable prefix the registry supersedes or
excludes: `registry.toml` is the one place those ids sit, and code reads its declared fields.

The ids are read from the packaged registry: each base provenance a UN agency supersedes,
the prefix its variable ids carry, its successor and the id segment the successor's
variables carry (`sdg/` reaches both `sdg/...` and `undata/sdg/...`), and each excluded UN
series with its code, which every variable of the series starts with. A string literal holding one is a hit wherever it sits
in `src/` or `packages/dfo-data-tools/src/`. Docstrings and comments are not literals, and
data files are not scanned.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from origin_catalogue.registry import load_registry

_SERIES_MARKER = "SERIES-"
"""The UN deployment's series ids read `undata/<segment>/SERIES-<code>`. The registry's
excluded series follow it, so the scan can name each code they carry."""

_REPO = Path(__file__).resolve().parents[2]
_SOURCE_ROOTS = (_REPO / "src", _REPO / "packages" / "dfo-data-tools" / "src")


def _registry_ids() -> list[tuple[str, str]]:
    """Each id the registry declares for a supersession or an exclusion, with what it is.
    One string declared as two things is listed twice. A literal holding one anywhere is a
    hit, except a `successor agency segment`, which is a hit only as the whole literal, since
    a short segment such as `sdg` sits inside unrelated words."""
    registry = load_registry()
    ids: list[tuple[str, str]] = []
    for exclusion, agency in registry.un_supersessions():
        ids.append((exclusion.provenance, "superseded provenance"))
        ids.append((exclusion.variable_prefix, "superseded variable prefix"))
        ids.append((exclusion.successor, "successor provenance"))
        ids.append((f"{agency.segment}/", "successor variable prefix"))
        ids.append((agency.segment, "successor agency segment"))
    for agency in registry.un_system:
        for exclusion in agency.exclusions:
            ids.append((exclusion.series, "excluded series"))
            ids.append((exclusion.series.rpartition(_SERIES_MARKER)[2], "excluded series code"))
    return ids


def _string_literals(source: str) -> Iterator[tuple[int, str]]:
    """Every string literal in `source` with its line, f-string parts included, docstrings
    and other bare string statements left out."""
    tree = ast.parse(source)
    statements = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in statements
        ):
            yield node.lineno, node.value


_EXACT = "successor agency segment"


def _hits(source: str, ids: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    return [
        (line, literal, kind)
        for line, literal in _string_literals(source)
        for declared, kind in ids
        if (literal == declared if kind == _EXACT else declared in literal)
    ]


def test_the_scan_reads_supersessions_and_exclusions_from_the_packaged_registry() -> None:
    kinds = {kind for _declared, kind in _registry_ids()}

    assert kinds == {
        "superseded provenance",
        "superseded variable prefix",
        "successor provenance",
        "successor variable prefix",
        "successor agency segment",
        "excluded series",
        "excluded series code",
    }


def test_the_scan_finds_a_declared_id_in_a_literal_and_skips_documentation() -> None:
    ids = _registry_ids()
    superseded = next(declared for declared, kind in ids if kind == "superseded provenance")
    prefix = next(declared for declared, kind in ids if kind == "superseded variable prefix")
    source = (
        f'"""Reads {superseded}."""\n'
        f"# {superseded}\n"
        f'HELD = "{superseded}"\n'
        "def superseded(dcid):\n"
        f'    """Whether `{prefix}` starts it."""\n'
        f'    return dcid.startswith("{prefix}")\n'
    )

    # The packaged prefix is also the successor agency's own.
    assert {(line, kind) for line, _literal, kind in _hits(source, ids)} >= {
        (3, "superseded provenance"),
        (6, "superseded variable prefix"),
    }
    assert {line for line, _literal, _kind in _hits(source, ids)} == {3, 6}


def test_the_scan_finds_a_successor_segment_only_as_a_whole_literal() -> None:
    ids = _registry_ids()
    segment = next(declared for declared, kind in ids if kind == _EXACT)
    source = (
        f'SEATED = agency.segment == "{segment}"\n'
        f'WORD = "{segment}x is not the segment"\n'
        f'joined = "undata/" + "{segment}"\n'
    )

    assert [(line, kind) for line, _literal, kind in _hits(source, ids)] == [
        (1, _EXACT),
        (3, _EXACT),
    ]


def test_no_python_source_names_an_id_the_registry_supersedes_or_excludes() -> None:
    ids = _registry_ids()
    files = sorted(path for root in _SOURCE_ROOTS for path in root.rglob("*.py"))
    assert len(files) > 100

    hits = [
        f"{path.relative_to(_REPO)}:{line}: {literal!r} holds a {kind}"
        for path in files
        for line, literal, kind in _hits(path.read_text(encoding="utf-8"), ids)
    ]

    assert hits == []
