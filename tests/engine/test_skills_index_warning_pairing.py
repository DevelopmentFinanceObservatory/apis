"""Checks that every response path calls both skills-index warning helpers.

The AST sweep records direct helper calls by enclosing function and pins the current call-site set.
"""

from __future__ import annotations

import ast
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SEARCH_SRC = _REPO_ROOT / "src" / "search"

_UNAVAILABLE = "skills_index_unavailable_warning"
_INCOMPLETE = "skills_index_incomplete_warning"
_PAIRED_HELPER_NAMES = frozenset({_UNAVAILABLE, _INCOMPLETE})


class _EnclosingFunctionCallCollector(ast.NodeVisitor):
    """Walks one module's AST, tracking the stack of enclosing function
    names, and records each direct call to a name in `_PAIRED_HELPER_NAMES`
    against the dotted function-name stack it was called from. A call
    outside any function (module level) is never recorded -- neither
    helper is ever called there today, and there is no "enclosing function"
    to key it against.
    """

    def __init__(self) -> None:
        self.calls_by_function: dict[str, set[str]] = {}
        self._stack: list[str] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name: str | None = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in _PAIRED_HELPER_NAMES and self._stack:
            qualname = ".".join(self._stack)
            self.calls_by_function.setdefault(qualname, set()).add(name)
        self.generic_visit(node)


def _sweep_paired_skills_warning_calls() -> dict[str, set[str]]:
    """Every "relpath::qualname" function anywhere under src/search that
    calls skills_index_unavailable_warning and/or
    skills_index_incomplete_warning directly, mapped to the subset of
    {_UNAVAILABLE, _INCOMPLETE} it calls. A function calling neither is
    absent from the map."""
    calls_by_site: dict[str, set[str]] = {}
    for path in sorted(_SEARCH_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        collector = _EnclosingFunctionCallCollector()
        collector.visit(tree)
        rel_path = path.relative_to(_SEARCH_SRC.parent).as_posix().removeprefix("search/")
        for qualname, names in collector.calls_by_function.items():
            calls_by_site[f"{rel_path}::{qualname}"] = names
    return calls_by_site


def test_every_call_site_of_either_skills_index_warning_calls_both():
    calls_by_site = _sweep_paired_skills_warning_calls()

    unavailable_without_incomplete = sorted(
        site
        for site, names in calls_by_site.items()
        if _UNAVAILABLE in names and _INCOMPLETE not in names
    )
    incomplete_without_unavailable = sorted(
        site
        for site, names in calls_by_site.items()
        if _INCOMPLETE in names and _UNAVAILABLE not in names
    )

    assert not unavailable_without_incomplete, (
        f"calls {_UNAVAILABLE}() without a paired {_INCOMPLETE}() call in the same "
        f"function, so that function's response can drop the skills_index_incomplete "
        f"warning silently: {unavailable_without_incomplete}"
    )
    assert not incomplete_without_unavailable, (
        f"calls {_INCOMPLETE}() without a paired {_UNAVAILABLE}() call in the same "
        f"function, so that function's response can drop the skills_index_unavailable "
        f"warning silently: {incomplete_without_unavailable}"
    )


def test_the_sweep_actually_finds_every_known_call_site():
    """A sweep that silently found zero call sites would pass the pairing
    test above for the wrong reason -- pin the exact set of functions this
    delivery wired, so a future call site is a deliberate addition to this
    list rather than an invisible one."""
    calls_by_site = _sweep_paired_skills_warning_calls()

    assert set(calls_by_site) == {
        "engine/core.py::_quick_no_data",
        "engine/conjoin.py::assemble_region",
        "engine/conjoin.py::combine_regions",
        "engine/entry/raw_text.py::_empty_query_response",
        "engine/entry/raw_text.py::_no_variables_response",
    }
    assert all(names == _PAIRED_HELPER_NAMES for names in calls_by_site.values())
