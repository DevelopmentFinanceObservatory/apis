"""Every literal `code=` and every refusal class's `code = "..."` under the Datasets and
materialise source trees must belong to this surface's served-code catalogue union. The AST
sweep fails on unresolved `code=` expressions. Known passthroughs use file, function, and
expression keys.
"""

from __future__ import annotations

import ast
import pathlib

from datasets_api.codes import DATASETS_CODE_CATALOGUE, SHARED_PLATFORM_CODES
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from materialise.bundle import BUNDLE_CODE_CATALOGUE
from materialise.codes import MATERIALISE_CODE_CATALOGUE
from materialise.reshape import RESHAPE_CODE_CATALOGUE

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DATASETS_SRC = _REPO_ROOT / "src" / "datasets_api"
_MATERIALISE_SRC = _REPO_ROOT / "packages" / "dfo-data-tools" / "src" / "materialise"

# The union contains this surface's refusals, shared platform codes, warnings, and
# materialise codes that can appear in its manifests.
_SERVED_DATASETS_CODES: frozenset[str] = (
    frozenset(DATASETS_CODE_CATALOGUE)
    | SHARED_PLATFORM_CODES
    | frozenset(DATASETS_WARNING_CATALOGUE)
    | frozenset(MATERIALISE_CODE_CATALOGUE)
    | frozenset(BUNDLE_CODE_CATALOGUE)
    | frozenset(RESHAPE_CODE_CATALOGUE)
)


# Exemptions use stable file/function/expression keys. Every other unresolved `code=` fails.
_KNOWN_PASSTHROUGHS: frozenset[tuple[str, str | None, str]] = frozenset(
    {
        ("src/datasets_api/version.py", "_render_body_exemplars", "code"),
        ("src/datasets_api/manifest.py", "warning_model", "cast(str, warning['code'])"),
        ("src/datasets_api/manifest.py", "reshape_warning_models", "code"),
        ("src/datasets_api/doors/meta.py", "classify_meta_exception", "exc.refusal_code"),
        # `RefreshNotPermitted.code`, a class attribute the sweep collects.
        ("src/datasets_api/live_refresh.py", "classify_live_refresh_exception", "exc.code"),
        (
            "src/datasets_api/doors/conversion_errors.py",
            "classify_conversion_exception",
            "exc.code",
        ),
        (
            "packages/dfo-data-tools/src/materialise/recipes.py",
            "cause_from_recipe_error",
            "exc.cause",
        ),
        (
            "packages/dfo-data-tools/src/materialise/group_facets.py",
            "_as_issue",
            "exc.code",
        ),
        # A codes-table entry's `code` is a data cell, not an error code.
        ("packages/dfo-data-tools/src/materialise/bundle_columns.py", "bundle_codes", "code"),
    }
)


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Returns module-level string constants by name for resolving `code=` arguments."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.value
    return constants


def _call_target_name(func: ast.expr) -> str | None:
    """Returns a call's bare callee name, or `None` for other expressions."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _CallVisitor(ast.NodeVisitor):
    """Collects call nodes with their enclosing function names for stable exemptions."""

    def __init__(self) -> None:
        self.calls: list[tuple[ast.Call, str | None]] = []
        self._stack: list[str] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append((node, ".".join(self._stack) or None))
        self.generic_visit(node)


def _sweep_dir(src_dir: pathlib.Path) -> tuple[set[str], list[str]]:
    """Returns resolved `code=` and class-attribute code literals, and failures for unresolved,
    unlisted `code=` expressions."""
    paths = sorted(src_dir.rglob("*.py"))
    trees = {path: ast.parse(path.read_text(), filename=str(path)) for path in paths}

    all_constants: dict[str, str] = {}
    for tree in trees.values():
        all_constants.update(_module_string_constants(tree))

    resolved: set[str] = set()
    failures: list[str] = []
    for path, tree in trees.items():
        # A refusal class names its code as a class attribute.
        resolved.update(
            statement.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "code" for t in statement.targets)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        rel_path = path.relative_to(_REPO_ROOT).as_posix()
        visitor = _CallVisitor()
        visitor.visit(tree)
        for node, enclosing_function in visitor.calls:
            code_kwarg = next((kw for kw in node.keywords if kw.arg == "code"), None)
            if code_kwarg is None:
                continue
            value = code_kwarg.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                resolved.add(value.value)
                continue
            if isinstance(value, ast.Name) and value.id in all_constants:
                resolved.add(all_constants[value.id])
                continue
            if (rel_path, enclosing_function, ast.unparse(value)) in _KNOWN_PASSTHROUGHS:
                continue
            failures.append(
                f"{rel_path}:{value.lineno}: code={ast.unparse(value)!r} on a call to "
                f"{ast.unparse(node.func)}(...) did not resolve to a string literal, a "
                "known module-level constant, or a documented passthrough in "
                "_KNOWN_PASSTHROUGHS"
            )

    return resolved, failures


# Computed at collection time from both source trees.
_MINTED_DATASETS_CODES, _DATASETS_FAILURES = _sweep_dir(_DATASETS_SRC)
_MINTED_MATERIALISE_CODES, _MATERIALISE_FAILURES = _sweep_dir(_MATERIALISE_SRC)
_MINTED_CODES = _MINTED_DATASETS_CODES | _MINTED_MATERIALISE_CODES
_UNRESOLVED = (*_DATASETS_FAILURES, *_MATERIALISE_FAILURES)


def test_every_code_kwarg_resolves_statically():
    """Every `code=` expression must resolve or appear in the passthrough allowlist."""
    assert not _UNRESOLVED, "unresolvable code= expression(s):\n" + "\n".join(_UNRESOLVED)


def test_sweep_found_a_plausible_number_of_mint_sites():
    """At least 40 codes must be found so a mispointed source path cannot pass vacuously."""
    assert len(_MINTED_CODES) >= 40, (
        f"only found {len(_MINTED_CODES)} minted code(s) -- far fewer than expected; "
        "check that _DATASETS_SRC/_MATERIALISE_SRC resolve to the real source directories"
    )


def test_every_minted_code_is_registered():
    unregistered = _MINTED_CODES - _SERVED_DATASETS_CODES
    assert not unregistered, (
        f"code(s) minted under src/datasets_api or packages/dfo-data-tools/src/materialise "
        f"but absent from this surface's own served-code union: {sorted(unregistered)}"
    )
