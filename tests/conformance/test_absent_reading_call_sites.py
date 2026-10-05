"""Check that production member-enumeration calls use a `readings` builder matching the live registry.
An empty map treats absent constraints as aggregate and can drop distinct CRS rows under
`not_declared`. The AST sweep rejects missing, empty, or unsupported readings expressions.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from collections.abc import Mapping

import pytest

from code_index.shared import get_index
from statvar_members.types import AbsentReading

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Scan both production roots because callers can live beside the main `src/` tree.
_SWEEP_ROOTS: tuple[pathlib.Path, ...] = (
    _REPO_ROOT / "src",
    _REPO_ROOT / "packages" / "dfo-data-tools" / "src",
)

# The library relays caller-supplied readings; this sweep checks its callers.
_EXCLUDED_DIR = _REPO_ROOT / "packages" / "dfo-data-tools" / "src" / "statvar_members"

# Import targets tracked by `_import_bindings` and resolved by `_resolve_dotted`.
_ENUMERATE_MEMBERS = "statvar_members.enumerate_members"
_MEMBER_ENUMERATOR = "statvar_members.MemberEnumerator"


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    """Return the dotted import name for `path` beneath `root`."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_bindings(tree: ast.Module) -> dict[str, str] | None:
    """Map direct or aliased imports to the tracked member-enumeration targets.

    Return None for wildcard imports, which the sweep cannot resolve.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "statvar_members"
            or (node.module is not None and node.module.startswith("statvar_members."))
        ):
            for alias in node.names:
                if alias.name == "*":
                    return None
                if alias.name == "enumerate_members":
                    bindings[alias.asname or alias.name] = _ENUMERATE_MEMBERS
                elif alias.name == "MemberEnumerator":
                    bindings[alias.asname or alias.name] = _MEMBER_ENUMERATOR
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "statvar_members":
                    bindings[alias.asname or "statvar_members"] = "statvar_members"
    return bindings


def _resolve_dotted(expr: ast.expr, bindings: dict[str, str]) -> str | None:
    """Resolve a dotted AST name through import bindings. Return None for other shapes."""
    parts: list[str] = []
    node = expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(bindings.get(node.id, node.id))
    return ".".join(reversed(parts))


def _member_enumerator_names(tree: ast.Module, bindings: dict[str, str]) -> set[str]:
    """Return names annotated or assigned as `MemberEnumerator` instances.

    The sweep matches `.enumerate(...)` calls on these names, including receiver attributes.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        annotation: ast.expr | None = None
        if isinstance(node, ast.arg):
            annotation = node.annotation
            target_name = node.arg
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotation = node.annotation
            target_name = node.target.id
        else:
            target_name = None
        if annotation is not None and _resolve_dotted(annotation, bindings) == _MEMBER_ENUMERATOR:
            names.add(target_name)  # type: ignore[arg-type]

        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _resolve_dotted(node.value.func, bindings) == _MEMBER_ENUMERATOR
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _call_targets(tree: ast.Module, bindings: dict[str, str]) -> list[ast.Call]:
    """Find direct `enumerate_members` calls and `.enumerate` calls on tracked instances."""
    enumerator_names = _member_enumerator_names(tree, bindings)
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _resolve_dotted(node.func, bindings) == _ENUMERATE_MEMBERS:
            calls.append(node)
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "enumerate"):
            continue
        receiver = func.value
        if (isinstance(receiver, ast.Name) and receiver.id in enumerator_names) or (
            isinstance(receiver, ast.Attribute) and receiver.attr in enumerator_names
        ):
            calls.append(node)
    return calls


def _registry_readings() -> Mapping[str, AbsentReading]:
    """Return code-system absence readings from the live registry."""
    return {
        code_system: declaration.absent_reading
        for code_system, declaration in get_index().declarations.items()
    }


class _CallSite:
    def __init__(self, *, rel_path: str, lineno: int, call: ast.Call, module_name: str) -> None:
        self.rel_path = rel_path
        self.lineno = lineno
        self.call = call
        self.module_name = module_name

    @property
    def label(self) -> str:
        return f"{self.rel_path}:{self.lineno}: {ast.unparse(self.call)}"


def _collect_call_sites() -> tuple[list[_CallSite], list[str]]:
    """Find production call sites and report wildcard imports the sweep cannot resolve."""
    sites: list[_CallSite] = []
    unresolved_modules: list[str] = []
    for root in _SWEEP_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if _EXCLUDED_DIR in path.parents:
                continue
            rel_path = path.relative_to(_REPO_ROOT).as_posix()
            tree = ast.parse(path.read_text(), filename=str(path))
            bindings = _import_bindings(tree)
            if bindings is None:
                unresolved_modules.append(
                    f"{rel_path}: wildcard import from statvar_members -- this sweep cannot "
                    "trace whether it binds enumerate_members/MemberEnumerator"
                )
                continue
            if not bindings:
                continue
            module_name = _module_name(path, root)
            sites.extend(
                _CallSite(rel_path=rel_path, lineno=call.lineno, call=call, module_name=module_name)
                for call in _call_targets(tree, bindings)
            )
    return sites, unresolved_modules


def _readings_value(site: _CallSite) -> Mapping[str, AbsentReading] | str:
    """Resolve the call's `readings` expression or return why the sweep cannot verify it."""
    readings_kwarg = next((kw for kw in site.call.keywords if kw.arg == "readings"), None)
    if readings_kwarg is None:
        return (
            f"{site.label} passes no readings= keyword at all -- it now silently inherits "
            "EMPTY_READINGS"
        )
    value = readings_kwarg.value
    if isinstance(value, ast.Dict) and not value.keys:
        return f"{site.label} passes readings={{}} explicitly -- as empty as no keyword at all"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and not value.keywords:
        module = importlib.import_module(site.module_name)
        builder = getattr(module, value.func.id, None)
        if builder is None:
            return (
                f"{site.label}: readings={ast.unparse(value)!r} names no attribute "
                f"{value.func.id!r} on {site.module_name!r}"
            )
        if not value.args:
            return builder()
        if len(value.args) == 1 and isinstance(value.args[0], ast.Name):
            # Some callers pass their resolved index to the builder. The live singleton checks
            # that the builder produces registry readings.
            return builder(get_index())
    return (
        f"{site.label}: readings={ast.unparse(value)!r} is neither an empty dict literal nor a "
        "bare zero-argument or single-argument call this sweep can resolve to an actual "
        "mapping -- extend this sweep, or the sweep in `_readings_value`, rather than let it "
        "pass unverified"
    )


_SITES, _UNRESOLVED_MODULES = _collect_call_sites()


def test_sweep_traced_every_statvar_members_import():
    """Fail when a wildcard import prevents the sweep from tracing member-enumeration calls."""
    assert not _UNRESOLVED_MODULES, "unresolvable statvar_members import(s):\n" + "\n".join(
        _UNRESOLVED_MODULES
    )


def test_sweep_found_the_known_production_call_sites():
    """Require the AST sweep to find the known production call site, preventing a vacuous pass."""
    found = {site.rel_path for site in _SITES}
    assert "src/datasets_api/adapter.py" in found


@pytest.mark.parametrize("site", _SITES, ids=[site.label for site in _SITES])
def test_call_site_passes_a_registry_backed_readings_map(site: _CallSite):
    result = _readings_value(site)
    assert not isinstance(result, str), result
    registry = _registry_readings()
    assert result, f"{site.label}: readings resolved to an empty mapping, not the live registry"
    assert result == registry, (
        f"{site.label}: readings resolved to {result!r}, not the live registry mapping {registry!r}"
    )
