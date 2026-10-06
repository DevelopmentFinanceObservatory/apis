"""AST sweeps of ``src/search`` check every Warning and EngineInputError code against the registry.
They cover unexecuted branches. Unsupported dynamic imports and lookups fail loudly.
The refusal scan includes reserved codes without an EngineInputError raise site.
"""

from __future__ import annotations

import ast
import pathlib

from search.engine.codes import REFUSAL_CODES, WARNING_CODES
from search.engine.region.codes import (
    NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    NO_MEMBER_HAS_DATA_AT_PLACE,
    ONLY_ALTERNATE_HAS_DATA,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SEARCH_SRC = _REPO_ROOT / "src" / "search"

# Every unresolved `code=` shape either sweep below finds is a hard failure
# (see test_every_warning_code_kwarg_resolves_statically and
# test_every_refusal_code_kwarg_resolves_statically); no site currently needs
# an exact (file, expression) exemption from the warning sweep.
_WARNING_KNOWN_PASSTHROUGHS: frozenset[tuple[str, str]] = frozenset()

# Every `EngineInputError(code=<non-literal expression>)` site the refusal
# sweep cannot resolve to a string, but that mints no new code: a rewritten
# re-raise of an already-caught EngineInputError (its own code was already
# resolved, and registered, at its original raise site), or a code fixed
# inside another exception class's own __init__ outside src/search
# (materialise.recipes' NonCurrencyUnitError/RecipeNotDeclaredError/
# RateCoverageGapError, whose literal `code=` this sweep's own directory
# scope cannot see).
_REFUSAL_KNOWN_PASSTHROUGHS: frozenset[tuple[str, str]] = frozenset(
    {
        # Rewrites a caught EngineInputError's pointer onto this candidate's
        # own slot in the request; the code travels unchanged.
        ("engine/entry/dataset_entries.py", "exc.code"),
        # Same rewrite, for the dataset_resubmit door's own entries.
        ("engine/entry/dataset_resubmit.py", "exc.code"),
        # Re-wraps an EngineInputError this module already built, onto a
        # renamed pointer; the code travels unchanged.
        ("engine/facet_refusals.py", "refusal.code"),
        # Carries the code fixed in NonCurrencyUnitError's/
        # RecipeNotDeclaredError's/RateCoverageGapError's own __init__
        # (materialise.recipes, outside src/search).
        ("engine/options/surface.py", "exc.code"),
        # Same as above, for the resubmit promote door.
        ("engine/region/promote_ground.py", "exc.code"),
        # Rewrites a caught EngineInputError's pointer onto the nested
        # identity contract; the code travels unchanged.
        ("engine/region/resubmit.py", "exc.code"),
    }
)


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    """Collect module-level string constants, including annotated assignments."""
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


# search.models.Warning is re-exported as search.Warning (see search/
# __init__.py's own docstring: "importable as search.Warning or via explicit
# from search import Warning") -- both dotted spellings name the identical
# class, so both are accepted resolution targets.
_WARNING_CLASS_TARGETS = frozenset({"search.models.Warning", "search.Warning"})

# EngineInputError has one canonical module, no re-export facade.
_ENGINE_INPUT_ERROR_CLASS_TARGETS = frozenset({"search.engine.errors.EngineInputError"})


def _warning_import_bindings(tree: ast.Module) -> dict[str, str] | None:
    """Resolve direct, aliased, and module-attribute imports of Warning.
    Return None for wildcard or submodule imports the resolver cannot trace.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "search.models":
            for alias in node.names:
                if alias.name == "*":
                    return None
                if alias.name == "Warning":
                    bindings[alias.asname or alias.name] = "search.models.Warning"
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("search.models.")
        ):
            # A submodule-form import (`from search.models.envelope import
            # Warning`) reaches into search/models/ past its facade -- this
            # resolver only traces the facade import forms above, so it
            # cannot tell whether such an import binds Warning. Treated as a
            # hard failure, same as a wildcard, rather than silently binding
            # nothing.
            return None
        elif isinstance(node, ast.ImportFrom) and node.module == "search":
            for alias in node.names:
                if alias.name == "*":
                    return None
                if alias.name == "models":
                    bindings[alias.asname or alias.name] = "search.models"
                elif alias.name == "Warning":
                    bindings[alias.asname or alias.name] = "search.models.Warning"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "search.models" and alias.asname is not None:
                    bindings[alias.asname] = "search.models"
                elif alias.name == "search" and alias.asname is not None:
                    bindings[alias.asname] = "search"
                # A bare (unaliased) `import search` / `import search.models`
                # binds the root name to itself; the identity fallback in
                # _resolve_dotted_target already resolves `search.models.
                # Warning(...)` / `search.Warning(...)` correctly without an
                # entry here.
    return bindings


def _engine_input_error_import_bindings(tree: ast.Module) -> dict[str, str] | None:
    """Resolve direct, aliased, and module-attribute imports of EngineInputError.
    Return None for wildcard imports the resolver cannot trace.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "search.engine.errors":
            for alias in node.names:
                if alias.name == "*":
                    return None
                if alias.name == "EngineInputError":
                    bindings[alias.asname or alias.name] = "search.engine.errors.EngineInputError"
        elif isinstance(node, ast.ImportFrom) and node.module == "search.engine":
            for alias in node.names:
                if alias.name == "*":
                    return None
                if alias.name == "errors":
                    bindings[alias.asname or alias.name] = "search.engine.errors"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "search.engine.errors" and alias.asname is not None:
                    bindings[alias.asname] = "search.engine.errors"
                # A bare (unaliased) `import search.engine.errors` binds the
                # root name to itself; the identity fallback in
                # _resolve_dotted_target already resolves
                # `search.engine.errors.EngineInputError(...)` correctly
                # without an entry here.
    return bindings


def _resolve_dotted_target(expr: ast.expr, bindings: dict[str, str]) -> str | None:
    """Resolve a Name/Attribute chain through its imported root binding."""
    parts: list[str] = []
    node = expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(bindings.get(node.id, node.id))
    return ".".join(reversed(parts))


class _RuntimeNodeCollector(ast.NodeVisitor):
    """Collects every node actually evaluated at runtime, skipping the
    subtrees reachable only through a type annotation (`AnnAssign.
    annotation`, a parameter's `arg.annotation`, a function's `returns`) or
    an `except` clause's own type expression (`ExceptHandler.type`). Every
    module under src/search opens with `from __future__ import
    annotations`, so annotations are stored as strings and never evaluated
    -- an ordinary `list[Warning]` field type or `-> EngineInputError`
    return annotation can never construct anything and must not be flagged
    as an indirect use of the class. An `except EngineInputError as exc:`
    clause IS evaluated at runtime (a real isinstance check), but it binds an
    already-raised instance rather than constructing a new one, so it is
    excluded the same way."""

    def __init__(self) -> None:
        self.nodes: list[ast.AST] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.nodes.append(node)
        super().generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.nodes.append(node)
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_arg(self, node: ast.arg) -> None:
        self.nodes.append(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.nodes.append(node)
        self.visit(node.value)
        # node.slice deliberately skipped: a subscript's slice is
        # overwhelmingly a generic type parameter in this codebase
        # (`tuple[..., Warning, ...]`), including old-style module-level
        # type aliases -- a plain `Assign`, not an `AnnAssign`, so it is NOT
        # caught by the annotation skip above (region/codes.py's
        # `_OptionProbeResult = tuple[tuple[_OptionEvent, ...], tuple[
        # Warning, ...]]` and grounding.py's `_GroundAnswerResult` are real
        # examples). A class sitting in a type-parameter position can never
        # construct an instance by itself.

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.nodes.append(node)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.nodes.append(node)
        for stmt in node.body:
            self.visit(stmt)


def _runtime_nodes(tree: ast.Module) -> list[ast.AST]:
    collector = _RuntimeNodeCollector()
    collector.visit(tree)
    return collector.nodes


def _indirect_class_uses(
    tree: ast.Module,
    bindings: dict[str, str],
    class_targets: frozenset[str],
    class_label: str,
) -> list[tuple[int, str]]:
    """Find class references the sweep cannot resolve to a direct constructor call.
    Untracked uses, including string-based ``getattr``, fail the scan.
    """
    exempt_ids = {id(call.func) for call in ast.walk(tree) if isinstance(call, ast.Call)}
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "isinstance"
            and len(call.args) == 2
        ):
            type_arg = call.args[1]
            if isinstance(type_arg, ast.Tuple):
                exempt_ids.update(id(elt) for elt in type_arg.elts)
            else:
                exempt_ids.add(id(type_arg))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Tuple, ast.List))
            and node.elts
            and all(isinstance(elt, (ast.Name, ast.Attribute)) for elt in node.elts)
        ):
            exempt_ids.update(id(elt) for elt in node.elts)
    violations: list[tuple[int, str]] = []
    for node in _runtime_nodes(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            if (
                _resolve_dotted_target(node, bindings) in class_targets
                and id(node) not in exempt_ids
            ):
                violations.append(
                    (
                        node.lineno,
                        f"{class_label} class referenced outside a direct call: `{ast.unparse(node)}`",
                    )
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and any(isinstance(arg, ast.Constant) and arg.value == class_label for arg in node.args)
        ):
            violations.append(
                (
                    node.lineno,
                    f'getattr(..., "{class_label}", ...) reaches the class by name: `{ast.unparse(node)}`',
                )
            )
    return violations


def _sweep_minted_warning_codes() -> tuple[frozenset[str], tuple[str, ...]]:
    """Collect resolvable Warning codes and report unresolved ``code=`` expressions."""
    paths = sorted(_SEARCH_SRC.rglob("*.py"))
    trees = {path: ast.parse(path.read_text(), filename=str(path)) for path in paths}

    # Flat name -> value lookup across the whole package. Constant names here
    # are unique across modules today (asserted by test_no_constant_name_
    # collides_across_modules below), so a flat lookup resolves every `code=
    # SOME_CONSTANT` reference regardless of which module actually imports it.
    all_constants: dict[str, str] = {}
    for tree in trees.values():
        all_constants.update(_module_string_constants(tree))

    resolved: set[str] = set()
    failures: list[str] = []
    for path, tree in trees.items():
        rel_path = path.relative_to(_SEARCH_SRC.parent).as_posix().removeprefix("search/")
        bindings = _warning_import_bindings(tree)
        if bindings is None:
            failures.append(
                f"{rel_path}: a wildcard import (`from search.models import *` "
                "or `from search import *`) or an import from a search.models "
                "submodule (`from search.models.<submodule> import ...`) cannot "
                "be traced statically -- add a targeted import so this sweep "
                "can tell whether it binds Warning"
            )
            continue
        for lineno, description in _indirect_class_uses(
            tree, bindings, _WARNING_CLASS_TARGETS, "Warning"
        ):
            failures.append(f"{rel_path}:{lineno}: {description}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _resolve_dotted_target(node.func, bindings) not in _WARNING_CLASS_TARGETS:
                continue
            code_kwarg = next((kw for kw in node.keywords if kw.arg == "code"), None)
            location = f"{rel_path}:{node.lineno}"
            if code_kwarg is None:
                failures.append(f"{location}: Warning(...) call has no code= kwarg")
                continue
            value = code_kwarg.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                resolved.add(value.value)
                continue
            if isinstance(value, ast.Name) and value.id in all_constants:
                resolved.add(all_constants[value.id])
                continue
            expr = ast.unparse(value)
            if (rel_path, expr) in _WARNING_KNOWN_PASSTHROUGHS:
                continue
            failures.append(
                f"{location}: code={expr!r} did not resolve to a string literal "
                "or a known module-level constant"
            )

    return frozenset(resolved), tuple(failures)


def _sweep_minted_refusal_codes() -> tuple[frozenset[str], tuple[str, ...]]:
    """Collect EngineInputError codes and report unresolved expressions outside known passthroughs.
    Raises without ``code=`` and explicit ``code=None`` contribute no code.
    """
    paths = sorted(_SEARCH_SRC.rglob("*.py"))
    trees = {path: ast.parse(path.read_text(), filename=str(path)) for path in paths}

    all_constants: dict[str, str] = {}
    for tree in trees.values():
        all_constants.update(_module_string_constants(tree))

    resolved: set[str] = set()
    failures: list[str] = []
    for path, tree in trees.items():
        rel_path = path.relative_to(_SEARCH_SRC.parent).as_posix().removeprefix("search/")
        bindings = _engine_input_error_import_bindings(tree)
        if bindings is None:
            failures.append(
                f"{rel_path}: a wildcard import (`from search.engine.errors "
                "import *` or `from search.engine import *`) cannot be traced "
                "statically -- add a targeted import so this sweep can tell "
                "whether it binds EngineInputError"
            )
            continue
        for lineno, description in _indirect_class_uses(
            tree, bindings, _ENGINE_INPUT_ERROR_CLASS_TARGETS, "EngineInputError"
        ):
            failures.append(f"{rel_path}:{lineno}: {description}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _resolve_dotted_target(node.func, bindings) not in _ENGINE_INPUT_ERROR_CLASS_TARGETS:
                continue
            code_kwarg = next((kw for kw in node.keywords if kw.arg == "code"), None)
            location = f"{rel_path}:{node.lineno}"
            if code_kwarg is None:
                continue
            value = code_kwarg.value
            if isinstance(value, ast.Constant) and value.value is None:
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                resolved.add(value.value)
                continue
            if isinstance(value, ast.Name) and value.id in all_constants:
                resolved.add(all_constants[value.id])
                continue
            expr = ast.unparse(value)
            if (rel_path, expr) in _REFUSAL_KNOWN_PASSTHROUGHS:
                continue
            failures.append(
                f"{location}: code={expr!r} did not resolve to a string literal, "
                "a known module-level constant, or a documented passthrough"
            )

    return frozenset(resolved), tuple(failures)


# Computed once at collection time: every warning code the engine actually
# mints, discovered by AST sweep rather than hand-mirrored.
_MINTED_WARNING_CODES, _UNRESOLVED_WARNING_CODE_KWARGS = _sweep_minted_warning_codes()

# Refusal codes with no EngineInputError raise site for the sweep above to
# find: the codes app.py's own exception handlers mint from a `ProblemSpec`
# (classify_search_exception's EngineInfraError/asyncio.TimeoutError
# branches), plus the codes the shared spec-kit registry reserves onto a
# Search endpoint with nothing in this package raising them.
_UNSWEPT_REFUSAL_CODES = {
    # classify_search_exception's EngineInfraError branch (upstream_status in
    # the 4xx range).
    "upstream_error",
    # classify_search_exception's EngineInfraError branch (every other case).
    "service_unavailable",
    # classify_search_exception's asyncio.TimeoutError branch.
    "timeout",
    # classify_search_exception's rate-table branch (RateTableInvalidError and
    # UpstreamVersionChangedError, raised by the materialise library).
    "capability_unavailable",
    "upstream_version_changed",
    # Registry-reserved compatibility code: canonical identity.entities.origin
    # accepts multiple origins, so Search code has no raise site for it.
    "provider_ambiguous",
    # Platform-shared registry-reserved code; no Search door raises it.
    "code_not_found",
    # Registry-reserved on search.options (422); no flow axis in this
    # package raises it.
    "directional_children_unsupported",
}

_SWEPT_REFUSAL_CODES, _UNRESOLVED_REFUSAL_CODE_KWARGS = _sweep_minted_refusal_codes()

# Computed once at collection time: every refusal code the engine actually
# mints, discovered by AST sweep plus the short explicit set above.
_MINTED_REFUSAL_CODES = _SWEPT_REFUSAL_CODES | _UNSWEPT_REFUSAL_CODES


def test_every_warning_code_kwarg_resolves_statically():
    """Every Warning code resolves to a literal or module constant."""
    assert not _UNRESOLVED_WARNING_CODE_KWARGS, "unresolvable code= expression(s):\n" + "\n".join(
        _UNRESOLVED_WARNING_CODE_KWARGS
    )


def test_every_refusal_code_kwarg_resolves_statically():
    """Every EngineInputError code resolves to a literal, module constant, or passthrough."""
    assert not _UNRESOLVED_REFUSAL_CODE_KWARGS, "unresolvable code= expression(s):\n" + "\n".join(
        _UNRESOLVED_REFUSAL_CODE_KWARGS
    )


def test_no_constant_name_collides_across_modules():
    """The package-wide constant lookup requires unique names across modules."""
    by_name: dict[str, set[str]] = {}
    for path in sorted(_SEARCH_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for name, value in _module_string_constants(tree).items():
            by_name.setdefault(name, set()).add(value)
    colliding = {name: values for name, values in by_name.items() if len(values) > 1}
    assert not colliding, f"constant name(s) with conflicting values across modules: {colliding}"


def test_warning_import_bindings_rejects_submodule_form_imports():
    """Unsupported submodule imports fail; both facade imports resolve Warning."""
    submodule_tree = ast.parse("from search.models.envelope import Warning\n")
    assert _warning_import_bindings(submodule_tree) is None

    facade_tree = ast.parse("from search.models import Warning\n")
    facade_bindings = _warning_import_bindings(facade_tree)
    assert facade_bindings is not None
    assert facade_bindings["Warning"] == "search.models.Warning"

    reexport_tree = ast.parse("from search import Warning\n")
    reexport_bindings = _warning_import_bindings(reexport_tree)
    assert reexport_bindings is not None
    assert reexport_bindings["Warning"] == "search.models.Warning"


def test_engine_input_error_import_bindings_rejects_wildcard_imports():
    """Wildcard imports fail; direct and module imports resolve EngineInputError."""
    wildcard_tree = ast.parse("from search.engine.errors import *\n")
    assert _engine_input_error_import_bindings(wildcard_tree) is None

    direct_tree = ast.parse("from search.engine.errors import EngineInputError\n")
    direct_bindings = _engine_input_error_import_bindings(direct_tree)
    assert direct_bindings is not None
    assert direct_bindings["EngineInputError"] == "search.engine.errors.EngineInputError"

    module_tree = ast.parse("from search.engine import errors\n")
    module_bindings = _engine_input_error_import_bindings(module_tree)
    assert module_bindings is not None
    assert module_bindings["errors"] == "search.engine.errors"


def test_warning_sweep_found_a_plausible_number_of_mint_sites():
    """At least 20 warning codes prevents a misdirected source path from passing vacuously.
    The current corpus has 29, leaving room for ordinary code changes.
    """
    assert len(_MINTED_WARNING_CODES) >= 20, (
        f"only found {len(_MINTED_WARNING_CODES)} minted warning codes -- far "
        "fewer than expected; check that _SEARCH_SRC resolves to the real "
        "src/search directory"
    )


def test_refusal_sweep_found_a_plausible_number_of_mint_sites():
    """At least 15 swept refusal codes guards source discovery before reserved codes are added.
    The current scan finds 22, leaving room for ordinary code changes.
    """
    assert len(_SWEPT_REFUSAL_CODES) >= 15, (
        f"only found {len(_SWEPT_REFUSAL_CODES)} minted refusal codes -- far "
        "fewer than expected; check that _SEARCH_SRC resolves to the real "
        "src/search directory"
    )


def test_every_minted_warning_code_is_registered():
    unregistered = _MINTED_WARNING_CODES - WARNING_CODES
    assert not unregistered, f"warning codes minted but not registered: {sorted(unregistered)}"


def test_every_minted_refusal_code_is_registered():
    unregistered = _MINTED_REFUSAL_CODES - REFUSAL_CODES
    assert not unregistered, f"refusal codes minted but not registered: {unregistered}"


def test_every_minted_warning_code_is_lower_snake_case():
    for code in _MINTED_WARNING_CODES:
        assert code == code.lower(), f"{code!r} is not lower snake_case"
        assert " " not in code


def test_every_registered_refusal_code_is_lower_snake_case():
    for code in REFUSAL_CODES:
        assert code == code.lower(), f"{code!r} is not lower snake_case"


def test_contract_named_warning_codes_are_registered():
    """The four warning codes the API contract names explicitly by string."""
    for code in (
        "date_coverage_partial",
        "binding_folded",
        "observations_omitted",
        "default_not_declared",
    ):
        assert code in WARNING_CODES


def test_contract_refusal_table_codes_are_registered():
    """Every refusal code the wire contract defines, verbatim."""
    for code in (
        "request_invalid",
        "query_too_long",
        "unsupported_schema_version",
        "shape_mismatch",
        "invalid_role_ref",
        "role_incompatible",
        "role_signature_incompatible",
        "facet_not_available",
        "resolution_ambiguous",
        "facet_binding_unsupported",
        "option_target_invalid",
    ):
        assert code in REFUSAL_CODES


def test_flagged_match_gate_codes_are_registered_refusal_only():
    """Gate codes share the refusal registry without being raised as refusals or warnings."""
    from search.engine.codes import CODE_CATALOG

    for code in (
        ONLY_ALTERNATE_HAS_DATA,
        NO_MEMBER_HAS_DATA_AT_PLACE,
        NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    ):
        assert code not in WARNING_CODES
        assert code in REFUSAL_CODES
        assert code in CODE_CATALOG
