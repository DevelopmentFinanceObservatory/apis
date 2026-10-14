"""Checks `NoData.category` derivation, the resolution order for every reason code, and
the election over it (`elect_no_data_reason`).

The table includes currency conversion's `rate_coverage_gap` and distinguishes unresolved entities
from absent data, including codes whose names contain `_not_`.
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Iterator
from pathlib import Path

import pytest

from search.models import NO_DATA_RESOLUTION_ORDER, NoData, NoDataReason, elect_no_data_reason

_SRC = Path(__file__).resolve().parents[2] / "src"

# code -> expected category, covering every NoDataReason value.
_CATEGORY_BY_REASON: dict[NoDataReason, str] = {
    "source_not_resolved": "unresolved",
    "source_ambiguous": "unresolved",
    "source_code_not_held": "unresolved",
    "source_code_ambiguous": "unresolved",
    "source_code_shares_variable": "unresolved",
    "entity_not_resolved": "unresolved",
    "entity_ambiguous": "unresolved",
    "entity_role_not_admitted": "unresolved",
    "entity_set_too_large": "unresolved",  # refused before any place is read
    "variable_not_resolved": "unresolved",
    "role_not_resolved": "unresolved",
    "unit_not_resolved": "unresolved",
    "no_matching_slice": "absent",
    "binding_mismatch": "absent",
    "denominator_not_available": "absent",  # naive-predicate trap: contains "_not_"
    "no_observations": "absent",
    "facet_not_available": "absent",  # naive-predicate trap: contains "_not_"
    "role_incompatible": "absent",
    "role_signature_incompatible": "absent",
    "rate_coverage_gap": "absent",
}


def test_category_derivation_probe_table():
    for reason, expected_category in _CATEGORY_BY_REASON.items():
        assert NoData(reason=reason).category == expected_category, reason


def test_category_is_derived_not_settable():
    """A caller-supplied category is always overwritten by the validator, so no
    constructor call can produce a contradictory pair."""
    no_data = NoData(reason="entity_not_resolved", category="absent")  # type: ignore[arg-type]
    assert no_data.category == "unresolved"

    no_data = NoData(reason="no_observations", category="unresolved")  # type: ignore[arg-type]
    assert no_data.category == "absent"


def test_resolution_order_covers_every_reason_exactly_once():
    assert len(NO_DATA_RESOLUTION_ORDER) == len(set(NO_DATA_RESOLUTION_ORDER))
    assert set(NO_DATA_RESOLUTION_ORDER) == set(NoDataReason.__args__)


def test_resolution_order_role_position():
    """The two role-compatibility codes sit at the role position, per contract."""
    role_index = NO_DATA_RESOLUTION_ORDER.index("role_not_resolved")
    assert NO_DATA_RESOLUTION_ORDER[role_index + 1] == "role_incompatible"
    assert NO_DATA_RESOLUTION_ORDER[role_index + 2] == "role_signature_incompatible"


def test_resolution_order_entity_position():
    """entity_ambiguous, entity_set_too_large and entity_role_not_admitted sit
    directly after entity_not_resolved, in that order, per contract: the four
    run by how far the entity got."""
    entity_index = NO_DATA_RESOLUTION_ORDER.index("entity_not_resolved")
    assert NO_DATA_RESOLUTION_ORDER[entity_index + 1] == "entity_ambiguous"
    assert NO_DATA_RESOLUTION_ORDER[entity_index + 2] == "entity_set_too_large"
    assert NO_DATA_RESOLUTION_ORDER[entity_index + 3] == "entity_role_not_admitted"


def test_resolution_order_slice_position():
    """binding_mismatch sits directly after no_matching_slice, per contract."""
    slice_index = NO_DATA_RESOLUTION_ORDER.index("no_matching_slice")
    assert NO_DATA_RESOLUTION_ORDER[slice_index + 1] == "binding_mismatch"


def test_resolution_order_currency_conversion_position():
    """rate_coverage_gap, currency-conversion's own no_data reason, sits
    directly after unit_not_resolved. recipe_not_declared, the other
    currency-conversion code, maps to unsupported_code rather than no_data_reason
    on the text doors and does not appear in this order."""
    unit_index = NO_DATA_RESOLUTION_ORDER.index("unit_not_resolved")
    assert NO_DATA_RESOLUTION_ORDER[unit_index + 1] == "rate_coverage_gap"


def test_election_is_order_independent():
    for a, b in itertools.product(NO_DATA_RESOLUTION_ORDER, repeat=2):
        assert elect_no_data_reason([a, b]) == elect_no_data_reason([b, a]), (a, b)


def test_adding_a_lower_ranked_reason_never_changes_the_winner():
    for a, b in itertools.product(NO_DATA_RESOLUTION_ORDER, repeat=2):
        winner = elect_no_data_reason([a, b])
        for lower in NO_DATA_RESOLUTION_ORDER[NO_DATA_RESOLUTION_ORDER.index(winner) + 1 :]:
            assert elect_no_data_reason([a, b, lower]) == winner, (a, b, lower)


@pytest.mark.parametrize("reasons", [[], ["no_observations", "not_a_reason"]])
def test_election_refuses_no_reason_and_an_unknown_one(reasons):
    with pytest.raises(ValueError):
        elect_no_data_reason(reasons)


def _literals(node: ast.expr | None) -> Iterator[str]:
    """The string literals `node` can evaluate to, through conditional expressions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.IfExp):
        yield from _literals(node.body)
        yield from _literals(node.orelse)


def _names_reason(annotation: ast.expr | None) -> bool:
    return annotation is not None and "NoDataReason" in ast.unparse(annotation)


def _reason_inputs(tree: ast.Module) -> Iterator[tuple[str, int | None, str]]:
    """Each callable's parameters and dataclass fields annotated `NoDataReason`, as
    (callable name, positional index or None, keyword)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            positional = [*node.args.posonlyargs, *node.args.args]
            if positional and positional[0].arg in {"self", "cls"}:
                positional = positional[1:]
            for index, arg in enumerate(positional):
                if _names_reason(arg.annotation):
                    yield node.name, index, arg.arg
            for arg in node.args.kwonlyargs:
                if _names_reason(arg.annotation):
                    yield node.name, None, arg.arg
        elif isinstance(node, ast.ClassDef):
            fields = [
                item
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
            for index, item in enumerate(fields):
                if _names_reason(item.annotation):
                    yield node.name, index, ast.unparse(item.target)


def _minted_reasons() -> list[tuple[str, int, str]]:
    """Every string literal `src` hands a `NoDataReason` parameter, field, default or
    return, as (file, line, literal): the reasons the code mints, found through the types
    of the no-data constructors (`ResolveState.no_data`, `NoDataDraft`, `RegionResult`)
    and of every other callable that takes or returns a reason, plus the values of the
    engine's narrower `...Reason` literal aliases."""
    trees = {str(path): ast.parse(path.read_text()) for path in sorted(_SRC.rglob("*.py"))}
    takes: dict[str, list[tuple[int | None, str]]] = {}
    for tree in trees.values():
        for name, index, keyword in _reason_inputs(tree):
            takes.setdefault(name, []).append((index, keyword))
    found: list[tuple[str, int, str]] = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.TypeAlias)
                and "/search/engine/" in path
                and ast.unparse(node.name).endswith("Reason")
                and isinstance(node.value, ast.Subscript)
                and ast.unparse(node.value.value) == "Literal"
            ):
                # A narrower reason vocabulary, such as `SourceRefusalReason`, whose values
                # reach a `NoDataReason` field through an attribute.
                values = node.value.slice
                elements = values.elts if isinstance(values, ast.Tuple) else [values]
                found.extend((path, node.lineno, v) for e in elements for v in _literals(e))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                args = node.args
                positional = [*args.posonlyargs, *args.args]
                with_defaults = positional[len(positional) - len(args.defaults) :]
                for arg, default in [
                    *zip(with_defaults, args.defaults, strict=True),
                    *zip(args.kwonlyargs, args.kw_defaults, strict=True),
                ]:
                    if _names_reason(arg.annotation):
                        found.extend((path, node.lineno, v) for v in _literals(default))
                if _names_reason(node.returns):
                    found.extend(
                        (path, ret.lineno, v)
                        for ret in ast.walk(node)
                        if isinstance(ret, ast.Return)
                        for v in _literals(ret.value)
                    )
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                for index, keyword in takes.get(name, ()):
                    passed = [kw.value for kw in node.keywords if kw.arg == keyword]
                    if index is not None and index < len(node.args):
                        passed.append(node.args[index])
                    found.extend((path, node.lineno, v) for arg in passed for v in _literals(arg))
    return found


def test_every_minted_reason_is_in_the_resolution_order():
    """Every reason a check mints has a rank, so `elect_no_data_reason` can elect it."""
    found = _minted_reasons()
    # The scan reaches the entity check and the engine's other refusals.
    assert {"entity_not_resolved", "entity_ambiguous", "entity_set_too_large"} <= {
        value for _path, _line, value in found
    }
    assert len(found) > 50
    unranked = [site for site in found if site[2] not in NO_DATA_RESOLUTION_ORDER]
    assert not unranked
