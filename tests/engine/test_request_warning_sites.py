"""Every write to a fresh pass's warnings is declared, so a definite answer read again within
its composite's population (`pipeline._within_population`) keeps exactly the warnings the
request's words raised.

`ResolveState.request_warnings` snapshots the pass's warnings before it interprets its read,
and a warning raised from the request's words after that is recorded through
`ResolveState.from_request`. The population read replaces every other warning about the
answer (`composite_answer.with_population_read`), so a write after the snapshot that is
neither derived from the read's rows nor recorded through `from_request` would drop a warning
the request's words raised. The sweep reads every write site under `search/engine` from the
source and holds it to `_SITES`: a new site fails until it is declared, and a site declared
`request` fails unless its function calls `state.from_request`.
"""

from __future__ import annotations

import ast
import pathlib
from collections import Counter
from typing import cast

_ENGINE = pathlib.Path(__file__).resolve().parents[2] / "src" / "search" / "engine"

# Callables that only read the list they are passed.
_READERS = frozenset(
    {"tuple", "len", "list", "set", "frozenset", "sorted", "enumerate", "any", "all", "iter"}
)
_MUTATORS = frozenset({"append", "extend", "insert", "remove", "clear", "pop", "sort", "reverse"})

type Site = tuple[str, str, str]


def _is_warnings(node: ast.AST, *, module: str) -> bool:
    """Whether `node` is a pass's warnings list: `state.warnings`, or `self.warnings` inside
    the state module."""
    if not (isinstance(node, ast.Attribute) and node.attr == "warnings"):
        return False
    owner = node.value
    names = {"state", "self"} if module == "region/state" else {"state"}
    return isinstance(owner, ast.Name) and owner.id in names


def _site_ops(node: ast.AST, *, module: str, writers: frozenset[str]) -> list[str]:
    """The writes `node` makes to a pass's warnings, one op each: a list method, a call that
    is passed the list, a call of a state method that writes it (`writers`), an alias of the
    list, or an assignment to it or an item of it."""
    ops: list[str] = []
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and _is_warnings(func.value, module=module):
            if func.attr in _MUTATORS:
                ops.append(func.attr)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "state"
            and func.attr in writers
        ):
            ops.append(f"state.{func.attr}")
        passed = [*node.args, *(keyword.value for keyword in node.keywords)]
        reader = isinstance(func, ast.Name) and func.id in _READERS
        if not reader and any(_is_warnings(arg, module=module) for arg in passed):
            ops.append(f"passed:{ast.unparse(func)}")
    if (
        isinstance(node, ast.Assign | ast.AnnAssign)
        and node.value is not None
        and _is_warnings(node.value, module=module)
    ):
        ops.append("alias")
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign | ast.Delete):
        targets = list(node.targets)
    elif isinstance(node, ast.AugAssign | ast.AnnAssign):
        targets = [node.target]
    for target in targets:
        held = target.value if isinstance(target, ast.Subscript) else target
        if _is_warnings(held, module=module):
            ops.append("assign")
    return ops


def _functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _trees() -> dict[str, ast.Module]:
    """Every module under `search/engine`, parsed, by its path below it without suffix."""
    return {
        path.relative_to(_ENGINE).with_suffix("").as_posix(): ast.parse(
            path.read_text(), filename=str(path)
        )
        for path in sorted(_ENGINE.rglob("*.py"))
    }


def _scanned(trees: dict[str, ast.Module]) -> dict[Site, int]:
    """Every write site under `search/engine`, by module, innermost function and op, with
    how many times it occurs there."""
    sites: Counter[Site] = Counter()

    def visit(node: ast.AST, module: str, function: str, writers: frozenset[str]) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        for op in _site_ops(node, module=module, writers=writers):
            sites[(module, function, op)] += 1
        for child in ast.iter_child_nodes(node):
            visit(child, module, function, writers)

    visit(trees["region/state"], "region/state", "<module>", frozenset())
    writers = frozenset(function for module, function, _ in sites if module == "region/state")
    for module, tree in trees.items():
        if module != "region/state":
            visit(tree, module, "<module>", writers)
    return dict(sites)


_R, _Q, _B, _P, _M = "read", "request", "before", "resubmit", "method"

# Each write site, by module, function and op, with the kind of each occurrence in source
# order:
# - before: runs before the pass snapshots its warnings (`ResolveState.request_warnings`);
# - read: derived from the rows the read returned, which the population read replaces;
# - request: raised from the request's words after the snapshot, so the function records it
#   through `state.from_request`;
# - keeps_key: rewrites a held warning's params or drops held warnings, raising none, so a
#   snapshot warning keeps its key (`departures.warning_key`);
# - resubmit: the resubmit pass (`_PromoteState`), which takes no snapshot;
# - method: a state method, whose callers are the sites.
_SITES: dict[Site, tuple[str, ...]] = {
    ("region/candidate_collapse", "_assemble_candidate_set", "append"): (_Q,),
    ("region/candidate_collapse", "_assemble_candidate_set", "extend"): (_R,),
    ("region/candidate_collapse", "_assemble_definite", "append"): (_R,),
    ("region/candidate_collapse", "_assemble_definite", "passed:_attach_observations"): (_R, _R),
    ("region/candidate_collapse", "_assemble_definite", "passed:install_place_warnings"): (_Q,),
    ("region/candidate_collapse", "assemble_candidates", "state.state_reelected_reading"): (
        "keeps_key",
    ),
    ("region/candidate_collapse", "_handle_sparse_survivors", "append"): (_Q, _Q),
    (
        "region/candidate_collapse",
        "_offer_place_warnings",
        "passed:stamp_offered_place_warnings",
    ): (_Q,),
    ("region/candidate_grounding", "_warn_for_candidates", "extend"): (_R, _R, _R),
    (
        "region/candidate_grounding",
        "restate_scope_warning",
        "passed:restate_role_scope_warnings",
    ): (_R,),
    ("region/candidate_options", "_attach_candidate_entity_options", "append"): (_R,),
    ("region/candidate_options", "_attach_candidate_facet_options", "append"): (_R,),
    ("region/candidate_options", "_attach_candidate_slot_options", "append"): (_R,),
    ("region/phase_answer", "_bound_unit_refusal", "append"): (_R,),
    ("region/phase_answer", "_finalize_definite_result", "passed:_attach_observations"): (_R, _R),
    ("region/phase_answer", "_assemble_definite_answer", "state.state_reelected_reading"): (
        "keeps_key",
    ),
    ("region/phase_answer", "_gather_definite_coverage", "append"): (_R, _R, _R),
    # A member's family total is absent from the rows read; the answer is no_data.
    ("region/phase_answer", "_ground_elected_answer", "state.depart"): (_R,),
    ("region/phase_answer", "_ground_elected_answer", "extend"): (_R,),
    (
        "region/phase_answer",
        "_install_filtered_answer",
        "passed:restate_role_scope_warnings",
    ): (_R,),
    # The years the request asked beside those the conversion tables cover; the answer's
    # identity holds the narrowed window, so its population read departs nothing.
    ("region/phase_answer", "_narrowed_to_period", "state.depart"): (_Q,),
    # A trailing request's stale and passed-over years name the span its words asked, then
    # the deflator omission the read's plan gives.
    ("region/phase_answer", "_over_conversion_period", "state.depart"): (_Q, _Q, _Q, _R),
    ("region/phase_answer", "_start_assemble_definite", "append"): (_R,),
    ("region/phase_election", "_install_settled", "extend"): (_B,),
    ("region/phase_election", "_repair_bindings", "append"): (_B, _B),
    ("region/phase_election", "_repair_bindings", "extend"): (_B, _B),
    ("region/phase_election", "_vet_election", "append"): (_B,),
    ("region/phase_place_materialise", "_install_shape", "passed:install_place_warnings"): (_B,),
    ("region/phase_place_materialise", "_unbind_refuted_claims", "state.forget_bound"): (
        "keeps_key",
    ),
    ("region/phase_place_materialise", "_nearest_real_for", "append"): (_R,),
    ("region/phase_place_materialise", "_nearest_real_for", "extend"): (_R,),
    ("region/phase_place_materialise", "_nearest_real_for", "passed:_attach_observations"): (_R,),
    (
        "region/phase_place_materialise",
        "_nearest_real_for",
        "passed:stamp_offered_place_warnings",
    ): (_Q,),
    ("region/phase_place_materialise", "_note_named_places", "append"): (_B,),
    (
        "region/phase_place_materialise",
        "_phase_materialise",
        "passed:interpret_materialised",
    ): (_R,),
    ("region/phase_place_materialise", "_phase_materialise", "passed:settle_place"): (_R,),
    ("region/phase_place_materialise", "_phase_place_bindings", "append"): (_B,),
    ("region/phase_place_materialise", "_reelect_zero_observation_sibling", "append"): (_B,),
    ("region/phase_place_materialise", "note_denominator_unverified", "append"): (_Q,),
    ("region/phase_rank_bind", "_phase_bind", "append"): (_B,),
    ("region/phase_rank_bind", "_phase_prune_and_rank", "extend"): (_B,),
    ("region/phase_rank_bind", "_read_availability", "append"): (_B,),
    ("region/phase_shared", "_elected_flagged_match", "extend"): (_R,),
    # The raw-text door's unit phrase the answer does not apply.
    ("region/phase_shared", "withdraw_typed_unit", "state.depart"): (_Q,),
    ("region/promote_finalize", "_bound_unit_refusal", "append"): (_P,),
    ("region/promote_ground", "_ground_and_build_spec", "state.depart"): (_P,),
    (
        "region/promote_ground",
        "_install_narrowed_answer",
        "passed:restate_role_scope_warnings",
    ): (_P,),
    ("region/promote_ground", "_narrowed_to_period", "append"): (_P,),
    ("region/promote_ground", "_narrowed_to_period", "state.depart"): (_P,),
    ("region/promote_ground", "_over_conversion_period", "append"): (_P,),
    ("region/promote_ground", "_over_conversion_period", "state.depart"): (_P,),
    ("region/promote_ground", "_phase_ground_build", "state.depart"): (_P,),
    ("region/promote_materialise", "_phase_materialise", "passed:interpret_materialised"): (_P,),
    ("region/promote_materialise", "_phase_materialise", "passed:settle_place"): (_P,),
    ("region/promote_places", "_phase_where_places", "extend"): (_P,),
    ("region/promote_reread", "_phase_reread_arcs", "append"): (_P,),
    ("region/promote_shape", "_phase_build_shape", "alias"): (_P,),
    ("region/state", "depart", "append"): (_M,),
    ("region/state", "forget_bound", "assign"): (_M,),
    ("region/state", "state_reelected_reading", "assign"): (_M,),
    ("region/unrealisable", "unbind_unrealisable", "state.forget_bound"): ("keeps_key",),
}


def _function(trees: dict[str, ast.Module], module: str, name: str) -> ast.AST:
    (function,) = [node for node in _functions(trees[module]) if node.name == name]
    return function


def test_every_write_to_a_pass_s_warnings_is_declared() -> None:
    """A write site the table does not declare, or declares a different number of times,
    fails until its kind is declared."""
    declared = {site: len(kinds) for site, kinds in _SITES.items()}

    assert _scanned(_trees()) == declared


def test_a_request_site_records_its_warning_through_from_request() -> None:
    """A function declaring a warning raised from the request's words after the snapshot
    calls `state.from_request`, so a population read keeps the warning."""
    trees = _trees()
    unrecorded = [
        (module, function)
        for (module, function, _), kinds in _SITES.items()
        if _Q in kinds
        and not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_request"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "state"
            for node in ast.walk(_function(trees, module, function))
        )
    ]

    assert unrecorded == []


def test_a_resubmit_site_writes_the_resubmit_pass_s_warnings() -> None:
    """A site declared `resubmit` writes a `_PromoteState`, the resubmit pass's state,
    which takes no snapshot."""
    trees = _trees()
    misplaced = [
        (module, function)
        for (module, function, _), kinds in _SITES.items()
        if _P in kinds
        and not any(
            arg.arg == "state"
            and arg.annotation is not None
            and ast.unparse(arg.annotation) == "_PromoteState"
            for arg in cast("ast.FunctionDef", _function(trees, module, function)).args.args
        )
    ]

    assert misplaced == []
