"""Derive seed constraints and verify them against a resubmit's base identity."""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.errors import EngineInputError
from search.engine.retrieval.bound_values import _expand_values
from search.engine.retrieval.facts import read_constraints, read_unresolved_constraints
from search.models import BaseIdentity, IdentityBindingSet, IdentityBindingValue


def _distinct_asserted_values(
    stat_var_dcids: list[str], constraints_by_sv: dict[str, dict[str, str]], prop: str
) -> list[str]:
    """Return distinct asserted values in SV order, skipping members without `prop`."""
    per_sv_values: list[str] = []
    for sv in stat_var_dcids:
        sv_constraints = constraints_by_sv[sv]
        if prop in sv_constraints:
            per_sv_values.append(sv_constraints[prop])
    return list(dict.fromkeys(per_sv_values))


def _seed_effective_constraints(
    stat_var_dcids: list[str], confirmed_arcs: dict[str, dict]
) -> dict[str, frozenset[str]]:
    """Map each asserted property to its distinct seed values.

    Frozensets make comparison order-independent. A seed can assert only part
    of a base binding or omit a property. Any asserted value outside the base
    binding fails the subset check.
    """
    constraints_by_sv = {sv: read_constraints(confirmed_arcs[sv]) for sv in stat_var_dcids}
    all_props: set[str] = set()
    for constraints in constraints_by_sv.values():
        all_props.update(constraints)

    effective: dict[str, frozenset[str]] = {}
    for prop in all_props:
        distinct_values = _distinct_asserted_values(stat_var_dcids, constraints_by_sv, prop)
        if distinct_values:
            effective[prop] = frozenset(distinct_values)
    return effective


def _seed_unresolved_props(
    stat_var_dcids: list[str], confirmed_arcs: dict[str, dict]
) -> frozenset[str]:
    """Return properties present but unresolved on any seed SV.

    These explicitly contradict a base-bound property. An absent property is
    treated as silent instead.
    """
    unresolved: set[str] = set()
    for sv in stat_var_dcids:
        unresolved.update(read_unresolved_constraints(confirmed_arcs[sv]))
    return frozenset(unresolved)


def _check_seed_matches_base_identity(
    base_identity: BaseIdentity,
    seed_constraints: dict[str, frozenset[str]],
    widened_props: set[str],
    unresolved_props: frozenset[str] = frozenset(),
) -> None:
    """Refuse seed values that contradict a base-bound what/how property.

    Every asserted seed value must belong to the base binding. Equality would
    reject valid narrowed seeds and members that omit a property. Unbound
    `widened_props` are exempt. A property in `unresolved_props` explicitly
    contradicts a base-bound axis because its asserted value is unknown.
    Check unresolved properties before subset comparison, which treats missing
    seed values as silent. This check compares the seed to the base identity.
    edit detection compares only the posted definition to the seed.

    Raise `request_invalid` (400) for the first contradiction in slot order.
    """
    for slot in base_identity.slots:
        if slot.property in widened_props:
            continue
        binding = slot.binding
        if isinstance(binding, IdentityBindingValue):
            posted = frozenset({binding.ref})
        elif isinstance(binding, IdentityBindingSet):
            posted = frozenset(binding.refs)
        else:
            continue
        # Expand rollups to leaf values to match the StatVar arcs' assertions.
        target = frozenset(_expand_values(get_index(), sorted(posted)))
        if slot.property in unresolved_props:
            raise EngineInputError(
                f"resolved_stat_var_dcids' own constraint for {slot.property!r} is "
                "present but unresolved (the graph could not read its value); "
                f"base_identity claims this axis is bound to ({sorted(posted)!r}), "
                "and a present-but-unresolved value cannot be confirmed to agree",
                code="request_invalid",
                pointer="/input/resolved_stat_var_dcids",
            )
        seed_values = seed_constraints.get(slot.property, frozenset())
        if not seed_values <= target:
            raise EngineInputError(
                f"resolved_stat_var_dcids' own constraint for {slot.property!r} "
                f"({sorted(seed_values)!r}) asserts a value outside "
                f"base_identity's bound value(s) ({sorted(posted)!r}); the "
                "seed does not correspond to the posted base",
                code="request_invalid",
                pointer="/input/resolved_stat_var_dcids",
            )
