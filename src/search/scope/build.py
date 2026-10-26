"""`build_part_scope`: one part's `PartScope`, from its base identity, entries and records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from search.engine.errors import EngineInputError
from search.models import (
    SCOPE_AXES,
    BaseIdentity,
    CompiledScope,
    Component,
    DroppedEntry,
    Entry,
    EntryProvenance,
    IdentityBindingValue,
    PartScope,
    ScopeAxis,
    SpecDefinition,
)
from search.scope import rules as scope_rules
from search.scope.compile import Compiled, Refusal, compile
from search.scope.rules import AxisRule
from search.scope.state import AxisRecord
from search.scope.tables import CompileFacts, Conflict, ScopeFacts


def build_part_scope(
    *,
    base: BaseIdentity,
    entries: Sequence[Entry],
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
    rules: Mapping[ScopeAxis, AxisRule] | None = None,
    conflicts: tuple[Conflict, ...] | None = None,
    refuse: bool = True,
) -> PartScope | Refusal:
    """The scope of a part whose identity is `base`, or the refusal of its entries.

    Each rule (`rules.AXIS_RULES` unless `rules` is given) reads `base`, its axis's record
    and `facts`, in `SCOPE_AXES` order, and the axes whose rule returns a table are
    compiled with `entries` under `conflicts` (`rules.CONFLICTS` unless given). The entry a
    binder applied from the request's words or the conversation (`AxisRecord.entry`) joins
    them after the posted entries when it is not among them and its axis's table holds the
    option the record settled (`AxisTable.held`): a part whose axis is absent, or whose base
    no longer holds the binder's option, adds nothing, so a binder's reading never refuses a
    part. With `refuse` false, as on a fresh search, an entry compile refuses is dropped
    instead, reported in `dropped_entries` with the refusal's cause (a stated entry as
    `stated_choice_not_applied`, a carried one as `carried_choice_not_applied`), and the
    rest compile without it, so the result is always a `PartScope`. The scope's id is the
    base identity's spec_id. The server keeps no ledger, so the scope's is []."""
    rules = scope_rules.AXIS_RULES if rules is None else rules
    conflicts = scope_rules.CONFLICTS if conflicts is None else conflicts
    tables = tuple(
        table
        for axis in SCOPE_AXES
        if (rule := rules.get(axis)) is not None
        and (table := rule(base, records.get(axis), facts)) is not None
    )
    held = {table.axis: table.held for table in tables}
    recorded = [
        record.entry
        for record in records.values()
        if record.entry is not None
        and record.entry not in entries
        and held.get(record.axis) == record.active
    ]
    entries = [*entries, *recorded]
    facts_read = CompileFacts(axes=tables, conflicts=conflicts)
    refused: list[DroppedEntry] = []
    while isinstance(compiled := compile(base, entries, facts_read), Refusal):
        if refuse:
            return compiled
        entry = entries.pop(compiled.entry)
        refused.append(
            DroppedEntry(entry=entry, code=_NOT_APPLIED[entry.provenance], cause=compiled.cause)
        )
    return _part_scope(
        base, compiled, refused, rule_version=scope_rules.rule_version(rules, conflicts)
    )


_NOT_APPLIED: Mapping[
    EntryProvenance, Literal["stated_choice_not_applied", "carried_choice_not_applied"]
] = {"explicit": "stated_choice_not_applied", "carried": "carried_choice_not_applied"}


def _part_scope(
    base: BaseIdentity, compiled: Compiled, refused: Sequence[DroppedEntry], *, rule_version: str
) -> PartScope:
    return PartScope(
        id=compiled.base_spec_id,
        base=SpecDefinition(spec_id=compiled.base_spec_id, identity=base),
        rule_version=rule_version,
        entries=list(compiled.entries),
        dropped_entries=[*refused, *compiled.dropped],
        choices=list(compiled.choices),
        compiled=served_scope(compiled.spec_id, compiled.identity),
        ledger=[],
    )


def served_scope(spec_id: str, identity: BaseIdentity) -> CompiledScope:
    """The compiled record of a plain Spec: `identity` under `spec_id`, served whole from
    the provenance it binds."""
    provenance = (identity.facets or {}).get("provenance")
    return CompiledScope(
        spec_id=spec_id,
        identity=identity,
        components=[
            Component(
                role="whole",
                provenance=(
                    provenance.ref if isinstance(provenance, IdentityBindingValue) else None
                ),
                identity=identity,
            )
        ],
    )


def refusal_error(refusal: Refusal, *, pointer: str | None) -> EngineInputError:
    """`refusal` as a 422 naming its cause and the entry blocking it, at `pointer`."""
    blocker = "" if refusal.blocker is None else f", blocked by scope entry {refusal.blocker}"
    message = f"Scope entry {refusal.entry} cannot apply to this part: {refusal.cause}{blocker}."
    if refusal.code == "requested_option_unavailable":
        return EngineInputError(
            message, code="requested_option_unavailable", status=422, pointer=pointer
        )
    return EngineInputError(
        message, code="conflicting_requested_options", status=422, pointer=pointer
    )
