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
from search.scope.tables import AxisTable, CompileFacts, Conflict, ScopeFacts


def build_part_scope(
    *,
    base: BaseIdentity,
    entries: Sequence[Entry],
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
    read: Sequence[Entry] = (),
    rules: Mapping[ScopeAxis, AxisRule] | None = None,
    conflicts: tuple[Conflict, ...] | None = None,
    refuse: bool = True,
) -> PartScope | Refusal:
    """The scope of a part whose identity is `base`, or the refusal of its entries.

    Each rule (`rules.AXIS_RULES` unless `rules` is given) reads `base`, its axis's record
    and `facts`, in `SCOPE_AXES` order, and the axes whose rule returns a table are
    compiled under `conflicts` (`rules.CONFLICTS` unless given). What compiles is
    `entries`, the entries posted for the part (a dataset_resubmit's `scope_entries`, a
    same-shape merge's members' entries), and the entry each binder applied
    (`AxisRecord.entry`) while its axis's table holds the option the record settled
    (`AxisTable.held`). An entry posted with no concept on an axis whose table was asked for
    one takes the table's concept, so the entry the scope echoes attaches only to parts of
    that concept. `read` are the entries a fresh search read for the part
    (`ScopeState.attached`): none of them compiles of its own, so the choices state only
    what the answer serves, and each no record applied is reported in `dropped_entries`
    (a stated entry as `stated_choice_not_applied`, a carried one as
    `carried_choice_not_applied`) with the cause `_not_served` gives. With
    `refuse` false an entry compile refuses is dropped and reported the same way with the
    refusal's cause, and the rest compile without it, so the result is always a
    `PartScope`. The scope's id is the base identity's spec_id. The server keeps no ledger,
    so the scope's is []."""
    rules = scope_rules.AXIS_RULES if rules is None else rules
    conflicts = scope_rules.CONFLICTS if conflicts is None else conflicts
    tables = {
        table.axis: table
        for axis in SCOPE_AXES
        if (rule := rules.get(axis)) is not None
        and (table := rule(base, records.get(axis), facts)) is not None
    }
    applied = [
        record.entry
        for record in records.values()
        if record.entry is not None
        and (table := tables.get(record.axis)) is not None
        and table.held == record.active
    ]
    posted = [_with_concept(entry, tables) for entry in entries]
    compiling = [*posted, *(entry for entry in applied if entry not in posted)]
    refused = [
        DroppedEntry(
            entry=entry,
            code=_NOT_APPLIED[entry.provenance],
            cause=_not_served(entry, tables, applied),
        )
        for n, entry in enumerate(read)
        if entry not in applied and entry not in read[:n]
    ]
    facts_read = CompileFacts(axes=tuple(tables.values()), conflicts=conflicts)
    while isinstance(compiled := compile(base, compiling, facts_read), Refusal):
        if refuse:
            return compiled
        entry = compiling.pop(compiled.entry)
        refused.append(
            DroppedEntry(entry=entry, code=_NOT_APPLIED[entry.provenance], cause=compiled.cause)
        )
    return _part_scope(
        base, compiled, refused, rule_version=scope_rules.rule_version(rules, conflicts)
    )


def _with_concept(entry: Entry, tables: Mapping[ScopeAxis, AxisTable]) -> Entry:
    """`entry` with the concept its axis's table was asked for, when it names none."""
    table = tables.get(entry.axis)
    if entry.concept is not None or table is None or table.concept is None:
        return entry
    return entry.model_copy(update={"concept": table.concept})


def _not_served(
    entry: Entry, tables: Mapping[ScopeAxis, AxisTable], applied: Sequence[Entry]
) -> str:
    """Why the binders did not serve `entry`: another entry they applied settles its axis
    (`entry_on_axis`), the axis is absent, the option is unknown or unavailable (its
    reason), or `not_served` when it is available."""
    if any(other.axis == entry.axis for other in applied):
        return "entry_on_axis"
    table = tables.get(entry.axis)
    if table is None:
        return "axis_absent"
    option = table.option(entry.option_id)
    if option is None:
        return "option_unknown"
    if option.availability.status == "unavailable":
        return option.availability.code or "option_unavailable"
    return "not_served"


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
    return EngineInputError(message, code=refusal.code, status=422, pointer=pointer)
