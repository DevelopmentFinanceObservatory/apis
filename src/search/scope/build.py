"""`build_part_scope`: one part's `PartScope`, from its base identity, entries and records."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

from dfo_platform.problem import Alternative
from search.engine.errors import EngineInputError
from search.engine.spec_id import identity_spec_id
from search.models import (
    SCOPE_AXES,
    BaseIdentity,
    CompiledScope,
    Component,
    DerivedSide,
    DerivedWhole,
    DroppedEntry,
    Entry,
    EntryProvenance,
    IdentityBindingValue,
    Option,
    PartScope,
    ProviderBinding,
    ScopeAxis,
    SideBinding,
    SpecDefinition,
)
from search.scope import rules as scope_rules
from search.scope.basis import SERVED_DIFFERS_RULE
from search.scope.compile import Compiled, Refusal, compile
from search.scope.composite import composite_components
from search.scope.edits import applies_in_pass, apply_binding
from search.scope.provider import elect_latest_window
from search.scope.rules import AxisRule
from search.scope.side import side_rule
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, CompileFacts, Conflict, ScopeFacts

WHOLE = "whole"
"""The option value an entry posts to ask for its axis's whole (`AxisTable.whole`)."""

SIDE_RULES: Mapping[ScopeAxis, AxisRule] = {
    "donor": side_rule("donor"),
    "recipient": side_rule("recipient"),
}
"""The side axes, read only where an entry posts their axis (`side`)."""


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
    compiled under `conflicts` (`rules.CONFLICTS` unless given) and the conflicts each table
    declares for its part (`AxisTable.conflicts`). What compiles is
    `entries`, the entries posted for the part (a dataset_resubmit's `scope_entries`, a
    same-shape merge's members' entries), and the entry each binder applied
    (`AxisRecord.entry`) while its axis's table holds the option the record settled
    (`AxisTable.held`), or the explicit entry on a partial option the record settled, which
    compile applies over the option's own window (`_compiles_over_window`). An entry posted with no concept on an axis whose table was asked for
    one takes the table's concept, so the entry the scope echoes attaches only to parts of
    that concept. `read` are the entries a fresh search read for the part
    (`ScopeState.attached`): none of them compiles of its own, so the choices state only
    what the answer serves, and each no record applied is reported in `dropped_entries`
    (a stated entry as `stated_choice_not_applied`, a carried one as
    `carried_choice_not_applied`) with the cause `_not_served` gives. With
    `refuse` false an entry compile refuses is dropped and reported the same way with the
    refusal's cause, and the rest compile without it, so the result is always a
    `PartScope`. An explicit entry is judged on `base` with the other explicit entries'
    edits applied (`_requested_tables`), so a request switching several axes compiles as
    the same switches one after the other. An option a binder settled over the identity
    it resolved that sets its source (the inclusive or bilateral provider scope, the OECD
    headline per year) is applied to `base` first (`_with_recorded_composites`), so the
    scope's base is the identity the answer serves.
    A composite asking for the latest year compiles to the one year its tables both cover
    (`provider.elect_latest_window`), while the base keeps the `latest` window. The scope's
    id is the base identity's spec_id. The server keeps no ledger, so the scope's is []."""
    rules = scope_rules.AXIS_RULES if rules is None else rules
    version_rules = rules
    asked = {entry.axis for entry in entries if entry.axis in SIDE_RULES}
    if asked:
        rules = {**rules, **{axis: SIDE_RULES[axis] for axis in asked}}
    conflicts = scope_rules.CONFLICTS if conflicts is None else conflicts
    tables = _tables(base, records, facts, rules)
    composed, settled = _with_recorded_composites(base, records, tables)
    if composed is not base or settled != records:
        base, records = composed, settled
        tables = _tables(base, records, facts, rules)
    applied = [
        record.entry
        for record in records.values()
        if record.entry is not None
        and (table := tables.get(record.axis)) is not None
        and (table.held == record.active or _compiles_over_window(record, table))
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
    asked_whole: Entry | None = None
    part_entries: list[Entry] = []
    whole_at = next(
        (
            n
            for n, entry in enumerate(compiling)
            if entry.provenance == "explicit" and entry.option_id.partition(":")[2] == WHOLE
        ),
        None,
    )
    if whole_at is not None:
        axes = _compile_facts(base, compiling, tables, records, facts, rules, conflicts).axes
        resolved = _whole_entry(compiling[whole_at], whole_at, axes)
        if isinstance(resolved, Refusal):
            return resolved
        compiling[whole_at] = asked_whole = resolved
        # The whole holds every axis but its own: the part's entry on that axis is the
        # part's, which the whole entry replaces.
        part_entries = [entry for entry in compiling if entry is not resolved]
        compiling = [e for e in compiling if e is resolved or e.axis != resolved.axis]
    while isinstance(
        compiled := compile(
            base,
            compiling,
            _compile_facts(base, compiling, tables, records, facts, rules, conflicts),
        ),
        Refusal,
    ):
        if refuse:
            return compiled
        entry = compiling.pop(compiled.entry)
        refused.append(
            DroppedEntry(entry=entry, code=_NOT_APPLIED[entry.provenance], cause=compiled.cause)
        )
    compiled = _served(compiled, facts)
    scope = _part_scope(
        base,
        compiled,
        refused,
        rule_version=scope_rules.rule_version(version_rules, conflicts),
        facts=facts,
    )
    if asked_whole is None or asked_whole not in compiled.entries:
        return scope
    without = part_entries
    part = compile(
        base, without, _compile_facts(base, without, tables, records, facts, rules, conflicts)
    )
    if isinstance(part, Refusal):
        return scope
    derived = _derived(_served(part, facts), compiled, asked_whole, facts=facts)
    return scope.model_copy(update={"derived": derived})


def _served(compiled: Compiled, facts: ScopeFacts) -> Compiled:
    """`compiled` over the one year its tables both cover where it asks for the latest
    (`provider.elect_latest_window`)."""
    served = elect_latest_window(compiled.identity, facts)
    if served is compiled.identity:
        return compiled
    return dataclasses.replace(compiled, identity=served, spec_id=identity_spec_id(served))


def _whole_entry(entry: Entry, index: int, axes: Sequence[AxisTable]) -> Entry | Refusal:
    """`entry`, asking for its axis's whole, as the entry of the option that holds it
    (`AxisTable.whole`), or the refusal of it: the axis is absent, it declares no whole (with
    the options a client may ask for in its place), or the part already holds it."""
    table = next((t for t in axes if t.axis == entry.axis), None)
    if table is None:
        return Refusal(code="requested_option_unavailable", entry=index, cause="axis_absent")
    if table.whole is None:
        return Refusal(
            code="requested_option_unavailable",
            entry=index,
            cause=table.whole_cause,
            alternatives=table.whole_alternatives,
        )
    if table.whole == table.held:
        return Refusal(code="requested_option_unavailable", entry=index, cause="part_is_whole")
    return entry.model_copy(update={"option_id": table.whole})


def _derived(part: Compiled, whole: Compiled, entry: Entry, *, facts: ScopeFacts) -> DerivedWhole:
    """What the part and its whole count: each one's spec_id, its option on the axis the
    whole opens and on the provider axis, the parties each side names, and the years both
    serve."""
    option = next(
        o
        for c in whole.choices
        if c.axis == entry.axis
        for o in c.options
        if o.option_id == entry.option_id
    )
    binding = option.binding
    kind = (
        "option"
        if not isinstance(binding, SideBinding)
        else "published_total"
        if binding.refs
        else "sum_of_members"
    )
    return DerivedWhole(
        axis=entry.axis,
        kind=kind,
        part=_derived_side(part, entry.axis),
        whole=_derived_side(whole, entry.axis),
        years=_shared_years(
            compiled_scope(part.spec_id, part.identity, facts=facts),
            compiled_scope(whole.spec_id, whole.identity, facts=facts),
        ),
    )


def _derived_side(compiled: Compiled, axis: ScopeAxis) -> DerivedSide:
    actives = {choice.axis: choice.active for choice in compiled.choices}
    return DerivedSide(
        spec_id=compiled.spec_id,
        option_id=actives[axis],
        provider=actives.get("provider"),
        populations={key: list(dcids) for key, dcids in compiled.identity.entities.items()},
    )


def _shared_years(part: CompiledScope, whole: CompiledScope) -> list[tuple[int, int]] | None:
    """The runs of years both figures serve, each over the union of its components'
    periods; None where neither states its periods."""
    served = [
        {
            year
            for component in scope.components
            for first, last in component.periods or ()
            for year in range(first, last + 1)
        }
        for scope in (part, whole)
        if any(component.periods is not None for component in scope.components)
    ]
    if not served:
        return None
    years = set.intersection(*served)
    runs: list[tuple[int, int]] = []
    for year in sorted(years):
        if runs and runs[-1][1] == year - 1:
            runs[-1] = (runs[-1][0], year)
        else:
            runs.append((year, year))
    return runs


def _compile_facts(
    base: BaseIdentity,
    entries: Sequence[Entry],
    tables: Mapping[ScopeAxis, AxisTable],
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
    rules: Mapping[ScopeAxis, AxisRule],
    conflicts: Sequence[Conflict],
) -> CompileFacts:
    """What `entries` compile under: the tables `_requested_tables` reads, and `conflicts`
    with those the tables declare for the part (`AxisTable.conflicts`)."""
    held = (*conflicts, *(c for table in tables.values() for c in table.conflicts))
    axes = _requested_tables(base, entries, tables, records, facts, rules, held)
    return CompileFacts(
        axes=axes, conflicts=(*conflicts, *(c for table in axes for c in table.conflicts))
    )


def _tables(
    base: BaseIdentity,
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
    rules: Mapping[ScopeAxis, AxisRule],
) -> dict[ScopeAxis, AxisTable]:
    """The table of every axis whose rule reads one off `base`, in `SCOPE_AXES` order."""
    return {
        table.axis: table
        for axis in SCOPE_AXES
        if (rule := rules.get(axis)) is not None
        and (table := rule(base, records.get(axis), facts)) is not None
    }


def _requested_tables(
    base: BaseIdentity,
    entries: Sequence[Entry],
    tables: Mapping[ScopeAxis, AxisTable],
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
    rules: Mapping[ScopeAxis, AxisRule],
    conflicts: Sequence[Conflict],
) -> tuple[AxisTable, ...]:
    """`tables` in order, the table of each axis an explicit entry names read off `base`
    with the edits of the explicit entries on the other axes applied, so each requested
    option is judged on the identity the other requested edits produce, and several edits
    in one request compile as they do one after the other. An entry edits `base` where its
    axis's table offers its option, `base` does not hold it, it is not unavailable, and it
    conflicts with no other explicit entry's option (`conflicts`), so compile's conflict
    rule decides such a pair on the tables of `base`. An axis whose rule reads no table off
    the edited identity keeps the table of `base`."""
    explicit = [entry for entry in entries if entry.provenance == "explicit"]
    pairs = {frozenset((c.first, c.second)) for c in conflicts}
    edits: dict[ScopeAxis, Option] = {}
    for entry in explicit:
        table = tables.get(entry.axis)
        if table is None or entry.axis in edits:
            continue
        option = table.option(entry.option_id)
        if (
            option is not None
            and option.option_id != table.held
            and option.availability.status != "unavailable"
            and not any(frozenset((option.option_id, e.option_id)) in pairs for e in explicit)
        ):
            edits[entry.axis] = option
    requested = dict(tables)
    for axis in {entry.axis for entry in explicit}:
        if axis not in tables or not edits.keys() - {axis}:
            continue
        edited = base
        for other, table in tables.items():
            if other != axis and (option := edits.get(other)) is not None:
                edited = apply_binding(edited, option.binding, [o.binding for o in table.options])
        if (table := rules[axis](edited, records.get(axis), facts)) is not None:
            requested[axis] = table
    return tuple(requested.values())


def _with_recorded_composites(
    base: BaseIdentity,
    records: Mapping[ScopeAxis, AxisRecord],
    tables: Mapping[ScopeAxis, AxisTable],
) -> tuple[BaseIdentity, dict[ScopeAxis, AxisRecord]]:
    """`base` with the source each option a binder settled over it and `base` does not
    hold yet sets (`edits.applies_in_pass`): the inclusive or bilateral provider scope
    (`provider.settle_provider`), or the OECD headline per year
    (`phase_election._record_basis`), with the records it then holds. The base keeps its
    own window: an option's window is applied when the scope compiles
    (`provider.elect_latest_window`). A partial option an explicit entry asked for is left
    to compile, which applies it over its own window (`_compiles_over_window`), so the base
    keeps the years the part asked for. Any other option the part cannot serve
    (`OptionAvailability.status` not `available`) is not applied. Its axis keeps the option
    the base holds, settled by default, under that option's own rule (`AxisTable.rule_of`),
    or `SERVED_DIFFERS_RULE` on the measure basis, since the served basis is then not the
    convention's. The scope's base is then the identity whose options the answer serves, so
    a resubmit of it keeps them."""
    settled = dict(records)
    for axis, table in tables.items():
        record = records.get(axis)
        if record is None or record.active == table.held:
            continue
        option = table.option(record.active)
        if (
            option is None
            or not applies_in_pass(option.binding)
            or _compiles_over_window(record, table)
        ):
            continue
        if option.availability.status != "available":
            settled[axis] = AxisRecord(
                axis=axis,
                rule=SERVED_DIFFERS_RULE if axis == "measure_basis" else table.rule_of(table.held),
                active=table.held,
                provenance="default",
            )
            continue
        binding = option.binding
        if isinstance(binding, ProviderBinding):
            binding = binding.model_copy(update={"window": None})
        base = apply_binding(base, binding, [o.binding for o in table.options])
    return base, settled


def _compiles_over_window(record: AxisRecord, table: AxisTable) -> bool:
    """Whether `record` settled a partial option of `table` on an explicit entry, which the
    scope compiles as a switch to it compiles: over the latest run of years the option
    covers, the window its binding sets."""
    option = table.option(record.active)
    return (
        record.entry is not None
        and record.entry.provenance == "explicit"
        and option is not None
        and option.availability.status == "partial"
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
    reason when it is partial or unavailable), or `not_served` when it is available."""
    if any(other.axis == entry.axis for other in applied):
        return "entry_on_axis"
    table = tables.get(entry.axis)
    if table is None:
        return "axis_absent"
    option = table.option(entry.option_id)
    if option is None:
        return "option_unknown"
    if option.availability.status != "available":
        return option.availability.code or f"option_{option.availability.status}"
    return "not_served"


_NOT_APPLIED: Mapping[
    EntryProvenance, Literal["stated_choice_not_applied", "carried_choice_not_applied"]
] = {"explicit": "stated_choice_not_applied", "carried": "carried_choice_not_applied"}


def _part_scope(
    base: BaseIdentity,
    compiled: Compiled,
    refused: Sequence[DroppedEntry],
    *,
    rule_version: str,
    facts: ScopeFacts,
) -> PartScope:
    return PartScope(
        id=compiled.base_spec_id,
        base=SpecDefinition(spec_id=compiled.base_spec_id, identity=base),
        rule_version=rule_version,
        entries=list(compiled.entries),
        dropped_entries=[*refused, *compiled.dropped],
        choices=list(compiled.choices),
        compiled=compiled_scope(compiled.spec_id, compiled.identity, facts=facts),
        ledger=[],
    )


def compiled_scope(spec_id: str, identity: BaseIdentity, *, facts: ScopeFacts) -> CompiledScope:
    """The compiled record of `identity` under `spec_id`: one component per component of a
    composite (`composite.composite_components`), with the years each serves in `facts`'
    coverage snapshot, else the plain Spec's one (`served_scope`)."""
    components = composite_components(identity, facts=facts)
    if components is None:
        return served_scope(spec_id, identity)
    return CompiledScope(spec_id=spec_id, identity=identity, components=components)


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
    alternatives = [Alternative(ref=ref, label=ref) for ref in refusal.alternatives] or None
    return EngineInputError(
        message, code=refusal.code, status=422, pointer=pointer, alternatives=alternatives
    )
