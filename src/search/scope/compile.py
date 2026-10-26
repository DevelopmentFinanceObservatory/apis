"""`compile`: a base identity, its scope entries and the axis tables to one selection.

Pure: it reads only its arguments and does no I/O. The rules it holds to:

- An explicit entry is applied or refused, never moved. One naming an absent axis, an
  option the axis does not offer, or an unavailable option refuses
  `requested_option_unavailable`; two that cannot hold together refuse
  `conflicting_requested_options`.
- Entries are weighed in three tiers: explicit entries read from the request's words,
  then carried entries, then switches (explicit entries whose origin is a choice). A
  carried entry yields to every explicit entry: one the part cannot take (its axis absent,
  its option unknown or unavailable, its axis settled by an explicit or earlier carried
  entry, or in conflict with an entry read from the words) is dropped and reported as
  `carried_choice_not_applied`. A switch that conflicts with an applied entry is refused,
  so a switch never moves an explicit or carried entry. Carried entries are weighed in
  axis order, then by option id, so the order they are posted in never matters.
- A partial option (one serving fewer years than the part asks for, with its own shorter
  window) applies only to a switch, since only a click may change the period asked for. A
  stated or carried entry asking for it is refused or dropped with the option's reason.
- Every other axis takes its derived default: the first option of its preference that is
  available and that no entry, and no default derived before it, conflicts with.
- An option that would conflict with an entry on another axis is unavailable, with
  `conflicts_with` naming the entry, which is the switch compile refuses. One that
  conflicts only with a default stays available, since the default moves.
- The identity is edited only on axes whose active option differs from the one the base
  holds. A selection the base already holds, a switch to the active option among them,
  compiles to the base and its spec_id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from search.engine.spec_id import identity_spec_id, mint_identity
from search.models import (
    BaseIdentity,
    ChoiceProvenance,
    DroppedEntry,
    Entry,
    Option,
    OptionAvailability,
    ScopeAxis,
    ScopeChoice,
)
from search.scope.edits import apply_binding
from search.scope.tables import AxisTable, CompileFacts, Conflict

RefusalCode = Literal["requested_option_unavailable", "conflicting_requested_options"]


@dataclass(frozen=True, kw_only=True)
class Compiled:
    """A compiled selection.

    `entries` are the entries applied, in axis order, and `dropped` the carried entries
    left out. `choices` hold one choice per axis with every option's availability.
    `identity` is what the active options compile to and `spec_id` its id; `base_spec_id`
    is the base identity's."""

    entries: tuple[Entry, ...]
    dropped: tuple[DroppedEntry, ...]
    choices: tuple[ScopeChoice, ...]
    identity: BaseIdentity
    spec_id: str
    base_spec_id: str


@dataclass(frozen=True, kw_only=True)
class Refusal:
    """A selection compile refuses. `entry` is the index of the refused entry among those
    posted, `blocker` the index of the entry it conflicts with, None when none does, and
    `cause` names what refused it."""

    code: RefusalCode
    entry: int
    cause: str
    blocker: int | None = None


@dataclass(frozen=True)
class _Settled:
    option_id: str
    provenance: ChoiceProvenance
    index: int | None


def _conflicts(conflicts: Sequence[Conflict]) -> Mapping[frozenset[str], str]:
    return {frozenset((c.first, c.second)): c.cause for c in conflicts}


def _unavailable(tables: Mapping[ScopeAxis, AxisTable], entry: Entry) -> str | None:
    """Why `entry` cannot apply on its own, None when it can."""
    table = tables.get(entry.axis)
    if table is None:
        return "axis_absent"
    option = table.option(entry.option_id)
    if option is None:
        return "option_unknown"
    if option.availability.status == "unavailable":
        return option.availability.code or "option_unavailable"
    switched = entry.provenance == "explicit" and entry.origin.kind == "choice"
    if option.availability.status == "partial" and not switched:
        return option.availability.code or "option_partial"
    return None


def _first_conflict(
    option_id: str,
    settled: Mapping[ScopeAxis, _Settled],
    conflicts: Mapping[frozenset[str], str],
    *,
    axis: ScopeAxis,
) -> tuple[_Settled, str] | None:
    """The first option settled on another axis that `option_id` conflicts with, and why."""
    for other_axis, other in settled.items():
        if other_axis == axis:
            continue
        cause = conflicts.get(frozenset((option_id, other.option_id)))
        if cause is not None:
            return other, cause
    return None


def compile(
    base: BaseIdentity, entries: Sequence[Entry], facts: CompileFacts
) -> Compiled | Refusal:
    """The selection `entries` make over `base`, under `facts`, or the refusal of it."""
    tables = {table.axis: table for table in facts.axes}
    order = [table.axis for table in facts.axes]
    conflicts = _conflicts(facts.conflicts)

    explicit = [(i, e) for i, e in enumerate(entries) if e.provenance == "explicit"]
    explicit_axes = {entry.axis for _, entry in explicit}
    fixed: dict[ScopeAxis, _Settled] = {}
    for i, entry in explicit:
        if entry.origin.kind == "text" and (
            refusal := _admit_explicit(i, entry, fixed, tables, conflicts)
        ):
            return refusal

    rank = {axis: n for n, axis in enumerate(order)}
    carried = sorted(
        ((i, e) for i, e in enumerate(entries) if e.provenance == "carried"),
        key=lambda pair: (rank.get(pair[1].axis, len(rank)), pair[1].option_id, pair[0]),
    )
    dropped: list[DroppedEntry] = []
    for i, entry in carried:
        if entry.axis in explicit_axes or entry.axis in fixed:
            cause = "entry_on_axis"
        else:
            cause = _unavailable(tables, entry)
            if cause not in (None, "axis_absent", "option_unknown"):
                cause = "option_unavailable"
            if cause is None and _first_conflict(
                entry.option_id, fixed, conflicts, axis=entry.axis
            ):
                cause = "conflicts_with_entry"
        if cause is not None:
            dropped.append(DroppedEntry(entry=entry, cause=cause))
            continue
        fixed[entry.axis] = _Settled(entry.option_id, "carried", i)

    for i, entry in explicit:
        if entry.origin.kind == "choice" and (
            refusal := _admit_explicit(i, entry, fixed, tables, conflicts)
        ):
            return refusal

    active: dict[ScopeAxis, _Settled] = {}
    for axis in order:
        if axis in fixed:
            active[axis] = fixed[axis]
            continue
        settled = _derived_default(tables[axis], fixed=fixed, derived=active, conflicts=conflicts)
        if isinstance(settled, Refusal):
            return settled
        active[axis] = settled

    base_spec_id = identity_spec_id(base)
    identity, spec_id = _compiled_identity(base, base_spec_id, tables, order, active)
    choices = tuple(
        ScopeChoice(
            id=f"{base_spec_id}:{axis}",
            axis=axis,
            rule=tables[axis].rule,
            provenance=active[axis].provenance,
            active=active[axis].option_id,
            options=[
                _offered(option, axis=axis, active=active[axis], fixed=fixed, conflicts=conflicts)
                for option in tables[axis].options
            ],
        )
        for axis in order
    )
    applied = tuple(
        entries[settled.index]
        for axis in order
        if (settled := fixed.get(axis)) is not None and settled.index is not None
    )
    return Compiled(
        entries=applied,
        dropped=tuple(dropped),
        choices=choices,
        identity=identity,
        spec_id=spec_id,
        base_spec_id=base_spec_id,
    )


def _admit_explicit(
    index: int,
    entry: Entry,
    fixed: dict[ScopeAxis, _Settled],
    tables: Mapping[ScopeAxis, AxisTable],
    conflicts: Mapping[frozenset[str], str],
) -> Refusal | None:
    """Settle explicit `entry` into `fixed`, or the refusal of it. A repeat of a settled
    entry is a no-op."""
    if (prior := fixed.get(entry.axis)) is not None:
        if prior.option_id == entry.option_id:
            return None
        return Refusal(
            code="conflicting_requested_options",
            entry=index,
            cause=f"both_on_{entry.axis}",
            blocker=prior.index,
        )
    if (cause := _unavailable(tables, entry)) is not None:
        return Refusal(code="requested_option_unavailable", entry=index, cause=cause)
    if (hit := _first_conflict(entry.option_id, fixed, conflicts, axis=entry.axis)) is not None:
        other, cause = hit
        return Refusal(
            code="conflicting_requested_options", entry=index, cause=cause, blocker=other.index
        )
    fixed[entry.axis] = _Settled(entry.option_id, "explicit", index)
    return None


def _derived_default(
    table: AxisTable,
    *,
    fixed: Mapping[ScopeAxis, _Settled],
    derived: Mapping[ScopeAxis, _Settled],
    conflicts: Mapping[frozenset[str], str],
) -> _Settled | Refusal:
    """The default of `table`'s axis: the first preferred option that is available and
    that no settled option conflicts with. A default derived earlier yields to an entry
    but not to another default, so when only defaults block every option the first one an
    entry allows is taken. When entries block every option, the first blocking entry is
    refused. An axis whose every preferred option is unavailable keeps the option the base
    holds."""
    servable = [
        option_id
        for option_id in table.preference
        if (option := table.option(option_id)) is not None
        and option.availability.status == "available"
    ]
    if not servable:
        return _Settled(table.held, "default", None)
    for option_id in servable:
        if _first_conflict(option_id, derived, conflicts, axis=table.axis) is None and (
            _first_conflict(option_id, fixed, conflicts, axis=table.axis) is None
        ):
            return _Settled(option_id, "default", None)
    for option_id in servable:
        if _first_conflict(option_id, fixed, conflicts, axis=table.axis) is None:
            return _Settled(option_id, "default", None)
    hit = _first_conflict(servable[0], fixed, conflicts, axis=table.axis)
    assert hit is not None
    blocker, cause = hit
    assert blocker.index is not None
    return Refusal(
        code="conflicting_requested_options",
        entry=blocker.index,
        cause=f"blocks_every_{table.axis}_option:{cause}",
    )


def _compiled_identity(
    base: BaseIdentity,
    base_spec_id: str,
    tables: Mapping[ScopeAxis, AxisTable],
    order: Sequence[ScopeAxis],
    active: Mapping[ScopeAxis, _Settled],
) -> tuple[BaseIdentity, str]:
    """`base` edited on every axis whose active option is not the one it holds, minted;
    `base` itself when there is none."""
    identity = base
    edited = False
    for axis in order:
        table = tables[axis]
        if active[axis].option_id == table.held:
            continue
        option = table.option(active[axis].option_id)
        assert option is not None
        identity = apply_binding(identity, option.binding, [o.binding for o in table.options])
        edited = True
    if not edited:
        return base, base_spec_id
    spec_id, minted = mint_identity(identity.model_dump(exclude_none=True))
    return minted, spec_id


def _offered(
    option: Option,
    *,
    axis: ScopeAxis,
    active: _Settled,
    fixed: Mapping[ScopeAxis, _Settled],
    conflicts: Mapping[frozenset[str], str],
) -> Option:
    """`option` as the part offers it: unavailable with `conflicts_with` when an entry on
    another axis blocks it, else as its table states it."""
    if option.availability.status == "unavailable" or option.option_id == active.option_id:
        return option
    hit = _first_conflict(option.option_id, fixed, conflicts, axis=axis)
    if hit is None:
        return option
    blocker, cause = hit
    return option.model_copy(
        update={
            "availability": OptionAvailability(
                status="unavailable",
                code="conflicts_with",
                detail={"entry": blocker.option_id, "cause": cause},
            )
        }
    )
