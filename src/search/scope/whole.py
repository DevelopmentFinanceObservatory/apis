"""A part's whole: the same figure switched on one axis, every other axis held.

A whole is a switch that keeps the part. `derive_whole` takes the scope a part's base and
entries compile to and compiles its whole with `build_part_scope` once more, from the same
base: one explicit entry per choice the part settled, holding the option the part holds,
and the whole's option as an ordinary explicit switch on its axis. Compile applies or
refuses every one of them, so a whole that cannot hold the part's option on another axis
refuses there, with compile's own cause. A pin whose option sets the role the whole's
option sets is left out, since the whole replaces those parties: a recipient whole drops the
region axis, a donor whole keeps it.

A provider table may declare that its whole rebinds the recipient side too
(`AxisTable.whole_rebind`, from `ProviderWhole.recipient`): DAC2a's total ODA over the whole
population reads total ODA to all recipients. That whole adds a second explicit switch, on
the recipient side axis, and leaves out the pins that set the recipients. Compile judges
total ODA on the identity the recipient edit produces, at all recipients, where France's
bilateral composite at the developing countries refuses total ODA alone.

Two rules a switch lacks: the whole compiles to another figure than the part
(`part_is_whole`), and its option serves every year the part asks for, so the whole reads
the part's window. The axis's table (`AxisTable.whole`, `AxisTable.whole_alternatives`),
read off the part's compiled identity, names the option a request with none asks for and
the options a client may ask for in its place; the alternatives a result lists are those
whose own derive succeeds.

The pins serve the whole's compile alone. The scope returned states the whole as a client
re-posts it: its base the identity it serves, its entries the part's own with the switch
on the whole's axis (none for a side axis, which `derived` states, as it states a rebind),
so the base and entries posted without `whole` compile the same Spec, and a choice the part
took by default keeps that provenance. A whole whose answer holds another option than the
part on an axis it does not set refuses `whole_moves_<axis>`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dfo_platform.problem import Alternative
from dfo_platform.roles import role_key_axis
from search.engine.errors import EngineInputError
from search.models import (
    ChoiceOrigin,
    CompiledScope,
    DerivedSide,
    DerivedWhole,
    Entry,
    PartScope,
    RoleAxis,
    ScopeAxis,
    SideBinding,
    WholeRequest,
)
from search.scope import rules as scope_rules
from search.scope.build import build_part_scope
from search.scope.compile import Refusal, RefusalCode
from search.scope.rules import AxisRule
from search.scope.side import side_rule, summed_members
from search.scope.state import AxisRecord
from search.scope.tables import AxisTable, Conflict, ScopeFacts
from search.scope.vocabulary import option_axis

SIDE_RULES: Mapping[ScopeAxis, AxisRule] = {
    "donor": side_rule("donor"),
    "recipient": side_rule("recipient"),
}
"""The side axes, read only for a whole request on their axis (`scope.side`)."""


@dataclass(frozen=True)
class WholeRefusal:
    """A whole request refused: the registry code, the cause, and the options a client may
    ask for as the whole in its place, each with its label."""

    code: RefusalCode
    cause: str
    alternatives: tuple[tuple[str, str], ...] = ()


def derive_whole(
    part: PartScope,
    request: WholeRequest,
    *,
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
    rules: Mapping[ScopeAxis, AxisRule] | None = None,
    conflicts: tuple[Conflict, ...] | None = None,
) -> PartScope | WholeRefusal:
    """The scope of `part`'s whole on `request.axis`, stating the pair in
    `PartScope.derived`, or the refusal of it with the alternatives that derive. Each
    candidate whole (the declared whole and its alternatives) is derived in full to learn
    whether it is offered, so a request runs up to three compiles per candidate (the whole's own, and two that
    restate it as a client re-posts it) beside its own."""
    posted = scope_rules.AXIS_RULES if rules is None else rules
    rules = {**posted}
    if request.axis in SIDE_RULES:
        rules[request.axis] = SIDE_RULES[request.axis]
    derive = _Derive(
        part=part, records=records, facts=facts, rules=rules, posted=posted, conflicts=conflicts
    )
    table = rules[request.axis](part.compiled.identity, records.get(request.axis), facts)
    if table is None:
        return WholeRefusal(code="requested_option_unavailable", cause="axis_absent")
    result = derive.whole(table, request.option_id)
    candidates = [o for o in (table.whole, *table.whole_alternatives) if o is not None]
    if isinstance(result, WholeRefusal):
        offered = [o for o in candidates if not isinstance(derive.whole(table, o), WholeRefusal)]
        return WholeRefusal(
            code=result.code, cause=result.cause, alternatives=_labelled(table, offered)
        )
    whole, asked = result
    offered = [
        o for o in candidates if o != asked and not isinstance(derive.whole(table, o), WholeRefusal)
    ]
    return whole.model_copy(
        update={"derived": derive.derived(table, whole, asked, alternatives=offered)}
    )


@dataclass(frozen=True)
class _Derive:
    part: PartScope
    records: Mapping[ScopeAxis, AxisRecord]
    facts: ScopeFacts
    rules: Mapping[ScopeAxis, AxisRule]
    posted: Mapping[ScopeAxis, AxisRule]
    conflicts: tuple[Conflict, ...] | None

    def held(self, table: AxisTable) -> str:
        """The option the part holds on `table`'s axis."""
        return next((c.active for c in self.part.choices if c.axis == table.axis), table.held)

    def whole(
        self, table: AxisTable, option_id: str | None
    ) -> tuple[PartScope, str] | WholeRefusal:
        """The whole on `option_id`, the axis's declared whole where None, and the option
        it holds, or the refusal of it."""
        asked = table.whole if option_id is None else option_id
        if asked is None:
            return WholeRefusal(code="requested_option_unavailable", cause=table.whole_cause)
        if asked not in (table.whole, *table.whole_alternatives):
            return WholeRefusal(code="requested_option_unavailable", cause="not_a_whole")
        rebind = _rebind(table, asked)
        if asked == self.held(table) and rebind is None:
            return WholeRefusal(code="requested_option_unavailable", cause="part_is_whole")
        option = table.option(asked)
        assert option is not None
        rules = dict(self.rules)
        switches = [_switch(self.part, table.axis, asked)]
        roles = {_role_axis(option.binding)}
        if rebind is not None:
            side = option_axis(rebind)
            rules[side] = SIDE_RULES[side]
            switches.append(_switch(self.part, side, rebind))
            roles |= _side_roles(self.part, side, rebind, records=self.records, facts=self.facts)
        pins, dropped = self._pins(table.axis, roles - {None})
        compiled = build_part_scope(
            base=self.part.base.identity,
            entries=[*pins, *switches],
            records=self.records,
            facts=self.facts,
            rules=rules,
            conflicts=self.conflicts,
        )
        if isinstance(compiled, Refusal):
            return WholeRefusal(code=compiled.code, cause=compiled.cause)
        if compiled.compiled.spec_id == self.part.compiled.spec_id:
            return WholeRefusal(code="requested_option_unavailable", cause="part_is_whole")
        served = next(
            o
            for c in compiled.choices
            if c.axis == table.axis
            for o in c.options
            if o.option_id == asked
        )
        if served.availability.status != "available":
            return WholeRefusal(
                code="requested_option_unavailable",
                cause=served.availability.code or f"option_{served.availability.status}",
            )
        # Compile reads every axis off the part's base, so the whole states a choice on each
        # axis the part settles; one it does not hold at the part's option moves.
        moved = next(
            (
                choice.axis
                for choice in self.part.choices
                if choice.axis != table.axis
                and choice.axis not in dropped
                and choice.active
                != next((c.active for c in compiled.choices if c.axis == choice.axis), None)
            ),
            None,
        )
        if moved is not None:
            return WholeRefusal(code="conflicting_requested_options", cause=f"whole_moves_{moved}")
        whole = self._posted(table.axis, asked, compiled, dropped)
        if isinstance(whole, WholeRefusal):
            return whole
        return whole, asked

    def _posted(
        self, axis: ScopeAxis, asked: str, compiled: PartScope, dropped: set[ScopeAxis]
    ) -> PartScope | WholeRefusal:
        """The whole as a client re-posts it: its base the identity it serves, its entries
        the part's own, less those on `axis` and on the `dropped` axes whose parties the
        whole replaces, with the switch to `asked` where `axis` is not a side axis, compiled
        over the axes a fresh answer reads. An entry on an axis the served identity offers
        no table for is left out, since the identity holds its option (a donor side summed
        over its members reads no provider axis on DAC1). Refuses `whole_not_restated`
        where that compile serves another figure than the whole's own."""
        entries = [e for e in self.part.entries if e.axis != axis and e.axis not in dropped]
        if axis not in SIDE_RULES:
            entries.append(_switch(self.part, axis, asked))

        def build(entries: list[Entry], *, refuse: bool) -> PartScope | Refusal:
            return build_part_scope(
                base=compiled.compiled.identity,
                entries=entries,
                records=self.records,
                facts=self.facts,
                rules=self.posted,
                conflicts=self.conflicts,
                refuse=refuse,
            )

        trial = build(entries, refuse=False)
        assert isinstance(trial, PartScope)
        unread = [d.entry for d in trial.dropped_entries if d.cause == "axis_absent"]
        whole = build([e for e in entries if e not in unread], refuse=True)
        if isinstance(whole, Refusal) or whole.compiled.spec_id != compiled.compiled.spec_id:
            return WholeRefusal(code="requested_option_unavailable", cause="whole_not_restated")
        return whole.model_copy(update={"dropped_entries": list(self.part.dropped_entries)})

    def _pins(
        self, axis: ScopeAxis, roles: set[RoleAxis | None]
    ) -> tuple[list[Entry], set[ScopeAxis]]:
        """An explicit entry per choice the part settled on an axis other than `axis`,
        holding its option, less one whose option sets the parties of one of the flow sides
        `roles` the whole sets, and the axes so left out. A choice the part holds on an
        unavailable option, kept because no option serves, is left to its default, which
        the whole's check that it moves no axis judges."""
        concepts = {entry.axis: entry.concept for entry in self.part.entries}
        pins: list[Entry] = []
        dropped: set[ScopeAxis] = set()
        for choice in self.part.choices:
            if choice.axis == axis:
                continue
            held = next(o for o in choice.options if o.option_id == choice.active)
            if _role_axis(held.binding) in roles:
                dropped.add(choice.axis)
                continue
            if held.availability.status == "unavailable":
                continue
            pins.append(
                Entry(
                    axis=choice.axis,
                    option_id=choice.active,
                    provenance="explicit",
                    concept=concepts.get(choice.axis),
                    origin=ChoiceOrigin(choice_id=choice.id),
                )
            )
        return pins, dropped

    def derived(
        self, table: AxisTable, whole: PartScope, asked: str, *, alternatives: Sequence[str]
    ) -> DerivedWhole:
        """What the part and its whole count, and the years both serve."""
        option = table.option(asked)
        assert option is not None
        binding = option.binding
        kind = (
            "option"
            if not isinstance(binding, SideBinding)
            else "published_total"
            if binding.refs
            else "sum_of_members"
        )
        return DerivedWhole(
            axis=table.axis,
            kind=kind,
            part=_side(self.part, table.axis, self.held(table)),
            whole=_side(whole, table.axis, asked),
            years=_shared_years(self.part.compiled, whole.compiled),
            members=summed_members(whole.compiled.identity, binding.role)
            if isinstance(binding, SideBinding) and not binding.refs
            else None,
            alternatives=list(alternatives),
            rebind=_rebind(table, asked),
        )


def _rebind(table: AxisTable, asked: str) -> str | None:
    """The side option the whole on `asked` takes beside it: the table's `whole_rebind`
    where `asked` is its declared whole, else None."""
    return table.whole_rebind if asked == table.whole else None


def _role_axis(binding: object) -> RoleAxis | None:
    """The flow side whose parties `binding` sets (`role_key_axis` of its role), None where
    it sets none."""
    role = getattr(binding, "role", None)
    return role_key_axis(role) if isinstance(role, str) else None


def _side_roles(
    part: PartScope,
    axis: ScopeAxis,
    option_id: str,
    *,
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
) -> set[RoleAxis | None]:
    """The flow side the side option `option_id` binds, read off the part's side table;
    none where the table offers no such option, which compile then refuses."""
    table = SIDE_RULES[axis](part.compiled.identity, records.get(axis), facts)
    option = table.option(option_id) if table is not None else None
    return {_role_axis(option.binding)} if option is not None else set()


def _switch(part: PartScope, axis: ScopeAxis, option_id: str) -> Entry:
    return Entry(
        axis=axis,
        option_id=option_id,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=f"{part.id}:{axis}"),
    )


def _side(scope: PartScope, axis: ScopeAxis, option_id: str) -> DerivedSide:
    actives = {choice.axis: choice.active for choice in scope.choices}
    return DerivedSide(
        spec_id=scope.compiled.spec_id,
        option_id=option_id,
        provider=actives.get("provider"),
        populations={key: list(dcids) for key, dcids in scope.compiled.identity.entities.items()},
    )


def _shared_years(part: CompiledScope, whole: CompiledScope) -> list[tuple[int, int]] | None:
    """The runs of years both figures serve, each over the union of its components'
    periods; None unless both state their periods."""
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
    if len(served) < 2:
        return None
    runs: list[tuple[int, int]] = []
    for year in sorted(set.intersection(*served)):
        if runs and runs[-1][1] == year - 1:
            runs[-1] = (runs[-1][0], year)
        else:
            runs.append((year, year))
    return runs


def _labelled(table: AxisTable, options: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (option_id, option.label)
        for option_id in options
        if (option := table.option(option_id)) is not None
    )


def whole_error(refusal: WholeRefusal, *, pointer: str) -> EngineInputError:
    """`refusal` as a 422 naming its cause, at `pointer`, with its alternatives."""
    alternatives = [Alternative(ref=ref, label=label) for ref, label in refusal.alternatives]
    return EngineInputError(
        f"The whole of this part cannot be derived: {refusal.cause}.",
        code=refusal.code,
        status=422,
        pointer=pointer,
        alternatives=alternatives or None,
    )
