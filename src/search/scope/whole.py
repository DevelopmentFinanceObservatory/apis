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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dfo_platform.problem import Alternative
from search.engine.errors import EngineInputError
from search.models import (
    ChoiceOrigin,
    CompiledScope,
    DerivedSide,
    DerivedWhole,
    Entry,
    PartScope,
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
    `PartScope.derived`, or the refusal of it with the alternatives that derive."""
    rules = {**(scope_rules.AXIS_RULES if rules is None else rules)}
    if request.axis in SIDE_RULES:
        rules[request.axis] = SIDE_RULES[request.axis]
    derive = _Derive(part=part, records=records, facts=facts, rules=rules, conflicts=conflicts)
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
        roles = {getattr(option.binding, "role", None)}
        if rebind is not None:
            side = option_axis(rebind)
            rules[side] = SIDE_RULES[side]
            switches.append(_switch(self.part, side, rebind))
            roles |= _side_roles(self.part, side, rebind, records=self.records, facts=self.facts)
        compiled = build_part_scope(
            base=self.part.base.identity,
            entries=[*self._pins(table.axis, roles - {None}), *switches],
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
        return _restated(compiled, self.part), asked

    def _pins(self, axis: ScopeAxis, roles: set[str | None]) -> list[Entry]:
        """An explicit entry per choice the part settled on an axis other than `axis`,
        holding its option, less one whose option sets one of the `roles` the whole sets. A
        choice the part holds on an unavailable option, kept because no option serves, is
        left to its default."""
        concepts = {entry.axis: entry.concept for entry in self.part.entries}
        pins: list[Entry] = []
        for choice in self.part.choices:
            if choice.axis == axis:
                continue
            held = next(o for o in choice.options if o.option_id == choice.active)
            if getattr(held.binding, "role", None) in roles:
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
        return pins

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


def _side_roles(
    part: PartScope,
    axis: ScopeAxis,
    option_id: str,
    *,
    records: Mapping[ScopeAxis, AxisRecord],
    facts: ScopeFacts,
) -> set[str | None]:
    """The role the side option `option_id` binds, read off the part's side table; none
    where the table offers no such option, which compile then refuses."""
    table = SIDE_RULES[axis](part.compiled.identity, records.get(axis), facts)
    option = table.option(option_id) if table is not None else None
    return {getattr(option.binding, "role", None)} if option is not None else set()


def _switch(part: PartScope, axis: ScopeAxis, option_id: str) -> Entry:
    return Entry(
        axis=axis,
        option_id=option_id,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=f"{part.id}:{axis}"),
    )


def _restated(whole: PartScope, part: PartScope) -> PartScope:
    """`whole` with each choice that holds the part's option stating the part's provenance,
    since the whole holds it as the part settled it."""
    settled = {choice.axis: choice for choice in part.choices}
    choices = [
        choice.model_copy(update={"provenance": held.provenance})
        if (held := settled.get(choice.axis)) is not None and held.active == choice.active
        else choice
        for choice in whole.choices
    ]
    return whole.model_copy(update={"choices": choices})


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
