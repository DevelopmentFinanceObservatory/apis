from __future__ import annotations

from pydantic import Field

from .base import StrictModel
from .grammar import GraphRef, SlotKey, SlotValue


class StatVarSlotValue(StrictModel):
    """A (key, value) pair recording one slot member that a StatVar realises."""

    key: SlotKey = Field(description="the slot identity this StatVar realises")
    value: SlotValue = Field(description="the slot member value this StatVar carries")


class DateRange(StrictModel):
    """The first and last dates of the points an answer serves for a StatVar."""

    start: str | None = Field(default=None, description="first served date, e.g. '2012'")
    end: str | None = Field(default=None, description="last served date, e.g. '2022'")


class StatVar(StrictModel):
    """A resolved StatVar inside the region."""

    ref: GraphRef = Field(description="the resolved StatVar; exists in the graph")
    shape_id: str = Field(description="back-reference to its Shape family")
    slot_values: list[StatVarSlotValue] = Field(
        description=(
            "the bound what/how slot members this StatVar's own arcs assert; "
            "a StatVar realising none of the spec's bound slots carries []"
        )
    )
    constraint_properties: list[str] | None = Field(
        description=(
            "every constraint property dcid this StatVar's own arcs assert, bound or "
            "not, sorted; the family's root lists exactly the spec's value-bound "
            "properties, a StatVar with no constraints carries [], and null means "
            "its constraints are not known"
        )
    )
    data_date_range: DateRange | None = Field(
        default=None,
        description=(
            "the first and last dates of the points the answer's read plan serves for this "
            "StatVar; null when it serves none"
        ),
    )
    data_confirmed_at_destination: bool | None = Field(
        default=None,
        description=(
            "True=the destination entity confirmed the StatVar directly, "
            "False=only the origin entity did, "
            "None=not checked"
        ),
    )


class ResolvedStatVar(StrictModel):
    """One resolved StatVar on the wire, under `resolved.stat_vars.items`: the
    flat projection of the engine's own `StatVar` -- `ref` collapses from a
    `GraphRef` to a bare dcid with `label` as a sibling. `slot_values`,
    `constraint_properties`, `data_date_range`, and `data_confirmed_at_destination`
    carry over unchanged: each has a real reader (rendering falls back to
    `data_date_range` for an unbound time slot's answer-derived text, slot_values,
    derived from this StatVar's own confirmed arcs, is how a multi-StatVar spec's
    per-variable constraint realisation is distinguished, and constraint_properties
    tells the family's root from a member with extra constraints). `shape_id` is
    dropped -- it only ever duplicated `resolved.shape.shape_id`, already on
    the wire at the Spec level.
    """

    ref: str = Field(description="the resolved StatVar's dcid")
    label: str | None = Field(default=None, description="plain-language display name")
    slot_values: list[StatVarSlotValue] = Field(
        description=(
            "the bound what/how slot members this StatVar's own arcs assert; "
            "a StatVar realising none of the spec's bound slots carries []"
        )
    )
    constraint_properties: list[str] | None = Field(
        description=(
            "every constraint property dcid this StatVar's own arcs assert, bound or "
            "not, sorted; the family's root lists exactly the spec's value-bound "
            "properties, a StatVar with no constraints carries [], and null means "
            "its constraints are not known"
        )
    )
    data_date_range: DateRange | None = Field(
        default=None,
        description=(
            "the first and last dates of the points the answer's read plan serves for this "
            "StatVar; null when it serves none"
        ),
    )
    data_confirmed_at_destination: bool | None = Field(
        default=None,
        description=(
            "True=the destination entity confirmed the StatVar directly, "
            "False=only the origin entity did, "
            "None=not checked"
        ),
    )


class ResolvedStatVars(StrictModel):
    """`resolved.stat_vars` on the wire: the identity's confirmed population, the
    StatVars confirmed to carry observations at this resolve's probe entities,
    plus the size of that population and whether items is a subset of it. An
    item the answer's read plan serves no point of carries a null
    `data_date_range`.

    member_count is the population confirmed to carry observations at this
    resolve's own probe entities, not resolved.shape.member_count's
    bindings-admitted population -- the two differ whenever a member the
    shape's bindings admit turns out to carry no observation at the resolved
    place. On a standard (non-role-bearing) shape this population can itself
    be a floor: the packaged code index does not source members for every
    shape family, so a shape outside it falls back to a recall sample capped
    at SEARCH_MAX_CONFIRM_CANDIDATES (40), and member_count cannot exceed
    what that sample admitted. A where-axis merge (same_shape.py, "population
    of Kenya and Uganda") publishes max() of its members' own member_count
    rather than the sum, since the same StatVar is typically confirmed at
    every member's own place and summing would double-count the
    intersection -- max() is a true lower bound of the union. member_count_exact
    records which of these applies. truncated is true exactly when items no
    longer carries every member member_count counts.
    """

    items: list[ResolvedStatVar] = Field(
        description=(
            "the identity's confirmed population: the StatVars confirmed to carry "
            "observations at this resolve's probe entities, >= 1 when grounded; an "
            "item the answer's read plan serves no point of carries a null "
            "data_date_range"
        )
    )
    member_count: int = Field(
        description=(
            "how many members are confirmed to carry observations at this "
            "resolve's own probe entities; see this model's own docstring for "
            "how this differs from resolved.shape.member_count"
        )
    )
    member_count_exact: bool = Field(
        default=True,
        description=(
            "True when member_count is a true total; False when it is a floor "
            "-- the population being at least member_count large, not exactly "
            "member_count -- because it was measured over a capped recall "
            "sample, or because it is a where-axis merge's max() of its "
            "members' own counts rather than a computed union. Defaults to "
            "True, correct for a caller that has not measured otherwise"
        ),
    )
    truncated: bool = Field(description="true when items is a subset of member_count")
