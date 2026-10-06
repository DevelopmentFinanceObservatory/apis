from __future__ import annotations

from collections.abc import Sequence

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


def served_date_range(members: Sequence[StatVar]) -> DateRange | None:
    """The earliest start and latest end over `members`' served date ranges, or `None` when
    no member states either bound. ISO dates of one precision order as strings, so the
    bounds compare as text."""
    ranges = [sv.data_date_range for sv in members if sv.data_date_range is not None]
    start = min(filter(None, (r.start for r in ranges)), default=None)
    end = max(filter(None, (r.end for r in ranges)), default=None)
    if start is None and end is None:
        return None
    return DateRange(start=start, end=end)


class SentenceFacts(StrictModel):
    """What the sentence renderer reads of a Spec's served members
    (`ResolvedSpec.sentence_facts`). The engine derives it from the members
    (`SentenceFacts.of`); the sentence golden stores it beside each Spec, since the
    wire carries no member list for a replay to read."""

    members: list[GraphRef] = Field(
        description=(
            "each served member's ref and label, in served order: one member names the "
            "measure, and a candidate's member names tell alike readings apart"
        )
    )
    constraint_properties: list[list[str]] = Field(
        description=(
            "the distinct constraint property lists the members state, sorted; a member "
            "whose constraints are not known states none. The flow family reads them"
        )
    )

    @classmethod
    def of(cls, members: Sequence[StatVar]) -> SentenceFacts:
        """The facts `members` give the renderer."""
        return cls(
            members=[sv.ref for sv in members],
            constraint_properties=[
                list(properties)
                for properties in sorted(
                    {
                        tuple(sv.constraint_properties)
                        for sv in members
                        if sv.constraint_properties is not None
                    }
                )
            ],
        )


class ResolvedStatVars(StrictModel):
    """`resolved.stat_vars` on the wire: how many StatVars the answer holds, how many a
    Datasets mint of the Spec's definition serves, and the dates the answer serves over
    them. The members themselves stay off the wire; `POST /api/datasets/{id}/members`
    lists the ones a mint serves.

    member_count is the population confirmed to carry observations at this
    resolve's own probe entities, not resolved.shape.member_count's
    bindings-admitted population. The two differ for two reasons. A member the
    shape's bindings admit can carry no observation at the resolved place. And on
    a role-bearing shape the answer reads one role class of the admitted members:
    where a total stands for its breakdowns the breakdowns are folded into it, so a
    donor-only answer can state 255 admitted members beside 1 confirmed while every
    one of the 255 holds data. On a standard (non-role-bearing) shape this population can itself
    be a floor: the packaged code index does not source members for every
    shape family, so a shape outside it falls back to a recall sample capped
    at SEARCH_MAX_CONFIRM_CANDIDATES (40), and member_count cannot exceed
    what that sample admitted. A where-axis merge (same_shape.py, "population
    of Kenya and Uganda") publishes max() of its members' own member_count
    rather than the sum, since the same StatVar is typically confirmed at
    every member's own place and summing would double-count the
    intersection -- max() is a true lower bound of the union. member_count_exact
    records which of these applies.

    served_count is the mint's own count, a function of the definition alone: the
    StatVars a Datasets mint of `definition` serves, the length of its
    `trace.resolved_stat_vars`. It is stated whenever Search settled the membership the
    mint resolves: a family the code index decides, or a family the mint's member walk
    settled for this definition. Null when Search answers from members it could not
    settle, which the part flags with a `membership_unverified` warning.
    """

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
    served_count: int | None = Field(
        default=None,
        description=(
            "how many StatVars a Datasets mint of this Spec's definition serves, the "
            "length of its trace.resolved_stat_vars; 1 for a single series. Null when "
            "the membership is unsettled: Search answered from members the mint may not "
            "serve, and the part carries membership_unverified"
        ),
    )
    data_date_range: DateRange | None = Field(
        default=None,
        description=(
            "the earliest first date and latest last date of the points the answer's read "
            "plan serves over its StatVars; null when it serves none"
        ),
    )
