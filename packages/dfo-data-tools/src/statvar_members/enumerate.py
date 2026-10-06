"""Enumerate StatVar members, confirm graph reads, and reduce to the minimal antichain.

Server-side filters cover concrete five-tuple values and `value` slots. Other
bindings need client checks. Re-probes reconcile dropped batch arcs before a
rejection or antichain winner is trusted. `typeOf` is checked on trusted winners.

`elect_role_class` is the reduction every member resolver shares: role coverage against
the bound role keys, then the antichain over constraint properties and over declared roles.
The graph walk here and `code_index.members` both call it.
"""

from __future__ import annotations

import time
import unicodedata
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from itertools import combinations
from types import MappingProxyType
from typing import Literal, cast

from dfo_platform.roles import declared_role_for
from statvar_members.types import (
    EMPTY_READINGS,
    UNRESOLVED_NODE,
    AbsentReading,
    FiveTuple,
    MemberArcReader,
    SubjectPage,
)

type SlotBinding = (
    tuple[Literal["value"], str]
    | tuple[Literal["set"], tuple[str, ...]]
    | tuple[Literal["unbound"]]
)
"""A slot binding keyed by constraint property, with refs as bare dcids."""


def slot_bindings_from_identity(
    slots: Sequence[Mapping[str, object]],
    *,
    widen: Callable[[str], Sequence[str]],
) -> dict[str, SlotBinding]:
    """Convert identity slots into `SlotBinding`s, widening hierarchical refs through
    the caller-supplied `widen` function. `value` and `set` refs are widened.
    Multiple descendants promote a `value` binding to `set`, which forfeits the server-side
    filter and the completeness proof. `exact` denotes its ref alone, so it maps to `value`
    without widening; unknown states map to `unbound`. Set refs are
    widened and deduplicated in first-seen order, and a set that widens to one ref maps to
    `value`, the binding the `value` spelling of the same ref maps to: the spec kit hashes
    the two spellings to one `spec_id`. The caller supplies the code
    index operation, keeping this package independent of `code_index`."""
    result: dict[str, SlotBinding] = {}
    for slot in slots:
        prop = cast(str, slot["property"])
        binding = cast("Mapping[str, object]", slot.get("binding", {}))
        state = binding.get("state")
        if state == "value":
            ref = cast(str, binding["ref"])
            widened = tuple(widen(ref))
            result[prop] = ("set", widened) if len(widened) > 1 else ("value", ref)
        elif state == "exact":
            result[prop] = ("value", cast(str, binding["ref"]))
        elif state == "set":
            refs = cast("Sequence[str]", binding["refs"])
            widened_refs = tuple(
                dict.fromkeys(descendant for ref in refs for descendant in widen(ref))
            )
            result[prop] = (
                ("set", widened_refs) if len(widened_refs) > 1 else ("value", widened_refs[0])
            )
        else:
            result[prop] = ("unbound",)
    return result


def five_tuple_from_identity(shape_member: Mapping[str, object]) -> FiveTuple:
    """Translate a described identity shape to `FiveTuple`. Absent members become `None`.

    An unresolved denominator sets `denominator_unresolved` without a ref.
    """
    denominator = cast("Mapping[str, object] | None", shape_member.get("measurement_denominator"))
    measurement_denominator: str | None = None
    denominator_unresolved = False
    if denominator is not None:
        if "ref" in denominator:
            measurement_denominator = cast(str, denominator["ref"])
        else:
            denominator_unresolved = True
    return FiveTuple(
        population_type=cast("str | None", shape_member.get("population_type")),
        measured_property=cast("str | None", shape_member.get("measured_property")),
        stat_type=cast("str | None", shape_member.get("stat_type")),
        measurement_qualifier=cast("str | None", shape_member.get("measurement_qualifier")),
        measurement_denominator=measurement_denominator,
        denominator_unresolved=denominator_unresolved,
    )


CANDIDATE_CAP = 4_000
"""Raw reverse-arc candidate ceiling, checked before forward reads. `node_subjects`
pages beyond the graph's 500-row page cap. This bound caps the candidate population.
Dev measurements (14 September 2026, three repeats): crs-grant-equivalent-kenya
had 3,013 candidates, 2 members, and 10.5 to 11.8 s runtime.
The `oda-grants-to-kenya` sample had 3,812 candidates, 466 members, and
23.2 to 28.5 s runtime. Their rates were
0.0035 to 0.0039 and 0.0062 to 0.0075 s/candidate. The latter projects 25 to 30 s at
4,000 candidates, within Search's 45 s and datasets' 120 s budgets. Measure a
wider identity before raising the cap."""

DEFAULT_MEMBER_CAP = 700
"""Default cap on the antichain-reduced result. `MemberEnumerator.member_cap` defaults to it,
so the Datasets doors and Search's walk enumerate under one cap."""

DEFAULT_SUBJECT_CEILING = 20_000
"""Subjects a full read takes when the first page truncates without proof. It covers the
measured deep-root sets (13,820 and 13,566 for Person/count by gender) and every dfo
five-tuple (largest 13,614). Search's resubmit and the Datasets mint both default to it.
`subject_ceiling` is part of the `MemberEnumerator` cache key, so one value lets them
share cached groups."""

_REPROBE_BUDGET = 32
"""Each reconciliation pass probes at most 32 suspects in one batch, then retries
uncertain results individually. Unprobed excess is reported as inconsistent."""

_EXTRA_CANDIDATE_PROBE_BUDGET = 32
"""`_confirm_group` applies this 32-probe budget only to caller-supplied
`extra_candidates`, which may be nonexistent and cost one `/v2/node` POST each.
Unprobed extras map to `None` and fail ordinary membership checks. Graph-proven
walked candidates remain unbudgeted. Unresolved reads mark the group inconsistent
and suppress `proven_complete`."""

_TWIN_PROBE_BUDGET = 8
"""Under a confirmed root, `_read_root_twins` reads at most 8 possible root twins one dcid
at a time: 8 arc reads beyond the reconciliation passes. Under an unresolved-denominator ask,
a twin naming a denominator target `_ProvisionalDenominatorReader` has not seen adds one
batched `typeOf` read, and each target that reads empty in a batch of several adds one
single-target re-read. The pass therefore costs at most 8 arc reads, 8 `typeOf` reads, and
one re-read per target that reads empty in a batch. Live families share one target
(`PerCapita`), which the root's read has seen, so they add none. Twins past the budget stay
in `undismissed_candidates`. Root-shaped suspects take the reconciliation
budgets first, so few reach this pass: dev walks of GDP per capita in Guatemala,
internet users in India, CO2 emissions in China, households in Kenya, and the
4,000-candidate `Count_Person` family left none."""

_ROLE_PROBE_BUDGET = 16
"""`_settle_role_absence` reads at most 16 members one dcid at a time: those whose place in
the role-class election rests on an `observationProperties` arc that two batch reads left
empty. A single-dcid read of the dev graph took a median 0.19 s and a p90 0.22 s (twenty
reads, 2 October 2026), so a spent budget adds about 3 s beside the 25 to 30 s a
4,000-candidate walk takes (`CANDIDATE_CAP`), inside Search's 45 s. A member that read alone
earlier in the walk costs nothing. Members past the budget stay in `undismissed_candidates`."""

_FIVE_TUPLE_PROPERTIES = (
    "populationType",
    "measuredProperty",
    "statType",
    "measurementQualifier",
    "measurementDenominator",
    "constraintProperties",
    "observationProperties",
    "typeOf",
)
"""Properties read to confirm candidates and reverify antichain winners. `typeOf`
identifies non-StatVar candidates before reduction."""

_EMPTY_ARCS: Mapping[str, dict[str, tuple[str, ...]] | None] = MappingProxyType({})

_STATISTICAL_VARIABLE = "StatisticalVariable"
_OBSERVATION_ABOUT = "observationAbout"
_MEASUREMENT_DENOMINATOR = "measurementDenominator"
_PROVISIONAL_NODE_TYPES = ("ProvisionalNode",)
"""The `typeOf` of a denominator target that Search reads as unresolved even though it
carries a dcid (`search.engine.retrieval.facts._is_provisional_target`). `PerCapita` is one."""


@dataclass(frozen=True)
class Member:
    stat_var: str
    constraints: Mapping[str, str]
    declared_roles: tuple[str, ...]


@dataclass(frozen=True)
class MemberGroup:
    five_tuple: FiveTuple
    members: tuple[Member, ...]
    family: tuple[str, ...]
    """Dcids of every confirmed candidate of the elected role class before antichain
    reduction, walked or caller-supplied, in confirmation order (`RoleClassElection.family`).
    `members` is its antichain-minimal subset, so a member a broader one dominates appears
    here alone. Dominated entries keep their batch-read confirmation; only `members` gets
    `_reverify_antichain`'s trusted re-read. A dominated member of the elected class whose
    `observationProperties` arc two batch reads dropped reads subject-only and is missing
    here, since `_settle_role_absence` reads alone only the members whose roles can change
    `members`."""
    candidates_considered: int
    candidate_cap: int
    """The `max_results` `_reverse_query` passed to `node_subjects` -- the
    window `candidates_considered` was read inside, and the bound a
    `truncated=True` group's own page crossed. A caller reporting the
    refusal a truncated page causes states this as the ceiling it crossed,
    alongside `candidates_considered` as the count: the candidate window is
    the bound this read measured, unlike the member count past it, which no
    bounded read settles. `CANDIDATE_CAP` for the first page, and the caller's
    `subject_ceiling` for a group the full read built (`full_read`). Meaningful
    only when `truncated` is `True`: a page that read every candidate was still
    read inside this window, which no candidate ever reached."""
    truncated: bool
    """`True` when `_reverse_query`'s page hit `candidate_cap` before
    exhausting the candidate population. A full read past `subject_ceiling`
    confirms nothing, so its group holds no members. `members` is confirmed over
    whatever page was read: confirmation and antichain domination depend
    only on that page, so a non-empty `members` here is as confirmed as a
    complete page's. Truncation withholds completeness, since a member
    that would dominate everything found so far could sit past the window.
    An empty `members` alongside `truncated=True` means no candidate
    inside the window dominated the rest; an empty `members` alongside
    `truncated=False` means a complete read found nothing.

    `truncated` records that the read hit its window. `proven_complete`
    records the separate fact of whether `_antichain_proven_complete`'s
    domination argument establishes that no candidate past the window
    dominates away anything `members` already holds. That argument still
    leaves a same-keyset twin, a literal duplicate declaration, unreported
    past the window, so `proven_complete=True` does not prove `members` is
    the identity's entire population. Read `proven_complete`, not
    `truncated` alone, to know whether `members` is free of a missed
    dominated member."""
    proven_complete: bool
    """`True` when no candidate past the page window could dominate away a member already held.

    Always `True` when `truncated` is `False`. With `truncated` true, it is `True` only when
    `_antichain_proven_complete`'s domination argument fires and the root declares only the
    roles that place the role keys (`_root_elects_role_class`), so no member past the window
    can declare fewer roles and replace it, and every open role key is among the keys the
    election read (`elected_open_role_keys`), since a twin past the window that places
    an open key no member on the page places would be elected over the root. It never proves the full
    population: a literal duplicate declaration with the same constraint-key set can sit past
    the window unreported, and the graph does carry duplicate StatVar declarations. The result
    is then narrower, never wrong. `_minimal_antichain` and `_minimal_constraint_svs` share this
    limit because both compare constraint-key sets alone."""
    root_confirmed: bool
    """`True` when `_antichain_proven_complete`'s domination argument holds: `members`
    includes a confirmed member whose constraint keys are exactly the `value`-bound slot
    keys and whose roles cover the elected role keys, a confirmed member of those
    constraint keys covers the bound role keys (`_root_covers_bound_keys`), no slot is
    `set`-bound, and every unbound property reads `aggregate`. Any
    candidate this group did not read, past a truncated window or in
    `inconsistent_candidates`, is then a non-member, a member that root dominates, or a
    same-keyset duplicate of it, so none of them can change `members` beyond omitting
    that duplicate. A duplicate is a survivor with unverified roles, so domination
    settles membership but not role agreement: `_read_root_twins` reads the unresolved
    candidates that may be duplicates, and `undismissed_candidates` names those it could
    not settle. A truncated group's `proven_complete` also needs the root's roles to be
    the fewest that place the role keys."""
    inconsistent_candidates: tuple[str, ...] = ()
    """Dcids excluded from `members` because the graph's forward read could
    not be trusted, rather than because they fail to match `slots`. Four
    sources feed this: a candidate the reverse-arc query proved exists and
    carries a filter-covered arc (`measurementDenominator` included, for a
    concrete-dcid ask), whose forward read -- empty outright, or merely
    contradicting that arc -- still would not settle after
    `_reconcile_filter_covered`'s re-probe; a candidate `_confirm_member`
    rejected for an absence-caused reason (`measurementDenominator` for a
    `denominator_unresolved` ask, a bound constraint's own value arc, or a
    bound constraint's name missing from `constraintProperties`) that
    `_reconcile_absence_rejected` never got the budget to re-probe at all;
    and an antichain winner `_reverify_antichain` could not re-verify,
    either because its dcid vanished on re-read or because its suspicious
    batch re-read exhausted `_REPROBE_BUDGET`'s single-dcid fallbacks; and a
    member whose role class rests on an empty `observationProperties` arc that
    `_settle_role_absence` could not read alone within `_ROLE_PROBE_BUDGET`.
    Excluded rather than raising: all four are a genuine inconsistency in
    the graph's own read path, distinct from a candidate that simply does
    not match `slots`. Under `root_confirmed`, a possible root twin that
    `_read_root_twins` settles leaves this list."""
    undismissed_candidates: tuple[str, ...] = ()
    """The `inconsistent_candidates` that could change `members` or their role agreement.
    Without `root_confirmed`, that is all of them, and role agreement is checked only when
    there are none. With it, `_possible_root_twins` keeps those whose reads name no
    constraint outside the `value`-bound slot keys, since each may be a same-keyset
    duplicate of the root whose roles disagree, and `_read_root_twins` drops each one its
    single-dcid read settles. Role agreement among `members` is then checked regardless,
    so a disagreement among confirmed members raises before any caller reads this."""
    full_read: bool = False
    """`True` when the walk read the whole filtered reverse set under a `subject_ceiling`,
    because the first page truncated without proof."""
    elected_open_role_keys: frozenset[str] = frozenset()
    """The open role keys the final election read beside the bound ones
    (`RoleClassElection.role_keys` over every confirmed member): the asked open keys the
    members serve. A walk of the same candidates that asks exactly these keys, open or
    bound, elects under the same keys, so it elects the same members."""

    @property
    def stated_family(self) -> frozenset[str] | None:
        """The family a caller states for this walk, by count and `member_digest`: `family`
        when the walk read every candidate, else `None`. A truncated page proven complete
        leaves candidates past its window unread, and a member that duplicates another's
        keys can sit among them. The Datasets mint (`trace.family_member_digest`) and
        Search (`resolved.shape.member_digest`) both state this set, so the two digests
        agree for one walk."""
        return None if self.truncated else frozenset(self.family)


class MemberReadDeadline(TimeoutError):
    """A walk's graph read started, or failed, past the caller's `deadline`.

    `candidate_cap` is the window being read when time ran out, `CANDIDATE_CAP` or the
    caller's `subject_ceiling`, and `candidates_considered` the subjects that window had
    returned, 0 before its page arrived. Running out of time says nothing about the set's
    size, so a caller treats it as a transient timeout. A retry walks again."""

    def __init__(
        self, *, five_tuple: FiveTuple, candidates_considered: int, candidate_cap: int
    ) -> None:
        self.five_tuple = five_tuple
        self.candidates_considered = candidates_considered
        self.candidate_cap = candidate_cap
        super().__init__(
            f"the member walk of {five_tuple!r} ran past its deadline after "
            f"{candidates_considered} candidates of a {candidate_cap}-candidate window"
        )


class MemberQueryUnanchored(Exception):
    """`measured_property` must anchor the reverse query."""

    def __init__(self, five_tuple: FiveTuple) -> None:
        self.five_tuple = five_tuple
        super().__init__(f"{five_tuple!r} names no measured_property to anchor the reverse query")


class MemberEnumerationTooLarge(Exception):
    """The confirmed, antichain-reduced member count exceeds `member_cap`."""

    def __init__(self, *, five_tuple: FiveTuple, count: int, member_cap: int) -> None:
        self.five_tuple = five_tuple
        self.count = count
        self.member_cap = member_cap
        super().__init__(
            f"{count} members are consistent with the slot bindings for {five_tuple!r}, "
            f"exceeding member_cap={member_cap}"
        )


class DegenerateObservationProperties(Exception):
    """A candidate declares `observationAbout` among its `observationProperties`."""

    def __init__(self, *, stat_var: str, observation_properties: tuple[str, ...]) -> None:
        self.stat_var = stat_var
        self.observation_properties = observation_properties
        super().__init__(
            f"{stat_var!r}'s observationProperties non-emptily declares observationAbout: "
            f"{observation_properties!r}"
        )


class MultiValuedConstraint(Exception):
    """A matching constraint property carries more than one value."""

    def __init__(self, *, stat_var: str, property: str, values: tuple[str, ...]) -> None:
        self.stat_var = stat_var
        self.property = property
        self.values = values
        super().__init__(
            f"{stat_var!r}'s {property!r} constraint arc carries {len(values)} bound values, "
            f"not one: {values!r}"
        )


class NotAStatisticalVariable(Exception):
    """A trusted antichain re-read identifies a non-StatVar candidate.

    This includes a dcid with no graph node, whose `typeOf` read back absent on the first read."""

    def __init__(self, *, stat_var: str, type_of: tuple[str, ...]) -> None:
        self.stat_var = stat_var
        self.type_of = type_of
        super().__init__(
            f"{stat_var!r}'s typeOf arc {type_of!r} doesn't include StatisticalVariable"
        )


class MemberRoleDisagreement(Exception):
    """Antichain members declare different role sets: a tie `elect_role_class` cannot
    break, between members no one of whose role sets sits inside all the others, or whose
    fewest-role members hold constraint keys the others lack.
    Checked only on final survivors, whose roles must agree for materialisation."""

    def __init__(
        self, *, five_tuple: FiveTuple, roles_by_stat_var: Mapping[str, tuple[str, ...]]
    ) -> None:
        self.five_tuple = five_tuple
        self.roles_by_stat_var = dict(roles_by_stat_var)
        super().__init__(
            f"members of {five_tuple!r} disagree on declared_roles: "
            f"{dict(sorted(self.roles_by_stat_var.items()))!r}"
        )


def enumerate_members(
    *,
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
    graph: MemberArcReader,
    readings: Mapping[str, AbsentReading] = EMPTY_READINGS,
    member_cap: int = DEFAULT_MEMBER_CAP,
    extra_candidates: Sequence[str] = (),
    role_keys: Collection[str] = (),
    subject_ceiling: int | None = None,
    deadline: float | None = None,
    full_read_deadline: float | None = None,
    open_role_keys: Collection[str] = (),
) -> MemberGroup:
    """Enumerate matching StatVars and reduce them to the minimal-constraint antichain.

    `role_keys` holds the role keys the ask binds entities under (`observationAbout`,
    `origin`, `destination`, or a declared role name), and `open_role_keys` the flow axes
    it asks every member of. `elect_role_class` keeps the members whose declared roles
    cover the bound keys and the open ones that join them (`electing_role_keys`) and
    reduces over constraint properties, then over declared roles, so a member lacking a
    role its sibling carries is the aggregate over that role.

    `extra_candidates` joins the reverse-query results before confirmation and
    follows the same membership, reduction, and trusted re-read checks. It skips
    filter reconciliation, which applies only to arcs established by the reverse query.
    A confirmed root extra can establish completeness on a truncated page.

    `readings` defines how absent constraints affect domination. Unspecified
    properties default to `aggregate`. Callers with `not_declared` properties
    must supply their registry-backed mapping. `member_cap` limits the reduced
    result and defaults to 700. `MemberEnumerator` passes its own `member_cap`.

    `subject_ceiling`, when set, settles a first page that truncates without proof by
    reading the whole filtered reverse set, up to that many subjects, and confirming
    it (`MemberGroup.full_read`). A complete read settles by construction. A set past
    the ceiling returns a truncated, unproven group with `candidate_cap` the ceiling.

    `deadline`, an absolute `time.monotonic()` instant, bounds every graph read: a read
    started past it, or failing once it has passed, raises `MemberReadDeadline`.
    `full_read_deadline` replaces it once the walk enters the full read, and defaults
    to `deadline`.

    Raises `MemberQueryUnanchored` when `measured_property` is absent,
    `MemberEnumerationTooLarge` above the reduced member cap,
    `MemberRoleDisagreement` for survivor roles the election cannot order,
    `MultiValuedConstraint` for a matching slot with multiple values, or
    `NotAStatisticalVariable` when a trusted re-read rejects a winner's type.
    Returns an empty group when all candidates fail the bindings.
    """
    return _enumerate_resolved(
        five_tuple,
        slots=slots,
        graph=graph,
        readings=readings,
        member_cap=member_cap,
        extra_candidates=extra_candidates,
        role_keys=role_keys,
        subject_ceiling=subject_ceiling,
        deadline=deadline,
        full_read_deadline=full_read_deadline,
        open_role_keys=open_role_keys,
    )


@dataclass
class _Window:
    """The reverse-arc window a walk is reading: its cap and subjects returned, for the
    refusal a deadline crossing raises, and the deadline its reads run under."""

    cap: int
    deadline: float | None
    considered: int = 0


def _enumerate_resolved(
    five_tuple: FiveTuple,
    *,
    slots: Mapping[str, SlotBinding],
    graph: MemberArcReader,
    readings: Mapping[str, AbsentReading],
    member_cap: int,
    extra_candidates: Sequence[str] = (),
    role_keys: Collection[str] = (),
    subject_ceiling: int | None = None,
    deadline: float | None = None,
    full_read_deadline: float | None = None,
    open_role_keys: Collection[str] = (),
) -> MemberGroup:
    """Run `_confirm_group` on the reverse-query page plus caller-supplied candidates,
    then, under `subject_ceiling`, on the whole filtered set when that page truncates
    without proof (`_read_whole_set`). Truncation passes through to `MemberGroup`. An
    empty truncated result leaves matching members beyond the page unaccounted for. An
    unresolved-denominator ask reads through `_ProvisionalDenominatorReader`, above the
    `_DeadlineReader` that bounds every read by the window's deadline: `deadline` for
    the first page, then `full_read_deadline`, which defaults to `deadline`."""
    if full_read_deadline is None:
        full_read_deadline = deadline
    window = _Window(cap=CANDIDATE_CAP, deadline=deadline)
    if deadline is not None or full_read_deadline is not None:
        graph = _DeadlineReader(graph, window)
    if five_tuple.denominator_unresolved:
        graph = _ProvisionalDenominatorReader(graph)
    try:
        page = _reverse_query(graph, five_tuple, slots=slots, max_results=CANDIDATE_CAP)
        window.considered = len(page.dcids)
        if subject_ceiling is None or not page.truncated:
            return _confirm_group(
                five_tuple,
                page.dcids,
                slots=slots,
                graph=graph,
                readings=readings,
                member_cap=member_cap,
                candidate_cap=CANDIDATE_CAP,
                truncated=page.truncated,
                extra_candidates=extra_candidates,
                role_keys=frozenset(role_keys),
                open_role_keys=frozenset(open_role_keys),
            )
        return _read_whole_set(
            five_tuple,
            page.dcids,
            window=window,
            subject_ceiling=subject_ceiling,
            full_read_deadline=full_read_deadline,
            slots=slots,
            graph=graph,
            readings=readings,
            member_cap=member_cap,
            extra_candidates=extra_candidates,
            role_keys=frozenset(role_keys),
            open_role_keys=frozenset(open_role_keys),
        )
    except _DeadlineCrossed as exc:
        raise MemberReadDeadline(
            five_tuple=five_tuple,
            candidates_considered=window.considered,
            candidate_cap=window.cap,
        ) from exc.__cause__


def _read_whole_set(
    five_tuple: FiveTuple,
    page: Sequence[str],
    *,
    window: _Window,
    subject_ceiling: int,
    full_read_deadline: float | None,
    slots: Mapping[str, SlotBinding],
    graph: MemberArcReader,
    readings: Mapping[str, AbsentReading],
    member_cap: int,
    extra_candidates: Sequence[str],
    role_keys: frozenset[str],
    open_role_keys: frozenset[str],
) -> MemberGroup:
    """Settle a truncated first page. Confirm it when it may hold a root that proves it
    complete, and otherwise read the whole filtered set up to `subject_ceiling` and
    confirm that.

    The first page's batch read serves both confirmations, so the full read reads only
    the arcs of subjects past the page. A page settles only under a confirmed root, a
    member whose constraint keys are exactly the `value`-bound slot keys. A read can only
    drop an arc, so a candidate whose read names a constraint outside those
    keys is no root, and a page holding only such candidates goes straight to the full
    read, as does a binding no root can prove complete (`root_can_prove_complete`). The
    full read runs under `full_read_deadline`."""
    properties = _candidate_properties(slots)
    page_arcs = graph.node_arcs_many(page, properties=properties)
    if extra_candidates or _page_may_hold_root(page_arcs, slots=slots, readings=readings):
        # A root past the page can dominate the members the page holds, so a cap or role
        # refusal of the page alone defers to the full read.
        with suppress(MemberEnumerationTooLarge, MemberRoleDisagreement):
            group = _confirm_group(
                five_tuple,
                page,
                slots=slots,
                graph=graph,
                readings=readings,
                member_cap=member_cap,
                candidate_cap=CANDIDATE_CAP,
                truncated=True,
                extra_candidates=extra_candidates,
                role_keys=role_keys,
                open_role_keys=open_role_keys,
                prior_arcs=page_arcs,
            )
            if group.proven_complete:
                return group
    window.cap, window.considered, window.deadline = subject_ceiling, 0, full_read_deadline
    whole = _reverse_query(graph, five_tuple, slots=slots, max_results=subject_ceiling)
    window.considered = len(whole.dcids)
    if whole.truncated:
        return MemberGroup(
            five_tuple=five_tuple,
            members=(),
            family=(),
            candidates_considered=len(whole.dcids),
            candidate_cap=subject_ceiling,
            truncated=True,
            proven_complete=False,
            root_confirmed=False,
            full_read=True,
        )
    group = _confirm_group(
        five_tuple,
        whole.dcids,
        slots=slots,
        graph=graph,
        readings=readings,
        member_cap=member_cap,
        candidate_cap=subject_ceiling,
        truncated=False,
        extra_candidates=extra_candidates,
        role_keys=role_keys,
        open_role_keys=open_role_keys,
        prior_arcs=page_arcs,
    )
    return replace(group, full_read=True)


def _page_may_hold_root(
    page_arcs: Mapping[str, Mapping[str, tuple[str, ...]] | None],
    *,
    slots: Mapping[str, SlotBinding],
    readings: Mapping[str, AbsentReading],
) -> bool:
    """Whether a confirmed root could sit among `page_arcs`: the binding lets a root prove
    the family complete, and some candidate's read, or missing read, names no constraint
    outside the `value`-bound slot keys."""
    if not root_can_prove_complete(slots=slots, readings=readings):
        return False
    bound_keys = frozenset(name for name, binding in slots.items() if binding[0] == "value")
    return any(
        arcs is None or frozenset(arcs.get("constraintProperties", ())) <= bound_keys
        for arcs in page_arcs.values()
    )


class _DeadlineCrossed(Exception):
    """A read under `_DeadlineReader` started or failed past its deadline. The walk turns it
    into `MemberReadDeadline` with the window it was reading."""


class _DeadlineReader:
    """Read through `graph` until the deadline of `window`, a `time.monotonic()` instant
    or `None` for no bound, read at each call so a walk entering the full read moves it.
    A read started past it raises `_DeadlineCrossed`, and so does any read failure once
    it has passed. The deadline rides on each read, so every page and request ends by it."""

    def __init__(self, graph: MemberArcReader, window: _Window) -> None:
        self._graph = graph
        self._window = window

    def _crossed(self) -> bool:
        return self._window.deadline is not None and time.monotonic() >= self._window.deadline

    def _bound(self, deadline: float | None) -> float | None:
        if self._window.deadline is None:
            return deadline
        return self._window.deadline if deadline is None else min(deadline, self._window.deadline)

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        if self._crossed():
            raise _DeadlineCrossed
        try:
            return _node_subjects(
                self._graph,
                dcid,
                property=property,
                filters=filters,
                max_results=max_results,
                deadline=self._bound(deadline),
            )
        except Exception as exc:
            if self._crossed():
                raise _DeadlineCrossed from exc
            raise

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        if self._crossed():
            raise _DeadlineCrossed
        try:
            return self._graph.node_arcs_many(
                dcids,
                properties=properties,
                individual_probe_budget=individual_probe_budget,
                deadline=self._bound(deadline),
            )
        except Exception as exc:
            if self._crossed():
                raise _DeadlineCrossed from exc
            raise


def _node_subjects(
    graph: MemberArcReader,
    dcid: str,
    *,
    property: str,
    filters: Mapping[str, str] | None,
    max_results: int,
    deadline: float | None,
) -> SubjectPage:
    """Forward a reverse read, passing `deadline` only when one is set, since a reader
    the walk wraps without a deadline may take none."""
    if deadline is None:
        return graph.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
    return graph.node_subjects(
        dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
    )


class _ProvisionalDenominatorReader:
    """Read arcs through `graph`, writing `UNRESOLVED_NODE` for a `measurementDenominator`
    target typed exactly `ProvisionalNode`.

    Search reads a denominator as unresolved when its target has no dcid or is a node
    typed exactly `ProvisionalNode`, and `node_arcs_many` keeps such a target's dcid
    (`PerCapita` reads as `("PerCapita",)`). This reader reads each new target's `typeOf`
    once and keeps the verdict for the enumeration, so every read path, re-probes
    included, sees one denominator state. `_enumerate_resolved` uses it only for an
    unresolved-denominator ask, so concrete and absent denominators read unchanged."""

    def __init__(self, graph: MemberArcReader) -> None:
        self._graph = graph
        self._provisional: dict[str, bool] = {}

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        return _node_subjects(
            self._graph,
            dcid,
            property=property,
            filters=filters,
            max_results=max_results,
            deadline=deadline,
        )

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        read = self._graph.node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )
        if _MEASUREMENT_DENOMINATOR not in properties:
            return read
        targets = {
            target
            for arcs in read.values()
            if arcs is not None
            for target in arcs[_MEASUREMENT_DENOMINATOR]
            if target != UNRESOLVED_NODE
        }
        self._read_types(sorted(targets - self._provisional.keys()))
        # Copies, so a reader that caches its results keeps the originals.
        return {
            dcid: None
            if arcs is None
            else {
                **arcs,
                _MEASUREMENT_DENOMINATOR: tuple(
                    dict.fromkeys(
                        UNRESOLVED_NODE if self._provisional.get(target) else target
                        for target in arcs[_MEASUREMENT_DENOMINATOR]
                    )
                ),
            }
            for dcid, arcs in read.items()
        }

    def _read_types(self, targets: Sequence[str]) -> None:
        """Record whether each target is typed exactly `ProvisionalNode`. A target whose
        `typeOf` reads empty in a batch is re-read alone, since a batch read can drop it."""
        if not targets:
            return
        read = self._graph.node_arcs_many(targets, properties=("typeOf",))
        for target in targets:
            type_of = (read.get(target) or {}).get("typeOf", ())
            if not type_of and len(targets) > 1:
                alone = self._graph.node_arcs_many([target], properties=("typeOf",))
                type_of = (alone.get(target) or {}).get("typeOf", ())
            self._provisional[target] = type_of == _PROVISIONAL_NODE_TYPES


def five_tuple_reverse_filters(
    *,
    population_type: str | None,
    stat_type: str | None,
    measurement_qualifier: str | None,
    measurement_denominator: str | None,
) -> dict[str, str]:
    """Return filters for present population type, stat type, qualifier, and concrete
    denominator values. Absent or unresolved arcs require client-side checks.
    Scalar arguments let other five-tuple types reuse this mapping."""
    filters: dict[str, str] = {}
    if population_type:
        filters["populationType"] = population_type
    if stat_type:
        filters["statType"] = stat_type
    if measurement_qualifier:
        filters["measurementQualifier"] = measurement_qualifier
    if measurement_denominator:
        filters["measurementDenominator"] = measurement_denominator
    return filters


def five_tuple_arcs_match(
    arcs: Mapping[str, tuple[str, ...]],
    *,
    population_type: str | None,
    measured_property: str | None,
    stat_type: str | None,
    measurement_qualifier: str | None,
    measurement_denominator: str | None,
    denominator_unresolved: bool = False,
) -> bool:
    """Check the five-tuple arcs exactly. A `None` field requires an empty candidate arc;
    unresolved denominators match `UNRESOLVED_NODE`. Pass `None` for fields already
    proven by the lead reverse arc. Scalar arguments support other tuple types."""
    if not _matches_optional_arc(arcs, "populationType", population_type):
        return False
    if not _matches_optional_arc(arcs, "measuredProperty", measured_property):
        return False
    if not _matches_optional_arc(arcs, "statType", stat_type):
        return False
    if not _matches_optional_arc(arcs, "measurementQualifier", measurement_qualifier):
        return False
    denominator_arc = arcs.get("measurementDenominator", ())
    if measurement_denominator is None and not denominator_unresolved:
        return not denominator_arc
    if not denominator_arc:
        return False
    if measurement_denominator is None:
        return all(value == UNRESOLVED_NODE for value in denominator_arc)
    return denominator_arc == (measurement_denominator,)


def _safe_slot_filters(slots: Mapping[str, SlotBinding]) -> dict[str, str]:
    """Return only `value` bindings as reverse-query filters. `set` and `unbound`
    bindings allow multiple values or absence, so confirmation stays client-side."""
    filters: dict[str, str] = {}
    for name, binding in slots.items():
        match binding:
            case ("value", ref):
                filters[name] = ref
    return filters


def _reverse_query(
    graph: MemberArcReader,
    five_tuple: FiveTuple,
    *,
    slots: Mapping[str, SlotBinding],
    max_results: int,
) -> SubjectPage:
    """Query up to `max_results` subjects by `measuredProperty`, with safe five-tuple and
    slot filters. The narrower measured-property arc avoids the much broader population
    axis. Raises `MemberQueryUnanchored` when `measured_property` is absent."""
    if five_tuple.measured_property is None:
        raise MemberQueryUnanchored(five_tuple)
    filters = five_tuple_reverse_filters(
        population_type=five_tuple.population_type,
        stat_type=five_tuple.stat_type,
        measurement_qualifier=five_tuple.measurement_qualifier,
        measurement_denominator=five_tuple.measurement_denominator,
    )
    filters.update(_safe_slot_filters(slots))
    return graph.node_subjects(
        five_tuple.measured_property,
        property="measuredProperty",
        filters=filters,
        max_results=max_results,
    )


def _candidate_properties(slots: Mapping[str, SlotBinding]) -> tuple[str, ...]:
    """The out-arcs `_confirm_group` reads for each candidate."""
    return _FIVE_TUPLE_PROPERTIES + tuple(sorted(slots))


def _confirm_group(
    five_tuple: FiveTuple,
    candidate_dcids: Sequence[str],
    *,
    slots: Mapping[str, SlotBinding],
    graph: MemberArcReader,
    readings: Mapping[str, AbsentReading],
    member_cap: int,
    candidate_cap: int,
    truncated: bool,
    extra_candidates: Sequence[str] = (),
    role_keys: frozenset[str] = frozenset(),
    prior_arcs: Mapping[str, dict[str, tuple[str, ...]] | None] = _EMPTY_ARCS,
    open_role_keys: frozenset[str] = frozenset(),
) -> MemberGroup:
    """Confirm walked and caller-supplied candidates, reduce survivors, and enforce
    `member_cap` on the antichain result. The reverse-query page is capped at
    `candidate_cap`. Supplied candidates use a separate probe budget. A walked
    candidate in `prior_arcs` takes that batch read in place of its own.
    `truncated` and `candidate_cap` pass through unchanged. Completeness is computed
    from the final antichain. Drop candidates with a present, non-StatVar `typeOf`.
    Missing `typeOf` is inconclusive, so winners receive a trusted re-read.

    Extras are deduplicated and receive the same membership and reduction checks,
    and skip filter reconciliation, which applies only to candidates the reverse
    query established. Their forward reads use a separate probe budget. Ordinary
    non-matches follow the membership checks.

    The reduction is `elect_role_class` under `role_keys` and `open_role_keys`. Each
    election, from the trusted re-read to the final one after the root twins join,
    resolves its keys with `electing_role_keys` over the members it elects from, so an
    open role's twin found late displaces the total and one settled out leaves it standing.
    A truncated page is proven complete only when the final election read every open key,
    so a page whose members serve no asked open role goes to the full read, where a twin
    past the page is found. The root that proves a page must also cover the bound keys
    (`_root_covers_bound_keys`), so that no member past the page changes which open keys
    join.
    A confirmed member that reads subject-only and fails a flow role key is re-read first
    (`_reconcile_role_rejected`),
    since a batch read can drop its `observationProperties` arc. After the trusted re-read
    of the winners, `_settle_role_absence` reads alone each member whose role class still
    rests on that arc reading empty.
    """
    deduped_extra = tuple(
        dcid for dcid in dict.fromkeys(extra_candidates) if dcid not in candidate_dcids
    )
    properties = _candidate_properties(slots)
    graph = single_reads = _SingleReads(graph, properties=properties)
    arcs_by_dcid = {dcid: prior_arcs[dcid] for dcid in candidate_dcids if dcid in prior_arcs}
    unread = [dcid for dcid in candidate_dcids if dcid not in arcs_by_dcid]
    if unread or not prior_arcs:
        arcs_by_dcid.update(graph.node_arcs_many(unread, properties=properties))
    if deduped_extra:
        arcs_by_dcid.update(
            graph.node_arcs_many(
                deduped_extra,
                properties=properties,
                individual_probe_budget=_EXTRA_CANDIDATE_PROBE_BUDGET,
            )
        )
    all_dcids = tuple(candidate_dcids) + deduped_extra
    arcs_by_dcid, inconsistent_candidates = _reconcile_filter_covered(
        five_tuple, candidate_dcids, arcs_by_dcid, slots=slots, properties=properties, graph=graph
    )
    consistent: list[Member] = []
    absence_suspects: list[str] = []
    for dcid in all_dcids:
        arcs = arcs_by_dcid.get(dcid)
        if arcs is None:
            # `_reconcile_filter_covered` re-probed empty reads. A remaining `None`
            # is already recorded as inconsistent, so skip it as a member candidate.
            continue
        if _type_of_names_a_non_stat_var(arcs):
            continue
        confirmation = _confirm_member(dcid, arcs, five_tuple=five_tuple, slots=slots)
        if confirmation.member is not None:
            consistent.append(confirmation.member)
        elif confirmation.absence_rejected:
            absence_suspects.append(dcid)
    recovered, unresolved_suspects = _reconcile_absence_rejected(
        _root_shaped_first(absence_suspects, arcs_by_dcid=arcs_by_dcid, slots=slots),
        five_tuple=five_tuple,
        slots=slots,
        properties=properties,
        graph=graph,
    )
    consistent.extend(recovered)
    consistent = _reconcile_role_rejected(
        consistent,
        role_keys=_elect(
            consistent, readings=readings, role_keys=role_keys, open_role_keys=open_role_keys
        ).role_keys,
        five_tuple=five_tuple,
        slots=slots,
        properties=properties,
        graph=graph,
    )
    inconsistent_candidates = inconsistent_candidates + unresolved_suspects
    _, confirmed, unverified = _reverify_antichain(
        five_tuple,
        consistent,
        slots=slots,
        graph=graph,
        properties=properties,
        readings=readings,
        role_keys=role_keys,
        open_role_keys=open_role_keys,
    )
    confirmed, role_unsettled = _settle_role_absence(
        confirmed,
        single_reads=single_reads,
        five_tuple=five_tuple,
        slots=slots,
        readings=readings,
        role_keys=role_keys,
        open_role_keys=open_role_keys,
    )
    election = _elect(
        confirmed, readings=readings, role_keys=role_keys, open_role_keys=open_role_keys
    )
    members = _sorted_members(confirmed, election.members)
    unresolved = inconsistent_candidates + unverified + role_unsettled
    root_confirmed = _antichain_proven_complete(
        members, slots=slots, readings=readings, role_keys=election.role_keys
    ) and _root_covers_bound_keys(confirmed, slots=slots, role_keys=role_keys)
    undismissed = unresolved
    if root_confirmed:
        twins = _possible_root_twins(
            unresolved, arcs_by_dcid=arcs_by_dcid, consistent=consistent, slots=slots
        )
        read_twins, undismissed = _read_root_twins(
            twins,
            walked=frozenset(candidate_dcids),
            five_tuple=five_tuple,
            slots=slots,
            graph=graph,
            properties=properties,
        )
        confirmed += read_twins
        settled = frozenset(twins) - frozenset(undismissed)
        unresolved = tuple(dcid for dcid in unresolved if dcid not in settled)
    election = _elect(
        confirmed, readings=readings, role_keys=role_keys, open_role_keys=open_role_keys
    )
    members = _sorted_members(confirmed, election.members)
    family = election.family
    if len(members) > member_cap:
        raise MemberEnumerationTooLarge(
            five_tuple=five_tuple, count=len(members), member_cap=member_cap
        )
    if root_confirmed or not unresolved:
        _check_role_agreement(five_tuple, members)
    return MemberGroup(
        five_tuple=five_tuple,
        members=members,
        family=family,
        candidates_considered=len(candidate_dcids),
        candidate_cap=candidate_cap,
        truncated=truncated,
        proven_complete=not truncated
        or (
            root_confirmed
            and open_role_keys <= election.role_keys
            and _root_elects_role_class(members, slots=slots, role_keys=election.role_keys)
        ),
        root_confirmed=root_confirmed,
        inconsistent_candidates=unresolved,
        undismissed_candidates=undismissed,
        elected_open_role_keys=election.role_keys - role_keys,
    )


def _antichain_proven_complete(
    members: Sequence[Member],
    *,
    slots: Mapping[str, SlotBinding],
    readings: Mapping[str, AbsentReading],
    role_keys: frozenset[str],
) -> bool:
    """A confirmed root member proves that unseen members with extra constraint keys
    would be dominated, whether they sit past a truncated page or behind a read that
    could not be trusted. Same-keyset twins may remain unseen.

    Let `B` be the `value`-bound slot keys. Every member includes `B`. A confirmed
    member with exactly those keys dominates members with extra keys when each
    extra property reads `aggregate`. This covers unbound slots because their
    keys can only extend `B`.

    The root must cover `role_keys` (`role_class_covers`). The election reduces constraint
    keys among the covering members only, so a root outside them dominates nothing, and an
    unread member that covers would replace it.

    The argument compares constraint keys. An unseen member of equal keys that declares
    fewer roles than the root would be elected over it, so a truncated page also needs
    `_root_elects_role_class`.

    Returns `False` when `root_can_prove_complete` rules the argument out for these slots."""
    if not root_can_prove_complete(slots=slots, readings=readings):
        return False
    return any(
        role_class_covers(root.declared_roles, role_keys) for root in _roots(members, slots=slots)
    )


def _root_covers_bound_keys(
    confirmed: Sequence[Member], *, slots: Mapping[str, SlotBinding], role_keys: frozenset[str]
) -> bool:
    """Whether a root among `confirmed` covers the bound `role_keys` (`role_class_covers`),
    so that no unread candidate changes which open keys join them.

    The open keys join by what the election under the bound keys alone reads
    (`electing_role_keys`). A root that covers the bound keys is a candidate of that
    election and dominates every candidate with more constraint keys, under the bindings
    `_antichain_proven_complete` accepts. The election then reads members of the root's
    constraint map whatever sits unread, and the members that place an open key for it
    hold that map too. A root that covers the elected keys covers a bound key among
    them. With no bound key only a subject-only root does: beside a role-bearing root
    alone, an unread subject-only member of more constraint keys would be read in its
    place, and an open key would join only if a twin of that member placed it."""
    return any(
        role_class_covers(root.declared_roles, role_keys) for root in _roots(confirmed, slots=slots)
    )


def _root_elects_role_class(
    members: Sequence[Member], *, slots: Mapping[str, SlotBinding], role_keys: frozenset[str]
) -> bool:
    """Whether a root among `members` covers `role_keys` and declares only roles that place
    one of them.

    A member that covers `role_keys` places each key with one declared role, since an alias
    two of its roles share places neither (`declared_role_for`). A member whose roles are a
    proper subset of such a root's lacks the role that places some key, so it does not cover
    the keys, and no unseen member of the root's constraint keys can be elected over the
    root. With no role key, only a subject-only member covers, and a subject-only root has
    the fewest roles. An unseen member of equal keys whose roles neither contain nor fall
    inside the root's ties with it, the limit `MemberGroup.proven_complete` states for a
    same-keyset member."""
    return any(
        declares_only_placing_roles(root.declared_roles, role_keys)
        for root in _roots(members, slots=slots)
    )


def declares_only_placing_roles(
    declared_roles: tuple[str, ...], role_keys: Collection[str]
) -> bool:
    """Whether a member declaring `declared_roles` covers `role_keys` and declares no role
    beyond the ones that place them, so a root declaring them elects the role class
    (`_root_elects_role_class`). With no role key, only a subject-only member does."""
    if not role_keys:
        return declared_roles == (_OBSERVATION_ABOUT,)
    return {declared_role_for(key, declared_roles) for key in role_keys} == set(declared_roles)


def _roots(members: Sequence[Member], *, slots: Mapping[str, SlotBinding]) -> list[Member]:
    """Return the members constrained by exactly the `value`-bound slot keys."""
    bound_keys = frozenset(name for name, binding in slots.items() if binding[0] == "value")
    return [member for member in members if frozenset(member.constraints) == bound_keys]


def root_can_prove_complete(
    *, slots: Mapping[str, SlotBinding], readings: Mapping[str, AbsentReading]
) -> bool:
    """Whether a confirmed root, a member constrained by exactly the `value`-bound slot
    keys, would prove a family under `slots` complete.

    `not_declared` properties preserve distinct populations. `set` slots can yield
    several refs with identical keys, which subset domination treats as ties. Returns `False`
    when any slot is `set`-bound or an unbound property reads `not_declared`."""
    if any(binding[0] == "set" for binding in slots.values()):
        return False
    unbound_properties = (name for name, binding in slots.items() if binding[0] == "unbound")
    return all(readings.get(name, "aggregate") == "aggregate" for name in unbound_properties)


def _possible_root_twins(
    unresolved: Sequence[str],
    *,
    arcs_by_dcid: Mapping[str, Mapping[str, tuple[str, ...]] | None],
    consistent: Sequence[Member],
    slots: Mapping[str, SlotBinding],
) -> tuple[str, ...]:
    """Return the unresolved candidates that may share a confirmed root's constraint keys.

    A read can drop an arc but never add one, so a constraint name any read showed
    outside the `value`-bound slot keys makes the candidate a non-member or a member the
    root dominates. Any other candidate may be a same-keyset twin of the root: a survivor
    whose roles no trusted read has checked. Call only when `_antichain_proven_complete`
    holds, so every unbound property reads `aggregate` and no slot is `set`-bound."""
    bound_keys = frozenset(name for name, binding in slots.items() if binding[0] == "value")
    confirmed_keys = {member.stat_var: frozenset(member.constraints) for member in consistent}
    return tuple(
        dcid
        for dcid in unresolved
        if confirmed_keys.get(dcid, frozenset())
        | frozenset((arcs_by_dcid.get(dcid) or {}).get("constraintProperties", ()))
        <= bound_keys
    )


def _read_root_twins(
    twins: Sequence[str],
    *,
    walked: frozenset[str],
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
    graph: MemberArcReader,
    properties: Sequence[str],
) -> tuple[tuple[Member, ...], tuple[str, ...]]:
    """Read the first `_TWIN_PROBE_BUDGET` possible root twins one dcid at a time.

    A single-dcid read is trusted, so a read settles its twin: a confirmed member
    joins the group (a root twin survives beside the root, any other the root dominates),
    and a non-match is dismissed. A twin in `walked` is the exception when its read lacks
    a filter-covered arc: the reverse query proved that arc, so the read contradicts the
    index and settles nothing, as in `_reconcile_filter_covered`. Returns the confirmed
    members and the twins still unsettled: those whose read came back empty or
    contradicted the index, and those past the budget."""
    scalar_expected, slot_expected = _filter_covered_expectations(five_tuple, slots)
    confirmed: list[Member] = []
    unsettled: list[str] = []
    for dcid in twins[:_TWIN_PROBE_BUDGET]:
        arcs = graph.node_arcs_many([dcid], properties=properties).get(dcid)
        if arcs is None:
            unsettled.append(dcid)
            continue
        if _type_of_names_a_non_stat_var(arcs):
            continue
        if dcid in walked and not _matches_filter_covered(arcs, scalar_expected, slot_expected):
            unsettled.append(dcid)
            continue
        member = _confirm_member(
            dcid, arcs, five_tuple=five_tuple, slots=slots, trust_type_of=True
        ).member
        if member is not None:
            confirmed.append(member)
    return tuple(confirmed), (*unsettled, *twins[_TWIN_PROBE_BUDGET:])


def _root_shaped_first(
    suspects: Sequence[str],
    *,
    arcs_by_dcid: Mapping[str, Mapping[str, tuple[str, ...]] | None],
    slots: Mapping[str, SlotBinding],
) -> list[str]:
    """Order `suspects` so those whose read names no constraint outside the `value`-bound
    slot keys come first, keeping each part's order. Such a suspect may be a root or a
    root's twin, so a reconciliation budget that reaches it first settles it before
    `_read_root_twins` has to."""
    bound_keys = frozenset(name for name, binding in slots.items() if binding[0] == "value")
    return sorted(
        suspects,
        key=lambda dcid: (
            not (
                frozenset((arcs_by_dcid.get(dcid) or {}).get("constraintProperties", ()))
                <= bound_keys
            )
        ),
    )


def _check_role_agreement(five_tuple: FiveTuple, members: Sequence[Member]) -> None:
    """Raise `MemberRoleDisagreement` if survivors declare different role sets, which is
    a tie `elect_role_class` left standing.

    Call when no unresolved candidate can remove a survivor: after every candidate is
    verified, or under a confirmed root, whose members hold exactly the `value`-bound
    keys that no unresolved candidate can dominate. An unresolved candidate can still
    add a role set, so a disagreement among the survivors is certain either way, and
    `undismissed_candidates` carries the candidates whose roles stay unread."""
    role_sets = {frozenset(member.declared_roles) for member in members}
    if len(role_sets) > 1:
        raise MemberRoleDisagreement(
            five_tuple=five_tuple,
            roles_by_stat_var={member.stat_var: member.declared_roles for member in members},
        )


def _filter_covered_expectations(
    five_tuple: FiveTuple, slots: Mapping[str, SlotBinding]
) -> tuple[Mapping[str, str], Mapping[str, str]]:
    """Return the values established by reverse-query filters, split by check type.
    Scalars must match as single-value arcs. `value` slot filters need only remain
    present because a constraint arc may be multi-valued. Absent fields,
    unresolved denominators, and `set` or `unbound` slots stay client-side."""
    assert five_tuple.measured_property is not None, "_reverse_query already required this"
    scalar_expectations: dict[str, str] = {"measuredProperty": five_tuple.measured_property}
    if five_tuple.population_type:
        scalar_expectations["populationType"] = five_tuple.population_type
    if five_tuple.stat_type:
        scalar_expectations["statType"] = five_tuple.stat_type
    if five_tuple.measurement_qualifier:
        scalar_expectations["measurementQualifier"] = five_tuple.measurement_qualifier
    if five_tuple.measurement_denominator:
        scalar_expectations["measurementDenominator"] = five_tuple.measurement_denominator
    return scalar_expectations, _safe_slot_filters(slots)


def _matches_filter_covered(
    arcs: Mapping[str, tuple[str, ...]],
    scalar_expected: Mapping[str, str],
    slot_expected: Mapping[str, str],
) -> bool:
    if any(arcs.get(property, ()) != (value,) for property, value in scalar_expected.items()):
        return False
    return all(value in arcs.get(property, ()) for property, value in slot_expected.items())


def _reconcile_filter_covered(
    five_tuple: FiveTuple,
    candidate_dcids: Sequence[str],
    arcs_by_dcid: Mapping[str, Mapping[str, tuple[str, ...]] | None],
    *,
    slots: Mapping[str, SlotBinding],
    properties: Sequence[str],
    graph: MemberArcReader,
) -> tuple[Mapping[str, Mapping[str, tuple[str, ...]] | None], tuple[str, ...]]:
    """Check that walked candidates' forward reads retain the facts established by
    `_reverse_query` filters. Batch reads can drop arcs, so probe up to 32
    contradictions in one batch and retry uncertain dcids individually. Persistent mismatches
    are returned as inconsistent, distinct from ordinary non-matches.

    Reconcile only walked candidates. Caller-supplied extras lack reverse-filter
    evidence and use ordinary membership checks."""
    scalar_expected, slot_expected = _filter_covered_expectations(five_tuple, slots)
    suspects = _root_shaped_first(
        [
            dcid
            for dcid in candidate_dcids
            if (arcs := arcs_by_dcid.get(dcid)) is None
            or not _matches_filter_covered(arcs, scalar_expected, slot_expected)
        ],
        arcs_by_dcid=arcs_by_dcid,
        slots=slots,
    )
    if not suspects:
        return arcs_by_dcid, ()

    budgeted = suspects[:_REPROBE_BUDGET]
    reconciled = dict(arcs_by_dcid)
    inconsistent: list[str] = []
    reprobed_by_dcid = graph.node_arcs_many(budgeted, properties=properties)
    for dcid in budgeted:
        reprobed = reprobed_by_dcid.get(dcid)
        still_suspect = reprobed is None or not _matches_filter_covered(
            reprobed, scalar_expected, slot_expected
        )
        if still_suspect and len(budgeted) > 1:
            reprobed = graph.node_arcs_many([dcid], properties=properties).get(dcid)
        if reprobed is not None and _matches_filter_covered(
            reprobed, scalar_expected, slot_expected
        ):
            reconciled[dcid] = reprobed
        else:
            inconsistent.append(dcid)
    inconsistent.extend(suspects[_REPROBE_BUDGET:])
    return reconciled, tuple(inconsistent)


@dataclass(frozen=True)
class _Confirmation:
    member: Member | None
    absence_rejected: bool
    """`True` when `member` is `None` because a required arc came back `()`
    rather than because a present value failed to match, an unnameable
    marker was read, or the candidate asserted something outside `slots`'s
    vocabulary. Covers `measurementDenominator` for a `denominator_unresolved`
    ask (a concrete-dcid ask's own denominator absence is filter-covered, so
    `_matches_five_tuple` never sets this flag for it), a bound constraint's
    own value arc, and a bound constraint's name missing from
    `constraintProperties`. Never `typeOf` missing `StatisticalVariable`:
    `_confirm_member` only enforces that with `trust_type_of=True`, which
    would raise rather than set this flag, and a `typeOf` arc absent from
    the read is not a verdict at all. This is exactly the distinction
    `_reconcile_absence_rejected` needs: a `node_arcs_many` batch can
    silently drop an arc, so a rejection resting on one being absent is
    untrusted until re-probed, while a rejection resting on a present value
    is not -- a drop can remove an arc, never fabricate one."""


def _confirm_member(
    dcid: str,
    arcs: Mapping[str, tuple[str, ...]],
    *,
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
    trust_type_of: bool = False,
) -> _Confirmation:
    """Exclude candidates whose five-tuple or slot bindings fail, required arcs are
    absent, or undeclared constraints appear. Identity slots require resolvable
    refs, so unbound unresolved markers fail. A matching constraint with multiple
    values raises `MultiValuedConstraint`.

    Check `typeOf` only when `trust_type_of=True`, on trusted winner re-reads.
    Missing `typeOf` in a batch remains inconclusive."""
    matches, absence_caused = _matches_five_tuple(arcs, five_tuple)
    if not matches:
        return _Confirmation(None, absence_rejected=absence_caused)
    constraint_names = arcs.get("constraintProperties", ())
    if any(name not in slots for name in constraint_names):
        return _Confirmation(None, absence_rejected=False)
    constraints: dict[str, str] = {}
    for name in constraint_names:
        values = arcs.get(name, ())
        if not values:
            return _Confirmation(None, absence_rejected=True)
        if not _binding_allows(slots[name], values[0]):
            return _Confirmation(None, absence_rejected=False)
        if len(values) > 1:
            raise MultiValuedConstraint(stat_var=dcid, property=name, values=values)
        # Per property, so a member is excluded at its first unresolved value before a
        # later property can raise.
        if not constraints_resolve((name,), {name: values[0]}):
            return _Confirmation(None, absence_rejected=False)
        constraints[name] = values[0]
    if any(
        binding[0] != "unbound" and name not in constraint_names for name, binding in slots.items()
    ):
        return _Confirmation(None, absence_rejected=True)
    if trust_type_of and _STATISTICAL_VARIABLE not in arcs.get("typeOf", ()):
        raise NotAStatisticalVariable(stat_var=dcid, type_of=arcs.get("typeOf", ()))
    roles = _declared_roles(dcid, arcs.get("observationProperties", ()))
    return _Confirmation(
        Member(stat_var=dcid, constraints=constraints, declared_roles=roles), absence_rejected=False
    )


def _reconcile_absence_rejected(
    suspects: Sequence[str],
    *,
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
    properties: Sequence[str],
    graph: MemberArcReader,
) -> tuple[tuple[Member, ...], tuple[str, ...]]:
    """Re-probe candidates rejected because a required arc was absent. Batch reads may
    drop arcs. Probe up to `_REPROBE_BUDGET` together, then retry unresolved
    candidates individually. Trusted absence is an ordinary non-match. Unprobed
    excess is inconsistent. Return recovered members and inconsistent ids."""
    budgeted = suspects[:_REPROBE_BUDGET]
    inconsistent = tuple(suspects[_REPROBE_BUDGET:])
    if not budgeted:
        return (), inconsistent

    recovered: list[Member] = []
    reprobed_by_dcid = graph.node_arcs_many(budgeted, properties=properties)
    for dcid in budgeted:
        confirmation = _confirm_absence_suspect(
            dcid, reprobed_by_dcid.get(dcid), five_tuple=five_tuple, slots=slots
        )
        if confirmation.member is None and confirmation.absence_rejected and len(budgeted) > 1:
            confirmation = _confirm_absence_suspect(
                dcid,
                graph.node_arcs_many([dcid], properties=properties).get(dcid),
                five_tuple=five_tuple,
                slots=slots,
            )
        if confirmation.member is not None:
            recovered.append(confirmation.member)
    return tuple(recovered), inconsistent


def _reconcile_role_rejected(
    consistent: Sequence[Member],
    *,
    role_keys: frozenset[str],
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
    properties: Sequence[str],
    graph: MemberArcReader,
) -> list[Member]:
    """Re-read confirmed members that role coverage would drop only because their
    `observationProperties` arc read empty.

    A batch read can drop the arc, and the member then reads subject-only and fails a
    flow role key. A present role cannot be fabricated, so a member that reads
    role-bearing is trusted. Suspects exist only when some member covers `role_keys`,
    since otherwise coverage drops nothing. One batch read covers every suspect, whatever
    their number: an empty arc is the ordinary state of a subject-only StatVar, so this read
    recovers an arc the first batch dropped. A suspect whose arc reads empty a second time
    stands as subject-only until `_settle_role_absence` reads it alone, and one that no
    longer confirms leaves the members. Return the members with each re-read one
    replaced."""
    members = list(consistent)
    if _OBSERVATION_ABOUT in role_keys or not any(
        covers_role_keys(member.declared_roles, role_keys) for member in members
    ):
        return members
    suspects = [
        member.stat_var
        for member in members
        if member.declared_roles == (_OBSERVATION_ABOUT,)
        and not covers_role_keys(member.declared_roles, role_keys)
    ]
    if not suspects:
        return members
    reprobed_by_dcid = graph.node_arcs_many(suspects, properties=properties)
    reread = {
        dcid: _confirm_absence_suspect(
            dcid, reprobed_by_dcid.get(dcid), five_tuple=five_tuple, slots=slots
        ).member
        for dcid in suspects
    }
    kept = [reread.get(member.stat_var, member) for member in members]
    return [member for member in kept if member is not None]


class _SingleReads:
    """Read arcs through `graph`, keeping each dcid's latest single-dcid read of `properties`.

    A batch read can drop an arc and a single-dcid read does not, so a kept read settles a
    decision that rests on an arc reading empty without another request."""

    def __init__(self, graph: MemberArcReader, *, properties: Sequence[str]) -> None:
        self._graph = graph
        self.properties = tuple(properties)
        self.arcs: dict[str, dict[str, tuple[str, ...]] | None] = {}

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        return _node_subjects(
            self._graph,
            dcid,
            property=property,
            filters=filters,
            max_results=max_results,
            deadline=deadline,
        )

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        read = self._graph.node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )
        if len(dcids) == 1 and tuple(properties) == self.properties:
            self.arcs[dcids[0]] = read.get(dcids[0])
        return read


def _settle_role_absence(
    confirmed: Sequence[Member],
    *,
    single_reads: _SingleReads,
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
    readings: Mapping[str, AbsentReading],
    role_keys: frozenset[str],
    open_role_keys: frozenset[str],
) -> tuple[tuple[Member, ...], tuple[str, ...]]:
    """Read alone each member whose role class rests on an empty `observationProperties` arc
    (`_role_absence_suspects`), one at a time, electing again after each read.

    A batch read can drop the arc on every read, so a member that declares a role reads
    subject-only, and the election places it in the wrong class. A member read alone
    earlier in the walk is settled from that read. Up to `_ROLE_PROBE_BUDGET` others are
    read now; one that reads no node, like a winner `_reverify_antichain` sees vanish, is
    unsettled. Once the budget is spent, every remaining suspect is unsettled. An unsettled
    member leaves the members, so the caller reports it among the candidates it could not
    dismiss. Returns the members with each read one replaced, and the unsettled dcids."""
    by_dcid = {member.stat_var: member for member in confirmed}
    settled: set[str] = set()
    unsettled: list[str] = []
    budget = _ROLE_PROBE_BUDGET
    while suspects := [
        dcid
        for dcid in _role_absence_suspects(
            tuple(by_dcid.values()),
            readings=readings,
            role_keys=role_keys,
            open_role_keys=open_role_keys,
        )
        if dcid not in settled
    ]:
        dcid = suspects[0]
        settled.add(dcid)
        if dcid in single_reads.arcs:
            arcs = single_reads.arcs[dcid]
        elif budget:
            budget -= 1
            arcs = single_reads.node_arcs_many([dcid], properties=single_reads.properties)[dcid]
        else:
            settled.update(suspects)
            unsettled.extend(suspects)
            for suspect in suspects:
                del by_dcid[suspect]
            continue
        if arcs is None:
            unsettled.append(dcid)
            del by_dcid[dcid]
            continue
        member = (
            None
            if _type_of_names_a_non_stat_var(arcs)
            else _confirm_member(
                dcid, arcs, five_tuple=five_tuple, slots=slots, trust_type_of=True
            ).member
        )
        if member is None:
            del by_dcid[dcid]
        else:
            by_dcid[dcid] = member
    return tuple(by_dcid.values()), tuple(unsettled)


def _role_absence_suspects(
    members: Sequence[Member],
    *,
    readings: Mapping[str, AbsentReading],
    role_keys: frozenset[str],
    open_role_keys: frozenset[str],
) -> list[str]:
    """Return the subject-only members whose place in the election would change if their
    `observationProperties` arc held a role, fewest constraint keys first.

    They are the suspects of the election under `role_keys` and `open_role_keys`
    (`_election_suspects`). When an open key joined, the suspects of the election under
    `role_keys` alone are suspects too, since the members that election reads decide which
    open keys join (`electing_role_keys`)."""
    election = _elect(
        members, readings=readings, role_keys=role_keys, open_role_keys=open_role_keys
    )
    suspects = _election_suspects(members, election, readings=readings)
    if election.role_keys != role_keys:
        bound = _elect(members, readings=readings, role_keys=role_keys)
        suspects += _election_suspects(members, bound, readings=readings)
        suspects = list({member.stat_var: member for member in suspects}.values())
    return [
        member.stat_var for member in sorted(suspects, key=lambda member: len(member.constraints))
    ]


def _election_suspects(
    members: Sequence[Member],
    election: RoleClassElection,
    *,
    readings: Mapping[str, AbsentReading],
) -> list[Member]:
    """The subject-only members whose place in `election`, run over `members`, would
    change if their `observationProperties` arc held a role.

    When a subject-only member does not cover the election's keys and another member does,
    coverage drops it. A member any winner's constraint keys dominate
    (`dominates_constraint_keys`) falls to that winner whatever it declares; every other
    one is a suspect. When a subject-only member covers, which needs no role key or
    `observationAbout` alone, a subject-only winner is a suspect once some member declares
    a role: declaring one would take it out of the class. A family in which every member
    reads subject-only offers the election no other class and yields no suspect, since
    reading each of its winners alone would cost one request per winner, up to
    `DEFAULT_MEMBER_CAP`."""
    role_keys = election.role_keys
    by_dcid = {member.stat_var: member for member in members}
    winners = [by_dcid[dcid] for dcid in election.members]
    subject_only = [member for member in members if member.declared_roles == (_OBSERVATION_ABOUT,)]
    if role_class_covers((_OBSERVATION_ABOUT,), role_keys):
        if len(subject_only) == len(members):
            return []
        return [member for member in winners if member.declared_roles == (_OBSERVATION_ABOUT,)]
    if any(role_class_covers(member.declared_roles, role_keys) for member in members):
        winner_keys = {frozenset(member.constraints) for member in winners}
        return [
            member
            for member in subject_only
            if not any(
                dominates_constraint_keys(keys, frozenset(member.constraints), readings=readings)
                for keys in winner_keys
            )
        ]
    return []


def _confirm_absence_suspect(
    dcid: str,
    arcs: Mapping[str, tuple[str, ...]] | None,
    *,
    five_tuple: FiveTuple,
    slots: Mapping[str, SlotBinding],
) -> _Confirmation:
    """Confirm a re-probed absence suspect. `_reverify_antichain` alone trusts `typeOf`.
    This read may still be batched and drop the arc."""
    if arcs is None:
        return _Confirmation(None, absence_rejected=True)
    return _confirm_member(dcid, arcs, five_tuple=five_tuple, slots=slots)


def _matches_optional_arc(
    arcs: Mapping[str, tuple[str, ...]], property: str, expected: str | None
) -> bool:
    """Match one value exactly. `None` represents an empty arc tuple."""
    return arcs.get(property, ()) == ((expected,) if expected else ())


def _matches_five_tuple(
    arcs: Mapping[str, tuple[str, ...]], five_tuple: FiveTuple
) -> tuple[bool, bool]:
    """Return `(matches, absence_caused)`. For unresolved-denominator asks, an empty
    denominator arc sets `absence_caused=True`; reverse filters establish value
    presence, while candidate reads establish absence. `UNRESOLVED_NODE` is a value.
    Unresolved asks match only markers, and concrete dcids reject them;
    `_ProvisionalDenominatorReader` has already written the marker for a target typed
    exactly `ProvisionalNode`."""
    if not _matches_optional_arc(arcs, "populationType", five_tuple.population_type):
        return False, False
    if not _matches_optional_arc(arcs, "measuredProperty", five_tuple.measured_property):
        return False, False
    if not _matches_optional_arc(arcs, "statType", five_tuple.stat_type):
        return False, False
    if not _matches_optional_arc(arcs, "measurementQualifier", five_tuple.measurement_qualifier):
        return False, False
    if five_tuple_arcs_match(
        arcs,
        population_type=five_tuple.population_type,
        measured_property=five_tuple.measured_property,
        stat_type=five_tuple.stat_type,
        measurement_qualifier=five_tuple.measurement_qualifier,
        measurement_denominator=five_tuple.measurement_denominator,
        denominator_unresolved=five_tuple.denominator_unresolved,
    ):
        return True, False
    absence_caused = (
        five_tuple.measurement_denominator is None
        and five_tuple.denominator_unresolved
        and not arcs.get("measurementDenominator", ())
    )
    return False, absence_caused


def constraints_resolve(declared: Collection[str], constraints: Mapping[str, str]) -> bool:
    """Whether every constraint property in `declared`, the properties a member lists in
    `constraintProperties`, carries a value in `constraints` that resolves to a node.

    The walk admits no member that declares a constraint property with no value, or whose
    value reads as `UNRESOLVED_NODE`, bound or not (`_confirm_member`). Search applies this
    same test to the members an answer lists."""
    return all(constraints.get(name, UNRESOLVED_NODE) != UNRESOLVED_NODE for name in declared)


def _binding_allows(binding: SlotBinding, value: str) -> bool:
    match binding:
        case ("unbound",):
            return True
        case ("value", ref):
            return ref == value
        case ("set", refs):
            return value in refs


def _declared_roles(stat_var: str, observation_properties: tuple[str, ...]) -> tuple[str, ...]:
    """Mirrors `materialise.roles.declared_roles`'s contract exactly
    (defined here to keep this package independent of `materialise`)."""
    if not observation_properties:
        return (_OBSERVATION_ABOUT,)
    if _OBSERVATION_ABOUT in observation_properties:
        raise DegenerateObservationProperties(
            stat_var=stat_var, observation_properties=observation_properties
        )
    return observation_properties


@dataclass(frozen=True, eq=False)
class RoleClass:
    """What `elect_role_class` reads of the ids that declare one role tuple and one set of
    constraint keys. `role_class_facts` builds one instance per pair, so instances compare
    by identity."""

    roles: tuple[str, ...]
    role_set: frozenset[str]
    keys: frozenset[str]


@dataclass(frozen=True)
class RoleClassFacts:
    """The per-id facts `elect_role_class` reads, built once (`role_class_facts`) so that
    elections over any subset of the ids read them."""

    class_of: Mapping[str, RoleClass]
    """Each id's role class."""
    twin_of: Mapping[str, tuple[frozenset[tuple[str, str]], frozenset[str]]]
    """Each id that may have a role twin, mapped to its constraint map and role set: the
    ids whose constraint map other ids hold under another role set."""
    smaller: Mapping[tuple[frozenset[tuple[str, str]], frozenset[str]], tuple[frozenset[str], ...]]
    """For each such constraint map and role set, the role sets held under that map that
    are strict subsets of it: the twins that displace it."""


def role_class_facts(
    constraints_by_id: Mapping[str, Mapping[str, str]],
    roles_by_id: Mapping[str, tuple[str, ...]],
) -> RoleClassFacts:
    """Return the facts `elect_role_class_of` reads for the ids of `roles_by_id`.

    Twin candidates group by constraint map, and each group compares its distinct role
    sets, so the work grows with the number of distinct sets, not the number of ids."""
    classes: dict[tuple[tuple[str, ...], frozenset[str]], RoleClass] = {}
    class_of: dict[str, RoleClass] = {}
    map_of: dict[str, frozenset[tuple[str, str]]] = {}
    role_sets_by_map: dict[frozenset[tuple[str, str]], set[frozenset[str]]] = {}
    for dcid, roles in roles_by_id.items():
        constraints = constraints_by_id[dcid]
        keys = frozenset(constraints)
        role_class = classes.get((roles, keys))
        if role_class is None:
            role_class = classes[roles, keys] = RoleClass(
                roles=roles, role_set=frozenset(roles), keys=keys
            )
        class_of[dcid] = role_class
        constraint_map = map_of[dcid] = frozenset(constraints.items())
        role_sets_by_map.setdefault(constraint_map, set()).add(role_class.role_set)
    smaller = {
        (constraint_map, role_set): nested
        for constraint_map, role_sets in role_sets_by_map.items()
        if len(role_sets) > 1
        for role_set in role_sets
        if (nested := tuple(other for other in role_sets if other < role_set))
    }
    twin_of = {
        dcid: (constraint_map, class_of[dcid].role_set)
        for dcid, constraint_map in map_of.items()
        if len(role_sets_by_map[constraint_map]) > 1
    }
    return RoleClassFacts(class_of=class_of, twin_of=twin_of, smaller=smaller)


def _displaced(facts: RoleClassFacts, order: Sequence[str]) -> set[str]:
    """The ids of `order` a role twin among `order` displaces: one with the same constraint
    map and a role set that is a strict subset of theirs. The id declaring the strict
    superset is a breakdown over the extra roles and the other is its total, so the
    breakdown is displaced. The result does not depend on the order given."""
    twins = [(dcid, facts.twin_of[dcid]) for dcid in order if dcid in facts.twin_of]
    present = {twin for _dcid, twin in twins}
    return {
        dcid
        for dcid, (constraint_map, role_set) in twins
        if any(
            (constraint_map, other) in present
            for other in facts.smaller.get((constraint_map, role_set), ())
        )
    }


def covers_role_keys(declared_roles: tuple[str, ...], role_keys: Collection[str]) -> bool:
    """Return whether `declared_roles` places every key of `role_keys`.

    A key is a declared role name or a flow-axis alias, resolved by
    `dfo_platform.roles.declared_role_for`, the resolver the mint's reproducibility check
    and Search's role path apply."""
    return all(declared_role_for(key, declared_roles) is not None for key in role_keys)


def electing_role_keys(
    facts: RoleClassFacts,
    order: Sequence[str],
    *,
    role_keys: Collection[str],
    open_role_keys: Collection[str],
    readings: Mapping[str, AbsentReading],
) -> frozenset[str]:
    """Return the role keys an election over the ids of `order` reads.

    `role_keys` are bound and `open_role_keys` open. An open key guides the election as a
    preference, and joins the bound keys only where the ids the bound keys alone elect
    publish a series by it: one of those ids, or an id of `order` holding the constraint
    map of one, its role twin, places it. A by-creditor twin places an open creditor and
    displaces its total. A two-role id of other constraint values is a twin of no total,
    so it places nothing for a question the totals answer, and the key is set aside.

    The keys are `role_keys` and the largest subset of `open_role_keys` that one such id
    covers together with them (`role_class_covers`). When two or more different subsets
    of that largest size are covered, no open key outranks another and the keys are
    `role_keys` alone, as they are when no such id covers any open key, so the election
    runs as it would were the role unnamed."""
    bound = frozenset(role_keys)
    open_keys = frozenset(open_role_keys) - bound
    if not open_keys:
        return bound
    placing = admitting_roles(facts, order, role_keys=bound, readings=readings)
    for size in range(len(open_keys), 0, -1):
        candidates = [bound | frozenset(subset) for subset in combinations(open_keys, size)]
        covered = [
            keys for keys in candidates if any(role_class_covers(roles, keys) for roles in placing)
        ]
        if covered:
            return covered[0] if len(covered) == 1 else bound
    return bound


def admitting_roles(
    facts: RoleClassFacts,
    order: Sequence[str],
    *,
    role_keys: Collection[str],
    readings: Mapping[str, AbsentReading],
) -> set[tuple[str, ...]]:
    """Return the role tuples of the series the ids of `order` publish for a question that
    binds `role_keys`: those of the ids `role_keys` alone elect, every id the election
    leaves when it ties, and those of the ids of `order` that hold the constraint map of
    one of them under another role set, its role twins.

    A key one of these tuples places is one the ids hold a series by, so it joins an
    election as an open key (`electing_role_keys`). Asked with no role key, they are the
    roles a dataset publishing the ids declares."""
    class_of = facts.class_of
    twin_of = facts.twin_of
    reading = _elect_under(facts, order, frozenset(role_keys), readings=readings).members
    roles = {class_of[dcid].roles for dcid in reading}
    maps = {twin_of[dcid][0] for dcid in reading if dcid in twin_of}
    if maps:
        roles.update(
            class_of[dcid].roles for dcid in order if dcid in twin_of and twin_of[dcid][0] in maps
        )
    return roles


@dataclass(frozen=True)
class RoleClassElection:
    """The role class `elect_role_class` elects and its reduction."""

    family: tuple[str, ...]
    """Ids of the elected class before antichain reduction, in the order given: the
    covering ids that declare the elected role set. Every covering id when the election
    ties."""
    members: tuple[str, ...]
    """The antichain of `family` over constraint keys, in the order given. Every id the
    constraint antichain and the role-twin collapse leave when the election ties."""
    tied: bool
    """Whether the ids the antichain and the twin collapse leave declare more than one role
    set and no one of them is elected, which the caller refuses."""
    role_keys: frozenset[str]
    """The keys the election ran under: the bound ones and the open ones that joined
    (`electing_role_keys`). An election over the same ids that binds these keys and
    leaves none open elects the same family and members."""


def elect_role_class(
    constraints_by_id: Mapping[str, Mapping[str, str]],
    roles_by_id: Mapping[str, tuple[str, ...]],
    order: Sequence[str],
    *,
    role_keys: Collection[str],
    open_role_keys: Collection[str] = (),
    readings: Mapping[str, AbsentReading],
) -> RoleClassElection:
    """Elect the role class that answers `role_keys` among matched members.

    `order` lists the ids that already match the five-tuple and every binding, with
    their constraint maps in `constraints_by_id` and declared roles in `roles_by_id`.

    `open_role_keys` join `role_keys` where the ids `role_keys` alone elect, or a role twin
    of one, place them, and are set aside otherwise (`electing_role_keys`).
    `RoleClassElection.role_keys` holds the keys the election ran under. Role coverage
    runs first: an id whose roles do not place every key (`covers_role_keys`) is dropped.
    When no id covers them, every id stays, and the caller's reproducibility check names
    the roles no member declares. An identity that binds no role key is about its
    subject, the role an entity named without one takes, so the subject-only ids cover
    it, and every id stays when none is subject-only.

    The antichain over constraint keys under `readings` then keeps the least-constrained
    ids, and role twins among them collapse to the total (`_displaced`): of two
    ids with equal constraint maps whose role sets nest, the one declaring fewer roles
    survives. A recipient-only total displaces its by-creditor twin when only the
    recipient is bound, and the twin alone covers a bound creditor.

    Of the role sets the ids left declare, the smallest is elected when every other one
    contains it and the constraint keys of each id left that declares it are among those
    of each id left that declares another: an id declaring a role the keys do not ask for
    leaves when ids with fewer roles and its own constraint keys can answer. A
    by-creditor series with no total at its own constraint values leaves for the totals
    that hold other values of the same keys, and answers once the creditor is bound. The
    family is every covering id of the elected role set and the members are the antichain
    of the family alone, since an id of a role set the election does not read dominates
    nothing. Role sets none of which sits inside all the others all survive, as do nested
    role sets whose ids hold other constraint keys: a series with fewer roles is the
    total of none that breaks down by another property. `tied` reports both."""
    return elect_role_class_of(
        role_class_facts(
            {dcid: constraints_by_id[dcid] for dcid in order},
            {dcid: roles_by_id[dcid] for dcid in order},
        ),
        order,
        role_keys=role_keys,
        open_role_keys=open_role_keys,
        readings=readings,
    )


def elect_role_class_of(
    facts: RoleClassFacts,
    order: Sequence[str],
    *,
    role_keys: Collection[str],
    open_role_keys: Collection[str] = (),
    readings: Mapping[str, AbsentReading],
) -> RoleClassElection:
    """`elect_role_class` over ids whose facts `role_class_facts` built, for a caller that
    elects over many subsets of one set of ids. Coverage, the antichain and the family
    are decided once per role class, so an election reads each id a few times, and an
    election with open keys runs twice: under the bound keys, then under the keys that
    joined."""
    return _elect_under(
        facts,
        order,
        electing_role_keys(
            facts, order, role_keys=role_keys, open_role_keys=open_role_keys, readings=readings
        ),
        readings=readings,
    )


def _elect_under(
    facts: RoleClassFacts,
    order: Sequence[str],
    role_keys: frozenset[str],
    *,
    readings: Mapping[str, AbsentReading],
) -> RoleClassElection:
    """`elect_role_class_of` under `role_keys` alone, with no open key."""
    class_of = facts.class_of
    present = {class_of[dcid] for dcid in order}
    covering = {
        role_class for role_class in present if role_class_covers(role_class.roles, role_keys)
    }
    candidates = covering or present
    covered = (
        tuple(order)
        if len(candidates) == len(present)
        else tuple(dcid for dcid in order if class_of[dcid] in candidates)
    )
    key_sets = {role_class.keys for role_class in candidates}
    minimal = {
        role_class
        for role_class in candidates
        if not any(
            dominates_constraint_keys(other, role_class.keys, readings=readings)
            for other in key_sets
        )
    }
    least_constrained = (
        covered
        if len(minimal) == len(candidates)
        else tuple(dcid for dcid in covered if class_of[dcid] in minimal)
    )
    displaced = _displaced(facts, least_constrained)
    survivors = (
        tuple(dcid for dcid in least_constrained if dcid not in displaced)
        if displaced
        else least_constrained
    )
    survivor_classes = {class_of[dcid] for dcid in survivors} if displaced else minimal
    role_sets = {role_class.role_set for role_class in survivor_classes}
    smallest = min(role_sets, key=len, default=frozenset())
    if not all(smallest <= role_set for role_set in role_sets) or not _keys_inside(
        survivor_classes, smallest
    ):
        return RoleClassElection(family=covered, members=survivors, tied=True, role_keys=role_keys)
    family_classes = {role_class for role_class in candidates if role_class.role_set == smallest}
    family = (
        covered
        if len(family_classes) == len(candidates)
        else tuple(dcid for dcid in covered if class_of[dcid] in family_classes)
    )
    if len(role_sets) <= 1:
        return RoleClassElection(family=family, members=survivors, tied=False, role_keys=role_keys)
    family_key_sets = {role_class.keys for role_class in family_classes}
    family_minimal = {
        role_class
        for role_class in family_classes
        if not any(
            dominates_constraint_keys(other, role_class.keys, readings=readings)
            for other in family_key_sets
        )
    }
    members = tuple(dcid for dcid in family if class_of[dcid] in family_minimal)
    return RoleClassElection(family=family, members=members, tied=False, role_keys=role_keys)


def _keys_inside(classes: Collection[RoleClass], role_set: frozenset[str]) -> bool:
    """Whether the constraint keys of each class of `classes` declaring `role_set` are
    among those of each class declaring another role set, the test under which
    `elect_role_class` reads the first as totals of the second."""
    own = {role_class.keys for role_class in classes if role_class.role_set == role_set}
    others = {role_class.keys for role_class in classes if role_class.role_set != role_set}
    return all(keys <= other for keys in own for other in others)


def role_class_covers(declared_roles: tuple[str, ...], role_keys: Collection[str]) -> bool:
    """`elect_role_class`'s coverage: `declared_roles` place every role key
    (`covers_role_keys`), or with no role key, they declare the subject alone."""
    if role_keys:
        return covers_role_keys(declared_roles, role_keys)
    return declared_roles == (_OBSERVATION_ABOUT,)


def _elect(
    members: Sequence[Member],
    *,
    readings: Mapping[str, AbsentReading],
    role_keys: frozenset[str],
    open_role_keys: frozenset[str] = frozenset(),
) -> RoleClassElection:
    """Run `elect_role_class` over confirmed members, keyed by StatVar."""
    by_dcid = {member.stat_var: member for member in members}
    return elect_role_class(
        {dcid: member.constraints for dcid, member in by_dcid.items()},
        {dcid: member.declared_roles for dcid, member in by_dcid.items()},
        tuple(by_dcid),
        role_keys=role_keys,
        open_role_keys=open_role_keys,
        readings=readings,
    )


def _sorted_members(members: Sequence[Member], kept: Sequence[str]) -> tuple[Member, ...]:
    """Return the members named in `kept`, sorted by NFC-normalised StatVar."""
    by_dcid = {member.stat_var: member for member in members}
    return tuple(
        sorted(
            (by_dcid[dcid] for dcid in kept),
            key=lambda member: unicodedata.normalize("NFC", member.stat_var),
        )
    )


def _minimal_antichain(
    members: Sequence[Member],
    *,
    readings: Mapping[str, AbsentReading],
    role_keys: frozenset[str] = frozenset(),
    open_role_keys: frozenset[str] = frozenset(),
) -> tuple[Member, ...]:
    """Keep the members `elect_role_class` leaves minimal under `role_keys` and
    `open_role_keys`.
    A proper subset of constraint keys dominates when each added property reads
    `aggregate`, and among survivors with equal constraint maps a proper subset of
    roles displaces its superset.
    `not_declared` properties represent distinct populations.

    An `unbound` slot selects the least-constrained member that covers it, while
    `set` expands across values. Equal key sets remain together because domination
    requires a proper subset. The shared reducer preserves order and keys by StatVar."""
    return _sorted_members(
        members,
        _elect(
            members, readings=readings, role_keys=role_keys, open_role_keys=open_role_keys
        ).members,
    )


def antichain_minimal_ids(
    keys_by_id: Mapping[str, frozenset[str]],
    order: Sequence[str],
    *,
    readings: Mapping[str, AbsentReading],
) -> tuple[str, ...]:
    """Retain ordered ids that survive domination. Keep unkeyed ids and ignore key sets
    outside `order`. Compare distinct key sets once, then apply results to matching ids."""
    ids_with_keys = frozenset(keys_by_id)
    distinct_key_sets = frozenset(
        keys_by_id[item_id] for item_id in order if item_id in ids_with_keys
    )
    dominated_key_sets = {
        keys
        for keys in distinct_key_sets
        if any(
            dominates_constraint_keys(other_keys, keys, readings=readings)
            for other_keys in distinct_key_sets
        )
    }
    return tuple(
        item_id
        for item_id in order
        if item_id not in ids_with_keys or keys_by_id[item_id] not in dominated_key_sets
    )


def dominates_constraint_keys(
    dominator_keys: frozenset[str],
    dominated_keys: frozenset[str],
    *,
    readings: Mapping[str, AbsentReading],
) -> bool:
    """Return whether `dominator_keys` is a proper subset and every missing property
    reads `aggregate`. Unspecified properties default to `aggregate`. Any
    `not_declared` property blocks domination."""
    if not dominator_keys < dominated_keys:
        return False
    return all(
        readings.get(property, "aggregate") == "aggregate"
        for property in dominated_keys - dominator_keys
    )


def _reverify_antichain(
    five_tuple: FiveTuple,
    consistent: Sequence[Member],
    *,
    slots: Mapping[str, SlotBinding],
    graph: MemberArcReader,
    properties: Sequence[str],
    readings: Mapping[str, AbsentReading],
    role_keys: frozenset[str] = frozenset(),
    open_role_keys: frozenset[str] = frozenset(),
) -> tuple[tuple[Member, ...], tuple[Member, ...], tuple[str, ...]]:
    """Re-read antichain winners' constraints, roles, and `typeOf`, then recompute the
    reduction when their data changes. Batch reads may drop these unfiltered arcs,
    so suspicious winners get a single-dcid fallback. `_REPROBE_BUDGET` caps
    fallbacks, and excess winners are reported as inconsistent.

    Re-reading only winners suffices: a dropped arc can create a wrongful winner,
    which a trusted read removes or replaces. Each round settles at least one
    dcid or leaves the antichain unchanged. A trusted missing `StatisticalVariable`
    raises `NotAStatisticalVariable`.

    Returns the antichain, every `consistent` member the re-reads kept, and the
    inconsistent dcids."""
    by_dcid = {member.stat_var: member for member in consistent}
    settled: set[str] = set()
    inconsistent: list[str] = []
    fallback_budget = _REPROBE_BUDGET
    antichain = _minimal_antichain(
        tuple(by_dcid.values()),
        readings=readings,
        role_keys=role_keys,
        open_role_keys=open_role_keys,
    )
    for _ in range(len(by_dcid) + 1):
        pending = [member for member in antichain if member.stat_var not in settled]
        if not pending:
            break
        reread_by_dcid = graph.node_arcs_many(
            [member.stat_var for member in pending], properties=properties
        )
        for member in pending:
            arcs = reread_by_dcid.get(member.stat_var)
            confirmation: _Confirmation | None = None
            needs_fallback = (
                arcs is None
                or _constraint_set_shrank(arcs, member)
                or _role_signature_changed(arcs, member)
                or _type_of_dropped(arcs)
            )
            if not needs_fallback:
                confirmation = _confirm_member(
                    member.stat_var,
                    arcs,
                    five_tuple=five_tuple,
                    slots=slots,
                )
                needs_fallback = confirmation.member is None
            if needs_fallback and len(pending) > 1:
                if fallback_budget == 0:
                    inconsistent.append(member.stat_var)
                    del by_dcid[member.stat_var]
                    continue
                fallback_budget -= 1
                arcs = graph.node_arcs_many([member.stat_var], properties=properties).get(
                    member.stat_var
                )
                confirmation = None
            if arcs is None:
                inconsistent.append(member.stat_var)
                del by_dcid[member.stat_var]
                continue
            if confirmation is None:
                confirmation = _confirm_member(
                    member.stat_var,
                    arcs,
                    five_tuple=five_tuple,
                    slots=slots,
                    trust_type_of=True,
                )
            reverified = confirmation.member
            if reverified is None:
                del by_dcid[member.stat_var]
            else:
                by_dcid[member.stat_var] = reverified
                settled.add(member.stat_var)
        antichain = _minimal_antichain(
            tuple(by_dcid.values()),
            readings=readings,
            role_keys=role_keys,
            open_role_keys=open_role_keys,
        )
    return antichain, tuple(by_dcid.values()), tuple(inconsistent)


def _constraint_set_shrank(arcs: Mapping[str, tuple[str, ...]], member: Member) -> bool:
    """Return whether the re-read omits a constraint already confirmed on `member`."""
    return not set(member.constraints) <= set(arcs.get("constraintProperties", ()))


def _role_signature_changed(arcs: Mapping[str, tuple[str, ...]], member: Member) -> bool:
    """Return whether re-derived roles disagree with `member`. A degenerate batch read
    also counts as suspicious. A trusted single-dcid read raises on that condition."""
    try:
        roles = _declared_roles(member.stat_var, arcs.get("observationProperties", ()))
    except DegenerateObservationProperties:
        return True
    return roles != member.declared_roles


def _type_of_dropped(arcs: Mapping[str, tuple[str, ...]]) -> bool:
    """Return whether `typeOf` omits `StatisticalVariable`. A batch result needs a
    trusted single-dcid read before it can exclude a winner."""
    return _STATISTICAL_VARIABLE not in arcs.get("typeOf", ())


def _type_of_names_a_non_stat_var(arcs: Mapping[str, tuple[str, ...]]) -> bool:
    """Return whether a present `typeOf` excludes `StatisticalVariable`. Missing
    `typeOf` is inconclusive because a batch read may drop the arc. A present
    value list is treated as complete by `node_arcs_many`."""
    type_of = arcs.get("typeOf", ())
    return bool(type_of) and _STATISTICAL_VARIABLE not in type_of
