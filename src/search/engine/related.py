"""Related measures: the other members of the smallest StatVarGroup a definite answer's one
StatVar belongs to, each with data at the answer's places (`ResolvedSpec.related_measures`).

The rule reads the graph alone. A group qualifies when it holds 2 to 40 members. The
smallest qualifies first, ties broken by group dcid. Its other members are kept when the
origin catalogue admits them as search candidates, their arcs resolve every constraint and
the denominator. Members sharing a five-tuple and constraint values share an identity, and
a Datasets mint of it reads every one of them, so they make one measure naming them all,
and members sharing the answer's StatVar's make none. A measure is kept when every place
holds a dated facet for its members, whose span at each place reaches into the requested
years when the answer names a range. Measures are ordered by the years their span shares
with the answer's StatVar's span at those places, then by dcids, and capped at 5.

`read_related` makes at most three serial reads: the StatVar's `memberOf` arcs when the
answer does not already hold them, one `<-memberOf` page per group capped at 41 subjects,
then the members' arcs beside their facets at the places, on a pool of two threads, each
facet read summary-only. `related_measures` mints each measure's identity from its
members' five-tuple and constraint values, carrying the answer's entities and window and dropping
its facets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from dc_graph import Facet
from dfo_platform.executor import submit_with_context
from origin_catalogue import current_catalogue
from search.engine.clients.graph import EngineGraphClient, SubjectsCoverage
from search.engine.retrieval.facts import (
    FiveTuple,
    _dcids_for,
    read_constraints,
    read_five_tuple,
    read_name,
    read_unresolved_constraints,
)
from search.engine.shape import identity_shape_for
from search.engine.spec_id import mint_identity
from search.models import (
    BaseIdentity,
    GraphRef,
    RelatedMeasure,
    RequestedWindow,
    SpecDefinition,
    TimeWindow,
)

RELATED_GROUP_MAX_MEMBERS = 40
RELATED_MEASURES_MAX = 5

type _Span = tuple[int, int]
type _IdentityKey = tuple[FiveTuple, tuple[tuple[str, str], ...]]


@dataclass(frozen=True, slots=True)
class RelatedMember:
    """One kept identity of the related group: the facts it is minted from, every member of
    the group that holds them, in dcid order, which a mint of the identity reads together,
    and the first of them, labelled, as the StatVar the measure reads."""

    dcids: tuple[str, ...]
    stat_var: GraphRef
    five_tuple: FiveTuple
    constraints: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Related:
    """The related group and its kept members. `group` is None, and `members` empty, when
    no group qualifies or none of its other members is kept."""

    group: GraphRef | None
    members: tuple[RelatedMember, ...]


NO_RELATED = Related(group=None, members=())


def read_related(
    *,
    stat_var: str,
    arcs: Mapping | None,
    places: Sequence[str],
    window: RequestedWindow | None,
    graph: EngineGraphClient,
    deadline: float,
) -> Related:
    """The related group of `stat_var` and its members with data at `places`.

    Args:
        stat_var: The answer's one StatVar.
        arcs: Its full arcs when the answer already holds them; None reads `memberOf`.
        places: The answer's places, each of which a kept member holds a facet at.
        window: The answer's requested window; a range drops members with no year in it.
        graph: The request's graph client.
        deadline: Absolute monotonic bound on every read.

    Raises:
        GraphInfraError, TimeoutError: A read failed or ran past `deadline`.
    """
    if arcs is None:
        read = graph.node_arcs_many([stat_var], properties=("memberOf",), deadline=deadline)
        groups = list((read.get(stat_var) or {}).get("memberOf", ()))
    else:
        groups = _dcids_for(dict(arcs), "memberOf")
    if not groups:
        return NO_RELATED

    pages = graph.node_subjects_many(
        groups,
        property="memberOf",
        max_results=RELATED_GROUP_MAX_MEMBERS,
        deadline=deadline,
    )
    sized = sorted(
        (len(page.dcids), group)
        for group, page in pages.items()
        if page.coverage is SubjectsCoverage.complete and len(page.dcids) >= 2
    )
    if not sized:
        return NO_RELATED
    group = sized[0][1]
    catalogue = current_catalogue()
    candidates = [
        dcid
        for dcid in pages[group].dcids
        if dcid != stat_var and catalogue.admit_candidate(dcid) is not None
    ]

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="related") as pool:
        arcs_read = submit_with_context(
            pool,
            graph.node_arcs_batch,
            [*candidates, group, *([stat_var] if arcs is None else [])],
            deadline=deadline,
        )
        # Summary facets carry each series' first and last date, all the filter reads.
        facets_read = submit_with_context(
            pool,
            graph.observation_facets_batch,
            [stat_var, *candidates],
            list(places),
            False,
            deadline=deadline,
        )
        member_arcs, facets = arcs_read.result(), facets_read.result()
    # A member's identity names its five-tuple and constraints alone, so members sharing
    # them share one identity, and a mint of it reads them all. They make one measure, and
    # one sharing the elected StatVar's makes none, since its mint reads the answer's too.
    elected_arcs = arcs if arcs is not None else member_arcs.get(stat_var)
    elected_key = _identity_key(elected_arcs) if elected_arcs is not None else None
    by_key: dict[_IdentityKey, dict[str, str]] = {}
    for dcid in candidates:
        held = member_arcs.get(dcid)
        if held is None or (key := _identity_key(held)) is None or key == elected_key:
            continue
        by_key.setdefault(key, {})[dcid] = read_name(held) or dcid

    elected_spans = _spans(facets.get(stat_var, {}), places)
    elected_span = _hull(elected_spans) if elected_spans is not None else None
    ranked: list[tuple[int, RelatedMember]] = []
    for (five_tuple, constraints), labels in by_key.items():
        dcids = sorted(labels)
        by_place = {
            place: [f for dcid in dcids for f in facets.get(dcid, {}).get(place, ())]
            for place in places
        }
        spans = _spans(by_place, places)
        # The card offers the measure at every place, so each must reach a requested year.
        if spans is None or not all(_meets(span, window) for span in spans):
            continue
        member = RelatedMember(
            dcids=tuple(dcids),
            stat_var=GraphRef(dcid=dcids[0], label=labels[dcids[0]]),
            five_tuple=five_tuple,
            constraints=dict(constraints),
        )
        ranked.append((_overlap(_hull(spans), elected_span), member))
    if not ranked:
        return NO_RELATED
    ranked.sort(key=lambda item: (-item[0], item[1].dcids))
    group_label = read_name(member_arcs.get(group) or {}) or group
    return Related(
        group=GraphRef(dcid=group, label=group_label),
        members=tuple(member for _, member in ranked[:RELATED_MEASURES_MAX]),
    )


def related_measures(related: Related, elected: BaseIdentity) -> list[RelatedMeasure]:
    """Each member of `related` as a measure a resubmit can switch to.

    A member's identity is built from its five-tuple (`identity_shape_for`) and its
    constraint values as bound what/how slots, with `elected`'s entities and requested
    window. `elected`'s facets are dropped, since a unit or source the elected measure pins
    says nothing about another measure. Its open roles are dropped too, since they name
    roles of the elected family that another member's family need not declare, so a related
    identity carries no open role.
    """
    base = elected.model_dump(exclude_none=True)
    base.pop("facets", None)
    base.pop("open_roles", None)
    measures: list[RelatedMeasure] = []
    for member in related.members:
        five_tuple = member.five_tuple
        shape = identity_shape_for(
            pop_type_dcid=five_tuple.pop_type_dcid,
            meas_prop_dcid=five_tuple.meas_prop_dcid,
            stat_type_dcid=five_tuple.stat_type_dcid,
            meas_qual_dcid=five_tuple.meas_qual_dcid,
            meas_denom_dcid=five_tuple.meas_denom_dcid,
        )
        slots = [
            {"property": prop, "binding": {"state": "value", "ref": value}}
            for prop, value in sorted(member.constraints.items())
        ]
        spec_id, identity = mint_identity({**base, "shape": shape, "slots": slots})
        measures.append(
            RelatedMeasure(
                stat_var=member.stat_var,
                definition=SpecDefinition(spec_id=spec_id, identity=identity),
            )
        )
    return measures


def _identity_key(arcs: Mapping) -> _IdentityKey | None:
    """The facts a member's identity is minted from, or None when an unresolved constraint
    or denominator leaves a value no identity can name."""
    held = dict(arcs)
    five_tuple = read_five_tuple(held)
    if five_tuple.meas_denom_unresolved or read_unresolved_constraints(held):
        return None
    return five_tuple, tuple(sorted(read_constraints(held).items()))


def _spans(by_place: Mapping[str, Sequence[Facet]], places: Sequence[str]) -> list[_Span] | None:
    """Each place's first and last year any dated facet holds there, or None when a place
    holds none."""
    spans: list[_Span] = []
    for place in places:
        held = [
            int(date[:4])
            for facet in by_place.get(place, ())
            for date in (facet.earliest_date, facet.latest_date)
            if date
        ]
        if not held:
            return None
        spans.append((min(held), max(held)))
    return spans


def _hull(spans: Sequence[_Span]) -> _Span:
    """The first and last year across `spans`."""
    return min(span[0] for span in spans), max(span[1] for span in spans)


def _meets(span: _Span, window: RequestedWindow | None) -> bool:
    """Whether `span` holds a year of `window` when it is a range; any span meets the rest."""
    if not isinstance(window, TimeWindow):
        return True
    start = window.start_year if window.start_year is not None else span[0]
    end = window.end_year if window.end_year is not None else span[1]
    return span[0] <= end and start <= span[1]


def _overlap(span: _Span, elected: _Span | None) -> int:
    """The years `span` shares with the elected StatVar's span."""
    if elected is None:
        return 0
    return max(0, min(span[1], elected[1]) - max(span[0], elected[0]) + 1)
