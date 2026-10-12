"""Confirm recall candidates and build one `ShapeDraft` per five-tuple, role class and
membership authority.

A five-tuple's subject-only members and its role-bearing members are separate shapes,
since no election orders one against the other. Members whose family the packaged code
index decides and members whose family the graph walk decides are separate shapes too
(`bound_values.member_family`), so a shape never holds members whose membership two
authorities decide, and members of two index families of one five-tuple, such as CRS and
DAC1 flows, are separate shapes.

Each shape carries confirmed member arcs and a slot taxonomy sourced from its
recall pool or the index, so bind needs no further graph reads.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from code_index import CodeIndex
from code_index.labels import index_family_label
from code_index.query import IndexFamily, row_five_tuple
from code_index.shared import get_index, get_measure_index
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.roles import DegenerateObservationProperties, declared_roles
from origin_catalogue import (
    DFO,
    DataCommons,
    SourceRef,
    current_catalogue,
)
from origin_catalogue.origins import UN_ID_PREFIX
from search.engine.clients.graph import EngineGraphClient, force_full_facets, stand_in
from search.engine.config import settings
from search.engine.degraded_reads import note_degraded_read
from search.engine.errors import GraphCapExceededError, RoleSignatureError
from search.engine.reading_coverage import ReadingCoverage, observed_years
from search.engine.retrieval.bound_values import (
    five_tuple_families,
    five_tuple_member_facts,
    member_family,
)
from search.engine.retrieval.facts import (
    FiveTuple,
    GroupRoles,
    _dcids_for,
    group_roles,
    read_constraints,
    read_declared_roles,
    read_five_tuple,
    read_name,
    read_slot_taxonomy,
    standard_shape_id_for,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.topics import is_topic_candidate, topic_leaves
from search.engine.shape import (
    STANDARD_SHAPE_LABEL,
    ShapeDraft,
    UpstreamSource,
    family_label,
    identity_shape_for,
    is_role_bearing,
    shape_draft_from,
)
from search.heading.measure_noun import denominator_noun
from statvar_members import FiveTuple as MemberFiveTuple

logger = logging.getLogger(__name__)

CONVERSION_TABLE_STAT_VARS: frozenset[str] = frozenset(
    stat_var
    for recipe in RECIPE_BODIES.values()
    for stat_var in (recipe.exchange_rate_variable, recipe.deflator_variable)
)
"""The StatVars the conversion tables serve, each recipe's rate and deflator under its
`table_provenance`. They are conversion inputs, so
`_rank_candidate_pool` drops them from the pool every recall route feeds."""

# Margin subtracted from the top real (non-topic) recall score when computing
# a topic-expansion-only member's expansion rank (see _split_and_expand_topics
# below). It keeps an expanded member ordered strictly below the best
# recalled candidate, so a tie-cluster of expanded leaves can never outrank
# the genuine candidate that set that ceiling under the
# settings().max_confirm_candidates cap sort or the bind menu's top-K cap.
_EXPANSION_RANK_EPSILON = 1e-4

# The members of one index family group an over-cap pool ranks at their score
# (`_family_spread`). One confirmed member builds the group's shape, since the index
# supplies the family's other rows (`five_tuple_member_facts`).
_FAMILY_CONFIRM_MEMBERS = 1

# An index row's derive group: its five-tuple, the properties of its index family, and
# whether it declares a role.
type _IndexGroupKey = tuple[MemberFiveTuple, frozenset[str], bool]

# A group of confirmed members: their five-tuple, whether they are role-bearing, and the
# properties of the index family they belong to, or None when the graph walk decides it.
type _GroupKey = tuple[FiveTuple, bool, frozenset[str] | None]

# The id suffix of each group class that is not its five-tuple's primary class, by
# (role-bearing, indexed). The primary class keeps the five-tuple's own slug
# (`group_shape_id`).
_GROUP_ID_SUFFIX: dict[tuple[bool, bool], str] = {
    (False, False): "_subject",
    (True, False): "_by_role",
    (False, True): "_subject_indexed",
}


def upstream_source(variables: Iterable[str]) -> UpstreamSource | None:
    """The single upstream source of a shape's `variables`, when the bound
    catalogue knows exactly one admitted (source, Data Commons) pair across
    all of them and DFO holds none of them, and `None` otherwise.

    `spec_build.build_shape_model` reads it over the members an answer serves.
    A variable DFO holds is served from DFO's facet wherever DFO has data, so
    its base rows name no source the answer comes from.
    """
    catalogue = current_catalogue()
    pairs: set[tuple[SourceRef, DataCommons]] = set()
    for variable in variables:
        rows = catalogue.variable_provenances(variable)
        if not rows or DFO in catalogue.data_commons(variable):
            return None
        pairs.update((row.source, row.data_commons) for row in rows)
    if len(pairs) != 1:
        return None
    ((source, origin),) = pairs
    return UpstreamSource(source=source, data_commons=origin)


def _is_role_bearing_candidate(observation_properties: tuple[str, ...]) -> bool:
    """Classify a candidate's roles for the cap's role-bearing/standard split.

    This runs before grouping, from the raw `observationProperties` arc. A
    degenerate arc counts as role-bearing here; `group_roles` later
    skips the member as unanswerable.
    """
    try:
        return declared_roles(observation_properties) != ("observationAbout",)
    except DegenerateObservationProperties:
        return True


@dataclasses.dataclass(frozen=True)
class _TopicExpansion:
    """Non-Topic recall candidates and BFS-expanded members with ranking metadata."""

    non_topic_svs: list[str]
    recall_svs: set[str]
    expanded_svs: list[str]
    expansion_rank: dict[str, float]
    # leaf -> the recall score of the topic root that reached it, uncapped.
    expansion_root_score: dict[str, float]


def _split_and_expand_topics(
    pool: list[str],
    *,
    sv_scores: dict[str, float],
    sv_types: dict[str, str | None],
    graph: EngineGraphClient,
) -> _TopicExpansion:
    """Expand strong Topic candidates into member StatVars before confirmation.

    Topics lack the five-tuple arcs confirmation needs. Some, such as dev-finance
    Scheme x Sector rollups, fan out to 1,700 to 5,200+ StatVars; mixing them into a
    `node_arcs_batch` with leaf StatVars can raise latency from ~200 ms to 3-5+ s.
    Excluding Topics from that read also frees confirm-cap slots.

    Expand roots within `topic_expand_margin` of the best pool score, up to
    `topic_expand_max_roots`; `topic_leaves` limits each root's size. The pool
    score includes Topics because the answer-bearing Topic can rank below other
    Topics and the best non-Topic candidate.
    """
    topic_svs = [sv for sv in pool if is_topic_candidate(sv, sv_types.get(sv))]
    non_topic_svs = [sv for sv in pool if not is_topic_candidate(sv, sv_types.get(sv))]

    expanded_svs: list[str] = []
    # leaf -> the topic root that reached it, and that leaf's own ordering
    # rank. A member reached by two roots keeps the first (strongest, roots
    # are score-descending) root's rank.
    expansion_rank: dict[str, float] = {}
    expansion_root_score: dict[str, float] = {}
    if topic_svs:
        pool_scores = [sv_scores.get(sv, 1.0) for sv in (*topic_svs, *non_topic_svs)]
        top_pool_score = max(pool_scores) if pool_scores else 1.0
        qualifying_roots = sorted(
            (
                sv
                for sv in topic_svs
                if sv_scores.get(sv, 1.0) >= top_pool_score - settings().topic_expand_margin
            ),
            key=lambda sv: sv_scores.get(sv, 1.0),
            reverse=True,
        )[: settings().topic_expand_max_roots]
        if qualifying_roots:
            leaf_to_root = topic_leaves(qualifying_roots, graph=graph)
            expanded_svs = list(leaf_to_root)
            # Expanded members have no measured recall score. Keep them out of
            # `sv_scores`; `expansion_rank` orders them without affecting confidence.
            # Cap each root below the best real score, stepping down by epsilon per
            # root to preserve order. This stops a topic self-match (GDP at 1.0)
            # displacing a scored StatVar (0.9084); use raw root scores when no
            # non-Topic candidate exists.
            non_topic_scores = [sv_scores.get(sv, 1.0) for sv in non_topic_svs]
            top_real_score = max(non_topic_scores) if non_topic_scores else None
            root_rank: dict[str, float] = {}
            root_scores: dict[str, float] = {}
            for position, root in enumerate(qualifying_roots):
                root_score = root_scores[root] = sv_scores.get(root, top_pool_score)
                if top_real_score is None:
                    root_rank[root] = root_score
                else:
                    ceiling = top_real_score - _EXPANSION_RANK_EPSILON * (position + 1)
                    root_rank[root] = min(root_score, ceiling)
            for sv, root in leaf_to_root.items():
                expansion_rank.setdefault(sv, root_rank[root])
                expansion_root_score.setdefault(sv, root_scores[root])

    return _TopicExpansion(
        non_topic_svs=non_topic_svs,
        recall_svs=set(non_topic_svs),
        expanded_svs=expanded_svs,
        expansion_rank=expansion_rank,
        expansion_root_score=expansion_root_score,
    )


def _rank_candidate_pool(
    non_topic_svs: list[str],
    expanded_svs: list[str],
    *,
    sv_scores: dict[str, float],
    expansion_rank: dict[str, float],
) -> list[str]:
    """De-duplicate and rank by recall score or, for expanded members, root rank.

    Conversion-table StatVars (`CONVERSION_TABLE_STAT_VARS`) leave the pool here.
    Stable sorting preserves expansion order for ties and unranked members, so
    ranked expansions survive alongside a wide recall pool.
    """
    pool_rank = {
        sv: (sv_scores[sv] if sv in sv_scores else expansion_rank.get(sv))
        for sv in dict.fromkeys((*non_topic_svs, *expanded_svs))
        if sv not in CONVERSION_TABLE_STAT_VARS
    }
    return sorted(
        pool_rank,
        key=lambda sv: (pool_rank[sv] is None, -(pool_rank[sv] or 0.0)),
    )


def _select_capped_candidates(
    candidate_pool: list[str],
    *,
    sv_scores: dict[str, float],
    recall_svs: set[str],
    graph: EngineGraphClient,
) -> set[str]:
    """Select candidates within `max_confirm_candidates`.

    If the pool exceeds the cap, reserve the top `min_recall_confirm_slots`
    recalled candidates, then up to `min_shape_class_confirm_slots` scored
    members from each role-bearing and standard class. Class reserves require
    scores at or above `unknown_score_threshold`; they use only remaining cap
    space. Fill any remaining slots by pool rank.
    Topic expansions have no score and receive no class reserve.
    """
    pool_exceeds_cap = len(candidate_pool) > settings().max_confirm_candidates
    role_bearing_by_sv: dict[str, bool] = {}
    if pool_exceeds_cap:
        # Classify only scored members above threshold; the roles-only batch is
        # cheaper than the full-arc confirm read, and Topics are already excluded.
        viable_pool = [
            sv
            for sv in candidate_pool
            if sv in sv_scores and sv_scores[sv] >= settings().unknown_score_threshold
        ]
        roles_by_dcid = graph.node_observation_properties_batch(viable_pool)
        for sv in viable_pool:
            role_bearing_by_sv[sv] = _is_role_bearing_candidate(roles_by_dcid[sv])

    # Filter the ranked pool to preserve the insertion order used by downstream probes.
    reserve = min(settings().min_recall_confirm_slots, settings().max_confirm_candidates)
    reserved: set[str] = set()
    for sv in candidate_pool:
        if len(reserved) >= reserve:
            break
        if sv in recall_svs:
            reserved.add(sv)

    if pool_exceeds_cap:
        class_floor = min(
            settings().min_shape_class_confirm_slots, settings().max_confirm_candidates
        )
        for want_role_bearing in (True, False):
            room = settings().max_confirm_candidates - len(reserved)
            if room <= 0:
                break
            class_members = [
                sv
                for sv in candidate_pool
                if sv in sv_scores
                and sv_scores[sv] >= settings().unknown_score_threshold
                and role_bearing_by_sv[sv] is want_role_bearing
            ]
            new_members = [sv for sv in class_members[:class_floor] if sv not in reserved]
            reserved.update(new_members[:room])

    selected = set(reserved)
    for sv in candidate_pool:
        if len(selected) >= settings().max_confirm_candidates:
            break
        selected.add(sv)

    if pool_exceeds_cap:
        naive_top = set(candidate_pool[: settings().max_confirm_candidates])
        claimed = reserved - naive_top
        if claimed:
            displaced = naive_top - selected
            logger.debug(
                "derive_shapes: capped %d candidates to %d (settings().max_confirm_candidates); "
                "reserve claimed %d slot(s), displacing %d rank-eligible candidate(s)",
                len(candidate_pool),
                settings().max_confirm_candidates,
                len(claimed),
                len(displaced),
            )
        else:
            logger.debug(
                "derive_shapes: capped %d candidates to %d (settings().max_confirm_candidates)",
                len(candidate_pool),
                settings().max_confirm_candidates,
            )

    return selected


@functools.lru_cache(maxsize=1)
def _index_group_keys(index: CodeIndex) -> dict[str, _IndexGroupKey]:
    """The derive group of each row of `index`, by StatVar dcid: the row's five-tuple, the
    properties of the family it belongs to (`CodeIndex.family_by_dcid`, as
    `bound_values.member_family` reads it), and whether it declares a role
    (`_is_role_bearing_candidate`). These are the parts of `_GroupKey` that `derive_shapes`
    reads from a confirmed member's arcs."""
    return {
        row.dcid: (
            row_five_tuple(row),
            index.family_by_dcid[row.dcid].properties,
            _is_role_bearing_candidate(row.observation_properties),
        )
        for row in index.statvars
    }


def _family_spread(candidate_pool: list[str]) -> list[str]:
    """Return `candidate_pool` with the members of each index family group beyond its first
    `_FAMILY_CONFIRM_MEMBERS` moved after every other candidate, rank order kept within both
    parts.

    A shape the index carries reads its slot keys and taxonomy from every row of its family
    (`five_tuple_member_facts`), so its further members would only spend confirm slots that
    other shapes need. StatVars the index does not decide keep their rank."""
    group_keys = _index_group_keys(get_index())
    taken: dict[_IndexGroupKey, int] = defaultdict(int)
    leading: list[str] = []
    surplus: list[str] = []
    for sv in candidate_pool:
        key = group_keys.get(sv)
        if key is not None and taken[key] >= _FAMILY_CONFIRM_MEMBERS:
            surplus.append(sv)
            continue
        if key is not None:
            taken[key] += 1
        leading.append(sv)
    return leading + surplus


def _stand_selection(
    candidate_pool: list[str],
    selected: set[str],
    *,
    sv_scores: dict[str, float],
    recall_svs: set[str],
    expansion_rank: dict[str, float],
    expansion_root_score: dict[str, float],
) -> tuple[list[str], set[str]]:
    """The ranked pool and the selection as the variables they stand as (`stand_in`), in the
    pool's order. The score, expansion rank and root score maps and `recall_svs` gain each
    successor under the best value of the variables standing as it, in place."""
    catalogue = current_catalogue()
    standing = {sv: stand_in(sv, catalogue=catalogue) for sv in candidate_pool}
    if all(standing[sv] == sv for sv in candidate_pool):
        return candidate_pool, selected
    for sv in candidate_pool:
        dcid = standing[sv]
        if dcid is None or dcid == sv:
            continue
        for values in (sv_scores, expansion_rank, expansion_root_score):
            if sv in values and values[sv] > values.get(dcid, float("-inf")):
                values[dcid] = values[sv]
        if sv in recall_svs:
            recall_svs.add(dcid)
    pool = list(dict.fromkeys(d for sv in candidate_pool if (d := standing[sv]) is not None))
    return pool, {d for sv in selected if (d := standing[sv]) is not None}


def _confirm_with_backfill(
    candidate_pool: list[str],
    selected: set[str],
    *,
    graph: EngineGraphClient,
) -> tuple[list[str], dict[str, dict | None]]:
    """Read full arcs for selected candidates, with one backfill batch for missing results.

    Fill each lost slot from the next unselected candidates in pool order.
    Backfill runs once; missing arcs in that batch leave slots empty. Mutates
    `selected` and returns confirmed candidates in pool order with their arcs.
    """
    candidates = [sv for sv in candidate_pool if sv in selected]
    arcs_by_dcid = graph.node_arcs_batch(candidates)

    dropped = [sv for sv in candidates if arcs_by_dcid.get(sv) is None]
    if dropped:
        unselected_pool = [sv for sv in candidate_pool if sv not in selected]
        backfill_pool = unselected_pool[: len(dropped)]
        if backfill_pool:
            backfill_arcs = graph.node_arcs_batch(backfill_pool)
            for sv in backfill_pool:
                arcs = backfill_arcs.get(sv)
                if arcs is not None:
                    selected.add(sv)
                    arcs_by_dcid[sv] = arcs
            candidates = [sv for sv in candidate_pool if sv in selected]

    return candidates, arcs_by_dcid


def _build_sv_facts(candidates: list[str], arcs_by_dcid: dict[str, dict | None]) -> dict[str, dict]:
    """Read confirmed arcs, five-tuples, and constraints.

    Five-tuples are partial; WHO GHED StatVars omit `measuredProperty`.
    Missing tuple members do not drop an SV. Materialisation's observation
    probe decides answerability.
    """
    sv_facts: dict[str, dict] = {}
    for sv_dcid in candidates:
        arcs = arcs_by_dcid.get(sv_dcid)
        if arcs is None:
            logger.debug("derive_shapes: dropped unconfirmable sv %r", sv_dcid)
            continue
        sv_facts[sv_dcid] = {
            "five_tuple": read_five_tuple(arcs),
            "constraints": read_constraints(arcs),
            "arcs": arcs,
        }
    return sv_facts


@dataclass(frozen=True)
class _PlaceProbe:
    """What the availability probe found for the confirmed candidates.

    `units` maps each dcid with observations at a probed place to the units of the facets
    holding them (`ShapeDraft.place_units`). `coverage` maps every probed dcid to the
    places where it has observations and the years its facets there hold
    (`ShapeDraft.member_coverage`).
    """

    units: dict[str, frozenset[str | None]]
    coverage: dict[str, ReadingCoverage]


def _probe_place_availability(
    sv_facts: dict[str, dict],
    *,
    place_dcids: list[str] | None,
    full_facets_max_candidates: int | None,
    graph: EngineGraphClient,
) -> _PlaceProbe | None:
    """Probe all confirmed candidates for observations at `place_dcids` in one batch.

    Return `None` when no place is known or the optional probe fails; otherwise
    each dcid's units and per-place year spans (`_PlaceProbe`). A dated request can widen
    this read under `force_full_facets(True)`. Recheck the confirmed count against
    `full_facets_max_candidates` because Topic expansion follows the caller's
    raw-count check; above the bound, `force_full_facets(False)` keeps this
    probe from fetching full history. Within the bound, the widened read shares
    the dated observation request's cache key. Graph errors fail open, and a failed probe
    is noted as a lost read (`degraded_reads.note_degraded_read`).
    """
    if not place_dcids:
        return None

    narrow_scope: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    if full_facets_max_candidates is not None and len(sv_facts) > full_facets_max_candidates:
        logger.info(
            "derive_shapes: %d confirmed candidates exceeds the %d-candidate "
            "full-facets bound; re-narrowing the availability floor probe to "
            "needs_dates=False",
            len(sv_facts),
            full_facets_max_candidates,
        )
        narrow_scope = force_full_facets(False)
    try:
        with narrow_scope:
            probe = graph.observation_facets_batch(
                list(sv_facts.keys()), list(dict.fromkeys(place_dcids)), False
            )
    except GraphCapExceededError as exc:
        logger.warning(
            "derive_shapes: place-availability probe for %d candidates against "
            "place_dcids=%r hit its %s cap (cap=%d, observed=%d); skipping the "
            "availability floor",
            len(sv_facts),
            place_dcids,
            exc.cap_kind,
            exc.cap,
            exc.observed,
        )
        return None
    except Exception:
        logger.warning(
            "derive_shapes: place-availability probe failed for %d candidates "
            "against place_dcids=%r; skipping the availability floor",
            len(sv_facts),
            place_dcids,
            exc_info=True,
        )
        note_degraded_read("place-availability probe")
        return None

    held = {
        sv: {
            place: held_facets
            for place, facets in by_entity.items()
            if (held_facets := [f for f in facets if f.obs_count > 0])
        }
        for sv, by_entity in probe.items()
    }
    return _PlaceProbe(
        units={
            sv: frozenset(f.unit for facets in by_place.values() for f in facets)
            for sv, by_place in held.items()
            if by_place
        },
        coverage={
            sv: {
                place: years
                for place, facets in by_place.items()
                if (years := observed_years(facets)) is not None
            }
            for sv, by_place in held.items()
        },
    )


def _fetch_constraint_labels(
    sv_facts: dict[str, dict], *, graph: EngineGraphClient
) -> tuple[dict[str, str], dict[str, str]]:
    """Fetch property, value and population-type labels in one graph call.

    Return raw graph labels and value labels. For indexed values, use the
    code-index label without its code-system suffix so ordinary query wording
    still matches (for example, `standard grant`). A UN System family is
    labelled by its population type (`_build_group_shape`), and every shape
    carries its population type's name (`ShapeDraft.pop_type_name`).
    """
    all_constraint_props = list({p for facts in sv_facts.values() for p in facts["constraints"]})
    all_constraint_values = list(
        {v for facts in sv_facts.values() for v in facts["constraints"].values()}
    )
    population_types = list(
        {
            facts["five_tuple"].pop_type_dcid
            for facts in sv_facts.values()
            if facts["five_tuple"].pop_type_dcid
        }
    )
    labels_raw = graph.node_labels_batch(
        [*all_constraint_props, *all_constraint_values, *population_types]
    )
    prop_labels_raw = labels_raw

    code_index = get_index()
    value_labels = dict(labels_raw)
    for value_dcid in all_constraint_values:
        row = code_index.code_by_dcid(value_dcid)
        if row is not None:
            value_labels[value_dcid] = row.label

    return prop_labels_raw, value_labels


def _shape_draft_for_group(
    *,
    shape_id: str,
    shape_label: str,
    pop_type_name: str,
    five_tuple: FiveTuple,
    group_arc_facts: dict[str, dict],
    constraint_props: list[str],
    prop_labels: dict[str, str],
    prop_observed_values: dict[str, list[str]],
    rep_score: float | None,
    expansion_rank_score: float | None,
    expansion_root_score: float | None,
    place_available: bool | None,
    place_units: frozenset[str | None] | None,
    roles: GroupRoles,
    group_recalled: bool,
    value_labels: dict[str, str],
) -> ShapeDraft:
    """Build a ShapeDraft from confirmed group facts.

    Index-sourced taxonomies call this before and after merging properties;
    each call must receive its current properties and labels.
    """
    return shape_draft_from(
        shape_id=shape_id,
        label=shape_label,
        pop_type_dcid=five_tuple.pop_type_dcid,
        pop_type_name=pop_type_name,
        meas_prop_dcid=five_tuple.meas_prop_dcid,
        stat_type_dcid=five_tuple.stat_type_dcid,
        meas_qual_dcid=five_tuple.meas_qual_dcid,
        meas_denom_dcid=five_tuple.meas_denom_dcid,
        meas_denom_unresolved=five_tuple.meas_denom_unresolved,
        constraint_props=constraint_props,
        prop_labels=prop_labels,
        prop_observed_values=dict(prop_observed_values),
        sv_arc_facts=group_arc_facts,
        # This is the settings().max_confirm_candidates-bounded confirm
        # fetch's own group, a recall sample by construction, regardless
        # of how many members it actually returned. See
        # ShapeDraft.sv_arc_facts_complete.
        sv_arc_facts_complete=False,
        representative_score=rep_score,
        expansion_rank_score=expansion_rank_score,
        expansion_root_score=expansion_root_score,
        place_available=place_available,
        place_units=place_units,
        value_labels=value_labels,
        declared_roles=roles.declared,
        optional_roles=roles.optional,
        recalled=group_recalled,
    )


def member_group_class(
    dcid: str, five_tuple: FiveTuple, arcs: dict
) -> tuple[bool, IndexFamily | None]:
    """Return member `dcid`'s group class: whether it declares a role, and the packaged code
    index family it belongs to (`bound_values.member_family`), or None when the graph walk
    decides its family.

    Derive groups members by five-tuple and class, and a resubmit reads the class of the
    members it re-reads to rebuild their shape's id (`group_shape_id`)."""
    return (
        _is_role_bearing_candidate(tuple(_dcids_for(arcs, "observationProperties"))),
        member_family(dcid, five_tuple, _dcids_for(arcs, "constraintProperties")),
    )


def group_shape_id(five_tuple: FiveTuple, *, role_bearing: bool, family: IndexFamily | None) -> str:
    """Return the shape id of a group of members of `five_tuple`.

    The id reads the group alone, so it is the same whatever else a derivation or a
    resubmit holds. The five-tuple's primary class keeps its slug: the role-bearing members
    of its first index family (CRS flows) when the packaged code index carries the
    five-tuple, and the subject-only members otherwise. Every other class takes its suffix
    (`_GROUP_ID_SUFFIX`), and the members of any other index family of the five-tuple
    (DAC1 flows beside CRS flows) take a tag of that family's properties too, so the groups
    a five-tuple splits into never share an id."""
    slug = standard_shape_id_for(five_tuple)
    families = five_tuple_families(five_tuple)
    primary_family = families[0].properties if families else None
    properties = family.properties if family is not None else None
    if families and role_bearing and properties == primary_family:
        return slug
    if not families and not role_bearing:
        return slug
    suffix = _GROUP_ID_SUFFIX.get((role_bearing, family is not None), "")
    if properties is not None and properties != primary_family:
        tag = hashlib.sha256("\n".join(sorted(properties)).encode()).hexdigest()[:8]
        suffix = f"{suffix}_family_{tag}"
    return f"{slug}{suffix}"


def index_shape_label(family: IndexFamily, *, population_type_label: str) -> str:
    """Return the label of a shape whose membership index `family` decides
    (`index_family_label`). Derive and a resubmit label such a shape with this."""
    return index_family_label(
        family,
        index=get_index(),
        measure_index=get_measure_index(),
        denominator_noun=denominator_noun(family.five_tuple.measurement_denominator),
        population_type_label=population_type_label,
    )


def _build_group_shape(
    five_tuple: FiveTuple,
    group_svs: list[str],
    *,
    shape_id: str,
    family: IndexFamily | None,
    sv_facts: dict[str, dict],
    sv_scores: dict[str, float],
    recall_svs: set[str],
    expansion_rank: dict[str, float],
    expansion_root_score: dict[str, float],
    place_probe: _PlaceProbe | None,
    prop_labels_raw: dict[str, str],
    value_labels: dict[str, str],
) -> tuple[ShapeDraft | None, bool, bool]:
    """Build one ShapeDraft for a confirmed group: the subject-only or the role-bearing
    members of one five-tuple that one authority decides the membership of.

    Return `(None, unusable_roles, zero_member)` when no single axis mapping
    covers the members' roles or the tuple has no schema member; otherwise
    return the shape with both flags false. Members that declare different
    roles on distinct axes share one shape carrying the union of their roles.
    `family` is the index family that decides the group's membership, or None when the
    graph walk decides it.
    """
    # One confirmed arc map feeds role validation, stamping, and downstream representatives.
    group_arc_facts = {sv: sv_facts[sv]["arcs"] for sv in group_svs}

    # Without a schema field, `identity_shape_for` yields `{"kind": "statvar_schema"}`,
    # which identity validation rejects.
    identity_member = identity_shape_for(
        pop_type_dcid=five_tuple.pop_type_dcid,
        meas_prop_dcid=five_tuple.meas_prop_dcid,
        stat_type_dcid=five_tuple.stat_type_dcid,
        meas_qual_dcid=five_tuple.meas_qual_dcid,
        meas_denom_dcid=five_tuple.meas_denom_dcid,
        meas_denom_unresolved=five_tuple.meas_denom_unresolved,
    )
    if identity_member.keys() == {"kind"}:
        logger.debug(
            "derive_shapes: skipped zero-member shape group covering %d confirmed sv(s): %r",
            len(group_svs),
            group_svs,
        )
        return None, False, True

    # Members must share one axis mapping. Mismatches are quiet; a degenerate arc
    # is a graph integrity fault, so log the offending members before skipping.
    try:
        roles = group_roles(group_arc_facts)
    except RoleSignatureError:
        return None, True, False
    except DegenerateObservationProperties:
        for _sv, _arcs in group_arc_facts.items():
            try:
                read_declared_roles(_arcs)
            except DegenerateObservationProperties:
                logger.error(
                    "derive_shapes: %r declares a degenerate observationProperties "
                    "arc (non-emptily names observationAbout): %r; group skipped "
                    "rather than answered under a corrupt role signature",
                    _sv,
                    _dcids_for(_arcs, "observationProperties"),
                )
        return None, True, False

    # Probe availability narrows taxonomy values only; slot keys still come
    # from the full group. An empty or unknown result keeps the full taxonomy.
    place_units: frozenset[str | None] | None = None
    if place_probe is not None:
        units_by_available_sv = place_probe.units
        group_available = {sv for sv in group_svs if sv in units_by_available_sv}
        scoped_svs = group_available or set(group_svs)
        place_available: bool | None = bool(group_available)
        if group_available:
            place_units = frozenset().union(*(units_by_available_sv[sv] for sv in group_available))
    else:
        scoped_svs = set(group_svs)
        # Unknown means the probe did not establish that the place lacks data.
        place_available = None

    # Slot keys use every confirmed member, independent of place availability;
    # otherwise unavailable members could hide a property that should be unbound.
    prop_observed_values: dict[str, list[str]] = defaultdict(list)
    for sv_dcid in group_svs:
        facts = sv_facts[sv_dcid]
        for prop_dcid, value_dcid in facts["constraints"].items():
            prop_observed_values[prop_dcid].append(value_dcid)

    constraint_props = list(prop_observed_values.keys())
    prop_labels: dict[str, str] = {p: prop_labels_raw.get(p, p) for p in constraint_props}

    # `group_svs` is score-descending. Use its first scored member for rank and
    # its highest-scoring recalled member for the label, falling back to that representative.
    rep_sv_dcid = next((sv for sv in group_svs if sv in sv_scores), group_svs[0])
    label_sv_dcid = next((sv for sv in group_svs if sv in recall_svs), rep_sv_dcid)
    pop_type_name = prop_labels_raw.get(five_tuple.pop_type_dcid, five_tuple.pop_type_dcid)
    # A member's name spells out its own constraint values, which an unbound slot spans, so
    # an index family reads as what all its rows share and a multi-member UN family as its
    # population type.
    population_type_label = (
        prop_labels_raw.get(five_tuple.pop_type_dcid)
        if len(group_svs) > 1 and all(sv.startswith(UN_ID_PREFIX) for sv in group_svs)
        else None
    )
    if family is not None:
        shape_label = index_shape_label(family, population_type_label=pop_type_name)
    elif population_type_label is not None:
        shape_label = family_label(
            population_type_label=population_type_label,
            member_constraints=[sv_facts[sv]["constraints"] for sv in group_svs],
            prop_labels=prop_labels,
            value_labels=value_labels,
        )
    else:
        # Prefer the recalled member's name, then other members, then the fallback
        # label; all arcs are already fetched.
        shape_label = next(
            (
                name
                for sv in (label_sv_dcid, *(s for s in group_svs if s != label_sv_dcid))
                if (name := read_name(group_arc_facts[sv])) is not None
            ),
            STANDARD_SHAPE_LABEL,
        )
    # Topic-only groups have no measured recall score.
    rep_score = sv_scores.get(rep_sv_dcid)
    # Ordering only for unscored groups; it never substitutes for confidence.
    expansion_rank_score = (
        None
        if rep_score is not None
        else max(
            (expansion_rank[sv] for sv in group_svs if sv in expansion_rank),
            default=None,
        )
    )
    # The measured evidence an unscored group meets the score floor on.
    group_root_score = (
        None
        if rep_score is not None
        else max(
            (expansion_root_score[sv] for sv in group_svs if sv in expansion_root_score),
            default=None,
        )
    )
    # True for non-Topic recall members, including backfill; false for expansion-only groups.
    group_recalled = any(sv in recall_svs for sv in group_svs)

    shape = _shape_draft_for_group(
        shape_id=shape_id,
        shape_label=shape_label,
        pop_type_name=pop_type_name,
        five_tuple=five_tuple,
        group_arc_facts=group_arc_facts,
        constraint_props=constraint_props,
        prop_labels=prop_labels,
        prop_observed_values=prop_observed_values,
        rep_score=rep_score,
        expansion_rank_score=expansion_rank_score,
        expansion_root_score=group_root_score,
        place_available=place_available,
        place_units=place_units,
        roles=roles,
        group_recalled=group_recalled,
        value_labels=value_labels,
    )

    # Read every index row of the five-tuple, across datasets, so the slot keys
    # stay those of the whole five-tuple; index options are unscoped because the
    # code index carries no place data.
    #
    # Starved-axis repair cannot match index-only values against sampled labels.
    member_facts, index_sourced = five_tuple_member_facts(shape)
    if not index_sourced:
        member_facts = {
            sv: ArcsMemberFacts(arcs=arcs)
            for sv, arcs in group_arc_facts.items()
            if sv in scoped_svs
        }
    else:
        # The index may add properties missed by the confirm sample; append them
        # in index row order.
        index_prop_values: dict[str, list[str]] = defaultdict(list)
        for member in member_facts.values():
            for prop_dcid, value_dcid in member.read_constraints().items():
                if value_dcid not in index_prop_values[prop_dcid]:
                    index_prop_values[prop_dcid].append(value_dcid)
        new_props = [p for p in index_prop_values if p not in prop_observed_values]
        if new_props:
            for prop_dcid in new_props:
                prop_observed_values[prop_dcid] = index_prop_values[prop_dcid]
            constraint_props = constraint_props + new_props
            prop_labels = {p: prop_labels_raw.get(p, p) for p in constraint_props}
            shape = _shape_draft_for_group(
                shape_id=shape_id,
                shape_label=shape_label,
                pop_type_name=pop_type_name,
                five_tuple=five_tuple,
                group_arc_facts=group_arc_facts,
                constraint_props=constraint_props,
                prop_labels=prop_labels,
                prop_observed_values=prop_observed_values,
                rep_score=rep_score,
                expansion_rank_score=expansion_rank_score,
                expansion_root_score=group_root_score,
                place_available=place_available,
                place_units=place_units,
                roles=roles,
                group_recalled=group_recalled,
                value_labels=value_labels,
            )

    taxonomy = read_slot_taxonomy(slot_keys=shape.slot_keys, member_facts=member_facts)
    # The probe reads subject facets, so a role-bearing member's coverage is unknown.
    member_coverage = (
        {sv: place_probe.coverage.get(sv, {}) for sv in group_svs}
        if place_probe is not None and not is_role_bearing(shape)
        else None
    )
    shape = dataclasses.replace(
        shape,
        slot_taxonomy=taxonomy,
        slot_taxonomy_index_sourced=index_sourced,
        member_coverage=member_coverage,
    )
    return shape, False, False


def derive_shapes(
    *,
    confirmed_svs: list[str],
    graph: EngineGraphClient,
    sv_scores: dict[str, float] | None = None,
    sv_types: dict[str, str | None] | None = None,
    rival_svs: frozenset[str] = frozenset(),
    place_dcids: list[str] | None = None,
    full_facets_max_candidates: int | None = None,
) -> list[ShapeDraft]:
    """Confirm recall candidates and build one shape per five-tuple and role class.

    Topic roots expand before capped full-arc confirmation. The cap reserves
    recalled candidates and both role classes; one backfill batch replaces
    unconfirmable members. Over the cap, each index family group confirms its top
    member before any of its others (`_family_spread`). Index-backed groups add
    properties beyond the sample.

    `confirmed_svs` and the Topic leaves (`topic_leaves`) enter under their own dcids,
    superseded ones included, so ranking and the confirm cap count each base variable once,
    as they did while its provenance was admitted. Only then does each selected variable
    stand as what Search reads (`clients.graph.stand_in`): a superseded one as its successor,
    read once however many selected variables stand as it, and one with no successor has
    none of its arcs or observations read. Its slot is spent either way, so supersession brings no candidate past the
    cap that would have stopped short of it. A successor takes the best score, expansion
    rank and recall standing of the variables standing as it.

    `rival_svs` are the rows recall added past its cut (`clients.graph.with_rival_families`).
    They are confirmed past the cap, which counts the other candidates alone, so a rival
    takes no candidate's slot, and a shape whose members are all rivals is marked
    `ShapeDraft.rival`.

    `sv_scores` supplies measured recall scores; expansion-only groups retain
    no score and carry a separate ordering rank. `sv_types` identifies Topic
    roots. `place_dcids` scopes taxonomy values, while slot keys use every group
    member. `full_facets_max_candidates` is re-checked against the confirmed count, which topic
    expansion can inflate past the caller's raw-count check; above it the probe narrows to
    `needs_dates=False`. `None` leaves the ambient facet scope unchanged.

    Return one shape per group: the members of one five-tuple that are subject-only or
    role-bearing, and whose family the packaged code index decides or does not
    (`bound_values.member_family`). A role-bearing shape declares every role its
    members declare (`facts.group_roles`), and marks optional the roles some member does
    not declare. Its label, slot keys and family all come from that one group.
    Groups without schema fields are skipped. A group whose roles put two names on one
    flow axis, or hold a degenerate arc, is skipped; raise `RoleSignatureError` only when
    every group with a schema field has unusable roles. A group with no place-available
    member, or a failed availability probe, keeps its full taxonomy.
    """
    # Do not mutate a score map the caller may reuse across variables.
    sv_scores = dict(sv_scores) if sv_scores else {}
    sv_types = sv_types or {}

    topic_expansion = _split_and_expand_topics(
        confirmed_svs, sv_scores=sv_scores, sv_types=sv_types, graph=graph
    )
    recall_svs = topic_expansion.recall_svs
    expansion_rank = topic_expansion.expansion_rank

    candidate_pool = _rank_candidate_pool(
        topic_expansion.non_topic_svs,
        topic_expansion.expanded_svs,
        sv_scores=sv_scores,
        expansion_rank=expansion_rank,
    )

    # Over the cap, an index family group's surplus members rank last, for the capped
    # selection and its backfill alike. Rivals sit outside the cap.
    capped_pool = [sv for sv in candidate_pool if sv not in rival_svs]
    if len(capped_pool) > settings().max_confirm_candidates:
        candidate_pool = _family_spread(candidate_pool)
        capped_pool = [sv for sv in candidate_pool if sv not in rival_svs]

    selected = _select_capped_candidates(
        capped_pool, sv_scores=sv_scores, recall_svs=recall_svs, graph=graph
    )
    selected.update(sv for sv in candidate_pool if sv in rival_svs)
    candidate_pool, selected = _stand_selection(
        candidate_pool,
        selected,
        sv_scores=sv_scores,
        recall_svs=recall_svs,
        expansion_rank=expansion_rank,
        expansion_root_score=topic_expansion.expansion_root_score,
    )

    # Full arcs supply five-tuples, constraints, and member facts; the roles-only read is insufficient.
    candidates, arcs_by_dcid = _confirm_with_backfill(candidate_pool, selected, graph=graph)

    sv_facts = _build_sv_facts(candidates, arcs_by_dcid)
    if not sv_facts:
        return []

    place_probe = _probe_place_availability(
        sv_facts,
        place_dcids=place_dcids,
        full_facets_max_candidates=full_facets_max_candidates,
        graph=graph,
    )

    prop_labels_raw, value_labels = _fetch_constraint_labels(sv_facts, graph=graph)

    # A degenerate arc reads as role-bearing, and `group_roles` refuses its group.
    groups: dict[_GroupKey, list[str]] = defaultdict(list)
    group_families: dict[_GroupKey, IndexFamily | None] = {}
    for sv_dcid, facts in sv_facts.items():
        five_tuple = facts["five_tuple"]
        role_bearing, family = member_group_class(sv_dcid, five_tuple, facts["arcs"])
        key = (five_tuple, role_bearing, family.properties if family is not None else None)
        groups[key].append(sv_dcid)
        group_families[key] = family

    shapes: list[ShapeDraft] = []
    unusable_role_groups = 0
    zero_member_groups = 0
    for key, group_svs in groups.items():
        five_tuple, role_bearing, _properties = key
        shape_id = group_shape_id(five_tuple, role_bearing=role_bearing, family=group_families[key])
        shape, unusable_roles, zero_member = _build_group_shape(
            five_tuple,
            group_svs,
            shape_id=shape_id,
            family=group_families[key],
            sv_facts=sv_facts,
            sv_scores=sv_scores,
            recall_svs=recall_svs,
            expansion_rank=expansion_rank,
            expansion_root_score=topic_expansion.expansion_root_score,
            place_probe=place_probe,
            prop_labels_raw=prop_labels_raw,
            value_labels=value_labels,
        )
        if zero_member:
            zero_member_groups += 1
            continue
        if unusable_roles:
            unusable_role_groups += 1
            continue
        assert shape is not None
        if rival_svs.issuperset(group_svs):
            shape = dataclasses.replace(shape, rival=True)
        shapes.append(shape)

    viable_groups = len(groups) - zero_member_groups
    if unusable_role_groups and unusable_role_groups == viable_groups:
        raise RoleSignatureError(
            f"all {unusable_role_groups} viable shape group(s) (those with at least "
            "one schema member) had entity roles no single axis mapping covers, or "
            "a degenerate role arc; no group had a role signature to answer under"
        )

    return shapes
