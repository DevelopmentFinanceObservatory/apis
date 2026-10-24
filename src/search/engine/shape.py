"""Shape classification and ShapeDraft construction.

Pure module: no I/O, no LLM, no graph calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from materialise.roles import RoleAxis, roles_by_axis
from origin_catalogue import DataCommons, SourceRef
from origin_catalogue.families import FamilyKey
from origin_catalogue.origins import UN_ID_PREFIX
from search.engine.axis import classify_axis
from search.engine.errors import RoleEntityUnplaced
from search.engine.reading_coverage import ReadingCoverage
from search.engine.unit.count import is_count_unit, is_person_count_unit

# search.models.Axis is the binder's slot axis (what/how/where/when), unrelated
# to materialise.roles.RoleAxis (origin/destination) imported above.
from search.models import Axis

# Fallback label when the shape group has no ->name arc. This leaf is imported
# by bind.py and derive.py, keeping resolution modules out of the bind path.
STANDARD_SHAPE_LABEL = "standard Data Commons"

type Pool = Literal["ours", "un_system"]
"""The resolver whose recall reached a shape: ours, or the UN System Data Commons'. Each
scores on its own scale, so each pool meets its own score floor. The UN pool's shapes are
elected on the UN pass (`region.pipeline`), and with `SEARCH_UN_ON_FIRST_MENU` also on our
pass's menu beside ours (`region.phase_intake._merge_un_pool`)."""

# ShapeDraft is the ungrounded shape representation used inside the pipeline.


@dataclass(frozen=True)
class SlotKeyDraft:
    """Ungrounded slot identity: axis + property dcid + display label.

    The property_dcid may be None for the when/source axis slots (observation-facet layer).
    """

    axis: Axis
    property_dcid: str | None  # None for when/source
    label: str


@dataclass(frozen=True, slots=True)
class UpstreamSource:
    """The one source, and the Data Commons serving it, that every member of
    an upstream shape comes through (`derive.upstream_source`)."""

    source: SourceRef
    data_commons: DataCommons


@dataclass(frozen=True)
class ShapeDraft:
    """Ungrounded shape: five-tuple dcids + ordered slot keys.

    sv_arc_facts and slot_taxonomy are stamped by derive.derive_shapes to
    avoid re-reading node_arcs per SV.
    """

    shape_id: str
    label: str

    # Five-tuple dcids. Required members use "" for absent arcs. A Shape groups on
    # whatever tuple subset the SV asserts (derive.derive_shapes).
    pop_type_dcid: str
    meas_prop_dcid: str
    stat_type_dcid: str
    meas_qual_dcid: str | None
    meas_denom_dcid: str | None

    # Slot keys in display order
    slot_keys: tuple[SlotKeyDraft, ...]

    # The population type's graph name, else its dcid, carried from derive_shapes. Every
    # member of the family shares it, so it describes the whole family. None on a shape
    # derive did not build.
    pop_type_name: str | None = field(default=None, compare=False)
    # True when the denominator arc points to a ProvisionalNode, including one
    # with a dcid. False with an absent arc. This distinction drives per-capita
    # checks and takes precedence over meas_denom_dcid in identity and shape ids.
    meas_denom_unresolved: bool = False
    # Per-SV arc facts dict (sv_dcid -> raw arcs), carried from derive_shapes.
    sv_arc_facts: dict[str, dict] | None = field(default=None, compare=False)
    # True when sv_arc_facts is complete and False when it is a
    # recall sample, however many members recall returned. Directly
    # built drafts default to complete, preventing consumers from widening facts
    # whose completeness the constructor leaves unknown.
    sv_arc_facts_complete: bool = field(default=True, compare=False)
    # The catalogue family whose members the shape stands for, stamped by derive_shapes on a
    # group whose membership the graph walk decides and whose family the catalogue's tables
    # hold (`derive.held_family`): its slot taxonomy reads that family, and so does
    # `bound_values.shape_member_facts`, beside sv_arc_facts. None on every other shape and
    # on one narrowed to some of its members.
    held_family: FamilyKey | None = field(default=None, compare=False)
    # The role set the role-class election narrowed a held shape to
    # (`bound_values._recalled_role_class_shape`): its population holds the members of
    # held_family that declare that set alone. None on a shape no election narrowed.
    held_roles: frozenset[str] | None = field(default=None, compare=False)
    # The members of sv_arc_facts the mint's walk of this shape is seeded with
    # (`bound_values.elected_role_class_shape`): on a shape narrowed from the members it
    # stood for (`bound_values.narrowed_to`), the ones it held before, so a walk is seeded
    # with recalled members and never with a held family's. None seeds every member.
    walk_seeds: frozenset[str] | None = field(default=None, compare=False)
    # The what and how slot properties a breakdown decision left out of slot_keys
    # (`phase_election._narrow_to_breakdown`). A member asserting one is no member of the
    # identity minted from the shape, so admission leaves it out
    # (`standard._admit_standard_members`). Empty on a shape no decision narrowed.
    omitted_slots: frozenset[str] = field(default=frozenset(), compare=False)
    # Per-shape bind taxonomy: "axis:property_dcid" -> a list of values a
    # member asserts (facts.read_slot_taxonomy).
    slot_taxonomy: dict[str, list[str]] | None = field(default=None, compare=False)
    # True when slot_taxonomy covers the full indexed population, and False when it
    # comes from recall-sampled sv_arc_facts. Bind display and SlotOptions.truncated
    # depend on this distinction. Directly built drafts default to False.
    slot_taxonomy_index_sourced: bool = field(default=False, compare=False)
    # Highest cosine score among the group's recalled members, or None when all
    # arrived through topic expansion. None supplies no strength evidence for margins or
    # confidence. Measure-family collapse may raise it to the family's max score.
    # It can differ from the member that supplied the label.
    representative_score: float | None = field(default=None, compare=False)
    # Ordering key for an expansion-only group: its strongest topic-root score,
    # capped below the best recalled candidate. Rank sorts use it. Floors and weak-score
    # checks use measured_score; confidence and margins use representative_score.
    expansion_rank_score: float | None = field(default=None, compare=False)
    # Recall score of the strongest topic root that reached an expansion-only group,
    # uncapped. None when a member was recalled. measured_score reads it.
    expansion_root_score: float | None = field(default=None, compare=False)
    # The topic root that reached an expansion-only group: the root of the leaf its
    # expansion rank comes from. None when a member was recalled. The bind menu seats a
    # root with every leaf or none (`region.seating.seat_menu`).
    expansion_root: str | None = field(default=None, compare=False)
    # Whether group members have observations at the named countries, read by derive. None
    # marks an unrun or failed-open probe, while False means it ran and found none. The
    # slot defaults read it (`bind_repair.apply_shape_slot_defaults`).
    place_available: bool | None = field(default=None, compare=False)
    # Per member, the places the part names at which it has observations, each with the
    # years its facets there hold (`reading_coverage.observed_years`), set by the
    # availability read over a menu shape when the part names a place other than a country
    # (`region.availability.read_menu_availability`). A member with data at no such place
    # maps to an empty mapping. None when that read did not run or failed for this shape.
    named_place_coverage: Mapping[str, ReadingCoverage] | None = field(default=None, compare=False)
    # The places the read behind the availability above covered: derive's probe at the
    # named countries, or the menu read at every place the part names. A read that found no
    # data establishes absence at these places only (`region.availability`). Empty when no
    # read ran.
    availability_places: frozenset[str] = field(default=frozenset(), compare=False)
    # Units of the facets holding the group's observations at the queried place, with
    # None for a facet that names no unit. None when the probe did not run or found no
    # data. `measures_count` reads it.
    place_units: frozenset[str | None] | None = field(default=None, compare=False)
    # Per member, the probed places at which it has observations, each with the years its
    # facets there hold (`reading_coverage.observed_years`). A member with data at no probed
    # place maps to an empty mapping. None when the derive probe did not run or failed, and
    # for a role-bearing shape, whose observations the subject-facet probe cannot see. The
    # default-reading collapse (`bound_values.collapse_measure_families`) reads it.
    member_coverage: Mapping[str, ReadingCoverage] | None = field(default=None, compare=False)
    # Labels for observed constraint-value dcids support query-word matching in
    # bind repair and qualifier detection. derive_shapes gets graph labels in a
    # batch, but uses CodeRow.label (without its system suffix) for indexed dcids.
    # None when another constructor built the draft.
    value_labels: dict[str, str] | None = field(default=None, compare=False)
    # Every role a confirmed group member declares, in first-seen arc order.
    # Subject-only groups declare ("observationAbout"). None when the draft
    # came from a constructor other than derive_shapes or promote.
    declared_roles: tuple[str, ...] | None = field(default=None, compare=False)
    # The subset of declared_roles that some member does not declare, such as
    # the creditor role of a total/by-creditor twin pair. A query may fill an
    # optional role; admission keeps the members that answer the roles it fills.
    optional_roles: frozenset[str] = field(default=frozenset(), compare=False)
    # True when any confirmed member is a direct recall. The bind menu reserves
    # space for recalled shapes, and family collapse carries this flag from any
    # recalled sibling.
    recalled: bool = field(default=False, compare=False)
    # The members retrieval scored against the question (`derive_shapes`), Topic expansions
    # excluded, which can sort ahead of them in sv_arc_facts. Empty on a shape derive did
    # not build.
    scored_members: frozenset[str] = field(default=frozenset(), compare=False)
    # True when every confirmed member is a row recall added for its five-tuple past the
    # cut (`clients.graph.with_rival_families`). Such a shape displaces only shapes of its
    # own five-tuple (`place_roles.without_unasked_axes`).
    rival: bool = field(default=False, compare=False)
    # The resolver whose recall reached the shape (`Pool`). A variable our resolver recalls
    # under a superseded dcid derives as its UN successor and stays in our pool.
    pool: Pool = field(default="ours", compare=False)
    # The datasets the part may answer this shape's five-tuple from, in serving order,
    # stamped by shape candidacy (`bound_values.shape_for_part`): the dataset the part
    # names, else the five-tuple's eligible publishers. None before candidacy and on a
    # shape no candidacy kept, whose pre-bind readers read every dataset. It enters
    # equality and the hash, since a named and an unnamed part read different rows.
    datasets: tuple[str, ...] | None = None
    # The dataset the part names, stamped by shape candidacy beside `datasets`: the one the
    # part asks for, or the one opt-in dataset an unnamed part may answer the shape from;
    # None otherwise. Only an identity naming a dataset implies its invariants
    # (`bound_values.shape_invariants`).
    named_dataset: str | None = None
    # False for shapes rebuilt from a posted identity, which have no retrieval
    # score. Their place confidence uses data availability alone. Expansion-only
    # Expansion-only shapes count as retrieved, so a missing score caps confidence.
    retrieved: bool = field(default=True, compare=False)

    @property
    def rank_score(self) -> float | None:
        """Ordering key: representative_score, else expansion_rank_score, else None.

        representative_score drives strength checks. expansion_rank_score only orders
        expansion-only shapes.
        """
        return (
            self.representative_score
            if self.representative_score is not None
            else self.expansion_rank_score
        )

    @property
    def measured_score(self) -> float | None:
        """The recall score a shape meets the score floor and the weak-match band on:
        representative_score, else the expansion's topic-root score, else None."""
        return (
            self.representative_score
            if self.representative_score is not None
            else self.expansion_root_score
        )


# Observation-facet slots
_WHEN_SLOT = SlotKeyDraft(axis="when", property_dcid=None, label="time window")
_SOURCE_SLOT = SlotKeyDraft(axis="source", property_dcid=None, label="data source")

# The subject-place slot has no constraint property, like when/source. It appears
# in resolved slots but stays out of the bind prompt and identity hash. The entity
# itself is already carried in resolved.entities.
_WHERE_SLOT = SlotKeyDraft(axis="where", property_dcid=None, label="place")


def is_ratio_backed_sv(sv_dcid: str) -> bool:
    """True when a dcid uses Data Commons' ``_AsAFractionOf_`` ratio suffix.

    Ratio suffixes fall outside sibling construction's token-per-constraint grammar.
    Refinement is disabled for those shapes.
    """
    return "_AsAFractionOf_" in sv_dcid


def identity_shape_for(
    *,
    pop_type_dcid: str | None,
    meas_prop_dcid: str | None,
    stat_type_dcid: str | None,
    meas_qual_dcid: str | None,
    meas_denom_dcid: str | None,
    meas_denom_unresolved: bool = False,
) -> dict:
    """Build the identity core's ``statvar_schema`` arm from five-tuple dcids.

    Omit empty dcids. An unresolved denominator takes precedence over its dcid,
    which may name a dangling ProvisionalNode. Resolved dcids retain their case.
    The plain dict is validated with the full identity projection by the caller.
    """
    shape: dict[str, object] = {"kind": "statvar_schema"}
    if pop_type_dcid:
        shape["population_type"] = pop_type_dcid
    if meas_prop_dcid:
        shape["measured_property"] = meas_prop_dcid
    if stat_type_dcid:
        shape["stat_type"] = stat_type_dcid
    if meas_qual_dcid:
        shape["measurement_qualifier"] = meas_qual_dcid
    if meas_denom_unresolved:
        shape["measurement_denominator"] = {"unresolved": True}
    elif meas_denom_dcid:
        shape["measurement_denominator"] = {"ref": meas_denom_dcid}
    return shape


def _declared_role_names(shape: ShapeDraft) -> tuple[str, ...]:
    """Return declared non-subject roles, or () for subject-only or unbuilt drafts."""
    if shape.declared_roles in (None, ("observationAbout",)):
        return ()
    return shape.declared_roles or ()


def required_role_names(shape: ShapeDraft) -> tuple[str, ...]:
    """Return the non-subject roles every member declares, in declared order."""
    return tuple(role for role in _declared_role_names(shape) if role not in shape.optional_roles)


def declares_shape_roles(shape: ShapeDraft, roles: Sequence[str]) -> bool:
    """True when `roles` is a role set some member of `shape` could declare.

    The set must hold every required role and nothing outside `declared_roles`.
    A draft without declared roles accepts any set.
    """
    if shape.declared_roles is None:
        return True
    declared = frozenset(roles)
    union = frozenset(shape.declared_roles)
    return union - shape.optional_roles <= declared <= union


def is_role_bearing(shape: ShapeDraft) -> bool:
    """True when the group's members declare a non-subject entity role."""
    return bool(_declared_role_names(shape))


def role_class(shape: ShapeDraft) -> frozenset[str]:
    """The non-subject roles the group's members declare, as a set: the form every
    role-class comparison and key reads, since `declared_roles` keeps arc order. Subject-only
    and unbuilt drafts give the empty set."""
    return frozenset(_declared_role_names(shape))


def measures_count(shape: ShapeDraft) -> bool:
    """True when the group's values are counts: its measured property is ``count``, or
    every facet holding its data at the queried place carries a count unit
    (`unit.count.is_count_unit`). An SDG or UN System series measures ``value`` and says it
    counts only through its unit."""
    if shape.meas_prop_dcid == "count":
        return True
    units = shape.place_units
    return bool(units) and all(is_count_unit(unit) for unit in units)


def counts_persons(shape: ShapeDraft) -> bool:
    """True when the group's values count persons: its population type is ``Person`` and
    it measures a count (`measures_count`), or every facet holding its data at the queried
    place carries a unit counting persons (`unit.count.is_person_count_unit`), as a UN
    System series says it does."""
    if shape.pop_type_dcid == "Person" and measures_count(shape):
        return True
    units = shape.place_units
    return bool(units) and all(is_person_count_unit(unit) for unit in units)


def is_un_shape(shape: ShapeDraft) -> bool:
    """True when every member dcid is a UN System Data Commons variable (`undata/`)."""
    members = shape.sv_arc_facts
    return bool(members) and all(dcid.startswith(UN_ID_PREFIX) for dcid in members)


def role_names_by_axis(shape: ShapeDraft) -> dict[RoleAxis, str]:
    """Map declared flow axes to role names, optional roles included.

    Returns ``{}`` for subject-only shapes and for roles the axis vocabulary cannot
    place. Use ``_declared_role_names`` to distinguish them. Raises ValueError when
    two roles name the same axis.
    """
    return roles_by_axis(_declared_role_names(shape))


def entities_by_role(
    *,
    shape: ShapeDraft,
    by_axis: Mapping[RoleAxis, Sequence[str]] | None = None,
    subject: Sequence[str] = (),
    role_names: Mapping[RoleAxis, str] | None = None,
) -> dict[str, list[str]]:
    """Map flow-axis entities to the role names used on the resolve wire.

    Subject-only shapes read ``subject``. Role-bearing shapes read ``by_axis`` and
    reject subject values. Empty axis lists are omitted. ``role_names`` overrides
    axis lookup for roles refreshed from the graph. Declaration checks read the shape's own declaration.
    Raises RoleEntityUnplaced for an unassignable entity and
    ValueError when declared roles share an axis.
    """
    declared = _declared_role_names(shape)
    if not declared:
        return {"observationAbout": list(subject)} if subject else {}
    if subject:
        raise RoleEntityUnplaced(
            f"shape {shape.shape_id!r} declares {declared!r} and no subject role; "
            f"{list(subject)!r} has no role to bind under"
        )
    names = dict(role_names) if role_names is not None else roles_by_axis(declared)
    bound: dict[str, list[str]] = {}
    for axis, dcids in (by_axis or {}).items():
        if not dcids:
            continue
        name = names.get(axis)
        if name is None:
            raise RoleEntityUnplaced(
                f"shape {shape.shape_id!r} declares {declared!r}, which names no "
                f"role on the {axis!r} flow axis; {list(dcids)!r} has no role to bind under"
            )
        bound[name] = list(dcids)
    return bound


def family_label(
    *,
    population_type_label: str,
    member_constraints: Sequence[Mapping[str, str]],
    prop_labels: Mapping[str, str],
    value_labels: Mapping[str, str],
) -> str:
    """Label a multi-member family by its population type and the values every member shares.

    A property enters the label only when every member holds it at one value, so an
    unbound slot's values never appear. Shared properties follow the first member's
    constraint order and read as UN System variable names do: ``Name [Property=Value]``.
    """
    first, *rest = member_constraints
    shared = [
        f"{prop_labels.get(prop, prop)}={value_labels.get(value, value)}"
        for prop, value in first.items()
        if all(member.get(prop) == value for member in rest)
    ]
    if not shared:
        return population_type_label
    return f"{population_type_label} [{', '.join(shared)}]"


def shape_draft_from(
    *,
    shape_id: str,
    label: str,
    pop_type_dcid: str,
    meas_prop_dcid: str,
    stat_type_dcid: str,
    meas_qual_dcid: str | None,
    meas_denom_dcid: str | None,
    constraint_props: list[str],
    pop_type_name: str | None = None,
    prop_labels: dict[str, str],
    prop_observed_values: dict[str, list[str]],
    meas_denom_unresolved: bool = False,
    sv_arc_facts: dict[str, dict] | None = None,
    sv_arc_facts_complete: bool = True,
    slot_taxonomy: dict[str, list[str]] | None = None,
    representative_score: float | None = None,
    expansion_rank_score: float | None = None,
    expansion_root_score: float | None = None,
    expansion_root: str | None = None,
    place_available: bool | None = None,
    place_units: frozenset[str | None] | None = None,
    value_labels: dict[str, str] | None = None,
    declared_roles: tuple[str, ...] | None = None,
    optional_roles: frozenset[str] = frozenset(),
    recalled: bool,
    retrieved: bool = True,
) -> ShapeDraft:
    """Build a ShapeDraft and classify each constraint property with ``classify_axis``.

    Axis classification uses observed values for date and place-fraction rules, alongside
    ``AXIS_OVERRIDES``. Adds the where, when, and source facet slots after constraint slots.

    An empty five-tuple dcid means the arc is absent. `meas_denom_unresolved` is true when
    the denominator target has no dcid or is a dcid-bearing ProvisionalNode.
    """
    constraint_slot_keys: list[SlotKeyDraft] = []
    for prop_dcid in constraint_props:
        observed = prop_observed_values.get(prop_dcid, [])
        axis = classify_axis(prop_dcid, observed)
        # Keep the graph's original label (including a dcid echo) intact in
        # the draft. The binder consumes these bytes, while presentation-only
        # humanization happens when the wire SlotKey is assembled.
        label_text = prop_labels.get(prop_dcid, prop_dcid)
        constraint_slot_keys.append(
            SlotKeyDraft(axis=axis, property_dcid=prop_dcid, label=label_text)
        )

    all_slots = (*tuple(constraint_slot_keys), _WHERE_SLOT, _WHEN_SLOT, _SOURCE_SLOT)

    return ShapeDraft(
        shape_id=shape_id,
        label=label,
        pop_type_dcid=pop_type_dcid,
        meas_prop_dcid=meas_prop_dcid,
        stat_type_dcid=stat_type_dcid,
        meas_qual_dcid=meas_qual_dcid,
        meas_denom_dcid=meas_denom_dcid,
        slot_keys=all_slots,
        meas_denom_unresolved=meas_denom_unresolved,
        pop_type_name=pop_type_name,
        sv_arc_facts=sv_arc_facts,
        sv_arc_facts_complete=sv_arc_facts_complete,
        slot_taxonomy=slot_taxonomy,
        representative_score=representative_score,
        expansion_rank_score=expansion_rank_score,
        expansion_root_score=expansion_root_score,
        expansion_root=expansion_root,
        place_available=place_available,
        place_units=place_units,
        value_labels=value_labels,
        declared_roles=declared_roles,
        optional_roles=optional_roles,
        recalled=recalled,
        retrieved=retrieved,
    )
