"""Ground answer data and build nearest-real suggestions.

This module depends on leaf helpers and sits below events and option coverage. It also builds
warnings for open role axes.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

from code_index import LevelOption
from code_index.members import offers_slot
from code_index.shared import get_index
from dfo_platform.executor import submit_with_context
from materialise import GraphReader
from materialise.roles import axis_noun
from search.engine.bind import SlotBindingDraft
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.errors import EngineInputError, GroundingMiss, RoleEntityUnplaced
from search.engine.extract import DateRequest
from search.engine.ground import graphref
from search.engine.region.arcs import (
    inapplicable_slots_from_member_facts,
    unresolved_slots_from_member_facts,
)
from search.engine.region.codes import (
    ENTITY_LABEL_UNRESOLVED,
    MEMBERSHIP_UNVERIFIED,
    OPTION_COVERAGE_PROBE_FAILED,
    ROLE_DECLARATION_UNVERIFIED,
    ROLE_SCOPE_DEFAULTED,
)
from search.engine.region.decisions import detect_set_ref
from search.engine.region.facets import answer_facet_dcids, answer_facets, cited_facets
from search.engine.region.refs import (
    _ref_or_miss,
    _refs_present,
    _where_slot_refs,
)
from search.engine.retrieval.bound_values import (
    RoleClassWalks,
    apply_declaration_version,
    apply_slot_expansions,
    elected_role_class_shape,
    entity_role_keys,
    five_tuple_member_facts,
    index_carries_shape,
    shape_query,
    slot_expansions,
)
from search.engine.retrieval.coverage import citable_source_ids
from search.engine.retrieval.elected_dataset import elected_datasets
from search.engine.retrieval.facts import read_slot_taxonomy
from search.engine.retrieval.member_facts import ArcsMemberFacts, MemberFacts
from search.engine.retrieval.standard import standard_option_coverage, standard_resolve
from search.engine.shape import (
    ShapeDraft,
    SlotKeyDraft,
    entities_by_role,
    is_role_bearing,
)
from search.engine.spec_build import (
    NamedGroup,
    bind_when_slot,
    build_shape_model,
    build_slot,
    build_spec,
    build_stat_vars,
    compute_member_digest,
    inapplicable_property_dcid,
    pair_slot_drafts,
)
from search.engine.unit.refs import unit_currency
from search.models import (
    Confidence,
    Entity,
    EntityRoleDirectional,
    EntityRoleKey,
    EntityRoleSubject,
    GraphRef,
    IdentityUnitRequest,
    NearestRealSpec,
    ProvenanceRef,
    Relaxation,
    RequestedWindow,
    SlotOption,
    SlotOptions,
    SlotValue,
    TimeWindow,
    Warning,
)
from search.observations import provenance_refs, unlisted_provenances

if TYPE_CHECKING:
    from materialise.place_scope import DefaultPresence
    from search.engine.region.place_context import PlaceContext
    from search.engine.retrieval.read_plan import ReadPlan
    from search.engine.retrieval.retrieve import Materialised
    from search.engine.retrieval.role_scope import DiscoveredScope
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

# role_scope_defaulted_message spells out this many bound entity labels at most.
# Longer lists collapse to a count.
_ROLE_SCOPE_MAX_NAMED_LABELS = 3

type EntityRoleBindings = list[tuple[str, EntityRoleKey]]
type EntityRoles = Mapping[str, EntityRoleKey] | EntityRoleBindings


def role_bindings(roles: EntityRoles) -> EntityRoleBindings:
    """Preserve every (dcid, role) pair while accepting legacy maps at seams."""
    if isinstance(roles, Mapping):
        # `isinstance` loses Mapping's generic arguments to ty.
        return cast(EntityRoleBindings, list(roles.items()))
    return roles


# Keep the tuple arity fixed so `ty` checks every unpack against all eleven fields.
GroundedAnswer = tuple[
    dict[str, GraphRef],  # five_tuple_refs
    dict[str, GraphRef],  # role_refs
    list,  # slots (list[Slot])
    list,  # slot_key_models (list[SlotKey])
    list[GraphRef],  # sv_refs
    list,  # entity_objects_data (list[tuple[EntityRoleKey, GraphRef, GraphRef | None]])
    list[ProvenanceRef],  # source_refs
    list[SlotOptions],  # slot_options
    list[str],  # inapplicable_slots (property dcids)
    list[Warning],  # warnings
    dict[str, str],  # labels: the one label batch, which `facets.answer_facets` reads
]


# ---------------------------------------------------------------------------
# Per-variable helper functions
# ---------------------------------------------------------------------------


def reject_reversed_window(window: RequestedWindow | None) -> None:
    """Raise `EngineInputError(code="request_invalid")` when start year exceeds end year.

    This prevents `compute_spec_id` from raising an untranslated `IdentityError`. None,
    `"latest"`, and ranges with at most one bound are unchanged.
    """
    if not isinstance(window, TimeWindow):
        return
    if (
        window.start_year is not None
        and window.end_year is not None
        and window.start_year > window.end_year
    ):
        raise EngineInputError(
            f"requested_window is reversed: start_year {window.start_year} "
            f"> end_year {window.end_year}",
            code="request_invalid",
        )


def requested_window_from_date_request(date_request: DateRequest | None) -> RequestedWindow | None:
    """Map a date request to identity data: `"latest"`, a concrete window, or None.

    The sentinel distinguishes an explicit latest request from no date request. Reversed
    windows raise `EngineInputError(code="request_invalid")` before spec construction.
    """
    if date_request is None:
        return None
    window = "latest" if date_request.latest else date_request.window
    reject_reversed_window(window)
    return window


def _grade_confidence(
    *,
    score: float | None,
    has_data: bool,
    available_at_place: bool,
    retrieved: bool = True,
    near_neighbor_relevant: bool = False,
) -> Confidence:
    """Grade available results against `weak_score_threshold`.

    Unavailable data is low confidence. An unscored retrieved shape is medium. A
    client-posted identity (`retrieved=False`) with data is high because it has no retrieval
    score to assess. A relevant near-neighbour tie caps a retrieved shape at medium, and
    a weak score beside it is low.
    """
    if not available_at_place:
        return "low"
    if has_data and not retrieved:
        return "high"
    weak = not has_data or score is None or score < settings().weak_score_threshold
    if weak and near_neighbor_relevant and score is not None:
        return "low"
    if weak or near_neighbor_relevant:
        return "medium"
    return "high"


def _place_confidence(
    *,
    destination_dcid: str | None,
    available_at_place: bool | None,
    representative_score: float | None,
    candidate: bool = False,
    near_neighbor_relevant: bool = False,
    has_data: bool = True,
    retrieved: bool = True,
) -> tuple[bool | None, Confidence | None]:
    """Derive place availability and confidence from one materialised result.

    Without a destination, availability is None. Confidence is also None when `has_data`
    is false. Retrieved shapes without a score cap at medium, while un-retrieved posted
    identities grade from availability alone. Unavailable data is low. Weak retrieval plus
    a relevant near-neighbour is low, and either signal alone is medium
    (`_grade_confidence`). `near_neighbor_relevant` is set for a candidate in a fired tie
    and for a definite answer whose tie stands unresolved (`ResolveState.tie_unresolved`).
    """
    if available_at_place is False and (candidate or destination_dcid is not None):
        return False, "low"
    if destination_dcid is None and not candidate and not has_data:
        return None, None
    # Place-less candidates get relevance confidence even when availability is unknown.
    return (None if destination_dcid is None else True), _grade_confidence(
        score=representative_score,
        has_data=True,
        available_at_place=True,
        retrieved=retrieved,
        near_neighbor_relevant=near_neighbor_relevant,
    )


def _shape_with_taxonomy(shape: ShapeDraft) -> ShapeDraft:
    """Fill missing slot taxonomy from pre-bind member facts.

    Exact posted sets use their own arcs. `index_carries_shape` controls whether slot options
    use level-scoped index menus. `five_tuple_member_facts` reports no index source for an
    exact posted set, so reusing it would flatten a resubmit's "how" menus. The taxonomy
    values come from every index row of the five-tuple, across datasets.
    """
    if shape.slot_taxonomy is not None:
        return shape
    member_facts, _ = five_tuple_member_facts(shape)
    taxonomy = read_slot_taxonomy(slot_keys=shape.slot_keys, member_facts=member_facts)
    return dataclasses.replace(
        shape, slot_taxonomy=taxonomy, slot_taxonomy_index_sourced=index_carries_shape(shape)
    )


def bound_opposite_role_dcids(discovered: DiscoveredScope, roles: EntityRoles) -> list[str]:
    """Return unique dcids bound to the role opposite the discovered axis, in binding order."""
    opposite_role: EntityRoleKey = "origin" if discovered.axis == "destination" else "destination"
    return list(dict.fromkeys(dcid for dcid, role in role_bindings(roles) if role == opposite_role))


def _role_scope_defaulted_base(axis: EntityRoleKey) -> str:
    noun = axis_noun(axis)
    return f"no {noun} was named; the answer was resolved across every {noun} with data"


def role_scope_defaulted_message(
    discovered: DiscoveredScope,
    *,
    member_count: int | None,
    bound_dcids: Sequence[str],
    label_of: Callable[[Sequence[str]], Sequence[str]],
) -> str:
    """Build the defaulted-scope warning with member count and any bound opposite-role entities.

    `member_count` is how many members of the open axis the materialised read plan serves
    (`Materialised.scope_member_dcids`), or None for a response whose answers serve different
    members, which the warning then counts as every member with data. Use `from` when
    destination is discovered, otherwise `to`. Spell out up to three labels; for four or
    more, report the count and opposite-axis noun without label lookup. `label_of` returns
    labels in the input order.
    """
    noun = axis_noun(discovered.axis)
    if member_count is None:
        base = _role_scope_defaulted_base(discovered.axis)
    else:
        noun_phrase = noun if member_count == 1 else f"{noun}s"
        base = (
            f"no {noun} was named; the answer was resolved across the {member_count} "
            f"{noun_phrase} with data"
        )
    if not bound_dcids:
        return f"{base}."
    preposition = "from" if discovered.axis == "destination" else "to"
    if len(bound_dcids) > _ROLE_SCOPE_MAX_NAMED_LABELS:
        opposite_noun = axis_noun("origin" if discovered.axis == "destination" else "destination")
        named = f"any of the {len(bound_dcids)} {opposite_noun}s"
    else:
        labels = label_of(bound_dcids)
        if len(labels) == 1:
            named = labels[0]
        elif len(labels) == 2:
            named = f"{labels[0]} or {labels[1]}"
        else:
            named = f"{', '.join(labels[:-1])}, or {labels[-1]}"
    return f"{base} {preposition} {named}."


def role_scope_defaulted_message_for_axis(axis: EntityRoleKey) -> str:
    """Build the unqualified warning for an open axis.

    This path has no discovered scope or member count.
    """
    return f"{_role_scope_defaulted_base(axis)}."


def memoised_labels(
    label_of: Callable[[Sequence[str]], Sequence[str]],
) -> Callable[[Sequence[str]], Sequence[str]]:
    """`label_of` reading each list of dcids once, so a warning restated after the facet
    filter (`restate_role_scope_warnings`) makes no second label request."""
    cache: dict[tuple[str, ...], Sequence[str]] = {}

    def lookup(dcids: Sequence[str]) -> Sequence[str]:
        key = tuple(dcids)
        if key not in cache:
            cache[key] = label_of(dcids)
        return cache[key]

    return lookup


def role_scope_defaulted_warnings(
    mat: Materialised,
    *,
    boundary_roles: EntityRoles,
    label_of: Callable[[Sequence[str]], Sequence[str]],
    members: Sequence[str] | None = None,
    state_count: bool = True,
) -> list[Warning]:
    """Build warnings for each open role axis in `mat`.

    A discovered scope reports how many members the answer serves (`members`, by default
    `mat.scope_member_dcids`, those of the materialised read plan), or with `state_count`
    false every member with data, and any bound opposite-role entities. A second open axis
    adds an unqualified warning. A bound
    destination with an open origin also warns, even when `open_origin_dcids` is empty.
    Return an empty list when no axis is open. `label_of` runs only for discovered scopes.
    """
    warnings: list[Warning] = []
    discovered = mat.discovered_scope
    if discovered is not None:
        warnings.append(
            Warning(
                code=ROLE_SCOPE_DEFAULTED,
                severity="warn",
                message=role_scope_defaulted_message(
                    discovered,
                    member_count=len(
                        members if members is not None else mat.scope_member_dcids or ()
                    )
                    if state_count
                    else None,
                    bound_dcids=bound_opposite_role_dcids(discovered, boundary_roles),
                    label_of=label_of,
                ),
            )
        )
        if mat.also_defaulted_axis is not None:
            # Both flow axes were unbound. The axis above is the one actually
            # discovered, and this companion axis defaulted right alongside it.
            warnings.append(
                Warning(
                    code=ROLE_SCOPE_DEFAULTED,
                    severity="warn",
                    message=role_scope_defaulted_message_for_axis(mat.also_defaulted_axis),
                )
            )
        return warnings
    if mat.open_origin_dcids is not None:
        warnings.append(
            Warning(
                code=ROLE_SCOPE_DEFAULTED,
                severity="warn",
                message=role_scope_defaulted_message_for_axis("origin"),
            )
        )
    return warnings


def restate_role_scope_warnings(
    warnings: list[Warning],
    mat: Materialised,
    *,
    members: Sequence[str] | None,
    boundary_roles: EntityRoles,
    label_of: Callable[[Sequence[str]], Sequence[str]],
    state_count: bool = True,
) -> None:
    """Restate, in place, the discovered-scope warning `interpret_materialised` built for
    `mat` so it counts `members`, the scope the answer lists under the request's facet
    filter (`scope.stated_scope_members`), or with `state_count` false names every member
    with data, for a response whose answers list different scopes. The warning is the
    `role_scope_defaulted` one opening with the same axis clause ("no recipient was named"),
    whatever count it states, so a response restates it once its candidates are grounded
    and again once they are filtered. It keeps its position, and `label_of`, the memoised
    lookup that built it (`memoised_labels`), makes no new request.

    An empty `members` is an answer that lists no member on the axis: it resolved across
    none, so the warning is removed."""
    if mat.discovered_scope is None or members is None:
        return
    fresh = role_scope_defaulted_warnings(
        mat,
        boundary_roles=boundary_roles,
        label_of=label_of,
        members=members,
        state_count=state_count,
    )
    # The discovered axis's warning leads. A companion axis's warning states no count, so it
    # stands as built.
    restated: Warning | None = fresh[0] if members or not state_count else None
    axis_clause = fresh[0].message.split(";", 1)[0]
    for index, old in enumerate(warnings):
        if old.code == ROLE_SCOPE_DEFAULTED and old.message.split(";", 1)[0] == axis_clause:
            if restated is None:
                del warnings[index]
            else:
                warnings[index] = restated
            break


def carrier_flag_warnings(
    *,
    role_declaration_unverified: bool,
    open_origin_defaulted: bool,
    membership_unverified: bool = False,
) -> list[Warning]:
    """Build one warning per true carrier flag.

    Candidate callers fold flags across alternatives before calling. A warning describes
    the answer even when several candidates carry the same condition.
    """
    warnings: list[Warning] = []
    if role_declaration_unverified:
        warnings.append(
            Warning(
                code=ROLE_DECLARATION_UNVERIFIED,
                severity="warn",
                message=role_declaration_unverified_message(),
            )
        )
    if open_origin_defaulted:
        # role_scope_defaulted_warnings' bound-destination arm, the same text
        # a definite answer carries for this state. That builder reads the
        # whole Materialised and would re-run its discovered-scope arm's label
        # read for every alternate. This arm needs neither.
        warnings.append(
            Warning(
                code=ROLE_SCOPE_DEFAULTED,
                severity="warn",
                message=role_scope_defaulted_message_for_axis("origin"),
            )
        )
    if membership_unverified:
        warnings.append(membership_unverified_warning())
    return warnings


def membership_unverified_warning() -> Warning:
    """Describe an answer that reads recalled members the mint's member walk did not settle
    (`bound_values.MembershipAuthority`)."""
    return Warning(
        code=MEMBERSHIP_UNVERIFIED,
        severity="warn",
        message=(
            "the full set of series behind this answer could not be confirmed, so it "
            "uses the series the search found; a dataset made from it may hold "
            "different series."
        ),
    )


def role_declaration_unverified_message() -> str:
    """Describe answers using indexed roles without a completed graph refresh.

    Refresh is skipped above `SEARCH_ROLE_REFRESH_MAX_MEMBERS`.
    """
    return (
        "the graph read confirming which entity roles these members declare "
        "was not completed, so the answer stands on the role declarations "
        "frozen in the packaged code index."
    )


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _GroundingLookup:
    """Shared lookup results for slot and entity grounding.

    `labels` comes from one graph batch. Slot pairs, source ids, taxonomy evidence, and
    role bindings are precomputed for assembly. Role names have no graph refs.
    """

    labels: dict[str, str]
    slot_pairs: list[tuple[SlotKeyDraft, SlotBindingDraft | None]]
    level_options_by_key: dict[str, tuple[LevelOption, ...]]
    # Constraint values the recall arcs or a served member assert, by `axis:property`.
    evidenced_values: dict[str, set[str]]
    prov_dcids: list[str]
    resolved_role_bindings: EntityRoleBindings
    five_tuple_refs: dict[str, GraphRef]
    role_refs: dict[str, GraphRef]


def _ground_answer_lookup(
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    sv_dcids: list[str],
    roles: EntityRoles,
    graph: EngineGraphClient,
    plan: ReadPlan | None,
    unit_request: IdentityUnitRequest | None,
    member_facts: Mapping[str, MemberFacts],
) -> _GroundingLookup:
    """Batch every graph-label lookup needed by grounding.

    Collect distinct ids across shapes, slots, StatVars, roles, and options for one
    `node_labels_batch` call. Missing graph labels fall back to packaged code-index labels.
    Source labels come from the origin catalogue; only provenance ids it does not list
    join the batch.
    Index-sourced taxonomies offer the bound value's parent and children. Unbound slots and
    bound sets offer each root and its children. Deeper levels are excluded from the lookup.
    A slot gets an index-sourced menu only when a dataset that can serve the family
    `bindings` and the roles' keys state publishes a row carrying its property
    (`code_index.members.offers_slot`), so a table without the slot's code system offers no
    menu when it is the one that serves.
    Sources are the citable provenance ids of the facets `plan` serves (`cited_facets`), and
    the methods and units `facets.answer_facets` reports join the batch
    (`facets.answer_facet_dcids`).
    The recall arcs and the `member_facts` of `sv_dcids` evidence bound constraint values.
    """
    five_tuple_dcids = []
    if shape_draft.pop_type_dcid:
        five_tuple_dcids.append(shape_draft.pop_type_dcid)
    if shape_draft.stat_type_dcid:
        five_tuple_dcids.append(shape_draft.stat_type_dcid)
    if shape_draft.meas_prop_dcid:
        five_tuple_dcids.append(shape_draft.meas_prop_dcid)
    if shape_draft.meas_qual_dcid:
        five_tuple_dcids.append(shape_draft.meas_qual_dcid)
    if shape_draft.meas_denom_dcid:
        five_tuple_dcids.append(shape_draft.meas_denom_dcid)

    # Both name collection and slot assembly reuse these pairs.
    slot_pairs = pair_slot_drafts(shape_draft.slot_keys, bindings)

    # Lookup order is irrelevant. Result order comes from the original dcid lists.
    name_dcids: set[str] = set(five_tuple_dcids)
    for slot_draft, b_draft in slot_pairs:
        if slot_draft.property_dcid:
            name_dcids.add(slot_draft.property_dcid)
        if b_draft and b_draft.kind in ("value", "set"):
            name_dcids.update(b_draft.value_dcids)
    name_dcids.update(sv_dcids)
    resolved_role_bindings = role_bindings(roles)
    name_dcids.update(dcid for dcid, _role in resolved_role_bindings)
    level_options_by_key: dict[str, tuple[LevelOption, ...]] = {}
    if shape_draft.slot_taxonomy_index_sourced:
        index = get_index()
        query = shape_query(
            shape_draft, bindings, frozenset(role for _dcid, role in resolved_role_bindings)
        )
        for slot_draft, b_draft in slot_pairs:
            prop_dcid = slot_draft.property_dcid
            if prop_dcid is None or not offers_slot(index, query, prop_dcid):
                continue
            # A bound *set* has no single node to scope a menu around. Read it
            # as unbound to use the classification's roots and their children.
            if b_draft and b_draft.kind == "value" and b_draft.value_dcids:
                bound_dcid = b_draft.value_dcids[0]
            else:
                bound_dcid = None
            key = f"{slot_draft.axis}:{prop_dcid}"
            opts = index.level_scoped_options(prop_dcid, bound_dcid)
            level_options_by_key[key] = opts
            name_dcids.update(o.dcid for o in opts)
    else:
        for option_dcids in (shape_draft.slot_taxonomy or {}).values():
            name_dcids.update(option_dcids)

    # Citations keep their first-seen order. The catalogue labels the ones it lists.
    prov_dcids = citable_source_ids(cited_facets(plan))
    name_dcids.update(unlisted_provenances(prov_dcids))
    name_dcids.update(answer_facet_dcids(plan, unit_request))

    labels = graph.node_labels_batch(list(name_dcids))
    # Fill graph label misses from the packaged code index.
    code_index = get_index()
    for dcid in name_dcids:
        if dcid in labels:
            continue
        row = code_index.code_by_dcid(dcid)
        if row is not None:
            labels[dcid] = row.label
    recalled_facts = {
        sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in (shape_draft.sv_arc_facts or {}).items()
    }
    served_facts = {sv: member_facts[sv] for sv in sv_dcids if sv in member_facts}
    evidenced_values: dict[str, set[str]] = {}
    for facts in (recalled_facts, served_facts):
        for key, values in read_slot_taxonomy(
            slot_keys=shape_draft.slot_keys, member_facts=facts
        ).items():
            evidenced_values.setdefault(key, set()).update(values)

    five_tuple_ref_list = _refs_present(five_tuple_dcids, labels)
    five_tuple_refs: dict[str, GraphRef] = {r.dcid: r for r in five_tuple_ref_list}

    # Role names are plain strings, so `_build_entity` uses a dcid-echo ref.
    role_refs: dict[str, GraphRef] = {}

    return _GroundingLookup(
        labels=labels,
        slot_pairs=slot_pairs,
        level_options_by_key=level_options_by_key,
        evidenced_values=evidenced_values,
        prov_dcids=prov_dcids,
        resolved_role_bindings=resolved_role_bindings,
        five_tuple_refs=five_tuple_refs,
        role_refs=role_refs,
    )


def _build_grounded_slots(
    shape_draft: ShapeDraft,
    lookup: _GroundingLookup,
    roles: EntityRoles,
    graph: EngineGraphClient,
    query_entity_text_by_dcid: Mapping[str, str],
) -> tuple[list, list, list[SlotOptions], list[str]]:
    """Build slots, option menus, and inapplicable-property ids from batched lookup data.

    Accepted entity text overrides graph labels for where values. Results follow
    `shape_draft.slot_keys` order.
    """
    labels = lookup.labels
    level_options_by_key = lookup.level_options_by_key

    slots = []
    slot_key_models = []
    slot_options: list[SlotOptions] = []
    inapplicable_slots: list[str] = []
    for slot_draft, b_draft in lookup.slot_pairs:
        prop_dcid = slot_draft.property_dcid
        inapplicable = inapplicable_property_dcid(slot_draft, b_draft)
        if inapplicable is not None:
            inapplicable_slots.append(inapplicable)

        prop_ref: GraphRef | None = None
        if prop_dcid:
            try:
                prop_ref = _ref_or_miss(prop_dcid, labels)
            except GroundingMiss:
                prop_ref = GraphRef(dcid=prop_dcid, label=prop_dcid)

        grounded_vals: list[GraphRef] = []
        if b_draft and b_draft.kind in ("value", "set"):
            if slot_draft.axis == "where":
                grounded_vals = _where_slot_refs(b_draft.value_dcids, labels, roles)
                grounded_vals = [
                    GraphRef(dcid=ref.dcid, label=query_entity_text_by_dcid[ref.dcid])
                    if ref.dcid in query_entity_text_by_dcid
                    else ref
                    for ref in grounded_vals
                ]
            else:
                # A bound value stays when the graph or code index labels it, or the recall
                # arcs or a served member assert it (shown under its dcid). Any other value
                # is one the model invented.
                evidenced = lookup.evidenced_values.get(f"{slot_draft.axis}:{prop_dcid}", set())
                grounded_vals = [
                    GraphRef(dcid=dcid, label=labels.get(dcid, dcid))
                    for dcid in b_draft.value_dcids
                    if dcid in labels or dcid in evidenced
                ]

        # `isPartOf` also models geographic containment, so set detection excludes `where`.
        slot_set_ref: GraphRef | None = None
        if (
            b_draft is not None
            and b_draft.kind == "set"
            and slot_draft.axis in ("what", "how")
            and len(grounded_vals) >= 2
        ):
            slot_set_ref = detect_set_ref(
                value_dcids=[gv.dcid for gv in grounded_vals], graph=graph
            )

        slot = build_slot(
            slot_draft,
            b_draft,
            grounded_vals,
            property_ref=prop_ref,
            set_ref=slot_set_ref,
        )
        slots.append(slot)
        slot_key_models.append(slot.key)

        # Omit empty menus. Index-sourced taxonomies use level-scoped options.
        if prop_dcid is not None:
            taxonomy_key = f"{slot_draft.axis}:{prop_dcid}"
            value_kind = "entity" if slot_draft.axis == "where" else "enum_value"
            if shape_draft.slot_taxonomy_index_sourced:
                level_opts = level_options_by_key.get(taxonomy_key, ())
                opt_by_dcid = {o.dcid: o for o in level_opts}
                option_refs = _refs_present([o.dcid for o in level_opts], labels)
                if option_refs:
                    slot_options.append(
                        SlotOptions(
                            key=slot.key,
                            options=[
                                SlotOption(
                                    value=SlotValue(ref=ref, value_kind=value_kind),
                                    relation=opt_by_dcid[ref.dcid].relation,
                                    member_count=opt_by_dcid[ref.dcid].member_count,
                                    parent=opt_by_dcid[ref.dcid].parent,
                                )
                                for ref in option_refs
                            ],
                            # The level-scoped index menu is exhaustive.
                            truncated=False,
                            source="observed",
                        )
                    )
            else:
                option_dcids = (shape_draft.slot_taxonomy or {}).get(taxonomy_key, [])
                option_refs = _refs_present(option_dcids, labels) if option_dcids else []
                if option_refs:
                    slot_options.append(
                        SlotOptions(
                            key=slot.key,
                            options=[
                                SlotOption(value=SlotValue(ref=ref, value_kind=value_kind))
                                for ref in option_refs
                            ],
                            truncated=True,
                            source="observed",
                        )
                    )

    return slots, slot_key_models, slot_options, inapplicable_slots


def _ground_entities_and_sources(
    sv_dcids: list[str],
    lookup: _GroundingLookup,
    graph: EngineGraphClient,
    query_entity_text_by_dcid: Mapping[str, str],
) -> tuple[list[GraphRef], list, list[ProvenanceRef], list[Warning]]:
    """Build StatVar, entity, and source refs from batched labels. Resolve entity types separately.

    A confirmed entity with no graph label keeps a dcid-echo ref and adds
    `ENTITY_LABEL_UNRESOLVED`. Accepted query text overrides its display label. Identity
    uses the dcid, so entities survive missing labels.
    """
    labels = lookup.labels

    sv_refs = _refs_present(sv_dcids, labels)

    entity_objects_data = []
    unresolved_entity_dcids: list[str] = []
    for dcid, role in lookup.resolved_role_bindings:
        try:
            entity_ref = _ref_or_miss(dcid, labels)
        except GroundingMiss:
            entity_ref = GraphRef(dcid=dcid, label=dcid)
            unresolved_entity_dcids.append(dcid)
        if query_text := query_entity_text_by_dcid.get(dcid):
            entity_ref = GraphRef(dcid=dcid, label=query_text)

        entity_type_ref: GraphRef | None = None
        node_type = graph.node_type(dcid)
        if node_type:
            try:
                entity_type_ref = graphref(node_type, graph=graph)
            except GroundingMiss:
                entity_type_ref = GraphRef(dcid=node_type, label=node_type)

        entity_objects_data.append((role, entity_ref, entity_type_ref))

    source_refs = provenance_refs(lookup.prov_dcids, labels)

    warnings: list[Warning] = []
    if unresolved_entity_dcids:
        warnings.append(
            Warning(
                code=ENTITY_LABEL_UNRESOLVED,
                severity="warn",
                message=(
                    "no ->name label found for entity dcid(s) "
                    f"{', '.join(unresolved_entity_dcids)}; presented with a dcid-echo label."
                ),
            )
        )

    return sv_refs, entity_objects_data, source_refs, warnings


def _ground_answer(
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    sv_dcids: list[str],
    roles: EntityRoles,
    graph: EngineGraphClient,
    plan: ReadPlan | None = None,
    query_entity_text_by_dcid: Mapping[str, str] | None = None,
    *,
    unit_request: IdentityUnitRequest | None = None,
    member_facts: Mapping[str, MemberFacts] | None = None,
) -> GroundedAnswer:
    """Synchronously ground answer refs using one batched label lookup.

    Returns all data needed for model assembly without further blocking graph calls.
    `plan` is the answer's read plan under the request's facet filter
    (`facets.FilteredAnswer.plan`): the source refs are the provenances of the facets it
    serves (`facets.cited_facets`), and none without a plan. The label batch also labels
    the methods and units of those facets and the currency `unit_request`, the request a
    preview converts by, names (`facets.answer_facet_dcids`), and is returned so the caller
    builds `resolved.facets` from it (`facets.answer_facets`) with no further request.
    `query_entity_text_by_dcid` overrides labels for resolver-accepted entity text. It
    changes labels only. `member_facts` are the materialised members' facts
    (`Materialised.member_facts`). Those of `sv_dcids`, the served members, evidence the
    bound constraint values they assert. None before materialise.
    """
    query_entity_text_by_dcid = query_entity_text_by_dcid or {}
    lookup = _ground_answer_lookup(
        shape_draft,
        bindings,
        sv_dcids,
        roles,
        graph,
        plan,
        unit_request,
        member_facts or {},
    )
    slots, slot_key_models, slot_options, inapplicable_slots = _build_grounded_slots(
        shape_draft, lookup, roles, graph, query_entity_text_by_dcid
    )
    sv_refs, entity_objects_data, source_refs, warnings = _ground_entities_and_sources(
        sv_dcids, lookup, graph, query_entity_text_by_dcid
    )
    return (
        lookup.five_tuple_refs,
        lookup.role_refs,
        slots,
        slot_key_models,
        sv_refs,
        entity_objects_data,
        source_refs,
        slot_options,
        inapplicable_slots,
        warnings,
        lookup.labels,
    )


def _build_entity(
    role: EntityRoleKey,
    entity_ref: GraphRef,
    entity_type_ref: GraphRef | None,
    role_refs: dict[str, GraphRef],
) -> Entity:
    """Build a grounded Entity from a role key.

    "observationAbout" builds a plain EntityRoleSubject. "origin" and
    "destination" carry the role-bearing shape's own axis. Declared role
    labels remain presentation data, while identity reads Entity.role.axis.
    """
    if role == "observationAbout":
        return Entity(ref=entity_ref, entity_type=entity_type_ref, role=EntityRoleSubject())
    role_ref = role_refs.get(role, GraphRef(dcid=role, label=role))
    return Entity(
        ref=entity_ref,
        entity_type=entity_type_ref,
        role=EntityRoleDirectional(role=role_ref, axis=role),
    )


def build_identity_entities(
    entity_data: list[tuple[EntityRoleKey, GraphRef, GraphRef | None]],
    role_refs: dict[str, GraphRef],
) -> list[Entity]:
    """Build identity entities from grounded refs without additional graph reads."""
    return [
        _build_entity(role, entity_ref, entity_type_ref, role_refs)
        for role, entity_ref, entity_type_ref in entity_data
    ]


# ---------------------------------------------------------------------------
# Axis-relaxation probe for the _suggest_nearest_real no-data path
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _RelaxationProbeContext:
    """Request-scoped inputs _suggest_nearest_real's own per-binding relaxation
    probe (_relaxed_binding_suggestion) reads on every loop iteration. It is
    immutable after construction and carries every _suggest_nearest_real parameter its probe
    needs except the per-iteration `relaxed` bindings and `relaxation` record,
    which the loop passes alongside this context.
    """

    graph: EngineGraphClient
    sdmx: EngineSdmxClient
    shape_draft: ShapeDraft
    # The original answer's place context and the request's presence read.
    place: PlaceContext
    presence: DefaultPresence | None
    date_request: DateRequest | None
    deadline: float | None
    coverage: CoverageSnapshot
    query_entity_text_by_dcid: Mapping[str, str] | None
    include_option_coverage: bool
    materialise_graph: GraphReader | None
    role_class_walks: RoleClassWalks | None
    text_recognized_properties: frozenset[str]
    measure_elected_properties: frozenset[str]
    defaulted_properties: frozenset[str]
    # The groups the request named, kept because a relaxation keeps their entities.
    named_groups: tuple[NamedGroup, ...]


@dataclasses.dataclass(slots=True, frozen=True)
class SuggestionRead:
    """What a nearest-real suggestion is read over: its read plan, its own place context,
    the original answer's context settled on the suggestion's result
    (`PlaceContext.settled`), as a resubmit of the suggestion settles it, and its members'
    facts (`Materialised.member_facts`), which label its preview rows."""

    plan: ReadPlan | None
    place: PlaceContext
    member_facts: Mapping[str, MemberFacts] | None


def _relaxation_places(place: PlaceContext) -> list[str]:
    """The places a relaxed suggestion is probed at in `place`: the context's probe places,
    else its one bound place."""
    if place.probe_dcids:
        return list(place.probe_dcids)
    bound = place.roles.obs_destination_dcid
    return [bound] if bound else []


@dataclasses.dataclass(frozen=True, slots=True)
class _WalkedRelaxation:
    """The shape a relaxed query answers with, and whether its own member walk ran and
    settled nothing."""

    shape: ShapeDraft
    unverified: bool


def _walked_relaxations(
    ctx: _RelaxationProbeContext, relaxations: Sequence[list[SlotBindingDraft]]
) -> list[_WalkedRelaxation | None]:
    """The shape each relaxed query answers with, in order: `ctx.shape_draft` narrowed to the
    family the mint's walk settles for that query (`elected_role_class_shape`), as the
    relaxed query's first answer narrows its own. The answer's shape holds the family its
    own bindings walked, which a relaxed binding widens. Every relaxation is walked, side by
    side, so the no-data path waits for one walk however many relaxations there are, and
    the caller then probes them in order. A walk that settles
    nothing leaves the shape's members to answer and marks the relaxation `unverified`, as
    the relaxed query's first answer warns `membership_unverified`. `None` marks a
    relaxation whose identity the mint refuses, or whose walk raised, which yields no
    suggestion. Without `ctx.role_class_walks` every relaxation answers with
    `ctx.shape_draft` and is not marked: the code index decides the relaxed query's members
    exactly when it decides the answer's, since relaxing a binding keeps its slot, so the
    answer's own reading of its members, and its warning, stand for the relaxation's."""
    walks = ctx.role_class_walks
    if walks is None or not relaxations:
        return [_WalkedRelaxation(ctx.shape_draft, unverified=False)] * len(relaxations)
    shape = ctx.shape_draft

    def walked(relaxed: list[SlotBindingDraft]) -> _WalkedRelaxation | None:
        try:
            resolution = elected_role_class_shape(
                shape,
                shape_query(
                    shape, relaxed, entity_role_keys(shape, ctx.place.roles.resolved_entities)
                ),
                walks=walks,
                arcs_graph=ctx.graph,
                deadline=ctx.deadline,
            )
        except Exception:
            logger.debug("nearest-real relaxation walk failed", exc_info=True)
            return None
        return _WalkedRelaxation(resolution.value, unverified=not resolution.authority.settled)

    with ThreadPoolExecutor(
        max_workers=len(relaxations), thread_name_prefix="nearest-real-walk"
    ) as pool:
        futures = [submit_with_context(pool, walked, relaxed) for relaxed in relaxations]
        return [future.result() for future in futures]


def _relaxations_presence(
    ctx: _RelaxationProbeContext, shapes: Sequence[ShapeDraft | None]
) -> DefaultPresence | None:
    """The presence the relaxations of a default-scoped answer are probed over: `ctx.presence`
    topped up, in one presence read (`place.read_default_presence`), with the members of
    the subject-only `shapes` it does not cover, since a walk can widen a relaxation's
    family past the members the request's read covered. Off a default scope, or with
    every member covered, `ctx.presence` itself. A read that fails leaves `ctx.presence`,
    and a relaxation whose members it lacks yields no suggestion."""
    from search.engine.place import read_default_presence

    presence = ctx.presence
    if presence is None or ctx.place.default_scope is None:
        return presence
    missing = list(
        dict.fromkeys(
            sv
            for shape in shapes
            if shape is not None and not is_role_bearing(shape)
            for sv in shape.sv_arc_facts or {}
            if sv not in presence.scopes
        )
    )
    if not missing:
        return presence
    try:
        return read_default_presence(
            graph=ctx.graph, sdmx=ctx.sdmx, stat_vars=missing, known=presence, deadline=ctx.deadline
        )
    except Exception:
        logger.warning("nearest-real relaxations' presence read failed", exc_info=True)
        return presence


def _relaxed_binding_suggestion(
    ctx: _RelaxationProbeContext,
    shape: ShapeDraft,
    relaxed: list[SlotBindingDraft],
    relaxation: Relaxation,
) -> tuple[NearestRealSpec | None, SuggestionRead | None, list[Warning] | None, Warning | None]:
    """Try one relaxed binding over `shape`, the shape its own query answers with
    (`_walked_relaxations`), and return `(suggestion, what it is read over,
    ground_warnings, probe_warning)`.

    No data, no StatVars, or any exception returns no suggestion. Ground warnings are None
    until grounding runs. Probe warnings report option-coverage failures, from a probe over
    the suggestion's own places. This optional no-data probe fails open and leaves the
    resolved response intact.
    """
    from search.engine.retrieval.retrieve import Materialised
    from search.engine.retrieval.role_resolve import graph_confirm_resolve

    roles = ctx.place.roles
    ground_warnings: list[Warning] | None = None
    probe_warning: Warning | None = None
    try:
        # A standard shape resolves as the standard route answers it, over the original
        # answer's entities and places or its default-scope presence read; a role-bearing
        # shape confirms against coverage.
        mat = (
            graph_confirm_resolve(
                shape=shape,
                bindings=relaxed,
                subject_dcids=_relaxation_places(ctx.place),
                origin_dcids=roles.origin_dcids,
                graph=ctx.graph,
                sdmx=ctx.sdmx,
                date_request=ctx.date_request,
                deadline=ctx.deadline,
                coverage_snapshot=ctx.coverage,
            )
            if is_role_bearing(shape)
            else standard_resolve(
                shape=shape,
                bindings=relaxed,
                entities=roles.resolved_entities,
                graph=ctx.graph,
                date_request=ctx.date_request,
                place_dcids=ctx.place.probe_places,
                default_presence=ctx.place.presence_for(ctx.presence),
            )
        )
        if not isinstance(mat, Materialised):
            return None, None, None, None
        # A flagged-unavailable carrier represents the original binding.
        if not mat.has_data:
            return None, None, None, None
        place = ctx.place.settled(mat)

        ground = _ground_answer(
            shape,
            relaxed,
            mat.sv_dcids,
            roles.roles,
            ctx.graph,
            mat.read_plan,
            ctx.query_entity_text_by_dcid,
            member_facts=mat.member_facts,
        )

        (
            ft_refs,
            role_refs,
            slots,
            slot_key_models,
            sv_refs,
            entity_data,
            source_refs,
            slot_opts,
            _draft_inapplicable_slots,  # unused: draft-kind-derived, see below
            ground_warnings,
            labels,
        ) = ground
        if not sv_refs:
            return None, None, ground_warnings, None

        # Use the probe's resolved facts. Index-backed shape arcs may cover another population.
        inapplicable_slots = inapplicable_slots_from_member_facts(
            slots=slots,
            sv_dcids=mat.sv_dcids,
            member_facts=mat.member_facts or {},
        )
        unresolved_slots = unresolved_slots_from_member_facts(
            slots=slots,
            sv_dcids=mat.sv_dcids,
            member_facts=mat.member_facts or {},
        )

        # Count the admitted population, not only the elected StatVars.
        shape_model = build_shape_model(
            shape,
            slot_key_models,
            ft_refs,
            member_count=(mat.population.admitted if mat.population is not None else len(sv_refs)),
            members=mat.served_members,
            member_digest=compute_member_digest(mat.population),
        )
        stat_vars_list = build_stat_vars(
            sv_refs,
            shape.shape_id,
            slots,
            plan=mat.read_plan,
            destination_confirmed=mat.destination_confirmed,
            member_facts=mat.member_facts or {},
        )
        entities_list = [_build_entity(rd, er, etr, role_refs) for rd, er, etr in entity_data]
        slots = bind_when_slot(slots, window=mat.coverage.window)
        slots = apply_slot_expansions(slots, slot_expansions(relaxed))

        available_at_place, confidence = _place_confidence(
            destination_dcid=roles.obs_destination_dcid,
            available_at_place=mat.available_at_place,
            representative_score=shape.representative_score,
            retrieved=shape.retrieved,
        )
        if ctx.include_option_coverage and slot_opts:
            # Keep the leaf dependency direction. Enrichment failure leaves options uncovered.
            #
            # Use the shape's declared roles, matching the primary resolve boundary.
            # An unplaced entity skips optional enrichment rather than the suggestion. The
            # probe reads every place the suggestion is read over, so no one named place
            # decides it.
            _probe_entities: dict[str, list[str]] | None
            _probe_places = _relaxation_places(place)
            try:
                if is_role_bearing(shape):
                    _probe_entities = entities_by_role(
                        shape=shape,
                        by_axis={"destination": _probe_places, "origin": roles.origin_dcids},
                    )
                else:
                    _probe_entities = entities_by_role(shape=shape, subject=_probe_places)
            except RoleEntityUnplaced:
                logger.warning(
                    "option coverage probe skipped for a nearest_real suggestion, "
                    "shape_id=%r: the resolved entity has no role to bind under",
                    shape.shape_id,
                )
                _probe_entities = None
            if _probe_entities is not None:
                try:
                    slot_opts = standard_option_coverage(
                        shape=shape,
                        slot_options=slot_opts,
                        entities=_probe_entities,
                        materialise_graph=ctx.materialise_graph,
                        date_request=ctx.date_request,
                        place_dcids=_probe_places,
                        datasets=elected_datasets(mat.member_facts),
                        query=shape_query(shape, relaxed, entity_role_keys(shape, _probe_entities)),
                    )
                except Exception:
                    logger.warning(
                        "option_coverage probe failed for a nearest_real suggestion, "
                        "shape_id=%r; returning uncovered slot_options",
                        shape.shape_id,
                        exc_info=True,
                    )
                    probe_warning = Warning(
                        code=OPTION_COVERAGE_PROBE_FAILED,
                        severity="warn",
                        message=(
                            "option coverage probe failed for a nearest_real suggestion "
                            f"on shape {shape.shape_id!r}; slot options returned "
                            "without coverage."
                        ),
                    )

        spec = build_spec(
            shape=shape_model,
            slots=slots,
            stat_vars=stat_vars_list,
            entities=entities_list,
            coverage=mat.coverage,
            plan=mat.read_plan,
            pipeline_trace=[],
            coverage_index=ctx.coverage.index,
            resolved_sources=source_refs,
            facets=answer_facets(
                mat.read_plan,
                labels,
                bound_axes={},
                request_basis=None,
                unit_request=None,
                currency_of=functools.partial(unit_currency, ctx.graph, deadline=ctx.deadline),
            ),
            slot_options=slot_opts,
            available_at_place=available_at_place,
            confidence=confidence,
            requested_window=requested_window_from_date_request(ctx.date_request),
            inapplicable_slots=inapplicable_slots,
            unresolved_slots=unresolved_slots,
            text_recognized_properties=ctx.text_recognized_properties,
            measure_elected_properties=ctx.measure_elected_properties,
            defaulted_properties=ctx.defaulted_properties,
            named_groups=ctx.named_groups,
            # These counts describe the confirmed half of the same population record.
            stat_vars_member_count=(
                mat.population.confirmed if mat.population is not None else None
            ),
            stat_vars_member_count_exact=(
                mat.population.exact if mat.population is not None else True
            ),
        )
        spec = apply_declaration_version(spec, relaxed)
    except Exception:
        return None, None, ground_warnings, probe_warning

    return (
        NearestRealSpec(
            spec_id=spec.spec_id,
            definition=spec.definition,
            resolved=spec.resolved,
            relaxation=relaxation,
        ),
        SuggestionRead(plan=mat.read_plan, place=place, member_facts=mat.member_facts),
        ground_warnings,
        probe_warning,
    )


def _suggest_nearest_real(
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    shape_draft: ShapeDraft,
    bindings: list[SlotBindingDraft],
    place: PlaceContext,
    presence: DefaultPresence | None,
    date_request: DateRequest | None,
    deadline: float | None = None,
    include_option_coverage: bool = False,
    materialise_graph: GraphReader | None = None,
    role_class_walks: RoleClassWalks | None = None,
    query_entity_text_by_dcid: Mapping[str, str] | None = None,
    text_recognized_properties: frozenset[str] = frozenset(),
    measure_elected_properties: frozenset[str] = frozenset(),
    defaulted_properties: frozenset[str] = frozenset(),
    named_groups: tuple[NamedGroup, ...] = (),
    n_max: int = 2,
    coverage: CoverageSnapshot,
) -> tuple[list[NearestRealSpec], list[Warning], dict[str, SuggestionRead], DefaultPresence | None]:
    """Relax one value/set binding at a time and return up to `n_max` confirmed suggestions,
    the warnings, what each suggestion is read over by spec id, and the presence the
    relaxations were probed over (`_relaxations_presence`), whose `unsettled` StatVars keep
    the answer out of the response cache.

    Runs synchronously on the no-observations path. Every relaxation first walks its own
    members with `role_class_walks`, all in one round, as the relaxed query's first answer
    does (`_walked_relaxations`), and is then probed in query order, in `place`, the
    original answer's place context, until `n_max` suggestions are found. A role-bearing
    shape confirms it against coverage at the context's probe places (else its observation
    destination) and origins; a standard shape resolves it through
    `standard.standard_resolve` over the context's entities and probe places, or `presence`
    on a default scope. The request's presence read covers every recalled member, so the
    relaxed queries read no scope beside the walks; one read after them adds the members
    the walks add (`_relaxations_presence`). A suggestion is what the standard route
    answers: a placeless shape's relaxation is probed over the default scope of the members
    it admits and settled on the StatVars it serves, as a resubmit of the suggestion is.
    Each suggestion is read over its own settled context (`SuggestionRead.place`). A
    suggestion whose walk settled nothing adds `membership_unverified`, as that first answer
    warns it. Each probe uses `deadline`, and any error yields no suggestion.
    Option-coverage failures add warnings. Callers fetch observations separately, from each
    suggestion's plan and places.
    `text_recognized_properties`, `measure_elected_properties`, and `defaulted_properties`
    preserve the original query's slot provenance after relaxation, and `named_groups` the
    groups the request named, which a suggestion states for the roles that still bind
    exactly their members.
    """
    ctx = _RelaxationProbeContext(
        graph=graph,
        sdmx=sdmx,
        shape_draft=shape_draft,
        place=place,
        presence=presence,
        date_request=date_request,
        deadline=deadline,
        coverage=coverage,
        query_entity_text_by_dcid=query_entity_text_by_dcid,
        include_option_coverage=include_option_coverage,
        materialise_graph=materialise_graph,
        role_class_walks=role_class_walks,
        text_recognized_properties=text_recognized_properties,
        measure_elected_properties=measure_elected_properties,
        defaulted_properties=defaulted_properties,
        named_groups=named_groups,
    )
    suggestions: list[NearestRealSpec] = []
    reads: dict[str, SuggestionRead] = {}
    probe_warnings: list[Warning] = []
    # The shared role map produces the same label warning on every iteration. Include it once.
    _entity_warnings: list[Warning] | None = None

    # Relax StatVar constraints only. The suggestion keeps its requested entity.
    relaxations: list[tuple[list[SlotBindingDraft], Relaxation]] = []
    for i, binding in enumerate(bindings):
        if binding.kind not in ("value", "set") or binding.axis in ("where", "when", "source"):
            continue
        relaxed = list(bindings)
        relaxed[i] = SlotBindingDraft(
            axis=binding.axis,
            property_dcid=binding.property_dcid,
            kind="unbound",
            value_dcids=[],
        )
        relaxation = Relaxation(
            axis=binding.axis,
            property_dcid=binding.property_dcid,
            dropped_value_dcids=list(binding.value_dcids),
        )
        relaxations.append((relaxed, relaxation))
    # One round of walks, so the no-data path waits for one walk: there is one relaxation,
    # and so at most one walk, per value or set binding outside where, when and source, at
    # most one per slot key of the shape. The request's presence read covers every recalled
    # member, so no scope is read beside the walks; a walk that settles a family can add
    # members, whose presence one read after the walks adds.
    walked = _walked_relaxations(ctx, [relaxed for relaxed, _relaxation in relaxations])
    ctx = dataclasses.replace(
        ctx, presence=_relaxations_presence(ctx, [w.shape if w else None for w in walked])
    )

    unverified = False
    for (relaxed, relaxation), walk in zip(relaxations, walked, strict=True):
        if len(suggestions) >= n_max:
            break
        if walk is None:
            continue
        suggestion, read, ground_warnings, probe_warning = _relaxed_binding_suggestion(
            ctx, walk.shape, relaxed, relaxation
        )
        if _entity_warnings is None and ground_warnings is not None:
            _entity_warnings = ground_warnings
        if probe_warning is not None:
            probe_warnings.append(probe_warning)
        if suggestion is not None and read is not None:
            suggestions.append(suggestion)
            unverified = unverified or walk.unverified
            reads[suggestion.spec_id] = read

    if _entity_warnings:
        probe_warnings.extend(_entity_warnings)
    if unverified:
        probe_warnings.append(membership_unverified_warning())

    return suggestions, probe_warnings, reads, ctx.presence
