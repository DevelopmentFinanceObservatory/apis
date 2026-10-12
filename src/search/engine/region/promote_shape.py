"""Reconstruct the resolved group's ShapeDraft from its re-read arc facts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from materialise.graph import UNRESOLVED_NODE
from search.engine.config import settings
from search.engine.envelope import now_ms
from search.engine.errors import (
    EngineInfraError,
    FamilyNotResolvableError,
    MembershipTooBroadError,
    RoleSignatureError,
)
from search.engine.region.arcs import declared_props_from_arcs
from search.engine.region.grounding import _shape_with_taxonomy, membership_unverified_warning
from search.engine.region.promote_results import _no_data_region, _scope_too_broad_region
from search.engine.region.promote_state import _PromoteState
from search.engine.region.result import RegionResult
from search.engine.retrieval.bound_values import elected_role_class_shape, identity_query
from search.engine.retrieval.derive import index_shape_label
from search.engine.retrieval.facts import (
    FiveTuple,
    group_roles,
    read_constraints,
    read_name,
)
from search.engine.shape import STANDARD_SHAPE_LABEL, shape_draft_from
from statvar_members import five_tuple_arcs_match, five_tuple_reverse_filters


def _confirm_axes_against_family(
    axes: Sequence[str],
    arcs_by_dcid: Mapping[str, dict[str, tuple[str, ...]] | None],
    *,
    ft: FiveTuple,
    measurement_qualifiers: frozenset[str | None],
) -> dict[str, list[str]]:
    """Return sampled values for axes declared by exact-family members.

    Reverse-filter results need client confirmation for `populationType`, `statType`,
    `measurementQualifier`, and `measurementDenominator`; `five_tuple_arcs_match` checks them.
    The query fixes `measuredProperty`.
    An axis is confirmed when `constraintProperties` names it, even if its values are
    unresolved. Return resolved values in graph order, excluding `UNRESOLVED_NODE`. Samples
    span the shape family without place scoping and may be incomplete when the subject page
    is truncated."""
    confirmed: dict[str, list[str]] = {}
    for arcs in arcs_by_dcid.values():
        if arcs is None:
            continue
        if not any(
            five_tuple_arcs_match(
                arcs,
                population_type=ft.pop_type_dcid or None,
                measured_property=None,
                stat_type=ft.stat_type_dcid or None,
                measurement_qualifier=measurement_qualifier,
                measurement_denominator=ft.meas_denom_dcid,
                denominator_unresolved=ft.meas_denom_unresolved,
            )
            for measurement_qualifier in measurement_qualifiers
        ):
            continue
        constraint_names = arcs.get("constraintProperties", ())
        for axis in axes:
            if axis not in constraint_names:
                continue
            values = confirmed.setdefault(axis, [])
            for value in arcs.get(axis, ()):
                if value != UNRESOLVED_NODE and value not in values:
                    values.append(value)
    return confirmed


def _phase_build_shape(state: _PromoteState) -> RegionResult | None:
    """Rebuild the ShapeDraft from resolved members' arcs.

    Fold constraints across resolved StatVars, limited to posted slot properties. Preserve
    every posted key; axes outside resolved declarations stay unbound and may receive sampled
    taxonomy values. Confirm them against the five-tuple family with one `node_subjects` and
    one `node_arcs_many` read. That batch read follows `nextToken`; EngineGraphClient's batch
    read does not. The lookup is skipped when `measuredProperty` is absent; a missing materialise
    client otherwise raises `EngineInfraError`. Only confirmed axes receive sampled taxonomy.
    Role sets the election cannot order return `no_data(role_signature_incompatible)`.
    Otherwise build the shape with observed-union taxonomy. The shape holds the family the
    mint resolves (`elected_role_class_shape`), whose arcs join `resolved_arcs`. A family
    the mint's walk does not settle stays as elected, under a warning."""
    t0 = now_ms()
    try:
        # This phase's warnings join those earlier phases raised, and a no_data return
        # carries them all.
        warnings = state.warnings

        # Use the first resolved member as representative; build shape facts from the full
        # resolved set. This draft is needed before place expansion to identify role shapes.
        state.rep_sv = state.resolved_stat_var_dcids[0]

        # Union constraints across resolved members; a representative may omit an axis the
        # shape declares. Preserve first-seen order.
        #
        # Declared keys come from the posted identity; other member constraints remain open
        # axes.
        identity_slots = state.inp.identity.slots
        posted_slot_props = {identity_slot.property for identity_slot in identity_slots}
        prop_observed_values: dict[str, list[str]] = {}
        for sv in state.resolved_stat_var_dcids:
            sv_c = read_constraints(state.resolved_arcs[sv])
            for prop, val in sv_c.items():
                if prop not in posted_slot_props:
                    continue
                prop_observed_values.setdefault(prop, []).append(val)

        # Keep every posted slot in the shape and its identity hash. Confirm axes absent
        # from resolved members against the family; their samples feed slot taxonomy separately
        # because `sv_arc_facts` stays scoped to resolved members.
        declared_props = declared_props_from_arcs(
            sv_dcids=state.resolved_stat_var_dcids, arcs_by_sv=state.resolved_arcs
        )
        unconfirmed_axes: list[str] = []
        for identity_slot in identity_slots:
            prop = identity_slot.property
            if prop in declared_props or prop in unconfirmed_axes:
                continue
            unconfirmed_axes.append(prop)

        # Confirm undeclared posted axes for slot-option taxonomy. Without measuredProperty,
        # The family query requires measuredProperty; a missing client is a wiring error.
        axis_taxonomy_additions: dict[str, list[str]] = {}
        if unconfirmed_axes and state.ft.meas_prop_dcid:
            if state.materialise_graph is None:
                raise EngineInfraError(
                    "materialise graph client not initialised; cannot confirm "
                    f"posted identity's declared axes {unconfirmed_axes!r} against "
                    "this shape's own five-tuple family"
                )
            family_filters = five_tuple_reverse_filters(
                population_type=state.ft.pop_type_dcid,
                stat_type=state.ft.stat_type_dcid,
                measurement_qualifier=state.ft.meas_qual_dcid,
                measurement_denominator=state.ft.meas_denom_dcid,
            )
            family_page = state.materialise_graph.node_subjects(
                state.ft.meas_prop_dcid,
                property="measuredProperty",
                filters=family_filters,
                max_results=settings().resubmit_axis_sample,
            )
            family_arcs = state.materialise_graph.node_arcs_many(
                family_page.dcids,
                properties=(
                    "constraintProperties",
                    "populationType",
                    "statType",
                    "measurementQualifier",
                    "measurementDenominator",
                    *unconfirmed_axes,
                ),
            )
            confirmed = _confirm_axes_against_family(
                unconfirmed_axes,
                family_arcs,
                ft=state.ft,
                measurement_qualifiers=state.family_measurement_qualifiers,
            )
            axis_taxonomy_additions = {
                prop: sample_values for prop, sample_values in confirmed.items() if sample_values
            }

        # Preserve unbound slots in the identity hash. Confirmed sample values belong in
        # Store family samples in slot taxonomy; resolved-member values stay empty for axes
        # those members do not assert.
        for identity_slot in identity_slots:
            prop_observed_values.setdefault(identity_slot.property, [])

        constraint_props = list(prop_observed_values.keys())

        # Read slot-property labels from the graph, with the population type's when the
        # index family labels the shape.
        label_dcids = constraint_props.copy()
        if state.index_family is not None:
            label_dcids.append(state.ft.pop_type_dcid)
        prop_label_map = state.graph.node_labels_batch(label_dcids) if label_dcids else {}
        prop_labels = {p: prop_label_map.get(p, p) for p in constraint_props}

        sv_arc_facts = state.sv_arc_facts = {
            sv: state.resolved_arcs[sv] for sv in state.resolved_stat_var_dcids
        }

        # A five-tuple omits observationProperties, so the role class is elected separately.
        query = identity_query(state.inp.identity)
        walks = state.role_class_walks
        try:
            roles = group_roles(sv_arc_facts)
        except RoleSignatureError:
            return _no_data_region(
                state.variable_text, "role_signature_incompatible", warnings=tuple(warnings)
            )

        # Label an index shape as derive does, else by the representative's name arc with the
        # standard label as fallback. The shape id carries identity; `_shape_with_taxonomy`
        # supplies slot options.
        if state.index_family is not None:
            pop_type = state.ft.pop_type_dcid
            shape_label = index_shape_label(
                state.index_family, population_type_label=prop_label_map.get(pop_type, pop_type)
            )
        else:
            shape_label = read_name(sv_arc_facts[state.rep_sv]) or STANDARD_SHAPE_LABEL

        state.shape_draft = _shape_with_taxonomy(
            shape_draft_from(
                shape_id=state.regenerated_shape_id,
                label=shape_label,
                pop_type_dcid=state.ft.pop_type_dcid,
                meas_prop_dcid=state.ft.meas_prop_dcid,
                stat_type_dcid=state.ft.stat_type_dcid,
                meas_qual_dcid=state.ft.meas_qual_dcid,
                meas_denom_dcid=state.ft.meas_denom_dcid,
                meas_denom_unresolved=state.ft.meas_denom_unresolved,
                constraint_props=constraint_props,
                prop_labels=prop_labels,
                prop_observed_values=prop_observed_values,
                sv_arc_facts=sv_arc_facts,
                # These arcs form the complete elected family.
                sv_arc_facts_complete=True,
                declared_roles=roles.declared,
                optional_roles=roles.optional,
                # The identity elected these members, so no recall or phrase score applies.
                recalled=False,
                representative_score=None,
                retrieved=False,
            )
        )

        # The mint serves the antichain-minimal members of the family its walk resolves, so
        # the shape holds that family.
        try:
            served = elected_role_class_shape(
                state.shape_draft,
                query,
                walks=walks,
                arcs_graph=state.graph,
                deadline=state.deadline,
            )
        except RoleSignatureError:
            return _no_data_region(
                state.variable_text, "role_signature_incompatible", warnings=tuple(warnings)
            )
        except FamilyNotResolvableError:
            return _no_data_region(
                state.variable_text, "variable_not_resolved", warnings=tuple(warnings)
            )
        except MembershipTooBroadError as exc:
            return _scope_too_broad_region(
                state.variable_text, state.inp.identity, str(exc), warnings=tuple(warnings)
            )
        if not served.authority.settled:
            warnings.append(membership_unverified_warning())
        state.shape_draft = served.value
        state.family_settled = served.authority.settled
        state.walked = served.walked
        state.resolved_arcs = {**(served.value.sv_arc_facts or {}), **state.resolved_arcs}

        # Merge family samples for axes absent from the resolved members into slot taxonomy.
        if axis_taxonomy_additions:
            merged_taxonomy = dict(state.shape_draft.slot_taxonomy or {})
            for slot_key in state.shape_draft.slot_keys:
                values = axis_taxonomy_additions.get(slot_key.property_dcid or "")
                if values:
                    merged_taxonomy[f"{slot_key.axis}:{slot_key.property_dcid}"] = values
            state.shape_draft = dataclasses.replace(
                state.shape_draft, slot_taxonomy=merged_taxonomy
            )

        return None
    finally:
        state.timing["shape"] = now_ms() - t0
