"""Build where and arc-derived constraint bindings for a standard resubmit."""

from __future__ import annotations

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.envelope import now_ms
from search.engine.region.arcs import standard_bindings_from_arcs
from search.engine.region.promote_state import _PromoteState
from search.engine.region.resubmit_request import (
    _date_request_for_resubmit,
    _reject_reversed_resubmit_window,
)
from search.engine.region.resubmit_seed import _distinct_asserted_values
from search.engine.retrieval.bound_values import _expand_values, identity_query, source_binding
from search.engine.retrieval.facts import read_constraints


def _covered_by_posted_binding(posted: list[str], vals: list[str]) -> bool:
    """Return whether every value is posted or its descendant; an empty `posted` set covers none.

    Use containment, since sparse rollups rarely have data at every leaf. Flat code values use
    equality because their expansion is empty."""
    if not posted:
        return False
    return set(vals) <= set(_expand_values(get_index(), posted))


def _phase_build_bindings(state: _PromoteState) -> None:
    """Build where and constraint bindings from the resolved StatVars and posted edits.

    Preserve multi-place sets, rebuild unedited constraints from final survivors, apply
    explicit widenings, and resolve the requested window. Validation runs before graph reads."""
    t0 = now_ms()
    try:
        # Multi-place where values use a set binding; one or zero places use a value binding.
        bindings: list[SlotBindingDraft] = []
        state.bindings = bindings
        if len(state.standard_where_dcids) >= 2:
            bindings.append(
                SlotBindingDraft(
                    axis="where",
                    property_dcid=None,
                    kind="set",
                    value_dcids=list(state.standard_where_dcids),
                )
            )
        elif state.entity_dcid:
            bindings.append(
                SlotBindingDraft(
                    axis="where",
                    property_dcid=None,
                    kind="value",
                    value_dcids=[state.entity_dcid],
                )
            )
        bindings.extend(
            standard_bindings_from_arcs(shape=state.shape_draft, rep_sv_dcid=state.rep_sv)
        )

        # Multi-value edits use requested order, filtered to values asserted by final survivors.
        # Other constraints are reconstructed from those survivors' own arcs below.
        resolved_constraints_by_sv = state.resolved_constraints_by_sv = {
            sv: read_constraints(state.resolved_arcs[sv]) for sv in state.resolved_stat_var_dcids
        }
        multi_value_edits = {prop: vals for prop, vals in state.edits.items() if len(vals) > 1}
        if multi_value_edits:
            for b in bindings:
                if b.property_dcid in multi_value_edits:
                    asserted = set(
                        _distinct_asserted_values(
                            state.resolved_stat_var_dcids,
                            resolved_constraints_by_sv,
                            b.property_dcid,
                        )
                    )
                    # Preserve requested order and keep only values backed by final survivors.
                    filtered_vals = [v for v in multi_value_edits[b.property_dcid] if v in asserted]
                    if len(filtered_vals) >= 2:
                        b.kind = "set"
                        b.value_dcids = filtered_vals
                    else:
                        b.kind = "value"
                        b.value_dcids = filtered_vals

        # Rebuild unedited constraints from final survivors; correlated axes can lose values
        # after a narrow or refine. Members assert rollup leaves, so preserve a posted rollup
        # only when survivors assert its full value set and descendants cover those values.
        for b in bindings:
            prop = b.property_dcid
            if prop is None or prop in state.edits:
                continue
            vals = _distinct_asserted_values(
                state.resolved_stat_var_dcids, resolved_constraints_by_sv, prop
            )
            posted = state.posted_bound_values.get(prop)
            if (
                posted is not None
                and set(vals) == state.union_constraints.get(prop, set())
                and _covered_by_posted_binding(posted, vals)
            ):
                b.kind = "set" if len(posted) >= 2 else "value"
                b.value_dcids = posted
                continue
            if len(vals) >= 2:
                b.kind = "set"
                b.value_dcids = vals
            elif len(vals) == 1:
                b.kind = "value"
                b.value_dcids = vals
            # No survivor asserts the property; keep the draft unbound.

        # Explicit widenings override arc-derived values. Election-forced properties are
        # excluded from widened_props and retain their elected bindings.
        # Posted bindings and widenings take precedence; resubmits lack the text evidence that
        # triggers declared defaults.
        for b in bindings:
            if b.property_dcid in state.widened_props:
                b.kind = "unbound"
                b.value_dcids = []

        # The dataset the posted identity names rides in the bindings, where every family
        # read after this phase takes it from.
        source = identity_query(state.inp.identity).source
        if source is not None:
            bindings.append(source_binding(source))

        # identity.requested_window == "latest" re-resolves to the current
        # latest year.
        date_request = state.date_request = _date_request_for_resubmit(state.inp)
        _reject_reversed_resubmit_window(state.inp, date_request)
    finally:
        state.timing["bind"] = now_ms() - t0
