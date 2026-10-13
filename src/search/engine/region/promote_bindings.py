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
from search.engine.region.unrealisable import disclose_posted_values, drop_posted_values
from search.engine.retrieval.bound_values import (
    _expand_values,
    identity_query,
    members_by_value,
    source_binding,
)
from search.engine.retrieval.facts import read_constraints


def _distinct_asserted_values(
    stat_var_dcids: list[str], constraints_by_sv: dict[str, dict[str, str]], prop: str
) -> list[str]:
    """Return distinct asserted values in SV order, skipping members without `prop`."""
    return list(
        dict.fromkeys(
            constraints_by_sv[sv][prop] for sv in stat_var_dcids if prop in constraints_by_sv[sv]
        )
    )


def _covered_by_posted_binding(posted: list[str], vals: list[str]) -> bool:
    """Return whether every value is posted or its descendant; an empty `posted` set covers none.

    Use containment, since sparse rollups rarely have data at every leaf. Flat code values use
    equality because their expansion is empty."""
    if not posted:
        return False
    return set(vals) <= set(_expand_values(get_index(), posted))


def _phase_build_bindings(state: _PromoteState) -> None:
    """Build where and constraint bindings from the resolved StatVars and posted bindings.

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

        # Multi-value edits use requested order, filtered to values final survivors assert
        # themselves or below (a posted rollup's members carry its leaves, as the narrow
        # admits them). Other constraints are reconstructed from those survivors' own arcs.
        index = get_index()
        resolved_constraints_by_sv = {
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
                    posted = multi_value_edits[b.property_dcid]
                    filtered_vals = [
                        v for v in posted if not asserted.isdisjoint(_expand_values(index, [v]))
                    ]
                    if uncarried := [v for v in posted if v not in filtered_vals]:
                        drop_posted_values(state, b.property_dcid, uncarried)
                    if len(filtered_vals) >= 2:
                        b.kind = "set"
                        b.value_dcids = filtered_vals
                    else:
                        b.kind = "value" if filtered_vals else "unbound"
                        b.value_dcids = filtered_vals

        # A single edited value is bound from the representative's own arcs, which hold
        # the posted value, a value below it, or another value when the elected members
        # stood without a narrow (`promote_survivors._narrowed_members`). A draft holding
        # anything else goes unbound, and the resubmit refuses on the posted value with
        # this relaxed read flagged (`unrealisable.drop_posted_values`).
        for prop, vals in state.edits.items():
            if len(vals) != 1:
                continue
            posted_or_below = {*vals, *_expand_values(index, vals)}
            drafts = [b for b in bindings if b.property_dcid == prop]
            if all(b.value_dcids and set(b.value_dcids) <= posted_or_below for b in drafts):
                continue
            drop_posted_values(state, prop, vals)
            for b in drafts:
                b.kind = "unbound"
                b.value_dcids = []

        # Rebuild unedited constraints from final survivors; correlated axes can lose values
        # after a narrow. Members assert rollup leaves, so preserve a posted rollup only when
        # survivors assert its full value set and descendants cover those values. A posted
        # value no survivor carries, itself or below (`members_by_value`), departs: every
        # value the members do carry lies inside the posted set, so edit detection saw
        # nothing, and the answer reads the others (`unrealisable.disclose_posted_values`).
        for b in bindings:
            prop = b.property_dcid
            if prop is None or prop in state.edits:
                continue
            vals = _distinct_asserted_values(
                state.resolved_stat_var_dcids, resolved_constraints_by_sv, prop
            )
            posted = state.posted_bound_values.get(prop)
            if posted is not None:
                carrying = members_by_value(
                    {sv: c.get(prop) for sv, c in resolved_constraints_by_sv.items()}, posted
                )
                if uncarried := [v for v in posted if not carrying[v]]:
                    disclose_posted_values(state, prop, uncarried)
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
