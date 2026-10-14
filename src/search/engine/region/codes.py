"""Warning codes and probe-result aliases shared by the region package.

This leaf module depends on ``search.engine.events`` and ``search.models``.
"""

from __future__ import annotations

from search.engine.events import (
    EntityOptionsEvent,
    FacetOptionsEvent,
    ObservationsEvent,
    SlotOptionsEvent,
)
from search.models import NoDataReason, Warning, elect_no_data_reason

RETRIEVAL_SCORE_WEAK = "retrieval_score_weak"
# An alternate reading that passed the admission guards was not evaluated: the presence
# read its default scope needs did not complete. The elected answer is served and the
# response is not cached.
ALTERNATES_UNEVALUATED = "alternates_unevaluated"
# The elected shape had no data and only an alternate survived. Return that
# Return the surviving shape through NoData.alternate_match.
ONLY_ALTERNATE_HAS_DATA = "only_alternate_has_data"
# The elected shape had no data and an alternate answered. This warning
# accompanies candidates with 2+ survivors, or a definite response with one
# measure-sibling survivor. A lone non-sibling alternate uses
# ONLY_ALTERNATE_HAS_DATA instead.
ELECTED_SHAPE_NO_DATA = "elected_shape_no_data"
# NoData.gate codes explain why a near-match is flagged. NO_MEMBER_HAS_DATA_AT_PLACE
# means no family member has data at the resolved place. NO_MEMBER_ASSERTS_BOUND_PROPERTIES
# means none asserts every bound constraint without a place. Both use flagged_match.
# ONLY_ALTERNATE_HAS_DATA uses alternate_match because no elected-shape Spec exists.
NO_MEMBER_HAS_DATA_AT_PLACE = "no_member_has_data_at_place"
NO_MEMBER_ASSERTS_BOUND_PROPERTIES = "no_member_asserts_bound_properties"
# Option-coverage failures distinguish probe errors and request-deadline timeouts
# from a completed probe with no results.
OPTION_COVERAGE_PROBE_FAILED = "option_coverage_probe_failed"
OPTION_COVERAGE_TIMED_OUT = "option_coverage_timed_out"
# Observation-fetch failures use separate codes from option-coverage probes.
# OBSERVATIONS_PROBE_FAILED means an upstream per-pair failure or no dispatchable
# facet. Requests over the preview ceilings use OBSERVATIONS_PREVIEW_TOO_LARGE.
OBSERVATIONS_PROBE_FAILED = "observations_probe_failed"
# The observation fetch crossed its deadline during per-pair or bulk reads, or
# a pending-task cancellation. A bound-unit request treats this
# as an unanswered check and returns a retryable 503.
OBSERVATIONS_TIMED_OUT = "observations_timed_out"
# No confirmed StatVar, entity, or complete role tuple could form a probe pair.
# This fires before dispatch. OBSERVATIONS_NO_POINTS means a fetch returned empty.
OBSERVATIONS_NOTHING_TO_PROBE = "observations_nothing_to_probe"
# Callers without graph or SDMX clients get this code. The live search surface
# supplies both clients.
OBSERVATIONS_CLIENTS_UNAVAILABLE = "observations_clients_unavailable"
# The dispatched pairs returned no points and no active facet filter caused
# the omission. Coverage may report observations for the same selection, so
# an empty preview can coexist with a non-empty data slice.
OBSERVATIONS_NO_POINTS = "observations_no_points"
# The availability read over the menu at the places a part names failed, so whether each
# reading has data there stays unknown (`availability.read_menu_availability`).
AVAILABILITY_PROBE_FAILED = "availability_probe_failed"
# A resubmit's source constraint matched no fetched facet. The fetch completed,
# distinguishing this from OBSERVATIONS_PROBE_FAILED and OBSERVATIONS_NO_POINTS.
OBSERVATIONS_NO_MATCHING_SOURCE = "observations_no_matching_source"
# Registered in the warning vocabulary, and no stage emits it: a sole-option binding is
# never dropped, since the catalogue gives every taxonomy whole.
REDUNDANT_BINDING_DROPPED = "redundant_binding_dropped"
# The bind response named a value or set binding with no dcids, which its schema refuses at
# parse (`bind.SlotBindingDraft`), so the part refuses `variable_not_resolved`. Raised by
# the bind phase (`phase_rank_bind._phase_bind`).
DEGENERATE_BINDING_DETECTED = "degenerate_binding_detected"
# A resolved role has no ->name label. Keep its dcid as the fallback label.
# Spec identity hashes dcids, so this preserves the entity's identity.
# One warning names all missed dcids.
ENTITY_LABEL_UNRESOLVED = "entity_label_unresolved"
# Place-as-constraint is disabled by SEARCH_SEAM_DEFAULT. Entities remain subjects.
SEAM_OFF_INFO_CODE = "place_constraint_seam_off"
# A standard query without a place used the default scope (place.DefaultPresence.scope).
# This warning appears on definite, candidates, and no-data responses using that scope.
PLACE_SCOPE_DEFAULTED = "place_scope_defaulted"
# Extraction found a ranking or selection qualifier the v1 identity cannot
# represent. The definite sentence repeats this warning in its trailing note.
QUERY_QUALIFIER_DROPPED = "query_qualifier_dropped"
# A role-bearing query left a flow axis open, so discover_role_scope enumerated
# its SDMX members. Both open axes can warn. Enumerating the second would require
# another SDMX read. The code is informational and appears on fresh and resubmit paths.
ROLE_SCOPE_DEFAULTED = "role_scope_defaulted"
# A declared default fixed a slot the query names by its plain name ("top 5 aid
# flows"), so the answer covers one value where the reader may expect them all.
# Raised by the election of a first answer; a resubmit posts the default and raises none.
SLOT_DEFAULTED = "slot_defaulted"
# A slot default the code index declares for the elected shape's population type was not
# applied, because no member reading the default value has data at the queried places while
# another member has, so the slot stays open and the family's total answers. Raised by the
# settle stage (`phase_election._repair_bindings`).
SLOT_DEFAULT_NOT_APPLIED = "slot_default_not_applied"
# The single member an answer names is not its family's total: the family holds
# none, or its total had no data at the place. Raised by the election of a first
# answer; a resubmit posts the member and raises none.
FAMILY_TOTAL_ABSENT = "family_total_absent"
# A role-bearing answer reads the members Search recalled, because the mint's member
# walk did not settle them (`bound_values.MembershipAuthority`): no walk ran, or it ran
# past its bound, a graph read failed, or it left candidates unread. The Datasets mint
# may resolve other members for the same identity, so the response cache does not
# store the answer.
MEMBERSHIP_UNVERIFIED = "membership_unverified"
# A resubmit's elected members include StatVars whose arcs the graph does not hold, so the
# answer leaves them out. The code index lists members the graph no longer serves, or the
# graph dropped a member between the election and the arc read. The message names them.
MEMBERS_ABSENT_FROM_GRAPH = "members_absent_from_graph"
# The question names a denominator and the answer's series declares no denominator arc, so
# its ratio base comes from the series name. The message names the denominator asked.
DENOMINATOR_UNVERIFIED = "denominator_unverified"
# A named place's direction is dropped when the elected shape cannot carry it,
# so the place becomes the subject. The warning carries this fact. A query with
# no direction hint emits none. Promote paths refuse role_incompatible instead.
DIRECTION_DISCARDED = "direction_discarded"
# The role refresh read failed for an index-sourced population. Its packaged
# role declarations remain in use. This warning is emitted during materialisation.
ROLE_DECLARATION_UNVERIFIED = "role_declaration_unverified"
# A scope-defaulted or role-bearing bulk SDMX read failed after coverage ran.
# The answer has no series, but retains coverage. Bound-unit checks escalate
# this upstream failure. Per-pair preview failures use OBSERVATIONS_PROBE_FAILED.
OBSERVATIONS_BULK_FETCH_FAILED = "observations_bulk_fetch_failed"
# The selected coverage-ranked facet misses some places. Series ship for
# covered places. The warning reports delivered and total place counts.
OBSERVATIONS_SCOPE_PARTIAL = "observations_scope_partial"
# More confirmed series exist than fit SEARCH_ROLE_BULK_SERIES_MAX, so only some ship:
# the most recent and largest when the whole read fits SEARCH_ROLE_BULK_RANK_MAX_ROWS
# (params.order "ranked"), else a fixed selection (params.order "fixed"). Every omitted
# pair has data. OBSERVATIONS_SCOPE_PARTIAL means some places lack coverage.
OBSERVATIONS_SERIES_TRUNCATED = "observations_series_truncated"
# The bulk facet merge yielded no dispatchable facet without an active filter.
# This includes candidates with no facet id. OBSERVATIONS_NO_MATCHING_SOURCE
# means a requested source filter caused the empty merge.
OBSERVATIONS_NO_DISPATCHABLE_FACET = "observations_no_dispatchable_facet"
# An attached conversion used a projected exchange-rate or deflator facet at
# the observation year or recipe base year.
DEFLATOR_PROJECTED = "deflator_projected"
# A conversion into each entity's own currency left out the rows of entities no single
# currency names through nationalCurrencyOf; the message names them.
CURRENCY_UNMAPPED_OMITTED = "currency_unmapped_omitted"
# A constant-price conversion left out each entity's rows in the years the deflator table
# holds no value for it, every row of one it holds no value for in the base year; the
# message names them, and the answer discloses a `year_unavailable` departure.
DEFLATOR_MISSING_OMITTED = "deflator_missing_omitted"
# The UN System Data Commons did not serve part of the answer. Either its resolve failed or
# timed out, so the candidates are ours alone, or its front door refused reads naming places,
# so the answer holds no UN data for the places the message names.
UN_SYSTEM_UNAVAILABLE = "un_system_unavailable"
# A no-data answer's elected series hold data only under provenances the origin
# registry does not admit; the message names them.
WITHHELD_BY_REGISTRY = "withheld_by_registry"
# The loaded origin catalogue is older than `origin_catalogue.STALE_AFTER`.
ORIGIN_CATALOGUE_STALE = "origin_catalogue_stale"
# The origin catalogue and coverage builds the request bound read different ingestions,
# so an answer may miss data the newer one added until both rebuild. The message names
# both builds and their ingestions.
ARTIFACT_PAIR_STALE = "artifact_pair_stale"
# The origin catalogue's arcs and the packaged code index disagree on the five-tuple,
# constraint properties or roles of a StatVar the answer serves. The message names them.
CATALOGUE_INDEX_DRIFT = "catalogue_index_drift"
# The app has loaded no conversion catalogue yet, so a currency answer states no
# unit_conversion offer; the null says nothing about whether it converts.
UNIT_CONVERSION_UNAVAILABLE = "unit_conversion_unavailable"
# A unit request's conversion tables cover only part of the years the answer serves, so the
# answer shows the longest run they cover (`convertibility.unit_request_period`); it
# discloses a `year_unavailable` departure.
DATE_COVERAGE_PARTIAL = "date_coverage_partial"


# Events and warnings from one trailing option-coverage probe.
_OptionEvent = SlotOptionsEvent | EntityOptionsEvent | FacetOptionsEvent
_OptionProbeResult = tuple[tuple[_OptionEvent, ...], tuple[Warning, ...]]

# One observation event and its warnings. The fetch always returns
# SpecObservations. An omitted preview is recorded in its `omitted` field.
# Its task relaunch rules differ from option coverage.
_ObsProbeResult = tuple[tuple[ObservationsEvent, ...], tuple[Warning, ...]]

# A grounded near-match Spec disproves entity- or variable-level failure for
# the response, whether it is same-variable/place or alternate-variable.
_INCOHERENT_WITH_GROUNDED_NEAR_MATCH = frozenset({"entity_not_resolved", "variable_not_resolved"})


def _elect_no_data_reason(
    *candidates: NoDataReason | None,
    fallback: NoDataReason,
    grounded_near_match: bool = False,
) -> NoDataReason:
    """Return the highest-priority candidate or ``fallback``.

    Ignore ``None`` candidates. ``fallback`` is always included. With a
    grounded near-match, discard entity- and variable-level failures from
    candidates only. Narrower failures remain eligible. ``entity_not_resolved``
    is unreachable here but stays in the exclusion set for completeness.
    """
    live = [c for c in candidates if c is not None]
    if grounded_near_match:
        live = [c for c in live if c not in _INCOHERENT_WITH_GROUNDED_NEAR_MATCH]
    return elect_no_data_reason([*live, fallback])
