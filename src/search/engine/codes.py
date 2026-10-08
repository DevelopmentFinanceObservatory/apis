"""Search's refusal and warning codes, plus its share of the platform registry.

Codes minted by both DFO APIs and routing refusals live in
``dfo_platform.codes.PLATFORM_CODE_CATALOGUE``. ``merge_catalogues`` rejects
all duplicate keys, including identical entries, so shared codes have one
source. ``SEARCH_REFUSAL_CODES`` holds contract codes and Search-specific
refusals, including:

- ``door_not_implemented`` marks an unserved defined door. ``facet_binding_unsupported``
  marks an unhonoured axis on a served door.
- ``unit_request_unsupported`` when an understood unit falls outside the
  currency/prices/base_year form. Parse failures use ``unit_not_resolved``.
  Unsupported axes use ``facet_binding_unsupported``.
- The anchor codes distinguish an unmapped type, a missing children anchor,
  mixed peers types, and multiple children anchors. Identity sorts and
  de-duplicates role entities, so each refusal avoids an answer dependent on
  anchor order or duplicates.
- ``provider_ambiguous`` is reserved by the shared registry. Search accepts multiple origins.
- Three ``NoData.gate`` codes belong to the contract's gate vocabulary. Gate is their only surface.
- ``population_empty`` marks a shape with no StatVar population in the code index or graph.
  ``exact_binding_unsupported`` marks a posted exact slot binding.
  ``directional_children_unsupported`` reserves flow-axis children scope.
  ``resolution_ambiguous`` is also contract-reserved.

``SEARCH_CODE_CATALOGUE`` supplies client-safe titles and default details.
Missing titles use ``title_for_status`` and generic details. ``CODE_CATALOG``
merges it with the platform catalogue, and its keys equal ``REFUSAL_CODES``.

Warning codes remain Search-specific because they have no problem ``type``
URI. ``facet_binding_unsupported`` appears in both registries because Search
can emit it as either a refusal or a warning. Gate codes appear only in the
refusal registry.
"""

from __future__ import annotations

from dfo_platform.codes import PLATFORM_CODE_CATALOGUE, merge_catalogues
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED
from search.observations import (
    OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
    OBSERVATIONS_PREVIEW_TOO_LARGE,
    OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
)

SEARCH_REFUSAL_CODES: frozenset[str] = frozenset(
    {
        # Search API contract refusal table.
        "query_too_long",
        "shape_mismatch",
        "option_target_invalid",
        # A resubmitted identity names an open role its elected members do not serve.
        "open_role_unavailable",
        # A contract-defined door omitted from this deployment.
        "door_not_implemented",
        # The unit binder understood a request outside its typed form.
        "unit_request_unsupported",
        # Open-axis discovery exceeds a year, row, byte, or value cap.
        "scope_too_broad",
        # A bound unit request exceeds the cap on facet combinations to probe.
        "facet_probe_ceiling",
        # observationAbout candidate-universe derivation.
        "unsupported_peers_anchor_type",
        "unsupported_children_anchor_type",
        # Children scope needs an anchor. Peers scope has a default.
        "children_anchor_required",
        # Bound peers anchors disagree on type, or children anchors on dcid.
        "mixed_peers_anchor_types",
        "children_anchor_ambiguous",
        # Reserved by the shared registry. Search accepts multiple origins.
        "provider_ambiguous",
        # NoData.gate codes belong to the contract's served registry.
        "only_alternate_has_data",
        "no_member_has_data_at_place",
        "no_member_asserts_bound_properties",
        # The identity shape has no StatVar population in the code index or graph.
        "population_empty",
        # Posted slot bindings with state="exact" are served only by datasets doors.
        "exact_binding_unsupported",
        # Flow axes lack a containment hierarchy for children scope.
        "directional_children_unsupported",
    }
)

SEARCH_CODE_CATALOGUE: dict[str, tuple[str, str]] = {
    # Code-specific title and fallback detail for problem responses.
    "query_too_long": (
        "Query Too Long",
        "The query text exceeds this deployment's maximum length.",
    ),
    "open_role_unavailable": (
        "Open Role Unavailable",
        "The resubmitted identity names an open role, and the StatVars it elects declare "
        "no role on that axis.",
    ),
    "shape_mismatch": (
        "Shape Mismatch",
        "The posted shape does not match the shape this request resolves to.",
    ),
    "option_target_invalid": (
        "Option Target Invalid",
        "The posted option target does not match any slot or entity this spec exposes.",
    ),
    "door_not_implemented": (
        "Door Not Implemented",
        "This deployment does not serve this input door; consult GET "
        "/api/search/capabilities for per-door liveness.",
    ),
    "unit_request_unsupported": (
        "Unit Request Unsupported",
        "The requested unit phrase was understood but cannot be represented "
        "in the currency/prices/base_year request form.",
    ),
    "scope_too_broad": (
        "Scope Too Broad",
        "The scope this query's open axis would discover exceeds this "
        "deployment's read caps; name an entity on this axis, narrow the "
        "requested time window, or bind a source/unit/method facet to "
        "narrow the scope.",
    ),
    "facet_probe_ceiling": (
        "Facet Probe Ceiling",
        "The bound provenance/measurementMethod combination would exceed "
        "this deployment's cap on facet probes; bind fewer refs on one "
        "axis, or bind a single ref instead of a Set on one axis.",
    ),
    "unsupported_peers_anchor_type": (
        "Unsupported Peers Anchor Type",
        "The anchor's type has no peers-scope mapping in this deployment.",
    ),
    "unsupported_children_anchor_type": (
        "Unsupported Children Anchor Type",
        "The anchor's type has no children-scope mapping in this deployment.",
    ),
    "children_anchor_required": (
        "Children Anchor Required",
        "The children scope requires a bound observationAbout entity to "
        "descend from; bind one, or request peers scope instead.",
    ),
    "mixed_peers_anchor_types": (
        "Mixed Peers Anchor Types",
        "The bound observationAbout anchors do not all resolve to the same "
        "entity type; peers scope requires one shared type.",
    ),
    "children_anchor_ambiguous": (
        "Children Anchor Ambiguous",
        "More than one distinct observationAbout anchor is bound; children "
        "scope requires exactly one to descend from.",
    ),
    "provider_ambiguous": (
        "Provider Ambiguous",
        "The origin-axis bindings cannot be represented unambiguously by this operation.",
    ),
    "only_alternate_has_data": (
        "Only Alternate Has Data",
        "The model's own elected shape returned no data; a distinct alternate "
        "shape with data is surfaced as a flagged match instead.",
    ),
    "no_member_has_data_at_place": (
        "No Member Has Data At Place",
        "No member of the matched variable's family has observations at the resolved place.",
    ),
    "no_member_asserts_bound_properties": (
        "No Member Asserts Bound Properties",
        "No member of the matched variable's family asserts every bound constraint property.",
    ),
    "population_empty": (
        "Population Empty",
        "The identity's own shape resolves to no StatVar population at all, from either the "
        "packaged code index or a graph enumeration.",
    ),
    "exact_binding_unsupported": (
        "Exact Binding Unsupported",
        "A posted slot binding carries state='exact', which no Search door serves; only the "
        "datasets doors read it.",
    ),
    "directional_children_unsupported": (
        "Directional Children Unsupported",
        "A flow role's options were requested with a children scope, but a flow axis has no "
        "containment hierarchy to expand and serves its members from one availability read.",
    ),
}

# Search's exception handlers resolve both Search and platform codes here.
CODE_CATALOG: dict[str, tuple[str, str]] = merge_catalogues(
    PLATFORM_CODE_CATALOGUE, SEARCH_CODE_CATALOGUE
)
REFUSAL_CODES: frozenset[str] = SEARCH_REFUSAL_CODES | frozenset(PLATFORM_CODE_CATALOGUE)

# Shared platform codes served by Search, including its exception mapping,
# facet/role errors, and materialisation recipe and rate-coverage outcomes.
# Contract-reserved code with no Search raise site.
SHARED_PLATFORM_CODES: frozenset[str] = frozenset(
    {
        "request_invalid",
        "unsupported_schema_version",
        "upstream_error",
        "service_unavailable",
        "timeout",
        "facet_not_available",
        "facet_binding_unsupported",
        "invalid_role_ref",
        "role_incompatible",
        "role_signature_incompatible",
        "resolution_ambiguous",
        "recipe_not_declared",
        "rate_coverage_gap",
        # classify_search_exception maps RateTableInvalidError and
        # UpstreamVersionChangedError to these two codes on every door.
        "capability_unavailable",
        "upstream_version_changed",
    }
)
if not frozenset(PLATFORM_CODE_CATALOGUE) >= SHARED_PLATFORM_CODES:
    raise AssertionError(
        f"SHARED_PLATFORM_CODES names code(s) "
        f"{sorted(SHARED_PLATFORM_CODES - frozenset(PLATFORM_CODE_CATALOGUE))} absent from "
        "PLATFORM_CODE_CATALOGUE"
    )

WARNING_CODES: frozenset[str] = frozenset(
    {
        # Named in the contract.
        "date_coverage_partial",
        "default_not_declared",
        # Conjunction and observation-inclusion outcomes.
        "binding_folded",
        "observations_omitted",
        "variables_clamped",
        # Skills-index availability and matching outcomes.
        "skills_index_unavailable",
        "skills_not_matchable",
        "skills_index_incomplete",
        # Region pipeline (region/codes.py).
        "retrieval_score_weak",
        "redundant_binding_dropped",
        "degenerate_binding_detected",
        "starved_axis_binding_restored",
        "near_neighbor_ambiguous",
        "near_neighbor_pool_widened",
        "alternates_unevaluated",
        "entity_label_unresolved",
        "elected_shape_no_data",
        "option_coverage_probe_failed",
        "option_coverage_timed_out",
        "observations_probe_failed",
        "observations_timed_out",
        "observations_no_matching_source",
        "observations_bulk_fetch_failed",
        "observations_scope_partial",
        "observations_series_truncated",
        OBSERVATIONS_WINDOWED_SCOPE_UNSUPPORTED,
        OBSERVATIONS_CONVERTED_SCOPE_UNSUPPORTED,
        "observations_no_dispatchable_facet",
        # Observation omission reasons are warning codes too.
        OBSERVATIONS_PREVIEW_TOO_LARGE,
        "observations_nothing_to_probe",
        "observations_clients_unavailable",
        "observations_no_points",
        "availability_probe_failed",
        "place_scope_defaulted",
        QUERY_QUALIFIER_DROPPED,
        # Role-bearing scope discovery (region/codes.py).
        "role_scope_defaulted",
        # A declared default fixed a slot the query names (region/codes.py).
        "slot_defaulted",
        # The single member an answer names is not its family's total
        # (region/codes.py).
        "family_total_absent",
        # A role-bearing answer reads recalled members the mint's walk did not settle
        # (region/codes.py).
        "membership_unverified",
        # Place-as-constraint (PAC) disabled info code (region/codes.py).
        "place_constraint_seam_off",
        "entity_role_disabled",
        # Parsed-door role lexicon warning, minted by entry/parsed.py.
        "role_text_not_recognised",
        # Warn when structured entities are absent and variable_text contains an entity-like name.
        "variable_text_entity_like",
        # Time-text warnings from parsed and dataset-entry handlers.
        "time_text_not_parsed",
        "time_window_defaulted",
        # Also a refusal code. When unit_text shadows source_text, this warning marks an
        # unhonoured axis. The unit_text binding determines the part's outcome.
        "facet_binding_unsupported",
        # A dataset_resubmit part restated as a stub without graph or LLM calls.
        "part_restated",
        # A projected rate or deflator supplied the conversion facet.
        "deflator_projected",
        # A conversion into each entity's own currency omitted entities no currency names
        # (region/codes.py).
        "currency_unmapped_omitted",
        # A constant-price conversion omitted the rows of entities in years the deflator
        # table lacks (region/codes.py).
        "deflator_missing_omitted",
        # No conversion catalogue is loaded, so a currency answer states no offer
        # (region/codes.py).
        "unit_conversion_unavailable",
        # A replayed selection collapsed to one definite candidate.
        "candidates_collapsed_definite",
        # A replayed selection has no surviving candidates.
        "candidates_collapsed_no_data",
        # The elected shape cannot carry the named place's flow direction.
        "direction_discarded",
        # Failed role-declaration reads leave candidates on the roles already read.
        "role_declaration_unverified",
        # Origin catalogue outcomes: a series withheld by the registry, a UN
        # resolve or place the UN System Data Commons did not serve, and a
        # loaded catalogue past its staleness limit.
        "withheld_by_registry",
        "un_system_unavailable",
        "origin_catalogue_stale",
    }
)
