"""Public exports for ``search.engine.region``.

The package exposes region symbols at one import path. Its submodules handle
matching, grounding, observations, candidates, and resubmission.

The root omits ``materialise_preview``, which tests patch in
``observations.fetch``. A root alias would let a stale patch succeed without
changing the global the fetch reads. The preview caps live in
``search.observations.ObservationSettings``, which tests replace through the
``search_settings`` fixture.
"""

from __future__ import annotations

from search.engine.envelope import _emit_step
from search.engine.region.arcs import (
    inapplicable_slots_from_arcs,
    standard_bindings_from_arcs,
    unresolved_slots_from_arcs,
)
from search.engine.region.candidates import _materialise_alt_shapes
from search.engine.region.codes import (
    AVAILABILITY_PROBE_FAILED,
    DEGENERATE_BINDING_DETECTED,
    DIRECTION_DISCARDED,
    NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    NO_MEMBER_HAS_DATA_AT_PLACE,
    OBSERVATIONS_NO_MATCHING_SOURCE,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_TIMED_OUT,
    ONLY_ALTERNATE_HAS_DATA,
    OPTION_COVERAGE_PROBE_FAILED,
    OPTION_COVERAGE_TIMED_OUT,
    PLACE_SCOPE_DEFAULTED,
    REDUNDANT_BINDING_DROPPED,
    RETRIEVAL_SCORE_WEAK,
    _ObsProbeResult,
    _OptionEvent,
    _OptionProbeResult,
)
from search.engine.region.decisions import detect_set_ref
from search.engine.region.grounding import (
    _build_entity,
    _grade_confidence,
    _ground_answer,
    _place_confidence,
    _shape_with_taxonomy,
    _suggest_nearest_real,
)
from search.engine.region.observations.dispatch import (
    _observations_relaunch_closure,
    _resolve_spec_observations,
)
from search.engine.region.observations.entities import _observation_entities
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.option_coverage import (
    _build_and_enrich_entity_options,
    _gather_slot_options_events,
    _launch_slot_options_task,
    _maybe_enrich_slot_options,
)
from search.engine.region.pipeline import resolve_variable
from search.engine.region.refs import (
    _build_source_refs,
    _ref_or_miss,
    _refs_present,
)
from search.engine.region.resubmit import resolve_spec_resubmit
from search.engine.region.result import RegionResult
from search.engine.region.sentence_events import (
    _build_shape_event,
)
from search.engine.region.weak_score import _representative_score_is_weak

__all__ = [
    "AVAILABILITY_PROBE_FAILED",
    "DEGENERATE_BINDING_DETECTED",
    "DIRECTION_DISCARDED",
    "NO_MEMBER_ASSERTS_BOUND_PROPERTIES",
    "NO_MEMBER_HAS_DATA_AT_PLACE",
    "OBSERVATIONS_NO_MATCHING_SOURCE",
    "OBSERVATIONS_PROBE_FAILED",
    "OBSERVATIONS_TIMED_OUT",
    "ONLY_ALTERNATE_HAS_DATA",
    "OPTION_COVERAGE_PROBE_FAILED",
    "OPTION_COVERAGE_TIMED_OUT",
    "PLACE_SCOPE_DEFAULTED",
    "REDUNDANT_BINDING_DROPPED",
    "RETRIEVAL_SCORE_WEAK",
    "RegionResult",
    "_ObsProbeResult",
    "_OptionEvent",
    "_OptionProbeResult",
    "_build_and_enrich_entity_options",
    "_build_entity",
    "_build_observations",
    "_build_shape_event",
    "_build_source_refs",
    "_emit_step",
    "_gather_slot_options_events",
    "_grade_confidence",
    "_ground_answer",
    "_launch_slot_options_task",
    "_materialise_alt_shapes",
    "_maybe_enrich_slot_options",
    "_observation_entities",
    "_observations_relaunch_closure",
    "_place_confidence",
    "_ref_or_miss",
    "_refs_present",
    "_representative_score_is_weak",
    "_resolve_spec_observations",
    "_shape_with_taxonomy",
    "_suggest_nearest_real",
    "detect_set_ref",
    "inapplicable_slots_from_arcs",
    "resolve_spec_resubmit",
    "resolve_variable",
    "standard_bindings_from_arcs",
    "unresolved_slots_from_arcs",
]
