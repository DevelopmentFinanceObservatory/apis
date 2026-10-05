"""Resolve a resubmitted identity through the standard promote/refine path.

Re-read StatVar arcs, then promote, narrow to posted values, or confirm sibling
StatVars for values outside the seed. `_PROMOTE_PHASES` fixes the execution order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.region.promote_bindings import _phase_build_bindings
from search.engine.region.promote_edits import _phase_detect_edits
from search.engine.region.promote_finalize import _phase_finalize
from search.engine.region.promote_ground import _phase_ground_build
from search.engine.region.promote_materialise import _phase_materialise
from search.engine.region.promote_places import _phase_where_places
from search.engine.region.promote_reread import _phase_reread_arcs
from search.engine.region.promote_shape import _phase_build_shape
from search.engine.region.promote_state import _PromoteState
from search.engine.region.promote_survivors import _phase_resolve_survivors
from search.engine.region.promote_validate import _phase_validate
from search.engine.region.related_read import settle_pending_related
from search.engine.region.result import RegionResult, with_departure_facts
from search.engine.retrieval.bound_values import RoleClassWalks
from search.models import PipelineStep, ResubmitIdentityPayload
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    # These imports stay type-only to avoid the options/population/validation cycle.
    from dc_sdmx import SdmxClient as MaterialiseSdmxClient
    from materialise import GraphReader
    from materialise.rate_tables import RateTableCache
    from search.engine.options.population_memo import OptionsPopulationMemo
    from search.engine.region.place_options import PlaceOptionsPool


def _resolve_standard_promote(
    inp: ResubmitIdentityPayload,
    graph: EngineGraphClient,
    pipeline_steps: list[PipelineStep],
    *,
    sdmx: EngineSdmxClient,
    include_option_coverage: bool = False,
    stream_slot_options: bool = False,
    include_observations: bool = False,
    stream_observations: bool = False,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    deadline: float | None = None,
    stat_vars_cap: int | None = settings().resolved_stat_vars_max,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    coverage: CoverageSnapshot,
) -> RegionResult:
    """Resolve a resubmit from graph-confirmed StatVar arcs.

    The request needs `base_spec_id` and `base_identity`. Standard shapes also
    need `resolved_stat_var_dcids` as their seed. An edit the seed cannot answer elects
    the edited identity's members as the Datasets mint does. Named measure families
    recover members from the code index. Missing base or seed data, malformed edits, and
    multiple five-tuples raise `request_invalid`. Base identity failures retain
    their specific codes. Incomplete edits raise `edit_incomplete`. Shape
    mismatches raise `shape_mismatch`. Donor refs on standard shapes raise
    `role_incompatible` (422). Absent observationAbout or donor refs raise
    `invalid_role_ref` (422).

    Matching what/how constraints promote. Values already covered by the seed
    narrow its confirmed StatVars. Other values elect the edited identity's family
    (`promote_survivors._elected_members`). Missing seeds and an election that settles
    no member return `no_data`.
    """
    state = _PromoteState(
        inp=inp,
        graph=graph,
        pipeline_steps=pipeline_steps,
        sdmx=sdmx,
        deadline=deadline,
        include_option_coverage=include_option_coverage,
        stream_slot_options=stream_slot_options,
        include_observations=include_observations,
        stream_observations=stream_observations,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        stat_vars_cap=stat_vars_cap,
        rate_cache=rate_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        coverage=coverage,
        role_class_walks=(
            RoleClassWalks(materialise_graph) if materialise_graph is not None else None
        ),
    )

    try:
        for phase in _PROMOTE_PHASES:
            result = phase(state)
            if result is not None:
                # Include timings from phases that returned early, and the departures raised
                # beside the warnings the result keeps. A resubmit posts no words for any
                # entity, so its departures quote none (`RegionResult.entity_words`).
                return with_departure_facts(
                    dataclasses.replace(
                        result, timing_by_step=dict(state.timing), entity_words=None
                    ),
                    state.departure_facts,
                )
        raise AssertionError("_phase_finalize is terminal")
    finally:
        # Grounding starts the related-measures reads and only finalisation takes them, so
        # however the pass ends it waits for them here, and none outlives the answer.
        settle_pending_related(state.related_read)


# Phases run in tuple order and stop at the first RegionResult. Validation,
# edit, binding, and place phases raise `EngineInputError` on refusal. Reread
# returns `variable_not_resolved`. Survivor resolution returns `no_matching_slice`.
# Shape building can return `no_matching_slice` or `role_signature_incompatible`.
# Binding reads no graph data. The place phase returns `role_incompatible`.
# Materialisation returns the draft's no-data reason or `variable_not_resolved`.
# Grounding returns
# `no_observations` (`NO_MEMBER_HAS_DATA_AT_PLACE`) or `no_matching_slice`
# (`NO_MEMBER_ASSERTS_BOUND_PROPERTIES`). It can refuse facet/unit bindings with
# `facet_not_available`, `facet_binding_unsupported`, `recipe_not_declared`,
# `rate_coverage_gap`, or `request_invalid`. Missing required clients raise
# `EngineInfraError`. Finalize returns `no_observations` for empty windows.
_PROMOTE_PHASES: tuple[Callable[[_PromoteState], RegionResult | None], ...] = (
    _phase_validate,
    _phase_reread_arcs,
    _phase_detect_edits,
    _phase_resolve_survivors,
    _phase_build_shape,
    _phase_build_bindings,
    _phase_where_places,
    _phase_materialise,
    _phase_ground_build,
    _phase_finalize,
)
