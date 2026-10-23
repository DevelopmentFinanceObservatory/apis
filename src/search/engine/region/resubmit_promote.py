"""Resolve a resubmitted identity through the standard promote path.

Elect the identity's members as the Datasets mint does, re-read their arcs, and
promote them. `_PROMOTE_PHASES` fixes the execution order.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from dfo_spec_kit.identity import IdentityError

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
from search.engine.region.requirements import with_requirement_departures
from search.engine.region.result import RegionResult, with_departure_facts
from search.engine.retrieval.bound_values import ResubmitWalk, RoleClassWalks
from search.engine.spec_id import identity_spec_id
from search.models import ResubmitIdentityPayload
from series_coverage.shared import CoverageSnapshot

if TYPE_CHECKING:
    # These imports stay type-only to avoid the options/population/validation cycle.
    from dc_sdmx import SdmxClient as MaterialiseSdmxClient
    from materialise import GraphReader
    from materialise.rate_tables import RateTableCache
    from search.engine.options.population_memo import OptionsPopulationMemo
    from search.engine.region.place_options import PlaceOptionsPool
    from search.models import BaseIdentity


def _resolve_standard_promote(
    inp: ResubmitIdentityPayload,
    graph: EngineGraphClient,
    *,
    sdmx: EngineSdmxClient,
    include_option_coverage: bool = False,
    stream_slot_options: bool = False,
    include_observations: bool = False,
    stream_observations: bool = False,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    deadline: float | None = None,
    full_read_deadline: float | None = None,
    rate_cache: RateTableCache | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    coverage: CoverageSnapshot,
) -> RegionResult:
    """Resolve a resubmit from the identity alone.

    The identity's members are elected as the Datasets mint elects them
    (`promote_reread._elected_members`): the release's code index for a named measure family or
    an identity it decides, else the mint's graph walk, with its first page under
    `deadline` and its full read of a deep-root family under `full_read_deadline`, which
    defaults to `deadline`. A walk that ran the full read moves every later phase to
    `full_read_deadline`. An election that settles no member returns `no_data`. A family
    too broad to settle returns the `scope_too_broad` part. A walk past its deadline or a
    failed graph read raises. Multiple five-tuples raise `request_invalid`. Shape
    mismatches raise `shape_mismatch`. Donor refs on standard shapes raise
    `role_incompatible` (422). Absent observationAbout or donor refs raise
    `invalid_role_ref` (422).

    The result's `reads_deadline` is the bound its later phases ran to. A read a phase
    swallows is noted as lost (`degraded_reads.note_degraded_read`), and
    `resolve_spec_resubmit` marks the region `degraded`.
    """
    state = _PromoteState(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        deadline=deadline,
        include_option_coverage=include_option_coverage,
        stream_slot_options=stream_slot_options,
        include_observations=include_observations,
        stream_observations=stream_observations,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        rate_cache=rate_cache,
        population_memo=population_memo,
        place_options_pool=place_options_pool,
        coverage=coverage,
        role_class_walks=(
            RoleClassWalks(
                materialise_graph,
                resubmit=ResubmitWalk(
                    subject_ceiling=settings().resubmit_subject_ceiling,
                    full_read_deadline=(
                        full_read_deadline if full_read_deadline is not None else deadline
                    ),
                    spec_id=_walk_spec_id(inp.identity),
                ),
            )
            if materialise_graph is not None
            else None
        ),
    )

    try:
        for phase in _PROMOTE_PHASES:
            result = phase(state)
            if result is not None:
                # Include timings from phases that returned early, the departures of the
                # posted values the answer leaves unapplied (`with_requirement_departures`),
                # and the departures raised beside the warnings the result keeps. A resubmit posts no words for any
                # entity, so its departures quote none (`RegionResult.entity_words`).
                return with_departure_facts(
                    with_requirement_departures(
                        dataclasses.replace(
                            result,
                            timing_by_step=dict(state.timing),
                            entity_words=None,
                            reads_deadline=state.deadline,
                        ),
                        state.requirements,
                        state.judgement,
                    ),
                    state.departure_facts,
                )
        raise AssertionError("_phase_finalize is terminal")
    finally:
        # Grounding starts the related-measures reads and only finalisation takes them, so
        # however the pass ends it waits for them here, and none outlives the answer.
        settle_pending_related(state.related_read)


def _walk_spec_id(definition: BaseIdentity) -> str | None:
    """The posted identity's spec id, which names its member walk in the walk's log line.
    None when the identity does not hash. The resubmit refuses it later, and the log line
    names no id."""
    with contextlib.suppress(IdentityError):
        return identity_spec_id(definition)
    return None


# Phases run in tuple order and stop at the first RegionResult. Validation,
# binding, and place phases raise `EngineInputError` on refusal. Reread returns
# the election's refusal: `variable_not_resolved`, `no_matching_slice`,
# `role_signature_incompatible`, or the `scope_too_broad` part. Shape building can
# return `role_signature_incompatible`, `variable_not_resolved` or `scope_too_broad`.
# Binding reads no graph data. The place phase returns `role_incompatible`.
# Materialisation returns the draft's no-data reason or `variable_not_resolved`, and
# raises `open_role_unavailable` for an open role of the identity its answer does not serve.
# Grounding returns
# `qualifier_not_bound` for a posted value the answer unbound, `no_observations`
# (`NO_MEMBER_HAS_DATA_AT_PLACE`) or `no_matching_slice`
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
