"""Entry point for standard identity resubmission.

Reject pinned StatVar identities, then delegate to `_resolve_standard_promote`,
which rebuilds the shape from graph-confirmed StatVar arcs and resolves edits.
"""

from __future__ import annotations

import logging

from materialise.member_conversion import request_conversion_reader
from origin_catalogue import recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.envelope import make_pipeline_step
from search.engine.errors import EngineInputError
from search.engine.identity_edit import check_described_arm
from search.engine.region.resubmit_promote import _resolve_standard_promote
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.region.un_system import with_un_unavailable_warning
from search.models import PipelineStep, ResubmitIdentityPayload
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

# The materialise-to-build_spec chain is synchronous, so callers can run it in `asyncio.to_thread`.


def _with_nested_identity_pointer(exc: EngineInputError) -> EngineInputError:
    """Translate legacy adapter pointers to the nested request contract."""
    pointer = exc.pointer
    if pointer is None:
        return exc
    if pointer == "/input/slots" or pointer.startswith("/input/slots/"):
        nested_pointer = pointer.replace("/input/slots", "/input/identity/slots", 1)
    elif pointer == "/input/facets" or pointer.startswith("/input/facets/"):
        nested_pointer = pointer.replace("/input/facets", "/input/identity/facets", 1)
    else:
        return exc
    return EngineInputError(
        *exc.args,
        code=exc.code,
        status=exc.status,
        pointer=nested_pointer,
        alternatives=exc.alternatives,
        cause=exc.cause,
        detail=exc.detail,
    )


def resolve_spec_resubmit(
    *,
    inp: ResubmitIdentityPayload,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    stat_vars_cap: int | None = settings().resolved_stat_vars_max,
    coverage: CoverageSnapshot,
    options: EngineOptions | None = None,
) -> RegionResult:
    """Resolve an identity without extract, recall, shape, or bind steps.

    Re-read posted StatVar arcs and rebuild the shape. Matching bindings promote.
    Edits narrow the seed or construct and graph-confirm sibling StatVars. The
    trace marks `extract` as skipped, and graph labels replace posted labels.

    Pinned StatVar identities raise `request_invalid` (400). Missing graph
    nodes produce `no_data`, except absent observationAbout role refs, which
    raise `invalid_role_ref` (422). Origin lists are capped at
    `max_standard_places`. Origin refs require a role-bearing shape.
    `stat_vars_cap` bounds the returned Spec. `None` serves uncapped internal
    options reads. `coverage` and `options` carry request-scoped enrichment
    inputs. `options=None` uses the un-enriched baseline.
    """
    check_described_arm(inp.identity.shape, pointer="/input/identity/shape")
    origin_dcids = inp.identity.entities.get("origin", [])
    if len(origin_dcids) > settings().max_standard_places:
        raise EngineInputError(
            f"{len(origin_dcids)} unique origin references exceed the "
            f"{settings().max_standard_places}-origin limit",
            code="request_invalid",
            pointer="/input/identity/entities/origin",
        )
    extract_step = make_pipeline_step("extract", ran=False)
    pipeline_steps: list[PipelineStep] = [extract_step]
    _options = options if options is not None else EngineOptions()

    # `_phase_materialise` checks origins after rebuilding the shape.
    # Role-bearing shapes bind origins on their origin axis. Standard shapes raise
    # `role_incompatible`.
    try:
        with recording_un_blocked_places() as un_blocked:
            result = _resolve_standard_promote(
                inp,
                graph,
                pipeline_steps,
                sdmx=sdmx,
                deadline=_options.deadline,
                # One conversion memo serves every unit decision, coverage check and preview.
                materialise_graph=(
                    request_conversion_reader(_options.materialise_graph)
                    if _options.materialise_graph is not None
                    else None
                ),
                materialise_sdmx=_options.materialise_sdmx,
                include_option_coverage=_options.include_option_coverage,
                stream_slot_options=_options.stream_slot_options,
                include_observations=_options.include_observations,
                stream_observations=_options.stream_observations,
                stat_vars_cap=stat_vars_cap,
                rate_cache=_options.rate_cache,
                population_memo=_options.population_memo,
                place_options_pool=_options.place_options_pool,
                coverage=coverage,
            )
    except EngineInputError as exc:
        raise _with_nested_identity_pointer(exc) from exc
    return with_un_unavailable_warning(result, un_blocked)
