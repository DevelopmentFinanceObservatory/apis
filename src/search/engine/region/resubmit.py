"""Entry point for standard identity resubmission.

Reject pinned StatVar identities, then delegate to `_resolve_standard_promote`,
which elects the identity's members and rebuilds the shape from their arcs.
"""

from __future__ import annotations

import dataclasses
import logging

from materialise.member_conversion import request_conversion_reader
from origin_catalogue import recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import EngineInputError
from search.engine.identity_edit import check_described_arm
from search.engine.region.resubmit_promote import _resolve_standard_promote
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.region.un_system import with_un_unavailable_warning
from search.models import ResubmitIdentityPayload
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
        str(exc),
        code=exc.code,
        status=exc.status,
        pointer=nested_pointer,
        alternatives=exc.alternatives,
        cause=exc.cause,
    )


def resolve_spec_resubmit(
    *,
    inp: ResubmitIdentityPayload,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    coverage: CoverageSnapshot,
    options: EngineOptions | None = None,
    full_read_deadline: float | None = None,
) -> RegionResult:
    """Resolve an identity without extract, recall, shape, or bind steps.

    Elect the identity's members as the Datasets mint does, re-read their arcs and
    rebuild the shape. Graph labels replace posted labels. `full_read_deadline` bounds a
    member walk's full read of a deep-root family and every phase after it
    (`_resolve_standard_promote`).

    Pinned StatVar identities raise `request_invalid` (400). Missing graph
    nodes produce `no_data`, except absent observationAbout role refs, which
    raise `invalid_role_ref` (422). An identity binding more places than a Spec holds
    raises `request_invalid` (400) in validation. Origin refs require a role-bearing shape.
    `coverage` and `options` carry request-scoped enrichment inputs. `options=None` uses
    the un-enriched baseline. The result is `degraded` when the promote swallowed a read
    (`degraded_reads`).
    """
    check_described_arm(inp.identity.shape, pointer="/input/identity/shape")
    _options = options if options is not None else EngineOptions()

    # `_phase_materialise` checks origins after rebuilding the shape.
    # Role-bearing shapes bind origins on their origin axis. Standard shapes raise
    # `role_incompatible`.
    try:
        with recording_degraded_reads() as lost, recording_un_blocked_places() as un_blocked:
            result = _resolve_standard_promote(
                inp,
                graph,
                sdmx=sdmx,
                deadline=_options.deadline,
                full_read_deadline=full_read_deadline,
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
                rate_cache=_options.rate_cache,
                population_memo=_options.population_memo,
                place_options_pool=_options.place_options_pool,
                coverage=coverage,
            )
    except EngineInputError as exc:
        raise _with_nested_identity_pointer(exc) from exc
    result = with_un_unavailable_warning(result, un_blocked)
    # A read the promote swallowed marks the part degraded, so no cache stores it.
    return dataclasses.replace(result, degraded=True) if lost.degraded else result
