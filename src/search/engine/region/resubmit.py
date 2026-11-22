"""Entry point for standard identity resubmission.

Reject pinned StatVar identities, then delegate to `_resolve_standard_promote`,
which elects the identity's members and rebuilds the shape from their arcs.
"""

from __future__ import annotations

import dataclasses
import logging

from code_index.shared import get_index
from materialise.member_conversion import request_conversion_reader
from origin_catalogue import current_catalogue, recording_un_blocked_places
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import EngineInputError
from search.engine.identity_edit import check_described_arm, check_not_composite, open_within
from search.engine.region.resubmit_promote import _resolve_standard_promote
from search.engine.region.resubmit_request import pinned_provenance_scope
from search.engine.region.result import RegionResult
from search.engine.region.state import EngineOptions
from search.engine.region.un_system import with_un_unavailable_warning
from search.engine.spec_id import identity_spec_id
from search.models import BaseIdentity, ResubmitIdentityPayload
from search.observations import previewing_years
from search.scope.composite import CompositeMint, served_components
from search.scope.tables import ScopeFacts
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


def composite_mint(identity: BaseIdentity, *, coverage: CoverageSnapshot) -> CompositeMint:
    """What a Spec answering composite `identity` is minted from (`CompositeMint`): its
    components with the years each serves (`served_components`), read with facts of the
    release's code index and `coverage` alone, and the catalogue bound when it is called,
    which its provenance facet and download read. A caller builds it before any read narrows
    the catalogue to a source scope (`resubmit_request.pinned_provenance_scope`)."""
    facts = ScopeFacts(index=get_index(), coverage=coverage)
    document = identity.model_dump(mode="json", exclude_none=True)
    return CompositeMint(
        identity=identity,
        spec_id=identity_spec_id(identity),
        served=tuple(served_components(document, facts=facts)),
        facts=facts,
        catalogue=current_catalogue(),
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
    rebuild the shape. Graph labels replace posted labels. A bound `facets.provenance`
    scopes every read to its refs before DFO precedence (`pinned_provenance_scope`), as the
    Datasets mint reads it. `full_read_deadline` bounds a
    member walk's full read of a deep-root family and every phase after it
    (`_resolve_standard_promote`).

    A role `inp.populations` lists is read as the open role the caller posted, within the
    population's members (`identity_edit.open_within`): the identity the promote reads opens
    it, and its role-scope discovery reads only those members
    (`role_scope.discover_role_scope`).

    Pinned StatVar identities and composite identities (`identity_edit.check_not_composite`)
    raise `request_invalid` (400). Missing graph
    nodes produce `no_data`, except absent observationAbout role refs, which
    raise `invalid_role_ref` (422). An identity binding more places than a Spec holds
    raises `request_invalid` (400) in validation. Origin refs require a role-bearing shape.
    `coverage` and `options` carry request-scoped enrichment inputs. `options=None` uses
    the un-enriched baseline. The result is `degraded` when the promote swallowed a read
    (`degraded_reads`). Its preview keeps points in `options.preview_years` alone
    (`observations.previewing_years`), where set.

    Where `inp.composite` names the composite `inp.identity` is the table identity of, the
    Spec is minted as that composite (`composite_mint`), under its spec_id, its warnings
    keyed to it and its defaulted scope counted over its components
    (`CompositeMint.role_scope_warnings`).
    """
    check_described_arm(inp.identity.shape, pointer="/input/identity/shape")
    check_not_composite(inp.identity, pointer="/input/identity/facets/provenance")
    _options = options if options is not None else EngineOptions()
    # Built before the identity's provenance pin narrows the catalogue.
    mint = None if inp.composite is None else composite_mint(inp.composite, coverage=coverage)
    opened, _ = open_within(inp.identity, inp.populations)
    if opened is not inp.identity:
        inp = inp.model_copy(update={"identity": opened})

    # `_phase_materialise` checks origins after rebuilding the shape.
    # Role-bearing shapes bind origins on their origin axis. Standard shapes raise
    # `role_incompatible`.
    try:
        with (
            recording_degraded_reads() as lost,
            recording_un_blocked_places() as un_blocked,
            pinned_provenance_scope(inp.identity.facets),
            previewing_years(_options.preview_years),
        ):
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
                composite=mint,
            )
    except EngineInputError as exc:
        raise _with_nested_identity_pointer(exc) from exc
    result = with_un_unavailable_warning(result, un_blocked)
    # A read the promote swallowed marks the part degraded, so no cache stores it.
    return dataclasses.replace(result, degraded=True) if lost.degraded else result
