"""Serve source documentation and the source catalogue.

The documentation route uses a separate literal prefix because the dcid
convertor accepts slashes. It serves the dependent provenance-document read
from `GraphDeps.provenance_docs`, a single-flight cache keyed by the dcid, the
request's origin catalogue build and the graph generation, holding at most
`GRAPH_PROVENANCE_DOC_CACHE_SIZE` documents. Concurrent requests for one
source share one read, a repeat request within a generation and build reads
nothing, and a read that fails is not kept. A miss runs the read as one task on
the shared DCP-read executor, under a flight deadline of
`GRAPH_NODE_TIMEOUT_SECONDS` from the moment the task is submitted. An executor
with every worker busy refuses the task with `DcpExecutorSaturated`, which every
request waiting on the flight answers as retryable 503. Each request waits until
its own deadline, set the same way before it looks the document up. Oversized
dcids return 400 before a read.

Currency and granularity stay out of the cached document. Each response looks
them up in the bound release's coverage index's per-provenance views, which the
index builds when it loads, and `materialise.provenance_doc.with_coverage_facts`
sets them and the gaps they decide, so a release swap shows on the next response.

The wire mapping converts the cadence token to its label, keeps missing scalars
null, maps the parsed transformation steps, changelog releases, methodology
and data notes, and passes `gaps` and `document_errors` through. Countries are
always empty and both usage entries are null. `details.source` and
`details.data_commons` come from the request's origin catalogue, read on the
request thread. A UN System Data Commons provenance (`undata/p/*`) is read
from the UN deployment by the origin graph client.

Currencies are the sorted ISO codes of every `Currency` unit on the facets of
the document's Provenances' coverage series. A Provenance without
currency-valued series adds none, and an empty list gaps the path.
Granularity needs an entry for every one of the document's Provenances, the
same every-child rule `license.type` and `citation` follow. It is
`"<period label>, by <origin noun> and <destination noun>"`, for example
`"Annual, by provider and recipient"`, when their coverage series share
exactly one observation period with a `materialise.cadence` label and the
union of their variables' role axes is the flow pair. The union is the finest
granularity. A Provenance with one-role totals and by-creditor variables
reads as a two-axis source, and one whose variables all sit on one axis
serves null. The nouns come from `materialise.roles.axis_noun`.
Anything else serves null and gaps the path, and the request still succeeds.
Both facts come from one index snapshot per response. A failure reading it logs
one WARNING and empties both facts, and a failure deriving one fact logs a
WARNING and empties only that fact. The release's coverage view holds only the
series of variables that declare roles, so a Provenance without coverage rows
reads as unknown, and the index cannot see a Provenance's other series.

The source index is a literal route on the same router and makes no graph
reads. It joins the packaged origin registry (`origin_catalogue.registry`) to
the request's origin catalogue. One entry per DFO source comes first, then each
registry entry, each with the provenances the catalogue admits under it,
grouped by Dataset, and the provenances the registry excludes under a base
source, each with its reason. A provenance the registry excludes is never also
listed as admitted. Each Dataset
group carries the card facts the catalogue job computed for it
(`origin_catalogue.card_facts`), with the cadence token mapped to its label
here as the documentation route maps it. Existing codes
are reused because a new code needs a `spec-kit`
registry row and a `rev` bump. They map missing nodes to 404, invalid source nodes and
oversized ids to 400, truncated children to non-retryable 503, deadline
expiry to 504, and DCP 4xx or other errors to
502 or 503 respectively.

A documentation response answers `public, max-age=300` with an `ETag` over its body, or
`no-store` when it asked for coverage facts while the coverage index was unreadable
(`dfo_platform.static_response.gapped_public_json`). The source index answers
`public, max-age=300` with a strong `ETag` and a gzip representation, both encoded once per
origin catalogue `build_id`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from concurrent.futures import Executor
from typing import TYPE_CHECKING

from dfo_provenance_docs import DocumentErrorReason
from fastapi import APIRouter, Request, Response

from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, path_registry_endpoints
from dfo_platform.refusal import TIMEOUT_DEFAULT
from dfo_platform.single_flight_cache import SingleFlightCache
from dfo_platform.static_response import FrozenJSON, freeze_public_json, gapped_public_json
from graph_api import routing
from graph_api.config import GRAPH_NODE_TIMEOUT_SECONDS, GRAPH_PROVENANCE_DOC_CACHE_SIZE
from graph_api.deps import GraphDeps, GraphDepsDep, log_queue_wait
from graph_api.doors.node import DcidTooLong
from graph_api.origin_models import DataCommonsRef, SourceNodeRef
from graph_api.source_models import (
    ChangeEntry,
    DatasetEntry,
    DocumentErrorEntry,
    Exclusion,
    Methodology,
    ProvenanceEntry,
    ScopeAndUnits,
    SourceCatalogue,
    SourceCatalogueEntry,
    SourceDetails,
    SourceDocumentation,
    SourceLicense,
    TransformationLogEntry,
    VersionHistoryEntry,
)
from graph_api.usage import usage_document
from graph_generation import GenerationProvider
from materialise.cadence import cadence_label
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from materialise.provenance_doc import (
    CoverageFacts,
    NotASourceNode,
    ProvenanceDoc,
    SourceDocumentNotFound,
    TooManyChildProvenances,
    coverage_facts_from_index,
    read_provenance_doc,
    with_coverage_facts,
)
from origin_catalogue import (
    BASE,
    DFO,
    UN_SYSTEM,
    CardFacts,
    Catalogue,
    DataCommons,
    ProvenanceRow,
    Registry,
    SourceRef,
    current_catalogue,
    entry_ref,
    load_registry,
)
from origin_catalogue.registry import Exclusion as RegistryExclusion

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


def _to_wire(doc: ProvenanceDoc) -> SourceDocumentation:
    """Map a provenance document to the wire model, converting the cadence token to its label.

    Missing scalars stay null, countries stays empty, a release date serves as
    `YYYY-MM-DD`, and `gaps` and `document_errors` pass through unchanged.
    """
    return SourceDocumentation(
        details=SourceDetails(
            name_of_source=doc.name_of_source,
            publisher=doc.publisher,
            source=SourceNodeRef.of(doc.source) if doc.source is not None else None,
            data_commons=(
                DataCommonsRef.of(doc.data_commons) if doc.data_commons is not None else None
            ),
            description=doc.description,
            source_url=doc.source_url,
            source_data_url=doc.source_data_url,
            citation=doc.citation,
        ),
        license=SourceLicense(
            type=doc.license_type,
            url=doc.license_url,
            update_cadence=cadence_label(doc.update_cadence),
            last_update=doc.last_update,
            next_expected_update=doc.next_expected_update,
        ),
        scope_and_units=ScopeAndUnits(
            countries=[],
            years_covered=doc.years_covered,
            granularity=doc.granularity,
            currencies=list(doc.currencies),
        ),
        methodology=Methodology(description=doc.methodology),
        data_notes=list(doc.data_notes),
        usage=usage_document(),
        transformation_log=[
            TransformationLogEntry(transformation=step.title, description=step.description)
            for step in doc.transformation_log
        ],
        version_history=[
            VersionHistoryEntry(
                date=release.date.isoformat(),
                title=release.title,
                changes=[
                    ChangeEntry(kind=change.kind.value, note=change.note)
                    for change in release.changes
                ],
            )
            for release in doc.version_history
        ],
        gaps=list(doc.gaps),
        document_errors=[
            DocumentErrorEntry(
                member=issue.member,
                line=issue.line,
                reason=(
                    issue.reason.value
                    if isinstance(issue.reason, DocumentErrorReason)
                    else issue.reason
                ),
            )
            for issue in doc.document_errors
        ],
    )


def _coverage_facts_of(
    deps: GraphDeps, provenance_dcids: tuple[str, ...]
) -> tuple[CoverageFacts, bool]:
    """Return the currencies and granularity the coverage index gives these Provenances, and
    whether the index could not be read, so the gaps those facts leave are transient: an
    unreadable index can answer a later response.

    Runs on the event loop, since `current()` returns the bound release's index without I/O
    and the index builds its per-provenance views when it loads. Both facts come from one
    index snapshot, so a release swap cannot mix two builds in one response.
    A failure reading the snapshot logs one WARNING and empties both facts.
    `materialise.provenance_doc.coverage_facts_from_index` then derives each
    fact under its own guard, so a failure logs a WARNING and empties only
    that fact. A failed snapshot read is transient.
    """
    try:
        index = deps.coverage.current()
    except Exception:
        logger.warning(
            "Graph sources door could not read the coverage index for Provenances %s; "
            "currencies serve [] and granularity serves null",
            list(provenance_dcids),
            exc_info=True,
        )
        return CoverageFacts(), True
    return coverage_facts_from_index(index, provenance_dcids), False


def new_provenance_doc_cache(
    *, executor: Executor, generation: GenerationProvider
) -> SingleFlightCache[tuple[str, str], ProvenanceDoc]:
    """The documentation route's cache of source documents, whose reads run on `executor`
    and whose entries key on `generation`'s stamp."""
    return SingleFlightCache(
        maxsize=GRAPH_PROVENANCE_DOC_CACHE_SIZE,
        flight_timeout_s=GRAPH_NODE_TIMEOUT_SECONDS,
        executor=executor,
        generation=generation,
    )


async def _read_provenance_doc(deps: GraphDeps, dcid: str) -> ProvenanceDoc:
    """Return `dcid`'s document without coverage facts from `deps.provenance_docs`, reading
    it on a miss. The key carries the request's catalogue build, which decides the children
    the document covers and its `source` and `data_commons`. Refresh the generation first,
    as the resolve door does, so the stamp is fresh after an idle gap.
    """
    await deps.generation.refresh_if_stale()
    deadline = time.monotonic() + GRAPH_NODE_TIMEOUT_SECONDS
    catalogue = current_catalogue()

    def read(flight_deadline: float) -> ProvenanceDoc:
        return read_provenance_doc(deps.graph, dcid, deadline=flight_deadline, catalogue=catalogue)

    return await deps.provenance_docs.aget_or_fetch(
        (dcid, catalogue.build_id), log_queue_wait(read), deadline=deadline
    )


async def _build_response(deps: GraphDeps, dcid: str) -> tuple[SourceDocumentation, bool]:
    """Read one source document, add this response's coverage facts and map it, and say
    whether any of its gaps is transient."""
    doc = await _read_provenance_doc(deps, dcid)
    facts, transient_gap = _coverage_facts_of(deps, doc.provenances)
    return _to_wire(with_coverage_facts(doc, facts)), transient_gap


def classify_sources_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify reader errors directly. The shared classifier chain also handles `DcidTooLong`
    from this route, so this door does not need a duplicate branch.
    """
    if isinstance(exc, SourceDocumentNotFound):
        return ProblemSpec(
            status=404, code="node_not_held", detail=str(exc), pointer="/dcid", retryable=False
        )
    if isinstance(exc, NotASourceNode):
        return ProblemSpec(
            status=400, code="request_invalid", detail=str(exc), pointer="/dcid", retryable=False
        )
    if isinstance(exc, TooManyChildProvenances):
        logger.error("Graph sources door refused a truncated child set: %s", exc)
        return ProblemSpec(status=503, code="service_unavailable", detail=str(exc), retryable=False)
    if isinstance(exc, GraphDeadlineExceededError):
        # This subclass maps to 504, so it must precede `GraphReadError`.
        return ProblemSpec(
            status=TIMEOUT_DEFAULT.status,
            code=TIMEOUT_DEFAULT.code,
            detail="the request's deadline expired during the graph documentation read; retry",
            retryable=TIMEOUT_DEFAULT.retryable,
        )
    if isinstance(exc, GraphReadError):
        logger.error("Graph sources door's DCP read failed: %s", exc, exc_info=exc)
        upstream_status = exc.upstream_status
        if isinstance(upstream_status, int) and 400 <= upstream_status < 500:
            return ProblemSpec(status=502, code="upstream_error", retryable=True)
        return ProblemSpec(status=503, code="service_unavailable", retryable=True)
    return None


async def get_documentation(dcid: str, deps: GraphDepsDep, request: Request) -> Response:
    if len(dcid) > routing.DCID_MAX_LENGTH:
        raise DcidTooLong(len(dcid))
    documentation, transient_gap = await asyncio.wait_for(
        _build_response(deps, dcid), timeout=GRAPH_NODE_TIMEOUT_SECONDS
    )
    return gapped_public_json(
        documentation.model_dump(mode="json"), request, transient_gap=transient_gap
    )


def _exclusion(exclusion: RegistryExclusion) -> Exclusion:
    return Exclusion(ref=exclusion.provenance, reason=exclusion.reason)


_NO_FACTS = CardFacts()


def _dataset_entry(ref: str, label: str, facts: CardFacts) -> DatasetEntry:
    return DatasetEntry(
        ref=ref,
        label=label,
        provenances=[],
        description=facts.description,
        license_type=facts.license_type,
        update_cadence=cadence_label(facts.update_cadence),
        years_covered=facts.years_covered,
        source_url=facts.source_url,
    )


def _datasets(rows: Iterable[ProvenanceRow], catalogue: Catalogue) -> list[DatasetEntry]:
    """Group provenance rows under their Dataset in first-seen order, each group
    with its card facts. A row with no Dataset forms its own group, keyed and
    labelled by the provenance."""
    groups: dict[str, DatasetEntry] = {}
    for row in rows:
        ref = entry_ref(row)
        group = groups.get(ref)
        if group is None:
            label = row.dataset.label if row.dataset is not None else row.name
            facts = catalogue.dataset_facts(ref) or _NO_FACTS
            group = groups[ref] = _dataset_entry(ref, label, facts)
        group.provenances.append(ProvenanceEntry(ref=row.id, label=row.name))
    return list(groups.values())


def build_source_catalogue(*, registry: Registry, catalogue: Catalogue) -> SourceCatalogue:
    """The source index: one entry per DFO source in provenance-table order, then each base
    registry entry, then each UN agency when the UN origin is on, each with the provenances
    `catalogue` admits under it. A provenance the registry excludes is listed under no
    source's datasets, whatever the loaded build holds. A registry entry the catalogue lists
    nothing under serves an empty `provenances`, as when the registry is newer than the loaded
    build."""
    excluded = {
        exclusion.provenance for base in registry.base for exclusion in base.exclusions or ()
    }
    rows_by_source: dict[tuple[DataCommons, str], list[ProvenanceRow]] = {}
    for row in catalogue.provenance_rows():
        if row.id not in excluded:
            rows_by_source.setdefault((row.data_commons, row.source.ref), []).append(row)

    def entry(
        source: SourceRef,
        origin: DataCommons,
        exclusions: tuple[Exclusion, ...] = (),
    ) -> SourceCatalogueEntry:
        return SourceCatalogueEntry(
            source=SourceNodeRef.of(source),
            data_commons=DataCommonsRef.of(origin),
            datasets=_datasets(rows_by_source.get((origin, source.ref), ()), catalogue),
            exclusions=list(exclusions),
        )

    dfo_sources = dict.fromkeys(
        row.source for row in catalogue.provenance_rows() if row.data_commons == DFO
    )
    return SourceCatalogue(
        sources=[
            *(entry(source, DFO) for source in dfo_sources),
            *(
                entry(
                    SourceRef(ref=base.source, label=base.label),
                    BASE,
                    tuple(_exclusion(exclusion) for exclusion in base.exclusions or ()),
                )
                for base in registry.base
            ),
            *(
                entry(SourceRef(ref=agency.provenance, label=agency.label), UN_SYSTEM)
                for agency in (registry.un_system if catalogue.un_system_enabled else ())
            ),
        ]
    )


@functools.cache
def _packaged_registry() -> Registry:
    return load_registry()


@functools.lru_cache(maxsize=1)
def _frozen_source_catalogue(build_id: str) -> FrozenJSON:
    """The source index for the bound catalogue, encoded once per catalogue build.

    `build_id` keys the cache and must be the bound catalogue's own. The packaged registry
    is fixed for the process, so the build alone decides the body."""
    catalogue = current_catalogue()
    document = build_source_catalogue(registry=_packaged_registry(), catalogue=catalogue)
    return freeze_public_json(document.model_dump(mode="json"), max_age=300)


async def list_sources(request: Request) -> Response:
    # This literal route makes no graph reads.
    return _frozen_source_catalogue(current_catalogue().build_id).respond(request)


def build_sources_door() -> DoorContribution:
    # Load the registry now, so an invalid one fails startup rather than the first request.
    _packaged_registry()
    router = APIRouter()
    # `response_model` documents the response body in OpenAPI.
    router.add_api_route(
        "/api/graph/documentation/{dcid:dcid}",
        get_documentation,
        methods=["GET"],
        response_model=SourceDocumentation,
    )
    router.add_api_route(
        "/api/graph/sources",
        list_sources,
        methods=["GET"],
        response_model=SourceCatalogue,
    )
    return DoorContribution(
        router=router,
        registry_endpoints=path_registry_endpoints(
            router, {"/api/graph/documentation/{dcid:dcid}": "graph.documentation"}
        ),
        classify=classify_sources_exception,
        exception_types=(
            SourceDocumentNotFound,
            NotASourceNode,
            TooManyChildProvenances,
            GraphDeadlineExceededError,
            GraphReadError,
        ),
        exception_exemplars=(
            SourceDocumentNotFound("exemplar/dcid"),
            NotASourceNode("exemplar/dcid", ("Country",)),
            TooManyChildProvenances("exemplar/dataset"),
            GraphDeadlineExceededError("exemplar"),
            GraphReadError("exemplar", upstream_status=502),
        ),
    )
