"""Materialises previews after the mint resolver has resolved each request.

A preview uses mint's graph checks, revision reader, and content-version calculation. A revision
change raises `UpstreamVersionChangedError`. Previews return before writes to the registry,
artifacts, versions, or heads.

`preview_requests` previews each request alone, as a collection mints each member alone:
`materialise_preview` probes facets and selects them for that one member. `preview_definition`
previews the members of one definition as the group its mint serves: `materialise_group_preview`
selects one facet set for the whole group, so a member with no observation at the bound entities
contributes no rows and the preview reports the definition mint's `total_rows` and
`content_version`. Either way the preview fetches every SDMX row a mint of the same identity
fetches, since `total_rows` and `content_version` cover every survivor; `limit` bounds only the
sample. A preview has no asynchronous path, so it is admitted as an inline mint is: one coverage
snapshot prices the requests, an estimate over the sync budget refuses `over_ceiling` before any
fetch, and the raw-fetch guard and survivor ceiling are the mint's (`serve.raw_fetch_guard`). A
member the estimate cannot price, such as a subject-only StatVar read at every country, therefore
refuses `over_ceiling` mid-stream where its mint would. The members `preview_requests` previews
share one `PreviewBudget`, so their raw reads draw on one guard and their survivors together
refuse past the ceiling, as a collection mint's do; `materialise_group_preview` pools a group's
survivors itself, as the group mint does. Every member `preview_requests` previews reads rate
tables through one `RateTableCache.for_request` view, so the members convert against one version
of each table even when a table refreshes mid-request. Either preview reads under one slot of the
materialisation semaphore a mint holds (`bounded_materialise_slot`), so it waits, or refuses with
retryable backpressure, as a mint does under the same load. Each preview carries the origin
catalogue's warnings (`datasets_api.origin`), and its `withheld_by_registry` re-read runs at most
once per preview call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from dfo_frame_grammar import canonical_frame_bytes

from datasets_api.admission import (
    bounded_materialise_slot,
    refuse_preview_over_budget,
    sync_estimate,
)
from datasets_api.manifest import build_citation, coverage_model, warning_model
from datasets_api.mint_deps import MintDeps
from datasets_api.origin import (
    origin_reads,
    stale_warnings,
    un_unavailable_warnings,
    withheld_warning,
    withheld_warnings,
)
from datasets_api.preview_models import PreviewEnvelope, PreviewFrameModel, PreviewMemberModel
from datasets_api.serve import raw_fetch_guard
from datasets_api.timings import phase
from materialise import (
    MaterialisationRequest,
    PreviewBudget,
    PreviewResult,
    materialise_group_preview,
    materialise_preview,
)
from materialise.member_conversion import refuse_if_every_row_omitted
from materialise.provenance_doc import read_source_citations
from materialise.rate_tables import RateTableCache
from origin_catalogue import UnBlockedPlaces, current_catalogue
from series_coverage.index import CoverageIndex


def _member_model(
    stat_var: str,
    outcome: PreviewResult,
    *,
    deps: MintDeps,
    un_blocked: UnBlockedPlaces,
) -> PreviewMemberModel:
    """The wire entry for one preview result, named `stat_var`."""
    # Read provenance from the selected facet. A missing ref yields empty citation input because
    # `build_citation` accepts strings.
    provenance = outcome.selections.get("provenance")
    provenance_refs = [provenance.ref] if provenance is not None and provenance.ref else []
    with phase("citation"):
        citation, citation_warnings = build_citation(
            provenance_refs=provenance_refs,
            source_citations=read_source_citations(
                deps.graph, provenance_refs, catalogue=current_catalogue()
            ),
        )
    warnings = [
        *(warning_model(w) for w in outcome.warnings),
        *citation_warnings,
        *un_unavailable_warnings(un_blocked),
        *stale_warnings(),
    ]
    # Decode the canonical frame1 document so preview rows match the bytes a mint stores.
    document = json.loads(canonical_frame_bytes(outcome.frame))
    return PreviewMemberModel(
        stat_var=stat_var,
        content_version=outcome.content_version,
        coverage=coverage_model(outcome.coverage),
        citation=citation,
        warnings=warnings,
        sample=PreviewFrameModel(columns=document["columns"], rows=document["rows"]),
        total_rows=outcome.total_rows,
        truncated=outcome.truncated,
    )


def _preview_member(
    request: MaterialisationRequest,
    *,
    deps: MintDeps,
    rate_cache: RateTableCache,
    limit: int,
    budget: PreviewBudget,
    coverage: CoverageIndex,
) -> tuple[PreviewMemberModel, PreviewResult]:
    """One member's preview, and the result `withheld_warning` reads."""
    with origin_reads(deps.graph, availability=deps.sdmx, deadline=deps.deadline) as un_blocked:
        outcome = materialise_preview(
            request,
            sdmx=deps.sdmx,
            graph=deps.graph,
            limit=limit,
            ceiling=budget.ceiling,
            revision_reader=deps.revision_reader,
            rate_cache=rate_cache,
            coverage=coverage,
            deadline=deps.deadline,
            refuse_over_ceiling=True,
            budget=budget,
            refuse_when_every_row_omitted=False,
        )
    member = _member_model(request.stat_var, outcome, deps=deps, un_blocked=un_blocked)
    return member, outcome


def _admit_preview(
    requests: Sequence[MaterialisationRequest], *, deps: MintDeps
) -> tuple[CoverageIndex, int]:
    """Price `requests` on one coverage snapshot and refuse a preview over the sync budget.

    Returns the snapshot's index, which every fetch reads, and the raw-fetch guard's ceiling.
    """
    with phase("coverage"):
        coverage = deps.coverage.snapshot().index
    with phase("estimate"):
        estimate = sync_estimate(
            requests, index=coverage, member_concurrency=deps.member_concurrency
        )
    refuse_preview_over_budget(
        estimate,
        sync_budget_rows=deps.settings.sync_budget_rows,
        sync_budget_requests=deps.settings.sync_budget_requests,
    )
    return coverage, raw_fetch_guard(estimate, floor=deps.ceiling)


def preview_definition(
    requests: Sequence[MaterialisationRequest], *, family_ref: str, deps: MintDeps, limit: int
) -> PreviewEnvelope:
    """Preview the members of one definition as the group its mint serves, in one entry.

    `materialise_group_preview` runs the group mint's generation, so the entry's `total_rows`
    and `content_version` are the mint's and its sample holds rows of every member. A
    one-member definition names its StatVar; a family names `family_ref`, the reference a
    summed frame of the definition carries as its `stat_var`.
    """
    coverage, ceiling = _admit_preview(requests, deps=deps)
    with (
        bounded_materialise_slot(
            deps.materialise_semaphore,
            wait_seconds=deps.settings.materialise_wait_seconds,
            deadline=deps.deadline,
        ),
        phase("preview"),
        origin_reads(deps.graph, availability=deps.sdmx, deadline=deps.deadline) as un_blocked,
    ):
        outcome = materialise_group_preview(
            requests,
            sdmx=deps.sdmx,
            graph=deps.graph,
            revision_reader=deps.revision_reader,
            limit=limit,
            ceiling=ceiling,
            member_concurrency=deps.member_concurrency,
            rate_cache=deps.rate_cache,
            coverage=coverage,
            deadline=deps.deadline,
        )
    stat_var = requests[0].stat_var if len(requests) == 1 else family_ref
    member = _member_model(stat_var, outcome, deps=deps, un_blocked=un_blocked)
    withheld = withheld_warnings(
        deps.graph,
        [(request, outcome.coverage) for request in requests],
        default_scope=outcome.default_scope,
        deadline=deps.deadline,
    )
    if withheld:
        member = member.model_copy(update={"warnings": [*member.warnings, *withheld]})
    return PreviewEnvelope(members=[member])


def preview_requests(
    requests: Sequence[MaterialisationRequest], *, deps: MintDeps, limit: int
) -> PreviewEnvelope:
    """Calls `materialise_preview` serially for each request and preserves request order.
    Refuses `currency_unmapped` or `deflator_missing` as a collection mint does: only when no
    member serves a row and some member's conversion omitted rows."""
    # The estimate and every fetch read one coverage snapshot, as a mint's do.
    coverage, ceiling = _admit_preview(requests, deps=deps)
    budget = PreviewBudget(ceiling)
    rate_cache = deps.rate_cache.for_request()
    with (
        bounded_materialise_slot(
            deps.materialise_semaphore,
            wait_seconds=deps.settings.materialise_wait_seconds,
            deadline=deps.deadline,
        ),
        phase("preview"),
    ):
        previewed = [
            _preview_member(
                request,
                deps=deps,
                rate_cache=rate_cache,
                limit=limit,
                budget=budget,
                coverage=coverage,
            )
            for request in requests
        ]
    # A collection mint refuses an omission only when no member serves a row.
    refuse_if_every_row_omitted(
        served_rows=sum(outcome.coverage.rows for _member, outcome in previewed),
        records=(outcome.conversion_record for _member, outcome in previewed),
    )
    members = [member for member, _outcome in previewed]
    # A member that binds no entity is checked across every member's default place scope, which
    # holds its own. A place where it holds no data at all has nothing withheld.
    default_scope = tuple(
        dict.fromkeys(place for _member, outcome in previewed for place in outcome.default_scope)
    )
    withheld = withheld_warning(
        deps.graph,
        [
            (request, outcome.coverage)
            for request, (_member, outcome) in zip(requests, previewed, strict=True)
        ],
        default_scope=default_scope,
        deadline=deps.deadline,
    )
    if withheld is not None:
        index, warning = withheld
        members[index] = members[index].model_copy(
            update={"warnings": [*members[index].warnings, warning]}
        )
    return PreviewEnvelope(members=members)
