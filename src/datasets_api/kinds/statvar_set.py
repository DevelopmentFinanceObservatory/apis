"""Resolves one StatVar and mints its fully pinned spec.

The shape resolves before any write. An unresolved StatVar refuses `member_not_resolvable`.
"""

from __future__ import annotations

from collections.abc import Sequence

from datasets_api.adapter import (
    build_identity_document,
    prime_head_signal,
    to_materialisation_request,
)
from datasets_api.admission import refuse_excel_over_estimate, sync_estimate, within_sync_budget
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.operations.submit import submit_over_budget
from datasets_api.serve import raw_fetch_guard, serve_definition
from datasets_api.shape import resolve_shape
from datasets_api.timings import phase
from spec_registry.identity import recompute_identity


def mint_statvar_set(
    statvars: Sequence[str],
    *,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
    trace_id: str | None = None,
) -> Manifest:
    """Mints the pinned spec of the one StatVar in `statvars`, which `StatVarSetInput` caps at
    one. The identity is the document the registration stores, so the mint materialises what
    the registry reads back. Resolution can raise `MemberNotResolvable`."""
    (stat_var,) = statvars
    with phase("resolve"):
        shape = resolve_shape(stat_var, graph=deps.graph)
        recomputed = recompute_identity(
            build_identity_document(shape, scope=scope, graph=deps.graph)
        )
        deps.registry.register_definition(
            identity_definition=recomputed.identity,
            schema_version=1,
            supplied_spec_id=recomputed.spec_id,
            trace_id=trace_id,
        )
        # No manifest returns before the registry write is read back.
        stored = deps.spec_reader.read_definition(recomputed.spec_id)
        assert stored is not None, "definition missing immediately after its own registration"
        request = to_materialisation_request(
            stored.identity, shape=shape, code_index=deps.code_index
        )

    # Keep the estimate and fetch on the same coverage index and stamp.
    coverage_snapshot = deps.coverage.snapshot()
    # Refuse oversized Excel requests before fetching. Submit over-budget work as an operation.
    estimate = sync_estimate(
        (request,), index=coverage_snapshot.index, member_concurrency=deps.member_concurrency
    )
    refuse_excel_over_estimate(estimate, format=options.format)
    submit_over_budget(
        subject_id=stored.spec_id,
        options=options,
        estimate=estimate,
        coverage_stamp=coverage_snapshot.stamp,
        operation_store=deps.operation_store,
        task_client=deps.task_client,
        within_budget=within_sync_budget(
            estimate,
            sync_budget_rows=deps.settings.sync_budget_rows,
            sync_budget_requests=deps.settings.sync_budget_requests,
        ),
        async_max_rows=deps.settings.async_max_rows,
        async_max_requests=deps.settings.async_max_requests,
        members=(request.stat_var,),
    )
    priming = prime_head_signal(
        (request,),
        subject_id=stored.spec_id,
        graph=deps.graph,
        sdmx=deps.sdmx,
        head_store=deps.head_store,
        materialise_wait_seconds=deps.settings.materialise_wait_seconds,
        revision_reader=deps.revision_reader,
        materialise_semaphore=deps.materialise_semaphore,
        coverage=coverage_snapshot.index,
        deadline=deps.deadline,
    )
    return serve_definition(
        stored=stored,
        requests=(request,),
        options=options,
        deps=deps.to_serve_deps(),
        trace_id=trace_id,
        signal=priming.signal,
        expected_content_version=priming.expected_content_version,
        precomputed_probe=priming.probe,
        # Lineage only. Excluded from content_version.
        ceiling=raw_fetch_guard(estimate, floor=deps.ceiling),
        coverage_stamp=coverage_snapshot.stamp,
        coverage_index=coverage_snapshot.index,
    )
