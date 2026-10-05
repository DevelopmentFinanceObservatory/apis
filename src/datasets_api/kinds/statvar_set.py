"""Resolves each StatVar, then mints one member spec or a collection of sorted unique spec ids.

All shapes resolve before writes. Unresolved members are reported together as
`member_not_resolvable`.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from datasets_api.adapter import (
    build_identity_document,
    prime_head_signal,
    to_materialisation_request,
)
from datasets_api.admission import refuse_excel_over_estimate, sync_estimate, within_sync_budget
from datasets_api.codes_materialise import MemberNotResolvable
from datasets_api.manifest_models import Manifest, WarningModel
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.operations.submit import submit_over_budget
from datasets_api.preview import preview_requests
from datasets_api.preview_models import PreviewEnvelope
from datasets_api.serve import raw_fetch_guard, serve_collection, serve_definition
from datasets_api.shape import StatVarShape, resolve_shape
from datasets_api.timings import phase
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from dfo_platform.executor import submit_with_context
from materialise import MaterialisationRequest
from spec_registry import MemberDefinition
from spec_registry.identity import recompute_identity
from spec_registry.reader import StoredDefinition


def _resolve_member(
    stat_var: str, *, scope: Scope | None, deps: MintDeps
) -> tuple[str, dict[str, object], StatVarShape]:
    """Resolves a StatVar shape and builds its canonical identity without writing.

    The identity is the document a registration stores, so a preview materialises what a mint
    reads back. Resolution can raise `MemberNotResolvable`. Callers aggregate failures before
    registering members.
    """
    shape = resolve_shape(stat_var, graph=deps.graph)
    recomputed = recompute_identity(build_identity_document(shape, scope=scope, graph=deps.graph))
    return recomputed.spec_id, recomputed.identity, shape


def mint_statvar_set(
    statvars: Sequence[str],
    *,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
    trace_id: str | None = None,
) -> Manifest:
    serve_deps = deps.to_serve_deps()
    # Keep the branch choice after `phase("resolve")` closes because serving runs outside it.
    single: bool

    with phase("resolve"):
        # Bound parallel resolution by `member_concurrency` and collect results in input order.
        pool = ThreadPoolExecutor(max_workers=max(1, min(deps.member_concurrency, len(statvars))))
        try:
            futures = [
                submit_with_context(pool, _resolve_member, stat_var, scope=scope, deps=deps)
                for stat_var in statvars
            ]
            failing: list[str] = []
            resolved: list[tuple[str, dict[str, object], StatVarShape]] = []
            for future in futures:
                try:
                    resolved.append(future.result())
                except MemberNotResolvable as exc:
                    failing.extend(exc.members)
        finally:
            # Cancel queued work and wait for running resolutions before returning.
            pool.shutdown(wait=True, cancel_futures=True)
        if failing:
            raise MemberNotResolvable(members=tuple(failing))

        single = len(resolved) == 1
        if single:
            spec_id, document, shape = resolved[0]
            deps.registry.register_definition(
                identity_definition=document,
                schema_version=1,
                supplied_spec_id=spec_id,
                trace_id=trace_id,
            )
            # No manifest returns before the registry write is read back.
            stored = deps.spec_reader.read_definition(spec_id)
            assert stored is not None, "definition missing immediately after its own registration"
            request = to_materialisation_request(
                stored.identity, shape=shape, code_index=deps.code_index
            )
        else:
            shapes_by_spec_id = {spec_id: shape for spec_id, _document, shape in resolved}
            registered_collection = deps.registry.register_collection(
                members=[
                    MemberDefinition(
                        identity_definition=document, schema_version=1, supplied_spec_id=spec_id
                    )
                    for spec_id, document, _shape in resolved
                ],
                trace_id=trace_id,
            )
            stored_collection = deps.spec_reader.read_collection(
                registered_collection.collection_id
            )
            assert stored_collection is not None, (
                "collection missing immediately after its own registration"
            )

            stored_members = deps.spec_reader.read_definitions(stored_collection.member_spec_ids)
            requests: list[tuple[StoredDefinition, MaterialisationRequest]] = []
            for member_spec_id in stored_collection.member_spec_ids:
                member_stored = stored_members[member_spec_id]
                shape = shapes_by_spec_id[member_spec_id]
                requests.append(
                    (
                        member_stored,
                        to_materialisation_request(
                            member_stored.identity, shape=shape, code_index=deps.code_index
                        ),
                    )
                )

            extra_warnings: tuple[WarningModel, ...] = ()
            if len(stored_collection.member_spec_ids) < len(resolved):
                _title, detail, severity = DATASETS_WARNING_CATALOGUE["members_deduplicated"]
                extra_warnings = (
                    WarningModel(code="members_deduplicated", severity=severity, message=detail),
                )

    # Keep the estimate and fetch on the same coverage index and stamp.
    coverage_snapshot = deps.coverage.snapshot()
    # Refuse oversized Excel requests before fetching. Submit over-budget work as an operation.
    # One signal for the single definition, or one per member in `requests`
    # order for the collection (`combine=True`/`False`).
    # See `prime_head_signal`'s own docstring.
    if single:
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
            combine=True,
            head_store=deps.head_store,
            materialise_wait_seconds=deps.settings.materialise_wait_seconds,
            revision_reader=deps.revision_reader,
            materialise_semaphore=deps.materialise_semaphore,
            coverage=coverage_snapshot.index,
            deadline=deps.deadline,
        )
        manifest = serve_definition(
            stored=stored,
            requests=(request,),
            options=options,
            deps=serve_deps,
            trace_id=trace_id,
            signal=priming.signal,
            expected_content_version=priming.expected_content_version,
            precomputed_probe=priming.probe,
            # Lineage only. Excluded from content_version.
            ceiling=raw_fetch_guard(estimate, floor=deps.ceiling),
            coverage_stamp=coverage_snapshot.stamp,
            coverage_index=coverage_snapshot.index,
        )
        return manifest
    bare_requests = [member_request for _member_stored, member_request in requests]
    estimate = sync_estimate(
        bare_requests, index=coverage_snapshot.index, member_concurrency=deps.member_concurrency
    )
    refuse_excel_over_estimate(estimate, format=options.format)
    submit_over_budget(
        subject_id=stored_collection.collection_id,
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
    )
    priming = prime_head_signal(
        bare_requests,
        subject_id=stored_collection.collection_id,
        graph=deps.graph,
        sdmx=deps.sdmx,
        combine=False,
        head_store=deps.head_store,
        materialise_wait_seconds=deps.settings.materialise_wait_seconds,
        revision_reader=deps.revision_reader,
        materialise_semaphore=deps.materialise_semaphore,
        coverage=coverage_snapshot.index,
        deadline=deps.deadline,
    )
    manifest = serve_collection(
        stored=stored_collection,
        requests=requests,
        options=options,
        deps=serve_deps,
        trace_id=trace_id,
        extra_warnings=extra_warnings,
        signal=priming.signal,
        expected_content_version=priming.expected_content_version,
        precomputed_probe=priming.probe,
        # Lineage only. Excluded from content_version.
        ceiling=raw_fetch_guard(estimate, floor=deps.ceiling),
        coverage_stamp=coverage_snapshot.stamp,
        coverage_index=coverage_snapshot.index,
    )
    return manifest


def preview_statvar_set(
    statvars: Sequence[str], *, scope: Scope | None, limit: int, deps: MintDeps
) -> PreviewEnvelope:
    """Write-free counterpart to `mint_statvar_set`.

    It resolves in the same bounded pool and aggregates `member_not_resolvable` failures.
    Requests use resolved identities. The preview returns one member per input StatVar without
    registering definitions or a collection.
    """
    with phase("resolve"):
        pool = ThreadPoolExecutor(max_workers=max(1, min(deps.member_concurrency, len(statvars))))
        try:
            futures = [
                submit_with_context(pool, _resolve_member, stat_var, scope=scope, deps=deps)
                for stat_var in statvars
            ]
            failing: list[str] = []
            resolved: list[tuple[str, dict[str, object], StatVarShape]] = []
            for future in futures:
                try:
                    resolved.append(future.result())
                except MemberNotResolvable as exc:
                    failing.extend(exc.members)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if failing:
            raise MemberNotResolvable(members=tuple(failing))

        requests = tuple(
            to_materialisation_request(document, shape=shape, code_index=deps.code_index)
            for _spec_id, document, shape in resolved
        )
    return preview_requests(requests, deps=deps, limit=limit)
