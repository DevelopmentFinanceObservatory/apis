"""Reads and materialises a registered definition, or registers a supplied sidecar first.

A miss requires a sidecar. Member resolution may refuse `member_not_resolvable`,
`spec_not_exportable`, `over_ceiling`, or `role_signature_incompatible`. Scope roles and
reproducibility are checked before the sidecar hash and registry write, so refusals leave no
row. Hash mismatches raise `SpecIdMismatch`. A registry hit serves stored data. The kind checks
supplied sidecars for hash consistency. Missing definitions raise `SpecNotFound`.

`resolved_stat_var_dcids` asserts that each posted StatVar is a member of the identity's
family, dominated or not. Member resolution takes the list as extra candidates, so a posted root
can prove a truncated page complete, and `_resolve_members` then refuses
`MemberSelectionMismatch` when a posted dcid is outside the family. The mint serves the
antichain-minimal members, posted or omitted. Every mint and preview path resolves through
`_resolve_members` before a registry write, so a refused selection leaves no row. Scope surgery
replaces supplied members, preserves the rest, and rehashes the document. `scope.facets` can bind
the provenance that names a family's dataset, so members resolve under the facets the scope
installs. Invalid roles refuse as
`InvalidRoleRef` or `RoleIncompatible` before a write. An unchanged identity reuses its record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from dfo_spec_kit.identity import IdentityError

from code_index.query import CodeIndex
from datasets_api.adapter import (
    build_identity_document,
    check_reproducible,
    prime_head_signal,
    resolve_definition_members,
    to_materialisation_request,
    verify_sidecar_definition,
)
from datasets_api.admission import refuse_excel_over_estimate, sync_estimate, within_sync_budget
from datasets_api.codes_materialise import (
    CurrencyRoleUndeclared,
    DefinitionNotReproducible,
    MalformedSidecarIdentity,
    MemberNotResolvable,
    MemberSelectionMismatch,
    SpecNotFound,
)
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.models import CanonicalDefinition
from datasets_api.operations.submit import submit_over_budget
from datasets_api.preview import preview_definition
from datasets_api.preview_models import PreviewEnvelope
from datasets_api.serve import raw_fetch_guard, serve_definition
from datasets_api.shape import StatVarShape
from datasets_api.surgery import apply_scope_surgery
from datasets_api.timings import phase
from materialise.graph import GraphReader
from spec_registry import RecomputedIdentity, SpecIdMismatch, recompute_identity
from statvar_members import MemberEnumerator

_POINTER = "/input/spec_id"
_STAT_VAR_PREFIX = "dcid:"


class MemberResolutionDeps(Protocol):
    """The graph reads sidecar admission makes. `MintDeps` and the definition door's
    `DefinitionDeps` both carry them."""

    graph: GraphReader
    enumerator: MemberEnumerator
    code_index: CodeIndex


def _resolve_members(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
    selection: Sequence[str] | None = None,
    scope: Scope | None = None,
) -> tuple[StatVarShape, ...]:
    """Resolves an identity and attaches `/input/spec_id` to `MemberNotResolvable`.

    Accepts a registered identity or an unregistered sidecar. A `scope` that supplies `facets`
    replaces the identity's own before resolution, as scope surgery does, and one that supplies
    `entities` supplies the role keys the role class is elected under, so the family is the
    one the surgered identity resolves. `selection` is the request's
    `resolved_stat_var_dcids`. It seeds member resolution, and each posted dcid must be in the
    family before antichain reduction, whichever path resolved it: the pinned arm, the code
    index, or the graph walk. A member a broader member dominates passes and is not served.
    Raises `MemberSelectionMismatch` naming each posted dcid outside the family. Returns the
    antichain-minimal shapes, so `None` and an empty list check nothing.
    """
    if scope is not None and scope.facets is not None:
        identity = apply_scope_surgery(
            identity, scope=scope.model_copy(update={"entities": None, "window": None})
        )
    try:
        resolved = resolve_definition_members(
            identity,
            spec_id=spec_id,
            graph=graph,
            enumerator=enumerator,
            code_index=code_index,
            seed=selection,
            role_keys=(
                frozenset(scope.entities)
                if scope is not None and scope.entities is not None
                else None
            ),
        )
    except MemberNotResolvable as exc:
        exc.pointer = _POINTER
        raise
    if selection:
        foreign = tuple(
            (index, dcid) for index, dcid in enumerate(selection) if dcid not in resolved.family
        )
        if foreign:
            raise MemberSelectionMismatch(foreign=foreign)
    return resolved.shapes


def _check_own_roles(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    scope: Scope | None,
    shapes: Sequence[StatVarShape],
    deps: MemberResolutionDeps,
    selection: Sequence[str] | None = None,
) -> None:
    """Checks `identity`'s own bound roles against the shape `identity` resolves to
    (`check_reproducible`).

    `shapes` are the request's, resolved under `scope`. A scope that binds entities under
    other role keys can elect another role class, so when the identity's roles fail against
    `shapes` under such a scope, the identity resolves again under its own role keys and
    the scope's facets, and is checked against that shape. An identity whose roles hold
    against `shapes` resolves once, so a truncated walk that a posted member proved complete
    is not walked again without it. `selection` seeds the second resolution and is not
    checked against its family, which is not the one served."""
    own_roles = frozenset(cast("Mapping[str, object]", identity.get("entities", {})))
    try:
        check_reproducible(identity, spec_id=spec_id, shape=shapes[0])
    except DefinitionNotReproducible:
        if scope is None or scope.entities is None or frozenset(scope.entities) == own_roles:
            raise
    else:
        return
    facets_only = scope.model_copy(update={"entities": None, "window": None})
    try:
        own = resolve_definition_members(
            apply_scope_surgery(identity, scope=facets_only)
            if scope.facets is not None
            else identity,
            spec_id=spec_id,
            graph=deps.graph,
            enumerator=deps.enumerator,
            code_index=deps.code_index,
            seed=selection,
        )
    except MemberNotResolvable as exc:
        exc.pointer = _POINTER
        raise
    check_reproducible(identity, spec_id=spec_id, shape=own.shapes[0])


def admit_sidecar(
    definition: CanonicalDefinition,
    *,
    spec_id: str,
    scope: Scope | None,
    deps: MemberResolutionDeps,
    selection: Sequence[str] | None,
) -> tuple[RecomputedIdentity, tuple[StatVarShape, ...]]:
    """Checks an unregistered sidecar and returns its recomputed identity (the canonical
    document, its bytes and digest) and resolved shapes.

    `recompute_identity` canonicalises the posted identity once. Member resolution, the scope
    role check and the reproducibility check read that canonical document, the one a
    registration stores, so a mint and a preview of one sidecar resolve and materialise the
    same identity whichever spelling the caller posted. Nothing is written.

    Raises `MalformedSidecarIdentity` for an identity the identity core refuses, each member
    resolution refusal, `InvalidRoleRef` or `RoleIncompatible` for a scope role, `SpecIdMismatch`
    when the identity does not hash to `spec_id`, and `DefinitionNotReproducible`, in that order.
    """
    try:
        recomputed = recompute_identity(definition.identity)
        identity = recomputed.identity
        shapes = _resolve_members(
            identity,
            spec_id=spec_id,
            graph=deps.graph,
            enumerator=deps.enumerator,
            code_index=deps.code_index,
            selection=selection,
            scope=scope,
        )
    except IdentityError as exc:
        raise MalformedSidecarIdentity(detail=str(exc), pointer=_POINTER) from exc
    if scope is not None and scope.entities is not None:
        for shape in shapes:
            build_identity_document(shape, scope=scope, graph=deps.graph)
    if recomputed.spec_id != spec_id:
        raise SpecIdMismatch(supplied=spec_id, computed=recomputed.spec_id)
    _check_own_roles(
        identity, spec_id=spec_id, scope=scope, shapes=shapes, deps=deps, selection=selection
    )
    return recomputed, shapes


def mint_spec_id(
    spec_id: str,
    *,
    definition: CanonicalDefinition | None,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
    resolved_stat_var_dcids: Sequence[str] | None = None,
    trace_id: str | None = None,
) -> Manifest:
    with phase("resolve"):
        stored = deps.spec_reader.read_definition(spec_id)
        # Members resolve before any write, under the facets and entities `scope` installs,
        # so the shapes are those of the surgered identity.
        if stored is None:
            if definition is None:
                raise SpecNotFound(subject_id=spec_id, pointer="/input/spec_id")
            # Resolve and validate the sidecar before writing. Refusals leave no registry row.
            _recomputed, shapes = admit_sidecar(
                definition,
                spec_id=spec_id,
                scope=scope,
                deps=deps,
                selection=resolved_stat_var_dcids,
            )
            deps.registry.register_definition(
                identity_definition=definition.identity,
                schema_version=definition.schema_version,
                supplied_spec_id=spec_id,
                trace_id=trace_id,
            )
            # No manifest returns before the registry write is read back.
            stored = deps.spec_reader.read_definition(spec_id)
            assert stored is not None, "definition missing immediately after its own registration"
        else:
            if definition is not None:
                # The registered identity supplies members. Verify the sidecar for consistency.
                try:
                    verify_sidecar_definition(definition, spec_id=spec_id)
                except IdentityError as exc:
                    raise MalformedSidecarIdentity(detail=str(exc), pointer=_POINTER) from exc
            shapes = _resolve_members(
                stored.identity,
                spec_id=stored.spec_id,
                graph=deps.graph,
                enumerator=deps.enumerator,
                code_index=deps.code_index,
                selection=resolved_stat_var_dcids,
                scope=scope,
            )
            _check_own_roles(
                stored.identity,
                spec_id=stored.spec_id,
                scope=scope,
                shapes=shapes,
                deps=deps,
                selection=resolved_stat_var_dcids,
            )
            if scope is not None and scope.entities is not None:
                for shape in shapes:
                    build_identity_document(shape, scope=scope, graph=deps.graph)

        if scope is not None:
            surgered = apply_scope_surgery(stored.identity, scope=scope)
            recomputed = recompute_identity(surgered)
            if recomputed.spec_id != stored.spec_id:
                deps.registry.register_definition(
                    identity_definition=surgered,
                    schema_version=stored.schema_version,
                    supplied_spec_id=recomputed.spec_id,
                    trace_id=trace_id,
                )
                # No manifest returns before the registry write is read back.
                stored = deps.spec_reader.read_definition(recomputed.spec_id)
                assert stored is not None, (
                    "surgered definition missing immediately after its own registration"
                )

        try:
            requests = tuple(
                to_materialisation_request(stored.identity, shape=shape, code_index=deps.code_index)
                for shape in shapes
            )
        except CurrencyRoleUndeclared as exc:
            exc.pointer = _POINTER
            raise
    # Keep the estimate and fetch on the same coverage index and stamp.
    with phase("coverage"):
        coverage_snapshot = deps.coverage.snapshot()
    # Refuse oversized Excel requests before fetching. Submit over-budget work as an operation.
    with phase("estimate"):
        estimate = sync_estimate(
            requests, index=coverage_snapshot.index, member_concurrency=deps.member_concurrency
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
        members=[request.stat_var for request in requests],
    )
    # One combined signal covers every member.
    priming = prime_head_signal(
        requests,
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
        requests=requests,
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
    return manifest


def preview_spec_id(
    spec_id: str,
    *,
    definition: CanonicalDefinition | None,
    scope: Scope | None,
    limit: int,
    deps: MintDeps,
    resolved_stat_var_dcids: Sequence[str] | None = None,
) -> PreviewEnvelope:
    """Write-free counterpart to `mint_spec_id` with the same resolution and validation.

    It materialises the canonical form of the sidecar or surgered identity, the document a mint
    registers and reads, without registering a definition or reporting a minted `spec_id`. The
    members preview as the one group the mint serves (`preview.preview_definition`), so the
    envelope holds one entry with the mint's `total_rows` and `content_version`.
    """
    with phase("resolve"):
        stored = deps.spec_reader.read_definition(spec_id)
        identity: Mapping[str, object]
        if stored is None:
            if definition is None:
                raise SpecNotFound(subject_id=spec_id, pointer="/input/spec_id")
            # The canonical identity is the document a mint of this sidecar registers and reads.
            recomputed, shapes = admit_sidecar(
                definition,
                spec_id=spec_id,
                scope=scope,
                deps=deps,
                selection=resolved_stat_var_dcids,
            )
            identity = recomputed.identity
        else:
            if definition is not None:
                try:
                    verify_sidecar_definition(definition, spec_id=spec_id)
                except IdentityError as exc:
                    raise MalformedSidecarIdentity(detail=str(exc), pointer=_POINTER) from exc
            identity = stored.identity
            shapes = _resolve_members(
                identity,
                spec_id=spec_id,
                graph=deps.graph,
                enumerator=deps.enumerator,
                code_index=deps.code_index,
                selection=resolved_stat_var_dcids,
                scope=scope,
            )
            _check_own_roles(
                identity,
                spec_id=spec_id,
                scope=scope,
                shapes=shapes,
                deps=deps,
                selection=resolved_stat_var_dcids,
            )
            if scope is not None and scope.entities is not None:
                for shape in shapes:
                    build_identity_document(shape, scope=scope, graph=deps.graph)

        if scope is not None:
            # Surgery replaces `entities`, `requested_window`, and `facets`, leaving `slots` and
            # `shape` intact. The shapes resolved under the scope's facets and entities are
            # those of the surgered identity, whose id a mint of this request reports. The preview reads
            # the surgered identity in the canonical form that mint registers.
            recomputed = recompute_identity(apply_scope_surgery(identity, scope=scope))
            identity, spec_id = recomputed.identity, recomputed.spec_id

        try:
            requests = tuple(
                to_materialisation_request(identity, shape=shape, code_index=deps.code_index)
                for shape in shapes
            )
        except CurrencyRoleUndeclared as exc:
            exc.pointer = _POINTER
            raise
    return preview_definition(
        requests, family_ref=f"{_STAT_VAR_PREFIX}{spec_id}", deps=deps, limit=limit
    )
