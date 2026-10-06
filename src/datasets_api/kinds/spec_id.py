"""Reads and materialises a registered definition, or registers a supplied sidecar first.

A miss requires a sidecar. Member resolution may refuse `member_not_resolvable`,
`spec_not_exportable`, `over_ceiling`, or `role_signature_incompatible`. Scope roles and
reproducibility are checked before the sidecar hash and registry write, so refusals leave no
row. Hash mismatches raise `SpecIdMismatch`. A registry hit serves stored data. The kind checks
supplied sidecars for hash consistency. Missing definitions raise `SpecNotFound`.

Members are a function of the identity, the graph and the code index. No request field adds or
names one. A described arm's walk reads the whole filtered reverse set, up to
`DatasetsBuildSettings.member_subject_ceiling` subjects, when its first page truncates without a
confirmed root (`adapter.member_read_bounds`). The mint serves the antichain-minimal members.
Every mint resolves through `_resolve_members` before a registry write, so a refusal leaves no
row. Scope surgery replaces supplied members, preserves the rest, and rehashes the document.
`scope.facets` can bind the provenance that names a family's dataset, so members resolve under
the facets the scope installs. Invalid roles refuse as `InvalidRoleRef` or `RoleIncompatible`
before a write. The mint then serves the
identity in canonical form (`code_index.members.canonical_identity`): each slot an invariant
of the dataset the identity names fixes binds the invariant's value, so definitions naming one
dataset that differ only in whether they bind an invariant or leave it unbound serve and
report one `spec_id`. An identity that names no dataset keeps the bindings it was posted with. An unchanged identity reuses its record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from dfo_spec_kit.identity import IdentityError

from code_index.members import canonical_identity
from code_index.query import CodeIndex
from datasets_api.adapter import (
    MemberReadBounds,
    build_identity_document,
    check_reproducible,
    member_read_bounds,
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
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.models import CanonicalDefinition
from datasets_api.operations.submit import submit_over_budget
from datasets_api.serve import raw_fetch_guard, serve_definition
from datasets_api.shape import StatVarShape
from datasets_api.surgery import apply_scope_surgery, open_roles_left_open
from datasets_api.timings import phase
from materialise.graph import GraphReader
from spec_registry import RecomputedIdentity, SpecIdMismatch, recompute_identity
from statvar_members import MemberEnumerator

_POINTER = "/input/spec_id"
_STAT_VAR_PREFIX = "dcid:"


class MemberResolutionDeps(Protocol):
    """The graph reads sidecar admission makes, the settings that bound its member walk, and
    the request's deadline. `MintDeps` and the definition door's `DefinitionDeps` both carry
    them."""

    graph: GraphReader
    enumerator: MemberEnumerator
    code_index: CodeIndex
    settings: DatasetsBuildSettings
    deadline: float | None


def _walk_bounds(deps: MemberResolutionDeps) -> MemberReadBounds:
    """The bounds of a request's member walks, counted from now."""
    return member_read_bounds(deps.settings, request_deadline=deps.deadline)


def _resolve_members(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
    bounds: MemberReadBounds,
    scope: Scope | None = None,
) -> tuple[StatVarShape, ...]:
    """Resolves an identity and attaches `/input/spec_id` to `MemberNotResolvable`.

    Accepts a registered identity or an unregistered sidecar. A `scope` that supplies `facets`
    replaces the identity's own before resolution, as scope surgery does, and one that supplies
    `entities` supplies the role keys the role class is elected under, and the open roles
    the scope leaves open (`open_roles_left_open`), so the family is the one the surgered
    identity resolves. `bounds` bounds the graph walk
    (`adapter.resolve_definition_members`). Returns the antichain-minimal shapes.
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
            bounds=bounds,
            role_keys=(
                frozenset(scope.entities)
                if scope is not None and scope.entities is not None
                else None
            ),
            open_role_keys=(
                open_roles_left_open(identity, scope.entities)
                if scope is not None and scope.entities is not None
                else None
            ),
        )
    except MemberNotResolvable as exc:
        exc.pointer = _POINTER
        raise
    return resolved.shapes


def _check_own_roles(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    scope: Scope | None,
    shapes: Sequence[StatVarShape],
    deps: MemberResolutionDeps,
    bounds: MemberReadBounds,
) -> None:
    """Checks `identity`'s own bound and open roles against the shape `identity` resolves to
    (`check_reproducible`).

    `shapes` are the request's, resolved under `scope`. A scope that elects under other bound
    or open role keys (`_resolve_members`) can elect another role class, so when the
    identity's roles fail against `shapes` under such a scope, the identity resolves again
    under its own role keys and the scope's facets, and is checked against that shape. An
    identity whose roles hold against `shapes` resolves once. The second resolution runs
    under the same `bounds`."""
    try:
        check_reproducible(identity, spec_id=spec_id, shape=shapes[0])
    except DefinitionNotReproducible:
        if scope is None or scope.entities is None:
            raise
        own_keys = (
            frozenset(cast("Mapping[str, object]", identity.get("entities", {}))),
            frozenset(cast("Sequence[str]", identity.get("open_roles", ()))),
        )
        scoped_keys = (
            frozenset(scope.entities),
            frozenset(open_roles_left_open(identity, scope.entities)),
        )
        if scoped_keys == own_keys:
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
            bounds=bounds,
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
) -> tuple[RecomputedIdentity, tuple[StatVarShape, ...]]:
    """Checks an unregistered sidecar and returns its recomputed identity (the canonical
    document, its bytes and digest) and resolved shapes.

    `recompute_identity` canonicalises the posted identity once. Member resolution, the scope
    role check and the reproducibility check read that canonical document, the one a
    registration stores, so a mint of one sidecar resolves and materialises the same identity
    whichever spelling the caller posted. Its member walks share one set of bounds counted from
    this call (`_walk_bounds`). Nothing is written.

    Raises `MalformedSidecarIdentity` for an identity the identity core refuses, each member
    resolution refusal, `InvalidRoleRef` or `RoleIncompatible` for a scope role, `SpecIdMismatch`
    when the identity does not hash to `spec_id`, and `DefinitionNotReproducible`, in that order.
    """
    bounds = _walk_bounds(deps)
    try:
        recomputed = recompute_identity(definition.identity)
        identity = recomputed.identity
        shapes = _resolve_members(
            identity,
            spec_id=spec_id,
            graph=deps.graph,
            enumerator=deps.enumerator,
            code_index=deps.code_index,
            bounds=bounds,
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
        identity, spec_id=spec_id, scope=scope, shapes=shapes, deps=deps, bounds=bounds
    )
    return recomputed, shapes


def _served_identity(
    identity: Mapping[str, object], *, scope: Scope | None, code_index: CodeIndex
) -> RecomputedIdentity | None:
    """The identity a mint of `identity` under `scope` serves, recomputed: surgered by `scope`
    (`apply_scope_surgery`), then in canonical form (`code_index.members.canonical_identity`).
    `None` when that is `identity` itself."""
    served = identity if scope is None else apply_scope_surgery(identity, scope=scope)
    served = canonical_identity(code_index, served)
    return None if served is identity else recompute_identity(served)


def mint_spec_id(
    spec_id: str,
    *,
    definition: CanonicalDefinition | None,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
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
            _recomputed, shapes = admit_sidecar(definition, spec_id=spec_id, scope=scope, deps=deps)
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
            bounds = _walk_bounds(deps)
            shapes = _resolve_members(
                stored.identity,
                spec_id=stored.spec_id,
                graph=deps.graph,
                enumerator=deps.enumerator,
                code_index=deps.code_index,
                bounds=bounds,
                scope=scope,
            )
            _check_own_roles(
                stored.identity,
                spec_id=stored.spec_id,
                scope=scope,
                shapes=shapes,
                deps=deps,
                bounds=bounds,
            )
            if scope is not None and scope.entities is not None:
                for shape in shapes:
                    build_identity_document(shape, scope=scope, graph=deps.graph)

        served = _served_identity(stored.identity, scope=scope, code_index=deps.code_index)
        if served is not None and served.spec_id != stored.spec_id:
            deps.registry.register_definition(
                identity_definition=served.identity,
                schema_version=stored.schema_version,
                supplied_spec_id=served.spec_id,
                trace_id=trace_id,
            )
            # No manifest returns before the registry write is read back.
            stored = deps.spec_reader.read_definition(served.spec_id)
            assert stored is not None, (
                "served definition missing immediately after its own registration"
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
