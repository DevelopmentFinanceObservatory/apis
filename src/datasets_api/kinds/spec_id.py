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

`resolve_served_definition` is the one place a definition becomes its served definition, and
every door that reads or writes subject state calls it. The mint serves the served
definition's current version through `current.serve_current`: a current head holding the
requested rendering answers with no SDMX call.
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
    resolve_definition_members,
    to_materialisation_request,
    verify_sidecar_definition,
)
from datasets_api.admission import (
    elected_provenances,
    identity_role_keys,
    refuse_excel_over_estimate,
    sync_estimate,
    within_sync_budget,
    xlsx_splits_totals,
)
from datasets_api.codes_materialise import (
    CurrencyRoleUndeclared,
    DefinitionNotReproducible,
    MalformedSidecarIdentity,
    MemberNotResolvable,
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import serve_current, serve_current_head
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintOptions, Scope
from datasets_api.models import CanonicalDefinition
from datasets_api.operations.submit import submit_over_budget
from datasets_api.render import linked_under
from datasets_api.shape import StatVarShape
from datasets_api.surgery import apply_scope_surgery, open_roles_left_open
from datasets_api.timings import phase
from materialise import MaterialisationRequest
from materialise.graph import GraphReader
from spec_registry import RecomputedIdentity, SpecIdMismatch, SpecRegistry, recompute_identity
from spec_registry.reader import SpecReader, StoredDefinition
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


class SidecarRegistrationDeps(MemberResolutionDeps, Protocol):
    """`MemberResolutionDeps` plus the registry a sidecar registration writes and the reader
    that reads it back."""

    registry: SpecRegistry
    spec_reader: SpecReader


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
        exc.issue_pointer = _POINTER
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
        exc.issue_pointer = _POINTER
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


def register_sidecar(
    definition: CanonicalDefinition,
    *,
    spec_id: str,
    scope: Scope | None,
    deps: SidecarRegistrationDeps,
    trace_id: str | None,
) -> tuple[StoredDefinition, tuple[StatVarShape, ...]]:
    """Admit the sidecar `definition` for the unregistered `spec_id` (`admit_sidecar`), register
    it, and read it back. Returns the stored definition and the shapes admission resolved.

    Admission runs before the registry write, so its refusals leave no registry row. The
    registry write is read back before the caller serves anything.
    """
    _recomputed, shapes = admit_sidecar(definition, spec_id=spec_id, scope=scope, deps=deps)
    deps.registry.register_definition(
        identity_definition=definition.identity,
        schema_version=definition.schema_version,
        supplied_spec_id=spec_id,
        trace_id=trace_id,
    )
    stored = deps.spec_reader.read_definition(spec_id)
    assert stored is not None, "definition missing immediately after its own registration"
    return stored, shapes


def _served_identity(
    identity: Mapping[str, object], *, scope: Scope | None, code_index: CodeIndex
) -> RecomputedIdentity | None:
    """The identity a mint of `identity` under `scope` serves, recomputed: surgered by `scope`
    (`apply_scope_surgery`), then in canonical form (`code_index.members.canonical_identity`).
    `None` when that is `identity` itself."""
    served = identity if scope is None else apply_scope_surgery(identity, scope=scope)
    served = canonical_identity(code_index, served)
    return None if served is identity else recompute_identity(served)


def resolve_served_definition(
    stored: StoredDefinition,
    *,
    code_index: CodeIndex,
    registry: SpecRegistry,
    scope: Scope | None = None,
    trace_id: str | None = None,
) -> StoredDefinition:
    """The served definition of the registered definition `stored`: surgered by `scope` when a
    mint supplies one, then in canonical form (`_served_identity`).

    One registry transaction reads the served definition, registers it when absent and, with
    no `scope`, binds its id to `stored`'s (`SpecRegistry.register_served_definition`).
    `stored`'s own id stays registered and resolves to the served one through this function,
    so every door keys heads, versions, operations and references on the served `spec_id`.
    Canonical form is a function of the code index, so the served id of one posted id can
    change with an index export. Each call resolves the served id under the index it is
    given, and the bindings keep every served id a posted id has resolved to, which is what a
    version-addressed read under the posted id reads (`subject_ids.served_subject_ids`). A
    scoped served id stays unbound: a mint with a scope hands out links naming the served id.
    Raises `IdentityQuarantined` when the served id, `stored`'s id or a served id already
    bound to it is quarantined, and `IdentityCollision` when the registry holds other
    identity bytes under the served id. A `stored` that is its own served definition is
    returned with no registry call: the read that produced it (`SpecReader.read_definition`)
    refuses a quarantined id and a quarantined served id bound to it.
    """
    served = _served_identity(stored.identity, scope=scope, code_index=code_index)
    if served is None or served.spec_id == stored.spec_id:
        return stored
    return registry.register_served_definition(
        served=served,
        schema_version=stored.schema_version,
        posted_spec_id=stored.spec_id if scope is None else None,
        trace_id=trace_id,
    )


def mint_spec_id(
    spec_id: str,
    *,
    definition: CanonicalDefinition | None,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
    trace_id: str | None = None,
) -> Manifest:
    """Mint the definition `spec_id` names, registering a supplied sidecar first, and serve its
    current version (`current.serve_current`).

    Members resolve before any write, under the facets and entities `scope` installs. The
    served definition (`resolve_served_definition`) keys every state write, and `serve_mint`
    serves it. With no `scope` the served id is the canonical form of `spec_id`, and the mint
    names `spec_id`: a head honours revocations under both ids, and the links name `spec_id`.
    With a `scope` the links name the served id.
    """
    with phase("resolve"):
        stored = deps.spec_reader.read_definition(spec_id)
        if stored is None:
            if definition is None:
                raise SpecNotFound(subject_id=spec_id, pointer="/input/spec_id")
            stored, shapes = register_sidecar(
                definition, spec_id=spec_id, scope=scope, deps=deps, trace_id=trace_id
            )
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

        stored = resolve_served_definition(
            stored,
            code_index=deps.code_index,
            registry=deps.registry,
            scope=scope,
            trace_id=trace_id,
        )
        try:
            requests = tuple(
                to_materialisation_request(stored.identity, shape=shape, code_index=deps.code_index)
                for shape in shapes
            )
        except CurrencyRoleUndeclared as exc:
            exc.pointer = _POINTER
            raise
    if scope is not None:
        # Scope surgery makes a served id no click on `spec_id` resolves to, so the links name
        # the served id.
        return serve_mint(stored, requests=requests, options=options, deps=deps, trace_id=trace_id)
    # The served id is the canonical form of `spec_id`. The mint names `spec_id`, so it honours
    # revocations under both ids and its links name `spec_id` (`render.linked_under`).
    manifest = serve_mint(
        stored,
        requests=requests,
        options=options,
        deps=deps,
        trace_id=trace_id,
        revoked_under=(spec_id,),
    )
    return linked_under(manifest, spec_id)


def serve_mint(
    stored: StoredDefinition,
    *,
    requests: Sequence[MaterialisationRequest],
    options: MintOptions,
    deps: MintDeps,
    trace_id: str | None,
    revoked_under: Sequence[str] = (),
) -> Manifest:
    """Serve a mint of the served definition `stored` over its resolved `requests`. A head
    answers under `current.load_stored`'s `revoked_under`.

    A current head that links the requested rendering answers first
    (`current.serve_current_head`). Otherwise admission runs once here. It refuses an Excel
    request over the row cap, submits work over the sync budget as an operation, and serves
    the current version (`current.serve_current`) under the admitted estimate, starting from
    the head and verdict `serve_current_head` reached, so the head is read once.

    One coverage snapshot serves the head check, the estimate, the budget decision and the
    fetch.
    """
    definition_deps = deps.to_definition_deps()

    with phase("coverage"):
        coverage_snapshot = deps.coverage.snapshot()
    # Keep the estimate and fetch on the same coverage index and stamp.
    served = serve_current_head(
        stored,
        requests=requests,
        options=options,
        trigger="mint",
        deps=definition_deps,
        trace_id=trace_id,
        coverage_snapshot=coverage_snapshot,
        revoked_under=revoked_under,
    )
    if isinstance(served, Manifest):
        return served
    with phase("estimate"):
        estimate = sync_estimate(
            requests, index=coverage_snapshot.index, member_concurrency=deps.member_concurrency
        )
        refuse_excel_over_estimate(
            estimate,
            format=options.format,
            splits_totals=xlsx_splits_totals(
                options,
                identity=stored.identity,
                role_columns=identity_role_keys(stored.identity),
                parties=definition_deps.code_index.parties,
                datasets=elected_provenances(requests),
            ),
        )
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
    return serve_current(
        stored,
        resolve_requests=lambda: requests,
        options=options,
        trigger="mint",
        deps=definition_deps,
        trace_id=trace_id,
        admitted=estimate,
        coverage_snapshot=coverage_snapshot,
        head_read=served,
        revoked_under=revoked_under,
    )
