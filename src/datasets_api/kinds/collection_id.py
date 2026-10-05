"""Reads and materialises a registered collection, or registers a supplied sidecar first.

A miss requires `members`. Each sidecar member must hash to its `spec_id`, resolve to exactly
one StatVar, and reproduce its identity against the graph. All member and scope checks finish
before any registry write. A registry hit serves stored data. The kind checks supplied sidecars
for consistency. Hash mismatches raise `SpecIdMismatch`. Invalid identities raise
`MalformedSidecarIdentity` with the member id. Missing collections raise `SpecNotFound`.
Scope role failures raise `InvalidRoleRef` or `RoleIncompatible` with the affected member id.

Scope surgery applies to each member independently. It may deduplicate members and emit
`members_deduplicated`. Unchanged identities reuse the registered collection.

A resolved shape is kept under `adapter.member_resolution_key`, the inputs member resolution
reads from an identity. `scope.facets` can bind or drop the provenance that names a member's
dataset, so a surgered identity is served by a shape resolved under an equal key and resolves
afresh under any other.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

from dfo_spec_kit.identity import IdentityError

from code_index.query import CodeIndex
from datasets_api.adapter import (
    build_identity_document,
    check_reproducible,
    member_resolution_key,
    prime_head_signal,
    resolve_definition_members,
    to_materialisation_request,
    verify_sidecar_definition,
)
from datasets_api.admission import refuse_excel_over_estimate, sync_estimate, within_sync_budget
from datasets_api.codes_materialise import (
    CurrencyRoleUndeclared,
    InvalidRoleRef,
    MalformedSidecarIdentity,
    MemberNotResolvable,
    RoleIncompatible,
    SpecNotFound,
)
from datasets_api.manifest_models import Manifest, WarningModel
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import CanonicalDefinition, MintOptions, Scope
from datasets_api.operations.submit import submit_over_budget
from datasets_api.preview import preview_requests
from datasets_api.preview_models import PreviewEnvelope
from datasets_api.serve import raw_fetch_guard, serve_collection
from datasets_api.shape import StatVarShape
from datasets_api.surgery import apply_scope_surgery
from datasets_api.timings import phase
from datasets_api.warnings import DATASETS_WARNING_CATALOGUE
from materialise import MaterialisationRequest
from materialise.errors import OverCeilingError
from materialise.graph import GraphReader
from spec_registry import MemberDefinition, SpecIdMismatch, sorted_unique_member_ids
from spec_registry import mint_collection_id as compute_collection_id
from spec_registry.identity import recompute_identity
from spec_registry.reader import StoredCollection, StoredDefinition
from statvar_members import MemberEnumerator

_POINTER = "/input/collection_id"


def _resolve_member(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
) -> StatVarShape:
    """Resolves one collection member and refuses `over_ceiling` if it names multiple StatVars.

    A collection member has one registered `spec_id` and one `StatVarShape`.
    """
    shapes = resolve_definition_members(
        identity,
        spec_id=spec_id,
        graph=graph,
        enumerator=enumerator,
        code_index=code_index,
        accepts_seed=False,
    ).shapes
    if len(shapes) > 1:
        raise OverCeilingError(
            f"collection member {spec_id!r} enumerates {len(shapes)} StatVar series against "
            "the graph, which a collection cannot express as a single member",
            ceiling_kind="member_cardinality",
            count=len(shapes),
            cap=1,
        )
    return shapes[0]


def _resolve_stored_member(
    identity: Mapping[str, object],
    *,
    spec_id: str,
    graph: GraphReader,
    enumerator: MemberEnumerator,
    code_index: CodeIndex,
) -> StatVarShape:
    """Resolves a member and attaches `/input/collection_id` to `MemberNotResolvable`.

    Accepts an identity document so preview can resolve surgered, unregistered members.
    """
    try:
        return _resolve_member(
            identity, spec_id=spec_id, graph=graph, enumerator=enumerator, code_index=code_index
        )
    except MemberNotResolvable as exc:
        exc.pointer = _POINTER
        raise


def _surgered_member_shape(
    identity: Mapping[str, object],
    surgered: Mapping[str, object],
    *,
    spec_id: str,
    scope: Scope,
    member_shapes: dict[Hashable, StatVarShape],
    deps: MintDeps,
) -> StatVarShape | None:
    """Returns the shape that serves a member's `surgered` identity, resolving it when the
    scope's checks need it, else `None`.

    A shape in `member_shapes` under the surgered identity's `member_resolution_key` serves
    it. Without one, the surgered identity resolves here when the scope binds entities,
    whose roles are checked against the shape, and when surgery changes the key, so a scope
    that changes what the member resolves is checked before the surgered collection is
    registered. `spec_id` names the member as the request does, and `identity` is the
    member before surgery, checked for reproducibility against the resolved shape."""
    key = member_resolution_key(surgered, code_index=deps.code_index)
    shape = member_shapes.get(key)
    if shape is not None:
        return shape
    if scope.entities is None and key == member_resolution_key(
        identity, code_index=deps.code_index
    ):
        return None
    shape = _resolve_stored_member(
        surgered,
        spec_id=spec_id,
        graph=deps.graph,
        enumerator=deps.enumerator,
        code_index=deps.code_index,
    )
    check_reproducible(identity, spec_id=spec_id, shape=shape)
    member_shapes[key] = shape
    return shape


def _read_back_collection(collection_id: str, *, deps: MintDeps) -> StoredCollection:
    stored = deps.spec_reader.read_collection(collection_id)
    assert stored is not None, "collection missing immediately after its own registration"
    return stored


def _verify_members_match_collection_id(
    members: Sequence[CanonicalDefinition], *, collection_id: str
) -> dict[str, Mapping[str, object]]:
    """Verifies member hashes and the collection id, and returns each member's canonical
    identity by its `spec_id`.

    The canonical document is the one a registration stores, so the callers resolve, check
    and materialise a sidecar member as its registered definition.

    Raises `SpecIdMismatch` for a mismatch, or `MalformedSidecarIdentity` with the invalid
    member's id when its identity fails the identity-core schema.
    """
    identities: dict[str, Mapping[str, object]] = {}
    for member in members:
        try:
            identities[member.spec_id] = verify_sidecar_definition(member, spec_id=member.spec_id)
        except IdentityError as exc:
            raise MalformedSidecarIdentity(
                detail=str(exc), pointer=_POINTER, member_spec_id=member.spec_id
            ) from exc
    expected_collection_id = compute_collection_id([member.spec_id for member in members])
    if expected_collection_id != collection_id:
        raise SpecIdMismatch(supplied=collection_id, computed=expected_collection_id)
    return identities


def mint_collection_id(
    collection_id: str,
    *,
    members: Sequence[CanonicalDefinition] | None,
    scope: Scope | None,
    options: MintOptions,
    deps: MintDeps,
    trace_id: str | None = None,
) -> Manifest:
    with phase("resolve"):
        stored = deps.spec_reader.read_collection(collection_id)
        # Each resolved shape, by the inputs member resolution reads from an identity.
        member_shapes: dict[Hashable, StatVarShape] = {}
        if stored is None:
            if not members:
                raise SpecNotFound(subject_id=collection_id, pointer="/input/collection_id")
            identities = _verify_members_match_collection_id(members, collection_id=collection_id)
            # Resolve, validate scope roles, and check reproducibility before any registry write.
            for member in members:
                identity = identities[member.spec_id]
                # Match the registry-hit path's error pointer.
                try:
                    shape = _resolve_member(
                        identity,
                        spec_id=member.spec_id,
                        graph=deps.graph,
                        enumerator=deps.enumerator,
                        code_index=deps.code_index,
                    )
                except MemberNotResolvable as exc:
                    exc.pointer = _POINTER
                    raise
                if scope is not None and scope.entities is not None:
                    try:
                        build_identity_document(shape, scope=scope, graph=deps.graph)
                    except (InvalidRoleRef, RoleIncompatible) as exc:
                        exc.member_spec_id = member.spec_id
                        raise
                check_reproducible(identity, spec_id=member.spec_id, shape=shape)
                member_shapes[member_resolution_key(identity, code_index=deps.code_index)] = shape
            deps.registry.register_collection(
                members=[
                    MemberDefinition(
                        identity_definition=member.identity,
                        schema_version=member.schema_version,
                        supplied_spec_id=member.spec_id,
                    )
                    for member in members
                ],
                trace_id=trace_id,
            )
            stored = _read_back_collection(collection_id, deps=deps)
        elif members is not None:
            _verify_members_match_collection_id(members, collection_id=collection_id)

        stored_members = deps.spec_reader.read_definitions(stored.member_spec_ids)

        extra_warnings: tuple[WarningModel, ...] = ()
        if scope is not None:
            surgered: list[tuple[str, dict[str, object], int]] = []
            for spec_id in stored.member_spec_ids:
                member_stored = stored_members[spec_id]
                surgered_document = apply_scope_surgery(member_stored.identity, scope=scope)
                shape = _surgered_member_shape(
                    member_stored.identity,
                    surgered_document,
                    spec_id=spec_id,
                    scope=scope,
                    member_shapes=member_shapes,
                    deps=deps,
                )
                if shape is not None and scope.entities is not None:
                    try:
                        build_identity_document(shape, scope=scope, graph=deps.graph)
                    except (InvalidRoleRef, RoleIncompatible) as exc:
                        exc.member_spec_id = spec_id
                        raise
                recomputed = recompute_identity(surgered_document)
                surgered.append(
                    (recomputed.spec_id, surgered_document, member_stored.schema_version)
                )

            new_collection_id = compute_collection_id([spec_id for spec_id, _doc, _sv in surgered])
            if new_collection_id != stored.collection_id:
                distinct_ids = sorted_unique_member_ids(
                    [spec_id for spec_id, _doc, _sv in surgered]
                )
                if len(distinct_ids) < len(surgered):
                    _title, detail, severity = DATASETS_WARNING_CATALOGUE["members_deduplicated"]
                    extra_warnings = (
                        WarningModel(
                            code="members_deduplicated", severity=severity, message=detail
                        ),
                    )
                deps.registry.register_collection(
                    members=[
                        MemberDefinition(
                            identity_definition=document,
                            schema_version=schema_version,
                            supplied_spec_id=spec_id,
                        )
                        for spec_id, document, schema_version in surgered
                    ],
                    trace_id=trace_id,
                )
                stored = _read_back_collection(new_collection_id, deps=deps)
                stored_members = deps.spec_reader.read_definitions(stored.member_spec_ids)

        # Keep this loop serial. Surgery can make a member depend on an earlier result.
        requests: list[tuple[StoredDefinition, MaterialisationRequest]] = []
        for spec_id in stored.member_spec_ids:
            member_stored = stored_members[spec_id]
            shape = member_shapes.get(
                member_resolution_key(member_stored.identity, code_index=deps.code_index)
            )
            if shape is None:
                # No shape resolved in this request serves this identity, so resolve it and
                # check reproducibility here.
                shape = _resolve_stored_member(
                    member_stored.identity,
                    spec_id=spec_id,
                    graph=deps.graph,
                    enumerator=deps.enumerator,
                    code_index=deps.code_index,
                )
                check_reproducible(member_stored.identity, spec_id=spec_id, shape=shape)
            try:
                request = to_materialisation_request(
                    member_stored.identity, shape=shape, code_index=deps.code_index
                )
            except CurrencyRoleUndeclared as exc:
                exc.pointer = _POINTER
                exc.member_spec_id = spec_id
                raise
            requests.append((member_stored, request))

    bare_requests = [request for _member_stored, request in requests]
    # Keep the estimate and fetch on the same coverage index and stamp.
    coverage_snapshot = deps.coverage.snapshot()
    # Refuse oversized Excel requests before fetching. Submit over-budget work as an operation.
    estimate = sync_estimate(
        bare_requests, index=coverage_snapshot.index, member_concurrency=deps.member_concurrency
    )
    refuse_excel_over_estimate(estimate, format=options.format)
    submit_over_budget(
        subject_id=stored.collection_id,
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
    # One signal per member, in `member_spec_ids` order.
    priming = prime_head_signal(
        bare_requests,
        subject_id=stored.collection_id,
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
        stored=stored,
        requests=requests,
        options=options,
        deps=deps.to_serve_deps(),
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


def preview_collection_id(
    collection_id: str,
    *,
    members: Sequence[CanonicalDefinition] | None,
    scope: Scope | None,
    limit: int,
    deps: MintDeps,
) -> PreviewEnvelope:
    """Write-free counterpart to `mint_collection_id` with the same resolution and validation.

    It materialises the canonical form of sidecar or surgered identities, the documents a mint
    registers and reads. Converged members share one
    recomputed `spec_id`. Previews do not emit `members_deduplicated` warnings.
    """
    with phase("resolve"):
        stored = deps.spec_reader.read_collection(collection_id)
        # Each resolved shape, by the inputs member resolution reads from an identity.
        member_shapes: dict[Hashable, StatVarShape] = {}
        member_identities: dict[str, Mapping[str, object]]
        if stored is None:
            if not members:
                raise SpecNotFound(subject_id=collection_id, pointer="/input/collection_id")
            member_identities = _verify_members_match_collection_id(
                members, collection_id=collection_id
            )
            for spec_id, identity in member_identities.items():
                try:
                    shape = _resolve_member(
                        identity,
                        spec_id=spec_id,
                        graph=deps.graph,
                        enumerator=deps.enumerator,
                        code_index=deps.code_index,
                    )
                except MemberNotResolvable as exc:
                    exc.pointer = _POINTER
                    raise
                if scope is not None and scope.entities is not None:
                    try:
                        build_identity_document(shape, scope=scope, graph=deps.graph)
                    except (InvalidRoleRef, RoleIncompatible) as exc:
                        exc.member_spec_id = spec_id
                        raise
                check_reproducible(identity, spec_id=spec_id, shape=shape)
                member_shapes[member_resolution_key(identity, code_index=deps.code_index)] = shape
        else:
            if members is not None:
                _verify_members_match_collection_id(members, collection_id=collection_id)
            stored_members = deps.spec_reader.read_definitions(stored.member_spec_ids)
            member_identities = {
                spec_id: stored_members[spec_id].identity for spec_id in stored.member_spec_ids
            }

        member_spec_ids = tuple(member_identities)

        if scope is not None:
            surgered_identities: dict[str, Mapping[str, object]] = {}
            for spec_id in member_spec_ids:
                identity = member_identities[spec_id]
                surgered_document = apply_scope_surgery(identity, scope=scope)
                shape = _surgered_member_shape(
                    identity,
                    surgered_document,
                    spec_id=spec_id,
                    scope=scope,
                    member_shapes=member_shapes,
                    deps=deps,
                )
                if shape is not None and scope.entities is not None:
                    try:
                        build_identity_document(shape, scope=scope, graph=deps.graph)
                    except (InvalidRoleRef, RoleIncompatible) as exc:
                        exc.member_spec_id = spec_id
                        raise
                # The canonical form is the document a mint registers for the surgered member.
                recomputed = recompute_identity(surgered_document)
                surgered_identities[recomputed.spec_id] = recomputed.identity
            member_identities = surgered_identities
            member_spec_ids = tuple(member_identities)

        # Resolve serially each member no shape resolved in this request serves.
        requests: list[MaterialisationRequest] = []
        for spec_id in member_spec_ids:
            identity = member_identities[spec_id]
            shape = member_shapes.get(member_resolution_key(identity, code_index=deps.code_index))
            if shape is None:
                shape = _resolve_stored_member(
                    identity,
                    spec_id=spec_id,
                    graph=deps.graph,
                    enumerator=deps.enumerator,
                    code_index=deps.code_index,
                )
                check_reproducible(identity, spec_id=spec_id, shape=shape)
            try:
                requests.append(
                    to_materialisation_request(identity, shape=shape, code_index=deps.code_index)
                )
            except CurrencyRoleUndeclared as exc:
                exc.pointer = _POINTER
                exc.member_spec_id = spec_id
                raise
    return preview_requests(requests, deps=deps, limit=limit)
