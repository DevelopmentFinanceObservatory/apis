"""Retrieve stored definitions with `GET /api/datasets/{id}` or its `/options` route.

The route accepts `spec_` ids. It reads the definition from the registry, resolves its served
definition (`kinds.spec_id.resolve_served_definition`), and serves that definition's current
version through `current.serve_current`, so every head, version and operation is keyed on the
served id. Grammar-valid ids absent from the registry return `spec_not_found`. Malformed ids
reach the surface's 404 refusal route. Rendering options use dotted query keys, and unsupported
options are refused.

`/options` is mounted before the snapshot version route, so the literal `options` suffix reaches
this handler. It rejects every query key with `request_invalid` and an `/options/<key>` pointer.
It is the GET with no rendering: a rebuild records the version and advances the head, and the
answer is the `OptionsEnvelope` of its manifest. Resolve refusals and `over_ceiling` match
retrieval behaviour. `POST /options` carries the spec's canonical definition
(`DefinitionOptionsRequest`), so it also answers for a spec no mint or register call has stored.
That definition is admitted and registered as a mint registers a sidecar, then served as a
registered id is, so the rebuild records the version and advances the head the next mint reads.

`POST /api/datasets/{id}/members` takes a `spec_` id and resolves the members a mint of it serves
without materialising a frame (`MembersEnvelope`): each member's ref, label, roles and
constraints, their count, and the family's count and digest. Its optional body carries the
canonical definition an unregistered id needs, admitted as a mint admits a sidecar, and nothing
is written. It refuses as member resolution does.

Retrieval delegates shared materialisation, registry, and snapshot errors to their classifiers.
This door adds registry-miss, backpressure, and stored-data errors. A refusal a rebuild's
admission or materialisation raises, whose mint classification points into a POST body, is
marked `strip_pointer` (`current.without_pointers`). Conversion refusals carry
no request pointer because their binding comes from the stored identity.

Rehydration checks stored identities against the current graph. It can refuse unresolvable members,
non-exportable definitions, role mismatches, over-ceiling enumerations, and non-reproducible
identities. `MemberNotResolvable` is wrapped so retrieval issues omit the mint body's
`/input/statvars` pointer. Other member errors abort retrieval immediately.

`GET` and `/options` first run on the surface's read executor, which answers from a head
confirmed inside the recheck window that holds the requested rendering (`serve_fresh_head`).
Any other call moves to the work executor, where the freshness check compares the head's
dependency fingerprint and rebuilds or serves the head, rendering a missing format from the
stored frame. `/members` runs on the read executor and reports the served id. Admitted
worker calls finish after a client timeout. The door reads its clients and stores from the
surface's `Deps` (`datasets_api.deps`) and builds none.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from itertools import chain
from typing import Any

from dfo_frame_grammar import FrameError
from dfo_spec_kit.identity import IdentityError
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from artifact_store import ArtifactInventoryError, ArtifactRecordConflictError, ArtifactWriteError
from dataset_store.snapshot import (
    SnapshotRevoked,
)

# registers the `dfo_spec_id` convertor
from datasets_api import routing  # noqa: F401
from datasets_api.adapter import (
    admitted_family,
    member_read_bounds,
    rehydrate_definition,
    to_materialisation_request,
    verify_sidecar_definition,
)
from datasets_api.backpressure import Backpressure
from datasets_api.codes_materialise import (
    CapabilityUnavailable,
    CurrencyRoleUndeclared,
    DefinitionNotReproducible,
    ExcludedByRegistry,
    MalformedSidecarIdentity,
    MemberNotResolvable,
    RoleSignatureIncompatible,
    SpecNotExportable,
    UnSystemUnavailable,
)
from datasets_api.current import (
    DefinitionDeps,
    HeadLoader,
    HeadRead,
    StoredTraceMissingResolvedStatVars,
    Trigger,
    serve_current,
    serve_fresh_head,
    without_pointers,
)
from datasets_api.deps import MATERIALISE_RAW_FETCH_GUARD_ROWS, Deps, DepsCell
from datasets_api.door_executor import DoorExecutor
from datasets_api.doors.mint_errors import MintBackpressure, classify_mint_exception
from datasets_api.doors.mint_inventory import MINT_EXCEPTION_EXEMPLARS
from datasets_api.doors.register import classify_register_exception
from datasets_api.headers import no_store_headers
from datasets_api.kinds.spec_id import (
    admit_sidecar,
    register_sidecar,
    resolve_served_definition,
)
from datasets_api.manifest_models import Manifest, OptionsEnvelope
from datasets_api.mint_models import (
    DefinitionMembersRequest,
    DefinitionOptionsRequest,
    MintOptions,
)
from datasets_api.models import CanonicalDefinition
from datasets_api.operation_store import OperationBacklogFull
from datasets_api.operations.submit import (
    OperationSubmitted,
    operation_submitted_resource,
    retry_after_headers,
)
from datasets_api.query_options import (
    QueryOptionsInvalid,
    parse_query_options,
    pointer_for_key,
)
from datasets_api.render import check_options_supported, linked_under
from datasets_api.retention import retaining
from datasets_api.schema import MembersEnvelope, OperationResource, ServedMember
from datasets_api.shape import StatVarShape
from datasets_api.store_errors import DatasetsStoreUnavailable
from datasets_api.timings import PhaseTimings, door_call, phase, run_with_timings
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dc_sdmx import (
    SdmxCapExceededError,
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxQueryError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxTransportError,
    SdmxUnsupportedFilterError,
)
from dfo_platform import ProblemIssue, ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.problem import Refusal
from dfo_platform.request_deadline import request_deadline
from dfo_platform.trace import get_trace_id
from materialise import (
    MaterialisationRequest,
)
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DatasetDisagreementError,
    DefaultFacetUndefinedError,
    FacetBindingUnsupportedError,
    FacetNotAvailableError,
    MembersNotHeldError,
    MixedGrainError,
    OverCeilingError,
    UpstreamResponseError,
    UpstreamVersionChangedError,
    VariableBreadthUnreadError,
)
from materialise.formats import ExcelIllegalCharacterError, ExcelLimitError, ParquetSizeBoundError
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from materialise.rate_tables import RateTableInvalidError
from materialise.recipes import NonCurrencyUnitError, RateCoverageGapError, RecipeNotDeclaredError
from materialise.reshape import (
    ColumnNotDroppableError,
    ColumnUnknownError,
    PivotNotHomogeneousError,
    PivotNotUniqueError,
    PivotOverCellCapError,
    PivotOverColumnCapError,
    RollupNotSummableError,
)
from spec_registry import SpecIdMismatch, SpecRegistryUnavailable
from spec_registry.errors import IdentityCollision, IdentityQuarantined
from spec_registry.reader import StoredDefinition
from statvar_members import MemberReadDeadline, shared_enumerator

logger = logging.getLogger(__name__)

# `DefinitionOptionsRequest` body paths the options route's refusals point at.
_DEFINITION_POINTER = "/definition"
_STAT_VAR_PREFIX = "dcid:"


def _unwind_seconds(timeout: float) -> float:
    """Up to one second, or 10% of the timeout, left after the work deadline for `wait_for` to
    unwind."""
    return min(1.0, timeout * 0.1)


def _sdmx_deadline(timeout: float) -> float:
    """The work deadline of a request answered within `timeout` from now."""
    return time.monotonic() + max(0.0, timeout - _unwind_seconds(timeout))


class SpecNotFound(Refusal):
    """A valid subject id is absent from the registry. It maps to `spec_not_found`, 404."""

    code = "spec_not_found"

    def __init__(self, *, id: str) -> None:
        self.id = id
        super().__init__(f"no definition is registered at id {id!r}")


class DefinitionBackpressure(Backpressure):
    """The retrieval semaphore is full. It maps to retryable `service_unavailable`, 503."""

    code = "service_unavailable"

    detail = (
        "the definition retrieval door is at its concurrent-materialisation capacity right "
        "now; retry"
    )


class DefinitionMemberNotResolvable(Refusal):
    """Unresolvable members on retrieval.

    It maps to `member_not_resolvable`, 422, without a pointer.

    `truncated` is true when the member list was capped.
    """

    code = "member_not_resolvable"

    @property
    def issues(self) -> tuple[ProblemIssue, ...]:
        return tuple(ProblemIssue(member=member) for member in self.members)

    def __init__(self, *, members: tuple[str, ...], truncated: bool = False) -> None:
        self.members = members
        self.truncated = truncated
        detail = f"member(s) did not resolve to exactly one registered shape: {members!r}"
        if truncated:
            detail += "; more members disagreed than are named here"
        super().__init__(detail)


def classify_definition_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify retrieval refusals, and the registry, materialisation, and snapshot errors a
    retrieval can raise."""
    return classify_register_exception(exc) or classify_mint_exception(exc)


def _served_or_none(
    *, spec_id: str, deps: DefinitionDeps, trace_id: str | None = None
) -> StoredDefinition | None:
    """The served definition of the registered id `spec_id` (`resolve_served_definition`),
    or `None` for an unregistered id."""
    stored = deps.spec_reader.read_definition(spec_id)
    if stored is None:
        return None
    return resolve_served_definition(
        stored,
        code_index=deps.code_index,
        registry=deps.registry,
        trace_id=trace_id,
    )


def _served_spec(
    *, spec_id: str, deps: DefinitionDeps, trace_id: str | None = None
) -> StoredDefinition:
    """`_served_or_none`, raising `SpecNotFound` for an unregistered id."""
    stored = _served_or_none(spec_id=spec_id, deps=deps, trace_id=trace_id)
    if stored is None:
        raise SpecNotFound(id=spec_id)
    return stored


def _resolve_spec_requests(
    *, spec_id: str, deps: DefinitionDeps
) -> tuple[StoredDefinition, Sequence[MaterialisationRequest]]:
    """Read a registered spec's served definition and rehydrate it.

    Wrap `MemberNotResolvable` so the GET route uses retrieval's pointer-free refusal.
    """
    with phase("resolve"):
        stored = _served_spec(spec_id=spec_id, deps=deps)
        return stored, _rehydrate_stored(stored, deps=deps)


def _rehydrate_stored(
    stored: StoredDefinition, *, deps: DefinitionDeps
) -> Sequence[MaterialisationRequest]:
    """Rehydrate a registered spec, its member walk bounded by `member_read_bounds`."""
    try:
        _shapes, requests = rehydrate_definition(
            stored,
            graph=deps.graph,
            enumerator=deps.enumerator,
            code_index=deps.code_index,
            bounds=member_read_bounds(deps.settings, request_deadline=deps.deadline),
        )
    except MemberNotResolvable as exc:
        raise DefinitionMemberNotResolvable(members=exc.members, truncated=exc.truncated) from exc
    return requests


def _verify_posted_definition(*, spec_id: str, definition: CanonicalDefinition) -> None:
    """Refuse a posted definition whose identity does not hash to `spec_id` or whose
    `definition.spec_id` names another id."""
    try:
        verify_sidecar_definition(definition, spec_id=spec_id)
    except IdentityError as exc:
        raise MalformedSidecarIdentity(detail=str(exc), pointer=_DEFINITION_POINTER) from exc
    if definition.spec_id != spec_id:
        # The identity hashes to the path id, so the definition's own id is the one at fault.
        raise SpecIdMismatch(supplied=definition.spec_id, computed=spec_id)


def _register_posted_definition(
    *, spec_id: str, body: DefinitionOptionsRequest, deps: DefinitionDeps, trace_id: str | None
) -> tuple[StoredDefinition, Sequence[MaterialisationRequest]]:
    """Register the definition `POST /options` carries for the unregistered `spec_id` as a
    `spec_id` mint registers a sidecar (`kinds.spec_id.register_sidecar`). Returns its served
    definition (`resolve_served_definition`) and the requests its admitted members build.

    `_verify_posted_definition` has already checked the body against `spec_id`. Admission
    refusals leave no registry row. Member refusals keep the GET route's pointer-free form.
    """
    try:
        registered, shapes = register_sidecar(
            body.definition, spec_id=spec_id, scope=None, deps=deps, trace_id=trace_id
        )
    except MemberNotResolvable as exc:
        raise DefinitionMemberNotResolvable(members=exc.members, truncated=exc.truncated) from exc
    stored = resolve_served_definition(
        registered, code_index=deps.code_index, registry=deps.registry, trace_id=trace_id
    )
    requests = tuple(
        to_materialisation_request(stored.identity, shape=shape, code_index=deps.code_index)
        for shape in shapes
    )
    return stored, requests


def _serve_spec(
    *,
    stored: StoredDefinition,
    options: MintOptions | None,
    trigger: Trigger,
    deps: DefinitionDeps,
    trace_id: str | None,
    load_head: HeadLoader | None = None,
    head_read: HeadRead | None = None,
    revoked_under: Sequence[str] = (),
    requests: Sequence[MaterialisationRequest] | None = None,
) -> Manifest:
    """Serve the served definition `stored` through `serve_current`, rehydrating it at most
    once and only when the head cannot answer. `requests`, the members a caller has already
    admitted, replace the rehydration. `load_head`, `head_read` and `revoked_under` are
    `serve_current`'s. A refusal the rebuild's admission or materialisation raises loses its
    mint-body pointer (`strip_pointer`)."""

    @functools.cache
    def _requests() -> Sequence[MaterialisationRequest]:
        if requests is not None:
            return requests
        with phase("resolve"):
            return _rehydrate_stored(stored, deps=deps)

    return serve_current(
        stored,
        resolve_requests=_requests,
        options=options,
        trigger=trigger,
        deps=deps,
        trace_id=trace_id,
        load_head=load_head,
        head_read=head_read,
        strip_pointer=True,
        revoked_under=revoked_under,
    )


def _serve_spec_options(
    *,
    spec_id: str,
    deps: DefinitionDeps,
    trace_id: str | None,
    body: DefinitionOptionsRequest | None,
    head_read: HeadRead | None,
) -> OptionsEnvelope:
    """`_serve_spec` with no rendering: a rebuild records the version and advances the head
    as a GET does, and the envelope carries the manifest's fields alone.

    `head_read` is the head `_serve_spec_options_head` read for the served definition, and
    `None` when `_serve_spec_options_head` found `spec_id` unregistered with a posted `body`.
    That function has already verified the body against `spec_id`. The body's definition is
    then registered (`_register_posted_definition`) and served as a registered id is, from the
    members its admission resolved, so a later mint or GET of `spec_id` reads the head this
    call advances.
    """
    requests: Sequence[MaterialisationRequest] | None = None
    if head_read is not None:
        stored = head_read.stored
    else:
        with phase("resolve"):
            served = _served_or_none(spec_id=spec_id, deps=deps, trace_id=trace_id)
            if served is None:
                if body is None:
                    raise SpecNotFound(id=spec_id)
                served, requests = _register_posted_definition(
                    spec_id=spec_id, body=body, deps=deps, trace_id=trace_id
                )
        stored = served
    manifest = _serve_spec(
        stored=stored,
        options=None,
        trigger="options",
        deps=deps,
        trace_id=trace_id,
        head_read=head_read,
        revoked_under=(spec_id,),
        requests=requests,
    )
    return _to_options_envelope(manifest)


def _resolve_spec_members(
    *, spec_id: str, definition: CanonicalDefinition | None, deps: DefinitionDeps
) -> tuple[str, Mapping[str, object], tuple[StatVarShape, ...]]:
    """Resolve the identity `spec_id` names and the members a mint of it serves, writing
    nothing but a served identity not yet registered. Returns the served id, its identity, and
    its shapes.

    A registered id resolves its served definition (`resolve_served_definition`) as the GET
    route does, with its roles checked (`rehydrate_definition`). An unregistered one needs
    `definition`, admitted as a mint admits a sidecar (`admit_sidecar`), and keeps its own id.
    Without it the call refuses `spec_not_found`. Member refusals take the GET route's
    pointer-free form.
    """
    with phase("resolve"):
        stored = _served_or_none(spec_id=spec_id, deps=deps, trace_id=None)
        if stored is None and definition is None:
            raise SpecNotFound(id=spec_id)
        try:
            if stored is not None:
                shapes, _requests = rehydrate_definition(
                    stored,
                    graph=deps.graph,
                    enumerator=deps.enumerator,
                    code_index=deps.code_index,
                    bounds=member_read_bounds(deps.settings, request_deadline=deps.deadline),
                )
                return stored.spec_id, stored.identity, shapes
            assert definition is not None
            recomputed, shapes = admit_sidecar(definition, spec_id=spec_id, scope=None, deps=deps)
        except MemberNotResolvable as exc:
            raise DefinitionMemberNotResolvable(
                members=exc.members, truncated=exc.truncated
            ) from exc
    return spec_id, recomputed.identity, shapes


def _serve_spec_members(
    *, spec_id: str, deps: DefinitionDeps, body: DefinitionMembersRequest | None
) -> MembersEnvelope:
    """The members a mint of `spec_id` serves, with one batched label read, and the family
    they come from, as the mint's trace states it (`admitted_family`). A posted definition is
    verified against `spec_id` first, registered id or not. A registered id reports its served
    id."""
    definition = body.definition if body is not None else None
    if definition is not None:
        with phase("resolve"):
            _verify_posted_definition(spec_id=spec_id, definition=definition)
    served_id, identity, shapes = _resolve_spec_members(
        spec_id=spec_id, definition=definition, deps=deps
    )
    with phase("labels"):
        labels = deps.graph.node_labels_batch(
            [shape.stat_var.removeprefix(_STAT_VAR_PREFIX) for shape in shapes]
        )
    family = admitted_family(identity, shapes[0])
    return MembersEnvelope(
        spec_id=served_id,
        members=[
            ServedMember(
                ref=shape.stat_var,
                label=labels.get(shape.stat_var.removeprefix(_STAT_VAR_PREFIX)),
                roles=list(shape.declared_roles),
                constraints=dict(shape.slots),
            )
            for shape in sorted(shapes, key=lambda shape: shape.stat_var)
        ],
        served_count=len(shapes),
        family_member_count=family.count if family is not None else None,
        family_member_digest=family.digest if family is not None else None,
    )


def _to_options_envelope(manifest: Manifest) -> OptionsEnvelope:
    """Return only the manifest fields exposed by the `/options` route."""
    return OptionsEnvelope(
        schema_version=manifest.schema_version,
        definition=manifest.definition,
        content_version=manifest.content_version,
        coverage=manifest.coverage,
        output_options=manifest.output_options,
        warnings=manifest.warnings,
        trace=manifest.trace,
    )


def definition_deps(deps: Deps, *, deadline: float | None) -> DefinitionDeps:
    """The retrieval view of the surface's `Deps`, with this request's SDMX deadline. Every
    request's view shares the surface's `RebuildFlights`."""
    settings = deps.settings
    return DefinitionDeps(
        registry=deps.registry,
        spec_reader=deps.spec_reader,
        graph=deps.graph,
        sdmx=deps.sdmx,
        artifact_store=deps.artifact_store,
        materialiser_build=deps.materialiser_build,
        capability_version=DATASETS_CAPABILITY_VERSION,
        ceiling=MATERIALISE_RAW_FETCH_GUARD_ROWS,
        member_concurrency=settings.mint_member_max_concurrency,
        enumerator=shared_enumerator(),
        enrichment=deps.enrichment,
        code_index=deps.code_index,
        rate_cache=deps.rate_cache,
        coverage=deps.coverage,
        version_writer=deps.version_writer,
        head_store=deps.head_store,
        snapshot_reader=deps.snapshot_reader,
        materialise_semaphore=deps.materialise_semaphore,
        head_recheck_interval=timedelta(seconds=settings.head_recheck_seconds),
        head_max_age=timedelta(seconds=settings.head_max_age_seconds),
        now=deps.now,
        deadline=deadline,
        operation_store=deps.operation_store,
        task_client=deps.task_client,
        settings=settings,
        rebuilds=deps.rebuilds,
        version_lister=deps.version_lister,
        variable_breadth=deps.variable_breadth,
    )


def request_definition_deps(deps: Deps) -> DefinitionDeps:
    """`definition_deps` under a deadline `definition_timeout_seconds` from now."""
    return definition_deps(deps, deadline=_sdmx_deadline(deps.settings.definition_timeout_seconds))


async def _run_retrieval[T](
    executor: DoorExecutor,
    fn: Callable[..., T],
    /,
    *,
    deps: DefinitionDeps,
    timings: PhaseTimings,
    refuse: Callable[[], Exception] = DefinitionBackpressure,
    **kwargs: object,
) -> T:
    """Run `fn(deps=deps, **kwargs)` on `executor` under `deps.deadline`, bound as the request
    deadline (`dfo_platform.request_deadline`) so every graph and SDMX read inside takes it.
    The wait ends with the unwind margin after the deadline, however many hops the request
    made before, and answers timeout (504). The worker finishes after a client timeout."""
    deadline = deps.deadline
    assert deadline is not None, "a request's retrieval deps carry its deadline"
    with request_deadline(deadline):
        return await asyncio.wait_for(
            executor.run(
                functools.partial(run_with_timings, fn, timings, deps=deps, **kwargs),
                refuse=refuse,
            ),
            timeout=max(0.0, deadline - time.monotonic())
            + _unwind_seconds(deps.settings.definition_timeout_seconds),
        )


def _serve_spec_head(
    *, spec_id: str, options: MintOptions, deps: DefinitionDeps, trace_id: str | None
) -> Manifest | HeadRead:
    """The served definition's answer from a current head that holds the requested rendering,
    or the head read for `_serve_spec` (`serve_fresh_head`). The head honours revocations
    under the served id and under `spec_id`, the id the request named."""
    with phase("resolve"):
        stored = _served_spec(spec_id=spec_id, deps=deps, trace_id=trace_id)
    return serve_fresh_head(
        stored,
        deps=deps,
        options=options,
        trigger="get",
        trace_id=trace_id,
        revoked_under=(spec_id,),
    )


def _serve_spec_options_head(
    *,
    spec_id: str,
    deps: DefinitionDeps,
    trace_id: str | None,
    body: DefinitionOptionsRequest | None,
) -> OptionsEnvelope | HeadRead | None:
    """`_serve_spec_options` answered from a current head alone, or the head read. A posted
    `body` is verified against `spec_id` first, since a current head answers without
    resolving; `_serve_spec_options` relies on this check. `None` means `spec_id` is
    unregistered and a `body` came with it, so the work executor registers it."""
    with phase("resolve"):
        if body is not None:
            _verify_posted_definition(spec_id=spec_id, definition=body.definition)
        stored = _served_or_none(spec_id=spec_id, deps=deps, trace_id=trace_id)
    if stored is None:
        if body is None:
            raise SpecNotFound(id=spec_id)
        return None
    manifest = serve_fresh_head(
        stored,
        deps=deps,
        options=None,
        trigger="options",
        trace_id=trace_id,
        revoked_under=(spec_id,),
    )
    return manifest if isinstance(manifest, HeadRead) else _to_options_envelope(manifest)


async def _get_definition_impl(id: str, request: Request, *, deps: Deps) -> JSONResponse:
    """Serve the spec from a fresh head on the read executor, else on the work executor."""
    options = parse_query_options(request.query_params.multi_items())
    check_options_supported(options, code_index=deps.code_index)
    trace_id = get_trace_id()
    request_deps = request_definition_deps(deps)
    submitted_resource: OperationResource | None = None
    with door_call("definition", logger=logger) as call:
        try:
            head_answer = await _run_retrieval(
                deps.read_executor,
                retaining(_serve_spec_head, reader=deps.snapshot_reader, now=deps.now),
                deps=request_deps,
                timings=call.timings,
                spec_id=id,
                options=options,
                trace_id=trace_id,
            )
            manifest = (
                await _run_retrieval(
                    deps.work_executor,
                    retaining(_serve_spec, reader=deps.snapshot_reader, now=deps.now),
                    timings=call.timings,
                    stored=head_answer.stored,
                    options=options,
                    trigger="get",
                    deps=request_deps,
                    trace_id=trace_id,
                    head_read=head_answer,
                    revoked_under=(id,),
                )
                if isinstance(head_answer, HeadRead)
                else head_answer
            )
        except OperationSubmitted as submitted:
            # Treat a 202 submission as a successful `door_call` outcome.
            submitted_resource = operation_submitted_resource(submitted, now=deps.now())
    if submitted_resource is not None:
        assert submitted_resource.operation.poll is not None
        return JSONResponse(
            submitted_resource.model_dump(mode="json", exclude_none=True),
            status_code=202,
            headers={
                **no_store_headers(server_timing=call.server_timing),
                "Location": submitted_resource.operation.poll,
                **retry_after_headers(submitted_resource.operation),
            },
        )
    # The links name the requested id, so a click honours the revocations this request did.
    return JSONResponse(
        linked_under(manifest, id).model_dump(mode="json", exclude_none=True),
        headers=no_store_headers(server_timing=call.server_timing),
    )


def _refuse_query_keys(request: Request, *, route: str) -> None:
    """Refuse the request's first query key, since `route` accepts none."""
    query_items = request.query_params.multi_items()
    if query_items:
        key, _value = query_items[0]
        raise QueryOptionsInvalid(
            pointer=pointer_for_key(key),
            detail=(f"the {route} route accepts no query parameters; remove {key!r}"),
        )


async def _definition_options_impl(
    id: str, request: Request, *, deps: Deps, body: DefinitionOptionsRequest | None
) -> JSONResponse:
    """Answer from a fresh head on the read executor, else rebuild on the work executor."""
    _refuse_query_keys(request, route="options")
    request_deps = request_definition_deps(deps)
    trace_id = get_trace_id()
    with door_call("definition_options", logger=logger) as call:
        head_answer = await _run_retrieval(
            deps.read_executor,
            _serve_spec_options_head,
            deps=request_deps,
            timings=call.timings,
            spec_id=id,
            trace_id=trace_id,
            body=body,
        )
        envelope = (
            head_answer
            if isinstance(head_answer, OptionsEnvelope)
            else await _run_retrieval(
                deps.work_executor,
                _serve_spec_options,
                deps=request_deps,
                timings=call.timings,
                spec_id=id,
                trace_id=trace_id,
                body=body,
                head_read=head_answer,
            )
        )
    return JSONResponse(
        envelope.model_dump(mode="json", exclude_none=True),
        headers=no_store_headers(server_timing=call.server_timing),
    )


async def _definition_members_impl(
    id: str,
    request: Request,
    *,
    deps: Deps,
    body: DefinitionMembersRequest | None,
) -> JSONResponse:
    """Resolve members on the read executor; nothing is materialised."""
    _refuse_query_keys(request, route="members")
    with door_call("definition_members", logger=logger) as call:
        envelope = await _run_retrieval(
            deps.read_executor,
            _serve_spec_members,
            timings=call.timings,
            spec_id=id,
            deps=request_definition_deps(deps),
            body=body,
        )
    return JSONResponse(
        envelope.model_dump(mode="json"),
        headers=no_store_headers(server_timing=call.server_timing),
    )


# The failures of member resolution: its graph reads fail or run past a deadline, and building
# requests from the identity raises `CurrencyRoleUndeclared` for a unit request whose currency
# role the StatVar shape lacks.
_RESOLUTION_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    GraphReadError,
    GraphDeadlineExceededError,
    MemberReadDeadline,
    CurrencyRoleUndeclared,
)

# The failures of a materialisation run, which the definition and options routes make.
_RUN_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    MixedGrainError,
    MembersNotHeldError,
    DatasetDisagreementError,
    CollectionProbeCeilingExceededError,
    UpstreamVersionChangedError,
    UpstreamResponseError,
    UnSystemUnavailable,
    # A rebuild at one place whose tie needs an unread whole-variable breadth refuses rather
    # than store another facet (`current.serve_current`).
    VariableBreadthUnreadError,
    # A materialisation and a head-hit render hold a permit of the materialise semaphore.
    MintBackpressure,
    # A frame the grammar refuses, built fresh or read back from the store.
    FrameError,
    SdmxQueryError,
    SdmxTransportError,
    SdmxHttpError,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxCapExceededError,
    SdmxMissingFilterError,
    SdmxUnsupportedFilterError,
    SdmxContainmentFilterError,
    SdmxIncompatibleVariablesError,
    # A rebuild stores the canonical frame and the statistics.
    ArtifactRecordConflictError,
    ArtifactWriteError,
    ArtifactInventoryError,
)

# The refusals every retrieval route can raise.
_ANY_ROUTE_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    SpecNotFound,
    DefinitionMemberNotResolvable,
    StoredTraceMissingResolvedStatVars,
    FacetNotAvailableError,
    QueryOptionsInvalid,
    DefinitionBackpressure,
    # Admission of a posted definition raises these. The mint and register classifiers map them.
    MalformedSidecarIdentity,
    DefinitionNotReproducible,
    SpecNotExportable,
    RoleSignatureIncompatible,
    OverCeilingError,
    # The surface composes the store classifier. Version and head writes can raise this type.
    DatasetsStoreUnavailable,
    # Resolving a registered id registers its served identity when that one is new
    # (`kinds.spec_id.resolve_served_definition`), which refuses a colliding identity.
    IdentityCollision,
    # The spec reader raises it for a quarantined id, which every route reads.
    IdentityQuarantined,
    # The spec reader, `register_definition` and `register_served_definition` raise it when the
    # registry pool has no connection free.
    SpecRegistryUnavailable,
    *_RESOLUTION_EXCEPTION_TYPES,
)

# The refusals of the currency check and the materialisation behind it, which the definition
# and options routes run and the members route does not. Resolution confirms registration and a
# head names a recorded version, so only a revoked version reaches this door from the snapshot
# loader. A rebuild over the inline budget submits an operation, which a full backlog refuses.
_MATERIALISING_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    OperationBacklogFull,
    DefaultFacetUndefinedError,
    ExcludedByRegistry,
    RecipeNotDeclaredError,
    RateCoverageGapError,
    NonCurrencyUnitError,
    FacetBindingUnsupportedError,
    RateTableInvalidError,
    SnapshotRevoked,
    *_RUN_EXCEPTION_TYPES,
)

# The refusals of the requested rendering, which only the definition route renders: a head hit
# renders a missing format from the stored frame, and a rebuild renders the request.
_RENDERING_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    CapabilityUnavailable,
    ExcelLimitError,
    ExcelIllegalCharacterError,
    ParquetSizeBoundError,
    RollupNotSummableError,
    PivotNotUniqueError,
    PivotOverColumnCapError,
    PivotOverCellCapError,
    PivotNotHomogeneousError,
    ColumnNotDroppableError,
    ColumnUnknownError,
)

# The types above the mint door raises too, whose exemplars it declares.
_SHARED_WITH_MINT_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    *_RESOLUTION_EXCEPTION_TYPES,
    *_RUN_EXCEPTION_TYPES,
    *_RENDERING_EXCEPTION_TYPES,
)

SERVING_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    *_ANY_ROUTE_EXCEPTION_TYPES,
    *_MATERIALISING_EXCEPTION_TYPES,
)
"""The refusals of serving a registered id's current version (`_served_spec`, then
`_serve_spec`), the requested rendering aside."""

ROUTE_EXCEPTION_TYPES: Mapping[str, tuple[type[Exception], ...]] = {
    "datasets.definition": (*SERVING_EXCEPTION_TYPES, *_RENDERING_EXCEPTION_TYPES),
    # A posted body is verified against the path id on the options and members routes.
    "datasets.options": (*SERVING_EXCEPTION_TYPES, SpecIdMismatch),
    "datasets.members": (*_ANY_ROUTE_EXCEPTION_TYPES, SpecIdMismatch),
}
"""The refusals each route of this door can raise, keyed by the code registry's endpoint name.
The door's `exception_types` is their union."""


OPERATION_SUBMITTED_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {
        "model": OperationResource,
        "description": (
            "The build runs as a background operation. Poll `Location` until the operation "
            "is done; the done poll carries the manifest."
        ),
        "headers": {
            "Location": {
                "description": "The operation's poll URL, `operation.poll` in the body.",
                "schema": {"type": "string"},
            },
            "Retry-After": {
                "description": (
                    "Whole seconds to wait before the next poll, rounded up from "
                    "`operation.retry_after_ms`. The value grows as the build runs."
                ),
                "schema": {"type": "integer", "minimum": 1},
            },
        },
    }
}
"""The `202` the mint and definition routes answer when a build exceeds the inline budget."""


def build_definition_door(*, deps: DepsCell) -> DoorContribution:
    """Build the retrieval door over the surface's dependency cell.

    Head age and recheck interval come from the bundle's settings, and the freshness clock from
    its `now`.
    """
    router = APIRouter()

    @router.get(
        "/api/datasets/{id:dfo_spec_id}",
        response_model=Manifest,
        responses=OPERATION_SUBMITTED_RESPONSES,
    )
    async def get_definition(id: str, request: Request) -> JSONResponse:
        return await _get_definition_impl(id, request, deps=deps.get())

    @router.get("/api/datasets/{id:dfo_spec_id}/options", response_model=OptionsEnvelope)
    async def get_definition_options(id: str, request: Request) -> JSONResponse:
        """The definition's current version with no rendering, reported as
        `output_options`: what the data honours. A rebuild records the version and advances
        the head as a GET does. Any query key refuses `request_invalid` with an
        `/options/<key>` pointer, the same `QueryOptionsInvalid` this door's `get_definition`
        raises for a query key outside its own namespace, through the same
        `query_options.pointer_for_key` mapping.
        """
        return await _definition_options_impl(id, request, deps=deps.get(), body=None)

    @router.post("/api/datasets/{id:dfo_spec_id}/options", response_model=OptionsEnvelope)
    async def post_definition_options(
        id: str, body: DefinitionOptionsRequest, request: Request
    ) -> JSONResponse:
        """The GET route's envelope for a spec id whose canonical `definition` the body
        carries, registered or not. An unregistered definition is admitted as a mint admits a
        sidecar, registered, and served as the GET route serves a registered one, so the
        mint that follows is a head hit. Query keys refuse as on the GET route.
        """
        return await _definition_options_impl(id, request, deps=deps.get(), body=body)

    @router.post("/api/datasets/{id:dfo_spec_id}/members", response_model=MembersEnvelope)
    async def post_definition_members(
        id: str, request: Request, body: DefinitionMembersRequest | None = None
    ) -> JSONResponse:
        """The members a mint of the spec serves, resolved without materialising a frame:
        each member's ref, label, declared roles and constraints, their count, and the
        family's count and digest. The body may carry the spec's canonical `definition`,
        which an unregistered id needs; it is admitted as a mint admits a sidecar and nothing
        is registered. Refusals are the mint's member refusals. Query keys refuse as on the
        options route.
        """
        return await _definition_members_impl(id, request, deps=deps.get(), body=body)

    return DoorContribution(
        router=router,
        registry_endpoints={
            "/api/datasets/{id:dfo_spec_id}": "datasets.definition",
            "/api/datasets/{id:dfo_spec_id}/options": "datasets.options",
            "/api/datasets/{id:dfo_spec_id}/members": "datasets.members",
        },
        classify=classify_definition_exception,
        exception_types=tuple(dict.fromkeys(chain.from_iterable(ROUTE_EXCEPTION_TYPES.values()))),
        exception_exemplars=(
            SpecNotFound(id="spec_exemplar"),
            DefinitionMemberNotResolvable(members=("spec_exemplar",)),
            StoredTraceMissingResolvedStatVars(
                subject_id="spec_exemplar", content_version="cv_exemplar"
            ),
            without_pointers(FacetNotAvailableError("exemplar", facet_pointer="/facets/exemplar")),
            without_pointers(
                ExcludedByRegistry(variables=("Exemplar",), provenances=("dc/base/Exemplar",))
            ),
            QueryOptionsInvalid(pointer="/options/exemplar", detail="exemplar"),
            DefinitionBackpressure(),
            MalformedSidecarIdentity(detail="exemplar", pointer=_DEFINITION_POINTER),
            SpecIdMismatch(supplied="spec_exemplar", computed="spec_exemplar_computed"),
            SpecRegistryUnavailable("registry write pool exhausted"),
            DefinitionNotReproducible(spec_ids=("spec_exemplar",), detail="exemplar"),
            SpecNotExportable(spec_id="spec_exemplar", detail="exemplar"),
            RoleSignatureIncompatible(
                spec_id="spec_exemplar", roles_by_stat_var={"dcid:exemplar": ("recipient",)}
            ),
            OverCeilingError("exemplar", ceiling_kind="rows", count=1, cap=1),
            RecipeNotDeclaredError("exemplar", cause="recipe_undeclared"),
            RateCoverageGapError(
                "exemplar", cause="currency_unmapped", entities=("CurrencyExemplar",)
            ),
            NonCurrencyUnitError("exemplar", unit="Exemplar"),
            FacetBindingUnsupportedError("exemplar"),
            RateTableInvalidError("exemplar"),
            DatasetsStoreUnavailable("exemplar"),
            SnapshotRevoked(id="spec_exemplar", content_version="cv_exemplar", reason=None),
            IdentityCollision(subject_id="spec_exemplar"),
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            # The mint's exemplars of the materialisation and rendering refusals both doors raise.
            *(
                exemplar
                for exemplar in MINT_EXCEPTION_EXEMPLARS
                if isinstance(exemplar, _SHARED_WITH_MINT_EXCEPTION_TYPES)
            ),
        ),
    )
