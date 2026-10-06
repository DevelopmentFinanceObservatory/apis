"""The register door writes definitions. Minting handles
materialisation and artifact writes.

It maps six exception types. Identity mismatches return 422. Collisions and
quarantines return non-retryable 500s. Registry and executor capacity failures
return retryable 503s. Invalid identity documents return `request_invalid`
(400). Platform validation handles other malformed bodies and unsupported
schema versions.

The fronting Data access proxy authenticates requests and enforces 401, 413,
415, and 429. The deployment platform limits request-body size.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from contextlib import asynccontextmanager

from dfo_spec_kit.identity import IdentityError
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse, Response

from dataset_store import ConnectionSettings, LiveDatasetStore, close_shared_pool, shared_pool
from datasets_api.door_executor import DoorExecutor, build_door_executor
from datasets_api.models import (
    CanonicalDefinition,
    DefinitionResponse,
    RegisterRequest,
)
from datasets_api.shutdown import ShutdownBudget
from datasets_api.timings import PhaseTimings, door_call, phase, run_with_timings
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.trace import get_trace_id
from spec_registry import (
    IdentityCollision,
    IdentityQuarantined,
    LiveSpecRegistry,
    SpecIdMismatch,
    SpecRegistry,
    SpecRegistryUnavailable,
)

logger = logging.getLogger(__name__)

# Matches the `DATASET_STORE_POOL_SIZE` default for injected registries.
_FALLBACK_POOL_SIZE = 2


class RegistryBackpressure(Exception):
    """Full capacity returns retryable `service_unavailable` (503) and sheds load
    during registry degradation.
    """


def classify_register_exception(exc: BaseException) -> ProblemSpec | None:
    """Map this door's six exception types. Defer others to platform handling."""
    if isinstance(exc, SpecIdMismatch):
        detail = (
            f"supplied spec_id {exc.supplied!r} does not match the "
            f"server-recomputed id {exc.computed!r}"
        )
        return ProblemSpec(status=422, code="spec_id_mismatch", detail=detail, retryable=False)

    if isinstance(exc, IdentityCollision):
        logger.error("Datasets register door surfaced an identity collision: %s", exc, exc_info=exc)
        return ProblemSpec(
            status=500,
            code="identity_collision",
            detail=f"identity collision at {exc.subject_id!r}: byte-different content already held at this id",
            retryable=False,
        )

    if isinstance(exc, IdentityQuarantined):
        logger.error(
            "Datasets register door refused an already-quarantined %s id %s",
            exc.id_kind,
            exc.subject_id,
        )
        return ProblemSpec(
            status=500,
            code="identity_quarantined",
            detail=(
                f"{exc.id_kind} id {exc.subject_id!r} is collision-quarantined; registration is "
                "refused until the quarantine is resolved"
            ),
            retryable=False,
        )

    if isinstance(exc, SpecRegistryUnavailable):
        logger.warning("Datasets register door's registry write pool is unavailable: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the registry write pool had no connection free right now; retry",
            retryable=True,
        )

    if isinstance(exc, RegistryBackpressure):
        logger.warning(
            "Datasets register door's write capacity gate refused admission: "
            "every dedicated executor worker slot is already occupied"
        )
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the registry door is at its concurrent-write capacity right now; retry",
            retryable=True,
        )

    if isinstance(exc, IdentityError):
        # The identity document failed schema validation. This differs from
        # `SpecIdMismatch`, where valid identity data hashes to another id.
        return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)

    return None


def build_register_door(
    *,
    registry: SpecRegistry | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build a register door with a dedicated executor and admission semaphore.

    Registry writes use this executor, leaving the process-wide pool free when
    the datastore blocks. Its size matches the live registry's pool. Full
    capacity raises `RegistryBackpressure` and sheds load. Injected registries
    default to `_FALLBACK_POOL_SIZE`. Live registries default to
    `DATASET_STORE_POOL_SIZE`. `pool_size` overrides either default.

    An omitted registry is built at startup and closed at shutdown. `budget` is
    shared across mounted doors. An omitted budget creates a private shutdown budget.
    """
    _budget = budget if budget is not None else ShutdownBudget()
    _shared_registry: SpecRegistry | None = None
    _door_executor: DoorExecutor | None = None

    if registry is not None:
        # An injected registry exists at construction, so build its executor
        # and semaphore then.
        _door_executor = build_door_executor(
            pool_size if pool_size is not None else _FALLBACK_POOL_SIZE, "registry"
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _shared_registry, _door_executor
        if registry is None:
            pool_opened_here = False
            try:
                connections = shared_pool()
                pool_opened_here = True
                _shared_registry = LiveSpecRegistry(store=LiveDatasetStore(connections=connections))
                effective_pool_size = pool_size
                if effective_pool_size is None:
                    effective_pool_size = ConnectionSettings.from_env(
                        user_var="DATASET_STORE_RUNTIME_DB_USER",
                        optional_password_var="DATASET_STORE_RUNTIME_DB_PASSWORD",
                        enable_iam_auth=True,
                    ).pool_size
                _door_executor = build_door_executor(effective_pool_size, "registry")
            except BaseException:
                # Close only if `shared_pool()` returned and registered a borrow.
                if pool_opened_here:
                    close_shared_pool()
                raise
        try:
            yield
        finally:
            # Cancel queued calls. Drain running calls before closing their
            # pool dependencies.
            if _door_executor is not None:
                _door_executor.executor.shutdown(wait=False, cancel_futures=True)
                await asyncio.to_thread(_door_executor.drain, _budget)
            if registry is None:
                close_shared_pool()

    def _registered_call(fn, /, **kwargs):
        """A registry write is one `register` phase because its implementation is opaque."""
        with phase("register"):
            return fn(**kwargs)

    async def _run_registered(fn, /, *, timings: PhaseTimings, **kwargs):
        """Run a registry call on the dedicated executor under the `register` phase."""
        assert _door_executor is not None  # lifespan guarantees this
        return await _door_executor.run(
            functools.partial(
                run_with_timings, functools.partial(_registered_call, fn), timings, **kwargs
            ),
            refuse=RegistryBackpressure,
        )

    router = APIRouter()

    @router.post("/api/datasets/register", response_model=DefinitionResponse)
    async def register_endpoint(body: RegisterRequest) -> Response:
        active_registry = registry if registry is not None else _shared_registry
        assert active_registry is not None  # lifespan guarantees this before serving
        trace_id = get_trace_id()
        definition = body.definition
        with door_call("register", logger=logger) as call:
            result = await _run_registered(
                active_registry.register_definition,
                timings=call.timings,
                identity_definition=definition.identity,
                schema_version=definition.schema_version,
                supplied_spec_id=definition.spec_id,
                trace_id=trace_id,
            )
            body_json = DefinitionResponse(
                definition=CanonicalDefinition(
                    # The request carries the validated `Literal[1]`.
                    # The registry result has no schema-version field.
                    schema_version=definition.schema_version,
                    spec_id=result.spec_id,
                    identity=definition.identity,
                )
            ).model_dump(mode="json")
        return JSONResponse(body_json, headers={"Server-Timing": call.server_timing})

    return DoorContribution(
        router=router,
        classify=classify_register_exception,
        exception_types=(
            SpecIdMismatch,
            IdentityCollision,
            IdentityQuarantined,
            SpecRegistryUnavailable,
            RegistryBackpressure,
            IdentityError,
        ),
        exception_exemplars=(
            SpecIdMismatch(supplied="urn:spec:supplied", computed="urn:spec:computed"),
            IdentityCollision(subject_id="urn:spec:collision"),
            IdentityQuarantined(subject_id="urn:spec:quarantined", id_kind="spec"),
            SpecRegistryUnavailable("registry write pool exhausted"),
            RegistryBackpressure(),
            IdentityError("schema_violation", "identity document failed schema validation"),
        ),
        lifespan=lifespan,
    )
