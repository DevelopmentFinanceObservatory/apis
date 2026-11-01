"""The register door writes definitions. Minting handles
materialisation and artifact writes.

It maps six exception types. Identity mismatches return 422. Collisions and
quarantines return non-retryable 500s. Registry and executor capacity failures
return retryable 503s. Invalid identity documents return `request_invalid`
(400). Platform validation handles other malformed bodies and unsupported
schema versions.

Callers reach the API through the website's proxy, which answers `429` once an
IP passes its allowance, 120 calls a minute on `/api/*` by default, counted in a
fixed window that every proxy instance shares in Postgres. A call whose count
fails goes through. The proxy passes `Authorization` upstream unread. The deployment platform
limits request-body size.
"""

from __future__ import annotations

import functools
import logging

from dfo_spec_kit.identity import IdentityError
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from datasets_api.backpressure import Backpressure
from datasets_api.deps import DepsCell
from datasets_api.models import (
    CanonicalDefinition,
    DefinitionResponse,
    RegisterRequest,
)
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.timings import door_call, phase, run_with_timings
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.trace import get_trace_id
from spec_registry import (
    IdentityCollision,
    IdentityQuarantined,
    SpecIdMismatch,
    SpecRegistryUnavailable,
)

logger = logging.getLogger(__name__)


class RegistryBackpressure(Backpressure):
    """Full capacity returns retryable `service_unavailable` (503) and sheds load
    during registry degradation.
    """

    code = "service_unavailable"

    detail = "the registry door is at its concurrent-write capacity right now; retry"


def _registry_backpressure() -> RegistryBackpressure:
    logger.warning(
        "Datasets register door's write capacity gate refused admission: "
        "every read executor slot is already occupied"
    )
    return RegistryBackpressure()


def classify_register_exception(exc: BaseException) -> ProblemSpec | None:
    """Map an identity document that fails schema validation, then any refusal."""
    if isinstance(exc, IdentityError):
        # This differs from `SpecIdMismatch`, where valid identity data hashes to another id.
        return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)
    return classify_datasets_refusal(exc)


def build_register_door(*, deps: DepsCell) -> DoorContribution:
    """Build the register door over the surface's dependency cell. Registry writes run on
    the read executor. A full executor raises `RegistryBackpressure` and sheds load.
    """

    def _registered_call(fn, /, **kwargs):
        """A registry write is one `register` phase because its implementation is opaque."""
        with phase("register"):
            return fn(**kwargs)

    router = APIRouter()

    @router.post("/api/datasets/register", response_model=DefinitionResponse)
    async def register(body: RegisterRequest) -> Response:
        current = deps.get()
        trace_id = get_trace_id()
        definition = body.definition
        with door_call("register", logger=logger) as call:
            result = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    functools.partial(_registered_call, current.registry.register_definition),
                    call.timings,
                    identity_definition=definition.identity,
                    schema_version=definition.schema_version,
                    supplied_spec_id=definition.spec_id,
                    trace_id=trace_id,
                ),
                refuse=_registry_backpressure,
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
        registry_endpoints=route_registry_endpoints(router, "datasets.register"),
        classify=classify_register_exception,
        exception_types=(
            SpecIdMismatch,
            IdentityCollision,
            IdentityQuarantined,
            SpecRegistryUnavailable,
            RegistryBackpressure,
            IdentityError,
        ),
        endpoint_exemplars={
            "datasets.register": (
                SpecIdMismatch(supplied="urn:spec:supplied", computed="urn:spec:computed"),
                IdentityCollision(subject_id="urn:spec:collision"),
                IdentityQuarantined(subject_id="urn:spec:quarantined", id_kind="spec"),
                SpecRegistryUnavailable("registry write pool exhausted"),
                RegistryBackpressure(),
                IdentityError("schema_violation", "identity document failed schema validation"),
            )
        },
    )
