"""Wire models for `POST /api/datasets/register`.

The request registers one canonical definition. Scope and rendering options belong to mint
requests. Identity schema validation and hash checks run in `spec_registry`. This module
validates the request shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dfo_platform.wire import ResponseModel


def _require_schema_version_int(value: object) -> object:
    """Rejects booleans and floats before Pydantic coerces them to `1`.

    A `Literal[1]` validator receives the raw value only in `mode="before"`. This check stays
    local to keep the Search and Datasets API surfaces independent.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("schema_version must be the integer 1")
    return value


class CanonicalDefinition(BaseModel):
    """One canonical definition, exactly as the contract shapes it: the wire
    version it was written under, its own spec_id (required -- the register
    door always verifies a supplied id, never mints one for a definition),
    and its identity-core document.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(
        description="the wire version this definition was written under"
    )
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    spec_id: str = Field(description="the caller-supplied spec_id this definition must hash to")
    identity: dict[str, Any] = Field(
        description="the identity-core document, shaped as search.models.identity.BaseIdentity "
        "and published at GET /api/search/schema"
    )


class RegisterRequest(BaseModel):
    """`POST /api/datasets/register`'s body: one canonical `definition`.
    `unsupported_schema_version` (the top-level `schema_version` field's own
    refusal) and any other shape failure surface through the platform's
    generic `RequestValidationError` handling.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(description="contract schema version; exactly 1 for v1")
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    definition: CanonicalDefinition = Field(description="the canonical definition to register")


class DefinitionResponse(ResponseModel):
    """The register door's response: the verified
    identity, echoed back whole."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    definition: CanonicalDefinition
