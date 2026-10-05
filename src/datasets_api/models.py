"""Wire models for `POST /api/datasets/register`.

The request registers one canonical definition or a collection of definitions. Scope and
rendering options belong to mint requests. Identity schema validation and hash checks run in
`spec_registry`. This module validates the request shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Published on the capabilities document as the register door's member cap.
MAX_COLLECTION_MEMBERS = 100


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
    """`POST /api/datasets/register`'s body: exactly one of `definition` or
    `members`. `unsupported_schema_version` (the top-level `schema_version`
    field's own refusal) and the exactly-one-of/cap checks below both
    surface through the platform's generic `RequestValidationError` handling
    -- no bespoke handler needed in this surface for either.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(description="contract schema version; exactly 1 for v1")
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    definition: CanonicalDefinition | None = Field(
        default=None, description="a single canonical definition to register"
    )
    members: list[CanonicalDefinition] | None = Field(
        default=None,
        description="canonical definitions to register together as one collection",
    )

    @model_validator(mode="after")
    def _exactly_one_of_definition_or_members(self) -> RegisterRequest:
        if (self.definition is None) == (self.members is None):
            raise ValueError(
                "exactly one of 'definition' or 'members' must be supplied, never both or neither"
            )
        if self.members is not None:
            if len(self.members) == 0:
                raise ValueError("'members' must name at least one canonical definition")
            if len(self.members) > MAX_COLLECTION_MEMBERS:
                raise ValueError(
                    f"'members' names {len(self.members)} definitions, exceeding this "
                    f"deployment's cap of {MAX_COLLECTION_MEMBERS}"
                )
        return self


class DefinitionResponse(BaseModel):
    """The register door's response for a single `definition`: the verified
    identity, echoed back whole."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    definition: CanonicalDefinition


class CollectionIdentity(BaseModel):
    """A registered collection's own identity: the server-minted
    `collection_id` and its sorted, de-duplicated member spec_ids -- the
    same list the registry itself stores (`datasets.collection_member`),
    not each member's full definition re-embedded (the caller already
    supplied those; the response's job is to carry back the verified
    collection identity, not restate what was posted).
    """

    model_config = ConfigDict(extra="forbid")

    collection_id: str
    members: list[str]


class CollectionResponse(BaseModel):
    """The register door's response for a `members` collection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    collection: CollectionIdentity
