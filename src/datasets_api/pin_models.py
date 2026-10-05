"""Pin requests name exact artifact bytes by digest. The subject and version come from the path."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datasets_api.models import _require_schema_version_int


class PinRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(
        default=1, description="contract schema version; exactly 1 for v1"
    )
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    artifact_digest: str = Field(
        pattern=r"^sha256:", description="the exact bytes to pin, e.g. 'sha256:3e7a...'"
    )


class PinnedCoordinates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="the spec_id or collection id from the path")
    content_version: str
    artifact_digest: str
    pinned_at: str


class PinResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    pinned: PinnedCoordinates
