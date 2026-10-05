"""`GET /api/datasets/schema` publishes contract-stable schemas with immutable `$id`s. Register
and mint schemas reuse their runtime models. The manifest, options, preview, pin, and operation
schemas define the fields their doors enforce. Nested graph-dependent shapes remain open.
Problem details embeds the platform-owned schema without a datasets `$id`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from datasets_api.mint_models import (
    DefinitionOptionsRequest,
    MintInput,
    MintOptions,
    PreviewRequest,
)
from datasets_api.models import (
    CollectionResponse,
    DefinitionResponse,
    RegisterRequest,
)
from dfo_platform.problem import ProblemDetails

_SCHEMA_BASE = "https://schemas.dfo.one.org/datasets/v1"


class MintRequest(BaseModel):
    """`POST /api/datasets`'s body: one of the four `input` kinds
    (`spec_id`, `collection_id`, `statvar_set`, `source_codes`) plus a
    rendering-only `options` block, never hashed into identity, and an
    optional `preview` block. `preview` set asks for a `PreviewEnvelope`
    instead of a `Manifest`; `options` has no effect on that response.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"$id": f"{_SCHEMA_BASE}/mint-request.json"}
    )

    schema_version: Literal[1] = 1
    input: MintInput
    options: MintOptions = Field(default_factory=MintOptions)
    preview: PreviewRequest | None = None


class Manifest(BaseModel):
    """The manifest every materialising door returns: exactly one of
    `definition` or `collection`, plus the immutable version record
    (`content_version`, `coverage`, `citation`, `trace`, `warnings`), the
    live artifact inventory (`artifacts`), and the number of distinct
    series this definition resolves (`series_count`, `null` on a version
    record minted before this field existed). The per-series detail itself
    is a `series-statistics/1` Parquet artifact listed in `artifacts`
    whenever `series_count` is not `null`, never inline on this shape.
    Nested shapes (coverage/citation/trace/artifact entry) are
    left as open objects here. Each door's own runtime models define their
    frozen field sets separately, so fixing them in this schema risks a
    second, competing shape. On definition retrieval, `trace` carries
    `served_from` (`head` or `materialise`) and `confirmed_at` (conformance
    clauses C150/C151, `docs/conformance-datasets-api-v1.md`), the
    currency-facing pair a caller checks alongside `content_version`.
    `series_coverage_stamp` is lineage only (C156): the
    `series_coverage` artifact this request's sync-admission estimate ran
    against, `null` when a served-from-head response ran no estimate this
    call. `content_version` does not fold it in. `prepared_at` is the
    RFC 3339 instant the version was first minted. Every later mint and
    re-render of the version reports the same instant.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"$id": f"{_SCHEMA_BASE}/manifest.json"}
    )

    schema_version: Literal[1] = 1
    definition: dict[str, object] | None = None
    collection: dict[str, object] | None = None
    content_version: str = Field(pattern=r"^cv_")
    prepared_at: AwareDatetime
    coverage: dict[str, object]
    citation: dict[str, object]
    trace: dict[str, object]
    warnings: list[dict[str, object]] = Field(default_factory=list)
    artifacts: list[dict[str, object]]
    series_count: int | None = None
    output_options: dict[str, object] | None = None
    series_coverage_stamp: str | None = None

    @model_validator(mode="after")
    def _definition_xor_collection(self) -> Manifest:
        if (self.definition is None) == (self.collection is None):
            raise ValueError("exactly one of 'definition' or 'collection' must be present")
        return self


class OptionsEnvelope(BaseModel):
    """`GET /api/datasets/{id}/options`'s response, the write-free
    projection of a manifest onto `content_version`, `coverage`,
    `output_options`, `warnings`, and `trace`. Nested shapes stay open
    objects here, the same as `Manifest`'s. `trace` carries the same
    `served_from`/`confirmed_at` currency pair a manifest's own `trace`
    does, read from the stored subject head under the same rule, though
    this route never writes it. `confirmed_at` has the same meaning on
    both routes, the instant the stored head was last confirmed. This
    route records nothing, so it reports the stored instant.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"$id": f"{_SCHEMA_BASE}/options-envelope.json"}
    )

    schema_version: Literal[1] = 1
    definition: dict[str, object] | None = None
    collection: dict[str, object] | None = None
    content_version: str = Field(pattern=r"^cv_")
    coverage: dict[str, object]
    output_options: dict[str, object] | None = None
    warnings: list[dict[str, object]] = Field(default_factory=list)
    trace: dict[str, object]

    @model_validator(mode="after")
    def _definition_xor_collection(self) -> OptionsEnvelope:
        if (self.definition is None) == (self.collection is None):
            raise ValueError("exactly one of 'definition' or 'collection' must be present")
        return self


class PreviewMember(BaseModel):
    """One preview entry: a definition, or one resolved
    `(stat_var, entity/role-tuple)` pair of a collection. A `spec_id` preview
    reports one entry for the whole definition; when the definition resolves
    to a family of StatVars, `stat_var` is the family reference
    `dcid:<spec_id>` and `sample` holds rows of its members.
    `content_version` and `total_rows` are those a mint of the same
    definition reports: both run the same generation over the same full
    survivor set, so a client holding a preview can compare the token against
    a mint's directly and skip a re-materialisation it already holds. Nested
    shapes (`coverage`/`citation`/`sample`) are left as open objects here,
    the same convention `Manifest` uses.
    """

    model_config = ConfigDict(extra="forbid")

    stat_var: str
    content_version: str | None
    coverage: dict[str, object]
    citation: dict[str, object]
    warnings: list[dict[str, object]] = Field(default_factory=list)
    sample: dict[str, object]
    total_rows: int
    truncated: bool


class PreviewEnvelope(BaseModel):
    """`POST /api/datasets`'s response to a `preview`-set `MintRequest`,
    never a `Manifest`: no top-level `content_version`, no `artifacts`, no
    `definition`/`collection` identity block, and never written to the
    registry, the artifact store, or a version/head record. `members`
    carries one entry for a `spec_id` preview, covering every StatVar the
    definition resolves to, and one entry per resolved
    `(stat_var, entity/role-tuple)` pair for a `statvar_set`,
    `source_codes` or `collection_id` preview, whose mint serves each pair
    alone. A single entry is still a one-entry list, never a bare top-level
    member. Each entry carries its own `content_version`; no aggregate spans
    them.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"$id": f"{_SCHEMA_BASE}/preview-envelope.json"}
    )

    schema_version: Literal[1] = 1
    members: list[PreviewMember] = Field(min_length=1)


class PinRequest(BaseModel):
    """`POST /api/datasets/{id}/{content_version}/pin`'s body: names the
    exact bytes to pin, not a rendering to re-derive.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"$id": f"{_SCHEMA_BASE}/pin-request.json"}
    )

    schema_version: Literal[1] = 1
    artifact_digest: str = Field(pattern=r"^sha256:")


class PinnedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content_version: str = Field(pattern=r"^cv_")
    artifact_digest: str = Field(pattern=r"^sha256:")
    pinned_at: str


class PinResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"$id": f"{_SCHEMA_BASE}/pin-response.json"}
    )

    schema_version: Literal[1] = 1
    pinned: PinnedArtifact


class OperationProgress(BaseModel):
    """`rows_written` against `estimate_rows`, read live off
    `datasets.operation` at poll time (`datasets_api.operation_store
    .Operation`). Present on a `pending`/`running` operation's own
    envelope; omitted from the `202` submit envelope (nothing has run
    yet) and from a terminal (`done`/`failed`) operation's own envelope,
    whose outcome the sibling `manifest` field, or the poll route's own
    `operation_failed` refusal, already carries in full.
    """

    model_config = ConfigDict(extra="forbid")

    rows_written: int
    estimate_rows: int


class OperationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(pattern=r"^op_")
    status: Literal["pending", "running", "done", "failed"]
    poll: str | None = None
    retry_after_ms: int | None = Field(
        default=None,
        description=(
            "populated only on the initial 202 submit response; always None on a "
            "pending/running poll -- a caller polling with no backoff strategy of "
            "its own gets no ongoing pacing guidance from this field past the "
            "first response"
        ),
    )
    progress: OperationProgress | None = None


class OperationResource(BaseModel):
    """`GET /api/datasets/operations/{operation_id}`'s poll response. The
    capability document reports `operation_envelope_active: true`, and this
    is the shape the route returns for a known operation id in `pending`,
    `running`, or `done` state. A `failed` operation raises the platform's
    ordinary refusal path instead (`operation_failed`), and an unknown or
    unauthorized-`done` id answers 404 `operation_not_found` -- neither
    returns this shape.

    `manifest` is set only once `operation.status == "done"`: the same
    manifest a synchronous mint returns, re-signed at read time
    (`datasets_api.doors.meta`'s operations route reuses `datasets_api
    .doors.snapshot.read_snapshot_manifest` against the operation's own
    stored `subject_id`/`content_version`). `error` models a `failed`
    operation's outcome inline with this resource's own shape, but the
    door never populates it: a `failed` operation raises the platform's
    ordinary refusal path instead (`operation_failed`, a bare RFC 9457
    body, the same pattern `operation_not_found` already uses on this
    route -- see the registry's own `datasets.operations` mapping), so a
    poll never reaches this model at all once the underlying operation
    has failed. `error` stays declared for schema completeness -- this is
    a contract-stable, immutable-`$id` shape -- but no code path fills it.
    """

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"$id": f"{_SCHEMA_BASE}/operation-resource.json"}
    )

    operation: OperationEnvelope
    manifest: dict[str, object] | None = None
    error: dict[str, object] | None = None

    @model_validator(mode="after")
    def _manifest_or_error_only_when_terminal(self) -> OperationResource:
        if self.manifest is not None and self.error is not None:
            raise ValueError("'manifest' and 'error' are mutually exclusive")
        return self


def build_schema_document() -> dict[str, object]:
    """Returns each contract schema and the capabilities URL."""
    return {
        "schema_version": 1,
        "mint_request": MintRequest.model_json_schema(),
        "register_request": RegisterRequest.model_json_schema(),
        "register_response": {
            "definition": DefinitionResponse.model_json_schema(),
            "collection": CollectionResponse.model_json_schema(),
        },
        "manifest": Manifest.model_json_schema(),
        "definition_options_request": DefinitionOptionsRequest.model_json_schema(),
        "options_envelope": OptionsEnvelope.model_json_schema(),
        "preview_envelope": PreviewEnvelope.model_json_schema(),
        "pin_request": PinRequest.model_json_schema(),
        "pin_response": PinResponse.model_json_schema(),
        "operation_resource": OperationResource.model_json_schema(),
        "problem_details_extensions": ProblemDetails.model_json_schema(),
        "capabilities": "/api/datasets/capabilities",
    }


def schema_document_response_schema() -> dict[str, object]:
    """Builds the OpenAPI response schema from `build_schema_document`'s actual value types."""
    document = build_schema_document()
    properties: dict[str, object] = {}
    for key, value in document.items():
        if isinstance(value, dict):
            properties[key] = {"type": "object", "description": f"JSON Schema for {key}"}
        elif isinstance(value, bool):
            properties[key] = {"type": "boolean", "const": value}
        elif isinstance(value, int):
            properties[key] = {"type": "integer", "const": value}
        elif isinstance(value, str):
            properties[key] = {"type": "string"}
        else:
            raise TypeError(
                f"schema document key {key!r} holds an unsupported value kind for the "
                f"OpenAPI response schema: {type(value)!r}"
            )
    return {
        "type": "object",
        "required": list(document),
        "properties": properties,
    }
