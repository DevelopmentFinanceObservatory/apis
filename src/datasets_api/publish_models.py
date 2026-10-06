"""The pinned frame grammar builds publish results from `role_keys` and `rows`. `spec_id` and
`content_version` use `ops_` and `ocv_` identities. Input versions accept both `cv_` and `ocv_` for
chained derivations. The JSON Schema holds identifier patterns.
This module duplicates them in Python. Keep both aligned with the envelope schema.

The fixed caps are 256 input versions (68 characters each), 16 role keys, 22 row keys, 1,024
characters per cell or key, 3 unit entries, and 1,000,000 rows. Role names use open vocabulary.
The grammar NFC-normalises and deduplicates names and reserves fixed columns. CRS shapes use two
role keys. Dcids and time periods stay under 100 characters. Row width is six fixed columns plus
the role-key limit.

Cells are scalars, with one level of scalar mapping under `unit` and keys `currency`, `prices`, or
`base_year`. The validator checks dict-key lengths itself to protect the JSON Pointer from echoing
overlong keys.

`MAX_ROWS` is a parse-time backstop before per-row validation and authorization. The separate
`PUBLISH_MAX_ROWS` ceiling is 250,000 rows, checked by `publish_frame` after validation and
authorization. That ceiling returns `over_ceiling` 422. `MAX_ROWS` returns `request_invalid` 400
before `publish_frame` runs.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datasets_api.models import _require_schema_version_int

_OPS_SPEC_ID_PATTERN = r"^ops_[0-9a-f]{6,64}$"
_OCV_CONTENT_VERSION_PATTERN = r"^ocv_[0-9a-f]{6,64}$"
_INPUT_CONTENT_VERSION_PATTERN = r"^(?:cv|ocv)_[0-9a-f]{6,64}$"

MAX_INPUT_CONTENT_VERSIONS = 256
MAX_ROLE_KEYS = 16
MAX_ROW_WIDTH = 6 + MAX_ROLE_KEYS
MAX_CELL_CHARS = 1024
MAX_UNIT_ENTRIES = 3
MAX_ROWS = 1_000_000

_RowScalar = Annotated[str, Field(max_length=MAX_CELL_CHARS)] | int | float | bool | None
_UnitMapping = Annotated[dict[str, _RowScalar], Field(max_length=MAX_UNIT_ENTRIES)]
# Keep the mapping arm first so Pydantic reports its bounded-map errors before scalar-union errors.
RowCellValue = _UnitMapping | _RowScalar


class PublishFrameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(
        default=1, description="contract schema version; exactly 1 for v1"
    )
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )

    spec_id: str = Field(
        pattern=_OPS_SPEC_ID_PATTERN,
        description="the derived frame's own op-spec hash id, ops_-prefixed",
    )
    content_version: str = Field(
        pattern=_OCV_CONTENT_VERSION_PATTERN,
        description="the derived frame's own toolkit content version, ocv_-prefixed",
    )
    role_keys: list[Annotated[str, Field(max_length=MAX_CELL_CHARS)]] = Field(
        max_length=MAX_ROLE_KEYS,
        description=(
            "`dfo_frame_grammar.Frame.build`'s own `role_keys` argument; empty is a valid "
            "subject-only frame with no role columns, `build_frame` accepts it, so no minimum "
            f"is imposed here. Bounded at {MAX_ROLE_KEYS} entries, each at most "
            f"{MAX_CELL_CHARS} characters, a schema fact rather than a deployment ceiling -- "
            "see this module's own docstring"
        ),
    )
    rows: list[Annotated[dict[str, RowCellValue], Field(max_length=MAX_ROW_WIDTH)]] = Field(
        max_length=MAX_ROWS,
        description=(
            "`dfo_frame_grammar.Frame.build`'s own `rows` argument; empty is a valid frame "
            "(an analysis verb whose filter matches nothing produces exactly this), "
            "`build_frame` accepts it, so no minimum is imposed here. Each row is bounded at "
            f"{MAX_ROW_WIDTH} keys; the outer list is bounded at {MAX_ROWS} rows, a fixed "
            "schema backstop, well above doors.publish's own fixed PUBLISH_MAX_ROWS that "
            "actually governs `len(rows)` at publish time -- see this module's own docstring "
            "for why the two are not the same kind of cap. A cell is a scalar, or under `unit` "
            f"alone a mapping of scalars bounded at {MAX_UNIT_ENTRIES} entries -- see this "
            "module's own docstring for the shapes refused at parse time"
        ),
    )
    input_content_versions: list[Annotated[str, Field(pattern=_INPUT_CONTENT_VERSION_PATTERN)]] = (
        Field(
            min_length=1,
            max_length=MAX_INPUT_CONTENT_VERSIONS,
            description=(
                "every input frame's own frame-grammar content_version this derived frame "
                f"was computed from. Bounded at {MAX_INPUT_CONTENT_VERSIONS} entries, a "
                "schema fact rather than a deployment ceiling -- see this module's own "
                "docstring"
            ),
        )
    )

    @field_validator("rows", mode="after")
    @classmethod
    def _bound_row_keys_and_unit_shape(
        cls, value: list[dict[str, RowCellValue]]
    ) -> list[dict[str, RowCellValue]]:
        """Bounds row and unit mapping keys and permits mappings only under `unit`. Dict-key type
        constraints would echo an overlong key into the JSON Pointer error. Pydantic has already
        checked cell shapes and row widths. This validator walks the bounded rows in place."""
        for row in value:
            for key, cell in row.items():
                if len(key) > MAX_CELL_CHARS:
                    raise ValueError(
                        f"row key {key[:80]!r} (length {len(key)}) exceeds "
                        f"MAX_CELL_CHARS={MAX_CELL_CHARS}"
                    )
                if key != "unit" and isinstance(cell, dict):
                    raise ValueError(
                        f"row key {key[:80]!r} carries a mapping value; only 'unit' may"
                    )
                if key == "unit" and isinstance(cell, dict):
                    for unit_key in cell:
                        if len(unit_key) > MAX_CELL_CHARS:
                            raise ValueError(
                                f"unit key {unit_key[:80]!r} (length {len(unit_key)}) exceeds "
                                f"MAX_CELL_CHARS={MAX_CELL_CHARS}"
                            )
        return value


class PublishedFrameCoordinates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str
    artifact_digest: str
    created: bool = Field(
        description=(
            "true when this call newly recorded the version row; false when an identical "
            "publish of this exact pair had already been recorded (the idempotent replay path)"
        )
    )


class PublishFrameResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    published: PublishedFrameCoordinates
