"""The pinned frame grammar builds publish results from `role_keys` and `rows`. `spec_id` and
`content_version` use `ops_` and `ocv_` identities. Input versions accept both `cv_` and `ocv_` for
chained derivations. The JSON Schema holds identifier patterns.
This module duplicates them in Python. Keep both aligned with the envelope schema.

The fixed caps are 256 input versions (68 characters each), 16 role keys, 22 row keys, 1,024
characters per cell or key, 3 unit entries, 200 characters per StatVar ref in a unit, and
1,000,000 rows. Role names use open vocabulary.
The grammar NFC-normalises and deduplicates names and reserves fixed columns. CRS shapes use two
role keys. Dcids and time periods stay under 100 characters. Row width is six fixed columns plus
the role-key limit.

Cells are scalars, except under `unit`, which also takes one of two mappings. A converted unit
is one level of scalars with keys `currency`, `prices`, or `base_year`. A composed unit is
`{"numerator": part, "denominator": part}`, where a part is a unit ref string, a converted unit, or
`{"stat_var": "dcid:..."}`. A mapping holding `numerator` or `denominator` is read as a composed
unit, the rule `dfo_frame_grammar.unit_cell_from_json` applies. This module bounds each mapping's
size and depth and `Frame.build` judges its members. A StatVar ref in a unit matches `^dcid:.` and
is at most 200 characters, the bound the components catalogue puts on `counted_stat_var`.
`Frame.build` stores a unit string in NFC, which can be longer than the string sent, so the 200
and 1,024 character bounds on a unit's strings hold for the NFC form too. A converted unit's
`base_year`, alone or as a side, is within 1900 to 2100, the range the components catalogue
gives `structured_unit`. A validator checks row and unit key lengths on the raw body and
refuses a mapping under any row key but `unit`, before any cell is parsed, so no error can
echo an overlong key into its JSON Pointer.

`composed_stat_vars` describes the StatVars an analysis sum, quotient or share wrote its rows
under, which the graph does not hold, so Presentation can word them. It takes at most 16
entries and 64 side refs across them. An entry is `{stat_var, relation, first, second}`:
`relation` is `quotient` (`first` divided by `second`), `sum`, or `share` (`first` as a
percentage of `second`). Each side is a non-empty list of StatVar refs, and every ref
matches `^dcid:.`, is at most 200 characters in NFC, the form the entry is stored in, and
holds no U+0000, which the version record's `jsonb` `composed_stat_vars` column cannot store.
`stat_var` values are distinct, and a side ref that names an entry names an earlier one, so the
list holds no cycle. A side ref no entry describes is valid. The same validator rule applies to
keys: an entry's keys are checked on the raw body, and a key outside the four is refused there.

`MAX_ROWS` is a parse-time backstop before per-row validation and authorization. The separate
`PUBLISH_MAX_ROWS` ceiling is 250,000 rows, checked by `publish_frame` after validation and
authorization. That ceiling returns `over_ceiling` 422. `MAX_ROWS` returns `request_invalid` 400
before `publish_frame` runs.

`Recipe` is the op-spec tree analysis-mcp ran to produce a derived frame. This module checks its
top-level fields and its `anchor`, and keeps `tree` and `roots` as opaque JSON whose shape
analysis-mcp owns. Each series the anchor left out must carry coordinates that name it
(`RecipeLeftOutSeries`), since the recipe store keeps a pair's first recipe and every load of
the frame labels those series. One recipe is at most `MAX_RECIPE_BYTES` (1,000,000) of compact JSON. That is
the allowance `doors.publish.MAX_PUBLISH_BODY_BYTES` adds to its row bytes for everything in a
publish body other than rows, so a recipe never needs more room than the publish body already
gives it. The same cap applies on the recipe write route.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from typing import Annotated, Literal, Self

from dfo_frame_grammar import (
    FACET_IDENTITY_COLUMNS,
    STAT_VAR_COLUMN,
    STATUS_COLUMN,
    TIME_PERIOD_COLUMN,
    UNIT_COLUMN,
    VALUE_COLUMN,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    Tag,
    field_validator,
    model_validator,
)

from datasets_api.models import _require_schema_version_int

_OPS_SPEC_ID_PATTERN = r"^ops_[0-9a-f]{6,64}$"
_OCV_CONTENT_VERSION_PATTERN = r"^ocv_[0-9a-f]{6,64}$"
_INPUT_CONTENT_VERSION_PATTERN = r"^(?:cv|ocv)_[0-9a-f]{6,64}$"

MAX_INPUT_CONTENT_VERSIONS = 256
MAX_ROLE_KEYS = 16
MAX_ROW_WIDTH = 6 + MAX_ROLE_KEYS
MAX_CELL_CHARS = 1024
MAX_UNIT_ENTRIES = 3
MAX_COMPOSED_UNIT_ENTRIES = 2
MAX_UNIT_STAT_VAR_CHARS = 200
CATALOGUE_UNIT_BASE_YEARS = range(1900, 2101)
MAX_ROWS = 1_000_000
MAX_RECIPE_BYTES = 1_000_000
MAX_ECHO_CHARS = 80
MAX_COMPOSED_STAT_VARS = 16
MAX_COMPOSED_STAT_VAR_SIDE_REFS = 64

_NORMAL_FORM_HASH_PATTERN = r"^rnf_[0-9a-f]{32}$"
_REPLAY_KEY_PATTERN = r"^rpk_[0-9a-f]{32}$"

# Columns that name no series. A left-out series' coordinates must hold `stat_var` or a column
# outside these, the columns `assembler.bind.series_label` labels a series by.
_UNNAMING_COLUMNS = frozenset(
    {TIME_PERIOD_COLUMN, VALUE_COLUMN, UNIT_COLUMN, STATUS_COLUMN, *FACET_IDENTITY_COLUMNS}
)

OpsSpecId = Annotated[str, Field(pattern=_OPS_SPEC_ID_PATTERN)]
OcvContentVersion = Annotated[str, Field(pattern=_OCV_CONTENT_VERSION_PATTERN)]
ReplayKey = Annotated[str, Field(pattern=_REPLAY_KEY_PATTERN)]

_BoundedText = Annotated[str, Field(min_length=1, max_length=MAX_CELL_CHARS)]

_COMPOSED_UNIT_KEYS = frozenset({"numerator", "denominator"})
_STAT_VAR_PART_KEY = "stat_var"


def _unit_part_arm(value: object) -> str:
    if not isinstance(value, dict):
        return "unit_ref_part"
    return "stat_var_part" if _STAT_VAR_PART_KEY in value else "unit_request_part"


def _row_cell_arm(value: object) -> str:
    if not isinstance(value, dict):
        return "scalar_cell"
    return "composed_unit_cell" if _COMPOSED_UNIT_KEYS & value.keys() else "unit_request_cell"


def _arms_as_any_of(schema: dict[str, object]) -> None:
    """Publishes a discriminated union as `anyOf`. Its mapping arms overlap, so the `oneOf`
    Pydantic writes for a discriminated union would refuse a value that parses."""
    schema["anyOf"] = schema.pop("oneOf")


_CellText = Annotated[str, Field(max_length=MAX_CELL_CHARS)]
_RowScalar = _CellText | int | float | bool | None
_UnitMapping = Annotated[dict[str, _RowScalar], Field(max_length=MAX_UNIT_ENTRIES)]
_UnitStatVarRef = Annotated[str, Field(pattern=r"^dcid:.", max_length=MAX_UNIT_STAT_VAR_CHARS)]
_StatVarPart = Annotated[dict[str, _UnitStatVarRef], Field(max_length=1)]
# Each discriminator picks one arm from the value's shape, so an error names the member at
# fault. Arm tags differ from every unit member name, which keeps them out of the JSON Pointer.
_UnitPart = Annotated[
    Annotated[_CellText, Tag("unit_ref_part")]
    | Annotated[_StatVarPart, Tag("stat_var_part")]
    | Annotated[_UnitMapping, Tag("unit_request_part")],
    Discriminator(_unit_part_arm),
    Field(json_schema_extra=_arms_as_any_of),
]
_ComposedUnitMapping = Annotated[dict[str, _UnitPart], Field(max_length=MAX_COMPOSED_UNIT_ENTRIES)]
RowCellValue = Annotated[
    Annotated[_ComposedUnitMapping, Tag("composed_unit_cell")]
    | Annotated[_UnitMapping, Tag("unit_request_cell")]
    | Annotated[_RowScalar, Tag("scalar_cell")],
    Discriminator(_row_cell_arm),
    Field(json_schema_extra=_arms_as_any_of),
]


def _bound_keys(keys: Iterable[object], *, kind: str) -> None:
    """Refuses a string key over `MAX_CELL_CHARS`, echoing its first `MAX_ECHO_CHARS` characters."""
    for key in keys:
        if isinstance(key, str) and len(key) > MAX_CELL_CHARS:
            raise ValueError(
                f"{kind} key {key[:MAX_ECHO_CHARS]!r} (length {len(key)}) exceeds "
                f"MAX_CELL_CHARS={MAX_CELL_CHARS}"
            )


def _bound_unit_values(unit: object) -> None:
    """Refuses a unit, or a side of a composed unit, that the components catalogue cannot
    carry: a unit ref or StatVar ref whose NFC form exceeds its bound (`MAX_CELL_CHARS`,
    `MAX_UNIT_STAT_VAR_CHARS`), and a converted unit whose `base_year` is outside
    `CATALOGUE_UNIT_BASE_YEARS`."""
    composed = isinstance(unit, dict) and _COMPOSED_UNIT_KEYS & unit.keys()
    for side in unit.values() if composed else (unit,):
        if isinstance(side, dict):
            base_year = side.get("base_year")
            if (
                isinstance(base_year, int)
                and not isinstance(base_year, bool)
                and base_year not in CATALOGUE_UNIT_BASE_YEARS
            ):
                raise ValueError(
                    f"unit base_year is outside {CATALOGUE_UNIT_BASE_YEARS.start} to "
                    f"{CATALOGUE_UNIT_BASE_YEARS.stop - 1}"
                )
            text, bound = side.get(_STAT_VAR_PART_KEY), MAX_UNIT_STAT_VAR_CHARS
        else:
            text, bound = side, MAX_CELL_CHARS
        if not isinstance(text, str) or unicodedata.is_normalized("NFC", text):
            continue
        length = len(unicodedata.normalize("NFC", text))
        if length > bound:
            raise ValueError(
                f"unit text {text[:MAX_ECHO_CHARS]!r} is {length} characters in NFC form, over {bound}"
            )


class ComposedStatVar(BaseModel):
    """One StatVar an analysis operation wrote its rows under, described by the StatVars it
    combines."""

    model_config = ConfigDict(extra="forbid")

    stat_var: _UnitStatVarRef = Field(description="the composed StatVar's `dcid:`-prefixed ref")
    relation: Literal["quotient", "sum", "share"] = Field(
        description=(
            "`quotient` for `first` divided by `second`, `sum` for `first` plus `second`, "
            "`share` for `first` as a percentage of `second`"
        )
    )
    first: list[_UnitStatVarRef] = Field(
        min_length=1,
        max_length=MAX_COMPOSED_STAT_VAR_SIDE_REFS,
        description="the StatVar refs of the first side: a quotient's or share's numerator",
    )
    second: list[_UnitStatVarRef] = Field(
        min_length=1,
        max_length=MAX_COMPOSED_STAT_VAR_SIDE_REFS,
        description="the StatVar refs of the second side: a quotient's or share's denominator",
    )


_COMPOSED_STAT_VAR_KEYS = frozenset(ComposedStatVar.model_fields)


class RecipeLeftOutSeries(BaseModel):
    """One series a rolling anchor left out: it holds no row at the anchor period.

    `coordinates` must name the series, since a reader labels it from them: they hold
    `stat_var`, or at least one entity-role column, which is any column other than the frame
    grammar's fixed columns (`stat_var`, `time_period`, `value`, `unit`, `status`) and its
    facet identity columns (`measurementMethod`, `provenance`). Empty coordinates, and
    coordinates of only those other columns, are refused."""

    model_config = ConfigDict(extra="forbid")

    root: int = Field(
        ge=0,
        description=(
            "index of the population the series belongs to, in the order of the recipe's "
            "populations: one per root dataset"
        ),
    )
    coordinates: dict[str, _BoundedText] = Field(
        description=(
            "the series' dimension values, column name to value. Holds `stat_var` or at "
            "least one entity-role column"
        )
    )
    label: _BoundedText | None = Field(
        default=None,
        description=(
            "the series' display name. Optional: a reader of the recipe resolves the name "
            "from `coordinates` when it is absent"
        ),
    )

    @field_validator("coordinates", mode="after")
    @classmethod
    def _name_the_series(cls, value: dict[str, str]) -> dict[str, str]:
        if STAT_VAR_COLUMN not in value and all(column in _UNNAMING_COLUMNS for column in value):
            raise ValueError(
                f"coordinates must hold {STAT_VAR_COLUMN!r} or an entity-role column; "
                f"{sorted(column[:80] for column in value)} name no series"
            )
        return value


class RecipeAnchor(BaseModel):
    """The period a rolling recipe's `latest` resolved to, and the series it left out."""

    model_config = ConfigDict(extra="forbid")

    period: _BoundedText = Field(description="a period label at the frame's grain")
    rule: Literal["half1"] = Field(
        description="the anchor rule: the newest period at least half of the series hold"
    )
    left_out: list[RecipeLeftOutSeries]


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(description="recipe schema version; exactly 1")
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )

    normal_form_hash: str = Field(
        pattern=_NORMAL_FORM_HASH_PATTERN,
        description=(
            "hash of the recipe's normal form, in which every derived input is replaced by "
            "its own recipe, every root by its dataset spec_id and rendering, and every "
            "literal a rolling overlay governs is dropped. Two frames replayed from one "
            "recipe on different data share it"
        ),
    )
    replay_key: ReplayKey = Field(
        description=(
            "hash of the normal form hash, the root dataset pairs, and the execution "
            "identity. It names the one frame replayed from them"
        ),
    )
    execution_identity: _BoundedText = Field(
        description="the engine, grammar, and reduction versions that ran the recipe"
    )
    tree: dict[str, JsonValue] = Field(
        description="the op-spec tree, opaque here; analysis-mcp owns its shape"
    )
    roots: list[JsonValue] = Field(
        description="how each root dataset was minted, opaque here; analysis-mcp owns its shape"
    )
    anchor: RecipeAnchor | None = Field(
        default=None, description="the rolling anchor, or null for a recipe with no rolling period"
    )

    @model_validator(mode="after")
    def _bound_size(self) -> Self:
        compact = json.dumps(
            self.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False
        )
        size = len(compact.encode("utf-8"))
        if size > MAX_RECIPE_BYTES:
            raise ValueError(
                f"recipe is {size} bytes of compact JSON, exceeding "
                f"MAX_RECIPE_BYTES={MAX_RECIPE_BYTES}"
            )
        return self


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
            f"alone a mapping of scalars bounded at {MAX_UNIT_ENTRIES} entries or a composed "
            "unit, `numerator` over `denominator`, each a unit ref, a mapping of scalars, or "
            "`{stat_var}` -- see this module's own docstring for the shapes refused at parse "
            "time"
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

    recipe: Recipe | None = Field(
        default=None,
        description=(
            "the recipe that produced this frame. The door stores it beside the version, "
            "outside the body a republish is compared on, so a republish may add one to a "
            "frame published without it and may omit one the frame already holds"
        ),
    )

    composed_stat_vars: list[ComposedStatVar] = Field(
        default_factory=list,
        max_length=MAX_COMPOSED_STAT_VARS,
        description=(
            "the StatVars this frame's rows name that an analysis sum, quotient or share composed, each "
            "described by its sides so Presentation can label it. At most "
            f"{MAX_COMPOSED_STAT_VARS} entries and {MAX_COMPOSED_STAT_VAR_SIDE_REFS} side refs "
            "across them; `stat_var` values are distinct, and a side ref that names an entry "
            "names an earlier one. Stored with the frame's first publish"
        ),
    )

    @field_validator("composed_stat_vars", mode="before")
    @classmethod
    def _bound_composed_stat_var_keys(cls, value: object) -> object:
        """Refuses an entry key outside the model's four, on the raw body, so no unknown or
        overlong key reaches a JSON Pointer. Other shapes are left to the model."""
        if not isinstance(value, list):
            return value
        for entry in value:
            if not isinstance(entry, dict):
                continue
            for key in entry:
                if key not in _COMPOSED_STAT_VAR_KEYS:
                    raise ValueError(
                        f"composed_stat_vars key {str(key)[:MAX_ECHO_CHARS]!r} is not one of "
                        f"{sorted(_COMPOSED_STAT_VAR_KEYS)}"
                    )
        return value

    @field_validator("composed_stat_vars", mode="after")
    @classmethod
    def _bound_composed_stat_vars(cls, value: list[ComposedStatVar]) -> list[ComposedStatVar]:
        """Puts every ref in NFC, the form `Frame.build` stores a `stat_var` cell in, and
        refuses a ref over `MAX_UNIT_STAT_VAR_CHARS` in that form, a ref holding U+0000 (the
        version's `composed_stat_vars` is a Postgres `jsonb` value, which cannot hold it), more than
        `MAX_COMPOSED_STAT_VAR_SIDE_REFS` side refs, a repeated `stat_var`, and a side ref that
        names its own entry or a later one. Messages name an entry by its position."""
        entries: list[ComposedStatVar] = []
        for position, entry in enumerate(value):
            normalised = {
                "stat_var": unicodedata.normalize("NFC", entry.stat_var),
                "first": [unicodedata.normalize("NFC", ref) for ref in entry.first],
                "second": [unicodedata.normalize("NFC", ref) for ref in entry.second],
            }
            refs = (normalised["stat_var"], *normalised["first"], *normalised["second"])
            if any(len(ref) > MAX_UNIT_STAT_VAR_CHARS for ref in refs):
                raise ValueError(
                    f"composed_stat_vars entry {position} holds a ref over "
                    f"{MAX_UNIT_STAT_VAR_CHARS} characters in NFC form"
                )
            if any("\x00" in ref for ref in refs):
                raise ValueError(
                    f"composed_stat_vars entry {position} holds a ref with U+0000, which the "
                    "version record cannot store"
                )
            entries.append(entry.model_copy(update=normalised))
        if sum(len(entry.first) + len(entry.second) for entry in entries) > (
            MAX_COMPOSED_STAT_VAR_SIDE_REFS
        ):
            raise ValueError(
                f"composed_stat_vars holds more than {MAX_COMPOSED_STAT_VAR_SIDE_REFS} side refs"
            )
        position_of: dict[str, int] = {}
        for position, entry in enumerate(entries):
            if entry.stat_var in position_of:
                raise ValueError(
                    f"composed_stat_vars entries {position_of[entry.stat_var]} and {position} "
                    "name the same stat_var"
                )
            position_of[entry.stat_var] = position
        for position, entry in enumerate(entries):
            if any(position_of.get(ref, -1) >= position for ref in (*entry.first, *entry.second)):
                raise ValueError(
                    f"composed_stat_vars entry {position} names itself or a later entry in a "
                    "side; a side ref that names an entry names an earlier one"
                )
        return entries

    @field_validator("rows", mode="before")
    @classmethod
    def _bound_row_and_unit_keys(cls, value: object) -> object:
        """Bounds the keys of each row, of its `unit` mapping, and of either side of a composed
        unit, on the raw body, and permits a mapping only under `unit`. A cell that fails its
        type is reported at a JSON Pointer that names every key above it, and dict-key type
        constraints would do the same, so the only mappings a cell parser sees are ones whose
        keys are bounded. Other shapes are left to the model."""
        if not isinstance(value, list):
            return value
        for row in value:
            if not isinstance(row, dict):
                continue
            _bound_keys(row, kind="row")
            for key, cell in row.items():
                if not isinstance(cell, dict):
                    continue
                if key != "unit":
                    raise ValueError(
                        f"row key {str(key)[:MAX_ECHO_CHARS]!r} carries a mapping value; only 'unit' may"
                    )
                _bound_keys(cell, kind="unit")
                for side in cell.values():
                    if isinstance(side, dict):
                        _bound_keys(side, kind="unit")
        return value

    @field_validator("rows", mode="after")
    @classmethod
    def _bound_units(cls, value: list[dict[str, RowCellValue]]) -> list[dict[str, RowCellValue]]:
        """Bounds the NFC form of a unit's strings and a converted unit's `base_year`.
        Pydantic has already checked cell shapes and row widths. This validator walks the
        bounded rows in place."""
        for row in value:
            if "unit" in row:
                _bound_unit_values(row["unit"])
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
