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
publish body other than rows and steps, so a recipe never needs more room than the publish body
already gives it. The same cap applies on the recipe write route.

`steps` are the frame's step records (`description.steps.Step`), one per operation that computed
it, in order. A step's shape is the kit's step schema, and its `op` is any bounded name, so an
operation dfo-spec-kit has not listed yet still publishes. The list meets dfo-spec-kit's structure
rules (`dfo_descriptions.check_steps`), checked in the kit's order: at most `MAX_STEPS` (62)
steps, at most `MAX_STEPS_BYTES` (65,536) bytes of RFC 8785 canonical JSON, step ids unique, and every input
that names a step naming an earlier one, so the steps form no cycle. A version reference is the
`input` of a `retrieve` step and of nothing else, and at least one step names one, so every
version the steps name is one of the frame's root inputs, which the record lists, and a record
with steps lists at least one. `input_content_versions` names the immediate input frames, a
different list for a chained derivation, so the steps are not checked against it. A
`continue_series` step states the fact a continuation is read from, so its `params` must hold
`method`, a non-empty string, and `periods`, two strings, the first and last period the
continuing series supplies. The byte bound
is the kit's own. The count bound is `description.steps.MAX_COMPUTED_STEPS`, the kit's 64 less the
two steps a card drawn from the frame adds, so the card's description stays within the kit's
bound. The largest chain the nine
analysis operations plausibly produce is 17 steps and 42,343 bytes: eight labelled retrievals,
one summing 64 variables and choosing among eight sources, then each operation once, with three
data gaps and a composed unit each, a rank covering 200 entities, and three steps covering 65
periods. The steps have their own allowance in `MAX_PUBLISH_BODY_BYTES`, since a recipe alone can
fill the allowance for everything else.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, Self

from dfo_descriptions import MAX_STEPS_BYTES, StepsError, StepsErrorReason, check_steps
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
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from datasets_api.models import _require_schema_version_int
from description.steps import CONTINUE_SERIES_OP, MAX_COMPUTED_STEPS, Step, VersionRef

_OPS_SPEC_ID_PATTERN = r"^ops_[0-9a-f]{6,64}$"
_OCV_CONTENT_VERSION_PATTERN = r"^ocv_[0-9a-f]{6,64}$"
_INPUT_CONTENT_VERSION_PATTERN = r"^(?:cv_[0-9a-f]{32}|ocv_[0-9a-f]{6,64})$"

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
MAX_STEPS = MAX_COMPUTED_STEPS

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


def _named_steps(step: Step) -> list[tuple[tuple[str, ...], str]]:
    """The step ids a step's inputs name, each with its location in the step, in order. An
    input that is a version reference names no step."""
    if isinstance(step.input, str):
        return [(("input",), step.input)]
    inputs = step.inputs if isinstance(step.inputs, dict) else {}
    return [(("inputs", role), named) for role, named in inputs.items() if isinstance(named, str)]


def _version_refs(step: Step) -> list[tuple[tuple[str, ...], VersionRef]]:
    """The version references a step's inputs name, each with its location in the step."""
    if isinstance(step.input, VersionRef):
        return [(("input",), step.input)]
    inputs = step.inputs if isinstance(step.inputs, dict) else {}
    return [
        (("inputs", role), named) for role, named in inputs.items() if isinstance(named, VersionRef)
    ]


def _continuation_params(params: Mapping[str, object]) -> bool:
    """Whether `params` state a continuation: `method`, a non-empty string, and `periods`, a
    list of two strings."""
    match params.get("method"), params.get("periods"):
        case str(method), [str(), str()] if method:
            return True
        case _:
            return False


def _kit_refusal(error: StepsError, steps: list[Step]) -> ValueError | ValidationError:
    """Restates the kit's structure refusal. The byte bound is refused at `steps`, and a value
    JSON cannot carry at `steps` with the kit's reason. Any other rule on the whole list, such
    as the kit's count bound, which `MAX_STEPS` keeps out of reach, is refused at `steps` under
    the kit's reason. A step's refusal is located under `steps` at the member it names: the
    repeated id, a compare's missing or mismatched `sides`, or the first input that names no
    earlier step."""
    if error.reason is StepsErrorReason.NOT_I_JSON:
        return _step_refusal(
            "not_i_json", f"steps hold a value JSON cannot carry: {error.detail}", loc=(), posted=""
        )
    if error.reason is StepsErrorReason.STEPS_TOO_LARGE:
        return ValueError(f"steps exceed MAX_STEPS_BYTES={MAX_STEPS_BYTES}: {error.detail}")
    if error.step is None:
        return ValueError(f"steps break dfo-spec-kit's {error.reason} rule: {error.detail}")
    step = steps[error.step]
    if error.reason is StepsErrorReason.DUPLICATE_STEP_ID:
        return _step_refusal(
            "duplicate_step_id",
            f"step id {step.id!r} is already taken by an earlier step",
            loc=(error.step, "id"),
            posted=step.id,
        )
    if error.reason is StepsErrorReason.MISSING_SIDES:
        return _step_refusal(
            "missing_sides",
            f"{step.op} step {step.id!r} records no sides, the entities each input holds",
            loc=(error.step, "sides"),
            posted=step.id,
        )
    if error.reason is StepsErrorReason.SIDES_ROLES_MISMATCH:
        return _step_refusal(
            "sides_roles_mismatch",
            f"sides of step {step.id!r} are keyed by other roles than its inputs",
            loc=(error.step, "sides"),
            posted=step.id,
        )
    earlier = {earlier_step.id for earlier_step in steps[: error.step]}
    loc, named = next((loc, named) for loc, named in _named_steps(step) if named not in earlier)
    if error.reason is StepsErrorReason.INPUT_NOT_EARLIER:
        return _step_refusal(
            "input_not_earlier",
            f"input names step {named!r}, which is this step or a later one",
            loc=(error.step, *loc),
            posted=named,
        )
    return _step_refusal(
        "unknown_input",
        f"input names step {named!r}, which no step holds",
        loc=(error.step, *loc),
        posted=named,
    )


def _step_refusal(
    reason: Literal[
        "duplicate_step_id",
        "unknown_input",
        "input_not_earlier",
        "missing_sides",
        "sides_roles_mismatch",
        "version_not_retrieved",
        "no_version_retrieved",
        "not_i_json",
        "params_invalid",
    ],
    message: str,
    *,
    loc: tuple[int | str, ...],
    posted: str,
) -> ValidationError:
    """A refusal of one step's member, located under `steps` so its pointer names the member.
    `reason` is the dfo-spec-kit structure rule the step breaks, or one of this door's three
    rules: `version_not_retrieved`, that only a `retrieve` step's `input` names a version,
    `no_version_retrieved`, that the steps name at least one, and `params_invalid`, that a
    `continue_series` step's `params` state its method and periods (`_continuation_params`)."""
    return ValidationError.from_exception_data(
        "steps",
        [
            InitErrorDetails(
                type=PydanticCustomError(reason, "{message}", {"message": message}),
                loc=loc,
                input=posted,
            )
        ],
    )


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

    steps: list[Step] | None = Field(
        default=None,
        description=(
            "the frame's step records, one per operation that computed it, in order, in the "
            f"shape of dfo-spec-kit's step schema. `op` is any operation name. At most {MAX_STEPS} "
            f"steps and {MAX_STEPS_BYTES} bytes of canonical JSON; step ids are distinct, an "
            "input that names a step names an earlier one, and only a `retrieve` step's "
            "`input` names a version, which at least one does. The versions `retrieve` steps "
            "name are the frame's root inputs, which "
            "`input_content_versions`, the immediate input frames, need not list. Absent when "
            "the publish carries none"
        ),
    )

    @field_validator("steps", mode="before")
    @classmethod
    def _bound_step_count(cls, value: object) -> object:
        """Refuses more than `MAX_STEPS` steps before any step is parsed."""
        if isinstance(value, list) and len(value) > MAX_STEPS:
            raise ValueError(f"steps holds {len(value)} steps, exceeding MAX_STEPS={MAX_STEPS}")
        return value

    @field_validator("steps", mode="after")
    @classmethod
    def _check_step_structure(cls, value: list[Step] | None) -> list[Step] | None:
        """Refuses steps that break dfo-spec-kit's structure rules
        (`dfo_descriptions.check_steps`) in their stored form: a value JSON
        cannot carry (a lone surrogate, a non-finite number or an integer past 2**53), more than
        `MAX_STEPS_BYTES` of RFC 8785 canonical JSON, or the first step whose id repeats an
        earlier one, whose input names neither an earlier step nor a version, or, for a compare,
        whose `sides` are missing or keyed by other roles than its inputs, at a pointer to that
        id, input or `sides`."""
        if value is None:
            return value
        try:
            check_steps([step.model_dump(mode="json") for step in value])
        except StepsError as error:
            raise _kit_refusal(error, value) from error
        return value

    @field_validator("steps", mode="after")
    @classmethod
    def _check_version_refs_are_retrieved(cls, value: list[Step] | None) -> list[Step] | None:
        """Refuses the first version reference anywhere but a `retrieve` step's `input`, at a
        pointer to it, then steps that retrieve no version, at `steps`. Every version the steps
        name is then one the record's inputs hold, and the record holds at least one input."""
        if value is None:
            return value
        retrieved = False
        for position, step in enumerate(value):
            for loc, ref in _version_refs(step):
                if step.op == "retrieve" and loc == ("input",):
                    retrieved = True
                    continue
                raise _step_refusal(
                    "version_not_retrieved",
                    f"a {step.op} step names version {ref.id!r} under {'/'.join(loc)}; only a "
                    "retrieve step's input names a version",
                    loc=(position, *loc),
                    posted=ref.content_version,
                )
        if not retrieved:
            raise _step_refusal(
                "no_version_retrieved",
                "steps retrieve no version; a computed frame's steps hold at least one retrieve "
                "step naming a version in its input",
                loc=(),
                posted="",
            )
        return value

    @field_validator("steps", mode="after")
    @classmethod
    def _check_continuation_params(cls, value: list[Step] | None) -> list[Step] | None:
        """Refuses the first `continue_series` step whose `params` do not state a continuation
        (`_continuation_params`), at a pointer to its `params`."""
        for position, step in enumerate(value or ()):
            if step.op != CONTINUE_SERIES_OP:
                continue
            params = step.params if isinstance(step.params, dict) else None
            if params is None or not _continuation_params(params):
                raise _step_refusal(
                    "params_invalid",
                    f"a {CONTINUE_SERIES_OP} step's params hold 'method', a non-empty string, "
                    "and 'periods', a list of two strings, the first and last period the "
                    "continuing series supplies",
                    loc=(position, "params"),
                    posted="",
                )
        return value

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
