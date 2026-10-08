"""Wire models for dataset mint inputs, scope, and rendering options.

The published mint schema and runtime validation use these same model definitions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationInfo,
    field_validator,
    model_validator,
)

from datasets_api.models import CanonicalDefinition, _require_schema_version_int
from dfo_platform.named import cut, named_capped

FACET_AXES: tuple[str, ...] = ("provenance", "measurementMethod", "unit")
NULL_REF_FACET_AXES: tuple[str, ...] = ("measurementMethod", "unit")
"""Axes whose ref binding may be null, as in the identity core: every facet carries a
provenance, so a null provenance ref names nothing."""

# Mirrors the SDMX client: `+` separates AND filters and `,` joins OR values.
# The client hardcodes both characters, so invalid values are rejected at this boundary.
_SDMX_RESERVED_VALUE_CHARS: tuple[str, ...] = ("+", ",")
# The deepest `groupby.levels` depth. A depth past a code's parent chain leaves the code as
# itself, so this refuses only depths no code hierarchy reaches, and it keeps a query depth
# far inside Python's integer string conversion limit.
MAX_GROUPBY_LEVEL_DEPTH = 99


class WindowAll(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["all"] = "all"


class WindowLatest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["latest"] = "latest"


class WindowRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["range"] = "range"
    start_year: int | None = None
    end_year: int | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> WindowRange:
        if self.start_year is None and self.end_year is None:
            raise ValueError("a range window requires start_year, end_year, or both")
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must not be greater than end_year")
        return self


WindowScope = Annotated[WindowAll | WindowLatest | WindowRange, Field(discriminator="kind")]


class FacetRefBinding(BaseModel):
    """A facet axis bound to a specific ref. Every axis accepts this form;
    the unit axis alone also accepts `FacetUnitRequestBinding`, the
    conversion-request form (`currency`/`prices`/`base_year`). A null `ref`
    binds the facets that declare no value on the axis, and only
    `NULL_REF_FACET_AXES` accept it.
    """

    model_config = ConfigDict(extra="forbid")
    ref: str | None


class UnitCurrencyRole(BaseModel):
    """A unit-request `currency` naming a spec role rather than a concrete
    ISO 4217 code: convert to the currency of whichever entity is bound in
    that role. The role must be one the resolved StatVar declares.
    """

    model_config = ConfigDict(extra="forbid")
    role: Literal["origin", "destination", "observationAbout"]


class UnitRequestBindingBase(BaseModel):
    """The `currency`/`prices` members every unit-request form shares: an
    upper-case ISO 4217 code or a role shape (`UnitCurrencyRole`), resolved
    per row to that entity's current currency, and the price basis each
    concrete form below narrows to its own `Literal`.
    """

    model_config = ConfigDict(extra="forbid")
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | UnitCurrencyRole
    prices: Literal["current", "constant"]


class UnitCurrentPricesRequestBinding(UnitRequestBindingBase):
    """The unit axis's own conversion-request form for `prices: "current"`:
    no `base_year` member.
    """

    prices: Literal["current"] = "current"


class UnitConstantPricesRequestBinding(UnitRequestBindingBase):
    """The unit axis's own conversion-request form for `prices: "constant"`:
    `base_year` is a required member.
    """

    prices: Literal["constant"] = "constant"
    base_year: int


FacetUnitRequestBinding = Annotated[
    UnitCurrentPricesRequestBinding | UnitConstantPricesRequestBinding,
    Field(discriminator="prices"),
]


def _facet_binding_discriminator(value: object) -> str:
    """Tags a value as `ref` when it has that key, otherwise `request`.

    Shape-based tagging preserves the relevant validation error for malformed bindings.
    """
    if isinstance(value, Mapping):
        return "ref" if "ref" in value else "request"
    return "ref" if isinstance(value, FacetRefBinding) else "request"


FacetBinding = Annotated[
    Annotated[FacetRefBinding, Tag("ref")] | Annotated[FacetUnitRequestBinding, Tag("request")],
    Discriminator(_facet_binding_discriminator),
]


class Scope(BaseModel):
    """The `scope` block a mint request may carry: an omitted member means
    unbound (everything). `window` narrows the result to `all`, `latest`,
    or a `start_year`/`end_year` range. `entities` binds role names to
    non-empty lists of dcids (a role's values must each be non-empty and
    must not contain `+` or `,`). `facets` binds a known axis
    (`provenance`, `measurementMethod`, `unit`) to a specific ref, or
    `measurementMethod` or `unit` to a null ref. The `unit` axis alone may
    instead carry a conversion request (`FacetUnitRequestBinding`).
    """

    # Validate roles and values against the SDMX client's filter rules before resolution. Empty
    # dcids are rejected here because the client allows them through to HTTP and no dcid grammar
    # accepts them. Identity-core `entities` allows only dcid arrays, so containment bindings fail.
    model_config = ConfigDict(extra="forbid")
    window: WindowScope | None = None
    entities: dict[str, list[str]] | None = None
    facets: dict[str, FacetBinding] | None = None

    @model_validator(mode="after")
    def _entities_are_valid(self) -> Scope:
        if self.entities is None:
            return self
        for role, refs in self.entities.items():
            if not role or not all(ch.isalnum() or ch == "_" for ch in role):
                raise ValueError(f"invalid entity role {cut(role)!r}")
            if not refs:
                raise ValueError(f"entity role {cut(role)!r} has no values")
            for ref in refs:
                if not ref:
                    raise ValueError(f"entity role {cut(role)!r} has an empty value")
                for char in _SDMX_RESERVED_VALUE_CHARS:
                    if char in ref:
                        raise ValueError(
                            f"entity role {cut(role)!r} has a value containing {char!r}: {cut(ref)!r}"
                        )
        return self

    @model_validator(mode="after")
    def _facets_axes_known(self) -> Scope:
        if self.facets is not None:
            unknown = set(self.facets) - set(FACET_AXES)
            if unknown:
                raise ValueError(f"unknown facet axis/axes: {named_capped(sorted(unknown))}")
        return self

    @model_validator(mode="after")
    def _request_form_only_on_unit(self) -> Scope:
        if self.facets is not None:
            for axis, binding in self.facets.items():
                if axis != "unit" and isinstance(binding, UnitRequestBindingBase):
                    raise ValueError(
                        f"facet axis {axis!r} accepts only the ref-binding form, not a "
                        "conversion request"
                    )
        return self

    @model_validator(mode="after")
    def _null_ref_only_on_method_or_unit(self) -> Scope:
        if self.facets is not None:
            for axis, binding in self.facets.items():
                if (
                    isinstance(binding, FacetRefBinding)
                    and binding.ref is None
                    and axis not in NULL_REF_FACET_AXES
                ):
                    raise ValueError(
                        "a null ref is only valid on the measurementMethod and unit axes "
                        f"(got {axis!r})"
                    )
        return self


class GroupbyOptions(BaseModel):
    """`options.groupby`: a safe roll-up that keeps exactly the axes named
    in `by`, one row per distinct combination of their values. Every other
    graph-declared summable axis a frame decoded is summed away, `stat_var`
    included. `by` names what a caller wants to see, the opposite of
    `groupby_sum`'s own "collapse this axis" reading. `sum` is the only op
    this closed reshape list allows.

    Every `by` entry names a code system some deployment declaration marks
    summable, a role/entity column (`materialise.roles.KNOWN_ROLE_NAMES`),
    `stat_var`, or a column every frame carries regardless of shape; any
    other name refuses `rollup_not_summable`
    (`datasets_api.groupby.refuse_unknown_by_axes`). A declared axis this
    frame's StatVars do not decode is left out of the collapse set
    (`datasets_api.groupby.collapse_axes`).

    `levels` rolls a kept hierarchical axis up to one depth of its parent
    chain before the sum: each code becomes its ancestor at that depth, and
    depth 0 is the root codes. Each key must also appear in `by`, and must
    be one of the capabilities document's `groupby_level_axes`
    (`datasets_api.render.check_options_supported`).
    """

    model_config = ConfigDict(extra="forbid")
    by: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, json_schema_extra={"uniqueItems": True}
    )
    op: Literal["sum"]
    levels: (
        dict[str, Annotated[int, Field(strict=True, ge=0, le=MAX_GROUPBY_LEVEL_DEPTH)]] | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "code system -> depth: roll each kept code up to its ancestor at that depth "
            "(0 = root codes); a code shallower than the depth stays as itself, under its "
            "own label"
        ),
    )

    @field_validator("by")
    @classmethod
    def _by_is_unique(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        duplicates = {name for name in value if name in seen or seen.add(name)}
        if duplicates:
            raise ValueError(f"duplicate groupby axis name(s): {named_capped(sorted(duplicates))}")
        return value

    @field_validator("levels")
    @classmethod
    def _levels_are_kept(
        cls, value: dict[str, int] | None, info: ValidationInfo
    ) -> dict[str, int] | None:
        by = info.data.get("by")
        if value is not None and by is not None:
            outside = sorted(set(value) - set(by))
            if outside:
                raise ValueError(
                    f"groupby.levels names axis(es) {named_capped(outside)} absent from by"
                )
        return value


class PivotOptions(BaseModel):
    """`options.pivot`: long-to-wide, `index` and `columns` each naming
    exactly one column.
    """

    model_config = ConfigDict(extra="forbid")
    index: str = Field(min_length=1)
    columns: str = Field(min_length=1)


class MintOptions(BaseModel):
    """Rendering-only, never hashed into identity. `format`/`bundle`/`pivot`/
    `columns`/`groupby` are this deployment's live rendering surface. A
    `groupby.by` entry names a code system some deployment declaration
    marks summable, a role/entity column, `stat_var`, or a column every
    frame carries regardless of shape; any other name refuses
    `rollup_not_summable` (`datasets_api.groupby.refuse_unknown_by_axes`).
    A declared axis this frame did not decode is simply absent from the
    collapse-set computation, not an error (see
    `datasets_api.groupby.collapse_axes`); an unsafe collapse refuses
    `rollup_not_summable` too, naming the axis and precondition
    (`materialise.reshape.groupby_sum`,
    `datasets_api.groupby.refuse_stat_var_collision`,
    `datasets_api.groupby.refuse_decode_gaps`). A `columns` entry naming a
    recognised-but-disabled enrichment field refuses `capability_unavailable`.
    `columns` selects the table and decoded columns it names. A list that names
    only enrichment fields selects nothing: every default column stays,
    decoded columns included, and the enrichments are added. A list that mixes
    a table column with an enrichment selects. Without a pivot, a list that
    names no decoded column keeps all of them, `status` is accepted and left
    out, `<role>_label` is written by default (a list naming it keeps
    `<role>`), and the written order follows the long layout
    (`materialise.projection`) whatever order the list names.

    `layout: "frame"` writes a bare parquet file in the frame layout instead
    of the long layout: the frame's grammar columns as reshaped, with
    decoded axes as `stat_var_<code_system>`, `status` kept, and a converted
    unit as its JCS request text. `columns` then selects over those columns,
    as it does for pivot output. It refuses with `bundle: true`, a `pivot`,
    or a format other than parquet.
    """

    # Disabled enrichment names refuse only when the frame lacks a real column with that name.
    model_config = ConfigDict(extra="forbid")
    format: Literal["csv", "xlsx", "parquet"] = "csv"
    bundle: bool = True
    groupby: GroupbyOptions | None = None
    pivot: PivotOptions | None = None
    columns: list[Annotated[str, Field(min_length=1)]] | None = Field(
        default=None, json_schema_extra={"uniqueItems": True}
    )

    @field_validator("columns")
    @classmethod
    def _columns_are_unique(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            seen: set[str] = set()
            duplicates = {name for name in value if name in seen or seen.add(name)}
            if duplicates:
                raise ValueError(f"duplicate column name(s): {named_capped(sorted(duplicates))}")
        return value

    # Declared last, so the validator sees every other field.
    layout: Literal["frame"] | None = None

    @field_validator("layout")
    @classmethod
    def _frame_layout_is_bare_parquet(
        cls, value: Literal["frame"] | None, info: ValidationInfo
    ) -> Literal["frame"] | None:
        if value is None:
            return value
        if info.data.get("bundle") is not False:
            raise ValueError('layout "frame" needs bundle: false; a bundle has its own layout')
        if info.data.get("format") != "parquet":
            raise ValueError('layout "frame" needs format: "parquet"')
        if info.data.get("pivot") is not None:
            raise ValueError('layout "frame" cannot be pivoted; a pivot is not a frame')
        return value


class StatVarSetInput(BaseModel):
    """`kind: "statvar_set"` -- resolves one `dcid:`-prefixed StatVar to its
    registered shape and mints a fully pinned spec for it. `statvars` holds
    exactly one StatVar.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["statvar_set"] = "statvar_set"
    statvars: list[Annotated[str, Field(pattern=r"^dcid:")]] = Field(min_length=1, max_length=1)
    scope: Scope | None = None


class SpecIdInput(BaseModel):
    """`kind: "spec_id"` -- an id already in the registry, with an optional
    canonical `definition` sidecar. `definition.spec_id`, when supplied,
    must equal the outer `spec_id`; the sidecar is required only when the
    registry does not already hold `spec_id`.
    """

    # The mint door handles this kind. A supplied identity must hash to `spec_id` on hits and
    # misses.
    model_config = ConfigDict(extra="forbid")
    kind: Literal["spec_id"] = "spec_id"
    spec_id: str = Field(pattern=r"^spec_")
    definition: CanonicalDefinition | None = None
    scope: Scope | None = None

    @model_validator(mode="after")
    def _sidecar_spec_id_matches_outer(self) -> SpecIdInput:
        if self.definition is not None and self.definition.spec_id != self.spec_id:
            raise ValueError(
                f"definition.spec_id {cut(self.definition.spec_id)!r} does not match "
                f"the outer spec_id {cut(self.spec_id)!r}"
            )
        return self


MintInput = Annotated[SpecIdInput | StatVarSetInput, Field(discriminator="kind")]


class DefinitionOptionsRequest(BaseModel):
    """Body of `POST /api/datasets/{id}/options`. It carries the canonical `definition` a search
    response gives for `id`, so the options route answers before any mint registers it.

    The route admits the definition as a mint admits a `spec_id` sidecar and writes nothing.
    `definition.spec_id` must equal the path id, and the identity must hash to it.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    definition: CanonicalDefinition


class DefinitionMembersRequest(BaseModel):
    """Body of `POST /api/datasets/{id}/members`. It may carry the canonical `definition` a
    search response gives for `id`, so the route answers before any mint registers it.

    A registered id needs no `definition`. A supplied one is admitted as a mint admits a
    `spec_id` sidecar: `definition.spec_id` must equal the path id, and the identity must hash
    to it. Nothing is written.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    definition: CanonicalDefinition | None = None


class MintRequest(BaseModel):
    """Body of `POST /api/datasets`: one `input` and the render `options`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    input: MintInput
    options: MintOptions = Field(default_factory=MintOptions)
