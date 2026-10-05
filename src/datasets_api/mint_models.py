"""Wire models for dataset mint inputs, scope, rendering options, and write-free previews.

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
    field_validator,
    model_validator,
)

from datasets_api.models import (
    MAX_COLLECTION_MEMBERS,
    CanonicalDefinition,
    _require_schema_version_int,
)
from statvar_members import POSTED_MEMBER_CAP

FACET_AXES: tuple[str, ...] = ("provenance", "measurementMethod", "unit")
NULL_REF_FACET_AXES: tuple[str, ...] = ("measurementMethod", "unit")
"""Axes whose ref binding may be null, as in the identity core: every facet carries a
provenance, so a null provenance ref names nothing."""

# Mirrors the SDMX client: `+` separates AND filters and `,` joins OR values.
# The client hardcodes both characters, so invalid values are rejected at this boundary.
_SDMX_RESERVED_VALUE_CHARS: tuple[str, ...] = ("+", ",")


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
                raise ValueError(f"invalid entity role {role!r}")
            if not refs:
                raise ValueError(f"entity role {role!r} has no values")
            for ref in refs:
                if not ref:
                    raise ValueError(f"entity role {role!r} has an empty value")
                for char in _SDMX_RESERVED_VALUE_CHARS:
                    if char in ref:
                        raise ValueError(
                            f"entity role {role!r} has a value containing {char!r}: {ref!r}"
                        )
        return self

    @model_validator(mode="after")
    def _facets_axes_known(self) -> Scope:
        if self.facets is not None:
            unknown = set(self.facets) - set(FACET_AXES)
            if unknown:
                raise ValueError(f"unknown facet axis/axes: {sorted(unknown)!r}")
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
    """

    model_config = ConfigDict(extra="forbid")
    by: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, json_schema_extra={"uniqueItems": True}
    )
    op: Literal["sum"]

    @field_validator("by")
    @classmethod
    def _by_is_unique(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        duplicates = {name for name in value if name in seen or seen.add(name)}
        if duplicates:
            raise ValueError(f"duplicate groupby axis name(s): {sorted(duplicates)!r}")
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
    a table column with an enrichment selects.
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
                raise ValueError(f"duplicate column name(s): {sorted(duplicates)!r}")
        return value


class StatVarSetInput(BaseModel):
    """`kind: "statvar_set"` -- resolves each `dcid:`-prefixed StatVar to its
    registered shape and mints one fully pinned member spec per StatVar.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["statvar_set"] = "statvar_set"
    statvars: list[Annotated[str, Field(pattern=r"^dcid:")]] = Field(
        min_length=1, max_length=MAX_COLLECTION_MEMBERS
    )
    scope: Scope | None = None


class SourceCodesInput(BaseModel):
    """`kind: "source_codes"` -- resolves codes to StatVars (scoped by
    `source` when supplied), then behaves as `statvar_set`.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["source_codes"] = "source_codes"
    codes: list[str] = Field(min_length=1, max_length=MAX_COLLECTION_MEMBERS)
    source: str | None = None
    scope: Scope | None = None

    @field_validator("codes")
    @classmethod
    def _codes_are_not_dcid_prefixed(cls, value: list[str]) -> list[str]:
        for code in value:
            if code.startswith("dcid:"):
                raise ValueError(f"code {code!r} must not be dcid:-prefixed")
        return value


def _require_bare_dcids(value: list[str] | None) -> list[str] | None:
    """Refuses a `resolved_stat_var_dcids` entry carrying the `dcid:` prefix."""
    for dcid in value or ():
        if dcid.startswith("dcid:"):
            raise ValueError(
                f"resolved_stat_var_dcids entry {dcid!r} must be a bare dcid, "
                "without the dcid: prefix"
            )
    return value


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
    resolved_stat_var_dcids: list[str] | None = Field(
        default=None,
        max_length=POSTED_MEMBER_CAP,
        description=(
            "bare StatVar dcids the caller holds as members of the identity, typically "
            "resolved.stat_vars.items[].ref from a search resolve. An assertion the mint "
            "and the preview check: after member resolution every posted dcid must be a "
            "member of the identity's family, or the request refuses member_not_resolvable "
            "(422) with one issue per posted non-member, pointing at "
            "/input/resolved_stat_var_dcids/<index>. A refused mint writes no registry "
            "row. The family is every StatVar consistent with the identity, including one "
            "a broader member dominates. The list does not narrow the dataset: the mint "
            "serves the family's antichain-minimal members, posted or omitted, so a "
            "posted dominated member is accepted and not served. An accepted list "
            "changes neither spec_id nor content_version, "
            "and an empty list behaves as an omitted field. The check covers every arm, "
            "so a pinned arm (kind: stat_var) accepts only its own one StatVar. The list "
            "also joins the graph's reverse-arc index walk as extra candidates "
            "(statvar_members.enumerate_members's extra_candidates), so a posted root, a "
            "member constrained by exactly the identity's single-code slots (state exact, "
            "or state value on a code with no descendants) that declares only the roles "
            "its entities are bound under, can prove complete a family "
            "too broad for that index to page through, unless a slot is bound to a set of "
            "codes (state set, or state value on a hierarchical code with descendants) or "
            "an unbound slot's code system reads absence as not_declared. "
            f"Capped at {POSTED_MEMBER_CAP} (statvar_members.POSTED_MEMBER_CAP), the "
            "ceiling search's resolved_stat_var_dcids shares, so a search response's member "
            "list posts whole. A sibling of definition and "
            "scope: the list is graph state and stays outside the identity hash"
        ),
    )

    _validate_resolved_stat_var_dcids = field_validator("resolved_stat_var_dcids")(
        _require_bare_dcids
    )

    @model_validator(mode="after")
    def _sidecar_spec_id_matches_outer(self) -> SpecIdInput:
        if self.definition is not None and self.definition.spec_id != self.spec_id:
            raise ValueError(
                f"definition.spec_id {self.definition.spec_id!r} does not match "
                f"the outer spec_id {self.spec_id!r}"
            )
        return self


class CollectionIdInput(BaseModel):
    """`kind: "collection_id"` -- a collection id already in the registry,
    with an optional `members` sidecar, each verified to hash to its own
    spec_id. `members`, when supplied, is a non-empty list, one per
    collection member; required only when the registry does not already
    hold `collection_id`. A `scope` block applies scope surgery to every
    member independently.
    """

    # An empty `members` list means no sidecar, so reject it at the wire boundary.
    model_config = ConfigDict(extra="forbid")
    kind: Literal["collection_id"] = "collection_id"
    collection_id: str = Field(pattern=r"^col_")
    members: list[CanonicalDefinition] | None = Field(
        default=None, min_length=1, max_length=MAX_COLLECTION_MEMBERS
    )
    scope: Scope | None = None


MintInput = Annotated[
    SpecIdInput | CollectionIdInput | StatVarSetInput | SourceCodesInput,
    Field(discriminator="kind"),
]


class PreviewRequest(BaseModel):
    """Set on `MintRequest.preview` to ask for a write-free preview of
    `input` instead of a mint: `POST /api/datasets` answers with a
    `datasets_api.preview_models.PreviewEnvelope` (never a `Manifest`) and
    touches no GCS object, no registry row, no version record --
    `datasets_api.preview`'s own module docstring covers what stays
    write-free and what a preview still costs.

    `limit` bounds the row sample each `PreviewEnvelope` member carries
    (`materialise.materialise_preview`'s own `limit` parameter). A preview
    reads every row a mint of the same request reads, so its `total_rows`
    and `content_version` cover every survivor, and it is admitted as an
    inline mint is. The upper bound here (1000) caps the response size.
    """

    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=100, ge=1, le=1000)


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
    resolved_stat_var_dcids: list[str] | None = Field(
        default=None,
        max_length=POSTED_MEMBER_CAP,
        description=(
            "bare StatVar dcids the caller holds as members of the identity, as on a "
            "spec_id mint input. For an unregistered definition they seed member "
            "resolution and each must be a member of the identity's family, or the request "
            "refuses member_not_resolvable (422) pointing at "
            "/resolved_stat_var_dcids/<index>. A registered id resolves as "
            "GET /api/datasets/{id}/options does and ignores the list"
        ),
    )
    _validate_resolved_stat_var_dcids = field_validator("resolved_stat_var_dcids")(
        _require_bare_dcids
    )


class MintRequest(BaseModel):
    """`options` is ignored when `preview` is set: a preview never renders
    or reshapes, so nothing under `options` has an effect on its response.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    input: MintInput
    options: MintOptions = Field(default_factory=MintOptions)
    preview: PreviewRequest | None = None
