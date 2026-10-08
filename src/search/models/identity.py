from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    Discriminator,
    Field,
    GetJsonSchemaHandler,
    SerializerFunctionWrapHandler,
    StrictInt,
    Tag,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError

from dfo_platform.named import named_capped

from .base import StrictModel
from .vocab import _FACET_AXIS_SCHEMA_VERSION, SCHEMA_VERSION, RoleAxis

# ---------------------------------------------------------------------------
# BaseIdentity mirrors the identity core's wire schema. It is the document
# ``build_spec_identity`` hashes and returns verbatim, separate from the richer
# engine response models.
# ---------------------------------------------------------------------------

# Mirror the identity core's role and facet-axis sets so unsupported names
# fail Pydantic validation. Keep every known axis here; BaseIdentity's
# validator distinguishes published axes from those requiring a newer schema version.
_IDENTITY_FACET_AXES = frozenset(_FACET_AXIS_SCHEMA_VERSION)
_IDENTITY_ENTITY_ROLE_KEYS = frozenset({"observationAbout", "origin", "destination"})

_MAX_KEY_CHARS = 32

_IDENTITY_MAX_SAFE_INTEGER = 2**53 - 1
"""JCS/I-JSON numbers represent integers exactly through ``2**53 - 1`` (IEEE-754 double)."""

IdentityString = Annotated[str, Field(max_length=1024)]
"""Identity string capped at 1,024 code points to match the identity core.

Empty strings remain valid because the core accepts them.
"""


class IdentityBindingValue(StrictModel):
    """A single bound dcid, in the identity core's own binding wire shape."""

    state: Literal["value"] = "value"
    ref: IdentityString = Field(description="the bound dcid")


class IdentityBindingSet(StrictModel):
    """A bound dcid set, in the identity core's own binding wire shape."""

    state: Literal["set"] = "set"
    refs: list[IdentityString] = Field(
        min_length=1,
        max_length=1024,
        description="the bound dcids (dfo_spec_kit sorts and de-duplicates)",
    )


class IdentityBindingUnbound(StrictModel):
    """An unbound slot/facet binding."""

    state: Literal["unbound"] = "unbound"


class IdentityBindingExact(StrictModel):
    """A single bound dcid, denoting that code alone -- never its subtree,
    even where the code is a hierarchical (non-leaf) code (see
    `docs/identity-v1.md` in dfo-spec-kit for the denotation rule: `value`
    denotes a hierarchical code's whole subtree, `exact` denotes the code
    alone). Slot-only: `_build_facet_binding` refuses `exact` on a facet.
    No engine `Binding` arm maps onto this state and no minting site
    produces it -- see `IdentityBinding`'s own docstring."""

    state: Literal["exact"] = "exact"
    ref: IdentityString = Field(description="the bound dcid, denoting that code alone")


IdentityBinding = Annotated[
    IdentityBindingValue | IdentityBindingSet | IdentityBindingUnbound | IdentityBindingExact,
    Field(discriminator="state"),
]
"""Identity-core slot binding union. The ``value``, ``set``, and ``unbound`` arms map to
engine bindings; ``exact`` is accepted only by dataset doors and has no Search minting path.
"""


_CURRENCY_PATTERN = r"^[A-Z]{3}$"

# Currency requests may name only roles that BaseIdentity can declare.
# The resolved shape is checked downstream for the named role.
_CURRENCY_ROLES = frozenset({"observationAbout", "origin", "destination"})


class IdentityCurrencyRole(StrictModel):
    """The role-shaped form of a unit request's `currency`: the currency of
    the entity bound to this spec role, resolved per row. Field-for-field mirror
    of the identity core's own `_CURRENCY_ROLE_KEYS`/`_CURRENCY_ROLES`
    (`dfo_spec_kit.identity`'s `_build_currency`, the object arm)."""

    role: Literal["observationAbout", "origin", "destination"] = Field(
        description="the entity role whose bound entity's currency this request means"
    )


IdentityCurrency = Annotated[
    Annotated[str, Field(pattern=_CURRENCY_PATTERN)] | IdentityCurrencyRole,
    Field(
        description=(
            "the requested currency: upper-case ISO 4217 (e.g. 'USD'), or "
            "{'role': ...} naming the spec role whose bound entity's currency to use"
        )
    ),
]
"""Upper-case ISO 4217 code or a role object; the JSON string and object forms are
unambiguous without a discriminator."""


class IdentityUnitRequest(StrictModel):
    """A completed unit-conversion request, in the identity core's own wire
    shape (`dfo_spec_kit.identity`'s `_build_unit_request`) -- legal only
    inside a unit-axis facet binding's `request` member
    (`IdentityFacetValueRequest`; see `BaseIdentity`'s model validator, which
    refuses this form on any other axis).

    Field-for-field mirror of the identity core's own `_UNIT_REQUEST_KEYS`
    schema: `currency` is upper-case ISO 4217 or a role object naming
    the spec role whose bound entity's currency to use (see
    `IdentityCurrency`; the identity core NFC-normalises a string spelling
    but does not case-fold, so a lower-case currency refuses here rather
    than silently normalising to a different meaning at hash time), `prices`
    is current (nominal) or constant (real, anchored to
    `base_year`), and `base_year` is present exactly when `prices=="constant"`
    -- both directions enforced below, matching the identity core's own
    `invalid_unit_request` pair of checks. `base_year` is `StrictInt`,
    matching `_require_integral_number`'s own exclusion of `bool` (a Python
    `int` subclass) and of numeric strings -- a lax `int` field would coerce
    JSON `true` to `1` and `"2023"` to `2023`, hashing a malformed request
    under an unintended base year instead of refusing it. Under simultaneous
    violations
    (e.g. a lower-case currency together with a missing base_year) the two
    layers can report a different invariant first: this model's per-field
    `currency` pattern validates ahead of `_base_year_matches_prices` (a
    `mode="after"` validator, which never runs once a field has already
    failed), while the identity core checks `prices`-vs-`base_year` before
    the currency regex. Same HTTP status/code either way -- message content
    only.
    """

    currency: IdentityCurrency = Field(
        description="the requested currency, upper-case ISO 4217 (e.g. 'USD'), or a role object"
    )
    prices: Literal["current", "constant"] = Field(
        description="'current' (nominal) prices, or 'constant' (real, base_year-anchored) prices"
    )
    base_year: StrictInt | None = Field(
        default=None,
        ge=-_IDENTITY_MAX_SAFE_INTEGER,
        le=_IDENTITY_MAX_SAFE_INTEGER,
        description="required exactly when prices=='constant'; absent when prices=='current'",
    )

    @model_validator(mode="after")
    def _base_year_matches_prices(self) -> IdentityUnitRequest:
        if self.prices == "constant" and self.base_year is None:
            raise ValueError("prices='constant' requires base_year")
        if self.prices == "current" and self.base_year is not None:
            raise ValueError("base_year is only valid when prices='constant'")
        return self


class IdentityFacetValueRequest(StrictModel):
    """A unit-axis facet bound via the request form: "serve whatever facet
    honours this completed currency/prices/base_year request" rather than a
    literal graph ref. `state` reuses the value-state spelling since this is
    still a single-valued binding to the identity core -- `request` is what
    distinguishes it from `IdentityBindingValue`'s `ref` form (see
    `_facet_binding_tag`, this union's discriminator)."""

    state: Literal["value"] = "value"
    request: IdentityUnitRequest = Field(description="the completed unit-conversion request")


NULL_REF_FACET_AXES = frozenset({"measurementMethod", "unit"})
"""Facet axes an `IdentityFacetValueAbsent` binding may name, mirroring the identity core's
null facet ref. Every facet carries a provenance, so a null provenance ref names nothing."""


class IdentityFacetValueAbsent(StrictModel):
    """A measurementMethod or unit facet bound to no declared value:
    `{"state": "value", "ref": null}` asks for the facets that declare no value
    on the axis, the ones Data Commons serves without that member. An unbound
    axis admits every facet; this binding admits only those without a value.
    Mirrors the identity core's null facet ref (`docs/identity-v1.md` in
    dfo-spec-kit). `ref` serialises as `null` even under `exclude_none`, since
    the identity core refuses a value binding with neither `ref` nor `request`."""

    state: Literal["value"] = "value"
    ref: None = Field(description="null: the facet declares no value on this axis")

    @model_serializer(mode="wrap")
    def _keep_null_ref(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        data["ref"] = None
        return data

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # The serializer only restores the `ref` that `exclude_none` strips, so the
        # validation schema describes the output too, and BaseIdentity keeps one
        # OpenAPI component rather than an -Input/-Output pair.
        if handler.mode == "serialization":
            return cls.model_json_schema(mode="validation")
        return handler(core_schema)


def _facet_binding_tag(value: Any) -> str | None:
    """Tag ``set`` by its state and distinguish ``ref`` from ``request`` under ``value``.

    Inspect raw data to keep malformed-input errors at the union's JSON Pointer. Smart-union
    resolution leaked the model name into ``loc`` (6 errors instead of 1). Unrecognised forms
    return ``None`` and produce one ``union_tag_not_found`` error at the union location.
    """
    if isinstance(value, dict):
        state = value.get("state")
        if state == "set":
            return "set"
        if state == "value":
            if "ref" in value and "request" in value:
                # Both present: report the cleaner tag-not-found shape
                # rather than tagging unit_request and letting
                # IdentityFacetValueRequest's extra="forbid" reject the
                # stray `ref` key with a less specific error.
                return None
            if "request" in value:
                return "unit_request"
            if "ref" in value:
                return "absent" if value["ref"] is None else "ref"
        return None
    if isinstance(value, IdentityBindingSet):
        return "set"
    if isinstance(value, IdentityFacetValueRequest):
        return "unit_request"
    if isinstance(value, IdentityFacetValueAbsent):
        return "absent"
    if isinstance(value, IdentityBindingValue):
        return "ref"
    return None


IdentityFacetBinding = Annotated[
    Annotated[IdentityBindingValue, Tag("ref")]
    | Annotated[IdentityBindingSet, Tag("set")]
    | Annotated[IdentityFacetValueRequest, Tag("unit_request")]
    | Annotated[IdentityFacetValueAbsent, Tag("absent")],
    Discriminator(_facet_binding_tag),
]
"""Facet binding union: value, set, a unit-request value, or a null-ref value naming no
declared value. Unbound facets are omitted from the sparse map; slots remain dense and use
``IdentityBindingUnbound``."""


class IdentitySlot(StrictModel):
    """One what/how slot's identity contribution: property dcid + binding state.

    Never a where/when/source slot: `where` is carried by entity roles instead,
    `when` is carried by `requested_window` instead, and `source` is carried by
    `facets.provenance` instead -- all three would otherwise hash resolved
    graph state rather than what was actually asked. Every IdentitySlot here
    carries a real constraint property by construction of the shape grammar.
    """

    property: IdentityString = Field(description="the constraint property dcid")
    binding: IdentityBinding = Field(description="this slot's bound state")


class IdentityWindowAll(StrictModel):
    """No date signal was requested at all."""

    kind: Literal["all"] = "all"


class IdentityWindowLatest(StrictModel):
    """An explicit latest-ask: re-resolve to whatever year is CURRENTLY latest."""

    kind: Literal["latest"] = "latest"


class IdentityWindowRange(StrictModel):
    """A concrete requested date range; at least one bound is always present."""

    kind: Literal["range"] = "range"
    start_year: int | None = Field(
        default=None,
        ge=-_IDENTITY_MAX_SAFE_INTEGER,
        le=_IDENTITY_MAX_SAFE_INTEGER,
        description="absent means open-ended start",
    )
    end_year: int | None = Field(
        default=None,
        ge=-_IDENTITY_MAX_SAFE_INTEGER,
        le=_IDENTITY_MAX_SAFE_INTEGER,
        description="absent means open-ended end",
    )

    @model_validator(mode="after")
    def _at_least_one_bound_and_not_reversed(self) -> IdentityWindowRange:
        if self.start_year is None and self.end_year is None:
            raise ValueError("a range requested_window needs at least one of start_year/end_year")
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError(
                f"requested_window is reversed: start_year {self.start_year} "
                f"> end_year {self.end_year}"
            )
        return self


IdentityWindow = Annotated[
    IdentityWindowAll | IdentityWindowLatest | IdentityWindowRange,
    Field(discriminator="kind"),
]
"""Tagged identity window. Engine ``None`` maps to ``IdentityWindowAll`` because
``BaseIdentity.requested_window`` is always present."""


class IdentityShapeDenominatorRef(StrictModel):
    """The described arm's `measurement_denominator`, resolved to a dcid."""

    ref: IdentityString = Field(description="the resolved denominator StatVar's dcid")


class IdentityShapeDenominatorUnresolved(StrictModel):
    """The described arm's `measurement_denominator`, present on the graph but
    never resolved to a dcid (e.g. a dcid-less ProvisionalNode) -- distinct
    from the member being absent entirely, which means no denominator arc
    exists at all."""

    unresolved: Literal[True] = Field(
        default=True, description="always the literal true -- the arc exists but names no dcid"
    )


def _shape_denominator_tag(value: Any) -> str | None:
    """Tag denominator arms by their sole member, ``ref`` or ``unresolved``."""
    if isinstance(value, dict):
        if "ref" in value and "unresolved" not in value:
            return "ref"
        if "unresolved" in value and "ref" not in value:
            return "unresolved"
        return None
    if isinstance(value, IdentityShapeDenominatorRef):
        return "ref"
    if isinstance(value, IdentityShapeDenominatorUnresolved):
        return "unresolved"
    return None


IdentityShapeDenominator = Annotated[
    Annotated[IdentityShapeDenominatorRef, Tag("ref")]
    | Annotated[IdentityShapeDenominatorUnresolved, Tag("unresolved")],
    Discriminator(_shape_denominator_tag),
]
"""A resolved denominator dcid or an unresolved marker."""


class IdentityShapeSchema(StrictModel):
    """The described-ask arm of the identity core's `shape` union
    (`kind: "statvar_schema"`): up to five sparse schema members mapping 1:1
    onto Search's FiveTuple (see `engine.shape.identity_shape_for`, the
    adapter that builds this from a ShapeDraft's or a grounded wire Shape's
    five-tuple dcids). Minted by Search for every standard and family shape.

    Every member is optional and sparse -- an absent member contributes no
    bytes, the same sparse rule the facets map follows -- but an arm
    describing nothing is invalid: at least one member must be present (see
    `_at_least_one_schema_member`)."""

    kind: Literal["statvar_schema"] = "statvar_schema"
    population_type: IdentityString | None = Field(
        default=None, description="the graph populationType dcid"
    )
    measured_property: IdentityString | None = Field(
        default=None, description="the graph measuredProperty dcid"
    )
    stat_type: IdentityString | None = Field(
        default=None, description="the graph statType dcid, e.g. measuredValue"
    )
    measurement_qualifier: IdentityString | None = Field(
        default=None,
        description="the graph measurementQualifier dcid, present only when the measured property is qualified",
    )
    measurement_denominator: IdentityShapeDenominator | None = Field(
        default=None,
        description=(
            "the graph measurementDenominator, present only on a ratio or per-capita "
            "measure: {'ref': dcid} when resolved, {'unresolved': true} when the arc "
            "exists but names no dcid"
        ),
    )

    @model_validator(mode="after")
    def _at_least_one_schema_member(self) -> IdentityShapeSchema:
        if not any(
            (
                self.population_type,
                self.measured_property,
                self.stat_type,
                self.measurement_qualifier,
                self.measurement_denominator,
            )
        ):
            raise ValueError("shape with kind='statvar_schema' requires at least one schema member")
        return self


class IdentityShapePin(StrictModel):
    """The pinned-ask arm of the identity core's `shape` union
    (`kind: "stat_var"`): a single StatVar dcid, minted by Datasets'
    statvar_set. See `BaseIdentity`'s cross-member validator -- when this arm
    is chosen, `slots` must be empty."""

    kind: Literal["stat_var"] = "stat_var"
    id: IdentityString = Field(description="the pinned StatVar's dcid")


IdentityShape = Annotated[
    IdentityShapeSchema | IdentityShapePin,
    Field(discriminator="kind"),
]
"""The identity core's closed two-arm `shape` union: the described ask
(`IdentityShapeSchema`) or the pinned ask (`IdentityShapePin`)."""


def _validate_facet_axis_names(data: Any) -> Any:
    """Validate facet axes before binding and after-model validation in both identity doors.

    A known axis requiring a newer schema raises ``facet_axis_higher_schema_version``
    (HTTP 422); an unknown axis raises ``ValueError`` (HTTP 400). This ordering prevents a
    malformed binding from hiding a higher-version axis error.
    """
    if not isinstance(data, dict):
        return data
    facets = data.get("facets")
    if not isinstance(facets, dict):
        return data
    unknown: list[str] = []
    higher_version: list[str] = []
    for axis in facets:
        version = _FACET_AXIS_SCHEMA_VERSION.get(axis)
        if version is None:
            unknown.append(axis)
        elif version > SCHEMA_VERSION:
            higher_version.append(axis)
    if higher_version:
        raise PydanticCustomError(
            "facet_axis_higher_schema_version",
            "facet axis name(s) {axes} require a higher schema_version than "
            "this deployment publishes",
            {"axes": ", ".join(sorted(higher_version))},
        )
    if unknown:
        raise ValueError(
            f"unrecognised facet axis name(s): {named_capped(sorted(unknown), max_chars=_MAX_KEY_CHARS)}"
        )
    return data


class BaseIdentity(StrictModel):
    """The spec1 identity-core projection document for one Spec: exactly what
    gets hashed to produce its spec_id, serialised once and echoed back
    verbatim -- never re-derived by a second, parallel implementation.

    Every Spec carries one, on `Spec.definition.identity` (see
    `SpecDefinition`). A dataset_resubmit identity entry posts a prior Spec's own
    `definition.identity` byte-for-byte as its edited `identity`. Directional
    entities live directly in `identity.entities.origin` and
    `identity.entities.destination`. StatVar members are graph state, derived from the
    identity, and no member here.

    The document is hashed whole. `"spec_"` plus the first 32 hex digits of
    the SHA-256 of its RFC 8785 serialisation, with null members omitted, is
    the Spec's spec_id.
    """

    identity_version: Literal["spec1"] = Field(
        default="spec1",
        description="the identity domain separator this Spec was hashed under",
    )
    shape: IdentityShape = Field(description="the shape this Spec was hashed under")
    slots: list[IdentitySlot] = Field(
        max_length=64,
        description="the what/how slots that feed identity -- where/when/source are excluded",
    )
    entities: dict[str, Annotated[list[IdentityString], Field(min_length=1, max_length=1024)]] = (
        Field(
            default_factory=dict,
            max_length=16,
            description=(
                "the bound entities, keyed by role (observationAbout/origin/destination); "
                "{} when none were bound -- always present, never omitted. A role that "
                "appears carries at least one dcid: the identity core refuses a role with "
                "zero members, so an empty list must not be expressible here"
            ),
        )
    )
    requested_window: IdentityWindow = Field(
        description=(
            "the requested date window this Spec was hashed under; always present -- "
            '{"kind": "all"} means no date was requested at all'
        )
    )
    facets: dict[str, IdentityFacetBinding] | None = Field(
        default=None,
        description=(
            "sparse facet bindings (provenance/measurementMethod/unit); omitted entirely "
            "when no facet axis is bound, never an empty object. A facet axis that appears "
            "is bound -- value, set, or (unit axis only) the request form -- never unbound: "
            "the identity core refuses an unbound facet binding state, since a sparse map "
            "omits what is not bound rather than carrying it as unbound"
        ),
    )

    open_roles: list[RoleAxis] | None = Field(
        default=None,
        description=(
            "the flow axes asked for member by member (every creditor, every donor), "
            "sorted and de-duplicated; omitted when none is. A role absent from both "
            "entities and open_roles asks for the total over that role. An open role is "
            "never also an entities key, and a pinned shape (kind='stat_var') carries none"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _classify_facet_axis_names(cls, data: Any) -> Any:
        return _validate_facet_axis_names(data)

    @model_validator(mode="after")
    def _entity_roles_and_request_form(self) -> BaseIdentity:
        unknown_roles = set(self.entities) - _IDENTITY_ENTITY_ROLE_KEYS
        if unknown_roles:
            raise ValueError(
                f"unrecognised entity role key(s): {named_capped(sorted(unknown_roles), max_chars=_MAX_KEY_CHARS)}"
            )
        for axis, binding in (self.facets or {}).items():
            if isinstance(binding, IdentityFacetValueRequest) and axis != "unit":
                raise ValueError(f"the request form is only valid on the unit axis (got {axis!r})")
            if isinstance(binding, IdentityFacetValueAbsent) and axis not in NULL_REF_FACET_AXES:
                raise ValueError(
                    f"a null ref is only valid on the measurementMethod and unit axes (got {axis!r})"
                )
        return self

    @model_validator(mode="after")
    def _pin_arm_requires_empty_slots(self) -> BaseIdentity:
        if isinstance(self.shape, IdentityShapePin) and self.slots:
            raise ValueError("a pinned shape (kind='stat_var') requires an empty slots array")
        return self

    @model_validator(mode="after")
    def _open_roles_unbound_and_described(self) -> BaseIdentity:
        # Entity keys here are the axis labels themselves, so a shared key is a shared axis.
        if not self.open_roles:
            self.open_roles = None
            return self
        bound = sorted(set(self.open_roles) & set(self.entities))
        if bound:
            raise ValueError(f"open_roles names role(s) entities also binds: {', '.join(bound)}")
        if isinstance(self.shape, IdentityShapePin):
            raise ValueError("a pinned shape (kind='stat_var') cannot carry open_roles")
        self.open_roles = sorted(set(self.open_roles))
        return self
