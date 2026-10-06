from __future__ import annotations

from typing import Literal

from pydantic import (
    ConfigDict,
    Field,
    field_validator,
)

from .base import StrictModel
from .envelope import Diagnostics
from .identity import BaseIdentity
from .options import SlotOption
from .vocab import SCHEMA_VERSION, _require_schema_version_int

# ---------------------------------------------------------------------------
# OptionsRequest/OptionsResponse: the options door
# (POST /api/search/options)
# ---------------------------------------------------------------------------


class OptionsFor(StrictModel):
    """Which resolved entity role the options door enumerates candidates for.

    Wrapped in its own object (rather than a bare string field) so the wire
    body reads `"for": {"role": "destination"}`, matching the contract's
    envelope shape on both the request and the echoed response.
    """

    role: str = Field(
        description=(
            "the resolved entity role to enumerate candidates for (e.g. "
            "'origin', 'destination', 'observationAbout' -- the vocabulary is "
            "derived per family, not frozen here). Deliberately a plain str, "
            "NEVER a Literal or enum: the target is checked only AFTER the "
            "posted identity resolves to StatVars, refusing "
            "option_target_invalid (422) for a role the resolved family does "
            "not carry. A Literal/enum would instead refuse an unrecognised "
            "role at the pydantic validation layer as request_invalid (400), "
            "before resolution ever runs -- do not tighten this to an enum."
        )
    )
    scope: Literal["peers", "children"] | None = Field(
        default=None,
        description=(
            "which candidate universe to enumerate for a role whose options are "
            "derived from the graph rather than a curated list; absent means "
            "'peers'. 'peers' enumerates entities sharing the bound entity's own "
            "type; 'children' enumerates entities contained in it. Roles backed "
            "by a curated candidate list ignore this member, having only one "
            "universe to draw from. Naming a role alone cannot distinguish the "
            "two readings -- peers of a country and the regions within it are "
            "both 'options for observationAbout' -- and only the peers reading "
            "is recoverable from the identity, so without this member the "
            "contained-in reading is unreachable rather than merely unrequested."
        ),
    )


class OptionsRequest(StrictModel):
    """Request body for the options door (POST /api/search/options): enumerate
    the candidate set for one resolved entity role of a posted identity.

    An identity block only. The door derives the identity's own StatVar
    population server-side (engine.options.population.options_population)
    rather than resolving a Spec through the resubmit path.

    validate_by_name=False (with validate_by_alias=True, the pydantic default
    pairing made explicit here): the published schema declares only the alias
    `for`, and this model is otherwise extra="forbid", so a wire body must be
    rejected for posting the internal Python attribute name `for_` instead --
    accepting both would let a client bypass the declared schema through the
    implementation-detail spelling. `for_` exists only because `for` is a
    Python reserved word and cannot itself be an attribute name; every
    constructor call (internal code, tests) builds this model through the
    alias instead, e.g. ``OptionsRequest(identity=..., **{"for": OptionsFor(...)})``.
    """

    model_config = ConfigDict(validate_by_alias=True, validate_by_name=False)

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="contract version the caller targets",
    )
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    identity: BaseIdentity = Field(
        description="the identity to resolve and enumerate candidate options for"
    )
    for_: OptionsFor = Field(
        alias="for",
        description="which resolved entity role to enumerate candidate options for",
    )
    include_coverage: bool = Field(
        default=False,
        description=(
            "when true, the engine probes each returned option for observation "
            "coverage (e.g. 'Kenya — 205 obs') and populates SlotOption.coverage; "
            "false (the default) never fires the extra probe, so this is opt-in "
            "latency, not a hidden cost on the hot path. The only resolution "
            "option this door reads -- unlike /resolve's ResolveOptions, this "
            "door has no candidate cap or sentence concept, so it "
            "carries this one field directly rather than nesting a "
            "response_options object with members it would never use."
        ),
    )


class OptionsSet(StrictModel):
    """The enumerated candidate set an options-door request returns.

    Items reuse SlotOption for shape parity with slot_options/entity_options
    ({value, coverage}) rather than a simpler {ref, label, obs_count} shape --
    the richer coverage form is what the engine already serves. An
    empty list is itself a valid result (e.g. the current recipient's
    variable truly has no other recipient with data), not an error.

    Carries no graph_revision -- see OptionsResponse's docstring for why.
    """

    items: list[SlotOption] = Field(
        description=(
            "candidate values, deterministically ordered; a dcid that failed "
            "to resolve a label is omitted rather than fabricated"
        )
    )
    truncated: bool = Field(
        description="true when items was capped before every candidate could be checked"
    )


class OptionsResponse(StrictModel):
    """Response for POST /api/search/options: `{schema_version, diagnostics,
    for, options}`. Carries no semantic status: the door either refuses
    (through the same shared identity validation a dataset_resubmit identity
    entry runs) or returns options -- possibly an empty
    list -- with no definite/candidates/no_data three-way to report on top.

    Omits graph_revision: no revision concept exists anywhere in src/ --
    nothing tracks the graph the way engine_build tracks the code build -- so
    stamping one here would assert a graph fact this deployment cannot back.
    Left out deliberately rather than filled with something untrue.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="contract schema version; exactly 1 for v1",
    )
    _validate_schema_version = field_validator("schema_version", mode="before")(
        _require_schema_version_int
    )
    diagnostics: Diagnostics = Field(description="run health, warnings, timing, build stamp")
    for_: OptionsFor = Field(
        alias="for",
        description=(
            "the role this response's options were enumerated for, echoed back from the request"
        ),
    )
    options: OptionsSet = Field(description="the enumerated candidate set")
