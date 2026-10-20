"""Build deployment capabilities from the limits and policies the Search engine enforces.

These values can change with configuration or deployment without a schema-version bump.
"""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel, ConfigDict, Field

from code_index.query import CodeIndex
from code_index.shared import get_index
from dfo_platform.build import build_commit
from search.engine.config import (
    SEARCH_MAX_ENTITY_TEXT_CHARS,
    SEARCH_MAX_PART_ENTITIES,
    SEARCH_MAX_PARTS,
    SEARCH_MAX_ROLE_TEXT_CHARS,
    SEARCH_MAX_SOURCE_TEXT_CHARS,
    SEARCH_MAX_TIME_TEXT_CHARS,
    SEARCH_MAX_UNIT_TEXT_CHARS,
    SEARCH_MAX_VARIABLE_TEXT_CHARS,
    SEARCH_RECIPE_CLASSES,
    SEARCH_RESOLVE_DOORS,
    SEARCH_UNIMPLEMENTED_DOORS,
    settings,
)
from search.models import Axis, FacetAxis
from search.observations import observation_settings

# Canonical resolve doors, also used in the capability-version hash.
_RESOLVE_DOORS: tuple[str, ...] = SEARCH_RESOLVE_DOORS


class CapabilityCaps(BaseModel):
    """Published caps and input limits, one field per enforced constant."""

    model_config = ConfigDict(title="SearchCapabilityCaps")

    max_query_chars: int = Field(
        description="raw_text.query character ceiling; over this refuses query_too_long"
    )
    max_candidates: int = Field(
        description=(
            "server ceiling enforced two ways: a resolved candidate set is clamped to "
            "this length, while a dataset_resubmit "
            "candidates entry posting more candidates than this refuses "
            "request_invalid at /input/entries/{i}/candidates rather than being clamped"
        )
    )
    max_variables: int = Field(
        description="variables resolved per query before the rest are dropped with variables_clamped"
    )
    obs_preview_points: int = Field(
        description="latest points per series carried once observation preview mode applies"
    )
    obs_inline_max: int = Field(
        description="total observation points above which every series switches to preview mode"
    )
    role_bulk_rank_max_rows: int = Field(
        description=(
            "rows a role-bearing bulk preview may read whole and rank, most recent and "
            "largest first; above this it shows a fixed selection, and its "
            "observations_series_truncated warning says so with params.order"
        )
    )
    max_options: int = Field(
        description=(
            "ceiling on the options door's candidate lists before truncation: "
            "graph-enumerated observationAbout candidates, and the sdmx-derived "
            "provider and recipient candidate lists alike"
        )
    )
    max_parts: int = Field(
        description=(
            "parsed.parts list length ceiling; a request over this refuses "
            "request_invalid. Distinct from max_variables: raw_text clamps an "
            "over-limit extraction with a variables_clamped warning and keeps "
            "resolving, while parsed enforces this as a hard schema cap"
        )
    )
    max_part_entities: int = Field(
        description="parsed part entities list length ceiling; over this refuses request_invalid"
    )
    max_variable_text_chars: int = Field(
        description="parsed part variable_text character ceiling; over this refuses request_invalid"
    )
    max_entity_text_chars: int = Field(
        description="parsed entity text character ceiling; over this refuses request_invalid"
    )
    max_role_text_chars: int = Field(
        description="parsed entity role_text character ceiling; over this refuses request_invalid"
    )
    max_time_text_chars: int = Field(
        description="parsed part time_text character ceiling; over this refuses request_invalid"
    )
    max_unit_text_chars: int = Field(
        description="parsed part unit_text character ceiling; over this refuses request_invalid"
    )
    max_source_text_chars: int = Field(
        description="parsed part source_text character ceiling; over this refuses request_invalid"
    )
    region_max_members: int = Field(
        description=(
            "ceiling on the alt-shape population region resolution pre-warms observation "
            "facets for; over this, the pre-warm is skipped rather than refused"
        )
    )
    scope_min_year: int = Field(
        description=(
            "earliest year role-scope discovery will request; an open-started "
            "requested window floors here rather than reaching further back"
        )
    )
    scope_max_window_years: int = Field(
        description=(
            "ceiling on a requested window's year span for a role-bearing read, "
            "named or discovered; over this refuses scope_too_broad"
        )
    )


class CodeIndexStamp(BaseModel):
    """Provenance for the packaged code index this deployment's constraint
    resolution reads: which export, and how many codes it carries per code
    system. Not one of `capability_version`'s hash inputs (see
    `search.engine.config.settings().capability_version`, computed from static
    config constants alone) -- published here as its own field so a caller
    can compare `content_hash` across two responses (e.g. a preview and a
    later download) and tell whether they resolved against the same index.
    """

    model_config = ConfigDict(title="SearchCodeIndexStamp")

    content_hash: str = Field(
        description="digest over the exported rows; equal across two exports of "
        "the same graph state"
    )
    exported_at: str = Field(description="when the packaged index was exported from the graph")
    code_counts: dict[str, int] = Field(
        description="code count per code system this index carries -- keyed by "
        "whatever code systems the index declares, not a fixed set of named fields"
    )


class CapabilityDoor(BaseModel):
    """One door's liveness in this deployment."""

    model_config = ConfigDict(title="SearchCapabilityDoor")

    name: str = Field(description="raw_text, parsed, dataset_resubmit, or options")
    live: bool = Field(description="whether this deployment serves this door")


class CapabilityDefaultPolicy(BaseModel):
    """One declared default policy: whether it is live, and its current
    default value when it is a simple boolean toggle.
    """

    name: str = Field(description="the declared default policy's name")
    live: bool = Field(description="whether this deployment honours the declared default")
    default: bool | None = Field(
        default=None,
        description="the current default value, when this policy is a boolean toggle",
    )


class Capabilities(BaseModel):
    """The full GET /api/search/capabilities document."""

    model_config = ConfigDict(title="SearchCapabilities")

    capability_version: str = Field(
        description=(
            "deterministic token identifying this capability set; distinct from "
            "schema_version and shared verbatim with problem-details bodies and "
            "resolved.trace"
        )
    )
    build: str = Field(
        description="the commit sha the running image was built from, the same "
        "value the x-dfo-commit response header carries"
    )
    axes: list[str] = Field(
        description=(
            "valid SlotKey.axis values. Contract-frozen (a schema_version-locked "
            "enum, search.models.Axis), republished here for convenience so a "
            "caller need not parse the full JSON Schema to enumerate them"
        )
    )
    facet_axes: list[str] = Field(
        description=(
            "valid facet-axis names (BaseIdentity.facets / ResubmitIdentityPayload.identity.facets "
            "keys), distinct from axes above -- search.models.FacetAxis, a separate "
            "vocabulary from SlotKey.axis. Schema-locked like axes: this vocabulary "
            "changes only with a schema_version bump, so unlike recipe_classes or the "
            "caps below, it is NOT one of capability_version's hash inputs -- "
            "schema_version already covers it"
        )
    )
    recipe_classes: list[str] = Field(
        description=(
            "the conversion recipe members this deployment applies "
            "(resolved.observations.series[].conversion.recipe_id), the sorted keys of "
            "materialise.recipe_bodies.RECIPE_BODIES, so the list and the applied "
            "recipes cannot drift apart"
        )
    )
    doors: list[CapabilityDoor] = Field(description="per-door liveness")
    default_policies: list[CapabilityDefaultPolicy] = Field(
        description="declared default policies and whether each is live"
    )
    caps: CapabilityCaps = Field(description="published caps and input limits")
    code_index: CodeIndexStamp = Field(description="provenance for the packaged code index")


def build_capabilities(index: CodeIndex | None = None) -> Capabilities:
    """Build capabilities from enforced config values and the stamp of `index`, by default the
    one `get_index()` returns.

    App construction loads the index, so a missing or malformed export fails startup.
    """
    index_stamp = (index if index is not None else get_index()).stamp
    doors = [
        CapabilityDoor(name=door, live=door not in SEARCH_UNIMPLEMENTED_DOORS)
        for door in _RESOLVE_DOORS
    ]
    # The options door is mounted and served in this deployment.
    doors.append(CapabilityDoor(name="options", live=True))

    return Capabilities(
        capability_version=settings().capability_version,
        build=build_commit(),
        axes=sorted(get_args(Axis)),
        facet_axes=sorted(get_args(FacetAxis)),
        recipe_classes=list(SEARCH_RECIPE_CLASSES),
        doors=doors,
        default_policies=[
            CapabilityDefaultPolicy(
                name="place_as_constraint", live=True, default=settings().seam_default
            ),
            # Unit completion takes the currency from SEARCH_UNIT_DEFAULT_CURRENCY, prices
            # as current, and the base year from the conversion tables' deflatorBaseYear.
            CapabilityDefaultPolicy(name="unit_currency", live=True),
            CapabilityDefaultPolicy(name="unit_prices", live=True),
            CapabilityDefaultPolicy(name="unit_base_year", live=True),
        ],
        caps=CapabilityCaps(
            max_query_chars=settings().max_query_chars,
            max_candidates=settings().max_candidates,
            max_variables=settings().max_variables,
            obs_preview_points=observation_settings().obs_preview_points,
            obs_inline_max=observation_settings().obs_inline_max,
            role_bulk_rank_max_rows=observation_settings().role_bulk_rank_max_rows,
            max_options=settings().max_options,
            max_parts=SEARCH_MAX_PARTS,
            max_part_entities=SEARCH_MAX_PART_ENTITIES,
            max_variable_text_chars=SEARCH_MAX_VARIABLE_TEXT_CHARS,
            max_entity_text_chars=SEARCH_MAX_ENTITY_TEXT_CHARS,
            max_role_text_chars=SEARCH_MAX_ROLE_TEXT_CHARS,
            max_time_text_chars=SEARCH_MAX_TIME_TEXT_CHARS,
            max_unit_text_chars=SEARCH_MAX_UNIT_TEXT_CHARS,
            max_source_text_chars=SEARCH_MAX_SOURCE_TEXT_CHARS,
            region_max_members=settings().region_max_members,
            scope_min_year=settings().scope_min_year,
            scope_max_window_years=settings().scope_max_window_years,
        ),
        code_index=CodeIndexStamp(
            content_hash=index_stamp.content_hash,
            exported_at=index_stamp.exported_at,
            code_counts=dict(index_stamp.code_counts),
        ),
    )
