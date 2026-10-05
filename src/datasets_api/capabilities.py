"""Build the mutable deployment capabilities document linked from `/schema`.

Published caps come from enforced constants. Mint input kinds and formats report dispatch
liveness. Revision provenance and graph-declared default facet policies are disabled. Unbound facet
unbound axes use this deployment's fallback policy when candidates exist.

Callers supply `capability_version`, `build`, `code_index_stamp`, and `declarations` to avoid a
self-referential version hash and keep this module independent of the code-index loader.
`decode_code_systems` comes from `code_index_stamp.code_counts`. Decoding is always enabled for
those systems, while `enrichment_fields` reflects the separately configurable field gate.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from code_index.types import CodeSystemDeclaration, IndexStamp, summable_axes
from dataset_store.snapshot import WORKING_TIER_RETENTION
from datasets_api.config import DatasetsBuildSettings
from datasets_api.models import MAX_COLLECTION_MEMBERS
from materialise.facets import DEPLOYMENT_FACET_POLICY, DEPLOYMENT_FACET_POLICY_VERSION
from materialise.recipe_bodies import RECIPE_CLASSES
from statvar_members.enumerate import CANDIDATE_CAP

# The four `kind`s the mint request's tagged union accepts
# (datasets_api.mint_models.MintInput), and whether each is actually
# dispatched by a live door.
_LIVE_MINT_INPUT_KINDS: frozenset[str] = frozenset(
    {"statvar_set", "source_codes", "spec_id", "collection_id"}
)
_MINT_INPUT_KINDS: tuple[str, ...] = ("spec_id", "collection_id", "statvar_set", "source_codes")

# The mint door renders CSV, XLSX, and Parquet. Bundle is a Frictionless Data Package delivery
# mode published beside those formats.
_LIVE_RENDERING_FORMATS: frozenset[str] = frozenset({"csv", "xlsx", "parquet"})
_RENDERING_FORMATS: tuple[str, ...] = ("csv", "xlsx", "parquet")

# The subset of `materialise.reshape.SUPPORTED_ENRICHMENTS` enabled here.
# `datasets_api.render._reshape` maps a qualified
# `{entity_column}_{field}` token whose `field` falls in
# `SUPPORTED_ENRICHMENTS - ENABLED_ENRICHMENTS` to `capability_unavailable`
# once `materialise.reshape.parse_column_tokens` has parsed it as an
# enrichment request against the materialised frame's own entity columns.
ENABLED_ENRICHMENTS: frozenset[str] = frozenset({"iso2", "iso3", "label", "type"})

# `groupby` reads summability from the packaged code index, loaded for the process lifetime.
# This capability is always live because the index is available whenever this module is used.
SUMMABILITY_DECLARATIONS_LIVE: bool = True


class CapabilityCaps(BaseModel):
    """Published input limits and response caps, one field per enforced
    constant."""

    max_collection_members: int = Field(
        description="ceiling on a register/mint collection's member count"
    )
    definition_candidate_cap: int = Field(
        description=(
            "ceiling on the reverse-arc candidate window a described-arm definition "
            "enumerates inside before resolve_definition_members raises OverCeilingError "
            "(candidate_window)"
        )
    )
    sync_budget_rows: int = Field(
        description=(
            "sync budget: a mint/definition request whose own coverage estimate "
            "(series_coverage.index.CoverageIndex.estimate) stays at or under this many "
            "rows, and at or under sync_budget_requests SDMX requests, runs inline"
        )
    )
    sync_budget_requests: int = Field(
        description="the SDMX request half of the sync budget -- see sync_budget_rows"
    )
    async_max_rows: int = Field(
        description=(
            "async ceiling: an over-sync-budget mint/definition request whose own coverage "
            "estimate (series_coverage.index.CoverageIndex.estimate) stays at or under this "
            "many rows, and at or under async_max_requests SDMX requests, is recorded as an "
            "operation and answered 202; over either, it is refused over_ceiling "
            "(ceiling_kind operation_rows) before any operation is recorded"
        )
    )
    async_max_requests: int = Field(
        description=(
            "the SDMX request half of the async ceiling -- see async_max_rows; crossing this "
            "one alone refuses over_ceiling with ceiling_kind operation_requests"
        )
    )
    pivot_max_output_cells: int = Field(
        description="cap on a pivot's total output cell count (rows x columns) before pivot_over_cell_cap"
    )
    operation_retention_days: int = Field(
        description=(
            "how long a terminal (done/failed) operation stays guaranteed pollable "
            "(contract, 'The operation envelope'). A done operation's own manifest is "
            "guaranteed servable for exactly this long, since it is bounded by the same "
            "working-tier artifact clock every other mint/definition response is "
            "(dataset_store.snapshot.WORKING_TIER_RETENTION) -- pinning it extends that "
            "the same way pinning any other snapshot does. A failed operation's own row "
            "carries no artifact and this deployment runs no janitor against "
            "datasets.operation, so it in practice stays pollable past this many days; "
            "this number is the guarantee, not an expiry this deployment enforces"
        )
    )


class CapabilityKind(BaseModel):
    """One mint-request `kind`'s liveness in this deployment."""

    name: str = Field(description="spec_id, collection_id, statvar_set, or source_codes")
    live: bool = Field(description="whether this deployment's mint door dispatches this kind")


class CapabilityFormat(BaseModel):
    """One output rendering's liveness in this deployment."""

    name: str = Field(description="csv, xlsx, or parquet")
    live: bool = Field(description="whether this deployment can render this format")


class CodeIndexStamp(BaseModel):
    """Provenance for the packaged code index this deployment's `source_codes`
    mint kind reads: which export, and how many codes it carries per code
    system. Published so a caller can compare `content_hash` across two
    responses (a preview and a later download, say) and tell whether they
    resolved against the same index.
    """

    content_hash: str = Field(
        description="digest over the exported rows; equal across two exports of "
        "the same graph state"
    )
    exported_at: str = Field(description="when the packaged index was exported from the graph")
    code_counts: dict[str, int] = Field(
        description="code count per code system this index carries -- keyed by "
        "whatever code systems the index declares, not a fixed set of named fields"
    )


class DeploymentFacetPolicy(BaseModel):
    """This deployment's own default-facet policy: what an unbound facet
    axis falls back to once `default_facet_policies_live` (a graph-declared,
    per-StatVar policy) is `false`. See `docs/datasets-facet-policy.md`.
    """

    policy: str = Field(description="the deployment default policy's own id")
    version: str = Field(description="the policy id's own version")


class Capabilities(BaseModel):
    """The full `GET /api/datasets/capabilities` document."""

    capability_version: str = Field(
        description=(
            "deterministic token identifying this capability set; distinct from "
            "schema_version and shared verbatim with problem-details bodies"
        )
    )
    build: str = Field(
        description="the commit sha the image was built from, the same value "
        "the x-dfo-commit response header carries"
    )
    schema_versions: list[int] = Field(description="schema_version values this deployment accepts")
    operation_envelope_active: bool = Field(
        description="whether POST mint/definition doors can answer 202 with a pollable operation"
    )
    input_kinds: list[CapabilityKind] = Field(description="per-kind mint-door liveness")
    formats: list[CapabilityFormat] = Field(description="per-format rendering liveness")
    bundle_active: bool = Field(
        description="whether a mint/definition response can deliver a Frictionless Data Package"
    )
    recipe_classes: list[str] = Field(
        description=(
            "the conversion recipes this deployment can apply, the ConversionRecipe "
            "members from materialise.recipe_bodies.RECIPE_CLASSES"
        )
    )
    programmatic_pinning: bool = Field(
        description="whether a caller other than the site's own share/publish flow may pin"
    )
    enrichment_fields: list[str] = Field(
        description=(
            "enrichment field names this deployment enables. A live token qualifies one as "
            "{entity_column}_{field} on the columns option"
        )
    )
    decode_code_systems: list[str] = Field(
        description=(
            "code systems the packaged code index declares, sorted. Decoding is always on "
            "when a rendered frame's StatVars declare one of these -- there is no opt-out "
            "-- and each appears on the frame as its own stat_var_{code_system} column, "
            "the StatVar's own code decoded to that system's label"
        )
    )
    revision_provenance_live: bool = Field(
        description="whether graph_revision/source_revisions bind to a real endpoint"
    )
    default_facet_policies_live: bool = Field(
        description=(
            "whether the graph declares its own per-StatVar/per-source default facet "
            "policy; false does not mean an unbound axis refuses -- see "
            "default_facet_policy"
        )
    )
    default_facet_policy: DeploymentFacetPolicy | None = Field(
        description=(
            "this deployment's own default-facet policy, consulted for an unbound axis "
            "once the graph declares none; null if this deployment enforces no fallback "
            "of its own"
        )
    )
    summability_declarations_live: bool = Field(
        description=(
            "whether the graph declares its own per-axis summability; true "
            "unconditionally, the same way decode carries no separate liveness gate"
        )
    )
    summable_axes: list[str] = Field(
        description=(
            "code systems the packaged code index declares summable, sorted -- the "
            "axes a groupby request may name in `by` to keep, with every other "
            "declared-summable axis this frame decoded collapsed into it"
        )
    )
    caps: CapabilityCaps = Field(description="published input limits and response caps")
    code_index: CodeIndexStamp = Field(description="provenance for the packaged code index")


def build_capabilities(
    *,
    capability_version: str,
    build: str,
    code_index_stamp: IndexStamp,
    declarations: Mapping[str, CodeSystemDeclaration],
    settings: DatasetsBuildSettings,
) -> Capabilities:
    """Assemble capabilities from caller-supplied settings and code-index data.

    The exported `code_counts` keys equal the declared code systems, including zero-count
    systems, so `decode_code_systems` needs no separate input. `summable_axes` uses
    `declarations`, since counts do not encode summability.
    """
    return Capabilities(
        capability_version=capability_version,
        build=build,
        schema_versions=[1],
        operation_envelope_active=True,
        input_kinds=[
            CapabilityKind(name=kind, live=kind in _LIVE_MINT_INPUT_KINDS)
            for kind in _MINT_INPUT_KINDS
        ],
        formats=[
            CapabilityFormat(name=fmt, live=fmt in _LIVE_RENDERING_FORMATS)
            for fmt in _RENDERING_FORMATS
        ],
        bundle_active=True,
        recipe_classes=list(RECIPE_CLASSES),
        programmatic_pinning=False,
        enrichment_fields=sorted(ENABLED_ENRICHMENTS),
        decode_code_systems=sorted(code_index_stamp.code_counts),
        revision_provenance_live=False,
        default_facet_policies_live=False,
        default_facet_policy=DeploymentFacetPolicy(
            policy=DEPLOYMENT_FACET_POLICY, version=DEPLOYMENT_FACET_POLICY_VERSION
        ),
        summability_declarations_live=SUMMABILITY_DECLARATIONS_LIVE,
        summable_axes=sorted(summable_axes(declarations)),
        caps=CapabilityCaps(
            max_collection_members=MAX_COLLECTION_MEMBERS,
            definition_candidate_cap=CANDIDATE_CAP,
            sync_budget_rows=settings.sync_budget_rows,
            sync_budget_requests=settings.sync_budget_requests,
            async_max_rows=settings.async_max_rows,
            async_max_requests=settings.async_max_requests,
            pivot_max_output_cells=settings.pivot_max_output_cells,
            operation_retention_days=WORKING_TIER_RETENTION.days,
        ),
        code_index=CodeIndexStamp(
            content_hash=code_index_stamp.content_hash,
            exported_at=code_index_stamp.exported_at,
            code_counts=dict(code_index_stamp.code_counts),
        ),
    )
