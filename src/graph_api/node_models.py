"""Request and response models for `GET /api/graph/node/{dcid}` and `POST /api/graph/nodes`.

The `shape` union distinguishes StatVars, sources, place-family entities,
and other held nodes. Every variant carries `dcid` and a nullable label.
`stat_var.facets` and `stat_var.sources` require entity-scoped observation
reads; `processing_level` has no graph convention. All three remain typed
nullable fields, so populating them later is additive.

A StatVar and a source node carry `data_commons`, the origins the request's
origin catalogue admits for them. Entities and generic nodes carry none,
because a place or a schema node is shared by every Data Commons.

`POST /api/graph/nodes` answers up to `NODES_MAX_DCIDS` dcids in one read, each in the
shape the GET route returns, with a `not_held` entry where the GET route answers 404.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from graph_api import routing
from graph_api.origin_models import DataCommonsRef

NODES_MAX_DCIDS = 50
"""The most dcids one `POST /api/graph/nodes` request names. Above it the request refuses
`request_invalid` (400)."""


class Freshness(BaseModel):
    """`source`'s own refresh cadence, read from three DCP provenance arcs
    populated unevenly upstream -- live-verified against
    `dc/base/WHO_COVID19` (carries both `release_frequency` and
    `refresh_type`) and `dc/base/BIS_CentralBankPolicyRate` (carries
    `refresh_type` but lacks `release_frequency`). Coverage varies node by
    node even within one publisher -- `dc/base/WHO_GHO` and `dc/base/UN_WHO`
    carry neither arc -- so "a WHO provenance" is not a reliable
    generalisation; only the two dcids named above are asserted here.
    Nullability is the contract, not a fallback.

    `release_frequency` is the raw `sourceReleaseFrequency` ISO 8601
    duration token (e.g. `P1M`); `release_frequency_label` is
    `materialise.cadence.cadence_label`'s own conversion of it to a display
    label (e.g. `Monthly`), or the raw token itself when that module does
    not recognise it. The raw token is always served alongside the label,
    never replaced by it.

    `refresh_type` is `automaticRefreshType`'s own value -- how this
    deployment refreshes the source, not a schedule. Not always a
    mechanism name: `dc/base/India_RBIStateDomesticProduct_AggCountry`
    carries the sentinel value `NoAutomaticRefresh`, live-verified,
    alongside mechanism names like `GCPCloudRunAutomaticRefresh`.
    `automaticRefreshSchedule`, the arc this field used to read, does not
    exist in the graph; every value it ever served was `None`.
    """

    release_frequency: str | None
    release_frequency_label: str | None
    refresh_type: str | None


class Publisher(BaseModel):
    """The `source` arc's target -- the organisation that publishes a
    `Provenance`/`Source` node's data (e.g. `dc/s/WorldBank`). `label` is
    `None` when that target is itself not held by this deployment's graph,
    carries no `name` arc, or carries only a `name` value equal to its own
    dcid (a junk triple some imports write) -- the entry appears either
    way, with a null label.
    """

    dcid: str
    label: str | None


class ParentDataset(BaseModel):
    """The `isPartOf` arc's target -- the `Dataset` node a `Provenance`
    rolls up under (e.g. `dc/d/WorldBank_WorldDevelopmentIndicators`).
    Nullability of `label` follows `Publisher.label`'s own rule. A
    `Provenance` with no parent Dataset (the ONE-minted
    `provenance/OECDCreditorReportingSystem` among them) carries `None`
    here instead of this model.
    """

    dcid: str
    label: str | None


class StatVarNode(BaseModel):
    shape: Literal["stat_var"] = "stat_var"
    dcid: str
    label: str | None
    definition: str | None = Field(
        description=(
            "the definition arc's value, the machine string that spells out the StatVar's "
            "constraints (`mq=Nominal,mp=amount,...`). None when the node has no definition arc"
        )
    )
    description: str | None = Field(
        description=(
            "the description arc's value, the publisher's prose gloss of the measure. None "
            "when the node has no description arc"
        )
    )
    roles: list[str] = Field(
        description=(
            "the entity roles this StatVar's observations carry, from its "
            "declared observationProperties, defaulting to observationAbout "
            "when it declares none"
        )
    )
    population_type: str | None = Field(
        description=(
            "the populationType arc's value: what this StatVar's "
            "observations measure a property of (None when the node does "
            "not declare it)"
        )
    )
    source_indicator_codes: list[str] = Field(
        description=(
            "the sourceIndicatorCode arc's values: the codes naming the source series this "
            "StatVar comes from, as the publisher codes it (IMF WEO GDP carries NGDP and "
            "NGDPD); empty when the node carries none"
        )
    )
    data_commons: list[DataCommonsRef] = Field(
        description=(
            "every Data Commons holding admitted data for this StatVar, DFO first, then "
            "base, then the UN; empty when the origin catalogue admits none"
        )
    )
    facets: None = None
    processing_level: None = None
    sources: None = None


class SourceNode(BaseModel):
    """`label` prefers the parent Dataset's own name (via `dataset.label`)
    and falls back to this node's own `name` arc value when there is no
    parent Dataset, or that Dataset carries no name of its own --
    `Provenance.name` is an import identifier (e.g.
    `GeoBoundaries_Jordan`), never a display title; the readable title
    lives on the parent Dataset (e.g. "Global Database of Political
    Administrative Boundaries Database").
    """

    shape: Literal["source"] = "source"
    dcid: str
    label: str | None
    license: str | None
    license_type: str | None
    description: str | None
    url: str | None
    source_data_url: str | None = Field(description="the sourceDataUrl arc's value")
    description_url: str | None = Field(description="the descriptionUrl arc's value")
    transformation_logic_url: str | None = Field(
        description="the dataTransformationLogic arc's value -- a URL to the transformation code, not prose"
    )
    last_data_refresh_date: str | None = Field(description="the lastDataRefreshDate arc's value")
    next_data_refresh_date: str | None = Field(description="the nextDataRefreshDate arc's value")
    earliest_observation_date: str | None = Field(
        description="the earliestObservationDate arc's value"
    )
    latest_observation_date: str | None = Field(description="the latestObservationDate arc's value")
    publisher: Publisher | None = Field(
        description="the source arc's target, resolved to a dcid+label; None when this node carries no source arc"
    )
    dataset: ParentDataset | None = Field(
        description="the isPartOf arc's target, resolved to a dcid+label; None when this node has no parent Dataset"
    )
    freshness: Freshness
    data_commons: DataCommonsRef | None = Field(
        description=(
            "the Data Commons serving this provenance, from the origin catalogue; null for a "
            "provenance the catalogue does not admit, a Dataset, or a Source"
        )
    )


class ContainedInPlace(BaseModel):
    """One `containedInPlace` target. `label` is `None` when that target is
    itself not held by this deployment's graph, or carries no `name` arc --
    the entry appears either way, with a null label.
    """

    dcid: str
    label: str | None


class EntityNode(BaseModel):
    shape: Literal["entity"] = "entity"
    dcid: str
    label: str | None
    entity_type: str | None
    contained_in: list[ContainedInPlace]


class GenericNode(BaseModel):
    """Every node this door doesn't recognize as a StatVar, a source, or a
    place-family entity. `properties` carries every non-empty allowlisted
    arc not already surfaced by a dedicated field (`types` covers `typeOf`,
    `label` covers `name`).
    """

    shape: Literal["node"] = "node"
    dcid: str
    label: str | None
    types: list[str]
    properties: dict[str, list[str]]


NodeResponse = Annotated[
    StatVarNode | SourceNode | EntityNode | GenericNode,
    Field(discriminator="shape"),
]


class NotHeldNode(BaseModel):
    """A batch entry for a dcid this deployment's graph does not hold, where
    `GET /api/graph/node/{dcid}` answers 404 `node_not_held`."""

    shape: Literal["not_held"] = "not_held"
    dcid: str


NodesEntry = Annotated[
    StatVarNode | SourceNode | EntityNode | GenericNode | NotHeldNode,
    Field(discriminator="shape"),
]

_Dcid = Annotated[
    str,
    StringConstraints(
        pattern=rf"^{routing.DcidConvertor.regex}$", max_length=routing.DCID_MAX_LENGTH
    ),
]


class NodesRequest(BaseModel):
    """`dcids`: 1 to `NODES_MAX_DCIDS` node dcids read as one batch under one DCP pool
    slot. Each dcid matches the GET route's `{dcid}` path syntax and its 512-character
    ceiling."""

    model_config = ConfigDict(extra="forbid", title="GraphNodesRequest")
    dcids: list[_Dcid] = Field(min_length=1, max_length=NODES_MAX_DCIDS)


class NodesResponse(BaseModel):
    """`nodes` is index-aligned with the request's `dcids`: a repeated dcid repeats its
    entry. Each entry is the body `GET /api/graph/node/{dcid}` returns for that dcid, or a
    `not_held` entry where that route answers 404."""

    nodes: list[NodesEntry]
