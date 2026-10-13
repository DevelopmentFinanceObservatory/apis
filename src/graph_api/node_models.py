"""Request and response models for `GET /api/graph/node/{dcid}` and `POST /api/graph/nodes`.

Every node is one envelope: `ref`, `label`, `kind`, `types`, `description`, `links`, `gaps`
and `built_from`, plus one detail block named for its kind (`graph_api.kinds`). `other` has no
block. Every reference is `{ref, label}`, with the dcid in `ref`. A value this door cannot know
is `null` and its member path is named in `gaps`; a verified empty list is `[]`.

The `stat_var` block holds the variable's `definition` (`graph_api.card_definition`) and its
`availability` (`graph_api.availability`). A source-family block carries `data_commons`, the
origin the request's origin catalogue admits for it.

`POST /api/graph/nodes` answers up to `NODES_MAX_DCIDS` dcids in one read, index-aligned,
each as the GET route's body, or `{ref, status: "not_held"}` where the GET route answers 404.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from dfo_platform.wire import ResponseModel
from entity_resolution.membership import DEFAULT_PROJECTION, Projection
from graph_api import routing
from graph_api.availability import Availability
from graph_api.card_definition import VariableDefinition
from graph_api.origin_models import DataCommonsRef
from graph_api.refs import NodeRef
from unit_words.unit_class import UnitKind

NODES_MAX_DCIDS = 50
"""The most dcids one `POST /api/graph/nodes` request names. Above it the request refuses
`request_invalid` (400)."""

LinkRelation = Literal[
    "codes", "children", "members", "countries", "datasets", "variables", "facets"
]
"""The list relations a node can link to, each served by
`GET /api/graph/node/{dcid}/members?relation=<relation>`."""

BuiltFromTable = Literal["code_index", "origin_catalogue", "coverage_index", "membership"]
"""The tables a node response can read, named as `/capabilities` names their builds."""

_MEMBERS_PATH = "/api/graph/node/{ref}/members"
_PROJECTED_RELATIONS: frozenset[LinkRelation] = frozenset({"members", "countries"})


def members_link(ref: str, relation: LinkRelation, *, as_of: date | None = None) -> str:
    """The list route a node's `links` names for `relation`. A group's `members` and a region's
    `countries` name the default projection, and the `as_of` date when the read named one."""
    link = f"{_MEMBERS_PATH.format(ref=ref)}?relation={relation}"
    if relation in _PROJECTED_RELATIONS:
        link += f"&projection={DEFAULT_PROJECTION.value}"
        if as_of is not None:
            link += f"&as_of={as_of.isoformat()}"
    return link


class Freshness(ResponseModel):
    """A source's own refresh cadence, read from DCP provenance arcs that imports populate
    unevenly: `dc/base/WHO_COVID19` carries both `sourceReleaseFrequency` and
    `automaticRefreshType`, `dc/base/BIS_CentralBankPolicyRate` only the second, and
    `dc/base/WHO_GHO` neither. Nullability is the contract.

    `release_frequency` is the raw `sourceReleaseFrequency` ISO 8601 duration (`P1M`);
    `release_frequency_label` is `materialise.cadence.cadence_label`'s display label for it
    (`Monthly`), or the raw token when that module does not recognise it.

    `refresh_type` is `automaticRefreshType`'s value: how this deployment refreshes the
    source, such as `GCPCloudRunAutomaticRefresh`, or the sentinel `NoAutomaticRefresh`.
    """

    release_frequency: str | None
    release_frequency_label: str | None
    refresh_type: str | None


class StatVarBlock(ResponseModel):
    """A variable's detail block: what it measures and where DFO holds observations of it."""

    model_config = ConfigDict(title="GraphStatVarBlock")

    definition: VariableDefinition
    availability: Availability | None = Field(
        description=(
            "where DFO holds observations of the variable; null, and named in gaps, when its "
            "read failed or the coverage index is not loaded"
        )
    )


class CodeSystemDeclarations(ResponseModel):
    """The rules a code system declares (`code_index.types.CodeSystemDeclaration`)."""

    model_config = ConfigDict(title="GraphCodeSystemDeclarations")

    hierarchical: bool = Field(description="whether its codes nest through specializationOf")
    disjoint_from_children: bool = Field(
        description=(
            "whether a parent code's records sit apart from its children's, so a parent's "
            "total is its own records plus its descendants' (CRS purposes); false where a "
            "parent row already totals its children (DAC1 aid types)"
        )
    )
    summable: bool = Field(description="whether sibling values can rebuild a parent total")
    absent_reading: Literal["not_declared", "aggregate"] = Field(
        description="how a variable that does not declare the property reads"
    )


class CodeSystemBlock(ResponseModel):
    """A code system's detail block."""

    model_config = ConfigDict(title="GraphCodeSystemBlock")

    property: str = Field(description="the constraint property the system governs")
    plain_name: str = Field(description="the property's plain name, such as `sector`")
    declarations: CodeSystemDeclarations
    code_count: int = Field(description="how many codes the code index holds for the system")


class CodeBlock(ResponseModel):
    """A code's detail block."""

    model_config = ConfigDict(title="GraphCodeBlock")

    code_system: NodeRef | None = Field(
        description="the system's graph node, such as `CRSPurposeCodeEnum`"
    )
    property: str = Field(description="the constraint property the code is a value of")
    code: str
    parent: NodeRef | None = Field(description="the code it specialises; null for a root")
    synonyms: list[str]


class Authority(ResponseModel):
    """Where a membership's rows come from, with the data version that reproduces them."""

    model_config = ConfigDict(title="GraphMembershipAuthority")

    name: str
    version: str


class ProjectionCount(ResponseModel):
    """One projection of a membership and its member count at the block's date."""

    model_config = ConfigDict(title="GraphProjectionCount")

    projection: Projection
    count: int | None = Field(
        description="members under the projection; null when the authority does not record it"
    )
    complete: bool = Field(
        description=(
            "false when the authority does not record the projection, or is known to hold "
            "it only in part (`membership_incomplete`)"
        )
    )


class MembershipBlock(ResponseModel):
    """A group's or region's membership under its declared authority."""

    model_config = ConfigDict(title="GraphMembershipBlock")

    authority: Authority
    as_of: date | None = Field(description="the date the read asked for; null for today")
    effective_date: date | None = Field(
        description=(
            "the latest membership change on or before the date; null for an undated membership"
        )
    )
    projections: list[ProjectionCount]


class PlaceBlock(ResponseModel):
    """A place's detail block. `membership` is set for a region the membership policy holds."""

    model_config = ConfigDict(title="GraphPlaceBlock")

    contained_in: list[NodeRef] = Field(description="the containedInPlace targets")
    membership: MembershipBlock | None


class PartyBlock(ResponseModel):
    """An organisation's or aggregate's detail block."""

    model_config = ConfigDict(title="GraphPartyBlock")

    roles: list[str] | None = Field(
        description=(
            "the role properties a node of these types can fill, by the role ranges the graph "
            "declares (rangeIncludes and its subClassOf closure); null when that read failed"
        )
    )


class PriceBasisWords(ResponseModel):
    """Current or constant prices, with the base year of constant prices when known."""

    model_config = ConfigDict(title="GraphPriceBasis")

    prices: Literal["current", "constant"]
    base_year: int | None


class UnitBlock(ResponseModel):
    """A unit's detail block: the words `unit_words` names it by."""

    model_config = ConfigDict(title="GraphUnitBlock")

    unit_class: UnitKind = Field(
        description="currency, purchasing_power, percent, or other (`unit_words.unit_class`)"
    )
    currency: str | None = Field(description="its ISO 4217 code when it is a currency")
    currency_noun: str | None = Field(
        description="the plural noun a currency reads as, such as `US dollars`"
    )
    price_basis: PriceBasisWords | None = Field(
        description="the price basis the unit itself states; null when it states none"
    )
    changes_magnitude: bool = Field(
        description="whether the unit changes what a number means, so a heading states it"
    )


class SourceBlock(ResponseModel):
    """A Source, Dataset or Provenance's detail block. `documentation` is the path of the
    node's source document on this API."""

    model_config = ConfigDict(title="GraphSourceBlock")

    publisher: NodeRef | None = Field(
        description="the source arc's target; null when the node carries no source arc"
    )
    parent: NodeRef | None = Field(
        description="the isPartOf arc's target; null when the node has no parent Dataset"
    )
    license: str | None
    license_type: str | None
    url: str | None
    source_data_url: str | None = Field(description="the sourceDataUrl arc's value")
    description_url: str | None = Field(description="the descriptionUrl arc's value")
    transformation_logic_url: str | None = Field(
        description="the dataTransformationLogic arc's value: a URL to the transformation code"
    )
    last_data_refresh_date: str | None = Field(description="the lastDataRefreshDate arc's value")
    next_data_refresh_date: str | None = Field(description="the nextDataRefreshDate arc's value")
    earliest_observation_date: str | None = Field(
        description="the earliestObservationDate arc's value"
    )
    latest_observation_date: str | None = Field(description="the latestObservationDate arc's value")
    freshness: Freshness
    data_commons: DataCommonsRef | None = Field(
        description=(
            "the Data Commons serving this provenance, from the origin catalogue; null for a "
            "provenance the catalogue does not admit, a Dataset, or a Source"
        )
    )
    documentation: str = Field(description="the path of the node's source document")


class _Envelope(ResponseModel):
    ref: str = Field(description="the node's dcid")
    label: str | None = Field(
        description=(
            "the node's name; for a Provenance, its parent Dataset's name where it has one. "
            "Null, and named in gaps, when the graph gives none"
        )
    )
    types: list[str] = Field(description="the node's typeOf values, as the graph holds them")
    description: str | None = Field(
        description="the node's prose in the graph; null when it has none"
    )
    links: dict[LinkRelation, str] = Field(
        description="each list relation the node has, mapped to its route; empty when none"
    )
    gaps: list[str] = Field(description="the member path of every value this door cannot know")
    built_from: dict[BuiltFromTable, str] = Field(
        description="the build of each table the response read"
    )


class StatVarNode(_Envelope):
    kind: Literal["stat_var"] = "stat_var"
    stat_var: StatVarBlock


class CodeSystemNode(_Envelope):
    kind: Literal["code_system"] = "code_system"
    code_system: CodeSystemBlock


class CodeNode(_Envelope):
    kind: Literal["code"] = "code"
    code: CodeBlock


class GroupNode(_Envelope):
    kind: Literal["group"] = "group"
    group: MembershipBlock


class CountryNode(_Envelope):
    kind: Literal["country"] = "country"
    country: PlaceBlock


class ContinentNode(_Envelope):
    kind: Literal["continent"] = "continent"
    continent: PlaceBlock


class SubregionNode(_Envelope):
    kind: Literal["subregion"] = "subregion"
    subregion: PlaceBlock


class ContinentalUnionNode(_Envelope):
    kind: Literal["continental_union"] = "continental_union"
    continental_union: PlaceBlock


class PlaceOtherNode(_Envelope):
    kind: Literal["place_other"] = "place_other"
    place_other: PlaceBlock


class OrganisationNode(_Envelope):
    kind: Literal["organisation"] = "organisation"
    organisation: PartyBlock


class AggregateNode(_Envelope):
    kind: Literal["aggregate"] = "aggregate"
    aggregate: PartyBlock


class UnitNode(_Envelope):
    kind: Literal["unit"] = "unit"
    unit: UnitBlock


class SourceNode(_Envelope):
    kind: Literal["source"] = "source"
    source: SourceBlock


class DatasetNode(_Envelope):
    kind: Literal["dataset"] = "dataset"
    dataset: SourceBlock


class ProvenanceNode(_Envelope):
    kind: Literal["provenance"] = "provenance"
    provenance: SourceBlock


class OtherNode(_Envelope):
    """A node no other kind fits. `types` and `description` carry what the graph holds."""

    kind: Literal["other"] = "other"


NodeResponse = Annotated[
    StatVarNode
    | CodeSystemNode
    | CodeNode
    | GroupNode
    | CountryNode
    | ContinentNode
    | SubregionNode
    | ContinentalUnionNode
    | PlaceOtherNode
    | OrganisationNode
    | AggregateNode
    | UnitNode
    | SourceNode
    | DatasetNode
    | ProvenanceNode
    | OtherNode,
    Field(discriminator="kind"),
]


class NotHeldEntry(ResponseModel):
    """A batch entry for a dcid where `GET /api/graph/node/{dcid}` answers 404
    `node_not_held`."""

    model_config = ConfigDict(title="GraphNotHeldEntry")

    ref: str
    status: Literal["not_held"] = "not_held"


NodesEntry = NodeResponse | NotHeldEntry

_Dcid = Annotated[
    str,
    StringConstraints(
        pattern=rf"^{routing.DcidConvertor.regex}$", max_length=routing.DCID_MAX_LENGTH
    ),
]


class NodesRequest(BaseModel):
    """`dcids`: 1 to `NODES_MAX_DCIDS` node dcids read as one batch, in one DCP-read
    executor job. Each dcid matches the GET route's `{dcid}` path syntax and its 512-character
    ceiling."""

    model_config = ConfigDict(extra="forbid", title="GraphNodesRequest")
    dcids: list[_Dcid] = Field(min_length=1, max_length=NODES_MAX_DCIDS)


class NodesResponse(ResponseModel):
    """`nodes` is index-aligned with the request's `dcids`: a repeated dcid repeats its
    entry. Each entry is the body `GET /api/graph/node/{dcid}` returns for that dcid, or a
    `not_held` entry where that route answers 404."""

    nodes: list[NodesEntry]
