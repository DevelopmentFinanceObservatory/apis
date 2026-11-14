"""Wire models for the deployment's schema: the document and the terms list.

`SchemaDocument` holds `generation`, the graph generation token the document was built
from, null where the token is unknown, and `nodes`, one `SchemaNode` per member of the
document, sorted by `ref`. A node, a term and a usage entry name a node by `ref`, its dcid,
and `label`, its name, as the node route's envelope does. The document holds every Class and
Property DFO declares, every node those name in
`subClassOf`, `subPropertyOf`, `domainIncludes` or `rangeIncludes`, every node a StatVar row
of the code index names, every listed instance of a Class, the `typeOf` targets of
every node that is neither a Class nor a Property, and every `subClassOf` and
`subPropertyOf` ancestor of all of these. A dcid the graph does not hold, such as the
literal range `Number`, is not a node, and the arc that names it keeps the dcid.

Members named after a Data Commons property carry that property's values in snake case:
`types` is `typeOf`, as on the node door's envelope, and `alternate_name`, `sub_class_of`,
`sub_property_of`, `specialization_of`, `domain_includes`, `range_includes` and
`classification_code` are the properties of the same names. Every list is sorted.

`other_properties` holds every other outgoing property the graph holds on the node, keyed by the
property's dcid. The ten named properties are a closed set and never appear in it; a property
promoted to a named member later leaves the map. Each value has exactly one member:
`{"ref": <string>}` for a node of the graph or `{"value": <string>}` for a literal, kept as the
string the graph holds and never parsed to a number or boolean. A value the graph repeats once
per declaring provenance appears once. Properties are sorted by dcid and values by their text, a
node reference before a literal of the same text. A literal longer than 2,000 characters, such
as a geometry, is left out, and a property keeps its first 100 values in that order. A node with
no other property holds `{}`.

`label` is `dc_graph.display_name` over the node's `name` values: the greatest by code point,
with a name equal to the dcid counting as no name, so the Graph surface names a node one way.
A term's `label` follows the same rule over the names its listing entries carry.
`description` is the smallest of the node's `description` values by code point.

`provenances` holds the dcids of the Provenances that a node's `typeOf` targets carry, verbatim
and sorted. A term's are those its listing entries carry. `declared_by` is the origin that
declares the node, by `graph_api.schema_document.declared_by`: `"dfo"` when any of its
provenances is DFO's, `"base"` when every one is base Data Commons', and `"un_system"`
otherwise. It is null when the graph names no declaring Provenance, which happens when a node has no `typeOf` arc, when none of its `typeOf` targets
carries one, or when a term's listing entries carry none. Its `provenances` is then empty.

`instances` is set on a Class that lists the nodes typed by it or by any of its subclasses:
every DFO-declared Class whose `subClassOf` chain reaches `Enumeration` or that the
`rangeIncludes` of a DFO-declared Property names, every DFO-declared subclass of one, and every
base Class whose chain reaches `Enumeration`. A DFO-declared Class lists, for itself and each
DFO-declared subclass, the subjects its `<-typeOf` listing read, up to a cap, and a base Class
or base subclass only the nodes StatVar rows use. `complete` is true only when every one of
those classes is DFO-declared and its listing read every subject, so `refs` is every instance
the graph holds. A list where it is false holds some of them. `instances` is null on every
other node, `Enumeration` and base ranges such as `Thing` included, and on a Class whose list
holds no node and is not complete.

`SchemaTerms` holds `generation`, null where the token is unknown, and `terms`, every Class
and Property on the graph with its `ref`, `label`, `types` (`Class`, `Property` or both),
`provenances` and `declared_by`, sorted by `ref`.

`DatasetUsage` says how each Provenance of the code index uses the model
(`graph_api.dataset_usage`): its dataset label and description, its StatVar count, the distinct
patterns of its StatVar rows, the values each constraint property takes, and the facet facts of
its coverage series.

`SchemaCheckRow` is a candidate StatisticalVariable definition and `SchemaFinding` one
fault `graph_api.schema_check` finds in it. `SchemaCheckRequest` carries 1 to
`CHECK_MAX_ROWS` rows to `POST /api/graph/schema/check`. Every string of a row is bounded at
512 characters, and an empty string counts as absent. A row names at most
`CHECK_MAX_CONSTRAINT_PROPERTIES` constraint properties and `CHECK_MAX_OBSERVATION_PROPERTIES`
observation properties. A constraint value is a string, a finite number within plus or minus
2**53, a boolean or null. The request model checks only these bounds; the check reports
everything else about a row's content as a finding.

`SchemaCheckResponse` answers with the generation the rows were checked against, null where
its token is unknown, `rows_checked`, the count of rows checked from the first, and every
finding in the rows checked. A caller resumes at `rows_checked` when it is less than the rows
sent. The check stops before the row that would take
its reads from the graph, ancestry included, past `CHECK_MAX_READ_IDS` distinct ids, its
findings past
`CHECK_MAX_FINDINGS` or its encoded body past `CHECK_MAX_BODY_BYTES`; the first row is always
taken, so every response checks at least one row.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
)

from dfo_platform.wire import ResponseModel
from graph_api.routing import DCID_MAX_LENGTH
from origin_catalogue import OriginId

CHECK_MAX_ROWS = 2_000
"""The most rows one `POST /api/graph/schema/check` request carries."""

CHECK_MAX_CONSTRAINT_PROPERTIES = 16
"""The most constraint properties one check row names, twice the most a StatVar row of the
served code index carries (7 on the dev graph)."""

CHECK_MAX_OBSERVATION_PROPERTIES = 4
"""The most observation properties one check row names, twice the most a StatVar row of the
served code index carries (2 on the dev graph)."""

CHECK_MAX_ROW_READ_IDS = 5 + 2 * CHECK_MAX_CONSTRAINT_PROPERTIES + CHECK_MAX_OBSERVATION_PROPERTIES
"""The most distinct ids one check row reads from the graph, 41: its 5 shape fields, each
constraint property and its value, and each observation property."""

CHECK_MAX_READ_IDS = 200
"""The most distinct ids one check reads from the graph, an internal budget the wire does not
publish. It counts the ids the rows name that the schema document does not hold, each
constraint value counted unless it is a literal or its property a document node that is no
Property, before any read (`graph_api.schema_check.plan_reads`), and the `subClassOf`
parents the range rule then reads for value types outside the document. The check takes rows
from the first while their ids stay within the budget, stops before the first row whose
ancestry would pass it, and answers where to resume. The budget is at least
`CHECK_MAX_ROW_READ_IDS`, so every check takes at least one row.

The check reads its planned ids in one `node_arcs_many` call, and each level of ancestry in
one more (`graph_api.schema_check` counts the POSTs): a batch POST per 50 ids, then each id
answered empty read alone, 16 at a time. The worst check names 200 ids the graph does not
hold, 204 POSTs in 13 rounds of 16. On the dev graph that took 2.6 to 4.4 s over eight runs
(POST median 170 to 210 ms), against 7.8 to 8.7 s for 41 ids read one at a time before the probes ran concurrently. Two checks at once, or other
node readers holding the client's 32 node slots, can halve the width to about 9 s, and a wait
for a cold schema build adds about 6.5 s, which leaves the worst check near 16 s, half of the
30 s `GRAPH_SCHEMA_CHECK_TIMEOUT_SECONDS`. The other half covers 5xx retries (0.5 s and 2 s
backoffs) and slower graph answers."""

CHECK_MAX_FINDINGS = 2_000
"""The most findings one check answers, an internal cap the wire does not publish. The check
stops before the row whose findings would pass it and answers where to resume. One row yields at most 79: 5 on its shape fields, 2 on each of its 16 constraint properties (a
domain warning and a value finding), 1 on each of its 4 observation properties, and 38
`definition_differs` warnings (5 shape fields, each constraint property the row or a served
StatVar of at most `CHECK_MAX_CONSTRAINT_PROPERTIES` names, the observation properties). A
row whose `dcid` is outside dcid syntax gets one finding on it and no `definition_differs`.
Every check takes at least one row."""

CHECK_MAX_BODY_BYTES = 4_000_000
"""The most bytes one check response encodes to, an internal budget the wire does not publish.
The check stops before the row whose findings would take the response past it and answers where
to resume. The first row is always taken, and fits: its at most 79 findings would encode to
2.20 MB even if each carried a 512-character `property` and four 512-character strings in
both `value` and `served`, every character escaped to six bytes (27,829 bytes a finding)."""

type DeclaredBy = OriginId
"""The origin that declares a node: `"dfo"`, `"base"` or `"un_system"`."""


class SchemaInstances(ResponseModel):
    """The nodes typed by a Class or by any of its subclasses."""

    refs: list[str] = Field(description="the dcids of the instances listed, sorted")
    complete: bool = Field(
        description="true when `refs` is every instance the graph holds: the class and each of "
        "its subclasses were listed from the graph and each listing read all of them; false "
        "when it holds some of them"
    )


class SchemaNodeReference(ResponseModel):
    """A value that names a node of the graph, by its dcid."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ref: str = Field(description="the node's dcid")


class SchemaLiteral(ResponseModel):
    """A literal value as the graph holds it, a string even where it reads as a number, date or
    boolean."""

    model_config = ConfigDict(extra="forbid", strict=True)

    value: str


class SchemaNode(ResponseModel):
    """One node of the schema document."""

    ref: str = Field(description="the node's dcid")
    label: str | None = Field(description="the node's name; null when the graph gives none")
    description: str | None = Field(
        description="the node's prose in the graph; null when it has none"
    )
    types: list[str] = Field(description="the node's typeOf values")
    alternate_name: list[str]
    sub_class_of: list[str]
    sub_property_of: list[str]
    specialization_of: list[str]
    domain_includes: list[str]
    range_includes: list[str]
    classification_code: list[str]
    other_properties: dict[str, list[SchemaNodeReference | SchemaLiteral]] = Field(
        description="every other outgoing property the graph holds on the node, by property "
        "dcid, sorted; the named members are a closed set, and a property promoted to one "
        "leaves this map. Each value has exactly one member, {ref: string} for a node or "
        "{value: string} for a literal, never parsed to a number or boolean, sorted by its "
        "text. A literal longer than 2,000 characters is left out, and a property keeps its "
        "first 100 values. {} when the node holds none"
    )
    provenances: list[str] = Field(
        description="the Provenances declaring the node's typeOf arcs, verbatim, empty when "
        "declared_by is null"
    )
    declared_by: DeclaredBy | None = Field(
        description="the origin declaring the node: dfo when any of its provenances is DFO's, "
        "base when every one is base Data Commons', un_system otherwise; null when the graph "
        "names no declaring Provenance for the node: it has no typeOf arc, or none of its "
        "typeOf targets carries one"
    )
    instances: SchemaInstances | None = Field(
        description="the nodes typed by the Class or any of its subclasses, on a DFO Class that "
        "descends from Enumeration or that a DFO property ranges on, a DFO subclass of one, or "
        "a base enumeration; null on every other node and on a Class whose list holds no node "
        "and is not complete"
    )


class SchemaDocument(ResponseModel):
    """The deployment's model as one document."""

    generation: str | None = Field(
        description="the graph generation the document was built from; null when unknown"
    )
    nodes: list[SchemaNode]


class SchemaTerm(ResponseModel):
    """One Class or Property on the graph."""

    ref: str = Field(description="the term's dcid")
    label: str | None = Field(description="the term's name; null when the graph gives none")
    types: list[Literal["Class", "Property"]] = Field(
        description="whether the term is typed Class, Property or both"
    )
    provenances: list[str] = Field(
        description="the Provenances declaring the term's typeOf arc, verbatim, empty when "
        "declared_by is null"
    )
    declared_by: DeclaredBy | None = Field(
        description="the origin declaring the term, by the node's rule; null when the graph "
        "names no declaring Provenance for the term: its listing entries carry none"
    )


class SchemaTerms(ResponseModel):
    """Every Class and Property the DFO graph lists by `typeOf`, base Data Commons terms
    included. The UN origin's terms are not listed, even where the deployment serves it."""

    generation: str | None = Field(
        description="the graph generation the list was built from; null when unknown"
    )
    terms: list[SchemaTerm]


class SchemaPattern(ResponseModel):
    """One distinct way a Provenance's StatVars use the model."""

    population_type: str | None
    measured_property: str
    stat_type: str | None
    measurement_qualifier: str | None
    measurement_denominator: str | None
    constraint_properties: list[str] = Field(
        description="the names of the constraint properties the StatVars set, sorted"
    )
    observation_properties: list[str] = Field(
        description="the observation properties the StatVars declare, sorted"
    )
    stat_var_count: int = Field(description="how many of the Provenance's StatVars follow it")
    examples: list[str] = Field(
        description="the first three of those StatVars' dcids in sorted order"
    )


class SchemaValueCount(ResponseModel):
    """One value a constraint property takes, and how many StatVars take it."""

    value: str
    stat_var_count: int


class SchemaConstraintValues(ResponseModel):
    """The values one constraint property takes over a Provenance's StatVars."""

    property: str
    values: list[SchemaValueCount] = Field(description="sorted by value")


class SchemaFacets(ResponseModel):
    """The facet facts of a Provenance's coverage series, each list sorted and distinct."""

    units: list[str]
    scaling_factors: list[str]
    measurement_methods: list[str]
    observation_periods: list[str] = Field(description="ISO 8601 durations, such as `P1Y`")
    currencies: list[str] = Field(description="ISO 4217 codes of the currency units")


class ProvenanceUsage(ResponseModel):
    """How one Provenance's StatVars use the model."""

    ref: str = Field(description="the Provenance's dcid")
    label: str | None = Field(
        description="the dataset's name, from its declaration; null when it has none"
    )
    description: str | None = Field(
        description="what the dataset measures, from its declaration; null when it has none"
    )
    stat_var_count: int
    patterns: list[SchemaPattern] = Field(
        description="sorted by StatVar count descending, then by the pattern's fields"
    )
    constraint_values: list[SchemaConstraintValues] = Field(
        description="one entry per constraint property the StatVars set, sorted by property"
    )
    facets: SchemaFacets = Field(
        description="facts of the coverage facets; empty lists when coverage holds none"
    )


class DatasetUsage(ResponseModel):
    """How each Provenance of the code index uses the model."""

    provenances: list[ProvenanceUsage] = Field(description="sorted by ref")


CONSTRAINT_NUMBER_BOUND = 2**53
"""The largest magnitude of a number a constraint value takes: every integer up to it is exact
as a JSON number in any client."""


def _refuse_integer(value: object) -> object:
    """Keep an integer out of the float branch, which would round one past
    `CONSTRAINT_NUMBER_BOUND` into it."""
    if isinstance(value, int) and not isinstance(value, bool):
        raise ValueError(f"an integer is within plus or minus {CONSTRAINT_NUMBER_BOUND}")
    return value


type ConstraintValue = (
    Annotated[str, StringConstraints(max_length=DCID_MAX_LENGTH)]
    | Annotated[int, Field(ge=-CONSTRAINT_NUMBER_BOUND, le=CONSTRAINT_NUMBER_BOUND)]
    | Annotated[
        float,
        BeforeValidator(_refuse_integer),
        Field(allow_inf_nan=False, ge=-CONSTRAINT_NUMBER_BOUND, le=CONSTRAINT_NUMBER_BOUND),
    ]
    | bool
    | None
)
"""A constraint value as sent: a node dcid, or a literal or null the check reports. A string
takes the 512-character dcid ceiling, and a number, integer or float, is finite and within
plus or minus `CONSTRAINT_NUMBER_BOUND`."""

FindingCode = Literal[
    "node_not_held",
    "not_a_node",
    "node_provisional",
    "node_untyped",
    "origin_not_served",
    "value_missing",
    "not_a_property",
    "not_a_class",
    "value_outside_range",
    "property_outside_domain",
    "definition_differs",
]
"""What a finding says about the cell it names.

- `node_not_held`: it is a well-formed dcid the graph holds no node for.
- `not_a_node`: a node belongs in the cell, which holds a number, a boolean or a string outside
  dcid syntax.
- `node_provisional`: the node of a constraint value, `stat_type`, `measurement_qualifier` or
  `measurement_denominator` has only the type `ProvisionalNode`.
- `node_untyped`: the node of a constraint value, `stat_type`, `measurement_qualifier` or
  `measurement_denominator` has no `typeOf`.
- `origin_not_served`: it is an id of the UN origin, which this deployment does not serve, so
  no read can establish whether the graph holds it.
- `value_missing`: the measured property is absent, or a constraint property has a null value.
- `not_a_property`: a constraint property, the measured property or an observation property
  is not typed `Property`.
- `not_a_class`: the population type is not typed `Class`.
- `value_outside_range`: every type of the value is a document node and none reaches the
  DFO-declared property's `rangeIncludes` by `subClassOf`.
- `property_outside_domain`: the population type is a document node that reaches none of the
  DFO-declared property's `domainIncludes` by `subClassOf`.
- `definition_differs`: the code index serves the row's dcid with another value in that cell,
  and `served` is the served value.

A constraint value whose property is not a held Property gets no finding of its own.
"""

ShapeField = Literal[
    "population_type",
    "measured_property",
    "stat_type",
    "measurement_qualifier",
    "measurement_denominator",
]
"""A StatVar shape field."""

FindingField = Literal["dcid", ShapeField, "constraint_properties", "observation_properties"]
"""The row member a finding is about."""

type _Text = Annotated[str, StringConstraints(max_length=DCID_MAX_LENGTH)]


def _blank_is_absent(value: str | None) -> str | None:
    return value or None


type _OptionalText = Annotated[_Text | None, AfterValidator(_blank_is_absent)]


def _blank_value_is_absent(values: dict[str, ConstraintValue]) -> dict[str, ConstraintValue]:
    return {prop: None if value == "" else value for prop, value in values.items()}


def _drop_blank(values: list[str]) -> list[str]:
    return [value for value in values if value]


class SchemaCheckRow(BaseModel):
    """A candidate StatisticalVariable definition. Each string is at most 512 characters, and
    an empty string counts as absent."""

    model_config = ConfigDict(extra="forbid", title="GraphSchemaCheckRow")

    dcid: _OptionalText = Field(
        default=None,
        description="the StatVar's dcid; when the code index serves it, the row is compared",
    )
    population_type: _OptionalText = None
    measured_property: _OptionalText = Field(
        default=None, description="absent or empty gets a `value_missing` finding"
    )
    stat_type: _OptionalText = None
    measurement_qualifier: _OptionalText = None
    measurement_denominator: _OptionalText = None
    constraint_properties: Annotated[
        dict[_Text, ConstraintValue], AfterValidator(_blank_value_is_absent)
    ] = Field(
        default_factory=dict,
        max_length=CHECK_MAX_CONSTRAINT_PROPERTIES,
        description="each constraint property and its value node; an empty value counts as null",
    )
    observation_properties: Annotated[list[_Text], AfterValidator(_drop_blank)] = Field(
        default_factory=list, max_length=CHECK_MAX_OBSERVATION_PROPERTIES
    )


class SchemaFinding(ResponseModel):
    """One fault in one checked row."""

    row: int = Field(description="the row's index in the request's `rows`")
    field: FindingField = Field(description="the row member the finding is about")
    property: str | None = Field(
        description="the constraint or observation property the finding is about; null for the "
        "other members"
    )
    subject: Literal["property", "value"] = Field(
        description="`property` when the finding judges a constraint or observation property "
        "itself, `value` otherwise"
    )
    value: ConstraintValue | list[str] = Field(
        description="what the row sent in the judged cell: the property's dcid when `subject` is "
        "`property`, a list for `observation_properties` on `definition_differs`, null where the "
        "row sent none"
    )
    served: ConstraintValue | list[str] = Field(
        description="for `definition_differs`, the value the code index serves (a sorted list "
        "for `observation_properties`, null where it serves none); null for every other code"
    )
    code: FindingCode
    severity: Literal["error", "warning"]


class SchemaCheckRequest(BaseModel):
    """`rows`: 1 to `CHECK_MAX_ROWS` candidate StatVar definitions, checked in order from the
    first."""

    model_config = ConfigDict(extra="forbid", title="GraphSchemaCheckRequest")

    rows: list[SchemaCheckRow] = Field(min_length=1, max_length=CHECK_MAX_ROWS)


class SchemaCheckResponse(ResponseModel):
    """The findings over the rows one check took, from the first."""

    generation: str | None = Field(
        description="the graph generation the rows were checked against; null when unknown"
    )
    rows_checked: int = Field(
        description="how many rows were checked, counted from the first row of the request; at "
        "least 1. A caller resumes at this index when it is less than the rows sent"
    )
    findings: list[SchemaFinding] = Field(
        description="one finding per fault in the rows checked, in row order; empty when none of "
        "them has a fault"
    )
