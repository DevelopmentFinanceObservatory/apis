"""Wire models for the instance's ontology: the document and the terms list.

`OntologyDocument` holds `generation`, the graph generation token the document was built
from, null where the token is unknown, and `nodes`, one `OntologyNode` per member of the
document, sorted by dcid. The document holds every Class and Property the instance declares, every node those name in
`subClassOf`, `subPropertyOf`, `domainIncludes` or `rangeIncludes`, every node a StatVar row
of the code index names, every listed member of an enumeration, the `typeOf` targets of
every node that is neither a Class nor a Property, and every `subClassOf` and
`subPropertyOf` ancestor of all of these. A dcid the graph does not hold, such as the
literal range `Number`, is not a node, and the arc that names it keeps the dcid.

Members named after a Data Commons property carry that property's values in snake case:
`types` is `typeOf`, as on the node door's envelope, and `alternate_name`, `sub_class_of`,
`sub_property_of`, `specialization_of`, `domain_includes`, `range_includes` and
`classification_code` are the properties of the same names. Every list is sorted.

`name` is `dc_graph.display_name` over the node's `name` values: the greatest by code point,
with a name equal to the dcid counting as no name, so the Graph surface names a node one way.
A term's `name` follows the same rule over the names its listing entries carry.
`description` is the smallest of the node's `description` values by code point.

`provenances` holds the dcids of the Provenances that a node's `typeOf` targets carry, verbatim
and sorted. A term's are those its listing entries carry. `declared_by` is `"base"` when every
one starts `dc/base/`, and `"instance"` otherwise. It is null when the graph names no declaring
Provenance, which happens when a node has no `typeOf` arc, when none of its `typeOf` targets
carries one, or when a term's listing entries carry none. Its `provenances` is then empty.

`enumeration` is set on a Class whose `subClassOf` chain reaches `Enumeration`, and null on
every other node. An instance-declared enumeration lists the members its `<-typeOf` listing
read, up to a cap; a base enumeration lists only the members StatVar rows use. `complete` is
true only when the build listed the class from the graph and the listing read every member,
so `members` is every member the graph holds. A list where it is false holds some of them.

`OntologyTerms` holds `generation`, null where the token is unknown, and `terms`, every Class
and Property on the graph with its `name`, `types` (`Class`, `Property` or both),
`provenances` and `declared_by`, sorted by dcid.

`OntologyProvenances` says how each Provenance of the code index uses the model
(`graph_api.ontology_provenances`): its dataset label and description, its StatVar count, the
distinct patterns of its StatVar rows, the values each constraint property takes, and the facet
facts of its coverage series.

`OntologyCheckRow` is a candidate StatisticalVariable definition and `OntologyFinding` one
fault `graph_api.ontology_check` finds in it. `OntologyCheckRequest` carries 1 to
`CHECK_MAX_ROWS` rows to `POST /api/graph/ontology/check`, and `OntologyCheckResponse` answers
with the generation the rows were checked against, null where its token is unknown, and every
finding. A row's shape fields, constraint properties and observation properties take the node
door's dcid syntax (`graph_api.node_models.Dcid`). A constraint value is any string up to the
same 512-character ceiling, a finite number within plus or minus 2**53, a boolean or null, and
`ref` takes the same ceiling. A row names at most `CHECK_MAX_CONSTRAINT_PROPERTIES` constraint
properties and `CHECK_MAX_OBSERVATION_PROPERTIES` observation properties. One request reads at
most `CHECK_MAX_READ_IDS` distinct ids from the graph (`graph_api.ontology_check`), a cap no
lower than the most one row names, and answers at most `CHECK_MAX_FINDINGS` findings, a cap
above what one row yields.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

from dfo_platform.wire import ResponseModel
from graph_api.node_models import Dcid
from graph_api.routing import DCID_MAX_LENGTH

CHECK_MAX_ROWS = 2_000
"""The most rows one `POST /api/graph/ontology/check` request carries."""

CHECK_MAX_CONSTRAINT_PROPERTIES = 16
"""The most constraint properties one check row names, twice the most a StatVar row of the
served code index carries (7 on the dev graph)."""

CHECK_MAX_OBSERVATION_PROPERTIES = 4
"""The most observation properties one check row names, twice the most a StatVar row of the
served code index carries (2 on the dev graph)."""

CHECK_MAX_READ_IDS = 5 + 2 * CHECK_MAX_CONSTRAINT_PROPERTIES + CHECK_MAX_OBSERVATION_PROPERTIES
"""The most distinct ids one check request may read from the graph, 41. They are the ids its
rows name that the ontology document does not hold, counted before any read, each constraint
value counted unless it is a literal or its property a document node that is no Property
(`graph_api.ontology_check.plan_reads`).
One row names at most this many (its 5 shape fields, each constraint property and its value,
each observation property), so a single row always fits. The check reads each once, in one
`node_arcs_many` call: at most 43 POSTs to `/v2/node` when the graph answers every POST, 45
where the deployment serves the UN origin, and at most 203 when the front door refuses every
POST naming more than one id, each tried up to 3 times, all within the 30 s
`GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS` (`graph_api.ontology_check` counts them). On the dev
graph, 41 absent ids took 43 POSTs and 7.4 to 8.5 s, under a third of that deadline."""

CHECK_MAX_FINDINGS = 2_000
"""The most findings one check answers. A check whose findings pass it stops and is refused.
One row yields at most 79: 5 on its shape fields, 2 on each of its 16 constraint properties (a
domain warning and a value finding), 1 on each of its 4 observation properties, and 38
`definition_differs` warnings (5 shape fields, each constraint property the row or a served
StatVar of at most `CHECK_MAX_CONSTRAINT_PROPERTIES` names, the observation properties). A
single row always fits."""

DeclaredBy = Literal["instance", "base"]
"""Who declares a node: `"base"` when every declaring provenance is a base Data Commons one."""


class OntologyEnumeration(ResponseModel):
    """The members of an enumeration Class."""

    members: list[str] = Field(description="the dcids of the members listed, sorted")
    complete: bool = Field(
        description="true when `members` is every member the graph holds: the class was listed "
        "from the graph and the listing read all of them; false when it holds some of them"
    )


class OntologyNode(ResponseModel):
    """One node of the ontology document."""

    dcid: str
    name: str | None = Field(description="the node's name; null when the graph gives none")
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
    provenances: list[str] = Field(
        description="the Provenances declaring the node's typeOf arcs, verbatim, empty when "
        "declared_by is null"
    )
    declared_by: DeclaredBy | None = Field(
        description="null when the graph names no declaring Provenance for the node: it has no "
        "typeOf arc, or none of its typeOf targets carries one"
    )
    enumeration: OntologyEnumeration | None = Field(
        description="the members when the node is an enumeration Class; null otherwise"
    )


class OntologyDocument(ResponseModel):
    """The instance's model as one document."""

    generation: str | None = Field(
        description="the graph generation the document was built from; null when unknown"
    )
    nodes: list[OntologyNode]


class OntologyTerm(ResponseModel):
    """One Class or Property on the graph."""

    dcid: str
    name: str | None = Field(description="the term's name; null when the graph gives none")
    types: list[Literal["Class", "Property"]] = Field(
        description="whether the term is typed Class, Property or both"
    )
    provenances: list[str] = Field(
        description="the Provenances declaring the term's typeOf arc, verbatim, empty when "
        "declared_by is null"
    )
    declared_by: DeclaredBy | None = Field(
        description="null when the graph names no declaring Provenance for the term: its "
        "listing entries carry none"
    )


class OntologyTerms(ResponseModel):
    """Every Class and Property the DFO graph lists by `typeOf`, base Data Commons terms
    included. The UN origin's terms are not listed, even where the deployment serves it."""

    generation: str | None = Field(
        description="the graph generation the list was built from; null when unknown"
    )
    terms: list[OntologyTerm]


class OntologyPattern(ResponseModel):
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


class OntologyValueCount(ResponseModel):
    """One value a constraint property takes, and how many StatVars take it."""

    value: str
    stat_var_count: int


class OntologyConstraintValues(ResponseModel):
    """The values one constraint property takes over a Provenance's StatVars."""

    property: str
    values: list[OntologyValueCount] = Field(description="sorted by value")


class OntologyFacets(ResponseModel):
    """The facet facts of a Provenance's coverage series, each list sorted and distinct."""

    units: list[str]
    scaling_factors: list[str]
    measurement_methods: list[str]
    observation_periods: list[str] = Field(description="ISO 8601 durations, such as `P1Y`")
    currencies: list[str] = Field(description="ISO 4217 codes of the currency units")


class OntologyProvenance(ResponseModel):
    """How one Provenance's StatVars use the model."""

    dcid: str
    label: str | None = Field(
        description="the dataset's name, from its declaration; null when it has none"
    )
    description: str | None = Field(
        description="what the dataset measures, from its declaration; null when it has none"
    )
    stat_var_count: int
    patterns: list[OntologyPattern] = Field(
        description="sorted by StatVar count descending, then by the pattern's fields"
    )
    constraint_values: list[OntologyConstraintValues] = Field(
        description="one entry per constraint property the StatVars set, sorted by property"
    )
    facets: OntologyFacets = Field(
        description="facts of the coverage facets; empty lists when coverage holds none"
    )


class OntologyProvenances(ResponseModel):
    """How each Provenance of the code index uses the model."""

    provenances: list[OntologyProvenance] = Field(description="sorted by dcid")


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
"""A constraint value as sent: a node dcid, or a literal or null the check refuses. A string
takes the 512-character dcid ceiling, and a number, integer or float, is finite and within
plus or minus `CONSTRAINT_NUMBER_BOUND`."""

FindingCode = Literal[
    "node_not_held",
    "node_provisional",
    "node_untyped",
    "value_missing",
    "not_a_property",
    "not_a_class",
    "value_outside_range",
    "property_outside_domain",
    "definition_differs",
]
"""What a finding says about the value it names.

- `node_not_held`: the graph holds no node for it, or a constraint value is a number, a
  boolean or a string outside dcid syntax.
- `node_provisional`: the node of a constraint value, `stat_type`, `measurement_qualifier` or
  `measurement_denominator` has only the type `ProvisionalNode`.
- `node_untyped`: the node of a constraint value, `stat_type`, `measurement_qualifier` or
  `measurement_denominator` has no `typeOf`.
- `value_missing`: a constraint property has a null value.
- `not_a_property`: a constraint property, the measured property or an observation property
  is not typed `Property`.
- `not_a_class`: the population type is not typed `Class`.
- `value_outside_range`: every type of the value is a document node and none reaches the
  instance property's `rangeIncludes` by `subClassOf`.
- `property_outside_domain`: the population type is a document node that reaches none of the
  instance property's `domainIncludes` by `subClassOf`.
- `definition_differs`: the code index serves the row's dcid with another value in that field,
  and `value` is the served value.

A constraint value whose property is not a held Property gets no finding of its own.
"""


class OntologyCheckRow(BaseModel):
    """A candidate StatisticalVariable definition."""

    model_config = ConfigDict(extra="forbid", title="GraphOntologyCheckRow")

    ref: str | None = Field(
        default=None,
        max_length=DCID_MAX_LENGTH,
        description="the caller's label, echoed on findings; at most 512 characters",
    )
    dcid: Dcid | None = Field(
        default=None,
        description="the StatVar's dcid; when the code index serves it, the row is compared",
    )
    population_type: Dcid | None = None
    measured_property: Dcid
    stat_type: Dcid | None = None
    measurement_qualifier: Dcid | None = None
    measurement_denominator: Dcid | None = None
    constraint_properties: dict[Dcid, ConstraintValue] = Field(
        default_factory=dict,
        max_length=CHECK_MAX_CONSTRAINT_PROPERTIES,
        description="each constraint property and its value node",
    )
    observation_properties: list[Dcid] = Field(
        default_factory=list, max_length=CHECK_MAX_OBSERVATION_PROPERTIES
    )


class OntologyFinding(ResponseModel):
    """One fault in one checked row."""

    ref: str | None = Field(description="the row's `ref`")
    field: str = Field(
        description=(
            "the row member: `population_type`, `measured_property`, `stat_type`, "
            "`measurement_qualifier`, `measurement_denominator`, `observation_properties`, or "
            "`constraint_properties.<property>` for a constraint property or its value"
        )
    )
    value: ConstraintValue | list[str] = Field(
        description=(
            "the value at fault; for `definition_differs`, the value the code index serves "
            "(a sorted list for `observation_properties`, null where it serves none)"
        )
    )
    code: FindingCode
    severity: Literal["error", "warning"]


class OntologyCheckRequest(BaseModel):
    """`rows`: 1 to `CHECK_MAX_ROWS` candidate StatVar definitions, checked as one batch."""

    model_config = ConfigDict(extra="forbid", title="GraphOntologyCheckRequest")

    rows: list[OntologyCheckRow] = Field(min_length=1, max_length=CHECK_MAX_ROWS)


class OntologyCheckResponse(ResponseModel):
    """The findings over every row of one check."""

    generation: str | None = Field(
        description="the graph generation the rows were checked against; null when unknown"
    )
    row_count: int = Field(description="how many rows were checked")
    findings: list[OntologyFinding] = Field(
        description="one finding per fault, in row order; empty when no row has a fault"
    )
