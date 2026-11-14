"""`check_rows` against the schema fixture graph, extended with the nodes the rules need."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

import pytest

from code_index import StatVarRow
from dc_graph import ArcTarget, GraphReadError
from graph_api import schema_check
from graph_api.reader import NodeArcsByDcid
from graph_api.schema_check import CheckedRows, SchemaLookup, check_rows, plan_reads
from graph_api.schema_document import Schema, build_schema
from graph_api.schema_models import (
    CHECK_MAX_BODY_BYTES,
    CHECK_MAX_CONSTRAINT_PROPERTIES,
    CHECK_MAX_FINDINGS,
    CHECK_MAX_OBSERVATION_PROPERTIES,
    CHECK_MAX_READ_IDS,
    SchemaCheckResponse,
    SchemaCheckRow,
)

from .conftest import FakeGraphNodeReader
from .schema_fixture import (
    CATALOGUE,
    DFO,
    INDEX,
    NODES,
    SCHEMA,
    TERMS,
    index_of,
    index_row,
    node_targets,
    schema_graph,
)

_DEADLINE = 1_000.0

_CHECK_TERMS = {
    **TERMS,
    "DAC1AidTypeEnum": ("Class", (DFO,)),
    "Person": ("Class", (SCHEMA,)),
    "Place": ("Class", (SCHEMA,)),
    "AdministrativeArea": ("Class", (SCHEMA,)),
    "Country": ("Class", (SCHEMA,)),
    "gender": ("Property", (SCHEMA,)),
    "recipientPlace": ("Property", (DFO,)),
    "financingPurpose": ("Property", (DFO,)),
    "purposeOrPlace": ("Property", (DFO,)),
    "recipientGender": ("Property", (DFO,)),
    "BeneficiaryGroupEnum": ("Class", (DFO,)),
    "beneficiaryGroup": ("Property", (DFO,)),
}

_CHECK_NODES = {
    **NODES,
    # A base property whose domain excludes FinancialFlow: base domains are not checked.
    "amount": {
        "typeOf": ("Property",),
        "rangeIncludes": ("Quantity",),
        "domainIncludes": ("Person",),
    },
    "DAC1AidTypeEnum": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)},
    "DAC1AidType1010": {"typeOf": ("DAC1AidTypeEnum",)},
    "Person": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "Place": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "AdministrativeArea": {"typeOf": ("Class",), "subClassOf": ("Place",)},
    "Country": {"typeOf": ("Class",), "subClassOf": ("AdministrativeArea",)},
    "country/UGA": {"typeOf": ("Country",)},
    "gender": {
        "typeOf": ("Property",),
        "domainIncludes": ("Person",),
        "rangeIncludes": ("GenderType",),
    },
    "recipientPlace": {
        "typeOf": ("Property",),
        "domainIncludes": ("FinancingEvent",),
        "rangeIncludes": ("Place",),
    },
    "financingPurpose": {
        "typeOf": ("Property",),
        "domainIncludes": ("FinancingEvent",),
        "rangeIncludes": ("FinancingPurposeEnum",),
    },
    "purposeOrPlace": {
        "typeOf": ("Property",),
        "domainIncludes": ("FinancingEvent",),
        "rangeIncludes": ("CRSPurposeCodeEnum", "Place"),
    },
    # A DFO property ranged on the base enumeration, whose member list is incomplete.
    "recipientGender": {
        "typeOf": ("Property",),
        "domainIncludes": ("FinancingEvent",),
        "rangeIncludes": ("GenderType",),
    },
    # A complete DFO enumeration (it has no member of its own) above the base one.
    "BeneficiaryGroupEnum": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)},
    "GenderType": {"typeOf": ("Class",), "subClassOf": ("BeneficiaryGroupEnum",)},
    "beneficiaryGroup": {
        "typeOf": ("Property",),
        "domainIncludes": ("FinancingEvent",),
        "rangeIncludes": ("BeneficiaryGroupEnum",),
    },
    # Held base properties outside the document, for a row naming the most ids a row can.
    **{
        f"baseProperty{i}": {"typeOf": ("Property",)}
        for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES)
    },
    "PlaceholderCode": {"typeOf": ("ProvisionalNode",)},
    "RealizedCode": {"typeOf": ("ProvisionalNode", "CRSPurposeCodeEnum")},
    "UntypedCode": {"name": ("Untyped code",)},
}


class _RecordingReader(FakeGraphNodeReader):
    """The check graph, recording the dcids, the properties and the probe budget of every
    `node_arcs_many` call."""

    def __init__(self) -> None:
        super().__init__(nodes=_CHECK_NODES)
        self.batches: list[list[str]] = []
        self.asked: list[tuple[tuple[str, ...], int | None]] = []

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> NodeArcsByDcid:
        self.batches.append(list(dcids))
        self.asked.append((tuple(properties), individual_probe_budget))
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


@cache
def _schema() -> Schema:
    graph = schema_graph(terms=_CHECK_TERMS, nodes=_CHECK_NODES)
    return build_schema(
        reader=graph, code_index=INDEX, catalogue=CATALOGUE, generation="g1", deadline=_DEADLINE
    )


@cache
def _lookup() -> SchemaLookup:
    return SchemaLookup.of(_schema().document)


def _row(**fields: Any) -> SchemaCheckRow:
    return SchemaCheckRow.model_validate(
        {
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            **fields,
        }
    )


def _as_row(served: StatVarRow, **fields: Any) -> SchemaCheckRow:
    return SchemaCheckRow.model_validate(
        {
            "dcid": served.dcid,
            "population_type": served.population_type,
            "measured_property": served.measured_property,
            "stat_type": served.stat_type,
            "measurement_qualifier": served.measurement_qualifier,
            "measurement_denominator": served.measurement_denominator,
            "constraint_properties": dict(served.constraint_properties),
            "observation_properties": list(served.observation_properties),
            **fields,
        }
    )


def _checked(
    rows: Sequence[SchemaCheckRow],
    reader: FakeGraphNodeReader | None = None,
    lookup: SchemaLookup | None = None,
    *,
    un_served: bool = True,
) -> CheckedRows:
    """`check_rows` over the rows `plan_reads` takes."""
    lookup = lookup or _lookup()
    plan = plan_reads(rows, lookup=lookup, un_served=un_served)
    return check_rows(
        rows[: plan.row_count],
        reads=plan.ids,
        lookup=lookup,
        reader=reader or _RecordingReader(),
        code_index=INDEX,
        un_served=un_served,
        generation="g1",
        deadline=_DEADLINE,
    )


type _Finding = tuple[str, str | None, str, object, object, str]


def _check(
    rows: Sequence[SchemaCheckRow],
    reader: FakeGraphNodeReader | None = None,
    lookup: SchemaLookup | None = None,
    *,
    un_served: bool = True,
) -> list[_Finding]:
    """The findings over every row, as `(field, property, subject, value, served, code)`."""
    checked = _checked(rows, reader, lookup, un_served=un_served)
    assert checked.rows_checked == len(rows)
    return [
        (
            finding.field,
            finding.property,
            finding.subject,
            finding.value,
            finding.served,
            finding.code,
        )
        for finding in checked.findings
    ]


def _cell(field: str, value: object, code: str) -> _Finding:
    return (field, None, "value", value, None, code)


def _prop(prop: str, code: str) -> _Finding:
    return ("constraint_properties", prop, "property", prop, None, code)


def _value(prop: str, value: object, code: str) -> _Finding:
    return ("constraint_properties", prop, "value", value, None, code)


def _observation(prop: str, code: str) -> _Finding:
    return ("observation_properties", prop, "property", prop, None, code)


_SERVED = INDEX.statvars[0]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode12220"}),
            [],
            id="held-value-domain-holds-an-ancestor",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpos": "CRSPurposeCode12220"}),
            [_prop("crsPurpos", "node_not_held")],
            id="key-not-held",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode99999"}),
            [_value("crsPurpose", "CRSPurposeCode99999", "node_not_held")],
            id="value-not-held-on-a-fully-listed-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"recipientPlace": "country/XYZ"}),
            [_value("recipientPlace", "country/XYZ", "node_not_held")],
            id="value-not-held-on-a-range-that-is-no-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "NoSuchGender"}),
            [_value("beneficiaryGender", "NoSuchGender", "node_not_held")],
            id="value-not-held-on-a-dfo-property-with-no-range",
        ),
        pytest.param(
            _row(population_type="Person", constraint_properties={"gender": "NoSuchGender"}),
            [_value("gender", "NoSuchGender", "node_not_held")],
            id="value-not-held-on-a-base-property",
        ),
        pytest.param(
            _row(
                constraint_properties={
                    "beneficiaryGender": "two words",
                    "amount": 5,
                    "gender": True,
                }
            ),
            [
                _value("amount", 5, "not_a_node"),
                _value("beneficiaryGender", "two words", "not_a_node"),
                _value("gender", True, "not_a_node"),
            ],
            id="literal-value",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "PlaceholderCode"}),
            [_value("beneficiaryGender", "PlaceholderCode", "node_provisional")],
            id="provisional-only",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "UntypedCode"}),
            [_value("beneficiaryGender", "UntypedCode", "node_untyped")],
            id="untyped-value-outside-the-document",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "RealizedCode"}),
            [],
            id="provisional-and-real-type",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": None}),
            [_value("crsPurpose", None, "value_missing")],
            id="null-value",
        ),
        pytest.param(
            _row(constraint_properties={"FinancialFlow": "CRSPurposeCode120"}),
            [_prop("FinancialFlow", "not_a_property")],
            id="key-is-class",
        ),
        pytest.param(
            _row(population_type="crsPurpose"),
            [_cell("population_type", "crsPurpose", "not_a_class")],
            id="population-type-is-property",
        ),
        pytest.param(
            _row(population_type=None, constraint_properties={"crsPurpose": "CRSPurposeCode12220"}),
            [],
            id="population-type-absent",
        ),
        pytest.param(
            _row(measured_property="FinancialFlow"),
            [_cell("measured_property", "FinancialFlow", "not_a_property")],
            id="measured-property-is-class",
        ),
        pytest.param(
            _row(observation_properties=["financingRecipient", "FinancialFlow"]),
            [_observation("FinancialFlow", "not_a_property")],
            id="observation-property-is-class",
        ),
        pytest.param(
            _row(stat_type="noSuchStatType"),
            [_cell("stat_type", "noSuchStatType", "node_not_held")],
            id="stat-type-not-held",
        ),
        pytest.param(
            _row(measurement_qualifier="noSuchQualifier"),
            [_cell("measurement_qualifier", "noSuchQualifier", "node_not_held")],
            id="qualifier-not-held",
        ),
        pytest.param(
            _row(measurement_denominator="noSuchDenominator"),
            [_cell("measurement_denominator", "noSuchDenominator", "node_not_held")],
            id="denominator-not-held",
        ),
        pytest.param(
            _row(stat_type="UntypedCode"),
            [_cell("stat_type", "UntypedCode", "node_untyped")],
            id="stat-type-untyped",
        ),
        pytest.param(
            _row(measurement_qualifier="PlaceholderCode"),
            [_cell("measurement_qualifier", "PlaceholderCode", "node_provisional")],
            id="qualifier-provisional-only",
        ),
        pytest.param(
            _row(measurement_denominator="UntypedCode"),
            [_cell("measurement_denominator", "UntypedCode", "node_untyped")],
            id="denominator-untyped",
        ),
        pytest.param(
            _row(measurement_qualifier="RealizedCode"), [], id="qualifier-provisional-and-real-type"
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "DAC1AidType1010"}),
            [_value("crsPurpose", "DAC1AidType1010", "value_outside_range")],
            id="member-of-another-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode120"}), [], id="parent-code"
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "Male"}), [], id="dfo-no-range"
        ),
        pytest.param(
            _row(constraint_properties={"financingPurpose": "CRSPurposeCode120"}),
            [],
            id="range-reached-by-subclass",
        ),
        pytest.param(
            _row(constraint_properties={"recipientPlace": "CRSPurposeCode120"}),
            [_value("recipientPlace", "CRSPurposeCode120", "value_outside_range")],
            id="held-value-typed-in-the-document-outside-a-class-range",
        ),
        pytest.param(
            _row(constraint_properties={"recipientPlace": "country/UGA"}),
            [],
            id="value-typed-outside-the-document",
        ),
        pytest.param(
            _row(constraint_properties={"purposeOrPlace": "CRSPurposeCode99999"}),
            [_value("purposeOrPlace", "CRSPurposeCode99999", "node_not_held")],
            id="value-outside-a-range-mixing-an-enumeration-and-a-class",
        ),
        pytest.param(
            _row(constraint_properties={"recipientGender": "NoSuchGender"}),
            [_value("recipientGender", "NoSuchGender", "node_not_held")],
            id="value-outside-an-incomplete-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGroup": "NoSuchGender"}),
            [_value("beneficiaryGroup", "NoSuchGender", "node_not_held")],
            id="value-outside-a-complete-enumeration-with-an-incomplete-sub-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"noSuchProperty": "NoSuchValue"}),
            [_prop("noSuchProperty", "node_not_held")],
            id="value-under-a-property-not-held",
        ),
        pytest.param(
            _row(constraint_properties={"noSuchProperty": None}),
            [_prop("noSuchProperty", "node_not_held")],
            id="null-value-under-a-property-not-held",
        ),
        pytest.param(_row(), [], id="base-measured-property-outside-domain"),
        pytest.param(
            _row(
                population_type="EconomicActivity",
                constraint_properties={"crsPurpose": "CRSPurposeCode120"},
            ),
            [_prop("crsPurpose", "property_outside_domain")],
            id="dfo-property-outside-domain",
        ),
        pytest.param(
            _row(
                population_type="Country", constraint_properties={"crsPurpose": "CRSPurposeCode120"}
            ),
            [],
            id="population-type-outside-the-document",
        ),
        pytest.param(
            _row(measured_property="grantEquivalent", population_type="FinancingEvent"),
            [_cell("measured_property", "grantEquivalent", "property_outside_domain")],
            id="dfo-measured-property-outside-domain",
        ),
        pytest.param(
            _row(population_type="Person", observation_properties=["crsPurpose"]),
            [],
            id="dfo-observation-property-outside-domain",
        ),
        pytest.param(
            _row(population_type="Person", constraint_properties={"gender": "Female"}),
            [],
            id="base-property-base-value",
        ),
        pytest.param(
            _row(constraint_properties={"amount": "CRSPurposeCode120"}),
            [],
            id="base-property-value-outside-its-range",
        ),
        pytest.param(
            _as_row(
                _SERVED,
                constraint_properties={
                    **_SERVED.constraint_properties,
                    "crsPurpose": "CRSPurposeCode120",
                },
            ),
            [
                (
                    "constraint_properties",
                    "crsPurpose",
                    "value",
                    "CRSPurposeCode120",
                    "CRSPurposeCode12220",
                    "definition_differs",
                )
            ],
            id="served-constraint-differs",
        ),
        pytest.param(
            _as_row(_SERVED, measured_property="grantEquivalent"),
            [
                (
                    "measured_property",
                    None,
                    "value",
                    "grantEquivalent",
                    "amount",
                    "definition_differs",
                )
            ],
            id="served-five-tuple-differs",
        ),
        pytest.param(
            _as_row(_SERVED, observation_properties=[]),
            [
                (
                    "observation_properties",
                    None,
                    "value",
                    [],
                    ["financingRecipient"],
                    "definition_differs",
                )
            ],
            id="served-observation-properties-differ",
        ),
        pytest.param(_as_row(_SERVED), [], id="served-same-definition"),
        pytest.param(
            _row(measured_property=None),
            [_cell("measured_property", None, "value_missing")],
            id="measured-property-absent",
        ),
        pytest.param(
            _row(measured_property=""),
            [_cell("measured_property", None, "value_missing")],
            id="measured-property-blank",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": ""}),
            [_value("crsPurpose", None, "value_missing")],
            id="constraint-value-blank",
        ),
        pytest.param(
            _row(stat_type="", observation_properties=[""]), [], id="blank-cells-are-absent"
        ),
        pytest.param(
            _row(population_type="Financial Flow"),
            [_cell("population_type", "Financial Flow", "not_a_node")],
            id="population-type-outside-dcid-syntax",
        ),
        pytest.param(
            _row(measured_property="dcid:amount"),
            [_cell("measured_property", "dcid:amount", "not_a_node")],
            id="measured-property-outside-dcid-syntax",
        ),
        pytest.param(
            _row(constraint_properties={"crs purpose": "CRSPurposeCode120"}),
            [_prop("crs purpose", "not_a_node")],
            id="key-outside-dcid-syntax",
        ),
        pytest.param(
            _row(observation_properties=["financing recipient"]),
            [_observation("financing recipient", "not_a_node")],
            id="observation-property-outside-dcid-syntax",
        ),
        pytest.param(
            _row(dcid="Amount CRS"),
            [_cell("dcid", "Amount CRS", "not_a_node")],
            id="dcid-outside-dcid-syntax",
        ),
    ],
)
def test_check_rules(row: SchemaCheckRow, expected: list[_Finding]) -> None:
    assert _check([row]) == expected


def test_findings_come_in_row_then_field_order_with_the_row_index() -> None:
    rows = [
        _row(stat_type="noSuchStatType", population_type="crsPurpose"),
        _row(constraint_properties={"crsPurpose": None}),
        _row(measurement_qualifier="noSuchQualifier"),
    ]

    findings = _checked(rows).findings

    assert [(finding.row, finding.field) for finding in findings] == [
        (0, "population_type"),
        (0, "stat_type"),
        (1, "constraint_properties"),
        (2, "measurement_qualifier"),
    ]


@pytest.mark.parametrize(
    ("code", "severity"),
    [
        pytest.param("node_not_held", "error", id="error"),
        pytest.param("not_a_node", "error", id="not-a-node"),
        pytest.param("property_outside_domain", "warning", id="domain"),
        pytest.param("definition_differs", "warning", id="differs"),
    ],
)
def test_a_finding_carries_its_code_severity(code: str, severity: str) -> None:
    rows = [
        _row(stat_type="noSuchStatType"),
        _row(measured_property="no such property"),
        _row(
            population_type="EconomicActivity",
            constraint_properties={"crsPurpose": "CRSPurposeCode120"},
        ),
        _as_row(_SERVED, measured_property="grantEquivalent"),
    ]

    severities = {finding.code: finding.severity for finding in _checked(rows).findings}

    assert severities[code] == severity


def test_every_served_statvar_sent_back_has_no_finding() -> None:
    rows = [_as_row(served) for served in (*INDEX.statvars, *INDEX.listed_statvars)]

    assert rows
    assert _check(rows) == []


def test_an_id_outside_the_document_is_read_once_for_every_row() -> None:
    reader = _RecordingReader()
    rows = [_row(constraint_properties={"beneficiaryGender": "Male"}) for _ in range(3)]

    assert _check(rows, reader) == []
    assert reader.batches == [["Male"]]


def test_an_id_named_in_two_roles_is_read_once() -> None:
    """An id named as an observation property and as the value of a property outside the
    document is read once, in the check's one read."""
    reader = _RecordingReader()
    rows = [
        *(_row(constraint_properties={"baseProperty0": f"NoSuchX{i}"}) for i in range(40)),
        *(_row(observation_properties=[f"NoSuchX{i}"]) for i in range(40)),
    ]

    _check(rows, reader)

    assert len(reader.batches) == 1
    assert len(reader.batches[0]) == len(set(reader.batches[0])) == CHECK_MAX_READ_IDS


def test_a_value_typed_outside_the_document_is_read_with_no_ancestry() -> None:
    reader = _RecordingReader()
    rows = [_row(constraint_properties={"recipientPlace": "country/UGA"}) for _ in range(2)]

    assert _check(rows, reader) == []
    assert reader.batches == [["country/UGA"]]


def test_a_value_under_a_property_not_held_is_read_and_gets_no_finding() -> None:
    reader = _RecordingReader()
    row = _row(constraint_properties={"gender": "NoSuchGender", "noSuchProperty": "NoSuchValue"})

    findings = _check([row], reader)

    assert findings == [
        _value("gender", "NoSuchGender", "node_not_held"),
        _prop("noSuchProperty", "node_not_held"),
    ]
    assert reader.batches == [["NoSuchGender", "NoSuchValue", "gender", "noSuchProperty"]]


@pytest.mark.parametrize(
    "constraints",
    [
        pytest.param({"crsPurpose": "CRSPurposeCode12220"}, id="document-holds-every-id"),
        pytest.param({"FinancialFlow": "NoSuchValue"}, id="value-under-a-class"),
        pytest.param(
            {
                "beneficiaryGender": "two words",
                "amount": 5,
                "crsPurpose": False,
                "recipientPlace": 1.5,
            },
            id="literals",
        ),
    ],
)
def test_no_read_where_the_document_or_the_value_decides(constraints: dict[str, object]) -> None:
    reader = _RecordingReader()

    _check([_row(constraint_properties=constraints)], reader)

    assert reader.batches == []
    assert sum(reader.calls.values()) == 0


@pytest.mark.parametrize(
    ("types", "expected", "batches"),
    [
        pytest.param(
            (ArcTarget(value="Property"),),
            [_value("crsPurpose", "CRSPurposeCode99999", "node_not_held")],
            [["CRSPurposeCode99999"]],
            id="no-type-sourced",
        ),
        pytest.param(
            (ArcTarget(value="Property", provenances=(DFO,)), ArcTarget(value="Thing")),
            [
                _prop("crsPurpose", "property_outside_domain"),
                _value("crsPurpose", "CRSPurposeCode99999", "node_not_held"),
            ],
            [["CRSPurposeCode99999"]],
            id="one-type-sourced-by-dfo",
        ),
    ],
)
def test_a_property_is_judged_by_the_provenances_its_types_carry(
    types: tuple[ArcTarget, ...],
    expected: list[_Finding],
    batches: list[list[str]],
) -> None:
    """`crsPurpose` with no sourced type is served `declared_by` null, so neither its range nor
    its domain is judged and its value is read. With one type sourced by DFO it stays a
    DFO-declared property, and both rules apply."""
    targets = {"crsPurpose": {**node_targets(_CHECK_NODES, _CHECK_TERMS)["crsPurpose"]}}
    targets["crsPurpose"]["typeOf"] = types
    graph = schema_graph(terms=_CHECK_TERMS, nodes=_CHECK_NODES, arc_targets=targets)
    document = build_schema(
        reader=graph, code_index=INDEX, catalogue=CATALOGUE, generation="g1", deadline=_DEADLINE
    )
    reader = _RecordingReader()
    row = _row(
        population_type="EconomicActivity",
        constraint_properties={"crsPurpose": "CRSPurposeCode99999"},
    )

    findings = _check([row], reader, SchemaLookup.of(document.document))

    assert findings == expected
    assert reader.batches == batches


def test_a_value_typed_below_a_complete_enumeration_outside_the_document_is_read() -> None:
    """`Male` is typed `GenderType`, a subclass of the complete `BeneficiaryGroupEnum`. With no
    reference bringing `GenderType` into the document, the check reads `Male` and finds no
    fault, because its type lies outside the document."""
    terms = {k: v for k, v in _CHECK_TERMS.items() if k not in ("gender", "recipientGender")}
    nodes = {k: v for k, v in _CHECK_NODES.items() if k not in ("gender", "recipientGender")}
    index = index_of(index_row("Amount_CRS12220", {"crsPurpose": "CRSPurposeCode12220"}))
    graph = schema_graph(terms=terms, nodes=nodes)
    document = build_schema(
        reader=graph, code_index=index, catalogue=CATALOGUE, generation="g1", deadline=_DEADLINE
    )
    lookup = SchemaLookup.of(document.document)
    reader = _RecordingReader()

    findings = _check([_row(constraint_properties={"beneficiaryGroup": "Male"})], reader, lookup)

    assert "GenderType" not in lookup.nodes
    assert findings == []
    assert reader.batches == [["Male"]]


def _absent_rows(count: int) -> list[SchemaCheckRow]:
    return [
        _row(constraint_properties={"beneficiaryGender": f"NoSuchGender{i}"}) for i in range(count)
    ]


def test_rows_naming_the_budget_of_ids_to_read_are_read() -> None:
    reader = _RecordingReader()

    findings = _check(_absent_rows(CHECK_MAX_READ_IDS), reader)

    assert [finding[-1] for finding in findings] == ["node_not_held"] * CHECK_MAX_READ_IDS
    assert len(reader.batches[0]) == CHECK_MAX_READ_IDS


def _maximal_row() -> SchemaCheckRow:
    """A row naming the most ids a row can, none of them in the document, every constraint
    property a held base property so its value is read too."""
    return _row(
        population_type="NoSuchClass",
        measured_property="noSuchMeasured",
        stat_type="noSuchStatType",
        measurement_qualifier="noSuchQualifier",
        measurement_denominator="noSuchDenominator",
        constraint_properties={
            f"baseProperty{i}": f"NoSuchValue{i}" for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES)
        },
        observation_properties=[
            f"noSuchObservation{i}" for i in range(CHECK_MAX_OBSERVATION_PROPERTIES)
        ],
    )


def test_a_single_row_naming_the_most_ids_a_row_can_is_read_within_the_budget() -> None:
    reader = _RecordingReader()

    findings = _check([_maximal_row()], reader)

    asked = {dcid for batch in reader.batches for dcid in batch}
    assert len(asked) == CHECK_MAX_READ_IDS
    assert len(reader.batches) == 1
    assert {finding[-1] for finding in findings} == {"node_not_held"}
    assert len(findings) == CHECK_MAX_READ_IDS - CHECK_MAX_CONSTRAINT_PROPERTIES


@pytest.mark.parametrize(
    ("rows", "taken"),
    [
        pytest.param(
            lambda: _absent_rows(CHECK_MAX_READ_IDS + 5),
            CHECK_MAX_READ_IDS,
            id="values-under-document-properties",
        ),
        pytest.param(lambda: [_maximal_row(), *_absent_rows(1)], 1, id="every-field-of-a-row"),
        pytest.param(
            lambda: [
                _row(constraint_properties={"crsPurpose": f"CRSPurposeCode9{i:04d}"})
                for i in range(CHECK_MAX_READ_IDS + 1)
            ],
            CHECK_MAX_READ_IDS,
            id="values-under-a-complete-enumeration",
        ),
        pytest.param(
            lambda: [*_absent_rows(CHECK_MAX_READ_IDS), _absent_rows(1)[0], _row()],
            CHECK_MAX_READ_IDS + 2,
            id="rows-reading-ids-already-taken",
        ),
    ],
)
def test_rows_past_the_read_budget_stop_before_the_row_that_passes_it(
    rows: Callable[[], list[SchemaCheckRow]], taken: int
) -> None:
    reader = _RecordingReader()
    sent = rows()

    plan = plan_reads(sent, lookup=_lookup(), un_served=True)
    checked = _checked(sent, reader)

    assert plan.row_count == checked.rows_checked == taken
    assert len(plan.ids) <= CHECK_MAX_READ_IDS
    assert reader.batches == [sorted(plan.ids)]
    assert {finding.row for finding in checked.findings} <= set(range(taken))


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "undata/NoSuchGroup"}),
            [_value("beneficiaryGender", "undata/NoSuchGroup", "origin_not_served")],
            id="value",
        ),
        pytest.param(
            _row(measured_property="undataAge"),
            [_cell("measured_property", "undataAge", "origin_not_served")],
            id="measured-property",
        ),
        pytest.param(
            _row(population_type="undata/Person"),
            [_cell("population_type", "undata/Person", "origin_not_served")],
            id="population-type",
        ),
        pytest.param(
            _row(stat_type="undata/Stat"),
            [_cell("stat_type", "undata/Stat", "origin_not_served")],
            id="stat-type",
        ),
        pytest.param(
            _row(constraint_properties={"undataSex": "undata/Female"}),
            [_prop("undataSex", "origin_not_served")],
            id="constraint-property-and-its-value",
        ),
        pytest.param(
            _row(observation_properties=["undataUnit"]),
            [_observation("undataUnit", "origin_not_served")],
            id="observation-property",
        ),
    ],
)
def test_a_un_id_gets_origin_not_served_with_no_read_while_the_un_origin_is_off(
    row: SchemaCheckRow, expected: list[_Finding]
) -> None:
    off = _RecordingReader()
    on = _RecordingReader()

    assert _check([row], off, un_served=False) == expected
    assert off.batches == []
    assert "origin_not_served" not in {finding[-1] for finding in _check([row], on)}
    assert on.batches != []


def test_every_read_asks_the_type_and_name_and_probes_every_empty_answer() -> None:
    reader = _RecordingReader()
    row = _row(constraint_properties={"gender": "NoSuchGender", "noSuchProperty": "NoSuchValue"})

    _check([row], reader)

    assert reader.asked == [(("typeOf", "name"), None)]


def test_a_held_node_with_neither_a_type_nor_a_name_reads_as_absent() -> None:
    class _DescribedOnly(_RecordingReader):
        def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
            answered = super().node_arcs_many(
                dcids,
                properties=properties,
                individual_probe_budget=individual_probe_budget,
                deadline=deadline,
            )
            return {**answered, "DescribedCode": dict.fromkeys(properties, ())}

    findings = _check(
        [_row(constraint_properties={"beneficiaryGender": "DescribedCode"})], _DescribedOnly()
    )

    assert findings == [_value("beneficiaryGender", "DescribedCode", "node_not_held")]


def _many_findings_row() -> SchemaCheckRow:
    """A row with one `not_a_node` finding on each constraint property."""
    return _row(
        constraint_properties={
            f"baseProperty{i}": f"not a dcid {i}" for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES)
        },
    )


def test_findings_past_the_cap_stop_before_the_row_that_passes_it() -> None:
    fit = CHECK_MAX_FINDINGS // CHECK_MAX_CONSTRAINT_PROPERTIES
    rows = [_many_findings_row() for _ in range(fit + 2)]

    checked = _checked(rows)

    assert checked.rows_checked == fit
    assert len(checked.findings) == fit * CHECK_MAX_CONSTRAINT_PROPERTIES <= CHECK_MAX_FINDINGS
    assert checked.findings[-1].row == fit - 1


def _escaped_row() -> SchemaCheckRow:
    """A row of 16 `not_a_node` findings, each echoing a 512-character value whose every
    character JSON escapes to six bytes."""
    escaped = "\x01" * 512
    return _row(
        constraint_properties={
            f"baseProperty{i}": escaped for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES)
        },
    )


def _encoded(checked: CheckedRows) -> bytes:
    return (
        SchemaCheckResponse(
            generation="g1", rows_checked=checked.rows_checked, findings=checked.findings
        )
        .model_dump_json()
        .encode()
    )


def test_findings_past_the_body_budget_stop_before_the_row_that_passes_it() -> None:
    rows = [_escaped_row() for _ in range(100)]

    checked = _checked(rows)

    body = _encoded(checked)
    last_row = [finding for finding in checked.findings if finding.row == checked.rows_checked - 1]
    next_row_size = sum(len(finding.model_dump_json().encode()) + 1 for finding in last_row)
    assert 1 < checked.rows_checked < len(rows)
    assert len(checked.findings) < CHECK_MAX_FINDINGS
    assert len(body) <= CHECK_MAX_BODY_BYTES < len(body) + next_row_size


def test_the_first_row_is_taken_past_the_body_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema_check, "CHECK_MAX_BODY_BYTES", 1)

    checked = _checked([_escaped_row(), _escaped_row()])

    assert checked.rows_checked == 1
    assert len(checked.findings) == CHECK_MAX_CONSTRAINT_PROPERTIES


def test_one_row_yields_fewer_findings_than_the_cap() -> None:
    """The most a row yields, from the per-row caps: 5 on the shape fields, 2 on each constraint
    property, 1 on each observation property, and a `definition_differs` warning on each shape
    field, each constraint property the row or a served StatVar names, and the observation
    properties."""
    served_most = max(len(row.constraint_properties) for row in INDEX.statvars)
    most = (
        5
        + 2 * CHECK_MAX_CONSTRAINT_PROPERTIES
        + CHECK_MAX_OBSERVATION_PROPERTIES
        + 5
        + 2 * CHECK_MAX_CONSTRAINT_PROPERTIES
        + 1
    )
    worst = _as_row(
        _SERVED,
        population_type="EconomicActivity",
        measured_property="noSuchMeasured",
        stat_type="noSuchStatType",
        measurement_qualifier="noSuchQualifier",
        measurement_denominator="noSuchDenominator",
        constraint_properties={
            "crsPurpose": "DAC1AidType1010",
            **{
                f"baseProperty{i}": f"not a dcid {i}"
                for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES - 1)
            },
        },
        observation_properties=["FinancialFlow"] * CHECK_MAX_OBSERVATION_PROPERTIES,
    )

    assert served_most <= CHECK_MAX_CONSTRAINT_PROPERTIES
    assert len(_check([worst])) <= most < CHECK_MAX_FINDINGS


class _FailingReader(_RecordingReader):
    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        raise GraphReadError("graph unavailable")


def test_a_read_error_propagates() -> None:
    with pytest.raises(GraphReadError):
        _check(
            [_row(constraint_properties={"beneficiaryGender": "NoSuchGender"})], _FailingReader()
        )
