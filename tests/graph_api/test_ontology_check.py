"""`check_rows` against the ontology fixture graph, extended with the nodes the rules need."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

import pytest

from code_index import StatVarRow
from dc_graph import ArcTarget, GraphReadError
from graph_api.ontology_check import (
    OntologyCheckTooManyFindings,
    OntologyCheckTooManyReads,
    OntologyCheckUnservedId,
    OntologyLookup,
    check_rows,
    plan_reads,
)
from graph_api.ontology_document import Ontology, build_ontology
from graph_api.ontology_models import (
    CHECK_MAX_CONSTRAINT_PROPERTIES,
    CHECK_MAX_FINDINGS,
    CHECK_MAX_OBSERVATION_PROPERTIES,
    CHECK_MAX_READ_IDS,
    OntologyCheckRow,
)
from graph_api.reader import NodeArcsByDcid

from .conftest import FakeGraphNodeReader
from .ontology_fixture import (
    DFO,
    INDEX,
    NODES,
    SCHEMA,
    TERMS,
    index_of,
    index_row,
    node_targets,
    ontology_graph,
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
    # An instance property ranged on the base enumeration, whose member list is incomplete.
    "recipientGender": {
        "typeOf": ("Property",),
        "domainIncludes": ("FinancingEvent",),
        "rangeIncludes": ("GenderType",),
    },
    # A complete instance enumeration (it has no member of its own) above the base one.
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
def _ontology() -> Ontology:
    graph = ontology_graph(terms=_CHECK_TERMS, nodes=_CHECK_NODES)
    return build_ontology(reader=graph, code_index=INDEX, generation="g1", deadline=_DEADLINE)


@cache
def _lookup() -> OntologyLookup:
    return OntologyLookup.of(_ontology().document)


def _row(**fields: Any) -> OntologyCheckRow:
    return OntologyCheckRow.model_validate(
        {
            "ref": "r1",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            **fields,
        }
    )


def _as_row(served: StatVarRow, **fields: Any) -> OntologyCheckRow:
    return OntologyCheckRow.model_validate(
        {
            "ref": served.dcid,
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


def _check(
    rows: Sequence[OntologyCheckRow],
    reader: FakeGraphNodeReader | None = None,
    lookup: OntologyLookup | None = None,
) -> list[tuple[str, object, str, str]]:
    lookup = lookup or _lookup()
    findings = check_rows(
        rows,
        reads=plan_reads(rows, lookup=lookup, un_served=True),
        lookup=lookup,
        reader=reader or _RecordingReader(),
        code_index=INDEX,
        deadline=_DEADLINE,
    )
    return [(finding.field, finding.value, finding.code, finding.severity) for finding in findings]


_SERVED = INDEX.statvars[0]
_CRS = "constraint_properties.crsPurpose"
_GENDER = "constraint_properties.beneficiaryGender"


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
            [("constraint_properties.crsPurpos", "crsPurpos", "node_not_held", "error")],
            id="key-not-held",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode99999"}),
            [(_CRS, "CRSPurposeCode99999", "node_not_held", "error")],
            id="value-not-held-on-a-fully-listed-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"recipientPlace": "country/XYZ"}),
            [("constraint_properties.recipientPlace", "country/XYZ", "node_not_held", "error")],
            id="value-not-held-on-a-range-that-is-no-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "NoSuchGender"}),
            [(_GENDER, "NoSuchGender", "node_not_held", "error")],
            id="value-not-held-on-an-instance-property-with-no-range",
        ),
        pytest.param(
            _row(population_type="Person", constraint_properties={"gender": "NoSuchGender"}),
            [("constraint_properties.gender", "NoSuchGender", "node_not_held", "error")],
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
                ("constraint_properties.amount", 5, "node_not_held", "error"),
                (_GENDER, "two words", "node_not_held", "error"),
                ("constraint_properties.gender", True, "node_not_held", "error"),
            ],
            id="literal-value",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "PlaceholderCode"}),
            [(_GENDER, "PlaceholderCode", "node_provisional", "error")],
            id="provisional-only",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "UntypedCode"}),
            [(_GENDER, "UntypedCode", "node_untyped", "error")],
            id="untyped-value-outside-the-document",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "RealizedCode"}),
            [],
            id="provisional-and-real-type",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": None}),
            [(_CRS, None, "value_missing", "error")],
            id="null-value",
        ),
        pytest.param(
            _row(constraint_properties={"FinancialFlow": "CRSPurposeCode120"}),
            [("constraint_properties.FinancialFlow", "FinancialFlow", "not_a_property", "error")],
            id="key-is-class",
        ),
        pytest.param(
            _row(population_type="crsPurpose"),
            [("population_type", "crsPurpose", "not_a_class", "error")],
            id="population-type-is-property",
        ),
        pytest.param(
            _row(population_type=None, constraint_properties={"crsPurpose": "CRSPurposeCode12220"}),
            [],
            id="population-type-absent",
        ),
        pytest.param(
            _row(measured_property="FinancialFlow"),
            [("measured_property", "FinancialFlow", "not_a_property", "error")],
            id="measured-property-is-class",
        ),
        pytest.param(
            _row(observation_properties=["financingRecipient", "FinancialFlow"]),
            [("observation_properties", "FinancialFlow", "not_a_property", "error")],
            id="observation-property-is-class",
        ),
        pytest.param(
            _row(stat_type="noSuchStatType"),
            [("stat_type", "noSuchStatType", "node_not_held", "error")],
            id="stat-type-not-held",
        ),
        pytest.param(
            _row(measurement_qualifier="noSuchQualifier"),
            [("measurement_qualifier", "noSuchQualifier", "node_not_held", "error")],
            id="qualifier-not-held",
        ),
        pytest.param(
            _row(measurement_denominator="noSuchDenominator"),
            [("measurement_denominator", "noSuchDenominator", "node_not_held", "error")],
            id="denominator-not-held",
        ),
        pytest.param(
            _row(stat_type="UntypedCode"),
            [("stat_type", "UntypedCode", "node_untyped", "error")],
            id="stat-type-untyped",
        ),
        pytest.param(
            _row(measurement_qualifier="PlaceholderCode"),
            [("measurement_qualifier", "PlaceholderCode", "node_provisional", "error")],
            id="qualifier-provisional-only",
        ),
        pytest.param(
            _row(measurement_denominator="UntypedCode"),
            [("measurement_denominator", "UntypedCode", "node_untyped", "error")],
            id="denominator-untyped",
        ),
        pytest.param(
            _row(measurement_qualifier="RealizedCode"), [], id="qualifier-provisional-and-real-type"
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "DAC1AidType1010"}),
            [(_CRS, "DAC1AidType1010", "value_outside_range", "error")],
            id="member-of-another-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode120"}), [], id="parent-code"
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "Male"}), [], id="instance-no-range"
        ),
        pytest.param(
            _row(constraint_properties={"financingPurpose": "CRSPurposeCode120"}),
            [],
            id="range-reached-by-subclass",
        ),
        pytest.param(
            _row(constraint_properties={"recipientPlace": "CRSPurposeCode120"}),
            [
                (
                    "constraint_properties.recipientPlace",
                    "CRSPurposeCode120",
                    "value_outside_range",
                    "error",
                )
            ],
            id="held-value-typed-in-the-document-outside-a-class-range",
        ),
        pytest.param(
            _row(constraint_properties={"recipientPlace": "country/UGA"}),
            [],
            id="value-typed-outside-the-document",
        ),
        pytest.param(
            _row(constraint_properties={"purposeOrPlace": "CRSPurposeCode99999"}),
            [
                (
                    "constraint_properties.purposeOrPlace",
                    "CRSPurposeCode99999",
                    "node_not_held",
                    "error",
                )
            ],
            id="value-outside-a-range-mixing-an-enumeration-and-a-class",
        ),
        pytest.param(
            _row(constraint_properties={"recipientGender": "NoSuchGender"}),
            [("constraint_properties.recipientGender", "NoSuchGender", "node_not_held", "error")],
            id="value-outside-an-incomplete-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"beneficiaryGroup": "NoSuchGender"}),
            [("constraint_properties.beneficiaryGroup", "NoSuchGender", "node_not_held", "error")],
            id="value-outside-a-complete-enumeration-with-an-incomplete-sub-enumeration",
        ),
        pytest.param(
            _row(constraint_properties={"noSuchProperty": "NoSuchValue"}),
            [("constraint_properties.noSuchProperty", "noSuchProperty", "node_not_held", "error")],
            id="value-under-a-property-not-held",
        ),
        pytest.param(
            _row(constraint_properties={"noSuchProperty": None}),
            [("constraint_properties.noSuchProperty", "noSuchProperty", "node_not_held", "error")],
            id="null-value-under-a-property-not-held",
        ),
        pytest.param(_row(), [], id="base-measured-property-outside-domain"),
        pytest.param(
            _row(
                population_type="EconomicActivity",
                constraint_properties={"crsPurpose": "CRSPurposeCode120"},
            ),
            [(_CRS, "crsPurpose", "property_outside_domain", "warning")],
            id="instance-property-outside-domain",
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
            [("measured_property", "grantEquivalent", "property_outside_domain", "warning")],
            id="instance-measured-property-outside-domain",
        ),
        pytest.param(
            _row(population_type="Person", observation_properties=["crsPurpose"]),
            [],
            id="instance-observation-property-outside-domain",
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
            [(_CRS, "CRSPurposeCode12220", "definition_differs", "warning")],
            id="served-constraint-differs",
        ),
        pytest.param(
            _as_row(_SERVED, measured_property="grantEquivalent"),
            [("measured_property", "amount", "definition_differs", "warning")],
            id="served-five-tuple-differs",
        ),
        pytest.param(
            _as_row(_SERVED, observation_properties=[]),
            [("observation_properties", ["financingRecipient"], "definition_differs", "warning")],
            id="served-observation-properties-differ",
        ),
        pytest.param(_as_row(_SERVED), [], id="served-same-definition"),
    ],
)
def test_check_rules(row: OntologyCheckRow, expected: list[tuple[str, object, str, str]]) -> None:
    assert _check([row]) == expected


def test_findings_come_in_row_then_field_order_with_the_row_ref() -> None:
    rows = [
        _row(ref="a", stat_type="noSuchStatType", population_type="crsPurpose"),
        _row(ref="b", constraint_properties={"crsPurpose": None}),
    ]

    findings = check_rows(
        rows,
        reads=plan_reads(rows, lookup=_lookup(), un_served=True),
        lookup=_lookup(),
        reader=_RecordingReader(),
        code_index=INDEX,
        deadline=_DEADLINE,
    )

    assert [(finding.ref, finding.field) for finding in findings] == [
        ("a", "population_type"),
        ("a", "stat_type"),
        ("b", _CRS),
    ]


def test_every_served_statvar_sent_back_has_no_finding() -> None:
    rows = [_as_row(served) for served in (*INDEX.statvars, *INDEX.listed_statvars)]

    assert rows
    assert _check(rows) == []


def test_an_id_outside_the_document_is_read_once_for_every_row() -> None:
    reader = _RecordingReader()
    rows = [_row(ref=str(i), constraint_properties={"beneficiaryGender": "Male"}) for i in range(3)]

    assert _check(rows, reader) == []
    assert reader.batches == [["Male"]]


def test_an_id_named_in_two_roles_is_read_once() -> None:
    """An id named as an observation property and as the value of a property outside the
    document is read once, in the check's one read."""
    reader = _RecordingReader()
    rows = [
        *(
            _row(ref=f"v{i}", constraint_properties={"baseProperty0": f"NoSuchX{i}"})
            for i in range(40)
        ),
        *(_row(ref=f"o{i}", observation_properties=[f"NoSuchX{i}"]) for i in range(40)),
    ]

    _check(rows, reader)

    assert len(reader.batches) == 1
    assert len(reader.batches[0]) == len(set(reader.batches[0])) == CHECK_MAX_READ_IDS


def test_a_value_typed_outside_the_document_is_read_with_no_ancestry() -> None:
    reader = _RecordingReader()
    rows = [
        _row(ref=str(i), constraint_properties={"recipientPlace": "country/UGA"}) for i in range(2)
    ]

    assert _check(rows, reader) == []
    assert reader.batches == [["country/UGA"]]


def test_a_value_under_a_property_not_held_is_read_and_gets_no_finding() -> None:
    reader = _RecordingReader()
    row = _row(constraint_properties={"gender": "NoSuchGender", "noSuchProperty": "NoSuchValue"})

    findings = _check([row], reader)

    assert findings == [
        ("constraint_properties.gender", "NoSuchGender", "node_not_held", "error"),
        ("constraint_properties.noSuchProperty", "noSuchProperty", "node_not_held", "error"),
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
            [(_CRS, "CRSPurposeCode99999", "node_not_held", "error")],
            [["CRSPurposeCode99999"]],
            id="no-type-sourced",
        ),
        pytest.param(
            (ArcTarget(value="Property", provenances=(DFO,)), ArcTarget(value="Thing")),
            [
                (_CRS, "crsPurpose", "property_outside_domain", "warning"),
                (_CRS, "CRSPurposeCode99999", "node_not_held", "error"),
            ],
            [["CRSPurposeCode99999"]],
            id="one-type-sourced-by-the-instance",
        ),
    ],
)
def test_a_property_is_judged_by_the_provenances_its_types_carry(
    types: tuple[ArcTarget, ...],
    expected: list[tuple[str, object, str, str]],
    batches: list[list[str]],
) -> None:
    """`crsPurpose` with no sourced type is served `declared_by` null, so neither its range nor
    its domain is judged and its value is read. With one type sourced by the instance it stays
    an instance property, and both rules apply."""
    targets = {"crsPurpose": {**node_targets(_CHECK_NODES, _CHECK_TERMS)["crsPurpose"]}}
    targets["crsPurpose"]["typeOf"] = types
    graph = ontology_graph(terms=_CHECK_TERMS, nodes=_CHECK_NODES, arc_targets=targets)
    document = build_ontology(reader=graph, code_index=INDEX, generation="g1", deadline=_DEADLINE)
    reader = _RecordingReader()
    row = _row(
        population_type="EconomicActivity",
        constraint_properties={"crsPurpose": "CRSPurposeCode99999"},
    )

    findings = _check([row], reader, OntologyLookup.of(document.document))

    assert findings == expected
    assert reader.batches == batches


def test_a_value_typed_below_a_complete_enumeration_outside_the_document_is_read() -> None:
    """`Male` is typed `GenderType`, a subclass of the complete `BeneficiaryGroupEnum`. With no
    reference bringing `GenderType` into the document, the check reads `Male` and finds no
    fault, because its type lies outside the document."""
    terms = {k: v for k, v in _CHECK_TERMS.items() if k not in ("gender", "recipientGender")}
    nodes = {k: v for k, v in _CHECK_NODES.items() if k not in ("gender", "recipientGender")}
    index = index_of(index_row("Amount_CRS12220", {"crsPurpose": "CRSPurposeCode12220"}))
    graph = ontology_graph(terms=terms, nodes=nodes)
    document = build_ontology(reader=graph, code_index=index, generation="g1", deadline=_DEADLINE)
    lookup = OntologyLookup.of(document.document)
    reader = _RecordingReader()

    findings = _check([_row(constraint_properties={"beneficiaryGroup": "Male"})], reader, lookup)

    assert "GenderType" not in lookup.nodes
    assert findings == []
    assert reader.batches == [["Male"]]


def _absent_rows(count: int) -> list[OntologyCheckRow]:
    return [
        _row(ref=str(i), constraint_properties={"beneficiaryGender": f"NoSuchGender{i}"})
        for i in range(count)
    ]


def test_rows_naming_the_cap_of_ids_to_read_are_read() -> None:
    reader = _RecordingReader()

    findings = _check(_absent_rows(CHECK_MAX_READ_IDS), reader)

    assert [code for _, _, code, _ in findings] == ["node_not_held"] * CHECK_MAX_READ_IDS
    assert len(reader.batches[0]) == CHECK_MAX_READ_IDS


def _maximal_row() -> OntologyCheckRow:
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


def test_a_single_row_naming_the_most_ids_a_row_can_is_read_within_the_cap() -> None:
    reader = _RecordingReader()

    findings = _check([_maximal_row()], reader)

    asked = {dcid for batch in reader.batches for dcid in batch}
    assert len(asked) == CHECK_MAX_READ_IDS
    assert len(reader.batches) == 1
    assert {code for _, _, code, _ in findings} == {"node_not_held"}
    assert len(findings) == CHECK_MAX_READ_IDS - CHECK_MAX_CONSTRAINT_PROPERTIES


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(
            lambda: _absent_rows(CHECK_MAX_READ_IDS + 1), id="values-under-document-properties"
        ),
        pytest.param(lambda: [_maximal_row(), *_absent_rows(1)], id="every-field-of-a-row"),
        pytest.param(
            lambda: [
                _row(ref=str(i), constraint_properties={"crsPurpose": f"CRSPurposeCode9{i:04d}"})
                for i in range(CHECK_MAX_READ_IDS + 1)
            ],
            id="values-under-a-complete-enumeration",
        ),
    ],
)
def test_rows_naming_one_id_past_the_cap_are_refused_with_no_read(
    rows: Callable[[], list[OntologyCheckRow]],
) -> None:
    with pytest.raises(OntologyCheckTooManyReads) as raised:
        plan_reads(rows(), lookup=_lookup(), un_served=True)

    assert raised.value.count == CHECK_MAX_READ_IDS + 1


@pytest.mark.parametrize(
    ("row", "read"),
    [
        pytest.param(
            _row(constraint_properties={"beneficiaryGender": "undata/NoSuchGroup"}),
            "undata/NoSuchGroup",
            id="value",
        ),
        pytest.param(_row(measured_property="undataAge"), "undataAge", id="property"),
    ],
)
def test_a_un_id_to_read_is_refused_while_the_un_origin_is_off(
    row: OntologyCheckRow, read: str
) -> None:
    with pytest.raises(OntologyCheckUnservedId) as raised:
        plan_reads([row], lookup=_lookup(), un_served=False)

    assert raised.value.dcid == read
    assert plan_reads([row], lookup=_lookup(), un_served=True) == {read}


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

    assert findings == [(_GENDER, "DescribedCode", "node_not_held", "error")]


def _many_findings_row(ref: str) -> OntologyCheckRow:
    """A row with one `node_not_held` finding on each constraint property, read from nothing."""
    return _row(
        ref=ref,
        constraint_properties={
            f"baseProperty{i}": f"not a dcid {i}" for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES)
        },
    )


def test_findings_past_the_cap_stop_the_check() -> None:
    rows = [
        _many_findings_row(str(i))
        for i in range(CHECK_MAX_FINDINGS // CHECK_MAX_CONSTRAINT_PROPERTIES + 1)
    ]

    assert len(_check(rows[:-1])) == CHECK_MAX_FINDINGS
    with pytest.raises(OntologyCheckTooManyFindings):
        _check(rows)


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
