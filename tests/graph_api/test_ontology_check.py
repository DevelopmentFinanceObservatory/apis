"""`check_rows` against the ontology fixture graph, extended with the nodes the rules need."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

import pytest

from code_index import StatVarRow
from dc_graph import GraphReadError
from graph_api.ontology_check import check_rows
from graph_api.ontology_document import Ontology, build_ontology
from graph_api.ontology_models import OntologyCheckRow
from graph_api.reader import NodeArcsByDcid

from .conftest import FakeGraphNodeReader
from .ontology_fixture import DFO, INDEX, NODES, SCHEMA, TERMS, ontology_graph

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
    "PlaceholderCode": {"typeOf": ("ProvisionalNode",)},
    "RealizedCode": {"typeOf": ("ProvisionalNode", "CRSPurposeCodeEnum")},
    "UntypedCode": {"name": ("Untyped code",)},
}


class _RecordingReader(FakeGraphNodeReader):
    """The check graph, recording the dcids of every `node_arcs_many` call."""

    def __init__(self) -> None:
        super().__init__(nodes=_CHECK_NODES)
        self.batches: list[list[str]] = []

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> NodeArcsByDcid:
        self.batches.append(list(dcids))
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


@cache
def _ontology() -> Ontology:
    graph = ontology_graph(terms=_CHECK_TERMS, nodes=_CHECK_NODES)
    return build_ontology(reader=graph, code_index=INDEX, generation="g1", deadline=_DEADLINE)


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
    rows: Sequence[OntologyCheckRow], reader: FakeGraphNodeReader | None = None
) -> list[tuple[str, object, str, str]]:
    findings = check_rows(
        rows,
        ontology=_ontology(),
        reader=reader or _RecordingReader(),
        code_index=INDEX,
        deadline=_DEADLINE,
    )
    return [(finding.field, finding.value, finding.code, finding.severity) for finding in findings]


_SERVED = INDEX.statvars[0]
_CRS = "constraint_properties.crsPurpose"


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode12220"}), [], id="held-value"
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpos": "CRSPurposeCode12220"}),
            [("constraint_properties.crsPurpos", "crsPurpos", "node_not_held", "error")],
            id="key-not-held",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode99999"}),
            [(_CRS, "CRSPurposeCode99999", "node_not_held", "error")],
            id="value-not-held",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "health", "amount": 5}),
            [
                ("constraint_properties.amount", 5, "node_not_held", "error"),
                (_CRS, "health", "node_not_held", "error"),
            ],
            id="literal-value",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "PlaceholderCode"}),
            [(_CRS, "PlaceholderCode", "node_provisional", "error")],
            id="provisional-only",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "UntypedCode"}),
            [(_CRS, "UntypedCode", "node_untyped", "error")],
            id="untyped-value",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "RealizedCode"}),
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
            _row(constraint_properties={"recipientPlace": "country/UGA"}),
            [],
            id="range-reached-by-subclass",
        ),
        pytest.param(_row(), [], id="base-measured-property-outside-domain"),
        pytest.param(
            _row(
                population_type="Person", constraint_properties={"crsPurpose": "CRSPurposeCode120"}
            ),
            [(_CRS, "crsPurpose", "property_outside_domain", "warning")],
            id="instance-property-outside-domain",
        ),
        pytest.param(
            _row(measured_property="grantEquivalent", population_type="FinancingEvent"),
            [("measured_property", "grantEquivalent", "property_outside_domain", "warning")],
            id="instance-measured-property-outside-domain",
        ),
        pytest.param(
            _row(constraint_properties={"crsPurpose": "CRSPurposeCode12220"}),
            [],
            id="domain-holds-an-ancestor",
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
        rows, ontology=_ontology(), reader=_RecordingReader(), code_index=INDEX, deadline=_DEADLINE
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


def test_ancestors_outside_the_document_are_read_one_level_per_batch() -> None:
    reader = _RecordingReader()
    rows = [
        _row(ref=str(i), constraint_properties={"recipientPlace": "country/UGA"}) for i in range(2)
    ]

    assert _check(rows, reader) == []
    assert reader.batches == [["country/UGA"], ["Country"], ["AdministrativeArea"]]


def test_no_read_when_the_document_holds_every_id() -> None:
    reader = _RecordingReader()

    assert _check([_row(constraint_properties={"crsPurpose": "CRSPurposeCode12220"})], reader) == []
    assert reader.calls["node_arcs_many"] == 0


class _FailingReader(_RecordingReader):
    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> NodeArcsByDcid:
        raise GraphReadError("graph unavailable")


def test_a_read_error_propagates() -> None:
    with pytest.raises(GraphReadError):
        _check(
            [_row(constraint_properties={"crsPurpose": "CRSPurposeCode99999"})], _FailingReader()
        )
