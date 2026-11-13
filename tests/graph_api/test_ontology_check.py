"""`check_rows` against the ontology fixture graph, extended with the nodes the rules need."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Any

import pytest

from code_index import StatVarRow
from dc_graph import GraphReadError
from graph_api.ontology_check import OntologyCheckTooManyReads, OntologyLookup, check_rows
from graph_api.ontology_document import Ontology, build_ontology
from graph_api.ontology_models import CHECK_MAX_READ_IDS, OntologyCheckRow, OntologyEnumeration

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
    """The check graph, recording the dcids of every `node_arcs_batch` call."""

    def __init__(self) -> None:
        super().__init__(nodes=_CHECK_NODES)
        self.batches: list[list[str]] = []

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        self.batches.append(list(dcids))
        return super().node_arcs_batch(dcids, deadline=deadline)


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
    findings = check_rows(
        rows,
        lookup=lookup or _lookup(),
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
            [(_CRS, "CRSPurposeCode99999", "value_outside_range", "error")],
            id="value-outside-a-fully-listed-enumeration",
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
        rows, lookup=_lookup(), reader=_RecordingReader(), code_index=INDEX, deadline=_DEADLINE
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


@pytest.mark.parametrize(
    "constraints",
    [
        pytest.param({"crsPurpose": "CRSPurposeCode12220"}, id="document-holds-every-id"),
        pytest.param({"crsPurpose": "CRSPurposeCode99999"}, id="fully-listed-enumeration"),
        pytest.param({"crsPurpose": "DAC1AidType1010"}, id="member-of-another-enumeration"),
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


def test_a_truncated_range_enumeration_reads_the_value() -> None:
    document = _ontology().document
    truncated = document.model_copy(
        update={
            "nodes": [
                node.model_copy(
                    update={
                        "enumeration": OntologyEnumeration(
                            members=node.enumeration.members, truncated=True
                        )
                    }
                )
                if node.dcid == "CRSPurposeCodeEnum" and node.enumeration is not None
                else node
                for node in document.nodes
            ]
        }
    )
    reader = _RecordingReader()

    findings = _check(
        [_row(constraint_properties={"crsPurpose": "CRSPurposeCode99999"})],
        reader,
        OntologyLookup.of(truncated),
    )

    assert findings == [(_CRS, "CRSPurposeCode99999", "node_not_held", "error")]
    assert reader.batches == [["CRSPurposeCode99999"]]


def _absent_rows(count: int) -> list[OntologyCheckRow]:
    return [
        _row(ref=str(i), constraint_properties={"beneficiaryGender": f"NoSuchGender{i}"})
        for i in range(count)
    ]


def test_rows_naming_more_ids_to_read_than_the_cap_are_refused_before_any_read() -> None:
    reader = _RecordingReader()

    with pytest.raises(OntologyCheckTooManyReads) as raised:
        _check(_absent_rows(CHECK_MAX_READ_IDS + 1), reader)

    assert raised.value.count == CHECK_MAX_READ_IDS + 1
    assert sum(reader.calls.values()) == 0


def test_rows_naming_the_cap_of_ids_to_read_are_read() -> None:
    reader = _RecordingReader()

    findings = _check(_absent_rows(CHECK_MAX_READ_IDS), reader)

    assert [code for _, _, code, _ in findings] == ["node_not_held"] * CHECK_MAX_READ_IDS
    assert len(reader.batches[0]) == CHECK_MAX_READ_IDS


def test_values_the_document_decides_count_against_no_cap() -> None:
    reader = _RecordingReader()
    rows = [
        _row(ref=str(i), constraint_properties={"crsPurpose": f"CRSPurposeCode9{i:04d}"})
        for i in range(CHECK_MAX_READ_IDS + 1)
    ]

    findings = _check(rows, reader)

    assert {code for _, _, code, _ in findings} == {"value_outside_range"}
    assert reader.batches == []


class _CompositionReader(_RecordingReader):
    """Answers `Male` with no arc in a batch of several, as the graph can."""

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        result = super().node_arcs_batch(dcids, deadline=deadline)
        if len(dcids) > 1:
            result["Male"] = None
        return result


def test_a_node_a_batch_answers_with_no_arc_is_read_alone() -> None:
    reader = _CompositionReader()
    rows = [
        _row(ref="a", constraint_properties={"beneficiaryGender": "Male"}),
        _row(ref="b", constraint_properties={"beneficiaryGender": "NoSuchGender"}),
    ]

    findings = _check(rows, reader)

    assert findings == [(_GENDER, "NoSuchGender", "node_not_held", "error")]
    assert reader.batches == [["Male", "NoSuchGender"], ["Male"], ["NoSuchGender"]]


class _FailingReader(_RecordingReader):
    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        raise GraphReadError("graph unavailable")


def test_a_read_error_propagates() -> None:
    with pytest.raises(GraphReadError):
        _check(
            [_row(constraint_properties={"beneficiaryGender": "NoSuchGender"})], _FailingReader()
        )
