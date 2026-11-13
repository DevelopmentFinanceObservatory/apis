"""A small fake graph shaped like the dev graph's ontology, and a code index naming it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from code_index import CodeIndex, IndexStamp, StatVarRow
from dc_graph import ArcSubject, ArcTarget

from .conftest import FakeGraphNodeReader

DFO = "provenance/DFOOntology"
SCHEMA = "dc/base/Schema"

# Each term: (kind, declaring provenances).
TERMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "FinancingEvent": ("Class", (DFO,)),
    "FinancialFlow": ("Class", (DFO,)),
    "CRSPurposeCodeEnum": ("Class", (DFO,)),
    "FinancingPurposeEnum": ("Class", (DFO,)),
    "EconomicActivity": ("Class", (SCHEMA,)),
    "Intangible": ("Class", (SCHEMA,)),
    "Thing": ("Class", (SCHEMA,)),
    "Enumeration": ("Class", (SCHEMA,)),
    "GenderType": ("Class", (SCHEMA,)),
    "UnusedBaseClass": ("Class", (SCHEMA,)),
    "crsPurpose": ("Property", (DFO,)),
    "grantEquivalent": ("Property", (DFO,)),
    "beneficiaryGender": ("Property", (DFO,)),
    "amount": ("Property", (SCHEMA,)),
    "financingRecipient": ("Property", (SCHEMA,)),
    "unusedBaseProperty": ("Property", (SCHEMA,)),
    "educationalAttainment": (
        "Property",
        ("dc/base/BLS_CPS", "dc/base/Census", "dc/base/Schema", "dc/base/WorldBank"),
    ),
}

NAMES = {"crsPurpose": "CRS purpose", "educationalAttainment": "Educational attainment"}
"""Listing names; every other term's listing name is its own dcid."""

NODES: dict[str, dict[str, tuple[str, ...]]] = {
    "FinancingEvent": {"typeOf": ("Class",), "subClassOf": ("EconomicActivity",)},
    "FinancialFlow": {"typeOf": ("Class",), "subClassOf": ("FinancingEvent",)},
    "EconomicActivity": {"typeOf": ("Class",), "subClassOf": ("Intangible",)},
    "Intangible": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "Thing": {"typeOf": ("Class",)},
    "Enumeration": {"typeOf": ("Class",), "subClassOf": ("Intangible",)},
    "CRSPurposeCodeEnum": {"typeOf": ("Class",), "subClassOf": ("FinancingPurposeEnum",)},
    "FinancingPurposeEnum": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)},
    "GenderType": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)},
    "UnusedBaseClass": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "crsPurpose": {
        "typeOf": ("Property",),
        "name": ("CRS purpose", "crsPurpose", "Crs purpose"),
        "description": ("The CRS purpose code of a flow.",),
        "alternateName": ("purpose", "CRS sector"),
        "domainIncludes": ("FinancingEvent",),
        "rangeIncludes": ("CRSPurposeCodeEnum",),
    },
    "grantEquivalent": {
        "typeOf": ("Property",),
        "domainIncludes": ("FinancialFlow",),
        "rangeIncludes": ("Number",),
    },
    "beneficiaryGender": {"typeOf": ("Property",), "domainIncludes": ("FinancialFlow",)},
    "amount": {"typeOf": ("Property",), "rangeIncludes": ("Quantity",)},
    "Quantity": {"typeOf": ("Class",)},
    "financingRecipient": {"typeOf": ("Property",)},
    "unusedBaseProperty": {"typeOf": ("Property",)},
    "educationalAttainment": {"typeOf": ("Property",)},
    "measuredValue": {"typeOf": ("StatType",)},
    "CRSPurposeCode120": {
        "typeOf": ("CRSPurposeCodeEnum",),
        "name": ("Education",),
        "classificationCode": ("120",),
    },
    "CRSPurposeCode12220": {
        "typeOf": ("CRSPurposeCodeEnum",),
        "name": ("Basic health care",),
        "specializationOf": ("CRSPurposeCode120",),
        "classificationCode": ("12220",),
    },
    "Female": {"typeOf": ("GenderType",)},
    "Male": {"typeOf": ("GenderType",)},
}

CRS_MEMBERS = (
    ArcSubject(
        dcid="CRSPurposeCode12220",
        names=("Basic health care",),
        types=("CRSPurposeCodeEnum",),
        provenances=(DFO,),
    ),
    ArcSubject(
        dcid="CRSPurposeCode120",
        names=("Education",),
        types=("CRSPurposeCodeEnum",),
        provenances=(DFO,),
    ),
)

GENDER_MEMBERS = tuple(
    ArcSubject(dcid=dcid, names=(dcid,), types=("GenderType",), provenances=(SCHEMA,))
    for dcid in ("Female", "Male")
)

ARC_TARGETS: dict[str, dict[str, tuple[ArcTarget, ...]]] = {
    "measuredValue": {"typeOf": (ArcTarget(value="StatType", provenances=(SCHEMA,)),)},
    "Female": {"typeOf": (ArcTarget(value="GenderType", provenances=("dc/base/Census",)),)},
}

STAMP = IndexStamp(
    graph_endpoint="test",
    exported_at="2026-01-01T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="test",
)


def index_row(dcid: str, constraints: Mapping[str, str]) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialFlow",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingRecipient",),
        constraint_properties=constraints,
    )


def index_of(*rows: StatVarRow, stamp: IndexStamp = STAMP) -> CodeIndex:
    return CodeIndex(stamp=stamp, declarations=(), codes=(), statvars=rows)


INDEX = index_of(
    index_row(
        "Amount_CRS12220_Female",
        {"crsPurpose": "CRSPurposeCode12220", "beneficiaryGender": "Female"},
    )
)


def listing(kind: str, terms: Mapping[str, tuple[str, tuple[str, ...]]]) -> list[ArcSubject]:
    return [
        ArcSubject(
            dcid=dcid, names=(NAMES.get(dcid, dcid),), types=(kind,), provenances=provenances
        )
        for dcid, (term_kind, provenances) in terms.items()
        if term_kind == kind
    ]


def arc_subjects(
    *,
    terms: Mapping[str, tuple[str, tuple[str, ...]]] = TERMS,
    members: Sequence[ArcSubject] = CRS_MEMBERS,
) -> dict[tuple[str, str], Sequence[ArcSubject]]:
    """The reverse `typeOf` subjects of the two listings and the two enumerations."""
    return {
        ("Class", "typeOf"): listing("Class", terms),
        ("Property", "typeOf"): listing("Property", terms),
        ("CRSPurposeCodeEnum", "typeOf"): members,
        ("GenderType", "typeOf"): GENDER_MEMBERS,
    }


def ontology_graph(
    *,
    terms: Mapping[str, tuple[str, tuple[str, ...]]] = TERMS,
    nodes: Mapping[str, Mapping[str, tuple[str, ...]]] = NODES,
    members: Sequence[ArcSubject] = CRS_MEMBERS,
    arc_targets: Mapping[str, Mapping[str, Sequence[ArcTarget]]] = ARC_TARGETS,
) -> FakeGraphNodeReader:
    return FakeGraphNodeReader(
        nodes=nodes,
        arc_subjects=arc_subjects(terms=terms, members=members),
        arc_targets=arc_targets,
    )
