"""`build_schema` against a small fake graph shaped like the dev graph's schema."""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable

import pytest

from code_index import CodeIndex
from dc_graph import UNRESOLVED_NODE, ArcSubject, ArcTarget, GraphReadError, parse_arc_targets
from graph_api import schema_document
from graph_api.schema_document import (
    Schema,
    SchemaListingTruncatedError,
    SchemaNoDfoDeclarationError,
    build_schema,
    declared_by,
)
from graph_api.schema_models import SchemaLiteral, SchemaNode, SchemaNodeReference
from origin_catalogue import BASE, UN_SYSTEM
from origin_catalogue import DFO as DFO_ORIGIN
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row

from .conftest import FakeGraphNodeReader
from .schema_fixture import (
    CATALOGUE,
    CRS_MEMBERS,
    DFO,
    INDEX,
    NODES,
    SCHEMA,
    TERMS,
    arc_subjects,
    index_of,
    index_row,
    listing,
    node_targets,
    schema_graph,
    subclass_subjects,
)

_DEADLINE = 1_000.0
_OTHER = "provenance/Other"


def _nodes(graph: FakeGraphNodeReader, index: CodeIndex = INDEX) -> dict[str, SchemaNode]:
    schema = build_schema(
        reader=graph, code_index=index, catalogue=CATALOGUE, generation="g1", deadline=_DEADLINE
    )
    return {node.ref: node for node in schema.document.nodes}


_LISTED = FakeCatalogue(
    provenances={
        "provenance/BaseListed": provenance_row("provenance/BaseListed", BASE),
        "dc/base/DfoListed": provenance_row("dc/base/DfoListed", DFO_ORIGIN),
        "undata/p/ILO": provenance_row("undata/p/ILO", UN_SYSTEM, segment="ILO"),
    }
)
"""A provenance table whose rows name origins their ids' prefixes would not."""


@pytest.mark.parametrize(
    ("provenances", "expected"),
    [
        ((SCHEMA,), "base"),
        (("dc/base/Census", SCHEMA), "base"),
        ((SCHEMA, DFO), "dfo"),
        ((DFO,), "dfo"),
        (("undata/p/SDG",), "un_system"),
        ((SCHEMA, "undata/p/SDG"), "un_system"),
        ((DFO, "undata/p/SDG"), "dfo"),
        # The provenance table's row decides over the prefix.
        (("provenance/BaseListed",), "base"),
        (("dc/base/DfoListed",), "dfo"),
        (("undata/p/ILO", SCHEMA), "un_system"),
    ],
)
def test_declared_by_names_the_origin_of_the_declaring_provenances(
    provenances: tuple[str, ...], expected: str
) -> None:
    assert declared_by(provenances, catalogue=_LISTED) == expected


def test_declared_by_refuses_no_provenance() -> None:
    with pytest.raises(ValueError):
        declared_by((), catalogue=CATALOGUE)


def test_a_provenance_the_catalogue_lists_as_base_declares_base_nodes() -> None:
    """A DFO-prefixed provenance the catalogue lists as base declares no DFO node, so the build
    of a graph whose only non-base provenance it is refuses."""
    catalogue = FakeCatalogue(provenances={DFO: provenance_row(DFO, BASE)})

    with pytest.raises(SchemaNoDfoDeclarationError):
        build_schema(
            reader=schema_graph(),
            code_index=INDEX,
            catalogue=catalogue,
            generation="g1",
            deadline=_DEADLINE,
        )


def test_document_holds_dfo_terms_their_references_index_names_and_ancestors() -> None:
    # The listed member CRSPurposeCode120, read after the closure, carries a second type.
    terms = {**TERMS, "SectorCode": ("Class", (SCHEMA,)), "CodeKind": ("Class", (SCHEMA,))}
    graph_nodes = {
        **NODES,
        "CRSPurposeCode120": {
            **NODES["CRSPurposeCode120"],
            "typeOf": ("CRSPurposeCodeEnum", "SectorCode"),
        },
        "SectorCode": {"typeOf": ("Class",), "subClassOf": ("CodeKind",)},
        "CodeKind": {"typeOf": ("Class",)},
    }
    arc_targets = {
        "CRSPurposeCode120": {
            "typeOf": (
                ArcTarget(value="CRSPurposeCodeEnum", provenances=(DFO,)),
                ArcTarget(value="SectorCode", provenances=(_OTHER,)),
            )
        },
    }
    nodes = _nodes(schema_graph(terms=terms, nodes=graph_nodes, arc_targets=arc_targets))

    assert sorted(nodes) == [
        "CRSPurposeCode120",  # listed member of a DFO enumeration
        "CRSPurposeCode12220",  # constraint value
        "CRSPurposeCodeEnum",
        "CodeKind",  # SectorCode's subClassOf
        "EconomicActivity",  # FinancingEvent's subClassOf
        "Enumeration",  # ancestor
        "Female",  # constraint value
        "FinancialFlow",
        "FinancingEvent",
        "FinancingPurposeEnum",
        "GenderType",  # Female's typeOf
        "Intangible",  # ancestor
        "SectorCode",  # CRSPurposeCode120's second typeOf
        "Thing",  # ancestor
        "amount",  # measured property
        "beneficiaryGender",
        "crsPurpose",
        "financingRecipient",  # observation property
        "grantEquivalent",
        "measuredValue",  # stat type
    ]
    # amount is base, so its range Quantity stays out though the graph holds it.
    # The member's second typeOf arc adds the provenance that declares it.
    assert nodes["CRSPurposeCode120"].provenances == [DFO, _OTHER]


def test_a_dcid_the_graph_does_not_hold_stays_in_the_arc_only() -> None:
    nodes = _nodes(schema_graph())

    assert "Number" not in nodes
    assert nodes["grantEquivalent"].range_includes == ["Number"]


def test_node_fields_and_declaring_provenances() -> None:
    nodes = _nodes(schema_graph())

    crs = nodes["crsPurpose"]
    # The greatest name by code point; the name equal to the dcid is no name.
    assert crs.label == "Crs purpose"
    assert crs.description == "The CRS purpose code of a flow."
    assert crs.types == ["Property"]
    assert crs.alternate_name == ["CRS sector", "purpose"]
    assert crs.domain_includes == ["FinancingEvent"]
    assert crs.range_includes == ["CRSPurposeCodeEnum"]
    assert crs.provenances == [DFO]
    assert crs.declared_by == "dfo"
    assert nodes["FinancialFlow"].sub_class_of == ["FinancingEvent"]
    assert nodes["CRSPurposeCode12220"].specialization_of == ["CRSPurposeCode120"]
    assert nodes["CRSPurposeCode12220"].classification_code == ["12220"]
    assert (nodes["amount"].provenances, nodes["amount"].declared_by) == ([SCHEMA], "base")
    assert nodes["EconomicActivity"].declared_by == "base"
    # Every node takes its provenances from the typeOf targets of its own read.
    assert (nodes["Female"].provenances, nodes["Female"].declared_by) == (
        ["dc/base/Census"],
        "base",
    )
    assert nodes["measuredValue"].provenances == [SCHEMA]


def test_enumerations_are_value_lists_of_their_instances() -> None:
    nodes = _nodes(schema_graph())

    lists = {dcid for dcid, node in nodes.items() if node.instances is not None}
    assert lists == {"CRSPurposeCodeEnum", "FinancingPurposeEnum", "GenderType"}
    crs = nodes["CRSPurposeCodeEnum"].instances
    assert crs is not None
    assert (crs.refs, crs.complete) == (["CRSPurposeCode120", "CRSPurposeCode12220"], True)
    assert nodes["CRSPurposeCode120"].label == "Education"
    assert nodes["CRSPurposeCode120"].provenances == [DFO]
    # A DFO enumeration lists the instances of its subclasses beside its own none.
    purpose = nodes["FinancingPurposeEnum"].instances
    assert purpose is not None
    assert (purpose.refs, purpose.complete) == (crs.refs, True)
    assert nodes["crsPurpose"].instances is None


def test_a_base_enumeration_lists_only_instances_the_index_uses() -> None:
    nodes = _nodes(schema_graph())

    # No DFO term names GenderType; it joins as the typeOf of the used value Female.
    assert not any(
        "GenderType" in values
        for dcid in TERMS
        if TERMS[dcid][1] == (DFO,)
        for values in NODES[dcid].values()
    )
    gender = nodes["GenderType"]
    assert (gender.declared_by, gender.sub_class_of) == ("base", ["Enumeration"])
    assert gender.instances is not None
    # The instances StatVar rows use are some of the instances the graph holds.
    assert (gender.instances.refs, gender.instances.complete) == (["Female"], False)
    assert "Male" not in nodes


_RANGE_TERMS = {
    **TERMS,
    "currencyOf": ("Property", (DFO,)),
    "recipientCountry": ("Property", (DFO,)),
    "hasMember": ("Property", (DFO,)),
    "Denomination": ("Class", (DFO,)),
    "Unit": ("Class", (DFO,)),
    "Country": ("Class", (SCHEMA,)),
    "Region": ("Class", (SCHEMA,)),
    "baseRanged": ("Property", (SCHEMA,)),
    "BaseRanged": ("Class", (DFO,)),
}

_RANGE_NODES = {
    **NODES,
    "currencyOf": {"typeOf": ("Property",), "rangeIncludes": ("Denomination",)},
    "recipientCountry": {"typeOf": ("Property",), "rangeIncludes": ("Region",)},
    "hasMember": {"typeOf": ("Property",), "rangeIncludes": ("Thing",)},
    "Denomination": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "Unit": {"typeOf": ("Class",), "subClassOf": ("Denomination",)},
    "UnitCHF": {"typeOf": ("Unit",), "name": ("Swiss franc",)},
    "UnitEUR": {"typeOf": ("Unit",), "name": ("Euro",)},
    "Region": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "Country": {"typeOf": ("Class",), "subClassOf": ("Region",)},
    "country/UGA": {"typeOf": ("Country",)},
    "country/KEN": {"typeOf": ("Country",)},
    "baseRanged": {"typeOf": ("Property",), "rangeIncludes": ("BaseRanged",)},
    "BaseRanged": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "BaseRanged1": {"typeOf": ("BaseRanged",)},
}


def _range_graph(
    *, unit_subjects: tuple[ArcSubject, ...] | None = None
) -> tuple[FakeGraphNodeReader, CodeIndex]:
    """The fixture graph with a DFO range `Denomination` whose instances are all typed by its
    DFO subclass `Unit`, a base range `Region` whose subclass `Country` types a used and an
    unused node, a DFO property ranged on the base `Thing`, and a DFO class `BaseRanged` that
    only the base property `baseRanged`, which a StatVar row names, ranges on."""
    units = tuple(
        ArcSubject(dcid=dcid, names=(dcid,), provenances=(DFO,)) for dcid in ("UnitCHF", "UnitEUR")
    )
    graph = FakeGraphNodeReader(
        nodes=_RANGE_NODES,
        arc_subjects={
            **subclass_subjects(_RANGE_NODES, _RANGE_TERMS),
            **arc_subjects(terms=_RANGE_TERMS),
            ("Unit", "typeOf"): units if unit_subjects is None else unit_subjects,
            ("BaseRanged", "typeOf"): (
                ArcSubject(dcid="BaseRanged1", names=(), provenances=(DFO,)),
            ),
            ("Country", "typeOf"): (
                ArcSubject(dcid="country/KEN", names=(), provenances=(SCHEMA,)),
                ArcSubject(dcid="country/UGA", names=(), provenances=(SCHEMA,)),
            ),
        },
        arc_targets=node_targets(_RANGE_NODES, _RANGE_TERMS),
    )
    row = index_row("Amount_UGA", {"recipientCountry": "country/UGA", "baseRanged": "Thing"})
    return graph, index_of(*INDEX.statvars, row)


def test_a_dfo_range_and_its_subclasses_list_every_instance() -> None:
    graph, index = _range_graph()

    nodes = _nodes(graph, index)

    for dcid in ("Denomination", "Unit"):
        instances = nodes[dcid].instances
        assert instances is not None
        assert (instances.refs, instances.complete) == (["UnitCHF", "UnitEUR"], True)
    # The instances join the document, though no StatVar row uses them.
    assert nodes["UnitCHF"].label == "Swiss franc"
    assert "UnitEUR" in nodes


def test_only_dfo_ranges_of_dfo_properties_and_base_enumerations_are_value_lists() -> None:
    graph, index = _range_graph()

    nodes = _nodes(graph, index)

    # A base range, its base subclass, the base root it shares with every node, and a DFO
    # subclass of that root list nothing, though StatVar rows use `country/UGA`.
    for dcid in ("Region", "Country", "Thing", "FinancingEvent", "Enumeration"):
        assert nodes[dcid].instances is None, dcid
    assert "country/KEN" not in nodes
    # A DFO class only a base property ranges on lists nothing, and its instance stays out.
    assert nodes["BaseRanged"].instances is None
    assert "BaseRanged1" not in nodes
    # A base enumeration still lists the instances StatVar rows use.
    gender = nodes["GenderType"].instances
    assert gender is not None
    assert (gender.refs, gender.complete) == (["Female"], False)


def _range_schema(graph: object, index: CodeIndex) -> Schema:
    return build_schema(
        reader=graph,  # type: ignore[arg-type]
        code_index=index,
        catalogue=CATALOGUE,
        generation="g1",
        deadline=_DEADLINE,
    )


def test_a_base_range_within_the_cap_joins_the_document_and_is_closed() -> None:
    graph, index = _range_graph()

    schema = _range_schema(graph, index)

    nodes = {node.ref: node for node in schema.document.nodes}
    # No read but `Thing`'s descendant listing reaches UnusedBaseClass.
    assert nodes["UnusedBaseClass"].sub_class_of == ["Thing"]
    # The DFO ranges, and the base ranges whose every descendant the document holds; the
    # graph holds no `Number`.
    assert schema.closed_ranges == {"CRSPurposeCodeEnum", "Denomination", "Region", "Thing"}
    # A base range's instances stay out.
    assert "country/KEN" not in nodes


def test_a_base_range_past_the_cap_adds_none_of_its_classes_and_is_not_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(schema_document, "RANGE_CLOSURE_CAP", 3)
    fake, index = _range_graph()
    graph = _Recorder(fake)

    schema = _range_schema(graph, index)

    nodes = {node.ref: node for node in schema.document.nodes}
    assert "UnusedBaseClass" not in nodes
    assert schema.closed_ranges == {"CRSPurposeCodeEnum", "Denomination", "Region"}
    # `Thing`'s first listing finds five subclasses, past the cap, so it is listed once and
    # no class below it is listed for it.
    listed = [dcid for chunk in _subclass_listings(graph) for dcid in chunk]
    assert listed.count("Thing") == 1
    assert not {"Intangible", "UnusedBaseClass", "BaseRanged"} & set(listed)


@pytest.mark.parametrize(
    ("cap", "closed"), [pytest.param(0, False, id="truncated"), pytest.param(1, True, id="at")]
)
def test_a_dfo_range_whose_subclass_listing_is_truncated_is_not_closed(
    cap: int, closed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schema_document, "SUBCLASS_LISTING_CAP", cap)
    graph, index = _range_graph()

    schema = _range_schema(graph, index)

    # Denomination has one subclass, Unit, which has none.
    assert ("Denomination" in schema.closed_ranges) is closed


def test_a_base_range_whose_subclass_listing_is_truncated_adds_none_of_its_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Area`, a base range within `RANGE_CLOSURE_CAP`, has two subclasses no other read
    reaches. Its listing capped at one subject leaves its tree partial, so neither joins."""
    monkeypatch.setattr(schema_document, "SUBCLASS_LISTING_CAP", 1)
    terms = {**_RANGE_TERMS, "inArea": ("Property", (DFO,)), "Area": ("Class", (SCHEMA,))}
    nodes = {
        **_RANGE_NODES,
        "inArea": {"typeOf": ("Property",), "rangeIncludes": ("Area",)},
        "Area": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
        "AreaA": {"typeOf": ("Class",), "subClassOf": ("Area",)},
        "AreaB": {"typeOf": ("Class",), "subClassOf": ("Area",)},
    }
    graph = FakeGraphNodeReader(
        nodes=nodes,
        arc_subjects={**subclass_subjects(nodes, terms), **arc_subjects(terms=terms)},
        arc_targets=node_targets(nodes, terms),
    )

    schema = _range_schema(graph, INDEX)

    held = {node.ref for node in schema.document.nodes}
    assert "Area" in held
    assert not {"AreaA", "AreaB"} & held
    assert "Area" not in schema.closed_ranges


_GROUP_TERMS = {
    **TERMS,
    "BeneficiaryGroupEnum": ("Class", (DFO,)),
    "AgeGroup": ("Class", (SCHEMA,)),
    "YouthGroup": ("Class", (SCHEMA,)),
}

_GROUP_NODES = {
    **NODES,
    "beneficiaryGender": {**NODES["beneficiaryGender"], "rangeIncludes": ("BeneficiaryGroupEnum",)},
    "BeneficiaryGroupEnum": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)},
    "GenderType": {"typeOf": ("Class",), "subClassOf": ("BeneficiaryGroupEnum",)},
    "Refugees": {"typeOf": ("BeneficiaryGroupEnum",)},
    "AgeGroup": {"typeOf": ("Class",), "subClassOf": ("BeneficiaryGroupEnum",)},
    "YouthGroup": {"typeOf": ("Class",), "subClassOf": ("AgeGroup",)},
}


def _group_nodes() -> dict[str, SchemaNode]:
    """The fixture graph with a DFO value list `BeneficiaryGroupEnum` holding one instance of
    its own above the base `GenderType`, whose instances no StatVar row uses, and the base
    chain `YouthGroup` under `AgeGroup`, which no read but a subclass listing reaches."""
    graph = FakeGraphNodeReader(
        nodes=_GROUP_NODES,
        arc_subjects={
            **subclass_subjects(_GROUP_NODES, _GROUP_TERMS),
            **arc_subjects(terms=_GROUP_TERMS),
            ("BeneficiaryGroupEnum", "typeOf"): (
                ArcSubject(dcid="Refugees", names=("Refugees",), provenances=(DFO,)),
            ),
        },
        arc_targets=node_targets(_GROUP_NODES, _GROUP_TERMS),
    )
    index = index_of(index_row("Amount_CRS12220", {"crsPurpose": "CRSPurposeCode12220"}))
    return _nodes(graph, index)


def test_a_dfo_value_list_with_an_unread_base_subclass_is_not_complete() -> None:
    nodes = _group_nodes()

    # Only the subclass listing reaches GenderType: no StatVar row uses Female or Male.
    assert "GenderType" in nodes
    assert "Female" not in nodes
    group = nodes["BeneficiaryGroupEnum"].instances
    assert group is not None
    assert (group.refs, group.complete) == (["Refugees"], False)


def test_a_dfo_value_list_reads_every_level_of_its_descendants() -> None:
    nodes = _group_nodes()

    assert nodes["YouthGroup"].sub_class_of == ["AgeGroup"]
    assert nodes["AgeGroup"].sub_class_of == ["BeneficiaryGroupEnum"]


@pytest.mark.parametrize(
    ("cap", "complete"), [pytest.param(0, False, id="over-the-cap"), pytest.param(1, True, id="at")]
)
def test_a_value_list_whose_subclass_listing_is_truncated_is_not_complete(
    cap: int, complete: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schema_document, "SUBCLASS_LISTING_CAP", cap)

    nodes = _nodes(schema_graph())

    # FinancingPurposeEnum has one subclass, CRSPurposeCodeEnum, which has none.
    purpose = nodes["FinancingPurposeEnum"].instances
    assert purpose is not None
    assert purpose.complete is complete
    crs = nodes["CRSPurposeCodeEnum"].instances
    assert crs is not None
    assert crs.complete is True


@pytest.mark.parametrize(
    ("cap", "complete"), [pytest.param(1, False, id="over-the-cap"), pytest.param(2, True, id="at")]
)
def test_a_value_list_over_the_listing_cap_is_not_complete(
    cap: int, complete: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schema_document, "INSTANCE_LISTING_CAP", cap)
    graph, index = _range_graph()

    nodes = _nodes(graph, index)

    crs = nodes["CRSPurposeCodeEnum"].instances
    assert crs is not None
    # The listing reads `CRSPurposeCode12220` first, and the index uses it.
    assert crs.complete is complete
    assert len(crs.refs) == cap
    # A subclass's truncated listing leaves its superclass's list incomplete too.
    denomination = nodes["Denomination"].instances
    assert denomination is not None
    assert (denomination.refs, denomination.complete) == (["UnitCHF", "UnitEUR"][:cap], complete)


def test_a_truncated_listing_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema_document, "TERM_LISTING_CAP", 3)

    with pytest.raises(SchemaListingTruncatedError):
        _nodes(schema_graph())


def test_a_listing_subject_with_no_provenance_is_served_undeclared(
    caplog: pytest.LogCaptureFixture,
) -> None:
    terms = {**TERMS, "educationalAttainment": ("Property", ())}

    with caplog.at_level("WARNING", logger=schema_document.__name__):
        schema = build_schema(
            reader=schema_graph(terms=terms),
            code_index=INDEX,
            catalogue=CATALOGUE,
            generation="g1",
            deadline=_DEADLINE,
        )

    term = {term.ref: term for term in schema.terms.terms}["educationalAttainment"]
    assert (term.provenances, term.declared_by) == ([], None)
    assert '"declared_by":null' in term.model_dump_json()
    assert [record.args for record in caplog.records] == [(1, ["educationalAttainment"], 0, [])]


def test_a_held_node_with_no_type_is_served_with_no_provenance() -> None:
    graph_nodes = {**NODES, "UntypedValue": {"name": ("Untyped value",)}}
    row = index_row("Amount_Untyped", {"beneficiaryGender": "UntypedValue"})

    graph = schema_graph(nodes=graph_nodes)
    schema = build_schema(
        reader=graph,
        code_index=index_of(*INDEX.statvars, row),
        catalogue=CATALOGUE,
        generation="g1",
        deadline=_DEADLINE,
    )

    untyped = {node.ref: node for node in schema.document.nodes}["UntypedValue"]
    assert (untyped.types, untyped.provenances, untyped.declared_by) == ([], [], None)
    assert '"declared_by":null' in untyped.model_dump_json()
    # Every node read is the all-properties read, which confirms each empty answer alone.
    assert graph.calls["node_all_arc_targets_many"]
    assert graph.calls["node_arc_targets_many"] == 0


def test_every_other_property_a_node_holds_is_served_typed_and_sorted() -> None:
    crs = node_targets(NODES)["CRSPurposeCode120"]
    arc_targets = {
        "CRSPurposeCode120": {
            **crs,
            "wbCode": (
                ArcTarget(value="EDU", literal=True),
                ArcTarget(value="DOD", literal=True),
            ),
            "isProvisional": (ArcTarget(value="True"),),
            "iso4217NumericCode": (ArcTarget(value="404", literal=True),),
            "combinesMethods": (
                ArcTarget(value="sameText", literal=True),
                ArcTarget(value="sameText"),
                ArcTarget(value="MethodA"),
            ),
            "descriptionUrl": (),
        }
    }

    nodes = _nodes(schema_graph(arc_targets=arc_targets))

    assert nodes["CRSPurposeCode120"].model_dump(mode="json")["other_properties"] == {
        "combinesMethods": [{"ref": "MethodA"}, {"ref": "sameText"}, {"value": "sameText"}],
        "isProvisional": [{"ref": "True"}],
        "iso4217NumericCode": [{"value": "404"}],
        "wbCode": [{"value": "DOD"}, {"value": "EDU"}],
    }
    # The named properties stay in their own members, and a node with no other property
    # holds an empty map.
    assert nodes["CRSPurposeCode120"].classification_code == ["120"]
    assert nodes["Female"].other_properties == {}


_SAME_TEXT_NODE = {"dcid": "sameText", "name": "Same text", "provenanceId": "provenance/A"}
_SAME_TEXT_LITERAL = {"value": "sameText", "provenanceId": "provenance/B"}


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([_SAME_TEXT_NODE, _SAME_TEXT_LITERAL], id="node-first"),
        pytest.param([_SAME_TEXT_LITERAL, _SAME_TEXT_NODE], id="literal-first"),
    ],
)
def test_a_node_and_a_literal_with_the_same_text_are_two_targets(entries: list[dict]) -> None:
    """From the wire, through `parse_arc_targets`, to the served node: each target keeps its
    own provenances, `other_properties` serves both, and a named property lists the text
    once."""
    wire = {"combinesMethods": {"nodes": entries}, "alternateName": {"nodes": entries}}

    parsed = parse_arc_targets(wire, ["combinesMethods", "alternateName"])
    crs = node_targets(NODES)["CRSPurposeCode120"]
    nodes = _nodes(schema_graph(arc_targets={"CRSPurposeCode120": {**crs, **parsed}}))

    node = ArcTarget(value="sameText", name="Same text", provenances=("provenance/A",))
    literal = ArcTarget(value="sameText", provenances=("provenance/B",), literal=True)
    assert parsed["combinesMethods"] == tuple(
        literal if "value" in entry else node for entry in entries
    )
    served = nodes["CRSPurposeCode120"].model_dump(mode="json")
    assert served["other_properties"] == {
        "combinesMethods": [{"ref": "sameText"}, {"value": "sameText"}]
    }
    assert served["alternate_name"] == ["sameText"]


def test_an_other_property_value_is_one_string_member() -> None:
    schema = SchemaNode.model_json_schema(mode="serialization")
    values = schema["properties"]["other_properties"]["additionalProperties"]["items"]["anyOf"]
    members = [schema["$defs"][value["$ref"].rsplit("/", 1)[1]] for value in values]

    assert [
        (member["required"], member["properties"], member["additionalProperties"])
        for member in members
    ] == [
        (
            ["ref"],
            {"ref": {"description": "the node's dcid", "title": "Ref", "type": "string"}},
            False,
        ),
        (["value"], {"value": {"title": "Value", "type": "string"}}, False),
    ]
    # A literal stays the graph's string, never a number or a boolean.
    for model, key in ((SchemaLiteral, "value"), (SchemaNodeReference, "ref")):
        for parsed in (404, True):
            with pytest.raises(ValueError):
                model.model_validate({key: parsed})


def test_other_properties_leave_out_long_literals_and_values_past_the_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    longest = "x" * schema_document.OTHER_LITERAL_MAX_CHARS
    many = schema_document.OTHER_VALUES_MAX + 5
    arc_targets = {
        "CRSPurposeCode120": {
            **node_targets(NODES)["CRSPurposeCode120"],
            "geoJsonCoordinates": (
                ArcTarget(value=longest + "x", literal=True),
                ArcTarget(value=longest, literal=True),
            ),
            "kmlCoordinates": (ArcTarget(value=longest + "x", literal=True),),
            "containsCode": tuple(
                ArcTarget(value=f"code{i:03d}", literal=True) for i in reversed(range(many))
            ),
        }
    }

    with caplog.at_level("WARNING", logger=schema_document.__name__):
        nodes = _nodes(schema_graph(arc_targets=arc_targets))

    others = nodes["CRSPurposeCode120"].model_dump(mode="json")["other_properties"]
    assert others["geoJsonCoordinates"] == [{"value": longest}]
    assert "kmlCoordinates" not in others
    assert others["containsCode"] == [
        {"value": f"code{i:03d}"} for i in range(schema_document.OTHER_VALUES_MAX)
    ]
    assert [record.args for record in caplog.records] == [
        (
            7,
            schema_document.OTHER_LITERAL_MAX_CHARS,
            schema_document.OTHER_VALUES_MAX,
            [
                ("CRSPurposeCode120", "containsCode"),
                ("CRSPurposeCode120", "geoJsonCoordinates"),
                ("CRSPurposeCode120", "kmlCoordinates"),
            ],
        )
    ]


def test_a_node_with_one_unsourced_type_is_declared_by_its_sourced_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arc_targets = {
        "crsPurpose": {
            **node_targets(NODES)["crsPurpose"],
            "typeOf": (ArcTarget(value="Property", provenances=(DFO,)), ArcTarget(value="Thing")),
        }
    }

    with caplog.at_level("WARNING", logger=schema_document.__name__):
        nodes = _nodes(schema_graph(arc_targets=arc_targets))

    crs = nodes["crsPurpose"]
    assert (crs.provenances, crs.declared_by) == ([DFO], "dfo")
    assert [record.args for record in caplog.records] == [(0, [], 1, ["crsPurpose"])]


def test_a_document_no_dfo_provenance_declares_refuses() -> None:
    base_terms = {dcid: (kind, (SCHEMA,)) for dcid, (kind, _) in TERMS.items()}
    base_targets = {
        dcid: {
            prop: tuple(
                ArcTarget(value=target.value, provenances=(SCHEMA,) if prop == "typeOf" else ())
                for target in targets
            )
            for prop, targets in arcs.items()
        }
        for dcid, arcs in node_targets(NODES, base_terms).items()
    }

    with pytest.raises(SchemaNoDfoDeclarationError):
        _nodes(schema_graph(terms=base_terms, arc_targets=base_targets))


def test_a_typed_node_whose_type_carries_no_provenance_is_served_undeclared(
    caplog: pytest.LogCaptureFixture,
) -> None:
    arc_targets = {"Female": {"typeOf": (ArcTarget(value="GenderType"),)}}

    with caplog.at_level("WARNING", logger=schema_document.__name__):
        nodes = _nodes(schema_graph(arc_targets=arc_targets))

    female = nodes["Female"]
    assert (female.types, female.provenances, female.declared_by) == (["GenderType"], [], None)
    assert [record.args for record in caplog.records] == [(0, [], 1, ["Female"])]


def test_a_term_that_is_also_a_member_is_declared_by_its_node_read_and_its_listing() -> None:
    # amount is a base Property and a DFO-declared member of CRSPurposeCodeEnum.
    graph_nodes = {
        **NODES,
        "amount": {**NODES["amount"], "typeOf": ("Property", "CRSPurposeCodeEnum")},
    }
    members = (
        *CRS_MEMBERS,
        ArcSubject(dcid="amount", names=(), provenances=(DFO,)),
    )
    amount = {
        "typeOf": (
            ArcTarget(value="Property", provenances=(SCHEMA,)),
            ArcTarget(value="CRSPurposeCodeEnum", provenances=(DFO,)),
        )
    }

    schema = build_schema(
        reader=schema_graph(nodes=graph_nodes, members=members, arc_targets={"amount": amount}),
        code_index=INDEX,
        catalogue=CATALOGUE,
        generation="g1",
        deadline=_DEADLINE,
    )

    node = {node.ref: node for node in schema.document.nodes}["amount"]
    term = {term.ref: term for term in schema.terms.terms}["amount"]
    # The node's provenances come from its own read, the term's from its Property listing.
    assert (node.provenances, node.declared_by) == ([SCHEMA, DFO], "dfo")
    assert (term.provenances, term.declared_by) == ([SCHEMA], "base")


def test_terms_list_every_class_and_property() -> None:
    graph = FakeGraphNodeReader(
        nodes=NODES,
        arc_targets=node_targets(NODES),
        arc_subjects={
            **arc_subjects(),
            ("Property", "typeOf"): [
                *listing("Property", TERMS),
                ArcSubject(dcid="FinancialFlow", names=("Financial flow",), provenances=(SCHEMA,)),
            ],
        },
    )

    terms = build_schema(
        reader=graph, code_index=INDEX, catalogue=CATALOGUE, generation="g1", deadline=_DEADLINE
    ).terms

    assert terms.generation == "g1"
    by_dcid = {term.ref: term for term in terms.terms}
    assert sorted(by_dcid) == sorted(TERMS)
    attainment = by_dcid["educationalAttainment"]
    assert attainment.declared_by == "base"
    assert attainment.provenances == sorted(TERMS["educationalAttainment"][1])
    crs = by_dcid["crsPurpose"]
    assert (crs.label, crs.types, crs.declared_by) == ("CRS purpose", ["Property"], "dfo")
    # A listing name equal to the dcid is no name.
    assert by_dcid["amount"].label is None
    # A dcid in both listings is one term with both kinds and every provenance.
    flow = by_dcid["FinancialFlow"]
    assert (flow.label, flow.types, flow.provenances, flow.declared_by) == (
        "Financial flow",
        ["Class", "Property"],
        [SCHEMA, DFO],
        "dfo",
    )


def test_an_unresolved_target_changes_no_served_byte_and_is_recorded() -> None:
    unresolved = ArcTarget(value=UNRESOLVED_NODE, provenances=(_OTHER,))
    targets = node_targets(NODES)
    marked = {
        "Female": {"typeOf": (*targets["Female"]["typeOf"], unresolved)},
        "GenderType": {
            **targets["GenderType"],
            "subClassOf": (*targets["GenderType"]["subClassOf"], unresolved),
        },
        # `grantEquivalent` ranges on `Number`, which the plain graph does not hold.
        "Number": {"typeOf": (unresolved,), "subClassOf": (unresolved,)},
    }

    plain, with_marks = (
        build_schema(
            reader=graph, code_index=INDEX, catalogue=CATALOGUE, generation="g1", deadline=_DEADLINE
        )
        for graph in (schema_graph(), schema_graph(arc_targets=marked))
    )

    assert with_marks.document.model_dump_json() == plain.document.model_dump_json()
    assert with_marks.terms.model_dump_json() == plain.terms.model_dump_json()
    assert (plain.unresolved_types, plain.unresolved_parents) == (frozenset(), frozenset())
    assert (with_marks.unresolved_types, with_marks.unresolved_parents) == (
        {"Female", "Number"},
        {"GenderType", "Number"},
    )
    # A node answered only with unresolved targets is neither held nor absent.
    assert "Number" in plain.absent
    assert with_marks.absent == plain.absent - {"Number"}


def test_two_builds_from_reordered_graphs_serialise_to_the_same_bytes() -> None:
    reordered_terms = dict(reversed(TERMS.items()))
    reordered_nodes = {
        dcid: {prop: tuple(reversed(values)) for prop, values in reversed(arcs.items())}
        for dcid, arcs in reversed(NODES.items())
    }
    builds = [
        build_schema(
            reader=graph, code_index=INDEX, catalogue=CATALOGUE, generation="g1", deadline=_DEADLINE
        )
        for graph in (
            schema_graph(),
            schema_graph(terms=reordered_terms, nodes=reordered_nodes, members=CRS_MEMBERS[::-1]),
        )
    ]

    documents = {build.document.model_dump_json() for build in builds}
    terms = {build.terms.model_dump_json() for build in builds}
    assert len(documents) == len(terms) == 1
    dcids = [node.ref for node in builds[0].document.nodes]
    assert dcids == sorted(dcids)


class _Recorder:
    """Pass every read to `reader` from any thread, recording each call's method, first
    argument, listed property and deadline, after waiting `delay(call)` seconds, where `call`
    is the method and first argument, and raising `fail(call)` where it returns an error."""

    def __init__(
        self,
        reader: FakeGraphNodeReader,
        *,
        delay: Callable[[tuple[str, object]], float] = lambda call: 0.0,
        fail: Callable[[tuple[str, object]], Exception | None] = lambda call: None,
    ) -> None:
        self._reader = reader
        self._delay = delay
        self._fail = fail
        self._lock = threading.Lock()
        self.started: list[tuple[str, object]] = []
        self.properties: list[object] = []
        self.landed: list[tuple[str, object]] = []
        self.deadlines: list[float | None] = []

    def __getattr__(self, name: str) -> Callable[..., object]:
        method = getattr(self._reader, name)

        def record(first: object, /, **kwargs: object) -> object:
            call = (name, first)
            with self._lock:
                self.started.append(call)
                self.properties.append(kwargs.get("property"))
                self.deadlines.append(kwargs.get("deadline"))  # type: ignore[arg-type]
            time.sleep(self._delay(call))
            if (error := self._fail(call)) is not None:
                raise error
            answer = method(first, **kwargs)
            with self._lock:
                self.landed.append(call)
            return answer

        return record


def _subclass_listings(graph: _Recorder) -> list[tuple[str, ...]]:
    """The classes of each `<-subClassOf` listing request `graph` started."""
    return [
        dcids  # type: ignore[misc]
        for (method, dcids), prop in zip(graph.started, graph.properties, strict=True)
        if method == "node_arc_subjects_many" and prop == "subClassOf"
    ]


def _build(graph: object, index: CodeIndex = INDEX) -> str:
    schema = build_schema(
        reader=graph,  # type: ignore[arg-type]
        code_index=index,
        catalogue=CATALOGUE,
        generation="g1",
        deadline=_DEADLINE,
    )
    return "\n".join(
        (
            schema.document.model_dump_json(),
            schema.terms.model_dump_json(),
            json.dumps(sorted(schema.closed_ranges)),
        )
    )


_LARGE_TERMS = {
    **TERMS,
    **{f"extraProperty{i}": ("Property", (DFO,)) for i in range(50)},
    "SubPurposeEnum": ("Class", (DFO,)),
    "BasePurposeEnum": ("Class", (SCHEMA,)),
    "rangedOnDenomination": ("Property", (DFO,)),
    "LateDenomination": ("Class", (DFO,)),
    "MidUnit": ("Class", (SCHEMA,)),
    "LateUnit": ("Class", (DFO,)),
    "GroupEnum": ("Class", (DFO,)),
    "BaseGroup": ("Class", (SCHEMA,)),
    "BaseSubGroup": ("Class", (SCHEMA,)),
}


_RANGED_TERMS = {
    "rangedOnThing": ("Property", (DFO,)),
    "rangedOnPlace": ("Property", (DFO,)),
    "Place": ("Class", (SCHEMA,)),
    "Country": ("Class", (SCHEMA,)),
    "City": ("Class", (SCHEMA,)),
}

_RANGED_NODES = {
    "rangedOnThing": {"typeOf": ("Property",), "rangeIncludes": ("Thing",)},
    "rangedOnPlace": {"typeOf": ("Property",), "rangeIncludes": ("Place",)},
    "Place": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
    "Country": {"typeOf": ("Class",), "subClassOf": ("Place",)},
    "City": {"typeOf": ("Class",), "subClassOf": ("Country",)},
}


def _large_graph(*, ranged: bool = False) -> tuple[FakeGraphNodeReader, CodeIndex]:
    """The fixture graph with fifty more DFO properties, each used by an index row with its own
    value, so the seed takes several node reads, a DFO enumeration `SubPurposeEnum` whose
    chain reaches `Enumeration` only through the base `BasePurposeEnum`, a level after the
    fixture's enumerations do, and a DFO class `LateUnit` that lists in full as a subclass of
    the DFO range `LateDenomination` only once its base parent `MidUnit` is read, and a DFO
    enumeration `GroupEnum` above the base chain `BaseSubGroup` under `BaseGroup`, which only
    its subclass listings reach. With `ranged`, DFO properties range on the base `Thing`,
    with four direct subclasses, and on the base `Place`, with the chain `City` under
    `Country` under it."""
    terms = {**_LARGE_TERMS, **(_RANGED_TERMS if ranged else {})}
    nodes = {
        **NODES,
        **{
            f"extraProperty{i}": {"typeOf": ("Property",), "domainIncludes": ("FinancialFlow",)}
            for i in range(50)
        },
        **{f"extraValue{i}": {"typeOf": ("StatType",)} for i in range(50)},
        "SubPurposeEnum": {"typeOf": ("Class",), "subClassOf": ("BasePurposeEnum",)},
        "BasePurposeEnum": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)},
        "SubPurpose1": {"typeOf": ("SubPurposeEnum",)},
        "rangedOnDenomination": {"typeOf": ("Property",), "rangeIncludes": ("LateDenomination",)},
        "LateDenomination": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
        "MidUnit": {"typeOf": ("Class",), "subClassOf": ("LateDenomination",)},
        "LateUnit": {"typeOf": ("Class",), "subClassOf": ("MidUnit",)},
        "LateUnit1": {"typeOf": ("LateUnit",)},
        "GroupEnum": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)},
        "BaseGroup": {"typeOf": ("Class",), "subClassOf": ("GroupEnum",)},
        "BaseSubGroup": {"typeOf": ("Class",), "subClassOf": ("BaseGroup",)},
        **(_RANGED_NODES if ranged else {}),
    }
    rows = [index_row(f"Extra{i}", {f"extraProperty{i}": f"extraValue{i}"}) for i in range(50)]
    graph = FakeGraphNodeReader(
        nodes=nodes,
        arc_subjects={
            **subclass_subjects(nodes, terms),
            **arc_subjects(terms=terms),
            ("SubPurposeEnum", "typeOf"): (
                ArcSubject(dcid="SubPurpose1", names=("Sub purpose",), provenances=(DFO,)),
            ),
            ("LateUnit", "typeOf"): (
                ArcSubject(dcid="LateUnit1", names=("Late unit",), provenances=(DFO,)),
            ),
        },
        arc_targets=node_targets(nodes, terms),
    )
    return graph, index_of(*INDEX.statvars, *rows)


def test_each_node_is_read_once_and_each_dfo_class_listed_once() -> None:
    fake, index = _large_graph()
    graph = _Recorder(fake)

    _build(graph, index)

    node_reads = [dcids for method, dcids in graph.started if method == "node_all_arc_targets_many"]
    read = [dcid for dcids in node_reads for dcid in dcids]  # type: ignore[attr-defined]
    assert len(read) == len(set(read))
    assert {"extraValue49", "Thing", "CRSPurposeCode120", "SubPurpose1", "LateUnit1"} <= set(read)
    assert max(len(dcids) for dcids in node_reads) == 12  # type: ignore[arg-type]
    listings = sorted(
        [
            *(
                (str(prop), str(dcid))
                for (method, dcid), prop in zip(graph.started, graph.properties, strict=True)
                if method == "node_arc_subjects"
            ),
            *(("subClassOf", dcid) for chunk in _subclass_listings(graph) for dcid in chunk),
        ]
    )
    dfo_classes = [
        dcid
        for dcid, (kind, provenances) in _LARGE_TERMS.items()
        if provenances == (DFO,) and kind == "Class"
    ]
    # Each DFO value list and each descendant its subclass listings find is listed once.
    descendants = [
        "Number",  # a base range with no subclass
        "BaseGroup",
        "BaseSubGroup",
        "CRSPurposeCodeEnum",
        "FinancingPurposeEnum",
        "GroupEnum",
        "LateDenomination",
        "LateUnit",
        "MidUnit",
        "SubPurposeEnum",
    ]
    assert listings == sorted(
        [
            ("typeOf", "Class"),
            ("typeOf", "Property"),
            *(("typeOf", dcid) for dcid in dfo_classes),
            *(("subClassOf", dcid) for dcid in descendants),
        ]
    )
    assert {method for method, _ in graph.started} == {
        "node_all_arc_targets_many",
        "node_arc_subjects",
        "node_arc_subjects_many",
    }


_WIDE = tuple(f"Wide{i:02d}" for i in range(30))


@pytest.mark.parametrize(
    ("parent", "terms", "nodes"),
    [
        pytest.param(
            "WideEnum",
            {"WideEnum": ("Class", (DFO,))},
            {"WideEnum": {"typeOf": ("Class",), "subClassOf": ("Enumeration",)}},
            id="dfo-value-list",
        ),
        pytest.param(
            "WideBase",
            {"rangedOnWide": ("Property", (DFO,)), "WideBase": ("Class", (SCHEMA,))},
            {
                "rangedOnWide": {"typeOf": ("Property",), "rangeIncludes": ("WideBase",)},
                "WideBase": {"typeOf": ("Class",), "subClassOf": ("Thing",)},
            },
            id="base-range",
        ),
    ],
)
def test_a_level_of_subclasses_is_listed_in_chunks(
    parent: str,
    terms: dict[str, tuple[str, tuple[str, ...]]],
    nodes: dict[str, dict[str, tuple[str, ...]]],
) -> None:
    """The 30 subclasses `parent`'s listing finds, none with a subclass, are listed in
    ceil(30 / 12) requests."""
    graph_nodes = {
        **NODES,
        **nodes,
        **{dcid: {"typeOf": ("Class",), "subClassOf": (parent,)} for dcid in _WIDE},
    }
    graph = _Recorder(schema_graph(terms={**TERMS, **terms}, nodes=graph_nodes))

    nodes_read = json.loads(_build(graph).split("\n")[0])["nodes"]

    wide = [chunk for chunk in _subclass_listings(graph) if set(chunk) & set(_WIDE)]
    chunk = schema_document._NODE_READ_CHUNK
    assert sorted(len(listing) for listing in wide) == [6, chunk, chunk]
    assert sorted(dcid for listing in wide for dcid in listing) == list(_WIDE)
    assert set(_WIDE) <= {node["ref"] for node in nodes_read}


def _reversed(call: tuple[str, object], calls: dict[tuple[str, object], int]) -> float:
    """The earlier a read starts, the later it lands."""
    calls.setdefault((call[0], str(call[1])), len(calls))
    return 0.002 * max(0, 40 - calls[(call[0], str(call[1]))])


@pytest.mark.parametrize(
    "ranged",
    [
        pytest.param(False, id="dfo-ranges"),
        # `Thing`'s first listing passes the cap of 3 and `Place`'s closure stays within it.
        pytest.param(True, id="base-ranges"),
    ],
)
@pytest.mark.parametrize(
    "order", ["term-listings-last", "instance-listings-last", "reversed", "seed-1", "seed-2"]
)
def test_answers_arriving_in_any_order_build_the_same_bytes(
    order: str, ranged: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schema_document, "RANGE_CLOSURE_CAP", 3)
    fake, index = _large_graph(ranged=ranged)
    with monkeypatch.context() as serial:
        serial.setattr(schema_document, "BUILD_CONCURRENCY", 1)
        reference = _build(_Recorder(fake), index)
    document, _, closed = reference.split("\n")
    nodes = {node["ref"]: node for node in json.loads(document)["nodes"]}
    if ranged:
        assert {"Country", "City"} <= nodes.keys()
        assert "UnusedBaseClass" not in nodes
        assert "Place" in json.loads(closed)
        assert "Thing" not in json.loads(closed)
    assert nodes["SubPurposeEnum"]["instances"]["refs"] == ["SubPurpose1"]
    assert "SubPurpose1" in nodes
    # MidUnit is base, so it lists only the used instances, and none are; LateDenomination
    # then cannot know the instances typed by MidUnit alone.
    assert nodes["MidUnit"]["instances"] is None
    assert nodes["LateUnit"]["instances"] == {"refs": ["LateUnit1"], "complete": True}
    assert nodes["LateDenomination"]["instances"] == {"refs": ["LateUnit1"], "complete": False}
    assert "LateUnit1" in nodes
    # Only GroupEnum's subclass listings reach its base descendants, which leave it incomplete
    # with no instance, so it lists nothing.
    assert nodes["BaseSubGroup"]["sub_class_of"] == ["BaseGroup"]
    assert nodes["GroupEnum"]["instances"] is None
    jitter = random.Random(order)
    calls: dict[tuple[str, object], int] = {}
    delays: dict[str, Callable[[tuple[str, object]], float]] = {
        "term-listings-last": lambda call: 0.05 if call[1] in ("Class", "Property") else 0.0,
        "instance-listings-last": lambda call: (
            0.05 if call[0] == "node_arc_subjects" and call[1] not in ("Class", "Property") else 0.0
        ),
        "reversed": lambda call: _reversed(call, calls),
        "seed-1": lambda call: jitter.uniform(0, 0.02),
        "seed-2": lambda call: jitter.uniform(0, 0.02),
    }
    graph = _Recorder(fake, delay=delays[order])

    assert _build(graph, index) == reference
    # The answers landed in another order than the reads started.
    assert graph.landed != graph.started


@pytest.mark.parametrize(
    "failing",
    [
        pytest.param(("node_all_arc_targets_many", "Intangible"), id="closure-level"),
        pytest.param(("node_arc_subjects", "FinancingEvent"), id="instance-listing"),
        pytest.param(("node_arc_subjects", "Property"), id="term-listing"),
    ],
)
def test_a_read_failing_mid_pipeline_ends_the_build_with_its_error(
    failing: tuple[str, str],
) -> None:
    error = GraphReadError("the mixer answered 503")
    method, dcid = failing

    def fail(call: tuple[str, object]) -> Exception | None:
        names = call[1] if isinstance(call[1], tuple) else (call[1],)
        return error if call[0] == method and dcid in names else None

    graph = _Recorder(schema_graph(), fail=fail)

    with pytest.raises(GraphReadError) as raised:
        _build(graph)

    assert raised.value is error
    assert graph.started


def test_a_failed_read_ends_the_build_without_waiting_for_the_reads_in_flight() -> None:
    error = GraphReadError("the mixer answered 400")
    release = threading.Event()

    def hold_node_reads(call: tuple[str, object]) -> float:
        if call[0] == "node_all_arc_targets_many":
            release.wait(timeout=10)
        return 0.0

    def fail_listing(call: tuple[str, object]) -> Exception | None:
        return error if call == ("node_arc_subjects", "Property") else None

    graph = _Recorder(schema_graph(), delay=hold_node_reads, fail=fail_listing)
    started = time.monotonic()
    try:
        with pytest.raises(GraphReadError) as raised:
            _build(graph)
        elapsed = time.monotonic() - started
        # The next build runs while the abandoned reads still hang.
        rebuilt = _build(schema_graph())
    finally:
        release.set()

    assert raised.value is error
    assert elapsed < 2
    assert graph.started.count(("node_arc_subjects", "Property")) == 1
    assert rebuilt == _build(schema_graph())


def test_every_read_carries_the_deadline() -> None:
    fake = schema_graph()
    graph = _Recorder(fake)

    _build(graph)

    assert len(graph.deadlines) == sum(fake.calls.values())
    assert set(graph.deadlines) == {_DEADLINE}
