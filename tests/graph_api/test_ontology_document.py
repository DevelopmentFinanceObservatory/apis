"""`build_ontology` against a small fake graph shaped like the dev graph's ontology."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from code_index import CodeIndex
from dc_graph import UNRESOLVED_NODE, ArcSubject, ArcTarget
from graph_api import ontology_document
from graph_api.ontology_document import (
    OntologyListingTruncatedError,
    OntologyProvenanceUnreadError,
    build_ontology,
    declared_by,
)
from graph_api.ontology_models import OntologyNode

from .conftest import FakeGraphNodeReader
from .ontology_fixture import (
    ARC_TARGETS,
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
    ontology_graph,
)

_DEADLINE = 1_000.0
_OTHER = "provenance/Other"


def _nodes(graph: FakeGraphNodeReader, index: CodeIndex = INDEX) -> dict[str, OntologyNode]:
    ontology = build_ontology(reader=graph, code_index=index, generation="g1", deadline=_DEADLINE)
    return {node.dcid: node for node in ontology.document.nodes}


@pytest.mark.parametrize(
    ("provenances", "expected"),
    [
        ((SCHEMA,), "base"),
        (("dc/base/Census", SCHEMA), "base"),
        ((SCHEMA, DFO), "instance"),
        ((DFO,), "instance"),
    ],
)
def test_declared_by_is_base_only_when_every_provenance_is_base(
    provenances: tuple[str, ...], expected: str
) -> None:
    assert declared_by(provenances) == expected


def test_declared_by_refuses_no_provenance() -> None:
    with pytest.raises(ValueError):
        declared_by(())


def test_document_holds_instance_terms_their_references_index_names_and_ancestors() -> None:
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
        **ARC_TARGETS,
        "CRSPurposeCode120": {
            "typeOf": (
                ArcTarget(value="CRSPurposeCodeEnum", provenances=(DFO,)),
                ArcTarget(value="SectorCode", provenances=(_OTHER,)),
            )
        },
    }
    nodes = _nodes(ontology_graph(terms=terms, nodes=graph_nodes, arc_targets=arc_targets))

    assert sorted(nodes) == [
        "CRSPurposeCode120",  # listed member of an instance enumeration
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


class _RecordingReader(FakeGraphNodeReader):
    """Records every dcid `node_arcs_many` is asked for."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.requested: set[str] = set()

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.requested.update(dcids)
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def test_an_unresolved_value_is_dropped_where_the_arcs_are_read() -> None:
    graph_nodes = {
        **NODES,
        "crsPurpose": {
            **NODES["crsPurpose"],
            "description": (UNRESOLVED_NODE,),
            "rangeIncludes": (UNRESOLVED_NODE, "CRSPurposeCodeEnum"),
        },
    }
    graph = _RecordingReader(
        nodes=graph_nodes, arc_subjects=arc_subjects(), arc_targets=ARC_TARGETS
    )

    nodes = _nodes(graph)

    crs = nodes["crsPurpose"]
    assert (crs.description, crs.range_includes) == (None, ["CRSPurposeCodeEnum"])
    assert UNRESOLVED_NODE not in graph.requested
    assert all(UNRESOLVED_NODE not in node.model_dump_json() for node in nodes.values())


def test_a_dcid_the_graph_does_not_hold_stays_in_the_arc_only() -> None:
    nodes = _nodes(ontology_graph())

    assert "Number" not in nodes
    assert nodes["grantEquivalent"].range_includes == ["Number"]


def test_node_fields_and_declaring_provenances() -> None:
    nodes = _nodes(ontology_graph())

    crs = nodes["crsPurpose"]
    # The greatest name by code point; the name equal to the dcid is no name.
    assert crs.name == "Crs purpose"
    assert crs.description == "The CRS purpose code of a flow."
    assert crs.types == ["Property"]
    assert crs.alternate_name == ["CRS sector", "purpose"]
    assert crs.domain_includes == ["FinancingEvent"]
    assert crs.range_includes == ["CRSPurposeCodeEnum"]
    assert crs.provenances == [DFO]
    assert crs.declared_by == "instance"
    assert nodes["FinancialFlow"].sub_class_of == ["FinancingEvent"]
    assert nodes["CRSPurposeCode12220"].specialization_of == ["CRSPurposeCode120"]
    assert nodes["CRSPurposeCode12220"].classification_code == ["12220"]
    assert (nodes["amount"].provenances, nodes["amount"].declared_by) == ([SCHEMA], "base")
    assert nodes["EconomicActivity"].declared_by == "base"
    # Nodes outside the listings take their provenances from their typeOf targets.
    assert (nodes["Female"].provenances, nodes["Female"].declared_by) == (
        ["dc/base/Census"],
        "base",
    )
    assert nodes["measuredValue"].provenances == [SCHEMA]


def test_enumerations_are_marked_and_list_members() -> None:
    nodes = _nodes(ontology_graph())

    enumerations = {dcid for dcid, node in nodes.items() if node.enumeration is not None}
    assert enumerations == {"CRSPurposeCodeEnum", "FinancingPurposeEnum", "GenderType"}
    crs = nodes["CRSPurposeCodeEnum"].enumeration
    assert crs is not None
    assert (crs.members, crs.truncated) == (["CRSPurposeCode120", "CRSPurposeCode12220"], False)
    assert nodes["CRSPurposeCode120"].name == "Education"
    assert nodes["CRSPurposeCode120"].provenances == [DFO]
    # An instance enumeration with no members of its own lists none.
    assert nodes["FinancingPurposeEnum"].enumeration is not None
    assert nodes["FinancingPurposeEnum"].enumeration.members == []


def test_a_base_enumeration_lists_only_members_the_index_uses() -> None:
    nodes = _nodes(ontology_graph())

    # No instance term names GenderType; it joins as the typeOf of the used value Female.
    assert not any(
        "GenderType" in values
        for dcid in TERMS
        if TERMS[dcid][1] == (DFO,)
        for values in NODES[dcid].values()
    )
    gender = nodes["GenderType"]
    assert (gender.declared_by, gender.sub_class_of) == ("base", ["Enumeration"])
    assert gender.enumeration is not None
    assert gender.enumeration.members == ["Female"]
    assert "Male" not in nodes


def test_an_enumeration_over_the_member_cap_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ontology_document, "ENUMERATION_MEMBER_CAP", 1)

    crs = _nodes(ontology_graph())["CRSPurposeCodeEnum"].enumeration

    assert crs is not None
    assert (crs.members, crs.truncated) == (["CRSPurposeCode12220"], True)


def test_a_truncated_listing_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ontology_document, "TERM_LISTING_CAP", 3)

    with pytest.raises(OntologyListingTruncatedError):
        _nodes(ontology_graph())


def test_a_listing_subject_with_no_provenance_refuses() -> None:
    terms = {**TERMS, "educationalAttainment": ("Property", ())}

    with pytest.raises(OntologyProvenanceUnreadError) as refused:
        _nodes(ontology_graph(terms=terms))

    assert refused.value.dcid == "educationalAttainment"


def test_a_held_node_with_no_type_is_served_with_no_provenance() -> None:
    graph_nodes = {**NODES, "UntypedValue": {"name": ("Untyped value",)}}
    row = index_row("Amount_Untyped", {"beneficiaryGender": "UntypedValue"})

    graph = ontology_graph(nodes=graph_nodes)
    ontology = build_ontology(
        reader=graph,
        code_index=index_of(*INDEX.statvars, row),
        generation="g1",
        deadline=_DEADLINE,
    )

    untyped = {node.dcid: node for node in ontology.document.nodes}["UntypedValue"]
    assert (untyped.types, untyped.provenances, untyped.declared_by) == ([], [], None)
    assert '"declared_by":null' in untyped.model_dump_json()
    # The client confirms the empty answer, so the build reads it as untyped.
    assert graph.confirm_empty == [True]


@pytest.mark.parametrize(
    "female",
    [
        pytest.param({"typeOf": (ArcTarget(value="GenderType"),)}, id="target-unsourced"),
        pytest.param(None, id="targets-unread"),
    ],
)
def test_a_typed_node_with_no_provenance_refuses(female: object) -> None:
    arc_targets = {dcid: targets for dcid, targets in ARC_TARGETS.items() if dcid != "Female"}
    if female is not None:
        arc_targets["Female"] = female  # type: ignore[assignment]

    with pytest.raises(OntologyProvenanceUnreadError) as refused:
        _nodes(ontology_graph(arc_targets=arc_targets))

    assert refused.value.dcid == "Female"


def test_a_term_that_is_also_a_member_keeps_every_provenance() -> None:
    # amount is a base Property and an instance member of CRSPurposeCodeEnum.
    graph_nodes = {
        **NODES,
        "amount": {**NODES["amount"], "typeOf": ("Property", "CRSPurposeCodeEnum")},
    }
    members = (
        *CRS_MEMBERS,
        ArcSubject(dcid="amount", names=(), types=("CRSPurposeCodeEnum",), provenances=(DFO,)),
    )

    ontology = build_ontology(
        reader=ontology_graph(nodes=graph_nodes, members=members),
        code_index=INDEX,
        generation="g1",
        deadline=_DEADLINE,
    )

    node = {node.dcid: node for node in ontology.document.nodes}["amount"]
    term = {term.dcid: term for term in ontology.terms.terms}["amount"]
    assert (node.provenances, node.declared_by) == ([SCHEMA, DFO], "instance")
    assert (term.provenances, term.declared_by) == (node.provenances, node.declared_by)


def test_terms_list_every_class_and_property() -> None:
    graph = FakeGraphNodeReader(
        nodes=NODES,
        arc_targets=ARC_TARGETS,
        arc_subjects={
            **arc_subjects(),
            ("Property", "typeOf"): [
                *listing("Property", TERMS),
                ArcSubject(
                    dcid="FinancialFlow", names=("Financial flow",), types=(), provenances=(SCHEMA,)
                ),
            ],
        },
    )

    terms = build_ontology(
        reader=graph, code_index=INDEX, generation="g1", deadline=_DEADLINE
    ).terms

    assert terms.generation == "g1"
    by_dcid = {term.dcid: term for term in terms.terms}
    assert sorted(by_dcid) == sorted(TERMS)
    attainment = by_dcid["educationalAttainment"]
    assert attainment.declared_by == "base"
    assert attainment.provenances == sorted(TERMS["educationalAttainment"][1])
    crs = by_dcid["crsPurpose"]
    assert (crs.name, crs.types, crs.declared_by) == ("CRS purpose", ["Property"], "instance")
    # A listing name equal to the dcid is no name.
    assert by_dcid["amount"].name is None
    # A dcid in both listings is one term with both kinds and every provenance.
    flow = by_dcid["FinancialFlow"]
    assert (flow.name, flow.types, flow.provenances, flow.declared_by) == (
        "Financial flow",
        ["Class", "Property"],
        [SCHEMA, DFO],
        "instance",
    )


def test_two_builds_from_reordered_graphs_serialise_to_the_same_bytes() -> None:
    reordered_terms = dict(reversed(TERMS.items()))
    reordered_nodes = {
        dcid: {prop: tuple(reversed(values)) for prop, values in reversed(arcs.items())}
        for dcid, arcs in reversed(NODES.items())
    }
    builds = [
        build_ontology(reader=graph, code_index=INDEX, generation="g1", deadline=_DEADLINE)
        for graph in (
            ontology_graph(),
            ontology_graph(terms=reordered_terms, nodes=reordered_nodes, members=CRS_MEMBERS[::-1]),
        )
    ]

    documents = {build.document.model_dump_json() for build in builds}
    terms = {build.terms.model_dump_json() for build in builds}
    assert len(documents) == len(terms) == 1
    dcids = [node.dcid for node in builds[0].document.nodes]
    assert dcids == sorted(dcids)


def _read_counts(graph: FakeGraphNodeReader) -> dict[str, int]:
    return {method: count for method, count in graph.calls.items() if count}


def test_reads_are_bounded_by_hierarchy_depth_not_node_count() -> None:
    small = ontology_graph()
    _nodes(small)

    # Fifty more instance properties, each used by an index row with its own value.
    extra = {f"extraProperty{i}": ("Property", (DFO,)) for i in range(50)}
    extra_nodes = {
        **NODES,
        **{dcid: {"typeOf": ("Property",), "domainIncludes": ("FinancialFlow",)} for dcid in extra},
        **{f"extraValue{i}": {"typeOf": ("StatType",)} for i in range(50)},
    }
    extra_targets = {
        **ARC_TARGETS,
        **{
            f"extraValue{i}": {"typeOf": (ArcTarget(value="StatType", provenances=(DFO,)),)}
            for i in range(50)
        },
    }
    large = ontology_graph(terms={**TERMS, **extra}, nodes=extra_nodes, arc_targets=extra_targets)
    rows = [index_row(f"Extra{i}", {f"extraProperty{i}": f"extraValue{i}"}) for i in range(50)]
    large_nodes = _nodes(large, index_of(*INDEX.statvars, *rows))

    assert "extraValue49" in large_nodes
    expected = {
        "node_arc_subjects": 2,  # Class and Property listings
        "node_arc_subjects_many": 1,  # members of every instance enumeration
        # seed, three ancestor levels (EconomicActivity, Intangible, Thing), unread members
        "node_arcs_many": 5,
        "node_arc_targets_many": 1,  # provenances of nodes outside the listings
    }
    assert _read_counts(small) == expected
    assert _read_counts(large) == expected


class _DeadlineRecorder:
    """Pass every read to `reader`, recording the `deadline` each one carries."""

    def __init__(self, reader: FakeGraphNodeReader) -> None:
        self._reader = reader
        self.deadlines: list[float | None] = []

    def __getattr__(self, name: str) -> Callable[..., object]:
        method = getattr(self._reader, name)

        def record(*args: object, **kwargs: object) -> object:
            self.deadlines.append(kwargs.get("deadline"))  # type: ignore[arg-type]
            return method(*args, **kwargs)

        return record


def test_every_read_carries_the_deadline() -> None:
    fake = ontology_graph()
    graph = _DeadlineRecorder(fake)

    build_ontology(reader=graph, code_index=INDEX, generation="g1", deadline=_DEADLINE)  # type: ignore[arg-type]

    assert len(graph.deadlines) == sum(fake.calls.values()) == 9
    assert set(graph.deadlines) == {_DEADLINE}
