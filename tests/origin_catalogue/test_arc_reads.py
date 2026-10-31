"""Tests the reads the arc tables are built from: Spanner through BigQuery, and our mixer."""

from __future__ import annotations

import gc
import random
import re
import weakref
from collections.abc import Sequence
from typing import Any

import pytest

from dc_graph import GraphReadError
from origin_catalogue.arc_reads import (
    EMPTY_RESOLVER_INDEX,
    SUBJECT_ROLE_PROPERTY,
    IndexSubjects,
    ResolverIndex,
    held_by_mixer,
    iter_base_index,
    read_catalogue_arcs,
    read_dfo_arcs,
    read_index_subjects,
    resolver_base_variables,
    schema_closure,
)
from origin_catalogue.arc_tables import ArcTablesBuilder
from origin_catalogue.artifact import _encode_arc_tables
from origin_catalogue.families import family_tables
from origin_catalogue.graph import _table_arcs, arcs_mirrored
from origin_catalogue.node_reads import iter_node_arcs
from origin_catalogue.un_mirror import EMPTY_UN_MIRROR, UnArc, wire_arcs
from tests.artifact_publication.support import FakeBigQuery
from tests.origin_catalogue.fake_catalogue import restrictive_catalogue
from tests.origin_catalogue.support import build_labelled
from tests.origin_catalogue.test_job import SPANNER_EDGES, FakeGraph, spanner_rows
from tests.origin_catalogue.un_fake import FakeUn, target

_CONNECTION = "proj.region.conn"

ODA = "Amount_ODA"
ORPHAN = "Amount_NoObservations"
POPULATION = "Count_Person"
LITERAL = "Official development assistance:9yQ1"


def _edge(subject: str, predicate: str, object_id: str) -> dict[str, str]:
    return {"subject_id": subject, "predicate": predicate, "object_id": object_id}


def _spanner(edges: list[dict[str, str]], nodes: list[dict[str, Any]] = ()) -> FakeBigQuery:
    return FakeBigQuery({SPANNER_EDGES: spanner_rows(edges, list(nodes))})


def _no_dfo() -> FakeBigQuery:
    return _spanner([])


def test_dfo_arcs_tell_entities_literals_and_base_graph_targets_apart() -> None:
    bq = _spanner(
        [
            _edge(ODA, "typeOf", "StatisticalVariable"),
            _edge(ODA, "name", LITERAL),
            # Another provenance's copy of an edge collapses.
            _edge(ODA, "typeOf", "StatisticalVariable"),
            _edge(ODA, "statType", "measuredValue"),
            _edge(ODA, "measurementDenominator", POPULATION),
            _edge(ORPHAN, "typeOf", "StatisticalVariable"),
        ],
        [
            {
                "subject_id": "StatisticalVariable",
                "value": "StatisticalVariable",
                "name": "Statistical Variable",
                "types": ["Class"],
            },
            {
                "subject_id": LITERAL,
                "value": "Official development assistance",
                "name": "",
                "types": [],
            },
        ],
    )
    graph = FakeGraph(
        arcs={
            "measuredValue": {"typeOf": ("Property",), "name": ("measuredValue",)},
            POPULATION: {"typeOf": ("StatisticalVariable",), "name": ("Population",)},
        }
    )

    builder = ArcTablesBuilder()
    read = read_dfo_arcs(
        bq, connection=_CONNECTION, observed=[ODA, POPULATION], graph=graph, into=builder
    )

    assert {variable: wire_arcs(rows) for variable, rows in builder.table("dfo").items()} == {
        ODA: {
            "typeOf": {
                "nodes": [
                    {
                        "dcid": "StatisticalVariable",
                        "name": "Statistical Variable",
                        "types": ["Class"],
                    }
                ]
            },
            "name": {"nodes": [{"value": "Official development assistance"}]},
            "statType": {
                "nodes": [{"dcid": "measuredValue", "name": "measuredValue", "types": ["Property"]}]
            },
            "measurementDenominator": {
                "nodes": [
                    {"dcid": POPULATION, "name": "Population", "types": ["StatisticalVariable"]}
                ]
            },
        }
    }
    # Observed with no Spanner node: its node, and so its arcs, are base Data Commons'.
    assert read.without_node == {POPULATION}
    assert read.variables == {ODA}
    assert read.base_targets == {"measuredValue", POPULATION}
    assert graph.calls == [("node_arcs_many", (POPULATION, "measuredValue"))]


def test_dfo_targets_take_the_mixer_s_order_whatever_order_bigquery_streams() -> None:
    edges = [
        _edge(variable, predicate, target)
        for variable in (ODA, "Amount_OOF")
        for predicate, target in (
            ("typeOf", "StatisticalVariable"),
            ("observationProperties", "financingRecipient"),
            ("observationProperties", "financingProvider"),
            ("memberOf", "dc/g/Oda"),
            ("memberOf", "dc/g/Aid"),
            ("memberOf", "dc/g/Flows"),
        )
    ]
    encoded = []
    for rows in (edges, edges[::-1], random.Random(7).sample(edges, len(edges))):
        builder = ArcTablesBuilder()
        read_dfo_arcs(
            _spanner(rows),
            connection=_CONNECTION,
            observed=[ODA, "Amount_OOF"],
            graph=FakeGraph(),
            into=builder,
        )
        tables = build_labelled(builder)
        roles = [arc.dcid for arc in tables.dfo[ODA] if arc.property == "observationProperties"]
        assert roles == ["financingProvider", "financingRecipient"]
        encoded.append(_encode_arc_tables(tables))

    assert encoded[0] == encoded[1] == encoded[2]


def test_a_base_graph_target_the_mixer_lacks_carries_its_dcid_alone() -> None:
    bq = _spanner([_edge(ODA, "typeOf", "StatisticalVariable"), _edge(ODA, "unit", "maturity")])

    builder = ArcTablesBuilder()
    read_dfo_arcs(bq, connection=_CONNECTION, observed=[ODA], graph=FakeGraph(), into=builder)

    assert wire_arcs(builder.table("dfo")[ODA])["unit"] == {"nodes": [{"dcid": "maturity"}]}


def test_dfo_arcs_come_from_one_query_reading_edge_once_and_root_partitionable() -> None:
    bq = _no_dfo()

    read_dfo_arcs(
        bq, connection=_CONNECTION, observed=[], graph=FakeGraph(), into=ArcTablesBuilder()
    )

    (sql,) = bq.queries
    inner = re.findall(rf'EXTERNAL_QUERY\("{_CONNECTION}", "([^"]*)"\)', sql)
    # One external query per table: `Edge` is exported once, not once per read.
    assert sorted(re.search(r"\bFROM (\w+)", text).group(1) for text in inner) == ["Edge", "Node"]
    for text in inner:
        assert not re.search(r"\b(DISTINCT|LIMIT|ORDER BY|JOIN|GROUP BY)\b", text), text


def _variable(index: int, arcs: int = 3) -> dict[str, list[dict[str, Any]]]:
    return {
        "name": [{"value": f"V{index}"}],
        "memberOf": [target(f"g/{i}") for i in range(arcs - 1)],
    }


def test_base_reads_name_25_variables_halve_a_paged_read_and_follow_one_variable_s_pages() -> None:
    nodes = {f"dc/base/V{i:02d}": _variable(i) for i in range(60)}
    nodes["dc/base/Wide"] = _variable(99, arcs=50)
    un = FakeUn(nodes)
    un.arc_targets_per_page = 40
    un.delay = 0.005
    listed = [*nodes, "dc/base/Gone"]

    arcs = read_catalogue_arcs(
        _no_dfo(),
        connection=_CONNECTION,
        observed=[],
        base_variables=listed,
        resolver_index=EMPTY_RESOLVER_INDEX,
        carried_base={},
        graph=FakeGraph(un=un),
    ).tables

    assert set(arcs.base) == set(listed)
    assert arcs.base["dc/base/Gone"] == ()
    assert len(arcs.base["dc/base/Wide"]) == 50
    assert all(len(arcs.base[f"dc/base/V{i:02d}"]) == 3 for i in range(60))
    reads = [request for request in un.requests if request["property"] == "->*"]
    assert max(len(read["nodes"]) for read in reads) == 25
    # A 25-variable read pages at 40 targets, so it is halved; the wide variable pages alone.
    assert any(len(read["nodes"]) == 12 for read in reads)
    assert any(read.get("nextToken") and read["nodes"] == ["dc/base/Wide"] for read in reads)
    assert un.max_in_flight <= 4


def _base_and_schema_reads(un: FakeUn) -> Any:
    return read_catalogue_arcs(
        _no_dfo(),
        connection=_CONNECTION,
        observed=[],
        base_variables=["dc/base/A", "dc/base/B", "dc/base/Gone"],
        resolver_index=EMPTY_RESOLVER_INDEX,
        carried_base={},
        graph=FakeGraph(un=un),
    ).tables


def test_a_node_a_read_naming_several_leaves_out_is_read_again_alone() -> None:
    nodes: dict[str, Any] = {
        "dc/base/A": _variable(1),
        "dc/base/B": {"observationProperties": [target("financingProvider")]},
        "financingProvider": {"rangeIncludes": [target("Organization")]},
        "Organization": {"typeOf": [target("Class")]},
    }
    un = FakeUn(nodes)
    # Each is left out of the read naming it beside other nodes, once.
    un.omitting = {"dc/base/A": 1, "financingProvider": 1}

    arcs = _base_and_schema_reads(un)

    assert len(arcs.base["dc/base/A"]) == 3
    assert arcs.base["dc/base/Gone"] == ()
    assert [row.dcid for row in arcs.schema["financingProvider"]] == ["Organization"]
    reads = [request["nodes"] for request in un.requests]
    assert ["dc/base/A"] in reads
    assert ["financingProvider"] in reads
    # An empty answer in a read naming several is read alone before it is held empty.
    assert ["dc/base/Gone"] in reads


def test_a_node_a_read_naming_it_alone_leaves_out_fails_the_read() -> None:
    un = FakeUn({"dc/base/A": _variable(1), "dc/base/B": _variable(2)})
    un.omitting = {"dc/base/A": 2}

    with pytest.raises(GraphReadError, match="dc/base/A"):
        _base_and_schema_reads(un)


def _labelled_reads(un: FakeUn) -> Any:
    return read_catalogue_arcs(
        _no_dfo(),
        connection=_CONNECTION,
        observed=[],
        base_variables=["dc/base/A"],
        resolver_index=EMPTY_RESOLVER_INDEX,
        carried_base={},
        graph=FakeGraph(un=un),
    )


def _labelled_nodes() -> dict[str, Any]:
    """A variable whose arcs name a node of two names by its first and a nameless node by
    its dcid, as our mixer's `->*` read names them."""
    return {
        "dc/base/A": {
            "name": [{"value": "A"}],
            "populationType": [target("Alcohol", "Alcohol")],
            "age": [target("Years0", "Years0")],
        },
        "Alcohol": {"name": [{"value": "Alcohol"}, {"value": "Alcoholic beverages"}]},
        "Years0": {"typeOf": [target("Quantity")]},
    }


@pytest.mark.parametrize("hazard", ["omitting", "emptying"])
def test_each_node_an_arc_names_takes_the_label_node_label_gives_it(hazard: str) -> None:
    """The label is the `label_or_self_name` of the `->name` values, or none, and the arc
    keeps the name `->*` gives. A node a `->name` read naming several leaves out or answers
    with no name is read again alone, as the `->*` and `->typeOf` reads read it, so
    `Years0`, which has no name, is read alone before it is held nameless."""
    un = FakeUn(_labelled_nodes())
    setattr(un, hazard, {"Alcohol": 1})

    built = _labelled_reads(un)

    assert built.tables.labels == {"Alcohol": "Alcoholic beverages", "Years0": None}
    assert built.tables.targets["Alcohol"].name == "Alcohol"
    assert built.tables.targets["Years0"].name == "Years0"
    assert built.labels_read == 2
    reads = [request["nodes"] for request in un.requests if request["property"] == "->name"]
    assert reads == [["Alcohol", "Years0"], ["Alcohol"], ["Years0"]]


def test_a_node_a_name_read_naming_it_alone_leaves_out_fails_the_read() -> None:
    un = FakeUn(_labelled_nodes())
    un.omitting = {"Alcohol": 2}

    with pytest.raises(GraphReadError, match="Alcohol"):
        _labelled_reads(un)


def test_labels_are_read_500_nodes_a_request() -> None:
    nodes: dict[str, Any] = {"dc/base/A": _variable(1, arcs=502)}
    nodes.update({f"g/{i}": {"name": [{"value": f"G{i}"}]} for i in range(501)})
    un = FakeUn(nodes)

    built = _labelled_reads(un)

    reads = [request["nodes"] for request in un.requests if request["property"] == "->name"]
    assert [len(nodes) for nodes in reads] == [500, 1]
    assert built.tables.labels == {f"g/{i}": f"G{i}" for i in range(501)}


def test_carried_base_variables_are_not_read_and_keep_their_rows() -> None:
    un = FakeUn({"dc/base/New": _variable(1)})
    kept = (UnArc(property="name", value="Old"),)

    built = read_catalogue_arcs(
        _no_dfo(),
        connection=_CONNECTION,
        observed=[],
        base_variables=["dc/base/New", "dc/base/Old"],
        resolver_index=EMPTY_RESOLVER_INDEX,
        carried_base={"dc/base/Old": kept, "dc/base/Dropped": kept},
        graph=FakeGraph(un=un),
    )

    assert built.tables.base["dc/base/Old"] == kept
    assert "dc/base/Dropped" not in built.tables.base
    assert (built.base_read, built.base_carried) == (1, 1)
    assert all("dc/base/Old" not in request["nodes"] for request in un.requests)


def test_an_observed_variable_spanner_declares_no_node_for_reads_as_a_base_node() -> None:
    """`Count_Person`'s observations are DFO's and its node is base Data Commons', so its
    arcs come from the mixer into the `base` table though no base list names it."""
    bq = _spanner([_edge(ODA, "typeOf", "StatisticalVariable")])
    un = FakeUn({POPULATION: _variable(1)})

    built = read_catalogue_arcs(
        bq,
        connection=_CONNECTION,
        observed=[ODA, POPULATION],
        base_variables=[],
        resolver_index=EMPTY_RESOLVER_INDEX,
        carried_base={},
        graph=FakeGraph(un=un),
    )

    assert built.dfo.without_node == {POPULATION}
    assert set(built.tables.dfo) == {ODA}
    assert [arc.value for arc in built.tables.base[POPULATION] if arc.property == "name"] == ["V1"]
    assert built.base_read == 1


def test_the_schema_closes_over_denominators_role_ranges_and_superclasses() -> None:
    answers: dict[str, dict[str, Any]] = {
        "Count_Household": {"measurementDenominator": {"nodes": [{"dcid": "Count_Dwelling"}]}},
        "Count_Dwelling": {},
        "financingProvider": {"rangeIncludes": {"nodes": [{"dcid": "Organization"}]}},
        "Organization": {"subClassOf": {"nodes": [{"dcid": "Thing"}]}},
        # A cycle in the class graph ends the walk.
        "Thing": {"subClassOf": {"nodes": [{"dcid": "Organization"}]}},
    }
    batches: list[Sequence[str]] = []

    def read(nodes: Sequence[str]) -> dict[str, Any]:
        batches.append(list(nodes))
        return {node: answers[node] for node in nodes if node in answers}

    schema = schema_closure(
        [
            (
                UnArc(property="measurementDenominator", dcid="Count_Household"),
                UnArc(property="measurementDenominator", dcid=POPULATION),
                UnArc(property="observationProperties", dcid="financingProvider"),
                # A variable's own range is not a schema dependency.
                UnArc(property="rangeIncludes", dcid="Number"),
            ),
            (),
        ],
        held=[POPULATION],
        read=read,
    )

    assert schema == {
        SUBJECT_ROLE_PROPERTY: {},
        **answers,
    }
    assert batches == [
        ["Count_Household", "financingProvider", SUBJECT_ROLE_PROPERTY],
        ["Count_Dwelling", "Organization"],
        ["Thing"],
    ]


class _Answers(list):
    """A read's targets for one property, which a test can tell is let go."""


def test_the_stage_lets_each_base_read_go_once_its_rows_are_held() -> None:
    answered: list[list[weakref.ref[_Answers]]] = []

    class Graph:
        def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict:
            targets = {node: _Answers([{"value": node}]) for node in payload["nodes"]}
            answered.append([weakref.ref(nodes) for nodes in targets.values()])
            return {
                "data": {node: {"arcs": {"name": {"nodes": targets[node]}}} for node in targets}
            }

    variables = [f"dc/base/V{i:03d}" for i in range(100)]
    builder = ArcTablesBuilder()
    for index, (variable, arcs) in enumerate(iter_node_arcs(Graph(), variables)):
        builder.add("base", variable, arcs)
        del arcs
        if index == 50:
            # The first read is let go while the third is being held.
            gc.collect()
            assert all(ref() is None for ref in answered[0])

    assert len(builder.table("base")) == 100


# --- Resolver indexes ---------------------------------------------------------

DROPPED = "Amount_FinancialFlow_Gross_Disbursement_CRSPurposeCode43010"
"""A subject of our index whose `Edge` rows an import dropped, as dev's CRS re-import did."""
BASE_HELD = "Count_Person_Urban"
"""A subject of our index Spanner declares no variable for and base Data Commons holds."""
INDEXED = "Count_MortalityEvent_Malaria"
"""A base variable the base index names that the build does not list."""


@pytest.mark.parametrize(
    ("unedged", "lone_reads"),
    [
        pytest.param(frozenset(), [[DROPPED]], id="edged"),
        pytest.param(frozenset({DROPPED, BASE_HELD}), [], id="unedged"),
    ],
)
def test_every_resolver_index_node_is_held_or_held_absent_and_routes_with_no_read(
    unedged: frozenset[str], lone_reads: list[list[str]]
) -> None:
    """Our index's subjects and the base index's nodes all read from the tables: a subject
    the graph dropped is held empty in the `dfo` table and reads absent, and one base Data
    Commons holds is read as a base node, as is each base index node the build does not
    list. Neither routes to a live read. A subject the batch answers untyped is read again
    alone, unless Spanner holds no `Edge` row of it, which agrees with the batch."""
    bq = _spanner([_edge(ODA, "typeOf", "StatisticalVariable")])
    un = FakeUn(
        {
            BASE_HELD: {"typeOf": [target("StatisticalVariable")], **_variable(1)},
            INDEXED: _variable(2),
        }
    )

    built = read_catalogue_arcs(
        bq,
        connection=_CONNECTION,
        observed=[ODA],
        base_variables=[],
        resolver_index=ResolverIndex(
            ours=frozenset({ODA, DROPPED, BASE_HELD}), base=frozenset({INDEXED}), unedged=unedged
        ),
        carried_base={},
        graph=FakeGraph(un=un),
    )

    assert built.tables.dfo[DROPPED] == ()
    assert set(built.tables.base) == {BASE_HELD, INDEXED}
    assert (built.index_absent, built.index_held_by_mixer, built.index_base) == (1, 1, 2)
    type_reads = [request["nodes"] for request in un.requests if request["property"] == "->typeOf"]
    assert type_reads == [sorted([BASE_HELD, DROPPED]), *lone_reads]
    catalogue = restrictive_catalogue(dfo_variables=[ODA], arcs=built.tables)
    for dcid in (DROPPED, BASE_HELD, INDEXED):
        assert arcs_mirrored(dcid, catalogue=catalogue)
    assert _table_arcs(catalogue, DROPPED) is None
    assert _table_arcs(catalogue, INDEXED) is not None


def test_a_mixer_presence_read_names_500_nodes_and_reads_a_left_out_node_alone() -> None:
    nodes = [f"dc/ours/V{i:03d}" for i in range(501)]
    un = FakeUn({node: {"typeOf": [target("StatisticalVariable")]} for node in nodes})
    un.omitting = {nodes[3]: 1}

    held = held_by_mixer(FakeGraph(un=un), nodes)

    assert held == set(nodes)
    assert sorted(len(request["nodes"]) for request in un.requests) == [1, 1, 500]


def test_a_node_a_presence_read_naming_several_answers_untyped_is_held_only_once_read_alone() -> (
    None
):
    """A batch can empty a node depending on which others share it. The node it answers
    with no type is read again alone, and only that read's empty answer counts as absent."""
    un = FakeUn({BASE_HELD: {"typeOf": [target("StatisticalVariable")]}})
    un.emptying = {BASE_HELD: 1}

    held = held_by_mixer(FakeGraph(un=un), [BASE_HELD, DROPPED])

    assert held == {BASE_HELD}
    assert sorted(request["nodes"] for request in un.requests) == sorted(
        [[BASE_HELD, DROPPED], [BASE_HELD], [DROPPED]]
    )


def test_a_node_a_presence_read_naming_it_alone_leaves_out_fails_the_read() -> None:
    un = FakeUn({})
    un.omitting = {DROPPED: 1}

    with pytest.raises(GraphReadError, match=DROPPED):
        held_by_mixer(FakeGraph(un=un), [DROPPED])


def test_our_index_subjects_come_from_one_root_partitionable_query() -> None:
    """Spanner answers whether it holds an `Edge` row of each subject by key, so the read
    exports no `Edge` row."""
    bq = FakeBigQuery(
        {
            "NodeEmbedding": [
                {"subject_id": ODA, "has_edges": True},
                {"subject_id": DROPPED, "has_edges": False},
            ]
        }
    )

    assert read_index_subjects(bq, connection=_CONNECTION) == IndexSubjects(
        subjects=frozenset({ODA, DROPPED}), unedged=frozenset({DROPPED})
    )
    (inner,) = re.findall(rf'EXTERNAL_QUERY\("{_CONNECTION}", "([^"]*)"\)', bq.queries[0])
    assert inner.endswith("FROM NodeEmbedding AS n")
    assert not re.search(r"\b(DISTINCT|LIMIT|ORDER BY|JOIN|GROUP BY)\b", inner)


def test_the_base_index_stands_a_topic_for_its_leaves_and_drops_un_dcids() -> None:
    topic, un_leaf = "dc/topic/Health", "undata/who/X"

    found = resolver_base_variables([INDEXED, topic], topics={topic: (BASE_HELD, un_leaf)})

    assert found == {INDEXED, BASE_HELD}


_ADMITTED_DEBT = "DebtOwed_WorldBank"
_WITHHELD_DEBT = "DebtOwed_ExcludedBank"


def _debt(creditor: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "typeOf": [target("StatisticalVariable")],
        "populationType": [target("Debt")],
        "measuredProperty": [target("amount")],
        "statType": [target("measuredValue")],
        "constraintProperties": [target("creditor")],
        "creditor": [target(creditor)],
        "name": [{"value": f"Debt to {creditor}"}],
    }


@pytest.mark.parametrize("side", ["ours", "base"])
def test_a_withheld_resolver_subject_is_in_no_table_and_no_family(side: str) -> None:
    """Admission refuses a withheld variable DFO holds no observations of, from our index
    or the base one alike, so no table holds it and its creditor joins no family."""
    un = FakeUn({_ADMITTED_DEBT: _debt("WorldBank"), _WITHHELD_DEBT: _debt("ExcludedBank")})
    subjects = frozenset({_WITHHELD_DEBT})

    built = read_catalogue_arcs(
        _no_dfo(),
        connection=_CONNECTION,
        observed=[],
        base_variables=[_ADMITTED_DEBT],
        resolver_index=ResolverIndex(
            ours=subjects if side == "ours" else frozenset(),
            base=subjects if side == "base" else frozenset(),
        ),
        carried_base={},
        withheld=subjects,
        graph=FakeGraph(un=un),
    )

    assert built.tables.arcs(_WITHHELD_DEBT) is None
    assert all(_WITHHELD_DEBT not in request["nodes"] for request in un.requests)
    [summary] = family_tables(
        dfo=built.tables.dfo, base=built.tables.base, un=EMPTY_UN_MIRROR
    ).arcs.values()
    assert summary.values == {"creditor": ("WorldBank",)}


def test_the_base_index_lists_every_dcid_of_its_dcid_column(tmp_path) -> None:
    index = tmp_path / "embeddings.csv"
    index.write_text(
        "0,1,dcid,sentence\n"
        f"0.1,0.2,{INDEXED},Malaria deaths\n"
        f'0.3,0.4,"{BASE_HELD},{POPULATION}","Population, urban"\n'
    )

    assert list(iter_base_index(index.as_uri())) == [INDEXED, BASE_HELD, POPULATION]
