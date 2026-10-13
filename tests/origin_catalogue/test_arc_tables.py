"""Tests the arc tables built from `->*` answers and from Spanner rows."""

from __future__ import annotations

import pytest

from origin_catalogue.arc_tables import (
    ArcTables,
    ArcTablesBuilder,
    ArcTablesError,
    SpannerNode,
    spanner_wire_target,
)
from origin_catalogue.artifact import _encode_arc_tables
from origin_catalogue.un_mirror import UnArc
from tests.origin_catalogue.support import (
    CRS_VARIABLE,
    GDP,
    HOUSEHOLDS,
    POPULATION,
    RECIPIENT,
    build_arc_tables,
    build_labelled,
    held_wire_arcs,
    sample_arc_tables,
)


def test_a_variable_reads_back_its_arcs_once_per_target_without_import_added_constraints() -> None:
    tables = sample_arc_tables()

    assert held_wire_arcs(tables, CRS_VARIABLE) == {
        "constraintProperties": {
            "nodes": [{"dcid": RECIPIENT, "name": RECIPIENT, "types": ["Property"]}]
        },
        "measurementDenominator": {
            "nodes": [{"dcid": HOUSEHOLDS, "types": ["StatisticalVariable"]}]
        },
        "name": {"nodes": [{"value": "Gross ODA disbursements"}]},
        "typeOf": {
            "nodes": [
                {"dcid": "StatisticalVariable", "name": "StatisticalVariable", "types": ["Class"]}
            ]
        },
    }


def test_a_listed_node_without_arcs_is_held_empty_and_an_unlisted_one_is_not_held() -> None:
    tables = sample_arc_tables()

    assert tables.arcs(POPULATION) == ()
    assert held_wire_arcs(tables, POPULATION) == {}
    assert held_wire_arcs(tables, "Place") == {}
    assert tables.arcs("Count_Person_Female") is None
    assert held_wire_arcs(tables, "Count_Person_Female") is None


def test_nodes_share_one_row_per_distinct_target() -> None:
    tables = sample_arc_tables()
    crs_type = tables.dfo[CRS_VARIABLE][-1]
    gdp_type = tables.base[GDP][-1]

    assert crs_type.property == gdp_type.property == "typeOf"
    assert crs_type is gdp_type


@pytest.mark.parametrize(
    ("dfo", "base", "schema", "subject"),
    [
        pytest.param({"a": None}, {"a": None}, {}, "a", id="node-in-two-tables"),
        pytest.param(
            {"a": {"measurementDenominator": {"nodes": [{"dcid": "d"}]}}},
            {},
            {},
            "a",
            id="variable-denominator-not-held",
        ),
        pytest.param(
            {},
            {},
            {"p": {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}}},
            "p",
            id="schema-range-not-held",
        ),
        pytest.param(
            {},
            {},
            {"Country": {"subClassOf": {"nodes": [{"dcid": "Place"}]}}},
            "Country",
            id="schema-superclass-not-held",
        ),
    ],
)
def test_tables_refuse_a_node_held_twice_or_a_dependency_they_do_not_hold(
    dfo: dict, base: dict, schema: dict, subject: str
) -> None:
    with pytest.raises(ArcTablesError) as raised:
        build_arc_tables(dfo=dfo, base=base, schema=schema)
    assert raised.value.subject == subject


@pytest.mark.parametrize(
    ("labels", "subject"),
    [
        pytest.param({}, "Years0", id="target-without-label"),
        pytest.param({"Years0": None, "Alcohol": "Alcohol"}, "Alcohol", id="label-without-target"),
    ],
)
def test_tables_refuse_labels_other_than_one_per_node_an_arc_names(
    labels: dict[str, str | None], subject: str
) -> None:
    rows = {"a": (UnArc(property="age", dcid="Years0", name="Years0"),)}

    with pytest.raises(ArcTablesError) as raised:
        ArcTables(dfo=rows, base={}, schema={}, labels=labels)
    assert raised.value.subject == subject


def test_a_variable_s_range_is_not_a_schema_dependency() -> None:
    # 470 WHO base variables carry `rangeIncludes Number`.
    tables = build_arc_tables(
        dfo={}, base={"who/AIR_11": {"rangeIncludes": {"nodes": [{"dcid": "Number"}]}}}, schema={}
    )
    assert tables.arcs("Number") is None


@pytest.mark.parametrize(
    ("object_id", "node", "target"),
    [
        pytest.param(
            "finalExpenditureType",
            SpannerNode(
                value="finalExpenditureType", name="final expenditure type", types=("Property",)
            ),
            {
                "dcid": "finalExpenditureType",
                "name": "final expenditure type",
                "types": ["Property"],
            },
            id="entity",
        ),
        pytest.param(
            "Gross capital fo:S2jZk872G5L9TLf1Z0ZK08fbS0EOppqOPWcivKmr8Js",
            SpannerNode(value="Gross capital formation, Percent of GDP", name=None, types=()),
            {"value": "Gross capital formation, Percent of GDP"},
            id="literal",
        ),
        pytest.param("RealValue", None, {"dcid": "RealValue"}, id="base-graph-node"),
    ],
)
def test_a_spanner_target_is_an_entity_a_literal_or_a_base_graph_node(
    object_id: str, node: SpannerNode | None, target: dict
) -> None:
    assert spanner_wire_target(object_id, node) == target


@pytest.mark.parametrize(
    "node",
    [
        pytest.param(SpannerNode(value="other", name=None, types=("Class",)), id="typed-literal"),
        pytest.param(SpannerNode(value="x", name=None, types=()), id="untyped-entity"),
        pytest.param(SpannerNode(value=None, name=None, types=()), id="no-value"),
    ],
)
def test_a_spanner_row_that_is_neither_entity_nor_literal_is_refused(node: SpannerNode) -> None:
    with pytest.raises(ArcTablesError):
        spanner_wire_target("x", node)


def test_tables_built_a_node_at_a_time_are_the_same_in_any_order() -> None:
    answers = {
        "b": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}, {"dcid": "StatisticalVariable"}]}
        },
        "a": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable", "name": "Statistical Variable"}]},
            "name": {"nodes": [{"value": "A"}]},
        },
        "c": None,
    }
    forward, backward = ArcTablesBuilder(), ArcTablesBuilder()
    for node in answers:
        forward.add("base", node, answers[node])
    for node in reversed(answers):
        backward.add("base", node, answers[node])

    tables = build_labelled(forward)

    assert tables == build_labelled(backward)
    assert _encode_arc_tables(tables) == _encode_arc_tables(build_labelled(backward))
    assert list(tables.base) == ["a", "b", "c"]
    # A target another provenance repeats is held once.
    assert held_wire_arcs(tables, "b") == {"typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]}}
    assert tables.targets["StatisticalVariable"].name == "Statistical Variable"
