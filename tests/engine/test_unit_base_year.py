"""The declared base year: read off the live conversion-table provenances, parsed strictly,
kept once per graph client, and refused as capability_unavailable when the tables disagree."""

from __future__ import annotations

import pytest

from materialise.recipe_bodies import RECIPE_BODIES
from search.engine.errors import GraphInfraError
from search.engine.unit.base_year import (
    BASE_YEAR_PROPERTY,
    TABLE_PROVENANCES,
    TableBaseYearError,
    declared_base_year,
)
from search.models import IdentityCurrencyRole, NoData, UnitAlternative
from tests.fixtures import FakeGraph

_DAC, _IMF = TABLE_PROVENANCES


def _table(*base_years: str, refresh: str | None = "2026-10-01") -> dict:
    arcs: dict = {BASE_YEAR_PROPERTY: {"nodes": [{"value": year} for year in base_years]}}
    if refresh is not None:
        arcs["lastDataRefreshDate"] = {"nodes": [{"value": refresh}]}
    return {"label": "Conversion tables", "arcs": arcs}


class _CountingGraph(FakeGraph):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reads: list[tuple[tuple[str, ...], tuple[str, ...], float | None]] = []

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.reads.append((tuple(dcids), tuple(properties), deadline))
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def _graph(dac: dict | None, imf: dict | None) -> _CountingGraph:
    nodes = {dcid: node for dcid, node in ((_DAC, dac), (_IMF, imf)) if node is not None}
    return _CountingGraph(nodes=nodes)


def test_table_provenances_are_every_recipe_s_table_provenance():
    assert set(TABLE_PROVENANCES) == {r.table_provenance for r in RECIPE_BODIES.values()}
    assert TABLE_PROVENANCES == (
        "provenance/ONEConversionTablesDAC",
        "provenance/ONEConversionTablesIMF",
    )


def test_the_year_the_graph_serves_as_text_parses_to_an_integer():
    graph = _graph(_table("2024"), _table("2024"))
    assert declared_base_year(graph, deadline=12.5) == 2024
    (read,) = graph.reads
    assert read == (TABLE_PROVENANCES, ("lastDataRefreshDate", BASE_YEAR_PROPERTY), 12.5)


def test_the_first_year_is_kept_for_the_life_of_the_graph_client():
    graph = _graph(_table("2024"), _table("2024"))
    assert declared_base_year(graph) == 2024
    assert declared_base_year(graph) == 2024
    assert len(graph.reads) == 1


def test_each_graph_client_reads_its_own_year():
    assert declared_base_year(_graph(_table("2024"), _table("2024"))) == 2024
    assert declared_base_year(_graph(_table("2025"), _table("2025"))) == 2025


def test_a_failed_read_is_not_kept():
    graph = _graph(_table("2024"), _table("2023"))
    with pytest.raises(TableBaseYearError):
        declared_base_year(graph)
    with pytest.raises(TableBaseYearError):
        declared_base_year(graph)
    assert len(graph.reads) == 2


def test_disagreeing_provenances_refuse_capability_unavailable():
    with pytest.raises(TableBaseYearError, match="different deflatorBaseYear") as exc_info:
        declared_base_year(_graph(_table("2024"), _table("2023")))
    assert exc_info.value.code == "capability_unavailable"
    assert exc_info.value.status == 501
    assert _DAC in exc_info.value.detail and _IMF in exc_info.value.detail


@pytest.mark.parametrize(
    ("dac", "imf", "match"),
    [
        pytest.param(None, _table("2024"), "absent from the graph", id="absent-provenance"),
        pytest.param(
            _table("2024", refresh=None), _table("2024"), "lastDataRefreshDate", id="no-refresh"
        ),
        pytest.param(_table(), _table("2024"), "declares 0", id="no-base-year"),
        pytest.param(_table("2024", "2023"), _table("2024"), "declares 2", id="two-base-years"),
    ],
)
def test_a_provenance_that_cannot_establish_a_year_refuses(dac, imf, match):
    with pytest.raises(TableBaseYearError, match=match):
        declared_base_year(_graph(dac, imf))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("2024.0", id="decimal"),
        pytest.param(" 2024", id="leading-space"),
        pytest.param("+2024", id="signed"),
        pytest.param("२०२४", id="non-ascii-digits"),
        pytest.param("1899", id="before-range"),
        pytest.param("2101", id="after-range"),
        pytest.param("24", id="two-digit"),
        pytest.param("2024-01-01", id="date"),
    ],
)
def test_anything_but_a_four_digit_year_in_range_refuses(text):
    with pytest.raises(TableBaseYearError, match="not a year"):
        declared_base_year(_graph(_table(text), _table(text)))


def test_a_transport_failure_raises_the_graph_client_s_own_error():
    with pytest.raises(GraphInfraError):
        declared_base_year(FakeGraph(raise_on_call=True))


class TestNoDataUnitAlternatives:
    def test_alternatives_and_their_truncation_flag_travel_together(self):
        with pytest.raises(ValueError, match="travel together"):
            NoData(reason="unit_not_resolved", unit_alternatives=[])

    def test_alternatives_belong_to_unit_not_resolved_only(self):
        with pytest.raises(ValueError, match="unit_not_resolved"):
            NoData(
                reason="no_observations",
                unit_alternatives=[UnitAlternative(currency="CHF", label="Swiss franc")],
                unit_alternatives_truncated=False,
            )

    def test_a_role_shape_serialises_as_the_object_a_caller_resubmits(self):
        no_data = NoData(
            reason="unit_not_resolved",
            unit_alternatives=[
                UnitAlternative(currency=IdentityCurrencyRole(role="origin"), label="provider")
            ],
            unit_alternatives_truncated=False,
        )
        dumped = no_data.model_dump(mode="json", exclude_none=True)
        assert dumped["unit_alternatives"] == [
            {"currency": {"role": "origin"}, "label": "provider"}
        ]
