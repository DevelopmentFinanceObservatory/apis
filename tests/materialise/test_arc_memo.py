"""Tests `materialise.arc_memo.MemoGraphReader` against the recording `FakeGraphReader`."""

from __future__ import annotations

from materialise.arc_memo import MemoGraphReader

from .conftest import FakeGraphReader

_ARCS = {
    "CurrencyEUR": {"typeOf": ("CurrencyUnit",), "nationalCurrencyOf": ("country/FRA",)},
    "provenance/Tables": {"lastDataRefreshDate": ("2026-01-01",)},
}


def test_a_repeat_read_of_a_remembered_property_is_served_from_memory():
    graph = FakeGraphReader(arcs=_ARCS)
    reader = MemoGraphReader(graph, properties=("typeOf",))

    first = reader.node_arcs_many(["CurrencyEUR", "CurrencyXXX"], properties=("typeOf",))
    second = reader.node_arcs_many(["CurrencyXXX", "CurrencyEUR"], properties=("typeOf",))

    assert first == {"CurrencyEUR": {"typeOf": ("CurrencyUnit",)}, "CurrencyXXX": None}
    assert second == {"CurrencyXXX": None, "CurrencyEUR": {"typeOf": ("CurrencyUnit",)}}
    assert graph.node_arcs_many_calls == [(("CurrencyEUR", "CurrencyXXX"), ("typeOf",))]


def test_a_read_fetches_only_the_dcids_the_memo_lacks():
    graph = FakeGraphReader(arcs={**_ARCS, "CurrencyUSD": {"typeOf": ("CurrencyUnit",)}})
    reader = MemoGraphReader(graph, properties=("typeOf",))

    reader.node_arcs_many(["CurrencyEUR"], properties=("typeOf",))
    reader.node_arcs_many(["CurrencyEUR", "CurrencyUSD"], properties=("typeOf",))

    assert graph.node_arcs_many_calls == [
        (("CurrencyEUR",), ("typeOf",)),
        (("CurrencyUSD",), ("typeOf",)),
    ]


def test_a_property_outside_the_memo_reaches_the_graph_every_time():
    graph = FakeGraphReader(arcs=_ARCS)
    reader = MemoGraphReader(graph, properties=("typeOf",))

    for _ in range(2):
        reader.node_arcs_many(["provenance/Tables"], properties=("lastDataRefreshDate",))

    assert graph.calls["node_arcs_many"] == 2


def test_a_repeat_reverse_read_is_served_from_memory():
    graph = FakeGraphReader(arcs=_ARCS)
    reader = MemoGraphReader(graph, properties=("nationalCurrencyOf",))

    pages = [
        reader.node_subjects("country/FRA", property="nationalCurrencyOf", max_results=2)
        for _ in range(3)
    ]

    assert {page.dcids for page in pages} == {("CurrencyEUR",)}
    assert graph.calls["node_subjects"] == 1


def test_a_batch_is_read_once_on_the_first_read_of_one_of_its_nodes():
    graph = FakeGraphReader(arcs={**_ARCS, "CurrencyUSD": {"typeOf": ("CurrencyUnit",)}})
    reader = MemoGraphReader(
        graph,
        properties=("typeOf", "nationalCurrencyOf"),
        batch=["CurrencyEUR", "CurrencyUSD"],
        batch_properties=("typeOf",),
    )

    reader.node_arcs_many(["CurrencyEUR"], properties=("typeOf",))
    reader.node_arcs_many(["CurrencyUSD"], properties=("typeOf",))

    assert graph.node_arcs_many_calls == [(("CurrencyEUR", "CurrencyUSD"), ("typeOf",))]


def test_a_batch_no_read_touches_is_never_fetched():
    graph = FakeGraphReader(arcs=_ARCS)
    reader = MemoGraphReader(
        graph, properties=("typeOf",), batch=["CurrencyEUR"], batch_properties=("typeOf",)
    )

    reader.node_arcs_many(["provenance/Tables"], properties=("typeOf",))

    assert graph.node_arcs_many_calls == [(("provenance/Tables",), ("typeOf",))]


_DECISION_PROPERTIES = ("measurementQualifier", "typeOf")
_STAT_VAR_ARCS = {
    "SV_A": {"measurementQualifier": ("Nominal",), "typeOf": ("StatisticalVariable",)},
    "SV_B": {"measurementQualifier": ("RealValue",), "typeOf": ("StatisticalVariable",)},
}


def test_a_batch_read_folds_in_the_triggering_reads_nodes_and_properties():
    """The read that triggers the batch rides on the batch's one call, with the union of
    the properties, so a later read of another batch node, of the triggering nodes, or of
    another property in the union is served from memory."""
    graph = FakeGraphReader(arcs={**_ARCS, **_STAT_VAR_ARCS})
    reader = MemoGraphReader(
        graph,
        properties=_DECISION_PROPERTIES,
        batch=["SV_A", "SV_B"],
        batch_properties=("measurementQualifier",),
    )

    first = reader.node_arcs_many(["SV_A", "CurrencyEUR"], properties=_DECISION_PROPERTIES)
    reader.node_arcs_many(["SV_B", "CurrencyEUR"], properties=_DECISION_PROPERTIES)
    reader.node_arcs_many(["SV_B"], properties=("typeOf",))

    assert first == {
        "SV_A": _STAT_VAR_ARCS["SV_A"],
        "CurrencyEUR": {"measurementQualifier": (), "typeOf": ("CurrencyUnit",)},
    }
    assert graph.node_arcs_many_calls == [
        (("SV_A", "SV_B", "CurrencyEUR"), ("measurementQualifier", "typeOf"))
    ]


def test_a_batch_read_through_a_nested_memo_fills_both_layers():
    """A batched memo over a request memo, as `conversion_graph_reader` nests over
    `request_conversion_reader`: the batch's one call lands in the inner memo, so a later
    read through the inner memo alone reaches the graph no more."""
    graph = FakeGraphReader(arcs={**_ARCS, **_STAT_VAR_ARCS})
    inner = MemoGraphReader(graph, properties=_DECISION_PROPERTIES)
    outer = MemoGraphReader(
        inner,
        properties=_DECISION_PROPERTIES,
        batch=["SV_A", "SV_B"],
        batch_properties=_DECISION_PROPERTIES,
    )

    outer.node_arcs_many(["SV_A", "CurrencyEUR"], properties=_DECISION_PROPERTIES)
    inner.node_arcs_many(["SV_B", "CurrencyEUR"], properties=_DECISION_PROPERTIES)

    assert len(graph.node_arcs_many_calls) == 1
