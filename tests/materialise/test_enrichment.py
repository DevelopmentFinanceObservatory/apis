"""Enrichment fields, LRU eviction, probe budgets, and concurrent cache reads."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import httpx

import materialise.enrichment as enrichment_module
from code_index.types import PartyRow
from dc_graph import GraphClient
from graph_release.context import bound_release, current_release
from materialise.enrichment import EnrichmentLookup, read_party_kinds
from tests.graph_release.support import release_at_revision

from .conftest import FakeGraphReader

_DEBT = "Amount_FinancialFlow_Debt_Nominal"


def _graph() -> FakeGraphReader:
    return FakeGraphReader(
        arcs={
            "country/KEN": {
                "countryAlpha3Code": ("KEN",),
                "countryAlpha2Code": ("KE",),
                "name": ("Kenya",),
            },
            "org/WORLDBANK": {
                "name": ("World Bank",),
            },
        }
    )


def test_ref_with_all_three_fields():
    lookup = EnrichmentLookup(_graph(), parties=lambda: {})

    result = lookup.lookup(["country/KEN"])

    assert result == {"country/KEN": {"iso3": "KEN", "iso2": "KE", "label": "Kenya"}}


def test_a_place_type_carries_every_type_of_the_place_sorted_and_comma_joined():
    # A place can hold several levels at once: nuts/FI2 is a country, an AdministrativeArea1
    # and a NUTS 2 region. The type rides the same batched read as the label.
    graph = FakeGraphReader(
        arcs={
            "nuts/FI2": {
                "typeOf": ("EurostatNUTS2", "Country", "AdministrativeArea1"),
                "name": ("Åland",),
            },
            "Earth": {"typeOf": ("Place",)},
        }
    )
    lookup = EnrichmentLookup(graph, parties=lambda: {})

    result = lookup.lookup(["nuts/FI2", "Earth"])

    assert result == {
        "nuts/FI2": {"type": "AdministrativeArea1,Country,EurostatNUTS2", "label": "Åland"},
        "Earth": {"type": "Place"},
    }
    assert graph.calls["node_arcs_many"] == 1


def test_a_partys_kind_comes_from_its_index_row_whatever_its_graph_types():
    """The index row decides the kind; a party it does not list has none, even when its graph
    types once implied one (`DACGroup` a total, `Organization` a party)."""
    graph = FakeGraphReader(
        arcs={
            "WorldBankGroup": {"typeOf": ("Organization",)},
            "IDA": {"typeOf": ("Organization", "ProvisionalNode")},
            "DACAfrica": {"typeOf": ("DACGroup",)},
            "Bare": {"label": ("No types",)},
        }
    )
    lookup = EnrichmentLookup(
        graph,
        parties=lambda: {
            "WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="total"),
            "Bare": PartyRow(dcid="Bare", kind="residual"),
        },
    )

    result = lookup.lookup(["WorldBankGroup", "IDA", "DACAfrica", "Bare"])

    assert {ref: fields.get("party_kind") for ref, fields in result.items()} == {
        "WorldBankGroup": "total",
        "IDA": None,
        "DACAfrica": None,
        "Bare": "residual",
    }
    assert result["DACAfrica"] == {"type": "DACGroup"}
    assert graph.calls["node_arcs_many"] == 1


def test_a_partys_kind_survives_a_graph_that_returns_no_node_for_it():
    # The graph has no node for WorldBankGroup (or its read failed), yet the index declares it
    # a total. A later call serves the kind from the cached entry without a graph read.
    graph = FakeGraphReader(arcs={})
    lookup = EnrichmentLookup(
        graph,
        parties=lambda: {"WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="total")},
    )

    first = lookup.lookup(["WorldBankGroup", "country/NOWHERE"])
    repeat = lookup.lookup(["WorldBankGroup", "country/NOWHERE"])

    expected = {"WorldBankGroup": {"party_kind": "total"}, "country/NOWHERE": {}}
    assert first == expected
    assert repeat == expected
    assert graph.calls["node_arcs_many"] == 1


def test_a_partys_kind_follows_the_party_rows_each_lookup_reads():
    """The rows can change between lookups, as a graph release swap replaces the code index. A
    cached ref takes the kind the current rows give, with no second graph read."""
    graph = FakeGraphReader(arcs={"WorldBankGroup": {"typeOf": ("Organization",)}})
    rows = {"WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="total")}
    lookup = EnrichmentLookup(graph, parties=lambda: rows)

    before = lookup.lookup(["WorldBankGroup"])
    rows = {"WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="party")}
    after = lookup.lookup(["WorldBankGroup"])

    assert before["WorldBankGroup"]["party_kind"] == "total"
    assert after["WorldBankGroup"] == {"type": "Organization", "party_kind": "party"}
    assert lookup.party_kind("WorldBankGroup") == "party"
    assert graph.calls["node_arcs_many"] == 1


def test_party_kinds_read_the_index_alone_for_flow_parties():
    """A party the index does not list has no kind, whatever its graph types, and a subject
    role holds no party."""
    kinds = read_party_kinds(
        [
            ("financingProvider", "WorldBankGroup"),
            ("origin", "country/USA"),
            ("destination", "DACAfrica"),
            ("creditor", "country/KEN"),
            ("observationAbout", "country/USA"),
        ],
        parties={
            "WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="total"),
            "country/USA": PartyRow(dcid="country/USA", kind="party"),
            "DACAfrica": PartyRow(dcid="DACAfrica", kind="total"),
        },
    )

    assert kinds == {
        ("financingProvider", "WorldBankGroup"): "total",
        ("origin", "country/USA"): "party",
        ("destination", "DACAfrica"): "total",
    }


def test_ref_with_only_a_label_omits_the_missing_fields():
    lookup = EnrichmentLookup(_graph(), parties=lambda: {})

    result = lookup.lookup(["org/WORLDBANK"])

    assert result == {"org/WORLDBANK": {"label": "World Bank"}}


def test_unknown_ref_maps_to_an_empty_mapping():
    lookup = EnrichmentLookup(_graph(), parties=lambda: {})

    result = lookup.lookup(["country/NOWHERE"])

    assert result == {"country/NOWHERE": {}}


def test_repeat_call_makes_no_graph_call():
    graph = _graph()
    lookup = EnrichmentLookup(graph, parties=lambda: {})
    lookup.lookup(["country/KEN", "org/WORLDBANK"])
    assert graph.calls["node_arcs_many"] == 1

    result = lookup.lookup(["country/KEN", "org/WORLDBANK"])

    assert graph.calls["node_arcs_many"] == 1
    assert result == {
        "country/KEN": {"iso3": "KEN", "iso2": "KE", "label": "Kenya"},
        "org/WORLDBANK": {"label": "World Bank"},
    }


def test_partial_repeat_fetches_only_the_new_ref():
    graph = _graph()
    lookup = EnrichmentLookup(graph, parties=lambda: {})
    lookup.lookup(["country/KEN"])
    assert graph.calls["node_arcs_many"] == 1

    result = lookup.lookup(["country/KEN", "org/WORLDBANK"])

    assert graph.calls["node_arcs_many"] == 2
    assert result == {
        "country/KEN": {"iso3": "KEN", "iso2": "KE", "label": "Kenya"},
        "org/WORLDBANK": {"label": "World Bank"},
    }


def test_empty_refs_makes_no_call():
    graph = _graph()
    lookup = EnrichmentLookup(graph, parties=lambda: {})

    result = lookup.lookup([])

    assert result == {}
    assert graph.calls["node_arcs_many"] == 0


def test_concurrent_lookups_of_the_same_uncached_refs_are_all_correct_then_cached():
    # Graph reads run outside the cache lock, so concurrent misses may fetch
    # the same refs. Results remain correct and later reads use the cache.
    graph = _graph()
    lookup = EnrichmentLookup(graph, parties=lambda: {})

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lookup.lookup, ["country/KEN", "org/WORLDBANK"]) for _ in range(8)]
        results = [future.result() for future in futures]

    for result in results:
        assert result == {
            "country/KEN": {"iso3": "KEN", "iso2": "KE", "label": "Kenya"},
            "org/WORLDBANK": {"label": "World Bank"},
        }
    calls_during_concurrency = graph.calls["node_arcs_many"]
    assert calls_during_concurrency >= 1

    result = lookup.lookup(["country/KEN", "org/WORLDBANK"])

    assert graph.calls["node_arcs_many"] == calls_during_concurrency
    assert result == {
        "country/KEN": {"iso3": "KEN", "iso2": "KE", "label": "Kenya"},
        "org/WORLDBANK": {"label": "World Bank"},
    }


def test_cache_evicts_the_least_recently_used_ref_at_the_ceiling(monkeypatch):
    monkeypatch.setattr(enrichment_module, "_CACHE_CEILING", 2)
    graph = FakeGraphReader(
        arcs={
            "country/KEN": {"name": ("Kenya",)},
            "country/TGO": {"name": ("Togo",)},
            "country/UGA": {"name": ("Uganda",)},
        }
    )
    lookup = EnrichmentLookup(graph, parties=lambda: {})
    lookup.lookup(["country/KEN"])
    lookup.lookup(["country/TGO"])
    # Reusing KEN makes TGO least-recently-used; UGA then evicts TGO.
    lookup.lookup(["country/KEN"])
    lookup.lookup(["country/UGA"])
    assert graph.calls["node_arcs_many"] == 3

    # KEN and UGA hit the cache; TGO's lookup makes a fresh graph call.
    assert lookup.lookup(["country/KEN"]) == {"country/KEN": {"label": "Kenya"}}
    assert lookup.lookup(["country/UGA"]) == {"country/UGA": {"label": "Uganda"}}
    assert graph.calls["node_arcs_many"] == 3

    result = lookup.lookup(["country/TGO"])

    assert graph.calls["node_arcs_many"] == 4
    assert result == {"country/TGO": {"label": "Togo"}}


def test_lookup_larger_than_the_ceiling_returns_every_requested_ref(monkeypatch):
    monkeypatch.setattr(enrichment_module, "_CACHE_CEILING", 2)
    graph = FakeGraphReader(
        arcs={
            "country/KEN": {"name": ("Kenya",)},
            "country/TGO": {"name": ("Togo",)},
            "country/UGA": {"name": ("Uganda",)},
        }
    )
    lookup = EnrichmentLookup(graph, parties=lambda: {})

    result = lookup.lookup(["country/KEN", "country/TGO", "country/UGA"])

    assert result == {
        "country/KEN": {"label": "Kenya"},
        "country/TGO": {"label": "Togo"},
        "country/UGA": {"label": "Uganda"},
    }
    assert len(lookup._cache) == 2


def test_refs_evicted_by_a_concurrent_store_are_still_returned(monkeypatch):
    # A concurrent store evicts fetched refs before this call returns. The
    # result must come from this call's fetched values, not the cache.
    monkeypatch.setattr(enrichment_module, "_CACHE_CEILING", 2)
    graph = FakeGraphReader(
        arcs={
            "country/KEN": {"name": ("Kenya",)},
            "country/TGO": {"name": ("Togo",)},
            "country/UGA": {"name": ("Uganda",)},
            "country/RWA": {"name": ("Rwanda",)},
            "country/BDI": {"name": ("Burundi",)},
        }
    )
    lookup = EnrichmentLookup(graph, parties=lambda: {})
    real_node_arcs_many = graph.node_arcs_many
    triggered = False

    def racing_node_arcs_many(dcids, **kwargs):
        nonlocal triggered
        result = real_node_arcs_many(dcids, **kwargs)
        if not triggered and set(dcids) == {"country/KEN", "country/TGO", "country/UGA"}:
            triggered = True
            # Stands in for another thread's concurrent lookup, whose
            # `_store` evicts entries from the cache before this call's own
            # `_store` and return.
            lookup.lookup(["country/RWA", "country/BDI"])
        return result

    monkeypatch.setattr(graph, "node_arcs_many", racing_node_arcs_many)

    result = lookup.lookup(["country/KEN", "country/TGO", "country/UGA"])

    assert result == {
        "country/KEN": {"label": "Kenya"},
        "country/TGO": {"label": "Togo"},
        "country/UGA": {"label": "Uganda"},
    }
    assert triggered
    assert len(lookup._cache) == 2


def test_budget_bounds_individual_probes_and_the_graph_logs_the_unprobed_count(monkeypatch, caplog):
    # Both absent refs remain suspects after the batch retry. With budget 1,
    # only AAA gets an individual probe; the graph logs BBB as unprobed.
    monkeypatch.setattr(enrichment_module, "_INDIVIDUAL_PROBE_BUDGET", 1)
    responses = [
        json.dumps({"data": {"country/AAA": {}, "country/BBB": {}}}),  # initial batch
        json.dumps({"data": {"country/AAA": {}, "country/BBB": {}}}),  # group retry
        json.dumps({"data": {"country/AAA": {}}}),  # individual probe, AAA only
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=responses[len(requests) - 1])

    graph = GraphClient(
        base_url="https://graph.example.org/core/api", transport=httpx.MockTransport(handler)
    )
    lookup = EnrichmentLookup(graph, parties=lambda: {})

    with caplog.at_level("INFO", logger="dc_graph.client"):
        result = lookup.lookup(["country/AAA", "country/BBB"])

    assert len(requests) == 3
    assert result == {"country/AAA": {}, "country/BBB": {}}
    assert any("1 dcid" in record.getMessage() for record in caplog.records)


def test_a_name_read_under_one_release_is_read_again_under_the_next(monkeypatch):
    """With `GRAPH_RELEASE_ORIGIN` on, the bound release answers a DFO variable's name, so a
    name cached under one release is not served under another."""
    monkeypatch.setenv("GRAPH_RELEASE_ORIGIN", "1")
    monkeypatch.setenv("GRAPH_RELEASE_COVERAGE", "1")
    first, second = release_at_revision("rev-1"), release_at_revision("rev-2")
    names = {first.release_id: "Debt flows", second.release_id: "Debt flows, renamed"}
    graph = FakeGraphReader(arcs={})

    def node_arcs_many(dcids, **_kwargs):
        return {dcid: {"name": (names[current_release().release_id],)} for dcid in dcids}

    monkeypatch.setattr(graph, "node_arcs_many", node_arcs_many)
    lookup = EnrichmentLookup(graph, parties=lambda: {})
    read = []
    for release in (first, second, first):
        with bound_release(release):
            read.append(lookup.lookup([_DEBT])[_DEBT]["label"])

    assert read == ["Debt flows", "Debt flows, renamed", "Debt flows"]
