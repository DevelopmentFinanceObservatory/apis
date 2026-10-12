"""Enrichment fields, LRU eviction, probe budgets, and concurrent cache reads."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import httpx

import materialise.enrichment as enrichment_module
from code_index.types import PartyRow
from dc_graph import GraphClient
from materialise.enrichment import EnrichmentLookup

from .conftest import FakeGraphReader


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
    lookup = EnrichmentLookup(_graph(), type_kinds={}, parties={})

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
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})

    result = lookup.lookup(["nuts/FI2", "Earth"])

    assert result == {
        "nuts/FI2": {"type": "AdministrativeArea1,Country,EurostatNUTS2", "label": "Åland"},
        "Earth": {"type": "Place"},
    }
    assert graph.calls["node_arcs_many"] == 1


def test_a_partys_kind_comes_from_its_types_under_the_declared_party_types():
    graph = FakeGraphReader(
        arcs={
            "country/IND": {"typeOf": ("Country", "Place")},
            "DACAfrica": {"typeOf": ("DACGroup",)},
            "DACAfricaUnallocated": {"typeOf": ("DACRecipientUnallocatedByCountry",)},
            "Earth": {"typeOf": ("Place",)},
            "nuts/FI2": {"typeOf": ("Country", "EurostatNUTS2")},
        }
    )
    lookup = EnrichmentLookup(
        graph,
        type_kinds={
            "Country": "party",
            "DACGroup": "total",
            "DACRecipientUnallocatedByCountry": "residual",
            "Place": None,
        },
        parties={},
    )

    result = lookup.lookup(
        ["country/IND", "DACAfrica", "DACAfricaUnallocated", "Earth", "nuts/FI2"]
    )

    assert {ref: fields.get("kind") for ref, fields in result.items()} == {
        "country/IND": "party",
        "DACAfrica": "total",
        "DACAfricaUnallocated": "residual",
        # Place alone carries no kind, and EurostatNUTS2 is not declared.
        "Earth": None,
        "nuts/FI2": None,
    }
    assert graph.calls["node_arcs_many"] == 1


def test_a_partys_own_kind_takes_precedence_over_the_kind_its_types_give():
    graph = FakeGraphReader(
        arcs={
            "WorldBankGroup": {"typeOf": ("Organization",)},
            "IDA": {"typeOf": ("Organization", "ProvisionalNode")},
            "Bare": {"label": ("No types",)},
        }
    )
    lookup = EnrichmentLookup(
        graph,
        type_kinds={"Organization": "party", "ProvisionalNode": None},
        parties={
            "WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="total"),
            "Bare": PartyRow(dcid="Bare", kind="residual"),
        },
    )

    result = lookup.lookup(["WorldBankGroup", "IDA", "Bare"])

    assert {ref: fields.get("kind") for ref, fields in result.items()} == {
        "WorldBankGroup": "total",
        "IDA": "party",
        "Bare": "residual",
    }


def test_a_partys_kind_survives_a_graph_that_returns_no_node_for_it():
    # The graph has no node for WorldBankGroup (or its read failed), yet the index declares it
    # a total. The kind is cached with the entry, so a later call serves it too.
    graph = FakeGraphReader(arcs={})
    lookup = EnrichmentLookup(
        graph,
        type_kinds={"Organization": "party"},
        parties={"WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="total")},
    )

    first = lookup.lookup(["WorldBankGroup", "country/NOWHERE"])
    repeat = lookup.lookup(["WorldBankGroup", "country/NOWHERE"])

    expected = {"WorldBankGroup": {"kind": "total"}, "country/NOWHERE": {}}
    assert first == expected
    assert repeat == expected
    assert graph.calls["node_arcs_many"] == 1


def test_ref_with_only_a_label_omits_the_missing_fields():
    lookup = EnrichmentLookup(_graph(), type_kinds={}, parties={})

    result = lookup.lookup(["org/WORLDBANK"])

    assert result == {"org/WORLDBANK": {"label": "World Bank"}}


def test_unknown_ref_maps_to_an_empty_mapping():
    lookup = EnrichmentLookup(_graph(), type_kinds={}, parties={})

    result = lookup.lookup(["country/NOWHERE"])

    assert result == {"country/NOWHERE": {}}


def test_repeat_call_makes_no_graph_call():
    graph = _graph()
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})
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
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})
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
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})

    result = lookup.lookup([])

    assert result == {}
    assert graph.calls["node_arcs_many"] == 0


def test_concurrent_lookups_of_the_same_uncached_refs_are_all_correct_then_cached():
    # Graph reads run outside the cache lock, so concurrent misses may fetch
    # the same refs. Results remain correct and later reads use the cache.
    graph = _graph()
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})

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
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})
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
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})

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
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})
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
    lookup = EnrichmentLookup(graph, type_kinds={}, parties={})

    with caplog.at_level("INFO", logger="dc_graph.client"):
        result = lookup.lookup(["country/AAA", "country/BBB"])

    assert len(requests) == 3
    assert result == {"country/AAA": {}, "country/BBB": {}}
    assert any("1 dcid" in record.getMessage() for record in caplog.records)
