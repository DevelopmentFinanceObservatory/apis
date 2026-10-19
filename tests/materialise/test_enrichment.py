"""Enrichment fields, LRU eviction, probe budgets, and concurrent cache reads."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

import materialise.enrichment as enrichment_module
from code_index.shared import get_index
from code_index.types import PartyRow, WholeMarker
from datasets_api.render import _RenderReads
from dc_graph import GraphClient
from materialise.enrichment import (
    EnrichmentLookup,
    PartyKindCell,
    party_kind_cell,
    period_year,
    read_party_kinds,
    row_party_kind_cell,
)

from .conftest import FakeGraphReader

_FIXTURES = Path(__file__).parent.parent / "fixtures"


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
    lookup = EnrichmentLookup(_graph(), parties={})

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
    lookup = EnrichmentLookup(graph, parties={})

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
        parties={
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
    # a total. The kind is cached with the entry, so a later call serves it too.
    graph = FakeGraphReader(arcs={})
    lookup = EnrichmentLookup(
        graph,
        parties={"WorldBankGroup": PartyRow(dcid="WorldBankGroup", kind="total")},
    )

    first = lookup.lookup(["WorldBankGroup", "country/NOWHERE"])
    repeat = lookup.lookup(["WorldBankGroup", "country/NOWHERE"])

    expected = {"WorldBankGroup": {"party_kind": "total"}, "country/NOWHERE": {}}
    assert first == expected
    assert repeat == expected
    assert graph.calls["node_arcs_many"] == 1


_DAC2A = "provenance/OECDDAC2A"
_DAC2B = "provenance/OECDDAC2B"

_ETHIOPIA_2021_PARTIES = {
    row.dcid: row
    for row in (
        PartyRow(dcid="country/FRA", kind="party"),
        PartyRow(dcid="GatesFoundation", kind="private"),
        PartyRow(dcid="DACCountriesTotal", kind="total"),
        PartyRow(
            dcid="DACOfficialDonors",
            kind="total",
            whole=(
                WholeMarker(dataset=_DAC2A, holds=("party",)),
                WholeMarker(dataset=_DAC2B, holds=("party",)),
            ),
        ),
        PartyRow(
            dcid="DACPrivateDonors",
            kind="total",
            whole=(WholeMarker(dataset=_DAC2A, holds=("private",)),),
        ),
        PartyRow(dcid="DACAfricaUnallocated", kind="residual"),
    )
}
"""The origin refs of DAC2A Ethiopia 2021 as the code index rows give them."""


def test_a_dac2a_ethiopia_2021_row_reads_private_and_whole_total_cells():
    cells = {
        ref: party_kind_cell(ref, parties=_ETHIOPIA_2021_PARTIES, year=2021, provenance=_DAC2A)
        for ref in (*_ETHIOPIA_2021_PARTIES, "country/NOWHERE")
    }

    assert cells == {
        "country/FRA": "party",
        "GatesFoundation": "private",
        "DACCountriesTotal": "total",
        "DACOfficialDonors": "whole_total",
        "DACPrivateDonors": "whole_total",
        "DACAfricaUnallocated": "residual",
        "country/NOWHERE": None,
    }


def test_official_donors_read_whole_total_in_both_dac_datasets_and_private_donors_in_one():
    def cell(ref: str, provenance: str) -> object:
        return party_kind_cell(
            ref, parties=_ETHIOPIA_2021_PARTIES, year=2021, provenance=provenance
        )

    assert [cell("DACOfficialDonors", dataset) for dataset in (_DAC2A, _DAC2B)] == [
        "whole_total",
        "whole_total",
    ]
    assert [cell("DACPrivateDonors", dataset) for dataset in (_DAC2A, _DAC2B)] == [
        "whole_total",
        "total",
    ]


def test_the_lookup_reads_a_whole_total_as_a_total_where_it_knows_no_dataset_or_year():
    """`fields_of` and `lookup` carry no row, so only a row-aware reader writes `whole_total`;
    every other kind reads as the row would."""
    reads = _RenderReads(
        enrichment=EnrichmentLookup(FakeGraphReader(arcs={}), parties=_ETHIOPIA_2021_PARTIES),
        known={},
        deadline=None,
    )

    cells = reads.fields_of([*_ETHIOPIA_2021_PARTIES], fields={"party_kind"})

    assert {ref: fields.get("party_kind") for ref, fields in cells.items()} == {
        "country/FRA": "party",
        "GatesFoundation": "private",
        "DACCountriesTotal": "total",
        "DACOfficialDonors": "total",
        "DACPrivateDonors": "total",
        "DACAfricaUnallocated": "residual",
    }


_WINDOWED = PartyRow(
    dcid="W",
    kind="total",
    whole=(WholeMarker(dataset=_DAC2A, holds=("party",), years=(2010, 2023)),),
)
_UNWINDOWED = PartyRow(
    dcid="W", kind="total", whole=(WholeMarker(dataset=_DAC2A, holds=("party",)),)
)


@pytest.mark.parametrize(
    ("row", "year", "provenance", "expected"),
    [
        pytest.param(_WINDOWED, 2021, _DAC2A, "whole_total", id="inside-the-window"),
        pytest.param(_WINDOWED, 2010, _DAC2A, "whole_total", id="window-start-is-inside"),
        pytest.param(_WINDOWED, 2023, _DAC2A, "whole_total", id="window-end-is-inside"),
        pytest.param(_WINDOWED, 2009, _DAC2A, "total", id="before-the-window"),
        pytest.param(_WINDOWED, 2024, _DAC2A, "total", id="after-the-window"),
        pytest.param(_WINDOWED, None, _DAC2A, "total", id="no-year-is-outside-the-window"),
        pytest.param(_WINDOWED, 2021, _DAC2B, "total", id="a-dataset-with-no-marker"),
        pytest.param(_WINDOWED, 2021, None, "total", id="no-provenance"),
        pytest.param(_UNWINDOWED, None, _DAC2A, "whole_total", id="no-window-needs-no-year"),
        pytest.param(_UNWINDOWED, 1960, _DAC2A, "whole_total", id="no-window-in-an-early-year"),
        pytest.param(_UNWINDOWED, 2021, None, "total", id="no-window-still-needs-a-dataset"),
        pytest.param(PartyRow(dcid="W", kind="total"), 2021, _DAC2A, "total", id="no-marker"),
        pytest.param(
            PartyRow(dcid="W", kind="private"), 2021, _DAC2A, "private", id="private-party"
        ),
    ],
)
def test_a_whole_total_reads_whole_total_only_in_the_dataset_and_years_of_its_marker(
    row: PartyRow, year: int | None, provenance: str | None, expected: PartyKindCell
):
    assert party_kind_cell("W", parties={"W": row}, year=year, provenance=provenance) == expected


def test_the_lookup_reads_a_whole_total_as_a_total_until_a_dataset_and_year_are_given():
    lookup = EnrichmentLookup(FakeGraphReader(arcs={}), parties={"W": _WINDOWED})

    assert lookup.party_kind("W") == "total"
    assert lookup.party_kind("W", year=2021) == "total"
    assert lookup.party_kind("W", year=2021, provenance=_DAC2A) == "whole_total"
    assert lookup.lookup(["W"]) == {"W": {"party_kind": "total"}}


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


def test_search_reads_a_whole_total_as_a_total():
    """Only a frame cell reads `whole_total`; the kind on the wire stays `total`."""
    kinds = read_party_kinds([("origin", "DACOfficialDonors")], parties=_ETHIOPIA_2021_PARTIES)

    assert kinds == {("origin", "DACOfficialDonors"): "total"}


def test_ref_with_only_a_label_omits_the_missing_fields():
    lookup = EnrichmentLookup(_graph(), parties={})

    result = lookup.lookup(["org/WORLDBANK"])

    assert result == {"org/WORLDBANK": {"label": "World Bank"}}


def test_unknown_ref_maps_to_an_empty_mapping():
    lookup = EnrichmentLookup(_graph(), parties={})

    result = lookup.lookup(["country/NOWHERE"])

    assert result == {"country/NOWHERE": {}}


def test_repeat_call_makes_no_graph_call():
    graph = _graph()
    lookup = EnrichmentLookup(graph, parties={})
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
    lookup = EnrichmentLookup(graph, parties={})
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
    lookup = EnrichmentLookup(graph, parties={})

    result = lookup.lookup([])

    assert result == {}
    assert graph.calls["node_arcs_many"] == 0


def test_concurrent_lookups_of_the_same_uncached_refs_are_all_correct_then_cached():
    # Graph reads run outside the cache lock, so concurrent misses may fetch
    # the same refs. Results remain correct and later reads use the cache.
    graph = _graph()
    lookup = EnrichmentLookup(graph, parties={})

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
    lookup = EnrichmentLookup(graph, parties={})
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
    lookup = EnrichmentLookup(graph, parties={})

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
    lookup = EnrichmentLookup(graph, parties={})
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
    lookup = EnrichmentLookup(graph, parties={})

    with caplog.at_level("INFO", logger="dc_graph.client"):
        result = lookup.lookup(["country/AAA", "country/BBB"])

    assert len(requests) == 3
    assert result == {"country/AAA": {}, "country/BBB": {}}
    assert any("1 dcid" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("cell", "year"),
    [
        pytest.param("2021", 2021, id="year"),
        pytest.param("2021-06", 2021, id="month"),
        pytest.param("2021-06-30", 2021, id="day"),
        pytest.param("2021/2021", 2021, id="interval-inside-a-year"),
        pytest.param("2021-01/2021-12", 2021, id="months-inside-a-year"),
        pytest.param(2021, 2021, id="integer-year"),
        pytest.param("2019/2021", None, id="interval-across-years"),
        pytest.param("2021-Q2", None, id="other-spelling"),
        pytest.param("", None, id="empty"),
        pytest.param(None, None, id="null"),
        pytest.param(True, None, id="boolean"),
    ],
)
def test_a_period_has_a_year_only_when_it_lies_in_one(cell: object, year: int | None):
    assert period_year(cell) == year


def test_a_recorded_facets_provenance_has_the_form_a_party_rows_datasets_hold():
    """A frame's `provenance` cell is its facet's `provenanceId`, the dataset the mint elected
    (`ElectedDataset.publishes`), which is also what `PartyRow.datasets` lists. Every provenance
    in the recorded SDMX facets is such a dcid, and the DAC datasets a whole total is marked in
    are spelled the same way."""
    recorded = json.loads((_FIXTURES / "sdmx_obs.json").read_text())
    provenances = {
        facet["provenanceId"]
        for key, facets in recorded.items()
        if not key.startswith("_")
        for facet in facets
    }
    parties = get_index().parties
    listed = {dataset for party in parties.values() for dataset in party.datasets}

    assert provenances
    assert provenances <= listed
    assert {_DAC2A, _DAC2B} <= set(parties["DACAllRecipients"].datasets)


@pytest.mark.parametrize(
    ("ref", "period", "provenance", "expected"),
    [
        pytest.param("W", "2021", _DAC2A, "whole_total", id="dataset-and-year-inside"),
        pytest.param("W", "2022", _DAC2A, "total", id="year-outside"),
        pytest.param("W", "2021", _DAC2B, "total", id="dataset-without-marker"),
        pytest.param("W", None, _DAC2A, "total", id="no-period"),
        pytest.param("W", "2021", None, "total", id="no-provenance"),
        pytest.param("W", "2021", 7, "total", id="provenance-not-text"),
        pytest.param(None, "2021", _DAC2A, None, id="ref-not-text"),
        pytest.param("X", "2021", _DAC2A, None, id="ref-not-listed"),
    ],
)
def test_a_rows_cells_give_its_party_kind_cell(ref, period, provenance, expected):
    parties = {
        "W": PartyRow(
            dcid="W",
            kind="total",
            whole=(WholeMarker(dataset=_DAC2A, holds=("party",), years=(2020, 2021)),),
        )
    }

    assert (
        row_party_kind_cell(ref, parties=parties, period=period, provenance=provenance) == expected
    )
