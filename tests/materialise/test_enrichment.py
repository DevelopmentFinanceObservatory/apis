"""Enrichment fields, LRU eviction, probe budgets, and concurrent cache reads."""

from __future__ import annotations

import contextvars
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
from graph_release.context import bound_release, current_release
from materialise.enrichment import (
    PARTY_FIELDS,
    EnrichmentLookup,
    is_whole_total,
    party_cell,
    period_year,
    read_party_kinds,
    row_party_cell,
)
from tests.graph_release.support import release_at_revision

from .conftest import FakeGraphReader

_FIXTURES = Path(__file__).parent.parent / "fixtures"
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
            "Bare": PartyRow(dcid="Bare", kind="residual", party_class="residual"),
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


_DAC2A = "provenance/OECDDAC2A"
_DAC2B = "provenance/OECDDAC2B"

_ETHIOPIA_2021_PARTIES = {
    row.dcid: row
    for row in (
        PartyRow(dcid="country/FRA", kind="party", party_class="official"),
        PartyRow(dcid="GatesFoundation", kind="party", party_class="private"),
        PartyRow(dcid="DACCountriesTotal", kind="total"),
        PartyRow(
            dcid="DACOfficialDonors",
            kind="total",
            party_class="official",
            whole=(
                WholeMarker(dataset=_DAC2A, covers=("official",)),
                WholeMarker(dataset=_DAC2B, covers=("official",)),
            ),
        ),
        PartyRow(
            dcid="DACPrivateDonors",
            kind="total",
            party_class="private",
            whole=(WholeMarker(dataset=_DAC2A, covers=("private",)),),
        ),
        PartyRow(dcid="DACAfricaUnallocated", kind="residual", party_class="residual"),
    )
}
"""The origin refs of DAC2A Ethiopia 2021 as the code index rows give them."""


def test_a_dac2a_ethiopia_2021_row_reads_each_partys_kind_class_and_whole_cells():
    cells = {
        ref: tuple(
            party_cell(field, ref, parties=_ETHIOPIA_2021_PARTIES, year=2021, provenance=_DAC2A)
            for field in PARTY_FIELDS
        )
        for ref in (*_ETHIOPIA_2021_PARTIES, "country/NOWHERE")
    }

    assert PARTY_FIELDS == ("party_kind", "party_class", "party_whole")
    assert cells == {
        "country/FRA": ("party", "official", "false"),
        "GatesFoundation": ("party", "private", "false"),
        "DACCountriesTotal": ("total", "", "false"),
        "DACOfficialDonors": ("total", "official", "true"),
        "DACPrivateDonors": ("total", "private", "true"),
        "DACAfricaUnallocated": ("residual", "residual", "false"),
        "country/NOWHERE": (None, None, None),
    }


def test_official_donors_are_whole_in_both_dac_datasets_and_private_donors_in_one():
    def whole(ref: str, provenance: str) -> object:
        return party_cell(
            "party_whole", ref, parties=_ETHIOPIA_2021_PARTIES, year=2021, provenance=provenance
        )

    assert [whole("DACOfficialDonors", dataset) for dataset in (_DAC2A, _DAC2B)] == [
        "true",
        "true",
    ]
    assert [whole("DACPrivateDonors", dataset) for dataset in (_DAC2A, _DAC2B)] == [
        "true",
        "false",
    ]


def test_the_render_leaves_a_whole_cell_empty_where_it_knows_no_dataset_or_year():
    """`fields_of` carries no row, so `party_whole` is empty for a party with whole markers and
    `false` for every other party; every other cell reads as the row would."""
    reads = _RenderReads(
        enrichment=EnrichmentLookup(
            FakeGraphReader(arcs={}), parties=lambda: _ETHIOPIA_2021_PARTIES
        ),
        known={},
        deadline=None,
    )

    cells = reads.fields_of([*_ETHIOPIA_2021_PARTIES, "country/NOWHERE"], fields=PARTY_FIELDS)

    assert cells == {
        "country/FRA": {"party_kind": "party", "party_class": "official", "party_whole": "false"},
        "GatesFoundation": {
            "party_kind": "party",
            "party_class": "private",
            "party_whole": "false",
        },
        "DACCountriesTotal": {"party_kind": "total", "party_class": "", "party_whole": "false"},
        "DACOfficialDonors": {
            "party_kind": "total",
            "party_class": "official",
            "party_whole": "",
        },
        "DACPrivateDonors": {
            "party_kind": "total",
            "party_class": "private",
            "party_whole": "",
        },
        "DACAfricaUnallocated": {
            "party_kind": "residual",
            "party_class": "residual",
            "party_whole": "false",
        },
        "country/NOWHERE": {},
    }
    assert reads.fields_of(["GatesFoundation"], fields={"party_class"}) == {
        "GatesFoundation": {"party_class": "private"}
    }


_WINDOWED = PartyRow(
    dcid="W",
    kind="total",
    party_class="official",
    whole=(WholeMarker(dataset=_DAC2A, covers=("official",), years=(2010, 2023)),),
)
_UNWINDOWED = PartyRow(
    dcid="W",
    kind="total",
    party_class="official",
    whole=(WholeMarker(dataset=_DAC2A, covers=("official",)),),
)


@pytest.mark.parametrize(
    ("row", "year", "provenance", "expected", "cell"),
    [
        pytest.param(_WINDOWED, 2021, _DAC2A, True, "true", id="inside-the-window"),
        pytest.param(_WINDOWED, 2010, _DAC2A, True, "true", id="window-start-is-inside"),
        pytest.param(_WINDOWED, 2023, _DAC2A, True, "true", id="window-end-is-inside"),
        pytest.param(_WINDOWED, 2009, _DAC2A, False, "false", id="before-the-window"),
        pytest.param(_WINDOWED, 2024, _DAC2A, False, "false", id="after-the-window"),
        pytest.param(_WINDOWED, None, _DAC2A, False, "", id="no-year-is-outside-the-window"),
        pytest.param(_WINDOWED, None, _DAC2B, False, "false", id="no-marker-needs-no-year"),
        pytest.param(_WINDOWED, 2021, _DAC2B, False, "false", id="a-dataset-with-no-marker"),
        pytest.param(_WINDOWED, 2021, None, False, "", id="no-provenance"),
        pytest.param(_UNWINDOWED, None, _DAC2A, True, "true", id="no-window-needs-no-year"),
        pytest.param(_UNWINDOWED, 1960, _DAC2A, True, "true", id="no-window-in-an-early-year"),
        pytest.param(_UNWINDOWED, 2021, None, False, "", id="no-window-still-needs-a-dataset"),
        pytest.param(PartyRow(dcid="W", kind="total"), None, None, False, "false", id="no-marker"),
        pytest.param(
            PartyRow(dcid="W", kind="party", party_class="private"),
            None,
            None,
            False,
            "false",
            id="private-party",
        ),
    ],
)
def test_a_total_is_whole_only_in_the_dataset_and_years_of_its_marker(
    row: PartyRow, year: int | None, provenance: str | None, expected: bool, cell: str
):
    """The cell is empty where the row lacks the dataset, or the year a window needs, that
    decides it; a party with no marker is `false` in every dataset and year."""
    assert is_whole_total(row, year=year, provenance=provenance) is expected
    assert (
        party_cell("party_whole", "W", parties={"W": row}, year=year, provenance=provenance) == cell
    )


def test_the_lookup_reads_only_the_kind_and_its_cells_no_whole_total():
    lookup = EnrichmentLookup(FakeGraphReader(arcs={}), parties=lambda: {"W": _WINDOWED})

    assert lookup.party_kind("W") == "total"
    assert lookup.party_cells("W") == {
        "party_kind": "total",
        "party_class": "official",
        "party_whole": "",
    }
    assert lookup.party_cells("X") == {}
    assert lookup.lookup(["W"]) == {"W": {"party_kind": "total"}}


def test_a_party_cell_refuses_a_field_that_is_not_a_party_field():
    with pytest.raises(ValueError, match="iso3"):
        party_cell("iso3", "W", parties={"W": _WINDOWED})


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


def test_search_reads_a_whole_total_as_a_total():
    """Only a frame's `party_whole` cell says a total is whole; the kind on the wire stays
    `total`."""
    kinds = read_party_kinds([("origin", "DACOfficialDonors")], parties=_ETHIOPIA_2021_PARTIES)

    assert kinds == {("origin", "DACOfficialDonors"): "total"}


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
    # the same refs. Results remain correct and later reads use the cache. Each
    # worker runs in a copy of the caller's context, as pooled request work does,
    # so it reads the release the caller bound.
    graph = _graph()
    lookup = EnrichmentLookup(graph, parties=lambda: {})

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                contextvars.copy_context().run, lookup.lookup, ["country/KEN", "org/WORLDBANK"]
            )
            for _ in range(8)
        ]
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
        pytest.param("W", "2021", _DAC2A, "true", id="dataset-and-year-inside"),
        pytest.param("W", "2022", _DAC2A, "false", id="year-outside"),
        pytest.param("W", "2021", _DAC2B, "false", id="dataset-without-marker"),
        pytest.param("W", None, _DAC2A, "", id="no-period"),
        pytest.param("W", "2020/2021", _DAC2A, "", id="period-across-years"),
        pytest.param("W", "2021", None, "", id="no-provenance"),
        pytest.param("W", "2021", 7, "", id="provenance-not-text"),
        pytest.param(None, "2021", _DAC2A, None, id="ref-not-text"),
        pytest.param("X", "2021", _DAC2A, None, id="ref-not-listed"),
    ],
)
def test_a_rows_cells_give_its_party_whole_cell(ref, period, provenance, expected):
    parties = {
        "W": PartyRow(
            dcid="W",
            kind="total",
            party_class="official",
            whole=(WholeMarker(dataset=_DAC2A, covers=("official",), years=(2020, 2021)),),
        )
    }

    assert (
        row_party_cell("party_whole", ref, parties=parties, period=period, provenance=provenance)
        == expected
    )


def test_a_name_read_under_one_release_is_read_again_under_the_next(monkeypatch):
    """The bound release answers a DFO variable's name, so a name cached under one release is
    not served under another."""
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
