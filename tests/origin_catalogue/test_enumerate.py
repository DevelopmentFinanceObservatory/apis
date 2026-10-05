"""Tests the base variable, withheld variable and topic tables built against a fake graph."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import pytest

from dc_graph import (
    GraphCapExceededError,
    GraphDeadlineExceededError,
    GraphReadError,
    SubjectPage,
)
from origin_catalogue import BASE, DFO, ScanLevel
from origin_catalogue.catalogue import BaseVariable, ProvenanceRow, SourceRef
from origin_catalogue.enumerate import (
    OBSERVATION_PATH,
    EnumerationError,
    EnumerationSettings,
    build_base_tables,
)
from origin_catalogue.registry import Registry, parse_registry
from origin_catalogue.topic_walk import walk_topics

WDI = "dc/base/WorldDevelopmentIndicators"
EUROSTAT = "dc/base/EurostatData"
UN_SDG = "dc/base/UN_SDG"  # not admitted, and shares a hostname with WDI
CRS = "provenance/OECDCreditorReportingSystem"

REGISTRY = parse_registry(
    b"""
[[base]]
source = "dc/s/WorldBank"
label = "World Bank"

[[base]]
source = "dc/s/EuropeanUnionEuEurostat"
label = "Eurostat"
scan_type = "EurostatNUTS2"

[[base]]
source = "dc/s/UsBureauOfLaborStatisticsBls"
label = "BLS"
country = "country/USA"
scan_type = "State"

[[un_system]]
provenance = "undata/p/WHO"
segment = "who"
label = "WHO"
"""
)


def _row(provenance_id: str, url: str) -> ProvenanceRow:
    return ProvenanceRow(
        id=provenance_id,
        data_commons=BASE,
        source=SourceRef(ref="dc/s/Any", label="Any"),
        name=provenance_id,
        url=url,
    )


ADMITTED = {
    WDI: _row(WDI, "https://datacatalog.worldbank.org/wdi"),
    EUROSTAT: _row(EUROSTAT, "https://ec.europa.eu/eurostat"),
}
FAST = EnumerationSettings(concurrency=1, retry_backoff_seconds=(0.0,))


@dataclass
class FakeGraph:
    """Places, topics and observations in memory.

    `existence` is the pass 1 answer per place; `facets` maps (variable,
    place) to provenance ids. A facet read answers every facet: the mixer's
    `filter.domains` would drop some, so a request carrying one fails. `max_pairs` answers a larger observation
    response with an empty 500, as Cloud Run does over 32 MiB. `fail` may
    raise for a payload before it is answered.
    """

    countries: list[str] = field(default_factory=lambda: ["country/KEN", "country/USA"])
    children: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    instances: dict[str, list[str]] = field(default_factory=dict)
    existence: dict[str, list[str]] = field(default_factory=dict)
    facets: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    arcs: dict[str, dict] = field(default_factory=dict)
    max_pairs: int | None = None
    listing_error: GraphReadError | None = None
    fail: Callable[[dict], None] | None = None
    payloads: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        if list(parent_dcids) == ["Earth"] and child_type == "Country":
            return {"Earth": list(self.countries)}
        return {p: list(self.children.get((p, child_type), [])) for p in parent_dcids}

    def node_subjects(
        self, dcid: str, *, property: str, max_results: int, deadline: float | None = None
    ) -> SubjectPage:
        assert property == "typeOf"
        if self.listing_error is not None:
            raise self.listing_error
        found = self.instances.get(dcid, [])
        return SubjectPage(dcids=tuple(found[:max_results]), truncated=len(found) > max_results)

    def node_arcs_batch(self, dcids: list[str]) -> Mapping[str, dict | None]:
        return {dcid: self.arcs.get(dcid) for dcid in dcids}

    def post_capped_bytes(
        self, path: str, payload: dict, *, max_bytes: int | None, deadline: float | None = None
    ) -> dict:
        assert path == OBSERVATION_PATH
        with self.lock:
            self.payloads.append(payload)
        if self.fail is not None:
            self.fail(payload)
        places = payload["entity"]["dcids"]
        if "variable" not in payload:
            pairs = [(v, p) for p in places for v in self.existence.get(p, [])]
            self._check_size(pairs)
            body: dict = {"byVariable": {}}
            for variable, place in pairs:
                body["byVariable"].setdefault(variable, {"byEntity": {}})["byEntity"][place] = {}
            return body
        assert "filter" not in payload
        body = {"byVariable": {}, "facets": {}}
        pairs = []
        for variable in payload["variable"]["dcids"]:
            for place in places:
                for provenance in self.facets.get((variable, place), []):
                    pairs.append((variable, place))
                    facet_id = f"f-{provenance}"
                    body["facets"][facet_id] = {"provenanceId": provenance}
                    entry = body["byVariable"].setdefault(variable, {"byEntity": {}})
                    entry["byEntity"].setdefault(place, {"orderedFacets": []})[
                        "orderedFacets"
                    ].append({"facetId": facet_id, "obsCount": 3})
        self._check_size(pairs)
        return body

    def _check_size(self, pairs: list) -> None:
        if self.max_pairs is not None and len(pairs) > self.max_pairs:
            raise GraphReadError("graph returned HTTP 500", upstream_status=500)

    def observation_reads(self, *, pass_: int) -> list[dict]:
        return [p for p in self.payloads if ("variable" in p) == (pass_ == 2)]


def _graph(**overrides: object) -> FakeGraph:
    """Earth, two countries, a continent, an empty UN region, NUTS 2 and two
    US states, with WDI and Eurostat admitted, the UN SDG copy not, and CRS
    DFO's. `Count_Phantom` is listed at the USA and has no facet."""
    graph = FakeGraph(
        children={("country/USA", "State"): ["geoId/06", "geoId/36"]},
        instances={
            "Continent": ["africa"],
            "UNGeoRegion": ["EmptyRegion"],
            "EurostatNUTS2": ["nuts/DE11"],
        },
        existence={
            "Earth": ["Count_Person"],
            "country/KEN": ["Count_Person", "sdg/SH_DTH", "Amount_CRS"],
            "country/USA": ["Count_Person", "Count_Phantom", "GDP"],
            "africa": ["Count_Person"],
            "nuts/DE11": ["GDP"],
            "geoId/06": ["Count_Person"],
        },
        facets={
            ("Count_Person", "Earth"): [WDI],
            ("Count_Person", "country/KEN"): [WDI],
            ("sdg/SH_DTH", "country/KEN"): [UN_SDG],
            ("Amount_CRS", "country/KEN"): [CRS],
            ("Count_Person", "country/USA"): [WDI],
            ("GDP", "country/USA"): [UN_SDG, WDI, EUROSTAT],
            ("Count_Person", "africa"): [WDI],
            ("GDP", "nuts/DE11"): [EUROSTAT],
        },
    )
    for key, value in overrides.items():
        setattr(graph, key, value)
    return graph


def _build(
    graph: FakeGraph,
    *,
    registry: Registry = REGISTRY,
    settings: EnumerationSettings = FAST,
    deadline: float | None = None,
):
    return build_base_tables(
        graph=graph,
        registry=registry,
        admitted_base_provenances=ADMITTED,
        dfo_provenances=frozenset({CRS}),
        settings=settings,
        deadline=deadline,
    )


def _stats(capsys: pytest.CaptureFixture[str]) -> dict:
    """The last `origin_catalogue_enumerated` event printed."""
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return [e for e in events if e["event"] == "origin_catalogue_enumerated"][-1]


# --- Scan places and pass 1 ---------------------------------------------------


def test_scan_places_keep_countries_and_earth_and_drop_empty_other_places(
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = _graph(countries=["country/USA", "country/KEN", "country/NRU"])

    tables = _build(graph)

    # Earth first, countries by dcid (Nauru kept with no data), then the other
    # places with data by dcid; the empty UN region and geoId/36 are dropped.
    assert tables.scan_places == (
        "Earth",
        "country/KEN",
        "country/NRU",
        "country/USA",
        "africa",
        "geoId/06",
        "nuts/DE11",
    )
    stats = _stats(capsys)
    assert (stats["candidate_places"], stats["scan_places"]) == (9, 7)
    read = {p for payload in graph.observation_reads(pass_=1) for p in payload["entity"]["dcids"]}
    assert {"EmptyRegion", "geoId/36"} <= read


def test_scan_levels_record_each_registry_entry_level() -> None:
    tables = _build(_graph())

    assert tables.scan_levels == {
        "dc/s/EuropeanUnionEuEurostat": (ScanLevel(place_type="EurostatNUTS2"),),
        "dc/s/UsBureauOfLaborStatisticsBls": (ScanLevel(place_type="State", within="country/USA"),),
    }


def test_pass1_names_places_per_request_and_no_variable() -> None:
    graph = _graph()

    _build(graph, settings=EnumerationSettings(concurrency=1, places_per_request=4))

    reads = graph.observation_reads(pass_=1)
    assert [len(r["entity"]["dcids"]) for r in reads] == [4, 4]
    assert all(r["select"] == ["variable", "entity"] for r in reads)


def test_pass1_splits_a_group_whose_response_is_too_large(
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = _graph(max_pairs=3)
    settings = EnumerationSettings(concurrency=1, places_per_request=9)

    tables = _build(graph, settings=settings)
    stats = _stats(capsys)

    unsplit = _build(_graph(), settings=settings)
    assert tables.scan_places == unsplit.scan_places
    assert tables.base_variables == unsplit.base_variables
    assert stats["pass1"]["splits"] > 0
    assert stats["pass1"]["requests"] > 1


# --- Pass 2 -------------------------------------------------------------------


def test_pass2_keeps_admitted_provenances_and_sets_place_bits() -> None:
    tables = _build(_graph())
    bit = {place: 1 << i for i, place in enumerate(tables.scan_places)}

    # geoId/06 lists Count_Person in pass 1 with no facet there, as the
    # existence table over-counts, so its bit stays clear. GDP's UN SDG facet
    # at the USA names no provenance in the table.
    assert tables.base_variables == {
        "Count_Person": BaseVariable(
            provenances=(WDI,),
            places=bit["Earth"] | bit["country/KEN"] | bit["country/USA"] | bit["africa"],
        ),
        "GDP": BaseVariable(
            provenances=(EUROSTAT, WDI), places=bit["country/USA"] | bit["nuts/DE11"]
        ),
    }


def test_pass2_records_where_each_admitted_provenance_holds_data() -> None:
    tables = _build(_graph())
    bit = {place: 1 << i for i, place in enumerate(tables.scan_places)}

    # WDI holds GDP at the USA only, so Eurostat alone has the NUTS region.
    assert tables.provenance_places == {
        EUROSTAT: bit["country/USA"] | bit["nuts/DE11"],
        WDI: bit["Earth"] | bit["country/KEN"] | bit["country/USA"] | bit["africa"],
    }
    assert not tables.provenance_places[WDI] & bit["nuts/DE11"]


def test_pass2_withholds_a_variable_whose_facets_are_all_unadmitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tables = _build(_graph())

    # sdg/SH_DTH has only the UN SDG copy's facet. Count_Phantom has no facet
    # anywhere, so it is in neither table.
    assert tables.withheld_variables == frozenset({"sdg/SH_DTH"})
    assert "Count_Phantom" not in tables.base_variables
    assert _stats(capsys)["withheld_variables"] == 1


def test_a_dfo_facet_neither_admits_nor_withholds_a_variable() -> None:
    graph = _graph()
    graph.existence["country/KEN"].append("Amount_Mixed")
    graph.facets[("Amount_Mixed", "country/KEN")] = [CRS, UN_SDG]

    tables = _build(graph)

    # Amount_CRS has only DFO's facet, so it is in neither table; Amount_Mixed
    # is withheld on its UN SDG facet.
    assert "Amount_CRS" not in tables.base_variables
    assert "Amount_CRS" not in tables.withheld_variables
    assert tables.withheld_variables == frozenset({"sdg/SH_DTH", "Amount_Mixed"})


def test_an_admitted_facet_at_any_scan_place_keeps_a_variable_out_of_the_withheld_set() -> None:
    graph = _graph()
    graph.existence["Earth"].append("sdg/SH_DTH")
    graph.facets[("sdg/SH_DTH", "Earth")] = [WDI]

    tables = _build(graph)

    assert tables.base_variables["sdg/SH_DTH"] == BaseVariable(provenances=(WDI,), places=1)
    assert tables.withheld_variables == frozenset()


def test_pass2_reads_each_place_by_name_with_no_domain_filter() -> None:
    graph = _graph()

    _build(graph)

    reads = graph.observation_reads(pass_=2)
    assert all("filter" not in r for r in reads)
    assert all(r["select"] == ["variable", "entity", "facet"] for r in reads)
    by_place = {r["entity"]["dcids"][0]: r["variable"]["dcids"] for r in reads}
    assert by_place["country/KEN"] == ["Amount_CRS", "Count_Person", "sdg/SH_DTH"]
    assert "geoId/36" not in by_place
    assert all(len(r["entity"]["dcids"]) == 1 for r in reads)


def test_pass2_batches_variables_per_request() -> None:
    graph = _graph()

    _build(graph, settings=EnumerationSettings(concurrency=1, variables_per_request=2))

    ken = [
        r["variable"]["dcids"]
        for r in graph.observation_reads(pass_=2)
        if r["entity"]["dcids"] == ["country/KEN"]
    ]
    assert ken == [["Amount_CRS", "Count_Person"], ["sdg/SH_DTH"]]


@pytest.mark.parametrize(
    "error",
    [
        GraphReadError("graph returned HTTP 500", upstream_status=500),
        GraphDeadlineExceededError("the request timed out"),
        GraphCapExceededError("too big", cap_kind="bytes", cap=1, observed=2),
    ],
)
def test_pass2_splits_a_batch_on_a_server_error_timeout_or_cap(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(payload: dict) -> None:
        if (
            payload["entity"]["dcids"] == ["country/KEN"]
            and len(payload.get("variable", {}).get("dcids", [])) > 1
        ):
            raise error

    tables = _build(_graph(fail=fail))
    stats = _stats(capsys)

    assert tables == _build(_graph())
    assert (stats["pass1"]["splits"], stats["pass2"]["splits"]) == (0, 2)


def test_concurrent_build_matches_sequential() -> None:
    settings = EnumerationSettings(
        concurrency=4, places_per_request=2, variables_per_request=1, retry_backoff_seconds=(0.0,)
    )

    assert _build(_graph(), settings=settings) == _build(_graph())


# --- Failures -----------------------------------------------------------------


def _fail_first(times: int, exc: Exception, *, when: Callable[[dict], bool]):
    remaining = [times]

    def fail(payload: dict) -> None:
        if when(payload) and remaining[0] > 0:
            remaining[0] -= 1
            raise exc

    return fail


def _single_place_pass2(payload: dict) -> bool:
    return payload["entity"]["dcids"] == ["nuts/DE11"] and "variable" in payload


@pytest.mark.parametrize(
    "error",
    [
        GraphReadError("transport error", upstream_status=None),
        GraphReadError("HTTP 503", upstream_status=503),
        GraphReadError("HTTP 429", upstream_status=429),
    ],
)
def test_transient_error_on_one_item_is_retried(error: GraphReadError) -> None:
    graph = _graph(fail=_fail_first(2, error, when=_single_place_pass2))

    tables = _build(graph)

    assert tables.base_variables["GDP"].provenances == (EUROSTAT, WDI)


def test_persistent_error_fails_the_build() -> None:
    error = GraphReadError("HTTP 503", upstream_status=503)
    graph = _graph(fail=_fail_first(99, error, when=_single_place_pass2))

    with pytest.raises(EnumerationError, match="failed 3 times") as info:
        _build(graph)
    assert info.value.__cause__ is error


def test_client_error_fails_without_retry() -> None:
    error = GraphReadError("HTTP 400", upstream_status=400)
    graph = _graph(fail=_fail_first(99, error, when=_single_place_pass2))

    with pytest.raises(EnumerationError, match="refused"):
        _build(graph)
    assert len([p for p in graph.payloads if _single_place_pass2(p)]) == 1


def test_byte_cap_on_one_item_fails_the_build() -> None:
    error = GraphCapExceededError("too big", cap_kind="bytes", cap=1, observed=2)
    graph = _graph(fail=_fail_first(1, error, when=_single_place_pass2))

    with pytest.raises(EnumerationError, match="exceeds"):
        _build(graph)


def test_node_read_failure_fails_the_build() -> None:
    graph = _graph(listing_error=GraphReadError("HTTP 502", upstream_status=502))

    with pytest.raises(EnumerationError, match="Continent"):
        _build(graph)


def test_truncated_listing_fails_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("origin_catalogue.enumerate._MAX_SUBJECTS", 1)
    graph = _graph(instances={"Continent": ["africa", "asia"]})

    with pytest.raises(EnumerationError, match="more than 1"):
        _build(graph)


def test_passed_deadline_fails_the_build_before_reading() -> None:
    graph = _graph()

    with pytest.raises(EnumerationError, match="deadline"):
        _build(graph, deadline=time.monotonic() - 1)
    assert graph.payloads == []


def test_provenance_set_must_be_non_empty_base_rows() -> None:
    with pytest.raises(ValueError, match="no admitted"):
        build_base_tables(
            graph=_graph(),
            registry=REGISTRY,
            admitted_base_provenances={},
            dfo_provenances=frozenset(),
        )
    dfo = {CRS: replace(_row(CRS, "https://data-explorer.oecd.org/crs"), data_commons=DFO)}
    with pytest.raises(ValueError, match="not base"):
        build_base_tables(
            graph=_graph(),
            registry=REGISTRY,
            admitted_base_provenances=dfo,
            dfo_provenances=frozenset(),
        )


def test_settings_refuse_non_positive_values() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        EnumerationSettings(concurrency=0)
    with pytest.raises(ValueError, match="retry_backoff_seconds"):
        EnumerationSettings(retry_backoff_seconds=())


# --- Topic table --------------------------------------------------------------


def _member(dcid: str, *types: str) -> dict:
    return {"dcid": dcid, "types": list(types)}


def _topic_graph() -> FakeGraph:
    return _graph(
        instances={
            "Continent": ["africa"],
            "Topic": [
                "dc/topic/Population",
                "dc/topic/SdgOnly",
                "dc/topic/TooBig",
                "dc/topic/Nested",
                "dc/topic/Missing",
            ],
        },
        arcs={
            "dc/topic/Population": {
                "relevantVariable": {"nodes": [_member("Count_Person"), _member("Count_Person")]}
            },
            "dc/topic/SdgOnly": {"relevantVariable": {"nodes": [_member("sdg/SH_DTH")]}},
            "dc/topic/TooBig": {
                "relevantVariable": {
                    "nodes": [_member("Count_Person"), _member("GDP"), _member("Amount_CRS")]
                }
            },
            "dc/topic/Nested": {
                "relevantVariable": {
                    "nodes": [_member("dc/svpg/Debt", "StatVarPeerGroup"), _member("GDP")]
                }
            },
            "dc/svpg/Debt": {"member": {"nodes": [_member("Amount_CRS"), _member("GDP")]}},
        },
    )


def test_topic_table_holds_each_topics_walk_leaves(capsys: pytest.CaptureFixture[str]) -> None:
    tables = _build(_topic_graph(), settings=EnumerationSettings(concurrency=1))

    # Every leaf is kept whatever its admission and with no member cap, in
    # walk order and without repeats; Missing has no arcs and is left out.
    assert tables.topics == {
        "dc/topic/Population": ("Count_Person",),
        "dc/topic/SdgOnly": ("sdg/SH_DTH",),
        "dc/topic/TooBig": ("Count_Person", "GDP", "Amount_CRS"),
        "dc/topic/Nested": ("GDP", "Amount_CRS"),
    }
    assert _stats(capsys)["topic_roots"] == 5


def test_walk_drops_a_root_over_the_cap_and_keeps_the_others() -> None:
    walked = walk_topics(
        ["dc/topic/TooBig", "dc/topic/Nested", "dc/topic/Nested", "dc/topic/Missing"],
        graph=_topic_graph(),
        max_members=2,
    )

    assert walked == {"dc/topic/Nested": ["GDP", "Amount_CRS"], "dc/topic/Missing": []}


def test_walk_with_admit_skips_rejected_leaves_before_the_cap() -> None:
    walked = walk_topics(
        ["dc/topic/TooBig", "dc/topic/Nested"],
        graph=_topic_graph(),
        max_members=2,
        admit=lambda dcid: dcid != "Amount_CRS",
    )

    # TooBig keeps two of its three leaves, within the cap once Amount_CRS
    # is rejected; the rejected leaf is dropped from Nested too.
    assert walked == {"dc/topic/TooBig": ["Count_Person", "GDP"], "dc/topic/Nested": ["GDP"]}
