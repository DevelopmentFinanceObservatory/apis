"""Guards parity between `dc_graph.GraphClient`'s dated read (Search's
`observation_facets_batch` with dates) and its undated read
(`variable_facets_batch`, used by materialise): for the same raw
/v2/observation response body, both fold repeated orderedFacets blocks into
one Facet per distinct facetId, in the same
(facet_id, obs_count, earliest_date, latest_date) order.
"""

from __future__ import annotations

import httpx

from dc_graph import GraphClient
from dc_graph.client import _parse_ordered_facets

_STAT_VAR = "Amount_SyntheticFlow_Test"
_ENTITY = "country/ZZZ"

_ORDERED_FACETS = [
    {
        "facetId": "A",
        "obsCount": 1,
        "earliestDate": "2021",
        "latestDate": "2021",
        "observations": [{"date": "2021", "value": 30.0}],
    },
    {
        "facetId": "A",
        "obsCount": 2,
        "earliestDate": "2019",
        "latestDate": "2020",
        "observations": [
            {"date": "2019", "value": 10.0},
            {"date": "2020", "value": 20.0},
        ],
    },
    {
        "facetId": "B",
        "obsCount": 1,
        "earliestDate": "2018",
        "latestDate": "2018",
        "observations": [{"date": "2018", "value": 9.0}],
    },
    {
        "obsCount": 1,
        "earliestDate": "2022",
        "latestDate": "2022",
        "observations": [{"date": "2022", "value": 4.0}],
    },
]

_TOP_FACETS_MAP = {
    "A": {"provenanceId": "prov/A", "measurementMethod": "Method", "unit": "USD"},
    "B": {"provenanceId": "prov/B", "measurementMethod": "Method", "unit": "EUR"},
}

_BODY = {
    "byVariable": {_STAT_VAR: {"byEntity": {_ENTITY: {"orderedFacets": _ORDERED_FACETS}}}},
    "facets": _TOP_FACETS_MAP,
}


def _materialise_reader() -> GraphClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_BODY)

    return GraphClient(
        base_url="https://graph.example.org/core/api",
        transport=httpx.MockTransport(handler),
    )


def test_duplicate_facet_id_merge_matches_between_search_and_materialise():
    search_facets = _materialise_reader().observation_facets_batch(
        [_STAT_VAR], [_ENTITY], needs_dates=True
    )[_STAT_VAR][_ENTITY]
    materialise_facets = _materialise_reader().variable_facets_batch(
        stat_var=_STAT_VAR, entities=[_ENTITY]
    )[_ENTITY]

    search_summary = [
        (f.facet_id, f.obs_count, f.earliest_date, f.latest_date) for f in search_facets
    ]
    materialise_summary = [
        (f.facet_id, f.obs_count, f.earliest_date, f.latest_date) for f in materialise_facets
    ]
    expected = [
        ("A", 3, "2019", "2021"),
        ("B", 1, "2018", "2018"),
        (None, 1, "2022", "2022"),
    ]
    assert search_summary == expected
    assert materialise_summary == expected


def test_merged_dates_and_values_stay_index_aligned_in_block_order():
    """dates/values concatenate block by block, never re-sorted: block one's
    "2021" precedes block two's "2019"/"2020" in the merged lists."""
    search_facets = _parse_ordered_facets(_ORDERED_FACETS, _TOP_FACETS_MAP, with_dates=True)
    merged_a = next(f for f in search_facets if f.facet_id == "A")

    assert len(merged_a.dates) == len(merged_a.values) == merged_a.obs_count == 3
    assert merged_a.dates == ("2021", "2019", "2020")
    assert merged_a.values == (30.0, 10.0, 20.0)
