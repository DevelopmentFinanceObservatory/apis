"""Tests the dependency fingerprint a materialisation produces (`materialise.dependencies`):
what `build_dependencies` records, how `dependency_changes` compares a stored fingerprint with a
current one, and that a run reads each dispatched provenance's refresh date from a precomputed
probe or, lacking one, from the graph, giving the same fingerprint either way."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_group
from materialise.content_version import recipe_dependency_fingerprint
from materialise.convert import ConversionRecord
from materialise.dependencies import (
    Dependencies,
    build_dependencies,
    decode_dependencies,
    dependency_changes,
    members_digest,
    refreshed_dependencies,
)
from materialise.facets import FacetBinding, ProvenanceRefresh, probe_collection_facets
from materialise.graph import Facet
from materialise.recipe_bodies import RECIPE_BODIES
from materialise.window import LatestYear, RequestedWindow, election_window
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from .conftest import FakeGraphReader

_STAT_VAR = "dcid:MalariaGrantDisbursements"
_BARE_STAT_VAR = "MalariaGrantDisbursements"
_RECIPE_ID = next(iter(sorted(RECIPE_BODIES)))

_LATEST_A = LatestYear(
    elected=2024, served=2023, facet_ids=("facet1",), omitted=(("recipient", "country/KEN"),)
)
_LATEST_YEARS = (("dcid:A", _LATEST_A),)

_BASE = Dependencies(
    members=members_digest(["A", "B"]),
    sources=(("provenance/OECDDAC2A", "2026-10-07"), ("provenance/IMF", None)),
    conversion_sources=(("provenance/ONEConversionTablesDAC", "2026-09-30"),),
    recipes=((_RECIPE_ID, "sha256:abc"),),
    window=(2015, 2026),
)


def _stored(**changes: Any) -> dict[str, object]:
    return replace(_BASE, **changes).as_json()


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(_BASE.as_json(), (), id="identical"),
        pytest.param(None, ("schema",), id="no-stored-fingerprint"),
        pytest.param({**_BASE.as_json(), "schema": 2}, ("schema",), id="unknown-schema"),
        pytest.param({"schema": 1, "members": "x"}, ("schema",), id="malformed"),
        pytest.param(_stored(members=members_digest(["A"])), ("members",), id="members"),
        pytest.param(
            _stored(sources=(("provenance/OECDDAC2A", "2026-10-01"), ("provenance/IMF", None))),
            ("sources:provenance/OECDDAC2A",),
            id="source-token",
        ),
        pytest.param(
            _stored(sources=(("provenance/OECDDAC2A", "2026-10-07"),)),
            ("sources:provenance/IMF",),
            id="source-missing-from-stored",
        ),
        pytest.param(
            _stored(sources=(*_BASE.sources, ("provenance/WB", "2026-01-01"))),
            ("sources:provenance/WB",),
            id="source-missing-from-current",
        ),
        pytest.param(
            _stored(
                sources=(("provenance/OECDDAC2A", "2026-10-07"), ("provenance/IMF", "2026-01-01"))
            ),
            (),
            id="source-date-to-none",
        ),
        pytest.param(
            _stored(conversion_sources=(("provenance/ONEConversionTablesDAC", None),)),
            ("conversion_sources:provenance/ONEConversionTablesDAC",),
            id="conversion-source-token",
        ),
        pytest.param(
            _stored(recipes=((_RECIPE_ID, "sha256:old"),)),
            (f"recipes:{_RECIPE_ID}",),
            id="recipe-digest",
        ),
        pytest.param(_stored(window=(2015, 2025)), ("window",), id="window"),
        pytest.param(_stored(window=None), ("window",), id="window-missing"),
        pytest.param(
            _stored(members=members_digest(["C"]), window=(2014, 2026)),
            ("members", "window"),
            id="two-members",
        ),
    ],
)
def test_dependency_changes_names_each_differing_member(stored, expected):
    assert dependency_changes(stored, _BASE) == expected


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(_stored(latest_years=_LATEST_YEARS), (), id="identical"),
        pytest.param(
            _stored(latest_years=(("dcid:A", replace(_LATEST_A, elected=2025)),)),
            ("latest_years",),
            id="elected-year",
        ),
        pytest.param(_BASE.as_json(), ("latest_years",), id="no-recorded-year"),
        pytest.param(
            {**_BASE.as_json(), "latest_years": [{"stat_var": "dcid:A", "elected": "2024"}]},
            ("schema",),
            id="malformed",
        ),
    ],
)
def test_dependency_changes_compares_the_years_a_latest_run_recorded(stored, expected):
    assert dependency_changes(stored, replace(_BASE, latest_years=_LATEST_YEARS)) == expected


def test_build_dependencies_records_each_latest_member_and_the_stored_shape_reads_back():
    """A `latest` member the run holds no year for records that it elected none, so a check
    of the same requests names the same members."""
    latest = RequestedWindow(kind="latest")

    dependencies = build_dependencies(
        [_request("dcid:B", latest), _request("dcid:A", latest)],
        source_refs=(),
        provenance_refresh={},
        conversion_records=(),
        latest_years={"dcid:A": _LATEST_A},
    )

    assert dependencies.latest_years == (
        ("dcid:A", _LATEST_A),
        ("dcid:B", LatestYear(elected=None, served=None)),
    )
    assert dependencies.as_json()["latest_years"] == [
        {
            "stat_var": "dcid:A",
            "elected": 2024,
            "served": 2023,
            "facet_ids": ["facet1"],
            "omitted": [{"role": "recipient", "entity": "country/KEN"}],
        },
        {"stat_var": "dcid:B", "elected": None, "served": None, "facet_ids": [], "omitted": []},
    ]
    assert decode_dependencies(dependencies.as_json()) == dependencies


def test_a_refreshed_fingerprint_reads_every_stored_ref_in_one_call():
    """The freshness side keeps the stored refs and recipe ids, dates every source and
    conversion source from one `node_arcs_many`, and takes members and window from the
    requests."""
    graph = FakeGraphReader(
        arcs={
            "provenance/OECDDAC2A": {"lastDataRefreshDate": ("2026-10-10",)},
            "provenance/ONEConversionTablesDAC": {"lastDataRefreshDate": ("2026-09-30",)},
        }
    )
    window = election_window(RequestedWindow(kind="range", start_year=2015), clock_year=2026)

    refreshed = refreshed_dependencies(
        _BASE, [_request("A", window), _request("B", window)], graph=graph, coverage=None
    )

    assert graph.node_arcs_many_calls == [
        (
            ("provenance/IMF", "provenance/OECDDAC2A", "provenance/ONEConversionTablesDAC"),
            ("lastDataRefreshDate",),
        )
    ]
    assert refreshed == replace(
        _BASE,
        sources=(("provenance/OECDDAC2A", "2026-10-10"), ("provenance/IMF", None)),
        recipes=(recipe_dependency_fingerprint(_RECIPE_ID),),
    )
    assert dependency_changes(_BASE.as_json(), refreshed) == (
        "recipes:" + _RECIPE_ID,
        "sources:provenance/OECDDAC2A",
    )


def test_a_source_with_no_date_on_both_sides_is_unchanged():
    current = replace(_BASE, sources=(("provenance/IMF", None),))
    assert dependency_changes(current.as_json(), current) == ()


def _record(*, recipe_id: str, provenance: str, refresh: str | None) -> ConversionRecord:
    return ConversionRecord(
        recipe_id=recipe_id,
        exchange_rate_table_version="sha256:rates",
        deflator_table_version=None,
        native_currency="USD",
        native_unit_ref="USDollar",
        table_provenance_id=provenance,
        table_refresh_date=refresh,
        fallback_count=0,
        fallback_examples=(),
        projected_count=0,
        projected_examples=(),
    )


def _request(stat_var: str, window: RequestedWindow) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={},
        declared_roles=("observationAbout",),
        facet_bindings={},
        window=window,
    )


def test_build_dependencies_records_what_the_run_dispatched():
    open_window = election_window(RequestedWindow(kind="range", start_year=2015), clock_year=2026)
    record = _record(recipe_id=_RECIPE_ID, provenance="provenance/Tables", refresh="2026-09-30")

    dependencies = build_dependencies(
        [_request("dcid:B", open_window), _request("A", open_window)],
        source_refs=["provenance/X", "provenance/X", "provenance/W"],
        provenance_refresh={
            "provenance/X": ProvenanceRefresh("2026-10-07"),
            "provenance/W": ProvenanceRefresh(None),
        },
        conversion_records=[record, None, record],
        latest_years={},
    )

    assert dependencies.members == members_digest(["A", "B"])
    assert dependencies.sources == (("provenance/W", None), ("provenance/X", "2026-10-07"))
    assert dependencies.conversion_sources == (("provenance/Tables", "2026-09-30"),)
    assert dependencies.recipes == (recipe_dependency_fingerprint(_RECIPE_ID),)
    assert dependencies.window == (2015, 2026)
    assert dependencies.as_json() == {
        "schema": 1,
        "members": members_digest(["A", "B"]),
        "sources": [
            {"ref": "provenance/W", "token": None},
            {"ref": "provenance/X", "token": "2026-10-07"},
        ],
        "conversion_sources": [{"ref": "provenance/Tables", "token": "2026-09-30"}],
        "recipes": [{"id": _RECIPE_ID, "digest": recipe_dependency_fingerprint(_RECIPE_ID)[1]}],
        "window": {"start_year": 2015, "end_year": 2026},
    }


@pytest.mark.parametrize(
    "window",
    [
        RequestedWindow(kind="range", start_year=2015, end_year=2020),
        RequestedWindow(kind="all"),
        RequestedWindow(kind="latest"),
    ],
)
def test_a_window_that_reads_no_clock_has_no_window_member(window):
    dependencies = build_dependencies(
        [_request(_STAT_VAR, window)],
        source_refs=(),
        provenance_refresh={},
        conversion_records=(),
        latest_years={},
    )
    assert dependencies.window is None


def test_members_digest_ignores_order_prefix_and_repeats():
    assert members_digest(["dcid:B", "A", "B"]) == members_digest(["A", "B"])
    assert members_digest(["A", "B"]).startswith("sha256:")
    assert members_digest(["A"]) != members_digest(["A", "B"])


def test_a_recipe_without_a_body_records_no_digest():
    dependencies = build_dependencies(
        [_request(_STAT_VAR, RequestedWindow(kind="all"))],
        source_refs=(),
        provenance_refresh={},
        conversion_records=[_record(recipe_id="NoSuchRecipe", provenance="p", refresh=None)],
        latest_years={},
    )
    assert dependencies.recipes == (("NoSuchRecipe", None),)


_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=2,
    earliest_date="2015",
    latest_date="2016",
)

# A second member, so a run of both takes the group path (`generation._generate_group`).
_SECOND_STAT_VAR = "dcid:MalariaGrantCommitments"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _csv(request: httpx.Request) -> str:
    """One observation for each StatVar the SDMX request names."""
    named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
    return _CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},country/TGO,USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
        for stat_var in (_STAT_VAR, _SECOND_STAT_VAR)
        if stat_var in named or stat_var.removeprefix("dcid:") in named
    )


def _materialise(*, with_probe: bool, refreshed_during_fetch: str | None = None, members: int = 1):
    """Materialise `members` members (one or two) against a fake graph whose `OECD_DAC2A`
    refresh date reads 2026-10-07, moving to `refreshed_during_fetch` on the first SDMX request
    when given. Returns the result, the refresh reads the run made, the graph and the
    requests."""
    stat_vars = (_STAT_VAR, _SECOND_STAT_VAR)[:members]
    bare = [stat_var.removeprefix("dcid:") for stat_var in stat_vars]
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "OECD_DAC2A": {"lastDataRefreshDate": ("2026-10-07",)}
    }

    def _respond(request: httpx.Request) -> httpx.Response:
        if refreshed_during_fetch is not None:
            arcs["OECD_DAC2A"] = {"lastDataRefreshDate": (refreshed_during_fetch,)}
        return httpx.Response(
            200,
            text=_csv(request),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(_respond)
    )
    graph = FakeGraphReader(
        arcs=arcs, facets={(stat_var, "country/TGO"): [_FACET] for stat_var in bare}
    )
    coverage = build_coverage_index(
        [(stat_var, "facet1", "country/DONOR", "country/TGO", [2015]) for stat_var in bare],
        role_names={stat_var: RoleNames(donor="donor", recipient="recipient") for stat_var in bare},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    requests = [
        MaterialisationRequest(
            stat_var=stat_var,
            role_bindings={"recipient": ["country/TGO"]},
            declared_roles=("recipient",),
            facet_bindings={
                "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
                "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
                "unit": FacetBinding(state="bound", ref="USD"),
            },
            window=RequestedWindow(kind="all"),
        )
        for stat_var in stat_vars
    ]
    probe = (
        probe_collection_facets(
            requests,
            graph=graph,
            availability=sdmx,
            settle=True,
            coverage=coverage,
        )
        if with_probe
        else None
    )
    reads_before = len(graph.node_arcs_many_calls)
    result = materialise_group(
        requests,
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-1",
        capability_version="cap-1",
        coverage=coverage,
        precomputed_probe=probe,
    )
    refresh_reads = [
        dcids
        for dcids, properties in graph.node_arcs_many_calls[reads_before:]
        if "lastDataRefreshDate" in properties
    ]
    return result, refresh_reads, graph, requests


def test_a_run_takes_source_tokens_from_its_probe_and_reads_them_without_one():
    probed, probed_reads, _, _ = _materialise(with_probe=True)
    unprobed, unprobed_reads, _, _ = _materialise(with_probe=False)

    assert probed.dependencies.sources == (("OECD_DAC2A", "2026-10-07"),)
    assert probed.dependencies == unprobed.dependencies
    assert probed_reads == []
    assert unprobed_reads == [("OECD_DAC2A",)]


@pytest.mark.parametrize("members", [1, 2], ids=["member", "group"])
@pytest.mark.parametrize("with_probe", [True, False], ids=["probed", "unprobed"])
def test_a_refresh_during_the_fetch_records_the_pre_fetch_token_and_the_next_check_sees_it(
    with_probe, members
):
    """A run records the refresh date it read before its first SDMX request, on the member
    path and the group path alike, so a source refresh that lands while the rows stream is
    caught by the next freshness check."""
    result, _, graph, requests = _materialise(
        with_probe=with_probe, refreshed_during_fetch="2026-10-11", members=members
    )

    assert result.coverage.rows == members
    assert result.dependencies.sources == (("OECD_DAC2A", "2026-10-07"),)
    current = refreshed_dependencies(result.dependencies, requests, graph=graph, coverage=None)
    assert dependency_changes(result.dependencies.as_json(), current) == ("sources:OECD_DAC2A",)
