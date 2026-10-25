"""Each request is built from the returned `output_options` document. Available and partial
formats must work; unavailable options must refuse with the stated code.

The five-row CRS fixture has duplicate `(recipient, time_period)` keys with different
providers, plus `country/UGA` without ISO2 or ISO3 codes. The complete `observationAbout`
fixture covers available pivots and enrichments.
"""

from __future__ import annotations

import httpx

from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=5,
    earliest_date="2015",
    latest_date="2016",
)

_SCOPE_FACETS = {
    "provenance": {"ref": "OECD_DAC2A"},
    "measurementMethod": {"ref": "GrantEquivalent"},
    "unit": {"ref": "USD"},
}

_CRS_STAT_VAR = "MalariaGrantDisbursementsConformanceCrs"
_CRS_STAT_VAR_REF = f"dcid:{_CRS_STAT_VAR}"

_CRS_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,provider,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _crs_csv_row(*, provider: str, recipient: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_CRS_STAT_VAR},{provider},{recipient},USD,"
        f"GrantEquivalent,NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


_CRS_CSV_BODY = (
    _CRS_CSV_HEADER
    + _crs_csv_row(provider="org/DAC1", recipient="country/KEN", year="2015", value="100")
    + _crs_csv_row(provider="org/DAC1", recipient="country/TGO", year="2016", value="200")
    + _crs_csv_row(provider="org/DAC2", recipient="country/UGA", year="2015", value="300")
    + _crs_csv_row(provider="org/DAC2", recipient="country/KEN", year="2016", value="400")
    + _crs_csv_row(provider="org/DAC2", recipient="country/KEN", year="2015", value="150")
)

_CRS_SCOPE = {
    "entities": {
        "provider": ["org/DAC1", "org/DAC2"],
        "recipient": ["country/KEN", "country/TGO", "country/UGA"],
    },
    "facets": _SCOPE_FACETS,
}


def _crs_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _CRS_STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("provider", "recipient"),
                "measuredProperty": ("amount",),
            },
            "country/KEN": {
                "countryAlpha3Code": ("KEN",),
                "countryAlpha2Code": ("KE",),
                "name": ("Kenya",),
            },
            "country/TGO": {
                "countryAlpha3Code": ("TGO",),
                "countryAlpha2Code": ("TG",),
                "name": ("Togo",),
            },
            # No ISO arcs leave `recipient_iso3` and `recipient_iso2` partial.
            "country/UGA": {"name": ("Uganda",)},
            # Provider nodes carry only their type, which the role kind check reads, so their
            # enrichments are omitted.
            "org/DAC1": {"typeOf": ("Organization",)},
            "org/DAC2": {"typeOf": ("Organization",)},
        },
        facets={
            (_CRS_STAT_VAR, "country/KEN"): [_FACET],
            (_CRS_STAT_VAR, "country/TGO"): [_FACET],
            (_CRS_STAT_VAR, "country/UGA"): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )


_OA_STAT_VAR = "MalariaGrantDisbursementsConformanceObservationAbout"
_OA_STAT_VAR_REF = f"dcid:{_OA_STAT_VAR}"
_OA_ENTITIES = ("country/KEN", "country/TGO", "country/UGA")
_OA_YEARS = ("2015", "2016")

_OA_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _oa_csv_row(*, entity: str, year: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{_OA_STAT_VAR_REF},{entity},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,facet1\r\n"
    )


_OA_CSV_BODY = _OA_CSV_HEADER + "".join(
    _oa_csv_row(entity=entity, year=year, value="10")
    for entity in _OA_ENTITIES
    for year in _OA_YEARS
)

_OA_SCOPE = {
    "entities": {"observationAbout": list(_OA_ENTITIES)},
    "facets": _SCOPE_FACETS,
}


def _oa_graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _OA_STAT_VAR: {
                "typeOf": ("StatisticalVariable",),
                # An absent arc defaults the subject role to `observationAbout`.
                "measuredProperty": ("amount",),
            },
            "country/KEN": {
                "countryAlpha3Code": ("KEN",),
                "countryAlpha2Code": ("KE",),
                "name": ("Kenya",),
            },
            "country/TGO": {
                "countryAlpha3Code": ("TGO",),
                "countryAlpha2Code": ("TG",),
                "name": ("Togo",),
            },
            "country/UGA": {
                "countryAlpha3Code": ("UGA",),
                "countryAlpha2Code": ("UG",),
                "name": ("Uganda",),
            },
        },
        facets={(_OA_STAT_VAR, entity): [_FACET] for entity in _OA_ENTITIES},
        labels={"OECD_DAC2A": "OECD DAC2a"},
    )


def _crs_coverage_service() -> FakeCoverageService:
    """Builds coverage for each distinct provider-recipient pair in the five CSV rows."""
    index = build_coverage_index(
        [
            (_CRS_STAT_VAR, "facet1", "org/DAC1", "country/KEN", [2015]),
            (_CRS_STAT_VAR, "facet1", "org/DAC1", "country/TGO", [2016]),
            (_CRS_STAT_VAR, "facet1", "org/DAC2", "country/UGA", [2015]),
            (_CRS_STAT_VAR, "facet1", "org/DAC2", "country/KEN", [2015, 2016]),
        ],
        role_names={_CRS_STAT_VAR: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _client(
    *, graph: FakeMintGraphReader, csv_body: str, coverage: FakeCoverageService | None = None
):
    store = FakeDatasetStore()
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(
            base_url=_BASE_URL,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text=csv_body,
                    headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
                )
            ),
        ),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage,
    )


def _mint_body(
    *,
    statvar_ref: str,
    scope: dict,
    format: str | None = None,
    pivot: dict | None = None,
    columns: list[str] | None = None,
) -> dict:
    options: dict = {"bundle": False}
    if format is not None:
        options["format"] = format
    if pivot is not None:
        options["pivot"] = pivot
    if columns is not None:
        options["columns"] = columns
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": [statvar_ref], "scope": scope},
        "options": options,
    }


def _expected_pivot_refusal(entry: dict, format_name: str) -> str | None:
    """Returns the refusal code promised by `entry` for this format, or `None` if available."""
    if entry["state"] == "unavailable":
        return entry["reason"]
    for unavailable in entry.get("unavailable_formats") or []:
        if unavailable["format"] == format_name:
            return unavailable["reason"]
    return None


def _assert_pivots_conform(client, document: dict, *, statvar_ref: str, scope: dict) -> None:
    format_names = [f["name"] for f in document["formats"]]
    for entry in document["pivots"]:
        for format_name in format_names:
            resp = client.post(
                "/api/datasets",
                json=_mint_body(
                    statvar_ref=statvar_ref,
                    scope=scope,
                    format=format_name,
                    pivot={"index": entry["index"], "columns": entry["columns"]},
                ),
            )
            expected_refusal = _expected_pivot_refusal(entry, format_name)
            if expected_refusal is None:
                assert resp.status_code == 200, (
                    f"output_options marked pivot index={entry['index']!r} "
                    f"columns={entry['columns']!r} format={format_name!r} as "
                    f"{entry['state']!r}, but the door refused: {resp.status_code} {resp.text}"
                )
            else:
                assert resp.status_code != 200, (
                    f"output_options marked pivot index={entry['index']!r} "
                    f"columns={entry['columns']!r} format={format_name!r} as refused "
                    f"({expected_refusal!r}), but the door served it: {resp.text}"
                )
                assert resp.json()["code"] == expected_refusal, (
                    f"pivot index={entry['index']!r} columns={entry['columns']!r} "
                    f"format={format_name!r}: output_options said {expected_refusal!r}, "
                    f"the door refused {resp.json()['code']!r}"
                )


def _assert_enrichments_conform(client, document: dict, *, statvar_ref: str, scope: dict) -> None:
    frame_columns = [c["name"] for c in document["columns"]]
    for entry in document["enrichments"]:
        token = entry["token"]
        resp = client.post(
            "/api/datasets",
            json=_mint_body(statvar_ref=statvar_ref, scope=scope, columns=[*frame_columns, token]),
        )
        assert resp.status_code == 200, (
            f"output_options marked enrichment {token!r} as {entry['state']!r}, but "
            f"requesting it refused: {resp.status_code} {resp.text}"
        )
        manifest = resp.json()
        unavailable_warning_present = any(
            w["code"] == "enrichment_unavailable" and token in w["message"]
            for w in manifest["warnings"]
        )
        if entry["state"] == "partial":
            assert unavailable_warning_present, (
                f"output_options marked enrichment {token!r} partial, but the mint "
                "carried no matching enrichment_unavailable warning"
            )
        else:
            assert entry["state"] == "available"
            assert not unavailable_warning_present, (
                f"output_options marked enrichment {token!r} available, but the mint "
                "carried an enrichment_unavailable warning for it anyway"
            )


def _assert_formats_conform(client, document: dict, *, statvar_ref: str, scope: dict) -> None:
    for entry in document["formats"]:
        resp = client.post(
            "/api/datasets",
            json=_mint_body(statvar_ref=statvar_ref, scope=scope, format=entry["name"]),
        )
        if entry["state"] == "available":
            assert resp.status_code == 200, (
                f"output_options marked format {entry['name']!r} available, but the "
                f"door refused: {resp.status_code} {resp.text}"
            )
        else:
            assert resp.status_code != 200, (
                f"output_options marked format {entry['name']!r} as refused "
                f"({entry['reason']!r}), but the door served it: {resp.text}"
            )
            assert resp.json()["code"] == entry["reason"]


def test_the_crs_two_role_fixtures_output_options_conforms_to_the_doors_own_behaviour():
    client = _client(graph=_crs_graph(), csv_body=_CRS_CSV_BODY, coverage=_crs_coverage_service())
    baseline = client.post(
        "/api/datasets", json=_mint_body(statvar_ref=_CRS_STAT_VAR_REF, scope=_CRS_SCOPE)
    )
    assert baseline.status_code == 200, baseline.text
    document = baseline.json()["output_options"]

    # UGA lacks codes; the fifth row duplicates a recipient-year key.
    assert any(e["state"] == "partial" for e in document["enrichments"])
    assert any(p["state"] == "unavailable" for p in document["pivots"])

    _assert_pivots_conform(client, document, statvar_ref=_CRS_STAT_VAR_REF, scope=_CRS_SCOPE)
    _assert_enrichments_conform(client, document, statvar_ref=_CRS_STAT_VAR_REF, scope=_CRS_SCOPE)
    _assert_formats_conform(client, document, statvar_ref=_CRS_STAT_VAR_REF, scope=_CRS_SCOPE)


def test_the_observation_about_single_role_fixtures_output_options_conforms_to_the_doors_own_behaviour():
    client = _client(graph=_oa_graph(), csv_body=_OA_CSV_BODY)
    baseline = client.post(
        "/api/datasets", json=_mint_body(statvar_ref=_OA_STAT_VAR_REF, scope=_OA_SCOPE)
    )
    assert baseline.status_code == 200, baseline.text
    document = baseline.json()["output_options"]

    assert document["pivots"]
    assert all(p["state"] == "available" for p in document["pivots"])
    assert document["enrichments"]
    assert all(e["state"] == "available" for e in document["enrichments"])

    _assert_pivots_conform(client, document, statvar_ref=_OA_STAT_VAR_REF, scope=_OA_SCOPE)
    _assert_enrichments_conform(client, document, statvar_ref=_OA_STAT_VAR_REF, scope=_OA_SCOPE)
    _assert_formats_conform(client, document, statvar_ref=_OA_STAT_VAR_REF, scope=_OA_SCOPE)
