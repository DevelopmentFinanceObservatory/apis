"""A small catalogue over the three origins, and a helper that publishes it."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from artifact_publication.uri import join_uri, read_uri_bytes_and_generation
from origin_catalogue import BASE, DFO, UN_SYSTEM, CatalogueTables, ScanLevel
from origin_catalogue.artifact import EncodedBuild, encode_build, publish_build
from origin_catalogue.catalogue import BaseVariable, CardFacts, DatasetRef
from origin_catalogue.schema import FORMAT_ROOT
from tests.origin_catalogue.fake_catalogue import provenance_row

CRS = "provenance/OECDCreditorReportingSystem"
WDI = "dc/base/WorldDevelopmentIndicators"
EUROSTAT_GDP = "dc/base/EurostatData_GDP"
UN_WHO = "undata/p/WHO"

CRS_VARIABLE = "Amount_FinancialFlow_Gross_Disbursement_Nominal_ODA"
POPULATION = "Count_Person"
GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
CENSUS_POVERTY = "Count_Person_BelowPovertyLevelInThePast12Months"

WDI_DATASET = "dc/d/WorldBank_WorldDevelopmentIndicators"

SCAN_PLACES = ("Earth", "country/KEN", "country/USA", "geoId/06", "nuts/DE11")

BUILT_AT = datetime(2026, 9, 28, 6, 0, tzinfo=UTC)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "origin_catalogue"
"""The sample build, published under `FIXTURE_BUILD_ID`, for a surface started outside
the suite, such as the docker job's read-only container. It holds a format root's
contents, which `scripts.stage_origin_catalogue_fixture.stage_fixture` presents as a
catalogue root. `scripts/record_origin_catalogue_fixture.py` rewrites it."""
FIXTURE_BUILD_ID = "sample"


def sample_tables() -> CatalogueTables:
    """DFO's CRS, two base provenances under two sources (WDI under its
    Dataset) that share a variable at different places, a variable only a non-admitted provenance holds, WHO on the
    UN, whose front door refuses two places, and the card facts of the three
    non-UN entries."""
    return CatalogueTables(
        provenances={
            row.id: row
            for row in (
                provenance_row(
                    CRS, DFO, source="dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"
                ),
                replace(
                    provenance_row(WDI, BASE, source="dc/s/WorldBank"),
                    dataset=DatasetRef(ref=WDI_DATASET, label="World Development Indicators"),
                ),
                provenance_row(EUROSTAT_GDP, BASE, source="dc/s/EuropeanUnionEuEurostat"),
                provenance_row(UN_WHO, UN_SYSTEM, segment="who"),
            )
        },
        dfo_variables=frozenset({CRS_VARIABLE, POPULATION}),
        base_variables={
            POPULATION: BaseVariable(provenances=(WDI,), places=0b00111),
            GDP: BaseVariable(provenances=(WDI, EUROSTAT_GDP), places=0b10110),
        },
        # WDI holds both variables at Earth, Kenya and the USA; Eurostat holds
        # GDP at its NUTS region only.
        provenance_places={WDI: 0b00111, EUROSTAT_GDP: 0b10000},
        withheld_variables=frozenset({CENSUS_POVERTY}),
        topics={"dc/topic/Demographics": (POPULATION, "Count_Person_Female")},
        scan_places=SCAN_PLACES,
        scan_levels={
            "dc/s/EuropeanUnionEuEurostat": (ScanLevel(place_type="EurostatNUTS2"),),
            "dc/s/UsBureauOfLaborStatisticsBls": (
                ScanLevel(place_type="State", within="country/USA"),
            ),
        },
        dataset_facts={
            CRS: CardFacts(
                description="Aid activity data reported by DAC members.",
                license_type="Creative Commons Attribution License",
                update_cadence="P1Y",
                years_covered="1973-2024",
                source_url="https://data-explorer.oecd.org",
            ),
            WDI_DATASET: CardFacts(
                license_type="Creative Commons Attribution License",
                update_cadence="P3M",
                years_covered="1960-2024",
                source_url="https://datacatalog.worldbank.org/dataset/world-development-indicators",
            ),
            EUROSTAT_GDP: CardFacts(),
        },
        un_blocked_places=frozenset({"country/GRC", "country/ESH"}),
    )


def encode_sample(*, build_id: str = "build-a", **overrides: object) -> EncodedBuild:
    fields: dict = {
        "build_id": build_id,
        "built_at": BUILT_AT,
        "build_seconds": 12.5,
        "identity_inputs": {"registry": "r1", "dfo": "f1", "week": "2026-W40"},
        "registry_hash": "r1",
    }
    fields.update(overrides)
    return encode_build(sample_tables(), **fields)


def publish_sample(
    base_uri: str, *, build_id: str = "build-a", **overrides: object
) -> EncodedBuild:
    """Publishes the sample build over whatever `current.json` holds."""
    build = encode_sample(build_id=build_id, **overrides)
    current = read_uri_bytes_and_generation(join_uri(base_uri, FORMAT_ROOT, "current.json"))
    assert publish_build(
        base_uri, build, expected_generation=None if current is None else current[1]
    )
    return build
