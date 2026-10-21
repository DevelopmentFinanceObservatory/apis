"""A small catalogue over the three origins, and a helper that publishes it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_publication.uri import join_uri, read_uri_bytes_and_generation
from origin_catalogue import BASE, DFO, UN_SYSTEM, CatalogueTables, ScanLevel
from origin_catalogue.arc_tables import ArcTables, ArcTablesBuilder, WireArcs
from origin_catalogue.artifact import EncodedBuild, encode_build, publish_build
from origin_catalogue.catalogue import (
    BaseVariable,
    CardFacts,
    CatalogueSnapshot,
    DatasetRef,
    Licence,
    ProvenanceRow,
    SourceDocumentation,
)
from origin_catalogue.families import FamilyKey, FamilySummary, ReleaseFamilies
from origin_catalogue.graph import _wire
from origin_catalogue.schema import FORMAT_ROOT, UnAgencyCounts
from origin_catalogue.un_mirror import UnArc, UnMirror, UnVariable
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, provenance_row


def named_labels(*tables: Mapping[str, tuple[UnArc, ...]]) -> dict[str, str | None]:
    """Each node an arc of `tables` names, labelled with the name its arc carries."""
    labels: dict[str, str | None] = {}
    for table in tables:
        for rows in table.values():
            for arc in rows:
                if arc.dcid is not None and labels.get(arc.dcid) is None:
                    labels[arc.dcid] = arc.name
    return labels


def build_labelled(
    builder: ArcTablesBuilder, *, labels: Mapping[str, str | None] | None = None
) -> ArcTables:
    """`builder`'s tables, each target labelled from `labels` or else with the name its arc
    carries."""
    named = named_labels(*(builder.table(name) for name in ("dfo", "base", "schema")))
    return builder.build(labels={**named, **(labels or {})})


def build_arc_tables(
    *,
    dfo: Mapping[str, WireArcs | None],
    base: Mapping[str, WireArcs | None],
    schema: Mapping[str, WireArcs | None],
    labels: Mapping[str, str | None] | None = None,
) -> ArcTables:
    """The arc tables of `->*` answers keyed by node, through `ArcTablesBuilder`, labelled
    as `build_labelled` labels them."""
    builder = ArcTablesBuilder()
    for name, answers in (("dfo", dfo), ("base", base), ("schema", schema)):
        for node, arcs in answers.items():
            builder.add(name, node, arcs)
    return build_labelled(builder, labels=labels)


def listing_base_variables(
    base: Mapping[str, WireArcs], *, provenances: Mapping[str, ProvenanceRow] | None = None
) -> FakeCatalogue:
    """A `FakeCatalogue` listing `provenances` whose base arc table lists each variable of
    `base` with its `->*` arcs, as a catalogue build lists every base variable. A family the
    bound release holds as DFO that also holds one of them is then walked on the graph whole
    (`graph_release.origin.ReleaseOrigin.holds_family`)."""
    return FakeCatalogue(
        provenances=provenances or {}, arcs=build_arc_tables(dfo={}, base=base, schema={})
    )


class JobTables:
    """`snapshot` answering each node's arcs and each family from its own tables, its DFO
    table included, as the catalogue job wrote them, in place of the bound release's DFO
    half, and everything else as it does. For a test whose subject is the job's DFO half."""

    def __init__(self, snapshot: CatalogueSnapshot) -> None:
        self._snapshot = snapshot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._snapshot, name)

    def held_arcs(self, dcid: str) -> tuple[UnArc, ...] | None:
        return self._snapshot.arcs.arcs(dcid)

    def held_family(self, key: FamilyKey) -> FamilySummary | None:
        # The build's own joined DFO and base summaries stand in for the release's part.
        families = self._snapshot.families
        return families.family(
            key,
            un_system_enabled=self._snapshot.un_system_enabled,
            release=ReleaseFamilies(dfo=families.arcs, base={}),
        )


def held_wire_arcs(tables: ArcTables, dcid: str) -> dict | None:
    """`dcid`'s held arcs as the origin client serves them, `None` for a node no table holds."""
    arcs = tables.arcs(dcid)
    return None if arcs is None else _wire(arcs)


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
    UN, whose front door refuses two places, the licences of CRS and WDI, and the
    card facts and documentation facts of the three non-UN entries."""
    return CatalogueTables(
        provenances={
            row.id: row
            for row in (
                replace(
                    provenance_row(
                        CRS,
                        DFO,
                        source="dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd",
                    ),
                    licence=Licence(
                        type="OECDTermsAndConditions",
                        label="OECD Terms and Conditions",
                        url="https://www.oecd.org/en/about/terms-conditions.html",
                    ),
                ),
                replace(
                    provenance_row(WDI, BASE, source="dc/s/WorldBank"),
                    dataset=DatasetRef(ref=WDI_DATASET, label="World Development Indicators"),
                    licence=Licence(
                        type="CreativeCommonsAttributionLicense",
                        label="Creative Commons Attribution License",
                    ),
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
        source_documentation={
            CRS: SourceDocumentation(
                provenances=(CRS,),
                publisher="OECD",
                source_data_url="https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet",
                citation="OECD (2026), Creditor Reporting System",
                license_url="https://www.oecd.org/en/about/terms-conditions.html",
                last_update="2026-10-02",
                next_expected_update="2027-01-02",
                documents={"dataNotes": ("- Figures for the latest year are preliminary.\\n",)},
            ),
            WDI_DATASET: SourceDocumentation(provenances=(WDI,), publisher="World Bank"),
            EUROSTAT_GDP: SourceDocumentation(provenances=(EUROSTAT_GDP,)),
        },
        un_blocked_places=frozenset({"country/GRC", "country/ESH"}),
    )


UN_SERIES = "undata/who/SERIES-MORT_U5_RATE"
UN_VARIABLE = "undata/who/MORT_U5_RATE"
UN_VARIABLE_FEMALE = "undata/who/MORT_U5_RATE.SEX--F"
UN_EXCLUDED_SERIES = "undata/who/SERIES-ODA_1_3"
UN_EXCLUDED_VARIABLE = "undata/who/ODA_1_3"
UN_TOPIC = "undata/topic/who/MORT_U5_RATE"


def sample_un_mirror() -> UnMirror:
    """WHO's under-five mortality series with a by-sex variable, one excluded
    aid series and one topic. The two variables share their series, type and
    measure arcs."""
    series = UnArc(
        property="populationType",
        dcid=UN_SERIES,
        name="Under-five mortality rate",
        types=("undata/who/SERIES",),
    )
    measure = UnArc(property="measuredProperty", dcid="value", name="value", types=("Property",))
    return UnMirror(
        variables={
            UN_VARIABLE: UnVariable(
                series=UN_SERIES,
                arcs=(measure, UnArc(property="name", value="Under-five mortality rate"), series),
            ),
            UN_VARIABLE_FEMALE: UnVariable(
                series=UN_SERIES,
                arcs=(
                    UnArc(
                        property="constraintProperties",
                        dcid="undataSex",
                        name="Sex",
                        types=("Property",),
                    ),
                    measure,
                    UnArc(property="name", value="Under-five mortality rate [Sex=Female]"),
                    series,
                    UnArc(
                        property="undataSex",
                        dcid="undata/SEX-F",
                        name="Female",
                        types=("undata/SEX",),
                    ),
                ),
            ),
        },
        excluded_variables={UN_EXCLUDED_VARIABLE: UN_EXCLUDED_SERIES},
        topics={UN_TOPIC: (UN_VARIABLE, UN_VARIABLE_FEMALE)},
        exclusions={UN_EXCLUDED_SERIES: CRS},
        excluded_series_names={UN_EXCLUDED_SERIES: "Net ODA to health"},
        agencies={
            "who": UnAgencyCounts(series=2, variables=3, excluded_series=1, excluded_variables=1)
        },
    )


HOUSEHOLDS = "Count_Household"
RECIPIENT = "financingRecipient"


def sample_arc_tables() -> ArcTables:
    """The CRS variable's arcs as the graph answers them, two provenances
    repeating its `typeOf`, with an import-added constraint property and a
    denominator no variable list holds; GDP's name; the population variable
    held empty; and the schema: the denominator, the recipient role with its
    range and the range's superclass chain."""
    statvar = {"dcid": "StatisticalVariable", "name": "StatisticalVariable", "types": ["Class"]}
    return build_arc_tables(
        dfo={
            CRS_VARIABLE: {
                "typeOf": {
                    "nodes": [
                        {**statvar, "provenanceId": "dc/base/CRS"},
                        {**statvar, "provenanceId": "dc/base/CRS_redeclared"},
                    ]
                },
                "constraintProperties": {
                    "nodes": [
                        {"dcid": RECIPIENT, "name": RECIPIENT, "types": ["Property"]},
                        {"dcid": "sourceIndicatorCode", "types": ["Property"]},
                    ]
                },
                "name": {"nodes": [{"value": "Gross ODA disbursements"}]},
                "measurementDenominator": {
                    "nodes": [{"dcid": HOUSEHOLDS, "types": ["StatisticalVariable"]}]
                },
            }
        },
        base={
            GDP: {
                "name": {"nodes": [{"value": "Gross domestic product"}]},
                "typeOf": {"nodes": [statvar]},
            },
            POPULATION: None,
        },
        schema={
            HOUSEHOLDS: {"typeOf": {"nodes": [statvar]}},
            RECIPIENT: {"rangeIncludes": {"nodes": [{"dcid": "Country", "types": ["Class"]}]}},
            "Country": {"subClassOf": {"nodes": [{"dcid": "Place", "types": ["Class"]}]}},
            "Place": {},
        },
        # The graph's arcs name the nameless role by its dcid; `node_label` gives it none.
        labels={RECIPIENT: None},
    )


def encode_sample(*, build_id: str = "build-a", **overrides: object) -> EncodedBuild:
    """The sample tables encoded as a build, holding `sample_arc_tables` unless `overrides`
    names its own `arcs`."""
    fields: dict = {
        "build_id": build_id,
        "built_at": BUILT_AT,
        "build_seconds": 12.5,
        "identity_inputs": {"registry": "r1", "dfo": "f1", "week": "2026-W40"},
        "registry_hash": "r1",
        "arcs": sample_arc_tables(),
    }
    fields.update(overrides)
    tables = replace(sample_tables(), superseded=fields.pop("superseded", {}))
    return encode_build(tables, **fields)


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
