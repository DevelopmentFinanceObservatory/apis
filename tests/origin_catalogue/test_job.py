"""Pins the catalogue job with fakes for BigQuery, both graphs and the base-table builder."""

from __future__ import annotations

import ast
import json
import time
import types
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pytest

from artifact_publication.jobs import LEASE_FILE, Ingestion, acquire_lease
from dc_graph import UNRESOLVED_NODE, GraphClient, GraphReadError, SubjectPage
from origin_catalogue import (
    BASE,
    DFO,
    UN_SYSTEM,
    BaseTables,
    BaseVariable,
    CardFacts,
    DatasetRef,
    ScanLevel,
    artifact,
    node_reads,
)
from origin_catalogue import job as catalogue_job
from origin_catalogue.artifact import load_catalogue, read_un_mirror
from origin_catalogue.catalogue import Licence, SourceDocumentation
from origin_catalogue.enumerate import EnumerationError
from origin_catalogue.graph import UN_BASE_URL
from origin_catalogue.registry import parse_registry
from origin_catalogue.schema import FORMAT_ROOT, CataloguePointer, IndexPresence, UnAgencyCounts
from origin_catalogue.successors import Successors
from origin_catalogue.un_mirror import UnMirror
from tests.artifact_publication.support import FakeBigQuery, history_row, import_key_responses
from tests.origin_catalogue.fake_catalogue import provenance_row, spread_provenance_places
from tests.origin_catalogue.support import held_wire_arcs
from tests.origin_catalogue.un_fake import (
    MORT,
    MORT_BY_SEX,
    MORT_FEMALE,
    MORT_MALE,
    ODA,
    ODA_SERIES,
    FakeUn,
)
from tests.origin_catalogue.un_fake import target as un_target

_CONNECTION = "proj.region.conn"

WORLD_BANK = "dc/s/WorldBank"
OECD = "dc/s/OrganisationForEconomicCo-operationAndDevelopmentOecd"
WDI = "dc/base/WorldDevelopmentIndicators"
WB_IDS = "dc/base/WorldBank_IDS"
OECD_MAIN = "dc/base/OECD_Main"
WB_DATASET = "dc/d/WorldBank_Something"
CRS = "provenance/OECDCreditorReportingSystem"
IDS = "provenance/WorldBankInternationalDebtStatistics"
WHO = "undata/p/WHO"
WHO_GROUP = "undata/g/who"
WDI_DATASET = "dc/d/WorldBank_WorldDevelopmentIndicators"
CRS_DATASET = "dc/d/OECD_CreditorReportingSystem"
UNSD = "undata/d/UNSD"

_UN_PROBE = "undata/who/MORT_U5_RATE"

CRS_VARIABLE = "Amount_ODA_CRS"
IDS_VARIABLE = "DT_DOD_DPPG_CD"
POPULATION = "Count_Person"
GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"

REGISTRY = f"""
[[base]]
source = "{WORLD_BANK}"
label = "World Bank"
exclusions = [{{ provenance = "{WB_IDS}", reason = "superseded_by_dfo" }}]

[[base]]
source = "{OECD}"
label = "Organisation for Economic Co-operation and Development (OECD)"
exclusions = []

[[un_system]]
provenance = "{WHO}"
segment = "who"
label = "World Health Organization (WHO)"
"""


def _provenance(
    url: str | None,
    *,
    source: str | None = None,
    name: str | None = None,
    part_of: str | None = None,
) -> dict:
    arcs: dict[str, tuple[str, ...]] = {"typeOf": ("Provenance",)}
    if url is not None:
        arcs["url"] = (url,)
    if source is not None:
        arcs["source"] = (source,)
    if name is not None:
        arcs["name"] = (name,)
    if part_of is not None:
        arcs["isPartOf"] = (part_of,)
    return arcs


class FakeGraph:
    """`UnGraph` over canned `<-source` subjects and node arcs. A dcid absent
    from `arcs` reads as a missing node. An observation read naming a place
    in `refused` finds it blocked, as the UN front door's refusal does; one
    naming a place in `refused_once` finds it blocked the first time only.
    A read whose method is in `failing` raises `GraphReadError`. `calls`
    records every read but the `/v2/node` POSTs, which `un` answers and
    records."""

    def __init__(
        self,
        *,
        subjects: dict[str, tuple[str, ...]] | None = None,
        arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
        refused: frozenset[str] = frozenset(),
        un: FakeUn | None = None,
    ) -> None:
        self.subjects = subjects or {}
        self.arcs = arcs or {}
        self.refused = refused
        self.un = un or FakeUn()
        self.refused_once: frozenset[str] = frozenset()
        self.failing: frozenset[str] = frozenset()
        self.named: set[str] = set()
        self.blocked: frozenset[str] = frozenset()
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def _fail(self, method: str) -> None:
        if method in self.failing:
            raise GraphReadError(f"graph returned HTTP 503 for {method}", upstream_status=503)

    def __enter__(self) -> FakeGraph:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def post(self, path, payload, *, deadline=None):
        self._fail("post")
        return self.un.post(path, payload, deadline=deadline)

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        assert property == "source"
        self.calls.append(("node_subjects", (dcid,)))
        found = self.subjects.get(dcid, ())
        return SubjectPage(dcids=tuple(found[:max_results]), truncated=len(found) > max_results)

    def observation_facets_batch(
        self, stat_vars, entities, needs_dates=False, *, max_bytes=None, domains=None
    ):
        self.calls.append(("observation_facets_batch", (*stat_vars, *entities)))
        self._fail("observation_facets_batch")
        first_time = self.refused_once - self.named
        self.named |= set(entities)
        self.blocked |= (self.refused | first_time) & set(entities)
        return {}

    def blocked_observation_entities(self) -> frozenset[str]:
        return self.blocked

    def clear(self) -> None:
        self.calls.append(("clear", ()))
        self.blocked = frozenset()

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self.calls.append(("node_arcs_many", tuple(dcids)))
        self._fail("node_arcs_many")
        return {
            dcid: (
                {prop: self.arcs[dcid].get(prop, ()) for prop in properties}
                if dcid in self.arcs
                else None
            )
            for dcid in dcids
        }


COUNT_HOUSEHOLD = "Count_Household"
FINANCING_PROVIDER = "financingProvider"
ORGANIZATION = "Organization"
THING = "Thing"
MEASURED_VALUE = "measuredValue"


def mixer_nodes() -> dict[str, Any]:
    """The `->*` arcs our mixer answers: one base variable with a
    denominator neither variable list holds, the role property a DFO variable
    names, and the class chain its range runs up. `IDS_VARIABLE` and
    `POPULATION` have no node."""
    return {
        GDP: {
            "typeOf": [un_target("StatisticalVariable", "StatisticalVariable", "Class")],
            "name": [{"value": "Nominal GDP"}],
            "measurementDenominator": [un_target(COUNT_HOUSEHOLD, "Households")],
        },
        COUNT_HOUSEHOLD: {"name": [{"value": "Households"}]},
        FINANCING_PROVIDER: {"rangeIncludes": [un_target(ORGANIZATION, "Organization", "Class")]},
        ORGANIZATION: {"subClassOf": [un_target(THING, "Thing", "Class")]},
        THING: {"typeOf": [un_target("Class", "Class", "Class")]},
    }


def mixer() -> FakeGraph:
    return FakeGraph(
        un=FakeUn(mixer_nodes()),
        subjects={
            WORLD_BANK: (WDI, WB_IDS, IDS, WB_DATASET),
            OECD: (CRS, OECD_MAIN),
        },
        arcs={
            WDI: _provenance(
                "https://data.worldbank.org", name="WorldDevelopmentIndicators", part_of=WDI_DATASET
            ),
            WDI_DATASET: {"typeOf": ("Dataset",), "name": ("World Development Indicators",)},
            WB_IDS: _provenance("https://datacatalog.worldbank.org/ids"),
            OECD_MAIN: _provenance("https://stats.oecd.org"),
            WB_DATASET: {"typeOf": ("Dataset",), "url": ("https://example.org",)},
            CRS: {
                **_provenance(
                    "https://data-explorer.oecd.org",
                    source=OECD,
                    name="OECDCreditorReportingSystem",
                    part_of=CRS_DATASET,
                ),
                "licenseType": ("CreativeCommonsAttributionLicense",),
                "license": ("https://www.oecd.org/en/about/terms-conditions.html",),
                "licenseAttribution": ("OECD (2026), Creditor Reporting System",),
                "lastDataRefreshDate": ("2026-10-02",),
                "sourceReleaseFrequency": ("P1Y",),
                "earliestObservationDate": ("1973",),
                "latestObservationDate": ("2024-12-31",),
                "description": ("Aid activity data.",),
                "dataNotes": ("- Figures for the latest year are preliminary.\\n",),
            },
            CRS_DATASET: {
                "typeOf": ("Dataset",),
                "name": ("OECD Creditor Reporting System",),
                "url": ("https://www.oecd.org/crs",),
                "isPartOf": (OECD,),
            },
            OECD: {"typeOf": ("Source",), "name": ("OECD",)},
            IDS: _provenance("https://datacatalog.worldbank.org/0038015", source=WORLD_BANK),
            "CreativeCommonsAttributionLicense": {
                "typeOf": ("LicenseTypeEnum",),
                "name": ("Creative Commons Attribution License",),
            },
            MEASURED_VALUE: {"typeOf": ("Property",), "name": ("measuredValue",)},
        },
    )


def un_deployment() -> FakeGraph:
    return FakeGraph(
        arcs={
            WHO: _provenance("https://www.who.int/data", name="WHO", part_of=UNSD),
            UNSD: {"typeOf": ("Dataset",), "name": ("United Nations Statistics Division",)},
            WHO_GROUP: {"typeOf": ("StatVarGroup",), "name": ("WHO (UN Data)",)},
        },
        refused=frozenset({"country/GRC", "country/ESH"}),
    )


class FakeBuilder:
    """`BaseTablesBuilder` returning `tables`. `during` runs inside the call,
    standing in for whatever happens while the base tables build."""

    def __init__(self, tables: BaseTables | None = None, *, during=None) -> None:
        self.tables = tables or base_tables()
        self.during = during
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        graph,
        registry,
        admitted_base_provenances,
        dfo_provenances,
        deadline=None,
    ):
        self.calls.append(
            {
                "provenances": sorted(admitted_base_provenances),
                "dfo_provenances": sorted(dfo_provenances),
                "deadline": deadline,
                "registry": registry,
            }
        )
        if self.during is not None:
            self.during()
        return self.tables


WITHHELD_VARIABLE = "Count_Person_BelowPovertyLevelInThePast12Months"


def base_tables(**overrides: Any) -> BaseTables:
    fields: dict[str, Any] = {
        "base_variables": {
            POPULATION: BaseVariable(provenances=(WDI,), places=0b11),
            GDP: BaseVariable(provenances=(WDI, OECD_MAIN), places=0b01),
            IDS_VARIABLE: BaseVariable(provenances=(WDI,), places=0b10),
        },
        "withheld_variables": frozenset({WITHHELD_VARIABLE}),
        "topics": {"dc/topic/Demographics": (POPULATION,)},
        "scan_places": ("Earth", "country/KEN", "country/GRC"),
        "scan_levels": {OECD: (ScanLevel(place_type="AdministrativeArea1"),)},
    }
    fields.update(overrides)
    fields.setdefault("provenance_places", spread_provenance_places(fields["base_variables"]))
    return BaseTables(**fields)


def spanner_edges(variables: list[str]) -> list[dict[str, str]]:
    """Each variable's Spanner `Edge` rows: its type, a name literal, a
    `statType` on a base Data Commons node, and for `CRS_VARIABLE` a
    denominator and a role property."""
    rows = []
    for variable in variables:
        arcs = [
            ("typeOf", "StatisticalVariable"),
            ("name", f"{variable}:hash"),
            ("statType", MEASURED_VALUE),
        ]
        if variable == CRS_VARIABLE:
            arcs += [
                ("measurementDenominator", POPULATION),
                ("observationProperties", FINANCING_PROVIDER),
            ]
        rows += [
            {"subject_id": variable, "predicate": predicate, "object_id": object_id}
            for predicate, object_id in arcs
        ]
    return rows


def spanner_nodes(variables: list[str]) -> list[dict[str, Any]]:
    """The Spanner `Node` rows of `spanner_edges`' targets but the base Data Commons ones."""
    return [
        {
            "subject_id": "StatisticalVariable",
            "value": "StatisticalVariable",
            "name": "StatisticalVariable",
            "types": ["Class"],
        },
        {
            "subject_id": FINANCING_PROVIDER,
            "value": FINANCING_PROVIDER,
            "name": "financingProvider",
            "types": ["Property"],
        },
        *(
            {"subject_id": f"{variable}:hash", "value": f"{variable} name", "name": "", "types": []}
            for variable in variables
        ),
    ]


SPANNER_EDGES = "FROM variable_edges AS e"
"""The marker of `arc_reads.dfo_edges_sql`, which `FakeBigQuery` checks before the
holdings query's `LEFT JOIN`."""


def spanner_rows(edges: list[dict[str, str]], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`dfo_edges_sql`'s rows: each `Edge` row with its target's `Node` row from
    `nodes`, all null for a target with none."""
    by_id = {node["subject_id"]: node for node in nodes}
    rows = []
    for edge in edges:
        node = by_id.get(edge["object_id"])
        rows.append(
            {
                **edge,
                "node_id": None if node is None else node["subject_id"],
                "value": None if node is None else node["value"],
                "name": None if node is None else node["name"],
                "types": None if node is None else node["types"],
            }
        )
    return rows


def bq_responses(
    *,
    execution: str = "wf-1",
    lock_owner: str | None = None,
    newest: dict[str, Any] | None = None,
    extra_crs_variables: int = 0,
) -> dict[str, list]:
    """Spanner holds observations of `POPULATION` and declares no node for
    it, as dev's Spanner does for `Count_Person`. `execution` names the
    newest successful import, and `newest` a newer row of another status."""
    declared = [
        CRS_VARIABLE,
        IDS_VARIABLE,
        *(f"{CRS_VARIABLE}_{i}" for i in range(extra_crs_variables)),
    ]
    return {
        SPANNER_EDGES: spanner_rows(spanner_edges(declared), spanner_nodes(declared)),
        **import_key_responses(execution, lock_owner=lock_owner, newest=newest),
        "LEFT JOIN": [
            {"variable": CRS_VARIABLE, "provenance": CRS},
            {"variable": IDS_VARIABLE, "provenance": IDS},
            {"variable": POPULATION, "provenance": CRS},
            *(
                {"variable": f"{CRS_VARIABLE}_{i}", "provenance": CRS}
                for i in range(extra_crs_variables)
            ),
        ],
        "NodeEmbedding": [{"subject_id": variable, "has_edges": True} for variable in declared],
    }


class Clock:
    """Wall-clock time shifted by `offset` seconds."""

    def __init__(self) -> None:
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


def settings(tmp_path: Path) -> catalogue_job.JobSettings:
    return catalogue_job.JobSettings(
        artifact_uri=f"file://{tmp_path}",
        bq_connection=_CONNECTION,
        graph_base_url="https://mixer/core/api",
    )


def run(
    tmp_path: Path,
    *,
    bq: FakeBigQuery | None = None,
    graph: FakeGraph | None = None,
    un_graph: FakeGraph | None = None,
    builder: FakeBuilder | None = None,
    registry: str = REGISTRY,
    clock: Clock | None = None,
    carry_un_from: str | None = None,
    base_index: Sequence[str] = (),
) -> None:
    un_graph = un_graph or un_deployment()
    catalogue_job.run(
        settings(tmp_path),
        bq=bq or FakeBigQuery(bq_responses()),
        # FakeGraph answers the node reads the provenance stage makes, and FakeBuilder reads none.
        graph=cast("GraphClient", graph or mixer()),
        un_graph=un_graph,
        un_group_counts=un_graph.un,
        build_base_tables=builder or FakeBuilder(),
        read_base_index=lambda uri: base_index,
        registry_raw=registry.encode("utf-8"),
        clock=clock or Clock(),
        carry_un_from=carry_un_from,
    )


_HEARTBEAT = "origin_catalogue_job_heartbeat"
_UN_ENUMERATED = "origin_catalogue_un_mirror_enumerated"


def logged(capsys) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def events(capsys) -> list[dict[str, Any]]:
    """The run's events without its heartbeat, which the monitoring tests pin,
    and without the UN enumeration's stats."""
    return [event for event in logged(capsys) if event["event"] not in {_HEARTBEAT, _UN_ENUMERATED}]


def root(tmp_path: Path) -> Path:
    return tmp_path / FORMAT_ROOT


def builds(tmp_path: Path) -> list[str]:
    versions = root(tmp_path) / "versions"
    return sorted(p.name for p in versions.iterdir()) if versions.is_dir() else []


# --- Publication -------------------------------------------------------------


def test_run_publishes_a_build_the_loader_accepts(tmp_path, capsys):
    builder = FakeBuilder()

    run(tmp_path, builder=builder)

    [published] = events(capsys)
    assert published["event"] == "origin_catalogue_job_published"
    assert published["base_tables"] == "built"
    snapshot = load_catalogue(f"file://{tmp_path}")
    rows = snapshot.tables.provenances
    assert {pid: row.data_commons for pid, row in rows.items()} == {
        WDI: BASE,
        OECD_MAIN: BASE,
        CRS: DFO,
        IDS: DFO,
        WHO: UN_SYSTEM,
    }
    assert rows[WDI].source.label == "World Bank"
    # Parent Dataset names, then the dcid for a provenance with neither.
    assert rows[WDI].name == "World Development Indicators"
    assert rows[CRS].name == "OECD Creditor Reporting System"
    assert rows[OECD_MAIN].name == OECD_MAIN
    assert rows[WDI].dataset == DatasetRef(ref=WDI_DATASET, label="World Development Indicators")
    assert rows[CRS].dataset == DatasetRef(ref=CRS_DATASET, label="OECD Creditor Reporting System")
    assert rows[OECD_MAIN].dataset is None
    # Every UN agency sits under the UN Statistics Division, so UN rows take the registry label.
    assert rows[WHO].name == "World Health Organization (WHO)"
    assert rows[WHO].dataset is None
    assert rows[CRS].source.label.startswith("Organisation for Economic")
    assert rows[WHO].segment == "who"
    assert rows[WHO].url == "https://www.who.int/data"
    assert snapshot.tables.dataset_facts == {
        WDI_DATASET: CardFacts(source_url="https://data.worldbank.org"),
        OECD_MAIN: CardFacts(source_url="https://stats.oecd.org"),
        CRS_DATASET: CardFacts(
            description="Aid activity data.",
            license_type="Creative Commons Attribution License",
            update_cadence="P1Y",
            years_covered="1973-2024",
            source_url="https://www.oecd.org/crs",
        ),
        IDS: CardFacts(source_url="https://datacatalog.worldbank.org/0038015"),
    }
    assert rows[CRS].licence == Licence(
        type="CreativeCommonsAttributionLicense",
        label="Creative Commons Attribution License",
        url="https://www.oecd.org/en/about/terms-conditions.html",
    )
    assert rows[WDI].licence == Licence()
    assert rows[WHO].licence == Licence()
    assert snapshot.tables.source_documentation == {
        WDI_DATASET: SourceDocumentation(provenances=(WDI,)),
        OECD_MAIN: SourceDocumentation(provenances=(OECD_MAIN,)),
        CRS_DATASET: SourceDocumentation(
            provenances=(CRS,),
            publisher="OECD",
            citation="OECD (2026), Creditor Reporting System",
            license_url="https://www.oecd.org/en/about/terms-conditions.html",
            last_update="2026-10-02",
            documents={"dataNotes": ("- Figures for the latest year are preliminary.\\n",)},
        ),
        # The graph holds no node for IDS's Source, so it names no publisher.
        IDS: SourceDocumentation(provenances=(IDS,)),
    }
    assert snapshot.tables.dfo_variables == {CRS_VARIABLE, IDS_VARIABLE, POPULATION}
    assert snapshot.manifest.overlaps == (POPULATION, IDS_VARIABLE)
    assert set(snapshot.manifest.identity_inputs) == {
        "format_version",
        "job_code",
        "enumeration_code",
        "base_arc_code",
        "un_mirror_code",
        "registry",
        "dfo_fingerprint",
        "iso_week",
        "base_index",
    }
    assert snapshot.manifest.identity_inputs["job_code"] == catalogue_job.source_digest(
        catalogue_job.JOB_MODULES
    )
    assert snapshot.manifest.identity_inputs["base_index"] == catalogue_job.BASE_INDEX_URI
    assert snapshot.manifest.un_carried_forward is None
    assert builder.calls[0]["provenances"] == [OECD_MAIN, WDI]
    assert not (root(tmp_path) / LEASE_FILE).exists()


def test_the_build_holds_every_node_a_resolver_index_names(tmp_path, capsys):
    """A subject of our index the graph dropped is held empty, and a base node the base
    index names is held with its arcs; a withheld one is left to admission, and a topic
    stands for its leaves, here a listed variable."""
    dropped, indexed = "Amount_FinancialFlow_CRSPurposeCode43010", "Count_Person_Urban"
    responses = bq_responses()
    responses["NodeEmbedding"] = [
        *responses["NodeEmbedding"],
        {"subject_id": dropped, "has_edges": False},
    ]
    graph = mixer()
    graph.un.nodes[indexed] = {"name": [{"value": "Urban population"}]}

    run(
        tmp_path,
        bq=FakeBigQuery(responses),
        graph=graph,
        base_index=[indexed, "dc/topic/Demographics", WITHHELD_VARIABLE],
    )

    [published] = events(capsys)
    assert (
        published["resolver_index_absent"],
        published["resolver_index_held_by_mixer"],
        published["resolver_index_base"],
    ) == (1, 0, 1)
    arcs = load_catalogue(f"file://{tmp_path}").arcs
    assert arcs.dfo[dropped] == ()
    assert [arc.value for arc in arcs.base[indexed]] == ["Urban population"]
    assert WITHHELD_VARIABLE not in arcs.base


def test_the_change_check_scans_no_observation(tmp_path, capsys):
    """A run that finds its identity published reads the import workflow's own rows only."""
    run(tmp_path)
    bq = FakeBigQuery(bq_responses())

    run(tmp_path, bq=bq)

    assert [e["event"] for e in events(capsys)][-1] == "origin_catalogue_job_noop"
    assert bq.queries
    assert not [sql for sql in bq.queries if "Observation" in sql or "last_update" in sql]


def test_run_skips_when_the_published_identity_matches(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    capsys.readouterr()
    graph = mixer()

    run(tmp_path, graph=graph, builder=builder)

    assert [e["event"] for e in events(capsys)] == ["origin_catalogue_job_noop"]
    assert len(builder.calls) == 1
    assert graph.calls == []


def test_a_registry_edit_changes_the_identity(tmp_path, capsys):
    run(tmp_path)
    capsys.readouterr()

    run(tmp_path, registry=REGISTRY + "\n# reviewed\n")

    assert events(capsys)[0]["event"] == "origin_catalogue_job_published"
    assert len(builds(tmp_path)) == 2


def _code(*, job: str, enumeration: str, un_mirror: str = "un-1", base_arc: str = "arc-1"):
    """A `source_digest` answering `enumeration` for the enumeration modules,
    `base_arc` for the base arc modules, `un_mirror` for the UN table modules
    and `job` for the job's."""

    def digest(modules: tuple[str, ...]) -> str:
        return {
            catalogue_job.UN_MIRROR_MODULES: un_mirror,
            catalogue_job.ENUMERATION_MODULES: enumeration,
            catalogue_job.BASE_ARC_MODULES: base_arc,
        }.get(modules, job)

    return digest


def test_a_job_code_change_publishes_and_carries_the_base_tables(tmp_path, capsys, monkeypatch):
    builder = FakeBuilder()
    monkeypatch.setattr(catalogue_job, "source_digest", _code(job="job-1", enumeration="enum-1"))
    run(tmp_path, builder=builder)
    capsys.readouterr()

    monkeypatch.setattr(catalogue_job, "source_digest", _code(job="job-2", enumeration="enum-1"))
    run(tmp_path, builder=builder)

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert published["base_arcs_read"] == 0
    assert published["base_arcs_carried"] > 0
    assert len(builder.calls) == 1
    assert load_catalogue(f"file://{tmp_path}").manifest.identity_inputs["job_code"] == "job-2"


def test_a_base_arc_code_change_reads_the_base_arcs_again(tmp_path, capsys, monkeypatch):
    builder = FakeBuilder()
    monkeypatch.setattr(catalogue_job, "source_digest", _code(job="job-1", enumeration="enum-1"))
    run(tmp_path, builder=builder)
    [first] = events(capsys)

    monkeypatch.setattr(
        catalogue_job, "source_digest", _code(job="job-2", enumeration="enum-1", base_arc="arc-2")
    )
    run(tmp_path, builder=builder)

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert len(builder.calls) == 1
    assert published["base_arcs_carried"] == 0
    assert published["base_arcs_read"] == first["base_arcs_read"] > 0


def test_an_enumeration_code_change_rebuilds_the_base_tables(tmp_path, capsys, monkeypatch):
    builder = FakeBuilder()
    monkeypatch.setattr(catalogue_job, "source_digest", _code(job="job-1", enumeration="enum-1"))
    run(tmp_path, builder=builder)
    capsys.readouterr()

    monkeypatch.setattr(catalogue_job, "source_digest", _code(job="job-2", enumeration="enum-2"))
    run(tmp_path, builder=builder)

    assert events(capsys)[0]["base_tables"] == "built"
    assert len(builder.calls) == 2


def _package_imports(module: str) -> set[str]:
    """The `origin_catalogue` modules `module` imports, as file names."""
    tree = ast.parse((resources.files("origin_catalogue") / module).read_text())
    return {
        f"{node.module.removeprefix('origin_catalogue.')}.py"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("origin_catalogue.")
    }


def test_the_job_modules_cover_every_package_module_the_job_imports():
    # The serving modules the job reads only types and constants from: the
    # catalogue's row types and invariants, and the UN GET budget.
    serving = {"catalogue.py", "graph.py"}
    imported = set().union(*(_package_imports(m) for m in catalogue_job.JOB_MODULES))
    assert imported - serving <= set(catalogue_job.JOB_MODULES)
    assert set(catalogue_job.ENUMERATION_MODULES) <= set(catalogue_job.JOB_MODULES)
    assert set(catalogue_job.BASE_ARC_MODULES) <= set(catalogue_job.JOB_MODULES)
    assert set(catalogue_job.UN_MIRROR_MODULES) <= set(catalogue_job.JOB_MODULES)
    assert not {"context.py", "service.py", "graph.py"} & set(catalogue_job.JOB_MODULES)
    # The card-fact derivation shapes every build, so a change to it rebuilds.
    assert "card_facts.py" in catalogue_job.JOB_MODULES


def test_the_base_arc_code_hashes_the_node_reads_and_not_the_un_enumeration():
    """A UN enumeration edit keeps the carried base arcs: the base arc modules take
    only the `UnArc` row type from `un_mirror.py`, and the reads they share with the UN
    tables live in `node_reads.py`, which both code hashes cover."""
    assert catalogue_job.BASE_ARC_MODULES == ("arc_reads.py", "arc_tables.py", "node_reads.py")
    assert "node_reads.py" in catalogue_job.UN_MIRROR_MODULES
    for module in catalogue_job.BASE_ARC_MODULES:
        tree = ast.parse((resources.files("origin_catalogue") / module).read_text())
        taken = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "origin_catalogue.un_mirror"
            for alias in node.names
        }
        assert taken <= {"UnArc"}, module


def test_the_source_digest_reads_the_named_modules():
    job_code = catalogue_job.source_digest(catalogue_job.JOB_MODULES)
    assert len(job_code) == 64
    assert job_code != catalogue_job.source_digest(catalogue_job.ENUMERATION_MODULES)
    with pytest.raises(FileNotFoundError):
        catalogue_job.source_digest(("missing.py",))


# --- UN blocked places -------------------------------------------------------


def test_the_build_records_the_scan_countries_the_un_front_door_refuses(tmp_path, capsys):
    un_graph = un_deployment()

    run(tmp_path, un_graph=un_graph)

    [published] = events(capsys)
    assert published["un_blocked_places"] == ["country/GRC"]
    probes = [call for call in un_graph.calls if call[0] in {"observation_facets_batch", "clear"}]
    # The place found refused is forgotten and read alone once more.
    assert probes == [
        ("observation_facets_batch", (_UN_PROBE, "country/KEN", "country/GRC")),
        ("clear", ()),
        ("observation_facets_batch", (_UN_PROBE, "country/GRC")),
    ]
    assert load_catalogue(f"file://{tmp_path}").un_blocked_places == {"country/GRC"}


def test_a_place_refused_once_is_not_recorded(tmp_path, capsys):
    un_graph = un_deployment()
    un_graph.refused = frozenset()
    un_graph.refused_once = frozenset({"country/KEN"})

    run(tmp_path, un_graph=un_graph)

    assert events(capsys)[0]["un_blocked_places"] == []
    assert load_catalogue(f"file://{tmp_path}").un_blocked_places == frozenset()


def test_a_probe_that_finds_nothing_reads_once(tmp_path, capsys):
    un_graph = un_deployment()
    un_graph.refused = frozenset()

    run(tmp_path, un_graph=un_graph)

    assert [call[0] for call in un_graph.calls if call[0] != "node_arcs_many"] == [
        "observation_facets_batch"
    ]


def test_a_carried_build_probes_again(tmp_path, capsys):
    run(tmp_path)
    capsys.readouterr()
    un_graph = un_deployment()
    un_graph.refused = frozenset({"country/KEN"})

    run(tmp_path, bq=FakeBigQuery(bq_responses(execution="wf-4")), un_graph=un_graph)

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert load_catalogue(f"file://{tmp_path}").un_blocked_places == {"country/KEN"}


# --- UN carry-forward --------------------------------------------------------


def _publish_then_fail_the_un(tmp_path, capsys, un_graph: FakeGraph, **kwargs: Any) -> list:
    """Publishes a build, then runs again after an ingestion with `un_graph`;
    returns the second run's events."""
    run(tmp_path)
    capsys.readouterr()
    run(tmp_path, bq=FakeBigQuery(bq_responses(execution="wf-4")), un_graph=un_graph, **kwargs)
    return events(capsys)


@pytest.mark.parametrize("failing", ["node_arcs_many", "observation_facets_batch"])
def test_a_un_read_failure_carries_the_un_rows_and_refused_places_forward(
    tmp_path, capsys, failing
):
    un_graph = un_deployment()
    un_graph.failing = frozenset({failing})

    carried, published = _publish_then_fail_the_un(tmp_path, capsys, un_graph)

    assert carried["event"] == "origin_catalogue_job_un_carried_forward"
    assert carried["severity"] == "WARNING"
    assert "UN deployment read failed" in carried["reason"]
    assert published["event"] == "origin_catalogue_job_published"
    first = carried["from_build"]
    assert published["un_carried_from"] == first
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert snapshot.build_id == published["build_id"] != first
    assert snapshot.tables.provenances[WHO].url == "https://www.who.int/data"
    assert snapshot.un_blocked_places == {"country/GRC"}
    note = snapshot.manifest.un_carried_forward
    assert note is not None
    assert (note.from_build, note.reason) == (first, carried["reason"])
    # DFO's update publishes regardless.
    assert snapshot.tables.dfo_variables == {CRS_VARIABLE, IDS_VARIABLE, POPULATION}


def test_a_un_refusal_refuses_even_with_a_published_build(tmp_path, capsys):
    run(tmp_path)
    first = CataloguePointer.from_json((root(tmp_path) / "current.json").read_bytes()).build_id
    capsys.readouterr()
    un_graph = un_deployment()
    del un_graph.arcs[WHO_GROUP]

    refused = _refused(
        tmp_path, capsys, bq=FakeBigQuery(bq_responses(execution="wf-4")), un_graph=un_graph
    )

    assert refused == [("un_agency_root_group_missing", WHO)]
    assert load_catalogue(f"file://{tmp_path}").build_id == first


def test_a_carried_row_whose_segment_the_registry_changed_is_refused(tmp_path, capsys):
    run(tmp_path)
    capsys.readouterr()
    un_graph = un_deployment()
    un_graph.failing = frozenset({"node_arcs_many"})

    refused = _refused(
        tmp_path,
        capsys,
        bq=FakeBigQuery(bq_responses(execution="wf-4")),
        un_graph=un_graph,
        registry=REGISTRY.replace('segment = "who"', 'segment = "who2"'),
    )

    # The carried tables also hold the old segment, which the registry no longer lists.
    assert refused == [("un_carried_segment_mismatch", WHO), ("un_carried_agency_mismatch", "who")]


def test_a_published_carry_forward_retries_the_un_on_an_unchanged_identity(tmp_path, capsys):
    down = un_deployment()
    down.failing = frozenset({"node_arcs_many"})
    _publish_then_fail_the_un(tmp_path, capsys, down)
    carried_build = CataloguePointer.from_json(
        (root(tmp_path) / "current.json").read_bytes()
    ).build_id
    ingested = bq_responses(execution="wf-4")

    # Still down: the retry publishes nothing new, and reads the UN before anything else,
    # so it reads no arc, provenance or holdings.
    bq, graph = FakeBigQuery(ingested), mixer()
    run(tmp_path, bq=bq, graph=graph, un_graph=down)
    [retry] = events(capsys)
    assert retry["event"] == "origin_catalogue_job_un_retry_failed"
    assert load_catalogue(f"file://{tmp_path}").build_id == carried_build
    assert (graph.calls, graph.un.requests) == ([], [])
    assert not [sql for sql in bq.queries if SPANNER_EDGES in sql or "NodeEmbedding" in sql]

    # Back up: the same identity publishes a build that read the UN.
    run(tmp_path, bq=FakeBigQuery(ingested))
    [published] = events(capsys)
    assert published["event"] == "origin_catalogue_job_published"
    assert published["un_carried_from"] is None
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert snapshot.build_id == published["build_id"] != carried_build
    assert snapshot.manifest.un_carried_forward is None

    # Then the identity skip holds again.
    run(tmp_path, bq=FakeBigQuery(ingested))
    assert [e["event"] for e in events(capsys)] == ["origin_catalogue_job_noop"]


def test_a_carried_carry_forward_names_the_build_that_read_the_un(tmp_path, capsys):
    run(tmp_path)
    first = CataloguePointer.from_json((root(tmp_path) / "current.json").read_bytes()).build_id
    down = un_deployment()
    down.failing = frozenset({"node_arcs_many"})
    run(tmp_path, bq=FakeBigQuery(bq_responses(execution="wf-4")), un_graph=down)
    capsys.readouterr()

    run(tmp_path, bq=FakeBigQuery(bq_responses(execution="wf-5")), un_graph=down)

    published = events(capsys)[-1]
    assert published["un_carried_from"] == first
    carried = load_catalogue(f"file://{tmp_path}").manifest.un_carried_forward
    assert carried is not None
    assert carried.from_build == first


def test_carried_un_rows_follow_the_current_registry(tmp_path, capsys):
    un_graph = un_deployment()
    un_graph.failing = frozenset({"node_arcs_many"})
    registry = REGISTRY.replace("World Health Organization (WHO)", "World Health Organization")
    registry += """
[[un_system]]
provenance = "undata/p/UNICEF"
segment = "unicef"
label = "UNICEF"
"""

    _publish_then_fail_the_un(tmp_path, capsys, un_graph, registry=registry)

    rows = load_catalogue(f"file://{tmp_path}").tables.provenances
    assert rows[WHO].name == rows[WHO].source.label == "World Health Organization"
    assert rows[WHO].segment == "who"
    # An agency the carried build never read stays out until the UN answers.
    assert "undata/p/UNICEF" not in rows


def test_a_un_read_failure_with_nothing_published_fails_the_build(tmp_path, capsys):
    un_graph = un_deployment()
    un_graph.failing = frozenset({"node_arcs_many"})

    with pytest.raises(catalogue_job.CatalogueBuildError, match="no published build"):
        run(tmp_path, un_graph=un_graph)

    assert events(capsys) == []
    _assert_unpublished(tmp_path)


# --- UN tables ---------------------------------------------------------------

_WEEK = 7 * 86_400.0


def _excluding(series: str, *, dfo_provenance: str = CRS) -> str:
    """`REGISTRY` with WHO excluding `series` as superseded by `dfo_provenance`."""
    return REGISTRY + (
        f'exclusions = [{{ series = "{series}", reason = "superseded_by_dfo", '
        f'dfo_provenance = "{dfo_provenance}" }}]\n'
    )


def _un_tables(tmp_path: Path) -> UnMirror:
    uri = f"file://{tmp_path}"
    return read_un_mirror(uri, load_catalogue(uri).manifest)


def test_the_build_publishes_the_un_tables_and_their_counts(tmp_path, capsys):
    run(tmp_path, registry=_excluding(ODA_SERIES))

    enumerated, published = (e for e in logged(capsys) if e["event"] != _HEARTBEAT)
    assert enumerated["event"] == _UN_ENUMERATED
    [who] = enumerated["agencies"]
    assert (who["segment"], who["variables"], who["counted"]) == ("who", 4, 4)
    assert published["un_mirror"] == "enumerated"
    tables = _un_tables(tmp_path)
    assert set(tables.variables) == {MORT, MORT_FEMALE, MORT_MALE}
    assert tables.excluded_variables == {ODA: ODA_SERIES}
    assert tables.exclusions == {ODA_SERIES: CRS}
    recorded = load_catalogue(f"file://{tmp_path}").manifest.un_mirror
    assert recorded.agencies == {
        "who": UnAgencyCounts(series=2, variables=4, excluded_series=1, excluded_variables=1)
    }
    assert recorded.row_counts == published["un_row_counts"]
    assert recorded.row_counts["un_variables"] == 3
    assert recorded.row_counts["un_excluded_variables"] == 1
    assert recorded.resident_bytes == published["un_resident_bytes"] > 0


def test_an_ingestion_copies_the_un_tables_without_enumerating_them(tmp_path, capsys, monkeypatch):
    un_graph = un_deployment()
    run(tmp_path, un_graph=un_graph)
    first = _un_tables(tmp_path)
    requests = len(un_graph.un.requests)
    capsys.readouterr()
    reads: list[str] = []
    read = artifact._read_un_mirror
    monkeypatch.setattr(
        artifact,
        "_read_un_mirror",
        lambda prefix, manifest: reads.append(prefix) or read(prefix, manifest),
    )

    run(tmp_path, bq=FakeBigQuery(bq_responses(execution="wf-4")), un_graph=un_graph)

    # The published build's UN tables are read once, by the copy, and not by the carry check.
    assert len(reads) == 1
    monkeypatch.undo()

    [published] = logged(capsys)[:-1]
    assert published["un_mirror"] == "copied"
    assert len(un_graph.un.requests) == requests
    assert _un_tables(tmp_path) == first
    assert load_catalogue(f"file://{tmp_path}").manifest.un_carried_forward is None


def test_a_new_iso_week_enumerates_the_un_tables_again(tmp_path, capsys):
    un_graph, clock = un_deployment(), Clock()
    run(tmp_path, un_graph=un_graph, clock=clock)
    requests = len(un_graph.un.requests)
    capsys.readouterr()

    clock.offset = _WEEK
    run(tmp_path, un_graph=un_graph, clock=clock)

    assert events(capsys)[0]["un_mirror"] == "enumerated"
    assert len(un_graph.un.requests) == 2 * requests


def test_a_un_table_code_change_enumerates_the_un_tables_again(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(catalogue_job, "source_digest", _code(job="job-1", enumeration="enum-1"))
    run(tmp_path)
    capsys.readouterr()

    monkeypatch.setattr(
        catalogue_job, "source_digest", _code(job="job-2", enumeration="enum-1", un_mirror="un-2")
    )
    run(tmp_path)

    [published] = events(capsys)
    assert (published["base_tables"], published["un_mirror"]) == ("carried", "enumerated")


def test_refuses_an_excluded_series_the_un_does_not_hold(tmp_path, capsys):
    refused = _refused(tmp_path, capsys, registry=_excluding("undata/who/SERIES-GONE"))

    assert refused == [("un_excluded_series_missing", "undata/who/SERIES-GONE")]
    _assert_unpublished(tmp_path)


def test_refuses_an_excluded_series_whose_dfo_provenance_dfo_does_not_hold(tmp_path, capsys):
    refused = _refused(
        tmp_path, capsys, registry=_excluding(ODA_SERIES, dfo_provenance="provenance/NotHeld")
    )

    assert refused == [("un_exclusion_provenance_not_held", ODA_SERIES)]
    _assert_unpublished(tmp_path)


@pytest.fixture(autouse=True)
def _no_wait_before_a_repeated_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node_reads, "_REPEATED_PAGE_WAITS_SECONDS", (0.0, 0.0, 0.0))


def _stalling(un_graph: FakeGraph) -> None:
    un_graph.un.page_size = 1
    un_graph.un.repeating = {MORT_BY_SEX}


def _truncating(un_graph: FakeGraph) -> None:
    un_graph.un.truncating = {MORT_BY_SEX: 1}


def _refusing(un_graph: FakeGraph) -> None:
    un_graph.un.refused = [{MORT_BY_SEX}]


_INCOMPLETE = pytest.mark.parametrize(
    ("hazard", "kind"),
    [
        (_stalling, "un_enumeration_stalled"),
        (_truncating, "un_variable_count_mismatch"),
        (_refusing, "un_read_refused"),
    ],
)


@_INCOMPLETE
def test_an_incomplete_enumeration_carries_the_un_tables_forward_and_publishes(
    tmp_path, capsys, hazard, kind
):
    run(tmp_path)
    first_build = load_catalogue(f"file://{tmp_path}").build_id
    first = _un_tables(tmp_path)
    capsys.readouterr()
    incomplete, clock = un_deployment(), Clock()
    hazard(incomplete)
    clock.offset = _WEEK

    run(tmp_path, un_graph=incomplete, clock=clock)

    by_event = {event["event"]: event for event in events(capsys)}
    assert "origin_catalogue_job_refused" not in by_event
    carried = by_event["origin_catalogue_job_un_carried_forward"]
    assert carried["severity"] == "WARNING"
    assert kind in carried["reason"]
    published = by_event["origin_catalogue_job_published"]
    assert (published["un_mirror"], published["un_carried_from"]) == ("carried", first_build)
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert snapshot.build_id == published["build_id"] != first_build
    assert _un_tables(tmp_path) == first
    note = snapshot.manifest.un_carried_forward
    assert note is not None
    assert (note.from_build, note.reason) == (first_build, carried["reason"])

    # The next run reads the UN again.
    run(tmp_path, clock=clock)
    by_event = {event["event"]: event for event in events(capsys)}
    assert by_event["origin_catalogue_job_published"]["un_mirror"] == "enumerated"


@_INCOMPLETE
def test_an_incomplete_enumeration_with_nothing_published_fails_the_build(
    tmp_path, capsys, hazard, kind
):
    un_graph = un_deployment()
    hazard(un_graph)

    with pytest.raises(catalogue_job.CatalogueBuildError, match=f"{kind}.*no published build"):
        run(tmp_path, un_graph=un_graph)

    assert "origin_catalogue_job_refused" not in {event["event"] for event in events(capsys)}
    _assert_unpublished(tmp_path)


_SUPERSEDED = {
    "ids/MORT.SEX--F": MORT_FEMALE,
    "ids/MORT": MORT,
    "ids/ODA_1": ODA,
    "ids/MORT.SEX--FEMALE": None,
    "ids/GONE": None,
}
"""Withheld variables of a provenance WHO supersedes, with the successor of
each: the UN variable of equal code and pairs, and none for a value WHO
spells otherwise or a code WHO does not hold."""


def _superseding(registry: str = REGISTRY) -> dict[str, Any]:
    """`run` arguments for a build whose registry has WHO supersede the
    World Bank's IDS copy, and whose withheld set holds `_SUPERSEDED`."""
    by_un = registry.replace(
        'reason = "superseded_by_dfo" }]\n\n[[base]]',
        f'reason = "superseded_by_un_system", successor = "{WHO}", '
        'variable_prefix = "ids/", code_delimiter = ".", '
        'pair_delimiter = "__", value_delimiter = "--" }]\n\n[[base]]',
    )
    assert by_un != registry
    withheld = frozenset({WITHHELD_VARIABLE, *_SUPERSEDED})
    return {"registry": by_un, "builder": FakeBuilder(base_tables(withheld_variables=withheld))}


def test_the_build_publishes_the_successor_of_each_superseded_variable(tmp_path, capsys):
    run(tmp_path, **_superseding())

    [published] = events(capsys)
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert snapshot.tables.superseded == _SUPERSEDED
    assert snapshot.successor("ids/MORT") == MORT
    assert snapshot.successor("ids/GONE") is None
    assert snapshot.is_superseded("ids/GONE")
    assert not snapshot.is_superseded(WITHHELD_VARIABLE)
    counts = published["row_counts"]
    assert (counts["superseded_variables"], counts["successor_variables"]) == (5, 3)
    assert published["successors"] == {
        "twins": 3,
        "reordered": 0,
        "series_excluded": 0,
        "unmatched": 2,
        "held_by_dfo": 0,
    }


def test_a_variable_of_an_excluded_series_has_no_successor(tmp_path, capsys):
    run(tmp_path, **_superseding(_excluding(ODA_SERIES)))

    [published] = events(capsys)
    superseded = load_catalogue(f"file://{tmp_path}").tables.superseded
    assert superseded == {v: None if s == ODA else s for v, s in _SUPERSEDED.items()}
    assert published["successors"]["series_excluded"] == 1


def test_carried_un_tables_give_the_same_successors(tmp_path, capsys):
    run(tmp_path, **_superseding())
    capsys.readouterr()
    down, clock = un_deployment(), Clock()
    down.failing = frozenset({"post"})
    clock.offset = _WEEK

    run(tmp_path, un_graph=down, clock=clock, **_superseding())

    by_event = {event["event"]: event for event in events(capsys)}
    assert by_event["origin_catalogue_job_published"]["un_mirror"] == "carried"
    assert load_catalogue(f"file://{tmp_path}").tables.superseded == _SUPERSEDED


def test_a_successor_the_un_tables_do_not_admit_refuses_the_build(tmp_path, capsys, monkeypatch):
    derived = catalogue_job.derive_successors
    monkeypatch.setattr(
        catalogue_job,
        "derive_successors",
        lambda *args, **kwargs: replace(
            derived(*args, **kwargs), rows={"ids/GONE": "undata/who/GONE"}
        ),
    )

    refused = _refused(tmp_path, capsys, **_superseding())

    assert refused == [("un_successor_not_held", "ids/GONE")]
    _assert_unpublished(tmp_path)


def test_an_ambiguous_successor_refuses_un_successor_ambiguous():
    successors = Successors(
        rows={"ids/X.A--1__B--2": None},
        ambiguous={"ids/X.A--1__B--2": ("undata/who/X.A--1__B--2", "undata/who/X.B--2__A--1")},
    )

    refused = catalogue_job.successor_refusals(
        successors,
        UnMirror(
            variables={},
            excluded_variables={},
            topics={},
            exclusions={},
            excluded_series_names={},
            agencies={},
        ),
    )

    assert [(r.kind, r.subject) for r in refused] == [
        ("un_successor_ambiguous", "ids/X.A--1__B--2")
    ]


def test_refuses_a_base_provenance_superseded_by_an_agency_the_build_has_no_row_for():
    registry = parse_registry(
        REGISTRY.replace(
            'reason = "superseded_by_dfo"',
            f'reason = "superseded_by_un_system", successor = "{WHO}", '
            'variable_prefix = "ids/", code_delimiter = ".", '
            'pair_delimiter = "__", value_delimiter = "--"',
        ).encode()
    )
    row = provenance_row(WHO, UN_SYSTEM, segment="who")

    def refusals(un_rows: dict) -> list[tuple[str, str]]:
        found = catalogue_job.un_registry_refusals(
            registry, dfo_provenances=frozenset({CRS}), un_rows=un_rows
        )
        return [(refusal.kind, refusal.subject) for refusal in found]

    assert refusals({WHO: row}) == []
    assert refusals({}) == [("un_successor_not_served", WB_IDS)]


def test_a_failed_enumeration_carries_the_un_tables_forward_whole(tmp_path, capsys):
    run(tmp_path)
    first_build = load_catalogue(f"file://{tmp_path}").build_id
    first = _un_tables(tmp_path)
    capsys.readouterr()
    down, clock = un_deployment(), Clock()
    down.failing = frozenset({"post"})
    clock.offset = _WEEK

    run(tmp_path, un_graph=down, clock=clock)

    by_event = {event["event"]: event for event in events(capsys)}
    assert "origin_catalogue_job_un_carried_forward" in by_event
    published = by_event["origin_catalogue_job_published"]
    assert (published["un_mirror"], published["un_carried_from"]) == ("carried", first_build)
    assert _un_tables(tmp_path) == first
    assert load_catalogue(f"file://{tmp_path}").manifest.un_carried_forward is not None

    # Carried tables are never copied: the next run enumerates the UN again.
    run(tmp_path, clock=clock)
    by_event = {event["event"]: event for event in events(capsys)}
    assert by_event["origin_catalogue_job_published"]["un_mirror"] == "enumerated"
    assert load_catalogue(f"file://{tmp_path}").manifest.un_carried_forward is None


@pytest.mark.parametrize(
    ("registry", "subject"),
    [
        (_excluding(ODA_SERIES), ODA_SERIES),
        (_excluding("undata/who/SERIES-GONE"), "undata/who/SERIES-GONE"),
    ],
)
def test_carried_un_tables_that_disagree_with_the_registry_are_refused(
    tmp_path, capsys, registry, subject
):
    run(tmp_path)
    first = load_catalogue(f"file://{tmp_path}").build_id
    capsys.readouterr()
    down = un_deployment()
    down.failing = frozenset({"post"})

    refused = _refused(tmp_path, capsys, un_graph=down, registry=registry)

    assert refused == [("un_carried_exclusions_mismatch", subject)]
    assert load_catalogue(f"file://{tmp_path}").build_id == first


def test_carried_un_tables_holding_an_agency_the_registry_dropped_are_refused(tmp_path, capsys):
    run(tmp_path)
    first = load_catalogue(f"file://{tmp_path}").build_id
    capsys.readouterr()
    down = un_deployment()
    down.failing = frozenset({"post"})
    without_who = "un_system = []\n" + REGISTRY[: REGISTRY.index("[[un_system]]")]

    refused = _refused(tmp_path, capsys, un_graph=down, registry=without_who)

    assert refused == [("un_carried_agency_mismatch", "who")]
    assert load_catalogue(f"file://{tmp_path}").build_id == first


def test_carried_un_tables_of_an_agency_renamed_under_its_segment_are_refused(tmp_path, capsys):
    run(tmp_path)
    first = load_catalogue(f"file://{tmp_path}").build_id
    capsys.readouterr()
    down = un_deployment()
    down.failing = frozenset({"node_arcs_many"})
    renamed = REGISTRY.replace(f'provenance = "{WHO}"', 'provenance = "undata/p/WHO_RENAMED"')
    assert renamed != REGISTRY

    refused = _refused(tmp_path, capsys, un_graph=down, registry=renamed)

    assert refused == [("un_carried_agency_mismatch", "who")]
    assert load_catalogue(f"file://{tmp_path}").build_id == first


# --- Carry-forward -------------------------------------------------------


def test_an_ingestion_carries_the_base_tables_into_a_new_build_directory(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    first = CataloguePointer.from_json((root(tmp_path) / "current.json").read_bytes()).build_id
    capsys.readouterr()

    first_arcs = load_catalogue(f"file://{tmp_path}").arcs
    graph = mixer()

    run(tmp_path, bq=FakeBigQuery(bq_responses(execution="wf-4")), graph=graph, builder=builder)

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert len(builder.calls) == 1
    # The base arcs carry with the base tables; the schema nodes are read again.
    assert (published["base_arcs_read"], published["base_arcs_carried"]) == (0, 2)
    read = {
        node
        for request in graph.un.requests
        if request["property"] == "->*"
        for node in request["nodes"]
    }
    assert read == {COUNT_HOUSEHOLD, FINANCING_PROVIDER, ORGANIZATION, THING, "observationAbout"}
    second = published["build_id"]
    assert (root(tmp_path) / "versions" / second / "base_variables.parquet").is_file()
    # The new build stands alone: it loads with the previous build deleted.
    for path in (root(tmp_path) / "versions" / first).iterdir():
        path.unlink()
    (root(tmp_path) / "versions" / first).rmdir()
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert snapshot.build_id == second
    assert snapshot.tables.base_variables == base_tables().base_variables
    assert snapshot.tables.withheld_variables == frozenset({WITHHELD_VARIABLE})
    assert snapshot.tables.topics == base_tables().topics
    assert snapshot.admit_candidate(WITHHELD_VARIABLE) is None
    assert snapshot.arcs == first_arcs


def test_a_new_dfo_variable_carries_the_base_tables_forward(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    capsys.readouterr()

    run(
        tmp_path,
        bq=FakeBigQuery(bq_responses(execution="wf-4", extra_crs_variables=1)),
        builder=builder,
    )

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert len(builder.calls) == 1
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert f"{CRS_VARIABLE}_0" in snapshot.tables.dfo_variables
    # DFO arcs are read on every ingestion.
    assert f"{CRS_VARIABLE}_0" in snapshot.arcs.dfo


def test_a_new_iso_week_rebuilds_the_base_tables(tmp_path, capsys):
    builder = FakeBuilder()
    clock = Clock()
    run(tmp_path, builder=builder, clock=clock)
    capsys.readouterr()

    clock.offset = 7 * 24 * 3600
    run(tmp_path, builder=builder, clock=clock)

    published = events(capsys)[0]
    assert published["base_tables"] == "built"
    assert len(builder.calls) == 2
    assert (published["base_arcs_read"], published["base_arcs_carried"]) == (2, 0)


def test_a_new_base_provenance_rebuilds_the_base_tables(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    capsys.readouterr()
    graph = mixer()
    graph.subjects[WORLD_BANK] = (*graph.subjects[WORLD_BANK], "dc/base/WorldBank_New")
    graph.arcs["dc/base/WorldBank_New"] = _provenance("https://data.worldbank.org/new")

    run(tmp_path, bq=FakeBigQuery(bq_responses(execution="wf-4")), graph=graph, builder=builder)

    assert events(capsys)[0]["base_tables"] == "built"
    assert builder.calls[1]["provenances"] == [OECD_MAIN, "dc/base/WorldBank_New", WDI]


def test_a_new_dfo_provenance_rebuilds_the_base_tables(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    capsys.readouterr()
    new = "provenance/NewDfoDataset"
    graph = mixer()
    graph.arcs[new] = _provenance("https://example.org/new", source="dc/s/NewSource")
    responses = bq_responses(execution="wf-4")
    responses["LEFT JOIN"].append({"variable": "Amount_New", "provenance": new})

    run(tmp_path, bq=FakeBigQuery(responses), graph=graph, builder=builder)

    assert events(capsys)[0]["base_tables"] == "built"
    assert builder.calls[0]["dfo_provenances"] == [CRS, IDS]
    assert builder.calls[1]["dfo_provenances"] == sorted([CRS, IDS, new])


_DROPPED = "Amount_FinancialFlow_CRSPurposeCode43010"
_DROPPED_LATER = "Amount_FinancialFlow_CRSPurposeCode43020"


def _indexing(*subjects: str, execution: str = "wf-1") -> FakeBigQuery:
    """Spanner's answers with `subjects` added to our resolver index, each a
    subject Spanner holds no `Edge` row of and the mixer holds no node for."""
    responses = bq_responses(execution=execution)
    responses["NodeEmbedding"] = [
        *responses["NodeEmbedding"],
        *({"subject_id": subject, "has_edges": False} for subject in subjects),
    ]
    return FakeBigQuery(responses)


def _type_read(graph: FakeGraph) -> set[str]:
    return {
        node
        for request in graph.un.requests
        if request["property"] == "->typeOf"
        for node in request["nodes"]
    }


def test_an_ingestion_reads_the_type_of_only_the_subjects_the_published_build_does_not_hold(
    tmp_path, capsys
):
    """A subject the published build holds absent stays absent with no `->typeOf` read,
    and a subject new to our index is read."""
    run(tmp_path, bq=_indexing(_DROPPED))
    capsys.readouterr()
    graph = mixer()

    run(tmp_path, bq=_indexing(_DROPPED, _DROPPED_LATER, execution="wf-4"), graph=graph)

    [published] = events(capsys)
    assert _type_read(graph) == {_DROPPED_LATER}
    assert (published["resolver_index_carried"], published["resolver_index_read"]) == (1, 1)
    assert published["resolver_index_absent"] == 2
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert snapshot.arcs.dfo[_DROPPED] == snapshot.arcs.dfo[_DROPPED_LATER] == ()
    assert snapshot.manifest.arc_tables.index_presence == IndexPresence(carried=1, read=1)


@pytest.mark.parametrize("change", ["iso_week", "base_arc_code"])
def test_a_new_week_or_base_arc_code_reads_the_type_of_every_subject(
    tmp_path, capsys, monkeypatch, change
):
    clock = Clock()
    monkeypatch.setattr(catalogue_job, "source_digest", _code(job="job-1", enumeration="enum-1"))
    run(tmp_path, bq=_indexing(_DROPPED), clock=clock)
    capsys.readouterr()
    if change == "iso_week":
        clock.offset = 7 * 24 * 3600
    else:
        monkeypatch.setattr(
            catalogue_job,
            "source_digest",
            _code(job="job-1", enumeration="enum-1", base_arc="arc-2"),
        )
    graph = mixer()

    run(tmp_path, bq=_indexing(_DROPPED, execution="wf-4"), graph=graph, clock=clock)

    published = events(capsys)[0]
    assert _type_read(graph) == {_DROPPED}
    assert (published["resolver_index_carried"], published["resolver_index_read"]) == (0, 1)
    assert load_catalogue(f"file://{tmp_path}").arcs.dfo[_DROPPED] == ()


# --- Arc tables --------------------------------------------------------------


def test_the_build_publishes_dfo_base_and_schema_arcs(tmp_path, capsys):
    run(tmp_path)

    [published] = events(capsys)
    snapshot = load_catalogue(f"file://{tmp_path}")
    arcs = snapshot.arcs
    assert set(arcs.dfo) == {CRS_VARIABLE, IDS_VARIABLE}
    assert held_wire_arcs(arcs, CRS_VARIABLE) == {
        # A base Data Commons node the mixer has no node for carries its dcid alone.
        "measurementDenominator": {"nodes": [{"dcid": POPULATION}]},
        "name": {"nodes": [{"value": f"{CRS_VARIABLE} name"}]},
        "observationProperties": {
            "nodes": [
                {"dcid": FINANCING_PROVIDER, "name": "financingProvider", "types": ["Property"]}
            ]
        },
        # Spanner has no node for it: the mixer names it.
        "statType": {
            "nodes": [{"dcid": MEASURED_VALUE, "name": "measuredValue", "types": ["Property"]}]
        },
        "typeOf": {
            "nodes": [
                {"dcid": "StatisticalVariable", "name": "StatisticalVariable", "types": ["Class"]}
            ]
        },
    }
    # IDS_VARIABLE's node is DFO's, so the DFO table holds it and the base table does not.
    assert set(arcs.base) == {GDP, POPULATION}
    assert arcs.arcs(POPULATION) == ()
    assert held_wire_arcs(arcs, GDP)["measurementDenominator"] == {
        "nodes": [{"dcid": COUNT_HOUSEHOLD, "name": "Households"}]
    }
    assert set(arcs.schema) == {
        COUNT_HOUSEHOLD,
        FINANCING_PROVIDER,
        ORGANIZATION,
        THING,
        "observationAbout",
    }
    assert arcs.arcs("observationAbout") == ()
    assert snapshot.manifest.ingestion == Ingestion(
        workflow_execution_id="wf-1", creation_timestamp="2026-09-24 10:00:00+00:00"
    )
    assert published["arc_row_counts"] == dict(snapshot.manifest.arc_tables.row_counts)
    assert published["arc_resident_bytes"] == snapshot.manifest.arc_tables.resident_bytes > 0
    assert (published["base_arcs_read"], published["base_arcs_carried"]) == (2, 0)


def test_a_variable_spanner_declares_no_node_for_stays_dfo_with_base_arcs(tmp_path, capsys):
    """Spanner holds observations of `POPULATION` and no node for it, as dev's
    Spanner does for `Count_Person`, whose node is base Data Commons'. Its
    observations keep it a DFO variable; its arcs are the base node's."""
    run(tmp_path)

    [published] = events(capsys)
    snapshot = load_catalogue(f"file://{tmp_path}")
    assert published["dfo_variables_without_node"] == [POPULATION]
    assert POPULATION in snapshot.tables.dfo_variables
    assert snapshot.data_commons(POPULATION) == (DFO, BASE)
    assert snapshot.manifest.overlaps == (POPULATION, IDS_VARIABLE)
    assert POPULATION in snapshot.arcs.base
    assert POPULATION not in snapshot.arcs.dfo


def _as_format_3(build: Path) -> None:
    """Rewrites `build` as a format-3 build of an earlier ISO week: no arc
    tables, no source documentation, no licence on any provenance row and no
    ingestion in its manifest."""
    manifest = json.loads((build / "manifest.json").read_text())
    for logical in ("dfo_arcs", "base_arcs", "schema_arcs", "arc_targets", "source_documentation"):
        (build / manifest["files"].pop(logical)).unlink()
    provenances = build / manifest["files"]["provenances"]
    rows = json.loads(provenances.read_text())
    for row in rows:
        del row["licence"]
    provenances.write_text(json.dumps(rows))
    del manifest["arc_tables"], manifest["ingestion"]
    manifest["format_version"] = 3
    manifest["identity_inputs"]["iso_week"] = "2026-W01"
    (build / "manifest.json").write_text(json.dumps(manifest))


def _old_build(tmp_path: Path, capsys) -> tuple[Path, UnMirror]:
    """A format-3 build directory holding the UN tables of a WHO excluding
    `ODA_SERIES`, and those tables."""
    old = tmp_path / "old"
    run(old, registry=_excluding(ODA_SERIES))
    tables = _un_tables(old)
    [build_id] = builds(old)
    build = root(old) / "versions" / build_id
    _as_format_3(build)
    capsys.readouterr()
    return build, tables


def _un_down() -> FakeGraph:
    down = un_deployment()
    down.failing = frozenset({"post", "node_arcs_many", "observation_facets_batch"})
    down.un.failing = True
    return down


def test_carry_un_from_takes_a_format_3_build_s_un_stage_without_reading_the_un(tmp_path, capsys):
    build, tables = _old_build(tmp_path, capsys)
    down = _un_down()

    run(
        tmp_path / "new",
        un_graph=down,
        registry=_excluding(ODA_SERIES),
        carry_un_from=build.as_uri(),
    )

    carried, published = events(capsys)
    assert carried["event"] == "origin_catalogue_job_un_carried_forward"
    assert published["un_mirror"] == "carried"
    assert down.calls == []
    assert down.un.requests == []
    assert _un_tables(tmp_path / "new") == tables
    snapshot = load_catalogue(f"file://{tmp_path / 'new'}")
    assert snapshot.un_blocked_places == {"country/GRC"}
    assert snapshot.tables.provenances[WHO].segment == "who"
    note = snapshot.manifest.un_carried_forward
    assert note is not None
    assert note.from_build == build.name
    assert note.reason == f"--carry-un-from build {build.name}, format 3"


def test_main_carries_the_un_stage_from_a_local_build(tmp_path, monkeypatch, capsys):
    build, tables = _old_build(tmp_path, capsys)

    _main(
        tmp_path,
        monkeypatch,
        builder=FakeBuilder(),
        expected_exit=0,
        un_graph=_un_down(),
        registry=_excluding(ODA_SERIES),
        argv=["--carry-un-from", str(build)],
    )

    assert _un_tables(tmp_path / "out") == tables


def test_carry_un_from_refuses_a_catalogue_uri_that_is_not_local(tmp_path):
    bq = FakeBigQuery(bq_responses())
    un_graph = un_deployment()

    with pytest.raises(catalogue_job.CatalogueBuildError, match="file://"):
        catalogue_job.run(
            catalogue_job.JobSettings(
                artifact_uri="gs://bucket/catalogue",
                bq_connection=_CONNECTION,
                graph_base_url="https://mixer/core/api",
            ),
            bq=bq,
            graph=cast("GraphClient", mixer()),
            un_graph=un_graph,
            un_group_counts=un_graph.un,
            build_base_tables=FakeBuilder(),
            read_base_index=lambda uri: (),
            registry_raw=REGISTRY.encode(),
            carry_un_from=(tmp_path / "old").as_uri(),
        )

    assert bq.queries == []


# --- Lease -------------------------------------------------------------------


def test_a_held_lease_skips_the_run_before_any_graph_read(tmp_path, capsys):
    assert acquire_lease(root(tmp_path).as_uri(), build_id="other", ttl_seconds=3600)
    graph, un_graph, bq = mixer(), un_deployment(), FakeBigQuery(bq_responses())

    run(tmp_path, bq=bq, graph=graph, un_graph=un_graph)

    [skipped] = events(capsys)
    assert skipped == {
        "event": "origin_catalogue_job_skipped",
        "identity": skipped["identity"],
        "reason": "lease_held",
        "severity": "INFO",
    }
    assert graph.calls == un_graph.calls == []
    assert not any("LEFT JOIN" in sql for sql in bq.queries)
    assert json.loads((root(tmp_path) / LEASE_FILE).read_text())["build_id"] == "other"


def test_an_expired_lease_is_reclaimed(tmp_path, capsys):
    lease_root = root(tmp_path).as_uri()
    assert acquire_lease(lease_root, build_id="dead", ttl_seconds=60, now=lambda: 0.0)

    run(tmp_path)

    assert events(capsys)[0]["event"] == "origin_catalogue_job_published"
    assert not (root(tmp_path) / LEASE_FILE).exists()


def test_the_lease_is_released_when_the_build_fails(tmp_path):
    def _fail() -> None:
        raise RuntimeError("mixer down")

    with pytest.raises(RuntimeError, match="mixer down"):
        run(tmp_path, builder=FakeBuilder(during=_fail))

    assert not (root(tmp_path) / LEASE_FILE).exists()


# --- Refusals ----------------------------------------------------------------


def _refused(tmp_path: Path, capsys, **kwargs: Any) -> list[tuple[str, str]]:
    """Runs a build that must be refused; returns each refusal's (kind, subject),
    which the log event and the raised error agree on."""
    with pytest.raises(catalogue_job.CatalogueRefusedError) as raised:
        run(tmp_path, **kwargs)
    [refused] = events(capsys)
    assert refused["event"] == "origin_catalogue_job_refused"
    logged = [(r["kind"], r["subject"]) for r in refused["refusals"]]
    assert logged == [(r.kind, r.subject) for r in raised.value.refusals]
    return logged


def _assert_unpublished(tmp_path: Path) -> None:
    assert not (root(tmp_path) / "current.json").exists()
    assert builds(tmp_path) == []
    assert not (root(tmp_path) / LEASE_FILE).exists()


def test_refuses_a_registered_source_with_no_admitted_variables(tmp_path, capsys):
    tables = base_tables(base_variables={POPULATION: BaseVariable(provenances=(WDI,), places=1)})

    assert _refused(tmp_path, capsys, builder=FakeBuilder(tables)) == [
        ("base_source_without_variables", OECD)
    ]
    _assert_unpublished(tmp_path)


def test_refuses_an_exclusion_its_source_does_not_hold(tmp_path, capsys):
    graph = mixer()
    graph.subjects[WORLD_BANK] = (WDI, IDS)

    assert _refused(tmp_path, capsys, graph=graph) == [("exclusion_not_held", WB_IDS)]
    _assert_unpublished(tmp_path)


def test_refuses_a_dfo_provenance_under_a_source_that_states_no_exclusions(tmp_path, capsys):
    registry = REGISTRY.replace("exclusions = []\n", "")

    assert _refused(tmp_path, capsys, registry=registry) == [
        ("dfo_provenance_under_unreviewed_source", CRS)
    ]
    _assert_unpublished(tmp_path)


def test_refuses_admitted_provenances_without_a_url_hostname(tmp_path, capsys):
    graph = mixer()
    del graph.arcs[OECD_MAIN]["url"]
    graph.arcs[IDS]["url"] = ("worldbank.org/ids",)
    tables = base_tables(
        base_variables={
            POPULATION: BaseVariable(provenances=(WDI,), places=1),
        }
    )

    assert _refused(tmp_path, capsys, graph=graph, builder=FakeBuilder(tables)) == [
        ("provenance_without_url", OECD_MAIN),
        ("provenance_without_url", IDS),
        # OECD's only admitted base provenance was refused, so it holds nothing.
        ("base_source_without_variables", OECD),
    ]
    _assert_unpublished(tmp_path)


@pytest.mark.parametrize("url", [None, "N/A"])
def test_refuses_a_un_provenance_without_a_url_hostname_on_a_first_build(tmp_path, capsys, url):
    un_graph = un_deployment()
    if url is None:
        del un_graph.arcs[WHO]["url"]
    else:
        un_graph.arcs[WHO]["url"] = (url,)

    assert _refused(tmp_path, capsys, un_graph=un_graph) == [("provenance_without_url", WHO)]
    _assert_unpublished(tmp_path)


def test_refuses_a_un_agency_whose_root_group_is_missing(tmp_path, capsys):
    un_graph = un_deployment()
    del un_graph.arcs[WHO_GROUP]

    assert _refused(tmp_path, capsys, un_graph=un_graph) == [("un_agency_root_group_missing", WHO)]
    _assert_unpublished(tmp_path)


def test_refuses_a_dfo_provenance_without_a_source(tmp_path, capsys):
    graph = mixer()
    del graph.arcs[CRS]["source"]

    assert _refused(tmp_path, capsys, graph=graph) == [("dfo_provenance_without_source", CRS)]
    _assert_unpublished(tmp_path)


def test_a_row_without_a_named_parent_takes_its_own_name_and_never_its_dcid(tmp_path, capsys):
    graph = mixer()
    graph.arcs[WDI_DATASET]["name"] = (WDI_DATASET,)
    graph.arcs[OECD_MAIN]["name"] = (OECD_MAIN, "OECD Main Economic Indicators")

    run(tmp_path, graph=graph)

    rows = load_catalogue(f"file://{tmp_path}").tables.provenances
    assert rows[WDI].name == "WorldDevelopmentIndicators"
    assert rows[OECD_MAIN].name == "OECD Main Economic Indicators"
    parent_reads = [c for c in graph.calls if c[0] == "node_arcs_many" and WDI_DATASET in c[1]]
    assert parent_reads == [("node_arcs_many", (CRS_DATASET, WDI_DATASET))]


def test_an_empty_name_is_skipped_for_the_next_one(tmp_path, capsys):
    graph = mixer()
    graph.arcs[WDI_DATASET]["name"] = ("", "World Development Indicators")
    graph.arcs[OECD_MAIN]["name"] = ("", "OECD Main Economic Indicators")

    run(tmp_path, graph=graph)

    rows = load_catalogue(f"file://{tmp_path}").tables.provenances
    assert rows[WDI].dataset == DatasetRef(ref=WDI_DATASET, label="World Development Indicators")
    assert rows[WDI].name == "World Development Indicators"
    assert rows[OECD_MAIN].name == "OECD Main Economic Indicators"


def test_a_row_belongs_to_its_first_parent_typed_dataset(tmp_path, capsys):
    graph = mixer()
    graph.arcs[WORLD_BANK] = {"typeOf": ("Source",), "name": ("World Bank",)}
    graph.arcs[WDI]["isPartOf"] = (WORLD_BANK, WDI_DATASET)
    graph.arcs[OECD_MAIN]["isPartOf"] = ("dc/d/NotHeld",)

    run(tmp_path, graph=graph)

    tables = load_catalogue(f"file://{tmp_path}").tables
    rows = tables.provenances
    assert rows[WDI].dataset == DatasetRef(ref=WDI_DATASET, label="World Development Indicators")
    assert rows[WDI].name == "World Development Indicators"
    # A parent the graph does not hold is no Dataset, so the row is its own entry.
    assert rows[OECD_MAIN].dataset is None
    assert rows[OECD_MAIN].name == OECD_MAIN
    assert set(tables.dataset_facts) == {WDI_DATASET, OECD_MAIN, CRS_DATASET, IDS}


def test_a_dfo_provenance_under_an_unregistered_source_takes_the_graph_label(tmp_path, capsys):
    graph = mixer()
    graph.arcs[IDS]["source"] = ("source/ONEData",)
    graph.arcs["source/ONEData"] = {"name": ("ONE Data",)}

    run(tmp_path, graph=graph)

    assert events(capsys)[0]["event"] == "origin_catalogue_job_published"
    row = load_catalogue(f"file://{tmp_path}").tables.provenances[IDS]
    assert (row.source.ref, row.source.label) == ("source/ONEData", "ONE Data")


# --- Discards and races ------------------------------------------------------


def test_discards_the_build_when_the_import_key_moves(tmp_path, capsys):
    bq = FakeBigQuery(bq_responses())

    def _ingest() -> None:
        bq._responses.update(import_key_responses("wf-99"))

    run(tmp_path, bq=bq, builder=FakeBuilder(during=_ingest))

    [discarded] = events(capsys)
    assert discarded["event"] == "origin_catalogue_job_discarded"
    assert discarded["reason"] == "fingerprint_moved_during_build"
    _assert_unpublished(tmp_path)


def test_discards_the_build_when_the_lock_is_taken(tmp_path, capsys):
    bq = FakeBigQuery(bq_responses())

    def _lock() -> None:
        bq._responses.update(import_key_responses(lock_owner="ingestion"))

    run(tmp_path, bq=bq, builder=FakeBuilder(during=_lock))

    assert events(capsys)[0]["reason"] == "ingestion_lock_acquired_during_build"
    _assert_unpublished(tmp_path)


def test_skips_while_the_ingestion_lock_is_held(tmp_path, capsys):
    graph = mixer()

    run(tmp_path, bq=FakeBigQuery(bq_responses(lock_owner="ingestion")), graph=graph)

    assert events(capsys)[0]["reason"] == "ingestion_lock_held"
    assert graph.calls == []


def test_loses_the_publish_race_cleanly(tmp_path, capsys):
    pointer = root(tmp_path) / "current.json"

    def _other_run_publishes() -> None:
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(CataloguePointer(build_id="other", identity="x").to_json())

    run(tmp_path, builder=FakeBuilder(during=_other_run_publishes))

    assert events(capsys)[0]["event"] == "origin_catalogue_job_lost_publish_race"
    assert json.loads(pointer.read_text())["build_id"] == "other"


# --- Monitoring --------------------------------------------------------------


def _locked() -> FakeBigQuery:
    return FakeBigQuery(bq_responses(lock_owner="ingestion"))


@pytest.mark.parametrize(
    "outcome",
    ["published", "noop", "ingestion_lock_held", "lease_held", "discarded", "lost_publish_race"],
)
def test_every_run_that_returns_logs_one_heartbeat_last(tmp_path, capsys, outcome):
    first = REGISTRY + "\n# first\n"
    run(tmp_path, registry=first)
    pointer = root(tmp_path) / "current.json"
    published = CataloguePointer.from_json(pointer.read_text())
    capsys.readouterr()
    bq = FakeBigQuery(bq_responses())
    kwargs: dict[str, Any] = {}
    match outcome:
        case "noop":
            kwargs["registry"] = first
        case "ingestion_lock_held":
            kwargs["bq"] = _locked()
        case "lease_held":
            assert acquire_lease(root(tmp_path).as_uri(), build_id="other", ttl_seconds=3600)
        case "discarded":
            moved = import_key_responses("wf-99")
            kwargs["bq"] = bq
            kwargs["builder"] = FakeBuilder(during=lambda: bq._responses.update(moved))
        case "lost_publish_race":
            other = CataloguePointer(build_id=published.build_id, identity="other")
            kwargs["builder"] = FakeBuilder(during=lambda: pointer.write_text(other.to_json()))

    run(tmp_path, **kwargs)

    lines = logged(capsys)
    assert [line["event"] for line in lines].count(_HEARTBEAT) == 1
    heartbeat = lines[-1]
    assert heartbeat["event"] == _HEARTBEAT
    current = CataloguePointer.from_json(pointer.read_text())
    assert (current.build_id != published.build_id) == (outcome == "published")
    assert heartbeat["build_id"] == current.build_id
    assert 0 <= heartbeat["age_seconds"] < 60
    assert heartbeat["built_at"].endswith("+00:00")


def test_a_heartbeat_with_nothing_published_names_no_build(tmp_path, capsys):
    run(tmp_path, bq=_locked())

    assert logged(capsys)[-1] == {
        "event": _HEARTBEAT,
        "severity": "INFO",
        "build_id": None,
        "built_at": None,
        "age_seconds": None,
    }


def test_a_refused_run_logs_no_heartbeat(tmp_path, capsys):
    tables = base_tables(base_variables={POPULATION: BaseVariable(provenances=(WDI,), places=1)})

    with pytest.raises(catalogue_job.CatalogueRefusedError):
        run(tmp_path, builder=FakeBuilder(tables))

    assert _HEARTBEAT not in [line["event"] for line in logged(capsys)]


def _stale(capsys) -> list[dict[str, Any]]:
    return [e for e in events(capsys) if e["event"] == "origin_catalogue_publish_stale"]


def test_a_held_lock_reports_the_identity_the_last_run_left_unpublished(tmp_path, capsys):
    """A refused build leaves its identity unpublished. A lock taken after it
    keeps every later run from computing an identity of its own."""
    tables = base_tables(base_variables={POPULATION: BaseVariable(provenances=(WDI,), places=1)})
    clock = Clock()
    with pytest.raises(catalogue_job.CatalogueRefusedError):
        run(tmp_path, builder=FakeBuilder(tables), clock=clock)
    run(tmp_path, bq=_locked(), clock=clock)
    assert _stale(capsys) == []

    clock.offset = 901.0
    run(tmp_path, bq=_locked(), clock=clock)

    [stale] = _stale(capsys)
    assert stale["reason"] == "identity_unpublished"
    assert stale["published_identity"] is None
    assert stale["seconds_behind"] > 900


def test_a_failed_import_publishes_nothing_and_alerts_after_a_cycle(tmp_path, capsys):
    clock = Clock()
    run(tmp_path, clock=clock)
    published = CataloguePointer.from_json((root(tmp_path) / "current.json").read_text())
    capsys.readouterr()
    failed = history_row(
        "wf-2", status="FAILURE", failed=True, created_at=datetime(2026, 9, 25, tzinfo=UTC)
    )

    for offset in (0.0, 901.0):
        clock.offset = offset
        run(tmp_path, bq=FakeBigQuery(bq_responses(newest=failed)), clock=clock)

    found = events(capsys)
    assert [(e["event"], e.get("reason")) for e in found] == [
        ("origin_catalogue_job_skipped", "import_not_successful"),
        ("origin_catalogue_publish_stale", "identity_unpublished"),
        ("origin_catalogue_job_skipped", "import_not_successful"),
    ]
    assert found[1]["published_identity"] == published.identity != found[1]["identity"]
    assert builds(tmp_path) == [published.build_id]


def test_a_held_lock_reports_nothing_while_the_last_identity_is_published(tmp_path, capsys):
    clock = Clock()
    run(tmp_path, clock=clock)
    clock.offset = 3600.0

    run(tmp_path, bq=_locked(), clock=clock)

    assert _stale(capsys) == []


def test_a_published_build_older_than_a_week_and_a_run_is_stale(tmp_path, capsys):
    clock = Clock()
    run(tmp_path, clock=clock)
    capsys.readouterr()
    limit = 7 * 86_400 + 7_200 + 900

    clock.offset = limit - 60
    run(tmp_path, bq=_locked(), clock=clock)
    assert _stale(capsys) == []

    clock.offset = limit + 60
    run(tmp_path, bq=_locked(), clock=clock)

    *_, stale, heartbeat = logged(capsys)
    assert stale["event"] == "origin_catalogue_publish_stale"
    assert stale["reason"] == "published_build_age"
    assert stale["limit_seconds"] == limit
    assert stale["build_id"] == heartbeat["build_id"]
    assert heartbeat["age_seconds"] > limit


# --- Stages ------------------------------------------------------------------


def test_read_dfo_holdings_refuses_an_observation_without_provenance():
    responses = bq_responses()
    responses["LEFT JOIN"].append({"variable": "Orphan", "provenance": None})

    with pytest.raises(catalogue_job.CatalogueBuildError, match="Orphan"):
        catalogue_job.read_dfo_holdings(FakeBigQuery(responses), connection=_CONNECTION)


def test_read_dfo_holdings_aggregates_outside_the_external_query():
    bq = FakeBigQuery(bq_responses())

    holdings = catalogue_job.read_dfo_holdings(bq, connection=_CONNECTION)

    assert holdings.provenances == {CRS, IDS}
    assert holdings.variables == {CRS_VARIABLE, IDS_VARIABLE, POPULATION}
    [sql] = bq.queries
    for inner in sql.split("EXTERNAL_QUERY(")[1:]:
        assert "DISTINCT" not in inner.split('")')[0]


def test_a_truncated_source_listing_fails_the_build(monkeypatch):
    monkeypatch.setattr(catalogue_job, "_MAX_SOURCE_SUBJECTS", 3)

    with pytest.raises(catalogue_job.CatalogueBuildError, match="WorldBank"):
        catalogue_job.build_provenance_table(
            registry=parse_registry(REGISTRY.encode()),
            dfo_provenances=frozenset({CRS, IDS}),
            graph=mixer(),
        )


def test_a_provenance_superseded_by_the_un_leaves_the_table_as_one_superseded_by_dfo_does():
    by_un = REGISTRY.replace(
        'reason = "superseded_by_dfo"',
        f'reason = "superseded_by_un_system", successor = "{WHO}", '
        'variable_prefix = "ids/", code_delimiter = ".", '
        'pair_delimiter = "__", value_delimiter = "--"',
    )
    assert by_un != REGISTRY

    by_dfo_stage, by_un_stage = (
        catalogue_job.build_provenance_table(
            registry=parse_registry(registry.encode()),
            dfo_provenances=frozenset({CRS, IDS}),
            graph=mixer(),
        )
        for registry in (REGISTRY, by_un)
    )

    assert WB_IDS not in by_un_stage.rows
    assert WDI in by_un_stage.rows
    assert by_un_stage == by_dfo_stage


def test_card_facts_label_each_licence_from_one_read_of_its_node():
    graph = mixer()
    graph.arcs[IDS] = {**graph.arcs[IDS], "licenseType": ("SomeLicenseTheGraphDoesNotHold",)}

    stage = catalogue_job.build_provenance_table(
        registry=parse_registry(REGISTRY.encode()),
        dfo_provenances=frozenset({CRS, IDS}),
        graph=graph,
    )

    assert stage.dataset_facts[CRS_DATASET].license_type == "Creative Commons Attribution License"
    # A licence node the graph does not hold serves its dcid.
    assert stage.dataset_facts[IDS].license_type == "SomeLicenseTheGraphDoesNotHold"
    licence_reads = [
        dcids
        for method, dcids in graph.calls
        if method == "node_arcs_many" and "CreativeCommonsAttributionLicense" in dcids
    ]
    assert [sorted(dcids) for dcids in licence_reads] == [
        ["CreativeCommonsAttributionLicense", "SomeLicenseTheGraphDoesNotHold"]
    ]


def test_an_unresolved_licence_name_serves_the_licence_dcid():
    graph = mixer()
    graph.arcs[IDS] = {**graph.arcs[IDS], "licenseType": ("SomeLicense",)}
    graph.arcs["SomeLicense"] = {"name": (UNRESOLVED_NODE,)}

    stage = catalogue_job.build_provenance_table(
        registry=parse_registry(REGISTRY.encode()),
        dfo_provenances=frozenset({CRS, IDS}),
        graph=graph,
    )

    assert stage.dataset_facts[IDS].license_type == "SomeLicense"


def test_provenances_naming_no_licence_cost_no_licence_read():
    graph = mixer()
    graph.arcs[CRS] = {k: v for k, v in graph.arcs[CRS].items() if k != "licenseType"}

    stage = catalogue_job.build_provenance_table(
        registry=parse_registry(REGISTRY.encode()),
        dfo_provenances=frozenset({CRS, IDS}),
        graph=graph,
    )

    assert stage.dataset_facts[CRS_DATASET].license_type is None
    # The provenance, parent and documentation reads are the only node reads.
    node_reads = [dcids for method, dcids in graph.calls if method == "node_arcs_many"]
    assert len(node_reads) == 3
    assert not any("CreativeCommonsAttributionLicense" in dcids for dcids in node_reads)


def test_iso_week_uses_the_iso_year():
    from datetime import UTC, datetime

    assert catalogue_job.iso_week(datetime(2027, 1, 1, tzinfo=UTC)) == "2026-W53"


# --- Settings and entry point ------------------------------------------------


def test_settings_read_their_aliases(monkeypatch):
    monkeypatch.setenv("ORIGIN_CATALOGUE_URI", "file:///tmp/x")
    monkeypatch.setenv("ORIGIN_CATALOGUE_BQ_CONNECTION", _CONNECTION)
    monkeypatch.setenv("DATASETS_GRAPH_BASE_URL", "https://mixer/core/api")
    monkeypatch.setenv("DATASETS_GRAPH_TOKEN", "")
    monkeypatch.delenv("UN_GRAPH_BASE_URL", raising=False)
    monkeypatch.delenv("ORIGIN_CATALOGUE_BQ_PROJECT", raising=False)

    loaded = catalogue_job.JobSettings()

    assert loaded.bq_project == "proj"
    assert loaded.graph_token is None
    assert loaded.cycle_seconds == 900.0
    assert loaded.un_base_url == UN_BASE_URL

    assert loaded.un_site_url == UN_BASE_URL.removesuffix("/core/api")

    monkeypatch.setenv("UN_GRAPH_BASE_URL", "https://un/core/api")
    assert catalogue_job.JobSettings().un_base_url == "https://un/core/api"
    assert catalogue_job.JobSettings().un_site_url == "https://un"

    # An API root the site root cannot be read from needs the site named.
    monkeypatch.setenv("UN_GRAPH_BASE_URL", "https://un/api")
    with pytest.raises(ValueError, match="UN_SITE_BASE_URL"):
        catalogue_job.JobSettings()
    monkeypatch.setenv("UN_SITE_BASE_URL", "https://un-site")
    assert catalogue_job.JobSettings().un_site_url == "https://un-site"


def _main(
    tmp_path: Path,
    monkeypatch,
    *,
    builder: FakeBuilder,
    expected_exit: int,
    un_graph: FakeGraph | None = None,
    registry: str = REGISTRY,
    argv: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Runs `main` over fakes, with a base resolver index naming `GDP`, and returns each graph
    client's options by base URL."""
    base_index = tmp_path / "embeddings.csv"
    base_index.write_text(f"0,dcid,sentence\n0.1,{GDP},Gross domestic product\n")
    for name, value in {
        "ORIGIN_CATALOGUE_BQ_CONNECTION": _CONNECTION,
        "DATASETS_GRAPH_BASE_URL": "https://mixer/core/api",
        "UN_GRAPH_BASE_URL": "https://un/core/api",
        "ORIGIN_CATALOGUE_BASE_INDEX_URI": base_index.as_uri(),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ORIGIN_CATALOGUE_URI", raising=False)
    graphs = {"https://mixer/core/api": mixer(), "https://un/core/api": un_graph or un_deployment()}
    options: dict[str, dict[str, Any]] = {}

    def _client(*, base_url: str, **kwargs: Any) -> FakeGraph:
        options[base_url] = kwargs
        return graphs[base_url]

    monkeypatch.setattr(catalogue_job, "GraphClient", _client)
    monkeypatch.setattr(
        catalogue_job,
        "UnSiteClient",
        lambda *, site_url, **_: (
            options.setdefault("site", {"site_url": site_url}) and graphs["https://un/core/api"].un
        ),
    )
    monkeypatch.setattr(catalogue_job, "LiveBigQuery", lambda **_: FakeBigQuery(bq_responses()))
    monkeypatch.setattr(
        catalogue_job,
        "packaged_registry",
        lambda: types.SimpleNamespace(read_bytes=lambda: registry.encode()),
    )
    monkeypatch.setattr(catalogue_job, "build_base_tables", builder)

    assert catalogue_job.main(["--local-dir", str(tmp_path / "out"), *argv]) == expected_exit
    return options


def test_main_publishes_under_the_local_dir(tmp_path, monkeypatch, capsys):
    options = _main(tmp_path, monkeypatch, builder=FakeBuilder(), expected_exit=0)

    assert events(capsys)[0]["event"] == "origin_catalogue_job_published"
    # The probe reads the UN by the same GET path, under the same budget, as the origin client.
    assert options["https://un/core/api"]["observation_get_url_budget_bytes"] == 7_000
    assert "observation_get_url_budget_bytes" not in options["https://mixer/core/api"]
    # The group counts are read from the UN site the API root sits under.
    assert options["site"] == {"site_url": "https://un"}
    assert load_catalogue((tmp_path / "out").as_uri()).tables.provenances.keys() >= {WDI, WHO}
    assert not (tmp_path / "out" / FORMAT_ROOT / LEASE_FILE).exists()


def test_main_exits_non_zero_on_a_refused_build(tmp_path, monkeypatch, capsys):
    tables = base_tables(base_variables={POPULATION: BaseVariable(provenances=(WDI,), places=1)})

    _main(tmp_path, monkeypatch, builder=FakeBuilder(tables), expected_exit=1)

    assert [e["event"] for e in events(capsys)] == ["origin_catalogue_job_refused"]
    assert not (tmp_path / "out" / FORMAT_ROOT / "current.json").exists()


def test_main_exits_zero_on_an_unchanged_identity_or_a_held_lease(tmp_path, monkeypatch, capsys):
    _main(tmp_path, monkeypatch, builder=FakeBuilder(), expected_exit=0)
    _main(tmp_path, monkeypatch, builder=FakeBuilder(), expected_exit=0)
    assert events(capsys)[-1]["event"] == "origin_catalogue_job_noop"

    (tmp_path / "out" / FORMAT_ROOT / "current.json").unlink()
    assert acquire_lease((tmp_path / "out" / FORMAT_ROOT).as_uri(), build_id="b", ttl_seconds=60)
    _main(tmp_path, monkeypatch, builder=FakeBuilder(), expected_exit=0)
    assert events(capsys)[-1]["reason"] == "lease_held"


def test_main_reports_an_enumeration_failure(tmp_path, monkeypatch, capsys):
    def _fail() -> None:
        raise EnumerationError("pass 2 ran past its deadline")

    _main(tmp_path, monkeypatch, builder=FakeBuilder(during=_fail), expected_exit=1)

    [failed] = events(capsys)
    assert failed["event"] == "origin_catalogue_job_failed"
    assert "deadline" in failed["error"]
    assert not (tmp_path / "out" / FORMAT_ROOT / "current.json").exists()


def test_main_reports_a_un_read_failure_with_nothing_published(tmp_path, monkeypatch, capsys):
    un_graph = un_deployment()
    un_graph.failing = frozenset({"node_arcs_many"})

    _main(tmp_path, monkeypatch, builder=FakeBuilder(), expected_exit=1, un_graph=un_graph)

    [failed] = events(capsys)
    assert failed["event"] == "origin_catalogue_job_failed"
    assert "UN deployment read failed" in failed["error"]
