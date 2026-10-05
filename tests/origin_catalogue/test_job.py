"""Pins the catalogue job with fakes for BigQuery, both graphs and the base-table builder."""

from __future__ import annotations

import ast
import json
import time
import types
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from artifact_publication.jobs import LEASE_FILE, acquire_lease
from dc_graph import UNRESOLVED_NODE, GraphReadError, SubjectPage
from origin_catalogue import (
    BASE,
    DFO,
    UN_SYSTEM,
    BaseTables,
    BaseVariable,
    CardFacts,
    DatasetRef,
    ScanLevel,
)
from origin_catalogue import job as catalogue_job
from origin_catalogue.artifact import load_catalogue
from origin_catalogue.enumerate import EnumerationError
from origin_catalogue.graph import UN_BASE_URL
from origin_catalogue.registry import parse_registry
from origin_catalogue.schema import FORMAT_ROOT, CataloguePointer
from tests.origin_catalogue.fake_catalogue import spread_provenance_places
from tests.series_coverage.test_job import FakeBigQuery

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
    records every read."""

    def __init__(
        self,
        *,
        subjects: dict[str, tuple[str, ...]] | None = None,
        arcs: dict[str, dict[str, tuple[str, ...]]] | None = None,
        refused: frozenset[str] = frozenset(),
    ) -> None:
        self.subjects = subjects or {}
        self.arcs = arcs or {}
        self.refused = refused
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


def mixer() -> FakeGraph:
    return FakeGraph(
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
                "sourceReleaseFrequency": ("P1Y",),
                "earliestObservationDate": ("1973",),
                "latestObservationDate": ("2024-12-31",),
                "description": ("Aid activity data.",),
            },
            CRS_DATASET: {
                "typeOf": ("Dataset",),
                "name": ("OECD Creditor Reporting System",),
                "url": ("https://www.oecd.org/crs",),
            },
            IDS: _provenance("https://datacatalog.worldbank.org/0038015", source=WORLD_BANK),
            "CreativeCommonsAttributionLicense": {
                "typeOf": ("LicenseTypeEnum",),
                "name": ("Creative Commons Attribution License",),
            },
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


def bq_responses(
    *, row_count: int = 3, lock_owner: str | None = None, extra_crs_variables: int = 0
) -> dict[str, list]:
    return {
        "IngestionLock": [{"LockOwner": lock_owner}],
        "IngestionHistory": [{"WorkflowExecutionID": "wf-1", "CreationTimestamp": "t1"}],
        "COUNT(*) AS row_count": [{"row_count": row_count, "max_ts": "t2"}],
        "LEFT JOIN": [
            {"variable": CRS_VARIABLE, "provenance": CRS},
            {"variable": IDS_VARIABLE, "provenance": IDS},
            *(
                {"variable": f"{CRS_VARIABLE}_{i}", "provenance": CRS}
                for i in range(extra_crs_variables)
            ),
        ],
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
) -> None:
    catalogue_job.run(
        settings(tmp_path),
        bq=bq or FakeBigQuery(bq_responses()),
        graph=graph or mixer(),  # type: ignore[arg-type]
        un_graph=un_graph or un_deployment(),
        build_base_tables=builder or FakeBuilder(),
        registry_raw=registry.encode("utf-8"),
        clock=clock or Clock(),
    )


_HEARTBEAT = "origin_catalogue_job_heartbeat"


def logged(capsys) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def events(capsys) -> list[dict[str, Any]]:
    """The run's events without its heartbeat, which the monitoring tests pin."""
    return [event for event in logged(capsys) if event["event"] != _HEARTBEAT]


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
    assert snapshot.tables.dfo_variables == {CRS_VARIABLE, IDS_VARIABLE}
    assert snapshot.manifest.overlaps == (IDS_VARIABLE,)
    assert set(snapshot.manifest.identity_inputs) == {
        "format_version",
        "job_code",
        "enumeration_code",
        "registry",
        "dfo_fingerprint",
        "iso_week",
    }
    assert snapshot.manifest.identity_inputs["job_code"] == catalogue_job.source_digest(
        catalogue_job.JOB_MODULES
    )
    assert snapshot.manifest.un_carried_forward is None
    assert builder.calls[0]["provenances"] == [OECD_MAIN, WDI]
    assert not (root(tmp_path) / LEASE_FILE).exists()


def test_the_fingerprint_counts_every_observation_row(tmp_path):
    """The catalogue's identity stamps this text's results, so a changed query moves every
    published identity."""
    bq = FakeBigQuery(bq_responses())

    run(tmp_path, bq=bq)

    stats = [sql for sql in bq.queries if "COUNT(*) AS row_count" in sql]
    assert stats
    assert set(stats) == {
        "SELECT COUNT(*) AS row_count, MAX(last_update_timestamp) AS max_ts\n"
        f'FROM EXTERNAL_QUERY("{_CONNECTION}", '
        '"SELECT last_update_timestamp FROM Observation")'
    }


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


def _code(*, job: str, enumeration: str):
    """A `source_digest` answering `enumeration` for the enumeration modules
    and `job` for the job's."""

    def digest(modules: tuple[str, ...]) -> str:
        return enumeration if modules == catalogue_job.ENUMERATION_MODULES else job

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
    assert len(builder.calls) == 1
    assert load_catalogue(f"file://{tmp_path}").manifest.identity_inputs["job_code"] == "job-2"


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
    assert not {"context.py", "service.py", "graph.py"} & set(catalogue_job.JOB_MODULES)
    # The card-fact derivation shapes every build, so a change to it rebuilds.
    assert "card_facts.py" in catalogue_job.JOB_MODULES


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

    run(tmp_path, bq=FakeBigQuery(bq_responses(row_count=4)), un_graph=un_graph)

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert load_catalogue(f"file://{tmp_path}").un_blocked_places == {"country/KEN"}


# --- UN carry-forward --------------------------------------------------------


def _publish_then_fail_the_un(tmp_path, capsys, un_graph: FakeGraph, **kwargs: Any) -> list:
    """Publishes a build, then runs again after an ingestion with `un_graph`;
    returns the second run's events."""
    run(tmp_path)
    capsys.readouterr()
    run(tmp_path, bq=FakeBigQuery(bq_responses(row_count=4)), un_graph=un_graph, **kwargs)
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
    assert snapshot.tables.dfo_variables == {CRS_VARIABLE, IDS_VARIABLE}


def test_a_un_refusal_refuses_even_with_a_published_build(tmp_path, capsys):
    run(tmp_path)
    first = CataloguePointer.from_json((root(tmp_path) / "current.json").read_bytes()).build_id
    capsys.readouterr()
    un_graph = un_deployment()
    del un_graph.arcs[WHO_GROUP]

    refused = _refused(
        tmp_path, capsys, bq=FakeBigQuery(bq_responses(row_count=4)), un_graph=un_graph
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
        bq=FakeBigQuery(bq_responses(row_count=4)),
        un_graph=un_graph,
        registry=REGISTRY.replace('segment = "who"', 'segment = "who2"'),
    )

    assert refused == [("un_carried_segment_mismatch", WHO)]


def test_a_published_carry_forward_retries_the_un_on_an_unchanged_identity(tmp_path, capsys):
    down = un_deployment()
    down.failing = frozenset({"node_arcs_many"})
    _publish_then_fail_the_un(tmp_path, capsys, down)
    carried_build = CataloguePointer.from_json(
        (root(tmp_path) / "current.json").read_bytes()
    ).build_id
    ingested = bq_responses(row_count=4)

    # Still down: the retry publishes nothing new.
    run(tmp_path, bq=FakeBigQuery(ingested), un_graph=down)
    [retry] = events(capsys)
    assert retry["event"] == "origin_catalogue_job_un_retry_failed"
    assert load_catalogue(f"file://{tmp_path}").build_id == carried_build

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
    run(tmp_path, bq=FakeBigQuery(bq_responses(row_count=4)), un_graph=down)
    capsys.readouterr()

    run(tmp_path, bq=FakeBigQuery(bq_responses(row_count=5)), un_graph=down)

    published = events(capsys)[-1]
    assert published["un_carried_from"] == first
    assert load_catalogue(f"file://{tmp_path}").manifest.un_carried_forward.from_build == first


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


# --- Carry-forward -------------------------------------------------------


def test_an_ingestion_carries_the_base_tables_into_a_new_build_directory(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    first = CataloguePointer.from_json((root(tmp_path) / "current.json").read_bytes()).build_id
    capsys.readouterr()

    run(tmp_path, bq=FakeBigQuery(bq_responses(row_count=4)), builder=builder)

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert len(builder.calls) == 1
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


def test_a_new_dfo_variable_carries_the_base_tables_forward(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    capsys.readouterr()

    run(
        tmp_path, bq=FakeBigQuery(bq_responses(row_count=4, extra_crs_variables=1)), builder=builder
    )

    [published] = events(capsys)
    assert published["base_tables"] == "carried"
    assert len(builder.calls) == 1
    assert f"{CRS_VARIABLE}_0" in load_catalogue(f"file://{tmp_path}").tables.dfo_variables


def test_a_new_iso_week_rebuilds_the_base_tables(tmp_path, capsys):
    builder = FakeBuilder()
    clock = Clock()
    run(tmp_path, builder=builder, clock=clock)
    capsys.readouterr()

    clock.offset = 7 * 24 * 3600
    run(tmp_path, builder=builder, clock=clock)

    assert events(capsys)[0]["base_tables"] == "built"
    assert len(builder.calls) == 2


def test_a_new_base_provenance_rebuilds_the_base_tables(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    capsys.readouterr()
    graph = mixer()
    graph.subjects[WORLD_BANK] = (*graph.subjects[WORLD_BANK], "dc/base/WorldBank_New")
    graph.arcs["dc/base/WorldBank_New"] = _provenance("https://data.worldbank.org/new")

    run(tmp_path, bq=FakeBigQuery(bq_responses(row_count=4)), graph=graph, builder=builder)

    assert events(capsys)[0]["base_tables"] == "built"
    assert builder.calls[1]["provenances"] == [OECD_MAIN, "dc/base/WorldBank_New", WDI]


def test_a_new_dfo_provenance_rebuilds_the_base_tables(tmp_path, capsys):
    builder = FakeBuilder()
    run(tmp_path, builder=builder)
    capsys.readouterr()
    new = "provenance/NewDfoDataset"
    graph = mixer()
    graph.arcs[new] = _provenance("https://example.org/new", source="dc/s/NewSource")
    responses = bq_responses(row_count=4)
    responses["LEFT JOIN"].append({"variable": "Amount_New", "provenance": new})

    run(tmp_path, bq=FakeBigQuery(responses), graph=graph, builder=builder)

    assert events(capsys)[0]["base_tables"] == "built"
    assert builder.calls[0]["dfo_provenances"] == [CRS, IDS]
    assert builder.calls[1]["dfo_provenances"] == sorted([CRS, IDS, new])


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


def test_discards_the_build_when_the_fingerprint_moves(tmp_path, capsys):
    bq = FakeBigQuery(bq_responses())

    def _ingest() -> None:
        bq._responses["COUNT(*) AS row_count"] = [{"row_count": 99, "max_ts": "t3"}]

    run(tmp_path, bq=bq, builder=FakeBuilder(during=_ingest))

    [discarded] = events(capsys)
    assert discarded["event"] == "origin_catalogue_job_discarded"
    assert discarded["reason"] == "fingerprint_moved_during_build"
    _assert_unpublished(tmp_path)


def test_discards_the_build_when_the_lock_is_taken(tmp_path, capsys):
    bq = FakeBigQuery(bq_responses())

    def _lock() -> None:
        bq._responses["IngestionLock"] = [{"LockOwner": "ingestion"}]

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
            moved = {"COUNT(*) AS row_count": [{"row_count": 99, "max_ts": "t3"}]}
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
    assert holdings.variables == {CRS_VARIABLE, IDS_VARIABLE}
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
    # The provenance read and the parent read are the only node reads.
    assert [method for method, _ in graph.calls].count("node_arcs_many") == 2


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

    monkeypatch.setenv("UN_GRAPH_BASE_URL", "https://un/core/api")
    assert catalogue_job.JobSettings().un_base_url == "https://un/core/api"


def _main(
    tmp_path: Path,
    monkeypatch,
    *,
    builder: FakeBuilder,
    expected_exit: int,
    un_graph: FakeGraph | None = None,
) -> dict[str, dict[str, Any]]:
    """Runs `main` over fakes and returns each graph client's options by base URL."""
    for name, value in {
        "ORIGIN_CATALOGUE_BQ_CONNECTION": _CONNECTION,
        "DATASETS_GRAPH_BASE_URL": "https://mixer/core/api",
        "UN_GRAPH_BASE_URL": "https://un/core/api",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ORIGIN_CATALOGUE_URI", raising=False)
    graphs = {"https://mixer/core/api": mixer(), "https://un/core/api": un_graph or un_deployment()}
    options: dict[str, dict[str, Any]] = {}

    def _client(*, base_url: str, **kwargs: Any) -> FakeGraph:
        options[base_url] = kwargs
        return graphs[base_url]

    monkeypatch.setattr(catalogue_job, "GraphClient", _client)
    monkeypatch.setattr(catalogue_job, "LiveBigQuery", lambda **_: FakeBigQuery(bq_responses()))
    monkeypatch.setattr(
        catalogue_job,
        "packaged_registry",
        lambda: types.SimpleNamespace(read_bytes=lambda: REGISTRY.encode()),
    )
    monkeypatch.setattr(catalogue_job, "build_base_tables", builder)

    assert catalogue_job.main(["--local-dir", str(tmp_path / "out")]) == expected_exit
    return options


def test_main_publishes_under_the_local_dir(tmp_path, monkeypatch, capsys):
    options = _main(tmp_path, monkeypatch, builder=FakeBuilder(), expected_exit=0)

    assert events(capsys)[0]["event"] == "origin_catalogue_job_published"
    # The probe reads the UN by the same GET path, under the same budget, as the origin client.
    assert options["https://un/core/api"]["observation_get_url_budget_bytes"] == 7_000
    assert "observation_get_url_budget_bytes" not in options["https://mixer/core/api"]
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
