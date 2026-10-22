"""Pins the builder's tick: each exit, the gate's two failure kinds, publication, pruning under
the lease, the backoff, hold and retry, the heartbeat and the stale line.

The tick builds the recorded slice (`tests.graph_release.recorded`) for real. Spanner is a fake
that answers the import-key and ingested-imports statements, with `read_inputs` returning the
recorded `RawInputs` under the key the fake snapshot holds. The Admin API is `CloudRunAdmin`
over an `httpx.MockTransport`; the mixer answers from the slice's own arcs and parties; SDMX
returns each series row's count; the release root is a `file://` directory.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from google.api_core.exceptions import ServiceUnavailable
from google.auth.exceptions import RefreshError
from pydantic import ValidationError

import code_index.export as code_index_export
from artifact_publication.jobs import acquire_lease, read_import_key, release_lease
from artifact_publication.uri import (
    join_uri,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes,
)
from dc_graph.client import SubjectPage
from dc_graph.errors import GraphReadError
from dc_sdmx.results import SdmxDataResult, SdmxObservation
from graph_release import job
from graph_release.format import Manifest, manifest_uri, pointer_uri, read_pointer
from graph_release.gate import (
    MAX_PART_BYTES,
    Dependency,
    GateResult,
    GateUnavailable,
    release_arcs,
    visible_edges,
)
from graph_release.hold import main as hold_main
from graph_release.origin import origin_view
from graph_release.publish import clear_hold, hold_release
from graph_release.serving_revision import CloudRunAdmin, ServingRevisionError
from tests.graph_generation.conftest import FakeCredentials
from tests.graph_release.recorded import (
    MIXER_PARTIES,
    RELEASE_PARTIES,
    SLICE_DECLARATIONS,
    UNHELD_PARTY,
    platform_parts,
    raw_inputs,
)

_SERVICE = "projects/p/locations/r/services/dc"
_CATALOGUE_JOB = "projects/p/locations/r/jobs/search-catalogue"
_OPERATION = "projects/p/locations/r/operations/op-1"
_COMPLETED = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
_NOW = _COMPLETED + timedelta(minutes=10)
_INGESTED = ("imp-a", "imp-b")


def _history(
    execution: str = "wf-1", *, status: str = "SUCCESS", completed: datetime | None = _COMPLETED
) -> tuple[object, ...]:
    return (execution, status, status != "SUCCESS", _COMPLETED, completed)


# The recorded `RawInputs` carry this key; a fake snapshot holding it re-reads it unchanged.
_RECORDED_ROW = _history()


class FakeSnapshot:
    def __init__(self, database: FakeDatabase) -> None:
        self._database = database

    def execute_sql(self, sql, params=None, param_types=None, *, request_options=None):
        database = self._database
        database.statements.append(sql)
        if sql.startswith("SELECT LockOwner"):
            return [] if database.lock_owner is None else [(database.lock_owner,)]
        if sql.startswith("SELECT IngestedImports"):
            assert params == {"execution": "wf-1"}
            return [(list(_INGESTED),)]
        if "Status = 'SUCCESS'" in sql:
            return [] if database.success is None else [database.success]
        if "FROM IngestionHistory" in sql:
            return [] if database.newest is None else [database.newest]
        raise AssertionError(f"unexpected statement: {sql}")


class FakeDatabase:
    """Holds the import workflow's rows; `snapshots` records each snapshot's options."""

    def __init__(self) -> None:
        self.lock_owner: str | None = None
        self.newest: tuple[object, ...] | None = _RECORDED_ROW
        self.success: tuple[object, ...] | None = _RECORDED_ROW
        self.error: Exception | None = None
        self.snapshots: list[dict[str, object]] = []
        self.statements: list[str] = []

    @contextmanager
    def snapshot(self, **options: object) -> Iterator[FakeSnapshot]:
        self.snapshots.append(options)
        if self.error is not None:
            raise self.error
        yield FakeSnapshot(self)


class FakeAdminApi:
    """The Admin API: serves `revisions` in turn (the last repeats), each created at
    `created`, and records each job run in `runs`, answering it with `run_status`."""

    def __init__(self) -> None:
        self.revisions = ["dc-00001"]
        self.created = _COMPLETED + timedelta(minutes=5)
        self.status = 200
        self.requests: list[str] = []
        self.run_status = 200
        self.runs: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.runs.append(request.url.path.removeprefix("/v2/"))
            if self.run_status != 200:
                return httpx.Response(self.run_status, text="denied")
            return httpx.Response(200, json={"name": _OPERATION})
        path = request.url.path.removeprefix(f"/v2/{_SERVICE}")
        self.requests.append(path)
        if self.status != 200:
            return httpx.Response(self.status, text="unavailable")
        if path == "":
            revision = self.revisions.pop(0) if len(self.revisions) > 1 else self.revisions[0]
            return httpx.Response(
                200, json={"trafficStatuses": [{"revision": revision, "percent": 100}]}
            )
        return httpx.Response(
            200, json={"createTime": self.created.isoformat().replace("+00:00", "Z")}
        )

    def client(self) -> CloudRunAdmin:
        return CloudRunAdmin(
            service=_SERVICE,
            transport=httpx.MockTransport(self.handle),
            credentials_factory=FakeCredentials,
        )


_PARTIES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    **{
        party: {"name": (name,), "typeOf": (kind,)}
        for party, (name, kind) in RELEASE_PARTIES.items()
    },
    **MIXER_PARTIES,
}


class FakeMixer:
    """Answers the slice's parties and each variable's arcs as the release holds them, and no
    node for the dcids in `missing`. Its `->*` and reverse reads serve the release's own
    arcs, and its arc and reverse reads the arcs of the `extra` variables too."""

    def __init__(self) -> None:
        parts = platform_parts()
        self._edges = visible_edges(parts.edges)
        self._nodes = parts.nodes
        self.error: Exception | None = None
        self.missing: set[str] = set()
        self.extra: dict[str, dict[str, tuple[str, ...]]] = {}

    def node_arcs_many(self, dcids: Sequence[str], *, properties: Sequence[str]):
        if self.error is not None:
            raise self.error
        arcs = release_arcs(self._edges, self._nodes, list(dcids))
        answers: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            found = (
                _PARTIES.get(dcid)
                or self.extra.get(dcid)
                or {prop: tuple(sorted(names)) for prop, names in arcs.get(dcid, {}).items()}
            )
            answers[dcid] = (
                {prop: tuple(found.get(prop, ())) for prop in properties}
                if found and dcid not in self.missing
                else None
            )
        return answers

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
    ) -> SubjectPage:
        if self.error is not None:
            raise self.error
        subjects = self._holding(property, dcid)
        for name, value in (filters or {}).items():
            subjects &= self._holding(name, value)
        return SubjectPage(dcids=tuple(sorted(subjects)), truncated=False)

    def _holding(self, predicate: str, target: str) -> set[str]:
        extra = {dcid for dcid, arcs in self.extra.items() if target in arcs.get(predicate, ())}
        return set(_subjects(self._edges, predicate, target)) | extra

    def node_arcs_batch(self, dcids: list[str]) -> dict[str, dict | None]:
        if self.error is not None:
            raise self.error
        arcs = release_arcs(self._edges, self._nodes, list(dcids))
        return {
            dcid: {
                prop: {"nodes": [{"dcid": name} for name in sorted(names)]}
                for prop, names in arcs[dcid].items()
            }
            if dcid in arcs
            else None
            for dcid in dcids
        }


def _subjects(edges: pa.Table, predicate: str, target: str) -> list[str]:
    rows = edges.filter(
        pc.and_(pc.equal(edges["predicate"], predicate), pc.equal(edges["object_id"], target))
    )
    return rows["subject_id"].to_pylist()


class FakeSdmx:
    """Returns `observation_count` rows for each series row, plus `extra`."""

    def __init__(self) -> None:
        self.extra = 0
        self._counts = {}
        for row in platform_parts().series.to_pylist():
            entities = tuple(e for e in (row["entity1"], row["entity2"], row.get("entity3")) if e)
            self._counts[(row["variable_measured"], entities, row["facet_id"])] = row[
                "observation_count"
            ]

    def data(self, *, variables, entities, facet_ids=None) -> SdmxDataResult:
        assert facet_ids is not None
        (variable,), (facet,) = variables, facet_ids
        count = self._counts[(variable, tuple(v for (v,) in entities.values()), facet)] + self.extra
        row = SdmxObservation(
            variable=variable,
            role_values={},
            unit=None,
            measurement_method=None,
            observation_period="P1Y",
            provenance=None,
            time_period="2020",
            value="1",
            scaling_factor=None,
            facet_id=facet,
        )
        return SdmxDataResult(roles=tuple(entities), rows=(row,) * count)


@dataclasses.dataclass
class Builder:
    root: str
    database: FakeDatabase
    admin: FakeAdminApi
    mixer: FakeMixer
    sdmx: FakeSdmx
    capsys: pytest.CaptureFixture[str]
    stamp: str
    now: datetime = _NOW
    catalogue_job: str = ""
    max_part_bytes: int = MAX_PART_BYTES

    def settings(self) -> job.JobSettings:
        return job.JobSettings(
            release_uri=self.root,
            spanner_project="p",
            spanner_instance="i",
            spanner_database="d",
            datacommons_service=_SERVICE,
            graph_base_url="http://graph.invalid/core/api",
            sdmx_base_url="http://graph.invalid/core/api",
            search_catalogue_job=self.catalogue_job,
            max_part_bytes=self.max_part_bytes,
        )

    def run(self) -> None:
        job.run(
            self.settings(),
            database=self.database,
            admin=self.admin.client(),
            graph=self.mixer,
            sdmx=self.sdmx,
            declarations=SLICE_DECLARATIONS,
            clock=self.now.timestamp,
        )

    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.capsys.readouterr().out.splitlines()]

    def tick(self) -> list[dict]:
        """Runs one tick and returns its events, checking the heartbeat closes it with the
        tick's outcome."""
        self.capsys.readouterr()
        self.run()
        events = self.events()
        assert events[-1]["event"] == "graph_release_heartbeat"
        assert sum(e["event"] == "graph_release_heartbeat" for e in events) == 1
        return events

    def state(self) -> dict:
        try:
            return json.loads(read_uri_bytes(join_uri(self.root, job.STATE_FILE)))
        except FileNotFoundError:
            return {}

    def lease_free(self) -> bool:
        return read_uri_bytes_and_generation(join_uri(self.root, "lease.json")) is None


def _outcome(events: list[dict]) -> str:
    return events[-1]["outcome"]


def _named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event["event"] == name]


@pytest.fixture
def builder(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch) -> Builder:
    def read_recorded(snapshot):
        return dataclasses.replace(raw_inputs(), import_key=read_import_key(snapshot))

    monkeypatch.setattr(job, "read_inputs", read_recorded)
    with FakeDatabase().snapshot() as snapshot:
        key = read_import_key(snapshot)
    assert key is not None
    return Builder(
        root=tmp_path.joinpath("releases").as_uri(),
        database=FakeDatabase(),
        admin=FakeAdminApi(),
        mixer=FakeMixer(),
        sdmx=FakeSdmx(),
        capsys=capsys,
        stamp=key.stamp,
    )


# ---------------------------------------------------------------- publish and step 3


def test_a_tick_publishes_the_release_and_the_next_exits_at_step_three(builder):
    events = builder.tick()

    assert _outcome(events) == "published"
    (published,) = _named(events, "graph_release_published")
    pointer = read_pointer(builder.root)
    assert pointer is not None and pointer.release_id == published["release_id"]
    manifest = Manifest.from_json(read_uri_bytes(manifest_uri(builder.root, pointer.release_id)))
    assert manifest.import_execution_id == "wf-1"
    assert manifest.import_completed_at == _COMPLETED.isoformat(timespec="microseconds")
    assert manifest.ingested_imports == _INGESTED
    assert manifest.dc_serving_revision == "dc-00001"
    assert manifest.declarations_hash == SLICE_DECLARATIONS.declarations_hash()
    assert manifest.gates["findings"] == [] and manifest.gates["unavailable"] == []
    assert published["rows"]["series"] == platform_parts().series.num_rows
    assert set(published["phases"]) == {"reads", "platform", "dfo", "gate", "publish"}
    assert all(phase["peak_rss_bytes"] > 0 for phase in published["phases"].values())
    # Step 1's key read and the build's read each take one multi-use snapshot.
    assert len(builder.database.snapshots) == 2
    assert builder.database.snapshots[0] == {"multi_use": True}
    assert builder.database.snapshots[1]["multi_use"] is True
    last_built = builder.state()["last_built"]
    assert last_built["inputs"]["import_execution_id"] == "wf-1"
    assert last_built["release_id"] == pointer.release_id
    assert builder.lease_free()
    assert builder.admin.runs == []

    events = builder.tick()

    assert _outcome(events) == "inputs_built"
    assert len(builder.database.snapshots) == 3
    assert read_pointer(builder.root) == pointer


def test_a_build_of_the_serving_release_records_its_inputs_and_publishes_nothing(
    builder, monkeypatch
):
    builder.tick()
    before = read_uri_bytes_and_generation(pointer_uri(builder.root))
    # A builder change that changes no part gives new inputs and the same release id.
    monkeypatch.setattr(job, "builder_code_hash", lambda: "another-builder")

    events = builder.tick()

    assert _outcome(events) == "release_unchanged"
    (unchanged,) = _named(events, "graph_release_unchanged")
    assert set(unchanged["phases"]) == {"reads", "platform", "dfo", "gate"}
    assert read_uri_bytes_and_generation(pointer_uri(builder.root)) == before
    assert builder.state()["last_built"]["inputs"]["builder_code_hash"] == "another-builder"
    assert _outcome(builder.tick()) == "inputs_built"


def test_only_a_publish_starts_the_search_catalogue_job(builder, monkeypatch):
    builder.catalogue_job = _CATALOGUE_JOB

    events = builder.tick()

    assert _outcome(events) == "published"
    (started,) = _named(events, "graph_release_catalogue_started")
    assert (started["job"], started["operation"]) == (_CATALOGUE_JOB, _OPERATION)
    assert builder.admin.runs == [f"{_CATALOGUE_JOB}:run"]

    assert _outcome(builder.tick()) == "inputs_built"
    monkeypatch.setattr(job, "builder_code_hash", lambda: "another-builder")
    assert _outcome(builder.tick()) == "release_unchanged"
    assert builder.admin.runs == [f"{_CATALOGUE_JOB}:run"]


def test_a_failed_catalogue_start_is_logged_and_the_tick_still_publishes(builder):
    builder.catalogue_job = _CATALOGUE_JOB
    builder.admin.run_status = 403

    events = builder.tick()

    assert _outcome(events) == "published"
    (failed,) = _named(events, "graph_release_catalogue_trigger_failed")
    assert (failed["severity"], failed["job"]) == ("WARNING", _CATALOGUE_JOB)
    assert _named(events, "graph_release_catalogue_started") == []
    assert read_pointer(builder.root) is not None


@pytest.mark.parametrize(
    ("package", "module"),
    [
        ("code_index", "loader.py"),
        ("code_index", "query.py"),
        ("code_index", "types.py"),
        ("series_coverage", "index.py"),
        ("series_coverage", "loader.py"),
        ("series_coverage", "schema.py"),
    ],
)
def test_a_change_to_a_loader_the_view_gate_applies_moves_the_builder_hash(
    monkeypatch, package, module
):
    """The gate refuses a release whose views these modules refuse, so a change to one must
    build the last failed inputs again."""
    baseline = job.builder_code_hash()
    files = job.resources.files

    class Edited:
        def __init__(self, path):
            self.path = path

        def read_bytes(self):
            return self.path.read_bytes() + b"\n# edited\n"

    class Package:
        def __init__(self, name):
            self.name, self.root = name, files(name)

        def joinpath(self, name):
            path = self.root.joinpath(name)
            return Edited(path) if (self.name, name) == (package, module) else path

    monkeypatch.setattr(job.resources, "files", Package)
    job.builder_code_hash.cache_clear()
    try:
        assert job.builder_code_hash() != baseline
    finally:
        job.builder_code_hash.cache_clear()


def test_a_held_pointer_stops_the_tick_before_it_builds(builder):
    builder.tick()
    pointer = read_pointer(builder.root)
    assert pointer is not None
    hold_release(builder.root, pointer.release_id)
    builder.admin.revisions = ["dc-00002"]

    events = builder.tick()

    assert _outcome(events) == "held"
    assert len(builder.database.snapshots) == 3


def test_a_held_pointer_raises_no_stale_and_clearing_it_republishes_the_newest_build(builder):
    builder.tick()
    first = read_pointer(builder.root)
    builder.admin.revisions = ["dc-00002"]
    assert _outcome(builder.tick()) == "published"
    newest = read_pointer(builder.root)
    assert first is not None and newest is not None and newest != first
    hold_release(builder.root, first.release_id)
    # A failed import 31 minutes ago would raise stale, but the hold is a deliberate rollback.
    builder.database.newest = _history("wf-2", status="FAILURE")
    builder.now = _COMPLETED + timedelta(minutes=31)

    events = builder.tick()

    assert _outcome(events) == "import_not_successful"
    assert _named(events, "graph_release_stale") == []

    clear_hold(builder.root)
    builder.database.newest = _RECORDED_ROW

    events = builder.tick()

    assert _outcome(events) == "published"
    assert read_pointer(builder.root) == newest
    assert _outcome(builder.tick()) == "inputs_built"


def test_a_held_lease_stops_the_tick(builder):
    acquire_lease(builder.root, build_id="other", ttl_seconds=600, now=builder.now.timestamp)

    events = builder.tick()

    assert _outcome(events) == "lease_held"
    assert _named(events, "graph_release_unavailable") == []
    assert read_pointer(builder.root) is None


# ---------------------------------------------------------------- steps 1 and 2


@pytest.mark.parametrize(
    ("change", "outcome"),
    [
        pytest.param({"newest": None, "success": None}, "no_import", id="no_history"),
        pytest.param({"lock_owner": "ingestion-wf-2"}, "ingestion_lock_held", id="lock_held"),
        pytest.param(
            {"newest": _history("wf-2", status="FAILURE")}, "import_not_successful", id="failed"
        ),
        pytest.param(
            {"newest": _history("wf-2", status="RUNNING", completed=None)},
            "import_not_successful",
            id="running",
        ),
    ],
)
def test_step_one_exits_unless_the_import_is_buildable(builder, change, outcome):
    for name, value in change.items():
        setattr(builder.database, name, value)

    events = builder.tick()

    assert _outcome(events) == outcome
    assert builder.admin.requests == []
    assert builder.database.snapshots == [{"multi_use": True}]


def test_step_two_waits_for_a_revision_created_after_the_import(builder):
    builder.admin.created = _COMPLETED - timedelta(seconds=1)

    events = builder.tick()

    assert _outcome(events) == "serving_revision_before_import"
    assert len(builder.database.snapshots) == 1


def test_the_oldest_serving_revision_decides_step_two() -> None:
    created = {"a": "2026-10-19T06:05:00Z", "b": "2026-10-19T05:00:00.123456Z"}

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix(f"/v2/{_SERVICE}")
        if path == "":
            statuses = [{"revision": "b", "percent": 10}, {"revision": "a", "percent": 90}]
            return httpx.Response(200, json={"trafficStatuses": statuses})
        return httpx.Response(200, json={"createTime": created[path.rsplit("/", 1)[-1]]})

    admin = CloudRunAdmin(
        service=_SERVICE, transport=httpx.MockTransport(handle), credentials_factory=FakeCredentials
    )
    serving = admin.serving()
    assert (serving.revisions, serving.token) == (("a", "b"), "a+b")
    assert admin.created_at("b") == datetime(2026, 10, 19, 5, 0, 0, 123456, tzinfo=UTC)

    def no_time(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"createTime": "2026-10-19T05:00:00"})

    naive = CloudRunAdmin(
        service=_SERVICE,
        transport=httpx.MockTransport(no_time),
        credentials_factory=FakeCredentials,
    )
    with pytest.raises(ServingRevisionError):
        naive.created_at("b")


# ---------------------------------------------------------------- gate outcomes


def test_a_finding_records_the_failed_inputs_and_the_next_tick_skips_them(builder):
    builder.sdmx.extra = 1

    events = builder.tick()

    assert _outcome(events) == "gate_failed"
    (failed,) = _named(events, "graph_release_gate_failed")
    assert [f["code"] for f in failed["findings"]] == ["sdmx_count_mismatch"]
    assert set(failed["phases"]) == {"reads", "platform", "dfo", "gate"}
    assert read_pointer(builder.root) is None
    assert builder.state()["last_failed"]["import_execution_id"] == "wf-1"
    assert builder.lease_free()

    events = builder.tick()

    assert _outcome(events) == "inputs_failed"
    assert len(builder.database.snapshots) == 3


def test_a_variable_sharing_a_release_five_tuple_publishes_with_a_notice(builder):
    parts = platform_parts()
    variable = min(origin_view(series=parts.series, edges=parts.edges, nodes=parts.nodes).arcs)
    arcs = release_arcs(visible_edges(parts.edges), parts.nodes, [variable])[variable]
    builder.mixer.extra = {
        "Base_Copy": {prop: tuple(sorted(names)) for prop, names in arcs.items()}
    }

    events = builder.tick()

    assert _outcome(events) == "published"
    (published,) = _named(events, "graph_release_published")
    assert [n["code"] for n in published["notices"]] == ["release_family_shared_with_base"]
    assert published["notices"][0]["examples"] == ["Base_Copy"]
    assert published["blind_spots"]["shared"] == 1


def test_a_release_past_the_size_ceiling_fails_the_gate(builder):
    builder.max_part_bytes = 1

    events = builder.tick()

    assert _outcome(events) == "gate_failed"
    (failed,) = _named(events, "graph_release_gate_failed")
    assert [f["code"] for f in failed["findings"]] == ["release_over_size_ceiling"]
    assert read_pointer(builder.root) is None


def _inconsistent_facets(builder: Builder, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = raw_inputs()
    facets = raw.facets.set_column(
        2, "facet_max", pa.array(["{}"] * raw.facets.num_rows, pa.string())
    )

    def read_inconsistent(snapshot):
        return dataclasses.replace(raw, import_key=read_import_key(snapshot), facets=facets)

    monkeypatch.setattr(job, "read_inputs", read_inconsistent)


def test_a_platform_refusal_is_a_finding(builder, monkeypatch):
    _inconsistent_facets(builder, monkeypatch)

    events = builder.tick()

    assert _outcome(events) == "gate_failed"
    (failed,) = _named(events, "graph_release_gate_failed")
    assert [f["code"] for f in failed["findings"]] == ["inconsistent_facet"]
    assert "last_failed" in builder.state()


def test_a_code_index_the_loader_refuses_is_a_gate_finding(builder, monkeypatch):
    """A built code index whose two code rows share a dcid stages, and the gate refuses it as
    `code_index_unloadable`, as for any document `parse_index` refuses."""
    export_graph_blocks = code_index_export.export_graph_blocks

    def duplicated_code(*args, **kwargs):
        blocks = export_graph_blocks(*args, **kwargs)
        blocks["codes"].append(dict(blocks["codes"][0]))
        return blocks

    monkeypatch.setattr(code_index_export, "export_graph_blocks", duplicated_code)

    events = builder.tick()

    assert _outcome(events) == "gate_failed"
    (failed,) = _named(events, "graph_release_gate_failed")
    assert "code_index_unloadable" in [f["code"] for f in failed["findings"]]
    assert read_pointer(builder.root) is None
    assert "last_failed" in builder.state()


def _unreadable_bucket(error: Exception):
    def unreadable(builder: Builder, monkeypatch: pytest.MonkeyPatch) -> None:
        def read(uri):
            raise error

        monkeypatch.setattr(job, "read_uri_bytes_and_generation", read)

    return unreadable


# An unreadable bucket fails the tick and then the stale check, so it logs two lines.
@pytest.mark.parametrize(
    ("break_it", "dependency", "lines"),
    [
        pytest.param(
            lambda b, m: setattr(b.mixer, "error", GraphReadError("connection reset")),
            "mixer",
            1,
            id="mixer",
        ),
        pytest.param(
            lambda b, m: b.mixer.missing.add(UNHELD_PARTY), "mixer", 1, id="mixer_without_a_party"
        ),
        pytest.param(lambda b, m: setattr(b.admin, "status", 503), "admin_api", 1, id="admin_api"),
        pytest.param(
            lambda b, m: setattr(b.database, "error", ServiceUnavailable("spanner down")),
            "spanner",
            1,
            id="spanner",
        ),
        pytest.param(
            lambda b, m: setattr(b.database, "error", RefreshError("token refresh failed")),
            "spanner",
            1,
            id="spanner_credentials",
        ),
        pytest.param(_unreadable_bucket(OSError("bucket down")), "bucket", 2, id="bucket"),
        pytest.param(
            _unreadable_bucket(RefreshError("token refresh failed")),
            "bucket",
            2,
            id="bucket_credentials",
        ),
    ],
)
def test_an_unavailable_dependency_records_nothing(
    builder, monkeypatch, break_it, dependency, lines
):
    break_it(builder, monkeypatch)

    events = builder.tick()

    assert _outcome(events) == "unavailable"
    unavailable = _named(events, "graph_release_unavailable")
    assert [u["dependency"] for e in unavailable for u in e["unavailable"]] == [dependency] * lines
    assert read_pointer(builder.root) is None
    assert "last_failed" not in builder.state()
    assert builder.lease_free()


def test_inputs_that_ended_unavailable_back_off_for_ten_minutes(builder):
    builder.mixer.error = GraphReadError("connection reset")

    events = builder.tick()

    assert _outcome(events) == "unavailable"
    (unavailable,) = _named(events, "graph_release_unavailable")
    assert set(unavailable["phases"]) == {"reads", "platform", "dfo"}
    builder.mixer.error = None
    builder.now += timedelta(seconds=job.UNAVAILABLE_BACKOFF_SECONDS - 1)

    events = builder.tick()

    assert _outcome(events) == "unavailable_backoff"
    assert len(builder.database.snapshots) == 3
    (repeated,) = _named(events, "graph_release_unavailable")
    assert repeated["backoff"] is True
    assert repeated["unavailable"] == unavailable["unavailable"]

    # The retry build holds the lease for minutes; the ticks that overlap it keep the line.
    builder.now += timedelta(seconds=1)
    retry = acquire_lease(
        builder.root, build_id="retry", ttl_seconds=600, now=builder.now.timestamp
    )
    assert retry is not None

    events = builder.tick()

    assert _outcome(events) == "lease_held"
    (overlapping,) = _named(events, "graph_release_unavailable")
    assert overlapping["backoff"] is True
    assert overlapping["unavailable"] == unavailable["unavailable"]

    release_lease(builder.root, retry)
    assert _outcome(builder.tick()) == "published"


def test_an_unclassified_error_logs_an_error_heartbeat_and_propagates(builder, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("declarations hold no datasets")

    monkeypatch.setattr(job, "build_dfo_parts", broken)
    builder.capsys.readouterr()

    with pytest.raises(ValueError, match="declarations"):
        builder.run()

    events = builder.events()
    assert [e["event"] for e in events].count("graph_release_heartbeat") == 1
    assert (events[-1]["outcome"], events[-1]["error"]) == ("error", "ValueError")
    assert builder.lease_free()


def test_a_stale_check_that_cannot_read_is_logged_when_the_tick_already_failed(
    builder, monkeypatch
):
    builder.mixer.error = GraphReadError("connection reset")

    def unreadable(root, key, *, now):
        raise job._Unavailable(GateUnavailable(Dependency.BUCKET, "pointer unreadable"))

    monkeypatch.setattr(job, "_track_stale", unreadable)

    events = builder.tick()

    assert _outcome(events) == "unavailable"
    logged = [
        [u["dependency"] for u in event["unavailable"]]
        for event in _named(events, "graph_release_unavailable")
    ]
    assert logged == [["mixer"], ["bucket"]]


# ---------------------------------------------------------------- moved inputs


def test_a_key_that_moved_before_the_build_snapshot_discards_the_build(builder, monkeypatch):
    def moved(snapshot):
        key = read_import_key(snapshot)
        assert key is not None
        return dataclasses.replace(
            raw_inputs(), import_key=dataclasses.replace(key, lock_owner="wf-2")
        )

    monkeypatch.setattr(job, "read_inputs", moved)

    events = builder.tick()

    assert _outcome(events) == "import_moved"
    assert read_pointer(builder.root) is None
    assert builder.state() == {}
    assert builder.lease_free()


def _sdmx_unavailable(builder: Builder, monkeypatch: pytest.MonkeyPatch) -> None:
    unavailable = GateUnavailable(Dependency.SDMX, "/data failed")
    monkeypatch.setattr(job, "run_gate", lambda staged, **_: GateResult(unavailable=(unavailable,)))


# Whatever the build decided, a moved revision records nothing, so the next tick at the
# original revision builds the same inputs again.
@pytest.mark.parametrize(
    ("break_it", "rebuilt"),
    [
        pytest.param(lambda b, m: None, "published", id="clean"),
        pytest.param(lambda b, m: setattr(b.sdmx, "extra", 1), "gate_failed", id="finding"),
        pytest.param(_inconsistent_facets, "gate_failed", id="platform_refusal"),
        pytest.param(_sdmx_unavailable, "unavailable", id="unavailable"),
        pytest.param(
            lambda b, m: setattr(b.mixer, "error", GraphReadError("connection reset")),
            "unavailable",
            id="mixer_unavailable_mid_build",
        ),
    ],
)
def test_a_serving_revision_that_moves_during_the_build_discards_it(
    builder, monkeypatch, break_it, rebuilt
):
    break_it(builder, monkeypatch)
    builder.admin.revisions = ["dc-00001", "dc-00002"]

    events = builder.tick()

    assert _outcome(events) == "serving_revision_moved"
    assert _named(events, "graph_release_gate_failed") == []
    assert _named(events, "graph_release_unavailable") == []
    assert read_pointer(builder.root) is None
    assert builder.state() == {}
    assert builder.lease_free()

    builder.admin.revisions = ["dc-00001"]
    assert _outcome(builder.tick()) == rebuilt


def test_an_admin_api_that_cannot_recheck_an_unavailable_build_joins_its_line(builder, monkeypatch):
    def unreadable(dcids, *, properties):
        builder.admin.status = 503
        raise GraphReadError("connection reset")

    monkeypatch.setattr(builder.mixer, "node_arcs_many", unreadable)

    events = builder.tick()

    assert _outcome(events) == "unavailable"
    (unavailable,) = _named(events, "graph_release_unavailable")
    assert [u["dependency"] for u in unavailable["unavailable"]] == ["mixer", "admin_api"]


# ---------------------------------------------------------------- lease and prune


def test_pruning_runs_under_the_lease_and_the_lease_is_released_after(builder, monkeypatch):
    calls = []
    prune, release = job.prune_releases, job.release_lease

    def pruning(root, *, now):
        calls.append(("prune", builder.lease_free()))
        return prune(root, now=now)

    def releasing(root, lease):
        calls.append(("release", builder.lease_free()))
        return release(root, lease)

    monkeypatch.setattr(job, "prune_releases", pruning)
    monkeypatch.setattr(job, "release_lease", releasing)

    assert _outcome(builder.tick()) == "published"
    assert calls == [("prune", False), ("release", False)]
    assert builder.lease_free()


@pytest.mark.parametrize("failing", ["prune_releases", "release_lease"])
def test_a_failed_prune_or_lease_release_leaves_the_outcome_published(
    builder, monkeypatch, failing
):
    def fail(*args, **kwargs):
        raise OSError("bucket down")

    monkeypatch.setattr(job, failing, fail)

    events = builder.tick()

    assert _outcome(events) == "published"
    assert _named(events, "graph_release_unavailable") == []
    assert len(_named(events, "graph_release_prune_failed")) == (failing == "prune_releases")


def test_a_failed_state_write_after_publishing_leaves_the_outcome_published(builder, monkeypatch):
    update = job._update_state

    def failing(root, **fields):
        if "last_built" in fields:
            raise OSError("bucket down")
        update(root, **fields)

    monkeypatch.setattr(job, "_update_state", failing)

    events = builder.tick()

    assert _outcome(events) == "published"
    assert len(_named(events, "graph_release_published")) == 1
    assert len(_named(events, "graph_release_state_write_failed")) == 1
    assert _named(events, "graph_release_unavailable") == []
    assert builder.state() == {}

    monkeypatch.setattr(job, "_update_state", update)

    assert _outcome(builder.tick()) == "release_unchanged"
    assert _outcome(builder.tick()) == "inputs_built"


def test_the_gate_reads_only_the_previous_release_edges(builder, monkeypatch):
    builder.tick()
    opened = []
    gate = job.run_gate

    def recording(staged, *, previous, **kwargs):
        opened.append(set(previous.tables) | set(previous.blobs))
        return gate(staged, previous=previous, **kwargs)

    monkeypatch.setattr(job, "run_gate", recording)
    builder.admin.revisions = ["dc-00002"]

    assert _outcome(builder.tick()) == "published"
    assert opened == [{"edges"}]


def test_hold_retry_lets_the_next_tick_build_the_failed_inputs_again(
    builder, tmp_path, monkeypatch
):
    builder.sdmx.extra = 1
    assert _outcome(builder.tick()) == "gate_failed"
    builder.sdmx.extra = 0
    assert _outcome(builder.tick()) == "inputs_failed"

    # A state file that changes under every write leaves the failed inputs recorded.
    with monkeypatch.context() as patched:
        patched.setattr(job, "write_uri_bytes_if_unchanged", lambda *a, **k: False)
        assert hold_main(["--local-dir", str(tmp_path / "releases"), "retry"]) == 1
    assert "changed under" in builder.capsys.readouterr().err
    assert "last_failed" in builder.state()

    assert hold_main(["--local-dir", str(tmp_path / "releases"), "retry"]) == 0

    (retried,) = _named(builder.events(), "graph_release_retry")
    assert retried["inputs"]["import_execution_id"] == "wf-1"
    assert "last_failed" not in builder.state()
    assert _outcome(builder.tick()) == "published"


@pytest.mark.parametrize("races", [1, job.STATE_WRITE_ATTEMPTS])
def test_a_state_write_keeps_what_another_writer_lands_after_its_read(builder, monkeypatch, races):
    uri = join_uri(builder.root, job.STATE_FILE)
    write_uri_bytes(uri, json.dumps({"last_failed": {"a": 1}, "last_built": {"b": 2}}).encode())
    pending = [races]

    def racing(read):
        def wrapped(target):
            found = read(target)
            if target == uri and pending[0]:
                # An overlapping tick records unavailable inputs after this read.
                pending[0] -= 1
                state = json.loads(read_uri_bytes(uri))
                state["last_unavailable"] = {"at": races - pending[0]}
                write_uri_bytes(uri, json.dumps(state).encode())
            return found

        return wrapped

    for name in ("read_uri_bytes", "read_uri_bytes_and_generation"):
        monkeypatch.setattr(job, name, racing(getattr(job, name)))

    if races == job.STATE_WRITE_ATTEMPTS:
        with pytest.raises(job.StateConflict):
            job.clear_failed(builder.root)
        assert "last_failed" in builder.state()
        return
    assert job.clear_failed(builder.root) == {"a": 1}
    assert builder.state() == {"last_built": {"b": 2}, "last_unavailable": {"at": 1}}


# ---------------------------------------------------------------- stale


def test_a_failed_import_raises_stale_after_thirty_minutes(builder):
    builder.database.newest = _history("wf-2", status="FAILURE")
    builder.now = _COMPLETED + timedelta(minutes=29)
    assert _named(builder.tick(), "graph_release_stale") == []

    builder.now = _COMPLETED + timedelta(minutes=31)
    events = builder.tick()

    (stale,) = _named(events, "graph_release_stale")
    assert (stale["import_execution_id"], stale["failed"]) == ("wf-2", True)
    assert _outcome(events) == "import_not_successful"


def test_a_published_import_is_never_stale(builder):
    builder.now = _COMPLETED + timedelta(hours=2)
    builder.admin.created = builder.now - timedelta(minutes=1)
    builder.tick()

    assert _named(builder.tick(), "graph_release_stale") == []


# ---------------------------------------------------------------- settings


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"datacommons_service": "dfo-dev-dc-datacommons-service"}, id="bare_service"),
        pytest.param({"search_catalogue_job": "search-catalogue"}, id="bare_catalogue_job"),
    ],
)
def test_settings_refuse_what_the_tick_cannot_use(builder, change):
    fields = builder.settings().model_dump() | change
    with pytest.raises(ValidationError):
        job.JobSettings(**fields)
