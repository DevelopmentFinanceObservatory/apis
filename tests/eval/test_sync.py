"""`search.eval.sync` against an in-memory Langfuse: the dataset mirror and the nightly."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from dfo_evals.langfuse_sink import DatasetItemRef, PushIncompleteError, PushTimeoutError
from dfo_evals.records import FaultClass, read_run
from dfo_evals.report import input_digest
from langfuse.api import DatasetStatus
from langfuse.api.core.api_error import ApiError
from pydantic_evals.reporting import EvaluationReport

import graph_release.service as release_service
import search.eval.sync as m
from dfo_platform.app import COMMIT_HEADER
from dfo_platform.singleton import RefCountedSingleton
from graph_release.context import URI_ENV_VAR, current_release
from search.engine.errors import GraphReadFailure
from search.eval.cases import load_case_files
from search.eval.checks import ScoreContext, WireSchema
from search.eval.corpus_dataset import SUITE, case_inputs
from search.eval.run_record import record_manifest, run_record
from search.eval.runner import CaseRun, Sample
from search.models import RESOLVE_RESPONSE_ADAPTER
from tests.conftest import _build_worked_example_response
from tests.origin_catalogue.fake_catalogue import fake_catalogue_service

ETHIOPIA = """
- id: eth
  query: health ODA to Ethiopia
  why: Ethiopia is the recipient.
  expect:
    definite:
      entities: [{dcid: country/ETH, role: destination}]
"""

KENYA = """
- id: ken
  query: health ODA to Kenya
  why: Kenya is the recipient.
  expect:
    definite:
      entities: [{dcid: country/KEN, role: destination}]
- id: ken-known
  query: health aid to Kenya
  why: Kenya is the recipient.
  known_failure: {issue: 1, checks: [entities]}
  expect:
    definite:
      entities: [{dcid: country/KEN, role: destination}]
- id: ken-holdout
  query: aid to Kenya
  slice: holdout
  why: Kenya is the recipient.
  expect:
    definite:
      entities: [{dcid: country/KEN, role: destination}]
"""


RESUBMIT = """
- id: resubmit
  query: "dataset_resubmit: aid to Kenya"
  door: dataset_resubmit
  why: A resubmitted identity builds its request from the bound release's code index.
  resubmit:
    variable_text: aid
    shape: {kind: statvar_schema, population_type: FinancialFlow, measured_property: amount, stat_type: measuredValue, measurement_qualifier: Nominal}
    slots: []
    entity_dcids: [country/KEN]
  expect:
    definite:
      entities: [{dcid: country/KEN, role: destination}]
"""

PROJECT_ID = "project-1"


class FakeLangfuse:
    """Datasets and items in memory, with the SDK calls `sync` makes."""

    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, Any]] = {}
        self.items: dict[str, dict[str, SimpleNamespace]] = {}
        self.api = SimpleNamespace(datasets=SimpleNamespace(list=self._list))

    def create_dataset(self, *, name: str, description: str, metadata: dict) -> None:
        self.datasets[name] = {"name": name, "metadata": metadata}
        self.items.setdefault(name, {})

    def create_dataset_item(self, *, dataset_name: str, id: str, status, **fields) -> None:
        item = self.items[dataset_name].setdefault(id, SimpleNamespace(id=id))
        item.status = status
        for key, value in fields.items():
            setattr(item, key, value)

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_dataset(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(items=list(self.items[name].values()))

    def _list(self, *, page: int, limit: int) -> SimpleNamespace:
        rows = [SimpleNamespace(**d) for d in self.datasets.values()]
        pages = [rows[i : i + 1] for i in range(len(rows))] or [[]]
        return SimpleNamespace(data=pages[page - 1], meta=SimpleNamespace(total_pages=len(pages)))


def _write(corpus: Path, files: dict[str, str]) -> dict:
    corpus.mkdir(exist_ok=True)
    for path in corpus.glob("*.yaml"):
        path.unlink()
    for stem, text in files.items():
        (corpus / f"{stem}.yaml").write_text(text)
    return load_case_files(corpus)


def _active(fake: FakeLangfuse, name: str) -> set[str]:
    return {i.id for i in fake.items[name].values() if i.status == DatasetStatus.ACTIVE}


def test_sync_keeps_one_dataset_per_file_and_archives_what_is_gone(tmp_path):
    corpus = tmp_path / "corpus"
    fake = FakeLangfuse()
    by_file = _write(corpus, {"eth": ETHIOPIA, "ken": KENYA})

    m.sync(by_file, langfuse=fake)

    assert _active(fake, "search-eth") == {"search-eth:search-case-eth"}
    item = fake.items["search-eth"]["search-eth:search-case-eth"]
    assert item.input == case_inputs(by_file["eth"][0]).model_dump(mode="json")
    assert item.input["case_id"] == "eth"
    assert item.expected_output == {
        "definite": {"entities": [{"dcid": "country/ETH", "role": "destination"}]}
    }
    assert item.metadata["id"] == "eth"
    assert len(_active(fake, "search-ken")) == 3

    # A dropped case and a deleted file are archived; a dataset sync did not create is left.
    fake.create_dataset(name="search-goldens-v1", description="", metadata={})
    fake.create_dataset_item(dataset_name="search-goldens-v1", id="old", status="ACTIVE")
    archived = m.sync(_write(corpus, {"ken": KENYA.split("- id: ken-known")[0]}), langfuse=fake)

    assert archived == {
        "search-ken": ["search-ken:search-case-ken-known", "search-ken:search-case-ken-holdout"],
        "search-eth": ["search-eth:search-case-eth"],
    }
    assert _active(fake, "search-ken") == {"search-ken:search-case-ken"}
    assert _active(fake, "search-eth") == set()
    assert _active(fake, "search-goldens-v1") == {"old"}


class AllExist:
    def exists_batch(self, dcids: list[str]) -> set[str]:
        return set(dcids)


def score_context(graph) -> ScoreContext:
    response = RESOLVE_RESPONSE_ADAPTER.json_schema(mode="serialization")
    schema = {"schema_version": 1, "request": {}, "response": response}
    return ScoreContext(schema=WireSchema.from_document(schema), graph=graph)


@pytest.fixture
def ctx() -> ScoreContext:
    return score_context(AllExist())


async def _ethiopia_run_cases(cases, **_):
    """The app answers Ethiopia to everything, with a trace id naming the case."""
    (case,) = cases
    body = copy.deepcopy(_build_worked_example_response())
    body["diagnostics"]["trace_id"] = f"t-{case.id}"
    sample = Sample(body, 200, False, False, 1, 0.1)
    return {case.id: CaseRun(case.id, {}, (sample,), sample, 1, None)}


async def _down_run_cases(cases, **_):
    (case,) = cases
    return {case.id: CaseRun(case.id, {}, (), None, 0, "HTTP 503")}


def _record(run: m.NightlyRun):
    started = datetime(2026, 11, 25, 4, 17, tzinfo=UTC)
    manifest = record_manifest(
        run_id="nightly-test",
        started_at=started,
        finished_at=started,
        revision="abc",
        environment={"url": "http://app", "samples": "1"},
        dataset_digest=run.dataset_digest,
    )
    return run_record(run.report, manifest=manifest)


def test_nightly_lists_only_unmarked_failures_of_eligible_cases(tmp_path, monkeypatch, ctx):
    """The app answers Ethiopia to everything, so `ken` fails `entities` unmarked."""
    by_file = _write(tmp_path / "corpus", {"eth": ETHIOPIA, "ken": KENYA})
    posted: list[str] = []

    async def run_cases(cases, *, gate, **kwargs):
        assert gate is not None, "every case shares one bound on posts"
        posted.append(cases[0].id)
        return await _ethiopia_run_cases(cases, **kwargs)

    monkeypatch.setattr(m, "run_cases", run_cases)

    run = m.nightly(by_file, url="http://app", ctx=ctx)

    assert sorted(posted) == ["eth", "ken", "ken-known"]
    assert [s.case.id for s in run.states] == ["eth", "ken", "ken-known"]
    markdown = m.nightly_markdown(run.states, experiment="exp", record=None)
    assert "`ken` (ken) `entities`" in markdown
    assert "`ken` (ken) `answer`: entities" in markdown
    assert "(trace `t-ken`)" in markdown
    assert "ken-known" not in markdown
    assert "`eth`" not in markdown
    assert "Langfuse experiment: `exp`" in markdown
    assert m.nightly_exit_code(run.states) == 1
    assert run.outputs[("ken", 1)]["diagnostics"]["trace_id"] == "t-ken"


def test_the_record_holds_one_trial_per_eligible_case(tmp_path, monkeypatch, ctx):
    by_file = _write(tmp_path / "corpus", {"eth": ETHIOPIA, "ken": KENYA})
    monkeypatch.setattr(m, "run_cases", _ethiopia_run_cases)

    record = _record(m.nightly(by_file, url="http://app", ctx=ctx))

    assert record.manifest.suite == SUITE
    assert record.manifest.headline_check == "answer"
    by_case = {r.case_id: r for r in record.results}
    assert sorted(by_case) == ["eth", "ken", "ken-known"]
    assert all(r.trial == 1 for r in record.results)
    assert by_case["ken"].tags == ("corpus/ken.yaml",)
    assert by_case["eth"].tags == ("corpus/eth.yaml",)
    assert by_case["ken"].trace_id == "t-ken"
    checks = {c.name: c.verdict for c in by_case["ken"].checks}
    assert checks["entities"] == "fail" and checks["answer"] == "fail"
    assert by_case["eth"].fault is None


def test_a_synced_item_holds_the_input_its_trial_ran_on(tmp_path, monkeypatch, ctx):
    """The sink pushes a trial only onto an item whose input digests as the trial's."""
    by_file = _write(tmp_path / "corpus", {"eth": ETHIOPIA, "ken": KENYA})
    fake = FakeLangfuse()
    m.sync(by_file, langfuse=fake)
    monkeypatch.setattr(m, "run_cases", _ethiopia_run_cases)

    record = _record(m.nightly(by_file, url="http://app", ctx=ctx))

    for result in record.results:
        stem = "eth" if result.case_id == "eth" else "ken"
        item = fake.items[f"search-{stem}"][f"search-{stem}:search-case-{result.case_id}"]
        assert input_digest(item.input) == result.input_digest


def test_nightly_reports_a_post_lost_to_infrastructure(tmp_path, monkeypatch, ctx):
    by_file = _write(tmp_path / "corpus", {"eth": ETHIOPIA})
    monkeypatch.setattr(m, "run_cases", _down_run_cases)

    run = m.nightly(by_file, url="http://app", ctx=ctx)

    assert m.nightly_exit_code(run.states) == 3
    assert "HTTP 503" in m.nightly_markdown(run.states, experiment=None, record=None)
    assert run.outputs == {("eth", 1): {"task_error": "HTTP 503"}}
    (result,) = _record(run).results
    assert result.fault == FaultClass.INFRA


SLOW = """
- id: slow
  query: health ODA to Ethiopia
  why: Ethiopia is the recipient.
  known_failure: {issue: 1, checks: [task_outcome]}
  expect:
    definite:
      entities: [{dcid: country/ETH, role: destination}]
"""


class DownGraph:
    def exists_batch(self, dcids: list[str]) -> set[str]:
        raise GraphReadFailure("graph down")


@pytest.mark.parametrize(
    ("run", "graph", "code"),
    [
        (CaseRun("slow", {}, (), None, 0, "HTTP 503"), AllExist(), 0),
        (None, DownGraph(), 3),
    ],
    ids=["failed run", "failed graph read"],
)
def test_a_known_task_failure_excuses_only_a_failed_run(tmp_path, monkeypatch, run, graph, code):
    by_file = _write(tmp_path / "corpus", {"slow": SLOW})

    async def run_cases(cases, **_):
        (case,) = cases
        if run is not None:
            return {case.id: run}
        sample = Sample(_build_worked_example_response(), 200, False, False, 1, 0.1)
        return {case.id: CaseRun(case.id, {}, (sample,), sample, 1, None)}

    monkeypatch.setattr(m, "run_cases", run_cases)

    nightly_run = m.nightly(by_file, url="http://app", ctx=score_context(graph))

    assert m.nightly_exit_code(nightly_run.states) == code
    # Either way the sample measures the environment, so the record marks it infra.
    (result,) = _record(nightly_run).results
    assert result.fault == FaultClass.INFRA


@pytest.fixture
def fresh_release_service(monkeypatch) -> None:
    """A release service singleton of the test's own, so `main` opens the release
    `GRAPH_RELEASE_URI` names rather than reading the session's service."""
    monkeypatch.setattr(
        release_service,
        "_singleton",
        RefCountedSingleton(build=release_service._build, label="graph release service"),
    )


def _set_release_uri(monkeypatch, tmp_path: Path, uri: str) -> None:
    """``session`` keeps the session release's URI, ``unset`` drops it and ``empty`` names a
    root with no release published."""
    if uri == "unset":
        monkeypatch.delenv(URI_ENV_VAR)
    elif uri == "empty":
        monkeypatch.setenv(URI_ENV_VAR, (tmp_path / "no-release").as_uri())


def _no_release_service():
    raise AssertionError("the release opened")


@pytest.mark.unbound_release
@pytest.mark.parametrize("corpus_text", [RESUBMIT, ETHIOPIA], ids=["resubmit", "raw_text"])
def test_a_plain_sync_opens_no_release(tmp_path, monkeypatch, corpus_text):
    """An item's input is the authored request, which reads no release, so a sync runs
    with no ``GRAPH_RELEASE_URI``."""
    corpus = tmp_path / "corpus"
    by_file = _write(corpus, {"main": corpus_text})
    fake = FakeLangfuse()
    monkeypatch.setattr(m, "_client", lambda: fake)
    monkeypatch.delenv(URI_ENV_VAR)
    monkeypatch.setattr(m, "get_release_service", _no_release_service)

    assert m.main([f"--corpus-dir={corpus}"]) == 0
    (case,) = by_file["main"]
    item = fake.items["search-main"][f"search-main:search-case-{case.id}"]
    assert item.input == case_inputs(case).model_dump(mode="json")


def _no_langfuse():
    raise RuntimeError("Langfuse credentials missing")


@pytest.mark.parametrize(
    ("corpus_text", "client"),
    [(ETHIOPIA, _no_langfuse), ("- not: a case\n", FakeLangfuse)],
    ids=["missing credentials", "corpus does not load"],
)
def test_a_crash_exits_2(tmp_path, monkeypatch, corpus_text, client):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "eth.yaml").write_text(corpus_text)
    monkeypatch.setattr(m, "_client", client)

    assert m.main([f"--corpus-dir={corpus}"]) == 2


def test_client_samples_every_span_whatever_the_sample_rate(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "0")
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    client = m._client()
    try:
        with client.start_as_current_observation(name="probe") as span:
            sampled = span._otel_span.get_span_context().trace_flags.sampled
    finally:
        client.shutdown()

    assert sampled


@pytest.mark.parametrize("value", ["false", "False"])
def test_client_refuses_disabled_tracing(monkeypatch, value):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", value)

    with pytest.raises(RuntimeError, match="LANGFUSE_TRACING_ENABLED"):
        m._client()


def _schema_response(
    status: int, url: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status, json={}, headers=headers, request=httpx.Request("GET", url))


def _refused(url: str, **_: object) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))


@pytest.mark.parametrize(
    ("headers", "revision"),
    [({COMMIT_HEADER: "abc123"}, "abc123"), ({}, "unknown"), ({COMMIT_HEADER: "a,b"}, "unknown")],
    ids=["named", "absent", "comma"],
)
def test_the_schema_read_names_the_apps_commit(monkeypatch, headers, revision):
    monkeypatch.setattr(m.httpx, "get", lambda url, **_: _schema_response(200, url, headers))

    assert m._schema("http://app") == ({}, revision)


@pytest.mark.parametrize(
    "get",
    [_refused, lambda url, **_: _schema_response(503, url)],
    ids=["connection refused", "503"],
)
def test_a_schema_that_does_not_load_exits_3(tmp_path, monkeypatch, get):
    """The schema loads before the release opens, so an app outage exits 3 whatever the
    release does."""
    corpus = tmp_path / "corpus"
    _write(corpus, {"main": RESUBMIT})
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", PROJECT_ID)
    monkeypatch.setattr(m, "_client", FakeLangfuse)
    monkeypatch.setattr(m.httpx, "get", get)
    monkeypatch.setattr(m, "get_release_service", _no_release_service)

    assert m.main([f"--corpus-dir={corpus}", "--nightly=http://app"]) == 3


@contextmanager
def _fake_score_clients():
    yield SimpleNamespace(graph=AllExist(), sdmx=None, coverage=None)


def _empty_run() -> m.NightlyRun:
    return m.NightlyRun(
        states=[], report=EvaluationReport(name=SUITE, cases=[]), dataset_digest="d", outputs={}
    )


@pytest.mark.unbound_release
@pytest.mark.parametrize(("uri", "code"), [("session", 0), ("unset", 2), ("empty", 3)])
def test_the_nightly_runs_under_the_release_main_opened(
    tmp_path, monkeypatch, fresh_release_service, uri, code
):
    """`_run_nightly` binds the release `main` opened from `GRAPH_RELEASE_URI` around the
    run, where a `dataset_resubmit` request reads it. A release that does not open exits 3,
    an unset URI 2."""
    corpus = tmp_path / "corpus"
    _write(corpus, {"main": ETHIOPIA})
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", PROJECT_ID)
    monkeypatch.setattr(m, "_client", FakeLangfuse)
    monkeypatch.setattr(m, "_connect", lambda project_id: FakeLangfuse())
    monkeypatch.setattr(m, "push_run", lambda **_: "exp")
    response = RESOLVE_RESPONSE_ADAPTER.json_schema(mode="serialization")
    schema = {"schema_version": 1, "request": {}, "response": response}
    monkeypatch.setattr(
        m.httpx,
        "get",
        lambda url, **_: httpx.Response(200, json=schema, request=httpx.Request("GET", url)),
    )
    monkeypatch.setattr(
        "origin_catalogue.service.get_catalogue_service",
        fake_catalogue_service,
    )
    monkeypatch.setattr("search.eval.live_clients.score_clients", _fake_score_clients)
    _set_release_uri(monkeypatch, tmp_path, uri)
    bound = []

    def spy_nightly(by_file, **_):
        bound.append(current_release())
        return _empty_run()

    monkeypatch.setattr(m, "nightly", spy_nightly)

    assert m.main([f"--corpus-dir={corpus}", "--nightly=http://app"]) == code
    if code == 0:
        assert [release.service for release in bound] == [release_service.get_release_service()]


class BuggyGraph:
    def exists_batch(self, dcids: list[str]) -> set[str]:
        raise ValueError("a bug in the check")


async def _raising_run_cases(cases, **_):
    raise RuntimeError("a bug in the task")


def _stub_nightly(monkeypatch, *, run_cases, graph, push_run) -> FakeLangfuse:
    """Stub everything the nightly reads beyond the corpus: the app's schema and commit
    (``abc``), the posts, the live clients, and both Langfuse clients, one fake."""
    fake = FakeLangfuse()
    monkeypatch.setenv("LANGFUSE_PROJECT_ID", PROJECT_ID)
    monkeypatch.setattr(m, "_client", lambda: fake)
    monkeypatch.setattr(m, "_connect", lambda project_id: fake)
    monkeypatch.setattr(m, "run_cases", run_cases)
    monkeypatch.setattr(m, "push_run", push_run)
    monkeypatch.setattr(
        m.httpx, "get", lambda url, **_: _schema_response(200, url, {COMMIT_HEADER: "abc"})
    )
    monkeypatch.setattr(
        m,
        "_run_nightly",
        lambda by_file, url, schema, release: m.nightly(by_file, url=url, ctx=score_context(graph)),
    )
    return fake


@pytest.mark.parametrize(
    ("corpus_text", "run_cases", "graph"),
    [(ETHIOPIA, _ethiopia_run_cases, BuggyGraph()), (SLOW, _raising_run_cases, AllExist())],
    ids=["raising check", "raising task on a known task failure"],
)
def test_a_task_or_check_that_raises_exits_2(tmp_path, monkeypatch, corpus_text, run_cases, graph):
    corpus = tmp_path / "corpus"
    _write(corpus, {"main": corpus_text})
    _stub_nightly(monkeypatch, run_cases=run_cases, graph=graph, push_run=lambda **_: "exp")

    assert m.main([f"--corpus-dir={corpus}", "--nightly=http://app"]) == 2


def test_main_writes_the_record_and_pushes_it_onto_the_synced_items(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    _write(corpus, {"eth": ETHIOPIA, "ken": KENYA})
    pushed: dict[str, Any] = {}

    def push_run(*, client, run, item_of, outputs, **_):
        refs = {r.case_id: item_of(r.case_id) for r in run.results}
        for result in run.results:
            item = client.items[refs[result.case_id].dataset][refs[result.case_id].item_id]
            assert input_digest(item.input) == result.input_digest
        pushed.update(run=run, outputs=outputs, refs=refs)
        return "apis-search/r1 abc 0123"

    _stub_nightly(monkeypatch, run_cases=_ethiopia_run_cases, graph=AllExist(), push_run=push_run)
    records, summary = tmp_path / "records", tmp_path / "nightly.md"

    code = m.main(
        [
            f"--corpus-dir={corpus}",
            "--nightly=http://app",
            f"--records={records}",
            "--run-id=r1",
            f"--summary={summary}",
        ]
    )

    assert code == 1
    record = read_run(records / SUITE / "r1")
    assert record == pushed["run"]
    assert record.manifest.revision == "abc"
    assert record.manifest.environment == {"url": "http://app", "samples": "1"}
    assert pushed["refs"] == {
        "eth": DatasetItemRef(dataset="search-eth", item_id="search-eth:search-case-eth"),
        "ken": DatasetItemRef(dataset="search-ken", item_id="search-ken:search-case-ken"),
        "ken-known": DatasetItemRef(
            dataset="search-ken", item_id="search-ken:search-case-ken-known"
        ),
    }
    assert set(pushed["outputs"]) == {("eth", 1), ("ken", 1), ("ken-known", 1)}
    markdown = summary.read_text()
    assert "Langfuse experiment: `apis-search/r1 abc 0123`" in markdown
    assert f"Run record: `{records / SUITE / 'r1'}`" in markdown


def test_a_nightly_without_a_project_id_exits_2_before_any_case_runs(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    _write(corpus, {"eth": ETHIOPIA})
    posted = []

    async def run_cases(cases, **kwargs):
        posted.append(cases[0].id)
        return await _ethiopia_run_cases(cases, **kwargs)

    _stub_nightly(monkeypatch, run_cases=run_cases, graph=AllExist(), push_run=lambda **_: "exp")
    monkeypatch.delenv("LANGFUSE_PROJECT_ID")

    assert m.main([f"--corpus-dir={corpus}", "--nightly=http://app"]) == 2
    assert posted == []


def _api_error(message: str) -> Exception:
    return ApiError(status_code=503, body=message)


def _transport_error(message: str) -> Exception:
    return httpx.ConnectError(message, request=httpx.Request("GET", "http://langfuse"))


@pytest.mark.parametrize(
    "error",
    [PushIncompleteError, PushTimeoutError, _api_error, _transport_error],
    ids=["incomplete", "timeout", "api error", "transport error"],
)
@pytest.mark.parametrize(
    ("corpus_text", "code"), [(ETHIOPIA, 3), (KENYA, 1)], ids=["checks pass", "checks fail"]
)
def test_a_push_langfuse_did_not_show_is_infrastructure(
    tmp_path, monkeypatch, error, corpus_text, code
):
    """An unmarked failure outranks the lost push, which otherwise exits 3."""
    corpus = tmp_path / "corpus"
    _write(corpus, {"main": corpus_text})
    records, summary = tmp_path / "records", tmp_path / "nightly.md"

    def push_run(**_):
        raise error("3 cases never appeared")

    _stub_nightly(monkeypatch, run_cases=_ethiopia_run_cases, graph=AllExist(), push_run=push_run)

    argv = [f"--corpus-dir={corpus}", "--nightly=http://app", f"--records={records}"]
    assert m.main([*argv, f"--summary={summary}"]) == code
    assert "the push did not complete" in summary.read_text()
    (written,) = (records / SUITE).iterdir()
    assert written.name.startswith("nightly-")


def test_another_push_error_exits_2_after_writing_the_summary(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    _write(corpus, {"main": KENYA})
    summary = tmp_path / "nightly.md"

    def push_run(**_):
        raise ValueError("revision holds a comma")

    _stub_nightly(monkeypatch, run_cases=_ethiopia_run_cases, graph=AllExist(), push_run=push_run)

    assert m.main([f"--corpus-dir={corpus}", "--nightly=http://app", f"--summary={summary}"]) == 2
    text = summary.read_text()
    assert "ValueError: revision holds a comma" in text
    assert "`ken` (main) `entities`" in text


@pytest.mark.parametrize(
    ("env", "base_url"),
    [
        ({"LANGFUSE_BASE_URL": "https://a", "LANGFUSE_HOST": "https://b"}, "https://a"),
        ({"LANGFUSE_HOST": "https://b"}, "https://b"),
        ({}, "https://cloud.langfuse.com"),
    ],
    ids=["base url", "host", "default"],
)
def test_the_client_reads_the_base_url_then_the_host(monkeypatch, env, base_url):
    for name in ("LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    seen = {}
    monkeypatch.setattr(m, "connect", lambda **kwargs: seen.update(kwargs))

    m._connect(PROJECT_ID)

    assert seen["base_url"] == base_url
