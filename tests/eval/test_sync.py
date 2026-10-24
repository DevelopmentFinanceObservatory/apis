"""`search.eval.sync` against an in-memory Langfuse: the dataset mirror and the nightly."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langfuse.api import DatasetStatus

import search.eval.sync as m
from search.engine.errors import GraphReadFailure
from search.eval.cases import load_case_files
from search.eval.checks import ScoreContext, WireSchema
from search.eval.runner import CaseRun, Sample, build_request
from search.models import RESOLVE_RESPONSE_ADAPTER
from tests.conftest import _build_worked_example_response

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


class FakeLangfuse:
    """Datasets and items in memory, with the SDK calls `sync` makes."""

    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, Any]] = {}
        self.items: dict[str, dict[str, SimpleNamespace]] = {}
        self.experiments: list[dict[str, Any]] = []
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

    def get_dataset(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(items=list(self.items[name].values()))

    def _list(self, *, page: int, limit: int) -> SimpleNamespace:
        rows = [SimpleNamespace(**d) for d in self.datasets.values()]
        pages = [rows[i : i + 1] for i in range(len(rows))] or [[]]
        return SimpleNamespace(data=pages[page - 1], meta=SimpleNamespace(total_pages=len(pages)))

    def run_experiment(self, *, data, task, evaluators, **kwargs) -> SimpleNamespace:
        """Runs the task and evaluators per item, as `Langfuse.run_experiment` does,
        swallowing what they raise as it does."""

        async def one(item) -> list:
            # The SDK logs and drops what a task or an evaluator raises.
            try:
                output = await task(item=item)
            except Exception:
                return []
            evaluations = []
            for evaluator in evaluators:
                try:
                    evaluations += await evaluator(
                        input=item.input,
                        output=output,
                        expected_output=item.expected_output,
                        metadata=item.metadata,
                    )
                except Exception:
                    continue
            return evaluations

        async def every() -> list:
            return [await one(item) for item in data]

        scores = asyncio.run(every())
        self.experiments.append({"items": [i.metadata["id"] for i in data], "scores": scores})
        return SimpleNamespace(dataset_run_url=f"run-{len(self.experiments)}")


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
    assert item.input == build_request(by_file["eth"][0])
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


def test_nightly_lists_only_unmarked_failures_of_eligible_cases(tmp_path, monkeypatch, ctx):
    """The app answers Ethiopia to everything, so `ken` fails `entities` unmarked."""
    fake = FakeLangfuse()
    by_file = _write(tmp_path / "corpus", {"eth": ETHIOPIA, "ken": KENYA})
    m.sync(by_file, langfuse=fake)
    posted: list[str] = []

    async def run_cases(cases, *, url, samples, concurrency):
        (case,) = cases
        posted.append(case.id)
        body = copy.deepcopy(_build_worked_example_response())
        body["diagnostics"]["trace_id"] = f"t-{case.id}"
        sample = Sample(body, 200, False, False, 1, 0.1)
        return {case.id: CaseRun(case.id, {}, (sample,), sample, 1, None)}

    monkeypatch.setattr(m, "run_cases", run_cases)

    states, run_urls = m.nightly(by_file, url="http://app", ctx=ctx, langfuse=fake)

    assert sorted(posted) == ["eth", "ken", "ken-known"]
    assert run_urls == {"search-eth": "run-1", "search-ken": "run-2"}
    markdown = m.nightly_markdown(states, run_urls)
    assert "`ken` (ken) `entities`" in markdown
    assert "`ken` (ken) `answer`: entities" in markdown
    assert "(trace `t-ken`)" in markdown
    assert "ken-known" not in markdown
    assert "`eth`" not in markdown
    assert m.nightly_exit_code(states) == 1
    # Each scored check reaches Langfuse as a 0/1 score named for the check.
    ken = fake.experiments[1]
    scores = {e.name: e.value for e in ken["scores"][ken["items"].index("ken")]}
    assert scores["entities"] == 0.0


def test_nightly_reports_a_post_lost_to_infrastructure(tmp_path, monkeypatch, ctx):
    fake = FakeLangfuse()
    by_file = _write(tmp_path / "corpus", {"eth": ETHIOPIA})
    m.sync(by_file, langfuse=fake)

    async def run_cases(cases, **_):
        (case,) = cases
        return {case.id: CaseRun(case.id, {}, (), None, 0, "HTTP 503")}

    monkeypatch.setattr(m, "run_cases", run_cases)

    states, run_urls = m.nightly(by_file, url="http://app", ctx=ctx, langfuse=fake)

    assert m.nightly_exit_code(states) == 3
    assert "HTTP 503" in m.nightly_markdown(states, run_urls)


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
    fake = FakeLangfuse()
    by_file = _write(tmp_path / "corpus", {"slow": SLOW})
    m.sync(by_file, langfuse=fake)

    async def run_cases(cases, **_):
        (case,) = cases
        if run is not None:
            return {case.id: run}
        sample = Sample(_build_worked_example_response(), 200, False, False, 1, 0.1)
        return {case.id: CaseRun(case.id, {}, (sample,), sample, 1, None)}

    monkeypatch.setattr(m, "run_cases", run_cases)

    states, _ = m.nightly(by_file, url="http://app", ctx=score_context(graph), langfuse=fake)

    assert m.nightly_exit_code(states) == code


@pytest.mark.unbound_release
def test_main_binds_the_served_release_before_it_builds_a_request(tmp_path, monkeypatch):
    """A `dataset_resubmit` request reads the bound release's code index, so `main` binds one
    before `sync` builds the dataset items."""
    corpus = tmp_path / "corpus"
    _write(corpus, {"resubmit": RESUBMIT})
    fake = FakeLangfuse()
    monkeypatch.setattr(m, "_client", lambda: fake)

    assert m.main([f"--corpus-dir={corpus}"]) == 0
    assert _active(fake, "search-resubmit") == {"search-resubmit:search-case-resubmit"}


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


def _schema_response(status: int, url: str) -> httpx.Response:
    return httpx.Response(status, json={}, request=httpx.Request("GET", url))


def _refused(url: str, **_: object) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))


@pytest.mark.parametrize(
    "get",
    [_refused, lambda url, **_: _schema_response(503, url)],
    ids=["connection refused", "503"],
)
def test_a_schema_that_does_not_load_exits_3(tmp_path, monkeypatch, get):
    corpus = tmp_path / "corpus"
    _write(corpus, {"main": ETHIOPIA})
    monkeypatch.setattr(m, "_client", FakeLangfuse)
    monkeypatch.setattr(m.httpx, "get", get)

    assert m.main([f"--corpus-dir={corpus}", "--nightly=http://app"]) == 3


class BuggyGraph:
    def exists_batch(self, dcids: list[str]) -> set[str]:
        raise ValueError("a bug in the check")


async def _answer_run_cases(cases, **_):
    (case,) = cases
    sample = Sample(_build_worked_example_response(), 200, False, False, 1, 0.1)
    return {case.id: CaseRun(case.id, {}, (sample,), sample, 1, None)}


async def _raising_run_cases(cases, **_):
    raise RuntimeError("a bug in the task")


@pytest.mark.parametrize(
    ("corpus_text", "run_cases", "graph"),
    [(ETHIOPIA, _answer_run_cases, BuggyGraph()), (SLOW, _raising_run_cases, AllExist())],
    ids=["raising check", "raising task on a known task failure"],
)
def test_a_task_or_check_that_raises_exits_2(tmp_path, monkeypatch, corpus_text, run_cases, graph):
    fake = FakeLangfuse()
    corpus = tmp_path / "corpus"
    _write(corpus, {"main": corpus_text})
    monkeypatch.setattr(m, "_client", lambda: fake)
    monkeypatch.setattr(m, "run_cases", run_cases)
    monkeypatch.setattr(m.httpx, "get", lambda url, **_: _schema_response(200, url))
    monkeypatch.setattr(
        m,
        "_run_nightly",
        lambda by_file, url, schema, langfuse: m.nightly(
            by_file, url=url, ctx=score_context(graph), langfuse=langfuse
        ),
    )

    assert m.main([f"--corpus-dir={corpus}", "--nightly=http://app"]) == 2
