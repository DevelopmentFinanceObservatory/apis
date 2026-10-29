"""`scripts/eval_paired.py` against two fake apps that replay a recorded response.

The base corpus is committed to a temporary git repository and the head corpus is its
working tree, so the driver reads the base through `git show` as it does in CI.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import scripts.eval_paired as m
from graph_release.context import bound_release_or_none
from graph_release.service import ReleaseService, get_release_service
from origin_catalogue import current_catalogue
from search.engine.errors import GraphReadFailure
from search.eval.cases import Case, asserted_checks, load_cases
from search.eval.checks import ANSWER, Result, Verdict, answer_result
from search.eval.gate import BLOCKING, Outcome
from search.eval.runner import run_cases
from search.models import RESOLVE_RESPONSE_ADAPTER
from tests.conftest import _build_worked_example_response
from tests.graph_release.support import build_view_release
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

REGRESSING = "health ODA grants to Ethiopia since 2015"
STEADY = "health ODA to Ethiopia"

CASES = f"""
- id: eth-health
  query: {REGRESSING}
  why: Ethiopia is the recipient.
  expect:
    definite:
      entities: [{{dcid: country/ETH, role: destination}}]
- id: eth-steady
  query: {STEADY}
  why: Ethiopia is the recipient.
  expect:
    definite:
      entities: [{{dcid: country/ETH, role: destination}}]
"""

GONE = """
- id: gone
  query: population of Kenya
  why: Count_Person answers.
  expect:
    definite: {}
"""


# The catalogue the driver binds; the graph refuses reads made under any other binding.
CATALOGUE = FakeCatalogue()


class AllExist:
    """A graph in which every dcid exists, read under the driver's catalogue."""

    def exists_batch(self, dcids: list[str]) -> set[str]:
        if current_catalogue() is not CATALOGUE:
            raise RuntimeError("the driver's catalogue is not bound on this thread")
        return set(dcids)

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False) -> dict:
        return {}


class ReleaseRecording(AllExist):
    """`AllExist` that records the graph release bound at each read, `None` for none."""

    def __init__(self) -> None:
        self.releases: list[str | None] = []

    def exists_batch(self, dcids: list[str]) -> set[str]:
        bound = bound_release_or_none()
        self.releases.append(bound.release_id if bound is not None else None)
        return super().exists_batch(dcids)


class Raising:
    """A graph whose every read raises ``error``."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def exists_batch(self, dcids: list[str]) -> set[str]:
        raise self.error

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False) -> dict:
        raise self.error


type Reply = Callable[[str, int], tuple[int, dict]]
"""The status and body an app answers a query with, given how many times it was posted."""


def answer() -> dict:
    """The worked example: a definite answer naming Ethiopia as the recipient."""
    return copy.deepcopy(_build_worked_example_response())


def kenya() -> dict:
    body = answer()
    body["parts"][0]["spec"]["definition"]["identity"]["entities"] = {
        "destination": ["country/KEN"]
    }
    return body


def no_data() -> dict:
    body = answer()
    variable_text = body["parts"][0]["variable_text"]
    body["status"] = "no_data"
    body["parts"] = [
        {
            "status": "no_data",
            "variable_text": variable_text,
            "no_data": {"reason": "no_observations"},
        }
    ]
    return body


def problem(status: int, code: str) -> dict:
    return {
        "type": f"/problems/{code}",
        "title": code,
        "status": status,
        "detail": code,
        "code": code,
        "trace_id": "0" * 32,
        "capability_version": "1",
        "retryable": False,
    }


def steady(query: str, n: int) -> tuple[int, dict]:
    return 200, answer()


def regressing(query: str, n: int) -> tuple[int, dict]:
    """Kenya as the recipient for ``REGRESSING``, the worked example for the rest."""
    return 200, kenya() if query == REGRESSING else answer()


def for_regressing(reply: Reply) -> Reply:
    """``reply`` for ``REGRESSING`` and the worked example for the rest."""
    return lambda query, n: reply(query, n) if query == REGRESSING else steady(query, n)


def fake_app(reply: Reply, *, posts: Counter) -> Starlette:
    """An app answering each post with ``reply``."""

    async def schema(request: Request) -> JSONResponse:
        response = RESOLVE_RESPONSE_ADAPTER.json_schema(mode="serialization")
        return JSONResponse({"schema_version": 1, "request": {}, "response": response})

    async def resolve(request: Request) -> JSONResponse:
        query = (await request.json())["input"]["query"]
        posts[query] += 1
        status, body = reply(query, posts[query])
        return JSONResponse(body, status_code=status)

    return Starlette(
        routes=[
            Route("/api/search/schema", schema),
            Route("/api/search/resolve", resolve, methods=["POST"]),
        ]
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_corpus(root: Path, *, base: str, head: str) -> Path:
    """``corpus/main.yaml`` committed as ``base`` and holding ``head`` in the work tree."""
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "main.yaml").write_text(base)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    (corpus / "main.yaml").write_text(head)
    return corpus


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A corpus whose base commit holds both cases and `gone`; the head deletes `gone`."""
    return make_corpus(tmp_path, base=CASES + GONE, head=CASES)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "run_cases", functools.partial(run_cases, backoff_s=0))


def _transports(head: Reply, base: Reply) -> tuple[dict, Counter, Counter]:
    head_posts: Counter = Counter()
    base_posts: Counter = Counter()
    transports = {
        "http://head": httpx.ASGITransport(app=fake_app(head, posts=head_posts)),
        "http://base": httpx.ASGITransport(app=fake_app(base, posts=base_posts)),
    }
    return transports, head_posts, base_posts


def run_driver(
    argv: list[str],
    *,
    head: Reply = regressing,
    base: Reply = steady,
    graph: AllExist | Raising | None = None,
    release: ReleaseService | None = None,
) -> tuple[int, Counter, Counter]:
    transports, head_posts, base_posts = _transports(head, base)
    args = m._parser().parse_args(argv)
    reads = m.Reads(
        catalogue=CATALOGUE, release=release or get_release_service(), graph=graph or AllExist()
    )

    async def go() -> int:
        async with httpx.AsyncClient(mounts=transports) as client:
            return await m.run(args, reads=reads, client=client)

    return asyncio.run(go()), head_posts, base_posts


def paired_args(corpus: Path, tmp_path: Path) -> list[str]:
    return [
        "--head-url=http://head",
        "--base-url=http://base",
        "--base-ref=HEAD",
        f"--corpus-dir={corpus}",
        f"--report={tmp_path / 'report.json'}",
        f"--summary={tmp_path / 'summary.md'}",
    ]


def test_a_scripted_regression_exits_1_naming_the_case_and_check(corpus, tmp_path):
    code, head_posts, base_posts = run_driver(paired_args(corpus, tmp_path))

    assert code == 1
    report = json.loads((tmp_path / "report.json").read_text())
    regressed = report["cases"]["eth-health"]
    assert regressed["checks"]["entities"]["verdict"] == "regression"
    assert regressed["checks"]["entities"]["head"][0]["verdict"] == "fail"
    assert regressed["samples"]["head"][0]["trace_id"] == "0" * 32
    assert report["cases"]["eth-steady"]["checks"]["entities"]["verdict"] == "pass"
    # Groundedness reads the graph on the scoring threads under the bound catalogue.
    assert report["cases"]["eth-steady"]["checks"]["groundedness"]["verdict"] == "pass"
    assert report["case_set"]["deleted"] == ["gone"]
    assert regressed["checks"]["answer"]["verdict"] == "regression"
    # The passing case takes one head sample. The regressing case takes all three and one
    # confirm sample, and only it runs at base.
    assert head_posts == {REGRESSING: 4, STEADY: 1}
    assert base_posts == {REGRESSING: 4}
    summary = (tmp_path / "summary.md").read_text()
    assert "| `eth-health` | entities | regression |" in summary
    assert summary.index("Blocking verdicts") < summary.index("Deleted cases")


def test_matching_apps_exit_0_on_one_head_sample_each_without_a_base_run(corpus, tmp_path):
    code, head_posts, base_posts = run_driver(paired_args(corpus, tmp_path), head=steady)

    assert code == 0
    assert head_posts == {REGRESSING: 1, STEADY: 1}
    assert not base_posts


def test_every_check_read_sees_the_served_graph_release(corpus, tmp_path):
    root = (tmp_path / "releases").as_uri()
    manifest = build_view_release(root)
    graph = ReleaseRecording()

    code, _, _ = run_driver(
        paired_args(corpus, tmp_path),
        head=steady,
        graph=graph,
        release=ReleaseService(uri=root, local_dir=tmp_path / "local"),
    )

    assert code == 0
    assert graph.releases and set(graph.releases) == {manifest.release_id}


def test_a_base_corpus_that_does_not_load_makes_every_case_new(corpus, tmp_path):
    (corpus / "main.yaml").write_text("- not: a case\n")
    _git(corpus.parent, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "bad")
    (corpus / "main.yaml").write_text(CASES)

    code, _, base_posts = run_driver(paired_args(corpus, tmp_path))

    assert code == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["base_corpus"]["loaded"] is False
    assert report["cases"]["eth-health"]["checks"]["entities"]["verdict"] == "new_case_failure"
    assert not base_posts


def test_single_mode_prints_each_failing_check_per_case(corpus, tmp_path, capsys):
    argv = ["--single-url=http://head", f"--corpus-dir={corpus}", "--case=eth-health"]

    code, head_posts, _ = run_driver(argv)

    assert code == 1
    assert head_posts == {REGRESSING: 3}
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"eth-health (main) {REGRESSING}"
    assert lines[1].split()[:5] == ["FAIL", "entities", "F", "F", "F"]


def _verdicts(tmp_path: Path, case_id: str) -> dict[str, str]:
    report = json.loads((tmp_path / "report.json").read_text())
    return {name: e["verdict"] for name, e in report["cases"][case_id]["checks"].items()}


def test_base_infra_costs_only_the_checks_that_sent_the_case_to_base(corpus, tmp_path):
    """`entities` needs base and base fails on infrastructure; the checks that passed at
    head stay pass, so the run exits 3 for `entities` and `answer` alone."""
    unavailable = for_regressing(lambda q, n: (503, problem(503, "service_unavailable")))

    code, _, _ = run_driver(paired_args(corpus, tmp_path), base=unavailable)

    assert code == 3
    verdicts = _verdicts(tmp_path, "eth-health")
    assert {n for n, v in verdicts.items() if v == "infra"} == {"entities", "answer"}
    assert {v for n, v in verdicts.items() if n not in ("entities", "answer")} == {"pass"}


def test_a_base_degraded_on_every_sample_is_base_inconclusive(corpus, tmp_path):
    """The base app times out its own leg on every attempt of every sample, as the slow base
    of synth-ghg-emissions-nigeria did, and answers right. A degraded answer is no baseline,
    so the head's failure is neither a regression nor a pass: the run exits 3 and the
    summary names the case."""

    def degraded(query: str, n: int) -> tuple[int, dict]:
        body = answer()
        body["diagnostics"]["timed_out"] = True
        return 200, body

    code, _, base_posts = run_driver(paired_args(corpus, tmp_path), base=for_regressing(degraded))

    assert code == 3
    verdicts = _verdicts(tmp_path, "eth-health")
    assert verdicts["entities"] == verdicts["answer"] == "base_inconclusive"
    assert not set(verdicts.values()) & {"regression", "pre_existing"}
    # Each base sample retried twice and stayed degraded; no confirm round ran.
    assert base_posts[REGRESSING] == 9
    summary = (tmp_path / "summary.md").read_text()
    assert "had only degraded base samples (exit 3)" in summary
    assert "- `eth-health` entities: every base sample stayed degraded" in summary


def test_a_head_500_against_a_base_200_is_a_regression(corpus, tmp_path):
    crashing = for_regressing(lambda q, n: (500, problem(500, "internal_error")))

    code, head_posts, _ = run_driver(paired_args(corpus, tmp_path), head=crashing)

    assert code == 1
    assert _verdicts(tmp_path, "eth-health")["disposition"] == "regression"
    # Each head sample and the confirm sample retried twice before its 500 was scored.
    assert head_posts[REGRESSING] == 12


def test_a_check_that_raises_an_error_that_is_no_read_failure_exits_2(
    corpus, tmp_path, monkeypatch
):
    transports, _, _ = _transports(steady, steady)

    @contextmanager
    def score_clients() -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(graph=Raising(ValueError("a bug")), sdmx=None, coverage=None)

    monkeypatch.setattr(m, "_load_catalogue", lambda: CATALOGUE)
    monkeypatch.setattr(m, "score_clients", score_clients)
    monkeypatch.setattr(
        httpx, "AsyncClient", functools.partial(httpx.AsyncClient, mounts=transports)
    )

    assert m.main(paired_args(corpus, tmp_path)) == 2


def test_a_rejected_head_request_passes_a_case_expecting_that_problem(tmp_path):
    case = f"""
- id: rejected
  query: {STEADY}
  why: The door refuses the request.
  expect:
    kind: problem
    problem: {{code: request_invalid}}
"""
    corpus = make_corpus(tmp_path, base=case, head=case)
    refusing = lambda q, n: (400, problem(400, "request_invalid"))  # noqa: E731

    code, head_posts, base_posts = run_driver(paired_args(corpus, tmp_path), head=refusing)

    assert code == 0
    assert set(_verdicts(tmp_path, "rejected").values()) == {"pass"}
    assert (head_posts, base_posts) == ({STEADY: 1}, {})


@pytest.mark.parametrize(
    ("head", "graph", "code"),
    [
        (lambda q, n: (503, problem(503, "service_unavailable")), None, 0),
        (steady, Raising(GraphReadFailure("graph down")), 3),
    ],
    ids=["failed run", "failed graph read"],
)
def test_a_known_task_failure_excuses_only_a_failed_run(tmp_path, head, graph, code):
    case = f"""
- id: slow
  query: {STEADY}
  why: Ethiopia is the recipient.
  known_failure: {{issue: 1, checks: [task_outcome]}}
  expect:
    definite:
      entities: [{{dcid: country/ETH, role: destination}}]
"""
    corpus = make_corpus(tmp_path, base=case, head=case)

    assert run_driver(paired_args(corpus, tmp_path), head=head, graph=graph)[0] == code


# Of 4 shards, eth-health and eth-steady fall in shard 2 and gone in shard 4, so the
# shard rows also pin the hash: a case that moved shard fails them.
@pytest.mark.parametrize(
    ("selection", "deleted", "ran"),
    [
        ([], ["gone"], {"eth-health", "eth-steady"}),
        (["--file=main"], ["gone"], {"eth-health", "eth-steady"}),
        (["--case=eth-steady"], [], {"eth-steady"}),
        (["--shard=2/4"], [], {"eth-health", "eth-steady"}),
        (["--shard=4/4"], ["gone"], set()),
        (["--shard=2/4", "--case=eth-steady"], [], {"eth-steady"}),
        (["--shard=4/4", "--case=eth-steady"], [], set()),
        (["--shard=4/4", "--file=main"], ["gone"], set()),
    ],
    ids=[
        "everything",
        "the file holding the deleted case",
        "one other case",
        "the shard of the head cases",
        "the shard of the deleted case",
        "a case in the shard",
        "a case outside the shard",
        "a shard of a file",
    ],
)
def test_the_case_set_lists_only_the_selected_cases(corpus, tmp_path, selection, deleted, ran):
    run_driver([*paired_args(corpus, tmp_path), *selection], head=steady)

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["case_set"]["deleted"] == deleted
    assert set(report["cases"]) == ran


def test_shards_partition_the_cases_by_a_stable_hash_of_the_id():
    ids = [f"case-{n}" for n in range(200)]
    shards = [m.Shard(k, 4) for k in range(1, 5)]

    owners = {case_id: [s.index for s in shards if s.holds(case_id)] for case_id in ids}

    assert all(len(owner) == 1 for owner in owners.values())
    assert all(any(owner == [k] for owner in owners.values()) for k in range(1, 5))


@pytest.mark.parametrize("mode", ["paired", "single"])
def test_a_shard_with_no_case_exits_0_and_says_so(corpus, tmp_path, mode):
    argv = paired_args(corpus, tmp_path)
    if mode == "single":
        argv = ["--single-url=http://head", *argv[3:]]

    code, head_posts, base_posts = run_driver([*argv, "--shard=1/4"])

    assert (code, head_posts, base_posts) == (0, {}, {})
    report = json.loads((tmp_path / "report.json").read_text())
    assert (report["shard"], report["cases"]) == ("1/4", {})
    summary = (tmp_path / "summary.md").read_text()
    assert "(shard 1/4)" in summary.splitlines()[0]
    assert "No eligible case falls in this selection" in summary


@pytest.mark.parametrize("spec", ["0/4", "5/4", "x/2", "1/0", "-1/2", "1", "1/2/3"])
def test_a_bad_shard_spec_is_a_usage_error(spec):
    with pytest.raises(SystemExit) as exit_:
        m.main(["--single-url=http://head", f"--shard={spec}"])

    assert exit_.value.code == 2


def test_a_head_wrong_differently_on_each_sample_regresses_on_answer(corpus, tmp_path):
    """No data on every head sample, then a definite answer naming the wrong recipient in
    the confirm round: no single check fails on every head sample, but the answer does."""
    wrong = for_regressing(lambda q, n: (200, no_data() if n <= 3 else kenya()))

    code, head_posts, _ = run_driver(paired_args(corpus, tmp_path), head=wrong)

    assert code == 1
    verdicts = _verdicts(tmp_path, "eth-health")
    assert {n for n, v in verdicts.items() if v in BLOCKING} == {"answer"}
    assert verdicts["answer"] == "regression"
    assert head_posts[REGRESSING] == 4
    summary = (tmp_path / "summary.md").read_text()
    assert (
        "| `eth-health` | answer | regression | disposition / disposition / disposition / entities |"
        in summary
    )


@pytest.mark.parametrize(
    ("known", "answer_verdict"),
    [
        ("", "pre_existing"),
        ("  known_failure: {issue: 1, checks: [entities]}\n", "wrong_answer_at_base"),
    ],
    ids=["entities asserted", "entities a known failure"],
)
def test_a_head_refusing_a_wrong_base_answer_does_not_block(tmp_path, known, answer_verdict):
    """Base answers definite on the wrong recipient and head answers no data. The head has
    no right answer to regress from on `disposition`, nor on `answer` when `entities` is a
    known failure that `answer` leaves out."""
    head_case = f"- id: eth-health\n  query: {REGRESSING}\n"
    cases = CASES.replace(head_case, head_case + known)
    corpus = make_corpus(tmp_path, base=cases, head=cases)
    refusing = for_regressing(lambda q, n: (200, no_data()))

    code, _, _ = run_driver(
        paired_args(corpus, tmp_path),
        head=refusing,
        base=for_regressing(lambda q, n: (200, kenya())),
    )

    assert code == 0
    verdicts = _verdicts(tmp_path, "eth-health")
    assert (verdicts["disposition"], verdicts["answer"]) == ("wrong_answer_at_base", answer_verdict)
    summary = (tmp_path / "summary.md").read_text()
    assert "### Wrong answers at base" in summary
    assert "- `eth-health` disposition: head status: got 'no_data'" in summary
    assert "; base entities: " in summary


def test_a_known_failure_passing_on_one_sample_of_three_is_not_fixed(tmp_path):
    case = f"""
- id: flaky
  query: {REGRESSING}
  why: Ethiopia is the recipient.
  known_failure: {{issue: 1, checks: [entities]}}
  expect:
    definite:
      entities: [{{dcid: country/ETH, role: destination}}]
"""
    corpus = make_corpus(tmp_path, base=case, head=case)
    right_once = lambda q, n: (200, answer() if n == 1 else kenya())  # noqa: E731

    code, head_posts, _ = run_driver(paired_args(corpus, tmp_path), head=right_once)

    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text())
    entities = report["cases"]["flaky"]["checks"]["entities"]
    assert (entities["verdict"], entities["fixed"]) == ("known_failure", False)
    assert head_posts == {REGRESSING: 3}


@pytest.mark.parametrize(
    ("base", "code", "verdict"),
    [
        (steady, 1, "regression"),
        (for_regressing(lambda q, n: (504, problem(504, "timeout"))), 3, "infra"),
    ],
    ids=["base answers", "base times out too"],
)
def test_a_head_timeout_is_judged_against_base(tmp_path, base, code, verdict):
    """A failed head run fails `task_outcome`; only a failed base run is infra. It is n/a
    on the other checks, and a known failure it covers is not fixed."""
    known = f"- id: eth-health\n  query: {REGRESSING}\n  known_failure: {{issue: 1, checks: [entities]}}\n"
    cases = CASES.replace(f"- id: eth-health\n  query: {REGRESSING}\n", known)
    corpus = make_corpus(tmp_path, base=cases, head=cases)
    timing_out = for_regressing(lambda q, n: (504, problem(504, "timeout")))

    result = run_driver(paired_args(corpus, tmp_path), head=timing_out, base=base)[0]

    assert result == code
    report = json.loads((tmp_path / "report.json").read_text())
    checks = report["cases"]["eth-health"]["checks"]
    assert (checks["task_outcome"]["verdict"], checks["answer"]["verdict"]) == (verdict, verdict)
    assert (checks["entities"]["verdict"], checks["entities"]["fixed"]) == ("known_failure", False)
    others = {
        e["verdict"] for n, e in checks.items() if n not in ("task_outcome", "answer", "entities")
    }
    assert others == {"pass"}


def test_a_head_run_with_one_failed_sample_passes_on_the_samples_that_answered(corpus, tmp_path):
    """The first head sample gets the app's timeout, which is not retried; the other two answer."""
    first_fails = for_regressing(
        lambda q, n: (504, problem(504, "timeout")) if n == 1 else (200, answer())
    )

    code, head_posts, base_posts = run_driver(paired_args(corpus, tmp_path), head=first_fails)

    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text())
    checks = report["cases"]["eth-health"]["checks"]
    outcome = {
        n: (checks[n]["verdict"], checks[n]["unstable"])
        for n in ("task_outcome", "answer", "entities")
    }
    assert outcome == {
        "task_outcome": ("pass", True),
        "answer": ("pass", True),
        "entities": ("pass", False),
    }
    assert head_posts[REGRESSING] == 3
    assert not base_posts


def _known_disposition_sample(case: Case, asserted: Verdict) -> m.SampleScore:
    """A sample of ``case`` on which `disposition`, its known failure, fails, every other
    check the case asserts reads ``asserted``, and the rest are n/a."""
    others = asserted_checks(case) - m.known_checks(case)

    def verdict(name: str) -> Verdict:
        if name == "disposition":
            return Verdict.FAIL
        return asserted if name in others else Verdict.NA

    results = {name: Result(verdict(name)) for name in m._checks()}
    results[ANSWER] = answer_result(results, known=m.known_checks(case))
    return m.SampleScore(sample=None, results=results, modal=True)


@pytest.mark.parametrize(("first", "more"), [(Verdict.UNREACHED, True), (Verdict.PASS, False)])
def test_a_first_head_sample_that_reaches_no_asserted_check_takes_more_samples(first, more):
    """dq-crs-015 lists `disposition` as a known failure, so a first answer in a status it
    does not accept leaves every other asserted check unreached. That settles nothing: the
    later samples may reach the checks and fail them."""
    case = next(c for c in load_cases() if c.id == "dq-crs-015")
    others = asserted_checks(case) - m.known_checks(case)
    assert others
    state = m.CaseState(case=case, file="crs", new_case=False)
    state.rounds["head"] = [_known_disposition_sample(case, first)]
    state.judge()
    assert state.wants_more_samples() is more
    if not more:
        return

    state.rounds["head"] += [_known_disposition_sample(case, Verdict.FAIL)] * 2
    state.judge()
    assert {state.judgements[c].outcome for c in {*others, ANSWER}} == {Outcome.NEEDS_BASE}
