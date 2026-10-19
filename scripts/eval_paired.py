#!/usr/bin/env python3
"""Score the eval cases on running Search apps: the paired gate, or one side alone.

Paired mode (``--head-url``, ``--base-url``, ``--base-ref``) loads the cases at head and
the base corpus from ``git show <base-ref>:corpus/...``, runs every eligible case on the
head app, and scores every sample. It re-runs at base, ``--samples`` times, only the cases
whose verdict needs base (``search.eval.gate``), then runs one interleaved confirm round,
a head sample and then a base sample, for each candidate regression. It writes a JSON
report and a Markdown summary and exits 0 when nothing blocks, 1 on a blocking verdict,
and 3 when a base run or a check's graph or coverage read failed on infrastructure. A
setup error, such as an unreadable head corpus or an unknown base ref, or any crash
exits 2.

A head sample that fails on infrastructure (a 502, 503 or 504, a timeout, a transport
error) is retried twice, except the app's own request timeout (a 504 whose problem body
has code ``timeout``), which a retry would only repeat. A sample still failing fails
``task_outcome`` and ``answer`` and is n/a on every other check. A query the change makes time out therefore goes to base, and it is a regression
when base answers and the confirm round agrees. On a case whose ``known_failure.checks``
holds ``task_outcome`` it is a known failure.

Head sampling is adaptive. Each case runs once at head. A case takes the rest of its
``--samples`` and is judged on all of them when a check outside its
``known_failure.checks`` did not pass or passed only because no sample reached it, or when
a known failure passed, which is reported fixed only if it passes on every sample. A head
sample that reaches the check and passes decides a pass, so for every check scored on each
sample the verdict is the one a run of every sample gives. A check unreached on the first
sample is not settled, since a later sample may reach it and fail.
``materialisation`` is scored on the modal sample of the samples taken. A case that stops
after one sample scores it on that sample, where a run of every sample scores it on the
modal sample of all of them, so its verdict can differ when the other samples agree on
another answer. A case that stops after one sample is not reported unstable.

Beside the per-check verdicts, each case is judged on the derived ``answer`` check
(``search.eval.checks.base.answer_result``): per sample, a failure when any check outside
the case's known failures fails on that sample. A head that answers differently wrongly on
each sample then fails ``answer`` on every sample, though each check passes on one.

Single mode (``--single-url``) runs one app and prints every check that did not pass on
every sample, case by case, with each response's trace id. It exits 1 when a check outside
its case's ``known_failure`` passed on no sample, else 3 on a failed graph or coverage
read, else 0. A sample that fails on infrastructure fails ``task_outcome``, as at head in
paired mode.

``--shard K/N`` (1 <= K <= N) runs the cases whose id hashes to K - 1 modulo N (the sha256
of the id, so a case stays in its shard across runs and machines), intersected with ``--case`` and
``--file``. The same rule picks the base corpus cases the case-set diff compares, so the N
shards together report every deleted, newly ineligible or newly known failure case
exactly once. The eval gate workflow runs N shards on parallel jobs, each against its own
head and base apps, and combines their exit codes: 1 if any shard exits 1, else 3 if any
exits 3, else 0. A shard that holds no case exits 0 and its summary says so.

Scoring reads the graph, SDMX and the coverage index (``search.eval.live_clients``), so
the environment needs the same graph settings and ``COVERAGE_ARTIFACT_URI`` as the apps.
``just eval-paired`` and ``just eval-run`` wrap the two modes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx

from graph_release.context import bound_served_release, optional_release_service
from origin_catalogue import Catalogue, bound_catalogue
from origin_catalogue.service import get_catalogue_service
from search.eval.apis_auth import apis_auth
from search.eval.cases import (
    CHECK_NAMES,
    DEFAULT_CORPUS_DIR,
    Case,
    CorpusError,
    expectation_hash,
    ineligibility,
    load_case_files,
)
from search.eval.checks import (
    ANSWER,
    CHECKS,
    MODAL_ONLY_CHECKS,
    Result,
    ScoreContext,
    Verdict,
    WireSchema,
    answer_result,
    score_with_answer,
)
from search.eval.gate import (
    Judgement,
    Observation,
    Outcome,
    exit_code,
    is_new_case,
    judge,
)
from search.eval.live_clients import score_clients
from search.eval.runner import (
    EVAL_HEADERS,
    CaseRun,
    Sample,
    extend_run,
    fetch_schema,
    members_door_reader,
    modal_sample,
    run_cases,
)
from search.eval.summary import paired_markdown, single_markdown, single_text

if TYPE_CHECKING:
    from graph_release.service import ReleaseService
    from search.engine.clients.sdmx import EngineSdmxClient
    from search.eval.graph import GraphClient
    from series_coverage.shared import CoverageService

type Side = Literal["head", "base"]
type Round = Literal["head", "base", "confirm_head", "confirm_base"]

# Above the engine's 45-second request deadline, as in search.eval.runner.
_TIMEOUT_S = 90.0


@dataclass(frozen=True)
class Reads:
    """The graph, SDMX and coverage reads the checks score against, the origin
    catalogue the graph client's reads need bound, and the release service whose served
    release a run binds beside it, `None` when `GRAPH_RELEASE_URI` is unset."""

    catalogue: Catalogue
    release: ReleaseService | None = None
    graph: GraphClient | None = None
    sdmx: EngineSdmxClient | None = None
    coverage: CoverageService | None = None

    def context(self, schema: Mapping[str, Any], side: Side, *, url: str) -> ScoreContext:
        """The context scoring answers of the app at `url`, whose members door names the
        StatVars its Specs serve."""
        return ScoreContext(
            schema=WireSchema.from_document(schema),
            graph=self.graph,
            sdmx=self.sdmx,
            coverage=self.coverage,
            side=side,
            members=members_door_reader(url),
        )


@dataclass(frozen=True)
class SampleScore:
    """Every check's result on one sample. ``sample`` is None for a sample that failed on
    infrastructure, which fails ``task_outcome`` and is n/a on every
    other check (``score_run``)."""

    sample: Sample | None
    results: dict[str, Result]
    modal: bool

    @property
    def infra(self) -> bool:
        return self.sample is None

    def observation(self, check: str, side: Side) -> Observation | None:
        """The check's observation on this sample. A failed run is infra at base and a
        scored sample at head, where the PR may be what made it fail."""
        result = self.results.get(check)
        if result is None:
            return None
        return Observation(
            verdict=result.verdict,
            infra="run" if self.infra and side == "base" else ("read" if result.infra else None),
            request_rejected=self.sample is not None and self.sample.request_rejected,
            degraded=self.sample is not None and self.sample.degraded,
            run_failed=self.infra and side == "head",
            parts=result.parts,
        )


@dataclass
class CaseState:
    """One case through the rounds the gate sends it to."""

    case: Case
    file: str
    new_case: bool
    rounds: dict[Round, list[SampleScore]] = field(default_factory=dict)
    infra_errors: dict[Round, str] = field(default_factory=dict)
    judgements: dict[str, Judgement] = field(default_factory=dict)

    def observations(self, round_: Round, check: str) -> list[Observation] | None:
        if round_ not in self.rounds:
            return None
        side: Side = "base" if round_ in ("base", "confirm_base") else "head"
        scores = self.rounds[round_]
        return [o for s in scores if (o := s.observation(check, side)) is not None]

    @property
    def known(self) -> frozenset[str]:
        return known_checks(self.case)

    def judge(self) -> None:
        known = self.known
        for check in [*_checks(), ANSWER]:
            confirm = None
            confirm_head = self.observations("confirm_head", check)
            confirm_base = self.observations("confirm_base", check)
            if confirm_head and confirm_base:
                confirm = (confirm_head[0], confirm_base[0])
            self.judgements[check] = judge(
                check=check,
                head=self.observations("head", check) or [],
                base=self.observations("base", check),
                confirm=confirm,
                known_failure=check in known,
                known_task_failure="task_outcome" in known,
                new_case=self.new_case,
            )

    def waiting_for(self, outcome: Outcome) -> bool:
        return any(j.outcome == outcome for j in self.judgements.values())

    def wants_more_samples(self) -> bool:
        """Whether more head samples could change a verdict: a check outside the known
        failures has no head pass from a sample that reached it, or a known failure passed,
        which is reported fixed only when it passes on every sample. Infra and a known task
        failure's failed run stay what they are on any number of samples."""
        settled = {Outcome.PASS, Outcome.INFRA, Outcome.KNOWN_FAILURE}
        return any(
            judgement.fixed
            if check in self.known
            else judgement.outcome not in settled or not judgement.reached
            for check, judgement in self.judgements.items()
        )


@dataclass(frozen=True)
class Shard:
    """Shard ``index`` of ``count``, numbered from 1: the cases whose id's sha256 is
    ``index - 1`` modulo ``count``. A digest, unlike ``hash``, puts a case in the same
    shard in every process."""

    index: int
    count: int

    def holds(self, case_id: str) -> bool:
        digest = hashlib.sha256(case_id.encode()).hexdigest()
        return int(digest, 16) % self.count == self.index - 1

    def __str__(self) -> str:
        return f"{self.index}/{self.count}"


def parse_shard(spec: str) -> Shard:
    """``K/N`` with ``1 <= K <= N``, as ``--shard`` takes it.

    Raises:
        argparse.ArgumentTypeError: ``spec`` is not of that form.
    """
    index, _, count = spec.partition("/")
    try:
        shard = Shard(int(index), int(count))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{spec!r} is not K/N") from None
    if not 1 <= shard.index <= shard.count:
        raise argparse.ArgumentTypeError(f"{spec!r} needs 1 <= K <= N")
    return shard


@dataclass(frozen=True)
class Selection:
    """Cases picked by ``--case``, ``--file`` and ``--shard``; empty tuples and no shard
    pick everything."""

    case_ids: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    shard: Shard | None = None

    def pick(self, by_file: Mapping[str, list[Case]]) -> dict[str, list[Case]]:
        """The selected cases, refusing a case id or file the corpus does not hold."""
        known_ids = {case.id for cases in by_file.values() for case in cases}
        unknown = [c for c in self.case_ids if c not in known_ids]
        unknown += [f"file {f}" for f in self.files if f not in by_file]
        if unknown:
            raise SetupError(f"not in the corpus: {', '.join(unknown)}")
        return self.filter(by_file)

    def filter(self, by_file: Mapping[str, list[Case]]) -> dict[str, list[Case]]:
        """The selected cases ``by_file`` holds."""
        return {
            stem: [case for case in cases if self.holds(case.id)]
            for stem, cases in by_file.items()
            if not self.files or stem in self.files
        }

    def holds(self, case_id: str) -> bool:
        """Whether the case ids and the shard pick ``case_id``."""
        if self.case_ids and case_id not in self.case_ids:
            return False
        return self.shard is None or self.shard.holds(case_id)


class SetupError(Exception):
    """The run cannot start: a corpus, a ref, or a filter is wrong."""


def _checks() -> list[str]:
    return [name for name in CHECK_NAMES if name in CHECKS]


def known_checks(case: Case) -> frozenset[str]:
    """The checks in the case's ``known_failure.checks``."""
    return frozenset(case.known_failure.checks) if case.known_failure else frozenset()


def load_base_corpus(
    ref: str, corpus_dir: Path = DEFAULT_CORPUS_DIR
) -> tuple[dict[str, list[Case]] | None, str]:
    """The corpus at ``ref``, from the same directory as ``corpus_dir``, loaded under the
    head case model. Returns None with the reason when it does not load.

    Raises:
        SetupError: git cannot read ``ref``.
    """
    try:
        listed = _git(corpus_dir, "ls-tree", "-r", "--name-only", ref, "--", ".")
    except subprocess.CalledProcessError as exc:
        raise SetupError(f"cannot list {corpus_dir} at {ref}: {exc.stderr.strip()}") from exc
    names = [
        name
        for name in listed.splitlines()
        if name.endswith(".yaml") and Path(name).parent.as_posix() in {".", "families"}
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            target = Path(tmp) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_git(corpus_dir, "show", f"{ref}:./{name}"), encoding="utf-8")
        try:
            return load_case_files(Path(tmp)), ""
        except CorpusError as exc:
            return None, f"{len(exc.errors)} error(s); first: {exc.errors[0]}"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def score_run(case: Case, run: CaseRun, ctx: ScoreContext) -> list[SampleScore]:
    """Score every sample of ``run``, with the derived ``answer`` check; checks in
    ``MODAL_ONLY_CHECKS`` run on the modal sample alone.

    A run that failed on infrastructure adds one failed sample after its completed ones:
    ``task_outcome`` scores its error and fails, every other check is n/a, and ``answer``
    follows from them. Its completed samples vote for their own modal sample.
    """
    modal = run.modal
    if modal is None and run.samples:
        modal = run.samples[modal_sample(run.samples)[0]]
    scores = [
        SampleScore(
            sample=sample,
            results=score_with_answer(
                case, sample.body, ctx, skip=() if sample is modal else MODAL_ONLY_CHECKS
            ),
            modal=sample is modal,
        )
        for sample in run.samples
    ]
    if run.infra_error is not None:
        error = {"task_error": run.infra_error}
        results = {name: Result(Verdict.NA) for name in _checks()}
        results["task_outcome"] = CHECKS["task_outcome"](case, error, ctx)
        results[ANSWER] = answer_result(results, known=known_checks(case))
        scores.append(SampleScore(sample=None, results=results, modal=modal is None))
    return scores


def _score_round(
    states: Sequence[CaseState],
    runs: Mapping[str, CaseRun],
    round_: Round,
    ctx: ScoreContext,
    *,
    workers: int,
) -> None:
    def one(state: CaseState) -> None:
        run = runs[state.case.id]
        state.rounds[round_] = score_run(state.case, run, ctx)
        if run.infra_error is not None:
            state.infra_errors[round_] = run.infra_error

    # Checks read the graph and coverage synchronously, so cases score on threads. Each
    # task runs in a copy of this context, which carries the bound origin catalogue and release.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(contextvars.copy_context().run, one, s) for s in states]
        for task in tasks:
            task.result()


async def run_paired(
    head_cases: Mapping[str, list[Case]],
    base_cases: Mapping[str, list[Case]] | None,
    *,
    head_url: str,
    base_url: str,
    samples: int,
    concurrency: int,
    client: httpx.AsyncClient,
    reads: Reads,
) -> list[CaseState]:
    """Run the gate over every eligible case in ``head_cases`` and return each case with
    its final verdicts. ``base_cases`` is None when the base corpus does not load."""
    base_hashes = (
        None
        if base_cases is None
        else {case.id: expectation_hash(case) for cases in base_cases.values() for case in cases}
    )
    states = [
        CaseState(case, stem, is_new_case(case.id, expectation_hash(case), base_hashes))
        for stem, cases in head_cases.items()
        for case in cases
        if not ineligibility(case, implemented_checks=CHECKS)
    ]
    head_ctx = reads.context(await fetch_schema(head_url, client=client), "head", url=head_url)
    await _run_head(
        states, url=head_url, samples=samples, concurrency=concurrency, client=client, ctx=head_ctx
    )

    to_base = [s for s in states if s.waiting_for(Outcome.NEEDS_BASE)]
    if not to_base:
        return states
    base_ctx = reads.context(await fetch_schema(base_url, client=client), "base", url=base_url)
    runs = await run_cases(
        [s.case for s in to_base],
        url=base_url,
        samples=samples,
        concurrency=concurrency,
        client=client,
    )
    _score_round(to_base, runs, "base", base_ctx, workers=concurrency)
    for state in to_base:
        state.judge()

    to_confirm = [s for s in to_base if s.waiting_for(Outcome.NEEDS_CONFIRM)]
    if not to_confirm:
        return states
    head_runs, base_runs = await _confirm_round(
        [s.case for s in to_confirm],
        head_url=head_url,
        base_url=base_url,
        concurrency=concurrency,
        client=client,
    )
    _score_round(to_confirm, head_runs, "confirm_head", head_ctx, workers=concurrency)
    _score_round(to_confirm, base_runs, "confirm_base", base_ctx, workers=concurrency)
    for state in to_confirm:
        state.judge()
    return states


async def _run_head(
    states: Sequence[CaseState],
    *,
    url: str,
    samples: int,
    concurrency: int,
    client: httpx.AsyncClient,
    ctx: ScoreContext,
) -> None:
    """Run and judge every case once at head, then give each case that
    ``wants_more_samples`` the rest of its ``samples`` and judge it on all of them."""
    first = await run_cases(
        [s.case for s in states], url=url, samples=1, concurrency=concurrency, client=client
    )
    _score_round(states, first, "head", ctx, workers=concurrency)
    for state in states:
        state.judge()
    more = [s for s in states if s.wants_more_samples()]
    if samples == 1 or not more:
        return
    rest = await run_cases(
        [s.case for s in more], url=url, samples=samples - 1, concurrency=concurrency, client=client
    )
    runs = {s.case.id: extend_run(first[s.case.id], rest[s.case.id]) for s in more}
    _score_round(more, runs, "head", ctx, workers=concurrency)
    for state in more:
        state.judge()


async def _confirm_round(
    cases: Sequence[Case],
    *,
    head_url: str,
    base_url: str,
    concurrency: int,
    client: httpx.AsyncClient,
) -> tuple[dict[str, CaseRun], dict[str, CaseRun]]:
    """One head sample and then one base sample per case, cases in parallel."""
    gate = asyncio.Semaphore(concurrency)

    async def pair(case: Case) -> tuple[CaseRun, CaseRun]:
        async with gate:
            head = await run_cases([case], url=head_url, samples=1, concurrency=1, client=client)
            base = await run_cases([case], url=base_url, samples=1, concurrency=1, client=client)
        return head[case.id], base[case.id]

    async with asyncio.TaskGroup() as tasks:
        pending = {case.id: tasks.create_task(pair(case)) for case in cases}
    pairs = {case_id: task.result() for case_id, task in pending.items()}
    return (
        {case_id: head for case_id, (head, _) in pairs.items()},
        {case_id: base for case_id, (_, base) in pairs.items()},
    )


async def run_single(
    cases: Mapping[str, list[Case]],
    *,
    url: str,
    samples: int,
    concurrency: int,
    client: httpx.AsyncClient,
    reads: Reads,
) -> list[CaseState]:
    """Run every eligible case on one app as head, sampling as the paired head does."""
    states = [
        CaseState(case, stem, new_case=True)
        for stem, stem_cases in cases.items()
        for case in stem_cases
        if not ineligibility(case, implemented_checks=CHECKS)
    ]
    ctx = reads.context(await fetch_schema(url, client=client), "head", url=url)
    await _run_head(
        states, url=url, samples=samples, concurrency=concurrency, client=client, ctx=ctx
    )
    return states


def case_set_changes(
    head_cases: Mapping[str, list[Case]],
    base_cases: Mapping[str, list[Case]] | None,
    selection: Selection,
) -> dict[str, Any]:
    """The selected cases the change deleted, made ineligible, or gave a new known failure.

    Only cases ``selection`` picks on either side count, so a filtered run does not report
    edits elsewhere in the corpus. A selected base case is deleted when no head file holds
    it. A base corpus that does not load counts as empty, so every selected ineligible case
    and known failure at head is listed.
    """
    head_ids = {case.id for cases in head_cases.values() for case in cases}
    head = {case.id: case for cases in selection.filter(head_cases).values() for case in cases}
    base = {
        case.id: case for cases in selection.filter(base_cases or {}).values() for case in cases
    }
    newly_ineligible = {
        case_id: reasons
        for case_id, case in head.items()
        if (reasons := ineligibility(case, implemented_checks=CHECKS))
        and (case_id not in base or not ineligibility(base[case_id], implemented_checks=CHECKS))
    }
    newly_known = {}
    for case_id, case in head.items():
        before = base.get(case_id)
        old = set(before.known_failure.checks) if before and before.known_failure else set()
        added = [
            c for c in (case.known_failure.checks if case.known_failure else []) if c not in old
        ]
        if added:
            newly_known[case_id] = added
    return {
        "deleted": sorted(base.keys() - head_ids),
        "newly_ineligible": newly_ineligible,
        "newly_known_failure": newly_known,
    }


def build_report(states: Sequence[CaseState], **header: Any) -> dict[str, Any]:
    """The JSON report: the run header and, per case, every sample and check verdict."""
    cases = {state.case.id: _case_report(state) for state in states}
    return {**header, "cases": cases}


def _case_report(state: CaseState) -> dict[str, Any]:
    case = state.case
    rounds = list(state.rounds)
    checks = {}
    for check, judgement in state.judgements.items():
        entry: dict[str, Any] = {
            "verdict": judgement.outcome.value,
            "blocks": judgement.blocks,
            "unstable": judgement.unstable,
            "fixed": judgement.fixed,
            "degraded": judgement.degraded,
        }
        for round_ in rounds:
            entry[round_] = [_result_entry(s.results.get(check)) for s in state.rounds[round_]]
        if check == ANSWER:
            entry["detail"] = _answer_detail(state)
        checks[check] = entry
    return {
        "file": state.file,
        "query": case.query,
        "new_case": state.new_case,
        "known_failure": list(case.known_failure.checks) if case.known_failure else [],
        "samples": {round_: [_sample_entry(s) for s in state.rounds[round_]] for round_ in rounds},
        "infra_errors": dict(state.infra_errors),
        "checks": checks,
    }


def _answer_detail(state: CaseState) -> str:
    """The check behind each head-side sample's ``answer``, in sample order:
    ``disposition / disposition / measure``."""
    scores = [*state.rounds.get("head", ()), *state.rounds.get("confirm_head", ())]
    return " / ".join(
        result.detail or result.verdict.value
        for score in scores
        if (result := score.results.get(ANSWER)) is not None
    )


def _result_entry(result: Result | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {"verdict": result.verdict.value, "detail": result.detail, "infra": result.infra}


def _sample_entry(score: SampleScore) -> dict[str, Any]:
    sample = score.sample
    if sample is None:
        return {"infra": True}
    diagnostics = sample.body.get("diagnostics")
    return {
        "status": sample.status,
        "modal": score.modal,
        "degraded": sample.degraded,
        "request_rejected": sample.request_rejected,
        "attempts": sample.attempts,
        "elapsed_s": round(sample.elapsed_s, 3),
        "trace_id": diagnostics.get("trace_id") if isinstance(diagnostics, dict) else None,
    }


def single_exit_code(states: Sequence[CaseState]) -> int:
    """1 when a check outside its case's known failures passed on no sample, else 3 on
    infrastructure, else 0. Single mode judges every case as new, so a check with no pass
    reads ``new_case_failure`` or ``structural_failure``."""
    return exit_code(j.outcome for s in states for j in s.judgements.values())


def write_responses(states: Sequence[CaseState], path: Path) -> None:
    """Every scored response body as JSON lines, keyed by case, round and sample index."""
    with path.open("w", encoding="utf-8") as fh:
        for state in states:
            for round_, scores in state.rounds.items():
                for index, score in enumerate(scores):
                    if score.sample is None:
                        continue
                    row = {
                        "case": state.case.id,
                        "round": round_,
                        "index": index,
                        "status": score.sample.status,
                        "body": score.sample.body,
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--head-url", help="the head app (paired mode)")
    mode.add_argument("--single-url", help="the one app to score (single mode)")
    parser.add_argument("--base-url", help="the base app (paired mode)")
    parser.add_argument("--base-ref", help="the base revision whose corpus/ is the baseline")
    parser.add_argument("--samples", type=int, default=3, help="samples per case (default 3)")
    parser.add_argument("--concurrency", type=int, default=16, help="posts in flight (default 16)")
    parser.add_argument("--case", action="append", default=[], help="run this case id only")
    parser.add_argument("--file", action="append", default=[], help="run this case file stem only")
    parser.add_argument(
        "--shard",
        type=parse_shard,
        metavar="K/N",
        help="run shard K of N (1 <= K <= N): the cases whose id hashes to K - 1 modulo N",
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    parser.add_argument("--summary", type=Path, help="write the Markdown summary here")
    parser.add_argument("--responses", type=Path, help="write every response body here (JSONL)")
    return parser


async def run(args: argparse.Namespace, *, reads: Reads, client: httpx.AsyncClient) -> int:
    """Run the mode ``args`` selects, write its outputs, and return the exit code."""
    with bound_catalogue(reads.catalogue), bound_served_release(reads.release):
        return await _run_bound(args, reads=reads, client=client)


async def _run_bound(args: argparse.Namespace, *, reads: Reads, client: httpx.AsyncClient) -> int:
    try:
        head_cases = load_case_files(args.corpus_dir)
    except CorpusError as exc:
        raise SetupError("\n".join(["the head corpus does not load:", *exc.errors])) from exc
    selection = Selection(case_ids=tuple(args.case), files=tuple(args.file), shard=args.shard)
    picked = selection.pick(head_cases)
    header: dict[str, Any] = {
        "samples": args.samples,
        "shard": str(args.shard) if args.shard else None,
    }

    if args.single_url:
        states = await run_single(
            picked,
            url=args.single_url,
            samples=args.samples,
            concurrency=args.concurrency,
            client=client,
            reads=reads,
        )
        code = single_exit_code(states)
        report = build_report(states, mode="single", url=args.single_url, exit_code=code, **header)
        print(single_text(report))
        markdown = single_markdown(report)
    else:
        base_cases, base_error = load_base_corpus(args.base_ref, args.corpus_dir)
        states = await run_paired(
            picked,
            base_cases,
            head_url=args.head_url,
            base_url=args.base_url,
            samples=args.samples,
            concurrency=args.concurrency,
            client=client,
            reads=reads,
        )
        code = exit_code(j.outcome for s in states for j in s.judgements.values())
        report = build_report(
            states,
            mode="paired",
            head_url=args.head_url,
            base_url=args.base_url,
            base_ref=args.base_ref,
            base_corpus={"loaded": base_cases is not None, "error": base_error},
            exit_code=code,
            case_set=case_set_changes(head_cases, base_cases, selection),
            **header,
        )
        markdown = paired_markdown(report)
        print(markdown)

    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.summary:
        args.summary.write_text(markdown)
    if args.responses:
        write_responses(states, args.responses)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.head_url and not (args.base_url and args.base_ref):
        parser.error("paired mode needs --base-url and --base-ref")
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    # A crash must not read as a blocking verdict (1) or infrastructure (3), which the
    # workflow's override label can release, so every unexpected error exits 2.
    try:
        catalogue = _load_catalogue()
        release = _load_release()
        with score_clients() as clients:
            reads = Reads(
                catalogue=catalogue,
                release=release,
                graph=clients.graph,
                sdmx=clients.sdmx,
                coverage=clients.coverage,
            )
            return asyncio.run(_main_async(args, reads))
    except SetupError as exc:
        print(f"eval setup error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


def _load_catalogue() -> Catalogue:
    """The origin catalogue ``ORIGIN_CATALOGUE_URI`` names, loaded as the app loads it."""
    try:
        return get_catalogue_service().snapshot()
    except Exception as exc:
        raise SetupError(f"the origin catalogue does not load: {exc!r}") from exc


def _load_release() -> ReleaseService | None:
    """The release service ``GRAPH_RELEASE_URI`` names, opened as the app opens it, or
    ``None`` when it is unset."""
    try:
        return optional_release_service()
    except Exception as exc:
        raise SetupError(f"the graph release does not load: {exc!r}") from exc


async def _main_async(args: argparse.Namespace, reads: Reads) -> int:
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_S, headers=EVAL_HEADERS, auth=apis_auth()
    ) as client:
        try:
            return await run(args, reads=reads, client=client)
        except httpx.HTTPError as exc:
            print(f"eval infrastructure error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3


if __name__ == "__main__":
    sys.exit(main())
