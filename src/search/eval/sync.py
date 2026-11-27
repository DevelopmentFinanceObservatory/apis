"""Mirror the case files to Langfuse, and run the nightly against a deployed app.

``python -m search.eval.sync`` keeps one Langfuse dataset per case file, ``search-<stem>``,
whose metadata names the file. It writes one item per case, keyed by case id, through
``dfo_evals.langfuse_sink.upsert_items``, which writes only the items Langfuse lacks or holds
differently. The item's input is the case's request as its case file authors it, keyed by
its case id (``corpus_dataset.CaseInputs``), and its expected output is the case's
``expect`` with families expanded. It archives every active item whose case is gone,
including the items of a deleted case file. A plain sync opens no graph release. It exits
3 when Langfuse or the network fails the project check, and 2 on keys Langfuse refuses or
any other error.

``python -m search.eval.sync --nightly URL`` loads the app's schema and the served graph
release, syncs the datasets, then runs every eligible case on the app at URL as the corpus
dataset (``search.eval.corpus_dataset``): one post per case, or ``repeat`` posts for a case
that sets it. It scores each case's modal answer with the gate's checks and the derived
``answer`` check. Unlike the gate, which fails ``task_outcome`` and marks every other check
n/a, it scores a case with a post that failed on infrastructure on the error body
``{"task_error": ...}`` with every check, so checks such as ``disposition`` fail too, even
when the case's other posts answered. It lists every check that did not pass and is outside
its case's ``known_failure``, and exits 0 when there is none, 1 when there is one, and 3
when the app's schema does not load, when the served graph release does not open, or when
a post or a check's read failed on infrastructure and nothing else failed. A post that
failed on infrastructure is a known failure on a case whose ``known_failure.checks`` holds
``task_outcome``, as in the gate, for the exit code; in the record and in Langfuse its
``answer`` still fails through the other checks. Any crash exits 2, an unset
``GRAPH_RELEASE_URI`` or ``LANGFUSE_PROJECT_ID`` among them.

The nightly's results become a ``dfo-evals`` run record (``search.eval.run_record``): one
trial per eligible case, scored by ``answer`` and tagged with its case file. With
``--records DIR`` the record is written under ``DIR/apis-search/<run id>/``. It is then
pushed to Langfuse as one experiment in each case file's dataset, all under one run name,
with each case's response body as its output (``dfo_evals.langfuse_sink.push_run``). Each
check scores 1 on a pass and 0 on a failure, and has no score when it is unverifiable, n/a
or unreached.

Langfuse is at fault when it does not show a push in full before its timeout, or answers
with a transport error, a 5xx or a 429. At the project check or during the sync, the
nightly then runs and records without syncing or pushing; during the push, the record is
already written. Either way it exits 1 when the checks gave 1, else 3, and the summary says
the push did not complete. Any other Langfuse error during the sync or the push, such as a
404 from an endpoint Langfuse removed, exits 2 once the record and summary are written;
keys Langfuse refuses, or keys of another project, exit 2 before anything runs. No record
or summary is written when the schema or the release fails (exit 3), or when a case's task
or the evaluator raises (exit 2, with the traceback). The experiments are named
``apis-search`` with the run's name, in place of the per-file ``nightly`` experiments of
earlier runs. The record's revision is the commit the app's ``x-dfo-commit`` header names,
or ``unknown``.

Requires ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_PROJECT_ID``, the
project the keys must name (``dfo_evals.langfuse_sink.connect``), and reads
``LANGFUSE_BASE_URL`` (or ``LANGFUSE_HOST``; default ``https://cloud.langfuse.com``). A
``dataset_resubmit`` case's request reads the
code index of the served graph release, and the checks read that release, so the nightly
reads ``GRAPH_RELEASE_URI`` and opens that release once, with Google credentials for a
``gs://`` URI. It also reads the graph settings and ``ORIGIN_CATALOGUE_URI`` the checks
score against.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import functools
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx
from dfo_evals.langfuse_sink import (
    DatasetItemRef,
    DatasetItemSpec,
    PushIncompleteError,
    PushTimeoutError,
    UnknownCaseError,
    connect,
    push_run,
    upsert_items,
)
from dfo_evals.records import Run, write_run
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from langfuse.api import DatasetStatus
from langfuse.api.core.api_error import ApiError
from pydantic_evals.reporting import EvaluationReport

from dfo_platform.app import COMMIT_HEADER
from graph_release.context import bound_release, release_uri
from graph_release.service import Release, get_release_service
from search.eval.apis_auth import apis_auth
from search.eval.cases import (
    CHECK_NAMES,
    DEFAULT_CORPUS_DIR,
    Case,
    expectation_hash,
    load_case_files,
)
from search.eval.checks import (
    ANSWER,
    CHECKS,
    Result,
    ScoreContext,
    Verdict,
    WireSchema,
    score_with_answer,
)
from search.eval.corpus_dataset import (
    SUITE,
    CaseInputs,
    CorpusCase,
    case_inputs,
    corpus_dataset,
    corpus_digest,
)
from search.eval.dataset import item_id_for_case
from search.eval.evaluate import evaluate_or_raise
from search.eval.evaluator import ScoredAnswer, SearchChecks, answer_is_infra
from search.eval.gate import Judgement, Observation, Outcome, exit_code, judge
from search.eval.run_record import parse_run_id, record_manifest, run_record
from search.eval.runner import (
    EVAL_HEADERS,
    SCHEMA_PATH,
    CaseRun,
    build_request,
    members_door_reader,
    run_cases,
)

if TYPE_CHECKING:
    from langfuse import Langfuse

DATASET_PREFIX = "search-"
_SOURCE_KEY = "source"
_NIGHTLY_CONCURRENCY = 8
_NIGHTLY_SAMPLES: Final = 1
_SCHEMA_TIMEOUT_S = 30.0
# Each case is one trial, its modal answer, so the record holds trial 1 of every case.
_TRIAL: Final = 1
_UNKNOWN_REVISION: Final = "unknown"
# Langfuse reads a comma in an experiment name as a list of names, so the sink refuses a
# revision holding one.
_REVISION_SEPARATOR: Final = ","
_PROJECT_ID_ENV: Final = "LANGFUSE_PROJECT_ID"
_DEFAULT_LANGFUSE_BASE_URL: Final = "https://cloud.langfuse.com"
_EXIT_FAILURE: Final = 1
_EXIT_INFRA: Final = 3

# Langfuse answers at or above this status when it, not the request, is at fault.
_SERVER_ERROR: Final = 500
# How much of an API error's body the summary and annotations quote.
_DESCRIBED_BODY_CHARS: Final = 200
# Langfuse answers this when it limits a client's rate, which a later run clears.
_TOO_MANY_REQUESTS: Final = 429

# What opening the served release raises when no release is published, or when its bucket or
# the credentials it needs cannot answer. `ReleaseUriMissingError` is a setup error and exits 2.
_RELEASE_INFRA_ERRORS = (OSError, GoogleAPIError, GoogleAuthError)


def dataset_name(stem: str) -> str:
    """The Langfuse dataset that mirrors ``corpus/<stem>.yaml``."""
    return f"{DATASET_PREFIX}{stem}"


def _source(stem: str) -> str:
    """The case file's path under the repository root."""
    return f"corpus/{stem}.yaml"


def case_item(case: Case, stem: str) -> DatasetItemSpec:
    """The dataset item for ``case``: its authored request keyed by its id, and its
    expectation.

    The input is ``case_inputs(case)`` as JSON, so its ``input_digest`` is the one a run
    record of the corpus dataset stores for the case.
    """
    return DatasetItemSpec(
        item_id=item_id_for_case(case, dataset_name(stem)),
        input=case_inputs(case).model_dump(mode="json"),
        expected_output=case.expect.model_dump(
            mode="json", exclude_unset=True, context={"expand": True}
        ),
        metadata={
            "id": case.id,
            "file": stem,
            "door": case.door,
            "slice": case.slice,
            "activation": case.activation,
            "tags": case.tags,
            "why": case.why,
            "known_failure": case.known_failure.model_dump(mode="json")
            if case.known_failure
            else None,
            "expectation_hash": expectation_hash(case),
        },
    )


@dataclass(frozen=True)
class Synced:
    """The item ids a sync wrote and archived, per dataset it touched."""

    written: dict[str, list[str]]
    archived: dict[str, list[str]]


def sync(by_file: Mapping[str, Sequence[Case]], *, langfuse: Langfuse) -> Synced:
    """Write every case to its file's dataset and archive the items whose case is gone.

    Each dataset is created or updated with metadata naming its case file, then its items
    go through ``upsert_items``, which writes only those Langfuse lacks or holds
    differently. A dataset whose metadata names a case file missing from ``by_file`` has
    every active item archived.
    """
    written: dict[str, list[str]] = {}
    archived: dict[str, list[str]] = {}
    for stem, cases in by_file.items():
        name = dataset_name(stem)
        langfuse.create_dataset(
            name=name,
            description=f"Search eval cases from {_source(stem)}",
            metadata={_SOURCE_KEY: _source(stem)},
        )
        items = [case_item(case, stem) for case in cases]
        written[name] = upsert_items(client=langfuse, dataset=name, items=items)
        archived[name] = _archive_absent(langfuse, name, {item.item_id for item in items})
    for name in _case_file_datasets(langfuse):
        if name not in archived:
            archived[name] = _archive_absent(langfuse, name, set())
    return Synced(written=written, archived=archived)


def _case_file_datasets(langfuse: Langfuse) -> list[str]:
    """Every dataset ``sync`` created, recognised by the case file its metadata names."""
    names, page = [], 1
    while True:
        listed = langfuse.api.datasets.list(page=page, limit=100)
        for dataset in listed.data:
            metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
            if str(metadata.get(_SOURCE_KEY, "")).startswith("corpus/"):
                names.append(dataset.name)
        if page >= listed.meta.total_pages:
            return names
        page += 1


def _archive_absent(langfuse: Langfuse, name: str, kept: set[str]) -> list[str]:
    """Archive every active item of dataset ``name`` outside ``kept``; return their ids."""
    archived = []
    for item in langfuse.get_dataset(name).items:
        if item.id not in kept and item.status == DatasetStatus.ACTIVE:
            langfuse.create_dataset_item(
                dataset_name=name, id=item.id, status=DatasetStatus.ARCHIVED
            )
            archived.append(item.id)
    return archived


def _known(case: Case) -> frozenset[str]:
    """The checks the case marks as known failures."""
    return frozenset(case.known_failure.checks) if case.known_failure else frozenset()


@dataclass
class NightlyCase:
    """One eligible case's run and check results; both stay None when its task raised."""

    case: Case
    file: str
    run: CaseRun | None = None
    results: dict[str, Result] | None = None

    def judgements(self) -> dict[str, Judgement]:
        """Each check judged as the gate judges a new case on one head sample."""
        known = _known(self.case)
        sample = self.run.modal if self.run is not None else None
        judgements = {}
        for check in [*(c for c in CHECK_NAMES if c in CHECKS), ANSWER]:
            result = (self.results or {}).get(check)
            if sample is None:
                infra = "run"
            elif result is None or result.infra:
                infra = "read"
            else:
                infra = None
            observation = Observation(
                verdict=result.verdict if result is not None else Verdict.UNVERIFIABLE,
                infra=infra,
                request_rejected=sample is not None and sample.request_rejected,
                degraded=sample is not None and sample.degraded,
            )
            judgements[check] = judge(
                check=check,
                head=[observation],
                known_failure=check in known,
                known_task_failure="task_outcome" in known,
                new_case=True,
            )
        return judgements

    def trace_id(self) -> str | None:
        """The trace id the modal sample's response carries, or None."""
        modal = self.run.modal if self.run is not None else None
        diagnostics = modal.body.get("diagnostics") if modal is not None else None
        trace_id = diagnostics.get("trace_id") if isinstance(diagnostics, dict) else None
        return trace_id if isinstance(trace_id, str) else None


@dataclass(frozen=True)
class NightlyRun:
    """The nightly's judged cases and what its run record and push read.

    Attributes:
        states: Every eligible case with its run and results, in corpus order.
        report: The Pydantic Evals report of the corpus dataset.
        dataset_digest: ``corpus_digest`` of the cases run.
        outputs: Each case's response body, or its task error, by case id and trial.
    """

    states: list[NightlyCase]
    report: EvaluationReport[CaseInputs, ScoredAnswer, CorpusCase]
    dataset_digest: str
    outputs: dict[tuple[str, int], dict[str, Any]]


def nightly(by_file: Mapping[str, Sequence[Case]], *, url: str, ctx: ScoreContext) -> NightlyRun:
    """Run every eligible case on ``url`` as the corpus dataset.

    Each case's task posts it once, or ``repeat`` times for a case that sets it, with at most
    ``_NIGHTLY_CONCURRENCY`` posts in flight across the run. It scores the modal answer with
    every check and ``answer``, or, when a post failed on infrastructure, the error body
    ``{"task_error": ...}`` with every check, and returns the results as a
    ``ScoredAnswer``, which ``SearchChecks`` labels.

    Args:
        by_file: Cases keyed by case file stem, as ``load_case_files`` returns them.
        url: The app's base URL.
        ctx: What the checks read beside the response.

    Returns:
        The judged cases, the report, the dataset digest and each case's output.

    Raises:
        Exception: The first exception a case's task raised, a fault in the nightly; the
            other cases are cancelled (``evaluate.evaluate_or_raise``).
        RuntimeError: The evaluator raised.
    """
    dataset = corpus_dataset(by_file, implemented_checks=CHECKS)
    dataset.add_evaluator(SearchChecks())
    by_id = {case.id: (case, stem) for stem, cases in by_file.items() for case in cases}
    states: dict[str, NightlyCase] = {}
    outputs: dict[tuple[str, int], dict[str, Any]] = {}

    async def evaluate() -> EvaluationReport[CaseInputs, ScoredAnswer, CorpusCase]:
        """Evaluate the dataset with one bound on posts in flight and one scoring pool
        across every case, stopping at the first case that raises."""
        gate = asyncio.Semaphore(_NIGHTLY_CONCURRENCY)
        loop = asyncio.get_running_loop()
        scorers = ThreadPoolExecutor(max_workers=_NIGHTLY_CONCURRENCY)

        async def task(inputs: CaseInputs) -> ScoredAnswer:
            """Post one case once and score its answer."""
            case, stem = by_id[inputs.case_id]
            state = states[case.id] = NightlyCase(case, stem)
            runs = await run_cases(
                [case], url=url, samples=_NIGHTLY_SAMPLES, concurrency=1, gate=gate
            )
            state.run = runs[case.id]
            modal = state.run.modal
            body = modal.body if modal is not None else {"task_error": state.run.infra_error}
            outputs[(case.id, _TRIAL)] = body
            # The checks read the graph under the bound catalogue and release, so the scoring
            # thread runs in a copy of this context.
            context = contextvars.copy_context()
            state.results = await loop.run_in_executor(
                scorers, functools.partial(context.run, score_with_answer, case, body, ctx)
            )
            return ScoredAnswer(
                results=state.results,
                infra=answer_is_infra(state.results, run_failed=modal is None),
                trace_id=state.trace_id(),
            )

        with scorers:
            return await evaluate_or_raise(
                dataset, task, name=SUITE, max_concurrency=_NIGHTLY_CONCURRENCY
            )

    # A case that cannot build its request raises here, before any post.
    for eval_case in dataset.cases:
        build_request(by_id[eval_case.inputs.case_id][0])
    report = asyncio.run(evaluate())
    return NightlyRun(
        states=[states[eval_case.inputs.case_id] for eval_case in dataset.cases],
        report=report,
        dataset_digest=corpus_digest(dataset),
        outputs=outputs,
    )


def push_nightly(
    run: Run,
    *,
    by_file: Mapping[str, Sequence[Case]],
    outputs: Mapping[tuple[str, int], object],
    client: Langfuse,
) -> str:
    """Push the nightly's run record to Langfuse, each case under the item ``sync`` keeps.

    Args:
        run: The nightly's run record.
        by_file: The cases the run and the sync read, keyed by case file stem.
        outputs: Each case's response body by case id and trial.
        client: A Langfuse client for the project (``dfo_evals.langfuse_sink.connect``).

    Returns:
        The experiment run name, shared by every case file's experiment.

    Raises:
        UnknownCaseError: The run holds a case no case file holds.
    """
    refs = {
        case.id: DatasetItemRef(
            dataset=dataset_name(stem), item_id=item_id_for_case(case, dataset_name(stem))
        )
        for stem, cases in by_file.items()
        for case in cases
    }

    def item_of(case_id: str) -> DatasetItemRef:
        """The dataset item that mirrors ``case_id``."""
        if case_id not in refs:
            raise UnknownCaseError(f"case {case_id!r} is in no case file")
        return refs[case_id]

    return push_run(client=client, run=run, item_of=item_of, outputs=outputs)


def nightly_markdown(
    states: Sequence[NightlyCase],
    *,
    experiment: str | None,
    record: Path | None,
    push_error: str | None = None,
) -> str:
    """The nightly summary: unmarked failures and infrastructure losses, check by check,
    then where the run went.

    Args:
        states: The nightly's judged cases.
        experiment: The Langfuse experiment run name, or None when the push did not
            complete.
        record: The run record's directory, or None when none was written.
        push_error: Why the push did not complete, or None.
    """
    failures, infra = [], []
    for state in states:
        where = f"`{state.case.id}` ({state.file})"
        for check, judgement in state.judgements().items():
            if judgement.outcome == Outcome.INFRA:
                error = state.run.infra_error if state.run is not None else "the task raised"
                infra.append(f"- {where} `{check}`: {error or 'a check read failed'}")
            elif judgement.blocks:
                result = (state.results or {}).get(check)
                detail = result.detail if result is not None and result.detail else "no pass"
                trace = state.trace_id()
                failures.append(
                    f"- {where} `{check}`: {detail}" + (f" (trace `{trace}`)" if trace else "")
                )
    lines = [
        "## Nightly Search eval",
        "",
        f"{len(states)} eligible cases; {len(failures)} unmarked failing checks; "
        f"{len(infra)} checks lost to infrastructure.",
    ]
    for title, entries in (("Unmarked failures", failures), ("Infrastructure", infra)):
        if entries:
            lines += ["", f"### {title}", "", *entries]
    if experiment is not None:
        langfuse = f"- Langfuse experiment: `{experiment}`"
    else:
        langfuse = (
            f"- Langfuse: the push did not complete ({push_error or 'no push'}); Langfuse has "
            "no experiment, or a partial one, for this night"
        )
        if record is not None:
            langfuse += ", and the run record holds it in full"
    record_line = (
        f"- Run record: `{record.parent.name}/{record.name}/`, kept as the workflow's "
        "`eval-records` artifact and, on `dev`, uploaded to the eval runs bucket"
        if record is not None
        else "- Run record: not written"
    )
    lines += ["", "### Run", "", langfuse, record_line]
    return "\n".join(lines) + "\n"


def nightly_exit_code(states: Sequence[NightlyCase]) -> int:
    """1 on an unmarked failure, else 3 on infrastructure, else 0 (``gate.exit_code``)."""
    return exit_code(j.outcome for s in states for j in s.judgements().values())


def _project_id() -> str:
    """The Langfuse project the sync and the nightly write to, from ``LANGFUSE_PROJECT_ID``.

    Raises:
        RuntimeError: The variable is unset or empty.
    """
    project_id = os.environ.get(_PROJECT_ID_ENV, "")
    if not project_id:
        raise RuntimeError(f"{_PROJECT_ID_ENV} is unset, so the keys' project cannot be checked")
    return project_id


def _connect(project_id: str) -> Langfuse:
    """A Langfuse client for ``project_id``, from the environment's keys and base URL.

    The client sends on a tracer provider of its own, so ``LANGFUSE_SAMPLE_RATE`` does not
    thin what it writes, and ``connect`` refuses keys of another project and
    ``LANGFUSE_TRACING_ENABLED=false``.

    Raises:
        _InfraError: Langfuse did not answer the project check: the network failed, or it
            answered 5xx or 429. Keys it refuses, or keys of another project, raise as they
            are.
    """
    try:
        return connect(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            base_url=os.environ.get("LANGFUSE_BASE_URL")
            or os.environ.get("LANGFUSE_HOST")
            or _DEFAULT_LANGFUSE_BASE_URL,
            project_id=project_id,
        )
    except (httpx.HTTPError, ApiError) as exc:
        if not _langfuse_at_fault(exc):
            raise
        raise _InfraError(f"Langfuse did not answer: {_describe(exc)}") from exc


def _describe(exc: BaseException) -> str:
    """A short, one-line account of an error, for the summary and annotations: an API error
    as its status and the start of its body, without the response headers it prints."""
    if isinstance(exc, ApiError):
        return f"ApiError {exc.status_code}: {repr(exc.body)[:_DESCRIBED_BODY_CHARS]}"
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def _command_text(text: str) -> str:
    """Escape text for a GitHub Actions workflow command, which ends at a newline."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _langfuse_at_fault(exc: BaseException) -> bool:
    """Whether ``exc`` is Langfuse or the network failing, which a later run clears: a push
    Langfuse did not show in full, a transport error, a 5xx, a 429, or an API error with no
    status. Any other API error, such as a 404 from an endpoint Langfuse removed, is a fault
    in the request that every run would repeat."""
    if isinstance(exc, PushIncompleteError | PushTimeoutError | httpx.HTTPError):
        return True
    if isinstance(exc, ApiError):
        status = exc.status_code
        return status is None or status >= _SERVER_ERROR or status == _TOO_MANY_REQUESTS
    return False


class _InfraError(Exception):
    """A read the run needs before any case failed on infrastructure; ``main`` exits 3."""


def _schema(url: str) -> tuple[Any, str]:
    """The search schema the app at ``url`` serves, and the commit its response names.

    The commit is ``unknown`` when the ``x-dfo-commit`` header is absent or empty, or when
    it holds a comma, which a pushed run's revision cannot hold.
    """
    try:
        response = httpx.get(
            f"{url.rstrip('/')}{SCHEMA_PATH}",
            timeout=_SCHEMA_TIMEOUT_S,
            headers=EVAL_HEADERS,
            auth=apis_auth(),
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise _InfraError(f"{type(exc).__name__}: {exc}") from exc
    revision = response.headers.get(COMMIT_HEADER, "").strip()
    if not revision or _REVISION_SEPARATOR in revision:
        revision = _UNKNOWN_REVISION
    return response.json(), revision


def _open_release() -> Release:
    """The release the process-wide release service serves, opened from ``GRAPH_RELEASE_URI``
    on the first call."""
    try:
        return get_release_service().snapshot()
    except _RELEASE_INFRA_ERRORS as exc:
        raise _InfraError(
            f"the graph release at {release_uri()} did not open: {type(exc).__name__}: {exc}"
        ) from exc


def _sync(by_file: Mapping[str, Sequence[Case]], langfuse: Langfuse) -> None:
    """``sync``, printing how many items it wrote and which it archived."""
    synced = sync(by_file, langfuse=langfuse)
    for name, ids in synced.archived.items():
        wrote = len(synced.written.get(name, []))
        print(
            f"{name}: wrote {wrote}, archived {len(ids)}" + (f" ({', '.join(ids)})" if ids else "")
        )


def _run_nightly(
    by_file: Mapping[str, Sequence[Case]],
    url: str,
    schema: Mapping[str, Any],
    release: Release,
) -> NightlyRun:
    """``nightly`` under the served origin catalogue and ``release``, with live clients."""
    from origin_catalogue import bound_catalogue
    from origin_catalogue.service import get_catalogue_service
    from search.eval.live_clients import score_clients

    # The graph client reads under a bound origin catalogue and graph release. The tasks, the
    # requests they build and the scoring threads inherit both bindings.
    catalogue = get_catalogue_service().snapshot()
    with score_clients() as clients, bound_catalogue(catalogue), bound_release(release):
        ctx = ScoreContext(
            schema=WireSchema.from_document(schema),
            graph=clients.graph,
            sdmx=clients.sdmx,
            coverage=clients.coverage,
            members=members_door_reader(url),
        )
        return nightly(by_file, url=url, ctx=ctx)


def main(argv: Sequence[str] | None = None) -> int:
    """Sync the case files, or run the nightly, and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--nightly", metavar="URL", help="run the nightly on URL")
    parser.add_argument("--summary", type=Path, help="write the nightly Markdown summary here")
    parser.add_argument(
        "--records",
        type=Path,
        help="write the nightly's run record under this root, as <root>/apis-search/<run id>/",
    )
    parser.add_argument(
        "--run-id",
        type=parse_run_id,
        help="the nightly run record's id (default: nightly-<UTC start time>)",
    )
    args = parser.parse_args(argv)
    if (args.records or args.run_id) and not args.nightly:
        parser.error("--records and --run-id are the nightly's")
    # A crash must not read as an unmarked failure (1) or infrastructure (3), which the
    # nightly workflow can report and pass, so every unexpected error exits 2.
    try:
        return _main(args)
    except Exception:
        traceback.print_exc()
        return 2


def _main(args: argparse.Namespace) -> int:
    """Run the sync, or the nightly, that ``args`` select; return the exit code."""
    by_file = load_case_files(args.corpus_dir)
    if args.nightly:
        return _main_nightly(args, by_file)
    try:
        langfuse = _connect(_project_id())
    except _InfraError as exc:
        print(f"eval infrastructure error: {exc}", file=sys.stderr)
        return _EXIT_INFRA
    try:
        _sync(by_file, langfuse)
    finally:
        langfuse.shutdown()
    return 0


def _main_nightly(args: argparse.Namespace, by_file: Mapping[str, Sequence[Case]]) -> int:
    """Sync, run the nightly, record and push it, and write its summary.

    One Langfuse client, checked against ``LANGFUSE_PROJECT_ID`` before anything is read or
    written, syncs the datasets and pushes the run, so keys of another project write nothing.
    When Langfuse or the network fails that check, the nightly still runs and writes its
    record, skipping the sync and the push, which the summary reports as lost to
    infrastructure.

    Returns:
        The nightly's exit code: 3 when Langfuse, the schema or the release does not answer,
        unless the checks found an unmarked failure (1).
    """
    project_id = _project_id()
    try:
        langfuse: Langfuse | None = _connect(project_id)
        unreachable = None
    except _InfraError as exc:
        print(f"eval infrastructure error: {exc}", file=sys.stderr)
        langfuse = None
        unreachable = (
            f"Langfuse did not answer at the start, so nothing was synced or pushed ({exc})"
        )
    try:
        return _nightly_with(args, by_file, langfuse, unreachable=unreachable)
    finally:
        if langfuse is not None:
            langfuse.shutdown()


def _nightly_with(
    args: argparse.Namespace,
    by_file: Mapping[str, Sequence[Case]],
    langfuse: Langfuse | None,
    *,
    unreachable: str | None,
) -> int:
    """``_main_nightly``'s body. With no client, ``unreachable`` says why, and the run
    skips the sync and the push."""
    started_at = datetime.now(UTC)
    try:
        # The app's schema loads first, so an outage of the app reads as one.
        schema, revision = _schema(args.nightly)
        release = _open_release()
        sync_crash: Exception | None = None
        if langfuse is not None:
            try:
                _sync(by_file, langfuse)
            except Exception as exc:
                # The run goes on without Langfuse either way, so its record and summary are
                # written; a sync error Langfuse is not at fault for is raised after them.
                langfuse = None
                if _langfuse_at_fault(exc):
                    print(
                        f"eval infrastructure error: the dataset sync failed: {_describe(exc)}",
                        file=sys.stderr,
                    )
                    unreachable = f"Langfuse failed during the dataset sync, so the run was not pushed ({_describe(exc)})"
                else:
                    sync_crash = exc
                    unreachable = (
                        f"the dataset sync failed, so the run was not pushed ({_describe(exc)})"
                    )
        run = _run_nightly(by_file, args.nightly, schema, release)
    except _InfraError as exc:
        print(f"eval infrastructure error: {exc}", file=sys.stderr)
        return _EXIT_INFRA

    manifest = record_manifest(
        run_id=args.run_id or started_at.strftime("nightly-%Y%m%dT%H%M%SZ"),
        started_at=started_at,
        finished_at=datetime.now(UTC),
        revision=revision,
        environment={"url": args.nightly, "samples": str(_NIGHTLY_SAMPLES)},
        dataset_digest=run.dataset_digest,
    )
    record = run_record(run.report, manifest=manifest)
    path = write_run(root=args.records, run=record) if args.records else None
    code = nightly_exit_code(run.states)

    # The summary is written whatever the push does, so a push that fails after a full run
    # keeps the run's findings. Langfuse or the network failing (``_langfuse_at_fault``) is
    # infrastructure, which an unmarked failure outranks. Any other push error is a crash,
    # raised once the summary is written.
    experiment, push_error, crash = None, None, sync_crash
    if langfuse is None:
        push_error = unreachable
        if crash is None:
            code = _EXIT_FAILURE if code == _EXIT_FAILURE else _EXIT_INFRA
    else:
        try:
            experiment = push_nightly(record, by_file=by_file, outputs=run.outputs, client=langfuse)
        except Exception as exc:
            push_error = _describe(exc)
            if _langfuse_at_fault(exc):
                code = _EXIT_FAILURE if code == _EXIT_FAILURE else _EXIT_INFRA
            else:
                crash = exc

    markdown = nightly_markdown(
        run.states, experiment=experiment, record=path, push_error=push_error
    )
    print(markdown)
    if args.summary:
        args.summary.write_text(markdown, encoding="utf-8")
    if push_error is not None and os.environ.get("GITHUB_ACTIONS") == "true":
        # The nightly's usual exit 1 would hide a failed push, so it gets its own annotation.
        print(
            f"::warning::{_command_text(f'The nightly Langfuse push did not complete: {push_error}')}"
        )
    if crash is not None:
        raise crash
    return code


if __name__ == "__main__":
    sys.exit(main())
