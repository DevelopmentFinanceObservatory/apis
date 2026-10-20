"""Mirror the case files to Langfuse, and run the nightly experiment against a deployed app.

``python -m search.eval.sync`` keeps one Langfuse dataset per case file, ``search-<stem>``.
It upserts one item per case, keyed by case id, whose input is the request the runner posts
and whose expected output is the case's ``expect`` with families expanded. It archives every
active item whose case is gone, including the items of a deleted case file.

``python -m search.eval.sync --nightly URL`` syncs, then runs every eligible case once on
the app at URL as one Langfuse experiment per case file, scoring each answer with the gate's
checks and the derived ``answer`` check as the gate scores them. It lists every check
that did not pass and is outside its case's ``known_failure``, and exits 0 when there is
none, 1 when there is one, and 3 when the app's schema does not load, or when a post or a
check's read failed on infrastructure and nothing else failed. A post that failed on infrastructure is a known failure on a case
whose ``known_failure.checks`` holds ``task_outcome``, as in the gate. Any crash exits 2.

Each check writes a Langfuse score of 1 on a pass and 0 on a failure, and no score when it
is unverifiable, n/a or unreached, since none of those says whether the answer was right.
``answer`` is unreached when a check it gathers never ran, so on a case whose
``disposition`` is a known failure, an answer in a status the case does not accept writes
no ``answer`` score.

Requires ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``. The nightly also reads the
graph settings, ``COVERAGE_ARTIFACT_URI`` and ``ORIGIN_CATALOGUE_URI`` the checks score
against.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from langfuse import Evaluation
from langfuse.api import DatasetStatus

from search.eval.apis_auth import apis_auth
from search.eval.cases import (
    CHECK_NAMES,
    DEFAULT_CORPUS_DIR,
    Case,
    expectation_hash,
    ineligibility,
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
from search.eval.dataset import item_id_for_case
from search.eval.gate import Judgement, Observation, Outcome, exit_code, judge
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
_SCHEMA_TIMEOUT_S = 30.0


def dataset_name(stem: str) -> str:
    """The Langfuse dataset that mirrors ``corpus/<stem>.yaml``."""
    return f"{DATASET_PREFIX}{stem}"


def _source(stem: str) -> str:
    return f"corpus/{stem}.yaml"


def case_item(case: Case, stem: str) -> dict[str, Any]:
    """The dataset item for ``case``: the request the runner posts, and its expectation."""
    return {
        "id": item_id_for_case(case, dataset_name(stem)),
        "input": build_request(case),
        "expected_output": case.expect.model_dump(
            mode="json", exclude_unset=True, context={"expand": True}
        ),
        "metadata": {
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
    }


def sync(by_file: Mapping[str, Sequence[Case]], *, langfuse: Langfuse) -> dict[str, list[str]]:
    """Upsert every case into its file's dataset and archive the items whose case is gone.

    A dataset whose metadata names a case file missing from ``by_file`` has every active
    item archived. Returns the archived item ids per dataset, for every dataset touched.
    """
    archived: dict[str, list[str]] = {}
    for stem, cases in by_file.items():
        name = dataset_name(stem)
        langfuse.create_dataset(
            name=name,
            description=f"Search eval cases from {_source(stem)}",
            metadata={_SOURCE_KEY: _source(stem)},
        )
        kept = set()
        for case in cases:
            item = case_item(case, stem)
            kept.add(item["id"])
            langfuse.create_dataset_item(dataset_name=name, status=DatasetStatus.ACTIVE, **item)
        archived[name] = _archive_absent(langfuse, name, kept)
    for name in _case_file_datasets(langfuse):
        if name not in archived:
            archived[name] = _archive_absent(langfuse, name, set())
    return archived


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
    archived = []
    for item in langfuse.get_dataset(name).items:
        if item.id not in kept and item.status == DatasetStatus.ACTIVE:
            langfuse.create_dataset_item(
                dataset_name=name, id=item.id, status=DatasetStatus.ARCHIVED
            )
            archived.append(item.id)
    return archived


def _known(case: Case) -> frozenset[str]:
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
        modal = self.run.modal if self.run is not None else None
        diagnostics = modal.body.get("diagnostics") if modal is not None else None
        return diagnostics.get("trace_id") if isinstance(diagnostics, dict) else None


def nightly(
    by_file: Mapping[str, Sequence[Case]],
    *,
    url: str,
    ctx: ScoreContext,
    langfuse: Langfuse,
) -> tuple[list[NightlyCase], dict[str, str | None]]:
    """Run every eligible case once on ``url``, one Langfuse experiment per case file.

    The datasets must be synced first. Returns every eligible case with its run and
    results, and each dataset's run URL. Raises the first exception a case's task or
    checks raised, once that file's experiment returns.
    """
    states: dict[str, NightlyCase] = {}
    run_urls: dict[str, str | None] = {}

    # `run_experiment` logs and drops what a task or an evaluator raises, which the summary
    # would then read as infrastructure. Both record it here, and the first is raised again
    # once the experiment returns.
    errors: list[Exception] = []

    async def task(*, item: Any, **_: Any) -> dict:
        try:
            state = states[item.metadata["id"]]
            runs = await run_cases([state.case], url=url, samples=1, concurrency=1)
        except Exception as exc:
            errors.append(exc)
            raise
        state.run = runs[state.case.id]
        modal = state.run.modal
        return modal.body if modal is not None else {"task_error": state.run.infra_error}

    async def checks(*, output: Any, metadata: dict[str, Any] | None, **_: Any) -> list[Evaluation]:
        assert metadata is not None, "every synced item carries its case id"
        state = states[metadata["id"]]
        try:
            state.results = await asyncio.to_thread(score_with_answer, state.case, output, ctx)
        except Exception as exc:
            errors.append(exc)
            raise
        return [
            Evaluation(
                name=name,
                value=1.0 if result.verdict == Verdict.PASS else 0.0,
                comment=result.detail or None,
            )
            for name, result in state.results.items()
            if result.verdict in (Verdict.PASS, Verdict.FAIL)
        ]

    for stem, cases in by_file.items():
        eligible = [case for case in cases if not ineligibility(case, implemented_checks=CHECKS)]
        if not eligible:
            continue
        name = dataset_name(stem)
        wanted = {item_id_for_case(case, name) for case in eligible}
        states.update({case.id: NightlyCase(case, stem) for case in eligible})
        # `DatasetClient.run_experiment` runs every item, archived and ineligible ones too.
        # The client's form takes the chosen items and still links each to the dataset run.
        items = [item for item in langfuse.get_dataset(name).items if item.id in wanted]
        result = langfuse.run_experiment(
            name="nightly",
            description=f"One sample per eligible case of {_source(stem)} against {url}",
            data=items,
            task=task,
            evaluators=[checks],
            max_concurrency=_NIGHTLY_CONCURRENCY,
            metadata={"url": url},
        )
        if errors:
            raise errors[0]
        run_urls[name] = result.dataset_run_url
    return list(states.values()), run_urls


def nightly_markdown(states: Sequence[NightlyCase], run_urls: Mapping[str, str | None]) -> str:
    """The nightly summary: unmarked failures and infrastructure losses, check by check."""
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
    lines += ["", "### Langfuse runs", ""]
    lines += [f"- `{name}`: {url or 'no run URL'}" for name, url in run_urls.items()]
    return "\n".join(lines) + "\n"


def nightly_exit_code(states: Sequence[NightlyCase]) -> int:
    """1 on an unmarked failure, else 3 on infrastructure, else 0 (``gate.exit_code``)."""
    return exit_code(j.outcome for s in states for j in s.judgements().values())


def _client() -> Langfuse:
    """A Langfuse client, raising clearly when the credentials are unset."""
    from langfuse import get_client

    missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Langfuse credentials missing: {', '.join(missing)}")
    return get_client()


def _run_nightly(
    by_file: Mapping[str, Sequence[Case]], url: str, schema: Mapping[str, Any], langfuse: Langfuse
) -> tuple[list[NightlyCase], dict[str, str | None]]:
    from graph_release.context import bound_served_release, optional_release_service
    from origin_catalogue import bound_catalogue
    from origin_catalogue.service import get_catalogue_service
    from search.eval.live_clients import score_clients

    # The graph client reads under a bound origin catalogue, and the served graph release when
    # `GRAPH_RELEASE_URI` is set. The experiment's tasks and the scoring threads inherit both
    # bindings.
    catalogue = get_catalogue_service().snapshot()
    with (
        score_clients() as clients,
        bound_catalogue(catalogue),
        bound_served_release(optional_release_service()),
    ):
        ctx = ScoreContext(
            schema=WireSchema.from_document(schema),
            graph=clients.graph,
            sdmx=clients.sdmx,
            coverage=clients.coverage,
            members=members_door_reader(url),
        )
        return nightly(by_file, url=url, ctx=ctx, langfuse=langfuse)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--nightly", metavar="URL", help="run the nightly experiment on URL")
    parser.add_argument("--summary", type=Path, help="write the nightly Markdown summary here")
    args = parser.parse_args(argv)
    # A crash must not read as an unmarked failure (1) or infrastructure (3), which the
    # nightly workflow can report and pass, so every unexpected error exits 2.
    try:
        return _main(args)
    except Exception:
        traceback.print_exc()
        return 2


def _main(args: argparse.Namespace) -> int:
    by_file = load_case_files(args.corpus_dir)
    langfuse = _client()
    try:
        for name, ids in sync(by_file, langfuse=langfuse).items():
            print(f"{name}: archived {len(ids)}" + (f" ({', '.join(ids)})" if ids else ""))
        if not args.nightly:
            return 0
        try:
            response = httpx.get(
                f"{args.nightly.rstrip('/')}{SCHEMA_PATH}",
                timeout=_SCHEMA_TIMEOUT_S,
                headers=EVAL_HEADERS,
                auth=apis_auth(),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"eval infrastructure error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        states, run_urls = _run_nightly(by_file, args.nightly, response.json(), langfuse)
    finally:
        langfuse.flush()
    markdown = nightly_markdown(states, run_urls)
    print(markdown)
    if args.summary:
        args.summary.write_text(markdown, encoding="utf-8")
    return nightly_exit_code(states)


if __name__ == "__main__":
    sys.exit(main())
