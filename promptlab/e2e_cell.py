"""Run fresh end-to-end resolves for each case and rung seed.

Both composed prompts pass through `LadderLLM`. The real guard, neighbour check,
repair, and twin diversion run. Captures supply raw-text cases and the original
`[[TODAY]]` date, while `GraphCache` serves captured graph reads. Live runs make
new LLM calls. Offline tests use `FakeLLM`. Each response is scored by the eval checks
(`search.eval.checks`) against the case, as the paired gate scores it. Only raw-text cases
exercise the ladder. Dataset-resubmit cases bypass extract and bind.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from unittest.mock import patch

from promptlab.cells import _run_concurrently
from promptlab.compose import compose
from promptlab.llm import CallRecord, LadderLLM
from promptlab.manifest import Rung
from promptlab.scoring import seed_flip_rate
from promptlab.store import CaptureStore, CellStore
from search import RESOLVE_RESPONSE_ADAPTER, ResolveRequest, ResolveResponse
from search.engine import config
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import LLM, SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.core import resolve_async
from search.eval.cases import CHECK_NAMES, Case
from search.eval.checks import ScoreContext, Verdict, WireSchema, score_results
from search.eval.graph import GraphClient
from series_coverage.shared import get_coverage_service

_TODAY_PLACEHOLDER = "[[TODAY]]"

# pipeline.py reads bind top-k from `settings().bind_top_k` on each call. Patch
# process settings for rung-specific top-k. The harness's `pac` override uses
# the same settings seam.


class _E2eGraphClient(EngineGraphClient, GraphClient, Protocol):
    """Graph methods required by both `resolve_async` and the checks."""


def _seeded(llm: SupportsLLM, seed: int) -> SupportsLLM:
    """Shallow-copy a real `LLM` with the seed set, preserving its other settings. Doubles
    such as `FakeLLM` pass through unchanged and replay the same answer per seed."""
    if not isinstance(llm, LLM):
        return llm
    clone = copy.copy(llm)
    clone._seed = seed
    return clone


def _check_rates(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Pass rate per check, as `<check>_pass_rate`, over the records that scored it.

    Checks that are n/a or unverifiable for a record leave it out of that check's rate, and a
    check with no scored record has no rate.
    """
    verdicts: dict[str, Counter[str]] = {name: Counter() for name in CHECK_NAMES}
    for record in records:
        for name, result in record["checks"].items():
            verdicts[name][result["verdict"]] += 1
    return {
        f"{name}_pass_rate": counts[Verdict.PASS] / scored
        for name, counts in verdicts.items()
        if (scored := counts[Verdict.PASS] + counts[Verdict.FAIL])
    }


def seed_flip_rate_mean(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Mean `seed_flip_rate` across cases, each over its own seeds' outputs.

    Returns None when every case ran once.
    """
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["case_id"], []).append(record)
    flips = [
        flip
        for case_records in by_case.values()
        if (
            flip := seed_flip_rate(
                [r["output"] for r in sorted(case_records, key=lambda r: r["seed"])]
            )
        )
        is not None
    ]
    return sum(flips) / len(flips) if flips else None


def _score_context(graph: _E2eGraphClient, sdmx: EngineSdmxClient) -> ScoreContext:
    """A head-side context whose wire schema is this process's own response model."""
    schema = WireSchema.from_document(
        {"response": RESOLVE_RESPONSE_ADAPTER.json_schema(mode="serialization")}
    )
    return ScoreContext(
        schema=schema, graph=graph, sdmx=sdmx, coverage=get_coverage_service(), side="head"
    )


async def run_e2e_cell(
    rung: Rung,
    *,
    capture: CaptureStore,
    graph: _E2eGraphClient,
    out: CellStore,
    llm: SupportsLLM,
    sdmx: EngineSdmxClient,
    concurrency: int = 16,
    cases: Sequence[Case],
) -> None:
    """Run one fresh resolve per captured raw-text case and rung seed, then score it.

    Requires a complete capture, which supplies the case ids and the original `[[TODAY]]`
    date only. Extract and bind calls run fresh.
    Both stage prompts are substituted through `LadderLLM`. Graph and SDMX clients
    feed the pipeline and the checks. A completed key in `out` is skipped, and
    `finish()` aggregates all stored records after the run.

    `concurrency` caps in-flight resolves. Non-raw-text cases raise
    `NotImplementedError`."""
    capture.require_complete()

    extract_system = compose(rung.extract).replace(
        _TODAY_PLACEHOLDER, capture.manifest.substituted_date
    )
    bind_system = compose(rung.bind)
    systems = {"extract": extract_system, "bind": bind_system}

    cases_by_id = {c.id: c for c in cases}
    ctx = _score_context(graph, sdmx)

    done = out.done_keys()
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(case_id: str, query: str, seed: int) -> None:
        # One record stores a full resolve, which may include several LLM calls. Resumes
        # key by case and seed with `leg=0`. All internal calls remain in `llm_calls`.
        if (case_id, 0, seed) in done:
            return
        case = cases_by_id[case_id]
        if case.door != "raw_text":
            raise NotImplementedError(
                f"run_e2e_cell only replays raw_text cases; {case_id!r} is "
                f"door={case.door!r}. See this module's docstring."
            )
        sink: list[CallRecord] = []
        seeded_llm = LadderLLM(_seeded(llm, seed), systems=systems, sink=sink)
        req = ResolveRequest.model_validate({"input": {"kind": "raw_text", "query": query}})
        async with semaphore:
            resp: ResolveResponse = await resolve_async(req, graph=graph, llm=seeded_llm, sdmx=sdmx)
        output = resp.model_dump(mode="json", exclude_none=True)

        # Scored in line, so a record is appended before the next resolve starts; resume
        # after a kill relies on that order.
        results = score_results(case, output, ctx)

        out.append(
            case_id=case_id,
            leg=0,
            seed=seed,
            call={
                "output": output,
                "checks": {
                    name: {"verdict": result.verdict.value, "detail": result.detail}
                    for name, result in results.items()
                },
                "llm_calls": [dataclasses.asdict(c) for c in sink],
            },
        )

    coros = (
        _run_one(record.case_id, record.query, seed)
        for record in capture.extract_records()
        for seed in rung.run.seeds
    )
    with patch.object(
        config,
        "_SETTINGS",
        config.settings().model_copy(update={"bind_top_k": rung.run.top_k}),
    ):
        await _run_concurrently(coros)

    out.finish(_aggregate(out.call_records(), seeds=rung.run.seeds))


def _aggregate(records: tuple[dict[str, Any], ...], *, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Aggregate all stored records, including those from earlier resumed runs. Report the
    pass rate of every check per seed, and `seed_flip_rate` across seeds. Omit
    `seed_flip_rate` when each case ran once."""
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for record in records:
        by_seed.setdefault(record["seed"], []).append(record)

    scores: dict[str, Any] = {
        "per_seed": {
            str(seed): _check_rates(seed_records) for seed, seed_records in by_seed.items()
        }
    }
    if (flip := seed_flip_rate_mean(records)) is not None:
        scores["seed_flip_rate"] = flip
    return scores
