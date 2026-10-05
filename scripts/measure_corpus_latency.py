"""Measure live latency for corpus cases whose ``why`` makes a timing claim.

Select cases whose ``why`` mentions seconds or a deadline. Run each case sequentially through
the eval runner's request builder (default 5 runs), with the shared skills index the service
acquires, so each run includes skill matching.

Each run records wall-clock seconds, status, sorted warning codes, and ``timed_out``. Each
artifact row stores those values, the ``[min, max]`` seconds, warning-code union, status-flip, and
timeout flags. The report flags cases whose maximum exceeds the deadline or whose status varies.

The JSON header records measurement kind, runs per case, engine build, and deadline. Host load
and deployment shape timing readings. Output defaults to ``baselines/eval-latency.json``. Pass
``--out`` to change it.

Uses ``GEMINI_API_KEY`` and defaults to the dfo-dev graph. Reads the cases in ``corpus/``.

Usage:
    uv run --env-file .env --extra engine --extra eval python scripts/measure_corpus_latency.py
    uv run --env-file .env --extra engine --extra eval python scripts/measure_corpus_latency.py \\
        --runs 5 --ids crs-gross-disbursements-kenya,crs-nominal-disbursements-kenya
    uv run --env-file .env --extra engine --extra eval python scripts/measure_corpus_latency.py \\
        --deadline-seconds 45 --out baselines/eval-latency.json
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from entity_resolution import acquire_shared, release_shared
from search.eval.cases import Case, load_cases
from search.eval.runner import build_request
from search.models import ResolveRequest

if TYPE_CHECKING:
    from entity_resolution import SharedResolver
    from search.models import ResolveResponse

    ResolveCallable = Callable[[ResolveRequest], Awaitable[ResolveResponse]]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "baselines" / "eval-latency.json"

_MEASUREMENT = "corpus_latency_runs"
_DEADLINE_MENTION_RE = re.compile(r"deadline|\bseconds?\b", re.IGNORECASE)
_ERROR_STATUS = "error"


def _acquire_latency_resolvekit() -> SharedResolver:
    """Acquire resolvekit before probing so cache setup errors fail at startup.

    Without a cache dir `acquire_shared()` raises `RuntimeError`, which `probe_case` would
    record as an ordinary error outcome. The pack builds offline under gitignored ``runs/``
    with ``RESOLVEKIT_OFFLINE=1``.
    """
    os.environ["RESOLVEKIT_OFFLINE"] = "1"
    if not os.environ.get("RESOLVEKIT_CACHE_DIR"):
        default_dir = _REPO_ROOT / "runs" / "measure_corpus_latency_resolvekit"
        default_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RESOLVEKIT_CACHE_DIR"] = str(default_dir)
    return acquire_shared()


def default_selection(cases: list[Case]) -> list[Case]:
    """Return id-sorted cases whose ``why`` carries a timing claim."""
    return sorted(
        (case for case in cases if _DEADLINE_MENTION_RE.search(case.why)), key=lambda case: case.id
    )


def select_cases(cases: list[Case], ids: str | None) -> list[Case]:
    """Select comma-separated ids sorted by id, or the default set when `ids` is None.

    Select ids present in the corpus. `main` exits when no requested id matches.
    """
    if ids is None:
        return default_selection(cases)
    wanted = {case_id.strip() for case_id in ids.split(",") if case_id.strip()}
    return sorted((c for c in cases if c.id in wanted), key=lambda case: case.id)


def _outcome_label(response: ResolveResponse) -> str:
    """Return a resolution's status, or the kind of a catalogue or declined answer."""
    return response.status if response.kind == "resolution" else response.kind


def summarize_runs(runs: list[dict]) -> dict:
    """Aggregate run times, statuses, warning codes, and timeout flags for one case.

    `status_flip` ignores ``error`` outcomes. Empty runs produce zero min/max times.
    """
    seconds = [run["seconds"] for run in runs]
    statuses = [run["status"] for run in runs]
    warning_codes = sorted({code for run in runs for code in run["warning_codes"]})
    observed = {status for status in statuses if status != _ERROR_STATUS}
    return {
        "runs": runs,
        "seconds": seconds,
        "min_seconds": min(seconds) if seconds else 0.0,
        "max_seconds": max(seconds) if seconds else 0.0,
        "statuses": statuses,
        "status_flip": len(observed) > 1,
        "warning_codes": warning_codes,
        "timed_out": any(run["timed_out"] for run in runs),
    }


def _run_record(seconds: float, *, status: str, warning_codes: list[str], timed_out: bool) -> dict:
    return {
        "seconds": seconds,
        "status": status,
        "warning_codes": warning_codes,
        "timed_out": timed_out,
    }


async def probe_case(case: Case, *, runs: int, resolve: ResolveCallable) -> dict:
    """Run the eval-built request sequentially.

    A raised call records ``error``, empty warnings, and ``timed_out=False``. Later runs continue.
    `resolve` is injected for live calls and tests.
    """
    request = ResolveRequest.model_validate(build_request(case))
    records = []
    for _ in range(runs):
        start = time.monotonic()
        try:
            response = await resolve(request)
        except Exception:
            records.append(
                _run_record(
                    time.monotonic() - start,
                    status=_ERROR_STATUS,
                    warning_codes=[],
                    timed_out=False,
                )
            )
            continue
        diagnostics = response.diagnostics
        records.append(
            _run_record(
                time.monotonic() - start,
                status=_outcome_label(response),
                warning_codes=sorted({warning.code for warning in diagnostics.warnings}),
                timed_out=bool(diagnostics.timed_out),
            )
        )
    return summarize_runs(records)


def _write_artifact(path: Path, artifact: dict) -> None:
    """Write JSON with sorted keys for stable output."""
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


async def measure(
    cases: list[Case],
    *,
    runs: int,
    resolve: ResolveCallable,
    out_path: Path,
    deadline_seconds: float,
    engine_build: str,
) -> dict:
    """Measure cases sequentially and save after each one for resumable runs.

    `resolve` and `engine_build` are injected for offline tests.
    """
    artifact = {
        "measurement": _MEASUREMENT,
        "runs_per_case": runs,
        "engine_build": engine_build,
        "deadline_seconds": deadline_seconds,
        "cases": {},
    }

    total = len(cases)
    for i, case in enumerate(cases, start=1):
        case_id = case.id
        print(f"[{i}/{total}] probing {case_id!r} ({runs} runs)...", flush=True)
        artifact["cases"][case_id] = await probe_case(case, runs=runs, resolve=resolve)
        _write_artifact(out_path, artifact)

    return artifact


def render_report(artifact: dict) -> str:
    """Render per-case time ranges and statuses, flagging deadline overruns and status flips."""
    deadline = artifact["deadline_seconds"]
    cases: dict[str, dict] = artifact["cases"]
    lines = [
        "",
        "=== corpus latency ===",
        f"cases: {len(cases)}, runs each: {artifact['runs_per_case']}, deadline: {deadline:.1f}s",
        f"{'id':<44} {'min':>8} {'max':>8}  statuses",
    ]
    over_deadline: list[tuple[str, float]] = []
    flipped: list[tuple[str, list[str]]] = []
    for case_id in sorted(cases):
        row = cases[case_id]
        statuses = sorted(set(row["statuses"]))
        lines.append(
            f"{case_id:<44} {row['min_seconds']:>8.2f} {row['max_seconds']:>8.2f}  "
            f"{', '.join(statuses)}"
        )
        if row["max_seconds"] > deadline:
            over_deadline.append((case_id, row["max_seconds"]))
        if row["status_flip"]:
            flipped.append((case_id, row["statuses"]))

    lines.append("")
    if over_deadline:
        lines.append(f"over the {deadline:.1f}s deadline (max wall-clock):")
        lines.extend(
            f"  {case_id}: max {max_seconds:.2f}s exceeds {deadline:.1f}s"
            for case_id, max_seconds in over_deadline
        )
    else:
        lines.append(f"no case exceeded the {deadline:.1f}s deadline")

    lines.append("")
    if flipped:
        lines.append("status differs between runs:")
        lines.extend(f"  {case_id}: {' -> '.join(statuses)}" for case_id, statuses in flipped)
    else:
        lines.append("no case changed status between runs")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=int, default=5, help="Sequential runs per case (default: 5)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output artifact path (default: baselines/eval-latency.json).",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "Comma-separated case ids to measure, replacing the default selection "
            "(every case whose why mentions a deadline or seconds)."
        ),
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=None,
        help=(
            "Service request deadline the timings are compared against (default: the "
            "engine's configured SEARCH_REQUEST_TIMEOUT_S)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected case ids and exit without building any client.",
    )
    args = parser.parse_args()

    selected = select_cases(load_cases(), args.ids)
    if not selected:
        sys.exit(
            "no cases selected -- the default selection is every case whose why "
            "mentions a deadline or seconds; pass --ids to name cases directly"
        )

    if args.dry_run:
        print(f"{len(selected)} case(s) selected:")
        for case in selected:
            print(f"  {case.id}")
        return

    # Delay engine imports until after --dry-run. They bind model settings from the environment.
    from search.engine.config import settings
    from search.engine.core import resolve_async
    from search.eval.live_clients import live_engine_clients, skills_index_snapshot

    deadline_seconds = (
        args.deadline_seconds if args.deadline_seconds is not None else settings().request_timeout_s
    )

    shared_resolvekit = _acquire_latency_resolvekit()
    try:
        print(f"Probing {len(selected)} case(s), {args.runs} runs each -> {args.out}")
        # The eval request asks for skills, so time the served path with the index in place.
        # live_engine_clients binds the run's catalogue around the build and every request.
        with live_engine_clients() as (graph, _sdmx):
            skills_index = skills_index_snapshot(graph)
            artifact = asyncio.run(
                measure(
                    selected,
                    runs=args.runs,
                    resolve=functools.partial(resolve_async, shared_skills_index=skills_index),
                    out_path=args.out,
                    deadline_seconds=deadline_seconds,
                    engine_build=settings().engine_build,
                )
            )
        print(render_report(artifact))
        print("Done.")
    finally:
        release_shared(shared_resolvekit)


if __name__ == "__main__":
    main()
