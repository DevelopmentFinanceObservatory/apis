#!/usr/bin/env python3
"""CLI for capturing production calls, running prompt rungs, and reporting scores.

`capture` records each case's extract and bind calls once. `run` replays a rung as a
bind, extract, or end-to-end cell. The CLI runs one cell per process because end-to-end
cells patch process-wide `bind_top_k`. Each cell controls its own resolve concurrency.
Run a multi-rung sweep as repeated CLI invocations.

`report` compares cells by metric and can rescore stored calls offline. `ablate` writes
leave-one-out rungs. `build` creates an upward rung sequence from measured rankings.

Usage:
    uv run --env-file .env python scripts/ladder.py capture --id k8-2026-08-31
    uv run --env-file .env python scripts/ladder.py run production --mode e2e
    uv run python scripts/ladder.py run production --mode bind --dry-run
    uv run python scripts/ladder.py ablate promptlab/rungs/production-{extract,bind}.json \\
        --capture k8-2026-08-31
    uv run python scripts/ladder.py report abl-bind-also-plausible-bind-... \\
        --against ceiling-e2e-...
    uv run python scripts/ladder.py build promptlab/rungs/production-{extract,bind}.json \\
        --ranking runs/ladder/report/ablation.json --capture k8-2026-08-31
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from promptlab import report, rungs_gen  # noqa: E402
from promptlab.bindmenu import BindMenuError, truncate_menu  # noqa: E402
from promptlab.cells import run_bind_cell, run_capture, run_extract_cell  # noqa: E402
from promptlab.compose import compose  # noqa: E402
from promptlab.e2e_cell import run_e2e_cell  # noqa: E402
from promptlab.graphcache import GraphCache  # noqa: E402
from promptlab.manifest import PROMPTLAB_ROOT, ManifestError, Rung, load_rung  # noqa: E402
from promptlab.store import (  # noqa: E402
    CaptureManifest,
    CaptureStore,
    CellManifest,
    CellStore,
    StoreError,
)
from search.engine import config  # noqa: E402
from search.engine.bind import _UNIVERSAL_BIND_SYSTEM_PROMPT  # noqa: E402
from search.engine.clients.llm import LLM  # noqa: E402
from search.engine.config import settings  # noqa: E402
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT  # noqa: E402
from search.eval.cases import Case, expectation_hash, load_case_files  # noqa: E402
from search.eval.live_clients import live_engine_clients  # noqa: E402

# pipeline reads bind_top_k from process settings. run_e2e_cell patches it.
# run_capture leaves process settings unchanged.


class _CliError(Exception):
    """An error `main` prints before returning exit code 1."""


def _resolve_runs_root(args: argparse.Namespace) -> Path:
    return args.runs_root if args.runs_root is not None else (_REPO_ROOT / "runs" / "ladder")


def _resolve_rung_paths(name: str, *, runs_root: Path) -> tuple[Path, Path]:
    """Find both rung files under `promptlab/rungs/`, then `<runs_root>/rungs/`."""
    for root in (PROMPTLAB_ROOT / "rungs", runs_root / "rungs"):
        extract_path = root / f"{name}-extract.json"
        bind_path = root / f"{name}-bind.json"
        if extract_path.is_file() and bind_path.is_file():
            return extract_path, bind_path
    raise _CliError(
        f"no rung named {name!r} under {PROMPTLAB_ROOT / 'rungs'} or {runs_root / 'rungs'} "
        f"(expected {name}-extract.json and {name}-bind.json)"
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _case_files() -> dict[str, list[Case]]:
    """The corpus cases by file stem, from `SEARCH_CORPUS_DIR` when set."""
    env_dir = os.environ.get("SEARCH_CORPUS_DIR")
    return load_case_files(Path(env_dir)) if env_dir else load_case_files()


def _corpus_sha256() -> str:
    """Hash every case's id and expectation, so editing a case's tags or `why` keeps captures valid."""
    lines = sorted(
        f"{case.id} {expectation_hash(case)}" for cases in _case_files().values() for case in cases
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _structured_door_ids(cases: Sequence[Case]) -> list[str]:
    return [c.id for c in cases if c.door != "raw_text"]


def _all_cases() -> list[Case]:
    return [case for cases in _case_files().values() for case in cases]


def _print_structured_door_exclusion() -> None:
    """Print excluded ids. Other doors bypass extract/bind and lack a rung prompt."""
    excluded = _structured_door_ids(_all_cases())
    if excluded:
        print(
            f"excluding {len(excluded)} non-raw_text case(s) -- no ladder-composed "
            f"prompt to replay: {', '.join(excluded)}"
        )


def _load_scoring_cases(
    *, corpus_file: str | None = None, ids: Sequence[str] | None = None
) -> list[Case]:
    """Load replayable cases, optionally filtered by case file stem and ids."""
    case_files = _case_files()
    if corpus_file is not None and corpus_file not in case_files:
        raise _CliError(f"no corpus case file {corpus_file!r}; choose from {sorted(case_files)}")
    pool = case_files[corpus_file] if corpus_file is not None else _all_cases()
    cases = [c for c in pool if c.door == "raw_text"]
    if ids is not None:
        id_set = set(ids)
        cases = [c for c in cases if c.id in id_set]
    return cases


def _capture_conflicts(rung: Rung, capture: CaptureStore) -> str | None:
    """Return a conflict message when the rung model differs from the capture model.

    `[[TODAY]]` uses the capture's recorded date, so the model is its only conflicting rung field.
    """
    if rung.run.model != capture.manifest.model:
        return (
            f"rung {rung.name!r} declares model {rung.run.model!r} but capture "
            f"{capture.manifest.capture_id!r} was taken under model "
            f"{capture.manifest.model!r}"
        )
    return None


def _require_corpus_match(capture: CaptureStore) -> None:
    """Raise `_CliError` if the corpus hash differs from the capture manifest.

    Replay and offline rescore both compare captured calls with current cases, so this check
    prevents scoring old calls against changed expectations.
    """
    current = _corpus_sha256()
    if current != capture.manifest.corpus_sha256:
        raise _CliError(
            f"capture {capture.manifest.capture_id!r} was taken over a corpus hashing to "
            f"{capture.manifest.corpus_sha256!r}, but the corpus on disk now hashes to "
            f"{current!r} -- the case corpus changed since this capture was taken; retake "
            "the capture or restore the corpus it was captured against"
        )


def _open_complete_capture(capture_root: Path) -> CaptureStore:
    try:
        capture = CaptureStore.open(capture_root)
        capture.require_complete()
    except StoreError as exc:
        raise _CliError(str(exc)) from exc
    _require_corpus_match(capture)
    return capture


def _print_rung(rung: Rung, extract_system: str, bind_system: str) -> None:
    print(f"rung {rung.name!r}: {rung.description}")
    print(f"  model={rung.run.model} top_k={rung.run.top_k} seeds={list(rung.run.seeds)}")
    print(f"--- extract system ({len(extract_system)} chars) ---")
    print(extract_system)
    print(f"--- bind system ({len(bind_system)} chars) ---")
    print(bind_system)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _check_bind_menus(rung: Rung, capture: CaptureStore) -> None:
    """Validate each captured bind menu at `top_k` before a live run spends API calls.

    A `BindMenuError` becomes `_CliError` with the offending record id.
    """
    records = capture.bind_records()
    for record in records:
        try:
            truncate_menu(record.prompt, rung.run.top_k)
        except BindMenuError as exc:
            raise _CliError(f"bind record {record.case_id!r} leg {record.leg}: {exc}") from exc
    print(f"checked {len(records)} bind record(s) against top_k={rung.run.top_k}: all parse")


def cmd_capture(args: argparse.Namespace) -> int:
    runs_root = _resolve_runs_root(args)
    capture_root = runs_root / "capture" / args.id

    _print_structured_door_exclusion()
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    cases = _load_scoring_cases(corpus_file=args.file, ids=ids)
    file_note = f" (file={args.file})" if args.file else ""
    print(f"capturing {len(cases)} case(s){file_note}")

    manifest_path = capture_root / "manifest.json"
    try:
        if manifest_path.is_file():
            if not args.resume:
                raise _CliError(
                    f"capture {args.id!r} already exists at {capture_root} -- pass --resume to "
                    "continue it"
                )
            store = CaptureStore.open(capture_root)
            if store.manifest.model != args.model:
                print(
                    f"warning: --model {args.model!r} ignored; resuming under the capture's own "
                    f"recorded model {store.manifest.model!r}"
                )
            if store.manifest.top_k != args.k:
                print(
                    f"warning: --k {args.k} ignored; resuming under the capture's own recorded "
                    f"top_k {store.manifest.top_k}"
                )
            model = store.manifest.model
            top_k = store.manifest.top_k
        else:
            if args.resume:
                print(f"note: no existing capture at {capture_root}; starting a fresh one")
            model = args.model
            top_k = args.k
            manifest = CaptureManifest(
                capture_id=args.id,
                git_sha=_git_sha(),
                engine_build_id=settings().engine_build,
                model=model,
                top_k=top_k,
                substituted_date=date.today().isoformat(),
                graph_base=settings().graph_base,
                corpus_sha256=_corpus_sha256(),
                seeds=(),
                extract_system_sha1=_sha1(_EXTRACTION_SYSTEM_PROMPT),
                bind_system_sha1=_sha1(_UNIVERSAL_BIND_SYSTEM_PROMPT),
            )
            store = CaptureStore.create(capture_root, manifest=manifest)
    except StoreError as exc:
        raise _CliError(str(exc)) from exc

    with store:
        llm = LLM(model=model)
        with live_engine_clients() as (live_graph, sdmx):
            graph = GraphCache(live_graph, path=store.graph_db_path)
            try:
                with patch.object(
                    config, "_SETTINGS", config.settings().model_copy(update={"bind_top_k": top_k})
                ):
                    asyncio.run(
                        run_capture(
                            cases,
                            out=store,
                            llm=llm,
                            graph=graph,
                            sdmx=sdmx,
                            concurrency=args.concurrency_llm,
                        )
                    )
            finally:
                graph.close()

        store.finish()
        print(
            f"capture {args.id!r} complete: {len(store.extract_records())} case(s), "
            f"{len(store.bind_records())} bind call(s)"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    runs_root = _resolve_runs_root(args)
    extract_path, bind_path = _resolve_rung_paths(args.rung, runs_root=runs_root)
    rung = load_rung(extract_path, bind_path)
    if args.seeds:
        try:
            seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
        except ValueError as exc:
            raise _CliError(
                f"--seeds {args.seeds!r} is not a comma-separated list of integers"
            ) from exc
        rung = dataclasses.replace(rung, run=dataclasses.replace(rung.run, seeds=seeds))

    extract_system = compose(rung.extract)
    bind_system = compose(rung.bind)
    _print_rung(rung, extract_system, bind_system)
    _print_structured_door_exclusion()

    capture_id = args.capture or rung.run.capture
    capture = _open_complete_capture(runs_root / "capture" / capture_id)
    with capture:
        conflict = _capture_conflicts(rung, capture)
        if conflict:
            raise _CliError(conflict)
        if args.mode == "bind" and rung.run.top_k > capture.manifest.top_k:
            raise _CliError(
                f"rung {rung.name!r} asks for top_k={rung.run.top_k} but capture {capture_id!r} "
                f"was only taken at top_k={capture.manifest.top_k}"
            )
        print(
            f"capture {capture_id!r}: complete, model={capture.manifest.model}, "
            f"{len(capture.extract_records())} case(s), dated {capture.manifest.substituted_date}"
        )

        if args.dry_run:
            if args.mode == "bind":
                _check_bind_menus(rung, capture)
            return 0

        cell_id = args.cell_id or f"{rung.name}-{args.mode}-{_timestamp()}"
        cell_root = runs_root / "cells" / cell_id
        manifest_path = cell_root / "manifest.json"
        try:
            if manifest_path.is_file():
                if not args.resume:
                    raise _CliError(
                        f"cell {cell_id!r} already exists at {cell_root} -- pass --resume to "
                        "continue it"
                    )
                cell_store = CellStore.open(cell_root)
            else:
                cell_manifest = CellManifest(
                    cell_id=cell_id,
                    rung=dataclasses.asdict(rung),
                    capture_id=capture_id,
                    git_sha=_git_sha(),
                    model=rung.run.model,
                    seeds=rung.run.seeds,
                    extract_system=extract_system,
                    bind_system=bind_system,
                )
                cell_store = CellStore.create(cell_root, manifest=cell_manifest)
        except StoreError as exc:
            raise _CliError(str(exc)) from exc

        with cell_store:
            # for_stage applies stage defaults with an unset thinking level.
            # Bind uses "minimal" when SEARCH_LLM_THINKING_LEVEL is unset.
            llm = LLM(model=rung.run.model)

            if args.mode == "bind":
                cases = _load_scoring_cases()
                asyncio.run(
                    run_bind_cell(
                        rung,
                        cases,
                        capture=capture,
                        out=cell_store,
                        llm=llm,
                        concurrency=args.concurrency,
                    )
                )
            elif args.mode == "extract":
                cases = _load_scoring_cases()
                with live_engine_clients() as (live_graph, _sdmx):
                    graph = GraphCache(live_graph, path=capture.graph_db_path)
                    try:
                        asyncio.run(
                            run_extract_cell(
                                rung,
                                cases,
                                capture=capture,
                                graph=graph,
                                out=cell_store,
                                llm=llm,
                                concurrency=args.concurrency,
                            )
                        )
                    finally:
                        graph.close()
            else:
                with live_engine_clients() as (live_graph, sdmx):
                    graph = GraphCache(live_graph, path=capture.graph_db_path)
                    try:
                        asyncio.run(
                            run_e2e_cell(
                                rung,
                                capture=capture,
                                graph=graph,
                                out=cell_store,
                                llm=llm,
                                sdmx=sdmx,
                                concurrency=args.concurrency,
                                cases=_load_scoring_cases(),
                            )
                        )
                    finally:
                        graph.close()

            print(f"cell {cell_id!r} complete: {cell_store.status()}")
        return 0


def _require_rescorable(cell_id: str, store: CellStore) -> None:
    """Reject incomplete cells before rescore marks their status complete.

    Applies to reported cells and the `--against` ceiling.
    """
    if not store.is_complete():
        raise _CliError(
            f"cannot --rescore cell {cell_id!r}: it is not complete (drop --rescore to view "
            "interim scores read-only, or finish the run first)"
        )


def _rescored_records(store: CellStore, *, runs_root: Path) -> list[dict]:
    """Rescore stored calls with a read-only graph cache and the live case corpus."""
    cases_by_id = {c.id: c for c in _all_cases()}
    with CaptureStore.open(runs_root / "capture" / store.manifest.capture_id) as capture:
        _require_corpus_match(capture)
        graph = report.open_read_only_graph_cache(capture.graph_db_path)
        try:
            return report.rescore_cell(store, graph=graph, cases_by_id=cases_by_id)
        finally:
            graph.close()


def cmd_report(args: argparse.Namespace) -> int:
    runs_root = _resolve_runs_root(args)
    cells_root = runs_root / "cells"

    with ExitStack() as stack:
        try:
            cells = [
                (
                    cell_id,
                    stack.enter_context(
                        report.open_cell(cells_root, cell_id, allow_partial=args.allow_partial)
                    ),
                )
                for cell_id in args.cell_ids
            ]
            if args.rescore:
                for cell_id, store in cells:
                    _require_rescorable(cell_id, store)

            against_table = None
            if args.against:
                ceiling_store = stack.enter_context(
                    report.open_cell(cells_root, args.against, allow_partial=args.allow_partial)
                )
                if args.rescore:
                    _require_rescorable(args.against, ceiling_store)
                report.require_minimal_ceiling(ceiling_store)
                ceiling_records = (
                    _rescored_records(ceiling_store, runs_root=runs_root)
                    if args.rescore
                    else ceiling_store.call_records()
                )
                against_table = report.build_metric_table_from_records(
                    cell_id=ceiling_store.manifest.cell_id,
                    rung_name=ceiling_store.manifest.rung.get(
                        "name", ceiling_store.manifest.cell_id
                    ),
                    records=ceiling_records,
                )

            tables = []
            for cell_id, store in cells:
                records = (
                    _rescored_records(store, runs_root=runs_root)
                    if args.rescore
                    else store.call_records()
                )
                tables.append(
                    report.build_metric_table_from_records(
                        cell_id=cell_id,
                        rung_name=store.manifest.rung.get("name", cell_id),
                        records=records,
                    )
                )
        except (StoreError, report.ReportError) as exc:
            raise _CliError(str(exc)) from exc

        if args.format == "json":
            print(json.dumps(report.render_json(tables, against=against_table), indent=2))
        else:
            print(report.render_markdown(tables, against=against_table), end="")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    runs_root = _resolve_runs_root(args)
    out_dir = args.out_dir or (runs_root / "rungs")
    try:
        base = load_rung(args.base_extract, args.base_bind)
        generated = rungs_gen.ablate(base, granularity=args.granularity)
        run = dataclasses.replace(base.run, capture=args.capture)
        written = rungs_gen.write_rungs(generated, out_dir=out_dir, run=run)
    except (ManifestError, rungs_gen.AblationError) as exc:
        raise _CliError(str(exc)) from exc

    print(f"wrote {len(written)} ablation rung(s) to {out_dir}")
    for g in generated:
        print(f"  {g.name}")
    return 0


def _merged_ranking(ranking_paths: Sequence[Path]) -> dict[str, float]:
    """Merge report rankings and raise `_CliError` for duplicate unit ids.

    Bind and extract families use separate ceiling rankings.
    """
    merged: dict[str, float] = {}
    for path in ranking_paths:
        ranking_data = json.loads(path.read_text())
        for unit, score in rungs_gen.ranking_from_report_json(ranking_data).items():
            if unit in merged:
                raise _CliError(
                    f"unit {unit!r} appears in more than one --ranking file (duplicated in {path})"
                )
            merged[unit] = score
    return merged


def cmd_build(args: argparse.Namespace) -> int:
    runs_root = _resolve_runs_root(args)
    out_dir = args.out_dir or (runs_root / "rungs")
    try:
        base = load_rung(args.base_extract, args.base_bind)
        rung0_extract, rung0_bind = _resolve_rung_paths("rung0", runs_root=runs_root)
        rung0 = load_rung(rung0_extract, rung0_bind)
        ranking = _merged_ranking(args.ranking)
        generated = rungs_gen.build_upward(rung0, base, ranking)
        run = dataclasses.replace(base.run, capture=args.capture)
        written = rungs_gen.write_rungs(generated, out_dir=out_dir, run=run)
    except (ManifestError, rungs_gen.AblationError, OSError, json.JSONDecodeError) as exc:
        raise _CliError(str(exc)) from exc

    print(f"wrote {len(written)} build-upward rung(s) to {out_dir}")
    for g in generated:
        print(f"  {g.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_cmd = subparsers.add_parser("capture", help="Run one live capture over the corpus.")
    capture_cmd.add_argument(
        "--id", required=True, help="Capture id (the directory name under capture/)."
    )
    capture_cmd.add_argument(
        "--k",
        type=int,
        default=settings().bind_top_k,
        help="Bind top-k to capture at (default: %(default)s, the production default).",
    )
    capture_cmd.add_argument(
        "--file",
        default=None,
        help="Restrict to the cases in this corpus file, by stem (for example standard or sdg).",
    )
    capture_cmd.add_argument("--ids", default=None, help="Comma-separated case ids to restrict to.")
    capture_cmd.add_argument(
        "--model",
        default="gemini-3.5-flash-lite",
        help="LLM model to capture under (default: %(default)s).",
    )
    capture_cmd.add_argument(
        "--concurrency-llm",
        type=int,
        default=16,
        dest="concurrency_llm",
        help="Max concurrent resolve_async calls (default: %(default)s).",
    )
    capture_cmd.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing, incomplete capture instead of refusing to overwrite it.",
    )
    capture_cmd.add_argument("--runs-root", type=Path, default=None, help=argparse.SUPPRESS)
    capture_cmd.set_defaults(func=cmd_capture)

    run_cmd = subparsers.add_parser(
        "run", help="Run a rung as one cell (bind, extract, or e2e) against a capture."
    )
    run_cmd.add_argument(
        "rung", help="Rung name (looked up under promptlab/rungs/, then runs/ladder/rungs/)."
    )
    run_cmd.add_argument(
        "--mode",
        choices=("bind", "extract", "e2e"),
        default="e2e",
        help="Which cell to run (default: %(default)s).",
    )
    run_cmd.add_argument(
        "--capture",
        default=None,
        help="Capture id to replay against (default: the rung's own run.capture).",
    )
    run_cmd.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed override (default: the rung's own run.seeds).",
    )
    run_cmd.add_argument(
        "--cell-id",
        default=None,
        dest="cell_id",
        help="Cell id (default: <rung>-<mode>-<timestamp>).",
    )
    run_cmd.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent live calls (default: %(default)s).",
    )
    run_cmd.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing, incomplete cell instead of refusing to overwrite it.",
    )
    run_cmd.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Compose the rung and check the capture; make no API calls.",
    )
    run_cmd.add_argument("--runs-root", type=Path, default=None, help=argparse.SUPPRESS)
    run_cmd.set_defaults(func=cmd_run)

    report_cmd = subparsers.add_parser(
        "report", help="Render one cell or several as a per-metric report."
    )
    report_cmd.add_argument(
        "cell_ids", nargs="+", metavar="CELL_ID", help="Cell id(s) under cells/ to report."
    )
    report_cmd.add_argument(
        "--against", default=None, help="Ceiling cell id to score every reported cell against."
    )
    report_cmd.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="Output format (default: %(default)s).",
    )
    report_cmd.add_argument(
        "--allow-partial",
        action="store_true",
        dest="allow_partial",
        help="Report on an incomplete cell instead of refusing it.",
    )
    report_cmd.add_argument(
        "--rescore",
        action="store_true",
        help=(
            "Re-derive each reported cell's scores from its own stored calls.jsonl, offline "
            "(zero LLM and zero live-graph traffic; see promptlab.report.rescore_cell). "
            "Persists the result to that cell's scores.json/status.json in place -- refuses "
            "an incomplete cell, or an incomplete --against ceiling, rather than mislabeling "
            "it complete."
        ),
    )
    report_cmd.add_argument("--runs-root", type=Path, default=None, help=argparse.SUPPRESS)
    report_cmd.set_defaults(func=cmd_report)

    ablate_cmd = subparsers.add_parser(
        "ablate", help="Write the leave-one-out ablation rung manifests."
    )
    ablate_cmd.add_argument("base_extract", type=Path, help="Base rung's <name>-extract.json.")
    ablate_cmd.add_argument("base_bind", type=Path, help="Base rung's <name>-bind.json.")
    ablate_cmd.add_argument(
        "--capture", required=True, help="Capture id the generated rungs will run against."
    )
    ablate_cmd.add_argument(
        "--granularity",
        choices=("unit", "block", "both"),
        default="both",
        help="Which ablation family to generate (default: %(default)s).",
    )
    ablate_cmd.add_argument(
        "--out-dir", type=Path, default=None, dest="out_dir", help=argparse.SUPPRESS
    )
    ablate_cmd.add_argument("--runs-root", type=Path, default=None, help=argparse.SUPPRESS)
    ablate_cmd.set_defaults(func=cmd_ablate)

    build_cmd = subparsers.add_parser(
        "build", help="Write the build-upward rung sequence from a measured ranking."
    )
    build_cmd.add_argument("base_extract", type=Path, help="Base rung's <name>-extract.json.")
    build_cmd.add_argument("base_bind", type=Path, help="Base rung's <name>-bind.json.")
    build_cmd.add_argument(
        "--ranking",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to a `ladder report --format json --against <ceiling>` artifact over the "
            "ablation cells (see promptlab.rungs_gen.ranking_from_report_json). Repeatable -- "
            "pass one per ceiling family (e.g. bind-family cells against a bind ceiling, "
            "extract-family cells against an extract ceiling); the rankings are merged, and a "
            "unit id appearing in more than one file is refused. The only input form this "
            "command reads, so a ranking is generated by `report`, never hand-edited."
        ),
    )
    build_cmd.add_argument(
        "--capture", required=True, help="Capture id the generated rungs will run against."
    )
    build_cmd.add_argument(
        "--out-dir", type=Path, default=None, dest="out_dir", help=argparse.SUPPRESS
    )
    build_cmd.add_argument("--runs-root", type=Path, default=None, help=argparse.SUPPRESS)
    build_cmd.set_defaults(func=cmd_build)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except _CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
