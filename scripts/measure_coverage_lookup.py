"""Measure `CoverageIndex` query latency and peak RSS against a published artifact.

Each mode runs in a spawned process and reports p50/p95/p99 wall time and peak RSS. Cold loads
use fresh-process repeats and report median and maximum RSS.

`cold_load` loads the full artifact. `named_bilateral` binds a real donor/recipient pair.
`named_single_side` binds a donor across ODA variables. `open_oda_family` leaves the
`CRSTypeOfFlow10` family open. `windowed` adds a representative variable's year range.
`miss` binds `_MISS_RECIPIENT` for an empty lookup. `pair_bound` groups donor-filtered pairs by
(variable, facet). `pair_open_oda` groups unbound pairs (1.64 million in the measured artifact).

`discover_open_oda` runs role discovery with both flow axes open and ``date_request=None``,
matching a bare "aid" query. `preview_open_oda` adds reads for the facet discovery elected: the
first `SEARCH_ROLE_BULK_SERIES_MAX` pairs and the observation count summed across each pair.
Both bind the origin catalogue at `--catalogue-uri`, which facet ranking reads. Each
measurement starts with fresh discovery.

`n_concurrent_threads` starts `--concurrency` threads at a barrier for the same bilateral
lookup. It reports burst and per-thread latency plus RSS. PyArrow compute kernels release the
GIL, so the calls overlap.

Other query modes load the immutable table once per process, then repeat reads in-process.
Peak RSS includes the artifact load and all queries.

Usage:
    uv run python scripts/measure_coverage_lookup.py
    uv run python scripts/measure_coverage_lookup.py --artifact-uri file:///tmp/coverage-live
    uv run python scripts/measure_coverage_lookup.py --catalogue-uri gs://<bucket>/origin-catalogue
    uv run python scripts/measure_coverage_lookup.py --out /tmp/coverage-lookup.json
    uv run python scripts/measure_coverage_lookup.py --repeats 1 --inner-repeats 5
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import resource
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPEATS = 5
"""Fresh-process repeats for cold loads."""

DEFAULT_INNER_REPEATS = 50
"""In-process query repeats after the artifact loads."""

DEFAULT_CONCURRENCY = 8

_MISS_RECIPIENT = "country/__measure_coverage_lookup_no_such_recipient__"
"""Reserved recipient dcid for a guaranteed-empty lookup as artifacts gain rows."""

_ODA_TOKEN = "CRSTypeOfFlow10"

_QUERY_MODES = (
    "named_bilateral",
    "named_single_side",
    "open_oda_family",
    "windowed",
    "miss",
    "pair_bound",
    "pair_open_oda",
    "discover_open_oda",
    "preview_open_oda",
)


def _max_rss_bytes() -> int:
    """Return process peak RSS in bytes. macOS reports bytes, and Linux reports KiB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(samples)

    def _at(p: float) -> float:
        index = max(0, min(len(ordered) - 1, round(p * (len(ordered) - 1))))
        return ordered[index]

    return {"p50": _at(0.50), "p95": _at(0.95), "p99": _at(0.99)}


# Discover real probe values in the parent, outside timed child processes.


def _discover_probe(artifact_uri: str) -> dict[str, Any]:
    from series_coverage.loader import load_coverage

    loaded = load_coverage(artifact_uri)
    index = loaded.index
    all_variables = list(index.role_names)
    if not all_variables:
        raise RuntimeError(f"artifact at {artifact_uri!r} carries no variables to probe")
    oda_variables = [v for v in all_variables if _ODA_TOKEN in v]
    if not oda_variables:
        print(
            f"no variable dcid contains {_ODA_TOKEN!r}; falling back to the first "
            f"{min(200, len(all_variables))} variables for the open-family/windowed modes",
            file=sys.stderr,
        )
        oda_variables = all_variables[: min(200, len(all_variables))]
    probe_variables = oda_variables[: min(len(oda_variables), 500)]
    rows = index.pair_rows(variables=probe_variables)
    if rows.num_rows == 0:
        rows = index.pair_rows(variables=all_variables)
    if rows.num_rows == 0:
        raise RuntimeError(f"artifact at {artifact_uri!r} carries no (donor, recipient) pairs")
    group = rows.groups()[0]
    bilateral_variable = group.variable
    donor, recipient, _n_obs = rows.pairs(variable=group.variable, facet_id=group.facet_id)[0]
    windowed = index.filter_candidates(variables=[bilateral_variable])
    first_year = windowed[0].first_year if windowed else None
    last_year = windowed[0].last_year if windowed else None
    return {
        "oda_variables": oda_variables,
        "bilateral_variable": bilateral_variable,
        "bilateral_donor": donor,
        "bilateral_recipient": recipient,
        "window_first_year": first_year,
        "window_last_year": last_year,
    }


# Import measured code in each child so its RSS excludes parent-process imports.


def _run_cold_load_in_child(artifact_uri: str, result_queue: multiprocessing.Queue) -> None:
    from series_coverage.loader import load_coverage

    started = time.perf_counter()
    load_coverage(artifact_uri)
    wall = time.perf_counter() - started
    result_queue.put({"wall_seconds": wall, "peak_rss_bytes": _max_rss_bytes()})


def _call_query_mode(loaded: Any, mode: str, probe: dict[str, Any]) -> None:
    index = loaded.index
    if mode == "named_bilateral":
        index.filter_candidates(
            variables=[probe["bilateral_variable"]],
            bound={
                "donor": frozenset({probe["bilateral_donor"]}),
                "recipient": frozenset({probe["bilateral_recipient"]}),
            },
        )
    elif mode == "named_single_side":
        index.filter_candidates(
            variables=probe["oda_variables"], bound={"donor": frozenset({probe["bilateral_donor"]})}
        )
    elif mode == "open_oda_family":
        index.filter_candidates(variables=probe["oda_variables"])
    elif mode == "windowed":
        index.filter_candidates(
            variables=probe["oda_variables"],
            bound={"donor": frozenset({probe["bilateral_donor"]})},
            first_year=probe["window_first_year"],
            last_year=probe["window_last_year"],
        )
    elif mode == "miss":
        index.filter_candidates(
            variables=[probe["bilateral_variable"]],
            bound={"recipient": frozenset({_MISS_RECIPIENT})},
        )
    elif mode == "pair_bound":
        index.pair_rows(
            variables=probe["oda_variables"], bound={"donor": frozenset({probe["bilateral_donor"]})}
        ).groups()
    elif mode == "pair_open_oda":
        index.pair_rows(variables=probe["oda_variables"]).groups()
    elif mode in ("discover_open_oda", "preview_open_oda"):
        from search.engine.retrieval.role_scope import discover_role_scope
        from series_coverage.shared import CoverageSnapshot

        result = discover_role_scope(
            surviving_svs=probe["oda_variables"],
            member_facts={},
            member_population_exact=True,
            open_flow_axis="destination",
            role_name="financingRecipient",
            opposite_role_name="financingProvider",
            opposite_dcids=(),
            facet_filters={},
            date_request=None,
            coverage_snapshot=CoverageSnapshot(
                index=index, manifest=loaded.manifest, stamp=loaded.manifest.stamp
            ),
        )
        if mode == "preview_open_oda":
            _preview(result)
    else:
        raise ValueError(f"unknown query mode {mode!r}")


def _preview(result: Any) -> None:
    """Plan the bulk preview's reads for the facet discovery elected: the configured leading
    pairs and the observations summed across each pair in the scope's time window."""
    from search.engine.region.observations.bulk_role_series import _plan_role_bulk_fetch

    plan = result.read_plan
    assert plan.elected is not None, "the open ODA scope reads above the pair cap"
    _plan_role_bulk_fetch(plan, plan.elected)


def _run_query_mode_in_child(
    artifact_uri: str,
    catalogue_uri: str,
    mode: str,
    probe: dict[str, Any],
    inner_repeats: int,
    result_queue: multiprocessing.Queue,
) -> None:
    from origin_catalogue import bind_catalogue, load_catalogue
    from series_coverage.loader import load_coverage

    loaded = load_coverage(artifact_uri)
    # Discovery ranks facets by DFO precedence, which reads the bound origin catalogue.
    bind_catalogue(load_catalogue(catalogue_uri))
    _call_query_mode(loaded, mode, probe)

    wall_samples = []
    for _ in range(inner_repeats):
        started = time.perf_counter()
        _call_query_mode(loaded, mode, probe)
        wall_samples.append(time.perf_counter() - started)
    result_queue.put({"wall_seconds_samples": wall_samples, "peak_rss_bytes": _max_rss_bytes()})


def _run_concurrency_in_child(
    artifact_uri: str, probe: dict[str, Any], concurrency: int, result_queue: multiprocessing.Queue
) -> None:
    import threading

    from series_coverage.loader import load_coverage

    loaded = load_coverage(artifact_uri)
    barrier = threading.Barrier(concurrency)
    latencies: list[float | None] = [None] * concurrency

    def _worker(slot: int) -> None:
        barrier.wait()
        started = time.perf_counter()
        _call_query_mode(loaded, "named_bilateral", probe)
        latencies[slot] = time.perf_counter() - started

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(concurrency)]
    overall_started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    overall_wall = time.perf_counter() - overall_started
    result_queue.put(
        {
            "overall_wall_seconds": overall_wall,
            "per_thread_wall_seconds_samples": [float(v) for v in latencies if v is not None],
            "peak_rss_bytes": _max_rss_bytes(),
            "concurrency": concurrency,
        }
    )


def _spawn(target, args: tuple) -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(target=target, args=(*args, result_queue))
    process.start()
    result = result_queue.get()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"child process for {target.__name__} exited {process.exitcode}")
    return result


def _report_repeats(
    label: str, samples: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    peaks_mb = sorted(s["peak_rss_bytes"] / (1024 * 1024) for s in samples)
    walls = sorted(s["wall_seconds"] for s in samples)
    percentiles = _percentiles(walls)
    median_mb = peaks_mb[len(peaks_mb) // 2]
    max_mb = peaks_mb[-1]
    print(
        f"{label:<26}  p50={percentiles['p50'] * 1000:>8.2f}ms  p95={percentiles['p95'] * 1000:>8.2f}ms  "
        f"p99={percentiles['p99'] * 1000:>8.2f}ms  median_rss={median_mb:>8.1f}MB  max_rss={max_mb:>8.1f}MB"
    )
    results.append(
        {
            "mode": label,
            "repeats": len(samples),
            "wall_seconds_percentiles": percentiles,
            "median_peak_rss_bytes": int(median_mb * 1024 * 1024),
            "max_peak_rss_bytes": int(max_mb * 1024 * 1024),
        }
    )


def _report_query_mode(label: str, result: dict[str, Any], results: list[dict[str, Any]]) -> None:
    percentiles = _percentiles(result["wall_seconds_samples"])
    peak_mb = result["peak_rss_bytes"] / (1024 * 1024)
    print(
        f"{label:<26}  p50={percentiles['p50'] * 1000:>8.2f}ms  p95={percentiles['p95'] * 1000:>8.2f}ms  "
        f"p99={percentiles['p99'] * 1000:>8.2f}ms  peak_rss={peak_mb:>8.1f}MB  "
        f"(n={len(result['wall_seconds_samples'])})"
    )
    results.append(
        {
            "mode": label,
            "wall_seconds_percentiles": percentiles,
            "peak_rss_bytes": result["peak_rss_bytes"],
            "inner_repeats": len(result["wall_seconds_samples"]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--artifact-uri",
        type=str,
        default=None,
        help=(
            "Base URI of a published coverage artifact (gs:// or file://). Defaults to this "
            "repo's tests/fixtures/coverage_slice."
        ),
    )
    parser.add_argument(
        "--catalogue-uri",
        type=str,
        default=f"file://{_REPO_ROOT / 'tests' / 'fixtures' / 'origin_catalogue_eval'}",
        help=(
            "Base URI of a published origin catalogue (gs:// or file://). Defaults to this "
            "repo's tests/fixtures/origin_catalogue_eval."
        ),
    )
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Fresh-process repeats for cold_load (default {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=DEFAULT_INNER_REPEATS,
        help=f"In-process repeats per query mode, once loaded (default {DEFAULT_INNER_REPEATS}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Threads for the n_concurrent_threads mode (default {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument("--skip-cold-load", action="store_true")
    parser.add_argument("--skip-concurrency", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=_QUERY_MODES,
        help="Query modes to run, overriding the default of all of them.",
    )
    args = parser.parse_args()

    artifact_uri = args.artifact_uri
    if artifact_uri is None:
        default_fixture = _REPO_ROOT / "tests" / "fixtures" / "coverage_slice"
        artifact_uri = f"file://{default_fixture}"

    print(f"artifact: {artifact_uri}")
    probe = _discover_probe(artifact_uri)
    print(
        f"probe: {len(probe['oda_variables'])} ODA-family variable(s), bilateral pair "
        f"({probe['bilateral_donor']}, {probe['bilateral_recipient']}), window "
        f"[{probe['window_first_year']}, {probe['window_last_year']}]"
    )

    results: list[dict[str, Any]] = []

    if not args.skip_cold_load:
        samples = [_spawn(_run_cold_load_in_child, (artifact_uri,)) for _ in range(args.repeats)]
        _report_repeats("cold_load", samples, results)

    for mode in args.modes or _QUERY_MODES:
        result = _spawn(
            _run_query_mode_in_child,
            (artifact_uri, args.catalogue_uri, mode, probe, args.inner_repeats),
        )
        _report_query_mode(mode, result, results)

    if not args.skip_concurrency:
        result = _spawn(_run_concurrency_in_child, (artifact_uri, probe, args.concurrency))
        percentiles = _percentiles(result["per_thread_wall_seconds_samples"])
        peak_mb = result["peak_rss_bytes"] / (1024 * 1024)
        print(
            f"{'n_concurrent_threads':<26}  overall={result['overall_wall_seconds'] * 1000:>8.2f}ms  "
            f"per-thread p50={percentiles['p50'] * 1000:>8.2f}ms  p99={percentiles['p99'] * 1000:>8.2f}ms  "
            f"peak_rss={peak_mb:>8.1f}MB  (concurrency={result['concurrency']})"
        )
        results.append(
            {
                "mode": "n_concurrent_threads",
                "concurrency": result["concurrency"],
                "overall_wall_seconds": result["overall_wall_seconds"],
                "per_thread_wall_seconds_percentiles": percentiles,
                "peak_rss_bytes": result["peak_rss_bytes"],
            }
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
