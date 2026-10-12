"""Bench the Datasets API's mint, retrieval and snapshot requests against a deployed base URL.

Two shapes, both definitions the deployed Search service returned and
`tests/render/heading_words_specs.json` records:

- q1, a walked family read subject-only: `weo-019-0`, the nominal economic-activity amounts of
  seven countries. The code index holds no `EconomicActivity` family, so its members are
  enumerated by a graph walk.
- q3, an index family read role-bearing: `crs-033`, CRS disbursements from Japan to Viet Nam.
  The code index decides its members, and both roles are bound.

Each shape runs these requests, in order:

- `cold_mint`: a mint of a definition this deployment holds no head for. Each sample narrows the
  shape's window to a range picked from the run's seed (`requested_window`), which gives a new
  `spec_id` and so no head. A sample that a head served anyway is counted in
  `served_from_head`.
- `warm_mint`: a repeat mint of the shape's own definition, after one priming mint.
- `cold_get`: a definition GET of another window variant, registered first, which rebuilds with
  `rebuild_reason=no_head`. A GET past the head's recheck window needs a 15-minute wait, so the
  bench does not measure one.
- `warm_get`: a definition GET the head answers inside its recheck window.
- `snapshot`: a GET of the version the priming mint returned.
- `concurrent_warm_get`: 20 warm GETs at once.
- `concurrent_cold_mint_3`, then `concurrent_cold_mint_6`: 3, then 6, cold mints at once.

A coverage reload during two mints needs a way to trigger a reload on the target, and the API
exposes none, so it is recorded as skipped.

Each request records its wall-clock latency, status, `trace.served_from` and the counts the
response's `Server-Timing` header carries as `name;desc=n` (`sdmx_requests`,
`graph_requests`, `db_transactions`, `sign_blob_calls`, `permit_wait_ms`), with the
header's `total`. Each request kind reports p50 and p95 latency and, per count, its p50, p95
and maximum. The JSON report also records the base URL, the seed, the sample count and, per
shape, the deployed build (`x-dfo-commit`).

Usage:
    uv run --extra datasets python scripts/bench_datasets.py --out bench.json
    uv run --extra datasets python scripts/bench_datasets.py \\
        --base-url https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app --samples 5 --seed 7
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx

from spec_registry.identity import recompute_identity

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RECORDED_DEFINITIONS = _REPO_ROOT / "tests" / "render" / "heading_words_specs.json"
DEFAULT_BASE_URL = "https://dfo-dev-apis-zms5lj4zhq-nn.a.run.app"

# Shape name, recorded definition key, and what the shape exercises.
SHAPES: tuple[tuple[str, str, str], ...] = (
    ("q1", "weo-019-0", "walked family, subject-only"),
    ("q3", "crs-033", "index family, role-bearing"),
)
_OPTIONS = {"format": "csv", "bundle": True}
_QUERY = {"format": "csv", "bundle": "true"}
# A mint or rebuild may run to the service's 120-second deadline.
_TIMEOUT_SECONDS = 150.0
_CONCURRENT_WARM_GETS = 20
_CONCURRENT_COLD_MINTS = (3, 6)
# Window starts and ends a cold variant picks from, so its range overlaps the recorded data.
_VARIANT_START_YEARS = range(1995, 2016)
_VARIANT_END_YEARS = range(2016, 2024)
_SKIPPED = {
    "coverage_reload_during_two_mints": (
        "the API exposes no way to trigger a coverage reload, so the bench cannot time one"
    )
}


@dataclass(frozen=True)
class Shape:
    """A definition the bench mints and reads, with the window variants its cold requests use."""

    name: str
    description: str
    definition: Mapping[str, object]
    variants: Callable[[], Mapping[str, object]]


@dataclass
class Sample:
    status: int
    latency_ms: float
    served_from: str | None
    server_total_ms: float | None
    counts: dict[str, int] = field(default_factory=dict)


def server_timing(header: str) -> tuple[float | None, dict[str, int]]:
    """The `total` duration and the counts (`name;desc=n`) a `Server-Timing` value carries."""
    total: float | None = None
    counts: dict[str, int] = {}
    for metric in filter(None, (part.strip() for part in header.split(","))):
        name, _, params = metric.partition(";")
        if params.startswith("dur=") and name == "total":
            total = float(params.removeprefix("dur="))
        elif params.startswith("desc="):
            counts[name] = int(params.removeprefix("desc="))
    return total, counts


def _sample(send: Callable[[], httpx.Response]) -> Sample:
    started = time.perf_counter()
    response = send()
    latency_ms = (time.perf_counter() - started) * 1000
    total, counts = server_timing(response.headers.get("server-timing", ""))
    served_from = None
    if response.status_code == 200:
        trace = response.json().get("trace")
        served_from = trace.get("served_from") if isinstance(trace, dict) else None
    return Sample(
        status=response.status_code,
        latency_ms=latency_ms,
        served_from=served_from,
        server_total_ms=total,
        counts=counts,
    )


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """The nearest-rank percentile of `values`, `None` when empty."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarise(samples: Sequence[Sample]) -> dict[str, object]:
    """Latency and count percentiles over `samples`, with each sample kept."""
    names = sorted({name for sample in samples for name in sample.counts})
    return {
        "n": len(samples),
        "statuses": sorted({sample.status for sample in samples}),
        "served_from_head": sum(sample.served_from == "head" for sample in samples),
        "latency_ms": {
            "p50": _percentile([s.latency_ms for s in samples], 0.5),
            "p95": _percentile([s.latency_ms for s in samples], 0.95),
        },
        "counts": {
            name: {
                "p50": _percentile(values, 0.5),
                "p95": _percentile(values, 0.95),
                "max": max(values),
            }
            for name in names
            if (values := [s.counts[name] for s in samples if name in s.counts])
        },
        "samples": [sample.__dict__ for sample in samples],
    }


def _mint_body(definition: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "input": {"kind": "spec_id", "spec_id": definition["spec_id"], "definition": definition},
        "options": _OPTIONS,
    }


def window_variants(
    definition: Mapping[str, object], *, rng: random.Random
) -> Callable[[], Mapping[str, object]]:
    """A source of distinct definitions, each `definition` with its window narrowed to a range
    `rng` picks, so each has its own `spec_id`."""
    used: set[tuple[int, int]] = set()

    def variant() -> Mapping[str, object]:
        while True:
            window = (rng.choice(_VARIANT_START_YEARS), rng.choice(_VARIANT_END_YEARS))
            if window not in used:
                used.add(window)
                break
        identity = copy.deepcopy(dict(cast("Mapping[str, object]", definition["identity"])))
        identity["requested_window"] = {
            "kind": "range",
            "start_year": window[0],
            "end_year": window[1],
        }
        recomputed = recompute_identity(identity)
        return {
            "schema_version": 1,
            "spec_id": recomputed.spec_id,
            "identity": recomputed.identity,
        }

    return variant


def recorded_shapes(*, seed: int) -> list[Shape]:
    """The q1 and q3 shapes, from the recorded definitions."""
    recorded = json.loads(_RECORDED_DEFINITIONS.read_text())
    rng = random.Random(seed)
    shapes = []
    for name, key, description in SHAPES:
        definition = recorded[key]["definition"]
        shapes.append(
            Shape(
                name=name,
                description=description,
                definition=definition,
                variants=window_variants(definition, rng=rng),
            )
        )
    return shapes


def _concurrently(sends: Sequence[Callable[[], httpx.Response]]) -> list[Sample]:
    with ThreadPoolExecutor(max_workers=len(sends)) as pool:
        return list(pool.map(_sample, sends))


def bench_shape(client: httpx.Client, shape: Shape, *, samples: int) -> dict[str, object]:
    """Every request kind of one shape, as `summarise` reports it."""

    def mint(definition: Mapping[str, object]) -> Callable[[], httpx.Response]:
        return lambda: client.post("/api/datasets", json=_mint_body(definition))

    def get(path: str) -> Callable[[], httpx.Response]:
        return lambda: client.get(path, params=_QUERY)

    def registered(definition: Mapping[str, object]) -> str:
        response = client.post(
            "/api/datasets/register", json={"schema_version": 1, "definition": definition}
        )
        response.raise_for_status()
        return str(definition["spec_id"])

    primed = client.post("/api/datasets", json=_mint_body(shape.definition))
    primed.raise_for_status()
    manifest = primed.json()
    spec_id = manifest["definition"]["spec_id"]
    snapshot_path = f"/api/datasets/{spec_id}/{manifest['content_version']}"
    results: dict[str, list[Sample]] = {
        "cold_mint": [_sample(mint(shape.variants())) for _ in range(samples)],
        "warm_mint": [_sample(mint(shape.definition)) for _ in range(samples)],
        "cold_get": [
            _sample(get(f"/api/datasets/{registered(shape.variants())}")) for _ in range(samples)
        ],
        "warm_get": [_sample(get(f"/api/datasets/{spec_id}")) for _ in range(samples)],
        "snapshot": [_sample(get(snapshot_path)) for _ in range(samples)],
        "concurrent_warm_get": _concurrently(
            [get(f"/api/datasets/{spec_id}")] * _CONCURRENT_WARM_GETS
        ),
    }
    for width in _CONCURRENT_COLD_MINTS:
        results[f"concurrent_cold_mint_{width}"] = _concurrently(
            [mint(shape.variants()) for _ in range(width)]
        )
    return {
        "description": shape.description,
        "build": primed.headers.get("x-dfo-commit"),
        "spec_id": spec_id,
        "requests": {kind: summarise(kind_samples) for kind, kind_samples in results.items()},
    }


def run_bench(
    client: httpx.Client, shapes: Sequence[Shape], *, samples: int, seed: int
) -> dict[str, object]:
    """The bench report for `shapes`, read through `client`."""
    return {
        "base_url": str(client.base_url),
        "seed": seed,
        "samples": samples,
        "shapes": {shape.name: bench_shape(client, shape, samples=samples) for shape in shapes},
        "skipped": _SKIPPED,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--samples", type=int, default=5, help="samples per sequential kind")
    parser.add_argument("--seed", type=int, default=None, help="window-variant seed")
    parser.add_argument("--out", type=Path, default=Path("bench-datasets.json"))
    args = parser.parse_args(argv)
    seed = args.seed if args.seed is not None else time.time_ns() % 1_000_000
    with httpx.Client(base_url=args.base_url, timeout=_TIMEOUT_SECONDS) as client:
        report = run_bench(client, recorded_shapes(seed=seed), samples=args.samples, seed=seed)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
