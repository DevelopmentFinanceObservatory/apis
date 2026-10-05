"""Measure peak RSS and write time for the three streaming format writers.

The real Parquet, CSV, and Excel writers consume synthetic CRS-shaped frames in 500-row
chunks. The measured row counts are 462,720 and 2,517,909. The Parquet row-group coalescing
threshold is 100,000 rows. Whole-table builds at these sizes used 636 MB to 3.28 GB.
Streaming runs target peak memory below 400 MB. The measurement runs offline on synthetic inputs.

Each row-count/format pair runs in a spawned process. The child reads ``ru_maxrss`` after the
write. macOS reports bytes and Linux KiB. The Excel measurement skips row counts above
``EXCEL_MAX_ROWS``. The 2,517,909-row case exceeds that limit. The sink counts output bytes
without retaining them, which keeps output memory out of peak RSS.

Usage:
    uv run python scripts/measure_writer_rss.py
    uv run python scripts/measure_writer_rss.py --out /tmp/writer-rss.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import resource
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

# Real row totals for the CRS families crs_family_233 and widest_shape_members.
ROW_COUNTS = (462_720, 2_517_909)

CHUNK_ROWS = 500
"""Synthetic rows per chunk, below the 100,000-row Parquet coalescing threshold."""

_STAT_VARS = tuple(f"dcid:CRSSyntheticFlow{i:03d}" for i in range(40))
_DONORS = tuple(f"country/DNR{i:02d}" for i in range(20))
_RECIPIENTS = tuple(f"country/RCP{i:02d}" for i in range(20))
_YEARS = tuple(str(year) for year in range(1973, 2025))
_UNITS = ("USD", None)


@dataclass(frozen=True)
class _SyntheticTable:
    """A literal implementation of the writers' ``Table`` protocol, without frame grammar."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


_COLUMNS = ("stat_var", "financingProvider", "financingRecipient", "time_period", "value", "unit")


def _synthetic_chunks(total_rows: int, *, chunk_rows: int) -> Iterator[_SyntheticTable]:
    """Yield fixed-pool StatVar/donor/recipient/year/unit chunks. The final chunk may be short.

    Deterministic rows keep input variation out of the measurement.
    """
    row_index = 0
    while row_index < total_rows:
        this_chunk_rows = min(chunk_rows, total_rows - row_index)
        rows = tuple(
            (
                _STAT_VARS[(row_index + i) % len(_STAT_VARS)],
                _DONORS[(row_index + i) % len(_DONORS)],
                _RECIPIENTS[(row_index + i) % len(_RECIPIENTS)],
                _YEARS[(row_index + i) % len(_YEARS)],
                float((row_index + i) % 10_000) + 0.5,
                _UNITS[(row_index + i) % len(_UNITS)],
            )
            for i in range(this_chunk_rows)
        )
        yield _SyntheticTable(columns=_COLUMNS, rows=rows)
        row_index += this_chunk_rows


def _max_rss_bytes() -> int:
    """Return process peak RSS in bytes. macOS reports bytes, and Linux reports KiB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _render_in_child(rows: int, format_name: str, result_queue: multiprocessing.Queue) -> None:
    """Run one writer and return output size, wall time, and peak RSS through the child queue.

    Import the writer in the child so each process measures its own import and write cost.
    """
    from materialise.formats import Rendering, write_csv, write_excel, write_parquet

    writers = {"csv": write_csv, "xlsx": write_excel, "parquet": write_parquet}
    writer = writers[format_name]

    class _DiscardingSink:
        """Count output bytes and discard the rendered data to keep sink memory small."""

        def __init__(self) -> None:
            self._size = 0

        def write(self, data: bytes, /) -> int:
            self._size += len(data)
            return len(data)

    sink = _DiscardingSink()
    started = time.perf_counter()
    artifact = writer(
        _synthetic_chunks(rows, chunk_rows=CHUNK_ROWS), rendering=Rendering(), sink=sink
    )
    wall_seconds = time.perf_counter() - started
    result_queue.put(
        {
            "bytes": artifact.size_bytes,
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": _max_rss_bytes(),
        }
    )


def _measure(rows: int, format_name: str) -> dict:
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(target=_render_in_child, args=(rows, format_name, result_queue))
    process.start()
    result = result_queue.get()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"child process for rows={rows} format={format_name} exited {process.exitcode}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help=(
            f"Row counts to measure, overriding the default research sizes "
            f"{ROW_COUNTS} (e.g. a small value for a quick smoke run)."
        ),
    )
    args = parser.parse_args()
    row_counts = tuple(args.rows) if args.rows else ROW_COUNTS

    from materialise.formats import EXCEL_MAX_ROWS

    results = []
    header = f"{'rows':>12}  {'format':<8}  {'peak RSS (MB)':>14}  {'wall (s)':>9}  {'bytes':>12}"
    print(header)
    print("-" * len(header))
    for rows in row_counts:
        for format_name in ("parquet", "csv", "xlsx"):
            if format_name == "xlsx" and rows > EXCEL_MAX_ROWS:
                print(f"{rows:>12,}  {format_name:<8}  {'skipped: over EXCEL_MAX_ROWS':>14}")
                continue
            measurement = _measure(rows, format_name)
            peak_mb = measurement["peak_rss_bytes"] / (1024 * 1024)
            print(
                f"{rows:>12,}  {format_name:<8}  {peak_mb:>14.1f}  "
                f"{measurement['wall_seconds']:>9.2f}  {measurement['bytes']:>12,}"
            )
            results.append({"rows": rows, "format": format_name, **measurement})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
