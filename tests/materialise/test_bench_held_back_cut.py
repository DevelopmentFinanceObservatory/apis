"""A timing bench of the held-back cut and the reported share on a synthetic CRS-shaped read.

Skipped unless `DFO_BENCH_HELD_BACK=1`, since it times work rather than checking it. Run it
alone, printing:

    DFO_BENCH_HELD_BACK=1 uv run pytest -s -p no:testmon \\
        tests/materialise/test_bench_held_back_cut.py

Three timings, each the best of `_REPEATS` runs:

- `member_mint` and `group_mint`: a range mint (`materialise_spec`, `materialise_group`) of one
  role-bearing StatVar with `_SERIES` series (20 providers by 100 recipients) over `_YEARS`
  years ending in 2025, whose source marks 2025 partial. The window is 1990 to 2020, so the
  rows of 2024 and 2025 are read only to count the share. The SDMX reads come from an
  in-process double, so each timing includes CSV parsing.
- `index_reads`: `_INDEX_READS` `CoverageIndex.series_years` reads under the default held-back
  cut, over an index of the same series from 1960, spread across 50 facets of the marked
  source.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest

from materialise import materialise_group, materialise_spec
from materialise.enrichment import EnrichmentLookup
from materialise.facets import FacetBinding
from materialise.rate_tables import RateTableCache
from materialise.window import RequestedWindow
from origin_catalogue import bound_catalogue
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.origin_catalogue.fake_catalogue import documenting

from .test_conversion_finish import (
    _BARE_STAT_VAR,
    _MEMBER_PROVENANCE,
    _ROLE_HEADER,
    _STAT_VAR,
    _base_arcs,
    _graph,
    _request,
    _role_row,
    _router_sdmx,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("DFO_BENCH_HELD_BACK") != "1", reason="a timing bench; DFO_BENCH_HELD_BACK=1"
)

_SERIES = 2000
_YEARS = 100
_REPEATS = 3
_INDEX_READS = 200
_LAST_YEAR = 2025
_FIRST_YEAR = _LAST_YEAR - _YEARS + 1
# The fixture horizon starts in 1960.
_INDEX_FIRST_YEAR = 1960
_WINDOW = RequestedWindow(kind="range", start_year=1990, end_year=2020)
_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_FACETS = tuple(f"f{index}" for index in range(50))


# 20 providers by 100 recipients, so the read's URL names 120 places.
_PROVIDERS = tuple(f"country/P{index:02d}" for index in range(20))
_RECIPIENTS = tuple(f"country/R{index:03d}" for index in range(_SERIES // len(_PROVIDERS)))


def _parties(series: int) -> tuple[str, str]:
    return _PROVIDERS[series % len(_PROVIDERS)], _RECIPIENTS[series // len(_PROVIDERS)]


def _index():
    rows = [
        (
            _BARE_STAT_VAR,
            _FACETS[series % len(_FACETS)],
            *_parties(series),
            list(range(_INDEX_FIRST_YEAR, _LAST_YEAR + 1)),
        )
        for series in range(_SERIES)
    ]
    return build_coverage_index(
        rows,
        role_names={_BARE_STAT_VAR: _ROLES},
        facet_metadata={
            facet: FacetMetadata(
                provenance_id=_MEMBER_PROVENANCE,
                measurement_method="GrantEquivalent",
                unit="CurrencyUSD",
                observation_period="P1Y",
            )
            for facet in _FACETS
        },
    )


def _mint(entry: str) -> Callable[[], object]:
    body = _ROLE_HEADER + "".join(
        _role_row(
            provider=_parties(series)[0],
            recipient=_parties(series)[1],
            year=str(year),
            value=str(year % 97 + 1),
            facet_id=_FACETS[series % len(_FACETS)],
        )
        for series in range(_SERIES)
        for year in range(_FIRST_YEAR, _LAST_YEAR + 1)
    )
    arcs = _base_arcs()
    request = _request(
        stat_var=_STAT_VAR,
        unit=FacetBinding(state="bound", ref="CurrencyUSD"),
        window=_WINDOW,
        role_bindings={
            "financingProvider": list(_PROVIDERS),
            "financingRecipient": list(_RECIPIENTS),
        },
    )
    coverage = _index()

    def run() -> object:
        sdmx, _ = _router_sdmx(role_bodies={_BARE_STAT_VAR: body})
        common = {
            "ceiling": 10_000_000,
            "sdmx": sdmx,
            "graph": _graph(arcs),
            "materialiser_build": "bench-build",
            "capability_version": "v1",
            "enrichment": EnrichmentLookup(_graph(arcs)),
            "parties": {},
            "rate_cache": RateTableCache(),
            "coverage": coverage,
        }
        if entry == "group":
            result = materialise_group([request], **common)
        else:
            result = materialise_spec(request, **common)
        assert result.year_totals, "the bench read counted no share"
        return result

    return run


def _index_reads() -> Callable[[], object]:
    index = _index()

    def run() -> object:
        for _ in range(_INDEX_READS):
            found = index.series_years(variables=[_BARE_STAT_VAR])
            assert found is not None
        return index

    return run


def _best(run: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(_REPEATS):
        started = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - started)
    return best


def test_bench_held_back_cut() -> None:
    marked = documenting({_MEMBER_PROVENANCE: {"partialObservationDate": (str(_LAST_YEAR),)}})
    with bound_catalogue(marked):
        timings = {
            "member_mint": _best(_mint("member")),
            "group_mint": _best(_mint("group")),
            "index_reads": _best(_index_reads()),
        }
    print(f"\n{_SERIES} series x {_YEARS} years, best of {_REPEATS}")
    for name, seconds in timings.items():
        print(f"{name:12s} {seconds:8.3f} s")
