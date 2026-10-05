"""The sync-materialisation size ceiling check."""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pytest

from dc_sdmx.results import observation_schema
from materialise.ceiling import check_ceiling
from materialise.errors import MaterialisationError, OverCeilingError
from materialise.facets import CoverageAdmission, FacetSelection
from materialise.generation import _collect_surviving_batches
from materialise.window import Coverage, RequestedWindow

from ._table_support import table_from_cells


def _coverage(rows: int) -> Coverage:
    return Coverage(rows=rows, dimensions=({"label": "years", "count": 3},), window=None)


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(10, id="under-ceiling"),
        pytest.param(100, id="at-inclusive-ceiling"),
    ],
)
def test_at_or_under_ceiling_does_not_raise(rows: int):
    check_ceiling(_coverage(rows), ceiling=100)


def test_one_over_ceiling_raises_over_ceiling():
    coverage = _coverage(101)

    with pytest.raises(MaterialisationError) as excinfo:
        check_ceiling(coverage, ceiling=100)

    assert excinfo.value.code == "over_ceiling"
    assert excinfo.value.status == 422
    assert excinfo.value.detail is not None
    assert "101" in excinfo.value.detail
    assert "100" in excinfo.value.detail
    assert "years" in excinfo.value.detail


def test_default_ceiling_used_when_not_specified():
    from materialise.ceiling import DEFAULT_SYNC_CEILING_ROWS

    check_ceiling(_coverage(DEFAULT_SYNC_CEILING_ROWS))
    with pytest.raises(MaterialisationError):
        check_ceiling(_coverage(DEFAULT_SYNC_CEILING_ROWS + 1))


_ROLES = ("recipient",)
_SELECTIONS = {
    axis: FacetSelection(
        axis=axis,
        ref=ref,
        basis="bound",
        default_policy=None,
        default_policy_version=None,
        selection_reason=None,
        dispatch_facet_id="facet1",
    )
    for axis, ref in (
        ("provenance", "OECD_DAC2A"),
        ("measurementMethod", "GrantEquivalent"),
        ("unit", "USD"),
    )
}


def _surviving_row(*, year: str) -> dict[str, str | None]:
    return {
        "variableMeasured": "CeilingOvershootBoundTest",
        "recipient": "country/TGO",
        "unit": "USD",
        "measurementMethod": "GrantEquivalent",
        "observationPeriod": None,
        "provenance": "OECD_DAC2A",
        "TIME_PERIOD": year,
        "OBS_VALUE": "1000",
        "scalingFactor": None,
        "facetId": "facet1",
    }


def _counting_batches(
    table: pa.Table, *, batch_size: int
) -> tuple[list[int], Iterator[pa.RecordBatch]]:
    """`table.to_batches(max_chunksize=batch_size)` wrapped to record each
    batch's own row count, in the order `_collect_surviving_batches`
    consumes them, into the returned list (mutated as the iterator is
    pulled, so a caller can inspect it after an exception interrupts the
    loop).
    """
    consumed: list[int] = []

    def generator() -> Iterator[pa.RecordBatch]:
        for batch in table.to_batches(max_chunksize=batch_size):
            consumed.append(batch.num_rows)
            yield batch

    return consumed, generator()


def test_survivor_ceiling_mid_stream_overshoot_bounded_to_one_batch():
    """With ceiling 5 and 20-row batches, the per-batch check can overshoot
    by at most 20. The 500 rows ensure the abort precedes end-of-stream.
    """
    ceiling = 5
    batch_size = 20
    table = table_from_cells((_surviving_row(year=str(2000 + i)) for i in range(500)), roles=_ROLES)
    consumed, batches = _counting_batches(table, batch_size=batch_size)

    with pytest.raises(OverCeilingError) as excinfo:
        _collect_surviving_batches(
            batches,
            schema=observation_schema(_ROLES),
            window=RequestedWindow(kind="all"),
            selections=_SELECTIONS,
            admission=CoverageAdmission(),
            ceiling=ceiling,
            stat_var="dcid:CeilingOvershootBoundTest",
        )

    assert excinfo.value.code == "over_ceiling"
    survivors_at_abort = sum(consumed)
    assert survivors_at_abort > ceiling
    assert survivors_at_abort <= ceiling + batch_size
    # The abort stops pulling batches before draining the whole 500-row table.
    assert survivors_at_abort < 500
