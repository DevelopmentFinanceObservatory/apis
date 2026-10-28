"""`_BatchRouter` routes runs of SDMX batches with the result routing each batch gives."""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pytest

from dc_sdmx.errors import SdmxResponseError
from materialise import generation as generation_module
from materialise.errors import OverCeilingError

_CEILING = 250


def _batches(*sizes: int, refusing: int | None = None) -> list[pa.RecordBatch]:
    """Batches of `sizes` rows. Batch `refusing` holds one row its route refuses."""
    batches = []
    start = 0
    for position, size in enumerate(sizes):
        values = [f"r{row}" for row in range(start, start + size)]
        if position == refusing:
            values[-1] = "refused"
        batches.append(pa.RecordBatch.from_pydict({"value": values}))
        start += size
    return batches


class _Member:
    """One member surviving every row, refusing a batch that holds a `refused` row, with
    `_CEILING` as its streaming ceiling."""

    def __init__(self) -> None:
        self.routed: list[int] = []
        self.rows: list[str] = []
        self.counts = {0: 0}

    def survivors(self, batch: pa.RecordBatch) -> Iterator[tuple[int, list[pa.RecordBatch]]]:
        self.routed.append(batch.num_rows)
        values = batch.column("value").to_pylist()
        if "refused" in values:
            raise SdmxResponseError(f"1 of {batch.num_rows} SDMX rows is refused")
        yield 0, [batch]

    def admit(self, index: int, member_batches: list[pa.RecordBatch]) -> None:
        for batch in member_batches:
            self.rows += batch.column("value").to_pylist()
            self.counts[index] += batch.num_rows
        if self.counts[index] > _CEILING:
            raise OverCeilingError(
                "over", ceiling_kind="rows", count=self.counts[index], cap=_CEILING
            )


def _route(batches, *, route_rows: int, monkeypatch, draw=lambda _batch: None):
    monkeypatch.setattr(generation_module, "_ROUTE_BATCH_ROWS", route_rows)
    member = _Member()
    router = generation_module._BatchRouter(
        member.survivors,
        member.admit,
        ceilings=[(0, _CEILING)],
        survivor_counts=member.counts,
    )
    try:
        router.route(batches, draw=draw)
    except (SdmxResponseError, OverCeilingError, RuntimeError) as error:
        return member, error
    return member, None


def test_a_run_of_batches_is_routed_as_one(monkeypatch):
    """Batches whose rows stay under the run size and the ceiling are routed together, with
    the rows routing each alone gives."""
    batches = _batches(60, 60, 60, 60)
    joined, _ = _route(batches, route_rows=100_000, monkeypatch=monkeypatch)
    alone, _ = _route(batches, route_rows=0, monkeypatch=monkeypatch)

    assert joined.routed == [240]
    assert alone.routed == [60, 60, 60, 60]
    assert joined.rows == alone.rows


@pytest.mark.parametrize(
    ("sizes", "refusing", "draw_fails_at"),
    [
        pytest.param((100,) * 10, None, None, id="ceiling"),
        pytest.param((40, 40, 40, 40), 2, None, id="refusal"),
        pytest.param((40, 40, 40, 40), 1, 3, id="refusal-before-a-draw-error"),
    ],
)
def test_a_run_meets_the_ceiling_or_refusal_routing_each_batch_meets(
    monkeypatch, sizes, refusing, draw_fails_at
):
    """A ceiling trips at the batch and count it trips at batch by batch. A refusal names the
    refusing batch's rows, and comes before a later batch's draw error."""
    drawn: list[int] = []

    def draw(batch: pa.RecordBatch) -> None:
        drawn.append(batch.num_rows)
        if len(drawn) == draw_fails_at:
            raise RuntimeError("draw refused")

    def outcome(route_rows: int):
        drawn.clear()
        member, error = _route(
            _batches(*sizes, refusing=refusing),
            route_rows=route_rows,
            monkeypatch=monkeypatch,
            draw=draw,
        )
        return type(error), str(error), getattr(error, "count", None), member.counts[0]

    alone = outcome(0)
    drawn_alone = list(drawn)
    joined = outcome(100_000)

    assert joined == alone
    assert alone[0] is not type(None)
    if refusing is None:
        assert drawn == drawn_alone
