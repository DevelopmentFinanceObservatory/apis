"""Test the materialise SDMX client's observation-read cache over a counting transport."""

from __future__ import annotations

import weakref
from contextlib import contextmanager

import httpx
import pyarrow as pa
import pytest

from dc_sdmx import SdmxClient
from graph_generation import FakeGeneration
from search.engine.clients.materialise_bridge import build_materialise_sdmx_client

_BODY = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId\r\n"
    "dataflow,DC:DF_OBS(1.0.0),I,Count_Person,country/KEN,Person,Census,P1Y,"
    "dc/base,2020,1.50,0,f1\r\n"
)


class _ConsumerError(Exception):
    pass


def _read(client, *, consume: str = "all") -> pa.Table | None:
    """Read once: `all` drains the stream, `none` leaves it unread, `raise` fails after the
    first batch."""
    with client.stream_data_batches(
        variables=["Count_Person"],
        entities={"observationAbout": ["country/KEN"]},
        facet_ids=["f1"],
        max_rows=100,
    ) as (_roles, batches):
        if consume == "raise":
            next(batches)
            raise _ConsumerError
        return pa.Table.from_batches(list(batches)) if consume == "all" else None


@pytest.mark.parametrize(
    ("first_consume", "next_token", "max_entry_bytes", "expected_requests"),
    [
        pytest.param("all", "gen-a", 1 << 20, 1, id="complete-read-is-served-from-cache"),
        pytest.param("none", "gen-a", 1 << 20, 2, id="abandoned-read-is-not-stored"),
        pytest.param("raise", "gen-a", 1 << 20, 2, id="read-failing-mid-drain-is-not-stored"),
        pytest.param("all", "gen-a", 1, 2, id="read-over-the-entry-cap-is-not-stored"),
        pytest.param("all", "gen-b", 1 << 20, 2, id="new-generation-reads-again"),
    ],
)
def test_second_identical_read_reuses_only_a_complete_read_of_the_same_generation(
    first_consume: str,
    next_token: str,
    max_entry_bytes: int,
    expected_requests: int,
    search_settings,
) -> None:
    search_settings(materialise_sdmx_cache_max_entry_bytes=max_entry_bytes)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=_BODY.encode(),
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    generation = FakeGeneration(token="gen-a")
    client = build_materialise_sdmx_client(
        base="https://graph.example.org",
        transport=httpx.MockTransport(handler),
        generation=generation,
    )

    if first_consume == "raise":
        with pytest.raises(_ConsumerError):
            _read(client, consume="raise")
        first = None
    else:
        first = _read(client, consume=first_consume)
    generation.advance(next_token)
    second = _read(client)

    assert len(requests) == expected_requests
    assert second is not None
    assert second.column("OBS_VALUE").to_pylist() == ["1.50"]
    if first is not None:
        assert second.equals(first)


def test_a_read_past_the_entry_cap_releases_its_kept_batches_before_the_next_is_read(
    monkeypatch, search_settings
) -> None:
    """Each batch the source yields after the cap trips finds every batch before the one in
    flight already freed, so an oversized read never holds more than one batch."""
    batch_bytes = pa.record_batch({"v": list(range(1000))}).nbytes
    search_settings(materialise_sdmx_cache_max_entry_bytes=batch_bytes * 3 // 2)
    yielded: list[weakref.ref[pa.RecordBatch]] = []
    retained_at_read: list[int] = []

    def source():
        for index in range(4):
            if index >= 2:
                retained_at_read.append(sum(ref() is not None for ref in yielded[:-1]))
            batch = pa.record_batch({"v": list(range(1000))})
            yielded.append(weakref.ref(batch))
            yield batch

    @contextmanager
    def stream_data_batches(self, **_kwargs):
        yield ("observationAbout",), source()

    monkeypatch.setattr(SdmxClient, "stream_data_batches", stream_data_batches)
    client = build_materialise_sdmx_client(
        base="https://graph.example.org", generation=FakeGeneration(token="gen-a")
    )

    with client.stream_data_batches(
        variables=["Count_Person"], entities={"observationAbout": ["country/KEN"]}
    ) as (_roles, batches):
        drained = sum(1 for _batch in batches)

    assert drained == 4
    assert retained_at_read == [0, 0]
    assert client.stats()["entries"] == 0


def test_empty_reads_are_charged_so_the_byte_budget_bounds_the_entry_count(
    monkeypatch, search_settings
) -> None:
    """A read yielding no batches holds no Arrow bytes, yet its entry still costs its key and
    the cache's bookkeeping, so distinct empty reads evict each other under the budget."""
    max_bytes = 16 * 1024
    search_settings(
        materialise_sdmx_cache_max_bytes=max_bytes,
        materialise_sdmx_cache_max_entry_bytes=max_bytes,
    )

    @contextmanager
    def stream_data_batches(self, **_kwargs):
        yield ("observationAbout",), iter(())

    monkeypatch.setattr(SdmxClient, "stream_data_batches", stream_data_batches)
    client = build_materialise_sdmx_client(
        base="https://graph.example.org", generation=FakeGeneration(token="gen-a")
    )

    reads = 1000
    for index in range(reads):
        with client.stream_data_batches(
            variables=[f"Count_Person_{index}"], entities={"observationAbout": ["country/KEN"]}
        ) as (_roles, batches):
            assert list(batches) == []

    stats = client.stats()
    assert 0 < stats["entries"] < reads
    assert 0 < stats["bytes"] <= max_bytes
