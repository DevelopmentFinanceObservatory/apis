"""Generation-keyed cache for the materialise SDMX client's ``stream_data_batches`` reads.

Search reads observations through the shared ``materialise`` core, which streams Arrow batches
from ``dc_sdmx.SdmxClient.stream_data_batches``. ``CachingMaterialiseSdmxClient`` records the
batches a read yields and stores them once the consumer exhausts the stream, so every entry is
a complete read. A read the consumer leaves early, or that raises, stores nothing. A hit replays
the stored roles and batches without a request. Each entry is weighed once, when it is stored,
by its Arrow bytes plus the heap its key and stored tuples hold and a fixed charge for the
cache's own bookkeeping, so an empty read costs more than zero and the byte budget bounds the
entry count. Entries expire with the graph generation they were read under. Other methods are
uncached.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import cachetools
import httpx
import pyarrow as pa

from dc_sdmx import ContainedIn, EntityFilter, SdmxClient
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from graph_generation.stamp import ttu_from_stamp_key
from search.engine.heap_weight import heap_bytes

type _EntitiesKey = tuple[tuple[str, tuple[str, ...] | ContainedIn], ...]
type _ReadKey = tuple[
    str,
    tuple[str, ...],
    _EntitiesKey,
    tuple[str, ...] | None,
    str | tuple[str, ...] | None,
    tuple[str, ...] | None,
    int | None,
    int | None,
]
type _Read = tuple[tuple[str, ...], tuple[pa.RecordBatch, ...]]

# The heap `cachetools.TLRUCache` keeps per entry beside its key and value, measured at about
# 220 bytes with tracemalloc on CPython 3.12.
_BOOKKEEPING_BYTES = 256


@dataclass(frozen=True, slots=True)
class _Entry:
    read: _Read
    size: int


def _entry_bytes(entry: _Entry) -> int:
    return entry.size


class CachingMaterialiseSdmxClient(SdmxClient):
    """``SdmxClient`` whose ``stream_data_batches`` reads are cached across requests.

    The key is every read argument except ``deadline``, under the generation stamp key, with
    the caller's order kept, since order can change the rows' order. ``max_bytes`` bounds the
    weight held in total, and ``0`` turns the cache off. A read weighing over
    ``max_entry_bytes`` is served and not kept, so one large read cannot evict the rest.
    Concurrent misses on one key each read.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None,
        timeout: float,
        transport: httpx.BaseTransport | None,
        chunk_concurrency: int,
        total_concurrency: threading.BoundedSemaphore,
        max_bytes: int,
        max_entry_bytes: int,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            base_url=base_url,
            token=token,
            timeout=timeout,
            transport=transport,
            chunk_concurrency=chunk_concurrency,
            total_concurrency=total_concurrency,
        )
        self._generation = generation
        self._max_bytes = max_bytes
        self._max_entry_bytes = min(max_entry_bytes, max_bytes)
        self._reads: cachetools.TLRUCache[_ReadKey, _Entry] = cachetools.TLRUCache(
            maxsize=max_bytes, ttu=ttu_from_stamp_key, timer=timer, getsizeof=_entry_bytes
        )
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Mapping[str, int]:
        """Hit, miss and entry counts and the weight held in bytes, for graph-generation
        registration."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._reads),
                "bytes": int(self._reads.currsize),
            }

    def clear(self) -> None:
        """Drop every stored read."""
        with self._lock:
            self._reads.clear()

    @contextmanager
    def stream_data_batches(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None = None,
        time_period: Literal["LATEST"] | Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> Iterator[tuple[tuple[str, ...], Iterator[pa.RecordBatch]]]:
        """``SdmxClient.stream_data_batches``, served from the cache when an equal read
        completed under the current generation."""
        if self._max_bytes == 0:
            with super().stream_data_batches(
                variables=variables,
                entities=entities,
                facet_ids=facet_ids,
                time_period=time_period,
                roles=roles,
                max_rows=max_rows,
                max_bytes=max_bytes,
                deadline=deadline,
            ) as streamed:
                yield streamed
            return
        key: _ReadKey = (
            self._generation.stamp().key,
            tuple(variables),
            tuple(
                (role, dcids if isinstance(dcids, ContainedIn) else tuple(dcids))
                for role, dcids in entities.items()
            ),
            None if facet_ids is None else tuple(facet_ids),
            time_period
            if time_period is None or isinstance(time_period, str)
            else tuple(time_period),
            None if roles is None else tuple(roles),
            max_rows,
            max_bytes,
        )
        with self._lock:
            cached = self._reads.get(key)
            if cached is None:
                self._misses += 1
            else:
                self._hits += 1
        if cached is not None:
            cached_roles, batches = cached.read
            yield cached_roles, iter(batches)
            return
        with super().stream_data_batches(
            variables=variables,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
            roles=roles,
            max_rows=max_rows,
            max_bytes=max_bytes,
            deadline=deadline,
        ) as (read_roles, read_batches):
            yield read_roles, self._recorded(key, read_roles, read_batches)

    def _recorded(
        self, key: _ReadKey, roles: tuple[str, ...], batches: Iterator[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        """Yield `batches`, then store them under `key` once they run out, unless the entry's
        weight passes the per-entry cap. Once the running Arrow total passes the cap the kept
        batches are dropped, so an oversized read holds only the batch in flight."""
        kept: list[pa.RecordBatch] = []
        size = 0
        for batch in batches:
            size += batch.nbytes
            if size > self._max_entry_bytes:
                kept.clear()
            else:
                kept.append(batch)
            yield batch
        if size > self._max_entry_bytes:
            return
        read = (roles, tuple(kept))
        weight = heap_bytes((key, read)) + _BOOKKEEPING_BYTES
        if weight > self._max_entry_bytes:
            return
        with self._lock:
            self._reads[key] = _Entry(read=read, size=weight)
