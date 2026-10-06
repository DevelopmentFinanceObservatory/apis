"""Heap weight of an object graph, shared by the engine's byte-bounded caches."""

from __future__ import annotations

import gc
import sys

import pyarrow as pa


def heap_bytes(
    root: object, *, shared: tuple[type, ...] = (type,), limit: int | None = None
) -> int:
    """`sys.getsizeof` summed over every object reachable from `root` once, through
    `gc.get_referents`, apart from instances of `shared`, which the process holds without
    it. A record batch counts its Python object and its Arrow buffers (`nbytes`), and the
    walk does not descend into it. The walk stops once the total passes `limit`, returning
    a total over it, so weighing a graph a cache refuses costs no more than weighing one it
    keeps."""
    seen: set[int] = set()
    pending: list[object] = [root]
    size = 0
    while pending and (limit is None or size <= limit):
        obj = pending.pop()
        if id(obj) in seen or isinstance(obj, shared):
            continue
        seen.add(id(obj))
        size += sys.getsizeof(obj)
        if isinstance(obj, pa.RecordBatch):
            size += obj.nbytes
        else:
            pending.extend(gc.get_referents(obj))
    return size
