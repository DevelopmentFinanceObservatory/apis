"""A bounded memo that keeps its most recently read entries, safe to share between threads."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable


class LruMemo[K: Hashable, V]:
    """At most `size` entries, the most recently read or kept. Reading an entry makes it
    the most recently read; keeping one past `size` drops the least recently read."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._entries: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """The entry under `key`, now the most recently read, or None."""
        with self._lock:
            found = self._entries.get(key)
            if found is not None:
                self._entries.move_to_end(key)
            return found

    def put(self, key: K, value: V) -> None:
        """Keep `value` under `key` as the most recently read entry."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["LruMemo"]
