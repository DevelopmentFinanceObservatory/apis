"""Built node envelopes kept across requests, so a repeated node read skips the graph reads.

An entry keys on the generation stamp key, the build of every table a node can read and the
dcid (`NodeKey`). A request looks an entry up under the builds it sees before its read. An entry
is stored under the builds its own `built_from` names, and the request's builds for the tables
it did not read, so a table swapped while the entry was built stores it where only a request
that sees the new build finds it. Entries expire with their stamp's lifetime
(`graph_generation.ttu_from_stamp_key`).

The cache takes no lock: only the event loop touches it. The graph lifespan registers it with
the generation provider as `graph_api.node_entries`, which reads `stats` and schedules `clear`
on the loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import NamedTuple

from cachetools import TLRUCache

from graph_api.config import GRAPH_NODE_ENTRY_CACHE_SIZE
from graph_api.node_models import NodeResponse
from graph_generation.stamp import ttu_from_stamp_key


class NodeKey(NamedTuple):
    """The generation stamp key, the build of each table a node can read, and the dcid.
    `membership` is the version of the membership authority that holds the dcid, `None` where
    none does; `origin_catalogue` and `coverage_index` are `None` where the surface serves none.
    The table fields carry the names `built_from` gives them."""

    stamp: str
    code_index: str
    origin_catalogue: str | None
    coverage_index: str | None
    membership: str | None
    dcid: str


class NodeEntryCache:
    """At most `maxsize` built node envelopes, each expiring with its generation stamp."""

    def __init__(
        self,
        *,
        maxsize: int = GRAPH_NODE_ENTRY_CACHE_SIZE,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._entries: TLRUCache[NodeKey, NodeResponse] = TLRUCache(
            maxsize=maxsize, ttu=ttu_from_stamp_key, timer=timer
        )
        self._hits = 0
        self._misses = 0

    def get(self, key: NodeKey) -> NodeResponse | None:
        """The live envelope under `key`, counting a hit, or `None`, counting a miss."""
        node = self._entries.get(key)
        if node is None:
            self._misses += 1
        else:
            self._hits += 1
        return node

    def put(self, key: NodeKey, node: NodeResponse) -> None:
        """Store `node`, looked up under `key`, with each build its `built_from` names in place
        of `key`'s."""
        built = node.built_from
        stored = key._replace(
            code_index=built.get("code_index", key.code_index),
            origin_catalogue=built.get("origin_catalogue", key.origin_catalogue),
            coverage_index=built.get("coverage_index", key.coverage_index),
            membership=built.get("membership", key.membership),
        )
        self._entries[stored] = node

    def stats(self) -> Mapping[str, int]:
        """Hit and miss counters and the current entry count."""
        return {"hits": self._hits, "misses": self._misses, "entries": len(self._entries)}

    def clear(self) -> None:
        """Drop every entry. Counters keep running."""
        self._entries.clear()
