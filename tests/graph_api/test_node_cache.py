"""`NodeEntryCache` stores an envelope under the builds its `built_from` names."""

from __future__ import annotations

import threading
import time

from pydantic import TypeAdapter

from graph_api.node_cache import NodeEntryCache, NodeKey
from graph_api.node_models import NodeResponse
from graph_generation.config import GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS
from graph_generation.stamp import UNKNOWN_KEY

_KEY = NodeKey(
    stamp="rev-1",
    code_index="index-a",
    origin_catalogue="catalogue-a",
    coverage_index="coverage-a",
    membership=None,
    dcid="dc/base/Foo",
)


def _node(built_from: dict[str, str]) -> NodeResponse:
    return TypeAdapter(NodeResponse).validate_python(
        {
            "ref": "dc/base/Foo",
            "label": "Some Node",
            "kind": "other",
            "types": ["SomeOtherType"],
            "description": None,
            "links": {},
            "gaps": [],
            "built_from": built_from,
        }
    )


def test_an_entry_built_from_a_swapped_table_is_stored_under_the_new_build():
    cache = NodeEntryCache()
    node = _node({"code_index": "index-a", "coverage_index": "coverage-b"})

    cache.put(_KEY, node)

    assert cache.get(_KEY) is None
    assert cache.get(_KEY._replace(coverage_index="coverage-b")) is node


def test_an_entry_expires_with_its_stamp():
    now = [0.0]
    cache = NodeEntryCache(timer=lambda: now[0])
    key = _KEY._replace(stamp=UNKNOWN_KEY)
    cache.put(key, _node({"code_index": "index-a"}))

    now[0] = GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS - 1
    assert cache.get(key) is not None
    now[0] = GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS + 1
    assert cache.get(key) is None


def test_stats_from_another_thread_waits_for_a_request_in_progress():
    # The timer runs inside `get`, after the request has entered the cache. From there it starts
    # `stats` on a second thread, the way the generation provider's refresh thread calls it, and
    # gives it 0.2 s to finish. `stats` expires entries through `len`, so it must wait for the
    # request instead of mutating the cache under it.
    outcome: dict[str, bool] = {}
    caller = threading.get_ident()

    def timer() -> float:
        if threading.get_ident() == caller and "stats_ran_inside" not in outcome:
            reader = threading.Thread(target=cache.stats)
            reader.start()
            reader.join(timeout=0.2)
            outcome["stats_ran_inside"] = not reader.is_alive()
        return time.monotonic()

    cache = NodeEntryCache(timer=timer)
    cache.get(_KEY)

    assert outcome == {"stats_ran_inside": False}
