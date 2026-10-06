"""Test the availability floor's facet scope against the standard probe.

Dated queries within the candidate bound can share one POST. Undated queries,
large candidate sets, and observation previews use distinct cache keys. The fake
records call arguments and cache-miss POSTs separately.
"""

from __future__ import annotations

import asyncio
import threading

from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet, _force_full_facets
from search.engine.extract import DateRequest
from search.engine.region import resolve_variable
from search.engine.region.phase_intake import _FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES
from search.engine.region.state import EngineOptions
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_SV = "sv/gdp"
_PLACE = "country/KEN"
_QUERY = "gdp"

_NODES: dict = {
    _SV: {
        "label": "GDP",
        "type": "StatisticalVariable",
        "arcs": {
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    _PLACE: {"label": "Kenya", "type": "Country"},
}
_DETECT: dict = {_QUERY: {"svs": [_SV]}}
# Unconfirmable members push raw candidates past the bound but leave one shape.
_DETECT_MANY_CANDIDATES: dict = {
    _QUERY: {
        "svs": [_SV]
        + [f"sv/unconfirmable-{i}" for i in range(_FLOOR_PROBE_FULL_FACETS_MAX_CANDIDATES)]
    }
}
_RESOLVE: dict = {"Kenya": _PLACE}
_OBS: dict = {f"{_SV}|{_PLACE}": [{"earliestDate": "2020", "latestDate": "2022", "obsCount": 5}]}

# One Topic expands to 37 confirmed leaves, exceeding the floor-probe bound.
_TOPIC = "dc/topic/BigDomain"
_TOPIC_LEAF_COUNT = 37


def _topic_expansion_nodes() -> dict:
    leaves = [f"sv/topic-leaf-{i}" for i in range(_TOPIC_LEAF_COUNT)]
    nodes: dict = {
        _TOPIC: {
            "arcs": {
                "relevantVariable": {
                    "nodes": [{"dcid": leaf, "types": ["StatisticalVariable"]} for leaf in leaves]
                }
            }
        },
        _PLACE: {"label": "Kenya", "type": "Country"},
    }
    for leaf in leaves:
        # Matching five-tuples place every leaf in one ShapeDraft.
        nodes[leaf] = {
            "type": "StatisticalVariable",
            "arcs": {
                "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
            },
        }
    return nodes


_DETECT_TOPIC_EXPANDS_PAST_BOUND: dict = {_QUERY: {"svs": [_TOPIC], "types": {_TOPIC: "Topic"}}}


class _PostCountingGraph(FakeGraph):
    """Model the live per-(SV, entity, needs_dates) cache.

    `calls` records invocations before lookup. `posts` records cache misses.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_count = 0
        self.calls: list[tuple[int, bool]] = []
        self.posts: list[tuple[int, bool, tuple[str, ...]]] = []
        self._cache: dict[tuple[str, str, bool], list[Facet]] = {}
        self._lock = threading.Lock()

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        needs_dates = needs_dates or _force_full_facets.get()
        self.calls.append((len(stat_vars), needs_dates))
        if not stat_vars or not entities:
            return {}
        result: dict[str, dict[str, list[Facet]]] = {sv: {} for sv in stat_vars}
        miss_pairs: list[tuple[str, str]] = []
        with self._lock:
            for sv in stat_vars:
                for entity in entities:
                    cache_key = (sv, entity, needs_dates)
                    if cache_key in self._cache:
                        result[sv][entity] = self._cache[cache_key]
                    else:
                        miss_pairs.append((sv, entity))
            if not miss_pairs:
                return result
            self.post_count += 1
            self.posts.append((len(stat_vars), needs_dates, tuple(entities)))
        for sv, entity in miss_pairs:
            facets = self.observation_facets(stat_var=sv, entity=entity, needs_dates=needs_dates)
            with self._lock:
                self._cache[(sv, entity, needs_dates)] = facets
            result[sv][entity] = facets
        return result


class _UnusedLLM:
    """Placeholder while bind_universal is patched in the test."""

    async def generate_structured(self, *, prompt, system, schema):
        raise AssertionError("bind_universal was replaced by a spy; the LLM must not be called")


def _spy_bind_universal(monkeypatch) -> None:
    """Elect the first shape without invoking an LLM."""

    async def _spy(
        variable,
        shapes,
        *,
        admitted_qualifiers=(),
        admitted_ties=(),
        withheld_measure_axes=frozenset(),
        dropped_qualifier_phrases=(),
        llm,
    ):
        return (
            _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=None, ask=None),
            None,
        )

    monkeypatch.setattr("search.engine.region.phase_rank_bind.bind_universal", _spy)


def _run(
    *,
    date_request: DateRequest | None,
    include_observations: bool,
    detect: dict | None = None,
    nodes: dict | None = None,
) -> _PostCountingGraph:
    graph = _PostCountingGraph(
        nodes=nodes if nodes is not None else _NODES,
        detect=detect if detect is not None else _DETECT,
        resolve=_RESOLVE,
        obs=_OBS,
        places={},
        children={},
    )
    asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=date_request,
            detect_query=_QUERY,
            role_query=f"{_QUERY} of Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_UnusedLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(include_observations=include_observations),
        )
    )
    return graph


def test_dated_query_shares_one_post_between_floor_and_standard_probes(monkeypatch):
    """A dated query shares the floor and standard probes' cache entry."""
    _spy_bind_universal(monkeypatch)

    graph = _run(date_request=DateRequest(window=None, latest=True), include_observations=False)

    assert graph.post_count == 1


def test_dated_query_over_candidate_bound_issues_two_posts(monkeypatch):
    """A dated query above the candidate bound uses two cache entries."""
    _spy_bind_universal(monkeypatch)

    graph = _run(
        date_request=DateRequest(window=None, latest=True),
        include_observations=False,
        detect=_DETECT_MANY_CANDIDATES,
    )

    assert graph.post_count == 2


def test_undated_observation_bearing_query_issues_two_posts(monkeypatch):
    """An undated query with an observations preview uses two cache entries."""
    _spy_bind_universal(monkeypatch)

    graph = _run(date_request=None, include_observations=True)

    assert graph.post_count == 2


def test_dated_query_topic_expansion_past_bound_keeps_floor_probe_narrow_single_group(monkeypatch):
    """One Topic expands to 37 members after the initial candidate check.

    The floor probe stays narrow for all members. The standard probe requests
    full facets for the same group, so two cache-miss POSTs are expected.
    """
    _spy_bind_universal(monkeypatch)

    graph = _run(
        date_request=DateRequest(window=None, latest=True),
        include_observations=False,
        detect=_DETECT_TOPIC_EXPANDS_PAST_BOUND,
        nodes=_topic_expansion_nodes(),
    )

    # Both probes cover the same 37 members with different needs_dates keys.
    assert graph.post_count == 2
    assert (_TOPIC_LEAF_COUNT, False, (_PLACE,)) in graph.posts
    assert (_TOPIC_LEAF_COUNT, True, (_PLACE,)) in graph.posts


# Two shape groups expand to 37 members. Binding elects the first group.
_TOPIC_MULTI_GROUP = "dc/topic/MultiGroupDomain"
_TOPIC_MULTI_GROUP_A_COUNT = 15
_TOPIC_MULTI_GROUP_B_COUNT = 22
_TOPIC_MULTI_GROUP_TOTAL = _TOPIC_MULTI_GROUP_A_COUNT + _TOPIC_MULTI_GROUP_B_COUNT


def _topic_multi_group_nodes() -> dict:
    group_a = [f"sv/multi-a-{i}" for i in range(_TOPIC_MULTI_GROUP_A_COUNT)]
    group_b = [f"sv/multi-b-{i}" for i in range(_TOPIC_MULTI_GROUP_B_COUNT)]
    leaves = [*group_a, *group_b]
    nodes: dict = {
        _TOPIC_MULTI_GROUP: {
            "arcs": {
                "relevantVariable": {
                    "nodes": [{"dcid": leaf, "types": ["StatisticalVariable"]} for leaf in leaves]
                }
            }
        },
        _PLACE: {"label": "Kenya", "type": "Country"},
    }
    for leaf in group_a:
        nodes[leaf] = {
            "type": "StatisticalVariable",
            "arcs": {
                "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
            },
        }
    for leaf in group_b:
        # PopulationType alone distinguishes this group from group_a.
        nodes[leaf] = {
            "type": "StatisticalVariable",
            "arcs": {
                "populationType": {"nodes": [{"dcid": "Person"}]},
                "measuredProperty": {"nodes": [{"dcid": "count"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
            },
        }
    return nodes


_DETECT_TOPIC_MULTI_GROUP: dict = {
    _QUERY: {"svs": [_TOPIC_MULTI_GROUP], "types": {_TOPIC_MULTI_GROUP: "Topic"}}
}


def test_dated_query_topic_expansion_multi_group_past_bound_avoids_full_history_waste(
    monkeypatch,
):
    """The 15- and 22-member groups exceed the bound.

    Only the first group reaches the standard probe.
    """
    _spy_bind_universal(monkeypatch)

    graph = _run(
        date_request=DateRequest(window=None, latest=True),
        include_observations=False,
        detect=_DETECT_TOPIC_MULTI_GROUP,
        nodes=_topic_multi_group_nodes(),
    )

    # The floor probe covers all 37 members. The standard probe covers the elected group.
    assert graph.post_count == 2
    assert (_TOPIC_MULTI_GROUP_TOTAL, False, (_PLACE,)) in graph.posts
    # Full-history reads cover only the elected group's members.
    wide_posts = [(count, entities) for count, needs_dates, entities in graph.posts if needs_dates]
    assert wide_posts
    assert all(count < _TOPIC_MULTI_GROUP_TOTAL for count, _entities in wide_posts)
