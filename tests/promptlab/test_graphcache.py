"""Checks GraphCache reads, writes, concurrency limits, and cache-only mode."""

from __future__ import annotations

import asyncio
import collections
import sqlite3
import threading
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from dc_graph import ArcTarget
from promptlab.graphcache import CacheMiss, GraphCache, StaleCaptureError, _key
from search.engine.clients.graph import (
    DetectedSV,
    EngineGraphClient,
    Facet,
    SubjectsCoverage,
    SubjectsPage,
    force_full_facets,
)
from search.engine.core import resolve_async
from search.engine.errors import GraphInfraError
from search.models import RawTextInput, ResolveRequest
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM

# Engine methods plus the evaluation-only methods and the per-resolve recall read GraphCache
# proxies.
_EXPECTED_METHODS = (
    "node_label",
    "node_arcs",
    "node_type",
    "default_facet_policy",
    "node_arcs_many",
    "node_arc_targets_many",
    "node_names_many",
    "node_subjects_many",
    "resolve_by_arc",
    "child_places_batch",
    "un_availability",
    "generation_stamp",
    "source_key",
    "entities_with_data_by_variable",
    "detect_svs",
    "observation_facets",
    "node_labels_batch",
    "node_types_batch",
    "node_arcs_batch",
    "node_observation_properties_batch",
    "observation_facets_batch",
    "observation_read_groups",
    "child_dcids",
    "enum_members",
    "currency_units",
    "entities_with_data",
    "exists_batch",
    "detect_legs",
)


class _CountingGraph:
    """Count calls forwarded to the wrapped FakeGraph."""

    def __init__(self, inner: FakeGraph | None = None):
        self._inner = inner if inner is not None else FakeGraph()
        self.calls: dict[str, int] = collections.defaultdict(int)
        self.facet_deadlines: list[float | None] = []

    def node_label(self, dcid):
        self.calls["node_label"] += 1
        return self._inner.node_label(dcid)

    def node_type(self, dcid):
        self.calls["node_type"] += 1
        return self._inner.node_type(dcid)

    def node_arcs(self, dcid):
        self.calls["node_arcs"] += 1
        return self._inner.node_arcs(dcid)

    def node_arcs_batch(self, dcids, *, deadline=None):
        self.calls["node_arcs_batch"] += 1
        return self._inner.node_arcs_batch(dcids)

    def node_labels_batch(self, dcids):
        self.calls["node_labels_batch"] += 1
        return self._inner.node_labels_batch(dcids)

    def node_types_batch(self, dcids):
        self.calls["node_types_batch"] += 1
        return self._inner.node_types_batch(dcids)

    def node_observation_properties_batch(self, dcids):
        self.calls["node_observation_properties_batch"] += 1
        return self._inner.node_observation_properties_batch(dcids)

    def node_arcs_many(
        self,
        dcids,
        *,
        properties,
        deadline: float | None = None,
    ):
        self.calls["node_arcs_many"] += 1
        return self._inner.node_arcs_many(dcids, properties=properties)

    def node_arc_targets_many(self, dcids, *, properties, deadline: float | None = None):
        self.calls["node_arc_targets_many"] += 1
        return self._inner.node_arc_targets_many(dcids, properties=properties)

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        self.calls["node_names_many"] += 1
        return self._inner.node_names_many(dcids)

    def node_subjects_many(self, dcids, *, property, max_results, deadline=None):
        self.calls["node_subjects_many"] += 1
        return self._inner.node_subjects_many(dcids, property=property, max_results=max_results)

    def resolve_by_arc(
        self,
        names,
        *,
        arc,
        deadline: float | None = None,
    ):
        self.calls["resolve_by_arc"] += 1
        return self._inner.resolve_by_arc(names, arc=arc)

    def child_places_batch(
        self,
        parent_dcids,
        child_type,
        *,
        deadline: float | None = None,
    ):
        self.calls["child_places_batch"] += 1
        return self._inner.child_places_batch(parent_dcids, child_type)

    def detect_svs(self, query):
        self.calls["detect_svs"] += 1
        return self._inner.detect_svs(query)

    def detect_legs(self, query):
        self.calls["detect_legs"] += 1
        return self._inner.detect_legs(query)

    def observation_facets(self, *, stat_var, entity, needs_dates=False):
        self.calls["observation_facets"] += 1
        return self._inner.observation_facets(
            stat_var=stat_var, entity=entity, needs_dates=needs_dates
        )

    def observation_facets_batch(
        self, stat_vars, entities, needs_dates=False, *, max_bytes=None, deadline=None
    ):
        self.calls["observation_facets_batch"] += 1
        self.facet_deadlines.append(deadline)
        return self._inner.observation_facets_batch(
            stat_vars, entities, needs_dates=needs_dates, max_bytes=max_bytes
        )

    def observation_read_groups(self, stat_vars, entities, needs_dates=False):
        self.calls["observation_read_groups"] += 1
        return self._inner.observation_read_groups(stat_vars, entities, needs_dates)

    def child_dcids(self, parent_dcid):
        self.calls["child_dcids"] += 1
        return self._inner.child_dcids(parent_dcid)

    def enum_members(self, property_dcid):
        self.calls["enum_members"] += 1
        return self._inner.enum_members(property_dcid)

    def entities_with_data(self, variable_dcids, entity_dcids, *, max_bytes=None):
        self.calls["entities_with_data"] += 1
        return self._inner.entities_with_data(variable_dcids, entity_dcids, max_bytes=max_bytes)


def _cache(tmp_path: Path, inner=None, *, read_only: bool = False) -> GraphCache:
    return GraphCache(
        inner if inner is not None else _CountingGraph(),
        path=tmp_path / "graph.sqlite",
        read_only=read_only,
    )


# ---------------------------------------------------------------------------
# Fail-closed method surface
# ---------------------------------------------------------------------------


def test_every_engine_graph_client_method_is_proxied():
    """GraphCache defines a method for every EngineGraphClient method and evaluation extra."""
    protocol_methods = {
        name
        for name in vars(EngineGraphClient)
        if not name.startswith("_") and callable(getattr(EngineGraphClient, name))
    }
    assert protocol_methods == set(_EXPECTED_METHODS) - {
        "exists_batch",
        "detect_legs",
    }
    for name in _EXPECTED_METHODS:
        assert callable(getattr(GraphCache, name, None)), f"{name} is not proxied on GraphCache"


def test_unknown_attribute_raises_not_implemented(tmp_path):
    cache = _cache(tmp_path)
    with pytest.raises(NotImplementedError, match="materialise_something_new"):
        _ = cache.materialise_something_new


def test_unknown_method_call_raises_not_implemented(tmp_path):
    cache = _cache(tmp_path)
    with pytest.raises(NotImplementedError):
        cache.some_future_engine_method("arg")


# ---------------------------------------------------------------------------
# Hit / miss / write-back, one representative per method family
# ---------------------------------------------------------------------------


def test_node_label_hit_miss_write_back(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    dcid = "measuredValue"

    first = cache.node_label(dcid)
    second = cache.node_label(dcid)

    assert first == second == "measuredValue"
    assert inner.calls["node_label"] == 1


def test_node_arcs_batch_shares_cache_with_node_arcs(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    dcid = "measuredValue"

    via_singular = cache.node_arcs(dcid)
    via_batch = cache.node_arcs_batch([dcid])[dcid]

    assert via_singular == via_batch
    # Both calls share the `node_arcs` cache entry.
    assert inner.calls["node_arcs_batch"] == 1


def test_node_labels_batch_reuses_node_label_cache_and_omits_absent(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    cache.node_label("measuredValue")  # warms the shared "node_label" entry
    result = cache.node_labels_batch(["measuredValue", "does-not-exist"])

    assert result == {"measuredValue": "measuredValue"}
    # Only the absent DCID reaches the inner batch call.
    assert inner.calls["node_labels_batch"] == 1


def test_node_observation_properties_batch_hit_miss_write_back(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    dcid = (
        "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
        "CRSPurposeCode99810_Nominal"
    )

    first = cache.node_observation_properties_batch([dcid])
    second = cache.node_observation_properties_batch([dcid])

    assert first == second == {dcid: ("financingProvider", "financingRecipient")}
    assert inner.calls["node_observation_properties_batch"] == 1


def test_node_observation_properties_batch_present_but_empty_entry_is_cached(tmp_path):
    """An empty tuple is a valid result and is cached."""
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    first = cache.node_observation_properties_batch(["measuredValue"])
    second = cache.node_observation_properties_batch(["measuredValue"])

    assert first == second == {"measuredValue": ()}
    assert inner.calls["node_observation_properties_batch"] == 1


def test_node_arcs_many_hit_miss_write_back(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    dcid = "country/KEN"

    first = cache.node_arcs_many([dcid], properties=("typeOf", "name"))
    second = cache.node_arcs_many([dcid], properties=("typeOf", "name"))

    assert first == second == {dcid: {"typeOf": ("Country",), "name": ("Kenya",)}}
    assert inner.calls["node_arcs_many"] == 1


def test_node_arcs_many_absent_dcid_caches_none(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    first = cache.node_arcs_many(["does-not-exist"], properties=("typeOf",))
    second = cache.node_arcs_many(["does-not-exist"], properties=("typeOf",))

    assert first == second == {"does-not-exist": None}
    assert inner.calls["node_arcs_many"] == 1


def test_node_arc_targets_many_serves_a_written_target_s_name_and_types(tmp_path):
    nodes = {
        "dc/base/WHO_GHO": {"label": "WHO_GHO", "arcs": {"isPartOf": {"nodes": [{"dcid": "d"}]}}},
        "d": {"label": "WHO Global Health Observatory", "type": "Dataset"},
    }
    inner = _CountingGraph(FakeGraph(nodes=nodes))

    written = _cache(tmp_path, inner).node_arc_targets_many(
        ["dc/base/WHO_GHO"], properties=("isPartOf",)
    )
    served = _cache(tmp_path, inner, read_only=True).node_arc_targets_many(
        ["dc/base/WHO_GHO"], properties=("isPartOf",)
    )

    assert served == written
    assert written["dc/base/WHO_GHO"]["isPartOf"] == (
        ArcTarget(value="d", name="WHO Global Health Observatory", types=("Dataset",)),
    )
    assert inner.calls["node_arc_targets_many"] == 1


def test_read_only_node_arc_targets_many_names_a_miss_from_its_captured_label(tmp_path):
    """A capture holding provenance names as `node_label` rows replays a name read."""
    path = tmp_path / "graph.sqlite"
    writer = GraphCache(
        _CountingGraph(FakeGraph(nodes={"dc/base/WHO_GHO": {"label": "WHO_GHO"}})), path=path
    )
    writer.node_labels_batch(["dc/base/WHO_GHO", "dc/base/Gone"])
    writer.close()
    read_only = GraphCache(FakeGraph(raise_on_call=True), path=path, read_only=True)

    served = read_only.node_arc_targets_many(
        ["dc/base/WHO_GHO", "dc/base/Gone"], properties=("isPartOf", "name", "typeOf")
    )

    assert served == {
        "dc/base/WHO_GHO": {"name": (ArcTarget(value="WHO_GHO"),)},
        "dc/base/Gone": None,
    }
    with pytest.raises(CacheMiss):
        read_only.node_arc_targets_many(["dc/base/Other"], properties=("name",))
    with pytest.raises(CacheMiss):
        read_only.node_arc_targets_many(["dc/base/WHO_GHO"], properties=("isPartOf",))


def test_node_subjects_many_hit_miss_write_back(tmp_path):
    nodes = {
        "child1": {"arcs": {"financingProvider": {"nodes": [{"dcid": "lead1"}]}}},
    }
    inner = _CountingGraph(FakeGraph(nodes=nodes))
    cache = _cache(tmp_path, inner)

    first = cache.node_subjects_many(["lead1"], property="financingProvider", max_results=25)
    second = cache.node_subjects_many(["lead1"], property="financingProvider", max_results=25)

    assert (
        first
        == second
        == {"lead1": SubjectsPage(dcids=("child1",), coverage=SubjectsCoverage.complete)}
    )
    assert inner.calls["node_subjects_many"] == 1


def test_node_subjects_many_no_subjects_caches_as_negative(tmp_path):
    inner = _CountingGraph(FakeGraph(nodes={}))
    cache = _cache(tmp_path, inner)

    first = cache.node_subjects_many(["lead1"], property="financingProvider", max_results=25)
    second = cache.node_subjects_many(["lead1"], property="financingProvider", max_results=25)

    assert first == second == {"lead1": SubjectsPage(dcids=(), coverage=SubjectsCoverage.complete)}
    assert inner.calls["node_subjects_many"] == 1


class _RecordingSubjectsGraph:
    """Record which DCIDs each inner batch fetch receives."""

    def __init__(self, inner: FakeGraph):
        self._inner = inner
        self.batches: list[list[str]] = []

    def node_subjects_many(self, dcids, *, property, max_results, deadline=None):
        self.batches.append(list(dcids))
        return self._inner.node_subjects_many(dcids, property=property, max_results=max_results)


def test_node_subjects_many_partial_warm_batch_fetches_only_misses(tmp_path):
    nodes = {
        "child1": {"arcs": {"financingProvider": {"nodes": [{"dcid": "lead1"}]}}},
        "child2": {"arcs": {"financingProvider": {"nodes": [{"dcid": "lead2"}]}}},
    }
    recorder = _RecordingSubjectsGraph(FakeGraph(nodes=nodes))
    cache = _cache(tmp_path, recorder)

    cache.node_subjects_many(["lead1"], property="financingProvider", max_results=25)
    result = cache.node_subjects_many(
        ["lead1", "lead2"], property="financingProvider", max_results=25
    )

    assert result == {
        "lead1": SubjectsPage(dcids=("child1",), coverage=SubjectsCoverage.complete),
        "lead2": SubjectsPage(dcids=("child2",), coverage=SubjectsCoverage.complete),
    }
    assert recorder.batches == [["lead1"], ["lead2"]]


def test_node_subjects_many_read_only_raises_cache_miss(tmp_path):
    path = tmp_path / "graph.sqlite"
    inner = _CountingGraph()
    read_only = GraphCache(inner, path=path, read_only=True)

    with pytest.raises(CacheMiss):
        read_only.node_subjects_many(["lead1"], property="financingProvider", max_results=25)
    assert inner.calls["node_subjects_many"] == 0


def test_node_subjects_many_keys_on_max_results(tmp_path):
    # The cap belongs in the key so a capped page cannot satisfy a larger request.
    nodes = {
        "child1": {"arcs": {"financingProvider": {"nodes": [{"dcid": "lead1"}]}}},
        "child2": {"arcs": {"financingProvider": {"nodes": [{"dcid": "lead1"}]}}},
        "child3": {"arcs": {"financingProvider": {"nodes": [{"dcid": "lead1"}]}}},
    }
    inner = _CountingGraph(FakeGraph(nodes=nodes))
    cache = _cache(tmp_path, inner)

    capped = cache.node_subjects_many(["lead1"], property="financingProvider", max_results=1)
    uncapped = cache.node_subjects_many(["lead1"], property="financingProvider", max_results=25)

    assert capped == {"lead1": SubjectsPage(dcids=("child1",), coverage=SubjectsCoverage.capped)}
    assert uncapped == {
        "lead1": SubjectsPage(
            dcids=("child1", "child2", "child3"), coverage=SubjectsCoverage.complete
        )
    }
    assert inner.calls["node_subjects_many"] == 2


def test_node_subjects_many_stale_capture_raises(tmp_path):
    """A cache row without `coverage` raises `StaleCaptureError`."""
    path = tmp_path / "graph.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE graph_cache (key TEXT PRIMARY KEY, method TEXT NOT NULL, "
        "value TEXT NOT NULL, written_at REAL NOT NULL)"
    )
    key = _key("node_subjects_many", "lead1", "financingProvider", 25)
    conn.execute(
        "INSERT INTO graph_cache (key, method, value, written_at) VALUES (?, ?, ?, ?)",
        (key, "node_subjects_many", '{"dcids": ["child1"], "truncated": false}', 0.0),
    )
    conn.commit()
    conn.close()

    cache = GraphCache(_CountingGraph(), path=path)
    with pytest.raises(StaleCaptureError, match=str(path)):
        cache.node_subjects_many(["lead1"], property="financingProvider", max_results=25)


def test_resolve_by_arc_hit_miss_write_back(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    arc = "<-description{typeOf:Country}->dcid"

    first = cache.resolve_by_arc(["Kenya"], arc=arc)
    second = cache.resolve_by_arc(["Kenya"], arc=arc)

    assert first == second == {"Kenya": ["country/KEN"]}
    assert inner.calls["resolve_by_arc"] == 1


def test_resolve_by_arc_miss_caches_empty_list(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    arc = "<-description{typeOf:Country}->dcid"

    first = cache.resolve_by_arc(["Nowhereland"], arc=arc)
    second = cache.resolve_by_arc(["Nowhereland"], arc=arc)

    assert first == second == {"Nowhereland": []}
    assert inner.calls["resolve_by_arc"] == 1


def test_child_places_batch_hit_miss_write_back(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    first = cache.child_places_batch(["africa"], "Country")
    second = cache.child_places_batch(["africa"], "Country")

    assert first == second
    assert first["africa"]
    assert inner.calls["child_places_batch"] == 1


def test_detect_svs_hit_miss_write_back(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    query = "government education spending Kenya"

    first = cache.detect_svs(query)
    second = cache.detect_svs(query)

    assert first.candidates
    assert first.candidates == second.candidates
    assert first.un.result() == second.un.result()
    assert inner.calls["detect_legs"] == 1


def test_observation_facets_hit_miss_write_back(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    sv = (
        "Amount_EconomicActivity_ExpenditureActivity_EducationExpenditure_Government_"
        "AsFractionOf_Amount_EconomicActivity_ExpenditureActivity_Government"
    )
    entity = "country/KEN"

    first = cache.observation_facets(stat_var=sv, entity=entity)
    second = cache.observation_facets(stat_var=sv, entity=entity)

    assert first == second
    assert inner.calls["observation_facets"] == 1


def test_observation_facets_batch_reuses_observation_facets_cache(tmp_path):
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)
    sv = (
        "Amount_EconomicActivity_ExpenditureActivity_EducationExpenditure_Government_"
        "AsFractionOf_Amount_EconomicActivity_ExpenditureActivity_Government"
    )
    entity = "country/KEN"

    cache.observation_facets(stat_var=sv, entity=entity)  # warms the shared "obs" entry
    result = cache.observation_facets_batch([sv], [entity])

    assert result[sv][entity] == cache.observation_facets(stat_var=sv, entity=entity)
    assert inner.calls["observation_facets"] == 1
    assert inner.calls["observation_facets_batch"] == 0


def test_observation_facets_batch_passes_its_deadline_to_a_miss_read(tmp_path):
    """A miss reads the inner graph under the caller's deadline, as the bound-unit gate's
    dated read (`convertibility._dated_plan`) passes it; a replay reads nothing."""
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    cache.observation_facets_batch(["SomeVar"], ["country/KEN"], needs_dates=True, deadline=5.0)
    cache.observation_facets_batch(["SomeVar"], ["country/KEN"], needs_dates=True, deadline=6.0)

    assert inner.facet_deadlines == [5.0]


def test_child_dcids_hit_miss_write_back(tmp_path):
    # This method expects each node value to contain an `arcs` mapping.
    nodes = {
        "child1": {"arcs": {"isPartOf": {"nodes": [{"dcid": "parent1"}]}}},
    }
    inner = _CountingGraph(FakeGraph(nodes=nodes))
    cache = _cache(tmp_path, inner)

    first = cache.child_dcids("parent1")
    second = cache.child_dcids("parent1")

    assert first == second == ["child1"]
    assert inner.calls["child_dcids"] == 1


def test_enum_members_hit_miss_write_back(tmp_path):
    inner = _CountingGraph(FakeGraph(enum_members={"prop/Foo": ["EnumVal1", "EnumVal2"]}))
    cache = _cache(tmp_path, inner)

    first = cache.enum_members("prop/Foo")
    second = cache.enum_members("prop/Foo")

    assert first == second == ["EnumVal1", "EnumVal2"]
    assert inner.calls["enum_members"] == 1


def test_entities_with_data_hit_miss_write_back(tmp_path):
    sv = (
        "Amount_EconomicActivity_ExpenditureActivity_EducationExpenditure_Government_"
        "AsFractionOf_Amount_EconomicActivity_ExpenditureActivity_Government"
    )
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    first = cache.entities_with_data([sv], ["country/KEN"])
    second = cache.entities_with_data([sv], ["country/KEN"])

    assert first == second == ["country/KEN"]
    assert inner.calls["entities_with_data"] == 1


def test_entities_with_data_preserves_current_call_order_across_a_cache_hit(tmp_path):
    """The cache keys the qualifying set without order and returns each call's input order."""
    sv = "FertilityRate_Person_Female"
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    forward = cache.entities_with_data([sv], ["country/KEN", "country/ETH"])
    reordered = cache.entities_with_data([sv], ["country/ETH", "country/KEN"])

    assert forward == ["country/KEN", "country/ETH"]
    assert reordered == ["country/ETH", "country/KEN"]
    assert inner.calls["entities_with_data"] == 1


def test_exists_batch_uses_label_batch_then_caches_singleton_evidence(tmp_path):
    """Resolve positives through labels, then confirm and cache singleton misses."""
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    present = cache.exists_batch(["measuredValue", "does-not-exist"])

    assert present == {"measuredValue"}
    assert inner.calls["node_labels_batch"] == 1
    assert inner.calls["node_arcs"] == 1
    again = cache.exists_batch(["measuredValue", "does-not-exist"])
    assert again == present
    assert inner.calls["node_labels_batch"] == 1
    assert inner.calls["node_arcs"] == 1


def test_exists_batch_does_not_cache_a_failed_singleton_confirmation(tmp_path):
    """A failed singleton lookup propagates and stays uncached."""

    class FailingSingletonGraph:
        def __init__(self) -> None:
            self.node_arcs_calls = 0

        def node_labels_batch(self, dcids):
            return {}

        def node_arcs(self, dcid):
            self.node_arcs_calls += 1
            raise GraphInfraError("malformed singleton evidence")

    inner = FailingSingletonGraph()
    cache = _cache(tmp_path, inner)

    for _ in range(2):
        with pytest.raises(GraphInfraError, match="malformed singleton evidence"):
            cache.exists_batch(["country/ARE"])

    assert inner.node_arcs_calls == 2


# ---------------------------------------------------------------------------
# Facet / DetectedSV round-trip through the real SQLite store
# ---------------------------------------------------------------------------


def test_facet_round_trips_through_sqlite_unchanged(tmp_path):
    sv = (
        "Amount_EconomicActivity_ExpenditureActivity_EducationExpenditure_Government_"
        "AsFractionOf_Amount_EconomicActivity_ExpenditureActivity_Government"
    )
    entity = "country/KEN"
    path = tmp_path / "graph.sqlite"

    writer = GraphCache(FakeGraph(), path=path)
    written = writer.observation_facets(stat_var=sv, entity=entity)
    writer.close()

    # The reader fails if the value is missing from SQLite.
    reader = GraphCache(FakeGraph(raise_on_call=True), path=path)
    replayed = reader.observation_facets(stat_var=sv, entity=entity)

    assert replayed == written
    assert all(isinstance(f, Facet) for f in replayed)


def test_detected_sv_round_trips_through_sqlite_unchanged(tmp_path):
    query = "government education spending Kenya"
    path = tmp_path / "graph.sqlite"

    writer = GraphCache(FakeGraph(), path=path)
    written = writer.detect_legs(query)
    writer.close()

    reader = GraphCache(FakeGraph(raise_on_call=True), path=path)
    replayed = reader.detect_legs(query)

    assert replayed == written
    assert all(isinstance(c, DetectedSV) for c in replayed.ours)


def test_failed_un_resolve_is_captured_per_leg(tmp_path):
    """A capture stores each resolve, so a replay reproduces a failed UN resolve."""
    query = "under-five mortality rate"
    path = tmp_path / "graph.sqlite"
    ours_sv = {"svs": ["Count_Death_Upto5Years"], "cosine_scores": [0.8]}

    writer = GraphCache(FakeGraph(detect={query: ours_sv}, un_detect={query: None}), path=path)
    written = writer.detect_svs(query)
    writer.close()
    reader = GraphCache(FakeGraph(raise_on_call=True), path=path)
    replayed = reader.detect_svs(query)

    assert replayed.candidates == written.candidates
    assert replayed.un.result() is None
    assert written.un.result() is None
    assert [c.dcid for c in replayed.candidates] == ["Count_Death_Upto5Years"]


# ---------------------------------------------------------------------------
# Inner failures never reach the cache
# ---------------------------------------------------------------------------


def test_inner_failure_propagates_and_writes_nothing(tmp_path):
    path = tmp_path / "graph.sqlite"
    failing = GraphCache(FakeGraph(raise_on_call=True), path=path)

    with pytest.raises(GraphInfraError):
        failing.node_label("measuredValue")

    # A failed call must leave no entry that masks the next inner lookup.
    healthy = _CountingGraph()
    recovered = GraphCache(healthy, path=path)
    assert recovered.node_label("measuredValue") == "measuredValue"
    assert healthy.calls["node_label"] == 1


# ---------------------------------------------------------------------------
# read_only mode
# ---------------------------------------------------------------------------


def test_read_only_raises_cache_miss_instead_of_falling_through(tmp_path):
    path = tmp_path / "graph.sqlite"
    inner = _CountingGraph()
    read_only = GraphCache(inner, path=path, read_only=True)

    with pytest.raises(CacheMiss):
        read_only.node_label("measuredValue")
    assert inner.calls["node_label"] == 0


def test_read_only_serves_an_existing_hit(tmp_path):
    path = tmp_path / "graph.sqlite"
    writer = GraphCache(_CountingGraph(), path=path)
    writer.node_label("measuredValue")
    writer.close()

    read_only = GraphCache(FakeGraph(raise_on_call=True), path=path, read_only=True)
    assert read_only.node_label("measuredValue") == "measuredValue"


# ---------------------------------------------------------------------------
# The fall-through semaphore admits at most 4
# ---------------------------------------------------------------------------


class _BlockingGraph:
    """Block four calls together and track the peak concurrent call count."""

    def __init__(self, barrier: threading.Barrier):
        self._barrier = barrier
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def node_label(self, dcid):
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        self._barrier.wait(timeout=5)
        with self._lock:
            self.current -= 1
        return dcid


def test_semaphore_admits_at_most_four_concurrent_inner_calls(tmp_path):
    barrier = threading.Barrier(4)
    inner = _BlockingGraph(barrier)
    cache = GraphCache(inner, path=tmp_path / "graph.sqlite")

    threads = [threading.Thread(target=cache.node_label, args=(f"dcid/{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()

    assert inner.peak == 4


# ---------------------------------------------------------------------------
# force_full_facets: key on the resolved needs_dates, not the raw argument
# ---------------------------------------------------------------------------


def test_force_full_facets_resolved_needs_dates_key(tmp_path):
    sv = (
        "Amount_EconomicActivity_ExpenditureActivity_EducationExpenditure_Government_"
        "AsFractionOf_Amount_EconomicActivity_ExpenditureActivity_Government"
    )
    entity = "country/KEN"
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    # Outside a scope, `False` resolves to false.
    cache.observation_facets(stat_var=sv, entity=entity, needs_dates=False)
    assert inner.calls["observation_facets"] == 1

    # The scope changes the resolved key, causing a second fetch.
    with force_full_facets(True):
        cache.observation_facets(stat_var=sv, entity=entity, needs_dates=False)
    assert inner.calls["observation_facets"] == 2

    # The explicit `True` call reuses the scoped entry.
    cache.observation_facets(stat_var=sv, entity=entity, needs_dates=True)
    assert inner.calls["observation_facets"] == 2

    # The original false-key entry remains available.
    cache.observation_facets(stat_var=sv, entity=entity, needs_dates=False)
    assert inner.calls["observation_facets"] == 2


def test_force_full_facets_nested_false_inside_true_restores_outer_on_exit(tmp_path):
    """Nested scopes restore the outer `True` value after the inner `False` scope exits."""
    sv = (
        "Amount_EconomicActivity_ExpenditureActivity_EducationExpenditure_Government_"
        "AsFractionOf_Amount_EconomicActivity_ExpenditureActivity_Government"
    )
    entity = "country/KEN"
    inner = _CountingGraph()
    cache = _cache(tmp_path, inner)

    with force_full_facets(True):
        # The nested false scope uses a separate key.
        with force_full_facets(False):
            cache.observation_facets(stat_var=sv, entity=entity, needs_dates=False)
        assert inner.calls["observation_facets"] == 1

        # The outer true scope resumes with its own key.
        cache.observation_facets(stat_var=sv, entity=entity, needs_dates=False)
    assert inner.calls["observation_facets"] == 2

    # Outside a scope, the raw false call reuses the nested entry.
    cache.observation_facets(stat_var=sv, entity=entity, needs_dates=False)
    assert inner.calls["observation_facets"] == 2


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_the_store_on_exit(tmp_path):
    with GraphCache(FakeGraph(), path=tmp_path / "graph.sqlite") as cache:
        cache.node_label("x")

    with pytest.raises(sqlite3.ProgrammingError):
        cache._store._conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Equivalence: GraphCache in front of resolve_async changes nothing observable
# ---------------------------------------------------------------------------


def _resolve(graph) -> dict:
    req = ResolveRequest(input=RawTextInput(query="health ODA grants from USA to Ethiopia"))
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        response = asyncio.run(
            resolve_async(req, graph=graph, llm=FakeLLM(), sdmx=FakeSdmxClient())
        )
    dumped = response.model_dump(mode="json")
    # Remove per-request telemetry before comparing pipeline results.
    dumped["diagnostics"].pop("trace_id", None)
    dumped["diagnostics"].pop("timing_ms", None)
    for part in dumped["parts"]:
        part.pop("timing_ms", None)
    return dumped


def test_graph_cache_is_transparent_to_resolve_async(tmp_path):
    baseline = _resolve(FakeGraph())
    cached = _resolve(GraphCache(FakeGraph(), path=tmp_path / "equivalence.sqlite"))

    assert cached == baseline
