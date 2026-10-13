"""Sources-door tests use in-memory arcs and recorded `/v2/node` responses."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from graph_api import app as app_module
from graph_api import deps as deps_module
from graph_api.config import DCP_QUEUE_WAIT_WARN_SECONDS
from graph_api.doors import sources as sources_door
from graph_api.version import GRAPH_CAPABILITY_VERSION
from graph_generation import FakeGeneration
from materialise import provenance_doc
from materialise.graph import GraphDeadlineExceededError, GraphReadError, SubjectPage, parse_arcs
from materialise.provenance_doc import (
    _DATASET_PROPERTIES,
    _NODE_PROPERTIES,
    _ROLLUP_PROPERTIES,
    _SOLE_CHILD_PROPERTIES,
    UNCONDITIONAL_GAPS,
    NotASourceNode,
    SourceDocumentNotFound,
    TooManyChildProvenances,
)
from origin_catalogue import BASE, UN_SYSTEM
from series_coverage.index import ProvenanceShape
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

from .conftest import FakeGraphNodeReader, make_client, send_past_a_full_pool, wait_until

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _wire_arcs(
    fixture_name: str, dcid: str, properties: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    body = json.loads((_FIXTURES_DIR / fixture_name).read_text())
    arcs = body["data"][dcid]["arcs"]
    return parse_arcs(arcs, properties, mark_unresolved=True)


def _wire_subject_dcids(fixture_name: str, dcid: str, property: str) -> tuple[str, ...]:
    body = json.loads((_FIXTURES_DIR / fixture_name).read_text())
    nodes = body["data"][dcid]["arcs"].get(property, {}).get("nodes", [])
    return tuple(n["dcid"] for n in nodes if "dcid" in n)


def _strings(value: object) -> Iterator[str]:
    """Every string inside a decoded JSON body."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


class TestWireMapping:
    def test_single_child_dataset_maps_every_graph_filled_member(self):
        graph = FakeGraphNodeReader(
            nodes={
                "dc/d/Dataset1": {
                    "typeOf": ("Dataset",),
                    "name": ("Readable Dataset Name",),
                    "url": ("https://dataset.example.org",),
                    "isPartOf": ("dc/s/Publisher1",),
                },
                "dc/s/Publisher1": {"name": ("Publisher One",)},
                "CreativeCommonsAttributionLicense": {
                    "name": ("Creative Commons Attribution License",)
                },
                "dc/base/Child1": {
                    "description": ("The child's own description.",),
                    "sourceDataUrl": ("https://dataset.example.org/data.csv",),
                    "licenseType": ("CreativeCommonsAttributionLicense",),
                    "license": ("https://dataset.example.org/terms",),
                    "sourceReleaseFrequency": ("P1M",),
                    "lastDataRefreshDate": ("2024-06-01",),
                    "nextDataRefreshDate": ("2024-07-01",),
                },
            },
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/Child1",), truncated=False
                )
            },
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/dc/d/Dataset1")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["details"] == {
            "name_of_source": "Readable Dataset Name",
            "publisher": "Publisher One",
            "source": None,
            "data_commons": None,
            "description": "The child's own description.",
            "source_url": "https://dataset.example.org",
            "source_data_url": "https://dataset.example.org/data.csv",
            "citation": None,
        }
        assert body["license"] == {
            # The wire-mapping layer converts the enum dcid to a display label.
            "type": "Creative Commons Attribution License",
            "url": "https://dataset.example.org/terms",
            "update_cadence": "Monthly",
            "last_update": "2024-06-01",
            "next_expected_update": "2024-07-01",
        }
        assert body["scope_and_units"] == {
            "countries": [],
            "years_covered": None,
            "granularity": None,
            "currencies": [],
        }
        assert body["methodology"] == {"description": None}
        assert body["data_notes"] == []
        assert body["usage"] == {"api_access": None, "python": None}
        assert body["transformation_log"] == []
        assert body["version_history"] == []
        assert body["document_errors"] == []
        # The one child carries no earliestObservationDate/latestObservationDate,
        # so years_covered stays a gap even though every other roll-up filled.
        assert body["gaps"] == [
            "details.source",
            "details.data_commons",
            "details.citation",
            "scope_and_units.countries",
            "scope_and_units.years_covered",
            "scope_and_units.granularity",
            "scope_and_units.currencies",
            "methodology.description",
            "data_notes",
            "usage.api_access",
            "usage.python",
            "transformation_log",
            "version_history",
        ]

    def test_absent_fields_serve_null_not_an_empty_string(self):
        """Missing scalars serialize as `null` and missing arrays as `[]`."""
        graph = FakeGraphNodeReader(
            nodes={"provenance/Orphan": {"typeOf": ("Provenance",)}},
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["details"] == {
            "name_of_source": None,
            "publisher": None,
            "source": None,
            "data_commons": None,
            "description": None,
            "source_url": None,
            "source_data_url": None,
            "citation": None,
        }
        assert body["license"] == {
            "type": None,
            "url": None,
            "update_cadence": None,
            "last_update": None,
            "next_expected_update": None,
        }
        assert body["methodology"] == {"description": None}
        assert body["data_notes"] == []
        assert body["transformation_log"] == []
        assert body["version_history"] == []
        assert body["document_errors"] == []
        assert set(body["gaps"]) == {
            "details.name_of_source",
            "details.publisher",
            "details.source",
            "details.data_commons",
            "details.description",
            "details.source_url",
            "details.source_data_url",
            "details.citation",
            "license.type",
            "license.url",
            "license.update_cadence",
            "license.last_update",
            "license.next_expected_update",
            "scope_and_units.years_covered",
            "scope_and_units.granularity",
            "methodology.description",
            "data_notes",
            "scope_and_units.currencies",
            "transformation_log",
            "version_history",
            *UNCONDITIONAL_GAPS,
        }

    def test_license_type_serves_the_licence_node_s_graph_name(self):
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "licenseType": ("InternationalMonetaryFundCopyrightAndUsage",),
                },
                "InternationalMonetaryFundCopyrightAndUsage": {
                    "name": ("IMF Copyright and Usage",)
                },
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        assert resp.json()["license"]["type"] == "IMF Copyright and Usage"

    def test_a_licence_node_the_graph_does_not_hold_serves_its_dcid(self):
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "licenseType": ("SomeLicenseTheGraphDoesNotHold",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        assert resp.json()["license"]["type"] == "SomeLicenseTheGraphDoesNotHold"


class TestScopeAndUnits:
    def test_years_covered_is_filled_and_dropped_from_gaps_when_the_graph_has_it(self):
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "earliestObservationDate": ("1973",),
                    "latestObservationDate": ("2024",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["scope_and_units"] == {
            "countries": [],
            "years_covered": "1973-2024",
            "granularity": None,
            "currencies": [],
        }
        assert "scope_and_units.years_covered" not in body["gaps"]
        # The graph has no country-coverage field, so `countries` remains a gap.
        assert "scope_and_units.countries" in body["gaps"]

    def test_years_covered_absent_leaves_it_in_gaps(self):
        graph = FakeGraphNodeReader(nodes={"provenance/Orphan": {"typeOf": ("Provenance",)}})
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["scope_and_units"]["years_covered"] is None
        assert "scope_and_units.years_covered" in body["gaps"]


class TestRouting:
    def test_slash_bearing_dcid_reaches_the_documentation_door_intact(self):
        """The route preserves all segments of a slash-separated dcid."""
        dcid = "dc/d/WorldBank_WorldDevelopmentIndicators"
        graph = FakeGraphNodeReader(
            nodes={dcid: {"typeOf": ("Dataset",)}},
            subjects={
                (dcid, "isPartOf"): SubjectPage(
                    dcids=("dc/base/WorldDevelopmentIndicators",), truncated=False
                )
            },
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get(f"/api/graph/documentation/{dcid}")

        assert resp.status_code == 200, resp.text
        assert "details" in resp.json()

    def test_node_door_still_serves_a_slash_bearing_dcid_unchanged(self):
        """The node route also preserves multi-segment dcids."""
        graph = FakeGraphNodeReader(
            nodes={"dc/d/WorldBank_WorldDevelopmentIndicators": {"typeOf": ("Dataset",)}}
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/node/dc/d/WorldBank_WorldDevelopmentIndicators")

        assert resp.status_code == 200, resp.text
        assert resp.json()["ref"] == "dc/d/WorldBank_WorldDevelopmentIndicators"


class TestRefusals:
    def test_unheld_dcid_refuses_node_not_held(self):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/dc/base/Nowhere")

        assert resp.status_code == 404, resp.text
        body = resp.json()
        assert body["code"] == "node_not_held"
        assert body["pointer"] == "/dcid"
        assert body["capability_version"] == GRAPH_CAPABILITY_VERSION

    def test_a_node_that_is_not_a_source_refuses_request_invalid(self):
        graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/country/KEN")

        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["code"] == "request_invalid"
        assert body["pointer"] == "/dcid"

    def test_a_source_dcid_given_directly_refuses_request_invalid(self):
        """The door accepts Dataset and Provenance roots. A publisher Source
        routed as orphan provenance would return a mostly empty document.
        """
        graph = FakeGraphNodeReader(
            nodes={"dc/s/WorldBank": {"typeOf": ("Source",), "name": ("World Bank",)}}
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/dc/s/WorldBank")

        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["code"] == "request_invalid"
        assert body["pointer"] == "/dcid"

    def test_oversized_dcid_refuses_request_invalid_before_any_read(self):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph)
        dcid = "a" * 513

        with client:
            resp = client.get(f"/api/graph/documentation/{dcid}")

        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["code"] == "request_invalid"
        assert body["pointer"] == "/dcid"
        assert graph.calls["node_arcs_many"] == 0

    def test_truncated_children_page_refuses_service_unavailable(self):
        graph = FakeGraphNodeReader(
            nodes={"dc/d/Dataset1": {"typeOf": ("Dataset",)}},
            subjects={
                ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                    dcids=("dc/base/ChildA",), truncated=True
                )
            },
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/dc/d/Dataset1")

        assert resp.status_code == 503, resp.text
        body = resp.json()
        assert body["code"] == "service_unavailable"
        assert body["retryable"] is False

    def test_classify_sources_exception_maps_every_reused_code_directly(self):
        assert (
            sources_door.classify_sources_exception(SourceDocumentNotFound("dcid")).code
            == "node_not_held"
        )
        assert (
            sources_door.classify_sources_exception(NotASourceNode("dcid", ("Country",))).code
            == "request_invalid"
        )
        assert (
            sources_door.classify_sources_exception(TooManyChildProvenances("dc/d/Dataset1")).code
            == "service_unavailable"
        )
        assert (
            sources_door.classify_sources_exception(
                GraphReadError("exemplar", upstream_status=404)
            ).code
            == "upstream_error"
        )
        assert sources_door.classify_sources_exception(ValueError("unrelated")) is None

    def test_classify_sources_exception_maps_deadline_exceeded_to_timeout(self):
        # `GraphDeadlineExceededError` subclasses `GraphReadError`. Its 504
        # arm must precede the general read-error arm.
        problem = sources_door.classify_sources_exception(GraphDeadlineExceededError("exemplar"))
        assert problem is not None
        assert problem.status == 504
        assert problem.code == "timeout"
        assert problem.retryable is True


class _BlockingGraphNodeReader:
    """Block `node_arcs_many` until the test releases it."""

    def __init__(self, delegate: FakeGraphNodeReader) -> None:
        self._delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()
        # One document read makes several graph calls under one deadline, so distinct
        # deadlines count the document reads that reached the reader.
        self.read_deadlines: set[float | None] = set()
        self.calls = delegate.calls

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.read_deadlines.add(deadline)
        self.entered.set()
        self.release.wait(timeout=5.0)
        return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        return self._delegate.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        return self._delegate.node_names_many(dcids)


_TWO_ORPHANS = {
    "provenance/Orphan": {"typeOf": ("Provenance",)},
    "provenance/OtherOrphan": {"typeOf": ("Provenance",)},
}


def test_a_read_past_the_pool_size_is_refused_at_once():
    """With every worker busy a read answers retryable 503 without reaching the reader. The
    in-pool read and a read after it succeed, and the refusal left nothing cached."""
    graph = _BlockingGraphNodeReader(FakeGraphNodeReader(nodes=_TWO_ORPHANS))
    client = make_client(graph=graph, node_pool_size=1)
    # Two sources, since two requests for one source share one read and never queue.
    paths = iter(
        [
            "/api/graph/documentation/provenance/Orphan",
            "/api/graph/documentation/provenance/OtherOrphan",
        ]
    )

    with client:
        responses = send_past_a_full_pool(
            client,
            lambda: client.get(next(paths)),
            entered=graph.entered,
            release=graph.release,
        )
        after = client.get("/api/graph/documentation/provenance/OtherOrphan")

    in_pool, refused = responses
    assert in_pool.status_code == 200, in_pool.text
    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"
    assert refused.json()["retryable"] is True
    assert after.status_code == 200, after.text
    assert len(graph.read_deadlines) == 2


def test_a_read_while_a_timed_out_read_holds_the_worker_is_refused_at_once(monkeypatch):
    """A read that outlives its door timeout answers 504 and keeps its worker until it
    returns, so a read sent meanwhile answers retryable 503 without reaching the reader. The
    next read runs once that worker returns, and the read that ended past its deadline left
    nothing cached."""
    monkeypatch.setattr(sources_door, "GRAPH_NODE_TIMEOUT_SECONDS", 0.2)
    graph = _BlockingGraphNodeReader(FakeGraphNodeReader(nodes=_TWO_ORPHANS))
    client = make_client(graph=graph, node_pool_size=1)

    with client:
        running = client.get("/api/graph/documentation/provenance/Orphan")
        assert graph.entered.wait(timeout=5.0), "the first read never started"
        refused = client.get("/api/graph/documentation/provenance/OtherOrphan")
        graph.release.set()
        wait_until(lambda: client.app.state.graph_deps.dcp_executor._in_flight == 0)
        recovered = client.get("/api/graph/documentation/provenance/Orphan")

    assert running.status_code == 504, running.text
    assert refused.status_code == 503, refused.text
    assert refused.json()["code"] == "service_unavailable"
    assert refused.json()["retryable"] is True
    assert recovered.status_code == 200, recovered.text
    assert len(graph.read_deadlines) == 2


@pytest.mark.parametrize(("waited_s", "level"), [(0.05, "DEBUG"), (1.5, "WARNING")])
def test_a_dcp_read_logs_its_queue_wait(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, waited_s: float, level: str
):
    """A wait longer than `DCP_QUEUE_WAIT_WARN_SECONDS` logs at WARNING with the threshold."""
    clock = iter([100.0, 100.0 + waited_s])
    monkeypatch.setattr(deps_module, "perf_counter", lambda: next(clock))

    def read() -> str:
        return "read"

    with caplog.at_level(logging.DEBUG, logger=deps_module.__name__):
        assert deps_module.log_queue_wait(read)() == "read"

    records = [record for record in caplog.records if record.name == deps_module.__name__]
    assert [record.levelname for record in records] == [level]
    args = records[0].args
    assert args[1] == pytest.approx(waited_s * 1000)
    assert args[2:] == ((DCP_QUEUE_WAIT_WARN_SECONDS * 1000,) if level == "WARNING" else ())


class _RaisingGraphNodeReader:
    """`node_arcs_many()` raises outright -- exercises the sources door's
    own classify branch end-to-end, through a real request. Mirrors
    `tests/graph_api/test_node_door.py`'s own `_RaisingGraphNodeReader`.
    """

    def __init__(self, exc: GraphReadError) -> None:
        self._exc = exc

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        raise self._exc

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        raise self._exc

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        raise self._exc


def test_graph_deadline_exceeded_end_to_end_maps_to_504_timeout():
    graph = _RaisingGraphNodeReader(GraphDeadlineExceededError("exemplar"))
    client = make_client(graph=graph)

    with client:
        resp = client.get("/api/graph/documentation/provenance/Orphan")

    assert resp.status_code == 504, resp.text
    problem = resp.json()
    assert problem["code"] == "timeout"
    assert problem["retryable"] is True


class _FailingOnceGraphNodeReader:
    """Raise `GraphReadError` from the first `node_arcs_many` call, then delegate."""

    def __init__(self, delegate: FakeGraphNodeReader) -> None:
        self._delegate = delegate
        self.calls = delegate.calls
        self._failed = False

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if not self._failed:
            self._failed = True
            raise GraphReadError("exemplar", upstream_status=503)
        return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)

    def node_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        return self._delegate.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )

    def node_names_many(self, dcids, *, deadline: float | None = None):
        return self._delegate.node_names_many(dcids)


_ORPHAN_PATH = "/api/graph/documentation/provenance/Orphan"


class TestDocumentCache:
    """`GraphDeps.provenance_docs` serves repeat and concurrent requests for one source from one
    graph read per generation and catalogue build."""

    def test_a_repeat_request_makes_no_graph_read(self):
        graph = FakeGraphNodeReader(nodes={"provenance/Orphan": {"typeOf": ("Provenance",)}})

        with make_client(graph=graph) as client:
            first = client.get(_ORPHAN_PATH)
            reads = graph.calls["node_arcs_many"]
            second = client.get(_ORPHAN_PATH)

        assert first.status_code == second.status_code == 200, (first.text, second.text)
        assert second.json() == first.json()
        assert reads > 0
        assert graph.calls["node_arcs_many"] == reads

    def test_concurrent_requests_for_one_source_share_one_read(self):
        graph = _BlockingGraphNodeReader(
            FakeGraphNodeReader(nodes={"provenance/Orphan": {"typeOf": ("Provenance",)}})
        )
        client = make_client(graph=graph, node_pool_size=2)
        responses: list[object] = []

        with client:
            docs = client.app.state.graph_deps.provenance_docs
            first = threading.Thread(target=lambda: responses.append(client.get(_ORPHAN_PATH)))
            first.start()
            assert graph.entered.wait(timeout=5.0), "the first read never started"
            second = threading.Thread(target=lambda: responses.append(client.get(_ORPHAN_PATH)))
            second.start()
            # The second request has looked the document up once it counts a miss.
            wait_until(lambda: docs.stats()["misses"] == 2)
            graph.release.set()
            first.join(timeout=5.0)
            second.join(timeout=5.0)

        assert [response.status_code for response in responses] == [200, 200]
        assert len(graph.read_deadlines) == 1

    def test_a_new_generation_reads_again(self, monkeypatch: pytest.MonkeyPatch):
        generation = FakeGeneration(token="gen-a")
        monkeypatch.setattr(app_module, "acquire_shared_generation", lambda **_kwargs: generation)
        monkeypatch.setattr(app_module, "release_shared_generation", lambda _shared: None)
        graph = FakeGraphNodeReader(nodes={"provenance/Orphan": {"typeOf": ("Provenance",)}})

        with make_client(graph=graph) as client:
            assert client.get(_ORPHAN_PATH).status_code == 200
            reads = graph.calls["node_arcs_many"]
            assert client.get(_ORPHAN_PATH).status_code == 200
            assert graph.calls["node_arcs_many"] == reads
            generation.set_unknown()
            assert client.get(_ORPHAN_PATH).status_code == 200

        # `set_unknown` changes the stamp without clearing caches, so the key alone reads again.
        assert graph.calls["node_arcs_many"] == 2 * reads

    def test_a_failed_read_is_read_again_on_the_next_request(self):
        graph = _FailingOnceGraphNodeReader(
            FakeGraphNodeReader(nodes={"provenance/Orphan": {"typeOf": ("Provenance",)}})
        )

        with make_client(graph=graph) as client:
            failed = client.get(_ORPHAN_PATH)
            recovered = client.get(_ORPHAN_PATH)
            reads = graph.calls["node_arcs_many"]
            repeat = client.get(_ORPHAN_PATH)

        assert failed.status_code == 503, failed.text
        assert recovered.status_code == repeat.status_code == 200, (recovered.text, repeat.text)
        assert reads > 0
        assert graph.calls["node_arcs_many"] == reads

    def test_coverage_facts_follow_the_coverage_index_of_each_response(self):
        """The cached document carries no coverage facts, so a refreshed index shows on the next
        response without a graph read."""
        graph = _two_child_dataset()
        both = ("dc/base/Child1", "dc/base/Child2")
        coverage = _SwappingCoverage(
            _FakeIndex(provenance_currencies={}, provenance_shapes={}),
            _FakeIndex(
                provenance_currencies=dict.fromkeys(both, frozenset({"USD"})),
                provenance_shapes={},
            ),
        )

        with make_client(graph=graph, coverage=coverage) as client:
            before = client.get("/api/graph/documentation/dc/d/Dataset1").json()
            reads = graph.calls["node_arcs_many"]
            after = client.get("/api/graph/documentation/dc/d/Dataset1").json()

        assert before["scope_and_units"]["currencies"] == []
        assert "scope_and_units.currencies" in before["gaps"]
        assert after["scope_and_units"]["currencies"] == ["USD"]
        assert "scope_and_units.currencies" not in after["gaps"]
        assert graph.calls["node_arcs_many"] == reads


class TestLiveFixtures:
    def test_world_development_indicators_end_to_end(self):
        dataset_arcs = _wire_arcs(
            "dataset_world_bank_wdi.json",
            "dc/d/WorldBank_WorldDevelopmentIndicators",
            _NODE_PROPERTIES,
        )
        publisher_arcs = _wire_arcs("source_world_bank_name.json", "dc/s/WorldBank", ("name",))
        child_arcs = _wire_arcs(
            "node_world_development_indicators.json",
            "dc/base/WorldDevelopmentIndicators",
            (*_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES),
        )
        graph = FakeGraphNodeReader(
            nodes={
                "dc/d/WorldBank_WorldDevelopmentIndicators": dataset_arcs,
                "dc/s/WorldBank": publisher_arcs,
                "dc/base/WorldDevelopmentIndicators": child_arcs,
                "CreativeCommonsAttributionLicense": {
                    "name": ("Creative Commons Attribution License",)
                },
            },
            subjects={
                ("dc/d/WorldBank_WorldDevelopmentIndicators", "isPartOf"): SubjectPage(
                    dcids=("dc/base/WorldDevelopmentIndicators",), truncated=False
                )
            },
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/dc/d/WorldBank_WorldDevelopmentIndicators")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["details"]["name_of_source"] == "World Development Indicators"
        assert body["details"]["publisher"] == "World Bank"
        assert body["license"]["type"] == "Creative Commons Attribution License"
        assert body["license"]["update_cadence"] == "Monthly"
        assert (
            body["details"]["source_data_url"]
            == "http://api.worldbank.org/v2/en/country/AFG?downloadformat=csv"
        )
        assert body["license"]["url"] == "https://datacatalog.worldbank.org/search/dataset/0037712"

    def test_us_census_acs_end_to_end_over_the_real_61_child_fan_in(self):
        dataset_arcs = _wire_arcs(
            "dataset_us_census_acs.json",
            "dc/d/UsCensusBureau_AmericanCommunitySurveyAcs",
            (*_DATASET_PROPERTIES, "typeOf"),
        )
        child_dcids = _wire_subject_dcids(
            "dataset_us_census_acs_children.json",
            "dc/d/UsCensusBureau_AmericanCommunitySurveyAcs",
            "isPartOf",
        )
        assert len(child_dcids) == 61
        nodes = {
            "dc/d/UsCensusBureau_AmericanCommunitySurveyAcs": dataset_arcs,
            "dc/s/UsCensusBureau": _wire_arcs(
                "source_us_census_bureau_name.json", "dc/s/UsCensusBureau", ("name",)
            ),
            "dc/base/CensusACS1YearSurvey": {"licenseType": ("PublicDomain",)},
            "dc/base/CensusACS5YearSurvey": {"licenseType": ("PublicDomain",)},
        }
        graph = FakeGraphNodeReader(
            nodes=nodes,
            subjects={
                ("dc/d/UsCensusBureau_AmericanCommunitySurveyAcs", "isPartOf"): SubjectPage(
                    dcids=child_dcids, truncated=False
                )
            },
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get(
                "/api/graph/documentation/dc/d/UsCensusBureau_AmericanCommunitySurveyAcs"
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["details"]["name_of_source"] == "American Community Survey (ACS)"
        assert body["details"]["publisher"] == "U.S. Census Bureau"
        # Multi-child Dataset descriptions are null.
        assert body["details"]["description"] is None
        # The fixture sets `licenseType` on 2 of 61 children; the live graph
        # sets it on 57. `_agree` requires all children, so report the gap.
        assert body["license"]["type"] is None
        assert "license.type" in body["gaps"]

    def test_oecd_crs_orphan_provenance_serves_the_real_recorded_years_covered(self):
        arcs = _wire_arcs(
            "node_oecd_crs_provenance.json",
            "provenance/OECDCreditorReportingSystem",
            (*_NODE_PROPERTIES, *_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES),
        )
        graph = FakeGraphNodeReader(nodes={"provenance/OECDCreditorReportingSystem": arcs})
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/OECDCreditorReportingSystem")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["scope_and_units"]["years_covered"] == "1973-2024"
        assert body["scope_and_units"]["countries"] == []
        assert "scope_and_units.years_covered" not in body["gaps"]

    def test_oecd_crs_documents_serve_decoded_steps_and_report_the_untitled_changelog(self):
        """`node_oecd_crs_provenance_documents.json` is the dev graph's
        `/v2/node` response for every property this reader requests of a
        sole child, recorded with the transformations and changelog the
        pipeline writes MCF-escaped.
        """
        dcid = "provenance/OECDCreditorReportingSystem"
        arcs = _wire_arcs(
            "node_oecd_crs_provenance_documents.json",
            dcid,
            (*_NODE_PROPERTIES, *_ROLLUP_PROPERTIES, *_SOLE_CHILD_PROPERTIES),
        )
        graph = FakeGraphNodeReader(nodes={dcid: arcs})
        client = make_client(graph=graph)

        with client:
            resp = client.get(f"/api/graph/documentation/{dcid}")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert [step["transformation"] for step in body["transformation_log"]] == [
            "Download the source data",
            "Exclude donor core contributions to multilaterals",
            "Standardise providers and recipients",
            "Select dimensions",
            "Aggregate totals",
        ]
        # The recorded changelog has bare-date headings, which the template refuses.
        assert body["version_history"] == []
        assert body["document_errors"] == [
            {"member": "version_history", "line": 1, "reason": "missing_release_title"}
        ]
        assert body["details"]["citation"].startswith("OECD (2026), Creditor Reporting System")
        assert (
            body["details"]["source_data_url"]
            == "https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS.parquet"
        )
        assert body["license"]["url"] == "https://www.oecd.org/en/about/terms-conditions.html"
        assert "version_history" in body["gaps"]
        assert "transformation_log" not in body["gaps"]
        assert "details.citation" not in body["gaps"]
        # The stored values hold escape sequences. No served string does.
        assert "\\n" in arcs["dataTransformationDescription"][0]
        served = list(_strings(body))
        assert not any("\\n" in text or "\\u0022" in text for text in served)


class TestOrigin:
    """`details.source` and `details.data_commons` come from the request's origin catalogue."""

    def test_an_admitted_provenance_serves_its_source_and_data_commons(self, serve_catalogue):
        dcid = "dc/base/WorldDevelopmentIndicators"
        serve_catalogue(
            restrictive_catalogue(provenances=[provenance_row(dcid, BASE, source="dc/s/WorldBank")])
        )
        graph = FakeGraphNodeReader(nodes={dcid: {"typeOf": ("Provenance",)}})
        client = make_client(graph=graph)

        with client:
            resp = client.get(f"/api/graph/documentation/{dcid}")

        assert resp.status_code == 200, resp.text
        details = resp.json()["details"]
        assert details["source"] == {"ref": "dc/s/WorldBank", "label": "dc/s/WorldBank"}
        assert details["data_commons"] == {"id": "base", "label": "Data Commons by Google"}
        assert "details.source" not in resp.json()["gaps"]

    def test_a_un_agency_provenance_is_documented_as_its_own_source(self, serve_catalogue):
        dcid = "undata/p/SDG"
        row = replace(provenance_row(dcid, UN_SYSTEM, segment="sdg"), name="SDG Global Database")
        serve_catalogue(restrictive_catalogue(provenances=[row]))
        graph = FakeGraphNodeReader(
            nodes={dcid: {"typeOf": ("Provenance",), "name": ("Graph name",)}}
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get(f"/api/graph/documentation/{dcid}")

        assert resp.status_code == 200, resp.text
        details = resp.json()["details"]
        assert details["name_of_source"] == "SDG Global Database"
        assert details["source"] == {"ref": dcid, "label": dcid}
        assert details["data_commons"] == {"id": "un_system", "label": "UN System Data Commons"}

    def test_the_graph_name_serves_as_name_of_source(self):
        dcid = "dc/d/OpenDataForAfrica_SouthafricaCensus"
        graph = FakeGraphNodeReader(
            nodes={dcid: {"typeOf": ("Dataset",), "name": ("SouthAfrica Census",)}},
            subjects={
                (dcid, "isPartOf"): SubjectPage(
                    dcids=("dc/base/OpenDataForAfrica_SouthafricaCensus",), truncated=False
                )
            },
        )
        client = make_client(graph=graph)

        with client:
            doc_resp = client.get(f"/api/graph/documentation/{dcid}")

        assert doc_resp.status_code == 200, doc_resp.text
        assert doc_resp.json()["details"]["name_of_source"] == "SouthAfrica Census"


def _escape(markdown: str) -> str:
    """The stored form of a Markdown document: dfo-pipelines'
    `escape_markdown` without its outer quotes, which the graph drops.
    """
    return json.dumps(markdown, ensure_ascii=False)[1:-1].replace('\\"', "\\u0022")


class TestDocumentMembers:
    def test_parsed_documents_map_to_the_wire_shape(self):
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "dataTransformationDescription": (_escape("## Step one\n\nDo it.\n"),),
                    "dataTransformationChangelog": (
                        _escape(
                            "## 2026-09-22 Second release\n\n### Changed\n\n- Changed a.\n\n"
                            "### Removed\n\n- Removed b.\n\n"
                            "## 2026-08-20 First release\n\n### Added\n\n- Added c.\n"
                        ),
                    ),
                    "sourceMethodologyDescription": (_escape("How it is *compiled*.\n"),),
                    "dataNotes": (_escape("- Note one.\n- Note two.\n"),),
                    "licenseAttribution": ("Cite the source.",),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["transformation_log"] == [
            {"transformation": "Step one", "description": "Do it."}
        ]
        assert body["version_history"] == [
            {
                "date": "2026-09-22",
                "title": "Second release",
                "changes": [
                    {"kind": "changed", "note": "Changed a."},
                    {"kind": "removed", "note": "Removed b."},
                ],
            },
            {
                "date": "2026-08-20",
                "title": "First release",
                "changes": [{"kind": "added", "note": "Added c."}],
            },
        ]
        assert body["methodology"] == {"description": "How it is *compiled*."}
        assert body["data_notes"] == ["Note one.", "Note two."]
        assert body["details"]["citation"] == "Cite the source."
        assert body["document_errors"] == []
        for path in (
            "details.citation",
            "methodology.description",
            "data_notes",
            "transformation_log",
            "version_history",
        ):
            assert path not in body["gaps"]

    def test_conflicting_document_values_are_reported_on_the_wire(self):
        graph = FakeGraphNodeReader(
            nodes={
                "provenance/Orphan": {
                    "typeOf": ("Provenance",),
                    "dataNotes": (_escape("- One.\n"), _escape("- Two.\n")),
                }
            }
        )
        client = make_client(graph=graph)

        with client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data_notes"] == []
        assert "data_notes" in body["gaps"]
        assert body["document_errors"] == [
            {"member": "data_notes", "line": None, "reason": "conflicting_values"}
        ]


@dataclass(frozen=True)
class _FakeIndex:
    provenance_currencies: Mapping[str, frozenset[str]]
    provenance_shapes: Mapping[str, ProvenanceShape]


class _BrokenShapesIndex:
    """An index whose currencies read and whose shapes raise."""

    provenance_currencies: Mapping[str, frozenset[str]] = {
        "dc/base/Child1": frozenset({"USD"}),
        "dc/base/Child2": frozenset({"USD"}),
    }

    @property
    def provenance_shapes(self) -> Mapping[str, ProvenanceShape]:
        raise KeyError("P1Y")


class _BrokenShapesCoverage:
    def current(self) -> _BrokenShapesIndex:
        return _BrokenShapesIndex()


class _FakeCoverage:
    """Stands in for `CoverageService`, whose `current()` is all the door reads."""

    def __init__(
        self,
        units: Mapping[str, frozenset[str]] | None = None,
        *,
        shapes: Mapping[str, ProvenanceShape] | None = None,
        unreadable: bool = False,
    ) -> None:
        self._units = units or {}
        self._shapes = shapes or {}
        self._unreadable = unreadable or (units is None and shapes is None)

    def current(self) -> _FakeIndex:
        if self._unreadable:
            raise RuntimeError("coverage artifact unreadable")
        return _FakeIndex(provenance_currencies=self._units, provenance_shapes=self._shapes)


def _two_child_dataset() -> FakeGraphNodeReader:
    return FakeGraphNodeReader(
        nodes={"dc/d/Dataset1": {"typeOf": ("Dataset",)}},
        subjects={
            ("dc/d/Dataset1", "isPartOf"): SubjectPage(
                dcids=("dc/base/Child1", "dc/base/Child2"), truncated=False
            )
        },
    )


def _currency_response(coverage: object | None) -> dict:
    with make_client(graph=_two_child_dataset(), coverage=coverage) as client:
        resp = client.get("/api/graph/documentation/dc/d/Dataset1")
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestCurrency:
    def test_one_currency_across_every_child_serves_a_one_code_list(self):
        body = _currency_response(
            _FakeCoverage(
                {
                    "dc/base/Child1": frozenset({"USD"}),
                    "dc/base/Child2": frozenset({"USD"}),
                    "dc/base/Unrelated": frozenset({"EUR"}),
                }
            )
        )

        assert body["scope_and_units"]["currencies"] == ["USD"]
        assert "scope_and_units.currencies" not in body["gaps"]

    def test_a_child_without_currency_facets_adds_none_to_the_list(self):
        body = _currency_response(_FakeCoverage({"dc/base/Child1": frozenset({"GBP"})}))

        assert body["scope_and_units"]["currencies"] == ["GBP"]
        assert "scope_and_units.currencies" not in body["gaps"]

    def test_several_currencies_serve_every_code_sorted(self):
        body = _currency_response(
            _FakeCoverage(
                {
                    "dc/base/Child1": frozenset({"USD", "EUR"}),
                    "dc/base/Child2": frozenset({"CAD", "USD"}),
                }
            )
        )

        assert body["scope_and_units"]["currencies"] == ["CAD", "EUR", "USD"]
        assert "scope_and_units.currencies" not in body["gaps"]

    def test_no_coverage_rows_serve_an_empty_list_and_gap_the_path(self):
        body = _currency_response(_FakeCoverage({}))

        assert body["scope_and_units"]["currencies"] == []
        assert "scope_and_units.currencies" in body["gaps"]

    def test_unset_coverage_serves_empty_and_logs_nothing_per_request(
        self, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.DEBUG, logger=sources_door.__name__):
            body = _currency_response(None)

        assert body["scope_and_units"]["currencies"] == []
        assert "scope_and_units.currencies" in body["gaps"]
        assert not [r for r in caplog.records if r.name == sources_door.__name__]

    def test_a_failing_read_serves_empty_and_warns_with_the_provenances(
        self, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.INFO, logger=sources_door.__name__):
            body = _currency_response(_FakeCoverage(None))

        assert body["scope_and_units"]["currencies"] == []
        assert body["scope_and_units"]["granularity"] is None
        assert {"scope_and_units.currencies", "scope_and_units.granularity"} <= set(body["gaps"])
        warnings = [
            record
            for record in caplog.records
            if record.name == sources_door.__name__ and record.levelno == logging.WARNING
        ]
        # One unreadable index empties both facts under one WARNING.
        assert len(warnings) == 1
        assert warnings[0].args == (["dc/base/Child1", "dc/base/Child2"],)
        assert warnings[0].exc_info is not None

    def test_a_failing_granularity_leaves_the_currencies_served(
        self, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING, logger=provenance_doc.__name__):
            body = _currency_response(_BrokenShapesCoverage())

        assert body["scope_and_units"]["currencies"] == ["USD"]
        assert body["scope_and_units"]["granularity"] is None
        assert "scope_and_units.granularity" in body["gaps"]
        assert [
            record.args[0] for record in caplog.records if record.name == provenance_doc.__name__
        ] == ["granularity"]

    def test_the_recorded_coverage_slice_serves_crs_in_usd(self):
        """The committed slice holds one CRS facet, in `CurrencyUSD`, so the
        real service and inversion resolve the orphan CRS Provenance to USD.
        """
        from series_coverage.shared import CoverageService
        from tests.series_coverage.support import index_for

        slice_uri = f"file://{Path(__file__).parent.parent / 'fixtures' / 'coverage_slice'}"
        coverage = CoverageService(
            artifact_uri=slice_uri, index=index_for(slice_uri), recheck_seconds=0
        )
        crs = "provenance/OECDCreditorReportingSystem"
        graph = FakeGraphNodeReader(nodes={crs: {"typeOf": ("Provenance",)}})

        with make_client(graph=graph, coverage=coverage) as client:
            resp = client.get(f"/api/graph/documentation/{crs}")

        assert resp.status_code == 200, resp.text
        assert resp.json()["scope_and_units"]["currencies"] == ["USD"]
        assert "scope_and_units.currencies" not in resp.json()["gaps"]
        # The slice's facets carry no `observation_period`, so granularity is
        # unknown.
        assert resp.json()["scope_and_units"]["granularity"] is None
        assert "scope_and_units.granularity" in resp.json()["gaps"]


def _documentation_response(coverage: object | None, headers: dict[str, str] | None = None):
    with make_client(graph=_two_child_dataset(), coverage=coverage) as client:
        return client.get("/api/graph/documentation/dc/d/Dataset1", headers=headers)


class TestCachePolicy:
    def test_permanent_gaps_serve_public_with_an_etag(self):
        resp = _documentation_response(_FakeCoverage({}))

        assert resp.status_code == 200, resp.text
        # An empty coverage index is a permanent answer, though it gaps the path.
        assert "scope_and_units.currencies" in resp.json()["gaps"]
        assert resp.headers["cache-control"] == "public, max-age=300"
        assert resp.headers["etag"]

    def test_a_matching_etag_answers_304(self):
        etag = _documentation_response(_FakeCoverage({})).headers["etag"]

        resp = _documentation_response(_FakeCoverage({}), {"if-none-match": etag})

        assert resp.status_code == 304

    @pytest.mark.parametrize(
        ("coverage", "load_failed"),
        [
            pytest.param(None, True, id="failed-to-load"),
            pytest.param(_FakeCoverage(None), False, id="unreadable"),
        ],
    )
    def test_a_coverage_gap_from_a_failed_or_unreadable_index_serves_no_store(
        self, coverage, load_failed
    ):
        with make_client(graph=_two_child_dataset(), coverage=coverage) as client:
            deps = client.app.state.graph_deps
            client.app.state.graph_deps = replace(deps, coverage_load_failed=load_failed)
            resp = client.get("/api/graph/documentation/dc/d/Dataset1")

        assert resp.status_code == 200, resp.text
        assert resp.headers["cache-control"] == "no-store"
        assert "etag" not in resp.headers

    def test_unconfigured_coverage_is_a_permanent_gap_and_serves_public(self):
        resp = _documentation_response(None)

        assert resp.status_code == 200, resp.text
        assert "scope_and_units.currencies" in resp.json()["gaps"]
        assert resp.headers["cache-control"] == "public, max-age=300"
        assert resp.headers["etag"]

    def test_the_policy_follows_the_coverage_state_across_a_source_document_cache_hit(self):
        with make_client(graph=_two_child_dataset()) as client:
            deps = client.app.state.graph_deps
            client.app.state.graph_deps = replace(deps, coverage_load_failed=True)
            first = client.get("/api/graph/documentation/dc/d/Dataset1")
            client.app.state.graph_deps = replace(
                deps, coverage=_FakeCoverage({}), coverage_load_failed=False
            )
            second = client.get("/api/graph/documentation/dc/d/Dataset1")

        assert first.headers["cache-control"] == "no-store"
        assert second.headers["cache-control"] == "public, max-age=300"
        assert second.headers["etag"]


_BOTH_AXES = frozenset({"origin", "destination"})


def _shape(*periods: str | None, axes: frozenset[str | None] = _BOTH_AXES) -> ProvenanceShape:
    return ProvenanceShape(observation_periods=frozenset(periods), role_axes=frozenset({axes}))


class TestGranularity:
    def test_one_period_and_both_axes_serve_the_phrase(self):
        body = _currency_response(
            _FakeCoverage(shapes={"dc/base/Child1": _shape("P1Y"), "dc/base/Child2": _shape("P1Y")})
        )

        assert body["scope_and_units"]["granularity"] == "Annual, by provider and recipient"
        assert "scope_and_units.granularity" not in body["gaps"]

    def test_one_role_and_two_role_sets_serve_the_union_phrase(self):
        one_and_two_roles = ProvenanceShape(
            observation_periods=frozenset({"P1Y"}),
            role_axes=frozenset({frozenset({"destination"}), _BOTH_AXES}),
        )
        body = _currency_response(
            _FakeCoverage(
                shapes={"dc/base/Child1": one_and_two_roles, "dc/base/Child2": _shape("P1Y")}
            )
        )

        assert body["scope_and_units"]["granularity"] == "Annual, by provider and recipient"
        assert "scope_and_units.granularity" not in body["gaps"]

    def test_one_role_only_serves_null_and_gaps_the_path(self):
        recipient_only = _shape("P1Y", axes=frozenset({"destination"}))
        body = _currency_response(
            _FakeCoverage(
                shapes={"dc/base/Child1": recipient_only, "dc/base/Child2": recipient_only}
            )
        )

        assert body["scope_and_units"]["granularity"] is None
        assert "scope_and_units.granularity" in body["gaps"]

    def test_disagreeing_periods_serve_null_even_with_the_union_of_axes(self):
        one_and_two_roles = ProvenanceShape(
            observation_periods=frozenset({"P1Y"}),
            role_axes=frozenset({frozenset({"destination"}), _BOTH_AXES}),
        )
        body = _currency_response(
            _FakeCoverage(
                shapes={"dc/base/Child1": one_and_two_roles, "dc/base/Child2": _shape("P1M")}
            )
        )

        assert body["scope_and_units"]["granularity"] is None
        assert "scope_and_units.granularity" in body["gaps"]

    def test_a_child_without_coverage_rows_serves_null_and_gaps_the_path(self):
        body = _currency_response(_FakeCoverage(shapes={"dc/base/Child2": _shape("P1M")}))

        assert body["scope_and_units"]["granularity"] is None
        assert "scope_and_units.granularity" in body["gaps"]

    @pytest.mark.parametrize(
        "shapes",
        [
            pytest.param(
                {"dc/base/Child1": _shape("P1Y"), "dc/base/Child2": _shape("P1M")},
                id="two-periods-across-children",
            ),
            pytest.param(
                {"dc/base/Child1": _shape("P1Y", "P1M"), "dc/base/Child2": _shape("P1Y")},
                id="two-periods-in-one",
            ),
            pytest.param(
                {"dc/base/Child1": _shape(None), "dc/base/Child2": _shape(None)}, id="no-period"
            ),
            pytest.param(
                {"dc/base/Child1": _shape("P1Y", None), "dc/base/Child2": _shape("P1Y")},
                id="period-and-no-period",
            ),
            pytest.param(
                {"dc/base/Child1": _shape("P7Y"), "dc/base/Child2": _shape("P7Y")},
                id="unknown-period-token",
            ),
            pytest.param(
                {
                    "dc/base/Child1": _shape("P1Y", axes=frozenset({"origin", None})),
                    "dc/base/Child2": _shape("P1Y", axes=frozenset({"origin", None})),
                },
                id="role-off-the-flow-axes",
            ),
            pytest.param(
                {
                    "dc/base/Child1": ProvenanceShape(
                        observation_periods=frozenset({"P1Y"}),
                        role_axes=frozenset({_BOTH_AXES, frozenset({"origin", None})}),
                    ),
                    "dc/base/Child2": _shape("P1Y"),
                },
                id="two-axis-sets",
            ),
            pytest.param({}, id="no-coverage-rows"),
        ],
    )
    def test_anything_ambiguous_or_absent_serves_null_and_gaps_the_path(
        self, shapes: Mapping[str, ProvenanceShape]
    ):
        body = _currency_response(_FakeCoverage(shapes=shapes))

        assert body["scope_and_units"]["granularity"] is None
        assert "scope_and_units.granularity" in body["gaps"]

    @pytest.mark.parametrize(
        "coverage", [None, _FakeCoverage(None)], ids=["not-configured", "read-fails"]
    )
    def test_unavailable_coverage_serves_null_and_succeeds(self, coverage: object | None):
        body = _currency_response(coverage)

        assert body["scope_and_units"]["granularity"] is None
        assert "scope_and_units.granularity" in body["gaps"]

    def test_a_stored_data_granularity_arc_is_not_served(self):
        graph = FakeGraphNodeReader(
            nodes={"provenance/Orphan": {"typeOf": ("Provenance",), "dataGranularity": ("Annual",)}}
        )

        with make_client(graph=graph) as client:
            resp = client.get("/api/graph/documentation/provenance/Orphan")

        assert resp.status_code == 200, resp.text
        assert resp.json()["scope_and_units"]["granularity"] is None


class _SwappingCoverage:
    """A service whose `current()` returns a different build on every call,
    as a background refresh between two reads would."""

    def __init__(self, *builds: _FakeIndex) -> None:
        self._builds = iter(builds)
        self.calls = 0

    def current(self) -> _FakeIndex:
        self.calls += 1
        return next(self._builds)


class TestCoverageSnapshot:
    def test_both_facts_come_from_one_index_snapshot(self):
        both = ("dc/base/Child1", "dc/base/Child2")
        first = _FakeIndex(
            provenance_currencies=dict.fromkeys(both, frozenset({"USD"})),
            provenance_shapes=dict.fromkeys(both, _shape("P1Y")),
        )
        second = _FakeIndex(provenance_currencies={}, provenance_shapes={})
        coverage = _SwappingCoverage(first, second)

        body = _currency_response(coverage)

        assert coverage.calls == 1
        assert body["scope_and_units"]["currencies"] == ["USD"]
        assert body["scope_and_units"]["granularity"] == "Annual, by provider and recipient"
