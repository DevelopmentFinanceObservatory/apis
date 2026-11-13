"""The ontology door: one cached build per generation behind the document, terms and check
routes, their waits, their refusals and the frozen bodies."""

from __future__ import annotations

import dataclasses
import json
import threading

import httpx
import pytest

from dc_graph import ArcSubject, ArcTarget
from graph_api import app as app_module
from graph_api.doors import ontology as ontology_door
from graph_api.ontology_document import OntologyListingTruncatedError
from graph_api.ontology_models import CHECK_MAX_ROWS
from graph_generation import FakeGeneration
from materialise.graph import GraphDeadlineExceededError, GraphReadError

from .conftest import FakeGraphNodeReader, make_client, wait_until
from .ontology_fixture import STAMP, index_of

_DFO = "provenance/DFOOntology"
_DOCUMENT = "/api/graph/ontology"
_TERMS = "/api/graph/ontology/terms"
_CHECK = "/api/graph/ontology/check"
_IDENTITY = {"Accept-Encoding": "identity"}

_ARC_SUBJECTS = {
    ("Class", "typeOf"): (
        ArcSubject(
            dcid="FinancialFlow", names=("Financial flow",), types=("Class",), provenances=(_DFO,)
        ),
        ArcSubject(
            dcid="Thing", names=("Thing",), types=("Class",), provenances=("dc/base/Schema",)
        ),
    ),
    ("Property", "typeOf"): (
        ArcSubject(
            dcid="crsPurpose", names=("CRS purpose",), types=("Property",), provenances=(_DFO,)
        ),
    ),
}
_NODES = {
    "FinancialFlow": {"typeOf": ("Class",), "subClassOf": ("Thing",), "name": ("Financial flow",)},
    "Thing": {"typeOf": ("Class",)},
    "crsPurpose": {"typeOf": ("Property",), "domainIncludes": ("FinancialFlow",)},
    "CRSPurposeCode120": {"typeOf": ("Thing",)},
}
_ARC_TARGETS = {"CRSPurposeCode120": {"typeOf": (ArcTarget(value="Thing", provenances=(_DFO,)),)}}


def _graph() -> FakeGraphNodeReader:
    return FakeGraphNodeReader(nodes=_NODES, arc_subjects=_ARC_SUBJECTS, arc_targets=_ARC_TARGETS)


def _builds(graph: FakeGraphNodeReader) -> int:
    """Every build opens with one `Class` and one `Property` listing."""
    return graph.calls["node_arc_subjects"] // 2


def _reads(graph: FakeGraphNodeReader) -> int:
    return sum(graph.calls.values())


class _BlockingReader(FakeGraphNodeReader):
    """Blocks the build's first read until `release` is set."""

    def __init__(self) -> None:
        super().__init__(nodes=_NODES, arc_subjects=_ARC_SUBJECTS, arc_targets=_ARC_TARGETS)
        self.entered = threading.Event()
        self.release = threading.Event()

    def node_arc_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        self.entered.set()
        self.release.wait(timeout=5.0)
        return super().node_arc_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )


class _RaisingReader(FakeGraphNodeReader):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def node_arc_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        raise self._error


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch) -> FakeGeneration:
    fake = FakeGeneration(token="gen-a")
    monkeypatch.setattr(app_module, "acquire_shared_generation", lambda **_kwargs: fake)
    monkeypatch.setattr(app_module, "release_shared_generation", lambda _shared: None)
    return fake


def _ontology_stats(client) -> dict[str, int]:
    return dict(client.app.state.graph_deps.ontology.stats())


class TestCache:
    def test_a_repeat_request_on_either_route_makes_no_graph_read(self, generation):
        graph = _graph()

        with make_client(graph=graph) as client:
            document = client.get(_DOCUMENT)
            reads = _reads(graph)
            again = client.get(_DOCUMENT)
            terms = client.get(_TERMS)

        assert [document.status_code, again.status_code, terms.status_code] == [200, 200, 200]
        assert reads > 0
        assert _reads(graph) == reads
        assert document.json()["generation"] == terms.json()["generation"] == "gen-a"

    def test_concurrent_first_requests_share_one_build(self, generation):
        graph = _BlockingReader()
        client = make_client(graph=graph, node_pool_size=2)
        responses: list[httpx.Response] = []

        with client:
            first = threading.Thread(target=lambda: responses.append(client.get(_DOCUMENT)))
            first.start()
            assert graph.entered.wait(timeout=5.0), "the build never started"
            second = threading.Thread(target=lambda: responses.append(client.get(_TERMS)))
            second.start()
            # The second request has looked the entry up once it counts a miss.
            wait_until(lambda: _ontology_stats(client)["misses"] == 2)
            graph.release.set()
            first.join(timeout=5.0)
            second.join(timeout=5.0)

        assert sorted(response.status_code for response in responses) == [200, 200]
        assert _builds(graph) == 1

    def test_a_generation_change_clears_the_entry_and_rebuilds(self, generation):
        graph = _graph()

        with make_client(graph=graph) as client:
            registered = generation.registered_names()
            before = client.get(_DOCUMENT).json()["generation"]
            generation.advance("gen-b")
            assert _ontology_stats(client)["entries"] == 0
            after = client.get(_TERMS).json()["generation"]

        assert "graph_api.ontology" in registered
        assert (before, after) == ("gen-a", "gen-b")
        assert _builds(graph) == 2

    def test_a_code_index_change_rebuilds(self, generation, monkeypatch: pytest.MonkeyPatch):
        graph = _graph()
        indexes = [
            index_of(stamp=dataclasses.replace(STAMP, content_hash=content_hash))
            for content_hash in ("index-a", "index-b")
        ]
        bound = [indexes[0]]
        monkeypatch.setattr(ontology_door, "get_index", lambda: bound[0])

        with make_client(graph=graph) as client:
            first = client.get(_DOCUMENT)
            bound[0] = indexes[1]
            second = client.get(_DOCUMENT)

        assert [first.status_code, second.status_code] == [200, 200]
        assert _builds(graph) == 2


class TestWaits:
    def test_a_build_slower_than_the_request_answers_503_and_lands_for_the_next(
        self, generation, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(ontology_door, "GRAPH_NODE_TIMEOUT_SECONDS", 0.2)
        graph = _BlockingReader()

        with make_client(graph=graph) as client:
            pending = client.get(_DOCUMENT)
            graph.release.set()
            wait_until(lambda: _ontology_stats(client)["entries"] == 1)
            landed = client.get(_DOCUMENT)

        assert pending.status_code == 503, pending.text
        assert pending.json()["code"] == "service_unavailable"
        assert pending.json()["retryable"] is True
        assert pending.headers["Retry-After"] == "5"
        assert landed.status_code == 200, landed.text
        assert _builds(graph) == 1

    def test_a_build_that_outlives_its_flight_answers_504_and_stores_nothing(
        self, generation, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(ontology_door, "ONTOLOGY_FLIGHT_TIMEOUT_SECONDS", 0.2)
        graph = _BlockingReader()

        with make_client(graph=graph) as client:
            expired = client.get(_DOCUMENT)
            graph.release.set()
            wait_until(lambda: client.app.state.graph_deps.dcp_executor._in_flight == 0)
            entries = _ontology_stats(client)["entries"]

        assert expired.status_code == 504, expired.text
        assert expired.json()["code"] == "timeout"
        assert entries == 0

    def test_a_saturated_pool_answers_503_and_the_next_request_builds(self, generation):
        graph = _graph()
        hold = threading.Event()

        with make_client(graph=graph, node_pool_size=1) as client:
            executor = client.app.state.graph_deps.dcp_executor
            executor.submit(hold.wait, 5.0)
            refused = client.get(_DOCUMENT)
            hold.set()
            wait_until(lambda: executor._in_flight == 0)
            recovered = client.get(_DOCUMENT)

        assert refused.status_code == 503, refused.text
        assert refused.json()["code"] == "service_unavailable"
        assert refused.json()["retryable"] is True
        assert "Retry-After" in refused.headers
        assert recovered.status_code == 200, recovered.text
        assert _builds(graph) == 1


@pytest.mark.parametrize(
    ("error", "status", "code", "retryable"),
    [
        (OntologyListingTruncatedError("Class listing capped"), 503, "service_unavailable", False),
        (GraphDeadlineExceededError("deadline"), 504, "timeout", True),
        (GraphReadError("bad request", upstream_status=400), 502, "upstream_error", True),
        (GraphReadError("unavailable", upstream_status=503), 503, "service_unavailable", True),
    ],
)
def test_a_failed_read_answers_its_problem_and_is_not_kept(
    generation, error: Exception, status: int, code: str, retryable: bool
):
    graph = _RaisingReader(error)

    with make_client(graph=graph) as client:
        response = client.get(_TERMS)
        entries = _ontology_stats(client)["entries"]

    assert response.status_code == status, response.text
    assert response.json()["code"] == code
    assert response.json()["retryable"] is retryable
    assert entries == 0


class TestFrozenBody:
    def test_an_etag_match_answers_304(self, generation):
        with make_client(graph=_graph()) as client:
            for path in (_DOCUMENT, _TERMS):
                first = client.get(path, headers=_IDENTITY)
                etag = first.headers["ETag"]
                again = client.get(path, headers={**_IDENTITY, "If-None-Match": etag})
                gzipped = client.get(path, headers={"Accept-Encoding": "gzip"})

                assert first.status_code == 200, first.text
                assert first.headers["Cache-Control"].startswith("public, max-age=")
                assert again.status_code == 304
                assert again.headers["ETag"] == etag
                assert gzipped.headers["Content-Encoding"] == "gzip"
                assert gzipped.json() == first.json()

    def test_two_builds_of_one_graph_give_the_same_sorted_bytes(self, generation):
        bodies: list[tuple[bytes, bytes, str]] = []
        for _ in range(2):
            with make_client(graph=_graph()) as client:
                document = client.get(_DOCUMENT, headers=_IDENTITY)
                terms = client.get(_TERMS, headers=_IDENTITY)
            bodies.append((document.content, terms.content, document.headers["ETag"]))

        assert bodies[0] == bodies[1]

        def keys_sorted(pairs: list[tuple[str, object]]) -> dict[str, object]:
            keys = [key for key, _ in pairs]
            assert keys == sorted(keys)
            return dict(pairs)

        for body in bodies[0][:2]:
            json.loads(body, object_pairs_hook=keys_sorted)


def _check_row(value: str) -> dict[str, object]:
    """A row the document answers except for its constraint value, which the check reads."""
    return {
        "ref": value,
        "population_type": "FinancialFlow",
        "measured_property": "crsPurpose",
        "constraint_properties": {"crsPurpose": value},
    }


class _FailingCheckReader(FakeGraphNodeReader):
    """Raises `error` on a read naming `BrokenCode`, which only the check reads."""

    def __init__(self, error: Exception) -> None:
        super().__init__(nodes=_NODES, arc_subjects=_ARC_SUBJECTS, arc_targets=_ARC_TARGETS)
        self._error = error

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if "BrokenCode" in dcids:
            raise self._error
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


class TestCheck:
    def test_a_faulty_row_answers_its_finding_with_the_generation(self, generation):
        rows = [_check_row("CRSPurposeCode120"), _check_row("NoSuchCode")]

        with make_client(graph=_graph()) as client:
            response = client.post(_CHECK, json={"rows": rows})

        assert response.status_code == 200, response.text
        assert response.json() == {
            "generation": "gen-a",
            "row_count": 2,
            "findings": [
                {
                    "ref": "NoSuchCode",
                    "field": "constraint_properties.crsPurpose",
                    "value": "NoSuchCode",
                    "code": "node_not_held",
                    "severity": "error",
                }
            ],
        }

    def test_clean_rows_answer_no_findings(self, generation):
        with make_client(graph=_graph()) as client:
            response = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})

        assert response.status_code == 200, response.text
        assert response.json()["findings"] == []
        assert response.json()["row_count"] == 1

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(
                {"rows": [_check_row("CRSPurposeCode120")] * (CHECK_MAX_ROWS + 1)}, id="too-many"
            ),
            pytest.param({"rows": []}, id="empty"),
            pytest.param(
                {"rows": [{**_check_row("CRSPurposeCode120"), "measured_property": "bad code"}]},
                id="malformed-dcid",
            ),
        ],
    )
    def test_a_body_outside_the_request_model_is_refused_before_any_read(self, generation, body):
        graph = _graph()

        with make_client(graph=graph) as client:
            response = client.post(_CHECK, json=body)

        assert response.status_code == 400, response.text
        assert response.json()["code"] == "request_invalid"
        assert _reads(graph) == 0

    def test_the_check_and_a_get_share_one_build(self, generation):
        graph = _graph()

        with make_client(graph=graph) as client:
            checked = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})
            document = client.get(_DOCUMENT)

        assert [checked.status_code, document.status_code] == [200, 200]
        assert _builds(graph) == 1

    def test_a_build_slower_than_the_request_answers_503_with_retry_after(
        self, generation, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(ontology_door, "GRAPH_NODE_TIMEOUT_SECONDS", 0.2)
        graph = _BlockingReader()

        with make_client(graph=graph) as client:
            pending = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})
            graph.release.set()

        assert pending.status_code == 503, pending.text
        assert pending.json()["code"] == "service_unavailable"
        assert pending.headers["Retry-After"] == "5"
        assert "findings" not in pending.json()

    @pytest.mark.parametrize(
        ("error", "status", "code"),
        [
            (GraphReadError("unavailable", upstream_status=503), 503, "service_unavailable"),
            (GraphReadError("bad request", upstream_status=400), 502, "upstream_error"),
            (GraphDeadlineExceededError("deadline"), 504, "timeout"),
        ],
    )
    def test_a_failed_check_read_answers_its_problem_and_no_findings(
        self, generation, error: Exception, status: int, code: str
    ):
        with make_client(graph=_FailingCheckReader(error)) as client:
            response = client.post(_CHECK, json={"rows": [_check_row("BrokenCode")]})

        assert response.status_code == status, response.text
        assert response.json()["code"] == code
        assert "findings" not in response.json()

    def test_the_check_read_is_one_executor_job(self, generation):
        graph = _graph()
        hold = threading.Event()

        with make_client(graph=graph, node_pool_size=1) as client:
            executor = client.app.state.graph_deps.dcp_executor
            built = client.get(_DOCUMENT)
            wait_until(lambda: executor._in_flight == 0)
            executor.submit(hold.wait, 5.0)
            refused = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})
            hold.set()
            wait_until(lambda: executor._in_flight == 0)
            checked = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})

        assert built.status_code == 200, built.text
        assert refused.status_code == 503, refused.text
        assert refused.json()["code"] == "service_unavailable"
        assert "Retry-After" in refused.headers
        assert checked.status_code == 200, checked.text
