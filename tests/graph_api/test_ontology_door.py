"""The ontology door: one cached build per generation behind the document, terms and check
routes, their waits, their refusals and the frozen bodies."""

from __future__ import annotations

import dataclasses
import json
import threading

import httpx
import pytest

from dc_graph import ArcSubject
from graph_api import app as app_module
from graph_api.config import GRAPH_ONTOLOGY_CHECK_CONCURRENCY
from graph_api.doors import ontology as ontology_door
from graph_api.ontology_check import OntologyLookup
from graph_api.ontology_document import (
    OntologyListingTruncatedError,
    OntologyNoInstanceDeclarationError,
)
from graph_api.ontology_models import (
    CHECK_MAX_CONSTRAINT_PROPERTIES,
    CHECK_MAX_FINDINGS,
    CHECK_MAX_OBSERVATION_PROPERTIES,
    CHECK_MAX_READ_IDS,
    CHECK_MAX_ROWS,
)
from graph_api.routing import DCID_MAX_LENGTH
from graph_generation import FakeGeneration
from materialise.graph import GraphDeadlineExceededError, GraphReadError
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from .conftest import FakeGraphNodeReader, make_client, wait_until
from .ontology_fixture import STAMP, index_of, index_row, node_targets

_DFO = "provenance/DFOOntology"
_DOCUMENT = "/api/graph/ontology"
_TERMS = "/api/graph/ontology/terms"
_CHECK = "/api/graph/ontology/check"
_IDENTITY = {"Accept-Encoding": "identity"}

_ARC_SUBJECTS = {
    ("Class", "typeOf"): (
        ArcSubject(dcid="FinancialFlow", names=("Financial flow",), provenances=(_DFO,)),
        ArcSubject(dcid="Thing", names=("Thing",), provenances=("dc/base/Schema",)),
    ),
    ("Property", "typeOf"): (
        ArcSubject(dcid="crsPurpose", names=("CRS purpose",), provenances=(_DFO,)),
    ),
}
_NODES = {
    "FinancialFlow": {"typeOf": ("Class",), "subClassOf": ("Thing",), "name": ("Financial flow",)},
    "Thing": {"typeOf": ("Class",)},
    "crsPurpose": {"typeOf": ("Property",), "domainIncludes": ("FinancialFlow",)},
    "CRSPurposeCode120": {"typeOf": ("Thing",)},
}
_ARC_TARGETS = node_targets(_NODES)


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
    """Raises `error` on every build's first read, counting the builds in `raised`."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error
        self.raised = 0

    def node_arc_subjects(self, dcid, *, property, filters=None, max_results, deadline=None):
        self.raised += 1
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

    def test_an_unknown_generation_request_keeps_the_known_entry(self, generation):
        graph = _graph()

        with make_client(graph=graph) as client:
            known = client.get(_DOCUMENT)
            generation.set_unknown()
            unknown = client.get(_DOCUMENT)
            generation.advance("gen-a")
            again = client.get(_DOCUMENT)

        assert [known.status_code, unknown.status_code, again.status_code] == [200, 200, 200]
        assert (known.json()["generation"], again.json()["generation"]) == ("gen-a", "gen-a")
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
    ("error", "status", "code", "retryable", "builds"),
    [
        pytest.param(
            GraphDeadlineExceededError("deadline"), 504, "timeout", True, 2, id="deadline"
        ),
        pytest.param(
            GraphReadError("bad request", upstream_status=400),
            502,
            "upstream_error",
            True,
            2,
            id="upstream-4xx",
        ),
        pytest.param(
            GraphReadError("unavailable", upstream_status=503),
            503,
            "service_unavailable",
            True,
            2,
            id="read-error",
        ),
    ],
)
def test_a_failed_build_answers_its_problem_and_is_not_kept(
    generation, error: Exception, status: int, code: str, retryable: bool, builds: int
):
    graph = _RaisingReader(error)

    with make_client(graph=graph) as client:
        responses = [client.get(_TERMS), client.get(_DOCUMENT)]

    for response in responses:
        assert response.status_code == status, response.text
        assert response.json()["code"] == code
        assert response.json()["retryable"] is retryable
    assert graph.raised == builds


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OntologyListingTruncatedError("Class listing capped"), id="truncated"),
        pytest.param(OntologyNoInstanceDeclarationError(3), id="no-instance-declaration"),
    ],
)
@pytest.mark.parametrize(
    ("known", "retryable", "builds"),
    [pytest.param(True, False, 1, id="known-kept"), pytest.param(False, True, 2, id="unknown")],
)
def test_a_refusal_is_kept_only_under_a_known_generation(
    generation, error: Exception, known: bool, retryable: bool, builds: int
):
    graph = _RaisingReader(error)
    if not known:
        generation.set_unknown()

    with make_client(graph=graph) as client:
        responses = [client.get(_TERMS), client.get(_DOCUMENT)]

    for response in responses:
        assert response.status_code == 503, response.text
        assert response.json()["code"] == "service_unavailable"
        assert response.json()["retryable"] is retryable
    assert graph.raised == builds


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

    def test_two_builds_of_one_graph_give_the_same_bytes(self, generation):
        bodies: list[tuple[bytes, bytes, str]] = []
        for _ in range(2):
            with make_client(graph=_graph()) as client:
                document = client.get(_DOCUMENT, headers=_IDENTITY)
                terms = client.get(_TERMS, headers=_IDENTITY)
            bodies.append((document.content, terms.content, document.headers["ETag"]))

        assert bodies[0] == bodies[1]


def _check_row(value: str) -> dict[str, object]:
    """A row the document answers except for its constraint value, which the check reads."""
    return {
        "ref": value,
        "population_type": "FinancialFlow",
        "measured_property": "crsPurpose",
        "constraint_properties": {"crsPurpose": value},
    }


def _nine_findings_row(ref: str) -> dict[str, object]:
    """A row with nine findings, each decided by the document with no read."""
    return {
        "ref": ref,
        "population_type": "crsPurpose",
        "measured_property": "FinancialFlow",
        "constraint_properties": {"crsPurpose": None, "FinancialFlow": "x", "Thing": "x"},
        "observation_properties": ["FinancialFlow"] * CHECK_MAX_OBSERVATION_PROPERTIES,
    }


class _FailingCheckReader(FakeGraphNodeReader):
    """Raises `error` on a read naming `BrokenCode`, which only the check reads."""

    def __init__(self, error: Exception) -> None:
        super().__init__(nodes=_NODES, arc_subjects=_ARC_SUBJECTS, arc_targets=_ARC_TARGETS)
        self._error = error

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        if "BrokenCode" in dcids:
            raise self._error
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


class _BlockingCheckReader(FakeGraphNodeReader):
    """Blocks every check read until `release` admits it, counting the reads that started in
    `waiting`."""

    def __init__(self) -> None:
        super().__init__(nodes=_NODES, arc_subjects=_ARC_SUBJECTS, arc_targets=_ARC_TARGETS)
        self.release = threading.Semaphore(0)
        self.waiting = 0
        self._lock = threading.Lock()

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        with self._lock:
            self.waiting += 1
        self.release.acquire(timeout=5.0)
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


class TestCheck:
    def test_a_faulty_row_answers_its_finding_with_the_generation(self, generation):
        rows = [_check_row("CRSPurposeCode120"), _check_row("NoSuchCode")]

        with make_client(graph=_graph()) as client:
            response = client.post(_CHECK, json={"rows": rows})

        assert response.status_code == 200, response.text
        assert response.json() == {
            "generation": "gen-a",
            "row_count": 2,
            "next_row": None,
            "findings": [
                {
                    "row": 1,
                    "ref": "NoSuchCode",
                    "field": "constraint_properties",
                    "property": "crsPurpose",
                    "subject": "value",
                    "value": "NoSuchCode",
                    "served": None,
                    "code": "node_not_held",
                    "severity": "error",
                }
            ],
        }

    def test_every_finding_member_and_next_row_is_encoded_with_nulls(self, generation):
        no_ref = {**_check_row("CRSPurposeCode120"), "measured_property": "", "ref": None}
        rows = [no_ref, *(_check_row(f"NoSuchCode{i}") for i in range(CHECK_MAX_READ_IDS + 1))]

        with make_client(graph=_graph()) as client:
            partial = client.post(_CHECK, json={"rows": rows})
            whole = client.post(_CHECK, json={"rows": [no_ref]})

        assert partial.status_code == whole.status_code == 200, partial.text
        members = ["row", "ref", "field", "property", "subject", "value", "served", "code"]
        first = json.loads(partial.content)["findings"][0]
        assert {key: first[key] for key in members} == {
            "row": 0,
            "ref": None,
            "field": "measured_property",
            "property": None,
            "subject": "value",
            "value": None,
            "served": None,
            "code": "value_missing",
        }
        for body in (json.loads(partial.content), json.loads(whole.content)):
            assert list(body) == ["generation", "row_count", "next_row", "findings"]
            for finding in body["findings"]:
                assert list(finding) == [*members, "severity"]
        assert json.loads(partial.content)["next_row"] == 1 + CHECK_MAX_READ_IDS
        assert b'"next_row":null' in whole.content

    def test_clean_rows_answer_no_findings(self, generation):
        with make_client(graph=_graph()) as client:
            response = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})

        assert response.status_code == 200, response.text
        assert response.json()["findings"] == []
        assert response.json()["row_count"] == 1
        assert response.json()["next_row"] is None

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(
                {"rows": [_check_row("CRSPurposeCode120")] * (CHECK_MAX_ROWS + 1)}, id="too-many"
            ),
            pytest.param({"rows": []}, id="empty"),
            pytest.param({}, id="rows-missing"),
            pytest.param(
                {"rows": [{**_check_row("CRSPurposeCode120"), "unit": "USDollar"}]},
                id="unknown-member",
            ),
            pytest.param(
                {
                    "rows": [
                        {
                            **_check_row("CRSPurposeCode120"),
                            "constraint_properties": {
                                f"p{i}": "CRSPurposeCode120"
                                for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES + 1)
                            },
                        }
                    ]
                },
                id="too-many-constraint-properties",
            ),
            pytest.param(
                {
                    "rows": [
                        {
                            **_check_row("CRSPurposeCode120"),
                            "observation_properties": ["crsPurpose"]
                            * (CHECK_MAX_OBSERVATION_PROPERTIES + 1),
                        }
                    ]
                },
                id="too-many-observation-properties",
            ),
            pytest.param(
                {"rows": [_check_row("x" * (DCID_MAX_LENGTH + 1))]},
                id="constraint-value-past-the-dcid-ceiling",
            ),
            pytest.param(
                {"rows": [{**_check_row("CRSPurposeCode120"), "ref": "r" * (DCID_MAX_LENGTH + 1)}]},
                id="ref-past-the-dcid-ceiling",
            ),
            *(
                pytest.param(
                    '{"rows": [{"measured_property": "crsPurpose", '
                    f'"constraint_properties": {{"crsPurpose": {literal}}}}}]}}',
                    id=f"constraint-value-{literal[:8]}",
                )
                for literal in ("NaN", "Infinity", "-Infinity", "1e400", "1" + "0" * 400)
            ),
        ],
    )
    def test_a_body_outside_the_request_model_is_refused_before_any_read(self, generation, body):
        graph = _graph()

        with make_client(graph=graph) as client:
            if isinstance(body, str):
                response = client.post(
                    _CHECK, content=body, headers={"content-type": "application/json"}
                )
            else:
                response = client.post(_CHECK, json=body)

        assert response.status_code == 400, response.text
        assert response.json()["code"] == "request_invalid"
        assert _reads(graph) == 0

    @pytest.mark.parametrize(
        ("row", "pointer"),
        [
            pytest.param(
                {
                    **_check_row("CRSPurposeCode120"),
                    "constraint_properties": {
                        f"p{i}": "CRSPurposeCode120"
                        for i in range(CHECK_MAX_CONSTRAINT_PROPERTIES + 1)
                    },
                },
                "/rows/2/constraint_properties",
                id="constraint-properties",
            ),
            pytest.param(
                {
                    **_check_row("CRSPurposeCode120"),
                    "population_type": "x" * (DCID_MAX_LENGTH + 1),
                },
                "/rows/2/population_type",
                id="shape-field",
            ),
        ],
    )
    def test_a_row_past_a_size_bound_is_refused_with_a_pointer_naming_it(
        self, generation, row, pointer
    ):
        rows = [_check_row("CRSPurposeCode120"), _check_row("CRSPurposeCode120"), row]

        with make_client(graph=_graph()) as client:
            response = client.post(_CHECK, json={"rows": rows})

        assert response.status_code == 400, response.text
        assert response.json()["code"] == "request_invalid"
        assert response.json()["pointer"] == pointer

    def test_a_malformed_cell_and_a_blank_measured_property_are_findings(self, generation):
        rows = [
            {**_check_row("CRSPurposeCode120"), "measured_property": "bad code"},
            {**_check_row("CRSPurposeCode120"), "measured_property": ""},
        ]

        with make_client(graph=_graph()) as client:
            response = client.post(_CHECK, json={"rows": rows})

        assert response.status_code == 200, response.text
        assert [
            (finding["row"], finding["field"], finding["value"], finding["code"])
            for finding in response.json()["findings"]
        ] == [
            (0, "measured_property", "bad code", "node_not_held"),
            (1, "measured_property", None, "value_missing"),
        ]

    def test_rows_past_the_read_budget_answer_a_prefix_and_the_next_row(self, generation):
        graph = _graph()
        rows = [_check_row(f"NoSuchCode{i}") for i in range(CHECK_MAX_READ_IDS + 3)]

        with make_client(graph=graph) as client:
            response = client.post(_CHECK, json={"rows": rows})

        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["row_count"], body["next_row"]) == (CHECK_MAX_READ_IDS, CHECK_MAX_READ_IDS)
        assert [finding["row"] for finding in body["findings"]] == list(range(CHECK_MAX_READ_IDS))
        assert graph.calls["node_arcs_many"] == 1

    def test_findings_past_the_cap_answer_the_rows_before_and_the_next_row(self, generation):
        fit = CHECK_MAX_FINDINGS // 9
        rows = [_nine_findings_row(str(i)) for i in range(fit + 3)]

        with make_client(graph=_graph()) as client:
            response = client.post(_CHECK, json={"rows": rows})

        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["row_count"], body["next_row"]) == (fit, fit)
        assert len(body["findings"]) == 9 * fit
        assert body["findings"][-1]["row"] == fit - 1

    def test_a_un_id_with_the_un_off_is_a_finding_with_no_read(self, generation, serve_catalogue):
        serve_catalogue(FakeCatalogue(un_system_enabled=False))
        graph = _graph()

        with make_client(graph=graph) as client:
            response = client.post(_CHECK, json={"rows": [_check_row("undata/NoSuchCode")]})

        assert response.status_code == 200, response.text
        assert [
            (finding["property"], finding["value"], finding["code"], finding["severity"])
            for finding in response.json()["findings"]
        ] == [("crsPurpose", "undata/NoSuchCode", "origin_not_served", "error")]
        assert graph.calls["node_arcs_many"] == 0

    def test_a_swapped_code_index_is_checked_against_its_own_build(
        self, generation, monkeypatch: pytest.MonkeyPatch
    ):
        built_with = index_of(
            index_row("Amount_X", {"crsPurpose": "CRSPurposeCode120"}),
            stamp=dataclasses.replace(STAMP, content_hash="index-a"),
        )
        swapped_in = index_of(
            index_row("Amount_X", {"crsPurpose": "OtherCode"}),
            stamp=dataclasses.replace(STAMP, content_hash="index-b"),
        )
        bound = [built_with]
        monkeypatch.setattr(ontology_door, "get_index", lambda: bound[0])
        served = built_with.statvars[0]
        row = {
            "ref": "x",
            "dcid": served.dcid,
            "population_type": served.population_type,
            "measured_property": served.measured_property,
            "stat_type": served.stat_type,
            "constraint_properties": dict(served.constraint_properties),
            "observation_properties": list(served.observation_properties),
        }

        graph = _graph()

        with make_client(graph=graph) as client:
            client.get(_DOCUMENT)
            bound[0] = swapped_in
            response = client.post(_CHECK, json={"rows": [row]})

        assert response.status_code == 200, response.text
        # The check builds for the swapped index and compares the row with its definition.
        assert _builds(graph) == 2
        assert "definition_differs" in {finding["code"] for finding in response.json()["findings"]}

    def test_the_lookup_is_built_once_per_entry(self, generation, monkeypatch: pytest.MonkeyPatch):
        built: list[object] = []
        build = OntologyLookup.of

        def counting(document):
            built.append(document)
            return build(document)

        monkeypatch.setattr(ontology_door.OntologyLookup, "of", counting)

        with make_client(graph=_graph()) as client:
            for _ in range(2):
                response = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})
                assert response.status_code == 200, response.text

        assert len(built) == 1

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
        monkeypatch.setattr(ontology_door, "GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS", 0.2)
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

    def test_a_check_with_reads_runs_while_the_shared_pool_is_full(self, generation):
        hold = threading.Event()

        with make_client(graph=_graph(), node_pool_size=1) as client:
            executor = client.app.state.graph_deps.dcp_executor
            built = client.get(_DOCUMENT)
            wait_until(lambda: executor._in_flight == 0)
            executor.submit(hold.wait, 5.0)
            checked = client.post(_CHECK, json={"rows": [_check_row("NoSuchCode")]})
            hold.set()

        assert built.status_code == 200, built.text
        assert checked.status_code == 200, checked.text
        assert [finding["code"] for finding in checked.json()["findings"]] == ["node_not_held"]

    def test_a_node_read_runs_while_every_check_worker_is_held(self, generation):
        hold = threading.Event()

        with make_client(graph=_graph()) as client:
            checks = client.app.state.graph_deps.check_executor
            for _ in range(GRAPH_ONTOLOGY_CHECK_CONCURRENCY):
                checks.submit(hold.wait, 5.0)
            refused = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})
            node = client.get("/api/graph/node/FinancialFlow")
            hold.set()

        assert refused.status_code == 503, refused.text
        assert node.status_code == 200, node.text

    def test_a_check_past_its_deadline_answers_504_and_holds_its_worker(
        self, generation, monkeypatch: pytest.MonkeyPatch
    ):
        graph = _BlockingCheckReader()

        with make_client(graph=graph) as client:
            checks = client.app.state.graph_deps.check_executor
            assert client.get(_DOCUMENT).status_code == 200
            monkeypatch.setattr(ontology_door, "GRAPH_ONTOLOGY_CHECK_TIMEOUT_SECONDS", 0.05)
            late = client.post(_CHECK, json={"rows": [_check_row("NoSuchCode")]})
            held = checks._in_flight
            graph.release.release()
            wait_until(lambda: checks._in_flight == 0)

        assert late.status_code == 504, late.text
        assert late.json()["code"] == "timeout"
        assert "findings" not in late.json()
        assert held == 1

    def test_a_cancelled_check_holds_its_worker_until_its_job_returns(self, generation):
        graph = _BlockingCheckReader()

        async def send(slot: int) -> httpx.Response:
            transport = httpx.ASGITransport(app=client.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as caller:
                return await caller.post(_CHECK, json={"rows": [_check_row(f"NoSuchCode{slot}")]})

        with make_client(graph=graph) as client:
            checks = client.app.state.graph_deps.check_executor
            assert client.get(_DOCUMENT).status_code == 200
            sent = [
                client.portal.start_task_soon(send, slot)
                for slot in range(GRAPH_ONTOLOGY_CHECK_CONCURRENCY)
            ]
            wait_until(lambda: graph.waiting == GRAPH_ONTOLOGY_CHECK_CONCURRENCY)
            for request in sent:
                request.cancel()
            wait_until(lambda: all(request.done() for request in sent))
            refused = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})
            graph.release.release()
            wait_until(lambda: checks._in_flight == GRAPH_ONTOLOGY_CHECK_CONCURRENCY - 1)
            admitted = client.post(_CHECK, json={"rows": [_check_row("CRSPurposeCode120")]})
            graph.release.release()
            wait_until(lambda: checks._in_flight == 0)

        assert all(request.cancelled() for request in sent)
        assert refused.status_code == 503, refused.text
        assert refused.json()["code"] == "service_unavailable"
        assert refused.json()["retryable"] is True
        assert refused.headers["Retry-After"] == "5"
        assert admitted.status_code == 200, admitted.text
