"""Resolve-door tests use the shared resolver and an in-memory graph reader.
The same fake serves entity confirmation and region expansion.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from entity_resolution import EntityResolutionError, NotFound, Resolution
from graph_api.deps import GraphDeps
from graph_api.doors import resolve as resolve_door
from materialise.graph import GraphDeadlineExceededError, GraphReadError

from .conftest import FakeGraphNodeReader, make_client, make_resolver


def test_kenya_resolved_and_confirmed(shared_resolver):
    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {
        "status": "resolved",
        "entity": {
            "dcid": "country/KEN",
            "name": "Kenya",
            "entity_type": "Country",
            "kind": "country",
        },
    }


def test_kenya_resolved_but_not_held_reports_not_held(shared_resolver):
    graph = FakeGraphNodeReader(nodes={})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {"status": "not_held", "name": "Kenya", "dcid": "country/KEN"}


def test_below_confidence_name_refuses_as_not_found_with_near_matches(shared_resolver):
    """Cash falls below the match threshold and returns the rejected names as near matches."""
    graph = FakeGraphNodeReader(nodes={})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Cash"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "not_found"
    assert result["near_matches"], "expected near matches for 'Cash'"
    assert all(isinstance(name, str) and name for name in result["near_matches"])


def test_nonsense_text_refuses_as_not_found_with_no_near_matches(shared_resolver):
    graph = FakeGraphNodeReader(nodes={})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["zzqqxv"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {"status": "not_found", "near_matches": []}


def test_congo_ambiguous_when_graph_holds_both_candidates(shared_resolver):
    graph = FakeGraphNodeReader(
        nodes={
            "country/COD": {"typeOf": ("Country",), "name": ("Congo [DRC]",)},
            "country/COG": {"typeOf": ("Country",), "name": ("Congo [Republic]",)},
        }
    )
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Congo"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "ambiguous"
    by_dcid = {c["dcid"]: c for c in result["candidates"]}
    assert set(by_dcid) == {"country/COD", "country/COG"}
    assert by_dcid["country/COD"]["kind"] == "country"
    assert by_dcid["country/COD"]["entity_type"] == "Country"


def test_congo_resolves_when_graph_holds_exactly_one_candidate(shared_resolver):
    graph = FakeGraphNodeReader(
        nodes={"country/COD": {"typeOf": ("Country",), "name": ("Congo [DRC]",)}}
    )
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Congo"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "resolved"
    assert result["entity"]["dcid"] == "country/COD"


def test_congo_not_held_when_graph_holds_neither_candidate(shared_resolver):
    graph = FakeGraphNodeReader(nodes={})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Congo"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "not_held"
    assert result["dcid"] == "country/COD"
    assert "country/COG" not in resp.text


def test_spelling_corrected_winner_demotes_to_not_found_with_correction_as_near_match(
    shared_resolver,
):
    """One-edit spelling correction maps Lagos to Laos, then the listed-name
    gate returns NotFound with Laos as a near match.
    """
    graph = FakeGraphNodeReader(nodes={"country/LAO": {"typeOf": ("Country",)}})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Lagos"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {"status": "not_found", "near_matches": ["Laos"]}


def test_typo_corrected_winner_demotes_to_not_found_with_correction_as_near_match(shared_resolver):
    graph = FakeGraphNodeReader(nodes={"country/NGA": {"typeOf": ("Country",)}})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Nigria"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {"status": "not_found", "near_matches": ["Nigeria"]}


@pytest.mark.parametrize(
    ("text", "dcid"),
    [
        ("Laos", "country/LAO"),
        ("Kenia", "country/KEN"),
        ("Britain", "country/GBR"),
        ("Côte d'Ivoire", "country/CIV"),
        ("DRC", "country/COD"),
        ("UK", "country/GBR"),
        ("USA", "country/USA"),
    ],
)
def test_listed_alias_or_code_stays_resolved(text: str, dcid: str, shared_resolver):
    """Aliases, country codes, and canonical names resolve directly."""
    graph = FakeGraphNodeReader(nodes={dcid: {"typeOf": ("Country",)}})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": [text]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "resolved", (text, result)
    assert result["entity"]["dcid"] == dcid


@pytest.mark.parametrize(
    ("text", "name", "dcid", "entity_type", "kind"),
    [
        (
            "African Development Bank",
            "African Development Bank",
            "AfricanDevelopmentBank",
            "Organization",
            "organisation",
        ),
        (
            "AfDB",
            "African Development Bank",
            "AfricanDevelopmentBank",
            "Organization",
            "organisation",
        ),
        (
            "EU Institutions",
            "European Union institutions",
            "EuropeanUnionInstitutions",
            "Organization",
            "organisation",
        ),
        (
            "Africa, regional",
            "Africa, unallocated by country (DAC)",
            "DACAfricaUnallocated",
            "DACRecipientUnallocatedByCountry",
            "aggregate",
        ),
    ],
)
def test_organisation_or_aggregate_resolves_with_kind(
    text: str, name: str, dcid: str, entity_type: str, kind: str, shared_resolver
):
    graph = FakeGraphNodeReader(nodes={dcid: {"typeOf": (entity_type,), "name": (name,)}})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": [text]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {
        "status": "resolved",
        "entity": {"dcid": dcid, "name": name, "entity_type": entity_type, "kind": kind},
    }


def test_world_bank_curated_ambiguity_names_ibrd_and_ida(shared_resolver):
    """The curated alias maps World Bank to both IBRD and IDA."""
    graph = FakeGraphNodeReader(
        nodes={
            "InternationalBankForReconstructionAndDevelopment": {"typeOf": ("Organization",)},
            "InternationalDevelopmentAssociation": {"typeOf": ("Organization",)},
        }
    )
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["World Bank"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "ambiguous"
    assert {c["dcid"] for c in result["candidates"]} == {
        "InternationalBankForReconstructionAndDevelopment",
        "InternationalDevelopmentAssociation",
    }
    assert all(c["kind"] == "organisation" for c in result["candidates"])


def _members_of_on_shared_thread(resolver, group: str) -> list[str]:
    return resolver.members_of(group)


@pytest.mark.parametrize("text", ["OECD", "G7"])
def test_oecd_and_g7_resolve_as_group_with_no_entity_reading(text: str, shared_resolver):
    """OECD and G7 resolve to group ids absent from the graph. Their held
    member countries return as `resolved_group` with `entity: null`. Member
    ids come from the shared resolver's `members_of` lookup.
    """
    member_ids = shared_resolver.run(
        _members_of_on_shared_thread, shared_resolver.resolver, text
    ).result()
    assert member_ids
    graph = FakeGraphNodeReader(
        nodes={dcid: {"typeOf": ("Country",), "name": (dcid,)} for dcid in member_ids}
    )
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": [text]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "resolved_group"
    assert result["entity"] is None
    assert {member["dcid"] for member in result["members"]} == set(member_ids)
    assert all(member["kind"] == "country" for member in result["members"])


def test_eu_resolves_as_group_with_both_entity_and_members(shared_resolver):
    """The held `EuropeanUnion` node and all 27 member countries appear in
    the result. This door has no aid role to choose between them.
    """
    member_ids = shared_resolver.run(
        _members_of_on_shared_thread, shared_resolver.resolver, "EU"
    ).result()
    assert member_ids
    nodes = {dcid: {"typeOf": ("Country",), "name": (dcid,)} for dcid in member_ids}
    nodes["EuropeanUnion"] = {"typeOf": ("ContinentalUnion",), "name": ("European Union",)}
    graph = FakeGraphNodeReader(nodes=nodes)
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["EU"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result["status"] == "resolved_group"
    assert result["entity"] == {
        "dcid": "EuropeanUnion",
        "name": "European Union",
        "entity_type": "ContinentalUnion",
        "kind": "continental_union",
    }
    assert {member["dcid"] for member in result["members"]} == set(member_ids)


def test_group_with_no_confirmed_member_and_no_entity_is_not_held(shared_resolver):
    """An unheld group with no held members returns `not_held` for its group id."""
    graph = FakeGraphNodeReader(nodes={})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["G7"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {"status": "not_held", "name": "Group of Seven", "dcid": "groups/G7"}


def test_sub_saharan_africa_and_africa_expand_through_containment_in_one_mixed_batch(
    shared_resolver,
):
    """A mixed request expands Sub-Saharan Africa and Africa in one containment
    call. Unlabelled member `country/BEN` uses its dcid as the display name.
    """
    graph = FakeGraphNodeReader(
        nodes={
            "SubSaharanAfrica": {"typeOf": ("UNGeoRegion",)},
            "africa": {"typeOf": ("Continent",)},
            "country/KEN": {"typeOf": ("Country",)},
            "country/AGO": {"typeOf": ("Country",), "name": ("Angola",)},
        },
        contained={
            ("SubSaharanAfrica", "Country"): ["country/AGO", "country/BEN"],
            ("africa", "Country"): ["country/AGO"],
        },
    )
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post(
            "/api/graph/resolve", json={"texts": ["Sub-Saharan Africa", "Kenya", "Africa"]}
        )

    assert resp.status_code == 200, resp.text
    assert "m49/" not in resp.text
    assert "wikidataId/" not in resp.text
    results = resp.json()["results"]
    assert len(results) == 3

    assert results[0] == {
        "status": "resolved_region",
        "region": {
            "dcid": "SubSaharanAfrica",
            "name": "Sub-Saharan Africa",
            "entity_type": "UNGeoRegion",
            "kind": "subregion",
        },
        "members": [
            {"dcid": "country/AGO", "name": "Angola"},
            {"dcid": "country/BEN", "name": "country/BEN"},
        ],
    }
    assert results[1]["status"] == "resolved"
    assert results[1]["entity"]["dcid"] == "country/KEN"
    assert results[2] == {
        "status": "resolved_region",
        "region": {
            "dcid": "africa",
            "name": "Africa",
            "entity_type": "Continent",
            "kind": "continent",
        },
        "members": [{"dcid": "country/AGO", "name": "Angola"}],
    }
    assert graph.calls["child_places_batch"] == 1


def test_resolving_a_subregion_and_a_continent_leaks_no_internal_dcid_prefix(shared_resolver):
    """Region responses must not expose `m49/` or `wikidataId/` dcids, which are internal graph
    namespaces.
    """
    graph = FakeGraphNodeReader(
        nodes={
            "SubSaharanAfrica": {"typeOf": ("UNGeoRegion",)},
            "africa": {"typeOf": ("Continent",)},
            "country/AGO": {"typeOf": ("Country",), "name": ("Angola",)},
        },
        contained={
            ("SubSaharanAfrica", "Country"): ["country/AGO"],
            ("africa", "Country"): ["country/AGO"],
        },
    )
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Sub-Saharan Africa", "Africa"]})

    assert resp.status_code == 200, resp.text
    assert "m49/" not in resp.text
    assert "wikidataId/" not in resp.text


def test_sub_saharan_africa_not_held_when_the_region_itself_is_not_confirmed(shared_resolver):
    """Confirmation precedes expansion. An unconfirmed region returns
    `not_held`, and the containment call count remains zero.
    """
    graph = FakeGraphNodeReader(nodes={})
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Sub-Saharan Africa"]})

    assert resp.status_code == 200, resp.text
    result = resp.json()["results"][0]
    assert result == {
        "status": "not_held",
        "name": "Sub-Saharan Africa",
        "dcid": "SubSaharanAfrica",
    }
    assert graph.calls["child_places_batch"] == 0


class TestGraphHoldingNothingNeverConfirms:
    """An empty reader returns `not_held` for listed entities and `not_found`
    for text with no listed candidate.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Kenya",
            "Cash",
            "zzqqxv",
            "Congo",
            "Sub-Saharan Africa",
            "Africa",
            "World Bank",
            "OECD",
            "G7",
            "EU",
        ],
    )
    def test_never_resolves(self, text: str, shared_resolver):
        graph = FakeGraphNodeReader(nodes={})
        client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

        with client:
            resp = client.post("/api/graph/resolve", json={"texts": [text]})

        assert resp.status_code == 200, resp.text
        status = resp.json()["results"][0]["status"]
        assert status in {"not_held", "not_found"}, (text, status)


class TestValidation:
    def test_empty_texts_list_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": []})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    def test_unknown_top_level_field_carries_pydantic_message_as_detail(self):
        with make_client() as client:
            resp = client.post(
                "/api/graph/resolve", json={"texts": ["Kenya"], "extra_field": "nope"}
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "Extra inputs are not permitted"

    def test_over_25_texts_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"] * 26})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    def test_empty_string_text_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": [""]})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"

    def test_201_char_text_refuses_request_invalid(self):
        with make_client() as client:
            resp = client.post("/api/graph/resolve", json={"texts": ["a" * 201]})
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == "request_invalid"


class _BlockingGraphNodeReader:
    """Block `node_arcs_many` until released to test semaphore admission.

    The reader ignores its deadline and waits at most 5 seconds.
    """

    def __init__(self, delegate: FakeGraphNodeReader) -> None:
        self._delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = delegate.calls

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.entered.set()
        self.release.wait(timeout=5.0)
        return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)

    def resolve_by_arc(self, names, *, arc, deadline=None):
        return self._delegate.resolve_by_arc(names, arc=arc, deadline=deadline)

    def child_places_batch(self, parent_dcids, *, child_type, deadline=None):
        return self._delegate.child_places_batch(
            parent_dcids, child_type=child_type, deadline=deadline
        )

    def node_names_many(self, dcids, *, deadline=None):
        return self._delegate.node_names_many(dcids, deadline=deadline)


class _DeadlineBoundedGraphNodeReader:
    """Stall selected reads until their deadline, then raise
    `GraphDeadlineExceededError`. Each stall is capped at 5 seconds. Later
    reads delegate, and `observed_deadlines` records every deadline.
    """

    _MAX_BLOCK_SECONDS = 5.0

    def __init__(self, delegate: FakeGraphNodeReader, *, stall_reads: int = 1) -> None:
        self._delegate = delegate
        self._remaining_stalls = stall_reads
        self.entered = threading.Event()
        self.entered_at: float | None = None
        self.observed_deadlines = []
        self.calls = delegate.calls

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        self.observed_deadlines.append(deadline)
        if self._remaining_stalls <= 0:
            return self._delegate.node_arcs_many(dcids, properties=properties, deadline=deadline)
        self._remaining_stalls -= 1
        self.entered_at = time.monotonic()
        self.entered.set()
        if deadline is None:
            return self._delegate.node_arcs_many(dcids, properties=properties)
        remaining = min(max(deadline - time.monotonic(), 0.0), self._MAX_BLOCK_SECONDS)
        time.sleep(remaining)
        raise GraphDeadlineExceededError("the read stopped at the door's own deadline")

    def resolve_by_arc(self, names, *, arc, deadline=None):
        return self._delegate.resolve_by_arc(names, arc=arc, deadline=deadline)

    def child_places_batch(self, parent_dcids, *, child_type, deadline=None):
        return self._delegate.child_places_batch(
            parent_dcids, child_type=child_type, deadline=deadline
        )

    def node_names_many(self, dcids, *, deadline=None):
        return self._delegate.node_names_many(dcids, deadline=deadline)


def test_semaphore_full_refuses_503_service_unavailable(shared_resolver):
    delegate = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
    graph = _BlockingGraphNodeReader(delegate)
    client = make_client(
        graph=graph, resolver=make_resolver(shared_resolver, graph=graph), node_pool_size=1
    )

    with client:
        first_response: list = []
        first_thread = threading.Thread(
            target=lambda: first_response.append(
                client.post("/api/graph/resolve", json={"texts": ["Kenya"]})
            )
        )
        first_thread.start()
        assert graph.entered.wait(timeout=5.0), "first request never reached the blocking read"

        second_response = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

        assert second_response.status_code == 503, second_response.text
        second_problem = second_response.json()
        assert second_problem["code"] == "service_unavailable"
        assert second_problem["retryable"] is True

        graph.release.set()
        first_thread.join(timeout=5.0)
        assert first_response, "first request never returned"
        assert first_response[0].status_code == 200, first_response[0].text


def test_timed_out_request_returns_the_read_at_its_deadline_and_frees_capacity(
    shared_resolver, monkeypatch: pytest.MonkeyPatch
):
    """A 1-second read deadline returns 504 and frees the sole semaphore slot.
    A second request must succeed within 0.5 seconds after the read exits.
    """
    monkeypatch.setattr(resolve_door, "GRAPH_RESOLVE_TIMEOUT_SECONDS", 1.0)
    delegate = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",)}})
    graph = _DeadlineBoundedGraphNodeReader(delegate)
    started_at = time.monotonic()
    client = make_client(
        graph=graph, resolver=make_resolver(shared_resolver, graph=graph), node_pool_size=1
    )

    with client:
        first = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})
        assert first.status_code == 504, first.text
        assert first.json()["code"] == "timeout"
        assert graph.entered.wait(timeout=5.0), "confirmation read never started"

        # The read stops at the door deadline. It is at most one timeout
        # after read entry, or earlier if resolvekit used part of the budget.
        assert graph.observed_deadlines, "the read was handed no deadline"
        door_deadline = graph.observed_deadlines[0]
        assert door_deadline is not None
        assert graph.entered_at is not None
        assert door_deadline - graph.entered_at <= resolve_door.GRAPH_RESOLVE_TIMEOUT_SECONDS
        assert time.monotonic() - started_at < resolve_door.GRAPH_RESOLVE_TIMEOUT_SECONDS + 1.0

        # The 504 can precede the worker's return by a few milliseconds, so
        # the first retry may get 503. Admission within 0.5 seconds proves
        # the completion callback freed the slot.
        freed_by = time.monotonic() + 0.5
        while True:
            second = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})
            if second.status_code != 503 or time.monotonic() > freed_by:
                break
            time.sleep(0.01)
        assert second.status_code == 200, second.text
        assert second.json()["results"] == [
            {
                "status": "resolved",
                "entity": {
                    "dcid": "country/KEN",
                    "name": "Kenya",
                    "entity_type": "Country",
                    "kind": "country",
                },
            }
        ]


class TestGraphReadErrorClassification:
    def test_upstream_4xx_maps_to_502_upstream_error(self):
        problem = resolve_door.classify_resolve_exception(
            GraphReadError("exemplar", upstream_status=404)
        )
        assert problem is not None
        assert problem.status == 502
        assert problem.code == "upstream_error"
        assert problem.retryable is True

    def test_no_upstream_status_maps_to_503_service_unavailable(self):
        problem = resolve_door.classify_resolve_exception(GraphReadError("exemplar"))
        assert problem is not None
        assert problem.status == 503
        assert problem.code == "service_unavailable"
        assert problem.retryable is True

    def test_upstream_5xx_maps_to_503_service_unavailable(self):
        problem = resolve_door.classify_resolve_exception(
            GraphReadError("exemplar", upstream_status=502)
        )
        assert problem is not None
        assert problem.status == 503
        assert problem.code == "service_unavailable"
        assert problem.retryable is True

    def test_deadline_exceeded_maps_to_504_timeout_before_the_general_arm(self):
        # `GraphDeadlineExceededError` subclasses `GraphReadError`, so the
        # deadline arm must precede the general arm. Combined classification
        # also uses it for node and sources door deadlines.
        problem = resolve_door.classify_resolve_exception(GraphDeadlineExceededError("exemplar"))
        assert problem is not None
        assert problem.status == 504
        assert problem.code == "timeout"
        assert problem.retryable is True


class _RaisingGraphNodeReader:
    """Raise `GraphReadError` from graph reads for end-to-end classification tests."""

    def __init__(self, exc: GraphReadError) -> None:
        self._exc = exc

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        raise self._exc

    def node_names_many(self, dcids, *, deadline=None):
        raise self._exc


def test_graph_read_error_end_to_end_maps_to_502(caplog: pytest.LogCaptureFixture, shared_resolver):
    graph = _RaisingGraphNodeReader(GraphReadError("upstream refused", upstream_status=404))
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client, caplog.at_level(logging.ERROR):
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 502, resp.text
    assert resp.json()["code"] == "upstream_error"


def test_graph_read_error_end_to_end_maps_to_503_without_upstream_status(shared_resolver):
    graph = _RaisingGraphNodeReader(GraphReadError("transport failure"))
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 503, resp.text
    assert resp.json()["code"] == "service_unavailable"


class _StubErrorResolver:
    """Stub that raises `EntityResolutionError` to test the door's 500 response."""

    async def resolve(self, texts: list[str], *, deadline: float | None = None) -> list[Resolution]:
        raise EntityResolutionError("resolvekit reported an internal error")


def test_resolvekit_error_status_raises_and_defaults_to_500_internal_error():
    client = make_client(resolver=_StubErrorResolver())

    with client:
        resp = client.post("/api/graph/resolve", json={"texts": ["Kenya"]})

    assert resp.status_code == 500, resp.text
    assert resp.json()["code"] == "internal_error"


class _StubNotFoundResolver:
    """Stub returning `NotFound` for every input."""

    async def resolve(self, texts: list[str], *, deadline: float | None = None) -> list[Resolution]:
        return [NotFound() for _ in texts]


def test_run_resolve_refreshes_the_generation_before_acquiring_the_semaphore():
    """`_run_resolve` refreshes generation once before acquiring the DCP semaphore."""
    calls: list[None] = []

    class _SpyGeneration:
        async def refresh_if_stale(self, *, timeout_s: float = 1.0) -> None:
            calls.append(None)

    deps = GraphDeps(
        graph=FakeGraphNodeReader(),
        resolver=_StubNotFoundResolver(),
        dcp_executor=ThreadPoolExecutor(max_workers=1),
        dcp_semaphore=threading.BoundedSemaphore(1),
        generation=_SpyGeneration(),
    )

    response = asyncio.run(resolve_door._run_resolve(deps, ["Kenya"]))

    assert calls == [None]
    assert response.results[0].status == "not_found"
