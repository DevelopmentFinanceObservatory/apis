"""`/v2/observation` reads by GET under a URL budget, and the observation-facet cache."""

from __future__ import annotations

import json
from collections import Counter

import httpx
import pytest

from dc_graph import Facet, GraphCapExceededError, GraphClient, GraphReadError
from graph_generation import FakeGeneration

_BASE_URL = "https://graph.example.org/core/api"
_OBSERVATION_PATH = "/core/api/v2/observation"


def _answer(variables: list[str], entities: list[str], select: list[str]) -> dict:
    """Answer an observation read over a synthetic graph where every pair holds one facet."""
    dated = "date" in select
    by_variable: dict[str, dict] = {}
    facets: dict[str, dict] = {}
    for variable in variables:
        facet_id = f"f-{variable}"
        facets[facet_id] = {"provenanceId": f"prov/{variable}", "unit": "USD"}
        block = {"facetId": facet_id, "obsCount": 2, "earliestDate": "2020", "latestDate": "2021"}
        if dated:
            block["observations"] = [
                {"date": "2020", "value": 1.0},
                {"date": "2021", "value": 2.0},
            ]
        by_variable[variable] = {
            "byEntity": {entity: {"orderedFacets": [block]} for entity in entities}
        }
    return {"byVariable": by_variable, "facets": facets if "facet" in select else {}}


class _ObservationServer:
    """Answer GET and POST observation reads and record what each request asked for."""

    def __init__(self, *, get_only: bool) -> None:
        self.get_only = get_only
        self.requests: list[httpx.Request] = []
        self.observation_body_sizes: list[int] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path != _OBSERVATION_PATH:
            return httpx.Response(200, json={"data": {}})
        if request.method == "POST":
            if self.get_only:
                return httpx.Response(403, json={"error": "POST is not allowed"})
            payload = json.loads(request.content)
            variables = payload["variable"]["dcids"]
            entities = payload["entity"]["dcids"]
            select = payload["select"]
        else:
            variables = request.url.params.get_list("variable.dcids")
            entities = request.url.params.get_list("entity.dcids")
            select = request.url.params.get_list("select")
        response = httpx.Response(200, json=_answer(variables, entities, select))
        self.observation_body_sizes.append(len(response.content))
        return response

    def observation_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == _OBSERVATION_PATH]


def _client(
    server: _ObservationServer, *, budget: int | None, obs_cache_size: int = 0
) -> GraphClient:
    return GraphClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(server.handler),
        observation_get_url_budget_bytes=budget,
        obs_cache_size=obs_cache_size,
        generation=FakeGeneration(token="rev-1"),
    )


def _long_dcids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}/{'x' * 100}_{i:03d}" for i in range(count)]


def test_get_read_sends_the_select_date_and_dcids_as_query_params():
    server = _ObservationServer(get_only=True)
    client = _client(server, budget=7_800)

    result = client.observation_facets_batch(["sv/A"], ["country/KEN"], needs_dates=True)

    [request] = server.observation_requests()
    assert request.method == "GET"
    assert request.content == b""
    params = request.url.params
    assert params.get_list("select") == ["variable", "entity", "date", "value", "facet"]
    assert params.get_list("date") == [""]
    assert params.get_list("variable.dcids") == ["sv/A"]
    assert params.get_list("entity.dcids") == ["country/KEN"]
    assert result["sv/A"]["country/KEN"] == [
        Facet(
            obs_count=2,
            earliest_date="2020",
            latest_date="2021",
            facet_id="f-sv/A",
            provenance_id="prov/sv/A",
            unit="USD",
            dates=("2020", "2021"),
            values=(1.0, 2.0),
        )
    ]


def test_undated_get_read_sends_no_date_param():
    server = _ObservationServer(get_only=True)
    client = _client(server, budget=7_800)

    client.observation_facets_batch_many(stat_vars=["sv/A"], entities=["country/KEN"])

    [request] = server.observation_requests()
    assert request.url.params.get_list("select") == ["variable", "entity", "facet"]
    assert "date" not in request.url.params


@pytest.mark.parametrize(
    ("variable_count", "entity_count", "budget"),
    [
        (60, 1, 2_000),  # one cheap entity stays whole, variables split
        (1, 60, 2_000),  # one cheap variable stays whole, entities split
        (40, 30, 2_000),  # both lists split, every group meets every group
        (3, 2, 7_800),  # everything fits one request
    ],
)
def test_get_groups_fit_the_budget_and_cover_every_pair_once(
    variable_count: int, entity_count: int, budget: int
):
    variables = _long_dcids("sv", variable_count)
    entities = _long_dcids("place", entity_count)
    server = _ObservationServer(get_only=True)
    client = _client(server, budget=budget)

    groups = client.observation_read_groups(variables, entities)
    assert server.requests == []
    result = client.observation_facets_batch(variables, entities)

    requests = server.observation_requests()
    assert all(len(request.url.query) <= budget for request in requests)
    # The plan names the variable groups the read sends, in order, without a request.
    assert groups == [
        list(group)
        for group in dict.fromkeys(
            tuple(request.url.params.get_list("variable.dcids")) for request in requests
        )
    ]
    pairs = Counter(
        (variable, entity)
        for request in requests
        for variable in request.url.params.get_list("variable.dcids")
        for entity in request.url.params.get_list("entity.dcids")
    )
    assert set(pairs) == {(v, e) for v in variables for e in entities}
    assert set(pairs.values()) == {1}
    assert all(len(result[v][e]) == 1 for v in variables for e in entities)
    if variable_count * entity_count > 6:
        assert len(requests) > 1


def test_get_and_post_reads_return_the_same_facets():
    variables = _long_dcids("sv", 40)
    entities = _long_dcids("place", 30)
    by_get = _client(_ObservationServer(get_only=True), budget=2_000)
    by_post = _client(_ObservationServer(get_only=False), budget=None)

    assert by_get.observation_facets_batch(
        variables, entities, needs_dates=True
    ) == by_post.observation_facets_batch(variables, entities, needs_dates=True)
    assert by_get.entities_with_data(variables, entities) == by_post.entities_with_data(
        variables, entities
    )


def test_post_mode_never_sends_a_get():
    server = _ObservationServer(get_only=False)
    client = _client(server, budget=None)

    client.observation_facets_batch(["sv/A"], ["country/KEN"])
    client.entities_with_data(["sv/A"], ["country/KEN"])

    assert {request.method for request in server.observation_requests()} == {"POST"}
    variables = _long_dcids("sv", 200)
    assert client.observation_read_groups(variables, ["country/KEN"]) == [variables]


def test_node_reads_stay_post_under_the_get_option():
    server = _ObservationServer(get_only=True)
    client = _client(server, budget=7_800)

    client.node_labels_batch(["country/KEN"])

    [request] = server.requests
    assert request.method == "POST"
    assert request.url.path == "/core/api/v2/node"


def test_a_dcid_longer_than_the_budget_refuses_before_any_request():
    server = _ObservationServer(get_only=True)
    client = _client(server, budget=300)

    with pytest.raises(ValueError, match="alone exceeds the observation GET URL budget"):
        client.observation_facets_batch(["sv/" + "x" * 400], ["country/KEN"])

    assert server.requests == []


def test_byte_cap_counts_every_get_request_together():
    """A cap each response fits under alone still trips once the responses add up past it."""
    variables = _long_dcids("sv", 40)
    uncapped = _ObservationServer(get_only=True)
    _client(uncapped, budget=2_000).observation_facets_batch(variables, ["country/KEN"])
    sizes = uncapped.observation_body_sizes
    assert len(sizes) > 2
    cap = sizes[0] + sizes[1] // 2
    assert max(sizes) < cap

    server = _ObservationServer(get_only=True)
    with pytest.raises(GraphCapExceededError) as exc_info:
        _client(server, budget=2_000).observation_facets_batch(
            variables, ["country/KEN"], max_bytes=cap
        )

    assert len(server.observation_requests()) == 2
    assert exc_info.value.cap == cap
    assert exc_info.value.observed > cap


def test_get_retries_a_5xx_like_post():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_answer(["sv/A"], ["country/KEN"], ["entity", "variable"]))

    client = GraphClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
        observation_get_url_budget_bytes=7_800,
    )
    client._await_retry = lambda attempt, deadline: attempt < 3  # no backoff sleep

    assert client.entities_with_data(["sv/A"], ["country/KEN"]) == ["country/KEN"]
    assert calls == ["GET", "GET"]


def test_observation_cache_serves_a_repeat_pair_and_keys_on_needs_dates():
    server = _ObservationServer(get_only=False)
    client = _client(server, budget=None, obs_cache_size=16)

    first = client.observation_facets_batch(["sv/A"], ["country/KEN"])
    second = client.observation_facets(stat_var="sv/A", entity="country/KEN")
    dated = client.observation_facets_batch(["sv/A"], ["country/KEN"], needs_dates=True)

    assert second == first["sv/A"]["country/KEN"]
    assert dated["sv/A"]["country/KEN"][0].dates == ("2020", "2021")
    assert len(server.observation_requests()) == 2
    stats = client.stats()
    assert (stats["obs_hits"], stats["obs_misses"], stats["obs_entries"]) == (1, 2, 2)

    client.clear()
    assert client.stats()["obs_entries"] == 0


def test_observation_cache_reads_only_the_missed_pairs():
    server = _ObservationServer(get_only=False)
    client = _client(server, budget=None, obs_cache_size=16)

    client.observation_facets_batch(["sv/A"], ["country/KEN"])
    client.observation_facets_batch(["sv/A", "sv/B"], ["country/KEN"])

    last = json.loads(server.observation_requests()[-1].content)
    assert last["variable"]["dcids"] == ["sv/B"]
    assert last["entity"]["dcids"] == ["country/KEN"]


def test_observation_facets_batch_many_stays_uncached():
    server = _ObservationServer(get_only=False)
    client = _client(server, budget=None, obs_cache_size=16)

    client.observation_facets_batch_many(stat_vars=["sv/A"], entities=["country/KEN"])
    client.observation_facets_batch_many(stat_vars=["sv/A"], entities=["country/KEN"])

    assert len(server.observation_requests()) == 2
    assert client.stats()["obs_entries"] == 0


def test_undated_read_marks_dates_complete_and_dated_read_flags_missing_dates():
    """`dates_complete` is `False` only when a dated read returned fewer dates than obsCount."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        block = {"facetId": "f1", "obsCount": 3, "earliestDate": "2019", "latestDate": "2021"}
        if "date" in payload["select"]:
            block["observations"] = [{"date": "2020", "value": 1.0}, {"value": 2.0}]
        body = {"byVariable": {"sv/A": {"byEntity": {"country/KEN": {"orderedFacets": [block]}}}}}
        return httpx.Response(200, json=body)

    client = GraphClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler))

    [undated] = client.observation_facets(stat_var="sv/A", entity="country/KEN")
    [dated] = client.observation_facets(stat_var="sv/A", entity="country/KEN", needs_dates=True)

    assert (undated.dates, undated.dates_complete) == ((), True)
    assert (dated.dates, dated.values, dated.dates_complete) == (("2020",), (1.0,), False)


@pytest.mark.parametrize("budget", [None, 7_800])
def test_facet_reads_send_domains_as_the_filter(budget: int | None):
    server = _ObservationServer(get_only=budget is not None)
    client = _client(server, budget=budget)

    client.observation_facets_batch_many(
        stat_vars=["sv/A"], entities=["country/KEN"], domains=("oecd.org", "worldbank.org")
    )

    [request] = server.observation_requests()
    if budget is None:
        assert json.loads(request.content)["filter"] == {"domains": ["oecd.org", "worldbank.org"]}
    else:
        assert request.url.params.get_list("filter.domains") == ["oecd.org", "worldbank.org"]


def test_the_observation_cache_keys_on_the_domain_filter():
    server = _ObservationServer(get_only=False)
    client = _client(server, budget=None, obs_cache_size=16)

    for domains in (None, ("oecd.org",), ("oecd.org",), None):
        client.observation_facets_batch(["sv/A"], ["country/KEN"], domains=domains)

    sent = [json.loads(r.content).get("filter") for r in server.observation_requests()]
    assert sent == [None, {"domains": ["oecd.org"]}]


# The UN front door's refusal page, as served for a GET naming `country/GRC`.
_FRONT_DOOR_PAGE = (
    '<!doctype html><meta charset="utf-8"><meta name=viewport '
    'content="width=device-width, initial-scale=1"><title>403</title>403 Forbidden'
)


class _Refusing(_ObservationServer):
    """Refuses any GET naming a place in `refused` with `status` and `page`, by default the UN
    front door's refusal."""

    def __init__(
        self,
        refused: set[str],
        *,
        status: int = 403,
        page: str = _FRONT_DOOR_PAGE,
        content_type: str = "text/html; charset=UTF-8",
    ) -> None:
        super().__init__(get_only=True)
        self.refused = refused
        self.status = status
        self.page = page
        self.content_type = content_type

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.refused & set(request.url.params.get_list("entity.dcids")):
            self.requests.append(request)
            return httpx.Response(
                self.status, text=self.page, headers={"content-type": self.content_type}
            )
        return super().handler(request)


def _places(count: int) -> list[str]:
    return [f"country/P{i:02d}" for i in range(count)]


@pytest.mark.parametrize("max_bytes", [None, 10_000_000])
def test_a_front_door_refusal_isolates_the_refused_places_and_serves_the_rest(
    max_bytes: int | None,
):
    places = _places(16)
    server = _Refusing({places[3], places[12]})
    client = _client(server, budget=7_800)

    result = client.observation_facets_batch_many(
        stat_vars=["sv/A"], entities=places, max_bytes=max_bytes
    )

    served = [place for place, facets in result["sv/A"].items() if facets]
    assert served == [p for p in places if p not in {places[3], places[12]}]
    assert client.blocked_observation_entities() == {places[3], places[12]}


def test_blocked_places_are_left_out_up_front_until_the_generation_changes():
    places = _places(8)
    server = _Refusing({places[5]})
    generation = FakeGeneration(token="rev-1")
    client = GraphClient(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(server.handler),
        observation_get_url_budget_bytes=7_800,
        generation=generation,
    )
    client.entities_with_data(["sv/A"], places)
    first = len(server.observation_requests())

    assert client.entities_with_data(["sv/A"], places) == [p for p in places if p != places[5]]
    [again] = server.observation_requests()[first:]
    assert places[5] not in again.url.params.get_list("entity.dcids")

    generation.advance("rev-2")
    assert client.blocked_observation_entities() == frozenset()
    client.entities_with_data(["sv/A"], places)
    assert len(server.observation_requests()) > first + 2


def test_seeded_places_are_never_named_and_cost_no_refusal():
    places = _places(8)
    server = _Refusing({places[2], places[6]})
    client = _client(server, budget=7_800)

    client.seed_blocked_observation_entities(frozenset({places[2], places[6]}))
    served = client.entities_with_data(["sv/A"], places)

    assert served == [p for p in places if p not in {places[2], places[6]}]
    [request] = server.observation_requests()
    assert set(request.url.params.get_list("entity.dcids")) == set(served)
    assert client.blocked_observation_entities() == {places[2], places[6]}


@pytest.mark.parametrize("status", [401, 404])
def test_statuses_other_than_403_still_fail_the_read(status: int):
    places = _places(4)
    server = _Refusing({places[0]}, status=status)
    client = _client(server, budget=7_800)

    with pytest.raises(GraphReadError):
        client.observation_facets_batch_many(stat_vars=["sv/A"], entities=places)
    assert client.blocked_observation_entities() == frozenset()


@pytest.mark.parametrize(
    ("page", "content_type"),
    [
        ('{"error": "forbidden"}', "application/json"),
        ("<html><title>403</title>Rate limit exceeded</html>", "text/html"),
        (_FRONT_DOOR_PAGE, "text/plain"),
    ],
)
@pytest.mark.parametrize("max_bytes", [None, 10_000_000])
def test_a_403_in_any_other_shape_fails_the_read_and_blocks_nothing(
    page: str, content_type: str, max_bytes: int | None
):
    places = _places(4)
    server = _Refusing({places[0]}, page=page, content_type=content_type)
    client = _client(server, budget=7_800)

    with pytest.raises(GraphReadError) as excinfo:
        client.observation_facets_batch_many(
            stat_vars=["sv/A"], entities=places, max_bytes=max_bytes
        )
    assert excinfo.value.upstream_status == 403
    assert client.blocked_observation_entities() == frozenset()
    assert len(server.observation_requests()) == 1
