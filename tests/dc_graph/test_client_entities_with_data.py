"""Tests batched entity presence reads, variable chunking, early exit, and empty inputs."""

import httpx
import pytest

from dc_graph import GraphCapExceededError, GraphClient, GraphReadError

# ---------------------------------------------------------------------------
# entities_with_data
# ---------------------------------------------------------------------------


def test_entities_with_data_posts_presence_only_select(monkeypatch) -> None:
    """The POST body uses the lean entity+variable select with explicit entity.dcids."""
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        return {
            "byVariable": {
                "sv/GDP": {"byEntity": {"country/GTM": {}, "country/USA": {}}},
            },
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.entities_with_data(["sv/GDP"], ["country/GTM", "country/USA", "country/VAT"])

    assert seen_bodies[0]["select"] == ["entity", "variable"]
    assert seen_bodies[0]["variable"] == {"dcids": ["sv/GDP"]}
    assert seen_bodies[0]["entity"] == {"dcids": ["country/GTM", "country/USA", "country/VAT"]}
    assert result == ["country/GTM", "country/USA"]


def test_entities_with_data_union_keeps_a_single_variable_member(monkeypatch) -> None:
    """A candidate present for only ONE of several requested variables still
    qualifies -- union, not intersection. country/USA has data for sv/A
    alone and must survive; country/ZZZ has data for neither and must not."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "byVariable": {
                "sv/A": {"byEntity": {"country/GTM": {}, "country/USA": {}}},
                "sv/B": {"byEntity": {"country/GTM": {}}},
            },
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.entities_with_data(
        ["sv/A", "sv/B"], ["country/GTM", "country/USA", "country/ZZZ"]
    )
    assert result == ["country/GTM", "country/USA"]


def test_entities_with_data_missing_variable_key_contributes_nothing(monkeypatch) -> None:
    """A variable entirely absent from byVariable contributes no entities to the
    union, but does not erase what another variable already contributed."""

    def fake_post(self, path, payload, *, deadline=None):
        return {"byVariable": {"sv/A": {"byEntity": {"country/GTM": {}}}}}  # sv/B missing

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.entities_with_data(["sv/A", "sv/B"], ["country/GTM"])
    assert result == ["country/GTM"]


def test_entities_with_data_empty_inputs_no_post(monkeypatch) -> None:
    post_called = []

    def fake_post(self, path, payload, *, deadline=None):
        post_called.append(payload)
        return {}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    assert client.entities_with_data([], ["country/GTM"]) == []
    assert client.entities_with_data(["sv/GDP"], []) == []
    assert post_called == []


def test_entities_with_data_raises_on_upstream_error(monkeypatch) -> None:
    def handler(request):
        return httpx.Response(503, json={})

    client = GraphClient(base_url="http://fake/core/api", transport=httpx.MockTransport(handler))

    with pytest.raises(GraphReadError):
        client.entities_with_data(["sv/GDP"], ["country/GTM"])


# ---------------------------------------------------------------------------
# entities_with_data chunking over variable_dcids (a server-derived options
# population can run to several thousand StatVars; see
# entities_with_data_variable_chunk_size)
# ---------------------------------------------------------------------------


def test_entities_with_data_chunks_variable_dcids(monkeypatch) -> None:
    """More variable_dcids than the chunk size issues one POST per chunk,
    each naming the FULL entity_dcids list (entity_dcids is never chunked
    here), and the results union across chunks."""
    seen_bodies: list[dict] = []

    entity_for_var = {"sv/0": "e0", "sv/1": "e1", "sv/2": "e2"}

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        # Each variable alone has data for one distinct entity, so every
        # chunk must be read for the full union to assemble.
        by_variable = {}
        for var in payload["variable"]["dcids"]:
            entity = entity_for_var.get(var)
            if entity is not None:
                by_variable[var] = {"byEntity": {entity: {}}}
        return {"byVariable": by_variable}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", entities_with_data_variable_chunk_size=2)

    result = client.entities_with_data(["sv/0", "sv/1", "sv/2"], ["e0", "e1", "e2", "e3"])

    # ceil(3 variable_dcids / chunk size 2) == 2 POSTs.
    assert len(seen_bodies) == 2
    assert seen_bodies[0]["variable"] == {"dcids": ["sv/0", "sv/1"]}
    assert seen_bodies[1]["variable"] == {"dcids": ["sv/2"]}
    # entity_dcids is posted in full on every chunk, never sliced.
    assert all(b["entity"] == {"dcids": ["e0", "e1", "e2", "e3"]} for b in seen_bodies)
    assert result == ["e0", "e1", "e2"]


def test_entities_with_data_stops_early_once_every_entity_is_present(monkeypatch) -> None:
    """Once the union already covers every entity_dcids member, a later chunk
    is never requested -- union is monotonic, so it can add nothing more."""
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        # The very first chunk (sv/0) already covers both candidates.
        return {"byVariable": {"sv/0": {"byEntity": {"e0": {}, "e1": {}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", entities_with_data_variable_chunk_size=1)

    result = client.entities_with_data(["sv/0", "sv/1", "sv/2"], ["e0", "e1"])

    assert len(seen_bodies) == 1, "a fully-covered union must not issue further chunk POSTs"
    assert result == ["e0", "e1"]


def test_entities_with_data_max_bytes_bounds_each_chunk(monkeypatch) -> None:
    """max_bytes reaches every chunk's own post_capped_bytes call; a trip on
    one chunk raises GraphCapExceededError without a prior chunk's result
    being discarded silently -- the exception propagates instead."""
    seen_max_bytes: list[int | None] = []

    def fake_post_capped_bytes(self, path, payload, *, max_bytes, deadline=None):
        seen_max_bytes.append(max_bytes)
        var = payload["variable"]["dcids"][0]
        if var == "sv/1":
            raise GraphCapExceededError(
                "simulated cap trip", cap_kind="bytes", cap=max_bytes, observed=max_bytes + 1
            )
        return {"byVariable": {var: {"byEntity": {}}}}

    monkeypatch.setattr(GraphClient, "post_capped_bytes", fake_post_capped_bytes)
    client = GraphClient(base_url="http://fake/core/api", entities_with_data_variable_chunk_size=1)

    with pytest.raises(GraphCapExceededError):
        client.entities_with_data(["sv/0", "sv/1"], ["e0"], max_bytes=4096)

    assert seen_max_bytes == [4096, 4096]


# ---------------------------------------------------------------------------
# entities_with_data_by_variable
# ---------------------------------------------------------------------------


def test_entities_with_data_by_variable_posts_presence_only_select(monkeypatch) -> None:
    """The POST body uses the same lean entity+variable select with explicit entity.dcids."""
    seen_bodies: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_bodies.append(payload)
        return {
            "byVariable": {
                "sv/GDP": {"byEntity": {"country/GTM": {}, "country/USA": {}}},
            },
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.entities_with_data_by_variable(
        ["sv/GDP"], ["country/GTM", "country/USA", "country/VAT"]
    )

    assert seen_bodies[0]["select"] == ["entity", "variable"]
    assert seen_bodies[0]["variable"] == {"dcids": ["sv/GDP"]}
    assert seen_bodies[0]["entity"] == {"dcids": ["country/GTM", "country/USA", "country/VAT"]}
    assert result == {"sv/GDP": ["country/GTM", "country/USA"]}


def test_entities_with_data_by_variable_keeps_variables_apart(monkeypatch) -> None:
    """A candidate present for only one of several requested variables still
    appears under that variable's own key, never merged or dropped."""

    def fake_post(self, path, payload, *, deadline=None):
        return {
            "byVariable": {
                "sv/A": {"byEntity": {"country/GTM": {}, "country/USA": {}}},
                "sv/B": {"byEntity": {"country/GTM": {}}},
            },
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.entities_with_data_by_variable(["sv/A", "sv/B"], ["country/GTM", "country/USA"])
    assert result == {"sv/A": ["country/GTM", "country/USA"], "sv/B": ["country/GTM"]}


def test_entities_with_data_by_variable_missing_variable_key_is_empty(monkeypatch) -> None:
    """A variable entirely absent from byVariable maps to [], never fabricated
    as having the same presence as a sibling variable."""

    def fake_post(self, path, payload, *, deadline=None):
        return {"byVariable": {"sv/A": {"byEntity": {"country/GTM": {}}}}}  # sv/B missing

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.entities_with_data_by_variable(["sv/A", "sv/B"], ["country/GTM"])
    assert result == {"sv/A": ["country/GTM"], "sv/B": []}


def test_entities_with_data_by_variable_empty_inputs_no_post(monkeypatch) -> None:
    post_called = []

    def fake_post(self, path, payload, *, deadline=None):
        post_called.append(payload)
        return {}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    assert client.entities_with_data_by_variable([], ["country/GTM"]) == {}
    assert client.entities_with_data_by_variable(["sv/GDP"], []) == {}
    assert post_called == []


def test_entities_with_data_by_variable_raises_on_upstream_error(monkeypatch) -> None:
    def handler(request):
        return httpx.Response(503, json={})

    client = GraphClient(base_url="http://fake/core/api", transport=httpx.MockTransport(handler))

    with pytest.raises(GraphReadError):
        client.entities_with_data_by_variable(["sv/GDP"], ["country/GTM"])


def test_entities_with_data_by_variable_passes_its_deadline_to_the_post(monkeypatch) -> None:
    seen_deadlines: list[float | None] = []

    def fake_post(self, path, payload, *, deadline=None):
        seen_deadlines.append(deadline)
        return {"byVariable": {}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    client.entities_with_data_by_variable(["sv/GDP"], ["country/GTM"], deadline=12.5)
    client.entities_with_data_by_variable(["sv/GDP"], ["country/GTM"])

    assert seen_deadlines == [12.5, None]
