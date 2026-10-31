"""Pins paging and error handling for `enum_members`'s `<-typeOf` hop at the 500-member page cap.
The fake graph models `->*` range lookup followed by paged `<-typeOf` reads."""

import pytest

from dc_graph import GraphClient, GraphNotAdvancingError, GraphReadError

_PAGE_SIZE = 500


def _rangeincludes_body(class_dcids: list[str]) -> dict:
    """Build the `->*` response listing a property's range classes."""
    return {
        "data": {
            "prop1": {"arcs": {"rangeIncludes": {"nodes": [{"dcid": d} for d in class_dcids]}}}
        }
    }


def test_enum_members_over_cap_class_returns_every_member(monkeypatch) -> None:
    """Paging returns all 1,200 members across three 500-member pages."""
    class_members = [f"member/{i}" for i in range(1200)]
    typeof_calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        prop = payload["property"]
        if prop == "->*":
            assert payload["nodes"] == ["prop1"]
            return _rangeincludes_body(["class1"])
        assert prop == "<-typeOf"
        assert payload["nodes"] == ["class1"]
        typeof_calls.append(dict(payload))
        token = payload.get("nextToken")
        start = 0 if token is None else int(token)
        page = class_members[start : start + _PAGE_SIZE]
        end = start + len(page)
        body: dict = {
            "data": {"class1": {"arcs": {"typeOf": {"nodes": [{"dcid": d} for d in page]}}}}
        }
        if end < len(class_members):
            body["nextToken"] = str(end)
        return body

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.enum_members("prop1")

    assert result == class_members
    assert len(typeof_calls) == 3, "1,200 members at 500/page must take 3 pages, not 1"


def test_enum_members_small_class_unchanged(monkeypatch) -> None:
    """A class well under the page cap resolves in one <-typeOf POST, with no
    nextToken handling triggered -- the ordinary case is untouched."""
    typeof_calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        prop = payload["property"]
        if prop == "->*":
            return _rangeincludes_body(["class1"])
        assert prop == "<-typeOf"
        assert "nextToken" not in payload
        typeof_calls.append(dict(payload))
        return {
            "data": {
                "class1": {
                    "arcs": {"typeOf": {"nodes": [{"dcid": "member/a"}, {"dcid": "member/b"}]}}
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.enum_members("prop1")

    assert result == ["member/a", "member/b"]
    assert len(typeof_calls) == 1


@pytest.mark.parametrize(
    "tokens",
    [
        pytest.param(["STUCK", "STUCK"], id="consecutive-repeated-token"),
        pytest.param(["A", "B", "A"], id="cycling-token"),
    ],
)
def test_enum_members_repeated_token_raises(monkeypatch, tokens) -> None:
    """Consecutive repeats and cycles raise instead of returning partial results."""
    calls = 0

    def fake_post(self, path, payload, *, deadline=None):
        nonlocal calls
        prop = payload["property"]
        if prop == "->*":
            return _rangeincludes_body(["class1"])
        assert prop == "<-typeOf"
        if calls == len(tokens):
            pytest.fail("enum_members requested a page after the repeated token")
        token = tokens[calls]
        calls += 1
        return {
            "data": {"class1": {"arcs": {"typeOf": {"nodes": [{"dcid": "member/a"}]}}}},
            "nextToken": token,
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    with pytest.raises(GraphReadError, match="repeated a nextToken"):
        client.enum_members("prop1")
    assert calls == len(tokens)


def test_enum_members_max_pages_exceeded_raises(monkeypatch) -> None:
    """A token still present after _ENUM_MEMBERS_TYPEOF_MAX_PAGES pages raises
    rather than returning a partial member list -- guards a graph that keeps
    advancing forever without ever finishing."""
    import dc_graph.client as graph_module

    def fake_post(self, path, payload, *, deadline=None):
        prop = payload["property"]
        if prop == "->*":
            return _rangeincludes_body(["class1"])
        assert prop == "<-typeOf"
        token = payload.get("nextToken")
        next_i = 0 if token is None else int(token) + 1
        return {
            "data": {"class1": {"arcs": {"typeOf": {"nodes": [{"dcid": f"member/{next_i}"}]}}}},
            "nextToken": str(next_i),
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    with pytest.raises(GraphReadError, match="still reported a nextToken") as raised:
        client.enum_members("prop1")
    assert isinstance(raised.value.__cause__, GraphNotAdvancingError)
    assert graph_module._ENUM_MEMBERS_TYPEOF_MAX_PAGES > 0
