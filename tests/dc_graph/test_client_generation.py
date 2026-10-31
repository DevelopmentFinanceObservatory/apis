"""Pins generation-keyed cache hits, expiry, in-flight reads, stats, and clearing.
The fake transport keeps every test offline."""

from __future__ import annotations

import threading

from dc_graph import GraphClient
from graph_generation import FakeGeneration
from graph_generation.stamp import UNKNOWN_KEY, GenerationStamp


def _fake_post_arcs(dcids: list[str], *, arc: str, target: str) -> dict:
    return {"data": {d: {"arcs": {arc: {"nodes": [{"dcid": target}]}}} for d in dcids}}


def _fake_post_name(dcids: list[str], *, label: str) -> dict:
    return {"data": {d: {"arcs": {"name": {"nodes": [{"value": label}]}}} for d in dcids}}


# ---------------------------------------------------------------------------
# Hit under the same token, miss after advance(), one per cache family.
# ---------------------------------------------------------------------------


def test_label_cache_hit_under_same_token_miss_after_advance(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return _fake_post_name(payload["nodes"], label="Kenya")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    assert client.node_label("country/KEN") == "Kenya"
    assert client.node_label("country/KEN") == "Kenya"
    assert len(calls) == 1, "the same token must serve the second call from cache"

    generation.advance("gen-b")
    assert client.node_label("country/KEN") == "Kenya"
    assert len(calls) == 2, "a token change must miss and re-fetch"


def test_arc_targets_cache_holds_answered_nodes_under_the_token(monkeypatch) -> None:
    """A node answered with arcs is served from the label-sized cache under the same token,
    whatever the order of `properties`; an empty answer and a token change read again."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(sorted(payload["nodes"]))
        return {
            "data": {
                d: {} if d == "dc/base/Gone" else _fake_post_name([d], label="WHO")["data"][d]
                for d in payload["nodes"]
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api", generation=generation, label_cache_size=100
    )

    first = client.node_arc_targets_many(
        ["dc/base/WHO_GHO", "dc/base/Gone"], properties=("name", "typeOf")
    )
    second = client.node_arc_targets_many(
        ["dc/base/Gone", "dc/base/WHO_GHO"], properties=("typeOf", "name")
    )

    assert second == first
    assert first["dc/base/Gone"] is None
    assert calls == [["dc/base/Gone", "dc/base/WHO_GHO"], ["dc/base/Gone"]]
    stats = client.stats()
    assert (stats["arc_targets_hits"], stats["arc_targets_misses"]) == (1, 3)
    assert stats["arc_targets_entries"] == 1

    generation.advance("gen-b")
    client.node_arc_targets_many(["dc/base/WHO_GHO"], properties=("name", "typeOf"))
    assert calls[-1] == ["dc/base/WHO_GHO"]
    client.clear()
    assert client.stats()["arc_targets_entries"] == 0


def test_type_cache_hit_under_same_token_miss_after_advance(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return _fake_post_arcs(payload["nodes"], arc="typeOf", target="Country")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    assert client.node_type("country/KEN") == "Country"
    assert client.node_type("country/KEN") == "Country"
    assert len(calls) == 1

    generation.advance("gen-b")
    assert client.node_type("country/KEN") == "Country"
    assert len(calls) == 2


def test_arcs_cache_hit_under_same_token_miss_after_advance(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload.get("nodes", []))
        calls.append(nodes)
        return _fake_post_arcs(nodes, arc="typeOf", target="Thing")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_arcs_batch(["country/KEN"])
    client.node_arcs_batch(["country/KEN"])
    assert len(calls) == 1

    generation.advance("gen-b")
    client.node_arcs_batch(["country/KEN"])
    assert len(calls) == 2


def test_observation_properties_cache_hit_under_same_token_miss_after_advance(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload["nodes"])
        calls.append(nodes)
        return _fake_post_arcs(nodes, arc="observationProperties", target="observationAbout")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_observation_properties_batch(["sv/A"])
    client.node_observation_properties_batch(["sv/A"])
    assert len(calls) == 1

    generation.advance("gen-b")
    client.node_observation_properties_batch(["sv/A"])
    assert len(calls) == 2


def test_earth_countries_cache_hit_under_same_token_miss_after_advance(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(list(payload["nodes"]))
        return {
            "data": {"Earth": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/KEN"}]}}}}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.child_places_batch(["Earth"], "Country")
    client.child_places_batch(["Earth"], "Country")
    assert len(calls) == 1

    generation.advance("gen-b")
    client.child_places_batch(["Earth"], "Country")
    assert len(calls) == 2


def test_currency_units_cache_hit_under_same_token_miss_after_advance(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        if payload["property"] == "<-typeOf":
            return {
                "data": {"CurrencyUnit": {"arcs": {"typeOf": {"nodes": [{"dcid": "CurrencyUSD"}]}}}}
            }
        return {
            "data": {"CurrencyUSD": {"arcs": {"iso4217AlphaCode": {"nodes": [{"value": "USD"}]}}}}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.currency_units()
    client.currency_units()
    assert len(calls) == 2, "the two hops of the ONE read; the second call issues none"

    generation.advance("gen-b")
    client.currency_units()
    assert len(calls) == 4, "a token change must miss and re-run both hops"


# ---------------------------------------------------------------------------
# Unknown and known stamps never share an entry.
# ---------------------------------------------------------------------------


def test_unknown_and_known_stamps_never_share_an_arcs_cache_entry(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload.get("nodes", []))
        calls.append(nodes)
        return _fake_post_arcs(nodes, arc="typeOf", target="Thing")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration()  # starts unknown
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_arcs_batch(["country/KEN"])
    assert len(calls) == 1

    # FakeGeneration.advance() from an initial None token fires no clear()
    # callback (a first-known-token transition, not a change between two
    # known tokens). The cache key still differs (UNKNOWN_KEY vs "gen-a"),
    # so the lookup misses on its own.
    generation.advance("gen-a")
    client.node_arcs_batch(["country/KEN"])
    assert len(calls) == 2, "a known stamp must never read an unknown-keyed entry"

    # Reverting to unknown, the original unknown-keyed entry, if not yet
    # expired or evicted, would still be a hit. This proves the unknown
    # entry was never overwritten by the known-token write above.
    generation.set_unknown()
    client.node_arcs_batch(["country/KEN"])
    assert len(calls) == 2, "the original unknown-keyed entry must still be live"


def test_unknown_key_literal_never_collides_with_a_real_token_named_unknown(monkeypatch) -> None:
    """The literal token `unknown` and `None` share a key but have different cache lifetimes."""
    known_as_literal_unknown = GenerationStamp(token=UNKNOWN_KEY)
    genuinely_unknown = GenerationStamp(token=None)
    assert known_as_literal_unknown.key == genuinely_unknown.key == UNKNOWN_KEY
    assert known_as_literal_unknown.max_age_s != genuinely_unknown.max_age_s


# ---------------------------------------------------------------------------
# Entries expire at the `max_age_s` of their captured stamp.
# ---------------------------------------------------------------------------


def test_arcs_cache_entry_expires_after_its_own_stamp_max_age_s(monkeypatch) -> None:
    calls: list[list[str]] = []
    fake_now = [0.0]

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload.get("nodes", []))
        calls.append(nodes)
        return _fake_post_arcs(nodes, arc="typeOf", target="Thing")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        timer=lambda: fake_now[0],
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_arcs_batch(["country/KEN"])
    max_age_s = generation.stamp().max_age_s
    fake_now[0] += max_age_s + 1.0
    client.node_arcs_batch(["country/KEN"])

    assert len(calls) == 2, "an entry must expire once its own stamp's max_age_s has elapsed"


def test_arcs_cache_entry_survives_just_under_its_own_stamp_max_age_s(monkeypatch) -> None:
    """Known-good control for the expiry test above. A read one second
    before max_age_s elapses must still be a cache hit."""
    calls: list[list[str]] = []
    fake_now = [0.0]

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload.get("nodes", []))
        calls.append(nodes)
        return _fake_post_arcs(nodes, arc="typeOf", target="Thing")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        timer=lambda: fake_now[0],
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_arcs_batch(["country/KEN"])
    max_age_s = generation.stamp().max_age_s
    fake_now[0] += max_age_s - 1.0
    client.node_arcs_batch(["country/KEN"])

    assert len(calls) == 1, "an entry must still be live just under its own stamp's max_age_s"


def test_unknown_generation_entry_has_the_shorter_max_age(monkeypatch) -> None:
    """An entry captured while the generation is unknown expires on the
    shorter unknown-generation lifetime."""
    calls: list[list[str]] = []
    fake_now = [0.0]

    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload.get("nodes", []))
        calls.append(nodes)
        return _fake_post_arcs(nodes, arc="typeOf", target="Thing")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration()  # unknown
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        timer=lambda: fake_now[0],
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_arcs_batch(["country/KEN"])
    unknown_max_age_s = generation.stamp().max_age_s
    known_max_age_s = GenerationStamp(token="gen-a").max_age_s
    assert unknown_max_age_s < known_max_age_s

    fake_now[0] += unknown_max_age_s + 1.0
    client.node_arcs_batch(["country/KEN"])

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Each generation uses its own in-flight fetch.
# ---------------------------------------------------------------------------


def test_a_post_flip_caller_never_joins_a_pre_flip_inflight_fetch(monkeypatch) -> None:
    """In-flight reads are keyed by generation and dcid. A post-change caller starts its own POST."""
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api",
        generation=generation,
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    call_count = [0]
    lock = threading.Lock()
    first_post_started = threading.Event()
    release_first_post = threading.Event()
    posted_chunks: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        chunk = list(payload["nodes"])
        with lock:
            posted_chunks.append(chunk)
            call_count[0] += 1
            is_first = call_count[0] == 1
        if is_first:
            first_post_started.set()
            released = release_first_post.wait(timeout=10.0)
            assert released, "test failed to release the pre-flip leader before its own timeout"
        return _fake_post_arcs(chunk, arc="observationProperties", target="role/X")

    monkeypatch.setattr(GraphClient, "post", fake_post)

    dcids = ["sv/A"]
    leader_result: dict[str, object] = {}

    def _leader() -> None:
        leader_result["result"] = client.node_observation_properties_batch(dcids)

    leader_thread = threading.Thread(target=_leader)
    leader_thread.start()
    assert first_post_started.wait(timeout=5.0), "the pre-flip leader never started its POST"

    # The token moves on WHILE the leader's fetch for "sv/A" is still in
    # flight, registered under the OLD stamp's key.
    generation.advance("gen-b")

    # A caller arriving now, under the new stamp, must become its own leader
    # for (new_stamp.key, "sv/A"). The join is bounded by a timeout so a
    # regression (incorrectly joining the pre-flip Future, which nothing
    # here ever releases) fails this test rather than hanging the suite.
    post_flip_result: dict[str, object] = {}

    def _post_flip_caller() -> None:
        post_flip_result["result"] = client.node_observation_properties_batch(dcids)

    post_flip_thread = threading.Thread(target=_post_flip_caller)
    post_flip_thread.start()
    post_flip_thread.join(timeout=5.0)

    assert not post_flip_thread.is_alive(), (
        "a post-flip caller must never block on a pre-flip in-flight fetch"
    )
    assert post_flip_result["result"] == {"sv/A": ("role/X",)}
    assert len(posted_chunks) == 2, (
        f"a post-flip caller must issue its own POST rather than join the "
        f"pre-flip in-flight fetch; got {posted_chunks!r}"
    )

    release_first_post.set()
    leader_thread.join(timeout=5.0)
    assert not leader_thread.is_alive(), "the pre-flip leader hung after being released"
    assert leader_result["result"] == {"sv/A": ("role/X",)}


# ---------------------------------------------------------------------------
# stats() / clear()
# ---------------------------------------------------------------------------


def test_stats_reports_hits_misses_and_entries(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload.get("nodes", []))
        return _fake_post_arcs(nodes, arc="typeOf", target="Thing")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api",
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_arcs_batch(["country/KEN"])  # miss
    client.node_arcs_batch(["country/KEN"])  # hit

    stats = client.stats()
    assert stats["arcs_misses"] == 1
    assert stats["arcs_hits"] == 1
    assert stats["arcs_entries"] == 1


def test_clear_empties_every_cache_and_the_inflight_registry(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        nodes = list(payload.get("nodes", []))
        return _fake_post_arcs(nodes, arc="typeOf", target="Thing")

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api",
        label_cache_size=100,
        type_cache_size=100,
        arcs_cache_size=100,
        obs_properties_cache_size=100,
        single_value_cache_enabled=True,
    )

    client.node_label("country/KEN")
    client.node_type("country/KEN")
    client.node_arcs_batch(["country/KEN"])
    client.node_observation_properties_batch(["country/KEN"])

    client.clear()

    stats = client.stats()
    assert stats["label_entries"] == 0
    assert stats["type_entries"] == 0
    assert stats["arcs_entries"] == 0
    assert stats["observation_properties_entries"] == 0
    assert stats["observation_properties_inflight"] == 0
    assert stats["earth_countries_entries"] == 0
    assert stats["currency_units_entries"] == 0
    assert client._earth_countries_cache is None
    assert client._currency_units_cache is None
