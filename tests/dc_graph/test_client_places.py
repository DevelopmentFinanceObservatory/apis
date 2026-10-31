"""Tests arc reads, name resolution, and child-place caching with a fake transport."""

import httpx
import pytest

import dc_graph.client as graph_module
from dc_graph import GraphClient, GraphNotAdvancingError, GraphOmittedDcidsError, GraphReadError
from dc_graph.client import UNRESOLVED_NODE
from entity_resolution import GraphLookup
from graph_generation import FakeGeneration
from graph_generation.stamp import UNKNOWN_STAMP
from materialise.place_scope import DefaultScopeUnknownError, earth_countries


def test_graph_client_satisfies_graph_lookup_protocol() -> None:
    assert isinstance(GraphClient(base_url="http://fake/core/api"), GraphLookup)


def test_node_arcs_many_single_property_request_shape(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        assert payload == {"nodes": ["country/KEN"], "property": "->typeOf"}

        return {"data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    assert client.node_arcs_many(["country/KEN"], properties=("typeOf",)) == {
        "country/KEN": {"typeOf": ("Country",)}
    }


def test_node_arcs_many_multi_property_request_shape(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        assert payload == {"nodes": ["country/KEN"], "property": "->[typeOf,name]"}

        return {
            "data": {
                "country/KEN": {
                    "arcs": {
                        "typeOf": {"nodes": [{"dcid": "Country"}]},
                        "name": {"nodes": [{"value": "Kenya"}]},
                    }
                }
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    assert client.node_arcs_many(["country/KEN"], properties=("typeOf", "name")) == {
        "country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}
    }


def test_node_arcs_many_single_dcid_present_but_empty_trusted_as_none(monkeypatch) -> None:
    """A lone present-but-empty node returns `None` without retry."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1

        return {"data": {"does/not/exist": {}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    assert client.node_arcs_many(["does/not/exist"], properties=("typeOf",)) == {
        "does/not/exist": None
    }
    assert call_count[0] == 1


def test_node_arcs_many_multi_dcid_empty_entry_retries_then_resolves(monkeypatch) -> None:
    """A present-but-empty entry in a multi-dcid batch is retried alone."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload["nodes"])

        if len(calls) == 1:
            return {
                "data": {
                    "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                    "country/ETH": {},
                }
            }
        return {"data": {"country/ETH": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_arcs_many(["country/KEN", "country/ETH"], properties=("typeOf",))
    assert result == {
        "country/KEN": {"typeOf": ("Country",)},
        "country/ETH": {"typeOf": ("Country",)},
    }
    assert calls == [["country/KEN", "country/ETH"], ["country/ETH"]]


def test_node_arcs_many_dcid_omitted_from_retry_raises(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    with pytest.raises(GraphReadError) as raised:
        client.node_arcs_many(["country/KEN", "country/ETH"], properties=("typeOf",))
    assert isinstance(raised.value.__cause__, GraphOmittedDcidsError)
    assert raised.value.__cause__.dcids == ("country/KEN", "country/ETH")


def test_node_arcs_many_group_retry_false_empty_caught_by_individual_probe(monkeypatch) -> None:
    """A second empty response triggers an individual probe."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload["nodes"])

        if len(calls) == 1:
            return {
                "data": {
                    "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                    "country/ETH": {},
                }
            }
        if len(calls) == 2:
            return {"data": {"country/ETH": {}}}
        return {"data": {"country/ETH": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_arcs_many(["country/KEN", "country/ETH"], properties=("typeOf",))
    assert result == {
        "country/KEN": {"typeOf": ("Country",)},
        "country/ETH": {"typeOf": ("Country",)},
    }
    assert calls == [["country/KEN", "country/ETH"], ["country/ETH"], ["country/ETH"]]


def test_node_arcs_many_still_empty_after_individual_probe_resolves_to_none(monkeypatch) -> None:
    """Three empty responses resolve the suspect to `None`."""
    calls: list[list[str]] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload["nodes"])

        if len(calls) == 1:
            return {
                "data": {
                    "country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}},
                    "country/ETH": {},
                }
            }
        return {"data": {"country/ETH": {}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_arcs_many(["country/KEN", "country/ETH"], properties=("typeOf",))
    assert result == {
        "country/KEN": {"typeOf": ("Country",)},
        "country/ETH": None,
    }
    assert calls == [["country/KEN", "country/ETH"], ["country/ETH"], ["country/ETH"]]


def test_node_arcs_many_multi_page_response_stitched(monkeypatch) -> None:
    """Values from `nextToken` pages concatenate in page order."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        if len(calls) == 1:
            assert "nextToken" not in payload
            return {
                "data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "Country"}]}}}},
                "nextToken": "tok1",
            }
        assert payload["nextToken"] == "tok1"
        return {"data": {"country/KEN": {"arcs": {"typeOf": {"nodes": [{"dcid": "SecondType"}]}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.node_arcs_many(["country/KEN"], properties=("typeOf",))
    assert result == {"country/KEN": {"typeOf": ("Country", "SecondType")}}
    assert len(calls) == 2


def test_node_arcs_many_page_budget_exhausted_raises(monkeypatch) -> None:
    # Each page carries a distinct token and no new dcids, so neither the
    # repeated-token check nor natural exhaustion ever fires. Only the page
    # budget can stop it. The page budget is lowered to 2 and exactly 2
    # responses are queued, so an undetected regression that keeps paging
    # exhausts the queue and fails with IndexError instead of hanging.
    monkeypatch.setattr(graph_module, "_NODE_ARCS_MANY_MAX_PAGES", 2)
    responses = [
        {
            "data": {"country/KEN": {"arcs": {"typeOf": {"nodes": []}}}},
            "nextToken": "tok1",
        },
        {
            "data": {"country/KEN": {"arcs": {"typeOf": {"nodes": []}}}},
            "nextToken": "tok2",
        },
    ]
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        return responses[len(calls) - 1]

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    with pytest.raises(GraphReadError) as raised:
        client.node_arcs_many(["country/KEN"], properties=("typeOf",))
    assert len(calls) == 2
    # The same read pages the same way again, which the cause marks.
    assert isinstance(raised.value.__cause__, GraphNotAdvancingError)


def test_node_arcs_many_empty_dcids_returns_empty_dict_without_network_call(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        raise AssertionError("must not POST for an empty dcids list")

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    assert client.node_arcs_many([], properties=("typeOf",)) == {}


def test_resolve_by_arc_request_shape_and_known_names(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        assert payload == {
            "nodes": ["Kenya", "Atlantis"],
            "property": "<-description{typeOf:Country}->dcid",
        }

        return {"entities": [{"node": "Kenya", "candidates": [{"dcid": "country/KEN"}]}]}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.resolve_by_arc(["Kenya", "Atlantis"], arc="<-description{typeOf:Country}->dcid")
    assert result == {"Kenya": ["country/KEN"], "Atlantis": []}


def test_resolve_by_arc_multiple_candidates_preserve_mixer_order(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "entities": [
                {
                    "node": "Congo",
                    "candidates": [{"dcid": "country/COD"}, {"dcid": "country/COG"}],
                }
            ]
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    assert client.resolve_by_arc(["Congo"], arc="<-description->dcid") == {
        "Congo": ["country/COD", "country/COG"]
    }


def test_resolve_by_arc_empty_names_returns_empty_dict_without_network_call(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        raise AssertionError("must not POST for an empty names list")

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    assert client.resolve_by_arc([], arc="<-description->dcid") == {}


def test_child_places_batch_known_and_absent_parent(monkeypatch) -> None:
    """Large multi-parent closures time out on the dev mixer, so each parent gets a POST."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        assert payload is not None
        assert payload["property"] == "<-containedInPlace+{typeOf:Country}"
        parent = payload["nodes"][0]

        if parent == "africa":
            return {
                "data": {
                    "africa": {
                        "arcs": {
                            # The "+" transitive-closure marker is preserved
                            # verbatim in the response arc key (confirmed live).
                            "containedInPlace+": {
                                "nodes": [
                                    {"dcid": "country/KEN"},
                                    {"dcid": "country/ETH"},
                                ]
                            }
                        }
                    }
                }
            }
        return {"data": {}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.child_places_batch(["africa", "absent/PLACE"], "Country")
    assert result["africa"] == ["country/KEN", "country/ETH"]
    assert result["absent/PLACE"] == []
    assert [c["nodes"] for c in calls] == [["africa"], ["absent/PLACE"]]


def test_child_places_batch_multi_page_response_stitched_per_parent(monkeypatch) -> None:
    """Each parent's pages concatenate in order, with a separate request per parent."""
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        if payload["nodes"] == ["africa"] and "nextToken" not in payload:
            return {
                "data": {
                    "africa": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/KEN"}]}}}
                },
                "nextToken": "tok1",
            }
        if payload["nodes"] == ["africa"] and payload.get("nextToken") == "tok1":
            return {
                "data": {
                    "africa": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/ETH"}]}}}
                }
            }
        return {
            "data": {
                "SubSaharanAfrica": {
                    "arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/NGA"}]}}
                }
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    result = client.child_places_batch(["africa", "SubSaharanAfrica"], "Country")
    assert result == {
        "africa": ["country/KEN", "country/ETH"],
        "SubSaharanAfrica": ["country/NGA"],
    }
    assert len(calls) == 3


def test_child_places_batch_page_budget_exhausted_raises(monkeypatch) -> None:
    monkeypatch.setattr(graph_module, "_NODE_ARCS_MANY_MAX_PAGES", 2)
    responses = [
        {"data": {"africa": {"arcs": {"containedInPlace+": {"nodes": []}}}}, "nextToken": "tok1"},
        {"data": {"africa": {"arcs": {"containedInPlace+": {"nodes": []}}}}, "nextToken": "tok2"},
    ]
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        return responses[len(calls) - 1]

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    with pytest.raises(GraphReadError):
        client.child_places_batch(["africa"], "Country")
    assert len(calls) == 2


def test_child_places_batch_repeated_token_raises(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {"africa": {"arcs": {"containedInPlace+": {"nodes": []}}}},
            "nextToken": "tok1",
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)
    with pytest.raises(GraphReadError):
        client.child_places_batch(["africa"], "Country")


def test_child_places_batch_empty_parent_list_returns_empty() -> None:
    client = GraphClient(base_url="http://fake/core/api")
    assert client.child_places_batch([], "Country") == {}


def test_child_places_batch_earth_countries_cache_hit_skips_post(monkeypatch) -> None:
    """A second child_places_batch(["Earth"], "Country") call within the TTL
    returns the memoised country list without issuing another POST."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return {
            "data": {
                "Earth": {
                    "arcs": {
                        "containedInPlace+": {
                            "nodes": [{"dcid": "country/KEN"}, {"dcid": "country/USA"}]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    first = client.child_places_batch(["Earth"], "Country")
    second = client.child_places_batch(["Earth"], "Country")

    assert first == {"Earth": ["country/KEN", "country/USA"]}
    assert second == {"Earth": ["country/KEN", "country/USA"]}
    assert call_count[0] == 1


def test_child_places_batch_reads_again_after_an_earth_listing_that_names_no_country(
    monkeypatch,
) -> None:
    """Earth holds every country, so a listing naming none is a failed read and is not
    memoised: the next call reads again, and the listing it gets is the one memoised."""
    answers = iter([[], ["country/KEN"]])
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        nodes = [{"dcid": dcid} for dcid in next(answers)]
        return {"data": {"Earth": {"arcs": {"containedInPlace+": {"nodes": nodes}}}}}

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    with pytest.raises(DefaultScopeUnknownError):
        earth_countries(client)
    assert earth_countries(client) == ("country/KEN",)
    assert earth_countries(client) == ("country/KEN",)
    assert call_count[0] == 2


def test_child_places_batch_earth_countries_cache_expires_after_generation_max_age(
    monkeypatch,
) -> None:
    """The memo expires after the stamp's `max_age_s` when its generation is unknown."""
    call_count = [0]
    fake_now = [1_000.0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return {
            "data": {"Earth": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/KEN"}]}}}}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(
        base_url="http://fake/core/api", timer=lambda: fake_now[0], single_value_cache_enabled=True
    )

    client.child_places_batch(["Earth"], "Country")
    fake_now[0] += UNKNOWN_STAMP.max_age_s + 1.0
    client.child_places_batch(["Earth"], "Country")

    assert call_count[0] == 2


def test_child_places_batch_earth_countries_cache_expires_on_generation_change(
    monkeypatch,
) -> None:
    """Changing the generation token invalidates the memo before `max_age_s`."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        return {
            "data": {"Earth": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/KEN"}]}}}}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    generation = FakeGeneration(token="gen-a")
    client = GraphClient(
        base_url="http://fake/core/api", generation=generation, single_value_cache_enabled=True
    )

    client.child_places_batch(["Earth"], "Country")
    generation.advance("gen-b")
    client.child_places_batch(["Earth"], "Country")

    assert call_count[0] == 2


def test_child_places_batch_non_earth_pair_bypasses_earth_countries_cache(monkeypatch) -> None:
    """Other parent/type pairs bypass the Earth-countries memo."""
    call_count = [0]

    def fake_post(self, path, payload, *, deadline=None):
        call_count[0] += 1
        parent = payload["nodes"][0]
        return {
            "data": {parent: {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/KEN"}]}}}}
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api", single_value_cache_enabled=True)

    client.child_places_batch(["africa"], "Country")
    client.child_places_batch(["Earth"], "State")

    assert call_count[0] == 2
    assert client._earth_countries_cache is None


def test_child_places_batch_dedups_a_child_declared_by_two_provenances(monkeypatch) -> None:
    # (#698) A child place asserted via ->containedInPlace by two
    # provenances reaches this read as two arc entries for the same dcid;
    # a genuinely distinct sibling survives once, undisturbed by the dedup.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "africa": {
                    "arcs": {
                        "containedInPlace+": {
                            "nodes": [
                                {"dcid": "country/KEN"},
                                {"dcid": "country/KEN"},
                                {"dcid": "country/NGA"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.child_places_batch(["africa"], "Country")

    assert result == {"africa": ["country/KEN", "country/NGA"]}


def test_child_places_batch_dedups_a_child_repeated_across_pages(monkeypatch) -> None:
    responses = [
        {
            "data": {
                "africa": {"arcs": {"containedInPlace+": {"nodes": [{"dcid": "country/KEN"}]}}}
            },
            "nextToken": "tok1",
        },
        {
            "data": {
                "africa": {
                    "arcs": {
                        "containedInPlace+": {
                            "nodes": [{"dcid": "country/KEN"}, {"dcid": "country/NGA"}]
                        }
                    }
                }
            }
        },
    ]
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)
        return responses[len(calls) - 1]

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.child_places_batch(["africa"], "Country")

    assert result == {"africa": ["country/KEN", "country/NGA"]}
    assert len(calls) == 2


def test_child_dcids_dedups_a_child_declared_by_two_provenances(monkeypatch) -> None:
    # (#698) child_dcids reads <-isPartOf, the non-geographic sibling of
    # child_places_batch's <-containedInPlace+ read; a child declared by two
    # provenances collapses the same way.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "Enum": {
                    "arcs": {
                        "isPartOf": {
                            "nodes": [
                                {"dcid": "MemberA"},
                                {"dcid": "MemberA"},
                                {"dcid": "MemberB"},
                            ]
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(GraphClient, "post", fake_post)
    client = GraphClient(base_url="http://fake/core/api")

    result = client.child_dcids("Enum")

    assert result == ["MemberA", "MemberB"]


def test_node_names_many_requests_both_arcs_and_merges_in_order(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        assert payload == {"nodes": ["country/KEN"], "property": "->[name,alternateName]"}

        return {
            "data": {
                "country/KEN": {
                    "arcs": {
                        "name": {"nodes": [{"value": "Kenya"}]},
                        "alternateName": {"nodes": [{"value": "Republic of Kenya"}]},
                    }
                }
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)

    assert client.node_names_many(["country/KEN"]) == {
        "country/KEN": ("Kenya", "Republic of Kenya")
    }


def test_node_names_many_collapses_a_name_declared_by_a_doubled_provenance(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "country/KEN": {
                    "arcs": {
                        "name": {"nodes": [{"value": "Kenya"}, {"value": "Kenya"}]},
                        "alternateName": {"nodes": [{"value": "Kenya"}, {"value": "Republic"}]},
                    }
                }
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)

    assert client.node_names_many(["country/KEN"]) == {"country/KEN": ("Kenya", "Republic")}


def test_node_names_many_unknown_dcid_maps_to_empty_tuple(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {"country/ZZZ": {}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)

    assert client.node_names_many(["country/ZZZ"]) == {"country/ZZZ": ()}


def test_node_names_many_duplicate_requested_dcids_collapse(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(self, path, payload, *, deadline=None):
        calls.append(payload)

        return {"data": {"country/KEN": {"arcs": {"name": {"nodes": [{"value": "Kenya"}]}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)

    result = client.node_names_many(["country/KEN", "country/KEN"])

    assert result == {"country/KEN": ("Kenya",)}
    assert len(calls) == 1


def test_node_names_many_empty_dcids_returns_empty_dict_without_network_call(monkeypatch) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        raise AssertionError("must not POST for an empty dcids list")

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)

    assert client.node_names_many([]) == {}


def test_node_names_many_transport_error_raises_graph_read_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = GraphClient(base_url="http://fake/core/api", transport=httpx.MockTransport(handler))

    with pytest.raises(GraphReadError):
        client.node_names_many(["country/KEN"])


def test_node_names_many_drops_the_unresolved_target_marker_beside_a_real_name(monkeypatch) -> None:
    # node_arcs_many marks an arc entry carrying neither dcid nor value with
    # UNRESOLVED_NODE; that entry carries no name, so only the real value is
    # reported.
    def fake_post(self, path, payload, *, deadline=None):
        return {
            "data": {
                "country/KEN": {"arcs": {"name": {"nodes": [{"value": "Kenya"}, {"foo": "bar"}]}}}
            }
        }

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)

    result = client.node_names_many(["country/KEN"])

    assert result == {"country/KEN": ("Kenya",)}
    assert UNRESOLVED_NODE not in result["country/KEN"]


def test_node_names_many_maps_a_name_arc_of_only_the_unresolved_target_marker_to_empty(
    monkeypatch,
) -> None:
    def fake_post(self, path, payload, *, deadline=None):
        return {"data": {"country/KEN": {"arcs": {"name": {"nodes": [{"foo": "bar"}]}}}}}

    client = GraphClient(base_url="http://fake/core/api")
    monkeypatch.setattr(GraphClient, "post", fake_post)

    assert client.node_names_many(["country/KEN"]) == {"country/KEN": ()}
