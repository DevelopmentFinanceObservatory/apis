"""Tests for fallback-ladder order, name checks, and rejected normalisations."""

from __future__ import annotations

import json

import httpx

from dc_graph import GraphClient
from entity_resolution.graph import resolve_fallback

from .conftest import FakeGraphLookup


def _normalize(text: str) -> str:
    """Casefold and collapse whitespace while preserving punctuation."""
    return " ".join(text.split()).casefold()


def _geo_normalize(shared_resolver):
    """Return the composed resolver's geo-pack normaliser."""
    return shared_resolver.resolver.pack_normalizers["geo"].normalize


def _fallback(graph, texts):
    return resolve_fallback(graph, texts, normalize=_normalize)


def test_ladder_tries_arcs_in_order_and_stops_at_first_hit():
    graph = FakeGraphLookup(
        nodes={"continent/AMB": {"typeOf": ("Continent",), "name": ("Ambigua",)}},
        arcs={
            ("<-description{typeOf:Continent}->dcid", "Ambigua"): ["continent/AMB"],
            ("<-description{typeOf:UNGeoRegion}->dcid", "Ambigua"): ["region/AMB"],
        },
    )

    result = _fallback(graph, {0: "Ambigua"})

    assert result == {0: ("continent/AMB",)}
    arcs_tried = [arc for _, arc in graph.resolve_by_arc_calls]
    assert arcs_tried == [
        "<-description{typeOf:Country}->dcid",
        "<-description{typeOf:Continent}->dcid",
    ]


def test_ladder_makes_no_calls_when_there_are_no_misses():
    graph = FakeGraphLookup()

    result = _fallback(graph, {})

    assert result == {}
    assert graph.resolve_by_arc_calls == []
    assert graph.node_names_many_calls == []


def test_ladder_batches_one_call_per_arc_across_every_miss():
    """Each of four rungs gets both misses in one call. Only the bare rung reads names."""
    graph = FakeGraphLookup(
        nodes={"dcid/nl": {"name": ("Never Land",)}},
        arcs={("<-description->dcid", "Neverland"): ["dcid/nl"]},
    )

    result = _fallback(graph, {0: "Nowhere", 1: "Neverland"})

    assert result == {}
    assert len(graph.resolve_by_arc_calls) == 4
    for names, _arc in graph.resolve_by_arc_calls:
        assert set(names) == {"Nowhere", "Neverland"}
    assert graph.node_names_many_calls == [("dcid/nl",)]


def test_ladder_carries_every_dcid_onward_for_a_multi_candidate_arc():
    """Both dcids list Manytown. Neither lists the one-edit match Manyton."""
    graph = FakeGraphLookup(
        nodes={
            "dcid/first": {"name": ("Manytown",)},
            "dcid/second": {"alternateName": ("Manytown",)},
        },
        arcs={
            ("<-description->dcid", "Manytown"): ["dcid/first", "dcid/second"],
            ("<-description->dcid", "Manyton"): ["dcid/first", "dcid/second"],
        },
    )

    result = _fallback(graph, {0: "Manytown", 1: "Manyton"})

    assert result == {0: ("dcid/first", "dcid/second")}


def test_ladder_does_not_retry_a_later_rung_after_a_multi_candidate_hit():
    graph = FakeGraphLookup(
        nodes={
            "country/AAA": {"name": ("Manytown",)},
            "country/BBB": {"name": ("Manytown",)},
        },
        arcs={
            ("<-description{typeOf:Country}->dcid", "Manytown"): ["country/AAA", "country/BBB"],
            ("<-description{typeOf:Continent}->dcid", "Manytown"): ["continent/CCC"],
        },
    )

    result = _fallback(graph, {0: "Manytown"})

    assert result == {0: ("country/AAA", "country/BBB")}
    arcs_tried = [arc for _, arc in graph.resolve_by_arc_calls]
    assert arcs_tried == ["<-description{typeOf:Country}->dcid"]


def test_ladder_dedupes_identical_texts_across_indices_in_one_call():
    graph = FakeGraphLookup(
        nodes={"country/XYZ": {"name": ("Twinsville",)}},
        arcs={("<-description{typeOf:Country}->dcid", "Twinsville"): ["country/XYZ"]},
    )

    result = _fallback(graph, {0: "Twinsville", 1: "Twinsville"})

    assert result == {0: ("country/XYZ",), 1: ("country/XYZ",)}
    assert len(graph.resolve_by_arc_calls) == 1
    names, _arc = graph.resolve_by_arc_calls[0]
    assert names == ("Twinsville",)


def test_ladder_admits_a_text_the_answering_node_lists_as_a_name_or_an_alternate_name():
    """Canonical and alternate names resolve. Canonical Cty is unlisted."""
    graph = FakeGraphLookup(
        nodes={
            "geoId/canonical": {"name": ("Canonical City",)},
            "geoId/alias": {"name": ("Elsewhere",), "alternateName": ("Alias City",)},
        },
        arcs={
            ("<-description->dcid", "Canonical City"): ["geoId/canonical"],
            ("<-description->dcid", "Alias City"): ["geoId/alias"],
            ("<-description->dcid", "Canonical Cty"): ["geoId/canonical"],
        },
    )

    result = _fallback(graph, {0: "Canonical City", 1: "Alias City", 2: "Canonical Cty"})

    assert result == {0: ("geoId/canonical",), 1: ("geoId/alias",)}


def test_ladder_drops_a_fuzzy_description_match_the_answering_node_does_not_list():
    """The description arc answers Nigria, but the node lists only Nigeria."""
    graph = FakeGraphLookup(
        nodes={"country/NGA": {"typeOf": ("Country",), "name": ("Nigeria",)}},
        arcs={("<-description->dcid", "Nigria"): ["country/NGA"]},
    )

    result = _fallback(graph, {0: "Nigria"})

    assert result == {}


def test_ladder_keeps_hyphens_significant_between_text_and_listed_name():
    """The normaliser keeps hyphens significant, so only Guinea-Bissau is listed."""
    graph = FakeGraphLookup(
        nodes={"country/GNB": {"typeOf": ("Country",), "name": ("Guinea-Bissau",)}},
        arcs={
            ("<-description{typeOf:Country}->dcid", "Guinea-Bissau"): ["country/GNB"],
            ("<-description{typeOf:Country}->dcid", "Guinea Bissau"): ["country/GNB"],
        },
    )

    result = _fallback(graph, {0: "Guinea Bissau", 1: "Guinea-Bissau"})

    assert result == {1: ("country/GNB",)}


def test_ladder_moves_a_filtered_text_to_a_later_rung_that_lists_it():
    graph = FakeGraphLookup(
        nodes={
            "country/X": {"typeOf": ("Country",), "name": ("Not The Text",)},
            "geoId/later": {"name": ("Late Town",)},
        },
        arcs={
            ("<-description{typeOf:Country}->dcid", "Late Town"): ["country/X"],
            ("<-description{typeOf:UNGeoRegion}->dcid", "Late Town"): ["geoId/later"],
        },
    )

    result = _fallback(graph, {0: "Late Town"})

    assert result == {0: ("geoId/later",)}
    arcs_tried = [arc for _, arc in graph.resolve_by_arc_calls]
    assert arcs_tried == [
        "<-description{typeOf:Country}->dcid",
        "<-description{typeOf:Continent}->dcid",
        "<-description{typeOf:UNGeoRegion}->dcid",
    ]


def test_ladder_drops_a_dcid_carrying_no_listed_names():
    graph = FakeGraphLookup(
        nodes={"dcid/nameless": {"typeOf": ("Country",)}},
        arcs={("<-description->dcid", "Nameless"): ["dcid/nameless"]},
    )

    result = _fallback(graph, {0: "Nameless"})

    assert result == {}


def test_ladder_reads_node_names_once_per_rung_that_answered_and_never_for_an_empty_rung():
    graph = FakeGraphLookup(
        nodes={"geoId/lagos": {"name": ("Lagos State",), "alternateName": ("Lagos",)}},
        arcs={
            ("<-description{typeOf:Continent}->dcid", "Lagos"): ["geoId/lagos"],
            ("<-description{typeOf:UNGeoRegion}->dcid", "Lagos"): ["geoId/lagos"],
        },
    )

    result = _fallback(graph, {0: "Lagos"})

    assert result == {0: ("geoId/lagos",)}
    assert graph.node_names_many_calls == [("geoId/lagos",)]


def test_ladder_refuses_a_text_it_cannot_normalise_and_answers_the_rest(shared_resolver):
    """The normaliser refuses ">" without raising or suppressing Nigeria's result."""
    graph = FakeGraphLookup(
        nodes={"country/NGA": {"typeOf": ("Country",), "name": ("Nigeria",)}},
        arcs={
            ("<-description->dcid", "Nigeria"): ["country/NGA"],
            ("<-description->dcid", ">"): ["country/NGA"],
        },
    )

    result = resolve_fallback(
        graph, {0: "Nigeria", 1: ">"}, normalize=_geo_normalize(shared_resolver)
    )

    assert result == {0: ("country/NGA",)}


def test_ladder_refuses_a_listed_name_it_cannot_normalise_without_raising(shared_resolver):
    """Nigeria resolves when the normaliser rejects another listed name."""
    graph = FakeGraphLookup(
        nodes={"country/NGA": {"typeOf": ("Country",), "name": (">", "Nigeria")}},
        arcs={("<-description->dcid", "Nigeria"): ["country/NGA"]},
    )

    result = resolve_fallback(graph, {0: "Nigeria"}, normalize=_geo_normalize(shared_resolver))

    assert result == {0: ("country/NGA",)}


def test_resolve_fallback_returns_distinct_dcids_per_name_against_a_dual_provenance_response():
    # (#698) GraphClient deduplicates the duplicate candidate before fallback sees it.
    resolve_body = json.dumps(
        {
            "entities": [
                {
                    "node": "Kenya",
                    "candidates": [{"dcid": "country/KEN"}, {"dcid": "country/KEN"}],
                }
            ]
        }
    )
    node_body = json.dumps(
        {"data": {"country/KEN": {"arcs": {"name": {"nodes": [{"value": "Kenya"}]}}}}}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        content = resolve_body if request.url.path.endswith("/v2/resolve") else node_body
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})

    graph = GraphClient(
        base_url="https://graph.example.org/core/api",
        timeout=30.0,
        transport=httpx.MockTransport(handler),
    )

    result = resolve_fallback(graph, {0: "Kenya"}, normalize=_normalize)

    assert result == {0: ("country/KEN",)}
