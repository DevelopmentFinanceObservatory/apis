"""A parsed entity's `name` is its identity within the part, and resolution reads its
`text` beside the name as the raw-text door reads a query's words beside an extracted
name (`ParsedEntity.name`).

Every case posts through the composed search app in process over offline doubles."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from search.engine.region import resolve_variable
from tests.engine.test_choice_options import (
    _COD,
    _COG,
    _FERTILITY,
    _FLOW_NODE,
    _FLOW_SV,
    _client,
    _fertility_client,
    _ShapeZeroLLM,
)
from tests.fixtures import FakeGraph

_DRC = "Democratic Republic of the Congo"
_ROC = "Republic of the Congo"


def _parts(client: TestClient, *parts: dict) -> list[dict]:
    response = client.post(
        "/api/search/resolve", json={"input": {"kind": "parsed", "parts": list(parts)}}
    )
    assert response.status_code == 200, response.text
    return response.json()["parts"]


@pytest.mark.parametrize(
    ("origin", "destination"),
    [(_DRC, _ROC), ("Congo [DRC]", "Congo [Republic]")],
    ids=["extracted names", "candidate labels"],
)
def test_two_names_written_alike_ask_two_choices(origin, destination):
    """Two rows written "Congo" under two names stay two entities, each asked apart in
    its words. Each option pins its own end alone, even where its candidate's label is the
    other end's name, and posted back asks the other end again."""
    graph = FakeGraph(
        nodes={
            _COD: {"label": _DRC, "type": "Country"},
            _COG: {"label": _ROC, "type": "Country"},
            _FLOW_SV: _FLOW_NODE,
        },
        detect={"aid": {"svs": [_FLOW_SV], "cosine_scores": [0.9]}},
        obs={},
    )
    client = _client(graph, _ShapeZeroLLM())

    (part,) = _parts(
        client,
        {
            "variable_text": "aid",
            "entities": [
                {"text": "Congo", "name": origin, "role_text": "from"},
                {"text": "Congo", "name": destination, "role_text": "to"},
            ],
        },
    )

    no_data = part["no_data"]
    assert no_data["reason"] == "entity_ambiguous"
    assert [
        (group["text"], {c["ref"] for c in group["candidates"]})
        for group in no_data["ambiguous_entities"]
    ] == [("Congo", {_COD, _COG})] * 2
    for option in no_data["options"]:
        rows = option["data_query"]["parts"][0]["entities"]
        assert len({row.get("name", row["text"]) for row in rows}) == 2, rows
        (again,) = _parts(client, option["data_query"]["parts"][0])
        (unchosen,) = [row["role_text"] for row in rows if "dcid" not in row]
        assert [group["text"] for group in again["no_data"]["ambiguous_entities"]] == ["Congo"]
        assert [
            [row["role_text"] for row in o["data_query"]["parts"][0]["entities"] if "dcid" in row]
            for o in again["no_data"]["options"]
        ] == [["from", "to"]] * 2, unchosen


_KENYA_PART = {"variable_text": _FERTILITY, "entities": [{"text": "Kenya"}]}


@pytest.mark.parametrize("beside", [(), (_KENYA_PART,)], ids=["one part", "two parts"])
def test_a_named_entity_resolves_its_words_beside_its_name(beside):
    """ "Congolese" alone names nothing; beside the name "Congo" it asks the choice "Congo"
    asks, quoted in the posted words. A second part takes the parsed door's batched
    resolution."""
    client = _fertility_client()

    alone, *_ = _parts(
        client, {"variable_text": _FERTILITY, "entities": [{"text": "Congolese"}]}, *beside
    )
    named, *_ = _parts(
        client,
        {"variable_text": _FERTILITY, "entities": [{"text": "Congolese", "name": "Congo"}]},
        *beside,
    )

    assert alone["no_data"]["reason"] == "entity_not_resolved"
    assert named["no_data"]["reason"] == "entity_ambiguous"
    (group,) = named["no_data"]["ambiguous_entities"]
    assert group["text"] == "Congolese"
    assert {_COD, _COG} <= {c["ref"] for c in group["candidates"]}


def test_a_name_beside_a_dcid_is_refused():
    """A pinned row's identity is its dcid, so a name beside it is refused at the row."""
    client = _fertility_client()

    response = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "parsed",
                "parts": [
                    {
                        "variable_text": _FERTILITY,
                        "entities": [{"text": "Congo", "name": _DRC, "dcid": _COD}],
                    }
                ],
            }
        },
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert (body["code"], body["pointer"]) == ("request_invalid", "/input/parts/0/entities/0")


def test_two_rows_pinning_one_dcid_under_two_texts_are_one_entity():
    """A pinned row's identity is its dcid, so two rows pinning one node in different
    words are one entity, read in the first row's words."""
    client = _fertility_client()

    (part,) = _parts(
        client,
        {
            "variable_text": _FERTILITY,
            "entities": [{"text": "Congo", "dcid": _COD}, {"text": "Congo [DRC]", "dcid": _COD}],
        },
    )

    assert part["status"] == "definite", part.get("no_data")
    assert part["spec"]["definition"]["identity"]["entities"] == {"observationAbout": [_COD]}


def test_each_part_resolves_its_own_words_for_a_shared_name(monkeypatch):
    """Two parts post the name "Congo" under different words, and each part's leg reads
    its own words."""
    seen: dict[int, dict[str, str] | None] = {}

    async def _recording(*args, **kwargs):
        seen[kwargs["variable_index"]] = kwargs["entity_surfaces"]
        return await resolve_variable(*args, **kwargs)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _recording)
    client = _fertility_client()

    _parts(
        client,
        {"variable_text": _FERTILITY, "entities": [{"text": "Congolese", "name": "Congo"}]},
        {"variable_text": _FERTILITY, "entities": [{"text": "the Congo", "name": "Congo"}]},
    )

    assert seen == {0: {"Congo": "Congolese"}, 1: {"Congo": "the Congo"}}


@pytest.mark.parametrize(
    "unpinned",
    [{"text": "Congo", "name": "country/KEN"}, {"text": "country/KEN"}],
    ids=["name", "text"],
)
@pytest.mark.parametrize(
    ("pin", "refused"),
    [({"text": "Kenya", "dcid": "country/KEN"}, True), ({"text": "Congo", "dcid": _COD}, False)],
    ids=["its own dcid", "another dcid"],
)
def test_an_unpinned_identity_another_row_pins_is_refused(unpinned, pin, refused):
    """An unpinned row whose name, or text, is a dcid another row pins would bind as that
    entity, so the part refuses at the unpinned row. Beside a pin to another dcid it
    answers."""
    client = _fertility_client(_ShapeZeroLLM())

    response = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "parsed",
                "parts": [{"variable_text": _FERTILITY, "entities": [pin, unpinned]}],
            }
        },
    )

    if refused:
        assert response.status_code == 400, response.text
        body = response.json()
        assert (body["code"], body["pointer"]) == ("request_invalid", "/input/parts/0/entities/1")
    else:
        assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "entities",
    [
        [{"text": "Congolese", "name": "Congo"}, {"text": "Congo"}],
        [{"text": "Congo"}, {"text": "Congolese", "name": "Congo"}],
        [{"text": "Congolese", "name": "Congo"}, {"text": "the Congo", "name": "Congo"}],
    ],
    ids=["named first", "named last", "two surfaces"],
)
def test_one_entity_posted_under_two_texts_is_refused(entities):
    """One entity resolves through one text, so a part posting it under two refuses."""
    client = _fertility_client()

    response = client.post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "parsed",
                "parts": [{"variable_text": _FERTILITY, "entities": entities}],
            }
        },
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert (body["code"], body["pointer"]) == ("request_invalid", "/input/parts/0/entities")


def test_words_that_differ_from_the_name_only_in_case_read_as_the_name(monkeypatch):
    """ "congo" under the name "Congo" are the name's own words, so resolution reads the
    name alone, as the raw-text door does (`surface_resolution.is_surface`)."""
    seen: dict[int, dict[str, str] | None] = {}

    async def _recording(*args, **kwargs):
        seen[kwargs["variable_index"]] = kwargs["entity_surfaces"]
        return await resolve_variable(*args, **kwargs)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _recording)
    client = _fertility_client()

    (part,) = _parts(
        client,
        {
            "variable_text": _FERTILITY,
            "entities": [{"text": "congo", "name": "Congo"}, {"text": "Congo"}],
        },
    )

    assert not seen[0]
    assert [group["text"] for group in part["no_data"]["ambiguous_entities"]] == ["Congo"]


def test_words_alike_but_for_case_under_one_name_are_one_text():
    """ "Congo" and "congo" under one name are the same words, so the part resolves the
    name through them once instead of refusing two texts."""
    client = _fertility_client()

    (part,) = _parts(
        client,
        {
            "variable_text": _FERTILITY,
            "entities": [{"text": "Congo", "name": _DRC}, {"text": "congo", "name": _DRC}],
        },
    )

    assert [group["text"] for group in part["no_data"]["ambiguous_entities"]] == ["Congo"]


@pytest.mark.parametrize("beside", [(), (_KENYA_PART,)], ids=["one part", "two parts"])
def test_a_pinned_row_s_words_reach_the_leg(monkeypatch, beside):
    """A pinned row's identity is its dcid, so its leg is handed the row's words, which
    bind masks in the variable phrase (`ResolveState.entity_texts`) though the row binds
    no role."""
    seen: dict[int, dict[str, str] | None] = {}

    async def _recording(*args, **kwargs):
        seen[kwargs["variable_index"]] = kwargs["entity_words"]
        return await resolve_variable(*args, **kwargs)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _recording)
    client = _fertility_client()

    _parts(
        client,
        {
            "variable_text": f"Congo {_FERTILITY}",
            "entities": [{"text": "Congo", "dcid": _COD}],
        },
        *beside,
    )

    assert seen[0] == {_COD: "Congo"}


@pytest.mark.parametrize(
    "kenya", [{"text": "Kenya"}, {"text": "Kenya", "dcid": "country/KEN"}], ids=["named", "pinned"]
)
def test_a_dropped_direction_quotes_the_rows_words(kenya):
    """A subject-only shape drops the direction a row states. Its warning and departure
    quote the row's words, a pinned row's included, whose identity is its dcid."""
    client = _fertility_client()

    (part,) = _parts(
        client, {"variable_text": _FERTILITY, "entities": [{**kenya, "role_text": "from"}]}
    )

    assert [w["message"] for w in part["warnings"] if w["code"] == "direction_discarded"] == [
        "Kenya was named as the provider; this shape reports the place as its own subject, "
        "so the direction was dropped."
    ]
    assert [(d["asked"], d["text"]) for d in part["departures"]] == [
        (
            "Kenya",
            "You named Kenya as a provider. This measure has no providers, so this shows the "
            "place itself.",
        )
    ]
