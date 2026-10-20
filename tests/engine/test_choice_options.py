"""A refusal that asks for a choice lists one ready request per choice
(`NoData.options`), and each request posted back unchanged through the parsed door does
not refuse with the same reason again.

Every round trip runs the composed search app in process over offline doubles. The
"fertility rate"/"Kenya" texts match the recorded LLM fixtures; the role-bearing cases
elect their one shape through a scripted bind."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import origin_catalogue.service as catalogue_service_module
from origin_catalogue import DFO, ProvenanceRow, SourceRef
from search.engine.bind import _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs
from tests.engine._quantity import NO_QUANTITY
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, fake_catalogue_service

_FERTILITY = "fertility rate"
_COD = "country/COD"
_FERTILITY_SV = "FertilityRate_Person_Female"
_FLOW_SV = "TESTFIXTURE/ChoiceOptions/Flow"
_PLACES = {
    "Norway": "country/NOR",
    "Kenya": "country/KEN",
    "Uganda": "country/UGA",
    "Rwanda": "country/RWA",
}


class _ShapeZeroLLM:
    """Elects shape 0 with no bindings, for the unrecorded role-bearing fixture."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def _fertility_client(llm: FakeLLM | _ShapeZeroLLM | None = None) -> TestClient:
    obs = {
        **_load_json("graph_obs.json"),
        f"FertilityRate_Person_Female|{_COD}": [
            {"earliestDate": "1960", "latestDate": "2023", "obsCount": 64}
        ],
    }
    graph = FakeGraph(obs=obs)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {key: facets for key, facets in obs.items() if key.startswith(_FERTILITY_SV)}
    )
    return TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm or FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        ),
        raise_server_exceptions=False,
    )


def _flow_client() -> TestClient:
    """A flow StatVar declaring a creditor and a debtor role, over four countries."""
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in _PLACES.items()},
            _FLOW_SV: {
                "label": "Test flow",
                "arcs": {
                    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                    "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
                },
            },
        },
        detect={
            text: {"svs": [_FLOW_SV], "cosine_scores": [0.9]} for text in ("aid", "aid to Norway")
        },
        resolve=dict(_PLACES),
        obs={},
    )
    return _client(graph, _ShapeZeroLLM())


def _client(graph: FakeGraph, llm: _ShapeZeroLLM) -> TestClient:
    return TestClient(
        search_app(graph=graph, resolver=make_resolver(graph), llm=llm, sdmx=FakeSdmxClient()),
        raise_server_exceptions=False,
    )


def _no_data(client: TestClient, part: dict) -> dict:
    """The no_data of the one part `part` answers, posted through the parsed door."""
    response = client.post(
        "/api/search/resolve", json={"input": {"kind": "parsed", "parts": [part]}}
    )
    assert response.status_code == 200, response.text
    (answered,) = response.json()["parts"]
    return answered.get("no_data") or {}


def _assert_options_settle(client: TestClient, no_data: dict) -> None:
    """Each option's request, posted unchanged, is a valid request that does not refuse
    with the same reason: it answers, or fails on something the choice does not decide."""
    for option in no_data["options"]:
        response = client.post("/api/search/resolve", json={"input": option["data_query"]})
        assert response.status_code not in (400, 422), (option["label"], response.text)
        if response.status_code != 200:
            continue
        (answered,) = response.json()["parts"]
        reason = (answered.get("no_data") or {}).get("reason")
        assert reason != no_data["reason"], option["label"]


def test_an_ambiguous_entity_offers_each_candidate_and_each_option_resolves_it():
    client = _fertility_client()

    no_data = _no_data(client, {"variable_text": _FERTILITY, "entities": [{"text": "Congo"}]})

    assert no_data["reason"] == "entity_ambiguous"
    (group,) = no_data["ambiguous_entities"]
    assert [(o["mentions"], o["label"]) for o in no_data["options"]] == [
        (["Congo"], candidate["label"]) for candidate in group["candidates"]
    ]
    assert [o["data_query"]["parts"][0]["entities"] for o in no_data["options"]] == [
        [{"text": "Congo", "dcid": candidate["ref"]}] for candidate in group["candidates"]
    ]
    _assert_options_settle(client, no_data)


def test_an_ambiguous_source_offers_each_candidate_and_each_option_resolves_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """Two publishers' tables share one name, so the name refuses source_ambiguous."""
    words = "Shared Tables"
    rows = [
        ProvenanceRow(
            id=f"dc/base/{publisher}Tables",
            data_commons=DFO,
            source=SourceRef(ref=f"dc/s/{publisher}", label=publisher),
            name=words,
            url="https://example.org",
        )
        for publisher in ("Alpha", "Beta")
    ]
    service = fake_catalogue_service(FakeCatalogue(provenances={row.id: row for row in rows}))
    with catalogue_service_module._singleton.poisoned_for_test(service):
        # A scoped answer reaches bind with a frame no recorded fixture holds.
        client = _fertility_client(_ShapeZeroLLM())
        no_data = _no_data(
            client,
            {
                "variable_text": _FERTILITY,
                "entities": [{"text": "Kenya"}],
                "sources": [{"text": words}],
            },
        )

        assert no_data["reason"] == "source_ambiguous"
        (group,) = no_data["ambiguous_sources"]
        assert [
            (o["mentions"], o["data_query"]["parts"][0]["sources"]) for o in no_data["options"]
        ] == [
            ([words], [{"text": words, "ref": candidate["ref"], "polarity": "include"}])
            for candidate in group["candidates"]
        ]
        _assert_options_settle(client, no_data)


@pytest.mark.parametrize(
    ("unit_text", "posted"),
    [
        ("francs", ["CHF", "XAF", "XOF"]),
        ("constant 2015 francs", ["constant 2015 CHF", "constant 2015 XAF", "constant 2015 XOF"]),
    ],
    ids=["bare", "beside-a-basis"],
)
def test_an_ambiguous_currency_offers_each_alternative_and_each_option_binds_it(unit_text, posted):
    client = _fertility_client()

    no_data = _no_data(
        client,
        {"variable_text": _FERTILITY, "entities": [{"text": "Kenya"}], "unit_text": unit_text},
    )

    assert no_data["reason"] == "unit_not_resolved"
    assert [o["label"] for o in no_data["options"]] == [
        alternative["label"] for alternative in no_data["unit_alternatives"]
    ]
    assert [o["data_query"]["parts"][0]["unit_text"] for o in no_data["options"]] == posted
    assert {tuple(o["mentions"]) for o in no_data["options"]} == {(unit_text,)}
    _assert_options_settle(client, no_data)


def _entities(*names: str, role_text: str | None = None) -> list[dict]:
    return [
        {"text": name} if role_text is None else {"text": name, "role_text": role_text}
        for name in names
    ]


@pytest.mark.parametrize(
    ("entities", "open_roles", "labels"),
    [
        pytest.param(
            _entities("Norway", "Kenya"),
            None,
            ["Norway to Kenya", "Kenya to Norway"],
            id="two-untold",
        ),
        pytest.param(
            _entities("Norway", "Kenya", "Uganda"),
            None,
            [
                "Norway and Kenya to Uganda",
                "Norway and Uganda to Kenya",
                "Norway to Kenya and Uganda",
                "Kenya and Uganda to Norway",
                "Kenya to Norway and Uganda",
                "Uganda to Norway and Kenya",
            ],
            id="three-untold",
        ),
        pytest.param(
            _entities("Norway", "Kenya", "Uganda", "Rwanda"), None, [], id="four-past-the-cap"
        ),
        pytest.param(
            _entities("Kenya", "Uganda"),
            ["creditor"],
            ["To Kenya and Uganda"],
            id="open-axis-takes-no-place",
        ),
        pytest.param(
            [{"text": "Norway", "role_text": "from"}, {"text": "Kenya"}],
            None,
            [],
            id="one-untold-read-both-ways-empty",
        ),
    ],
)
def test_an_untold_direction_offers_each_reading_within_the_cap(entities, open_roles, labels):
    client = _flow_client()
    part = {"variable_text": "aid", "entities": entities}
    if open_roles is not None:
        part["open_roles"] = open_roles

    no_data = _no_data(client, part)

    assert no_data["reason"] == "role_not_resolved"
    assert [o["label"] for o in no_data["options"]] == labels
    _assert_options_settle(client, no_data)


def test_a_role_option_states_a_direction_for_every_untold_name():
    client = _flow_client()

    no_data = _no_data(client, {"variable_text": "aid", "entities": _entities("Norway", "Kenya")})

    assert [o["mentions"] for o in no_data["options"]] == [["Norway", "Kenya"]] * 2
    assert [o["data_query"]["parts"][0]["entities"] for o in no_data["options"]] == [
        [{"text": "Norway", "role_text": "from"}, {"text": "Kenya", "role_text": "to"}],
        [{"text": "Norway", "role_text": "to"}, {"text": "Kenya", "role_text": "from"}],
    ]


def test_a_reading_its_own_wording_contradicts_is_not_offered():
    """ "aid to Norway" contradicts Norway as the creditor. The one reading that fills the
    creditor axis puts Norway back on it, which the parsed door refuses again, so the
    refusal offers nothing."""
    client = _flow_client()

    no_data = _no_data(
        client,
        {
            "variable_text": "aid to Norway",
            "entities": [
                {"text": "Norway", "role_text": "from"},
                {"text": "Kenya", "role_text": "to"},
            ],
        },
    )

    assert no_data["reason"] == "role_not_resolved"
    assert no_data["options"] == []
