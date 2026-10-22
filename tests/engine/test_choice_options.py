"""A refusal that asks for a choice lists one ready request per choice
(`NoData.options`), and each request posted back unchanged through the parsed door does
not refuse with the same reason again.

Every round trip runs the composed search app in process over offline doubles. The
"fertility rate"/"Kenya" texts match the recorded LLM fixtures. The role-bearing cases
elect their one shape through a scripted bind."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import origin_catalogue.service as catalogue_service_module
from origin_catalogue import DFO, ProvenanceRow, SourceRef
from search.engine.quantity_read import QuantityOutput
from series_coverage.schema import RoleNames
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import patch_coverage
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
    "European Union": "EuropeanUnion",
}


class _ShapeZeroLLM:
    """Elects shape 0 with no bindings, for the unrecorded role-bearing fixture."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0), None


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
    """A flow StatVar declaring a creditor and a debtor role, over four countries and the
    European Union."""
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
    """Each option's request, posted unchanged, succeeds and does not refuse with the same
    reason: it answers, or refuses on something the choice does not decide."""
    for option in no_data["options"]:
        response = client.post("/api/search/resolve", json={"input": option["data_query"]})
        assert response.status_code == 200, (option["label"], response.text)
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
            ["Norway to Kenya"],
            id="one-untold-read-unknown",
        ),
    ],
)
def test_an_untold_direction_offers_each_reading_within_the_cap(entities, open_roles, labels):
    """The coverage index holds no row of the flow, so no direction is read empty."""
    client = _flow_client()
    part = {"variable_text": "aid", "entities": entities}
    if open_roles is not None:
        part["open_roles"] = open_roles

    no_data = _no_data(client, part)

    assert no_data["reason"] == "role_not_resolved"
    assert [o["label"] for o in no_data["options"]] == labels
    _assert_options_settle(client, no_data)


def test_an_untold_direction_read_empty_every_way_offers_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """The coverage index holds the flow between Uganda and Rwanda only, so the one reading
    of Kenya is read and holds no data."""
    patch_coverage(
        monkeypatch,
        [(_FLOW_SV, "f_flow", _PLACES["Uganda"], _PLACES["Rwanda"], [2020])],
        role_names={_FLOW_SV: RoleNames(donor="creditor", recipient="debtor")},
    )
    client = _flow_client()

    no_data = _no_data(
        client,
        {
            "variable_text": "aid",
            "entities": [{"text": "Norway", "role_text": "from"}, {"text": "Kenya"}],
        },
    )

    assert no_data["reason"] == "role_not_resolved"
    assert no_data["options"] == []


def test_a_role_option_states_a_direction_for_every_untold_name():
    client = _flow_client()

    no_data = _no_data(client, {"variable_text": "aid", "entities": _entities("Norway", "Kenya")})

    assert [o["mentions"] for o in no_data["options"]] == [["Norway", "Kenya"]] * 2
    assert [o["data_query"]["parts"][0]["entities"] for o in no_data["options"]] == [
        [{"text": "Norway", "role_text": "from"}, {"text": "Kenya", "role_text": "to"}],
        [{"text": "Norway", "role_text": "to"}, {"text": "Kenya", "role_text": "from"}],
    ]


@pytest.mark.parametrize(
    ("entities", "names", "untold", "labels"),
    [
        pytest.param(
            [{"text": "KE", "dcid": "country/KEN"}, {"text": "Norway"}],
            ["KE", "Norway"],
            'Could not tell who is giving and who is receiving among "KE" and "Norway"',
            ["KE to Norway", "Norway to KE"],
            id="pinned-only-text",
        ),
        pytest.param(
            [{"text": "Kenya"}, {"text": "Norway"}, {"text": "KEN", "dcid": "country/KEN"}],
            ["Kenya", "Norway", "KEN"],
            'Could not tell who is giving and who is receiving among "Kenya" ("KEN") and "Norway"',
            ['Kenya ("KEN") to Norway', 'Norway to Kenya ("KEN")'],
            id="two-texts-of-one-set-apart",
        ),
        pytest.param(
            [
                {"text": "KE", "dcid": "country/KEN"},
                {"text": "Kenya"},
                {"text": "Norway", "role_text": "from"},
            ],
            ["KE", "Kenya"],
            'Could not tell whether "KE" ("Kenya") is giving or receiving',
            ['Norway to KE ("Kenya")'],
            id="one-set-named-twice",
        ),
        pytest.param(
            [
                {"text": "the union", "dcid": "EuropeanUnion"},
                {"text": "European Union"},
                {"text": "Kenya", "role_text": "to"},
            ],
            ["the union", "European Union"],
            'Could not tell whether "the union" ("European Union") is giving or receiving',
            ['the union ("European Union") to Kenya'],
            id="pinned-group-named-twice",
        ),
    ],
)
def test_a_role_refusal_names_every_untold_text_in_query_order(entities, names, untold, labels):
    """The options' mentions and the refusal's entity terms name the same request texts in
    query order, a pin's key included in place of its node's label. The message and the
    option labels name each untold place once, its further texts in parentheses."""
    client = _flow_client()

    no_data = _no_data(client, {"variable_text": "aid", "entities": entities})

    assert no_data["reason"] == "role_not_resolved"
    assert no_data["options"]
    assert [o["mentions"] for o in no_data["options"]] == [names] * len(no_data["options"])
    assert no_data["entity_terms"] == names
    assert no_data["message"].split(". ", 1)[0] == untold
    assert [o["label"] for o in no_data["options"]] == labels


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
