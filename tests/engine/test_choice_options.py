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
from search.engine.extract import ExtractedEntity, Extraction
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
_COG = "country/COG"
_FERTILITY_SV = "FertilityRate_Person_Female"
_FLOW_SV = "TESTFIXTURE/ChoiceOptions/Flow"
_PLACES = {
    "Norway": "country/NOR",
    "Kenya": "country/KEN",
    "Uganda": "country/UGA",
    "Rwanda": "country/RWA",
    "European Union": "EuropeanUnion",
}

# A flow StatVar declaring a creditor and a debtor role.
_FLOW_NODE = {
    "label": "Test flow",
    "arcs": {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "creditor"}, {"dcid": "debtor"}]},
    },
}


class _ShapeZeroLLM:
    """Elects shape 0 with no bindings, for the unrecorded role-bearing fixture."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0), None


class _ExtractionLLM(_ShapeZeroLLM):
    """Reads every query as `variable` over `entities`, and elects shape 0 with no
    bindings."""

    def __init__(self, variable: str, *entities: ExtractedEntity) -> None:
        self.variable = variable
        self.entities = list(entities)

    async def generate_structured(self, *, prompt, system, schema):
        if schema is Extraction:
            extraction = Extraction(
                kind="resolution", variables=[self.variable], entities=self.entities
            )
            return extraction, None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _congo_extraction_llm(span: str) -> _ExtractionLLM:
    """Reads every query as the fertility rate of the Democratic Republic of the Congo,
    named in the query by `span`."""
    return _ExtractionLLM(
        _FERTILITY, ExtractedEntity(text="Democratic Republic of the Congo", span=span)
    )


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
            _FLOW_SV: _FLOW_NODE,
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


def _asked(no_data: dict) -> set[str]:
    """The request words a refusal asks the caller to choose or correct."""
    groups = [*(no_data.get("ambiguous_sources") or ()), *(no_data.get("ambiguous_entities") or ())]
    return {
        *(group["text"] for group in groups),
        *(no_data.get("entity_terms") or ()),
        *(no_data.get("source_terms") or ()),
    }


def _assert_options_settle(client: TestClient, no_data: dict) -> None:
    """Each option's request, posted unchanged, is accepted and settles the choice. The
    part answers, or refuses without asking about the option's words again. A part has one
    unit text, so a currency choice settles when the answer offers no currencies."""
    for option in no_data["options"]:
        response = client.post("/api/search/resolve", json={"input": option["data_query"]})
        assert response.status_code == 200, (option["label"], response.text)
        (answered,) = response.json()["parts"]
        again = answered.get("no_data") or {}
        assert not set(option["mentions"]) & _asked(again), (option["label"], again)
        assert not (no_data.get("unit_alternatives") and again.get("unit_alternatives")), (
            option["label"],
            again,
        )


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


def _raw_part(client: TestClient, query: str) -> dict:
    """The one part the raw-text door answers for `query`."""
    response = client.post(
        "/api/search/resolve", json={"input": {"kind": "raw_text", "query": query}}
    )
    assert response.status_code == 200, response.text
    (answered,) = response.json()["parts"]
    return answered


def test_a_name_the_query_words_leave_open_asks_its_choice_in_those_words():
    """The extraction names "Congo" the Democratic Republic of the Congo, but "Congo" names
    two countries, so the choice quotes the user's word. An option posts that word pinned
    to its candidate, and settles the part under it."""
    client = _fertility_client(_congo_extraction_llm("Congo"))

    part = _raw_part(client, "fertility rate in Congo")

    no_data = part["no_data"]
    assert no_data["reason"] == "entity_ambiguous"
    assert no_data["entity_terms"] == ["Congo"]
    (group,) = no_data["ambiguous_entities"]
    assert group["text"] == "Congo"
    assert {c["ref"] for c in group["candidates"]} == {_COD, _COG}
    assert [o["mentions"] for o in no_data["options"]] == [["Congo"], ["Congo"]]
    assert [o["entity_choice"] for o in no_data["options"]] == [0, 0]
    assert [t.get("entity_choice") for t in part["sentence_plan"] if t["kind"] == "slot"] == [0]
    assert [e["text"] for e in part["restatement"]["entities"]] == ["Congo"]
    assert [o["data_query"]["parts"][0]["entities"] for o in no_data["options"]] == [
        [{"text": "Congo", "dcid": c["ref"]}] for c in group["candidates"]
    ]
    for option, candidate in zip(no_data["options"], group["candidates"], strict=True):
        response = client.post("/api/search/resolve", json={"input": option["data_query"]})
        (answered,) = response.json()["parts"]
        assert answered["restatement"]["entities"] == [{"text": "Congo", "dcid": candidate["ref"]}]
        assert answered["entity_refs"] == [{"text": "Congo", "refs": [candidate["ref"]]}]
    _assert_options_settle(client, no_data)


@pytest.mark.parametrize(
    ("name", "span", "query", "asked"),
    [
        ("Congo", "congo", "fertility rate in congo", "Congo"),
        ("Democratic Republic of the Congo", "congo", "fertility rate in congo", "congo"),
        ("Congo", "Kongo", "fertility rate in Congo", "Congo"),
    ],
    ids=["case only", "lower-case span", "ungrounded span"],
)
def test_a_choice_quotes_the_words_its_rows_post(name, span, query, asked):
    """The refusal quotes the words its resolution read, the extracted name where the span
    differs from it only in case or is not in the query, and its restatement row posts
    those words, so posting the restatement back asks the same choice."""
    client = _fertility_client(_ExtractionLLM(_FERTILITY, ExtractedEntity(text=name, span=span)))

    part = _raw_part(client, query)

    no_data = part["no_data"]
    (group,) = no_data["ambiguous_entities"]
    assert group["text"] == asked
    assert no_data["entity_terms"] == [asked]
    assert [e["text"] for e in part["restatement"]["entities"]] == [asked]
    assert [o["mentions"] for o in no_data["options"]] == [[asked]] * 2
    again = _no_data(client, part["restatement"])
    assert again["ambiguous_entities"] == no_data["ambiguous_entities"]
    _assert_options_settle(client, no_data)


def test_an_ambiguous_name_beside_an_unresolved_one_is_quoted_in_the_users_word():
    """ "Narnia" names no place, so the part refuses entity_not_resolved, and the refusal
    quotes "Congo" as the user wrote it."""
    llm = _ExtractionLLM(
        _FERTILITY,
        ExtractedEntity(text="Democratic Republic of the Congo", span="Congo"),
        ExtractedEntity(text="Narnia"),
    )
    client = _fertility_client(llm)

    part = _raw_part(client, "fertility rate in Congo and Narnia")

    no_data = part["no_data"]
    assert no_data["reason"] == "entity_not_resolved"
    assert no_data["entity_terms"] == ["Congo", "Narnia"]
    assert no_data["ambiguous_entities"] == []


_DRC = "Democratic Republic of the Congo"
_ROC = "Republic of the Congo"
_IBRD = "InternationalBankForReconstructionAndDevelopment"
_IDA = "InternationalDevelopmentAssociation"
# "World Bank" names both of its lending arms.
_WORLD_BANK = ExtractedEntity(text="World Bank", span="World Bank")


def _congo_flow_client(*entities: ExtractedEntity) -> TestClient:
    """A flow StatVar over the two Congos and the World Bank's two lending arms, with
    every query read as aid over `entities`."""
    graph = FakeGraph(
        nodes={
            _COD: {"label": _DRC, "type": "Country"},
            _COG: {"label": _ROC, "type": "Country"},
            _IBRD: {
                "label": "International Bank for Reconstruction and Development",
                "type": "Organization",
            },
            _IDA: {"label": "International Development Association", "type": "Organization"},
            _FLOW_SV: _FLOW_NODE,
        },
        detect={"aid": {"svs": [_FLOW_SV], "cosine_scores": [0.9]}},
        obs={},
    )
    return _client(graph, _ExtractionLLM("aid", *entities))


def _congo_to_congo(*extra: ExtractedEntity) -> tuple[TestClient, dict]:
    """The part "aid from Congo to Congo" answers, read as the Democratic Republic of the
    Congo to the Republic of the Congo, beside `extra`, and its client."""
    client = _congo_flow_client(
        ExtractedEntity(text=_DRC, span="Congo", flow_role="origin", role_span="from"),
        ExtractedEntity(text=_ROC, span="Congo", flow_role="destination", role_span="to"),
        *extra,
    )
    query = "aid from Congo to Congo" + (" by the World Bank" if extra else "")
    return client, _raw_part(client, query)


def _posted(option: dict) -> list[dict]:
    """The entity rows an option posts."""
    return option["data_query"]["parts"][0]["entities"]


def _congo_rows(rows: list[dict]) -> list[tuple[str, str | None, str | None]]:
    """The words, name and direction of each Congo row of `rows`."""
    return [
        (row["text"], row.get("name"), row.get("role_text"))
        for row in rows
        if row["text"] == "Congo"
    ]


def test_two_names_written_alike_ask_each_choice_apart():
    """ "Congo" names two countries, and the extraction reads "aid from Congo to Congo" as
    the Democratic Republic of the Congo to the Republic of the Congo. Each end is asked
    apart in the user's word, and each row posts that word under its own name. An option
    settles its own end and asks the other again, and an option of that settles the
    part."""
    client, part = _congo_to_congo()

    no_data = part["no_data"]
    assert no_data["reason"] == "entity_ambiguous"
    assert no_data["entity_terms"] == ["Congo", "Congo"]
    assert [
        (group["text"], {c["ref"] for c in group["candidates"]})
        for group in no_data["ambiguous_entities"]
    ] == [("Congo", {_COD, _COG})] * 2
    assert [o["mentions"] for o in no_data["options"]] == [["Congo"]] * 4
    ends = [("Congo", _DRC, "from"), ("Congo", _ROC, "to")]
    assert _congo_rows(part["restatement"]["entities"]) == ends
    assert [_pins(o) for o in no_data["options"]] == [
        [("from", _COD), ("to", None)],
        [("from", _COG), ("to", None)],
        [("from", None), ("to", _COD)],
        [("from", None), ("to", _COG)],
    ]
    # Each slot and each option names its own group, so a menu per group pins one end.
    assert [
        (token["text"], token.get("role"), token.get("entity_choice"))
        for token in part["sentence_plan"]
        if token["kind"] == "slot"
    ] == [("Congo", "origin", 0), ("Congo", "destination", 1)]
    menus: dict[int, list] = {}
    for option in no_data["options"]:
        menus.setdefault(option["entity_choice"], []).append(_pins(option))
    assert menus == {
        0: [[("from", _COD), ("to", None)], [("from", _COG), ("to", None)]],
        1: [[("from", None), ("to", _COD)], [("from", None), ("to", _COG)]],
    }
    for option in no_data["options"]:
        # The pinned end posts its words and the dcid alone, and the other keeps its name.
        assert [
            (text, None if dcid else name, role_text)
            for (text, name, role_text), (_, dcid) in zip(ends, _pins(option), strict=True)
        ] == _congo_rows(_posted(option))
        response = client.post("/api/search/resolve", json={"input": option["data_query"]})
        assert response.status_code == 200, response.text
        (answered,) = response.json()["parts"]
        again = answered["no_data"]
        assert [group["text"] for group in again["ambiguous_entities"]] == ["Congo"]
        # Each option of the end asked again pins that end alone, and the first pick holds.
        settled = {role: dcid for role, dcid in _pins(option) if dcid}
        assert [_pins(o) for o in again["options"]] == [
            [(role, settled.get(role, ref)) for role in ("from", "to")] for ref in (_COD, _COG)
        ]
        _assert_options_settle(client, again)


def test_one_name_on_both_ends_asks_each_choice_apart():
    """The live extraction names "Congo" twice, so one name holds both ends. Each end still
    asks its own choice, each option pins its own row alone, the restatement states both
    rows, and two picks bind each end to its own country."""
    client = _congo_flow_client(
        ExtractedEntity(text="Congo", flow_role="origin", role_span="from"),
        ExtractedEntity(text="Congo", flow_role="destination", role_span="to"),
    )

    part = _raw_part(client, "aid from Congo to Congo")

    no_data = part["no_data"]
    assert no_data["reason"] == "entity_ambiguous"
    assert no_data["entity_terms"] == ["Congo", "Congo"]
    assert [group["text"] for group in no_data["ambiguous_entities"]] == ["Congo", "Congo"]
    assert _congo_rows(part["restatement"]["entities"]) == [
        ("Congo", None, "from"),
        ("Congo", None, "to"),
    ]
    assert [(o["entity_choice"], _pins(o)) for o in no_data["options"]] == [
        (0, [("from", _COD), ("to", None)]),
        (0, [("from", _COG), ("to", None)]),
        (1, [("from", None), ("to", _COD)]),
        (1, [("from", None), ("to", _COG)]),
    ]
    for option in no_data["options"]:
        answered = client.post("/api/search/resolve", json={"input": option["data_query"]})
        assert answered.status_code == 200, answered.text
        again = answered.json()["parts"][0]["no_data"]
        assert [group["text"] for group in again["ambiguous_entities"]] == ["Congo"]
        _assert_options_settle(client, again)

    again = _no_data(client, no_data["options"][0]["data_query"]["parts"][0])
    (chosen,) = [o for o in again["options"] if _pins(o) == [("from", _COD), ("to", _COG)]]
    settled = client.post("/api/search/resolve", json={"input": chosen["data_query"]})
    assert settled.status_code == 200, settled.text
    (answered,) = settled.json()["parts"]
    assert not answered.get("no_data", {}).get("ambiguous_entities")
    assert answered["entity_refs"] == [
        {"text": "Congo", "refs": [_COD]},
        {"text": "Congo", "refs": [_COG]},
    ]


def test_a_candidate_named_by_another_row_s_words_is_pinned_apart_from_it():
    """The destination posts the candidate label "Congo [Republic]" pinned to the Republic,
    and the origin "Congo" is asked. Every option answers, and the Republic option pins
    both ends to the Republic without posting one entity under two texts."""
    client = _congo_flow_client()

    no_data = _no_data(
        client,
        {
            "variable_text": "aid",
            "entities": [
                {"text": "Congo", "role_text": "from"},
                {"text": "Congo [Republic]", "role_text": "to", "dcid": _COG},
            ],
        },
    )

    assert no_data["reason"] == "entity_ambiguous"
    for option in no_data["options"]:
        response = client.post("/api/search/resolve", json={"input": option["data_query"]})
        assert response.status_code == 200, (option["label"], response.text)
    (republic,) = [o for o in no_data["options"] if o["label"] == "Congo [Republic]"]
    assert _pins(republic) == [("from", _COG), ("to", _COG)]


def test_the_restatement_of_two_names_written_alike_asks_each_choice_again():
    """Posted back, the restatement of "aid from Congo to Congo" asks each end apart."""
    client, part = _congo_to_congo()

    again = _no_data(client, part["restatement"])

    assert again["reason"] == "entity_ambiguous"
    assert [group["text"] for group in again["ambiguous_entities"]] == ["Congo", "Congo"]
    assert [_pins(o) for o in again["options"]] == [_pins(o) for o in part["no_data"]["options"]]


def test_choosing_another_entity_keeps_two_names_written_alike_apart():
    """Choosing the World Bank's lender in "aid from Congo to Congo by the World Bank"
    posts both Congo rows under their names, so the next refusal asks each end apart and
    each of its options pins its own end alone."""
    client, part = _congo_to_congo(_WORLD_BANK)

    options = part["no_data"]["options"]
    (bank_option, _) = [o for o in options if o["mentions"] == ["World Bank"]]
    assert _congo_rows(_posted(bank_option)) == [
        ("Congo", _DRC, "from"),
        ("Congo", _ROC, "to"),
    ]

    again = _no_data(client, bank_option["data_query"]["parts"][0])

    assert again["reason"] == "entity_ambiguous"
    assert [group["text"] for group in again["ambiguous_entities"]] == ["Congo", "Congo"]
    assert [_pins(o) for o in again["options"]] == [
        [("from", ref), ("to", None), (None, _IBRD)] for ref in (_COD, _COG)
    ] + [[("from", None), ("to", ref), (None, _IBRD)] for ref in (_COD, _COG)]


def test_choosing_another_entity_keeps_a_name_its_words_leave_open_ambiguous():
    """The extraction names "Congolese" Congo, which names two countries, and "Congolese"
    alone names none. Choosing the World Bank's lender posts the Congo row in the user's
    word under its name, so the next refusal asks the Congo choice again."""
    client = _congo_flow_client(
        ExtractedEntity(text="World Bank", span="World Bank", flow_role="origin", role_span="from"),
        ExtractedEntity(text="Congo", span="Congolese", flow_role="destination", role_span="to"),
    )
    part = _raw_part(client, "aid from the World Bank to Congolese")

    options = part["no_data"]["options"]
    (bank_option, _) = [o for o in options if o["mentions"] == ["World Bank"]]
    (_, congo) = _posted(bank_option)
    assert congo == {"text": "Congolese", "name": "Congo", "role_text": "to"}

    again = _no_data(client, bank_option["data_query"]["parts"][0])

    assert again["reason"] == "entity_ambiguous"
    assert [group["text"] for group in again["ambiguous_entities"]] == ["Congolese"]
    assert [_pins(o) for o in again["options"]] == [
        [("from", _IBRD), ("to", ref)] for ref in (_COD, _COG)
    ]


def _pins(option: dict) -> list[tuple[str | None, str | None]]:
    """The direction and pinned dcid of each entity row an option posts."""
    return [
        (row.get("role_text"), row.get("dcid"))
        for row in option["data_query"]["parts"][0]["entities"]
    ]


def test_a_name_the_query_words_settle_keeps_its_name():
    """ "DR Congo" names one country, so the part answers under the extraction's name."""
    client = _fertility_client(_congo_extraction_llm("DR Congo"))

    part = _raw_part(client, "fertility rate in DR Congo")

    assert part["status"] == "definite"
    assert part["restatement"]["entities"] == [{"text": "Democratic Republic of the Congo"}]
    assert part["entity_refs"] == [{"text": "Democratic Republic of the Congo", "refs": [_COD]}]


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


@pytest.mark.parametrize("excluded_kind", ["provenance", "source"])
def test_source_choices_omit_a_candidate_removed_by_existing_exclusions(excluded_kind):
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
    excluded = rows[0].id if excluded_kind == "provenance" else rows[0].source.ref
    service = fake_catalogue_service(FakeCatalogue(provenances={row.id: row for row in rows}))
    with catalogue_service_module._singleton.poisoned_for_test(service):
        client = _fertility_client(_ShapeZeroLLM())
        no_data = _no_data(
            client,
            {
                "variable_text": _FERTILITY,
                "entities": [{"text": "Kenya"}],
                "sources": [{"text": words}, {"ref": excluded, "polarity": "exclude"}],
            },
        )
        assert no_data["reason"] == "source_ambiguous"
        (option,) = no_data["options"]
        assert option["data_query"]["parts"][0]["sources"] == [
            {"text": words, "ref": rows[1].id, "polarity": "include"},
            {"ref": excluded, "polarity": "exclude"},
        ]
        _assert_options_settle(client, no_data)


def test_a_source_wording_both_included_and_excluded_is_refused():
    """Every reading of the wording includes and excludes one candidate, so no choice can
    leave the part a source: the request is refused at the excluded mention."""
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
        response = _fertility_client(_ShapeZeroLLM()).post(
            "/api/search/resolve",
            json={
                "input": {
                    "kind": "parsed",
                    "parts": [
                        {
                            "variable_text": _FERTILITY,
                            "entities": [{"text": "Kenya"}],
                            "sources": [
                                {"text": words, "polarity": "include"},
                                {"text": words, "polarity": "exclude"},
                            ],
                        }
                    ],
                }
            },
        )
    assert response.status_code == 400, response.text
    problem = response.json()
    assert (problem["code"], problem["pointer"]) == ("request_invalid", "/input/parts/0/sources/1")


def test_a_contradiction_over_two_source_wordings_is_refused_once_one_is_chosen():
    """Each wording is both included and excluded, so every pair of choices empties the
    scope. Each reading of the first wording leaves the second ambiguous, so the part
    answers source_ambiguous. Once the first is chosen, every reading of the second
    empties the scope and the request is refused at its excluded mention."""
    rows = [
        ProvenanceRow(
            id=f"dc/base/{publisher}{words.replace(' ', '')}",
            data_commons=DFO,
            source=SourceRef(ref=f"dc/s/{publisher}", label=publisher),
            name=words,
            url="https://example.org",
        )
        for words, publishers in (
            ("Shared Tables", ("Alpha", "Beta")),
            ("Joint Ledgers", ("Gamma", "Delta")),
        )
        for publisher in publishers
    ]
    part = {
        "variable_text": _FERTILITY,
        "entities": [{"text": "Kenya"}],
        "sources": [
            {"text": "Shared Tables", "polarity": "include"},
            {"text": "Shared Tables", "polarity": "exclude"},
            {"text": "Joint Ledgers", "polarity": "include"},
            {"text": "Joint Ledgers", "polarity": "exclude"},
        ],
    }
    service = fake_catalogue_service(FakeCatalogue(provenances={row.id: row for row in rows}))
    with catalogue_service_module._singleton.poisoned_for_test(service):
        client = _fertility_client(_ShapeZeroLLM())
        no_data = _no_data(client, part)
        assert no_data["reason"] == "source_ambiguous"
        assert {group["text"] for group in no_data["ambiguous_sources"]} == {
            "Shared Tables",
            "Joint Ledgers",
        }
        chosen = [
            {**mention, "ref": rows[0].id} if mention["text"] == "Shared Tables" else mention
            for mention in part["sources"]
        ]
        response = client.post(
            "/api/search/resolve",
            json={"input": {"kind": "parsed", "parts": [{**part, "sources": chosen}]}},
        )
    assert response.status_code == 400, response.text
    problem = response.json()
    assert (problem["code"], problem["pointer"]) == ("request_invalid", "/input/parts/0/sources/3")


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
    assert [o.get("entity_choice") for o in no_data["options"]] == [None, None]
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


def test_an_untold_direction_tells_apart_two_places_written_alike():
    """ "Congo" and "Congo", pinned to the two countries, with no direction told. Each place
    keeps its own term, and the message and the option labels add each one's graph label.
    Every option answers."""
    client = _congo_flow_client()

    no_data = _no_data(
        client,
        {
            "variable_text": "aid",
            "entities": [
                {"text": "Congo", "dcid": _COD},
                {"text": "Congo", "dcid": _COG},
            ],
        },
    )

    assert no_data["reason"] == "role_not_resolved"
    assert no_data["entity_terms"] == ["Congo", "Congo"]
    assert [o["mentions"] for o in no_data["options"]] == [["Congo", "Congo"]] * 2
    assert [o["label"] for o in no_data["options"]] == [
        f'Congo ("{_DRC}") to Congo ("{_ROC}")',
        f'Congo ("{_ROC}") to Congo ("{_DRC}")',
    ]
    assert no_data["message"].split(". ", 1)[0] == (
        "Could not tell who is giving and who is receiving among "
        f'"Congo" ("{_DRC}") and "Congo" ("{_ROC}")'
    )
    for option in no_data["options"]:
        response = client.post("/api/search/resolve", json={"input": option["data_query"]})
        assert response.status_code == 200, (option["label"], response.text)
