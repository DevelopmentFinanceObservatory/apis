"""A coded refusal's wire `detail` names the values it refuses plainly and within the cap.

Each case raises one refusal at its door and reads the `detail` off the wire: the first
five values appear without a list or tuple repr around them, the rest appear only as a
count. A string the caller posted comes back cut, in the `detail` and in the `pointer`.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from code_index.members import SourceNotInFamily
from dfo_platform.named import MAX_VALUE_CHARS
from search.engine.options import population
from search.engine.retrieval import bound_values
from search.models import OptionsFor, OptionsRequest
from statvar_members import (
    DEFAULT_MEMBER_CAP,
    FiveTuple,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
)
from tests.engine._base_identity import GDP_SHAPE, make_identity, origin_entity, subject_entity
from tests.engine._builders import value_slot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._options_fixtures import (
    _GDP_PLACE_NODES,
    _LIFE_EXPECTANCY_ARCS,
    _LIFE_EXPECTANCY_SUBJECTS,
    _LIFE_EXPECTANCY_SV,
    _options_identity,
)
from tests.engine._refusal_detail import assert_cut, assert_names_plainly
from tests.engine._search_app import search_app
from tests.engine.test_app import make_client
from tests.fixtures import FakeGraph, FakeLLM, make_resolver
from tests.materialise.conftest import FakeGraphReader

_SEVEN = 7
_OVERSIZED = "k" * 4000
_LONG_DCID = "d" * 1000


def _dcids(prefix: str) -> list[str]:
    return [f"{prefix}/ZZ{chr(ord('A') + i)}" for i in range(_SEVEN)]


def _options_post(
    entities: dict[str, list[str]],
    *,
    node_types: dict[str, str],
    declared_roles: tuple[str, ...] = (),
    scope: str | None = None,
):
    """POST `/api/search/options` for the life-expectancy shape, whose members declare
    `declared_roles`, over a graph typing each dcid of `node_types`."""
    graph = FakeGraph(nodes={dcid: {"label": dcid, "type": t} for dcid, t in node_types.items()})
    arcs = {
        _LIFE_EXPECTANCY_SV: {
            **_LIFE_EXPECTANCY_ARCS[_LIFE_EXPECTANCY_SV],
            "observationProperties": declared_roles,
        }
    }
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=FakeGraphReader(arcs=arcs, subjects=_LIFE_EXPECTANCY_SUBJECTS),
    )
    request = OptionsRequest(
        identity=_options_identity(entities=entities),
        **{"for": OptionsFor(role="observationAbout", scope=scope)},
    )
    return TestClient(app, raise_server_exceptions=False).post(
        "/api/search/options", json=request.model_dump(by_alias=True, mode="json")
    )


def _resubmit_post(
    entities,
    *,
    monkeypatch=None,
    source_candidates=None,
    source="provenance/Elsewhere",
    places=(),
):
    """POST a GDP `dataset_resubmit` identity binding `entities`, over a graph that holds
    the GDP StatVar and every dcid of `places`."""
    activity_source = value_slot(
        "how", "activitySource", "activity source", "GrossDomesticProduction"
    )
    identity = make_identity(GDP_SHAPE, [activity_source], entities=entities)
    if source_candidates is not None:

        def refuse(_index, _query):
            raise SourceNotInFamily(source=source, candidates=source_candidates)

        monkeypatch.setattr(bound_values, "index_family", refuse)
    nodes = {**_GDP_PLACE_NODES, **{dcid: {"label": dcid, "type": "Country"} for dcid in places}}
    return make_client(graph=FakeGraph(nodes=nodes)).post(
        "/api/search/resolve",
        json={
            "input": {
                "kind": "dataset_resubmit",
                "entries": [
                    {
                        "entry_kind": "identity",
                        "variable_text": "a resubmitted identity",
                        "identity": identity.model_dump(mode="json"),
                    }
                ],
            }
        },
    )


def _absent_refs(monkeypatch):
    values = _dcids("country")
    return _resubmit_post([subject_entity(dcid) for dcid in values]), values, ()


def _origin_on_a_standard_shape(monkeypatch):
    values = _dcids("country")
    entities = [subject_entity("country/IND"), *(origin_entity(dcid) for dcid in values)]
    return _resubmit_post(entities, places=[*values, "country/IND"]), values, ()


def _source_outside_family(monkeypatch):
    values = _dcids("provenance")
    response = _resubmit_post(
        [subject_entity("country/IND")], monkeypatch=monkeypatch, source_candidates=tuple(values)
    )
    return response, values, ()


def _direction_on_a_subject_only_population(monkeypatch):
    values = _dcids("country")
    nodes = dict.fromkeys(values, "Country")
    return _options_post({"destination": values}, node_types=nodes), values, ()


def _axis_the_population_declares_no_role_for(monkeypatch):
    values = _dcids("country")
    nodes = dict.fromkeys(values, "Country")
    response = _options_post(
        {"origin": values}, node_types=nodes, declared_roles=("financingRecipient",)
    )
    return response, values, ("destination",)


def _anchors_of_two_types(monkeypatch):
    nodes = {"country/ZZA": "Country", "state/ZZB": "State"}
    response = _options_post({"observationAbout": list(nodes)}, node_types=nodes, scope="peers")
    return response, [], ("Country", "State")


def _enumeration_raising(monkeypatch, error: Exception):
    """POST `/api/search/options` over a member walk that raises `error`."""

    class _Refusing:
        member_cap = DEFAULT_MEMBER_CAP

        def enumerate(self, **kwargs):
            raise error

    monkeypatch.setattr(population, "shared_enumerator", _Refusing)
    return _options_post({}, node_types={})


def _candidate_with_many_constraint_values(monkeypatch):
    values = [f"Lender{chr(ord('A') + i)}" for i in range(_SEVEN)]
    error = MultiValuedConstraint(
        stat_var=_LIFE_EXPECTANCY_SV, property="lendingEntity", values=tuple(values)
    )
    return _enumeration_raising(monkeypatch, error), values, (_LIFE_EXPECTANCY_SV,)


def _candidate_of_many_types(monkeypatch):
    values = [f"Type{chr(ord('A') + i)}" for i in range(_SEVEN)]
    error = NotAStatisticalVariable(stat_var=_LIFE_EXPECTANCY_SV, type_of=tuple(values))
    return _enumeration_raising(monkeypatch, error), values, (_LIFE_EXPECTANCY_SV,)


def _members_of_two_role_signatures(monkeypatch):
    values = _dcids("dfo/member")
    roles = {dcid: ("financingRecipient",) if i % 2 else () for i, dcid in enumerate(values)}
    error = MemberRoleDisagreement(five_tuple=FiveTuple(), roles_by_stat_var=roles)
    return _enumeration_raising(monkeypatch, error), values, ()


_CASES: dict[str, tuple[Callable, str]] = {
    "refs-absent-from-the-graph": (_absent_refs, "invalid_role_ref"),
    "origin-on-a-standard-shape": (_origin_on_a_standard_shape, "role_incompatible"),
    "source-outside-the-family": (_source_outside_family, "facet_not_available"),
    "direction-on-a-subject-only-population": (
        _direction_on_a_subject_only_population,
        "role_incompatible",
    ),
    "axis-the-population-declares-no-role-for": (
        _axis_the_population_declares_no_role_for,
        "role_incompatible",
    ),
    "anchors-of-two-types": (_anchors_of_two_types, "mixed_peers_anchor_types"),
    "candidate-with-many-constraint-values": (
        _candidate_with_many_constraint_values,
        "request_invalid",
    ),
    "candidate-of-many-types": (_candidate_of_many_types, "request_invalid"),
    "members-of-two-role-signatures": (
        _members_of_two_role_signatures,
        "role_signature_incompatible",
    ),
}


@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_a_refusal_names_its_values_plainly_and_capped(case, monkeypatch):
    """Values are named plainly: seven dcids come back as five and a count of the other two."""
    raise_it, code = case
    response, values, always = raise_it(monkeypatch)

    assert response.status_code in (400, 422), response.text
    body = response.json()
    assert body["code"] == code
    assert_names_plainly(body["detail"], values, always=always)


def _gdp_identity() -> dict:
    activity_source = value_slot(
        "how", "activitySource", "activity source", "GrossDomesticProduction"
    )
    return make_identity(GDP_SHAPE, [activity_source]).model_dump(mode="json")


def _raw_json(body: dict) -> dict:
    """`content` and `headers` that post `body` as ASCII JSON, so a key holding a lone
    surrogate reaches the door. The client's own `json=` refuses to encode it."""
    return {"content": json.dumps(body), "headers": {"content-type": "application/json"}}


def _resolve_post(input_: dict):
    return make_client().post("/api/search/resolve", **_raw_json({"input": input_}))


def _resubmit_identity(edit: Callable[[dict], object], **entry_members):
    """POST a `dataset_resubmit` of the GDP identity after `edit` changed it in place."""
    identity = _gdp_identity()
    edit(identity)
    entry = {
        "entry_kind": "identity",
        "variable_text": "an identity",
        "identity": identity,
        **entry_members,
    }
    return _resolve_post({"kind": "dataset_resubmit", "entries": [entry]})


def _options_identity_post(edit: Callable[[dict], object], **for_members):
    """POST `/api/search/options` for the life-expectancy identity after `edit` changed it."""
    request = OptionsRequest(
        identity=_options_identity(), **{"for": OptionsFor(role="observationAbout")}
    )
    body = request.model_dump(by_alias=True, mode="json")
    edit(body["identity"])
    body["for"].update(for_members)
    graph = FakeGraph()
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(observations={}),
        materialise_graph=FakeGraphReader(
            arcs=_LIFE_EXPECTANCY_ARCS, subjects=_LIFE_EXPECTANCY_SUBJECTS
        ),
    )
    return TestClient(app, raise_server_exceptions=False).post(
        "/api/search/options", **_raw_json(body)
    )


_IDENTITY_DOORS = pytest.mark.parametrize(
    "post", [_resubmit_identity, _options_identity_post], ids=["resubmit", "options"]
)
_KEYED = pytest.mark.parametrize(
    ("member", "value"),
    [("entities", ["country/KEN"]), ("facets", {"ref": "x"})],
    ids=["entity-role", "facet-axis"],
)


@_IDENTITY_DOORS
@_KEYED
def test_unknown_identity_keys_are_named_plainly_and_capped(post, member, value):
    """Eight unknown keys come back as the first five and a count of the other three."""
    keys = [f"bad_{letter}" for letter in "abcdefgh"]
    response = post(lambda identity: identity.update({member: dict.fromkeys(keys, value)}))

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert_names_plainly(body["detail"], keys)


@_IDENTITY_DOORS
@_KEYED
def test_an_oversized_identity_key_is_named_by_its_first_characters(post, member, value):
    """A 4,000-character key is named by its first 32 characters."""
    response = post(lambda identity: identity.update({member: {_OVERSIZED: value}}))

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert _OVERSIZED[:32] in detail
    assert _OVERSIZED[:33] not in detail


@_IDENTITY_DOORS
@_KEYED
def test_a_key_that_utf8_cannot_encode_is_refused_with_a_bounded_body(post, member, value):
    """A lone surrogate is valid JSON and cannot be encoded in a message. The door answers
    400 as it does for any other unknown key."""
    response = post(lambda identity: identity.update({member: {"\ud83d": value}}))

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
    assert len(response.content) < MAX_VALUE_CHARS * 4


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_a_raw_query_holding_a_lone_surrogate_is_refused(surrogate):
    """A raw query the response could not echo is refused at the door, as a parsed part's
    text is."""
    response = _resolve_post({"kind": "raw_text", "query": f"aid to Kenya{surrogate}"})

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input/query"


@_IDENTITY_DOORS
def test_an_open_role_the_entities_bind_is_named_plainly(post):
    """The roles both members name come back without a list repr around them."""
    roles = ["destination", "origin"]
    response = post(
        lambda identity: identity.update(
            entities={role: ["country/KEN"] for role in roles}, open_roles=roles
        )
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["code"] == "request_invalid"
    assert_names_plainly(body["detail"], roles)


def _set_member(member: str, value: object) -> Callable[[dict], None]:
    return lambda identity: identity.update({member: value})


def _set_slot(**members) -> Callable[[dict], None]:
    """Give the identity one slot: the GDP identity's own, with `members` over it."""
    slot = {**_gdp_identity()["slots"][0], **members}
    return lambda identity: identity.update(slots=[slot])


_IDENTITY_ECHOES: dict[str, tuple[Callable[[dict], object], dict, str, str, str]] = {
    # edit of the identity, members posted beside it, the oversized string, code, pointer tail
    "role-key-over-a-valid-list": (
        _set_member("entities", {_OVERSIZED: ["country/KEN"]}),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity",
    ),
    "role-key-over-an-empty-list": (
        _set_member("entities", {_OVERSIZED: []}),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity/entities",
    ),
    "role-key-over-a-non-string-member": (
        _set_member("entities", {_OVERSIZED: [1]}),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity/entities",
    ),
    "facet-axis-key-over-a-binding": (
        _set_member("facets", {_OVERSIZED: {"ref": "x"}}),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity",
    ),
    "facet-axis-key-over-a-non-binding": (
        _set_member("facets", {_OVERSIZED: 1}),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity",
    ),
    "member-the-identity-does-not-declare": (
        _set_member(_OVERSIZED, 1),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity",
    ),
    "shape-kind": (
        lambda identity: identity["shape"].update(kind=_OVERSIZED),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity/shape",
    ),
    "slot-binding-state": (
        _set_slot(binding={"state": _OVERSIZED}),
        {},
        _OVERSIZED,
        "request_invalid",
        "/identity/slots/0/binding",
    ),
    "exact-binding-on-a-long-property": (
        _set_slot(property=_LONG_DCID, binding={"state": "exact", "ref": "x"}),
        {},
        _LONG_DCID,
        "exact_binding_unsupported",
        "/identity/slots/0/binding",
    ),
    "slot-ref-the-identity-core-refuses": (
        _set_slot(binding={"state": "value", "ref": f"a {_LONG_DCID}"}),
        {},
        _LONG_DCID,
        "request_invalid",
        "/identity",
    ),
}


@_IDENTITY_DOORS
@pytest.mark.parametrize("case", _IDENTITY_ECHOES.values(), ids=_IDENTITY_ECHOES.keys())
def test_an_identity_refusal_echoes_no_oversized_string(post, case):
    """A long key, tag or dcid in a posted identity comes back cut. The body never holds it
    whole, and the pointer stops at the object that holds an oversized key."""
    edit, beside, oversized, code, pointer_tail = case
    response = post(edit, **beside)

    assert response.status_code in (400, 501), response.text
    body = response.json()
    assert body["code"] == code
    assert body["pointer"].endswith(pointer_tail)
    assert_cut(response, oversized)


def _resubmit_entry(entry: dict):
    return _resolve_post({"kind": "dataset_resubmit", "entries": [entry]})


def _parsed_part(**members):
    part = {"variable_text": "gdp", "entities": [{"text": "Kenya"}], **members}
    return _resolve_post({"kind": "parsed", "parts": [part]})


def _long_source_outside_the_family(monkeypatch):
    return _resubmit_post(
        [subject_entity("country/IND")],
        monkeypatch=monkeypatch,
        source_candidates=tuple(f"{_LONG_DCID}{i}" for i in range(_SEVEN)),
        source=_LONG_DCID,
    )


_DOOR_ECHOES: dict[str, tuple[Callable, str, str, str | None]] = {
    # the post, the oversized string, code, pointer
    "entity-groups-key": (
        lambda _: _resubmit_identity(lambda _identity: None, entity_groups={_OVERSIZED: {}}),
        _OVERSIZED,
        "request_invalid",
        "/input/entries/0/entity_groups",
    ),
    "entry-member-the-model-does-not-declare": (
        lambda _: _resubmit_identity(lambda _identity: None, **{_OVERSIZED: 1}),
        _OVERSIZED,
        "request_invalid",
        "/input/entries/0",
    ),
    "entry-kind": (
        lambda _: _resubmit_entry({"entry_kind": _OVERSIZED}),
        _OVERSIZED,
        "request_invalid",
        "/input/entries/0",
    ),
    "input-kind": (
        lambda _: _resolve_post({"kind": _OVERSIZED}),
        _OVERSIZED,
        "request_invalid",
        "/input",
    ),
    "placeholder-code": (
        lambda _: _resubmit_entry(
            {
                "entry_kind": "placeholder",
                "variable_text": "a part",
                "status": "unsupported",
                "code": _OVERSIZED,
            }
        ),
        _OVERSIZED,
        "request_invalid",
        "/input/entries/0/code",
    ),
    "refs-absent-from-the-graph": (
        lambda _: _resubmit_post([subject_entity(_LONG_DCID)]),
        _LONG_DCID,
        "invalid_role_ref",
        "/input/entries/0/identity/entities/observationAbout",
    ),
    "source-outside-the-family": (
        _long_source_outside_the_family,
        _LONG_DCID,
        "facet_not_available",
        "/input/entries/0/identity/facets/provenance",
    ),
    "part-pinning-one-text-twice": (
        lambda _: _parsed_part(
            entities=[
                {"text": "Kenya", "dcid": f"a{_LONG_DCID}"},
                {"text": "Kenya", "dcid": f"b{_LONG_DCID}"},
            ]
        ),
        _LONG_DCID,
        "request_invalid",
        "/input/parts/0/entities",
    ),
    "part-member-the-model-does-not-declare": (
        lambda _: _parsed_part(**{_OVERSIZED: 1}),
        _OVERSIZED,
        "request_invalid",
        "/input/parts/0",
    ),
    "options-target-role": (
        lambda _: _options_identity_post(lambda _identity: None, role=_OVERSIZED),
        _OVERSIZED,
        "option_target_invalid",
        "/for/role",
    ),
    "options-target-member-the-model-does-not-declare": (
        lambda _: _options_identity_post(lambda _identity: None, **{_OVERSIZED: 1}),
        _OVERSIZED,
        "request_invalid",
        "/for",
    ),
}


@pytest.mark.parametrize("case", _DOOR_ECHOES.values(), ids=_DOOR_ECHOES.keys())
def test_a_refusal_echoes_no_oversized_string(case, monkeypatch):
    """A long key, tag, code or dcid posted outside an identity comes back cut."""
    post, oversized, code, pointer = case
    response = post(monkeypatch)

    assert response.status_code in (400, 422), response.text
    body = response.json()
    assert body["code"] == code
    assert body["pointer"] == pointer
    assert_cut(response, oversized)
