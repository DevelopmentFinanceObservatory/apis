"""`EntityResolver.resolve` with a `kinds` filter, the `code` kind, `matched` and `reading`,
registry-backed groups, and the deadline of a job queued on the shared worker.

Each test uses resolvekit's composed pack and a fresh FakeGraphLookup."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from dataclasses import replace

import pytest
from resolvekit.core.model.result import CandidateSummary

import entity_resolution.resolver as resolver_module
import entity_resolution.shared as shared_module
from code_index.shared import get_index
from dc_graph.errors import GraphDeadlineExceededError
from entity_resolution import (
    Ambiguous,
    CodeHit,
    Entity,
    EntityResolver,
    Group,
    Matched,
    NotFound,
    NotHeld,
    Resolved,
)
from entity_resolution.membership import DEFAULT_PROJECTION, MembershipPolicy
from entity_resolution.shared import shared_memberships
from entity_resolution.types import DEFAULT_KINDS

from .conftest import FakeGraphLookup

_LAST_RUNG_ARC = "<-description->dcid"
_LAGOS_STATE, _LAGOS_CITY = "wikidataId/Q815913", "wikidataId/Q8673"
_PACIFIC_TOWN = "geoId/5352495"
_UNICEF = "United Nations Children's Fund"
_OPEC = "Organization of the Petroleum Exporting Countries"


def _node(graph_type: str, *names: str) -> dict[str, tuple[str, ...]]:
    return {"typeOf": (graph_type,), "name": names or ("graph-name-unused",)}


_NODES = {
    "country/KEN": _node("Country"),
    "country/COD": _node("Country"),
    "country/COG": _node("Country"),
    "country/FSM": _node("Country"),
    "Micronesia": _node("UNGeoRegion"),
    "EuropeanUnion": _node("ContinentalUnion", "European Union", "Qqqunion"),
    _LAGOS_STATE: _node("AdministrativeArea1", "Lagos"),
    _LAGOS_CITY: _node("City", "Lagos"),
    _PACIFIC_TOWN: _node("City", "Pacific"),
}
_ARCS = {
    (_LAST_RUNG_ARC, "Lagos"): [_LAGOS_STATE, _LAGOS_CITY],
    (_LAST_RUNG_ARC, "Pacific"): [_PACIFIC_TOWN],
    (_LAST_RUNG_ARC, "Qqqunion"): ["EuropeanUnion"],
}


def _run(coro):
    return asyncio.run(coro)


def _resolve_one(shared_resolver, text: str, **kwargs):
    resolver = EntityResolver(
        graph=FakeGraphLookup(nodes=_NODES, arcs=_ARCS), shared=shared_resolver
    )
    (result,) = _run(resolver.resolve([text], **kwargs))
    return result


def _refs(result) -> set[str]:
    if isinstance(result, Resolved):
        return {result.entity.dcid}
    if isinstance(result, Group):
        return {result.dcid}
    if isinstance(result, Ambiguous):
        return {candidate.dcid for candidate in result.candidates}
    return set()


@pytest.mark.parametrize(
    ("text", "kinds", "outcome", "refs"),
    [
        pytest.param("Lagos", None, Ambiguous, {_LAGOS_STATE, _LAGOS_CITY}, id="lagos-unfiltered"),
        pytest.param(
            "Lagos", ["place_other"], Ambiguous, {_LAGOS_STATE, _LAGOS_CITY}, id="lagos-places"
        ),
        pytest.param(
            "Micronesia", ["country"], Resolved, {"country/FSM"}, id="micronesia-one-left"
        ),
        pytest.param(
            "Congo", ["country"], Ambiguous, {"country/COD", "country/COG"}, id="congo-both-kept"
        ),
        pytest.param("Kenya", ["country"], Resolved, {"country/KEN"}, id="kenya-kept"),
        pytest.param("EU", ["group"], Group, {"EuropeanUnion"}, id="eu-as-a-group"),
    ],
)
def test_a_kinds_filter_keeps_the_candidates_of_the_requested_kinds(
    shared_resolver, text, kinds, outcome, refs
):
    result = _resolve_one(shared_resolver, text, kinds=kinds)

    assert isinstance(result, outcome), result
    assert _refs(result) == refs


@pytest.mark.parametrize(
    ("text", "kinds", "removed"),
    [
        pytest.param(
            "Lagos",
            ["country"],
            {(_LAGOS_STATE, "place_other"), (_LAGOS_CITY, "place_other")},
            id="lagos-fallback",
        ),
        pytest.param(
            "Commonwealth",
            ["country"],
            {("groups/Commonwealth", "group")},
            id="commonwealth-group",
        ),
        pytest.param("Pacific", ["country"], {(_PACIFIC_TOWN, "place_other")}, id="pacific-town"),
        pytest.param("G7", ["country"], {("GroupOfSeven", "group")}, id="g7-registry-group"),
        # The registry holds the EU, so its `ContinentalUnion` node is a group, as the node
        # route classes it, on the survivor path and on the last rung alike.
        pytest.param(
            "EU", ["continental_union"], {("EuropeanUnion", "group")}, id="eu-node-is-a-group"
        ),
        pytest.param(
            "Qqqunion",
            ["continental_union"],
            {("EuropeanUnion", "group")},
            id="eu-node-on-the-last-rung-is-a-group",
        ),
        pytest.param(
            "Micronesia",
            ["organisation"],
            {("country/FSM", "country"), ("Micronesia", "subregion")},
            id="micronesia-hits",
        ),
    ],
)
def test_a_filter_that_removes_every_candidate_names_them_with_their_kinds(
    shared_resolver, text, kinds, removed
):
    result = _resolve_one(shared_resolver, text, kinds=kinds)

    assert isinstance(result, NotFound), result
    assert {(entity.dcid, entity.kind) for entity in result.removed} == removed


def test_a_registry_group_rejected_by_the_gate_is_a_group_near_match(shared_resolver):
    result = _resolve_one(shared_resolver, "European Onion")

    assert isinstance(result, NotFound), result
    assert ("EuropeanUnion", "group") in {(e.dcid, e.kind) for e in result.near_matches}


def test_a_confirmed_node_takes_the_most_specific_kind_of_all_its_types(shared_resolver):
    nodes = {"country/KEN": {"typeOf": ("Place", "Country"), "name": ("Kenya",)}}
    resolver = EntityResolver(graph=FakeGraphLookup(nodes=nodes), shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Kenya"], kinds=["country"]))

    assert isinstance(result, Resolved), result
    assert result.entity.kind == "country"


@pytest.mark.parametrize("graph_type", ["EurostatNUTS2", "GeoRegion", "SeaIce"])
def test_a_last_rung_node_of_any_place_type_is_a_place_the_default_kinds_keep(
    shared_resolver, graph_type
):
    """`SeaIce` sits four `subClassOf` hops below `Place`."""
    graph = FakeGraphLookup(
        nodes={"nuts/QQ11": _node(graph_type, "Qqqshire")},
        arcs={(_LAST_RUNG_ARC, "Qqqshire"): ["nuts/QQ11"]},
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Qqqshire"], kinds=DEFAULT_KINDS))

    assert isinstance(result, Resolved), result
    assert result.entity.kind == "place_other"


@pytest.mark.parametrize(
    ("kinds", "outcome"),
    [
        pytest.param(
            ["country"],
            NotFound(removed=(Entity("UnitedNationsChildrensFund", _UNICEF, "organisation", ""),)),
            id="excluded-kind-not-found",
        ),
        pytest.param(
            ["organisation"],
            NotHeld(dcid="UnitedNationsChildrensFund", name=_UNICEF),
            id="admitted-kind-not-held",
        ),
    ],
)
def test_a_hit_the_graph_does_not_hold_is_filtered_by_its_resolvekit_kind(
    shared_resolver, kinds, outcome
):
    resolver = EntityResolver(graph=FakeGraphLookup(), shared=shared_resolver)

    (result,) = _run(resolver.resolve(["UNICEF"], kinds=kinds))

    assert result == outcome


@pytest.mark.parametrize(
    ("kinds", "outcome"),
    [
        pytest.param(None, NotHeld(dcid="groups/OPEC", name=_OPEC), id="default-kinds-not-held"),
        pytest.param(
            ["country"],
            NotFound(removed=(Entity("groups/OPEC", _OPEC, "organisation", ""),)),
            id="excluded-kind-not-found",
        ),
    ],
)
def test_an_unheld_membership_group_the_registry_lacks_is_an_organisation(
    shared_resolver, monkeypatch, kinds, outcome
):
    """Every resolvekit group is in the registry today, so the registry and the policy are
    made to miss OPEC."""
    held = shared_memberships()
    policy = held.policy
    monkeypatch.setattr(resolver_module, "_group_hit", lambda _shared, _candidate: None)
    monkeypatch.setattr(
        shared_module,
        "_memberships",
        replace(
            held,
            policy=MembershipPolicy(
                [m for m in policy if m.ref != "groups/OPEC"],
                omitted_regions=policy.omitted_regions,
            ),
        ),
    )
    resolver = EntityResolver(graph=FakeGraphLookup(), shared=shared_resolver)

    (result,) = _run(resolver.resolve(["OPEC"], kinds=DEFAULT_KINDS if kinds is None else kinds))

    assert result == outcome


@pytest.mark.parametrize(
    ("kinds", "outcome"),
    [
        pytest.param(None, Resolved, id="default-kinds-resolve-the-group"),
        pytest.param(["country"], NotFound, id="excluded-kind-removes-a-group"),
    ],
)
def test_a_registry_group_with_no_node_that_is_not_the_top_survivor_is_a_group(
    shared_resolver, monkeypatch, kinds, outcome
):
    """A registry group survivor the graph holds no node for, where the text names no group
    (`_GateOutcome.group` is `None`, as for any survivor below the top one), is kind `group`
    with the policy's label, as on every other path."""
    monkeypatch.setattr(resolver_module, "_group_hit", lambda _shared, _candidate: None)
    resolver = EntityResolver(graph=FakeGraphLookup(), shared=shared_resolver)
    group = Entity("groups/OPEC", shared_resolver.policy.get("groups/OPEC").label, "group", "")

    (result,) = _run(resolver.resolve(["OPEC"], kinds=DEFAULT_KINDS if kinds is None else kinds))

    assert isinstance(result, outcome), result
    found = result.entity if isinstance(result, Resolved) else result.removed[0]
    assert found == group


def test_kinds_join_the_cache_key_and_their_order_does_not(shared_resolver, monkeypatch):
    calls: list[list[str]] = []
    original = resolver_module._resolve_batch_job

    def counting(shared, code_index, texts):
        calls.append(list(texts))
        return original(shared, code_index, texts)

    monkeypatch.setattr(resolver_module, "_resolve_batch_job", counting)
    resolver = EntityResolver(
        graph=FakeGraphLookup(nodes=_NODES, arcs=_ARCS), shared=shared_resolver
    )

    (unfiltered,) = _run(resolver.resolve(["Lagos"]))
    (places,) = _run(resolver.resolve(["Lagos"], kinds=["place_other", "country"]))
    (again,) = _run(resolver.resolve(["Lagos"], kinds=["country", "place_other"]))
    (countries,) = _run(resolver.resolve(["Lagos"], kinds=["country"]))

    assert isinstance(unfiltered, Ambiguous)
    assert again == places
    assert isinstance(countries, NotFound)
    assert len(calls) == 3


# --- matched and reading -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "matched"),
    [
        ("Kenya", Matched(text="Kenya", matched_as="name")),
        ("KE", Matched(text="KE", matched_as="code")),
        ("KEN", Matched(text="KEN", matched_as="code")),
        ("Burma", Matched(text="Burma", matched_as="alias")),
        ("G20", Matched(text="G20", matched_as="alias")),
        # A query alias row is the listed text.
        ("EU countries", Matched(text="EU countries", matched_as="alias")),
        # The retry matched the stripped text.
        ("the EU", Matched(text="EU", matched_as="alias")),
    ],
)
def test_a_result_says_which_listed_text_it_matched_and_how(shared_resolver, text, matched):
    nodes = {**_NODES, "country/MMR": _node("Country")}
    resolver = EntityResolver(graph=FakeGraphLookup(nodes=nodes), shared=shared_resolver)

    (result,) = _run(resolver.resolve([text]))

    assert isinstance(result, Resolved | Group), result
    assert result.matched == matched


@pytest.mark.parametrize(
    ("graph_name", "alternate_name", "matched_as"),
    [("Qqqlandia", "Qqq", "name"), ("Qqq", "Qqqlandia", "alias")],
)
def test_a_fallback_hit_matches_as_its_name_or_as_an_alternate_name(
    shared_resolver, graph_name, alternate_name, matched_as
):
    graph = FakeGraphLookup(
        nodes={"dcid/one": {**_node("Country", graph_name), "alternateName": (alternate_name,)}},
        arcs={("<-description{typeOf:Country}->dcid", "Qqqlandia"): ["dcid/one"]},
    )
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["Qqqlandia"]))

    assert isinstance(result, Resolved), result
    assert result.matched == Matched(text="Qqqlandia", matched_as=matched_as)


@pytest.mark.parametrize(
    ("text", "reading"),
    [
        ("EU countries", "members"),
        ("DAC members", "members"),
        ("EU", "entity"),
        ("Kenya", "entity"),
    ],
)
def test_reading_is_members_only_for_a_text_naming_a_groups_members(shared_resolver, text, reading):
    result = _resolve_one(shared_resolver, text)

    assert isinstance(result, Resolved | Group), result
    assert result.reading == reading


# --- groups from the registry --------------------------------------------------------------


def test_a_group_resolves_from_the_registry_with_its_id_label_and_count(shared_resolver):
    graph = FakeGraphLookup()
    resolver = EntityResolver(graph=graph, shared=shared_resolver)

    (result,) = _run(resolver.resolve(["G20"]))

    membership = shared_resolver.groups.get("groups/G20")
    assert isinstance(result, Group), result
    assert result.dcid == "groups/G20"
    assert result.label == membership.label
    assert result.member_count == membership.member_count(DEFAULT_PROJECTION) == 19
    assert [member.dcid for member in result.members] == list(membership.members())
    assert graph.node_arcs_many_calls == [(("groups/G20",), ("typeOf", "name"))]


def test_the_shared_resolver_carries_the_registry_and_the_policy(shared_resolver):
    assert "groups/G20" in shared_resolver.groups
    assert "groups/G20" in shared_resolver.policy
    assert shared_resolver.policy.members("groups/G20").members == (
        shared_resolver.groups.get("groups/G20").members()
    )


# --- the code kind -------------------------------------------------------------------------

_CODE = Entity(
    dcid="CRSPurposeCode12220",
    name="Basic health care",
    kind="code",
    graph_type="CRSPurposeCodeEnum",
)


_CODE_HIT = CodeHit(entity=_CODE, text="basic health care", matched_as="name")


class _FakeCodes:
    def __init__(self, hits: dict[str, Sequence[CodeHit]]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, str | None]] = []

    def codes_by_text(self, text: str, *, code_system: str | None) -> Sequence[CodeHit]:
        self.calls.append((text, code_system))
        return [
            hit
            for hit in self.hits.get(text, ())
            if code_system is None or hit.entity.graph_type == code_system
        ]


def test_the_code_kind_alone_confirms_codes_in_one_read_and_runs_no_resolvekit_job(
    shared_resolver, monkeypatch
):
    batch_jobs: list[object] = []
    monkeypatch.setattr(resolver_module, "_resolve_batch_job", lambda *a: batch_jobs.append(a))
    codes = _FakeCodes({"basic health care": [_CODE_HIT]})
    graph = FakeGraphLookup(nodes={_CODE.dcid: _node("CRSPurposeCodeEnum")})
    resolver = EntityResolver(
        graph=graph, shared=shared_resolver, codes=codes, code_index=get_index
    )

    (hit,) = _run(resolver.resolve(["basic health care"], kinds=["code"]))
    (narrowed,) = _run(
        resolver.resolve(["basic health care"], kinds=["code"], code_system="DAC1AidTypeEnum")
    )

    assert hit == Resolved(
        entity=_CODE, matched=Matched(text="basic health care", matched_as="name")
    )
    assert isinstance(narrowed, NotFound)
    assert codes.calls[-1] == ("basic health care", "DAC1AidTypeEnum")
    assert batch_jobs == []
    assert graph.node_arcs_many_calls == [((_CODE.dcid,), ("typeOf", "name"))]
    assert graph.resolve_by_arc_calls == []


def test_a_code_the_graph_does_not_hold_is_not_a_candidate(shared_resolver):
    codes = _FakeCodes({"basic health care": [_CODE_HIT]})
    resolver = EntityResolver(graph=FakeGraphLookup(), shared=shared_resolver, codes=codes)

    (result,) = _run(resolver.resolve(["basic health care"], kinds=["code"]))

    assert result == NotFound()


def test_a_code_and_an_entity_listing_one_text_are_ambiguous_after_one_read(shared_resolver):
    code = Entity(dcid="SomeCode", name="Kenya", kind="code", graph_type="SomeEnum")
    codes = _FakeCodes({"kenya": [CodeHit(entity=code, text="Kenya", matched_as="alias")]})
    graph = FakeGraphLookup(nodes={**_NODES, "SomeCode": _node("SomeEnum")})
    resolver = EntityResolver(graph=graph, shared=shared_resolver, codes=codes)

    (result,) = _run(resolver.resolve(["Kenya"], kinds=["country", "code"]))

    assert isinstance(result, Ambiguous), result
    assert {c.dcid for c in result.candidates} == {"country/KEN", "SomeCode"}
    assert len(graph.node_arcs_many_calls) == 1


_AID_CODE = "DAC1AidType2102"


@pytest.mark.parametrize("path", ["shared_name", "resolvekit_hit", "fallback"])
@pytest.mark.parametrize(
    ("code_system", "admitted"),
    [("CRSPurposeCodeEnum", False), ("DAC1AidTypeEnum", True), (None, True)],
)
def test_a_code_system_narrows_the_codes_every_path_finds(
    shared_resolver, monkeypatch, path, code_system, admitted
):
    """A code a path other than the code index finds is admitted under `code_system` only
    when the code index places it in that system, as a code-index hit is: a graph node
    sharing a hit's text ("European Union" beside the EU), a resolvekit survivor, and a
    node the miss ladder's last rung answers."""
    text = {"shared_name": "European Union", "resolvekit_hit": "Kenya", "fallback": "Qqqaid"}[path]
    if path == "resolvekit_hit":
        survivor = CandidateSummary(
            entity_id=f"dfo_graph/{_AID_CODE}",
            confidence=0.9,
            canonical_name="Multilateral contribution to EU institutions",
            pack_id="org",
        )
        monkeypatch.setattr(
            resolver_module,
            "_resolve_batch_job",
            lambda *_: [resolver_module._GateOutcome(survivors=(survivor,), near_matches=())],
        )
    graph = FakeGraphLookup(
        nodes={**_NODES, _AID_CODE: _node("DAC1AidTypeEnum", text)},
        arcs={(_LAST_RUNG_ARC, text): ["EuropeanUnion", _AID_CODE]},
    )
    resolver = EntityResolver(
        graph=graph, shared=shared_resolver, codes=_FakeCodes({}), code_index=get_index
    )

    (result,) = _run(
        resolver.resolve(
            [text], kinds=["country", "code"], code_system=code_system, near_matches=True
        )
    )

    if admitted:
        assert isinstance(result, Resolved), result
        assert result.entity.dcid == _AID_CODE
    else:
        assert isinstance(result, NotFound), result


@pytest.mark.parametrize(
    ("kinds", "code_system", "codes"),
    [
        pytest.param(["country"], "CRSPurposeCodeEnum", _FakeCodes({}), id="system-without-code"),
        pytest.param(None, "CRSPurposeCodeEnum", _FakeCodes({}), id="system-unfiltered"),
        pytest.param(["code"], None, None, id="code-without-lookup"),
        pytest.param(
            ["code"], "CRSPurposeCodeEnum", _FakeCodes({}), id="system-without-code-index"
        ),
    ],
)
def test_a_code_request_the_resolver_cannot_answer_is_refused(
    shared_resolver, kinds, code_system, codes
):
    resolver = EntityResolver(graph=FakeGraphLookup(), shared=shared_resolver, codes=codes)

    with pytest.raises(ValueError):
        _run(resolver.resolve(["x"], kinds=kinds, code_system=code_system))


# --- the deadline of a queued job ----------------------------------------------------------


def test_a_job_queued_past_its_deadline_does_not_start(shared_resolver, monkeypatch):
    batch_jobs: list[object] = []
    monkeypatch.setattr(resolver_module, "_resolve_batch_job", lambda *a: batch_jobs.append(a))
    resolver = EntityResolver(graph=FakeGraphLookup(nodes=_NODES), shared=shared_resolver)
    started, release = threading.Event(), threading.Event()

    def hold_the_worker() -> None:
        started.set()
        release.wait(timeout=5)

    async def resolve_behind_a_running_job():
        queued = asyncio.ensure_future(
            resolver.resolve(["Kenya"], deadline=time.monotonic() + 0.05)
        )
        await asyncio.sleep(0.15)
        release.set()
        return await queued

    shared_resolver.run(hold_the_worker)
    try:
        assert started.wait(timeout=5)
        with pytest.raises(GraphDeadlineExceededError):
            _run(resolve_behind_a_running_job())
    finally:
        release.set()

    assert batch_jobs == []


def test_a_group_no_longer_returns_not_held_when_the_graph_holds_none_of_it(shared_resolver):
    result = _resolve_one(shared_resolver, "G7")

    assert not isinstance(result, NotHeld)
    assert isinstance(result, Group)
    assert result.member_count == 7
