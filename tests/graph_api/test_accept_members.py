"""Acceptance cases for `GET /api/graph/node/{dcid}/members`: page bounds, the cursor's binding
to its node and `as_of` date, a walk across two code index builds, group membership under the
design's group ids (P6), and region membership under the declared default projection (P7).

The rows come from `accept-graph.md`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from resolvekit.core.model.result import CandidateSummary

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import CodeRow
from entity_resolution.resolver import _near_matches
from graph_api import cursor
from graph_api.doors.node import NODE_MEMBERSHIP_SLOT
from tests.graph_release.support import serving_index

from .conftest import FakeGraphNodeReader, make_client, make_resolver

_CRS_PURPOSES = "CRSPurposeCodeEnum"
_CRS_FINANCE = "CRSTypeOfFinanceEnum"


def _members(client: TestClient, dcid: str, **params: str | int) -> Any:
    return client.get(f"/api/graph/node/{dcid}/members", params=params)


def _page(client: TestClient, dcid: str, **params: str | int) -> dict[str, Any]:
    response = _members(client, dcid, **params)
    assert response.status_code == 200, response.text
    return response.json()


def _walk(client: TestClient, dcid: str, **params: str | int) -> list[dict[str, Any]]:
    """Every page of a list, in order."""
    pages = [_page(client, dcid, **params)]
    while pages[-1]["next_cursor"] is not None:
        pages.append(_page(client, dcid, **{**params, "cursor": pages[-1]["next_cursor"]}))
    return pages


def _refs(pages: list[dict[str, Any]]) -> list[str]:
    return [item["ref"] for page in pages for item in page["items"]]


def _assert_refused_at_the_cursor(response: Any) -> None:
    assert response.status_code == 400, response.text
    problem = response.json()
    assert problem["code"] == "request_invalid"
    assert problem["pointer"] == "/cursor"


@contextmanager
def _serving_code_index(index: CodeIndex) -> Iterator[None]:
    with serving_index(index):
        yield


# --- page bounds ---------------------------------------------------------------------------


def test_a_picker_gets_the_largest_code_system_in_one_page():
    with make_client() as client:
        page = _page(client, _CRS_PURPOSES, relation="codes", limit=1000)

    assert len(page["items"]) == 289
    assert page["next_cursor"] is None
    assert page["complete"] is True
    assert "code_index" in page["built_from"]


@pytest.mark.parametrize(("limit", "status"), [(200, 200), (1000, 200), (1001, 400), (0, 400)])
def test_a_page_holds_up_to_a_thousand_items(limit, status):
    with make_client() as client:
        response = _members(client, _CRS_PURPOSES, relation="codes", limit=limit)

    assert response.status_code == status
    if status == 400:
        assert response.json()["pointer"] == "/limit"


def test_a_default_page_holds_two_hundred_items():
    with make_client() as client:
        page = _page(client, _CRS_PURPOSES, relation="codes")

    assert len(page["items"]) == 200
    assert page["next_cursor"] is not None
    assert page["complete"] is False


# --- cursor binding ------------------------------------------------------------------------


def test_a_cursor_continues_its_own_walk_without_repeating_an_item():
    with make_client() as client:
        first = _page(client, _CRS_PURPOSES, relation="codes", limit=5)
        second = _page(
            client, _CRS_PURPOSES, relation="codes", limit=5, cursor=first["next_cursor"]
        )

    assert not {i["ref"] for i in first["items"]} & {i["ref"] for i in second["items"]}


def test_a_cursor_sent_for_another_node_is_refused():
    with make_client() as client:
        first = _page(client, _CRS_PURPOSES, relation="codes", limit=5)
        response = _members(
            client, _CRS_FINANCE, relation="codes", limit=5, cursor=first["next_cursor"]
        )

    _assert_refused_at_the_cursor(response)


def test_a_cursor_sent_with_another_as_of_date_is_refused():
    with make_client() as client:
        first = _page(client, "EuropeanUnion", relation="members", limit=5, as_of="2015-06-01")
        response = _members(
            client,
            "EuropeanUnion",
            relation="members",
            limit=5,
            as_of="2021-06-01",
            cursor=first["next_cursor"],
        )

    _assert_refused_at_the_cursor(response)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"relation": "children"}, id="C8-relation"),
        pytest.param({"order": "dcid"}, id="C10-order"),
    ],
)
def test_a_cursor_bound_to_another_relation_or_order_of_the_node_is_refused(change):
    """No node serves two relations and the wire names no order, so the cursor is one the route
    would issue for the same node with only the relation or the order changed."""
    own = cursor.Binding(
        node=_CRS_PURPOSES, relation="codes", projection=None, as_of=None, order="code"
    )
    with make_client() as client:
        first = _page(client, _CRS_PURPOSES, relation="codes", limit=5)
        # The route's own binding decodes its cursor, so `own` differs only by `change`.
        position = cursor.decode(first["next_cursor"], own)
        response = _members(
            client,
            _CRS_PURPOSES,
            relation="codes",
            limit=5,
            cursor=cursor.encode(replace(own, **change), position),
        )

    _assert_refused_at_the_cursor(response)


def test_a_tampered_cursor_is_refused():
    with make_client() as client:
        response = _members(client, _CRS_PURPOSES, relation="codes", cursor="not-a-cursor")

    _assert_refused_at_the_cursor(response)


# --- a walk across two builds --------------------------------------------------------------


def _rebuilt(index: CodeIndex, *, drop: str, add: CodeRow) -> CodeIndex:
    """`index` without the code `drop` and with `add`, under a new content hash."""
    return CodeIndex(
        stamp=replace(index.stamp, content_hash="0" * 64),
        declarations=index.declarations.values(),
        codes=[*(row for row in index.codes if row.dcid != drop), add],
        statvars=index.statvars,
        datasets=index.datasets.values(),
        party_types=index.party_types,
        parties=index.parties.values(),
        listed_statvars=index.listed_statvars,
    )


def test_a_walk_across_two_builds_repeats_no_item_and_skips_none_held_by_both():
    """Two instances serve different code index builds, as during a rollout. Page one comes
    from the old build; the rest from the new one, which drops a leaf code the first page
    returned, so an offset cursor would skip an item, and adds a code past the cursor."""
    old = get_index()
    with _serving_code_index(old), make_client() as client:
        first = _page(client, _CRS_PURPOSES, relation="codes", limit=5)
    parents = {row.parent for row in old.codes}
    dropped = next(item["ref"] for item in first["items"][:-1] if item["ref"] not in parents)
    added = CodeRow(
        dcid="CRSPurposeCode99999",
        code_system="crsPurpose",
        code="99999",
        label="a code the new build adds",
        synonyms=(),
        parent=None,
    )
    new = _rebuilt(old, drop=dropped, add=added)

    with _serving_code_index(new), make_client() as client:
        rest = _walk(client, _CRS_PURPOSES, relation="codes", limit=5, cursor=first["next_cursor"])

    walked = _refs([first, *rest])
    in_both = {row.dcid for row in old.codes if row.code_system == "crsPurpose"} - {dropped}
    assert len(walked) == len(set(walked))
    assert in_both <= set(walked)
    assert added.dcid in walked
    assert rest[0]["built_from"]["code_index"] != first["built_from"]["code_index"]
    assert rest[-1]["complete"] is True


# --- groups (P6) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dcid", "count"),
    [
        ("groups/G20", 19),
        ("GroupOfSeven", 7),
        ("undata-geo/G00406000", 38),
        ("EuropeanUnion", 27),
        ("groups/EU27_2020", 27),
        ("groups/UN.SIDS", 37),
        ("groups/Commonwealth", 56),
    ],
)
def test_a_group_lists_its_members_under_the_design_id(dcid, count):
    with make_client() as client:
        pages = _walk(client, dcid, relation="members")

    refs = _refs(pages)
    assert len(refs) == count
    assert all(ref.startswith("country/") for ref in refs)
    assert pages[-1]["complete"] is True


@pytest.mark.parametrize(
    ("dcid", "as_of", "count"),
    [
        ("EuropeanUnion", "2015-06-01", 28),
        ("EuropeanUnion", "2021-06-01", 27),
    ],
)
def test_a_dated_group_lists_its_members_at_a_date(dcid, as_of, count):
    with make_client() as client:
        refs = _refs(_walk(client, dcid, relation="members", as_of=as_of))

    assert len(refs) == count
    assert ("country/GBR" in refs) is (as_of < "2020-02-01")


def test_an_undated_group_refuses_as_of():
    with make_client() as client:
        response = _members(client, "groups/AU", relation="members", as_of="2015-06-01")

    assert response.status_code == 400
    assert response.json()["pointer"] == "/as_of"


# --- group projections (P7) ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("projection", "count", "pse"),
    [("countries", 133, False), ("other_states", 1, True)],
)
def test_g77_splits_palestine_into_other_states(projection, count, pse):
    """Dev's 134 G77 members include Palestine, which the policy holds outside UN membership."""
    with make_client() as client:
        refs = _refs(_walk(client, "groups/G77", relation="members", projection=projection))

    assert len(refs) == count
    assert ("country/PSE" in refs) is pse


# --- regions (P7) --------------------------------------------------------------------------

# Graph containment as dev recorded it (live-probes.md section A; measure-regions.md): it files
# territories and dissolved states as countries, Congo under Western Africa and Turkey, Cyprus
# and Armenia under Europe. Membership comes from resolvekit's M49 containment, so every test
# below serves this graph to show the containment is ignored.
_CONTAINED = {
    "africa": ("country/AGO", "country/KEN", "country/NGA", "country/ATF", "country/ESH"),
    "europe": ("country/DEU", "country/FRA", "country/TUR", "country/CYP", "country/ARM"),
    "WesternAfrica": ("country/BEN", "country/NGA", "country/COG"),
    "MiddleAfrica": ("country/CMR",),
    "northamerica": ("country/USA", "country/CAN", "country/PRI"),
}

_REGION_TYPES = {"africa": "Continent", "europe": "Continent", "northamerica": "Continent"}


def _region_graph() -> FakeGraphNodeReader:
    places = {place for members in _CONTAINED.values() for place in members}
    return FakeGraphNodeReader(
        nodes={
            **{
                region: {"typeOf": (_REGION_TYPES.get(region, "UNGeoRegion"),), "name": (region,)}
                for region in _CONTAINED
            },
            **{dcid: {"typeOf": ("Country",), "name": (dcid,)} for dcid in places},
        },
        contained={(region, "Country"): list(members) for region, members in _CONTAINED.items()},
    )


@pytest.mark.parametrize(
    ("region", "projection", "inside", "outside", "count"),
    [
        pytest.param(
            "africa",
            "countries",
            ("country/AGO", "country/KEN", "country/NGA"),
            ("country/ESH", "country/ATF", "country/MYT", "country/SHN"),
            54,
            id="africa-countries",
        ),
        pytest.param(
            "europe",
            "countries",
            ("country/DEU", "country/FRA", "country/NOR"),
            ("country/TUR", "country/CYP", "country/ARM", "country/GIB", "country/YUG"),
            None,
            id="europe-countries",
        ),
        pytest.param(
            "northamerica",
            "countries",
            ("country/USA", "country/CAN", "country/MEX", "country/TTO"),
            ("country/PRI",),
            23,
            id="north-america-countries",
        ),
        pytest.param(
            "WesternAfrica", "countries", ("country/NGA",), ("country/COG",), None, id="west-africa"
        ),
        pytest.param("MiddleAfrica", "countries", ("country/COG",), (), None, id="middle-africa"),
        pytest.param(
            "africa",
            "territories",
            ("country/ESH", "country/MYT", "country/SHN"),
            ("country/KEN",),
            None,
            id="africa-territories",
        ),
        pytest.param(
            "europe", "other_states", ("country/VAT",), ("country/DEU",), None, id="europe-other"
        ),
    ],
)
def test_a_region_lists_its_members_under_a_projection(region, projection, inside, outside, count):
    with make_client(graph=_region_graph()) as client:
        refs = set(_refs(_walk(client, region, relation="countries", projection=projection)))

    assert set(inside) <= refs
    assert not set(outside) & refs
    if count is not None:
        assert len(refs) == count


@pytest.mark.parametrize(
    ("region", "projection", "incomplete"),
    [
        ("africa", "territories", True),
        ("europe", "former_states", True),
        ("africa", "countries", False),
        ("europe", "other_states", True),
    ],
)
def test_a_regions_territories_and_former_states_name_their_gap(region, projection, incomplete):
    """Resolvekit holds 18 territories as no country, places almost no former state, and
    places Kosovo in no region, so those three region projections say their list is
    incomplete; `countries` does not."""
    with make_client(graph=_region_graph()) as client:
        page = _page(client, region, relation="countries", projection=projection)

    assert ("membership_incomplete" in page["gaps"]) is incomplete


def test_a_cursor_sent_with_another_projection_is_refused():
    with make_client(graph=_region_graph()) as client:
        first = _page(client, "africa", relation="countries", projection="countries", limit=5)
        response = _members(
            client,
            "africa",
            relation="countries",
            projection="territories",
            limit=5,
            cursor=first["next_cursor"],
        )

    _assert_refused_at_the_cursor(response)


def test_a_resolved_regions_count_is_the_length_of_its_default_walk(shared_resolver):
    graph = _region_graph()
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))
    # An injected resolver leaves the policy to its caller; publish the shared one, as the
    # live lifespan does.
    NODE_MEMBERSHIP_SLOT.set(client.app, shared_resolver.policy)

    with client:
        (result,) = client.post("/api/graph/resolve", json={"texts": ["Africa"]}).json()["results"]
        refs = _refs(_walk(client, "africa", relation="countries"))

    assert result["status"] == "resolved"
    assert result["entity"]["kind"] == "continent"
    assert "relation=countries" in result["members"]
    assert result["member_count"] == 54
    assert result["member_count"] == len(refs)
    assert "country/ESH" not in refs


# The `typeOf` dev's graph gives each node of a policy region or group it holds, read from
# dev's mixer. The Americas (`undata-geo/G00134000`) and the OECD (`undata-geo/G00406000`) are
# `GeoRegion` nodes; a group missing here has no graph node.
_DEV_TYPES = {
    **dict.fromkeys(
        ("africa", "asia", "europe", "northamerica", "oceania", "southamerica"), "Continent"
    ),
    **dict.fromkeys(
        (
            "AustraliaAndNewZealand",
            "Caribbean",
            "CentralAmerica",
            "CentralAsia",
            "EasternAfrica",
            "EasternAsia",
            "EasternEurope",
            "LatinAmericaAndCaribbean",
            "Melanesia",
            "Micronesia",
            "MiddleAfrica",
            "NorthernAfrica",
            "NorthernEurope",
            "Polynesia",
            "SouthEasternAsia",
            "SouthernAfrica",
            "SouthernAsia",
            "SouthernEurope",
            "SubSaharanAfrica",
            "WesternAfrica",
            "WesternAsia",
            "WesternEurope",
        ),
        "UNGeoRegion",
    ),
    "undata-geo/G00134000": "GeoRegion",
    "EuropeanUnion": "ContinentalUnion",
    "GroupOfSeven": "Grouping",
    "undata-geo/G00406000": "GeoRegion",
}
_RELATION = {"region": "countries", "group": "members"}


@pytest.mark.parametrize("membership_kind", ["region", "group"])
def test_every_policy_region_and_group_reads_alike_on_resolve_the_node_and_its_walk(
    shared_resolver, membership_kind
):
    """For every region the real membership policy holds, and every group, over the graph
    types dev gives their nodes: resolving the label under the policy kinds gives the kind the
    node route gives, and a `member_count` equal to the length of the members walk. A hit the
    graph does not confirm, which a filter that admits none of the policy kinds names among
    its removed candidates, and a near match the gate rejected, take that kind too."""
    policy = shared_resolver.policy
    held = [m for m in policy if m.kind == membership_kind]
    missing = [m.ref for m in held if m.kind == "region" and m.ref not in _DEV_TYPES]
    assert not missing, f"no recorded graph type for regions {missing}"
    graph = FakeGraphNodeReader(
        nodes={ref: {"typeOf": (t,), "name": (ref,)} for ref, t in _DEV_TYPES.items()}
    )
    client = make_client(graph=graph, resolver=make_resolver(shared_resolver, graph=graph))
    NODE_MEMBERSHIP_SLOT.set(client.app, policy)
    unconfirmed = make_client(
        graph=FakeGraphNodeReader(),
        resolver=make_resolver(shared_resolver, graph=FakeGraphNodeReader()),
    )
    NODE_MEMBERSHIP_SLOT.set(unconfirmed.app, policy)
    kinds = ["continent", "subregion", "group"]

    mismatches = {}
    with client, unconfirmed:
        for m in held:
            body = {"texts": [m.label], "kinds": kinds}
            (result,) = client.post("/api/graph/resolve", json=body).json()["results"]
            entity = result.get("entity") or {}
            got = (result["status"], entity.get("ref"), entity.get("kind"), result["member_count"])
            node_kind = client.get(f"/api/graph/node/{m.ref}").json()["kind"]
            walk = _refs(_walk(client, m.ref, relation=_RELATION[m.kind]))
            expected = ("resolved", m.ref, node_kind, len(walk))
            if got != expected:
                mismatches[m.ref] = {"resolve": got, "node and walk": expected}

            body = {"texts": [m.label], "kinds": ["other"]}
            (result,) = unconfirmed.post("/api/graph/resolve", json=body).json()["results"]
            removed = {near["ref"]: near["kind"] for near in result.get("near_matches", [])}
            if removed.get(m.ref) != node_kind:
                mismatches[m.ref] = {"unconfirmed": removed.get(m.ref), "node": node_kind}

            hint = CandidateSummary(
                entity_id=m.entity_id, canonical_name=m.label, entity_type=m.region_class
            )
            (near,) = _near_matches([hint], [], policy, get_index())
            if (near.dcid, near.kind) != (m.ref, node_kind):
                mismatches[m.ref] = {"near match": (near.dcid, near.kind), "node": node_kind}

    assert held
    assert mismatches == {}
