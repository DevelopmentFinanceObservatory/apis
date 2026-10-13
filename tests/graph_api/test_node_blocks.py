"""The node envelope's detail blocks: code systems and codes from the code index, groups and
regions from the membership policy, roles from the declared role ranges, and unit words."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date

import pytest
from fastapi.testclient import TestClient

from entity_resolution.admission import Admission
from entity_resolution.groups import build_group_registry
from entity_resolution.membership import (
    COUNTRY_CLASS,
    Authority,
    Membership,
    MembershipPolicy,
    Projection,
    Row,
    Sovereignty,
    build_policy,
)
from entity_resolution.shared import SharedResolver
from graph_api.deps import DcpReadExecutor
from graph_api.doors import node as node_door
from graph_api.doors.node import NODE_MEMBERSHIP_SLOT
from materialise.graph import GraphReadError

from .conftest import FakeGraphNodeReader, make_client

_AUTHORITY = Authority(name="curated", version="v1")
# AAA is a UN member since 1958; TER is a territory whose membership ended in 2020.
_SOVEREIGNTY = Sovereignty(
    states={"country/AAA": (Row(member="country/AAA", member_class=COUNTRY_CLASS),)},
    dissolved=frozenset(),
    other_states=frozenset(),
)
_DATED = Membership(
    ref="groups/T",
    entity_id="groups/T",
    label="Test group",
    kind="group",
    authority=_AUTHORITY,
    dated=True,
    rows=(
        Row(member="country/AAA", member_class=COUNTRY_CLASS, valid_from=date(1958, 1, 1)),
        Row(
            member="country/TER",
            member_class=COUNTRY_CLASS,
            valid_from=date(1973, 1, 1),
            valid_until=date(2020, 2, 1),
        ),
    ),
    sovereignty=_SOVEREIGNTY,
)
_UNDATED = Membership(
    ref="groups/U",
    entity_id="groups/U",
    label="Undated group",
    kind="group",
    authority=_AUTHORITY,
    dated=False,
    rows=(Row(member="country/AAA", member_class=COUNTRY_CLASS),),
    sovereignty=_SOVEREIGNTY,
)
_REGION = Membership(
    ref="africa",
    entity_id="m49/002",
    label="Africa",
    kind="region",
    authority=_AUTHORITY,
    dated=False,
    rows=(Row(member="country/AAA", member_class=COUNTRY_CLASS),),
    sovereignty=_SOVEREIGNTY,
    incomplete=frozenset({Projection.TERRITORIES, Projection.FORMER_STATES}),
    region_class="geo.continent",
)
_POLICY = MembershipPolicy([_DATED, _UNDATED, _REGION])


@contextmanager
def _client(
    graph: FakeGraphNodeReader, policy: MembershipPolicy | None = _POLICY
) -> Iterator[TestClient]:
    """A running client serving `policy` in place of the one its lifespan published, or no
    policy at all for `None`."""
    with make_client(graph=graph) as client:
        if policy is None:
            NODE_MEMBERSHIP_SLOT.clear(client.app)
        else:
            NODE_MEMBERSHIP_SLOT.set(client.app, policy)
        yield client


def _counts(block: dict) -> dict[str, int | None]:
    return {entry["projection"]: entry["count"] for entry in block["projections"]}


class TestGroup:
    def test_a_group_opens_from_the_policy_with_no_graph_node(self):
        with _client(FakeGraphNodeReader()) as client:
            response = client.get("/api/graph/node/groups/T")

        assert response.status_code == 200, response.text
        node = response.json()
        assert node["kind"] == "group"
        assert node["label"] == "Test group"
        assert node["types"] == []
        assert node["links"] == {
            "members": "/api/graph/node/groups/T/members?relation=members&projection=countries"
        }
        group = node["group"]
        assert group["authority"] == {"name": "curated", "version": "v1"}
        assert group["effective_date"] == "2020-02-01"
        assert _counts(group) == {
            "countries": 1,
            "other_states": 0,
            "territories": 0,
            "former_states": 0,
            "institutions": None,
        }
        assert node["gaps"] == ["group.projections.institutions"]
        complete = {entry["projection"]: entry["complete"] for entry in group["projections"]}
        assert (complete["countries"], complete["institutions"]) == (True, False)
        assert node["built_from"]["membership"] == "v1"
        # The institutions gap is permanent, so the response is cacheable.
        assert response.headers["cache-control"] == "public, max-age=300"
        assert response.headers["etag"]

    @pytest.mark.parametrize(
        ("dcid", "types", "kind", "label"),
        [
            pytest.param("groups/T", ("Place",), "group", "Test group", id="group"),
            pytest.param("africa", ("Continent",), "continent", "Africa", id="region"),
        ],
    )
    def test_a_held_membership_takes_the_authoritys_name(self, dcid, types, kind, label):
        graph = FakeGraphNodeReader(nodes={dcid: {"typeOf": types, "name": ("Graph name",)}})

        with _client(graph) as client:
            node = client.get(f"/api/graph/node/{dcid}").json()

        assert (node["kind"], node["label"]) == (kind, label)

    def test_as_of_dates_the_counts_and_the_members_link(self):
        with _client(FakeGraphNodeReader()) as client:
            node = client.get("/api/graph/node/groups/T", params={"as_of": "2019-01-01"}).json()

        assert node["group"]["as_of"] == "2019-01-01"
        assert node["group"]["effective_date"] == "1973-01-01"
        assert _counts(node["group"])["territories"] == 1
        assert node["links"]["members"].endswith("&as_of=2019-01-01")

    def test_an_undated_group_serves_no_effective_date_and_names_the_gap(self):
        with _client(FakeGraphNodeReader()) as client:
            node = client.get("/api/graph/node/groups/U").json()

        assert node["group"]["effective_date"] is None
        assert "group.effective_date" in node["gaps"]

    @pytest.mark.parametrize("dcid", ["groups/U", "africa"])
    def test_as_of_on_an_undated_membership_is_refused(self, dcid):
        graph = FakeGraphNodeReader(nodes={"africa": {"typeOf": ("Continent",)}})

        with _client(graph) as client:
            response = client.get(f"/api/graph/node/{dcid}", params={"as_of": "2015-06-01"})

        assert response.status_code == 400, response.text
        assert response.json()["code"] == "request_invalid"
        assert response.json()["pointer"] == "/as_of"

    def test_a_group_id_is_not_held_while_no_policy_is_wired(self):
        with _client(FakeGraphNodeReader(), policy=None) as client:
            response = client.get("/api/graph/node/groups/T")

        assert response.status_code == 404, response.text
        assert response.json()["code"] == "node_not_held"

    def test_a_curated_group_opens_under_the_shared_policy(self, shared_resolver: SharedResolver):
        registry = shared_resolver.run(build_group_registry, shared_resolver.resolver).result()
        policy = shared_resolver.run(
            lambda: build_policy(shared_resolver.resolver, groups=registry)
        ).result()

        with _client(FakeGraphNodeReader(), policy=policy) as client:
            node = client.get("/api/graph/node/groups/G20").json()

        assert node["kind"] == "group"
        assert _counts(node["group"])["countries"] == 19


class TestRegion:
    def test_a_region_is_a_place_with_its_membership_and_a_countries_link(self):
        graph = FakeGraphNodeReader(
            nodes={"africa": {"typeOf": ("Continent",), "name": ("Africa",)}}
        )

        with _client(graph) as client:
            node = client.get("/api/graph/node/africa").json()

        assert node["kind"] == "continent"
        assert node["links"] == {
            "countries": "/api/graph/node/africa/members?relation=countries&projection=countries"
        }
        complete = {
            entry["projection"]: entry["complete"]
            for entry in node["continent"]["membership"]["projections"]
        }
        assert complete["countries"] is True
        assert complete["territories"] is False
        assert complete["former_states"] is False
        assert "continent.membership.effective_date" in node["gaps"]


class TestCodes:
    def test_a_code_system_counts_its_codes_and_names_its_property(self):
        graph = FakeGraphNodeReader(nodes={"CRSPurposeCodeEnum": {"typeOf": ("Class",)}})

        with _client(graph, policy=None) as client:
            node = client.get("/api/graph/node/CRSPurposeCodeEnum").json()

        code_system = node["code_system"]
        assert code_system["property"] == "crsPurpose"
        assert code_system["plain_name"] == "sector"
        assert code_system["declarations"]["hierarchical"] is True
        assert code_system["code_count"] == 289
        assert node["links"] == {
            "codes": "/api/graph/node/CRSPurposeCodeEnum/members?relation=codes"
        }

    def test_a_leaf_code_names_its_system_node_and_links_no_children(self):
        graph = FakeGraphNodeReader(
            nodes={
                "CRSPurposeCode12220": {
                    "typeOf": ("CRSPurposeCodeEnum",),
                    "name": ("Basic health care",),
                },
                "CRSPurposeCodeEnum": {"name": ("CRS purpose code",)},
            }
        )

        with _client(graph, policy=None) as client:
            node = client.get("/api/graph/node/CRSPurposeCode12220").json()

        code = node["code"]
        assert code["code_system"] == {"ref": "CRSPurposeCodeEnum", "label": "CRS purpose code"}
        assert code["property"] == "crsPurpose"
        assert code["parent"] == {"ref": "CRSPurposeCode122", "label": "basic health"}
        assert "primary health care" in code["synonyms"]
        assert node["links"] == {}
        assert node["built_from"]["code_index"]


class _RangeGraph(FakeGraphNodeReader):
    """Declares `financingProvider`'s range as organisations and countries, and
    `financingRecipient`'s as countries. `fail_ranges` makes the range read fail."""

    def __init__(self, *, nodes, fail_ranges: bool = False) -> None:
        super().__init__(
            nodes={
                **nodes,
                "financingProvider": {"rangeIncludes": ("Organization", "Country")},
                "financingRecipient": {"rangeIncludes": ("Country",)},
            }
        )
        self._fail_ranges = fail_ranges

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if self._fail_ranges and tuple(properties) == ("rangeIncludes",):
            raise GraphReadError("range read failed")
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


class TestParties:
    def test_an_organisation_lists_the_roles_its_types_can_fill(self):
        graph = _RangeGraph(nodes={"dc/o/Fund": {"typeOf": ("Organization",), "name": ("A fund",)}})

        with _client(graph, policy=None) as client:
            node = client.get("/api/graph/node/dc/o/Fund").json()

        assert node["kind"] == "organisation"
        assert node["organisation"] == {"roles": ["financingProvider"]}

    def test_an_aggregate_lists_its_roles(self):
        graph = _RangeGraph(nodes={"DACAfrica": {"typeOf": ("DACGroup",)}})

        with _client(graph, policy=None) as client:
            node = client.get("/api/graph/node/DACAfrica").json()

        assert node["kind"] == "aggregate"
        assert node["aggregate"] == {"roles": []}

    def test_a_failed_range_read_serves_null_roles_uncached(self):
        graph = _RangeGraph(
            nodes={"dc/o/Fund": {"typeOf": ("Organization",), "name": ("A fund",)}},
            fail_ranges=True,
        )

        with _client(graph, policy=None) as client:
            response = client.get("/api/graph/node/dc/o/Fund")

        assert response.status_code == 200, response.text
        assert response.json()["organisation"] == {"roles": None}
        assert response.json()["gaps"] == ["organisation.roles"]
        assert response.headers["cache-control"] == "no-store"
        assert "etag" not in response.headers

    @pytest.mark.parametrize("refusal", ["saturated", "shut_down"])
    def test_a_range_read_the_dcp_executor_refuses_serves_null_roles_uncached(
        self, monkeypatch, refusal: str
    ):
        """The role reads go through a one-worker DCP-read executor that is either busy with
        another read or shut down, so the read is refused before it starts."""
        graph = _RangeGraph(nodes={"dc/o/Fund": {"typeOf": ("Organization",), "name": ("A fund",)}})
        executor = DcpReadExecutor(max_workers=1, thread_name_prefix="test-dcp")
        release = threading.Event()
        if refusal == "saturated":
            executor.submit(release.wait, 5)
        else:
            executor.shutdown()
        monkeypatch.setattr(
            node_door,
            "_admission",
            lambda _request, _deps: Admission(graph, graph_executor=executor),
        )
        try:
            with _client(graph, policy=None) as client:
                response = client.get("/api/graph/node/dc/o/Fund")
        finally:
            release.set()
            executor.shutdown(wait=True)

        assert response.status_code == 200, response.text
        assert response.json()["organisation"] == {"roles": None}
        assert response.json()["gaps"] == ["organisation.roles"]
        assert response.headers["cache-control"] == "no-store"


class TestUnits:
    @pytest.mark.parametrize(
        ("dcid", "arcs", "expected"),
        [
            (
                "CurrencyUSD",
                {"typeOf": ("CurrencyUnit",), "iso4217AlphaCode": ("USD",), "name": ("USD",)},
                {
                    "unit_class": "currency",
                    "currency": "USD",
                    "currency_noun": "US dollars",
                    "price_basis": None,
                    "changes_magnitude": False,
                },
            ),
            (
                "KES",
                {"typeOf": ("Currency",), "name": ("Kenyan shilling",)},
                {
                    "unit_class": "currency",
                    "currency": "KES",
                    "currency_noun": "Kenyan shillings",
                    "price_basis": None,
                    "changes_magnitude": True,
                },
            ),
            (
                "ChainedUSDollarBaseYr2017",
                {"typeOf": ("UnitOfMeasure",)},
                {
                    "unit_class": "currency",
                    "currency": "USD",
                    "currency_noun": "US dollars",
                    "price_basis": {"prices": "constant", "base_year": 2017},
                    "changes_magnitude": False,
                },
            ),
            (
                "Percent",
                {"typeOf": ("UnitOfMeasure",)},
                {
                    "unit_class": "percent",
                    "currency": None,
                    "currency_noun": None,
                    "price_basis": None,
                    "changes_magnitude": True,
                },
            ),
        ],
    )
    def test_a_unit_carries_the_words_it_is_named_by(self, dcid, arcs, expected):
        with _client(FakeGraphNodeReader(nodes={dcid: arcs}), policy=None) as client:
            node = client.get(f"/api/graph/node/{dcid}").json()

        assert node["kind"] == "unit"
        assert node["unit"] == expected


def test_a_node_with_no_name_names_the_label_gap():
    graph = FakeGraphNodeReader(nodes={"dc/base/Foo": {"typeOf": ("SomeOtherType",)}})

    with _client(graph, policy=None) as client:
        node = client.get("/api/graph/node/dc/base/Foo").json()

    assert node["label"] is None
    assert node["gaps"] == ["label"]
