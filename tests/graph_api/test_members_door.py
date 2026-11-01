"""The members and roots doors through a minimal surface that mounts them ahead of the node door,
as the Graph surface must: the relations' owners and orders, the partial-list gaps, the
refusals and their pointers, and the cursor's binding to every field of its walk."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from code_index.shared import get_index
from dfo_platform.app import Surface, create_app
from dfo_platform.doors import combine_classifiers, flatten
from entity_resolution.membership import (
    COUNTRY_CLASS,
    Authority,
    Membership,
    MembershipPolicy,
    Projection,
    Row,
    Sovereignty,
)
from entity_resolution.shared import SharedResolver
from graph_api import cursor
from graph_api.codes import GRAPH_CODE_CATALOGUE
from graph_api.deps import GRAPH_DEPS_SLOT, DcpReadExecutor, GraphDeps
from graph_api.doors.members import build_members_door
from graph_api.doors.node import NODE_MEMBERSHIP_SLOT, build_node_door
from graph_api.doors.sources import new_provenance_doc_cache
from graph_api.version import GRAPH_CAPABILITY_VERSION
from graph_generation import UNKNOWN_GENERATION
from materialise.graph import GraphReadError
from origin_catalogue import DFO, Catalogue, DatasetRef, ProvenanceRow, SourceRef, bound_catalogue
from origin_catalogue.service import get_catalogue_service
from series_coverage.shared import get_coverage_service
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from .conftest import FakeGraphNodeReader, make_client, make_resolver

_AUTHORITY = Authority(name="curated", version="v1")
_SOVEREIGNTY = Sovereignty(
    states={
        dcid: (Row(member=dcid, member_class=COUNTRY_CLASS),)
        for dcid in ("country/AAA", "country/BBB", "country/CCC")
    },
    dissolved=frozenset(),
    other_states=frozenset(),
)
_GROUP = Membership(
    ref="groups/T",
    entity_id="groups/T",
    label="Test group",
    kind="group",
    authority=_AUTHORITY,
    dated=False,
    rows=tuple(
        Row(member=dcid, member_class=COUNTRY_CLASS)
        for dcid in ("country/CCC", "country/AAA", "country/BBB")
    ),
    sovereignty=_SOVEREIGNTY,
    member_labels={"country/AAA": "Aaa", "country/BBB": "Bbb"},
)
_REGION = Membership(
    ref="testregion",
    entity_id="m49/999",
    label="Test region",
    kind="region",
    authority=_AUTHORITY,
    dated=False,
    rows=(
        Row(member="country/AAA", member_class=COUNTRY_CLASS),
        Row(member="country/TER", member_class=COUNTRY_CLASS),
    ),
    sovereignty=_SOVEREIGNTY,
    incomplete=frozenset({Projection.TERRITORIES}),
    region_class="geo.subregion",
)
_POLICY = MembershipPolicy([_GROUP, _REGION])

_CRS = "provenance/OECDCreditorReportingSystem"
_DAC1 = "provenance/OECDDAC1"
_UNINDEXED = "provenance/NotIndexed"
_SOURCE = SourceRef(ref="dc/s/Test", label="Test source")


def _row(provenance: str, dataset: str) -> ProvenanceRow:
    return ProvenanceRow(
        id=provenance,
        data_commons=DFO,
        source=_SOURCE,
        name=provenance,
        url="",
        dataset=DatasetRef(ref=dataset, label=f"label of {dataset}"),
    )


_CATALOGUE = FakeCatalogue(
    provenances={
        row.id: row
        for row in (
            _row(_DAC1, "dc/d/Zeta"),
            _row(_CRS, "dc/d/Alpha"),
            _row(_UNINDEXED, "dc/d/Alpha"),
            _row("provenance/AlsoNotIndexed", "dc/d/Beta"),
        )
    }
)


class _NameReadFails(FakeGraphNodeReader):
    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if tuple(properties) == ("name",):
            raise GraphReadError("names", upstream_status=503)
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def _surface_app(
    *,
    shared: SharedResolver,
    policy: MembershipPolicy | None,
    graph: FakeGraphNodeReader | None = None,
) -> FastAPI:
    graph = graph if graph is not None else FakeGraphNodeReader()
    doors = (build_members_door(), build_node_door())
    catalogue = get_catalogue_service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with DcpReadExecutor(max_workers=1, thread_name_prefix="test-dcp") as dcp:
            GRAPH_DEPS_SLOT.set(
                app,
                GraphDeps(
                    graph=graph,
                    resolver=make_resolver(shared, graph=graph),
                    dcp_executor=dcp,
                    generation=UNKNOWN_GENERATION,
                    provenance_docs=new_provenance_doc_cache(
                        executor=dcp, generation=UNKNOWN_GENERATION
                    ),
                    coverage=get_coverage_service(),
                ),
            )
            if policy is not None:
                NODE_MEMBERSHIP_SLOT.set(app, policy)
            yield

    return create_app(
        [
            Surface(
                name="graph",
                prefix="/api/graph",
                router=flatten(doors),
                capability_version=GRAPH_CAPABILITY_VERSION,
                codes=GRAPH_CODE_CATALOGUE,
                classify=combine_classifiers(*(door.classify for door in doors)),
                exception_types=tuple(t for door in doors for t in door.exception_types),
                exception_exemplars=tuple(e for door in doors for e in door.exception_exemplars),
                lifespan=lifespan,
                request_scope=lambda: bound_catalogue(catalogue.snapshot()),
            )
        ]
    )


_HELD_SOURCE = "dc/s/Held"
_HELD_DATASET = "dc/d/Held"
_HELD_PLACE = "country/AAA"
_GRAPH = FakeGraphNodeReader(
    nodes={
        _HELD_SOURCE: {"typeOf": ("Source",)},
        _HELD_DATASET: {"typeOf": ("Dataset",)},
        _HELD_PLACE: {"typeOf": ("Country",)},
    }
)


@pytest.fixture
def client(
    shared_resolver: SharedResolver, serve_catalogue: Callable[[Catalogue], None]
) -> TestClient:
    serve_catalogue(_CATALOGUE)
    return TestClient(
        _surface_app(shared=shared_resolver, policy=_POLICY, graph=_GRAPH),
        raise_server_exceptions=False,
    )


def _get(client: TestClient, dcid: str, **params: str | int) -> Any:
    return client.get(f"/api/graph/node/{dcid}/members", params=params)


def _page(client: TestClient, dcid: str, **params: str | int) -> dict[str, Any]:
    response = _get(client, dcid, **params)
    assert response.status_code == 200, response.text
    return response.json()


def _walk_refs(client: TestClient, dcid: str, **params: str | int) -> list[str]:
    page = _page(client, dcid, **params)
    refs = [item["ref"] for item in page["items"]]
    while page["next_cursor"] is not None:
        page = _page(client, dcid, **{**params, "cursor": page["next_cursor"]})
        refs += [item["ref"] for item in page["items"]]
    return refs


def _assert_refused(response: Any, pointer: str) -> None:
    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
    assert response.json()["pointer"] == pointer


class TestRouting:
    def test_the_members_route_wins_over_the_node_route_for_a_slash_bearing_dcid(self, client):
        with client:
            listed = _get(client, "groups/T", relation="members")
            node = client.get("/api/graph/node/groups/T")

        assert listed.json()["ref"] == "groups/T"
        assert node.json()["kind"] == "group"
        assert node.json()["links"]["members"].startswith("/api/graph/node/groups/T/members?")


class TestCodes:
    def test_a_codes_walk_holds_every_code_once_in_code_order(self, client):
        index = get_index()
        with client:
            refs = _walk_refs(client, "DAC2bAidTypeEnum", relation="codes", limit=5)

        rows = sorted(
            (row for row in index.codes if row.code_system == "dac2bAidType"),
            key=lambda row: row.code,
        )
        assert refs == [row.dcid for row in rows]

    def test_a_codes_item_carries_its_code_and_parent(self, client):
        child = next(
            row for row in get_index().codes if row.code_system == "crsPurpose" and row.parent
        )
        assert child.parent is not None
        with client:
            items = _page(client, child.parent, relation="children", limit=1000)["items"]

        assert next(item for item in items if item["ref"] == child.dcid) == {
            "ref": child.dcid,
            "label": child.label,
            "code": child.code,
            "parent": child.parent,
        }

    def test_a_leaf_lists_no_children_and_is_complete(self, client):
        index = get_index()
        leaf = next(row for row in index.codes if not index.children(row.dcid))
        with client:
            page = _page(client, leaf.dcid, relation="children")

        assert (page["items"], page["complete"], page["total"]) == ([], True, 0)
        assert set(page["built_from"]) == {"code_index"}


class TestCatalogueLists:
    def test_a_sources_datasets_are_ordered_by_dcid(self, client):
        with client:
            refs = _walk_refs(client, _SOURCE.ref, relation="datasets", limit=1)

        assert refs == ["dc/d/Alpha", "dc/d/Beta", "dc/d/Zeta"]

    def test_a_provenance_lists_the_variables_the_code_index_associates(self, client):
        with client:
            page = _page(client, _DAC1, relation="variables", limit=1000)

        assert [item["ref"] for item in page["items"]] == list(
            get_index().variables_by_provenance[_DAC1]
        )
        assert (page["complete"], page["gaps"]) == (True, [])

    @pytest.mark.parametrize(
        "unnamed_arcs",
        [
            pytest.param(lambda dcid: {"name": (dcid,)}, id="self-named"),
            pytest.param(
                lambda dcid: {"name": (dcid,), "alternateName": ("Alternate",)},
                id="self-named-with-alternate-name",
            ),
            pytest.param(lambda dcid: {"alternateName": ("Alternate",)}, id="alternate-name-only"),
        ],
    )
    def test_a_variables_page_takes_its_labels_from_one_graph_name_read(
        self, shared_resolver, serve_catalogue, unnamed_arcs
    ):
        """The first variable has a name. The second has no name but its own dcid, so it
        takes no label, and an `alternateName` never becomes one, as on the node door."""
        named, unnamed, *_ = get_index().variables_by_provenance[_DAC1]
        graph = FakeGraphNodeReader(
            nodes={named: {"name": ("Named variable",)}, unnamed: unnamed_arcs(unnamed)}
        )
        serve_catalogue(_CATALOGUE)
        app = _surface_app(shared=shared_resolver, policy=_POLICY, graph=graph)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = _get(client, _DAC1, relation="variables", limit=3)

        page = response.json()
        assert [item["label"] for item in page["items"]][:2] == ["Named variable", None]
        assert page["gaps"] == []
        assert response.headers["cache-control"].startswith("public")
        assert graph.calls["node_arcs_many"] == 1

    def test_a_failed_label_read_leaves_labels_null_and_is_not_stored(
        self, shared_resolver, serve_catalogue
    ):
        serve_catalogue(_CATALOGUE)
        app = _surface_app(shared=shared_resolver, policy=_POLICY, graph=_NameReadFails())
        with TestClient(app, raise_server_exceptions=False) as client:
            response = _get(client, _DAC1, relation="variables", limit=3)

        assert response.status_code == 200, response.text
        page = response.json()
        assert [item["label"] for item in page["items"]] == [None, None, None]
        assert page["gaps"] == ["labels_not_read"]
        assert response.headers["cache-control"] == "no-store"

    def test_a_label_read_the_full_pool_refuses_leaves_labels_null_and_is_not_stored(self, client):
        held = threading.Event()
        with client:
            dcp = client.app.state.graph_deps.dcp_executor
            dcp.submit(held.wait, 5.0)
            try:
                response = _get(client, _DAC1, relation="variables", limit=3)
            finally:
                held.set()

        assert response.status_code == 200, response.text
        page = response.json()
        assert [item["label"] for item in page["items"]] == [None, None, None]
        assert page["gaps"] == ["labels_not_read"]
        assert response.headers["cache-control"] == "no-store"

    @pytest.mark.parametrize(
        ("dataset", "provenance"),
        [("dc/d/Zeta", _DAC1), ("dc/d/Alpha", _CRS), ("dc/d/Beta", None)],
    )
    def test_a_dataset_lists_its_indexed_provenances_variables(self, client, dataset, provenance):
        """Alpha holds CRS and a provenance the code index does not associate, so it lists CRS's
        variables as a partial list; Beta holds only an unindexed one, so it lists none."""
        with client:
            page = _page(client, dataset, relation="variables", limit=1)

        expected = get_index().variables_by_provenance[provenance][:1] if provenance else ()
        assert [item["ref"] for item in page["items"]] == list(expected)
        whole = dataset == "dc/d/Zeta"
        assert page["complete"] is False
        assert page["gaps"] == ([] if whole else ["variables_not_indexed"])
        assert (page["total"] is None) is not whole


class TestMembership:
    def test_a_groups_members_are_ordered_by_dcid(self, client):
        with client:
            page = _page(client, "groups/T", relation="members")

        assert page["items"] == [
            {"ref": "country/AAA", "label": "Aaa"},
            {"ref": "country/BBB", "label": "Bbb"},
            {"ref": "country/CCC", "label": None},
        ]
        assert (page["projection"], page["complete"], page["total"]) == ("countries", True, 3)
        assert page["built_from"] == {"membership": "v1"}

    @pytest.mark.parametrize(
        ("projection", "refs", "gaps"),
        [
            ("countries", ["country/AAA"], []),
            ("territories", ["country/TER"], ["membership_incomplete"]),
            ("institutions", [], ["projection_not_recorded"]),
        ],
    )
    def test_a_regions_projection_names_its_gap(self, client, projection, refs, gaps):
        with client:
            page = _page(client, "testregion", relation="countries", projection=projection)

        assert [item["ref"] for item in page["items"]] == refs
        assert page["gaps"] == gaps
        assert page["complete"] is not gaps

    def test_an_unpublished_policy_refuses_retryably(self, shared_resolver):
        app = _surface_app(shared=shared_resolver, policy=None)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = _get(client, "groups/T", relation="members")

        assert response.status_code == 503
        assert response.json()["code"] == "service_unavailable"


class TestRefusals:
    @pytest.mark.parametrize(
        ("dcid", "params", "pointer"),
        [
            pytest.param("groups/T", {"relation": "codes"}, "/relation", id="codes-of-a-group"),
            pytest.param(
                "testregion", {"relation": "members"}, "/relation", id="members-of-region"
            ),
            pytest.param(
                "groups/T", {"relation": "countries"}, "/relation", id="countries-of-group"
            ),
            pytest.param(
                _HELD_SOURCE, {"relation": "datasets"}, "/relation", id="source-without-datasets"
            ),
            pytest.param(
                _HELD_DATASET, {"relation": "variables"}, "/relation", id="unlisted-dataset"
            ),
            pytest.param(
                _HELD_PLACE, {"relation": "children"}, "/relation", id="children-of-place"
            ),
            pytest.param(
                "DAC2bAidTypeEnum",
                {"relation": "codes", "projection": "countries"},
                "/projection",
                id="projection-on-codes",
            ),
            pytest.param(
                "DAC2bAidTypeEnum",
                {"relation": "codes", "as_of": "2020-01-01"},
                "/as_of",
                id="as-of-on-codes",
            ),
            pytest.param(
                "groups/T",
                {"relation": "members", "as_of": "2020-01-01"},
                "/as_of",
                id="as-of-undated",
            ),
            pytest.param("groups/T", {"relation": "members", "limit": 0}, "/limit", id="limit-0"),
        ],
    )
    def test_a_request_the_list_cannot_take_is_refused_at_its_member(
        self, client, dcid, params, pointer
    ):
        with client:
            _assert_refused(_get(client, dcid, **params), pointer)


class TestUnheldNode:
    @pytest.mark.parametrize(
        "relation", ["codes", "children", "datasets", "variables", "members", "countries"]
    )
    def test_a_dcid_the_deployment_does_not_hold_answers_404_for_every_relation(
        self, client, relation
    ):
        with client:
            response = _get(client, "dc/Nothing", relation=relation)

        assert response.status_code == 404, response.text
        assert response.json()["code"] == "node_not_held"
        assert response.json()["pointer"] == "/dcid"

    @pytest.mark.parametrize(
        ("dcid", "relation"),
        [
            pytest.param("groups/T", "countries", id="group-with-no-graph-node"),
            pytest.param("testregion", "members", id="region-with-no-graph-node"),
        ],
    )
    def test_a_group_or_region_the_policy_holds_never_answers_404(self, client, dcid, relation):
        with client:
            _assert_refused(_get(client, dcid, relation=relation), "/relation")

    def test_a_policy_held_group_with_no_graph_node_lists_its_members(self, client):
        with client:
            assert _page(client, "groups/T", relation="members")["total"] == 3

    def test_a_served_list_reads_no_node(self, shared_resolver, serve_catalogue):
        serve_catalogue(_CATALOGUE)
        graph = FakeGraphNodeReader()
        with TestClient(
            _surface_app(shared=shared_resolver, policy=_POLICY, graph=graph),
            raise_server_exceptions=False,
        ) as client:
            _page(client, "DAC2bAidTypeEnum", relation="codes")
            _page(client, "groups/T", relation="members")

        assert graph.calls["node_arcs_many"] == 0


_BINDING = cursor.Binding(node="n", relation="codes", projection=None, as_of=None, order="code")


@pytest.mark.parametrize(
    "changed",
    [
        {"node": "m"},
        {"relation": "children"},
        {"projection": "countries"},
        {"as_of": date(2020, 1, 1)},
        {"order": "dcid"},
        {"filters": (("measurement_method", None),)},
    ],
    ids=lambda changed: next(iter(changed)),
)
def test_a_cursor_decodes_only_under_its_own_binding(changed):
    """Each field of the binding is checked, the relation and order among them, which no
    request can change alone."""
    token = cursor.encode(_BINDING, cursor.Position(key="110", dcid="x"))

    assert cursor.decode(token, _BINDING) == cursor.Position(key="110", dcid="x")
    with pytest.raises(cursor.CursorInvalid):
        cursor.decode(token, replace(_BINDING, **changed))


@pytest.mark.parametrize("token", ["", "not-a-cursor", "WzFd", "e30"])
def test_a_token_that_is_no_cursor_is_refused(token):
    with pytest.raises(cursor.CursorInvalid):
        cursor.decode(token, _BINDING)


def test_the_roots_name_each_list_with_its_count(client):
    index = get_index()
    with client:
        response = client.get("/api/graph/roots")

    assert response.status_code == 200, response.text
    roots = response.json()
    assert {entry["ref"]: entry["count"] for entry in roots["code_systems"]} == {
        d.enum_dcid: len(index.codes_by_system[d.code_system])
        for d in index.declarations.values()
        if d.enum_dcid is not None
    }
    assert roots["groups"] == [{"ref": "groups/T", "label": "Test group", "count": 3}]
    assert roots["regions"] == [{"ref": "testregion", "label": "Test region", "count": 1}]
    assert {"ref": _SOURCE.ref, "label": _SOURCE.label, "count": 3} in roots["sources"]
    assert set(roots["built_from"]) == {"code_index", "origin_catalogue", "membership"}


class _HoldReadFails(FakeGraphNodeReader):
    def node_arcs_many(self, dcids, *, properties, deadline=None):
        raise GraphReadError("arcs", upstream_status=404)


def test_a_failed_hold_read_is_classified_and_logged_by_the_members_door(caplog):
    """Every Graph door recognises `GraphReadError`. On the composed surface the members
    route's own classifier answers, so the line logged is the members door's."""
    client = make_client(graph=_HoldReadFails())

    with client, caplog.at_level(logging.ERROR):
        response = _get(client, "dc/Nothing", relation="codes")

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "upstream_error"
    assert [
        record.getMessage().split(" door's")[0]
        for record in caplog.records
        if record.getMessage().startswith("graph: Graph ")
    ] == ["graph: Graph members"]
