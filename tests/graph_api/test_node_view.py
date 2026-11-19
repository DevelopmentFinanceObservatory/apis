"""`build_node_view` against a fake graph of base, UN and value-list nodes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest

from dc_graph import ArcTarget, GraphClient, InArcs, InArcSubject
from graph_api import node_view
from graph_api.node_view import NodeNotHeldError, NodeViewReader, build_node_view
from graph_api.node_view_models import SchemaNodeView
from graph_api.schema_models import SchemaInstances, SchemaNode
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

_BASE = "dc/base/Schema"
_UN = "undata/eclac"
_DEADLINE = 1_000.0

# (subject, property, object) for every node arc of the fake graph.
_ARCS: tuple[tuple[str, str, str], ...] = (
    ("Thing", "typeOf", "Class"),
    ("Intangible", "typeOf", "Class"),
    ("Intangible", "subClassOf", "Thing"),
    ("Enumeration", "typeOf", "Class"),
    ("Enumeration", "subClassOf", "Intangible"),
    ("Person", "typeOf", "Class"),
    ("Person", "subClassOf", "Thing"),
    ("Student", "typeOf", "Class"),
    ("Student", "subClassOf", "Person"),
    ("age", "typeOf", "Property"),
    ("age", "domainIncludes", "Person"),
    ("age", "rangeIncludes", "Number"),
    ("gender", "typeOf", "Property"),
    ("gender", "domainIncludes", "Person"),
    ("gender", "rangeIncludes", "GenderType"),
    ("GenderType", "typeOf", "Class"),
    ("GenderType", "subClassOf", "Enumeration"),
    ("Female", "typeOf", "GenderType"),
    ("Male", "typeOf", "GenderType"),
    ("undata/Series", "typeOf", "Class"),
    ("undata/Series", "subClassOf", "Enumeration"),
    ("undata/eclac/SERIES", "typeOf", "Class"),
    ("undata/eclac/SERIES", "subClassOf", "undata/Series"),
    ("undata/eclac/SI_POV", "typeOf", "undata/eclac/SERIES"),
    ("undata/eclac/SI_POV", "unConcept", "undata/eclac/POVERTY"),
    ("undataSeries", "typeOf", "Property"),
    ("undataSeries", "rangeIncludes", "undata/eclac/SERIES"),
)

_NAMES = {
    "Person": "A person",
    "Student": "A student",
    "age": "Age",
    "gender": "Gender",
    "GenderType": "Gender type",
    "Female": "Female gender",
    "Male": "Male gender",
    "Enumeration": "An enumeration",
    "undata/eclac/SERIES": "Series",
    "undata/eclac/SI_POV": "Poverty headcount",
    "undataSeries": "Series",
}

_PROVENANCES = {"undata/eclac/SERIES": _UN, "undata/eclac/SI_POV": _UN, "undataSeries": _UN}

_LITERALS = {("undata/eclac/SI_POV", "unCode"): ("SI_POV",)}


class _FakeGraph:
    """`_ARCS` as the four reads answer them, each call recorded as (method, dcids, properties)."""

    def __init__(self, arcs: Sequence[tuple[str, str, str]] = _ARCS) -> None:
        self.arcs = tuple(arcs)
        self.calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def _types(self, dcid: str) -> tuple[str, ...]:
        return tuple(sorted(o for s, p, o in self.arcs if s == dcid and p == "typeOf"))

    def _target(self, subject: str, prop: str, value: str) -> ArcTarget:
        return ArcTarget(
            value=value,
            name=_NAMES.get(value),
            types=self._types(value),
            provenances=(_PROVENANCES.get(subject, _BASE),) if prop == "typeOf" else (),
        )

    def _read(
        self, dcid: str, properties: Sequence[str] | None
    ) -> dict[str, tuple[ArcTarget, ...]] | None:
        entry: dict[str, list[ArcTarget]] = {}
        for s, p, o in self.arcs:
            if s == dcid and (properties is None or p in properties):
                entry.setdefault(p, []).append(self._target(s, p, o))
        if dcid in _NAMES and (properties is None or "name" in properties):
            entry["name"] = [ArcTarget(value=_NAMES[dcid], literal=True)]
        for (s, p), values in _LITERALS.items():
            if s == dcid and properties is None:
                entry[p] = [ArcTarget(value=value, literal=True) for value in values]
        return {p: tuple(targets) for p, targets in entry.items()} or None

    def node_all_arc_targets_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        self.calls.append(("->*", tuple(dcids), ()))
        return {dcid: self._read(dcid, None) for dcid in dcids}

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        self.calls.append(("->", tuple(dcids), tuple(properties)))
        return {dcid: self._read(dcid, properties) for dcid in dcids}

    def node_in_arcs(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        max_results: int,
        deadline: float | None = None,
    ) -> Mapping[str, InArcs]:
        self.calls.append(("<-", tuple(dcids), tuple(properties)))
        answers: dict[str, InArcs] = {}
        for dcid in dcids:
            subjects: list[InArcSubject] = []
            capped: set[str] = set()
            for prop in properties:
                found = sorted({s for s, p, o in self.arcs if p == prop and o == dcid})
                if len(found) > max_results:
                    capped.add(prop)
                subjects.extend(
                    InArcSubject(
                        dcid=s,
                        property=prop,
                        name=_NAMES.get(s),
                        types=self._types(s),
                        provenances=(),
                    )
                    for s in found[:max_results]
                )
            answers[dcid] = InArcs(subjects=tuple(subjects), capped=frozenset(capped))
        return answers

    def node_closure(
        self, dcid: str, *, property: str, deadline: float | None = None
    ) -> tuple[ArcTarget, ...]:
        self.calls.append(("+", (dcid,), (property,)))
        seen: set[str] = set()
        stack = [dcid]
        while stack:
            for s, p, o in self.arcs:
                if s == stack[-1] and p == property and o not in seen:
                    seen.add(o)
                    stack.append(o)
                    break
            else:
                stack.pop()
        return tuple(self._target(dcid, property, value) for value in sorted(seen))

    def in_arc_reads(self, prop: str) -> list[str]:
        return [
            dcid
            for method, dcids, props in self.calls
            if method == "<-" and prop in props
            for dcid in dcids
        ]


def _client_reads_the_view(client: GraphClient) -> NodeViewReader:
    """Fails type checking when `GraphClient` stops satisfying `NodeViewReader`."""
    return client


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(max_workers=4) as pool:
        yield pool


def _view(dcid: str, graph: _FakeGraph, executor: ThreadPoolExecutor) -> SchemaNodeView:
    return build_node_view(
        dcid,
        reader=graph,
        catalogue=FakeCatalogue(),
        generation="g1",
        executor=executor,
        deadline=_DEADLINE,
    )


def _nodes(view: SchemaNodeView) -> dict[str, SchemaNode]:
    return {node.ref: node for node in view.nodes}


def test_a_class_carries_its_arcs_and_no_member_read(executor: ThreadPoolExecutor) -> None:
    """Person outside `Enumeration` lists no members: `<-typeOf` is never requested."""
    graph = _FakeGraph()
    view = _view("Person", graph, executor)

    assert view.generation == "g1"
    assert view.centre.ref == "Person"
    assert view.centre.declared_by == "base"
    assert view.centre.provenances == [_BASE]
    assert view.centre.sub_class_of == ["Thing"]
    assert view.centre.instances is None
    nodes = _nodes(view)
    assert sorted(nodes) == ["Class", "Student", "Thing", "age", "gender"]
    assert nodes["Student"].sub_class_of == ["Person"]
    assert nodes["Student"].label == "A student"
    assert nodes["age"].domain_includes == ["Person"]
    assert nodes["age"].types == ["Property"]
    assert nodes["age"].declared_by is None
    assert nodes["Thing"].sub_class_of == []
    assert graph.in_arc_reads("typeOf") == []
    assert view.capped == []


def test_a_value_list_lists_its_members(executor: ThreadPoolExecutor) -> None:
    """GenderType's closure holds `Enumeration`, so its members are read and listed."""
    graph = _FakeGraph()
    view = _view("GenderType", graph, executor)

    assert view.centre.instances == SchemaInstances(refs=["Female", "Male"], complete=True)
    nodes = _nodes(view)
    assert nodes["Female"].types == ["GenderType"]
    assert nodes["Female"].label == "Female gender"
    assert nodes["gender"].range_includes == ["GenderType"]
    assert nodes["Enumeration"].label == "An enumeration"
    # Enumeration is a direct parent, so no chain edges are read.
    assert [call for call in graph.calls if call[0] == "->"] == []


def test_a_property_carries_its_domain_and_range(executor: ThreadPoolExecutor) -> None:
    graph = _FakeGraph()
    view = _view("age", graph, executor)

    assert view.centre.domain_includes == ["Person"]
    assert view.centre.range_includes == ["Number"]
    nodes = _nodes(view)
    assert nodes["Person"].label == "A person"
    assert nodes["Person"].types == ["Class"]
    assert nodes["Number"].types == []
    assert graph.in_arc_reads("typeOf") == []
    assert graph.in_arc_reads("rangeIncludes") == ["age"]


def test_a_value_reads_its_list(executor: ThreadPoolExecutor) -> None:
    """Female's type class descends from `Enumeration`, so it lists Female, and the property
    ranging on it joins."""
    graph = _FakeGraph()
    view = _view("Female", graph, executor)

    assert view.centre.types == ["GenderType"]
    assert view.centre.instances is None
    nodes = _nodes(view)
    assert nodes["GenderType"].instances == SchemaInstances(refs=["Female"], complete=False)
    assert nodes["GenderType"].sub_class_of == ["Enumeration"]
    assert nodes["gender"].range_includes == ["GenderType"]
    assert nodes["Enumeration"].instances is None
    assert graph.in_arc_reads("typeOf") == []
    assert len(graph.calls) == 6


def test_a_un_series_reads_its_list_through_the_chain(executor: ThreadPoolExecutor) -> None:
    """A UN series' type class reaches `Enumeration` through a parent, and every edge of the
    chain in the view is one the graph answered."""
    graph = _FakeGraph()
    view = _view("undata/eclac/SI_POV", graph, executor)

    assert view.centre.declared_by == "un_system"
    assert view.centre.other_properties["unCode"][0].model_dump() == {"value": "SI_POV"}
    nodes = _nodes(view)
    series = nodes["undata/eclac/SERIES"]
    assert series.instances == SchemaInstances(refs=["undata/eclac/SI_POV"], complete=False)
    assert series.sub_class_of == ["undata/Series"]
    assert nodes["undata/Series"].sub_class_of == ["Enumeration"]
    assert nodes["undata/Series"].instances == SchemaInstances(refs=[], complete=False)
    assert nodes["undataSeries"].range_includes == ["undata/eclac/SERIES"]
    assert "undata/eclac/POVERTY" not in nodes
    assert nodes["Intangible"].instances is None


def test_a_deep_value_list_reads_its_chain_and_marks_narrower_lists(
    executor: ThreadPoolExecutor,
) -> None:
    """The class undata/Series reaches `Enumeration` directly and has a narrower list, so its
    own list is incomplete and the narrower class lists instances too; the deeper
    undata/eclac/SERIES reads the edges to `Enumeration`."""
    graph = _FakeGraph()
    series = _view("undata/Series", graph, executor)
    assert series.centre.instances == SchemaInstances(refs=[], complete=False)
    assert _nodes(series)["undata/eclac/SERIES"].instances == SchemaInstances(
        refs=[], complete=False
    )

    deep = _view("undata/eclac/SERIES", graph, executor)
    assert deep.centre.instances == SchemaInstances(refs=["undata/eclac/SI_POV"], complete=True)
    assert _nodes(deep)["undata/Series"].sub_class_of == ["Enumeration"]
    assert _nodes(deep)["undata/Series"].instances == SchemaInstances(refs=[], complete=False)


def test_a_member_list_past_the_cap_is_capped(
    executor: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node_view, "MEMBER_CAP", 2)
    graph = _FakeGraph((*_ARCS, ("Other", "typeOf", "GenderType")))
    view = _view("GenderType", graph, executor)

    assert view.centre.instances == SchemaInstances(refs=["Female", "Male"], complete=False)
    assert view.capped == ["typeOf"]


def test_a_dcid_the_graph_lacks_is_refused(executor: ThreadPoolExecutor) -> None:
    with pytest.raises(NodeNotHeldError) as raised:
        _view("NoSuchNode", _FakeGraph(), executor)
    assert raised.value.dcid == "NoSuchNode"
