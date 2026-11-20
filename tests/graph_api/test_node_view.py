"""`build_node_view` against a fake graph of base, UN and value-list nodes."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest

from dc_graph import (
    ArcTarget,
    GraphCapExceededError,
    GraphClient,
    GraphReadError,
    InArcs,
    InArcSubject,
)
from graph_api import node_view, schema_document
from graph_api.node_view import NodeNotHeldError, NodeViewReader, build_node_view
from graph_api.node_view_models import CappedListing, SchemaNodeView
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
    "undata/eclac/POVERTY": "Poverty",
}

_PROVENANCES = {"undata/eclac/SERIES": _UN, "undata/eclac/SI_POV": _UN, "undataSeries": _UN}

_LITERALS = {("undata/eclac/SI_POV", "unCode"): ("SI_POV",)}


class _FakeGraph:
    """`_ARCS` as the four reads answer them, each call recorded as (method, dcids, properties)
    and each byte cap sent in `max_bytes`, the `->*` read's in `centre_max_bytes`. A dcid of `empty_in_batches` read beside others under
    `->p` answers empty unless each empty answer is read again alone, as `GraphClient` does
    with `individual_probe_budget=None`."""

    def __init__(
        self,
        arcs: Sequence[tuple[str, str, str]] = _ARCS,
        *,
        empty_in_batches: frozenset[str] = frozenset(),
        literals: Mapping[tuple[str, str], tuple[str, ...]] = _LITERALS,
    ) -> None:
        self.arcs = tuple(arcs)
        self.literals = literals
        self.empty_in_batches = empty_in_batches
        self.calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self.max_bytes: set[int | None] = set()
        self.centre_max_bytes: set[int | None] = set()
        self.in_arc_answers: list[tuple[tuple[str, ...], str, int]] = []

    def _types(self, dcid: str) -> tuple[str, ...]:
        return tuple(sorted(o for s, p, o in self.arcs if s == dcid and p == "typeOf"))

    def _target(self, subject: str, prop: str, value: str) -> ArcTarget:
        return ArcTarget(
            value=value,
            name=_NAMES.get(value),
            types=self._types(value),
            provenances=(_PROVENANCES.get(subject, _BASE),),
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
        for (s, p), values in self.literals.items():
            if s == dcid and properties is None:
                entry[p] = [ArcTarget(value=value, literal=True) for value in values]
        return {p: tuple(targets) for p, targets in entry.items()} or None

    def node_all_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        self.calls.append(("->*", tuple(dcids), ()))
        self.centre_max_bytes.add(max_bytes)
        return {dcid: self._read(dcid, None) for dcid in dcids}

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = 0,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        self.calls.append(("->", tuple(dcids), tuple(properties)))
        self.max_bytes.add(max_bytes)
        return {
            dcid: None
            if dcid in self.empty_in_batches and len(dcids) > 1 and individual_probe_budget == 0
            else self._read(dcid, properties)
            for dcid in dcids
        }

    def node_in_arcs(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        max_results: int,
        max_total: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> Mapping[str, InArcs]:
        # One walk per property in turn, as `GraphClient.node_in_arcs` reads them, each holding
        # at most `max_total` subjects over every dcid. `in_arc_answers` records how many each
        # walk answered.
        for prop in properties:
            self.calls.append(("<-", tuple(dcids), (prop,)))
            self._walked(prop, dcids)
        self.max_bytes.add(max_bytes)
        subjects: dict[str, list[InArcSubject]] = {dcid: [] for dcid in dcids}
        capped: dict[str, set[str]] = {dcid: set() for dcid in dcids}
        for prop in properties:
            room = len(self.arcs) if max_total is None else max_total
            for dcid in dcids:
                found = sorted({s for s, p, o in self.arcs if p == prop and o == dcid})
                kept = found[: min(max_results, room)]
                room -= len(kept)
                if len(found) > len(kept):
                    capped[dcid].add(prop)
                subjects[dcid].extend(
                    InArcSubject(
                        dcid=s,
                        property=prop,
                        name=_NAMES.get(s),
                        types=self._types(s),
                        provenances=(_PROVENANCES.get(s, _BASE),),
                    )
                    for s in kept
                )
            self.in_arc_answers.append(
                (
                    tuple(dcids),
                    prop,
                    sum(1 for d in dcids for x in subjects[d] if x.property == prop),
                )
            )
        return {
            dcid: InArcs(subjects=tuple(subjects[dcid]), capped=frozenset(capped[dcid]))
            for dcid in dcids
        }

    def _walked(self, prop: str, dcids: Sequence[str]) -> None:
        """Called after each in-arc walk is sent."""

    def node_closure(
        self,
        dcid: str,
        *,
        property: str,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> tuple[ArcTarget, ...]:
        self.calls.append(("+", (dcid,), (property,)))
        self.max_bytes.add(max_bytes)
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
    """Person is outside `Enumeration`, so it lists no members and `<-typeOf` is not requested."""
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
    # An in-arc subject is declared by its arc's provenance, and an out-arc target, whose arc
    # names the centre's declarer, by none.
    assert nodes["age"].declared_by == "base"
    assert nodes["age"].provenances == [_BASE]
    assert nodes["Thing"].declared_by is None
    assert nodes["Thing"].provenances == []
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
    # Number's closure holds no `Enumeration`, so it is no value list.
    assert nodes["Number"].instances is None
    assert graph.in_arc_reads("typeOf") == []
    assert graph.in_arc_reads("rangeIncludes") == ["age"]


def test_a_property_reads_its_range_class_as_a_value_list(executor: ThreadPoolExecutor) -> None:
    """undataSeries ranges on undata/eclac/SERIES, which reaches `Enumeration` through a parent,
    so the range class reads as a value list from its own closure."""
    graph = _FakeGraph()
    view = _view("undataSeries", graph, executor)

    series = _nodes(view)["undata/eclac/SERIES"]
    assert series.instances == SchemaInstances(refs=[], complete=False)
    assert ("+", ("undata/eclac/SERIES",), ("subClassOf",)) in graph.calls
    assert graph.in_arc_reads("typeOf") == []


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
    # Round one walks each of the five in-arcs apart, so a 429 sends one walk again.
    assert Counter(method for method, _, _ in graph.calls) == {"->*": 1, "<-": 6, "+": 2, "->": 1}


def test_a_un_series_reads_its_list_through_the_chain(executor: ThreadPoolExecutor) -> None:
    """A UN series' type class reaches `Enumeration` through a parent, and every edge of the
    chain in the view is one the graph answered, even where the batch read of the chain's edges
    answers a class empty."""
    graph = _FakeGraph(empty_in_batches=frozenset({"undata/Series"}))
    view = _view("undata/eclac/SI_POV", graph, executor)

    assert view.centre.declared_by == "un_system"
    assert view.centre.other_properties["unCode"][0].model_dump() == {"value": "SI_POV"}
    nodes = _nodes(view)
    series = nodes["undata/eclac/SERIES"]
    # The chain read the type class's own subClassOf edges, so their provenance declares it.
    assert series.declared_by == "un_system"
    assert nodes["undataSeries"].declared_by == "un_system"
    assert series.instances == SchemaInstances(refs=["undata/eclac/SI_POV"], complete=False)
    assert series.sub_class_of == ["undata/Series"]
    assert nodes["undata/Series"].sub_class_of == ["Enumeration"]
    assert nodes["undata/Series"].instances == SchemaInstances(refs=[], complete=False)
    assert nodes["undataSeries"].range_includes == ["undata/eclac/SERIES"]
    # An out-arc target outside the schema arcs joins with the label `->*` carries.
    assert nodes["undata/eclac/POVERTY"].label == "Poverty"
    assert nodes["Intangible"].instances is None
    assert {method for method, _, _ in graph.calls} == {"->*", "<-", "+", "->"}
    assert graph.max_bytes == {node_view.NODE_VIEW_MAX_BYTES}
    assert graph.centre_max_bytes == {node_view.CENTRE_MAX_BYTES}


def test_a_literal_property_past_its_byte_limit_is_left_out_and_named_capped(
    executor: ThreadPoolExecutor,
) -> None:
    """A literal property whose values total past `LARGE_LITERAL_BYTES` leaves the centre and
    is named in `capped`, though each value is short and they number under 100. A smaller
    literal property and a node-valued one stay."""
    shape = ("x" * 400,) * 90
    graph = _FakeGraph(literals={**_LITERALS, ("undata/eclac/SI_POV", "geoJsonCoordinates"): shape})
    view = _view("undata/eclac/SI_POV", graph, executor)

    assert sum(len(value) for value in shape) > node_view.LARGE_LITERAL_BYTES
    assert "geoJsonCoordinates" not in view.centre.other_properties
    assert view.capped == [CappedListing(ref="undata/eclac/SI_POV", property="geoJsonCoordinates")]
    assert view.centre.other_properties["unCode"][0].model_dump() == {"value": "SI_POV"}
    assert view.centre.other_properties["unConcept"][0].model_dump() == {
        "ref": "undata/eclac/POVERTY"
    }


def test_a_deep_value_list_reads_its_chain_and_marks_narrower_lists(
    executor: ThreadPoolExecutor,
) -> None:
    """The class undata/Series reaches `Enumeration` directly and has a narrower list, so its
    own list is incomplete and the narrower class lists instances too. The deeper
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


_GRADUATES = (
    ("Graduate", "typeOf", "Class"),
    ("Graduate", "subClassOf", "Student"),
    ("Doctoral", "typeOf", "Class"),
    ("Doctoral", "subClassOf", "Graduate"),
)


@pytest.mark.parametrize(
    ("centre", "extra", "cap", "capped"),
    [
        pytest.param(
            "GenderType",
            (("Other", "typeOf", "GenderType"),),
            (node_view, "MEMBER_CAP", 2),
            ("GenderType", "typeOf"),
            id="value-list-members",
        ),
        pytest.param(
            "Person",
            (("Teacher", "typeOf", "Class"), ("Teacher", "subClassOf", "Person")),
            (node_view, "IN_ARC_CAP", 1),
            ("Person", "subClassOf"),
            id="centre-schema-arc",
        ),
        pytest.param(
            "Female",
            (("sex", "typeOf", "Property"), ("sex", "rangeIncludes", "GenderType")),
            (node_view, "IN_ARC_CAP", 1),
            ("GenderType", "rangeIncludes"),
            id="value-type-class-range",
        ),
        pytest.param(
            "Person",
            _GRADUATES,
            (node_view, "DESCENDANT_CAP", 2),
            ("Person", "subClassOf+"),
            id="descendant-count",
        ),
        pytest.param(
            "Person",
            _GRADUATES,
            (node_view, "DESCENDANT_LEVELS", 2),
            ("Person", "subClassOf+"),
            id="descendant-levels",
        ),
        pytest.param(
            "undata/eclac/SI_POV",
            (("undata/eclac/SI_POV", "unConcept", "undata/eclac/HUNGER"),),
            (schema_document, "OTHER_VALUES_MAX", 1),
            ("undata/eclac/SI_POV", "unConcept"),
            id="centre-other-property",
        ),
        pytest.param(
            "gender",
            (),
            (node_view, "RANGE_CLASS_CAP", 0),
            ("gender", "rangeIncludes/subClassOf+"),
            id="property-range-classes",
        ),
        pytest.param(
            "Female",
            (),
            (node_view, "VALUE_TYPE_CAP", 0),
            ("Female", "typeOf/rangeIncludes"),
            id="value-type-classes",
        ),
    ],
)
def test_a_listing_past_its_cap_is_named_capped(
    executor: ThreadPoolExecutor,
    monkeypatch: pytest.MonkeyPatch,
    centre: str,
    extra: tuple[tuple[str, str, str], ...],
    cap: tuple[object, str, int],
    capped: tuple[str, str],
) -> None:
    """Each capped listing is named by the node listed and the property, so a value's type
    class listing reads apart from the centre's own, and a cut walk below a class names
    `subClassOf+`."""
    monkeypatch.setattr(*cap)
    view = _view(centre, _FakeGraph((*_ARCS, *extra)), executor)

    assert CappedListing(ref=capped[0], property=capped[1]) in view.capped
    if centre == "GenderType":
        assert view.centre.instances == SchemaInstances(refs=["Female", "Male"], complete=False)


@pytest.mark.parametrize(
    "cap",
    [
        pytest.param(("DESCENDANT_CAP", 3), id="count"),
        pytest.param(("DESCENDANT_LEVELS", 3), id="levels"),
    ],
)
def test_a_walk_ending_at_its_limit_with_nothing_below_names_nothing_capped(
    executor: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch, cap: tuple[str, int]
) -> None:
    """Student, Graduate and Doctoral fill the limit exactly, and Doctoral has no narrower
    class, so the view below Person is complete."""
    monkeypatch.setattr(node_view, *cap)
    view = _view("Person", _FakeGraph((*_ARCS, *_GRADUATES)), executor)

    assert _nodes(view)["Doctoral"].sub_class_of == ["Graduate"]
    assert view.capped == []


def test_a_level_below_a_class_reads_no_more_than_the_room_left(
    executor: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two narrower classes with three classes each below them, and room for two more classes.
    The level's read answers two subjects over both parents."""
    monkeypatch.setattr(node_view, "DESCENDANT_CAP", 4)
    below = tuple(
        arc
        for parent in ("Student", "Teacher")
        for child in (f"{parent}{i}" for i in range(3))
        for arc in ((child, "typeOf", "Class"), (child, "subClassOf", parent))
    )
    graph = _FakeGraph(
        (*_ARCS, ("Teacher", "typeOf", "Class"), ("Teacher", "subClassOf", "Person"), *below)
    )
    view = _view("Person", graph, executor)

    assert [n for dcids, prop, n in graph.in_arc_answers if dcids == ("Student", "Teacher")] == [2]
    assert len([ref for ref in _nodes(view) if ref.startswith(("Student", "Teacher"))]) == 4
    assert CappedListing(ref="Person", property="subClassOf+") in view.capped


class _TooLargeWalk(_FakeGraph):
    """`_FakeGraph` whose in-arc walk of `dcids` under `prop` answers past its byte cap."""

    def __init__(self, dcids: tuple[str, ...], prop: str) -> None:
        super().__init__()
        self.failing = (dcids, prop)

    def _walked(self, prop: str, dcids: Sequence[str]) -> None:
        if (tuple(dcids), prop) == self.failing:
            raise GraphCapExceededError("too large", cap_kind="bytes", cap=1, observed=2)


@pytest.mark.parametrize(
    ("centre", "walk", "capped", "unlisted"),
    [
        pytest.param(
            "Person", ("Person", "domainIncludes"), ("Person", "domainIncludes"), "age", id="in-arc"
        ),
        pytest.param(
            "GenderType", ("GenderType", "typeOf"), ("GenderType", "typeOf"), "Female", id="members"
        ),
        pytest.param(
            "Person", ("Student", "subClassOf"), ("Person", "subClassOf+"), None, id="descendants"
        ),
        pytest.param(
            "Female",
            ("GenderType", "rangeIncludes"),
            ("GenderType", "rangeIncludes"),
            "gender",
            id="value-range",
        ),
    ],
)
def test_a_listing_past_the_byte_cap_is_named_capped_and_the_view_builds(
    executor: ThreadPoolExecutor,
    centre: str,
    walk: tuple[str, str],
    capped: tuple[str, str],
    unlisted: str | None,
) -> None:
    view = _view(centre, _TooLargeWalk((walk[0],), walk[1]), executor)

    assert view.centre.ref == centre
    assert CappedListing(ref=capped[0], property=capped[1]) in view.capped
    if unlisted is not None:
        assert unlisted not in _nodes(view)
    if centre == "GenderType":
        assert view.centre.instances == SchemaInstances(refs=[], complete=False)


def test_a_class_lists_every_class_below_it_with_its_edges(executor: ThreadPoolExecutor) -> None:
    """The classes below a narrower class join with the `subClassOf` edge to the class above,
    so a walk down from the centre reaches each, and an uncut walk names nothing capped."""
    graph = _FakeGraph((*_ARCS, *_GRADUATES))
    view = _view("Person", graph, executor)

    nodes = _nodes(view)
    assert nodes["Graduate"].sub_class_of == ["Student"]
    assert nodes["Doctoral"].sub_class_of == ["Graduate"]
    assert nodes["Doctoral"].declared_by == "base"
    assert view.capped == []
    assert graph.in_arc_reads("subClassOf") == ["Person", "Student", "Graduate", "Doctoral"]


def test_a_value_list_with_an_untyped_narrower_class_is_incomplete(
    executor: ThreadPoolExecutor,
) -> None:
    """A `subClassOf` subject the graph answers with no types still narrows the list, so its
    members are not all listed, and each class below it lists instances too."""
    graph = _FakeGraph(
        (
            *_ARCS,
            ("SubGender", "subClassOf", "GenderType"),
            ("SubSubGender", "subClassOf", "SubGender"),
        )
    )
    view = _view("GenderType", graph, executor)

    assert view.centre.instances == SchemaInstances(refs=["Female", "Male"], complete=False)
    for ref in ("SubGender", "SubSubGender"):
        assert _nodes(view)[ref].instances == SchemaInstances(refs=[], complete=False)


def test_a_value_with_no_type_class_sends_no_empty_read(executor: ThreadPoolExecutor) -> None:
    graph = _FakeGraph((*_ARCS, ("Somewhere", "typeOf", "Thing")))
    _view("Somewhere", graph, executor)

    assert Counter(method for method, _, _ in graph.calls) == {"->*": 1, "<-": 5, "+": 1}


class _RateLimited(_FakeGraph):
    """`_FakeGraph` whose first `failures` `->*` reads answer HTTP 429."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def node_all_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        if self.failures:
            self.failures -= 1
            self.calls.append(("429", tuple(dcids), ()))
            raise GraphReadError("rate limited", upstream_status=429)
        return super().node_all_arc_targets_many(dcids, max_bytes=max_bytes, deadline=deadline)


@pytest.mark.parametrize("failures", [1, 2], ids=["within-budget", "past-budget"])
def test_a_rate_limited_read_is_sent_again_within_the_budget(
    executor: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    monkeypatch.setattr(node_view, "RATE_LIMIT_WAITS_SECONDS", (0.0,))
    graph = _RateLimited(failures)

    def view() -> SchemaNodeView:
        # A wait is taken only when it ends before the deadline, so the deadline is real.
        return build_node_view(
            "Person",
            reader=graph,
            catalogue=FakeCatalogue(),
            generation="g1",
            executor=executor,
            deadline=time.monotonic() + 60,
        )

    if failures == 1:
        assert view().centre.ref == "Person"
    else:
        with pytest.raises(GraphReadError) as raised:
            view()
        assert raised.value.upstream_status == 429
    assert [call[0] for call in graph.calls].count("429") == failures


class _RateLimitedWalk(_FakeGraph):
    """`_FakeGraph` whose first `rangeIncludes` in-arc walk answers HTTP 429."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def _walked(self, prop: str, dcids: Sequence[str]) -> None:
        if prop == "rangeIncludes" and not self.failed:
            self.failed = True
            raise GraphReadError("rate limited", upstream_status=429)


def test_a_rate_limited_walk_sends_only_that_walk_again(
    executor: ThreadPoolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node_view, "RATE_LIMIT_WAITS_SECONDS", (0.0,))
    graph = _RateLimitedWalk()
    view = build_node_view(
        "Person",
        reader=graph,
        catalogue=FakeCatalogue(),
        generation="g1",
        executor=executor,
        deadline=time.monotonic() + 60,
    )

    assert _nodes(view)["age"].domain_includes == ["Person"]
    assert graph.in_arc_reads("rangeIncludes") == ["Person", "Person"]
    assert graph.in_arc_reads("subPropertyOf") == ["Person"]


def test_chain_marks_follow_from_the_edges_in_any_order() -> None:
    """A class that reaches `Enumeration` only through a class on the walk's path is marked,
    whichever parent the edges list first."""
    for parents in (["B", "Enumeration"], ["Enumeration", "B"]):
        neighbours = node_view._Neighbours("centre")
        for ref in ("A", "B"):
            neighbours.add(ref)
        node_view._mark_chain({"A": parents, "B": ["A"]}, exclude="centre", neighbours=neighbours)

        listed = SchemaInstances(refs=[], complete=False)
        assert {ref: n.instances for ref, n in neighbours.by_ref.items()} == {
            "A": listed,
            "B": listed,
        }


def test_a_dcid_the_graph_lacks_is_refused(executor: ThreadPoolExecutor) -> None:
    with pytest.raises(NodeNotHeldError) as raised:
        _view("NoSuchNode", _FakeGraph(), executor)
    assert raised.value.dcid == "NoSuchNode"
