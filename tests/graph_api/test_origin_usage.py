"""`build_origin_usage` over a small origin catalogue, and the two routes that serve it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from code_index.shared import get_index
from graph_api.dataset_usage import build_usage
from graph_api.doors import schema as schema_door
from graph_api.origin_usage import build_origin_usage
from origin_catalogue import BASE, UN_SYSTEM, CardFacts, Catalogue, CatalogueSnapshot
from origin_catalogue import service as catalogue_service_module
from origin_catalogue.arc_tables import ArcTables
from origin_catalogue.service import CatalogueService
from origin_catalogue.un_mirror import UnArc
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)
from tests.origin_catalogue.support import named_labels

from .conftest import FakeGraphNodeReader, make_client

_WHO = "dc/base/UN_WHO"
_WB = "dc/base/WorldBankDatasets"
_UN_WHO = "undata/p/WHO"
_UN_ILO = "undata/p/ILO"
_IDENTITY = {"Accept-Encoding": "identity"}


def _arcs(
    population: str,
    *,
    constraints: dict[str, str | None] | None = None,
    qualifier: UnArc | None = None,
    observation_properties: tuple[str, ...] = (),
) -> tuple[UnArc, ...]:
    """A base variable's arcs. A constraint mapped to `None` is declared with no value arc."""
    constraints = constraints or {}
    rows = [
        *(UnArc(property="constraintProperties", dcid=prop, name=prop) for prop in constraints),
        UnArc(property="measuredProperty", dcid="count", name="count"),
        *(UnArc(property="observationProperties", dcid=p, name=p) for p in observation_properties),
        UnArc(property="populationType", dcid=population, name=population),
        UnArc(property="statType", dcid="measuredValue", name="measuredValue"),
        *(
            UnArc(property=prop, dcid=value, name=value)
            for prop, value in constraints.items()
            if value is not None
        ),
    ]
    if qualifier is not None:
        rows.append(qualifier)
    return tuple(sorted(rows, key=lambda arc: arc.property))


_BASE_ARCS = {
    "Count_Person_Female": _arcs("Person", constraints={"gender": "Female"}),
    "Count_Person_Male": _arcs("Person", constraints={"gender": "Male"}),
    "Count_Person_Unresolved": _arcs(
        "Person",
        constraints={"gender": None},
        qualifier=UnArc(
            property="measurementQualifier", dcid="Provisional", types=("ProvisionalNode",)
        ),
    ),
    "Count_Household": _arcs("Household", observation_properties=("observationAbout",)),
    # The registry excludes the provenance holding it, so the catalogue withholds it.
    "Count_Withheld": _arcs("Person"),
    # Superseded by `undata/WHO/SV_A`.
    "Count_Superseded": _arcs("Person"),
}

_UN_VARIABLES = {
    "undata/WHO/SV_A": un_variable(
        "undata/WHO/SERIES_A",
        name="A",
        constraints={"undata/WHO/SEX": ("Sex", "undata/WHO/SEX_F", "Female")},
    ),
    "undata/WHO/SV_B": un_variable("undata/WHO/SERIES_A", name="B"),
}


def _catalogue(*, un_system_enabled: bool = True, build_id: str | None = None) -> CatalogueSnapshot:
    return restrictive_catalogue(
        provenances=[
            provenance_row(_WHO, BASE, name="WHO via Data Commons"),
            provenance_row(_WB, BASE, name="World Bank"),
            provenance_row(_UN_WHO, UN_SYSTEM, segment="WHO", name="World Health Organization"),
            provenance_row(_UN_ILO, UN_SYSTEM, segment="ILO", name="ILO"),
        ],
        base_variables={
            "Count_Person_Female": ((_WHO, _WB), ()),
            "Count_Person_Male": ((_WHO,), ()),
            "Count_Person_Unresolved": ((_WHO,), ()),
            "Count_Household": ((_WB,), ()),
        },
        withheld_variables=("Count_Withheld", "Count_Superseded"),
        successors={"Count_Superseded": "undata/WHO/SV_A"},
        arcs=ArcTables(dfo={}, base=_BASE_ARCS, schema={}, labels=named_labels(_BASE_ARCS)),
        un_mirror=un_mirror(
            _UN_VARIABLES, excluded={"undata/WHO/SV_EXCLUDED": "undata/WHO/SERIES_X"}
        ),
        dataset_facts={_WHO: CardFacts(description="Health statistics."), _WB: CardFacts()},
        un_system_enabled=un_system_enabled,
        build_id=build_id,
    )


def _entries(catalogue: CatalogueSnapshot, origin) -> dict[str, dict]:
    body = build_origin_usage(catalogue, origin).model_dump(mode="json")
    return {entry["ref"]: entry for entry in body["provenances"]}


def test_the_base_document_counts_only_admitted_variables_under_each_provenance() -> None:
    """The withheld and superseded variables have arcs but no entry in the base table, and a
    variable two provenances hold counts under both."""
    entries = _entries(_catalogue(), "base")

    assert list(entries) == [_WHO, _WB]
    who, wb = entries[_WHO], entries[_WB]
    assert (who["label"], who["description"]) == ("WHO via Data Commons", "Health statistics.")
    assert (wb["label"], wb["description"]) == ("World Bank", None)
    assert (who["stat_var_count"], wb["stat_var_count"]) == (3, 2)
    examples = {
        dcid for entry in entries.values() for p in entry["patterns"] for dcid in p["examples"]
    }
    assert examples == {
        "Count_Person_Female",
        "Count_Person_Male",
        "Count_Person_Unresolved",
        "Count_Household",
    }
    assert who["facets"] == {
        "units": [],
        "scaling_factors": [],
        "measurement_methods": [],
        "observation_periods": [],
        "currencies": [],
    }


def test_an_unresolved_constraint_counts_in_its_pattern_and_adds_no_value() -> None:
    """`Count_Person_Unresolved` declares `gender` with no value arc, and its qualifier is a
    `ProvisionalNode`, which reads as none."""
    who = _entries(_catalogue(), "base")[_WHO]

    assert [
        (
            p["population_type"],
            p["constraint_properties"],
            p["measurement_qualifier"],
            p["stat_var_count"],
        )
        for p in who["patterns"]
    ] == [("Person", ["gender"], None, 3)]
    assert who["constraint_values"] == [
        {
            "property": "gender",
            "values": [
                {"value": "Female", "stat_var_count": 1},
                {"value": "Male", "stat_var_count": 1},
            ],
        }
    ]


@pytest.mark.parametrize("origin", ["base", "un_system"])
def test_a_source_scope_changes_nothing_in_the_document(origin) -> None:
    """The cache keys on the origin alone, so a scoped request must build the unscoped body."""
    catalogue = _catalogue()

    scoped = build_origin_usage(catalogue.scoped(frozenset({_WB})), origin)

    assert scoped == build_origin_usage(catalogue, origin)


def test_the_un_document_counts_each_agency_variable() -> None:
    """The ILO row holds no variable, so it has no entry, and UN rows carry no card facts."""
    entries = _entries(_catalogue(), "un_system")

    assert list(entries) == [_UN_WHO]
    who = entries[_UN_WHO]
    assert (who["label"], who["description"], who["stat_var_count"]) == (
        "World Health Organization",
        None,
        2,
    )
    assert [p["examples"] for p in who["patterns"]] == [["undata/WHO/SV_B"], ["undata/WHO/SV_A"]]
    assert who["constraint_values"] == [
        {
            "property": "undata/WHO/SEX",
            "values": [{"value": "undata/WHO/SEX_F", "stat_var_count": 1}],
        }
    ]


_RATIOS = "dc/base/Ratios"


def _ratio(denominator: UnArc | None) -> tuple[UnArc, ...]:
    rows = [
        UnArc(property="measuredProperty", dcid="count", name="Count"),
        UnArc(property="populationType", dcid="Person", name="Person", types=("Class",)),
        UnArc(property="statType", dcid="measuredValue", name="Measured value"),
    ]
    if denominator is not None:
        rows.append(denominator)
    return tuple(sorted(rows, key=lambda arc: arc.property))


def _ratio_catalogue() -> CatalogueSnapshot:
    """One provenance holding a variable with no denominator, one whose denominator arc names
    no value and one whose denominator is a `ProvisionalNode`."""
    arcs = {
        "Ratio_None": _ratio(None),
        "Ratio_Valueless": _ratio(UnArc(property="measurementDenominator", name="Unnamed")),
        "Ratio_Provisional": _ratio(
            UnArc(
                property="measurementDenominator", dcid="dc/p/Pending", types=("ProvisionalNode",)
            )
        ),
    }
    schema = {"dc/p/Pending": (UnArc(property="typeOf", dcid="ProvisionalNode"),)}
    return restrictive_catalogue(
        provenances=[provenance_row(_RATIOS, BASE, name="Ratios")],
        base_variables=dict.fromkeys(arcs, ((_RATIOS,), ())),
        # The arc tables hold every denominator a variable names.
        arcs=ArcTables(dfo={}, base=arcs, schema=schema, labels=named_labels(arcs, schema)),
    )


def test_an_unresolved_denominator_keeps_its_pattern_apart_from_no_denominator() -> None:
    patterns = _entries(_ratio_catalogue(), "base")[_RATIOS]["patterns"]

    assert [
        (p["measurement_denominator"], p["measurement_denominator_unresolved"], p["examples"])
        for p in patterns
    ] == [
        (None, False, ["Ratio_None"]),
        (None, True, ["Ratio_Valueless"]),
        ("dc/p/Pending", True, ["Ratio_Provisional"]),
    ]


def test_the_terms_label_every_node_the_patterns_and_values_name() -> None:
    """Each term reads its name and types from the variables' arcs, and a name equal to the
    dcid is no label. The `StatisticalVariable` type and each variable's own name are named by
    no pattern, so they are no terms."""
    body = build_origin_usage(_catalogue(), "un_system").model_dump(mode="json")

    assert body["terms"] == [
        {"ref": "measuredValue", "label": None, "types": []},
        {"ref": "undata/WHO/SERIES_A", "label": "undata/WHO/SERIES_A name", "types": ["Class"]},
        {"ref": "undata/WHO/SEX", "label": "Sex", "types": ["Property"]},
        {"ref": "undata/WHO/SEX_F", "label": "Female", "types": ["Enumeration"]},
        {"ref": "value", "label": None, "types": ["Property"]},
    ]


def test_the_base_terms_leave_out_a_provisional_qualifier() -> None:
    """`Count_Person_Unresolved`'s qualifier reads as none, so no pattern names it."""
    body = build_origin_usage(_catalogue(), "base").model_dump(mode="json")

    assert [term["ref"] for term in body["terms"]] == [
        "Female",
        "Household",
        "Male",
        "Person",
        "count",
        "gender",
        "measuredValue",
        "observationAbout",
    ]


# The routes


@dataclass
class _Counted:
    """Counts `build_origin_usage` calls by origin."""

    origins: list[str] = field(default_factory=list)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = schema_door.build_origin_usage

        def counted(catalogue, origin):
            self.origins.append(origin)
            return real(catalogue, origin)

        monkeypatch.setattr(schema_door, "build_origin_usage", counted)


@pytest.mark.parametrize(("origin", "ref"), [("base", _WHO), ("un_system", _UN_WHO)])
def test_each_route_answers_its_origin_once_per_catalogue_build_with_an_etag(
    origin: str,
    ref: str,
    serve_catalogue: Callable[[Catalogue], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds = _Counted()
    builds.install(monkeypatch)
    route = f"/api/graph/schema/usage/{origin}"
    serve_catalogue(_catalogue(build_id="build-a"))

    with make_client() as client:
        first = client.get(route, headers=_IDENTITY)
        etag = first.headers["ETag"]
        again = client.get(route, headers={**_IDENTITY, "If-None-Match": etag})
        gzipped = client.get(route, headers={"Accept-Encoding": "gzip"})

    assert first.status_code == 200, first.text
    assert first.headers["Cache-Control"] == "public, max-age=300"
    assert again.status_code == 304
    assert gzipped.headers["Content-Encoding"] == "gzip"
    assert gzipped.json() == first.json()
    assert ref in [entry["ref"] for entry in first.json()["provenances"]]
    assert builds.origins == [origin]


def test_a_request_bound_to_another_catalogue_build_builds_the_body_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service loads a second build mid-run, so the next request binds it and the cache's
    generation stamp, the bound build, misses the first build's entry."""
    builds = _Counted()
    builds.install(monkeypatch)
    loaded = [_catalogue(build_id="build-a")]
    service = CatalogueService(
        artifact_uri=f"file://{tmp_path / 'catalogue'}",
        recheck_seconds=0,
        loader=lambda _uri: loaded[0],
    )

    with (
        catalogue_service_module._singleton.poisoned_for_test(service),
        make_client() as client,
    ):
        first = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)
        loaded[0] = _catalogue(build_id="build-b")
        assert service.snapshot_current().build_id == "build-b"
        changed = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)
        repeat = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)

    assert [first.status_code, changed.status_code, repeat.status_code] == [200, 200, 200]
    assert changed.headers["ETag"] == first.headers["ETag"]
    assert builds.origins == ["base", "base"]


def test_the_un_route_with_the_un_origin_off_refuses_and_builds_nothing(
    serve_catalogue: Callable[[Catalogue], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = _Counted()
    builds.install(monkeypatch)
    serve_catalogue(_catalogue(un_system_enabled=False))

    with make_client() as client:
        refused = client.get("/api/graph/schema/usage/un_system", headers=_IDENTITY)
        base = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)

    assert refused.status_code == 404, refused.text
    body = refused.json()
    assert (body["code"], body["retryable"], body["cause"]["code"]) == (
        "not_found",
        False,
        "origin_not_served",
    )
    assert refused.headers["Cache-Control"] == "no-store"
    assert base.status_code == 200, base.text
    assert builds.origins == ["base"]


@dataclass
class _EmptyCoverageIndex:
    facet_metadata: dict = field(default_factory=dict)
    provenance_currencies: dict = field(default_factory=dict)


@dataclass
class _Manifest:
    build_id: str = "coverage-empty"


@dataclass
class _Snapshot:
    index: _EmptyCoverageIndex = field(default_factory=_EmptyCoverageIndex)
    manifest: _Manifest = field(default_factory=_Manifest)


class _FakeCoverage:
    def snapshot(self) -> _Snapshot:
        return _Snapshot()


@pytest.mark.parametrize("catalogue", [FakeCatalogue, _catalogue], ids=["fake", "base-and-un"])
def test_the_default_usage_body_is_build_usages_whatever_the_catalogue_holds(
    catalogue: Callable[[], Catalogue], serve_catalogue: Callable[[Catalogue], None]
) -> None:
    """The graph holds no Provenance node, so `build_usage` reads no node facts. The body
    carries no `terms` and no pattern's `measurement_denominator_unresolved`."""
    serve_catalogue(catalogue())
    expected = schema_door._freeze(
        build_usage(code_index=get_index(), coverage=_EmptyCoverageIndex(), nodes={})
    )

    with make_client(graph=FakeGraphNodeReader(), coverage=_FakeCoverage()) as client:
        response = client.get("/api/graph/schema/usage", headers=_IDENTITY)

    assert response.status_code == 200, response.text
    assert response.content == expected.identity
    body = response.json()
    assert "terms" not in body
    assert body["provenances"]
    assert all(
        "measurement_denominator_unresolved" not in pattern
        for entry in body["provenances"]
        for pattern in entry["patterns"]
    )
