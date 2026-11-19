"""`build_origin_usage` over a small origin catalogue, and the two routes that serve it."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from graph_api.doors import schema as schema_door
from graph_api.origin_usage import build_origin_usage
from origin_catalogue import BASE, UN_SYSTEM, CardFacts, Catalogue, CatalogueSnapshot
from origin_catalogue.arc_tables import ArcTables
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


def test_a_provenance_the_scope_leaves_out_gets_no_entry() -> None:
    entries = _entries(_catalogue().scoped(frozenset({_WB})), "base")

    assert list(entries) == [_WB]
    assert entries[_WB]["stat_var_count"] == 2


def test_the_un_document_counts_each_agency_variable_and_no_excluded_one() -> None:
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
    serve_catalogue: Callable[[Catalogue], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = _Counted()
    builds.install(monkeypatch)
    bound = [_catalogue(build_id="build-a")]
    serve_catalogue(bound[0])
    monkeypatch.setattr(schema_door, "current_catalogue", lambda: bound[0])

    with make_client() as client:
        first = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)
        bound[0] = _catalogue(build_id="build-b")
        changed = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)
        repeat = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)

    assert [first.status_code, changed.status_code, repeat.status_code] == [200, 200, 200]
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


# Digest of the default usage body over the suite's bound code index, a coverage index holding
# nothing and a graph holding no Provenance node. The outside-origin routes leave it unchanged.
_USAGE_DIGEST = "4f2ab9de5585a9b135692d7e1ade382d6af41f23d1fe95a74588bfa5451808a1"


@dataclass
class _EmptyCoverageIndex:
    facet_metadata: dict = field(default_factory=dict)
    provenance_currencies: dict = field(default_factory=dict)


@dataclass
class _Manifest:
    build_id: str = "coverage-pinned"


@dataclass
class _Snapshot:
    index: _EmptyCoverageIndex = field(default_factory=_EmptyCoverageIndex)
    manifest: _Manifest = field(default_factory=_Manifest)


class _FakeCoverage:
    def snapshot(self) -> _Snapshot:
        return _Snapshot()


@pytest.mark.parametrize("catalogue", [FakeCatalogue, _catalogue], ids=["fake", "base-and-un"])
def test_the_default_usage_body_is_byte_identical_whatever_the_catalogue_holds(
    catalogue: Callable[[], Catalogue], serve_catalogue: Callable[[Catalogue], None]
) -> None:
    serve_catalogue(catalogue())

    with make_client(graph=FakeGraphNodeReader(), coverage=_FakeCoverage()) as client:
        response = client.get("/api/graph/schema/usage", headers=_IDENTITY)

    assert response.status_code == 200, response.text
    assert hashlib.sha256(response.content).hexdigest() == _USAGE_DIGEST
