"""`build_origin_usage` over a small origin catalogue, and the two routes that serve it."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import pytest

from code_index.shared import get_index
from graph_api import deps as deps_module
from graph_api.dataset_usage import build_usage
from graph_api.doors import schema as schema_door
from graph_api.origin_usage import build_origin_usage
from graph_release import service as release_service_module
from graph_release.context import bound_release
from graph_release.service import Release, ReleaseService
from origin_catalogue import BASE, UN_SYSTEM, CardFacts, Catalogue, CatalogueSnapshot
from origin_catalogue import service as catalogue_service_module
from origin_catalogue.arc_tables import ArcTables
from origin_catalogue.service import CatalogueService
from origin_catalogue.un_mirror import UnArc
from tests.graph_release.support import origin_release, release_at_revision
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


_TERMS = "dc/base/Terms"


def test_a_term_reads_the_label_its_node_card_gives() -> None:
    """The node card labels a node by the display name of its own names. `Count_Person` is a
    held denominator whose names differ from the one its arc carries, and `Years10To14` is an
    arc target the tables label apart from its arc's name. `Person`'s only name is its dcid."""
    arcs = {
        "Count_Person_10To14": (
            UnArc(property="age", dcid="Years10To14", name="Years 10 To 14"),
            UnArc(property="constraintProperties", dcid="age", name="age"),
            UnArc(property="measuredProperty", dcid="count", name="count"),
            UnArc(property="measurementDenominator", dcid="Count_Person", name="Population"),
            UnArc(property="populationType", dcid="Person", name="Person"),
            UnArc(property="statType", dcid="measuredValue", name="measuredValue"),
        ),
        "Count_Person": (
            UnArc(property="name", value="Population"),
            UnArc(property="name", value="Total population"),
        ),
    }
    labels = {**named_labels(arcs), "Years10To14": "Years10To14 range"}
    catalogue = restrictive_catalogue(
        provenances=[provenance_row(_TERMS, BASE, name="Terms")],
        base_variables={"Count_Person_10To14": ((_TERMS,), ())},
        arcs=ArcTables(dfo={}, base=arcs, schema={}, labels=labels),
    )

    terms = {
        term["ref"]: term["label"]
        for term in build_origin_usage(catalogue, "base").model_dump(mode="json")["terms"]
    }

    assert (terms["Count_Person"], terms["Years10To14"], terms["Person"]) == (
        "Total population",
        "Years10To14 range",
        None,
    )


_OVERLAP_SOURCE = "dc/base/Overlap"


def _person_arcs(population: str) -> tuple[UnArc, ...]:
    return (
        UnArc(property="measuredProperty", dcid="count", name="count"),
        UnArc(property="populationType", dcid=population, name=population),
        UnArc(property="statType", dcid="measuredValue", name="measuredValue"),
    )


def test_a_base_variable_whose_node_is_dfo_s_reads_the_bound_release() -> None:
    """Both variables are DFO's nodes, held in the build's `dfo` arc table and listed in the
    base variable table. The release holds `Count_Released` with another population, and does
    not hold `Count_Unreleased`, whose node `/node/{dcid}` then reads from the graph."""
    build_arcs = {
        "Count_Released": _person_arcs("Person"),
        "Count_Unreleased": _person_arcs("Person"),
    }
    catalogue = restrictive_catalogue(
        provenances=[provenance_row(_OVERLAP_SOURCE, BASE, name="Overlap")],
        dfo_variables=build_arcs,
        base_variables=dict.fromkeys(build_arcs, ((_OVERLAP_SOURCE,), ())),
        arcs=ArcTables(dfo=build_arcs, base={}, schema={}, labels=named_labels(build_arcs)),
    )
    released = {
        "Count_Released": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "measuredProperty": {"nodes": [{"dcid": "count", "name": "count"}]},
            "populationType": {
                "nodes": [{"dcid": "Household", "name": "Household", "types": ["Class"]}]
            },
            "statType": {"nodes": [{"dcid": "measuredValue", "name": "measuredValue"}]},
        }
    }

    with bound_release(origin_release(released)):
        entry = _entries(catalogue, "base")[_OVERLAP_SOURCE]

    assert entry["stat_var_count"] == 1
    assert [(p["population_type"], p["examples"]) for p in entry["patterns"]] == [
        ("Household", ["Count_Released"])
    ]


class _CountedRows(dict):
    """Provenance rows that count the scans of every row."""

    scans = 0

    def values(self):
        self.scans += 1
        return super().values()


def test_the_un_document_scans_the_provenance_rows_once_per_agency() -> None:
    """Both UN variables sit under the WHO segment, so one scan of the rows answers both."""
    catalogue = _catalogue()
    rows = _CountedRows(catalogue.tables.provenances)
    catalogue = replace(catalogue, tables=replace(catalogue.tables, provenances=rows))
    rows.scans = 0

    build_origin_usage(catalogue, "un_system")

    assert rows.scans == 1


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


class _SwappableReleases:
    """A release service serving `release`, which a test swaps between requests."""

    def __init__(self, release: Release) -> None:
        self.release = release

    def snapshot(self) -> Release:
        return self.release

    def serves(self, release: Release) -> bool:
        return release is self.release

    def register_swap_guard(self, guard: object) -> Callable[[], None]:
        return lambda: None

    def on_swap(self, callback: object) -> Callable[[], None]:
        return lambda: None

    def close(self) -> None:
        pass


@contextmanager
def _serving_releases(releases: _SwappableReleases) -> Iterator[None]:
    with release_service_module._singleton.poisoned_for_test(cast("ReleaseService", releases)):
        yield


def test_a_request_bound_to_another_release_builds_the_body_again(
    serve_catalogue: Callable[[Catalogue], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The body reads the release's arcs, so the cache keys on the release under one
    catalogue build."""
    builds = _Counted()
    builds.install(monkeypatch)
    serve_catalogue(_catalogue(build_id="build-a"))
    releases = _SwappableReleases(release_at_revision("rev-1"))

    with _serving_releases(releases), make_client() as client:
        first = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)
        releases.release = release_at_revision("rev-2")
        swapped = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)
        repeat = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)

    assert [first.status_code, swapped.status_code, repeat.status_code] == [200, 200, 200]
    assert builds.origins == ["base", "base"]


@pytest.mark.parametrize("origin", ["base", "un_system"])
def test_a_catalogue_with_no_tables_serves_no_origin_usage(
    origin: str, serve_catalogue: Callable[[Catalogue], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    builds = _Counted()
    builds.install(monkeypatch)
    serve_catalogue(FakeCatalogue())

    with make_client() as client:
        refused = client.get(f"/api/graph/schema/usage/{origin}", headers=_IDENTITY)

    assert refused.status_code == 404, refused.text
    body = refused.json()
    assert (body["code"], body["retryable"], body["cause"]["code"]) == (
        "not_found",
        False,
        "origin_not_served",
    )
    assert builds.origins == []


def test_an_origin_build_logs_no_dcp_queue_wait(
    serve_catalogue: Callable[[Catalogue], None],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The build reads no DCP, so a wait for its worker is no DCP read's wait, even past a
    threshold every wait passes."""
    monkeypatch.setattr(deps_module, "DCP_QUEUE_WAIT_WARN_SECONDS", -1.0)
    serve_catalogue(_catalogue())

    with caplog.at_level(logging.DEBUG, logger=deps_module.logger.name), make_client() as client:
        response = client.get("/api/graph/schema/usage/base", headers=_IDENTITY)

    assert response.status_code == 200, response.text
    assert not [
        r
        for r in caplog.records
        if r.name == deps_module.logger.name and r.levelno >= logging.WARNING
    ]


def test_every_schema_path_a_usage_description_names_is_served() -> None:
    with make_client() as client:
        spec = client.get("/openapi.json").json()

    named = set(re.findall(r"`(/schema/[^`]*)`", str(spec["components"]["schemas"])))

    assert named
    assert {f"/api/graph{path}" for path in named} <= set(spec["paths"])


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
