"""The upstreams of the local stack, answered from `fixtures/readings.json`.

The Datasets surface reads a graph, an SDMX endpoint and a coverage index. The local stack
reaches none of the live ones, so `local_app.py` hands the surface these doubles, built from
one fixture: they serve each reading's StatVar arcs, labels, facets and points. The mint and
observations doors then read the fixture as they would read the live graph. `seed.py` mints
each reading through the running server.

The Presentation surface builds its own graph client from `DATASETS_GRAPH_BASE_URL`, which
`local.env` points at the server's own `/local-stack/graph` route. `node_reply` answers that
client's `/v2/node` reads from the same fixture: each node's out-arcs, a labelled node's
`name`, and no in-arcs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx

from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

READINGS = Path(__file__).with_name("fixtures") / "readings.json"
PLACE_ROLE = "observationAbout"
_SDMX_BASE_URL = "http://local-stack.invalid/core/api"
_SDMX_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"
_NOT_APPLICABLE = "NotApplicable"


@dataclass(frozen=True)
class FacetFixture:
    facet_id: str
    provenance: str
    method: str | None
    unit: str | None


@dataclass(frozen=True)
class Series:
    """One series of a StatVar: its entity under each role, its facet and its points."""

    entities: Mapping[str, str]
    facet: FacetFixture
    points: tuple[tuple[str, float], ...]

    def graph_facet(self) -> Facet:
        return Facet(
            facet_id=self.facet.facet_id,
            provenance_id=self.facet.provenance,
            measurement_method=self.facet.method,
            unit=self.facet.unit,
            obs_count=len(self.points),
            earliest_date=self.points[0][0],
            latest_date=self.points[-1][0],
        )


@dataclass(frozen=True)
class Reading:
    name: str
    stat_var: str
    scope: Mapping[str, object]
    series: tuple[Series, ...]


@dataclass(frozen=True)
class StatVar:
    arcs: Mapping[str, tuple[str, ...]]
    label: str
    roles: RoleNames | None

    def role_columns(self) -> tuple[str, ...]:
        """The SDMX entity columns of this StatVar's rows, sorted by role name."""
        if self.roles is None:
            return (PLACE_ROLE,)
        return tuple(sorted((self.roles.donor or "", self.roles.recipient or "")))


@dataclass(frozen=True)
class Fixture:
    places: Mapping[str, str]
    labels: Mapping[str, str]
    stat_vars: Mapping[str, StatVar]
    readings: tuple[Reading, ...]


def load_fixture(path: Path = READINGS) -> Fixture:
    raw = json.loads(path.read_text())
    stat_vars = {
        dcid: StatVar(
            arcs={prop: tuple(values) for prop, values in body["arcs"].items()},
            label=body["label"],
            roles=RoleNames(**body["roles"]) if body["roles"] is not None else None,
        )
        for dcid, body in raw["stat_vars"].items()
    }
    readings = tuple(
        Reading(
            name=body["name"],
            stat_var=body["stat_var"],
            scope=body["scope"],
            series=tuple(
                Series(
                    entities=series["entities"],
                    facet=FacetFixture(**series["facet"]),
                    points=tuple((date, float(value)) for date, value in series["points"]),
                )
                for series in body["series"]
            ),
        )
        for body in raw["readings"]
    )
    unknown = {reading.stat_var for reading in readings} - stat_vars.keys()
    if unknown:
        raise ValueError(f"readings name StatVars the fixture does not define: {sorted(unknown)}")
    return Fixture(
        places=raw["places"], labels=raw["labels"], stat_vars=stat_vars, readings=readings
    )


def _anchor(stat_var: StatVar, series: Series) -> str:
    """The entity a graph facet read keys the series under: its place, or its recipient."""
    if stat_var.roles is None:
        return series.entities[PLACE_ROLE]
    assert stat_var.roles.recipient is not None
    return series.entities[stat_var.roles.recipient]


def graph_reader(fixture: Fixture) -> FakeMintGraphReader:
    """The graph the mint reads: each StatVar's arcs and label, each place as a `Country`
    node, and each series' facet under its StatVar and anchor entity."""
    arcs: dict[str, Mapping[str, tuple[str, ...]]] = {
        dcid: stat_var.arcs for dcid, stat_var in fixture.stat_vars.items()
    }
    for place, name in fixture.places.items():
        arcs[place] = {"typeOf": ("Country",), "name": (name,)}
    facets: dict[tuple[str, str], list[Facet]] = {}
    for reading in fixture.readings:
        stat_var = fixture.stat_vars[reading.stat_var]
        for series in reading.series:
            key = (reading.stat_var, _anchor(stat_var, series))
            held = facets.setdefault(key, [])
            if all(facet.facet_id != series.facet.facet_id for facet in held):
                held.append(series.graph_facet())
    labels = {
        **fixture.labels,
        **fixture.places,
        **{dcid: stat_var.label for dcid, stat_var in fixture.stat_vars.items()},
    }
    return FakeMintGraphReader(arcs=arcs, facets=facets, labels=labels)


def graph_nodes(fixture: Fixture) -> dict[str, dict[str, tuple[str, ...]]]:
    """Every node the fixture names, by dcid, with its out-arcs: a StatVar's arcs, each place
    as a `Country` node, and a `name` for each labelled node."""
    nodes: dict[str, dict[str, tuple[str, ...]]] = {
        dcid: dict(stat_var.arcs) for dcid, stat_var in fixture.stat_vars.items()
    }
    for place, name in fixture.places.items():
        nodes[place] = {"typeOf": ("Country",), "name": (name,)}
    for dcid, label in fixture.labels.items():
        nodes.setdefault(dcid, {}).setdefault("name", (label,))
    return nodes


def node_reply(nodes: Mapping[str, Mapping[str, tuple[str, ...]]], payload: Mapping) -> dict:
    """The `/v2/node` reply to `payload`: for an out-arc read (`->p`, `->[p,q]` or `->*`) each
    node's arcs among those named, `{}` for a node holding none, and for an in-arc read (`<-p`)
    no subjects, since the local stack reads none. One page holds every node."""
    spec = cast(str, payload["property"])
    requested = cast("list[str]", payload["nodes"])
    if spec.startswith("<-"):
        return {"data": {dcid: {} for dcid in requested}}
    if not spec.startswith("->"):
        raise ValueError(f"local-stack graph double: unexpected arc spec {spec!r}")
    named = spec[2:].strip("[]")
    data: dict[str, object] = {}
    for dcid in requested:
        arcs = nodes.get(dcid, {})
        properties = arcs if named == "*" else named.split(",")
        held = {
            prop: {"nodes": [{"value": value} for value in arcs[prop]]}
            for prop in properties
            if arcs.get(prop)
        }
        data[dcid] = {"arcs": held} if held else {}
    return {"data": data}


def _all_series(fixture: Fixture) -> dict[tuple[str, tuple[tuple[str, str], ...]], Series]:
    """Every series once, keyed by StatVar and sorted role bindings; two readings may share
    one."""
    held: dict[tuple[str, tuple[tuple[str, str], ...]], Series] = {}
    for reading in fixture.readings:
        for series in reading.series:
            held[(reading.stat_var, tuple(sorted(series.entities.items())))] = series
    return held


def sdmx_rows(fixture: Fixture, params: Mapping[str, str]) -> str:
    """The SDMX CSV body a data read with `params` filters receives: one StatVar's rows, the
    filters on its role columns and facet id applied."""
    stat_var_filter = params.get("c[variableMeasured]")
    if stat_var_filter is None:
        raise ValueError(f"local-stack SDMX double: a data read without a StatVar filter: {params}")
    (dcid,) = stat_var_filter.split(",")
    stat_var = fixture.stat_vars[dcid]
    columns = stat_var.role_columns()

    def admitted(key: str, value: str) -> bool:
        return key not in params or value in params[key].split(",")

    lines = [
        ",".join(
            (
                "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured",
                *columns,
                "unit,measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
                "scalingFactor,facetId",
            )
        )
    ]
    for (series_var, _bindings), series in _all_series(fixture).items():
        if series_var != dcid:
            continue
        entities = tuple(series.entities[column] for column in columns)
        if not all(
            admitted(f"c[{column}]", entity)
            for column, entity in zip(columns, entities, strict=True)
        ) or not admitted("c[facetId]", series.facet.facet_id):
            continue
        lines.extend(
            ",".join(
                (
                    f"dataflow,DC:DF_OBS(1.0.0),I,{dcid}",
                    *entities,
                    series.facet.unit or _NOT_APPLICABLE,
                    series.facet.method or _NOT_APPLICABLE,
                    _NOT_APPLICABLE,
                    series.facet.provenance,
                    date,
                    repr(value),
                    "",
                    series.facet.facet_id,
                )
            )
            for date, value in series.points
        )
    return "\r\n".join(lines) + "\r\n"


def sdmx_client(fixture: Fixture) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/sdmx/v3/data/" not in request.url.path:
            raise ValueError(
                f"local-stack SDMX double: unexpected request {request.method} {request.url}"
            )
        body = sdmx_rows(fixture, dict(request.url.params))
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV})

    return SdmxClient(base_url=_SDMX_BASE_URL, transport=httpx.MockTransport(handler))


def coverage(fixture: Fixture) -> FakeCoverageService:
    """Pair coverage for each role-bearing StatVar's series, which the mint reads to bind
    donor and recipient; place-keyed StatVars need none."""
    rows = []
    metadata: dict[str, FacetMetadata] = {}
    role_names: dict[str, RoleNames] = {}
    for (dcid, _bindings), series in _all_series(fixture).items():
        roles = fixture.stat_vars[dcid].roles
        if roles is None:
            continue
        assert roles.donor is not None and roles.recipient is not None
        role_names[dcid] = roles
        rows.append(
            (
                dcid,
                series.facet.facet_id,
                series.entities[roles.donor],
                series.entities[roles.recipient],
                [int(date) for date, _value in series.points],
            )
        )
        metadata[series.facet.facet_id] = FacetMetadata(
            provenance_id=series.facet.provenance,
            measurement_method=series.facet.method,
            unit=series.facet.unit,
        )
    return FakeCoverageService(
        build_coverage_index(rows, role_names=role_names, facet_metadata=metadata)
    )
