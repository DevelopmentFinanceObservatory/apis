"""A minted version's frozen observations equal the live preview of the same spec.

Each case builds one upstream: the graph the engine reads (`FakeGraph`), the same nodes as the
materialise graph reader, and one SDMX double that serves every facet's rows and the DAC
conversion tables. Live resolve (`resolve_variable` with `include_observations`) builds its
preview through `_build_observations`. The spec's definition is then minted through
`POST /api/datasets` (`serve_definition`) over the same upstream, and the minted version is read
back through `GET /api/datasets/{spec_id}/{content_version}/observations`. The two bodies must be
equal, so a mint that elects another facet, records another conversion, or labels a constrained
member another way fails here.

Cases under small preview caps drive each preview mode end to end, so the mode the frozen side
derives from what the mint stored (bound pairs, series count, `default_scope_plural`) is
compared with the mode live plans: role bulk, scope bulk, the too-large omission and inline
truncation.

The `recorded-count-person-rural` case replays
`tests/fixtures/mint_live_parity_count_person_rural.json`, recorded from the dev graph and SDMX:
`node_arcs_many` for the StatVar's arcs and constraint value, `node_labels_batch` for its
labels, `variable_facets_batch` at Kenya and Ghana, and one `SdmxClient.data` read of the
StatVar at both places, each point filed under its row's facet. Every facet's recorded point
count equals the graph's `obs_count`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import httpx
import pytest

import search.observations
from artifact_store import LiveArtifactInventory
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.doors.observations import observations_body
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient, SdmxObservation
from materialise.graph import Facet
from materialise.rate_tables import RateTableCache
from origin_catalogue import BASE, Catalogue
from origin_catalogue.arc_tables import WireArcs
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet as EngineFacet
from search.engine.quantity_read import QuantityOutput
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.models import Spec
from search.observations import OBSERVATIONS_PREVIEW_TOO_LARGE, ObservationSettings
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.datasets_api._app_factories import make_client
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._version_record_store import FakeVersionRecordDatasetStore
from tests.datasets_api.conftest import FakeDatasetStore
from tests.datasets_api.test_pin_door import FakeArtifactReferenceWriter
from tests.datasets_api.test_snapshot_lifecycle import _InventoryRecordingArtifactStore
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise._rate_table_support import (
    RateTableServer,
    conversion_arcs,
    recipe_table_bodies,
)
from tests.materialise._scope_availability import ScopeAvailability
from tests.origin_catalogue.fake_catalogue import (
    fake_catalogue_service,
    provenance_row,
)
from tests.origin_catalogue.support import listing_base_variables

_SDMX_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"
_SDMX_BASE_URL = "https://example.org/core/api"


def _sdmx_header(entity_columns: tuple[str, ...]) -> str:
    return ",".join(
        (
            "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured",
            *entity_columns,
            "unit,measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
            "scalingFactor,facetId\r\n",
        )
    )


_NOT_APPLICABLE = "NotApplicable"

_SV = "TESTFIXTURE/MintLiveParity/Amount"
_SV_NAME = "Mint live parity amount"
_ALT_SV = "TESTFIXTURE/MintLiveParity/AltAmount"
_QUERY = "mint live parity amount"
_KEN, _GHA, _NOR, _UGA = "country/KEN", "country/GHA", "country/NOR", "country/UGA"
_PLACE_NAMES = {_KEN: "Kenya", _GHA: "Ghana", _NOR: "Norway", _UGA: "Uganda"}
# A role-bearing StatVar's roles. Destination comes first in the plan's role names, so an
# unsorted role tuple would list the debtor before the creditor.
_DONOR_ROLE, _RECIPIENT_ROLE = "creditor", "debtor"
_ROLE_QUERY = "mint live parity amount from Norway"
_PROV_A, _PROV_B = "dc/base/MintLiveParityA", "dc/base/MintLiveParityB"
_PURPOSE_PROPERTY = "crsPurpose"
_PURPOSE = "TESTFIXTURE/MintLiveParity/HealthPurpose"
_RECIPE = "ConversionRecipeDAC"
_YEARS = range(2018, 2024)

_LABELS = {
    _PROV_A: "Parity Source A",
    _PROV_B: "Parity Source B",
    "Estimated": "Estimated",
    "CurrencyUSD": "US dollar",
    "USDollar": "US dollar",
    "CurrencyEUR": "Euro",
    _PURPOSE: "Parity health purpose",
    **_PLACE_NAMES,
}


@dataclass(frozen=True)
class _FacetFixture:
    """One facet of the StatVar at one place, with its points."""

    facet_id: str
    provenance: str
    points: tuple[tuple[str, float], ...]
    method: str | None = None
    unit: str | None = "CurrencyUSD"

    def facet(self) -> Facet:
        return Facet(
            facet_id=self.facet_id,
            provenance_id=self.provenance,
            measurement_method=self.method,
            unit=self.unit,
            obs_count=len(self.points),
            earliest_date=self.points[0][0],
            latest_date=self.points[-1][0],
        )

    def engine_facet(self) -> EngineFacet:
        return EngineFacet(
            facet_id=self.facet_id,
            provenance_id=self.provenance,
            measurement_method=self.method,
            unit=self.unit,
            obs_count=len(self.points),
            earliest_date=self.points[0][0],
            latest_date=self.points[-1][0],
            dates=tuple(date for date, _value in self.points),
            values=tuple(value for _date, value in self.points),
        )

    def graph_obs(self) -> dict[str, object]:
        return {
            "facetId": self.facet_id,
            "earliestDate": self.points[0][0],
            "latestDate": self.points[-1][0],
            "obsCount": len(self.points),
            "provenanceId": self.provenance,
            **({"measurementMethod": self.method} if self.method else {}),
            **({"unit": self.unit} if self.unit else {}),
            "dates": [date for date, _value in self.points],
            "values": [value for _date, value in self.points],
        }

    def csv_rows(self, stat_var: str, entities: tuple[str, ...]) -> str:
        return "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{','.join(entities)},"
            f"{self.unit or _NOT_APPLICABLE},"
            f"{self.method or _NOT_APPLICABLE},{_NOT_APPLICABLE},{self.provenance},{date},"
            f"{value},,{self.facet_id}\r\n"
            for date, value in self.points
        )


@dataclass(frozen=True)
class _Upstream:
    """One recorded upstream: the StatVar's facets at each place, whether the StatVar is a
    constrained member, and the requested unit. With a `donor` the StatVar is role-bearing:
    each place in `facets` receives from `donor`, and the query names `donor` as origin and
    each place as destination. With an `alternate`, detection also offers `_ALT_SV`, which
    holds no data, so the answer collapses to the lone surviving candidate
    (`candidate_collapse`). A `scoped` query names no place, so both sides read the default
    place scope: the places in `facets`. `stat_var_arcs`, in the graph's node form, replace the
    synthetic StatVar's arcs, and `labels` adds node labels."""

    facets: Mapping[str, tuple[_FacetFixture, ...]]
    constrained: bool = False
    unit_text: str | None = None
    donor: str | None = None
    alternate: bool = False
    scoped: bool = False
    stat_var: str = _SV
    stat_var_arcs: Mapping[str, object] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    @property
    def query(self) -> str:
        if self.donor is None:
            return _QUERY
        return f"{_ROLE_QUERY} to {' and '.join(_PLACE_NAMES[p] for p in self.facets)}"

    @property
    def entities(self) -> list[str]:
        if self.scoped:
            return []
        places = [self.donor, *self.facets] if self.donor is not None else list(self.facets)
        return [_PLACE_NAMES[place] for place in places]

    def _entity_columns(self) -> tuple[str, ...]:
        if self.donor is None:
            return ("observationAbout",)
        return (_DONOR_ROLE, _RECIPIENT_ROLE)

    def _entity_values(self, place: str) -> tuple[str, ...]:
        return (place,) if self.donor is None else (self.donor, place)

    def _nodes(self) -> dict[str, dict[str, object]]:
        arcs: dict[str, object] = {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": _SV_NAME}]},
            "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        if self.donor is not None:
            arcs["observationProperties"] = {
                "nodes": [{"dcid": _DONOR_ROLE}, {"dcid": _RECIPIENT_ROLE}]
            }
        if self.constrained:
            arcs["constraintProperties"] = {"nodes": [{"dcid": _PURPOSE_PROPERTY}]}
            arcs[_PURPOSE_PROPERTY] = {"nodes": [{"dcid": _PURPOSE}]}
        alt_arcs = {**arcs, "measuredProperty": {"nodes": [{"dcid": "altAmount"}]}}
        return {
            self.stat_var: {"label": _SV_NAME, "arcs": self.stat_var_arcs or arcs},
            **(
                {_ALT_SV: {"label": "Alternate amount", "arcs": alt_arcs}} if self.alternate else {}
            ),
            **{place: {"label": name, "type": "Country"} for place, name in _PLACE_NAMES.items()},
            **{
                dcid: {"label": label}
                for dcid, label in {**_LABELS, **self.labels}.items()
                if dcid not in _PLACE_NAMES
            },
        }

    def stat_var_arcs_by_dcid(self) -> dict[str, WireArcs]:
        """The arcs of each StatVar detection offers, in the graph's node form."""
        return {
            dcid: cast("WireArcs", node["arcs"])
            for dcid, node in self._nodes().items()
            if "arcs" in node
        }

    def engine_graph(self) -> FakeGraph:
        return FakeGraph(
            nodes=self._nodes(),
            detect={
                self.query: {
                    "svs": [self.stat_var, _ALT_SV] if self.alternate else [self.stat_var],
                    "cosine_scores": [0.9, 0.88] if self.alternate else [0.9],
                }
            },
            resolve={name: place for place, name in _PLACE_NAMES.items()},
            obs={
                f"{self.stat_var}|{place}": [fixture.graph_obs() for fixture in fixtures]
                for place, fixtures in self.facets.items()
            },
            children={"Earth": {"Country": list(self.facets)}},
        )

    def materialise_graph(self) -> FakeMintGraphReader:
        arcs = self.engine_graph().member_arcs()
        arcs.update(
            conversion_arcs(
                recipe_id=_RECIPE,
                provenance_id=_PROV_A,
                currencies=["CurrencyUSD", "CurrencyEUR"],
            )
        )
        return FakeMintGraphReader(
            arcs=arcs,
            facets={
                (self.stat_var, place): [fixture.facet() for fixture in fixtures]
                for place, fixtures in self.facets.items()
            },
            labels={dcid: str(node["label"]) for dcid, node in self._nodes().items()},
            child_places={("Earth", "Country"): list(self.facets)},
        )

    def _scope_availability(self) -> ScopeAvailability:
        """The default place scope's availability read: the places holding a facet."""
        return ScopeAvailability(data=[(self.stat_var, place) for place in self.facets])

    def engine_sdmx(self) -> FakeSdmxClient:
        """The engine's SDMX client, which serves a role bulk read (`role_series`), a default
        place scope's latest points (`subject_observations`) and its availability."""
        return FakeSdmxClient(
            observations=(
                {}
                if self.donor is None
                else {
                    self.stat_var: {
                        ((_DONOR_ROLE, self.donor), (_RECIPIENT_ROLE, place)): [
                            fixture.engine_facet() for fixture in fixtures
                        ]
                        for place, fixtures in self.facets.items()
                    }
                }
            ),
            availability={},
            subject_observations={
                self.stat_var: {
                    place: [
                        SdmxObservation(
                            variable=self.stat_var,
                            role_values={"observationAbout": place},
                            unit=fixture.unit,
                            measurement_method=fixture.method,
                            observation_period=None,
                            provenance=fixture.provenance,
                            time_period=date,
                            value=str(value),
                            scaling_factor=None,
                            facet_id=fixture.facet_id,
                        )
                        for fixture in fixtures
                        for date, value in fixture.points[-1:]
                    ]
                    for place, fixtures in self.facets.items()
                }
            },
            scope_availability=self._scope_availability(),
        )

    def _data(self, request: httpx.Request) -> httpx.Response:
        """Every facet's rows the request's StatVar, place and facet filters admit, or the
        default place scope's availability."""
        if (listed := self._scope_availability().respond(request)) is not None:
            return listed
        params = request.url.params

        def admitted(key: str, value: str) -> bool:
            return key not in params or value in params[key].split(",")

        columns = self._entity_columns()
        body = _sdmx_header(columns)
        for place, fixtures in self.facets.items():
            entities = self._entity_values(place)
            if not admitted("c[variableMeasured]", self.stat_var) or not all(
                admitted(f"c[{column}]", entity)
                for column, entity in zip(columns, entities, strict=True)
            ):
                continue
            body += "".join(
                fixture.csv_rows(self.stat_var, entities)
                for fixture in fixtures
                if admitted("c[facetId]", fixture.facet_id)
            )
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV})

    def sdmx(self) -> SdmxClient:
        tables = recipe_table_bodies(
            _RECIPE,
            rates={
                "CurrencyUSD": dict.fromkeys(_YEARS, 1.0),
                "CurrencyEUR": dict.fromkeys(_YEARS, 0.9),
            },
        )
        return RateTableServer(tables, fallback=self._data).client(base_url=_SDMX_BASE_URL)

    def coverage(self) -> FakeCoverageService:
        """A role-bearing StatVar's pair coverage, which both the live plan and the mint
        read; empty for a place-keyed one."""
        if self.donor is None:
            return FakeCoverageService(build_coverage_index([], role_names={}))
        rows = [
            (
                self.stat_var,
                fixture.facet_id,
                self.donor,
                place,
                [int(d) for d, _v in fixture.points],
            )
            for place, fixtures in self.facets.items()
            for fixture in fixtures
        ]
        return FakeCoverageService(
            build_coverage_index(
                rows,
                role_names={self.stat_var: RoleNames(donor=_DONOR_ROLE, recipient=_RECIPIENT_ROLE)},
                facet_metadata={
                    fixture.facet_id: FacetMetadata(
                        provenance_id=fixture.provenance,
                        measurement_method=fixture.method,
                        unit=fixture.unit,
                    )
                    for fixtures in self.facets.values()
                    for fixture in fixtures
                },
            )
        )


def _catalogue(upstream: _Upstream) -> Catalogue:
    """Lists the upstream's StatVars as base variables, so their `FinancialFlow` amount family,
    which the bound release holds as DFO, is walked on the upstream's graph."""
    return listing_base_variables(
        upstream.stat_var_arcs_by_dcid(),
        provenances={
            _PROV_A: provenance_row(_PROV_A, BASE, source="ParitySourceA"),
            _PROV_B: provenance_row(_PROV_B, BASE, source="ParitySourceB"),
        },
    )


class _FixedBindLLM:
    """Elects the first shape, flagging the shapes `also_plausible` names as alternates."""

    def __init__(self, also_plausible: list[int]) -> None:
        self._also_plausible = also_plausible

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=self._also_plausible
        ), None


def _resolve_live(upstream: _Upstream) -> Spec:
    """The live answer for the upstream's places, its preview built by `_build_observations`."""
    graph = upstream.engine_graph()
    region = asyncio.run(
        resolve_variable(
            "amount",
            entities=upstream.entities,
            date_request=None,
            detect_query=upstream.query,
            role_query=upstream.query,
            entity_roles=door_roles_for_test_query(upstream.query, upstream.entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=upstream.engine_sdmx(),
            llm=_FixedBindLLM([1] if upstream.alternate else []),
            base_timing={},
            unit_text=upstream.unit_text,
            coverage=upstream.coverage().snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=upstream.materialise_graph(),
                materialise_sdmx=upstream.sdmx(),
                rate_cache=RateTableCache(),
            ),
        )
    )
    assert region.status == "definite", region.no_data_reason
    (spec,) = region.specs
    return spec


def _mint_and_read_frozen(upstream: _Upstream, spec: Spec) -> dict[str, object]:
    """Mint `spec`'s definition over the upstream, then read the minted version's
    observations from the observations door."""
    version_store = FakeVersionRecordDatasetStore()
    spec_store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=spec_store),
        spec_reader=LiveSpecReader(store=spec_store),
        snapshot_reader=LiveSnapshotReader(store=version_store),
        version_writer=LiveVersionRecordWriter(store=version_store),
        artifact_store=_InventoryRecordingArtifactStore(
            inventory=LiveArtifactInventory(store=version_store)
        ),
        graph=upstream.materialise_graph(),
        sdmx=upstream.sdmx(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=upstream.coverage(),
        rate_cache=RateTableCache(),
        catalogue=fake_catalogue_service(_catalogue(upstream)),
    )
    assert spec.definition is not None
    minted = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": spec.spec_id,
                "definition": spec.definition.model_dump(mode="json", exclude_none=True),
            },
            "options": {"bundle": False},
        },
    )
    assert minted.status_code == 200, minted.text
    content_version = minted.json()["content_version"]
    frozen = client.get(f"/api/datasets/{spec.spec_id}/{content_version}/observations")
    assert frozen.status_code == 200, frozen.text
    return frozen.json()


def _points(*values: float, start: int = 2019) -> tuple[tuple[str, float], ...]:
    return tuple((str(start + offset), value) for offset, value in enumerate(values))


_RECORDED_A = _FacetFixture("f-a", _PROV_A, _points(10.0, 11.0, 12.0, 13.0))
# The base Data Commons `USDollar` unit, whose ref differs from the request-derived
# `CurrencyUSD`, so a frozen read must take the traced ref.
_RECORDED_A_USDOLLAR = replace(_RECORDED_A, unit="USDollar")
_ESTIMATED_B = _FacetFixture("f-b", _PROV_B, _points(20.0, 21.0, start=2021), method="Estimated")


def _multi_facet(spec: Spec) -> None:
    """Kenya holds both facets and Ghana only the estimated one. Live serves both places from
    the estimated facet, which covers both, over Kenya's longer recorded facet."""
    observations = spec.resolved.observations
    assert observations is not None
    provenances = {
        series.entities[0].ref: series.facet["provenance"].ref for series in observations.series
    }
    assert provenances == {_KEN: _PROV_B, _GHA: _PROV_B}


def _constrained(spec: Spec) -> None:
    """The member's label composes its constraint value's label."""
    observations = spec.resolved.observations
    assert observations is not None
    (series,) = observations.series
    assert series.stat_var.label == _LABELS[_PURPOSE]


def _converted(spec: Spec) -> None:
    """The EUR request converts the series served from the convertible source A."""
    observations = spec.resolved.observations
    assert observations is not None
    (series,) = observations.series
    assert series.conversion is not None
    assert series.facet["unit"].ref == "CurrencyEUR"


def _natively_satisfied(spec: Spec) -> None:
    """The USD request names the series' own currency, so live converts nothing and keeps its
    native unit."""
    observations = spec.resolved.observations
    assert observations is not None
    (series,) = observations.series
    assert series.conversion is None
    assert series.facet["unit"].ref == "USDollar"


def _role_bearing(spec: Spec) -> None:
    """The series lists its role tuple sorted by role name, as the mint keys it
    (`series_key`): the creditor before the debtor. Its label composes the member's
    constraint value's label."""
    observations = spec.resolved.observations
    assert observations is not None
    (series,) = observations.series
    assert [(entity.role, entity.ref) for entity in series.entities] == [
        ("origin", _NOR),
        ("destination", _KEN),
    ]
    assert series.stat_var.label == _LABELS[_PURPOSE]


_RECORDED_PATH = Path(__file__).parents[1] / "fixtures" / "mint_live_parity_count_person_rural.json"


def _recorded_upstream() -> _Upstream:
    """The recorded `Count_Person_Rural` upstream at Kenya and Ghana."""
    recorded = json.loads(_RECORDED_PATH.read_text())
    return _Upstream(
        facets={
            place: tuple(
                _FacetFixture(
                    facet["facet_id"],
                    facet["provenance"],
                    tuple((date, value) for date, value in facet["points"]),
                    method=facet["method"],
                    unit=facet["unit"],
                )
                for facet in facets
            )
            for place, facets in recorded["facets"].items()
        },
        stat_var=recorded["stat_var"],
        stat_var_arcs={
            prop: {"nodes": [{"value" if prop == "name" else "dcid": v} for v in values]}
            for prop, values in recorded["arcs"].items()
        },
        labels=recorded["labels"],
    )


_WDI = "dc/base/WorldDevelopmentIndicators"


def _recorded(spec: Spec) -> None:
    """Kenya holds the World Development Indicators facet and two census facets, Ghana the
    first alone. Live serves both places in full from that facet, which covers both."""
    observations = spec.resolved.observations
    assert observations is not None
    assert not observations.truncated
    assert {
        series.entities[0].ref: (series.facet["provenance"].ref, len(series.points))
        for series in observations.series
    } == {_KEN: (_WDI, 65), _GHA: (_WDI, 65)}


# Three places on one facet, each with its own values, over a preview cap of two series and a
# role bulk of two shown series.
_THREE_PLACES = {
    place: (_FacetFixture("f-a", _PROV_A, _points(*(10.0 * scale + year for year in range(4)))),)
    for scale, place in enumerate((_KEN, _GHA, _UGA), start=1)
}
_SMALL_CAPS = ObservationSettings(obs_preview_max_pairs=2, role_bulk_series_max=2)


def _role_bulk(spec: Spec) -> None:
    """Three bound pairs exceed the cap of two, so live shows the role bulk's two series."""
    observations = spec.resolved.observations
    assert observations is not None
    assert len(observations.series) == _SMALL_CAPS.role_bulk_series_max
    assert {series.entities[0].ref for series in observations.series} == {_NOR}


def _scope_bulk(spec: Spec) -> None:
    """A query naming no place shows one latest point at each place of the default scope."""
    observations = spec.resolved.observations
    assert observations is not None
    assert {
        series.entities[0].ref: [point.value for point in series.points]
        for series in observations.series
    } == {_KEN: [13.0], _GHA: [23.0], _UGA: [33.0]}


_THREE_PLACES_USDOLLAR = {
    place: tuple(replace(facet, unit="USDollar") for facet in facets)
    for place, facets in _THREE_PLACES.items()
}


def _native_units(spec: Spec) -> None:
    """The USD request names each series' own currency, so live converts nothing and every
    series keeps its native unit."""
    observations = spec.resolved.observations
    assert observations is not None
    assert {series.facet["unit"].ref for series in observations.series} == {"USDollar"}
    assert all(series.conversion is None for series in observations.series)


def _role_bulk_natively_satisfied(spec: Spec) -> None:
    _role_bulk(spec)
    _native_units(spec)


def _scope_bulk_natively_satisfied(spec: Spec) -> None:
    _scope_bulk(spec)
    _native_units(spec)


def _too_large(spec: Spec) -> None:
    """Three named places exceed the cap of two series, so live omits the preview."""
    observations = spec.resolved.observations
    assert observations is not None
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PREVIEW_TOO_LARGE
    assert (observations.omitted.pairs, observations.omitted.cap) == (3, 2)


def _inline_truncated(spec: Spec) -> None:
    """Eight points exceed the inline cap of four, so each series keeps its latest two."""
    observations = spec.resolved.observations
    assert observations is not None
    assert observations.truncated
    assert [len(series.points) for series in observations.series] == [2, 2]


@pytest.mark.parametrize(
    ("upstream", "check_live", "caps"),
    [
        pytest.param(
            _Upstream(facets={_KEN: (_RECORDED_A, _ESTIMATED_B), _GHA: (_ESTIMATED_B,)}),
            _multi_facet,
            None,
            id="multi-facet",
        ),
        pytest.param(
            _Upstream(facets={_KEN: (_RECORDED_A,)}, constrained=True),
            _constrained,
            None,
            id="constrained-member",
        ),
        pytest.param(
            _Upstream(facets={_KEN: (_RECORDED_A, _ESTIMATED_B)}, unit_text="EUR"),
            _converted,
            None,
            id="converted",
        ),
        pytest.param(
            _Upstream(facets={_KEN: (_RECORDED_A_USDOLLAR,)}, unit_text="USD"),
            _natively_satisfied,
            None,
            id="natively-satisfied",
        ),
        pytest.param(
            _Upstream(facets={_KEN: (_RECORDED_A,)}, constrained=True, donor=_NOR),
            _role_bearing,
            None,
            id="role-bearing",
        ),
        pytest.param(
            _Upstream(facets={_KEN: (_RECORDED_A,)}, constrained=True, donor=_NOR, alternate=True),
            _role_bearing,
            None,
            id="role-bearing-collapsed",
        ),
        pytest.param(
            _Upstream(facets=_THREE_PLACES, donor=_NOR), _role_bulk, _SMALL_CAPS, id="role-bulk"
        ),
        pytest.param(
            _Upstream(facets=_THREE_PLACES, scoped=True),
            _scope_bulk,
            _SMALL_CAPS,
            id="scope-bulk",
        ),
        pytest.param(
            _Upstream(facets=_THREE_PLACES_USDOLLAR, donor=_NOR, unit_text="USD"),
            _role_bulk_natively_satisfied,
            _SMALL_CAPS,
            id="role-bulk-natively-satisfied",
        ),
        pytest.param(
            _Upstream(facets=_THREE_PLACES_USDOLLAR, scoped=True, unit_text="USD"),
            _scope_bulk_natively_satisfied,
            _SMALL_CAPS,
            id="scope-bulk-natively-satisfied",
        ),
        pytest.param(_Upstream(facets=_THREE_PLACES), _too_large, _SMALL_CAPS, id="too-large"),
        pytest.param(
            _Upstream(facets={_KEN: (_RECORDED_A, _ESTIMATED_B), _GHA: (_RECORDED_A,)}),
            _inline_truncated,
            ObservationSettings(obs_inline_max=4, obs_preview_points=2),
            id="inline-truncated",
        ),
        pytest.param(_recorded_upstream(), _recorded, None, id="recorded-count-person-rural"),
    ],
)
def test_the_frozen_observations_equal_the_live_preview(
    upstream: _Upstream,
    check_live: Callable[[Spec], None],
    caps: ObservationSettings | None,
    bind_catalogue: Callable[[Catalogue], None],
    monkeypatch: pytest.MonkeyPatch,
):
    """`check_live` first asserts the live preview shows what the case covers, so an upstream
    that stops exercising it fails rather than passing trivially."""
    if caps is not None:
        monkeypatch.setattr(search.observations, "_SETTINGS", caps)
    bind_catalogue(_catalogue(upstream))
    spec = _resolve_live(upstream)
    check_live(spec)
    live = spec.resolved.observations
    assert live is not None

    assert _mint_and_read_frozen(upstream, spec) == observations_body(live)
