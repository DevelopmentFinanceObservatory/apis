"""A small world of place sets and the three doors that read it.

The places: the continent Africa with five member countries, one of them (Saint Helena)
holding no data; the continent Asia with three; the subregion Sub-Saharan Africa with
three of Africa's; the European Union, a resolver group whose node the graph holds, with
three members; the African Union, a resolver group with no node, with four. Africa's node
also contains a subregion and a city directly, so its single-hop containment differs from
its expansion to countries.

The series: `WDI` holds observations at countries only. `WHO` holds them at the `africa`
and `EuropeanUnion` nodes in one span of years and at countries in a later span, and none
at `asia`. `DEBT` is a flow series with a recipient role and `AID` one with a provider and
a recipient role. Both roles' ranges admit `Country` and a subregion's type, the provider
role's also the European Union's type, and neither a continent's. `AID` holds rows from
France and from the `EuropeanUnion` node to African countries, from France to the
`SubSaharanAfrica` node, from that node to Ghana, from Kenya to that node and from Ghana
to Kenya, and none to `africa`. `UN_SERIES` is a series of
the UN System Data Commons that holds no observation anywhere.

`Doors` runs the parsed door, the `dataset_resubmit` door and the Datasets mint over one
set of doubles built from those tables, and keeps each door's observation reads. The
functions below it read a Spec off the wire, compute what the tables imply for it, and
check one against the other.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from dc_sdmx import SdmxClient
from entity_resolution import Admission
from materialise.graph import Facet as MaterialiseFacet
from origin_catalogue import note_un_blocked_places
from search.engine.bind import _UniversalBindOutput
from search.engine.core import resolve_async
from search.engine.departures import spec_departure_facts
from search.engine.errors import GraphInfraError
from search.engine.extract import ExtractedEntity, ExtractedOperation, Extraction
from search.engine.spec_id import mint_identity
from search.models import (
    BaseIdentity,
    CoverageExact,
    DatasetResubmitInput,
    EntityGroup,
    Operation,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    Resolution,
    ResolveOptions,
    ResolveRequest,
    ResolveResponse,
    ResubmitIdentityEntry,
    ResubmitParsedEntry,
    Spec,
)
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.datasets_api._app_factories import make_client
from tests.datasets_api._artifact_doubles import FakeArtifactStore
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.datasets_api._snapshot_doubles import FakeSnapshotReader
from tests.datasets_api.conftest import FakeDatasetStore
from tests.datasets_api.test_pin_door import FakeArtifactReferenceWriter
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import run_pinned
from tests.fixtures import FakeGraph, make_resolver

# Places.
AFRICA, ASIA, SUB_SAHARAN, EU = "africa", "asia", "SubSaharanAfrica", "EuropeanUnion"
GHA, KEN, NGA, TZA, SHN = (
    "country/GHA",
    "country/KEN",
    "country/NGA",
    "country/TZA",
    "country/SHN",
)
DEU, FRA, ITA = "country/DEU", "country/FRA", "country/ITA"
IDN, IND, JPN = "country/IDN", "country/IND", "country/JPN"
_CITY = "PlaceSetCity"

# What each set expands to: a region's `Country` children, and for the European Union the
# members of the resolver's group that the graph holds.
EXPANSION: Mapping[str, tuple[str, ...]] = {
    AFRICA: (GHA, KEN, NGA, SHN, TZA),
    ASIA: (IDN, IND, JPN),
    SUB_SAHARAN: (GHA, KEN, TZA),
    EU: (DEU, FRA, ITA),
}
# The members of the resolver's African Union group that the graph holds. The graph holds
# no node for the group.
AFRICAN_UNION = (GHA, KEN, NGA, TZA)
NODE_TYPE: Mapping[str, str] = {
    AFRICA: "Continent",
    ASIA: "Continent",
    SUB_SAHARAN: "UNGeoRegion",
    EU: "ContinentalUnion",
}
LABEL: Mapping[str, str] = {
    AFRICA: "Africa",
    ASIA: "Asia",
    SUB_SAHARAN: "Sub-Saharan Africa",
    EU: "European Union",
    GHA: "Ghana",
    KEN: "Kenya",
    NGA: "Nigeria",
    TZA: "Tanzania",
    SHN: "Saint Helena",
    DEU: "Germany",
    FRA: "France",
    ITA: "Italy",
    IDN: "Indonesia",
    IND: "India",
    JPN: "Japan",
}
COUNTRIES = tuple(dcid for dcid in LABEL if dcid.startswith("country/"))

# Series, and the variable text that recalls each.
WDI = "Amount_PlaceSetHealthSpending"
WHO = "Amount_PlaceSetGovernmentHealthSpending"
DEBT = "Amount_PlaceSetExternalDebt"
AID = "Amount_PlaceSetAid"
UN_SERIES = "undata/placeset/HealthIndex"
VARIABLE_TEXT: Mapping[str, str] = {
    WDI: "health spending",
    WHO: "government health spending",
    DEBT: "external debt stocks",
    AID: "aid",
    UN_SERIES: "health index",
}
_POPULATION_TYPE: Mapping[str, str] = {
    WDI: "PlaceSetHealthSpending",
    WHO: "PlaceSetGovernmentHealthSpending",
    DEBT: "PlaceSetExternalDebt",
    AID: "PlaceSetAid",
    UN_SERIES: "PlaceSetHealthIndex",
}

PROVIDER, RECIPIENT = "financingProvider", "financingRecipient"
ROLE_NAMES: Mapping[str, RoleNames] = {
    DEBT: RoleNames(donor=None, recipient=RECIPIENT),
    AID: RoleNames(donor=PROVIDER, recipient=RECIPIENT),
}
# The types each flow role's range admits. Both admit a subregion, the provider role the
# European Union's type too, and neither a continent.
ROLE_RANGE: Mapping[str, tuple[str, ...]] = {
    PROVIDER: ("Country", "ContinentalUnion", "UNGeoRegion"),
    RECIPIENT: ("Country", "UNGeoRegion"),
}
_AXIS = {"observationAbout": "observationAbout", PROVIDER: "origin", RECIPIENT: "destination"}

# A node's span of years ends before a country's begins, so a window can hold one and not
# the other, and `NO_YEAR` lies in neither.
NODE_YEARS = (2004, 2005, 2006)
COUNTRY_YEARS = (2017, 2018, 2019, 2020)
NODE_YEAR, COUNTRY_YEAR, NO_YEAR = 2005, 2018, 1990

# Where each subject series holds observations, and the years it holds there.
SUBJECT_DATA: Mapping[str, Mapping[str, tuple[int, ...]]] = {
    WDI: dict.fromkeys((GHA, KEN, TZA, DEU, FRA, ITA, IDN, IND, JPN), COUNTRY_YEARS),
    WHO: {
        AFRICA: NODE_YEARS,
        EU: NODE_YEARS,
        **dict.fromkeys((GHA, KEN, DEU, FRA, IDN, JPN), COUNTRY_YEARS),
    },
    UN_SERIES: {},
}
# The `(provider, recipient)` rows each flow series holds, each in every year of
# `COUNTRY_YEARS`. `DEBT` declares no provider.
FLOW_DATA: Mapping[str, tuple[tuple[str, str], ...]] = {
    DEBT: (("", GHA), ("", KEN), ("", TZA)),
    AID: (
        (FRA, GHA),
        (FRA, KEN),
        (EU, KEN),
        (EU, TZA),
        (DEU, NGA),
        (FRA, SUB_SAHARAN),
        (SUB_SAHARAN, GHA),
        (KEN, SUB_SAHARAN),
        (GHA, KEN),
    ),
}

_SOURCE = "dc/base/PlaceSetFixture"
_FACET_ID = "f0"
_BASE_URL = "https://example.org/core/api"
_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"
_CSV_PREFIX = "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured"
_CSV_TRAILING = (
    "unit,measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId"
)


def _statvar(stat_var: str, label: str, roles: tuple[str, ...] = ()) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": _POPULATION_TYPE[stat_var]}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


def _place(dcid: str) -> dict:
    """A place node, contained in each set that expands to it. The subregion and the city
    sit in Africa's node directly."""
    parents = [parent for parent, members in EXPANSION.items() if dcid in members]
    if dcid in (SUB_SAHARAN, _CITY):
        parents = [AFRICA]
    return {
        "label": LABEL.get(dcid, dcid),
        "type": "City" if dcid == _CITY else NODE_TYPE.get(dcid, "Country"),
        "arcs": {"containedInPlace": {"nodes": [{"dcid": parent} for parent in parents]}},
    }


_NODES: dict = {
    **{dcid: _place(dcid) for dcid in (*LABEL, _CITY)},
    WDI: _statvar(WDI, "Health spending"),
    WHO: _statvar(WHO, "Government health spending"),
    DEBT: _statvar(DEBT, "External debt stocks", (RECIPIENT,)),
    AID: _statvar(AID, "Aid disbursements", (PROVIDER, RECIPIENT)),
    UN_SERIES: _statvar(UN_SERIES, "Health index"),
    **{
        role: {
            "label": role,
            "type": "Property",
            "arcs": {"rangeIncludes": {"nodes": [{"dcid": admitted} for admitted in types]}},
        }
        for role, types in ROLE_RANGE.items()
    },
}
_CHILDREN = {
    **{node: {"Country": list(EXPANSION[node])} for node in (AFRICA, ASIA, SUB_SAHARAN)},
    "Earth": {"Country": list(COUNTRIES)},
}
# A text that names Africa and is not the graph's label for it.
AFRICA_ALIAS = "Afrique"
_PLACES = {
    LABEL[node]: {"dcid": node, "type_of": NODE_TYPE[node]} for node in (AFRICA, ASIA, SUB_SAHARAN)
} | {AFRICA_ALIAS: {"dcid": AFRICA, "type_of": NODE_TYPE[AFRICA]}}
# A second text of a request that names Sub-Saharan Africa: the resolver reads it as the
# first.
SUB_SAHARAN_ALIAS = LABEL[SUB_SAHARAN].lower()
_OBS = {
    f"{stat_var}|{place}": [
        {
            "facetId": _FACET_ID,
            "provenanceId": _SOURCE,
            "earliestDate": str(min(years)),
            "latestDate": str(max(years)),
            "obsCount": len(years),
            "dates": [str(year) for year in years],
            "values": [float(year) for year in years],
        }
    ]
    for stat_var, held in SUBJECT_DATA.items()
    for place, years in held.items()
}


@dataclass(frozen=True)
class Read:
    """One observation read a door made: the client method it went through, the series it
    named and the dcids it named on each role."""

    door: str
    via: str
    stat_vars: tuple[str, ...]
    entities: Mapping[str, tuple[str, ...]]

    @property
    def dcids(self) -> frozenset[str]:
        return frozenset(dcid for dcids in self.entities.values() for dcid in dcids)


# The reads that fetch observations. A door's other reads are facet and presence reads.
DATA_READS = frozenset({"materialise.sdmx", "sdmx.role_series"})


class _Recorder:
    """Files each read under the door running when it is made, except a read a double makes
    to answer another (`outermost`)."""

    def __init__(self) -> None:
        self.door = "setup"
        self.reads: list[Read] = []
        self._lock = threading.Lock()
        self._nested = threading.local()

    def add(
        self, via: str, stat_vars: Iterable[str], entities: Mapping[str, Iterable[str]]
    ) -> None:
        if getattr(self._nested, "depth", 0):
            return
        read = Read(
            self.door,
            via,
            tuple(stat_vars),
            {role: tuple(dcids) for role, dcids in entities.items()},
        )
        with self._lock:
            self.reads.append(read)

    @contextmanager
    def outermost(
        self, via: str, stat_vars: Iterable[str], entities: Mapping[str, Iterable[str]]
    ) -> Iterator[None]:
        """File one read, and none of those the double makes while it answers it."""
        self.add(via, stat_vars, entities)
        self._nested.depth = getattr(self._nested, "depth", 0) + 1
        try:
            yield
        finally:
            self._nested.depth -= 1


class _Graph(FakeGraph):
    """The world's graph. Records each facet and presence read. With `lose_node_read` set,
    the next batched facet read naming a set's node fails as a lost graph request does."""

    lose_node_read = False

    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(
            nodes=_NODES,
            obs=_OBS,
            detect={
                text: {"svs": [stat_var], "cosine_scores": [0.9]}
                for stat_var, text in VARIABLE_TEXT.items()
                if stat_var != UN_SERIES
            },
            resolve={LABEL[dcid]: dcid for dcid in COUNTRIES},
            places=_PLACES,
            children=_CHILDREN,
        )
        self._recorder = recorder

    def recall(self, text: str, stat_vars: Sequence[str]) -> None:
        """Make `text` recall `stat_vars`, best first."""
        scores = [0.9 - 0.01 * rank for rank in range(len(stat_vars))]
        self._detect[text] = {"svs": list(stat_vars), "cosine_scores": scores}

    def observation_facets(self, *, stat_var, entity, needs_dates=False):
        self._recorder.add("graph.facets", [stat_var], {"observationAbout": [entity]})
        return super().observation_facets(stat_var=stat_var, entity=entity, needs_dates=needs_dates)

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **kwargs):
        if self.lose_node_read and not EXPANSION.keys().isdisjoint(entities):
            self.lose_node_read = False
            raise GraphInfraError("the facet read was lost")
        with self._recorder.outermost("graph.facets", stat_vars, {"observationAbout": entities}):
            return super().observation_facets_batch(stat_vars, entities, needs_dates, **kwargs)

    def entities_with_data(self, variable_dcids, entity_dcids, *, max_bytes=None):
        with self._recorder.outermost(
            "graph.presence", variable_dcids, {"observationAbout": entity_dcids}
        ):
            return super().entities_with_data(variable_dcids, entity_dcids, max_bytes=max_bytes)

    def entities_with_data_by_variable(self, variable_dcids, entity_dcids, *, deadline=None):
        with self._recorder.outermost(
            "graph.presence", variable_dcids, {"observationAbout": entity_dcids}
        ):
            return super().entities_with_data_by_variable(
                variable_dcids, entity_dcids, deadline=deadline
            )


class _MintGraph(FakeMintGraphReader):
    """The graph the mint reads, and Search's observation reads with it. Records each facet
    and presence read.

    Two faults can be switched on. A place in `un_refused` is one the UN System Data Commons
    front door refuses, and a facet read of a UN series naming it reports the refusal as
    the live client does. Provenances in `withheld` hold data the origin registry does not
    admit, at every place a re-read asks about."""

    def __init__(self, recorder: _Recorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._recorder = recorder
        self.un_refused: frozenset[str] = frozenset()
        self.withheld: tuple[str, ...] = ()

    def _note_refused(self, stat_vars: Iterable[str], entities: Iterable[str]) -> None:
        if any(stat_var.startswith("undata/") for stat_var in stat_vars):
            note_un_blocked_places(e for e in entities if e in self.un_refused)

    def variable_facets_batch(self, *, stat_var, entities, deadline=None):
        self._recorder.add("materialise.facets", [stat_var], {"observationAbout": entities})
        self._note_refused([stat_var], entities)
        return super().variable_facets_batch(
            stat_var=stat_var, entities=entities, deadline=deadline
        )

    def observation_facets_batch_many(self, *, stat_vars, entities, deadline=None):
        self._recorder.add("materialise.facets", stat_vars, {"observationAbout": entities})
        self._note_refused(stat_vars, entities)
        return super().observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )

    def entities_with_data_by_variable(self, variable_dcids, entity_dcids, *, deadline=None):
        self._recorder.add(
            "materialise.presence", variable_dcids, {"observationAbout": entity_dcids}
        )
        return super().entities_with_data_by_variable(
            variable_dcids, entity_dcids, deadline=deadline
        )

    def withheld_at(self, variables, entities, *, deadline=None, max_bytes=None):
        return dict.fromkeys(entities, self.withheld) if self.withheld else {}


class _Sdmx(FakeSdmxClient):
    """Search's SDMX client. Records each series read."""

    def __init__(self, recorder: _Recorder, graph: _Graph) -> None:
        super().__init__(
            observations={}, availability={}, scope_availability=graph.scope_availability
        )
        self._recorder = recorder

    def role_series(self, *, variables, entities, **kwargs):
        self._recorder.add("sdmx.role_series", variables, entities)
        return super().role_series(variables=variables, entities=entities, **kwargs)


def _rows(stat_var: str) -> list[tuple[dict[str, str], int]]:
    """Every observation `stat_var` holds, as its role cells and its year."""
    if stat_var in SUBJECT_DATA:
        return [
            ({"observationAbout": place}, year)
            for place, years in SUBJECT_DATA[stat_var].items()
            for year in years
        ]
    names = ROLE_NAMES[stat_var]
    rows: list[tuple[dict[str, str], int]] = []
    for provider, recipient in FLOW_DATA[stat_var]:
        cells = {RECIPIENT: recipient}
        if names.donor:
            cells[names.donor] = provider
        rows.extend((cells, year) for year in COUNTRY_YEARS)
    return rows


def _data_transport(recorder: _Recorder) -> httpx.MockTransport:
    """Serves each data read the rows its filters name and records it. A role the read
    leaves unfiltered is open."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        stat_vars = [
            name.removeprefix("dcid:") for name in params["c[variableMeasured]"].split(",")
        ]
        filters = {
            role: params[f"c[{role}]"].split(",") for role in _AXIS if f"c[{role}]" in params
        }
        recorder.add("materialise.sdmx", stat_vars, filters)
        rows = [
            (stat_var, cells, year)
            for stat_var in stat_vars
            for cells, year in _rows(stat_var)
            if all(cells.get(role) in wanted for role, wanted in filters.items())
        ]
        roles = list(dict.fromkeys(role for _sv, cells, _year in rows for role in cells))
        lines = [",".join((_CSV_PREFIX, *(roles or filters), _CSV_TRAILING))]
        lines.extend(
            ",".join(
                (
                    "dataflow",
                    "DC:DF_OBS(1.0.0)",
                    "I",
                    f"dcid:{stat_var}",
                    *(cells[role] for role in roles),
                    *("NotApplicable",) * 3,
                    _SOURCE,
                    str(year),
                    str(float(year)),
                    "",
                    _FACET_ID,
                )
            )
            for stat_var, cells, year in rows
        )
        return httpx.Response(
            200, text="\r\n".join(lines) + "\r\n", headers={"content-type": _CSV_CONTENT_TYPE}
        )

    return httpx.MockTransport(handler)


def _materialise_facets() -> dict[tuple[str, str], list[MaterialiseFacet]]:
    return {
        (stat_var, place): [
            MaterialiseFacet(
                facet_id=_FACET_ID,
                provenance_id=_SOURCE,
                obs_count=len(years),
                earliest_date=str(min(years)),
                latest_date=str(max(years)),
            )
        ]
        for stat_var, held in SUBJECT_DATA.items()
        for place, years in held.items()
    }


class _Bind:
    """Elects the first shape and binds nothing. It names the shapes at `also_plausible`
    as other readings of the request."""

    def __init__(self, also_plausible: Sequence[int] = ()) -> None:
        self._also_plausible = list(also_plausible) or None

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=self._also_plausible, ask=None
        ), None


class _Extract(_Bind):
    """Reads every query as one extraction, and binds as `_Bind` does."""

    def __init__(self, extraction: Extraction) -> None:
        super().__init__()
        self._extraction = extraction

    async def generate_structured(self, *, prompt, system, schema):
        if schema is Extraction:
            return self._extraction, None
        return await super().generate_structured(prompt=prompt, system=system, schema=schema)


def _part(
    stat_var: str,
    entities: Sequence[str | ParsedEntity],
    *,
    year: int | None = None,
    operations: Sequence[Operation] = (),
    text: str | None = None,
) -> ParsedPart:
    """One parsed part asking for `stat_var` at `entities`, under `text` when one is given."""
    return ParsedPart(
        variable_text=VARIABLE_TEXT[stat_var] if text is None else text,
        entities=[
            entity if isinstance(entity, ParsedEntity) else ParsedEntity(text=entity)
            for entity in entities
        ],
        time_text=None if year is None else str(year),
        operations=list(operations),
    )


class Doors:
    """The three doors over one set of doubles, with every observation read each made."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._recorder = _Recorder()
        self._graph = self.graph = _Graph(self._recorder)
        self._sdmx = _Sdmx(self._recorder, self._graph)
        index = patch_coverage(
            monkeypatch,
            [
                (stat_var, _FACET_ID, provider, recipient, list(COUNTRY_YEARS))
                for stat_var, rows in FLOW_DATA.items()
                for provider, recipient in rows
            ],
            role_names=dict(ROLE_NAMES),
            facet_metadata={_FACET_ID: FacetMetadata(provenance_id=_SOURCE)},
        )
        self._coverage = FakeCoverageService(index)
        self.mint_graph = _MintGraph(
            self._recorder,
            facets=_materialise_facets(),
            labels={**LABEL, _SOURCE: "Place set fixture"},
            arcs=self._graph.member_arcs(),
            child_places={
                (parent, child_type): children
                for parent, by_type in _CHILDREN.items()
                for child_type, children in by_type.items()
            },
        )
        self._materialise_sdmx = SdmxClient(
            base_url=_BASE_URL, transport=_data_transport(self._recorder)
        )

    def reads(self, door: str) -> list[Read]:
        """The observation reads `door` made: `first_answer`, `resubmit` or `mint`."""
        return [read for read in self._recorder.reads if read.door == door]

    def _resolve(
        self, door: str, request: ResolveRequest, *, llm: _Bind | None = None
    ) -> ResolveResponse:
        self._recorder.door = door
        try:
            return run_pinned(
                resolve_async(
                    request,
                    graph=self._graph,
                    resolver=make_resolver(self._graph),
                    admission=Admission(self._graph),
                    llm=llm or _Bind(),
                    sdmx=self._sdmx,
                    materialise_graph=self.mint_graph,
                    materialise_sdmx=self._materialise_sdmx,
                )
            )
        finally:
            self._recorder.door = "setup"

    def _ask(
        self,
        door: str,
        stat_var: str,
        entities: Sequence[str | ParsedEntity],
        year: int | None,
        operations: Sequence[Operation],
        *,
        text: str | None = None,
        also_plausible: Sequence[int] = (),
    ) -> ResolveResponse:
        part = _part(stat_var, entities, year=year, operations=operations, text=text)
        return self._resolve(
            door,
            ResolveRequest(
                input=ParsedInput(parts=[part]),
                response_options=ResolveOptions(include_observations=True),
            ),
            llm=_Bind(also_plausible),
        )

    def parts(self, *asks: tuple[str, Sequence[str | ParsedEntity]]) -> ResolveResponse:
        """The parsed door's answer to one request of several parts, each a series and
        the entities it is asked at."""
        return self._resolve(
            "first_answer",
            ResolveRequest(
                input=ParsedInput(parts=[_part(stat_var, entities) for stat_var, entities in asks])
            ),
        )

    def resubmit_part(self, stat_var: str, *entities: str | ParsedEntity) -> ResolveResponse:
        """The `dataset_resubmit` door's answer to a parsed entry asking for `stat_var` at
        `entities`."""
        entry = ResubmitParsedEntry(part=_part(stat_var, entities))
        return self._resolve(
            "resubmit", ResolveRequest(input=DatasetResubmitInput(entries=[entry]))
        )

    def first_answer(
        self,
        stat_var: str,
        *entities: str | ParsedEntity,
        year: int | None = None,
        operations: Sequence[Operation] = (),
    ) -> ResolveResponse:
        """The parsed door's answer to one part asking for `stat_var` at `entities`, in
        `year` when one is given, with its observations."""
        return self._ask("first_answer", stat_var, entities, year, operations)

    def readings(
        self,
        stat_vars: Sequence[str],
        *entities: str | ParsedEntity,
        year: int | None = None,
        operations: Sequence[Operation] = (),
    ) -> ResolveResponse:
        """The parsed door's answer to one part whose text recalls each of `stat_vars`, in
        that order, where bind elects the first and names the rest as other readings."""
        text = " or ".join(VARIABLE_TEXT[stat_var] for stat_var in stat_vars)
        self._graph.recall(text, stat_vars)
        return self._ask(
            "first_answer",
            stat_vars[0],
            entities,
            year,
            operations,
            text=text,
            also_plausible=range(1, len(stat_vars)),
        )

    def raw_text(
        self,
        query: str,
        stat_var: str,
        *entities: ExtractedEntity,
        operations: Sequence[ExtractedOperation] = (),
    ) -> ResolveResponse:
        """The raw-text door's answer to `query`, whose extraction is `stat_var`'s measure
        at `entities` with `operations`, with its observations."""
        extraction = Extraction(
            kind="resolution",
            variables=[VARIABLE_TEXT[stat_var]],
            entities=list(entities),
            operations=list(operations),
        )
        return self._resolve(
            "first_answer",
            ResolveRequest(
                input=RawTextInput(query=query),
                response_options=ResolveOptions(include_observations=True),
            ),
            llm=_Extract(extraction),
        )

    def identity(
        self, stat_var: str, entities: Mapping[str, Sequence[str]], *, year: int | None = None
    ) -> BaseIdentity:
        """The identity of `stat_var` binding `entities` by role, as a caller would post
        it: the identity Search serves for the series at one country, with its entities
        replaced. The reads that builds it are filed under no door."""
        named: tuple[str | ParsedEntity, ...] = ("Kenya",)
        if stat_var == AID:
            named = (origin("France"), destination("Kenya"))
        # Search recalls no UN series here, so its identity is `WDI`'s on its own shape.
        recalled = WDI if stat_var == UN_SERIES else stat_var
        document = the_spec(self._ask("setup", recalled, named, year, ())).definition.identity
        projection = document.model_dump(mode="json", exclude_none=True)
        projection["shape"]["population_type"] = _POPULATION_TYPE[stat_var]
        projection["entities"] = {role: list(dcids) for role, dcids in entities.items()}
        return mint_identity(projection)[1]

    def post(self, part: ParsedPart) -> ResolveResponse:
        """The parsed door's answer to `part` as a client posts it, with its observations."""
        return self._resolve(
            "first_answer",
            ResolveRequest(
                input=ParsedInput(parts=[part]),
                response_options=ResolveOptions(include_observations=True),
            ),
        )

    def resubmit(
        self,
        stat_var: str,
        identity: BaseIdentity,
        entity_groups: Mapping[str, EntityGroup] | None = None,
    ) -> ResolveResponse:
        """The `dataset_resubmit` door's answer to `identity` posted with `entity_groups`."""
        entry = ResubmitIdentityEntry.model_validate(
            {
                "variable_text": VARIABLE_TEXT[stat_var],
                "identity": identity,
                "entity_groups": None if entity_groups is None else dict(entity_groups),
            }
        )
        return self._resolve(
            "resubmit",
            ResolveRequest(
                input=DatasetResubmitInput(entries=[entry]),
                response_options=ResolveOptions(include_observations=True),
            ),
        )

    def mint(self, identity: BaseIdentity) -> httpx.Response:
        """The Datasets mint's response to `identity` posted as a definition."""
        store = FakeDatasetStore()
        client = make_client(
            registry=LiveSpecRegistry(store=store),
            spec_reader=LiveSpecReader(store=store),
            graph=self.mint_graph,
            sdmx=self._materialise_sdmx,
            artifact_store=FakeArtifactStore(),
            snapshot_reader=FakeSnapshotReader(),
            reference_writer=FakeArtifactReferenceWriter(),
            coverage=self._coverage,
        )
        projection = identity.model_dump(mode="json", exclude_none=True)
        spec_id = mint_identity(projection)[0]
        definition = {"schema_version": 1, "spec_id": spec_id, "identity": projection}
        self._recorder.door = "mint"
        try:
            return client.post(
                "/api/datasets",
                json={
                    "schema_version": 1,
                    "input": {"kind": "spec_id", "spec_id": spec_id, "definition": definition},
                    "options": {"bundle": False},
                },
            )
        finally:
            self._recorder.door = "setup"


# Reading an answer off the wire.


def origin(text: str, *, members: bool = False) -> ParsedEntity:
    """`text` named as a flow's origin, asked for as its members when `members`."""
    return ParsedEntity(text=text, role_text="from", members=members)


def destination(text: str, *, members: bool = False) -> ParsedEntity:
    """`text` named as a flow's destination, asked for as its members when `members`."""
    return ParsedEntity(text=text, role_text="to", members=members)


def members(text: str) -> ParsedEntity:
    """`text` asked for as the members of the set it names."""
    return ParsedEntity(text=text, members=True)


def pinned(dcid: str, text: str | None = None, *, members: bool = False) -> ParsedEntity:
    """The node `dcid` pinned under `text`, its label by default, asked for as its members
    when `members`."""
    return ParsedEntity(text=text or LABEL[dcid], dcid=dcid, members=members)


def the_spec(response: ResolveResponse) -> Spec:
    """The Spec of a response that answers its one part definitely."""
    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert part.status == "definite", (part.status, getattr(part, "no_data", None))
    return part.spec


def the_specs(response: ResolveResponse) -> list[Spec]:
    """The Spec of each part of a response that answers every part definitely."""
    assert isinstance(response, Resolution)
    assert [part.status for part in response.parts] == ["definite"] * len(response.parts)
    return [part.spec for part in response.parts]


def the_readings(response: ResolveResponse) -> dict[str, Spec]:
    """The Specs of a response that offers its one part several readings, by the series
    each reads."""
    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert part.status == "candidates", part.status
    series = {population: stat_var for stat_var, population in _POPULATION_TYPE.items()}
    return {
        series[spec.definition.identity.shape.population_type]: spec
        for spec in part.candidates.specs
    }


def no_data_reason(response: ResolveResponse) -> str:
    """The reason of a response that answers its one part with no data."""
    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert part.status == "no_data", part.status
    return part.no_data.reason


def no_data_terms(response: ResolveResponse) -> tuple[str, list[str] | None]:
    """The reason and entity terms of a response that answers its one part with no
    data."""
    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert part.status == "no_data", part.status
    return part.no_data.reason, part.no_data.entity_terms


# One `entity_unavailable` departure: the places it names one by one, and for each set
# whose members it counts, the set's label, its members without values and its size.
type Unavailable = tuple[tuple[str, ...], tuple[tuple[str, int, int], ...]]


def unavailable(spec: Spec) -> list[Unavailable]:
    """The `entity_unavailable` departures of `spec`, each as the places it names one by
    one and the count it states for each set. A role carrying a group names no place one
    by one and counts the group."""
    facts = [
        fact
        for fact in spec_departure_facts(spec, operations=())
        if fact.code == "entity_unavailable"
    ]
    return [
        ((), ((fact.group, len(fact.entities), fact.group_size),))
        if fact.group is not None and fact.group_size is not None
        else (
            fact.entities,
            tuple((members.label, len(members.entities), members.size) for members in fact.sets),
        )
        for fact in facts
    ]


def departure_asks(response: ResolveResponse, code: str) -> list[str | None]:
    """What each departure of `code` on a response's one part says the request asked
    for."""
    assert isinstance(response, Resolution)
    (part,) = response.parts
    return [d.asked for d in part.departures if d.code == code]


def bound(spec: Spec) -> dict[str, list[str]]:
    """The dcids `spec`'s identity binds, by role, sorted."""
    return {role: sorted(dcids) for role, dcids in spec.definition.identity.entities.items()}


def bound_dcids(identity: BaseIdentity) -> set[str]:
    return {dcid for dcids in identity.entities.values() for dcid in dcids}


def preview(spec: Spec) -> dict[str, set[str]]:
    """The dcids `spec`'s observation series are about, by role."""
    observations = spec.resolved.observations
    assert observations is not None
    served: dict[str, set[str]] = {}
    for series in observations.series:
        for entity in series.entities:
            served.setdefault(entity.role, set()).add(entity.ref)
    return served


def stated_observations(spec: Spec) -> int:
    """The observation count `spec`'s coverage states."""
    coverage = spec.resolved.coverage
    assert isinstance(coverage, CoverageExact), coverage
    return coverage.observation_count


def group(spec: Spec, role: str = "observationAbout") -> dict[str, Any] | None:
    """The entity group `spec` serves on `role`, as the wire carries it."""
    groups = spec.resolved.entity_groups or {}
    return groups[role].model_dump(mode="json") if role in groups else None


def warning_codes(manifest: Mapping[str, Any]) -> set[str]:
    return {warning["code"] for warning in manifest["warnings"]}


def served_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    """How many dcids a minted manifest's rows hold on each role that holds any."""
    return {
        _AXIS[dimension["label"]]: dimension["count"]
        for dimension in manifest["coverage"]["dimensions"]
        if dimension["label"] in _AXIS and dimension["count"]
    }


# What the tables imply.


def places_with_data(stat_var: str, places: Iterable[str], year: int | None = None) -> list[str]:
    """Those of `places` where the subject series `stat_var` holds an observation, in
    `year` when one is given."""
    held = SUBJECT_DATA[stat_var]
    return [p for p in places if p in held and (year is None or year in held[p])]


def subject_reading(stat_var: str, node: str, year: int | None = None) -> list[str]:
    """What the set at `node` settles to as the subject of `stat_var`: its node where the
    series holds an observation there, in `year` when one is given, else its members. The
    subject role's range admits every place type."""
    return [node] if places_with_data(stat_var, [node], year) else sorted(EXPANSION[node])


def members_reading(*nodes: str) -> list[str]:
    """What the sets at `nodes` settle to on one role when each is asked for as its
    members: every member of any of them."""
    return sorted({member for node in nodes for member in EXPANSION[node]})


def flow_reading(stat_var: str, node: str, role: str, across: Iterable[str] = ("",)) -> list[str]:
    """What the set at `node` settles to on the flow role `role` of `stat_var`, where the
    other role holds `across`: its node when the role's range admits the node's type and
    the series holds a row from or to it across those dcids, else its members."""
    pairs = {(node, other) if role == PROVIDER else (other, node) for other in across}
    admitted = NODE_TYPE[node] in ROLE_RANGE[role]
    return [node] if admitted and pairs & set(FLOW_DATA[stat_var]) else sorted(EXPANSION[node])


def _flow_rows(stat_var: str, entities: Mapping[str, Sequence[str]]) -> list[tuple[str, str]]:
    """The rows of the flow series `stat_var` that `entities` binds. A role it leaves out
    is open."""
    return [
        (provider, recipient)
        for provider, recipient in FLOW_DATA[stat_var]
        if provider in entities.get("origin", [provider])
        and recipient in entities.get("destination", [recipient])
    ]


def held_observations(
    stat_var: str, entities: Mapping[str, Sequence[str]], year: int | None = None
) -> int:
    """The observations the world holds of `stat_var` at exactly the dcids `entities`
    binds, in `year` when one is given."""
    if stat_var in SUBJECT_DATA:
        held = SUBJECT_DATA[stat_var]
        places = places_with_data(stat_var, entities["observationAbout"], year)
        return sum(1 if year is not None else len(held[place]) for place in places)
    in_year = len(COUNTRY_YEARS) if year is None else int(year in COUNTRY_YEARS)
    return len(_flow_rows(stat_var, entities)) * in_year


def held_at(
    stat_var: str, entities: Mapping[str, Sequence[str]], year: int | None = None
) -> dict[str, set[str]]:
    """Those of the dcids `entities` binds, by role, at which the world holds an
    observation of `stat_var`, in `year` when one is given."""
    served: dict[str, set[str]] = {}
    if stat_var in SUBJECT_DATA:
        if places := places_with_data(stat_var, entities["observationAbout"], year):
            served["observationAbout"] = set(places)
        return served
    if year is not None and year not in COUNTRY_YEARS:
        return served
    for provider, recipient in _flow_rows(stat_var, entities):
        if provider:
            served.setdefault("origin", set()).add(provider)
        served.setdefault("destination", set()).add(recipient)
    return served


# Checking one against the other.


def compares_a_node_with_its_members(read: Read) -> bool:
    """Whether `read` is a facet read naming a set's node beside every member of the set.
    A Spec whose read decides a set reads the node and the members in one batch, and so
    does the sample that tells which shapes hold data before one is elected. Neither is a
    read of a settled Spec."""
    return read.via not in DATA_READS and any(
        {node, *members} <= read.dcids for node, members in EXPANSION.items()
    )


def check_reading(doors: Doors, stat_var: str, spec: Spec, year: int | None = None) -> None:
    """`spec`, served or offered by the first answer of `doors`, states as many observations
    as the world holds at the dcids its identity binds. Every read of its series names those
    dcids alone: its observation fetches, and its facet and presence reads but those that
    compare a set's node with its members (`compares_a_node_with_its_members`)."""
    assert stated_observations(spec) == held_observations(stat_var, bound(spec), year)
    for read in doors.reads("first_answer"):
        if stat_var in read.stat_vars and not compares_a_node_with_its_members(read):
            assert read.dcids <= bound_dcids(spec.definition.identity), read


def check_first_answer(doors: Doors, stat_var: str, spec: Spec, year: int | None = None) -> None:
    """`spec`, served by the first answer of `doors`, is read at the dcids its identity
    binds (`check_reading`), and its observation series are about those of them the world
    holds an observation at."""
    check_reading(doors, stat_var, spec, year)
    assert preview(spec) == held_at(stat_var, bound(spec), year)


def check_posted(
    doors: Doors,
    stat_var: str,
    identity: BaseIdentity,
    entity_groups: Mapping[str, EntityGroup] | None = None,
    year: int | None = None,
) -> Spec:
    """A resubmit of `identity` keeps its `spec_id`, states the observations the world
    holds at the dcids it binds, and makes no observation read of another dcid. Returns the
    Spec it answers with."""
    again = the_spec(doors.resubmit(stat_var, identity, entity_groups))
    entities = bound(again)
    assert again.spec_id == mint_identity(identity.model_dump(mode="json", exclude_none=True))[0]
    assert entities == {role: sorted(dcids) for role, dcids in identity.entities.items()}
    assert preview(again) == held_at(stat_var, entities, year)
    assert stated_observations(again) == held_observations(stat_var, entities, year)
    for read in doors.reads("resubmit"):
        assert read.dcids <= bound_dcids(identity), read
    return again


def check_minted(
    doors: Doors, stat_var: str, identity: BaseIdentity, year: int | None = None
) -> dict[str, Any]:
    """A mint of `identity` writes the observations the world holds at the dcids it binds,
    over the dcids that hold them, from reads of those dcids alone. Returns the manifest."""
    entities = {role: sorted(dcids) for role, dcids in identity.entities.items()}
    minted = doors.mint(identity)
    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    assert manifest["coverage"]["rows"] == held_observations(stat_var, entities, year)
    assert served_counts(manifest) == {
        role: len(dcids) for role, dcids in held_at(stat_var, entities, year).items()
    }
    for read in doors.reads("mint"):
        assert read.dcids <= bound_dcids(identity), read
    return manifest


def check_three_doors(doors: Doors, stat_var: str, spec: Spec, year: int | None = None) -> None:
    """`spec`, its resubmit and its mint read one list of dcids, the one its identity
    binds, and state the same observations (`check_first_answer`, `check_posted`,
    `check_minted`)."""
    check_first_answer(doors, stat_var, spec, year)
    identity = spec.definition.identity
    again = check_posted(doors, stat_var, identity, spec.resolved.entity_groups, year)
    assert again.resolved.coverage == spec.resolved.coverage
    assert preview(again) == preview(spec)
    check_minted(doors, stat_var, identity, year)


def check_groups(spec: Spec) -> None:
    """Each group `spec` serves with a dcid names a set whose members are the dcids its
    role binds."""
    for role, served in (spec.resolved.entity_groups or {}).items():
        if served.dcid is not None:
            assert bound(spec)[role] == sorted(EXPANSION[served.dcid]), (role, served)
