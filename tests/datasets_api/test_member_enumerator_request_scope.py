"""The definition door reads member groups through the process's shared enumerator, so a graph
write reaches a GET at the next graph generation, and each request reuses one `DefinitionDeps`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import httpx

import datasets_api.doors.definition as definition_module
from datasets_api.doors.definition import DefinitionDeps
from dc_sdmx import SdmxClient
from graph_generation import FakeGeneration, GenerationStamp
from materialise.graph import Facet, FacetDefaultPolicy
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.identity import recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._identity_builders import register_body
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore, wire_definition_for
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_RECIPIENT = "country/KEN"

_ODA_GRANTS = "ODAGrantsAmount"
_ODA_GRANTS_REF = f"dcid:{_ODA_GRANTS}"
_ODA_LOANS = "ODALoansAmount"
_ODA_LOANS_REF = f"dcid:{_ODA_LOANS}"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_DESCRIBED_TWO_MEMBER_IDENTITY: dict = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [
        {
            "property": "DevelopmentFinanceScheme",
            "binding": {"state": "set", "refs": ["ODAGrants", "ODALoans"]},
        }
    ],
    "entities": {"recipient": [_RECIPIENT]},
    "facets": {
        "provenance": {"state": "value", "ref": "OECD_DAC2A"},
        "measurementMethod": {"state": "value", "ref": "GrantEquivalent"},
        "unit": {"state": "value", "ref": "USD"},
    },
    "requested_window": {"kind": "all"},
}


class _MutableGraph:
    """Mutable graph double with the reads needed to enumerate and materialise described specs."""

    def __init__(
        self,
        *,
        arcs: Mapping[str, Mapping[str, tuple[str, ...]]],
        facets: Mapping[tuple[str, str], list[Facet]] = {},
        labels: Mapping[str, str] = {},
    ) -> None:
        self._arcs = arcs
        self._facets = facets
        self._labels = labels
        self.generation = FakeGeneration(token="rev1")

    def generation_stamp(self) -> GenerationStamp:
        return self.generation.stamp()

    def node_label(self, dcid: str) -> str | None:
        return self._labels.get(dcid)

    def observation_facets_batch(
        self, *, stat_var: str, entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, list[Facet]]:
        return {entity: self._facets.get((stat_var, entity), []) for entity in entities}

    def observation_facets_batch_many(
        self, *, stat_vars: Sequence[str], entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, list[Facet]]]:
        return {
            stat_var: {entity: self._facets.get((stat_var, entity), []) for entity in entities}
            for stat_var in stat_vars
        }

    def declares_facet_defaults(self) -> bool:
        return False

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        return None

    def node_arcs(self, dcid: str, *, properties: Sequence[str]) -> dict[str, tuple[str, ...]]:
        node = self._arcs.get(dcid, {})
        return {prop: node.get(prop, ()) for prop in properties}

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ):
        filters = filters or {}
        matches = sorted(
            candidate
            for candidate, node_arcs in self._arcs.items()
            if dcid in node_arcs.get(property, ())
            and all(
                filter_value in node_arcs.get(filter_property, ())
                for filter_property, filter_value in filters.items()
            )
        )
        return _SubjectPage(
            dcids=tuple(matches[:max_results]), truncated=len(matches) > max_results
        )

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for candidate in dcids:
            node = self._arcs.get(candidate)
            result[candidate] = (
                None if node is None else {prop: node.get(prop, ()) for prop in properties}
            )
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        return {
            candidate: tuple(
                dict.fromkeys(
                    value
                    for prop in ("name", "alternateName")
                    for value in self._arcs.get(candidate, {}).get(prop, ())
                )
            )
            for candidate in dict.fromkeys(dcids)
        }


@dataclasses.dataclass(frozen=True)
class _SubjectPage:
    dcids: tuple[str, ...]
    truncated: bool


def _member_arcs_for(scheme_value: str) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("recipient",),
        "constraintProperties": ("DevelopmentFinanceScheme",),
        "DevelopmentFinanceScheme": (scheme_value,),
    }


def _sdmx_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        variable_param = request.url.params.get("c[variableMeasured]", "")
        rows = ""
        if _ODA_GRANTS in variable_param:
            rows += (
                f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_GRANTS},{_RECIPIENT},USD,GrantEquivalent,"
                "NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
            )
        if _ODA_LOANS in variable_param:
            rows += (
                f"dataflow,DC:DF_OBS(1.0.0),I,{_ODA_LOANS},{_RECIPIENT},USD,GrantEquivalent,"
                "NotApplicable,OECD_DAC2A,2020,500,,facet1\r\n"
            )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_service() -> FakeCoverageService:
    """One-facet coverage for both StatVars, matching `_FACET` and their 2020 rows."""
    index = build_coverage_index(
        [
            (_ODA_GRANTS, "facet1", ABSENT_ENTITY, _RECIPIENT, [2020]),
            (_ODA_LOANS, "facet1", ABSENT_ENTITY, _RECIPIENT, [2020]),
        ],
        role_names={
            _ODA_GRANTS: RoleNames(donor=None, recipient="recipient"),
            _ODA_LOANS: RoleNames(donor=None, recipient="recipient"),
        },
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
            )
        },
    )
    return FakeCoverageService(index)


def _register_and_client() -> tuple:
    """Registers the described identity and returns its client, graph, mutable arcs, and spec
    id."""
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        _ODA_GRANTS: _member_arcs_for("ODAGrants"),
    }
    graph = _MutableGraph(
        arcs=arcs,
        facets={
            (_ODA_GRANTS, _RECIPIENT): [_FACET],
            (_ODA_LOANS, _RECIPIENT): [_FACET],
        },
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )
    store = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=_coverage_service(),
    )
    spec_id = recompute_identity(_DESCRIBED_TWO_MEMBER_IDENTITY).spec_id
    reg = client.post(
        "/api/datasets/register",
        json=register_body(
            definition=wire_definition_for(spec_id=spec_id, identity=_DESCRIBED_TWO_MEMBER_IDENTITY)
        ),
    )
    assert reg.status_code == 200, reg.text
    return client, graph, arcs, spec_id


def test_a_graph_write_between_two_retrievals_reaches_the_next_graph_generation():
    """A GET under the generation that resolved a group serves it again, and the first GET
    after the generation moves reads the write."""
    client, graph, arcs, spec_id = _register_and_client()

    def resolved_refs() -> set[str]:
        response = client.get(f"/api/datasets/{spec_id}")
        assert response.status_code == 200, response.text
        return {entry["ref"] for entry in response.json()["trace"]["resolved_stat_vars"]}

    assert resolved_refs() == {_ODA_GRANTS_REF}
    arcs[_ODA_LOANS] = _member_arcs_for("ODALoans")
    assert resolved_refs() == {_ODA_GRANTS_REF}
    graph.generation.advance("rev2")
    assert resolved_refs() == {_ODA_GRANTS_REF, _ODA_LOANS_REF}


def test_every_retrieval_shares_the_surfaces_rebuilds_and_enrichment(monkeypatch):
    """Each request's `DefinitionDeps` wraps the surface's one `RebuildFlights` and enrichment
    cache, so concurrent rebuilds of one subject meet and labels are read once."""
    views: list[DefinitionDeps] = []
    real_definition_deps = definition_module.definition_deps

    def _spy(deps, **kwargs):
        view = real_definition_deps(deps, **kwargs)
        views.append(view)
        return view

    monkeypatch.setattr(definition_module, "definition_deps", _spy)

    client, _graph, _arcs, spec_id = _register_and_client()

    client.get(f"/api/datasets/{spec_id}")
    client.get(f"/api/datasets/{spec_id}")

    first, second = views
    assert first.rebuilds is second.rebuilds
    assert first.enrichment is second.enrichment
