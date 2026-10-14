"""Graph facet-probe deadline breaches classify as retryable 504 timeouts on mint and
retrieval routes. The fixture uses the default `observationAbout` role to reach the graph probe.
"""

from __future__ import annotations

import httpx

from datasets_api.doors.mint_errors import classify_mint_exception
from dc_sdmx import SdmxClient
from materialise.graph import Facet, GraphDeadlineExceededError, GraphReadError
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR = "LifeExpectancy_Person"
_STAT_VAR_REF = f"dcid:{_STAT_VAR}"
_ENTITY = "country/KEN"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},{_ENTITY},Years,dc-aggregate,"
    "NotApplicable,WorldBank,2020,61.4,,facet1\r\n"
)

_FACET = Facet(
    facet_id="facet1",
    provenance_id="WorldBank",
    measurement_method="dc-aggregate",
    unit="Years",
    obs_count=1,
    earliest_date="2020",
    latest_date="2020",
)

_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "measuredProperty": ("lifeExpectancy",),
    "populationType": ("Person",),
}

_SCOPE = {
    "entities": {"observationAbout": [_ENTITY]},
    "facets": {
        "provenance": {"ref": "WorldBank"},
        "measurementMethod": {"ref": "dc-aggregate"},
        "unit": {"ref": "Years"},
    },
}

_MINT_BODY = {
    "schema_version": 1,
    "input": {"kind": "statvar_set", "statvars": [_STAT_VAR_REF], "scope": _SCOPE},
    "options": {"bundle": False},
}


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={_STAT_VAR: _ARCS},
        facets={(_STAT_VAR, _ENTITY): [_FACET]},
        labels={"WorldBank": "World Bank", _ENTITY: "Kenya"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


class _DeadlineExceededGraphReader:
    """Raises `GraphDeadlineExceededError` on facet reads with a deadline and delegates other
    graph reads."""

    def __init__(self, delegate: FakeMintGraphReader) -> None:
        self._delegate = delegate

    def node_label(self, dcid: str) -> str | None:
        return self._delegate.node_label(dcid)

    def variable_facets_batch(self, *, stat_var, entities, deadline: float | None = None):
        if deadline is not None:
            raise GraphDeadlineExceededError("simulated graph read past its absolute deadline")
        return self._delegate.variable_facets_batch(
            stat_var=stat_var, entities=entities, deadline=deadline
        )

    def observation_facets_batch_many(self, *, stat_vars, entities, deadline: float | None = None):
        if deadline is not None:
            raise GraphDeadlineExceededError("simulated graph read past its absolute deadline")
        return self._delegate.observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )

    def generation_stamp(self):
        return self._delegate.generation_stamp()

    def default_facet_policy(self, *, stat_var: str, axis: str):
        return self._delegate.default_facet_policy(stat_var=stat_var, axis=axis)

    def node_arc_values(self, dcid: str, *, properties):
        return self._delegate.node_arc_values(dcid, properties=properties)

    def node_subjects(self, dcid: str, *, property, filters=None, max_results: int):
        return self._delegate.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )

    def node_names_many(
        self,
        dcids,
        *,
        deadline: float | None = None,
    ):
        return self._delegate.node_names_many(dcids)

    def node_arcs_many(
        self,
        dcids,
        *,
        properties,
        individual_probe_budget=None,
        deadline: float | None = None,
    ):
        return self._delegate.node_arcs_many(
            dcids, properties=properties, individual_probe_budget=individual_probe_budget
        )


def _client(*, graph, store: FakeDatasetStore):
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport()),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
    )


def test_a_graph_deadline_breach_yields_504_timeout_from_the_mint_door():
    """A facet read beyond the mint deadline returns a retryable timeout."""
    store = FakeDatasetStore()
    client = _client(graph=_DeadlineExceededGraphReader(_graph()), store=store)

    resp = client.post("/api/datasets", json=_MINT_BODY)

    assert resp.status_code == 504, resp.text
    problem = resp.json()
    assert problem["code"] == "timeout"
    assert problem["retryable"] is True


def test_a_graph_deadline_breach_yields_504_timeout_from_the_definition_door():
    """Without a head writer, retrieval reaches the same deadline-bound facet probe."""
    store = FakeDatasetStore()
    minted = _client(graph=_graph(), store=store).post("/api/datasets", json=_MINT_BODY)
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]

    client = _client(graph=_DeadlineExceededGraphReader(_graph()), store=store)
    resp = client.get(f"/api/datasets/{spec_id}")

    assert resp.status_code == 504, resp.text
    problem = resp.json()
    assert problem["code"] == "timeout"
    assert problem["retryable"] is True


def test_graph_deadline_exceeded_classifies_as_timeout_before_the_general_graph_read_arm():
    """The deadline arm must precede `GraphReadError` so it maps to 504, not 502."""
    deadline_problem = classify_mint_exception(GraphDeadlineExceededError("exemplar"))
    assert deadline_problem is not None
    assert deadline_problem.status == 504
    assert deadline_problem.code == "timeout"
    assert deadline_problem.retryable is True

    read_problem = classify_mint_exception(GraphReadError("exemplar"))
    assert read_problem is not None
    assert read_problem.status == 502
    assert read_problem.code == "upstream_error"
