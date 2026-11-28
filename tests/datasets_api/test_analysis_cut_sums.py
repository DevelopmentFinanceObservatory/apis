"""A sum read over a party-totals cut version: the cut and the sum rendered together from the
cut's base version, so the read holds when the cut version's own file is gone.

The fixture is a flow dataset of two StatVars on one summable axis, `testAxis`, each with an
open `provider` side whose providers are France and the DAC total, and a `recipient` bound to
Kenya."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from artifact_store.errors import ArtifactNotFoundError
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import PartyRow, StatVarRow
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_definition_door_currency import _FakeHeadStore
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_MEMBERS = {"FlowMemberA": "TestAxisA", "FlowMemberB": "TestAxisB"}
_FRANCE = "country/FRA"
_DAC_TOTAL = "DACTotal"
_KENYA = "country/KEN"
_FACET_ID = "facet1"
_PROVENANCE = "OECD_DAC2A"
_VALUES = {
    ("FlowMemberA", _DAC_TOTAL): 900,
    ("FlowMemberA", _FRANCE): 100,
    ("FlowMemberB", _DAC_TOTAL): 450,
    ("FlowMemberB", _FRANCE): 50,
}
_CSV = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,provider,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
) + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,{member},{provider},{_KENYA},USD,GrantEquivalent,"
    f"NotApplicable,{_PROVENANCE},2020,{value},,{_FACET_ID}\r\n"
    for (member, provider), value in _VALUES.items()
)
_IDENTITY = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [
        {"property": "testAxis", "binding": {"state": "set", "refs": sorted(_MEMBERS.values())}}
    ],
    "entities": {"recipient": [_KENYA]},
    "open_roles": ["origin"],
    "requested_window": {"kind": "all"},
}
_BARE = {"format": "parquet", "bundle": False, "layout": "frame"}


def _graph() -> FakeMintGraphReader:
    arcs = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("DevelopmentFinance",),
        "measuredProperty": ("amount",),
        "observationProperties": ("provider", "recipient"),
        "constraintProperties": ("testAxis",),
    }
    return FakeMintGraphReader(
        arcs={member: {**arcs, "testAxis": (code,)} for member, code in _MEMBERS.items()},
        labels={_PROVENANCE: "OECD DAC2a", _KENYA: "Kenya", _FRANCE: "France"},
    )


def _code_index() -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-09-15T00:00:00Z",
            code_counts={"testAxis": 2},
            statvar_shape_counts={},
            content_hash="sha256:fixture",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system="testAxis",
                hierarchical=False,
                disjoint_from_children=False,
                summable=True,
                absent_reading="not_declared",
                plain_name="test axis",
            ),
        ),
        codes=tuple(
            CodeRow(
                dcid=code,
                code_system="testAxis",
                code=code[-1],
                label=f"Axis value {code[-1]}",
                synonyms=(),
                parent=None,
            )
            for code in _MEMBERS.values()
        ),
        statvars=tuple(
            StatVarRow(
                dcid=member,
                population_type="DevelopmentFinance",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("provider", "recipient"),
                constraint_properties={"testAxis": code},
                datasets=(_PROVENANCE,),
            )
            for member, code in _MEMBERS.items()
        ),
        parties=(
            PartyRow(dcid=_DAC_TOTAL, kind="total", published=((_PROVENANCE, "origin"),)),
            PartyRow(dcid=_FRANCE, kind="party", published=((_PROVENANCE, "origin"),)),
            PartyRow(dcid=_KENYA, kind="party", published=((_PROVENANCE, "destination"),)),
        ),
    )


def _coverage_service() -> FakeCoverageService:
    return FakeCoverageService(
        build_coverage_index(
            [
                (member, _FACET_ID, provider, _KENYA, [2020])
                for member in _MEMBERS
                for provider in (_DAC_TOTAL, _FRANCE)
            ],
            role_names={
                member: RoleNames(donor="provider", recipient="recipient") for member in _MEMBERS
            },
            facet_metadata={
                _FACET_ID: FacetMetadata(
                    provenance_id=_PROVENANCE, measurement_method="GrantEquivalent", unit="USD"
                )
            },
        )
    )


def _mint(client, *, spec_id: str, sidecar: dict, options: dict) -> dict:
    response = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
            "options": options,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_a_sum_over_a_cut_version_whose_file_is_gone_renders_from_its_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank over a version a party-totals cut minted sums its two StatVars. The cut
    version's file is gone, so the cut and the sum render together from its base version, and
    the rank reads France's 150 with the DAC total left out."""
    store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                text=_CSV,
                headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
            )
        ),
        version_store=store,
        code_index=_code_index(),
        coverage=_coverage_service(),
        head_writer=_FakeHeadStore(),
        recipe_store=LiveRecipeStore(store=store),
        version_lister=LiveVersionLister(store=store),
    )
    recomputed = recompute_identity(_IDENTITY)
    spec_id = recomputed.spec_id
    sidecar = {"schema_version": 1, "spec_id": spec_id, "identity": _IDENTITY}
    raw = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_BARE)
    cut = _mint(
        client, spec_id=spec_id, sidecar=sidecar, options={**_BARE, "without_party_totals": True}
    )
    assert cut["content_version"] != raw["content_version"]
    gone = {
        location.artifact_digest
        for location in LiveSnapshotReader(store=store).read_artifact_locations(
            spec_id, cut["content_version"]
        )
    }
    get = _InventoryRecordingArtifactStore.get

    def without_the_cut(self, *, artifact_digest: str, tier: Any) -> bytes:
        if artifact_digest in gone:
            raise ArtifactNotFoundError(digest=artifact_digest, tier=tier)
        return get(self, artifact_digest=artifact_digest, tier=tier)

    monkeypatch.setattr(_InventoryRecordingArtifactStore, "get", without_the_cut)
    root = {
        "spec_id": spec_id,
        "content_version": cut["content_version"],
        "rendering": {"columns": []},
    }
    tree = {
        "verb": "rank",
        "arguments": {"rank_by": ["provider"], "aggregation": "sum"},
        "inputs": {"data": root},
    }

    response = client.post("/api/datasets/analysis", json={"tree": tree})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 1
    assert [summed["count"] for summed in body["stat_vars_summed"]] == [2]
    assert body["derived_roots"] == []
