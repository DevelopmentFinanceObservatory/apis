"""A mint with `options.without_party_totals` through `POST /api/datasets` and
`GET /api/datasets/{id}`: the version it records and the totals its manifest names.

The fixture is a flow dataset with an open `provider` side, whose providers are France and the
DAC total, and a `recipient` bound to Kenya. It uses the real registry, materialisation and
SQLite-backed version storage.
"""

from __future__ import annotations

import httpx

from code_index import CodeIndex, IndexStamp
from code_index.types import PartyRow, StatVarRow
from datasets_api.recorded_rendering import MintingOptions, recorded_mintings
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._description_doubles import read_record
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_definition_door_currency import _FakeHeadStore

_MEMBER = "FlowMember"
_FRANCE = "country/FRA"
_DAC_TOTAL = "DACTotal"
_KENYA = "country/KEN"
_FACET_ID = "facet1"
_PROVENANCE = "OECD_DAC2A"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,provider,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = _CSV_HEADER + "".join(
    f"dataflow,DC:DF_OBS(1.0.0),I,{_MEMBER},{provider},{_KENYA},USD,GrantEquivalent,"
    f"NotApplicable,{_PROVENANCE},2020,{value},,{_FACET_ID}\r\n"
    for provider, value in ((_DAC_TOTAL, 900), (_FRANCE, 100))
)

_IDENTITY = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [],
    "entities": {"recipient": [_KENYA]},
    "open_roles": ["origin"],
    "requested_window": {"kind": "all"},
}


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _MEMBER: {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("DevelopmentFinance",),
                "measuredProperty": ("amount",),
                "observationProperties": ("provider", "recipient"),
            }
        },
        labels={_PROVENANCE: "OECD DAC2a", _KENYA: "Kenya", _FRANCE: "France"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _coverage_service() -> FakeCoverageService:
    return FakeCoverageService(
        build_coverage_index(
            [(_MEMBER, _FACET_ID, provider, _KENYA, [2020]) for provider in (_DAC_TOTAL, _FRANCE)],
            role_names={_MEMBER: RoleNames(donor="provider", recipient="recipient")},
            facet_metadata={
                _FACET_ID: FacetMetadata(
                    provenance_id=_PROVENANCE, measurement_method="GrantEquivalent", unit="USD"
                )
            },
        )
    )


def _code_index() -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-09-15T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="sha256:fixture",
        ),
        declarations=(),
        codes=(),
        statvars=(
            StatVarRow(
                dcid=_MEMBER,
                population_type="DevelopmentFinance",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("provider", "recipient"),
                constraint_properties={},
                provenances=(_PROVENANCE,),
            ),
        ),
        parties=(
            PartyRow(dcid=_DAC_TOTAL, kind="total", datasets=(_PROVENANCE,), axes=("origin",)),
            PartyRow(dcid=_FRANCE, kind="party", datasets=(_PROVENANCE,), axes=("origin",)),
            PartyRow(dcid=_KENYA, kind="party", datasets=(_PROVENANCE,), axes=("destination",)),
        ),
    )


def _rig():
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
        head_writer=_FakeHeadStore(),
    )
    return client, version_store, *_spec_id_and_sidecar()


def _spec_id_and_sidecar() -> tuple[str, dict]:
    recomputed = recompute_identity(_IDENTITY)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": _IDENTITY,
    }


def _mint(client, *, spec_id: str, sidecar: dict, options: dict) -> dict:
    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
            "options": options,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


_CUT = {"format": "csv", "bundle": False, "without_party_totals": True}
_LEFT_OUT = [{"role": "provider", "entities": [_DAC_TOTAL]}]


def test_a_cut_mint_answers_with_its_own_version_naming_the_totals_it_left_out():
    client, version_store, spec_id, sidecar = _rig()

    # With no head yet, the cut mint materialises, and the plain mint then reads the head.
    cut = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_CUT)
    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    assert cut["trace"]["served_from"] == "materialise"
    assert cut["content_version"] != base["content_version"]
    assert cut["trace"]["party_totals_left_out"] == _LEFT_OUT
    assert "party_totals_left_out" not in base["trace"]
    assert (base["coverage"]["rows"], cut["coverage"]["rows"]) == (2, 1)
    assert {d["label"]: d["count"] for d in cut["coverage"]["dimensions"]}["provider"] == 1

    # The cut version is recorded with its base and its cut frame, and a snapshot read of it
    # names the totals.
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT base_content_version, frame_summed FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, cut["content_version"]),
        )
    assert row == {"base_content_version": base["content_version"], "frame_summed": False}
    snapshot = client.get(f"/api/datasets/{spec_id}/{cut['content_version']}")
    assert snapshot.status_code == 200, snapshot.text
    assert snapshot.json()["trace"]["party_totals_left_out"] == _LEFT_OUT
    assert recorded_mintings(a["rendering"] for a in snapshot.json()["artifacts"]) == [
        MintingOptions(without_party_totals=True)
    ]

    # The cut's description record counts its own rows, keeps its base's steps, and sums
    # nothing.
    base_record = read_record(
        version_store, subject_id=spec_id, content_version=base["content_version"]
    )
    cut_record = read_record(
        version_store, subject_id=spec_id, content_version=cut["content_version"]
    )
    assert base_record is not None and cut_record is not None
    assert (base_record.series.rows, cut_record.series.rows) == (2, 1)
    assert cut_record.steps == base_record.steps


def test_a_cut_read_on_a_head_hit_and_by_query_parameter_serves_the_cut_version():
    """The query-parameter form renders the cut from the head's stored frame, records it with
    its description, and answers the version a mint of the same options records."""
    client, version_store, spec_id, sidecar = _rig()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    read = client.get(
        f"/api/datasets/{spec_id}",
        params={"format": "csv", "bundle": "false", "without_party_totals": "true"},
    )
    assert read.status_code == 200, read.text
    cut = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_CUT)

    assert read.json()["content_version"] == cut["content_version"] != minted["content_version"]
    assert read.json()["trace"]["party_totals_left_out"] == _LEFT_OUT
    assert read.json()["coverage"]["rows"] == 1
    assert read.json()["trace"]["served_from"] == "head"
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT frame_summed FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, cut["content_version"]),
        )
    assert row == {"frame_summed": False}
    head_record = read_record(
        version_store, subject_id=spec_id, content_version=minted["content_version"]
    )
    cut_record = read_record(
        version_store, subject_id=spec_id, content_version=read.json()["content_version"]
    )
    assert head_record is not None and cut_record is not None
    assert cut_record.series.rows == 1
    assert cut_record.steps == head_record.steps
