"""A full groupby collapse records raw and collapsed versions under one spec id through
`POST /api/datasets`.

The test uses real registry, materialisation, and SQLite-backed version storage. Its two
members differ on the synthetic summable `testAxis`.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import httpx
import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar import Frame

import datasets_api.serve as serve_module
import materialise.generation as generation_module
from artifact_store import LiveArtifactInventory
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, DatasetDeclaration, IndexStamp
from code_index.types import StatVarRow
from origin_catalogue import DFO, Catalogue
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    dev_provenance_nodes,
    documenting,
    provenance_row,
)

from ._frame_doubles import _canonical_frame_row, _version_artifact_rows
from ._graph_doubles import FakeMintGraphReader
from ._release_doubles import IndexServingRelease
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_MEMBER_A = "FamilyMemberA"
_MEMBER_B = "FamilyMemberB"
_RECIPIENT = "country/KEN"
_FACET_ID = "facet1"
_PROVENANCE = "OECD_DAC2A"
_MEASUREMENT_METHOD = "GrantEquivalent"
_UNIT = "USD"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)
_CSV_BODY = (
    _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_MEMBER_A},{_RECIPIENT},USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    + f"dataflow,DC:DF_OBS(1.0.0),I,{_MEMBER_B},{_RECIPIENT},USD,GrantEquivalent,"
    f"NotApplicable,OECD_DAC2A,2020,500,,facet1\r\n"
)

_MEMBER_ARCS = {
    "typeOf": ("StatisticalVariable",),
    "populationType": ("DevelopmentFinance",),
    "measuredProperty": ("amount",),
    "observationProperties": ("recipient",),
    "constraintProperties": ("testAxis",),
}

_IDENTITY = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "DevelopmentFinance",
        "measured_property": "amount",
    },
    "slots": [
        {
            "property": "testAxis",
            "binding": {"state": "set", "refs": ["TestAxisA", "TestAxisB"]},
        }
    ],
    "entities": {"recipient": [_RECIPIENT]},
    "requested_window": {"kind": "all"},
}


def _graph() -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={
            _MEMBER_A: {**_MEMBER_ARCS, "testAxis": ("TestAxisA",)},
            _MEMBER_B: {**_MEMBER_ARCS, "testAxis": ("TestAxisB",)},
        },
        labels={"OECD_DAC2A": "OECD DAC2a", _RECIPIENT: "Kenya"},
    )


def _sdmx_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )
    )


def _coverage_service(
    *, stat_vars: tuple[str, ...] = (_MEMBER_A, _MEMBER_B), unit: str = _UNIT
) -> FakeCoverageService:
    """Builds a one-facet index matching `_CSV_BODY` for 2020.

    `unit` overrides the observed unit for converted-base fixtures.
    """
    index = build_coverage_index(
        [(stat_var, _FACET_ID, ABSENT_ENTITY, _RECIPIENT, [2020]) for stat_var in stat_vars],
        role_names={
            stat_var: RoleNames(donor=None, recipient="recipient") for stat_var in stat_vars
        },
        facet_metadata={
            _FACET_ID: FacetMetadata(
                provenance_id=_PROVENANCE, measurement_method=_MEASUREMENT_METHOD, unit=unit
            )
        },
    )
    return FakeCoverageService(index)


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
        # `code_index.members`'s index-backed described-arm enumeration
        # (`datasets_api.adapter._resolve_index_members`) matches a posted
        # identity's own slot refs -- always bare dcids, `dfo_spec_kit`'s
        # own identity grammar refuses a `dcid:`-prefixed one outright --
        # directly against `StatVarRow.constraint_properties` values, with
        # no prefix translation in between (mirrors
        # `tests.datasets_api.test_resolve_definition_members_index`'s own
        # fixtures). `CodeRow.dcid` matches that same bare form so decode
        # can look up each value's own label.
        codes=(
            CodeRow(
                dcid="TestAxisA",
                code_system="testAxis",
                code="A",
                label="Axis value A",
                synonyms=(),
                parent=None,
            ),
            CodeRow(
                dcid="TestAxisB",
                code_system="testAxis",
                code="B",
                label="Axis value B",
                synonyms=(),
                parent=None,
            ),
        ),
        statvars=(
            StatVarRow(
                dcid=_MEMBER_A,
                population_type="DevelopmentFinance",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("recipient",),
                constraint_properties={"testAxis": "TestAxisA"},
                datasets=("OECD_DAC2A",),
            ),
            StatVarRow(
                dcid=_MEMBER_B,
                population_type="DevelopmentFinance",
                measured_property="amount",
                stat_type=None,
                measurement_qualifier=None,
                measurement_denominator=None,
                observation_properties=("recipient",),
                constraint_properties={"testAxis": "TestAxisB"},
                datasets=("OECD_DAC2A",),
            ),
        ),
    )


def _code_index_no_summable_axis() -> CodeIndex:
    """Copies `_code_index`'s members with zero summable declarations, so groupby has an empty
    collapse set.
    """
    base = _code_index()
    return CodeIndex(
        stamp=base.stamp,
        declarations=(),
        codes=base.codes,
        statvars=base.statvars,
    )


def _spec_id_and_sidecar(*, window: dict | None = None) -> tuple[str, dict]:
    """`_IDENTITY`'s spec id and sidecar, its `requested_window` replaced by `window`."""
    identity = _IDENTITY if window is None else {**_IDENTITY, "requested_window": window}
    recomputed = recompute_identity(identity)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": identity,
    }


_ALL_WITH_PARTIAL = {"kind": "all", "include": ["partial"]}
"""A window serving the partial periods a test's catalogue marks."""


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


def test_a_full_collapse_mints_its_own_content_version_and_records_both():
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    collapsed = _mint(
        client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
    )

    assert collapsed["content_version"] != base["content_version"]
    assert collapsed["definition"]["spec_id"] == spec_id == base["definition"]["spec_id"]

    # The sum is a step of the collapsed version's description record, not a warning.
    assert "stat_vars_summed" not in {w["code"] for w in collapsed["warnings"]}

    # The collapse changes output identity. Materialisation resolves the same two members.
    assert len(collapsed["trace"]["resolved_stat_vars"]) == 2
    assert len(base["trace"]["resolved_stat_vars"]) == 2

    # Two distinct rows under this same identity's single `spec_id`: the raw
    # materialisation and the collapsed rendering both stay independently
    # resolvable.
    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT content_version FROM datasets.version_record WHERE subject_id = %s "
            "ORDER BY content_version",
            (spec_id,),
        )
    recorded_versions = {row["content_version"] for row in rows}
    assert recorded_versions == {base["content_version"], collapsed["content_version"]}


def test_a_repeat_collapsed_mint_answers_from_the_recorded_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """The second mint finds the collapsed version on record and answers from its stored facts,
    with the warnings the first mint returned."""
    persisted_versions: list[str] = []
    persisted_response = serve_module.persisted_response

    def _recording_persisted_response(**kwargs):
        persisted_versions.append(kwargs["persisted"].content_version)
        return persisted_response(**kwargs)

    monkeypatch.setattr(serve_module, "persisted_response", _recording_persisted_response)
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    options = {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}}

    first = _mint(client, spec_id=spec_id, sidecar=sidecar, options=options)
    second = _mint(client, spec_id=spec_id, sidecar=sidecar, options=options)

    assert persisted_versions == [first["content_version"]]
    assert second["warnings"] == first["warnings"]


_STORED_SUMMED = {
    "code": "stat_vars_summed",
    "severity": "info",
    "message": "The series sums observations from more than one StatVar.",
}


@pytest.mark.parametrize(
    "stored",
    [
        _STORED_SUMMED,
        {
            **_STORED_SUMMED,
            "contributing_member_count": 2,
            "stat_var_ref": "dcid:spec_recorded",
            "stat_var_label": "Label recorded before a code-index re-export",
        },
    ],
    ids=["without_structured_fields", "structured"],
)
def test_a_repeat_collapsed_mint_keeps_a_stored_summed_warning(stored: dict):
    """A collapsed version minted while collapses reported their sum as a warning stores a
    `stat_vars_summed` entry, with or without the structured fields. The re-mint returns it
    unchanged, once, and equals a snapshot read of the same version."""
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    options = {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}}

    first = _mint(client, spec_id=spec_id, sidecar=sidecar, options=options)
    content_version = first["content_version"]
    with version_store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT warnings FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (spec_id, content_version),
        )
        assert row is not None
        tx.execute(
            "UPDATE datasets.version_record SET warnings = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps([stored, *cast(list, row["warnings"])]), spec_id, content_version),
        )

    second = _mint(client, spec_id=spec_id, sidecar=sidecar, options=options)
    snapshot = client.get(f"/api/datasets/{spec_id}/{content_version}")
    assert snapshot.status_code == 200, snapshot.text

    assert second["content_version"] == content_version
    assert [w for w in second["warnings"] if w["code"] == "stat_vars_summed"] == [stored]
    assert second["warnings"] == snapshot.json()["warnings"]


def test_a_multi_chunk_group_mint_streams_the_canonical_frame_of_its_single_chunk_run(
    monkeypatch: pytest.MonkeyPatch,
):
    """One member per chunk gives the canonical frame writer two chunks; its stored frame has the
    digest the default single-chunk run stores."""
    chunk_counts: list[int] = []
    render_and_store_canonical_frame = serve_module.render_and_store_canonical_frame

    def _counting_render_and_store_canonical_frame(*, frame_chunks, **kwargs):
        chunks: list[Frame] = list(frame_chunks)
        chunk_counts.append(len(chunks))
        return render_and_store_canonical_frame(frame_chunks=chunks, **kwargs)

    monkeypatch.setattr(
        serve_module,
        "render_and_store_canonical_frame",
        _counting_render_and_store_canonical_frame,
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    frame_digests: list[str] = []
    for target_rows in (100_000, 1):
        monkeypatch.setattr(generation_module, "_FRAME_CHUNK_TARGET_ROWS", target_rows)
        version_store = FakeVersionRecordDatasetStore()
        client = _client(
            graph=_graph(),
            sdmx_transport=_sdmx_transport(),
            version_store=version_store,
            code_index=_code_index(),
            coverage=_coverage_service(),
        )
        manifest = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
        rows = _version_artifact_rows(
            version_store, subject_id=spec_id, content_version=manifest["content_version"]
        )
        frame_digests.append(cast(str, _canonical_frame_row(rows)["artifact_digest"]))

    assert chunk_counts == [1, 2]
    assert frame_digests[0] == frame_digests[1]


def test_a_collapse_writes_the_canonical_frame_of_the_base_and_the_collapsed_version():
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    collapsed = _mint(
        client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
    )

    with version_store.transaction() as tx:
        versions = tx.fetch_all(
            "SELECT content_version FROM datasets.version_record WHERE subject_id = %s",
            (spec_id,),
        )
    (base_version,) = {row["content_version"] for row in versions} - {collapsed["content_version"]}
    for content_version in (base_version, collapsed["content_version"]):
        _canonical_frame_row(
            _version_artifact_rows(
                version_store, subject_id=spec_id, content_version=content_version
            )
        )


def test_a_full_collapse_reports_the_collapsed_coverage_not_the_raw_one():
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    collapsed = _mint(
        client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
    )

    # Two members, same recipient/period, sum to one row.
    assert base["coverage"]["rows"] == 2
    assert collapsed["coverage"]["rows"] == 1
    assert collapsed["series_count"] == 1


_SOLE_IDENTITY = {
    **_IDENTITY,
    "slots": [{"property": "testAxis", "binding": {"state": "set", "refs": ["TestAxisA"]}}],
}


def _spec_id_and_sidecar_sole_member() -> tuple[str, dict]:
    recomputed = recompute_identity(_SOLE_IDENTITY)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": _SOLE_IDENTITY,
    }


def _sole_member_sdmx_transport() -> httpx.MockTransport:
    body = (
        _CSV_HEADER + f"dataflow,DC:DF_OBS(1.0.0),I,{_MEMBER_A},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,2020,1000,,facet1\r\n"
    )
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )
    )


def test_a_sole_contributor_full_collapse_mints_the_identical_raw_content_version():
    """The single-contributor no-op case: a described-arm spec resolving exactly one member's own
    full collapse produces a frame byte-identical to the raw one, so the
    collapsed `content_version` equals the raw one and only one version
    record is written under this `spec_id` -- exercised end to end through
    `datasets_api.serve` (`POST /api/datasets`), not a synthetic `_reshape`
    fixture.
    """
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sole_member_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(stat_vars=(_MEMBER_A,)),
    )
    spec_id, sidecar = _spec_id_and_sidecar_sole_member()

    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    collapsed = _mint(
        client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
    )

    assert collapsed["content_version"] == base["content_version"]
    assert "stat_vars_summed" not in {w["code"] for w in collapsed["warnings"]}

    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT content_version FROM datasets.version_record WHERE subject_id = %s",
            (spec_id,),
        )
    assert {row["content_version"] for row in rows} == {base["content_version"]}


def test_a_by_omitting_stat_var_refuses_when_this_deployment_declares_no_summable_axis():
    """Two described members have no summable axis in this index. Omitting `stat_var` raises
    `rollup_not_summable` because the request asks for one series.
    """
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index_no_summable_axis(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar()

    base = _mint(client, spec_id=spec_id, sidecar=sidecar, options={"bundle": False})
    assert base["coverage"]["rows"] == 2

    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
            "options": {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
        },
    )

    assert resp.status_code == 422, resp.text
    problem = resp.json()
    assert problem["code"] == "rollup_not_summable"
    assert problem["pointer"] == "/options/groupby"
    assert "no summable axis" in problem["detail"]


def test_a_sole_contributor_with_no_summable_axis_still_serves_unchanged():
    """With one resolved member, grouping preserves the row even when the index has no
    summable axes.
    """
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sole_member_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index_no_summable_axis(),
        coverage=_coverage_service(stat_vars=(_MEMBER_A,)),
    )
    spec_id, sidecar = _spec_id_and_sidecar_sole_member()

    resp = client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": spec_id, "definition": sidecar},
            "options": {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["coverage"]["rows"] == 1


def _marking(serve_catalogue: Callable[[Catalogue], None], mark: str) -> None:
    """Serve a catalogue documenting `_PROVENANCE` with `mark` partial."""
    arcs = {**dev_provenance_nodes()["provenance/OECDDAC2A"], "partialObservationDate": (mark,)}
    serve_catalogue(documenting({_PROVENANCE: arcs}))


def _records(version_store: FakeVersionRecordDatasetStore) -> dict[str, dict[str, object]]:
    with version_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT content_version, incomplete_after, frame_summed FROM datasets.version_record"
        )
    return {row["content_version"]: row for row in rows}


def _partial(stat_vars: list[str], mark: str) -> list[dict[str, object]]:
    return [
        {
            "provenance": _PROVENANCE,
            "stat_vars": stat_vars,
            "source": _PROVENANCE,
            "status": "partial",
            "partial_period": mark,
        }
    ]


def test_a_summed_version_stores_the_raw_judgement_of_its_rows(
    serve_catalogue: Callable[[Catalogue], None],
) -> None:
    """The source marks 2020 partial and the window opts in to it. The sum of both members
    names its row by the spec id under the members' provenance, so it is partial from the
    members' mark."""
    _marking(serve_catalogue, "2020")
    version_store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=version_store,
        code_index=_code_index(),
        coverage=_coverage_service(),
    )
    spec_id, sidecar = _spec_id_and_sidecar(window=_ALL_WITH_PARTIAL)

    collapsed = _mint(
        client,
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
    )

    recorded = _records(version_store)
    summed = recorded.pop(collapsed["content_version"])
    (raw,) = recorded.values()
    assert summed["frame_summed"]
    assert raw["incomplete_after"] == _partial([f"dcid:{_MEMBER_A}", f"dcid:{_MEMBER_B}"], "2020")
    assert summed["incomplete_after"] == _partial([f"dcid:{spec_id}"], "2020")


# A dataset of several member provenances: `dataset/Ledger` files member A's series under one
# provenance and member B's under another, as a register filed by publisher does.
# `dataset/Solo` is a declared Dataset whose one member provenance files both.
_LEDGER = "dataset/Ledger"
_NORTH = "provenance/LedgerNorth"
_SOUTH = "provenance/LedgerSouth"
_SOLO = "dataset/Solo"
_SOLO_MEMBER = "provenance/SoloMember"
_MEMBERS = (_NORTH, _SOUTH, _SOLO_MEMBER)
_FACETS_BY_MEMBER = {_NORTH: "facetNorth", _SOUTH: "facetSouth", _SOLO_MEMBER: "facetSolo"}
_LEDGER_PUBLISHER = "dc/s/LedgerPublisher"


@dataclass(frozen=True)
class _Declared:
    """A declared dataset and the member provenance filing each of `_MEMBER_A` (1000) and
    `_MEMBER_B` (500)."""

    dataset: str
    filed_a: str
    filed_b: str

    @property
    def rows(self) -> tuple[tuple[str, str, int], ...]:
        return ((_MEMBER_A, self.filed_a, 1000), (_MEMBER_B, self.filed_b, 500))

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.filed_a, self.filed_b)))


_TWO_MEMBERS = _Declared(_LEDGER, _NORTH, _SOUTH)
_ONE_MEMBER = _Declared(_SOLO, _SOLO_MEMBER, _SOLO_MEMBER)


def _ledger_code_index(declared: _Declared = _TWO_MEMBERS) -> CodeIndex:
    base = _code_index()
    return CodeIndex(
        stamp=base.stamp,
        declarations=tuple(base.declarations.values()),
        codes=base.codes,
        statvars=tuple(replace(row, datasets=(declared.dataset,)) for row in base.statvars),
        datasets=(
            DatasetDeclaration(
                id=declared.dataset,
                label=declared.dataset.removeprefix("dataset/"),
                description="A test register filed by publisher.",
                opt_in=False,
                absent_observation="unknown",
                account="provider",
                rank=0,
            ),
        ),
        dataset_members={declared.dataset: declared.members},
    )


def _ledger_graph(
    *, dataset_refreshed: str | None = None, declared: _Declared = _TWO_MEMBERS
) -> FakeMintGraphReader:
    """The members' arcs, and each member provenance's refresh date, the later member newer.
    The Dataset node states `dataset_refreshed` as its own when given."""
    graph = _graph()
    label = declared.dataset.removeprefix("dataset/")
    return FakeMintGraphReader(
        arcs={
            **{member: graph._arcs[member] for member in (_MEMBER_A, _MEMBER_B)},
            declared.dataset: {
                "typeOf": ("Dataset",),
                "name": (label,),
                **({"lastDataRefreshDate": (dataset_refreshed,)} if dataset_refreshed else {}),
            },
            **{
                member: {
                    "typeOf": ("Provenance",),
                    "isPartOf": (declared.dataset,),
                    "lastDataRefreshDate": (refreshed,),
                }
                for member, refreshed in zip(
                    declared.members, ("2026-03-01", "2026-05-01"), strict=False
                )
            },
        },
        labels={declared.dataset: label, _RECIPIENT: "Kenya"},
    )


def _ledger_sdmx_transport(declared: _Declared = _TWO_MEMBERS) -> httpx.MockTransport:
    body = _CSV_HEADER + "".join(
        f"dataflow,DC:DF_OBS(1.0.0),I,{member},{_RECIPIENT},USD,GrantEquivalent,"
        f"NotApplicable,{provenance},2020,{value},,{_FACETS_BY_MEMBER[provenance]}\r\n"
        for member, provenance, value in declared.rows
    )
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )
    )


def _ledger_coverage_service(declared: _Declared = _TWO_MEMBERS) -> FakeCoverageService:
    index = build_coverage_index(
        [
            (member, _FACETS_BY_MEMBER[provenance], ABSENT_ENTITY, _RECIPIENT, [2020])
            for member, provenance, _value in declared.rows
        ],
        role_names={
            member: RoleNames(donor=None, recipient="recipient")
            for member in (_MEMBER_A, _MEMBER_B)
        },
        facet_metadata={
            _FACETS_BY_MEMBER[provenance]: FacetMetadata(
                provenance_id=provenance,
                measurement_method=_MEASUREMENT_METHOD,
                unit=_UNIT,
                observation_period="P1Y",
            )
            for provenance in declared.members
        },
    )
    return FakeCoverageService(index)


def _ledger_client(
    *,
    dataset_refreshed: str | None = None,
    declared: _Declared = _TWO_MEMBERS,
    version_store: FakeVersionRecordDatasetStore | None = None,
    artifact_store=None,
):
    """A client serving a release whose code index declares `declared`, as a production
    request reads its release's index."""
    return _client(
        graph=_ledger_graph(dataset_refreshed=dataset_refreshed, declared=declared),
        sdmx_transport=_ledger_sdmx_transport(declared),
        version_store=version_store
        if version_store is not None
        else FakeVersionRecordDatasetStore(),
        artifact_store=artifact_store,
        release=IndexServingRelease(_ledger_code_index(declared)),
        coverage=_ledger_coverage_service(declared),
    )


def _ledger_spec(pin: dict | None = None) -> tuple[str, dict]:
    identity = _IDENTITY if pin is None else {**_IDENTITY, "facets": {"provenance": pin}}
    recomputed = recompute_identity(identity)
    return recomputed.spec_id, {
        "schema_version": 1,
        "spec_id": recomputed.spec_id,
        "identity": identity,
    }


@pytest.mark.parametrize(
    "pin",
    [None, {"state": "value", "ref": _LEDGER}, {"state": "set", "refs": [_LEDGER]}],
    ids=["unpinned", "pinned", "pinned_set"],
)
def test_a_mint_of_a_dataset_of_two_member_provenances_serves_both_members(pin):
    """The read of an identity pinning the dataset is scoped to its member provenances
    (`adapter.provenance_pin`), so it serves what the unpinned identity serves."""
    spec_id, sidecar = _ledger_spec(pin)

    manifest = _mint(_ledger_client(), spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    counts = {
        entry["ref"]: entry["observation_count"]
        for entry in manifest["trace"]["resolved_stat_vars"]
    }
    assert counts == {f"dcid:{_MEMBER_A}": 1, f"dcid:{_MEMBER_B}": 1}
    assert [
        entry["ref"] for entry in manifest["trace"]["facets"] if entry["axis"] == "provenance"
    ] == [_LEDGER]
    assert "coverage_below_scope" not in {w["code"] for w in manifest["warnings"]}


@pytest.mark.parametrize(
    ("dataset_refreshed", "token"),
    [(None, "2026-05-01"), ("2026-06-01", "2026-06-01"), ("2026-04-01", "2026-05-01")],
    ids=["newest_member", "dataset_node", "member_newer_than_node"],
)
def test_a_dataset_of_two_member_provenances_cites_its_id_with_one_refresh_date(
    dataset_refreshed, token
):
    """The source token of a dataset of several members is the newest refresh date of its
    Dataset node and every member (`facets._probe_provenance_refresh`), so a member's new
    import moves it."""
    spec_id, sidecar = _ledger_spec()

    manifest = _mint(
        _ledger_client(dataset_refreshed=dataset_refreshed),
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False},
    )

    assert manifest["trace"]["source_revisions"] == [{"source": _LEDGER, "revision": token}]


@pytest.mark.parametrize("target_rows", [100_000, 1], ids=["one_chunk", "chunk_per_member"])
def test_a_sum_over_two_member_provenances_of_one_dataset_folds_to_one_total(
    target_rows, monkeypatch: pytest.MonkeyPatch
):
    """The members' rows carry different provenances of one dataset, which agree
    (`render._reshape_groupby_collapse`), so the sum is one row for the recipient."""
    monkeypatch.setattr(generation_module, "_FRAME_CHUNK_TARGET_ROWS", target_rows)
    spec_id, sidecar = _ledger_spec()

    collapsed = _mint(
        _ledger_client(),
        spec_id=spec_id,
        sidecar=sidecar,
        options={"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
    )

    assert collapsed["coverage"]["rows"] == 1


@pytest.mark.parametrize(
    "pin",
    [{"state": "value", "ref": _NORTH}, {"state": "set", "refs": [_NORTH]}],
    ids=["value", "set"],
)
def test_a_pin_on_a_member_provenance_serves_that_member_alone(pin):
    """A member pin scopes the read to that member (`adapter.provenance_pin`) and binds the
    dataset's id on the provenance axis, so the mint serves North's member alone and cites
    the dataset."""
    spec_id, sidecar = _ledger_spec(pin)

    manifest = _mint(_ledger_client(), spec_id=spec_id, sidecar=sidecar, options={"bundle": False})

    counts = {
        entry["ref"]: entry["observation_count"]
        for entry in manifest["trace"]["resolved_stat_vars"]
    }
    assert counts == {f"dcid:{_MEMBER_A}": 1, f"dcid:{_MEMBER_B}": 0}
    assert [
        entry["ref"] for entry in manifest["trace"]["facets"] if entry["axis"] == "provenance"
    ] == [_LEDGER]


def _served_texts(data: bytes) -> list[str]:
    """`data` as text: each file of a zip, the rows of a Parquet file, else the bytes."""
    if data.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            return [text for name in bundle.namelist() for text in _served_texts(bundle.read(name))]
    if data.startswith(b"PAR1"):
        return [json.dumps(pq.read_table(io.BytesIO(data)).to_pylist(), default=str)]
    return [data.decode("utf-8", errors="replace")]


@pytest.mark.parametrize("declared", [_TWO_MEMBERS, _ONE_MEMBER], ids=["two-members", "one-member"])
def test_a_mint_of_a_declared_dataset_names_its_members_only_as_reporters(
    declared: _Declared, serve_catalogue: Callable[[Catalogue], None]
):
    """Every row, ref and citation a mint serves or stores names the dataset: its manifests,
    traces and citations, and every file it writes (frame, CSV, bundle, summed version). The
    dataset's citation takes the source and Data Commons its members' catalogue rows share,
    as Search cites it. A member appears only in a file's `reporting_organisation` column:
    each unsummed row names the member that filed it, and the sum over the recipient keeps
    the member its rows share and names none where two members filed them."""
    serve_catalogue(
        FakeCatalogue(
            provenances={
                member: provenance_row(member, DFO, source=_LEDGER_PUBLISHER)
                for member in declared.members
            }
        )
    )
    version_store = FakeVersionRecordDatasetStore()
    artifacts = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    client = _ledger_client(
        declared=declared, version_store=version_store, artifact_store=artifacts
    )
    spec_id, sidecar = _ledger_spec()

    manifests = [
        _mint(client, spec_id=spec_id, sidecar=sidecar, options=options)
        for options in (
            {"bundle": False},
            {"bundle": True},
            {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}},
        )
    ]

    assert [manifest["coverage"]["rows"] for manifest in manifests] == [2, 2, 1]
    (cited,) = manifests[0]["citation"]["sources"]
    assert (cited["ref"], cited["source"]["ref"], cited["data_commons"]["id"]) == (
        declared.dataset,
        _LEDGER_PUBLISHER,
        "dfo",
    )
    texts = [json.dumps(manifest) for manifest in manifests]
    assert not [member for member in _MEMBERS if any(member in text for text in texts)]
    files = [text for data in artifacts._objects.values() for text in _served_texts(data)]
    assert any(declared.dataset in text for text in files)
    for text in files:
        if any(member in text for member in _MEMBERS):
            assert "reporting_organisation" in text
    # The canonical frames of the version and its sum, beside the series statistics tables.
    frames = [
        pq.read_table(io.BytesIO(data))
        for data in artifacts._objects.values()
        if data.startswith(b"PAR1")
    ]
    reported = {
        tuple(sorted(table.column("reporting_organisation").to_pylist(), key=str))
        for table in frames
        if "time_period" in table.column_names
    }
    shared = declared.filed_a if declared.filed_a == declared.filed_b else None
    assert reported == {tuple(sorted((declared.filed_a, declared.filed_b))), (shared,)}
