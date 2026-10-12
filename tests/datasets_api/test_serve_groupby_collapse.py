"""A full groupby collapse records raw and collapsed versions under one spec id through
`POST /api/datasets`.

The test uses real registry, materialisation, and SQLite-backed version storage. Its two
members differ on the synthetic summable `testAxis`.
"""

from __future__ import annotations

import json
from typing import cast

import httpx
import pytest
from dfo_frame_grammar import Frame

import datasets_api.serve as serve_module
import materialise.generation as generation_module
from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from series_coverage.schema import ABSENT_ENTITY, FacetMetadata, RoleNames
from spec_registry.identity import recompute_identity
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._frame_doubles import _canonical_frame_row, _version_artifact_rows
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client

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
                provenances=("OECD_DAC2A",),
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
                provenances=("OECD_DAC2A",),
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

    # A `stat_vars_summed` warning discloses that this value sums more
    # than one StatVar's own series into the family ref.
    warning_codes = {w["code"] for w in collapsed["warnings"]}
    assert "stat_vars_summed" in warning_codes
    summed_warning = next(w for w in collapsed["warnings"] if w["code"] == "stat_vars_summed")
    assert f"dcid:{spec_id}" in summed_warning["message"]
    assert "2" in summed_warning["message"]

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


def test_a_repeat_collapsed_mint_returns_the_recorded_summed_warning_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """The second mint finds the collapsed version on record and answers from its stored facts,
    which keeps the warning's structured fields and gains no second copy of it."""
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
    summed = [w for w in second["warnings"] if w["code"] == "stat_vars_summed"]
    assert len(summed) == 1
    assert summed[0]["contributing_member_count"] == 2
    assert summed[0]["stat_var_ref"] == f"dcid:{spec_id}"
    assert summed[0]["stat_var_label"] == "Axis value A/Axis value B"
    assert second["warnings"] == first["warnings"]


def _without_structured_fields(warning: dict) -> dict:
    return {key: warning[key] for key in ("code", "severity", "message")}


def _relabelled(warning: dict) -> dict:
    return {**warning, "stat_var_label": "Label recorded before a code-index re-export"}


@pytest.mark.parametrize("stored_warning", [_without_structured_fields, _relabelled])
def test_a_repeat_collapsed_mint_keeps_the_stored_summed_warning_when_this_call_differs(
    stored_warning,
):
    """A stored `stat_vars_summed` entry that differs from this call's (recorded without the
    structured fields, or under an earlier code-index label) is the one the re-mint returns, once,
    and the re-mint equals a snapshot read of the same version."""
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
        warnings = [
            stored_warning(w) if w["code"] == "stat_vars_summed" else w
            for w in cast(list, row["warnings"])
        ]
        tx.execute(
            "UPDATE datasets.version_record SET warnings = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps(warnings), spec_id, content_version),
        )
    (stored,) = [w for w in warnings if w["code"] == "stat_vars_summed"]

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
