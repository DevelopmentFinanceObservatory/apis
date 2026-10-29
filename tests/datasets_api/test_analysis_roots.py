"""How the live root source names and reads a root: its default enrichment, its mint options,
its frame pair, a frame-layout file decoded, a root read summed by a groupby, and a stored
recipe root it cannot read, and a root under `latest` read in the rendering it records."""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rfc8785
from dfo_frame_grammar.grammar import UnitRequest

from datasets_api import analysis_roots
from datasets_api.analysis import AnalysisPending, RootRendering, RootRenderingUnreadable, SpecRoot
from datasets_api.analysis_roots import (
    FLOW_ENRICHMENT_COLUMNS,
    LiveRootSource,
    enrichment_columns,
    frame_from_frame_layout,
    frame_ref_for,
    mint_options,
)
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.operations.submit import OperationSubmitted
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister

from ._deps_doubles import Unwired
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door import _client
from .test_definition_door_groupby_collapse import _hierarchical_code_index
from .test_meta_doors import _operation
from .test_refresh_door import (
    _GROUPBY,
    _MEMBER_A_ROW,
    _MEMBER_B_ROW,
    _PLAIN_MINT,
    _SUMMED_MINT,
    _move_to,
    _one_contributor_rig,
)
from .test_serve_groupby_collapse import (
    _coverage_service,
    _graph,
    _mint,
    _sdmx_transport,
    _spec_id_and_sidecar,
)

_SPEC = "spec_" + "a" * 32
_CV = "cv_" + "1" * 32


@pytest.mark.parametrize(
    ("identity", "columns"),
    [
        pytest.param({"entities": {"origin": ["country/DEU"]}}, FLOW_ENRICHMENT_COLUMNS, id="flow"),
        pytest.param(
            {"entities": {"financingProvider": ["country/DEU"]}},
            FLOW_ENRICHMENT_COLUMNS,
            id="flow-column",
        ),
        pytest.param({"open_roles": ["destination"]}, FLOW_ENRICHMENT_COLUMNS, id="open-flow"),
        pytest.param(
            {"entities": {"observationAbout": ["country/KEN"]}},
            ("observationAbout_type",),
            id="place",
        ),
    ],
)
def test_a_root_takes_the_enrichment_its_roles_need(identity: dict, columns: tuple) -> None:
    assert enrichment_columns(identity) == columns


def test_mint_options_are_the_frame_layout_with_the_root_s_columns_and_cut() -> None:
    flow = {"entities": {"origin": ["country/DEU"]}}

    assert mint_options(RootRendering(columns=()), identity=flow) == {
        "format": "parquet",
        "bundle": False,
        "layout": "frame",
    }
    assert mint_options(RootRendering(without_party_totals=True), identity=flow) == {
        "format": "parquet",
        "bundle": False,
        "layout": "frame",
        "columns": list(FLOW_ENRICHMENT_COLUMNS),
        "without_party_totals": True,
    }
    assert mint_options(
        RootRendering(columns=(), groupby={"by": ["time_period"], "op": "sum"}), identity=flow
    ) == {
        "format": "parquet",
        "bundle": False,
        "layout": "frame",
        "groupby": {"by": ["time_period"], "op": "sum"},
    }


def test_a_frame_of_columns_is_named_by_them_and_by_its_kinds_stamp() -> None:
    plain = frame_ref_for(spec_id=_SPEC, content_version=_CV, columns=(), enrichment_stamp="s")
    typed = frame_ref_for(
        spec_id=_SPEC, content_version=_CV, columns=("observationAbout_type",), enrichment_stamp="s"
    )
    kinds = frame_ref_for(
        spec_id=_SPEC, content_version=_CV, columns=("party_kind",), enrichment_stamp="s"
    )

    assert (plain.spec_id, plain.content_version) == (_SPEC, _CV)
    preimage = {"columns": ["observationAbout_type"], "content_version": _CV}
    assert typed.content_version == "cv_" + hashlib.sha256(rfc8785.dumps(preimage)).hexdigest()[:32]
    salted = {"columns": ["party_kind"], "content_version": _CV, "enrichment_stamp": "s"}
    assert kinds.content_version == "cv_" + hashlib.sha256(rfc8785.dumps(salted)).hexdigest()[:32]
    summed = frame_ref_for(
        spec_id=_SPEC,
        content_version=_CV,
        columns=(),
        groupby={"by": ["time_period"], "op": "sum"},
        enrichment_stamp=None,
    )
    grouped = {
        "columns": [],
        "content_version": _CV,
        "groupby": {"by": ["time_period"], "op": "sum"},
    }
    assert summed.content_version == "cv_" + hashlib.sha256(rfc8785.dumps(grouped)).hexdigest()[:32]


def test_a_frame_layout_file_decodes_as_data_mcp_reads_it() -> None:
    converted = json.dumps(
        {"base_year": 2021, "currency": "EUR", "prices": "constant"}, separators=(",", ":")
    )
    table = pa.table(
        {
            "stat_var": ["dcid:Amount", "dcid:Amount"],
            "financingRecipient": ["country/KEN", "country/NGA"],
            "financingRecipient_party_kind": ["party", ""],
            "time_period": ["2022", "2022"],
            "value": [1.5, 2.5],
            "unit": ["USDollar", converted],
            "status": pa.nulls(2, pa.string()),
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)

    frame = frame_from_frame_layout(sink.getvalue())

    assert frame.table.column("financingRecipient_party_kind").to_pylist() == ["party", "untyped"]
    units = [row[frame.columns.index("unit")] for row in frame.rows]
    assert units[0] == "USDollar"
    assert isinstance(units[1], UnitRequest)


def test_a_stored_root_read_with_options_the_runner_does_not_read_refuses() -> None:
    roots = LiveRootSource(deps=Unwired("deps"), recipes=Unwired("recipes"))  # ty: ignore[invalid-argument-type]
    bare = {"format": "parquet", "bundle": False, "layout": "frame"}
    record = {"dataset": {"spec_id": _SPEC, "content_version": _CV}, "options": bare}

    for options in (
        {**bare, "pivot": {"index": "time_period", "columns": "stat_var"}},
        {**bare, "groupby": {"axis": "x"}},
    ):
        with pytest.raises(RootRenderingUnreadable):
            roots.dataset_root({**record, "options": options})
    assert roots.dataset_root(record).rendering == RootRendering(columns=())
    summed = {**bare, "groupby": {"by": ["time_period"], "op": "sum"}}
    assert roots.dataset_root({**record, "options": summed}).rendering == RootRendering(
        columns=(), groupby={"by": ["time_period"], "op": "sum"}
    )


_BARE = {"format": "parquet", "bundle": False, "layout": "frame"}
# `get_data`'s `stat_vars="sum"` and a `breakdown` of `testAxis` at level 0.
_STAT_VARS_SUM = {"by": ["time_period"], "op": "sum"}
_BREAKDOWN = {"by": ["time_period", "testAxis"], "op": "sum", "levels": {"testAxis": 0}}


@pytest.mark.parametrize(
    "groupby", [_STAT_VARS_SUM, _BREAKDOWN], ids=["stat_vars_sum", "breakdown"]
)
def test_a_root_read_with_a_groupby_is_the_summed_version_its_mint_names(groupby: dict) -> None:
    """A root of the raw version read with a groupby reads the version that groupby's mint
    names, as does a root of that version itself, and its recipe, normal form and manifest
    roots name the groupby."""
    store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=store,
        code_index=_hierarchical_code_index(),
        coverage=_coverage_service(),
        recipe_store=LiveRecipeStore(store=store),
        version_lister=LiveVersionLister(store=store),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    raw = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_BARE)["content_version"]
    summed = _mint(client, spec_id=spec_id, sidecar=sidecar, options={**_BARE, "groupby": groupby})
    assert summed["content_version"] != raw

    def run(content_version: str, rendering: dict) -> dict:
        root = {"spec_id": spec_id, "content_version": content_version, "rendering": rendering}
        tree = {
            "verb": "aggregate",
            "arguments": {"aggregation": "sum", "group_by": ["stat_var"]},
            "inputs": {"data": root},
        }
        response = client.post("/api/datasets/analysis", json={"tree": tree})
        assert response.status_code == 200, response.text
        return response.json()["ref"]["members"][0]

    plain = run(raw, {"columns": []})
    from_raw = run(raw, {"columns": [], "groupby": groupby})
    from_summed = run(summed["content_version"], {"columns": [], "groupby": groupby})

    assert from_summed == from_raw
    rendering = {**_BARE, "groupby": groupby}
    assert from_raw["roots"] == [
        {"spec_id": spec_id, "content_version": summed["content_version"], "rendering": rendering}
    ]
    hashes = {member["spec"]["recipe"]["normal_form_hash"] for member in (plain, from_raw)}
    assert len(hashes) == 2
    derived = client.get(
        f"/api/datasets/derived/{from_raw['spec']['spec_id']}/{from_raw['content_version']}"
    ).json()
    assert derived["roots"] == from_raw["roots"]
    (record,) = derived["recipe"]["roots"]
    assert record["options"] == rendering
    assert (
        record["frame"]
        == frame_ref_for(
            spec_id=spec_id,
            content_version=summed["content_version"],
            columns=(),
            groupby=groupby,
            enrichment_stamp=None,
        ).model_dump()
    )


def test_a_live_root_s_recorded_rows_refuse_a_run_before_its_frame_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeVersionRecordDatasetStore()
    client = _client(
        graph=_graph(),
        sdmx_transport=_sdmx_transport(),
        version_store=store,
        code_index=_hierarchical_code_index(),
        coverage=_coverage_service(),
        recipe_store=LiveRecipeStore(store=store),
        version_lister=LiveVersionLister(store=store),
        settings=DatasetsBuildSettings(analysis_max_root_rows=1),
    )
    spec_id, sidecar = _spec_id_and_sidecar()
    minted = _mint(client, spec_id=spec_id, sidecar=sidecar, options=_BARE)
    assert minted["coverage"]["rows"] > 1

    def unread(self, root):
        raise AssertionError("a root past the row bound was read")

    monkeypatch.setattr(LiveRootSource, "read", unread)
    root = {"spec_id": spec_id, "content_version": minted["content_version"]}
    tree = {
        "verb": "aggregate",
        "arguments": {"aggregation": "sum", "group_by": ["stat_var"]},
        "inputs": {"data": {**root, "rendering": {"columns": []}}},
    }
    response = client.post("/api/datasets/analysis", json={"tree": tree})

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "over_ceiling"
    assert body["fix"] == {
        "ceiling_kind": "analysis_root_rows",
        "count": minted["coverage"]["rows"],
        "cap": 1,
    }


@pytest.mark.parametrize(
    ("rendering", "summed"),
    [
        pytest.param({"columns": [], "groupby": _GROUPBY}, True, id="groupby"),
        pytest.param({"columns": []}, False, id="plain"),
    ],
)
def test_a_root_under_latest_follows_new_data_in_the_rendering_it_records(
    rendering: dict, summed: bool
) -> None:
    """A root version recording its plain and summed renderings, while the sum is the plain
    frame, follows new data that splits them in the rendering the root names."""
    shared = _one_contributor_rig(_PLAIN_MINT, _SUMMED_MINT)
    _move_to(shared, _MEMBER_A_ROW, _MEMBER_B_ROW)
    root = {
        "spec_id": shared.spec_id,
        "content_version": shared.base_content_version,
        "rendering": rendering,
    }
    tree = {
        "verb": "aggregate",
        "arguments": {"aggregation": "sum", "group_by": ["stat_var"]},
        "inputs": {"data": root},
    }

    resp = shared.rig.client.post("/api/datasets/analysis", json={"tree": tree, "latest": True})

    assert resp.status_code == 200, resp.text
    ((moved,),) = (member["roots"] for member in resp.json()["ref"]["members"])
    new_head = shared.rig.head_store.head_for(shared.spec_id).content_version
    assert new_head != shared.base_content_version
    if summed:
        assert shared.version_rows()[moved["content_version"]] == new_head
        assert moved["rendering"]["groupby"] == _GROUPBY
    else:
        assert moved["content_version"] == new_head
        assert "groupby" not in moved["rendering"]


@pytest.mark.parametrize(
    ("state", "refused"),
    [
        pytest.param("pending", AnalysisPending, id="pending-waits"),
        pytest.param("failed", RefreshRebuildFailed, id="failed-refuses"),
    ],
)
def test_a_root_whose_rebuild_was_queued_waits_unless_the_rebuild_failed(
    monkeypatch: pytest.MonkeyPatch, state: str, refused: type[Exception]
) -> None:
    """A rebuild that failed for good refuses as the refresh door refuses it, so a caller that
    follows `Retry-After` is not told to wait for it forever."""
    operation = _operation(state=state, refusal_code="over_ceiling" if state == "failed" else None)

    def submitted(**_: object) -> None:
        raise OperationSubmitted(operation=operation)

    monkeypatch.setattr(
        analysis_roots, "served_spec", lambda **_: SimpleNamespace(spec_id=_SPEC, identity={})
    )
    monkeypatch.setattr(analysis_roots, "serve_spec", submitted)
    roots = LiveRootSource(deps=Unwired("deps"), recipes=Unwired("recipes"))  # ty: ignore[invalid-argument-type]

    with pytest.raises(refused):
        roots.resolve(SpecRoot(spec_id=_SPEC), latest=False, trace_id=None)
