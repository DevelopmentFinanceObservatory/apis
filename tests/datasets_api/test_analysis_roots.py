"""How the live root source names and reads a root: its default enrichment, its mint options,
its frame pair, a frame-layout file decoded, and a stored recipe root it cannot read."""

from __future__ import annotations

import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rfc8785
from dfo_frame_grammar.grammar import UnitRequest

from datasets_api.analysis import RootRendering, RootRenderingUnreadable
from datasets_api.analysis_roots import (
    FLOW_ENRICHMENT_COLUMNS,
    LiveRootSource,
    enrichment_columns,
    frame_from_frame_layout,
    frame_ref_for,
    mint_options,
)

from ._deps_doubles import Unwired

_SPEC = "spec_" + "a" * 32
_CV = "cv_" + "1" * 32


@pytest.mark.parametrize(
    ("identity", "columns"),
    [
        pytest.param({"entities": {"origin": ["country/DEU"]}}, FLOW_ENRICHMENT_COLUMNS, id="flow"),
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
    record = {
        "dataset": {"spec_id": _SPEC, "content_version": _CV},
        "options": {
            "format": "parquet",
            "bundle": False,
            "layout": "frame",
            "groupby": {"axis": "x"},
        },
    }

    with pytest.raises(RootRenderingUnreadable):
        roots.dataset_root(record)
    readable = {**record, "options": {"format": "parquet", "bundle": False, "layout": "frame"}}
    assert roots.dataset_root(readable).rendering == RootRendering(columns=())
