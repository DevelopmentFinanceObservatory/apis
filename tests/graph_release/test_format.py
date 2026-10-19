"""Pins the release format: part round trips, the provenance split, the
manifest refusals and the download hash check."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from artifact_publication.uri import write_uri_bytes
from graph_release import format as release_format
from graph_release.format import (
    PART_SCHEMAS,
    SERIES,
    Manifest,
    PartRef,
    ReleaseFormatError,
    manifest_uri,
    open_release,
    part_uri,
    write_part,
)
from tests.graph_release.support import build_test_release, sample_tables


def _rows(table: pa.Table, key: tuple[str, ...]) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda row: tuple(row[k] for k in key))


@pytest.mark.parametrize("name", sorted(PART_SCHEMAS))
def test_each_part_round_trips_sorted_by_its_key(tmp_path, name):
    tables = sample_tables()
    written = {name: tables[name]} | ({"facets": tables["facets"]} if name == "series" else {})
    build_test_release(f"file://{tmp_path / 'bucket'}", tables=written)

    loaded = open_release(f"file://{tmp_path / 'bucket'}", local_dir=tmp_path / "local")

    part = PART_SCHEMAS[name]
    table = loaded.tables[name]
    assert table.schema == part.schema
    expected = _rows(tables[name].cast(_input_schema(tables[name], part)), part.key)
    if name == "series":
        for row in expected:
            row["entity3"] = None
    assert _rows(table, part.key) == expected
    for file_table in _file_tables(loaded.local_dir, loaded.manifest, name):
        assert file_table.to_pylist() == _rows(file_table, part.key)
        for column in file_table.columns:
            array = column.chunk(0)
            if pa.types.is_list(array.type):
                array = array.values
            if pa.types.is_dictionary(array.type):
                dictionary = array.dictionary.to_pylist()
                assert dictionary == sorted(dictionary)


def _input_schema(table: pa.Table, part: release_format.PartSchema) -> pa.Schema:
    # The sample's strings are plain, so compare values.
    return pa.schema(
        part.schema.field(name).with_type(_plain(part.schema.field(name).type))
        for name in table.column_names
    )


def _plain(data_type: pa.DataType) -> pa.DataType:
    if pa.types.is_dictionary(data_type):
        return pa.string()
    if data_type == pa.list_(pa.dictionary(pa.int32(), pa.string())):
        return pa.list_(pa.string())
    return data_type


def _file_tables(local_dir: Path, manifest: Manifest, name: str) -> list[pa.Table]:
    return [
        pa.ipc.open_file(pa.memory_map(str(local_dir / ref.file_name))).read_all()
        for ref in manifest.parts
        if ref.name == name
    ]


def test_blob_parts_round_trip_byte_for_byte_with_their_registered_suffix(tmp_path):
    blobs = {
        "declarations": '{"a":"é"}'.encode(),
        "code_index_blocks": gzip.compress(b'{"blocks":{}}', mtime=0),
    }
    manifest = build_test_release(f"file://{tmp_path}", tables={}, blobs=blobs)

    loaded = open_release(f"file://{tmp_path}", local_dir=tmp_path / "local")

    assert loaded.blobs == blobs
    assert {ref.name: (ref.layer, ref.suffix) for ref in manifest.parts} == {
        "declarations": ("dfo", ".json"),
        "code_index_blocks": ("dfo", ".json.gz"),
    }


def test_manifest_round_trips_through_json():
    manifest = Manifest(
        import_execution_id="execution",
        import_completed_at="2026-10-18T23:50:00+00:00",
        ingested_imports=("ImportA", "ImportB"),
        read_timestamp="2026-10-19T00:00:00+00:00",
        dc_serving_revision="revision",
        builder_code_hash="code",
        declarations_hash="declarations",
        base_un_build_id="base-un-build",
        parts=(
            PartRef(
                name="nodes",
                layer="platform",
                sha256="0" * 64,
                bytes=10,
                rows=1,
                columns=("subject_id",),
                source="spanner.node",
                produced_by_execution="execution",
            ),
        ),
        gates={"popcount_mismatch": 0},
    )

    assert Manifest.from_json(manifest.to_json()) == manifest


def test_row_order_and_dictionary_order_do_not_change_part_bytes(tmp_path):
    edges = sample_tables()["edges"].filter(pc.field("provenance") == "dc/p2")
    kwargs = {
        "name": "edges",
        "layer": "platform",
        "source": "test",
        "produced_by_execution": "e",
        "provenance": "dc/p2",
    }

    forward = write_part(edges, staging_dir=tmp_path / "a", **kwargs)
    reversed_rows = edges.take(pa.array(range(edges.num_rows - 1, -1, -1)))
    backward = write_part(
        reversed_rows.set_column(0, "subject_id", reversed_rows["subject_id"].dictionary_encode()),
        staging_dir=tmp_path / "b",
        **kwargs,
    )

    assert [part.ref.sha256 for part in forward] == [part.ref.sha256 for part in backward]


@pytest.mark.parametrize(
    ("rows_per_variable", "file_rows"),
    [(10, [100, 100, 50]), (30, [90, 90, 70]), (150, [150, 100])],
    ids=["even", "cut-back-to-a-variable-boundary", "oversized-variable-stays-whole"],
)
def test_series_splits_on_variable_boundaries_at_max_rows(
    tmp_path, monkeypatch, rows_per_variable, file_rows
):
    monkeypatch.setattr(release_format, "MAX_ROWS", 100)
    table = _series_rows(250, rows_per_variable=rows_per_variable)

    staged = write_part(
        table,
        staging_dir=tmp_path,
        name="series",
        layer="platform",
        source="test",
        produced_by_execution="e",
        provenance="dc/p1",
    )

    assert [part.ref.rows for part in staged] == file_rows
    files = [pa.ipc.open_file(pa.memory_map(str(part.path))).read_all() for part in staged]
    variables = [file["variable_measured"].to_pylist() for file in files]
    for part, values in zip(staged, variables, strict=True):
        assert part.ref.key_range == (values[0], values[-1])
    for earlier, later in itertools.pairwise(variables):
        assert earlier[-1] < later[0]
    assert [value for values in variables for value in values] == sorted(
        table["variable_measured"].to_pylist()
    )


def _series_rows(rows: int, *, rows_per_variable: int) -> pa.Table:
    variables = [f"Var_{n // rows_per_variable:03d}" for n in range(rows)]
    return pa.table(
        {
            "variable_measured": variables[::-1],
            "entity1": [f"country/{n:04d}" for n in range(rows)],
            "extra_entities_id": [""] * rows,
            "facet_id": ["f1"] * rows,
            "entity2": [None] * rows,
            "observation_count": [1] * rows,
            "earliest_date": [date(2000, 1, 1)] * rows,
            "latest_date": [date(2000, 1, 1)] * rows,
            "period_mask_0": [1 << 40] * rows,
            "period_mask_1": [0] * rows,
        }
    )


def _rewrite_manifest(root: Path, release_id: str, /, **changes: object) -> None:
    path = Path(manifest_uri(f"file://{root}", release_id).removeprefix("file://"))
    payload = json.loads(path.read_bytes())
    path.write_text(json.dumps(payload | changes))


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"format": 2}, "format"),
        ({"format": None}, "format"),
        ({"import_execution_id": "another-execution"}, "contents give"),
    ],
    ids=["newer-format", "no-format", "contents-not-matching-release-id"],
)
def test_reader_refuses_a_manifest_it_was_not_built_for(tmp_path, changes, reason):
    manifest = build_test_release(f"file://{tmp_path}", tables={"nodes": sample_tables()["nodes"]})
    _rewrite_manifest(tmp_path, manifest.release_id, **changes)

    with pytest.raises(ReleaseFormatError, match=reason):
        open_release(f"file://{tmp_path}", local_dir=tmp_path / "local")


def test_reader_refuses_a_part_whose_bytes_do_not_match_its_hash(tmp_path):
    manifest = build_test_release(f"file://{tmp_path}", tables={"nodes": sample_tables()["nodes"]})
    (ref,) = manifest.parts
    stored = Path(part_uri(f"file://{tmp_path}", ref).removeprefix("file://"))
    stored.write_bytes(stored.read_bytes()[:-1] + b"\x00")

    with pytest.raises(ReleaseFormatError, match="SHA-256"):
        open_release(f"file://{tmp_path}", local_dir=tmp_path / "local")
    assert list((tmp_path / "local").iterdir()) == []


def test_reader_ignores_unknown_parts_and_columns_and_nulls_absent_optional_ones(tmp_path):
    root = f"file://{tmp_path}"
    series = pa.table(
        {
            "variable_measured": pa.array(["Var_0"]).dictionary_encode(),
            "entity1": pa.array(["country/GHA"]).dictionary_encode(),
            "extra_entities_id": pa.array([""]).dictionary_encode(),
            "facet_id": pa.array(["f1"]).dictionary_encode(),
            "entity2": pa.array(["country/DEU"]).dictionary_encode(),
            "observation_count": pa.array([1], pa.uint32()),
            "earliest_date": pa.array([date(2000, 1, 1)]),
            "latest_date": pa.array([date(2000, 1, 1)]),
            "period_mask_0": pa.array([1 << 40], pa.int64()),
            "period_mask_1": pa.array([0], pa.int64()),
            "future_column": pa.array([7]),
        }
    )
    refs = [_publish_file(root, series, name="series"), _publish_file(root, series, name="future")]
    manifest = Manifest(
        import_execution_id="e",
        import_completed_at="t",
        ingested_imports=(),
        read_timestamp="t",
        dc_serving_revision="r",
        builder_code_hash="c",
        declarations_hash="d",
        parts=tuple(refs),
    )
    write_uri_bytes(manifest_uri(root, manifest.release_id), manifest.to_json())

    loaded = open_release(root, local_dir=tmp_path / "local", release_id=manifest.release_id)

    assert set(loaded.tables) == {"series"}
    assert loaded.tables["series"].schema == SERIES.schema
    assert loaded.tables["series"]["entity3"].to_pylist() == [None]


def _publish_file(root: str, table: pa.Table, *, name: str) -> PartRef:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    data = sink.getvalue().to_pybytes()
    ref = PartRef(
        name=name,
        layer="platform",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        rows=table.num_rows,
        columns=tuple(table.column_names),
        source="test",
        produced_by_execution="e",
        provenance="dc/p1",
        key_range=("Var_0", "Var_0"),
    )
    write_uri_bytes(part_uri(root, ref), data)
    return ref


def test_open_release_without_a_pointer_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_release(f"file://{tmp_path}", local_dir=tmp_path / "local")
