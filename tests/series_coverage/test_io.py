"""`series_coverage.io`'s writers, checked against the shape each one
produces (`series_coverage.loader`'s own tests already cover the real read
path for the coverage table, role-names JSON, and facet-metadata JSON:
`series_coverage.loader.load_coverage`'s own private readers, not this
module's writers' counterpart). `read_manifest`/`read_current` are read
back here directly since no other test exercises them.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from series_coverage.io import (
    read_current,
    read_manifest,
    write_coverage_table,
    write_current,
    write_manifest,
)
from series_coverage.schema import CoverageManifest, CoveragePointer, YearHorizon

_HORIZON = YearHorizon(epoch_year=1973, mask_words=1)


def _one_row_table() -> pa.Table:
    return pa.table(
        {
            "variable": ["V1"],
            "facet_id": ["f1"],
            "donor": ["country/FRA"],
            "recipient": ["country/KEN"],
            "n_obs": pa.array([1], type=pa.uint32()),
            "first_year": pa.array([2020], type=pa.uint16()),
            "last_year": pa.array([2020], type=pa.uint16()),
            "year_mask_0": pa.array([1 << 47], type=pa.uint64()),
        },
        schema=_HORIZON.schema,
    )


def test_write_coverage_table_refuses_wrong_schema(tmp_path: Path) -> None:
    bad = pa.table({"variable": ["V1"]})
    with pytest.raises(pa.ArrowInvalid, match="does not match"):
        write_coverage_table(bad, tmp_path / "bad.parquet", horizon=_HORIZON)


def test_coverage_table_is_zstd_compressed(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    path = tmp_path / "coverage.parquet"
    write_coverage_table(_one_row_table(), path, horizon=_HORIZON)
    metadata = pq.ParquetFile(path).metadata
    codec = metadata.row_group(0).column(0).compression
    assert codec.upper() == "ZSTD"


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = CoverageManifest(
        stamp="s1",
        build_id="b1",
        built_at="2026-09-24T00:00:00Z",
        source="test",
        row_counts={"rows": 4},
        files={"coverage": "coverage.parquet", "role_names": "role_names.json"},
        horizon=_HORIZON,
    )
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


def test_read_manifest_raises_on_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"stamp": "s1"}', encoding="utf-8")
    with pytest.raises(KeyError):
        read_manifest(path)


def test_current_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "current.json"
    pointer = CoveragePointer(stamp="s1", build_id="20260927T142252Z-3f9a0c1e")
    write_current(pointer, path)
    assert read_current(path) == pointer


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"stamp": "s1"}',
        '{"build_id": "b1"}',
        '{"stamp": "s1", "build_id": ""}',
        '{"stamp": "s1", "build_id": "../b1"}',
        '{"stamp": "s1", "build_id": "a/b"}',
        '["s1", "b1"]',
    ],
)
def test_read_current_refuses_a_pointer_without_a_stamp_and_a_one_segment_build_id(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "current.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        read_current(path)
