"""Pins artifact loading, format checks, and failures for incomplete or inconsistent files."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from series_coverage.io import (
    read_manifest,
    write_coverage_table,
    write_current,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.loader import (
    CoverageArtifactError,
    format_root,
    load_coverage,
    read_published_pointer,
)
from series_coverage.schema import (
    BUILD_EPOCH_YEAR,
    COVERAGE_FORMAT_VERSION,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)
from tests.series_coverage.support import (
    KEN,
    V1,
    build_dir,
    build_single_mask_artifact,
    build_test_artifact,
    pointer_path,
)

_HORIZON = YearHorizon(epoch_year=BUILD_EPOCH_YEAR, mask_words=1)


def _point_at(base: Path, *, stamp: str = "s1", build_id: str = "b1") -> Path:
    """Writes `base`'s pointer naming `build_id` and returns that build's
    (created) directory."""
    prefix = build_dir(base, build_id)
    prefix.mkdir(parents=True, exist_ok=True)
    write_current(CoveragePointer(stamp=stamp, build_id=build_id), pointer_path(base))
    return prefix


def _manifest(**overrides: object) -> CoverageManifest:
    fields: dict = {
        "stamp": "s1",
        "build_id": "b1",
        "built_at": "2026-09-24T00:00:00Z",
        "source": "test",
        "row_counts": {"rows": 1},
        "files": {
            "coverage": "coverage.parquet",
            "role_names": "role_names.json",
            "facets": "facets.json",
        },
        "horizon": _HORIZON,
    }
    fields.update(overrides)
    return CoverageManifest(**fields)


def test_load_coverage_reads_a_real_shaped_artifact(tmp_path: Path) -> None:
    uri = build_test_artifact(tmp_path / "coverage")
    loaded = load_coverage(uri)
    assert loaded.manifest.stamp == "test-coverage-stamp"
    assert loaded.index.num_rows > 0
    assert loaded.index.filter_candidates(variables=[V1], bound={"recipient": frozenset({KEN})})


def test_build_test_artifact_writes_the_format_root_layout(tmp_path: Path) -> None:
    # The layout itself, independent of load_coverage's read path: the
    # pointer and every build file live under format-<N>/, the build's files
    # under versions/<build_id>/, and nothing sits directly under the base.
    base = tmp_path / "coverage"
    build_test_artifact(base, build_id="b1")
    root = base / f"format-{COVERAGE_FORMAT_VERSION}"
    assert (root / "current.json").is_file()
    for name in ("manifest.json", "coverage.parquet", "role_names.json", "facets.json"):
        assert (root / "versions" / "b1" / name).is_file(), name
    assert sorted(p.name for p in base.iterdir()) == [root.name]


def test_load_coverage_raises_for_missing_current_json(tmp_path: Path) -> None:
    with pytest.raises(CoverageArtifactError, match=re.escape("no current.json")):
        load_coverage(f"file://{tmp_path / 'nothing-here'}")


def test_load_coverage_raises_for_a_pointer_without_a_build_id(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    build_test_artifact(base)
    pointer_path(base).write_text('{"stamp": "test-coverage-stamp"}', encoding="utf-8")
    with pytest.raises(CoverageArtifactError, match=r"current\.json .* is malformed"):
        load_coverage(f"file://{base}")


def test_load_coverage_raises_for_missing_manifest(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    _point_at(base)
    with pytest.raises(CoverageArtifactError, match=re.escape("no manifest.json")):
        load_coverage(f"file://{base}")


def test_load_coverage_raises_when_manifest_names_a_missing_file(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    prefix = _point_at(base)
    write_manifest(_manifest(row_counts={}), prefix / "manifest.json")
    with pytest.raises(CoverageArtifactError, match="coverage file"):
        load_coverage(f"file://{base}")


def test_load_coverage_raises_on_stamp_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    prefix = _point_at(base)
    write_manifest(_manifest(stamp="s2"), prefix / "manifest.json")
    with pytest.raises(CoverageArtifactError, match="disagree"):
        load_coverage(f"file://{base}")


def test_load_coverage_raises_on_build_id_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    prefix = _point_at(base)
    write_manifest(_manifest(build_id="b2"), prefix / "manifest.json")
    with pytest.raises(CoverageArtifactError, match="disagree"):
        load_coverage(f"file://{base}")


def test_load_coverage_reads_only_its_own_format_root(tmp_path: Path) -> None:
    # A complete artifact under the next format's root is invisible to this
    # code: the base holds no format-<N> root, so there is nothing to load.
    base = tmp_path / "coverage"
    build_test_artifact(base)
    newer = base / f"format-{COVERAGE_FORMAT_VERSION + 1}"
    (base / f"format-{COVERAGE_FORMAT_VERSION}").rename(newer)
    uri = f"file://{base}"

    assert read_published_pointer(uri) is None
    with pytest.raises(CoverageArtifactError, match=re.escape("no current.json")):
        load_coverage(uri)
    assert format_root(uri) == f"{uri}/format-{COVERAGE_FORMAT_VERSION}"


def test_load_coverage_refuses_an_artifact_under_the_format_1_root(tmp_path: Path) -> None:
    # Format 1 artifacts hold no one-role rows, and a reader at the current
    # format never reads that root, even when it is the only one published.
    base = tmp_path / "coverage"
    build_test_artifact(base)
    (base / f"format-{COVERAGE_FORMAT_VERSION}").rename(base / "format-1")
    uri = f"file://{base}"

    assert COVERAGE_FORMAT_VERSION == 2
    assert format_root(uri) == f"{uri}/format-2"
    assert read_published_pointer(uri) is None
    with pytest.raises(CoverageArtifactError, match=re.escape("no current.json")):
        load_coverage(uri)


def test_load_coverage_refuses_a_format_1_manifest_in_the_current_root(tmp_path: Path) -> None:
    uri = build_test_artifact(tmp_path / "coverage")
    _rewrite_manifest(tmp_path / "coverage", format_version=1)

    with pytest.raises(CoverageArtifactError, match="format_version 1, but"):
        load_coverage(uri)


def _write_bad_invariant_artifact(base: Path) -> None:
    table = pa.table(
        {
            "variable": ["V1"],
            "facet_id": ["f1"],
            "donor": ["country/FRA"],
            "recipient": ["country/KEN"],
            "n_obs": pa.array([5], type=pa.uint32()),  # disagrees with the mask below (1 bit set)
            "first_year": pa.array([2020], type=pa.uint16()),
            "last_year": pa.array([2020], type=pa.uint16()),
            "year_mask_0": pa.array([1 << 47], type=pa.uint64()),
        },
        schema=_HORIZON.schema,
    )
    _write_single_variable_artifact(base, table)


def _write_single_variable_artifact(base: Path, table: pa.Table) -> None:
    """An artifact at `base` holding `table`, whose rows all name variable
    `V1` under facet `f1`."""
    prefix = _point_at(base)
    write_coverage_table(table, prefix / "coverage.parquet", horizon=_HORIZON)
    write_role_names(
        {"V1": RoleNames(donor="financingProvider", recipient="financingRecipient")},
        prefix / "role_names.json",
    )
    write_facet_metadata(
        {"f1": FacetMetadata(provenance_id="provenance/Test")}, prefix / "facets.json"
    )
    write_manifest(_manifest(), prefix / "manifest.json")


def test_load_coverage_raises_on_n_obs_year_mask_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    _write_bad_invariant_artifact(base)
    with pytest.raises(CoverageArtifactError, match="bit count"):
        load_coverage(f"file://{base}")


def test_load_coverage_reads_the_key_columns_already_dictionary_encoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CoverageIndex encodes any plain-string key column it is handed; with
    # that encoder disabled, a load that still succeeds proves the loader
    # read every key column dictionary-encoded itself.
    import series_coverage.index as index_module

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the loader handed CoverageIndex a plain-string key column")

    monkeypatch.setattr(index_module.pc, "dictionary_encode", _refuse)
    loaded = load_coverage(build_test_artifact(tmp_path / "coverage"))
    for name in ("variable", "facet_id", "donor", "recipient"):
        assert pa.types.is_dictionary(loaded.index._table.schema.field(name).type), name


def test_load_coverage_raises_on_a_repeated_series_key(tmp_path: Path) -> None:
    # Two rows share (V1, f1, FRA, KEN); each row alone satisfies the popcount
    # invariant.
    table = pa.table(
        {
            "variable": ["V1", "V1"],
            "facet_id": ["f1", "f1"],
            "donor": ["country/FRA", "country/FRA"],
            "recipient": ["country/KEN", "country/KEN"],
            "n_obs": pa.array([1, 1], type=pa.uint32()),
            "first_year": pa.array([2020, 2021], type=pa.uint16()),
            "last_year": pa.array([2020, 2021], type=pa.uint16()),
            "year_mask_0": pa.array([1 << 47, 1 << 48], type=pa.uint64()),
        },
        schema=_HORIZON.schema,
    )
    base = tmp_path / "coverage"
    _write_single_variable_artifact(base, table)
    with pytest.raises(CoverageArtifactError, match=r"1 row\(s\) repeating"):
        load_coverage(f"file://{base}")


def test_load_coverage_exposes_facet_metadata_on_the_index(tmp_path: Path) -> None:
    from tests.series_coverage.support import FACET_ACTUALS, FACET_METADATA

    uri = build_test_artifact(tmp_path / "coverage")
    loaded = load_coverage(uri)
    assert loaded.index.facet_metadata == FACET_METADATA
    assert loaded.index.facet_metadata[FACET_ACTUALS].unit == "CurrencyUSD"
    assert loaded.index.facet_metadata[FACET_ACTUALS].observation_period == "P1Y"


def test_load_coverage_reads_facets_json_written_without_observation_period(
    tmp_path: Path,
) -> None:
    # Artifacts published before the job read `observation_period` have no such key.
    from tests.series_coverage.support import FACET_ACTUALS

    uri = build_test_artifact(tmp_path / "coverage")
    facets_path = build_dir(tmp_path / "coverage", "test-coverage-stamp-build") / "facets.json"
    payload = json.loads(facets_path.read_text(encoding="utf-8"))
    for fields in payload.values():
        del fields["observation_period"]
    facets_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_coverage(uri)
    assert {meta.observation_period for meta in loaded.index.facet_metadata.values()} == {None}
    assert loaded.index.facet_metadata[FACET_ACTUALS].unit == "CurrencyUSD"


def test_load_coverage_raises_when_manifest_has_no_facets_entry(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    prefix = _point_at(base)
    table = pa.table(
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
    write_coverage_table(table, prefix / "coverage.parquet", horizon=_HORIZON)
    write_role_names(
        {"V1": RoleNames(donor="financingProvider", recipient="financingRecipient")},
        prefix / "role_names.json",
    )
    write_manifest(
        _manifest(files={"coverage": "coverage.parquet", "role_names": "role_names.json"}),
        prefix / "manifest.json",
    )
    with pytest.raises(CoverageArtifactError, match="facets"):
        load_coverage(f"file://{base}")


def _rewrite_manifest(base: Path, *, format_version: int) -> None:
    path = build_dir(base, "test-coverage-stamp-build") / "manifest.json"
    manifest = read_manifest(path)
    assert manifest.format_version == COVERAGE_FORMAT_VERSION
    write_manifest(replace(manifest, format_version=format_version), path)


@pytest.mark.parametrize(
    "format_version",
    [COVERAGE_FORMAT_VERSION + 1, COVERAGE_FORMAT_VERSION - 1],
    ids=["newer", "previous"],
)
def test_load_coverage_refuses_a_manifest_of_another_format_version(
    tmp_path: Path, format_version: int
) -> None:
    # A manifest labelled with a format_version this code doesn't read must
    # never load: a different bit-packing convention would give every
    # windowed lookup the wrong answer with no error at all.
    uri = build_test_artifact(tmp_path / "coverage")
    _rewrite_manifest(tmp_path / "coverage", format_version=format_version)

    with pytest.raises(CoverageArtifactError, match=f"format_version {format_version}"):
        load_coverage(uri)


def test_load_coverage_refuses_a_manifest_with_no_format_version(tmp_path: Path) -> None:
    base = tmp_path / "coverage"
    uri = build_test_artifact(base)
    path = build_dir(base, "test-coverage-stamp-build") / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["format_version"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CoverageArtifactError, match="missing required field 'format_version'"):
        load_coverage(uri)


def test_load_coverage_refuses_a_single_mask_artifact_labelled_with_this_format(
    tmp_path: Path,
) -> None:
    # A manifest carrying this code's format_version but no year horizon (the
    # single-mask layout) is refused by name, before the table is read.
    uri = build_single_mask_artifact(tmp_path / "coverage")
    with pytest.raises(
        CoverageArtifactError, match=r"missing required field\(s\) \['epoch_year', 'mask_words'\]"
    ):
        load_coverage(uri)


def test_load_coverage_refuses_a_table_without_the_first_mask_word(tmp_path: Path) -> None:
    # A well-formed manifest over a single-mask Parquet file (`year_mask` in
    # place of `year_mask_0`) fails the schema check by name.
    base = tmp_path / "coverage"
    uri = build_test_artifact(base)
    path = build_dir(base, "test-coverage-stamp-build") / "coverage.parquet"
    table = pq.read_table(path)
    pq.write_table(
        table.rename_columns(
            ["year_mask" if name == "year_mask_0" else name for name in table.column_names]
        ),
        path,
    )

    with pytest.raises(CoverageArtifactError, match="does not match the manifest's"):
        load_coverage(uri)


def test_load_coverage_raises_when_a_coverage_facet_id_has_no_metadata_entry(
    tmp_path: Path,
) -> None:
    base = tmp_path / "coverage"
    prefix = _point_at(base)
    table = pa.table(
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
    write_coverage_table(table, prefix / "coverage.parquet", horizon=_HORIZON)
    write_role_names(
        {"V1": RoleNames(donor="financingProvider", recipient="financingRecipient")},
        prefix / "role_names.json",
    )
    # facets.json exists but has no entry for "f1", the only facet_id the
    # coverage table carries.
    write_facet_metadata({}, prefix / "facets.json")
    write_manifest(_manifest(), prefix / "manifest.json")
    with pytest.raises(CoverageArtifactError, match="f1"):
        load_coverage(f"file://{base}")
