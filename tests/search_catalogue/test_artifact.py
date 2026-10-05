"""Pins where a catalogue version may point and that a loaded build is the
one its pointer names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from search.catalogue.artifact import (
    BUILD_FILE,
    FORMAT_ROOT,
    CatalogueArtifactError,
    CataloguePointer,
    load_build,
    load_published,
)
from tests.search_catalogue.support import publish_catalogue, sample_catalogue

_TRAVERSALS = ("../outside", "..", ".hidden", "a/b", "", "a\\b")


@pytest.mark.parametrize("version", _TRAVERSALS)
def test_a_version_outside_one_path_segment_never_names_a_build(tmp_path: Path, version: str):
    with pytest.raises(ValidationError):
        sample_catalogue(version)
    with pytest.raises(ValidationError):
        CataloguePointer(catalogue_version=version, key="k")
    with pytest.raises(CatalogueArtifactError, match="one path segment"):
        load_build(tmp_path.as_uri(), version)


def test_a_pointer_naming_a_build_outside_versions_loads_nothing(tmp_path: Path):
    base = tmp_path / "bucket"
    artifact = publish_catalogue(base.as_uri())
    root = base / FORMAT_ROOT
    # A build planted beside the format root, which `../../` from `versions/` reaches.
    (base / "planted").mkdir()
    (base / "planted" / BUILD_FILE).write_text(artifact.model_dump_json())
    pointer = {"catalogue_version": "../../planted", "key": artifact.key}
    (root / "current.json").write_text(json.dumps(pointer))
    with pytest.raises(CatalogueArtifactError, match=r"no readable current\.json"):
        load_published(base.as_uri())


def test_a_build_carrying_another_version_than_its_pointer_does_not_load(tmp_path: Path):
    uri = tmp_path.as_uri()
    published = publish_catalogue(uri, sample_catalogue("20260930T000000Z-00000001"))
    other = sample_catalogue("20260930T000000Z-00000002")
    build = tmp_path / FORMAT_ROOT / "versions" / published.catalogue_version / BUILD_FILE
    build.write_text(other.model_dump_json())
    with pytest.raises(CatalogueArtifactError, match="carries version"):
        load_published(uri)


def test_the_published_build_loads(tmp_path: Path):
    uri = tmp_path.as_uri()
    published = publish_catalogue(uri)
    assert load_published(uri) == published
