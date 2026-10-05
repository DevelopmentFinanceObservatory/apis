"""Pin manifest digest coverage and rejection of modified catalogue files."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from assembler.catalogue import CatalogueIntegrityError, load_release


def test_manifest_digests_match_files_on_disk(catalogue_dir):
    manifest = json.loads((catalogue_dir / "manifest.json").read_text())
    for relative_path, expected in manifest["artifacts"].items():
        artifact_path = catalogue_dir / relative_path
        actual = "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert actual == expected, f"{relative_path} digest mismatch"


def test_manifest_lists_every_json_artifact_except_itself(catalogue_dir):
    manifest = json.loads((catalogue_dir / "manifest.json").read_text())
    on_disk = {
        str(path.relative_to(catalogue_dir))
        for path in catalogue_dir.rglob("*.json")
        if path.name != "manifest.json"
    }
    assert set(manifest["artifacts"]) == on_disk


def test_load_release_refuses_a_tampered_artifact(catalogue_dir, tmp_path):
    _copy_catalogue(catalogue_dir, tmp_path)
    tampered = tmp_path / "schemas" / "envelope.schema.json"
    tampered.write_text(tampered.read_text() + "\n")

    with pytest.raises(CatalogueIntegrityError) as excinfo:
        load_release(catalogue_dir=tmp_path)
    assert excinfo.value.path == "schemas/envelope.schema.json"


def _copy_catalogue(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)
