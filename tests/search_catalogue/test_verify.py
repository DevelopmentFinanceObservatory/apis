"""Pins `python -m search.catalogue.verify`: the exit status and outcome for a published build,
a pointer or build that does not load, a format root with nothing published, and no URI."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from artifact_publication import POINTER_FILE, VERSIONS_PREFIX
from search.catalogue import verify
from search.catalogue.artifact import BUILD_FILE, FORMAT_ROOT
from tests.search_catalogue.support import publish_catalogue


def _logged(capsys: pytest.CaptureFixture[str]) -> dict:
    (line,) = capsys.readouterr().out.splitlines()
    return json.loads(line)


def test_a_published_build_opens(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SEARCH_CATALOGUE_URI", tmp_path.as_uri())
    published = publish_catalogue(tmp_path.as_uri())

    assert verify.main() == verify.OPENED
    logged = _logged(capsys)
    assert logged["outcome"] == "opened"
    assert (logged["catalogue_version"], logged["key"]) == (
        published.catalogue_version,
        published.key,
    )
    assert logged["build_commit"] == published.inputs.build_commit


def _malformed_pointer(root: Path) -> None:
    (root / POINTER_FILE).write_text("{")


def _missing_build(root: Path) -> None:
    for build in (root / VERSIONS_PREFIX).iterdir():
        (build / BUILD_FILE).unlink()


def _unparsable_build(root: Path) -> None:
    for build in (root / VERSIONS_PREFIX).iterdir():
        (build / BUILD_FILE).write_text("{")


@pytest.mark.parametrize(
    "damage",
    [_malformed_pointer, _missing_build, _unparsable_build],
    ids=["malformed-pointer", "missing-build", "unparsable-build"],
)
def test_a_build_that_does_not_load_is_refused(
    tmp_path, monkeypatch, capsys, damage: Callable[[Path], None]
):
    monkeypatch.setenv("SEARCH_CATALOGUE_URI", tmp_path.as_uri())
    publish_catalogue(tmp_path.as_uri())
    damage(tmp_path / FORMAT_ROOT)

    assert verify.main() == verify.REFUSED
    logged = _logged(capsys)
    assert (logged["outcome"], logged["severity"]) == ("refused", "ERROR")
    assert logged["error"] == "CatalogueArtifactError"


def test_a_format_root_with_nothing_published_has_no_pointer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SEARCH_CATALOGUE_URI", tmp_path.as_uri())

    assert verify.main() == verify.NO_POINTER
    assert _logged(capsys)["outcome"] == "no_pointer"


def test_no_uri_exits_without_reading(monkeypatch, capsys):
    monkeypatch.delenv("SEARCH_CATALOGUE_URI", raising=False)

    assert verify.main() == verify.NO_URI
    assert _logged(capsys)["outcome"] == "no_uri"
