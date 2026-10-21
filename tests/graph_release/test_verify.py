"""Pins `python -m graph_release.verify`: the exit status and outcome for a release that opens
with its views, one whose views do not build, a format with no pointer, and no URI."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from graph_release import verify
from graph_release.format import Manifest
from tests.graph_release.support import build_test_release, build_view_release, sample_tables


def _logged(capsys: pytest.CaptureFixture[str]) -> dict:
    (line,) = capsys.readouterr().out.splitlines()
    return json.loads(line)


@pytest.mark.parametrize(
    ("publish", "status", "outcome"),
    [
        (build_view_release, verify.OPENED, "opened"),
        (lambda root: build_test_release(root, tables=sample_tables()), verify.REFUSED, "refused"),
        (lambda _root: None, verify.NO_POINTER, "no_pointer"),
    ],
    ids=["opens", "views-refused", "no-pointer"],
)
def test_verify_opens_the_pointers_release_with_its_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    publish: Callable[[str], Manifest | None],
    status: int,
    outcome: str,
):
    root = (tmp_path / "releases").as_uri()
    published = publish(root)
    monkeypatch.setenv("GRAPH_RELEASE_URI", root)

    assert verify.main() == status
    logged = _logged(capsys)
    assert logged["event"] == "graph_release_verify"
    assert logged["outcome"] == outcome
    if outcome == "opened":
        assert published is not None
        assert logged["release_id"] == published.release_id
        assert logged["severity"] == "INFO"
    else:
        assert logged["severity"] == "ERROR"
    if outcome == "refused":
        assert logged["error"] == "ViewRefused"


def test_verify_refuses_without_a_uri(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("GRAPH_RELEASE_URI", raising=False)

    assert verify.main() == verify.NO_URI
    assert _logged(capsys)["outcome"] == "no_uri"
