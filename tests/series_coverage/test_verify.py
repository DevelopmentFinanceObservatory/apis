"""`series_coverage.verify.main`: the deploy-time gate that stops a deploy
when no compatible coverage artifact has been published. Exercises the
failure/success shapes `.github/workflows/deploy-dev.yml`'s verify step
depends on: no artifact published in this code's format root, a published
artifact this code refuses, and a real, loadable artifact.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from series_coverage.io import read_manifest, write_manifest
from series_coverage.schema import COVERAGE_FORMAT_VERSION
from series_coverage.verify import main
from tests.series_coverage.support import (
    build_dir,
    build_single_mask_artifact,
    build_test_artifact,
)

_COVERAGE_SLICE_URI = (
    f"file://{Path(__file__).resolve().parents[1] / 'fixtures' / 'coverage_slice'}"
)


def test_verify_succeeds_on_a_real_loadable_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([_COVERAGE_SLICE_URI])
    assert exit_code == 0
    out = capsys.readouterr()
    assert "loaded a compatible coverage artifact" in out.out
    assert "build '20260927T142252Z-slice'" in out.out
    assert out.err == ""


def test_verify_reads_the_artifact_uri_env_var_when_no_argument_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COVERAGE_ARTIFACT_URI", _COVERAGE_SLICE_URI)
    assert main([]) == 0


def test_verify_fails_loudly_with_no_argument_and_no_env_var(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("COVERAGE_ARTIFACT_URI", raising=False)
    exit_code = main([])
    assert exit_code == 1
    assert "COVERAGE_ARTIFACT_URI" in capsys.readouterr().err


def test_verify_fails_when_no_pointer_is_published(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = f"file://{tmp_path / 'nothing-here'}"
    exit_code = main([uri])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert uri in err
    assert "current.json" in err


def test_verify_fails_on_a_format_version_this_code_does_not_recognise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "coverage"
    uri = build_test_artifact(base)
    manifest_path = build_dir(base, "test-coverage-stamp-build") / "manifest.json"
    manifest = read_manifest(manifest_path)
    assert manifest.format_version == COVERAGE_FORMAT_VERSION
    write_manifest(replace(manifest, format_version=COVERAGE_FORMAT_VERSION + 1), manifest_path)

    exit_code = main([uri])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "format_version" in err


def test_verify_fails_on_a_single_mask_artifact_labelled_with_this_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = build_single_mask_artifact(tmp_path / "coverage")

    assert main([uri]) == 1
    err = capsys.readouterr().err
    assert "missing required field(s) ['epoch_year', 'mask_words']" in err


def test_verify_fails_when_only_another_format_root_is_published(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "coverage"
    uri = build_test_artifact(base)
    (base / f"format-{COVERAGE_FORMAT_VERSION}").rename(
        base / f"format-{COVERAGE_FORMAT_VERSION + 1}"
    )

    assert main([uri]) == 1
    assert "no current.json" in capsys.readouterr().err


def test_verify_rejects_more_than_one_argument(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["a", "b"])
    assert exit_code == 2
    assert "usage" in capsys.readouterr().err
