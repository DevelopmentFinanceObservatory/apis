"""`origin_catalogue.verify.main`: the deploy-time gate that stops a deploy
when no loadable origin catalogue has been published."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from origin_catalogue import STALE_AFTER, artifact
from origin_catalogue.schema import FORMAT_ROOT, UN_VARIABLES_FILE, UnCarryForward
from origin_catalogue.verify import main
from tests.origin_catalogue.support import BUILT_AT, publish_sample, sample_un_mirror


def _uri(tmp_path: Path) -> str:
    return f"file://{tmp_path / 'catalogue'}"


def test_verify_prints_the_build_row_counts_and_staleness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri)

    assert main([uri]) == 0

    out = capsys.readouterr()
    assert out.err == ""
    assert "build 'build-a'" in out.out
    assert f"built {BUILT_AT.isoformat()}" in out.out
    staleness = "stale" if datetime.now(UTC) - BUILT_AT > STALE_AFTER else "fresh"
    assert f"days ago, {staleness})" in out.out
    loaded = out.out.splitlines()[0]
    assert json.loads(loaded.split("row counts ", 1)[1]) == dict(build.manifest.row_counts)


def test_verify_prints_the_un_table_counts_from_one_read_of_each_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri, un_mirror=sample_un_mirror())
    reads: list[str] = []
    read = artifact._read
    monkeypatch.setattr(artifact, "_read", lambda uri: reads.append(uri) or read(uri))

    assert main([uri]) == 0

    assert any(Path(read).name == UN_VARIABLES_FILE for read in reads)
    assert len(reads) == len(set(reads))
    un_tables = capsys.readouterr().out.splitlines()[1]
    counts, rest = un_tables.split("row counts ", 1)[1].split(", ", 1)[0], un_tables
    assert '"un_variables": 2' in rest
    assert f"{build.manifest.un_mirror.resident_bytes} bytes in memory" in rest
    agencies = json.loads(rest.split("agencies ", 1)[1])
    assert agencies == {
        "who": {"series": 2, "variables": 3, "excluded_series": 1, "excluded_variables": 1}
    }
    assert counts.startswith("{")


def test_verify_fails_when_the_un_tables_do_not_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = _uri(tmp_path)
    build = publish_sample(uri, un_mirror=sample_un_mirror())
    build_dir = tmp_path / "catalogue" / FORMAT_ROOT / "versions" / build.manifest.build_id
    (build_dir / "un_variables.parquet").unlink()

    assert main([uri]) == 1
    assert "un_variables.parquet" in capsys.readouterr().err


def test_verify_notes_a_un_carry_forward(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = _uri(tmp_path)
    publish_sample(
        uri, un_carried_forward=UnCarryForward(from_build="build-0", reason="UN read failed")
    )

    assert main([uri]) == 0

    assert (
        "carries its UN rows, refused places and UN tables from build 'build-0': UN read failed"
        in capsys.readouterr().out
    )


def test_verify_reads_the_uri_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    monkeypatch.setenv("ORIGIN_CATALOGUE_URI", uri)

    assert main([]) == 0


def test_verify_fails_with_no_uri(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ORIGIN_CATALOGUE_URI", raising=False)

    assert main([]) == 1
    assert "ORIGIN_CATALOGUE_URI" in capsys.readouterr().err


def test_verify_fails_when_nothing_is_published(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = _uri(tmp_path)

    assert main([uri]) == 1
    err = capsys.readouterr().err
    assert uri in err
    assert "current.json" in err


def test_verify_applies_the_services_memory_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = _uri(tmp_path)
    publish_sample(uri)
    monkeypatch.setenv("ORIGIN_CATALOGUE_MAX_RESIDENT_BYTES", "1")

    assert main([uri]) == 1
    assert "budget" in capsys.readouterr().err


def test_verify_refuses_extra_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["a", "b"]) == 2
    assert "usage" in capsys.readouterr().err
