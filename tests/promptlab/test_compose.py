"""Checks manifest loading and composition with synthetic prompt blocks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptlab.compose import compose
from promptlab.manifest import BlockRef, ManifestError, StageSpec, load_rung, units

FIXTURES = Path(__file__).parent / "fixtures"
EXTRACT_MANIFEST = FIXTURES / "rungs" / "synthetic-extract.json"
BIND_MANIFEST = FIXTURES / "rungs" / "synthetic-bind.json"


def test_load_rung_composes_both_stages_byte_for_byte():
    rung = load_rung(EXTRACT_MANIFEST, BIND_MANIFEST, root=FIXTURES)

    expected_extract = (FIXTURES / "composed" / "synthetic-extract.txt").read_text()
    expected_bind = (FIXTURES / "composed" / "synthetic-bind.txt").read_text()

    assert compose(rung.extract, root=FIXTURES) == expected_extract
    assert compose(rung.bind, root=FIXTURES) == expected_bind


def test_load_rung_reads_name_description_and_run_from_the_bind_file():
    rung = load_rung(EXTRACT_MANIFEST, BIND_MANIFEST, root=FIXTURES)

    assert rung.name == "synthetic"
    assert rung.description == "A minimal two-block bind stage for compose and load_rung tests."
    assert rung.run.capture == "synthetic-capture"
    assert rung.run.top_k == 3
    assert rung.run.seeds == (1, 2, 3)
    assert rung.run.model == "test-model"
    assert rung.run.thinking == {"extract": None, "bind": "minimal"}


def test_blank_separator_emits_a_blank_line_before_a_non_first_block():
    stage = StageSpec(
        stage="bind",
        blocks=(BlockRef(id="00-gamma"), BlockRef(id="01-delta", sep="blank")),
    )
    assert compose(stage, root=FIXTURES) == "Gamma block content.\n\nDelta block content.\n"


def test_line_separator_emits_a_single_newline_before_a_non_first_block():
    stage = StageSpec(
        stage="extract",
        blocks=(BlockRef(id="00-alpha"), BlockRef(id="01-beta", sep="line")),
    )
    assert compose(stage, root=FIXTURES) == "Alpha block content.\nBeta block content.\n"


def test_first_blocks_separator_is_ignored():
    # The first block's separator does not add a leading newline.
    stage = StageSpec(
        stage="extract",
        blocks=(BlockRef(id="00-alpha", sep="line"),),
    )
    assert compose(stage, root=FIXTURES) == "Alpha block content.\n"


def test_missing_block_raises_from_compose():
    stage = StageSpec(stage="extract", blocks=(BlockRef(id="99-missing"),))
    with pytest.raises(ManifestError, match="99-missing"):
        compose(stage, root=FIXTURES)


def test_missing_block_raises_from_load_rung(tmp_path):
    data = json.loads(EXTRACT_MANIFEST.read_text())
    data["blocks"].append({"id": "99-missing"})
    bad_manifest = tmp_path / "bad-extract.json"
    bad_manifest.write_text(json.dumps(data))

    with pytest.raises(ManifestError, match="99-missing"):
        load_rung(bad_manifest, BIND_MANIFEST, root=FIXTURES)


def test_trailing_blank_line_composes_identically_to_a_single_newline():
    with_blank_line = StageSpec(stage="extract", blocks=(BlockRef(id="00-alpha-blankline"),))
    without_blank_line = StageSpec(stage="extract", blocks=(BlockRef(id="00-alpha"),))
    assert compose(with_blank_line, root=FIXTURES) == compose(without_blank_line, root=FIXTURES)


def test_load_rung_rejects_a_manifest_missing_required_fields(tmp_path):
    bad_manifest = tmp_path / "bad-bind.json"
    bad_manifest.write_text(json.dumps({"name": "x", "stage": "bind", "blocks": []}))

    with pytest.raises(ManifestError):
        load_rung(EXTRACT_MANIFEST, bad_manifest, root=FIXTURES)


def test_load_rung_rejects_a_bind_file_missing_run(tmp_path):
    data = json.loads(BIND_MANIFEST.read_text())
    del data["run"]
    bad_manifest = tmp_path / "no-run-bind.json"
    bad_manifest.write_text(json.dumps(data))

    with pytest.raises(ManifestError):
        load_rung(EXTRACT_MANIFEST, bad_manifest, root=FIXTURES)


def test_load_rung_rejects_a_stage_field_that_does_not_match_its_position(tmp_path):
    # Reject the bind manifest in the extract position.
    with pytest.raises(ManifestError, match="stage must be 'extract'"):
        load_rung(BIND_MANIFEST, BIND_MANIFEST, root=FIXTURES)


def test_load_rung_rejects_a_name_mismatch_between_stage_files(tmp_path):
    data = json.loads(EXTRACT_MANIFEST.read_text())
    data["name"] = "a-different-name"
    mismatched_extract = tmp_path / "mismatched-extract.json"
    mismatched_extract.write_text(json.dumps(data))

    with pytest.raises(ManifestError, match="a-different-name"):
        load_rung(mismatched_extract, BIND_MANIFEST, root=FIXTURES)


def test_units_groups_by_unit_and_defaults_a_bare_block_to_its_own_id():
    stage = StageSpec(
        stage="bind",
        blocks=(
            BlockRef(id="00-gamma"),
            BlockRef(id="01-delta", unit="shared"),
            BlockRef(id="02-epsilon", unit="shared"),
        ),
    )
    grouped = units(stage)
    assert grouped["00-gamma"] == (BlockRef(id="00-gamma"),)
    assert grouped["shared"] == (
        BlockRef(id="01-delta", unit="shared"),
        BlockRef(id="02-epsilon", unit="shared"),
    )
