"""Test dcid, system-scoped code, label, and synonym lookups across 398 captured CRS codes."""

from __future__ import annotations

import json

import pytest

from code_index import CodeIndex

from .conftest import FIXTURE_PATH

_FIXTURE_CODES = json.loads(FIXTURE_PATH.read_text())["codes"]


def test_fixture_carries_all_398_captured_code_rows():
    assert len(_FIXTURE_CODES) == 398


@pytest.mark.parametrize("row", _FIXTURE_CODES, ids=lambda row: row["dcid"])
def test_every_captured_code_row_resolves_by_dcid_code_and_label(row, code_index: CodeIndex):
    by_dcid = code_index.code_by_dcid(row["dcid"])
    assert by_dcid is not None
    assert by_dcid.dcid == row["dcid"]
    assert by_dcid.code_system == row["code_system"]
    assert by_dcid.code == row["code"]
    assert by_dcid.label == row["label"]
    assert by_dcid.synonyms == tuple(row["synonyms"])
    assert by_dcid.parent == row["parent"]
    assert by_dcid.short_name == row["short_name"]

    by_code = code_index.code_by_code(row["code_system"], row["code"])
    assert by_code is not None
    assert by_code.dcid == row["dcid"]

    label_matches = code_index.codes_by_text(row["label"])
    assert any(candidate.dcid == row["dcid"] for candidate in label_matches)

    for synonym in row["synonyms"]:
        synonym_matches = code_index.codes_by_text(synonym)
        assert any(candidate.dcid == row["dcid"] for candidate in synonym_matches)


@pytest.mark.parametrize(
    ("dcid", "short_name"),
    [("CRSTypeOfFlow10", "ODA"), ("CRSTypeOfFlow21", "OOF"), ("CRSPurposeCode110", None)],
)
def test_short_name_is_set_for_flow_types_and_null_for_other_codes(
    dcid: str, short_name: str | None, code_index: CodeIndex
):
    row = code_index.code_by_dcid(dcid)
    assert row is not None
    assert row.short_name == short_name


def test_code_lookup_is_scoped_to_its_code_system(code_index: CodeIndex):
    # Code "110" occurs in both systems, so lookups must include the code system.
    purpose = code_index.code_by_code("crsPurpose", "110")
    finance = code_index.code_by_code("crsTypeOfFinance", "110")
    assert purpose is not None
    assert finance is not None
    assert purpose.dcid != finance.dcid
    assert purpose.dcid == "CRSPurposeCode110"
    assert finance.dcid == "CRSTypeOfFinance110"


def test_an_unknown_code_returns_none(code_index: CodeIndex):
    assert code_index.code_by_code("crsPurpose", "not-a-real-code") is None
    assert code_index.code_by_dcid("NotADcid") is None


def test_a_label_shared_across_code_systems_resolves_to_every_row_that_carries_it(
    code_index: CodeIndex,
):
    # Both rows carry "budget support" as a label in the captured data.
    matches = code_index.codes_by_text("budget support")
    dcids = {row.dcid for row in matches}
    assert dcids == {"CRSPurposeCode510", "CRSCooperationModalityA"}


def test_unmatched_text_returns_no_rows(code_index: CodeIndex):
    assert code_index.codes_by_text("no code carries this text") == ()
