"""Test token-sequence code lookup: case, separators, and real cross-row and same-row
collisions."""

from __future__ import annotations

import pytest

from code_index import CodeIndex, CodeRow, default_index_path, load_index
from code_index.tokens import tokenize


def lookup(code_index: CodeIndex, text: str) -> tuple[CodeRow, ...]:
    return code_index.codes_by_tokens(tokenize(text))


def test_matches_regardless_of_case(code_index: CodeIndex):
    exact = code_index.code_by_dcid("CRSTypeOfFlow10")
    assert exact is not None
    assert exact.synonyms == ("ODA",)

    assert lookup(code_index, "ODA") == (exact,)
    assert lookup(code_index, "oda") == (exact,)
    assert lookup(code_index, "OdA") == (exact,)


@pytest.mark.parametrize(
    ("text", "dcids"),
    [
        ("ODA", ["CRSTypeOfFlow10"]),
        ("oda", ["CRSTypeOfFlow10"]),
        ("  Basic   Health Care ", ["CRSPurposeCode12220"]),
        # The label and a synonym of 600 fold alike, and the row appears once.
        ("ACTION RELATING TO DEBT", ["CRSPurposeCode600", "CRSPurposeCode60010"]),
        # Whole-text lookup reads no separator away.
        ("basic health, care", []),
    ],
)
def test_whole_text_lookup_folds_case_and_whitespace(
    code_index: CodeIndex, text: str, dcids: list[str]
):
    assert [row.dcid for row in code_index.codes_by_text(text)] == dcids


def test_ignores_surrounding_whitespace(code_index: CodeIndex):
    exact = code_index.code_by_dcid("CRSTypeOfFlow10")
    assert lookup(code_index, "  ODA  ") == (exact,)


def test_real_cross_row_case_collision_upper_secondary_education(code_index: CodeIndex):
    lower = code_index.code_by_dcid("CRSPurposeCode113")
    title = code_index.code_by_dcid("CRSPurposeCode11320")
    assert lower is not None
    assert title is not None
    assert lower.label == "upper secondary education"
    assert "Upper Secondary Education" in title.synonyms

    matches = lookup(code_index, "upper secondary education")
    assert {row.dcid for row in matches} == {"CRSPurposeCode113", "CRSPurposeCode11320"}


def test_real_cross_row_case_collision_food_assistance(code_index: CodeIndex):
    # The fixture has this text as a label on 520 and a synonym on 52010.
    matches = lookup(code_index, "Food Assistance")
    assert {row.dcid for row in matches} == {"CRSPurposeCode520", "CRSPurposeCode52010"}


def test_real_same_row_case_collision_does_not_duplicate_the_row(code_index: CodeIndex):
    # Deduplicate 600's label and synonym while retaining the distinct 60010 row.
    row_600 = code_index.code_by_dcid("CRSPurposeCode600")
    assert row_600 is not None
    assert row_600.label == "action relating to debt"
    assert "Action Relating to Debt" in row_600.synonyms

    matches = lookup(code_index, "action relating to debt")
    dcids = [row.dcid for row in matches]
    assert dcids.count("CRSPurposeCode600") == 1
    assert set(dcids) == {"CRSPurposeCode600", "CRSPurposeCode60010"}


def test_unmatched_text_returns_no_rows(code_index: CodeIndex):
    assert lookup(code_index, "no code carries this text") == ()


@pytest.mark.parametrize(
    ("text", "dcids"),
    [
        # The label word for word, its commas included.
        ("population policy, programme, and reproductive health", {"CRSPurposeCode130"}),
        # A query gap with no separator reads a label gap holding one.
        ("population policy programme and reproductive health", {"CRSPurposeCode130"}),
        ("Population policies/programmes and reproductive health", {"CRSPurposeCode130"}),
        ("population policies programmes and reproductive health", {"CRSPurposeCode130"}),
        ("waste management disposal", {"CRSPurposeCode14050"}),
        # A query separator reads only the same separator in the label.
        ("population policy; programme, and reproductive health", set()),
        ("waste management, disposal", set()),
        # A query separator matches no label gap that holds none.
        ("upper secondary, education", set()),
        # Separators before the first word and after the last are ignored.
        ("(Fuelwood/Charcoal)", {"CRSPurposeCode31261"}),
    ],
)
def test_separators_in_label_and_query(code_index: CodeIndex, text: str, dcids: set[str]):
    assert {row.dcid for row in lookup(code_index, text)} == dcids


@pytest.mark.parametrize(
    ("text", "dcids"),
    [
        # DFO convention: the bare phrase names the whole sector, and "care" narrows it to 13020.
        ("reproductive health", {"CRSPurposeCode130"}),
        ("reproductive health care", {"CRSPurposeCode13020"}),
    ],
)
def test_committed_index_reads_reproductive_health_as_sector_130(text: str, dcids: set[str]):
    assert {row.dcid for row in lookup(load_index(default_index_path()), text)} == dcids


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("HIV/AIDS", [("HIV", "word", 0), ("/", "sep", 3), ("AIDS", "word", 4)]),
        ("policy,  programme", [("policy", "word", 0), (",", "sep", 6), ("programme", "word", 9)]),
        # Inner hyphens and apostrophes stay in the word, and letters can be non-ASCII.
        ("non-concessional", [("non-concessional", "word", 0)]),
        ("Côte d'Ivoire", [("Côte", "word", 0), ("d'Ivoire", "word", 5)]),
        # A typographic apostrophe joins as a straight one does, and the word writes it so.
        ("Côte d\u2019Ivoire", [("Côte", "word", 0), ("d'Ivoire", "word", 5)]),
        (
            "(I.5)",
            [("(", "sep", 0), ("I", "word", 1), (".", "sep", 2), ("5", "word", 3), (")", "sep", 4)],
        ),
    ],
)
def test_tokenize_keeps_separators_with_offsets(text: str, tokens: list[tuple[str, str, int]]):
    found = tokenize(text)
    assert [(t.text, t.kind, t.start) for t in found] == tokens
    assert all(text[t.start : t.end].replace("\u2019", "'") == t.text for t in found)
