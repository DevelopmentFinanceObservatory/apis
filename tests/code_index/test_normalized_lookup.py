"""Test case-insensitive code lookup, including real cross-row and same-row collisions."""

from __future__ import annotations

from code_index import CodeIndex


def test_matches_regardless_of_case(code_index: CodeIndex):
    exact = code_index.code_by_dcid("CRSTypeOfFlow10")
    assert exact is not None
    assert exact.synonyms == ("ODA",)

    assert code_index.codes_by_normalized_text("ODA") == (exact,)
    assert code_index.codes_by_normalized_text("oda") == (exact,)
    assert code_index.codes_by_normalized_text("OdA") == (exact,)


def test_exact_lookup_stays_case_sensitive(code_index: CodeIndex):
    # Exact lookup remains case-sensitive.
    assert code_index.codes_by_text("oda") == ()
    assert len(code_index.codes_by_text("ODA")) == 1


def test_strips_surrounding_whitespace(code_index: CodeIndex):
    exact = code_index.code_by_dcid("CRSTypeOfFlow10")
    assert code_index.codes_by_normalized_text("  ODA  ") == (exact,)


def test_real_cross_row_case_collision_upper_secondary_education(code_index: CodeIndex):
    lower = code_index.code_by_dcid("CRSPurposeCode113")
    title = code_index.code_by_dcid("CRSPurposeCode11320")
    assert lower is not None
    assert title is not None
    assert lower.label == "upper secondary education"
    assert "Upper Secondary Education" in title.synonyms

    matches = code_index.codes_by_normalized_text("upper secondary education")
    assert {row.dcid for row in matches} == {"CRSPurposeCode113", "CRSPurposeCode11320"}


def test_real_cross_row_case_collision_food_assistance(code_index: CodeIndex):
    # The fixture has this text as a label on 520 and a synonym on 52010.
    matches = code_index.codes_by_normalized_text("Food Assistance")
    assert {row.dcid for row in matches} == {"CRSPurposeCode520", "CRSPurposeCode52010"}


def test_real_same_row_case_collision_does_not_duplicate_the_row(code_index: CodeIndex):
    # Deduplicate 600's label and synonym while retaining the distinct 60010 row.
    row_600 = code_index.code_by_dcid("CRSPurposeCode600")
    assert row_600 is not None
    assert row_600.label == "action relating to debt"
    assert "Action Relating to Debt" in row_600.synonyms

    matches = code_index.codes_by_normalized_text("action relating to debt")
    dcids = [row.dcid for row in matches]
    assert dcids.count("CRSPurposeCode600") == 1
    assert set(dcids) == {"CRSPurposeCode600", "CRSPurposeCode60010"}


def test_unmatched_text_returns_no_rows(code_index: CodeIndex):
    assert code_index.codes_by_normalized_text("no code carries this text") == ()
