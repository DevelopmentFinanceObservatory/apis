"""Test claimed spans, including ambiguity, and the one-row `match_qualifiers` projection."""

from __future__ import annotations

from code_index.shared import get_index, get_measure_index
from search.engine.qualifier_match import (
    QualifierSpan,
    match_qualifiers,
    recognise_spans,
)

_REAL_INDEX = get_index()


def _by_text(spans: tuple[QualifierSpan, ...]) -> dict[str, QualifierSpan]:
    return {s.matched_text: s for s in spans}


# ---------------------------------------------------------------------------
# recognise_spans surfaces what match_qualifiers silently drops
# ---------------------------------------------------------------------------


def test_recognise_spans_surfaces_the_budget_support_ambiguity_match_qualifiers_drops():
    # The real index maps "budget support" to CRSCooperationModalityA and
    # CRSPurposeCode510. `recognise_spans` reports both; `match_qualifiers` refuses the tie.
    spans = _by_text(recognise_spans("ODA grants for budget support", _REAL_INDEX))
    budget_support = spans["budget support"]
    assert len(budget_support.rows) == 2
    assert {row.dcid for row in budget_support.rows} == {
        "CRSCooperationModalityA",
        "CRSPurposeCode510",
    }
    matches = match_qualifiers("ODA grants for budget support", _REAL_INDEX)
    assert "budget support" not in {m.matched_text for m in matches}


def test_recognise_spans_surfaces_the_grants_ambiguity_match_qualifiers_drops():
    # The index maps "grants" to both CRSTypeOfFinance100 and 110.
    spans = _by_text(recognise_spans("ODA grants for budget support", _REAL_INDEX))
    grants = spans["grants"]
    assert len(grants.rows) == 2
    assert {row.dcid for row in grants.rows} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}


def test_recognise_spans_surfaces_the_ta_acronym_ambiguity():
    # The index maps "TA" to both the modality root and CRSCooperationModalityD02.
    spans = _by_text(recognise_spans("financed through TA", _REAL_INDEX))
    ta = spans["TA"]
    assert len(ta.rows) == 2
    assert {row.dcid for row in ta.rows} == {"CRSCooperationModalityD", "CRSCooperationModalityD02"}


def test_recognise_spans_unambiguous_spans_carry_exactly_one_row():
    spans = _by_text(recognise_spans("ODA standard grants for health", _REAL_INDEX))
    assert len(spans["ODA"].rows) == 1
    assert spans["ODA"].rows[0].dcid == "CRSTypeOfFlow10"
    assert len(spans["standard grants"].rows) == 1
    assert spans["standard grants"].rows[0].dcid == "CRSTypeOfFinance110"
    assert len(spans["health"].rows) == 1
    assert spans["health"].rows[0].dcid == "CRSPurposeCode120"


def test_recognise_spans_bare_code_ambiguity_across_systems_stays_unclaimed():
    # The bare-code pass leaves ambiguous codes unclaimed. Label and synonym
    # matches remain claimed even when ambiguous.
    spans = recognise_spans("ODA grants for records coded exactly to code 110", _REAL_INDEX)
    assert "110" not in {s.matched_text for s in spans}


# ---------------------------------------------------------------------------
# match_qualifiers is recognise_spans' one-row projection
# ---------------------------------------------------------------------------


def test_match_qualifiers_is_the_one_row_projection_of_recognise_spans():
    query = "ODA grants for malaria control through modality D02"
    spans = recognise_spans(query, _REAL_INDEX)
    matches = match_qualifiers(query, _REAL_INDEX)
    one_row_spans = [s for s in spans if len(s.rows) == 1]
    assert len(one_row_spans) == len(matches)
    for span, match in zip(one_row_spans, matches, strict=True):
        assert span.matched_text == match.matched_text
        assert span.rows[0].dcid == match.code_dcid


def test_match_qualifiers_is_the_one_row_projection_with_a_measure_index():
    measure_index = get_measure_index()
    query = "grant equivalent"
    spans = recognise_spans(query, _REAL_INDEX, measure_index)
    matches = match_qualifiers(query, _REAL_INDEX, measure_index)
    one_row_spans = [s for s in spans if len(s.rows) == 1]
    assert len(one_row_spans) == len(matches) == 1
    assert one_row_spans[0].rows[0].dcid == matches[0].code_dcid == "grantEquivalent"


def test_recognise_spans_empty_query_yields_no_spans():
    assert recognise_spans("", _REAL_INDEX) == ()


def test_a_dataset_mention_owns_its_words_against_every_code_label():
    """ "imputed multilateral ODA" names the imputed table and is also the label of the aid
    type its rows carry. The mention owns the span, so no qualifier is recognised inside it,
    while "ODA" outside a mention still is."""
    text = "education ODA excluding imputed multilateral ODA"

    spans = recognise_spans(text, _REAL_INDEX, get_measure_index())

    assert [span.matched_text for span in spans] == ["education", "ODA"]
    assert all(span.end <= text.index("imputed") for span in spans)
