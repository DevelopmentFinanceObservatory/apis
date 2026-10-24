"""Test claimed spans per vocabulary, including ambiguity, and the one-row projections."""

from __future__ import annotations

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.types import CodeRow
from search.engine.qualifier_match import QualifierSpan, VocabularyReading, match_qualifiers
from tests.engine._index_slots import crs_vocabulary, read_text

_REAL_INDEX = get_index()
_CRS = crs_vocabulary()


def _by_text(spans: tuple[QualifierSpan, ...]) -> dict[str, QualifierSpan]:
    return {s.matched_text: s for s in spans}


def _crs(query: str, index: CodeIndex = _REAL_INDEX) -> VocabularyReading:
    """`query` read in a CRS shape's vocabulary."""
    return read_text(query, index, vocabularies=[_CRS]).by_vocabulary[_CRS]


def _with_dac2a_oda_grant() -> CodeIndex:
    """The fixture index with DAC2a aid type 201, "ODA grant", whose label covers "ODA
    grants" in every code system together. The row is added when the fixture index does
    not already hold it."""
    oda_grant = CodeRow(
        dcid="DAC2aAidType201",
        code_system="dac2aAidType",
        code="201",
        label="ODA grant",
        synonyms=(),
        parent=None,
    )
    held = _REAL_INDEX.code_by_dcid(oda_grant.dcid) is not None
    return CodeIndex(
        stamp=_REAL_INDEX.stamp,
        declarations=_REAL_INDEX.declarations.values(),
        codes=_REAL_INDEX.codes if held else (*_REAL_INDEX.codes, oda_grant),
        statvars=_REAL_INDEX.statvars,
        datasets=_REAL_INDEX.datasets.values(),
    )


# ---------------------------------------------------------------------------
# A vocabulary's spans keep the ambiguity its matches drop
# ---------------------------------------------------------------------------


def test_the_crs_vocabulary_reads_budget_support_as_a_tie_it_does_not_match():
    """The index maps "budget support" to CRSCooperationModalityA and CRSPurposeCode510,
    and to two DAC1 aid types. The CRS reading's span carries its two CRS rows and its
    matches leave the tie out, as the reading in every system does."""
    query = "ODA grants for budget support"
    reading = _crs(query)
    assert {row.dcid for row in _by_text(reading.spans)["budget support"].rows} == {
        "CRSCooperationModalityA",
        "CRSPurposeCode510",
    }
    assert "budget support" not in {m.matched_text for m in reading.matches}
    assert "budget support" not in {m.matched_text for m in match_qualifiers(query, _REAL_INDEX)}


@pytest.mark.parametrize(
    ("index", "claimed_everywhere"),
    [(_REAL_INDEX, ()), (_with_dac2a_oda_grant(), ("ODA grants",))],
    ids=["fixture", "with-dac2a-oda-grant"],
)
def test_the_crs_vocabulary_reads_grants_as_the_type_of_finance_tie(index, claimed_everywhere):
    """The CRS reading of "grants" is the CRSTypeOfFinance100/110 tie, also beside a DAC2a
    "ODA grant" label that claims "ODA grants" in every system together. The every-system
    reading of the fixture index is left unasserted, since it claims "ODA grants" once
    the fixture index declares DAC2a."""
    query = "ODA grants for budget support"
    everywhere = _by_text(read_text(query, index).everywhere.spans)
    for text in claimed_everywhere:
        assert text in everywhere
    reading = _crs(query, index)
    grants = _by_text(reading.spans)["grants"]
    assert {row.dcid for row in grants.rows} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}
    (tie,) = (t for t in reading.ties if t.matched_text == "grants")
    assert {row.code_dcid for row in tie.rows} == {"CRSTypeOfFinance100", "CRSTypeOfFinance110"}


def test_ta_reads_as_a_modality_tie_in_every_reading():
    # The index maps "TA" to both the modality root and CRSCooperationModalityD02.
    reading = read_text("financed through TA", _REAL_INDEX, vocabularies=[_CRS])
    for own in (reading.everywhere, reading.by_vocabulary[_CRS]):
        ta = _by_text(own.spans)["TA"]
        assert len(ta.rows) == 2
        assert {row.dcid for row in ta.rows} == {
            "CRSCooperationModalityD",
            "CRSCooperationModalityD02",
        }


def test_a_span_naming_one_code_of_a_vocabulary_carries_one_row():
    """ "ODA" names a code of each of three families' systems, so read in every system it
    carries all three, and the CRS reading carries the CRS flow type alone."""
    query = "ODA standard grants for health"
    everywhere = _by_text(read_text(query, _REAL_INDEX).everywhere.spans)
    assert {row.dcid for row in everywhere["ODA"].rows} == {
        "CRSTypeOfFlow10",
        "DAC1AidType1010",
        "DAC2aAidType206",
    }
    spans = _by_text(_crs(query).spans)
    assert {span: [row.dcid for row in spans[span].rows] for span in spans} == {
        "ODA": ["CRSTypeOfFlow10"],
        "standard grants": ["CRSTypeOfFinance110"],
        "health": ["CRSPurposeCode120"],
    }


def test_bare_code_ambiguity_across_systems_stays_unclaimed():
    # The bare-code pass leaves ambiguous codes unclaimed. Label and synonym
    # matches remain claimed even when ambiguous.
    reading = read_text(
        "ODA grants for records coded exactly to code 110", _REAL_INDEX, vocabularies=[_CRS]
    )
    for spans in (reading.everywhere.spans, reading.by_vocabulary[_CRS].spans):
        assert "110" not in {s.matched_text for s in spans}


# ---------------------------------------------------------------------------
# Matches are the spans' one-row projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "measure_index"),
    [
        ("ODA grants for malaria control through modality D02", None),
        ("grant equivalent", get_measure_index()),
    ],
)
def test_matches_are_the_one_row_projection_of_the_spans(query, measure_index):
    reading = read_text(query, _REAL_INDEX, measure_index, vocabularies=[_CRS])
    for own in (reading.everywhere, reading.by_vocabulary[_CRS]):
        one_row_spans = [s for s in own.spans if len(s.rows) == 1]
        assert [(s.matched_text, s.rows[0].dcid) for s in one_row_spans] == [
            (m.matched_text, m.code_dcid) for m in own.matches
        ]
    assert reading.everywhere.matches == match_qualifiers(query, _REAL_INDEX, measure_index)


def test_the_measure_value_is_one_row_in_every_reading():
    reading = read_text("grant equivalent", _REAL_INDEX, get_measure_index(), vocabularies=[_CRS])
    for own in (reading.everywhere, reading.by_vocabulary[_CRS]):
        assert [m.code_dcid for m in own.matches] == ["grantEquivalent"]


def test_an_empty_query_yields_no_spans():
    reading = read_text("", _REAL_INDEX, vocabularies=[_CRS])
    assert reading.everywhere.spans == reading.by_vocabulary[_CRS].spans == ()
