"""Test text and code matching against the fixture code index.

Synthetic indexes cover code collisions and token cases absent from the CRS
corpus. The smaller test index lacks synonyms needed by the grant cases.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from code_index.declarations import MEASURE_WORD_SYNONYMS_KEPT
from code_index.query import CodeIndex
from code_index.shared import get_index, get_measure_index
from code_index.tokens import tokenize, word_tokens
from code_index.types import (
    CodeRow,
    CodeSystemDeclaration,
    IndexStamp,
    MeasureIndex,
    MeasureValueRow,
    StatVarRow,
)
from search.engine.qualifier_match import (
    QualifierMatch,
    QualifierTie,
    VocabularyReading,
    _all_normalized_text_ranges,
    _all_system_markers,
    _max_phrase_words,
    match_qualifier_readings,
    match_qualifier_ties,
    match_qualifiers,
    match_unsupported_qualifier_phrases,
    qualifier_occurrences,
    read_by_vocabulary,
)
from tests.engine._index_slots import crs_vocabulary, read_text

# ---------------------------------------------------------------------------
# The fixture index, the code index the session release serves.
# ---------------------------------------------------------------------------

_REAL_INDEX = get_index()
_CRS = crs_vocabulary()
# The vocabularies of the DAC1 flow shape and the imputed multilateral shape, as the
# committed index's families declare them.
_DAC1 = frozenset({"dac1AidType", "dacTypeOfFinance", "financialFlowType", "nettingBasis"})
_IMPUTED = frozenset(
    {"crsPurpose", "crsTypeOfFlow", "dac2aAidType", "financialFlowType", "nettingBasis"}
)


def _read(query: str, vocabulary: frozenset[str] = _CRS) -> VocabularyReading:
    """`query` read in `vocabulary`, a CRS shape's by default, as the prune step reads it."""
    return read_text(query, _REAL_INDEX, vocabularies=[vocabulary]).by_vocabulary[vocabulary]


def _matches(query: str) -> tuple[QualifierMatch, ...]:
    """The matches of `query` with a CRS shape in play, as the prune step reads them."""
    return _read(query).matches


def _by_system(matches: tuple[QualifierMatch, ...]) -> dict[str, QualifierMatch]:
    return {m.code_system: m for m in matches}


# ---------------------------------------------------------------------------
# The six-phrase regression
# ---------------------------------------------------------------------------


def test_plain_health_query_recovers_no_finance_or_modality_qualifier():
    # "health" is the CRSPurposeCode120 label; ODA is a flow synonym.
    # The phrase contains no finance-type or modality term.
    by_system = _by_system(_matches("ODA grants for health"))
    assert set(by_system) == {"crsTypeOfFlow", "crsPurpose"}
    oda = by_system["crsTypeOfFlow"]
    assert oda.code_dcid == "CRSTypeOfFlow10"
    assert oda.source == "synonym"
    assert oda.matched_text == "ODA"
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode120"


def test_plain_malaria_query_recovers_no_finance_or_modality_qualifier():
    by_system = _by_system(_matches("ODA grants for malaria control"))
    assert set(by_system) == {"crsTypeOfFlow", "crsPurpose"}
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode12262"


def test_standard_grants_recovers_the_finance_type_qualifier():
    by_system = _by_system(_matches("ODA standard grants for health"))
    finance = by_system["crsTypeOfFinance"]
    assert finance.code_dcid == "CRSTypeOfFinance110"
    assert finance.label == "standard grant"
    assert finance.source == "label"
    # The plural allowance maps "standard grants" to the registered
    # singular label "standard grant"; matched_text keeps the query form.
    assert finance.matched_text == "standard grants"
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode120"
    assert by_system["crsTypeOfFlow"].code_dcid == "CRSTypeOfFlow10"


def test_modality_d02_recovers_the_modality_qualifier():
    by_system = _by_system(_matches("ODA grants for malaria control through modality D02"))
    modality = by_system["crsCooperationModality"]
    assert modality.code_dcid == "CRSCooperationModalityD02"
    assert modality.source == "code"
    assert modality.matched_text == "D02"
    assert "crsTypeOfFinance" not in by_system


def test_purpose_code_120_recovers_the_exact_code_qualifier():
    by_system = _by_system(_matches("ODA grants for records coded exactly to purpose code 120"))
    purpose = by_system["crsPurpose"]
    assert purpose.code_dcid == "CRSPurposeCode120"
    assert purpose.source == "code"
    assert purpose.matched_text == "120"
    assert "crsTypeOfFinance" not in by_system


def test_entities_only_query_is_the_control_and_carries_only_the_flow_slot():
    # ODA matches the flow synonym. "Grants" maps to two finance rows and
    # stays ambiguous; France and Togo add no qualifier.
    by_system = _by_system(_matches("ODA grants from France to Togo"))
    assert set(by_system) == {"crsTypeOfFlow"}
    assert by_system["crsTypeOfFlow"].code_dcid == "CRSTypeOfFlow10"


def test_bare_grants_alone_never_matches_a_finance_type():
    for query in (
        "ODA grants for health",
        "ODA grants for malaria control",
        "ODA grants from France to Togo",
    ):
        by_system = _by_system(_matches(query))
        assert "crsTypeOfFinance" not in by_system, query


def test_a_no_vocabulary_term_matches_only_as_a_whole_word():
    terms = frozenset({"non-concessional"})
    assert match_unsupported_qualifier_phrases("non-concessional aid", terms=terms) == (
        "non-concessional",
    )
    assert match_unsupported_qualifier_phrases("non-concessionality aid", terms=terms) == ()
    assert match_unsupported_qualifier_phrases("non-concessional-like aid", terms=terms) == ()


# ---------------------------------------------------------------------------
# Matches come back in query order (not the width-descending search order)
# ---------------------------------------------------------------------------


def test_matches_are_returned_in_query_order_not_search_order():
    # Search tries the widest span first; results retain query order.
    matches = _matches("ODA standard grants for health")
    assert [m.matched_text for m in matches] == ["ODA", "standard grants", "health"]


# ---------------------------------------------------------------------------
# Ordinary words must match a registered form or its allowed plural
# ---------------------------------------------------------------------------


def test_ordinary_words_that_merely_start_with_a_registered_text_do_not_match():
    # Each word starts with a label or synonym, but matches neither the
    # registered form nor its permitted plural.
    for query in (
        "healthcare spending in Kenya",
        "coalition building in fragile states",
        "median income by district",
        "golden rice adoption",
        "granting agencies in europe",
    ):
        assert _matches(query) == (), query


def test_grants_is_refused_while_standard_grants_resolves():
    # "grants" maps to two rows; "standard grants" matches one plural form.
    assert "crsTypeOfFinance" not in _by_system(_matches("ODA grants from France to Togo"))
    finance = _by_system(_matches("ODA standard grants for health"))["crsTypeOfFinance"]
    assert finance.code_dcid == "CRSTypeOfFinance110"


def test_unaffected_exact_matches_still_resolve():
    # Registered texts that are already the same word as the query's own
    # span still resolve directly, with no plural allowance involved.
    assert (
        _by_system(_matches("roads maintenance"))["crsPurpose"].code_dcid == "CRSPurposeCode21020"
    )
    assert (
        _by_system(_matches("statistics office"))["crsPurpose"].code_dcid == "CRSPurposeCode16062"
    )


def test_plural_allowance_resolves_a_genuine_singular_label():
    # "elections" is registered nowhere; "election" is, as the label of
    # exactly one row -- the plural allowance is what recovers it.
    match = _by_system(_matches("elections monitoring"))["crsPurpose"]
    assert match.code_dcid == "CRSPurposeCode15151"
    assert match.label == "election"
    assert match.matched_text == "elections"


# ---------------------------------------------------------------------------
# Real corpus: ambiguity the real data already exposes
# ---------------------------------------------------------------------------


def test_label_present_in_two_real_code_systems_stays_unmatched():
    # "budget support" is a real collision: registered as the label on both
    # CRSCooperationModalityA and CRSPurposeCode510, two systems one CRS shape declares. A
    # wrong guess here is worse than none -- left to the binder.
    rows = _REAL_INDEX.codes_by_tokens(tokenize("budget support"))
    assert len([row for row in rows if row.code_system in _CRS]) == 2
    by_system = _by_system(_matches("ODA grants for budget support"))
    assert "crsPurpose" not in by_system
    assert "crsCooperationModality" not in by_system


def test_ta_acronym_present_in_two_real_code_systems_stays_unmatched():
    # "TA" is registered on both CRSCooperationModalityD (the root) and
    # CRSCooperationModalityD02 -- proof that the ambiguity call alone,
    # with no acronym notion layered on top, correctly refuses even a
    # clean acronym when it collides.
    assert len(_REAL_INDEX.codes_by_tokens(tokenize("TA"))) == 2
    by_system = _by_system(_matches("financed through TA"))
    assert "crsCooperationModality" not in by_system


def test_bare_code_without_a_marker_word_stays_unmatched():
    # "D02" alone, with no code system's own vocabulary named anywhere
    # earlier in the query, is never looked up as a code at all -- a
    # character shape ("D02" mixes a letter and a digit) is not, on its
    # own, grounds to trust a bare token.
    assert _matches("financed through D02") == ()


def test_bare_code_shared_by_two_systems_stays_unmatched_with_no_marker():
    # "110" is a real code string shared by CRSPurposeCode110 and
    # CRSTypeOfFinance110 -- naming it with no code system's vocabulary
    # present leaves it ineligible for lookup in either.
    by_system = _by_system(_matches("ODA grants for records coded exactly to code 110"))
    assert "crsPurpose" not in by_system
    assert "crsTypeOfFinance" not in by_system


def test_code_colliding_across_systems_resolves_under_its_own_systems_marker():
    # Naming a system's vocabulary limits lookup to that system, where 110 is unique.
    purpose = _by_system(_matches("ODA grants for records coded exactly to purpose code 110"))
    assert purpose["crsPurpose"].code_dcid == "CRSPurposeCode110"
    assert "crsTypeOfFinance" not in purpose

    finance = _by_system(
        _matches("ODA grants for records coded exactly to type of finance code 110")
    )
    assert finance["crsTypeOfFinance"].code_dcid == "CRSTypeOfFinance110"
    assert "crsPurpose" not in finance


def test_code_colliding_across_systems_stays_refused_once_crs_is_no_marker():
    # "crs" appears in every system name, so it identifies none of them and
    # cannot make 110 eligible for lookup.
    by_system = _by_system(_matches("ODA grants for records coded exactly to crs code 110"))
    assert "crsPurpose" not in by_system
    assert "crsTypeOfFinance" not in by_system


def test_year_shaped_number_matching_a_real_code_stays_unmatched_when_unmarked():
    # 1100 is both a real finance code and a plausible year. Without a system
    # marker, the numeric token remains unclaimed.
    by_system = _by_system(_matches("ODA grants for health in 1100"))
    assert "crsTypeOfFinance" not in by_system


def test_year_shaped_number_matches_only_when_its_systems_vocabulary_is_named():
    # "Type of finance" marks the system, so the same 4-digit code resolves.
    by_system = _by_system(
        _matches("ODA grants for the guarantee coded as type of finance code 1100")
    )
    assert by_system["crsTypeOfFinance"].code_dcid == "CRSTypeOfFinance1100"


# ---------------------------------------------------------------------------
# "of" and "type" are spelled out inside both crsTypeOfFinance's and
# crsTypeOfFlow's own names -- neither word may leak into either system's
# marker words just because it happens to be grammatical filler shared by
# the two names
# ---------------------------------------------------------------------------


def test_ordinary_preposition_or_filler_word_does_not_reach_either_system():
    # "Of" and "type" are shared by both system names; alone, they mark neither.
    for query in (
        "amount of aid coded 110",
        "share of total assistance coded 1100",
        "what type of support was 110",
    ):
        by_system = _by_system(_matches(query))
        assert "crsTypeOfFinance" not in by_system, query
        assert "crsTypeOfFlow" not in by_system, query


def test_each_systems_own_distinctive_word_still_reaches_it():
    # "Finance" and "flow" uniquely mark their respective systems.
    assert _by_system(_matches("purpose code 12220"))["crsPurpose"].code_dcid == (
        "CRSPurposeCode12220"
    )
    assert _by_system(_matches("type of finance 110"))["crsTypeOfFinance"].code_dcid == (
        "CRSTypeOfFinance110"
    )
    assert _by_system(_matches("flow type 10"))["crsTypeOfFlow"].code_dcid == "CRSTypeOfFlow10"
    assert _by_system(_matches("cooperation modality C01"))["crsCooperationModality"].code_dcid == (
        "CRSCooperationModalityC01"
    )
    assert _by_system(_matches("aid with purpose 12220"))["crsPurpose"].code_dcid == (
        "CRSPurposeCode12220"
    )
    assert _by_system(_matches("modality D02"))["crsCooperationModality"].code_dcid == (
        "CRSCooperationModalityD02"
    )


# ---------------------------------------------------------------------------
# A span must not bridge two tokens across punctuation the tokenizer
# discards -- two clauses separated by a period or comma are not a phrase
# ---------------------------------------------------------------------------


def test_span_does_not_bridge_a_sentence_boundary():
    # The period separates the words, so they cannot form "standard grants".
    assert _matches("This was a standard. Grants like these are common") == ()


def test_span_does_not_bridge_a_comma():
    assert _matches("the loan was standard, grants are separate") == ()


@pytest.mark.parametrize(
    ("needle", "haystack", "ranges"),
    [
        ("aid", "Straße AID", [(7, 10)]),
        # "ß" folds to two characters, and the range covers the one it came from.
        ("strasse aid", "STRAßE aid", [(0, 10)]),
        ("health", "health and HEALTH", [(0, 6), (11, 17)]),
    ],
)
def test_text_ranges_are_offsets_in_the_haystack_itself(needle, haystack, ranges):
    assert _all_normalized_text_ranges(needle, haystack) == ranges


@pytest.mark.parametrize(
    ("query", "dcid", "reads_code"),
    [
        (
            "ODA for population policy, programme, and reproductive health",
            "CRSPurposeCode130",
            True,
        ),
        # 14050's synonym "Waste management/disposal" holds "/", the query ",". No code reads
        # "waste management" or "disposal" alone, so only the separator rule decides.
        ("ODA for waste management/disposal", "CRSPurposeCode14050", True),
        ("ODA for waste management, disposal", "CRSPurposeCode14050", False),
        # Whitespace is no token, so a run of spaces keeps the words of a label adjacent.
        (
            "ODA   for   population   policy,   programme,   and   reproductive   health",
            "CRSPurposeCode130",
            True,
        ),
    ],
)
def test_occurrences_read_the_separators_the_scan_reads(query: str, dcid: str, reads_code: bool):
    # The response cache keys on these records, so they follow the scan's separator rule.
    records = qualifier_occurrences(query, _REAL_INDEX, get_measure_index())
    found = any(
        isinstance(rows := r.get("rows"), list) and ["crsPurpose", dcid] in rows for r in records
    )
    assert found is reads_code
    assert any(m.code_dcid == dcid for m in _matches(query)) is reads_code


def test_span_does_not_bridge_a_separator_its_label_lacks():
    # The label's gap holds "/", the query's ",".
    dcids = {m.code_dcid for m in _matches("ODA for waste management, disposal")}
    assert "CRSPurposeCode14050" not in dcids


@pytest.mark.parametrize(
    ("query", "dcid", "matched_text"),
    [
        # Whitespace preserves adjacency between phrase words.
        ("ODA   standard   grants   for   health", "CRSTypeOfFinance110", "standard   grants"),
        # A label's separators read the same separators in the query, or none.
        (
            "gross disbursements for population policy, programme, and reproductive health",
            "CRSPurposeCode130",
            "population policy, programme, and reproductive health",
        ),
        (
            "population policy programme and reproductive health ODA",
            "CRSPurposeCode130",
            "population policy programme and reproductive health",
        ),
        (
            "ODA for STD control including HIV/AIDS in Kenya",
            "CRSPurposeCode13040",
            "STD control including HIV/AIDS",
        ),
        ("ODA for waste management/disposal", "CRSPurposeCode14050", "waste management/disposal"),
    ],
)
def test_a_span_reads_the_separators_its_label_holds(query: str, dcid: str, matched_text: str):
    (match,) = [m for m in _matches(query) if m.code_dcid == dcid]
    assert match.matched_text == matched_text
    # A variable's match carries its offsets in the text read.
    assert match.start is not None
    assert query[match.start : match.end] == matched_text
    # Offsets take no part in a match's equality.
    assert match == replace(match, start=None, end=None)
    # A match of other text carries no offsets.
    assert all(m.start is None and m.end is None for m in match_qualifiers(query, _REAL_INDEX))


# ---------------------------------------------------------------------------
# Synthetic index: cases the real corpus cannot expose
# ---------------------------------------------------------------------------


def _synthetic_index(*rows: CodeRow, code_system: str = "syntheticAxis") -> CodeIndex:
    """Build an in-memory index with one non-hierarchical system."""
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="synthetic",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={code_system: len(rows)},
            statvar_shape_counts={},
            content_hash="synthetic",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system=code_system,
                hierarchical=False,
                disjoint_from_children=True,
                summable=True,
                absent_reading="not_declared",
                plain_name="axis",
            ),
        ),
        codes=rows,
        statvars=(),
    )


def test_bare_year_colliding_with_a_synthetic_code_still_needs_the_marker():
    # The unique code "2020" still needs the system's "synthetic" or "axis" marker.
    index = _synthetic_index(
        CodeRow(
            dcid="SyntheticAxisCode2020",
            code_system="syntheticAxis",
            code="2020",
            label="vintage two thousand twenty",
            synonyms=(),
            parent=None,
        )
    )
    unmarked = match_qualifiers("spending reported for 2020", index)
    assert unmarked == ()
    marked = match_qualifiers("spending reported for synthetic axis code 2020", index)
    assert len(marked) == 1
    assert marked[0].code_dcid == "SyntheticAxisCode2020"


def test_ambiguous_short_word_does_not_fall_back_from_a_longer_candidate():
    # The ambiguous short word "cash" must not override the longer plural match.
    index = _synthetic_index(
        CodeRow(
            dcid="SyntheticAxisCashA",
            code_system="syntheticAxis",
            code="A",
            label="cash",
            synonyms=(),
            parent=None,
        ),
        CodeRow(
            dcid="SyntheticAxisCashB",
            code_system="syntheticAxis",
            code="B",
            label="petty cash",
            synonyms=("cash",),
            parent=None,
        ),
        CodeRow(
            dcid="SyntheticAxisCashTransfer",
            code_system="syntheticAxis",
            code="C",
            label="cash transfer",
            synonyms=(),
            parent=None,
        ),
    )
    # The bare word alone is genuinely ambiguous (two rows).
    assert len(index.codes_by_tokens(tokenize("cash"))) == 2
    assert match_qualifiers("financed by cash alone", index) == ()
    # The longer phrase is registered nowhere verbatim, so it falls to the
    # plural allowance ("cash transfer" plus a trailing "s") and resolves
    # unambiguously -- with no fallback attempted on "cash".
    matches = match_qualifiers("financed by cash transfers to households", index)
    assert len(matches) == 1
    assert matches[0].code_dcid == "SyntheticAxisCashTransfer"
    assert matches[0].source == "label"
    assert matches[0].matched_text == "cash transfers"


def test_bare_root_letter_code_needs_the_marker_even_when_unambiguous():
    # A single-letter code needs a system marker, even when the code is unique.
    index = _synthetic_index(
        CodeRow(
            dcid="SyntheticAxisRootA",
            code_system="syntheticAxis",
            code="A",
            label="root category a",
            synonyms=(),
            parent=None,
        )
    )
    assert match_qualifiers("a grant for schools", index) == ()
    marked = match_qualifiers("a grant coded as axis code A", index)
    assert len(marked) == 1
    assert marked[0].code_dcid == "SyntheticAxisRootA"


def test_mixed_alnum_code_still_needs_a_marker():
    # A letter-and-digit shape ("D02") is no longer trusted on its own --
    # it needs the same marker-word signal as any other bare code.
    index = _synthetic_index(
        CodeRow(
            dcid="SyntheticAxisD02",
            code_system="syntheticAxis",
            code="D02",
            label="delivery mode two",
            synonyms=(),
            parent=None,
        )
    )
    assert match_qualifiers("financed through mode D02", index) == ()
    matches = match_qualifiers("financed through synthetic axis mode D02", index)
    assert len(matches) == 1
    assert matches[0].code_dcid == "SyntheticAxisD02"
    assert matches[0].source == "code"


def test_empty_query_matches_nothing():
    assert match_qualifiers("", _REAL_INDEX) == ()
    assert match_qualifiers("   ", _REAL_INDEX) == ()


# ---------------------------------------------------------------------------
# Measure values: a `MeasureValueRow`, offered via an optional `measure_index`
# argument, is reachable by the same label/synonym pass as a `CodeRow` --
# see qualifier_match.py's own module docstring and `_row_system`.
# ---------------------------------------------------------------------------


def test_measure_word_matches_with_its_axis_as_code_system():
    matches = match_qualifiers("net", _REAL_INDEX, get_measure_index())
    assert len(matches) == 1
    assert matches[0].code_system == "nettingBasis"
    assert matches[0].code_dcid == "Net"


def test_measure_value_plural_reaches_its_singular_registered_label():
    matches = match_qualifiers("commitments", _REAL_INDEX, get_measure_index())
    assert len(matches) == 1
    assert matches[0].code_system == "financialObligationType"
    assert matches[0].code_dcid == "Commitment"


def test_gross_and_nominal_measure_words_are_unambiguous_against_the_real_index():
    gross_matches = match_qualifiers("gross", _REAL_INDEX, get_measure_index())
    assert len(gross_matches) == 1
    assert gross_matches[0].code_dcid == "Gross"

    nominal_matches = match_qualifiers("nominal GDP", _REAL_INDEX, get_measure_index())
    by_system = _by_system(nominal_matches)
    assert by_system["measurementQualifier"].code_dcid == "Nominal"


@pytest.mark.parametrize(
    "query",
    [
        "real GDP of Zambia",
        "GDP of Zambia in real terms",
        "GDP of Zambia at constant prices",
        "GDP of Zambia, constant prices",
        "GDP of Zambia at constant 2015 prices",
        "GDP of Zambia in constant US dollars",
        "inflation-adjusted GDP of Zambia",
        "GDP of Zambia adjusted for inflation",
        "inflation adjusted GDP of Zambia",
    ],
)
def test_each_real_price_basis_phrasing_names_the_real_value_reading(query: str):
    # "inflation adjusted" is InflationAdjusted's own label, so it resolves to the canonical row.
    by_system = _by_system(match_qualifiers(query, _REAL_INDEX, get_measure_index()))
    assert by_system["measurementQualifier"].code_dcid == "RealValue"


def test_purchasing_power_parity_names_its_price_basis():
    matches = match_qualifiers(
        "GDP purchasing power parity Kenya", _REAL_INDEX, get_measure_index()
    )
    assert _by_system(matches)["measurementQualifier"].code_dcid == "PurchasingPowerParity"


def test_ppp_is_one_match_with_a_code_and_a_measure_reading():
    # The graph lists "PPP" (public-private partnership) as a synonym of CRSPurposeCode25010.
    # Text cannot choose, so `match_qualifiers` refuses the span and the readings carry both.
    assert match_qualifiers("GDP PPP of Kenya", _REAL_INDEX, get_measure_index()) == ()
    (readings,) = match_qualifier_readings("GDP PPP of Kenya", _REAL_INDEX, get_measure_index())
    assert (readings.code.code_system, readings.code.code_dcid) == (
        "crsPurpose",
        "CRSPurposeCode25010",
    )
    assert (readings.measure.code_system, readings.measure.code_dcid) == (
        "measurementQualifier",
        "PurchasingPowerParity",
    )


def _published_vocabulary(provenance: str) -> frozenset[str]:
    """Every property a family `provenance` publishes carries."""
    return frozenset(
        prop
        for families in _REAL_INDEX.families_by_five_tuple.values()
        for family in families
        if provenance in family.datasets
        for prop in family.properties
    )


@pytest.mark.parametrize(
    "provenance",
    ["provenance/OECDCreditorReportingSystem", "provenance/OECDDAC1", "provenance/OECDDAC2A"],
)
@pytest.mark.parametrize(
    ("query", "basis"),
    [("gross ODA", "Gross"), ("net ODA", "Net"), ("net debt relief", "Net")],
)
def test_a_basis_word_before_a_code_text_reads_the_netting_basis(
    query: str, basis: str, provenance: str
):
    """A code synonym holding the basis word ("Gross ODA", "Net debt relief") would claim the
    whole span and leave the basis unread in every vocabulary."""
    vocabulary = _published_vocabulary(provenance)
    reading = read_text(
        query, _REAL_INDEX, get_measure_index(), vocabularies=[vocabulary]
    ).by_vocabulary[vocabulary]

    assert ("nettingBasis", basis) in {(m.code_system, m.code_dcid) for m in reading.matches}


@pytest.mark.parametrize(
    ("query", "basis"), [("gross ODA loans", "Gross"), ("net ODA loans", "Net")]
)
def test_a_basis_word_reads_beside_the_code_text_left_without_it(query: str, basis: str):
    """ "Gross ODA Loans" on DAC2aAidType218 is exported as "ODA Loans", so the code reads in
    the remainder and the basis in its own word."""
    vocabulary = _published_vocabulary("provenance/OECDDAC2A")
    reading = read_text(
        query, _REAL_INDEX, get_measure_index(), vocabularies=[vocabulary]
    ).by_vocabulary[vocabulary]

    assert {(m.code_system, m.code_dcid) for m in reading.matches} >= {
        ("nettingBasis", basis),
        ("dac2aAidType", "DAC2aAidType218"),
    }


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ("ODA grant equivalents of bilateral loans", "DAC1AidType11017"),
        ("ODA grant equivalents of debt relief operations", "DAC1AidType11026"),
        ("grant equivalents of debt relief", "DAC1AidType11026"),
        ("ODA grant equivalents of private sector instruments", "DAC1AidType11040"),
    ],
)
def test_a_synonym_naming_a_measure_its_code_implies_reads_the_code(query: str, code: str):
    """Every series of these codes is a grant equivalent, so their synonyms keep the measure
    words and a question in DAC1's wording reads the code."""
    vocabulary = _published_vocabulary("provenance/OECDDAC1")
    reading = read_text(
        query, _REAL_INDEX, get_measure_index(), vocabularies=[vocabulary]
    ).by_vocabulary[vocabulary]

    assert ("dac1AidType", code) in {(m.code_system, m.code_dcid) for m in reading.matches}


_SHAPE_MEASURE_FIELDS = {
    "populationType": "population_type",
    "measuredProperty": "measured_property",
    "measurementQualifier": "measurement_qualifier",
}


def _carries(statvar: StatVarRow, value: MeasureValueRow) -> bool:
    if value.axis in _SHAPE_MEASURE_FIELDS:
        return getattr(statvar, _SHAPE_MEASURE_FIELDS[value.axis]) == value.dcid
    return statvar.constraint_properties.get(value.axis) == value.dcid


def test_no_code_synonym_takes_a_measure_value_its_code_does_not_imply():
    """A code synonym the matcher reads whole, holding a measure value's words, would take
    them from the measure reading in every vocabulary
    (`code_index.export._strip_measure_words`). It may hold them only when every
    series carrying its code has that measure, as DAC1AidType11017's "ODA grant equivalents
    of bilateral loans" does. A synonym that is itself a measure text ("PPP") keeps both
    readings on its one span, and a declared one names no measure."""
    synonyms = sorted(
        text
        for text in {text for row in _REAL_INDEX.codes for text in row.synonyms}
        if text.casefold() not in MEASURE_WORD_SYNONYMS_KEPT
    )
    together = read_by_vocabulary(synonyms, _REAL_INDEX, get_measure_index(), vocabularies=())
    measures_alone = read_by_vocabulary(
        synonyms, _synthetic_index(), get_measure_index(), vocabularies=()
    )
    measure_spans = {
        text: measure_reading.everywhere.spans
        for text, reading, measure_reading in zip(synonyms, together, measures_alone, strict=True)
        for span in reading.everywhere.spans
        if span.start == 0
        and span.end == len(text)
        and not any(isinstance(row, MeasureValueRow) for row in span.rows)
    }
    series_by_code: dict[str, list[StatVarRow]] = {}
    for statvar in _REAL_INDEX.statvars:
        for value in statvar.constraint_properties.values():
            series_by_code.setdefault(value, []).append(statvar)

    taking = [
        (row.dcid, text)
        for row in _REAL_INDEX.codes
        for text in row.synonyms
        for span in measure_spans.get(text, ())
        if not series_by_code.get(row.dcid)
        or not all(
            any(
                _carries(statvar, value)
                for value in span.rows
                if isinstance(value, MeasureValueRow)
            )
            for statvar in series_by_code[row.dcid]
        )
    ]

    assert measure_spans
    assert taking == []


def test_a_declared_measure_word_synonym_keeps_its_code_and_reads_no_basis():
    """The word "net" in "social safety net" names no netting basis
    (`code_index.declarations.MEASURE_WORD_SYNONYMS_KEPT`)."""
    reading = read_text(
        "social safety nets in Kenya", _REAL_INDEX, get_measure_index(), vocabularies=[_CRS]
    ).by_vocabulary[_CRS]

    by_system = _by_system(reading.matches)
    assert by_system["crsPurpose"].code_dcid == "CRSPurposeCode16010"
    assert "nettingBasis" not in by_system


def test_a_span_naming_two_code_rows_and_a_measure_value_has_no_readings():
    code_index = _synthetic_index(
        CodeRow(
            dcid="SyntheticA",
            code_system="syntheticAxis",
            code="A",
            label="widget",
            synonyms=(),
            parent=None,
        ),
        CodeRow(
            dcid="SyntheticB",
            code_system="syntheticAxis",
            code="B",
            label="widget",
            synonyms=(),
            parent=None,
        ),
    )
    measure_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="SyntheticMeasure",
            axis="syntheticMeasureAxis",
            label="widget",
            synonyms=(),
            short_name=None,
            matchable=True,
        )
    )
    assert match_qualifier_readings("widget", code_index, measure_index) == ()
    assert match_qualifiers("widget", code_index, measure_index) == ()


def test_alias_and_canonical_measure_rows_sharing_a_text_are_one_reading():
    measure_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="RealValue",
            axis="measurementQualifier",
            label="real terms",
            synonyms=("inflation adjusted",),
            short_name=None,
            matchable=True,
        ),
        MeasureValueRow(
            dcid="InflationAdjusted",
            axis="measurementQualifier",
            label="inflation adjusted",
            synonyms=(),
            short_name=None,
            matchable=True,
            same_reading_as="RealValue",
        ),
    )
    matches = match_qualifiers("inflation adjusted GDP", _synthetic_index(), measure_index)
    assert [(m.code_system, m.code_dcid) for m in matches] == [
        ("measurementQualifier", "RealValue")
    ]


def test_no_measure_index_offered_leaves_matching_unchanged():
    # A caller offering no measure family at all (the `measure_index=None`
    # default) or a deployment that declares one with no values both get
    # exactly the matching behaviour this module had before either existed.
    baseline = match_qualifiers("ODA standard grants for health", _REAL_INDEX)
    empty_measure_index = MeasureIndex(axes=(), values=(), partition_labels=())
    assert match_qualifiers("ODA standard grants for health", _REAL_INDEX, None) == baseline
    assert (
        match_qualifiers("ODA standard grants for health", _REAL_INDEX, empty_measure_index)
        == baseline
    )


def _synthetic_measure_index(*rows: MeasureValueRow) -> MeasureIndex:
    """A minimal in-memory measure index carrying only `rows` -- the
    `MeasureValueRow` counterpart to `_synthetic_index`, kept out of any
    committed export per generalisation-rule.md."""
    return MeasureIndex(axes=(), values=rows, partition_labels=())


def test_synthetic_measure_index_matches_only_its_own_declared_synonym():
    # Proves the matcher reads whatever a MeasureIndex declares, not a
    # phrase hard-coded anywhere in this module: the multi-word synonym
    # below exists nowhere but this test's own fixture.
    declaring_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="SyntheticMeasureWidgetTotal",
            axis="syntheticMeasureAxis",
            label="widget total",
            synonyms=("grand widget sum",),
            short_name=None,
            matchable=True,
        )
    )
    empty_code_index = _synthetic_index()
    query = "please total the grand widget sum for Q1"
    matches = match_qualifiers(query, empty_code_index, declaring_index)
    assert len(matches) == 1
    assert matches[0].code_system == "syntheticMeasureAxis"
    assert matches[0].code_dcid == "SyntheticMeasureWidgetTotal"
    assert matches[0].source == "synonym"

    # A second declaration for the same dcid, with no such synonym, never
    # matches the same query -- the declaration is what is read, not the
    # dcid or axis name.
    silent_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="SyntheticMeasureWidgetTotal",
            axis="syntheticMeasureAxis",
            label="widget total",
            synonyms=(),
            short_name=None,
            matchable=True,
        )
    )
    assert match_qualifiers(query, empty_code_index, silent_index) == ()


def test_code_row_and_measure_value_sharing_a_label_are_ambiguous():
    # A `CodeRow` and a `MeasureValueRow` registering the identical
    # normalized text must refuse the same way two colliding `CodeRow`s do
    # -- neither source silently wins over the other.
    code_index = _synthetic_index(
        CodeRow(
            dcid="SyntheticAxisSharedLabel",
            code_system="syntheticAxis",
            code="SHARED",
            label="shared measure label",
            synonyms=(),
            parent=None,
        )
    )
    measure_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="SyntheticMeasureSharedLabel",
            axis="syntheticMeasureAxis",
            label="shared measure label",
            synonyms=(),
            short_name=None,
            matchable=True,
        )
    )
    query = "reported under shared measure label today"
    assert match_qualifiers(query, code_index, measure_index) == ()


def test_unmatchable_measure_row_never_matches_its_label_or_synonyms():
    # A row with matchable=False registers neither its label nor its
    # synonyms -- a matchable sibling in the same index still matches.
    measure_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="SyntheticMeasureVisible",
            axis="syntheticMeasureAxis",
            label="visible measure",
            synonyms=("seen measure",),
            short_name=None,
            matchable=True,
        ),
        MeasureValueRow(
            dcid="SyntheticMeasureHidden",
            axis="syntheticMeasureAxis",
            label="hidden measure",
            synonyms=("unseen measure",),
            short_name=None,
            matchable=False,
        ),
    )
    empty_code_index = _synthetic_index()

    visible = match_qualifiers("report the visible measure", empty_code_index, measure_index)
    assert len(visible) == 1
    assert visible[0].code_dcid == "SyntheticMeasureVisible"

    assert match_qualifiers("report the hidden measure", empty_code_index, measure_index) == ()
    assert match_qualifiers("report the unseen measure", empty_code_index, measure_index) == ()


def test_unmatchable_measure_row_does_not_widen_the_matched_span_width():
    # A multi-word label on an unmatchable row must not extend the span
    # width the matcher tries -- there is no lookup entry it could ever
    # resolve to, so trying the wider span buys nothing and only this test,
    # not any observable match outcome, would catch a regression here.
    code_index = _synthetic_index(
        CodeRow(
            dcid="SyntheticAxisShort",
            code_system="syntheticAxis",
            code="SHORT",
            label="short",
            synonyms=(),
            parent=None,
        )
    )
    measure_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="SyntheticMeasureHiddenPhrase",
            axis="syntheticMeasureAxis",
            label="a much longer hidden phrase",
            synonyms=(),
            short_name=None,
            matchable=False,
        )
    )
    assert _max_phrase_words(code_index, measure_index) == 1

    # A matchable row of the same width does widen it, confirming the
    # assertion above tests the flag and not some other ceiling.
    matchable_measure_index = _synthetic_measure_index(
        MeasureValueRow(
            dcid="SyntheticMeasureVisiblePhrase",
            axis="syntheticMeasureAxis",
            label="a much longer visible phrase",
            synonyms=(),
            short_name=None,
            matchable=True,
        )
    )
    assert _max_phrase_words(code_index, matchable_measure_index) == 5


def test_a_matching_call_tokenises_the_code_labels_once_per_index(monkeypatch):
    # Every matching call builds a lexicon; tokenising each code label on each call cost
    # several milliseconds a call on the committed index, four or more calls a request.
    match_qualifiers("health ODA", _REAL_INDEX)
    labels = {text for row in _REAL_INDEX.codes for text in (row.label, *row.synonyms)}
    tokenised: list[str] = []

    def counting_word_tokens(text: str):
        tokenised.append(text)
        return word_tokens(text)

    monkeypatch.setattr("search.engine.qualifier_match.word_tokens", counting_word_tokens)
    match_qualifiers("health ODA", _REAL_INDEX)

    assert not [text for text in tokenised if text in labels]


def test_grant_equivalent_recognized_amount_not_against_the_real_index():
    grant_equivalent_matches = match_qualifiers(
        "grant equivalent", _REAL_INDEX, get_measure_index()
    )
    by_system = _by_system(grant_equivalent_matches)
    assert by_system["measuredProperty"].code_dcid == "grantEquivalent"


# ---------------------------------------------------------------------------
# match_qualifier_ties: the ambiguous counterpart to match_qualifiers' own
# one-row projection -- slice 6 fix 1's cross-system tie surface.
# ---------------------------------------------------------------------------


def _ties_by_text(ties: tuple[QualifierTie, ...]) -> dict[str, QualifierTie]:
    return {t.matched_text: t for t in ties}


def test_budget_support_yields_one_tie_of_two_rows():
    # "budget support" is registered as the label on both
    # CRSCooperationModalityA and CRSPurposeCode510 -- match_qualifiers
    # refuses it outright (see test_qualifier_spans' own
    # coverage); match_qualifier_ties surfaces it instead, with the rows a CRS shape
    # declares.
    ties = _read("budget support").ties
    assert len(ties) == 1
    tie = ties[0]
    assert tie.matched_text == "budget support"
    assert len(tie.rows) == 2
    assert {row.code_dcid for row in tie.rows} == {
        "CRSCooperationModalityA",
        "CRSPurposeCode510",
    }
    assert {row.code_system for row in tie.rows} == {
        "crsCooperationModality",
        "crsPurpose",
    }
    for row in tie.rows:
        assert row.matched_text == "budget support"


def test_debt_relief_grants_yields_two_ties_four_rows_and_two_rows():
    ties = _ties_by_text(_read("debt relief grants").ties)
    assert set(ties) == {"debt relief", "grants"}

    debt_relief = ties["debt relief"]
    assert len(debt_relief.rows) == 4
    assert {row.code_dcid for row in debt_relief.rows} == {
        "CRSCooperationModalityF01",
        "CRSPurposeCode600",
        "CRSPurposeCode60020",
        "CRSTypeOfFinance600",
    }

    grants = ties["grants"]
    assert len(grants.rows) == 2
    assert {row.code_dcid for row in grants.rows} == {
        "CRSTypeOfFinance100",
        "CRSTypeOfFinance110",
    }


def test_match_qualifier_ties_excludes_a_measure_value_only_tie():
    # A tie is recorded only when every row is a CodeRow -- a
    # MeasureValueRow-touching span stays refused by this function too
    # (_measure_query_matches, region/pipeline.py, owns a same-axis
    # measure conflict instead).
    ties = match_qualifier_ties("grant equivalent", _REAL_INDEX, get_measure_index())
    assert ties == ()


def test_match_qualifier_ties_empty_when_nothing_is_ambiguous():
    assert _read("ODA standard grants for health").ties == ()


def test_a_text_two_families_share_is_a_tie_in_every_system_together():
    """ "ODA" names the CRS flow type and the DAC1 aid type. Read in every code system at
    once, it names two rows and is a tie, not a match."""
    assert match_qualifier_ties("ODA", _REAL_INDEX) != ()
    assert [m.code_dcid for m in match_qualifiers("ODA", _REAL_INDEX)] == []
    assert match_qualifiers("amount", _REAL_INDEX, get_measure_index()) == ()


@pytest.mark.parametrize(
    ("query", "vocabulary", "matches", "ties"),
    [
        ("ODA", _CRS, ["CRSTypeOfFlow10"], []),
        ("ODA", _DAC1, ["DAC1AidType1010"], []),
        (
            "ODA grants for health",
            _CRS,
            ["CRSTypeOfFlow10", "CRSPurposeCode120"],
            [{"CRSTypeOfFinance100", "CRSTypeOfFinance110"}],
        ),
        (
            "bilateral ODA grants to Kenya",
            _CRS,
            ["CRSTypeOfFlow10"],
            [{"CRSTypeOfFinance100", "CRSTypeOfFinance110"}],
        ),
        ("bilateral ODA grants to Kenya", _DAC1, ["DAC1AidType11016"], []),
        ("multilateral ODA", _CRS, ["CRSTypeOfFlow10"], []),
        # The imputed sectors table declares the CRS flow type and the DAC2a aid type, and
        # both read "ODA".
        ("multilateral ODA", _IMPUTED, [], [{"CRSTypeOfFlow10", "DAC2aAidType206"}]),
        ("multilateral ODA", _DAC1, ["DAC1AidType2000"], []),
        ("budget support", _CRS, [], [{"CRSCooperationModalityA", "CRSPurposeCode510"}]),
        ("aid type 106", _IMPUTED, ["DAC2aAidType106"], []),
        ("aid type 106", _CRS, [], []),
    ],
)
def test_each_vocabulary_reads_the_query_in_its_own_systems(query, vocabulary, matches, ties):
    """A shape's vocabulary reads only its own systems' labels. A longer label of another
    family ("bilateral ODA grants", DAC1 11016; "multilateral ODA", DAC1 2000) claims its
    words only in that family's vocabulary, so a CRS shape still reads ODA and the "grants"
    tie. A tie remains where one vocabulary's own systems share a text ("budget
    support"). A bare code after another system's marker ("aid type 106", DAC2a) reads only
    in a vocabulary holding that system."""
    reading = _read(query, vocabulary)
    assert [m.code_dcid for m in reading.matches] == matches
    assert [{row.code_dcid for row in tie.rows} for tie in reading.ties] == ties


@pytest.mark.parametrize(
    ("query", "vocabulary", "matches"),
    [
        ("net ODA to Kenya", _DAC1, ["Net", "DAC1AidType1010"]),
        ("gross national income of Ghana", _DAC1, []),
        ("gross national income of Ghana", _CRS, []),
        ("PPP", _DAC1, []),
    ],
)
def test_a_measure_value_keeps_its_reading_in_every_system(query, vocabulary, matches):
    """A vocabulary drops another system's code rows and the codes no shape can bind, but
    keeps the cover their labels give a measure word. "gross" inside "gross national
    income", a CRS type of finance no CRS row holds, is no netting basis on a DAC1 or a CRS
    shape, and "PPP", a CRS purpose and a price basis, stays a two-reading span there
    (`match_qualifier_readings`)."""
    reading = read_text(
        query, _REAL_INDEX, get_measure_index(), vocabularies=[vocabulary]
    ).by_vocabulary[vocabulary]
    assert [m.code_dcid for m in reading.matches] == matches


# ---------------------------------------------------------------------------
# Marker words that also name the measure
# ---------------------------------------------------------------------------


def test_each_code_systems_markers_come_from_its_name():
    markers = _all_system_markers(_REAL_INDEX)

    assert {system: sorted(m.words) for system, m in markers.items()} == {
        "crsCooperationModality": ["cooperation", "modality"],
        "crsPurpose": ["purpose"],
        "crsTypeOfFinance": ["finance"],
        "crsTypeOfFlow": ["flow"],
        "dac1AidType": ["dac1"],
        "dac2aAidType": ["dac2a"],
        "dac2bAidType": ["dac2b"],
    }
    # "aid" names every aid-type system, so it marks none, measure word or not.
    assert {
        system: sorted(m.measure_words) for system, m in markers.items() if m.measure_words
    } == {}


@pytest.mark.parametrize(
    ("text", "code_dcid"),
    [
        ("modality D02", "CRSCooperationModalityD02"),
        ("cooperation modality C01", "CRSCooperationModalityC01"),
        ("purpose code 120", "CRSPurposeCode120"),
        ("aid for purpose code 110", "CRSPurposeCode110"),
        ("type of finance 110", "CRSTypeOfFinance110"),
        ("finance 110", "CRSTypeOfFinance110"),
        ("flow type 10", "CRSTypeOfFlow10"),
        ("aid flow type 10", "CRSTypeOfFlow10"),
        ("aid type 106", "DAC2aAidType106"),
        ("dac2a 106", "DAC2aAidType106"),
    ],
)
def test_a_bare_code_after_its_systems_marker_matches(text: str, code_dcid: str):
    matches = match_qualifiers(text, _REAL_INDEX)
    assert [m.code_dcid for m in matches if m.source == "code"] == [code_dcid]


@pytest.mark.parametrize(
    "text", ["aid to 106 countries", "education aid 106", "aid 106", "type 106"]
)
def test_a_word_that_names_the_measure_marks_no_code_on_its_own(text: str):
    """ "aid" is the word a declared default reads as naming the measure, and a word of both
    `dac2aAidType` and `dac1AidType`, so it marks neither system alone."""
    assert [m for m in match_qualifiers(text, _REAL_INDEX) if m.source == "code"] == []
