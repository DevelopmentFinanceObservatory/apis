"""Tests for scripts/intake_sheet.py's CSV grammar parser and row pipeline."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load by path because scripts/ has no `__init__.py`.
# Register first so dataclasses can resolve postponed annotations.
_SPEC = importlib.util.spec_from_file_location(
    "intake_sheet", Path(__file__).parents[2] / "scripts" / "intake_sheet.py"
)
assert _SPEC is not None and _SPEC.loader is not None
intake_sheet = importlib.util.module_from_spec(_SPEC)
sys.modules["intake_sheet"] = intake_sheet
_SPEC.loader.exec_module(intake_sheet)

CellParseError = intake_sheet.CellParseError
apply_corrections = intake_sheet.apply_corrections
domain_for = intake_sheet.domain_for
parse_expected_answer = intake_sheet.parse_expected_answer
parse_rows = intake_sheet.parse_rows
slugify_id = intake_sheet.slugify_id


def _row(
    question: str,
    behaviour: str,
    raw_cell: str = "",
    *,
    difficulty: str = "",
    notes: str = "",
) -> list[str]:
    """Build a 14-column sheet row with only the fields tests care about set."""
    return [
        "",  # sheet_id (always blank in practice)
        question,
        behaviour,
        raw_cell,
        "",  # multi_turn
        "",  # follow_up
        "",  # how_established (ignored)
        "",  # independent (ignored)
        "",  # status_trust (ignored)
        "",  # origin (ignored)
        difficulty,
        "",  # source_area
        notes,
        "",  # contributor (ignored)
    ]


class TestGrammarHappyPaths:
    def test_single_statvar_single_place(self):
        parsed = parse_expected_answer("Count_Person; places: country/IND")
        assert parsed.is_candidates is False
        assert parsed.stat_vars == ["Count_Person"]
        assert parsed.places == ["country/IND"]
        assert parsed.date is None

    def test_multi_statvar_candidates(self):
        parsed = parse_expected_answer(
            "candidates: Count_Person, UnemploymentRate_Person; places: country/FRA"
        )
        assert parsed.is_candidates is True
        assert parsed.stat_vars == ["Count_Person", "UnemploymentRate_Person"]

    def test_multi_place(self):
        parsed = parse_expected_answer("worldBank/EG_ELC_ACCS_ZS; places: country/GHA, country/NGA")
        assert parsed.places == ["country/GHA", "country/NGA"]

    def test_single_year_date(self):
        parsed = parse_expected_answer("Count_Person; places: country/ESP; date: 2020")
        assert parsed.date == "2020"

    def test_year_range_date(self):
        parsed = parse_expected_answer("Count_Person; places: country/ESP; date: 2015-2020")
        assert parsed.date == "2015-2020"

    def test_no_date_is_none(self):
        parsed = parse_expected_answer("Count_Person; places: country/ESP")
        assert parsed.date is None

    def test_empty_cell(self):
        parsed = parse_expected_answer("")
        assert parsed.stat_vars == []
        assert parsed.places == []
        assert parsed.date is None


class TestMalformedCells:
    def test_date_leaked_into_places_segment_rejects(self):
        with pytest.raises(CellParseError, match="bare date"):
            parse_expected_answer(
                "Count_Person, UnemploymentRate_Person; places: country/FRA, 2022"
            )

    def test_trailing_bare_year_without_date_label_rejects(self):
        with pytest.raises(CellParseError, match="missing the 'date:' label"):
            parse_expected_answer(
                "Mean_Inflation_EconomicActivity_ConsumerPriceIndex_GeneralSector; "
                "places: country/ARG; 2023"
            )

    def test_missing_places_segment_rejects(self):
        with pytest.raises(CellParseError, match="places"):
            parse_expected_answer("Count_Person; date: 2020")

    def test_empty_places_segment_rejects_with_distinct_message(self):
        with pytest.raises(CellParseError, match="no place dcid"):
            parse_expected_answer("Count_Person; places:")

    def test_unrecognized_segment_rejects(self):
        with pytest.raises(CellParseError, match="unrecognized segment"):
            parse_expected_answer("Count_Person; places: country/ESP; garbage: yes")

    def test_malformed_date_value_rejects(self):
        with pytest.raises(CellParseError, match="year or year-range"):
            parse_expected_answer("Count_Person; places: country/ESP; date: banana")


class TestDomainAssignment:
    def test_standard_domain(self):
        assert domain_for(["Count_Person"]) == "standard"

    def test_sdg_domain_from_prefix(self):
        assert domain_for(["sdg/SH_STA_MORT.SEX--F"]) == "sdg"

    def test_sdg_domain_wins_when_mixed(self):
        assert domain_for(["Count_Person", "sdg/EG_FEC_RNEW"]) == "sdg"

    def test_worldbank_and_who_are_standard(self):
        assert domain_for(["worldBank/EG_ELC_ACCS_ZS"]) == "standard"
        assert domain_for(["who/mslv"]) == "standard"


class TestCorrections:
    def test_vietnam_place_corrected(self):
        corrected, applied = apply_corrections(["country/VTN"])
        assert corrected == ["country/VNM"]
        assert len(applied) == 1
        assert "country/VTN" in applied[0]
        assert "country/VNM" in applied[0]

    def test_who_mslv_case_corrected(self):
        corrected, applied = apply_corrections(["WHO/mslv"])
        assert corrected == ["who/mslv"]
        assert len(applied) == 1

    def test_unaffected_tokens_pass_through(self):
        corrected, applied = apply_corrections(["country/KEN", "Count_Person"])
        assert corrected == ["country/KEN", "Count_Person"]
        assert applied == []


class TestIdSlugification:
    def test_drops_stopwords(self):
        assert slugify_id("What is the GDP of Kenya?") == slugify_id("GDP Kenya")

    def test_distinct_questions_distinct_slugs(self):
        assert slugify_id("GDP in Southern African countries") != slugify_id(
            "GDP in Southern Africa"
        )

    def test_slug_is_lowercase_kebab_case(self):
        slug = slugify_id("How long are women in China expected to live?")
        assert slug == slug.lower()
        assert all(c.isalnum() or c == "-" for c in slug)


class TestBehaviourMapping:
    def test_value_maps_to_definite(self):
        result = parse_rows(
            [_row("total population India", "value", "Count_Person; places: country/IND")],
            existing_ids=set(),
        )
        assert len(result.parsed) == 1
        assert result.parsed[0].record["expected_status"] == "definite"
        assert {"behaviour": "definite"} in result.parsed[0].record["tags"]

    def test_candidates_maps_to_candidates(self):
        result = parse_rows(
            [
                _row(
                    "income in California",
                    "candidates",
                    "candidates: Median_Income_Person, Median_Income_Household; places: geoId/06",
                )
            ],
            existing_ids=set(),
        )
        record = result.parsed[0].record
        assert record["expected_status"] == "candidates"
        assert record["candidate_count"] == 2

    def test_no_data_maps_to_no_data(self):
        result = parse_rows(
            [_row("nominal GDP of Atlantis", "no-data", "")],
            existing_ids=set(),
        )
        record = result.parsed[0].record
        assert record["expected_status"] == "no_data"
        assert record["expected_stat_vars"] == []
        assert record["expected_entities"] == []

    def test_answer_redirect_maps_to_definite_and_leaves_operations_to_the_maintainer(self):
        result = parse_rows(
            [
                _row(
                    "Is South Africa's GDP larger than Nigeria?",
                    "answer-redirect",
                    "Amount_EconomicActivity_GrossDomesticProduction_Nominal; "
                    "places: country/ZAF, country/NGA",
                )
            ],
            existing_ids=set(),
        )
        record = result.parsed[0].record
        assert record["expected_status"] == "definite"
        assert not any("derivation" in tag for tag in record["tags"])
        assert "expected_operations" not in record
        assert "expected_operations" in record["notes"]

    def test_value_row_notes_do_not_ask_for_operations(self):
        result = parse_rows(
            [_row("total population India", "value", "Count_Person; places: country/IND")],
            existing_ids=set(),
        )
        assert "expected_operations" not in result.parsed[0].record["notes"]

    def test_refuse_redirect_is_parked_not_rejected(self):
        result = parse_rows(
            [_row("How educated is India's population?", "refuse-redirect", "")],
            existing_ids=set(),
        )
        assert result.parsed == []
        assert result.rejected == []
        assert len(result.parked) == 1

    def test_unknown_behaviour_rejects(self):
        result = parse_rows(
            [_row("some question", "mystery-behaviour", "")],
            existing_ids=set(),
        )
        assert result.parsed == []
        assert len(result.rejected) == 1
        assert "unknown behaviour" in result.rejected[0].reason

    def test_no_data_row_with_nonempty_cell_rejects(self):
        result = parse_rows(
            [_row("nominal GDP of Atlantis", "no-data", "Count_Person; places: country/KEN")],
            existing_ids=set(),
        )
        assert result.parsed == []
        assert len(result.rejected) == 1
        assert "non-empty" in result.rejected[0].reason

    @pytest.mark.parametrize(
        ("question", "behaviour"),
        [
            pytest.param("total population India", "value", id="value"),
            pytest.param(
                "Is South Africa's GDP larger than Nigeria?",
                "answer-redirect",
                id="answer_redirect",
            ),
        ],
    )
    def test_value_and_answer_redirect_rows_with_blank_cell_reject(self, question, behaviour):
        result = parse_rows(
            [_row(question, behaviour, "")],
            existing_ids=set(),
        )
        assert result.parsed == []
        assert len(result.rejected) == 1
        assert "blank" in result.rejected[0].reason


class TestMalformedRowsRejectWithUsefulMessages:
    def test_date_leaked_into_places_row_rejects(self):
        result = parse_rows(
            [
                _row(
                    "Population and unemployment rate of France in 2022",
                    "candidates",
                    "candidates: Count_Person, UnemploymentRate_Person; places: country/FRA, 2022",
                )
            ],
            existing_ids=set(),
        )
        assert result.parsed == []
        assert len(result.rejected) == 1
        assert "bare date" in result.rejected[0].reason
        assert "2022" in result.rejected[0].reason

    def test_trailing_bare_year_row_rejects(self):
        result = parse_rows(
            [
                _row(
                    "Argentina's inflation rate in 2023",
                    "value",
                    "Mean_Inflation_EconomicActivity_ConsumerPriceIndex_GeneralSector; "
                    "places: country/ARG; 2023",
                )
            ],
            existing_ids=set(),
        )
        assert result.parsed == []
        assert len(result.rejected) == 1
        assert "date:" in result.rejected[0].reason
        assert "2023" in result.rejected[0].reason


class TestIdCollisionDetection:
    def test_collision_against_existing_corpus_id_is_flagged_not_rejected(self):
        result = parse_rows(
            [_row("total population India", "value", "Count_Person; places: country/IND")],
            existing_ids={slugify_id("total population India")},
        )
        assert len(result.parsed) == 1  # The collision is flagged without rejecting the row.
        golden = result.parsed[0]
        assert golden.collision_with is not None
        assert "COLLISION" in golden.record["notes"]

    def test_collision_within_the_same_batch_is_flagged(self):
        rows = [
            _row("total population India", "value", "Count_Person; places: country/IND"),
            _row("total population India", "value", "Count_Person; places: country/IND"),
        ]
        result = parse_rows(rows, existing_ids=set())
        assert len(result.parsed) == 2
        assert result.parsed[0].collision_with is None
        assert result.parsed[1].collision_with is not None

    def test_no_collision_when_ids_differ(self):
        rows = [
            _row("total population India", "value", "Count_Person; places: country/IND"),
            _row("total population Kenya", "value", "Count_Person; places: country/KEN"),
        ]
        result = parse_rows(rows, existing_ids=set())
        assert all(g.collision_with is None for g in result.parsed)


class TestUnderivableFieldsLeftEmpty:
    """`parse_rows` leaves engine-derived shape and no-data fields empty for every status."""

    @pytest.mark.parametrize(
        ("question", "behaviour", "raw_cell"),
        [
            pytest.param(
                "total population India",
                "value",
                "Count_Person; places: country/IND",
                id="definite_row",
            ),
            pytest.param(
                "birth rate Ethiopia",
                "candidates",
                "candidates: Count_BirthEvent_LiveBirth_AsFractionOf_Count_Person, "
                "FertilityRate_Person_Female; places: country/ETH",
                id="candidates_row",
            ),
        ],
    )
    def test_shape_and_slots_are_empty_for_definite_and_candidates_rows(
        self, question, behaviour, raw_cell
    ):
        result = parse_rows(
            [_row(question, behaviour, raw_cell)],
            existing_ids=set(),
        )
        record = result.parsed[0].record
        assert record["expected_shape"] is None
        assert record["expected_slots"] == []

    def test_no_data_row_leaves_reason_and_category_empty(self):
        result = parse_rows(
            [_row("nominal GDP of Atlantis", "no-data", "")],
            existing_ids=set(),
        )
        record = result.parsed[0].record
        assert record["expected_no_data_reason"] is None
        assert record["expected_no_data_category"] is None
        assert "expected_no_data_reason" in record["notes"]
        assert "expected_no_data_category" in record["notes"]

    def test_stat_vars_and_entities_are_still_populated(self):
        # The parser preserves the sheet's authoritative dcids and places.
        result = parse_rows(
            [_row("total population India", "value", "Count_Person; places: country/IND")],
            existing_ids=set(),
        )
        record = result.parsed[0].record
        assert record["expected_stat_vars"] == ["Count_Person"]
        assert record["expected_entities"] == [
            {"dcid": "country/IND", "role_kind": "subject", "role_axis": None, "role_dcid": None}
        ]
