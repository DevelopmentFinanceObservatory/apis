"""Tests time-text parsing, relative ranges, refusals, and conversion to `DateRequest`."""

from __future__ import annotations

import random
from typing import ClassVar

import pytest

from search.engine.extract import DateRequest, dates_to_request
from search.engine.time_bind import TIME_TEXT_FORMS, TIME_TEXT_NOT_PARSED, parse_time_text
from search.models import Warning

_CURRENT_YEAR = 2026


class TestGrammar:
    def test_bare_year(self) -> None:
        dates, warning = parse_time_text("2015", current_year=_CURRENT_YEAR)
        assert warning is None
        assert len(dates) == 1
        assert dates[0].kind == "point"
        assert dates[0].start == "2015"
        assert dates[0].end is None

    @pytest.mark.parametrize("expr", ["2015 to 2024", "2015-2024", "2015–2024"])
    def test_ranges(self, expr: str) -> None:
        dates, warning = parse_time_text(expr, current_year=_CURRENT_YEAR)
        assert warning is None
        assert len(dates) == 1
        assert dates[0].kind == "range"
        assert dates[0].start == "2015"
        assert dates[0].end == "2024"

    def test_since(self) -> None:
        dates, warning = parse_time_text("since 2015", current_year=_CURRENT_YEAR)
        assert warning is None
        assert len(dates) == 1
        assert dates[0].kind == "range"
        assert dates[0].start == "2015"
        assert dates[0].end is None

    @pytest.mark.parametrize("word", ["latest", "current", "now"])
    def test_latest_words(self, word: str) -> None:
        dates, warning = parse_time_text(word, current_year=_CURRENT_YEAR)
        assert warning is None
        assert len(dates) == 1
        assert dates[0].kind == "latest"
        assert dates[0].start is None
        assert dates[0].end is None

    def test_case_and_whitespace_insensitive(self) -> None:
        dates, warning = parse_time_text("  Since 2015  ", current_year=_CURRENT_YEAR)
        assert warning is None
        assert dates[0].kind == "range"
        assert dates[0].start == "2015"

    @pytest.mark.parametrize("word", ["LATEST", " Current ", "NOW"])
    def test_latest_words_casefolded(self, word: str) -> None:
        dates, warning = parse_time_text(word, current_year=_CURRENT_YEAR)
        assert warning is None
        assert dates[0].kind == "latest"


class TestReversedRange:
    def test_reversed_range_is_unparsed(self) -> None:
        """A reversed range ("2024 to 2015") has no representation in the
        request grammar -- dates_to_request does not validate start<=end, so
        passing it through would silently produce an inverted window.
        Treated as unparsed instead of refused or corrected."""
        dates, warning = parse_time_text("2024 to 2015", current_year=_CURRENT_YEAR)
        assert dates == []
        assert warning is not None
        assert warning.code == TIME_TEXT_NOT_PARSED


class TestRelativeSpans:
    @pytest.mark.parametrize(
        ("phrase", "digits"),
        [
            ("last 10 years", 10),
            ("past 10 years", 10),
            ("last ten years", 10),
            ("past ten years", 10),
            ("over the last 10 years", 10),
            ("over the past ten years", 10),
            ("last 1 year", 1),
            ("last one year", 1),
            ("last 20 years", 20),
            ("last twenty years", 20),
        ],
    )
    def test_last_n_years(self, phrase: str, digits: int) -> None:
        dates, warning = parse_time_text(phrase, current_year=_CURRENT_YEAR)
        assert warning is None
        assert len(dates) == 1
        assert dates[0].kind == "range"
        assert dates[0].start == str(_CURRENT_YEAR - digits + 1)
        assert dates[0].end == str(_CURRENT_YEAR)

    @pytest.mark.parametrize(
        "word",
        [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ],
    )
    def test_number_words_up_to_twenty(self, word: str) -> None:
        expected = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
        }[word]
        dates, warning = parse_time_text(f"last {word} years", current_year=_CURRENT_YEAR)
        assert warning is None
        assert dates[0].start == str(_CURRENT_YEAR - expected + 1)
        assert dates[0].end == str(_CURRENT_YEAR)

    @pytest.mark.parametrize(
        "phrase",
        ["last decade", "past decade", "over the last decade", "over the past decade"],
    )
    def test_decade_phrases(self, phrase: str) -> None:
        dates, warning = parse_time_text(phrase, current_year=_CURRENT_YEAR)
        assert warning is None
        assert len(dates) == 1
        assert dates[0].kind == "range"
        assert dates[0].start == str(_CURRENT_YEAR - 10 + 1)
        assert dates[0].end == str(_CURRENT_YEAR)

    def test_resolves_against_the_caller_supplied_year(self) -> None:
        """The relative-span rows have no notion of "today" of their own --
        changing current_year moves the resolved range, proving the module
        reads no clock."""
        dates, _ = parse_time_text("last 5 years", current_year=2000)
        assert dates[0].start == "1996"
        assert dates[0].end == "2000"

    def test_case_and_whitespace_insensitive(self) -> None:
        dates, warning = parse_time_text("  Last 5 Years  ", current_year=_CURRENT_YEAR)
        assert warning is None
        assert dates[0].start == str(_CURRENT_YEAR - 5 + 1)

    @pytest.mark.parametrize(
        ("text", "start"),
        [
            ("the last 5 years", "2022"),
            ("in the last 5 years", "2022"),
            ("during the last 5 years", "2022"),
            ("for the past 5 years", "2022"),
            ("within the last five years", "2022"),
            ("over the past 10 years", "2017"),
            ("the past three years", "2024"),
            ("in the past decade", "2017"),
            ("the past decade", "2017"),
            ("the last two decades", "2007"),
            ("over the past 2 decades", "2007"),
            ("recent years", "2022"),
            ("in recent years", "2022"),
            ("over the last few years", "2022"),
            ("los últimos 10 años", "2017"),
            ("en los últimos diez años", "2017"),
            ("últimos 5 años", "2022"),
            ("ultimos 5 anos", "2022"),
            ("durante la última década", "2017"),
            ("las últimas dos décadas", "2007"),
            ("en los últimos años", "2022"),
            ("au cours des dix dernières années", "2017"),
            ("les 5 derniers ans", "2022"),
            ("derniers 5 ans", "2022"),
            ("pendant les dernières dix années", "2017"),
            ("la dernière décennie", "2017"),
            ("sur les deux dernières décennies", "2007"),
            ("ces dernières années", "2022"),
            ("past year", "2025"),
            ("the past year", "2025"),
            ("in the past year", "2025"),
            ("over the past year", "2025"),
            ("the last year", "2025"),
            ("in the last year", "2025"),
            ("over the last year", "2025"),
            ("el último año", "2025"),
            ("en el último año", "2025"),
            ("la dernière année", "2025"),
            ("au cours de la dernière année", "2025"),
        ],
    )
    def test_spans_measured_from_now(self, text: str, start: str) -> None:
        """A lead, an article, "past" or "last", a count in digits or words, decades,
        "recent years" and the Spanish and French forms all close at current_year."""
        dates, warning = parse_time_text(text, current_year=_CURRENT_YEAR)
        assert warning is None
        assert [(d.kind, d.start, d.end) for d in dates] == [("range", start, "2026")]

    @pytest.mark.parametrize(
        ("text", "year"),
        [
            ("last year", "2025"),
            ("for last year", "2025"),
            ("this year", "2026"),
            ("for this year", "2026"),
            ("next year", "2027"),
            ("el año pasado", "2025"),
            ("este año", "2026"),
            ("el próximo año", "2027"),
            ("el año que viene", "2027"),
            ("l'année dernière", "2025"),
            ("l'année passée", "2025"),
            ("l\u2019an dernier", "2025"),
            ("cette année", "2026"),
            ("l'année prochaine", "2027"),
            ("l'an prochain", "2027"),
        ],
    )
    def test_years_measured_from_now_are_points(self, text: str, year: str) -> None:
        dates, warning = parse_time_text(text, current_year=_CURRENT_YEAR)
        assert warning is None
        assert [(d.kind, d.start, d.end) for d in dates] == [("point", year, None)]

    def test_relative_wording_needs_a_relative_word(self) -> None:
        """Leads alone make nothing relative: "in 2015" stays outside the grammar."""
        dates, warning = parse_time_text("in 2015", current_year=_CURRENT_YEAR)
        assert dates == []
        assert warning is not None
        assert warning.code == TIME_TEXT_NOT_PARSED

    def test_same_years_anaphor_stays_unparsed(self) -> None:
        """ "the same years" is resolved by the harness, not this grammar."""
        dates, warning = parse_time_text("the same years", current_year=_CURRENT_YEAR)
        assert dates == []
        assert warning is not None
        assert warning.code == TIME_TEXT_NOT_PARSED

    def test_zero_years_stays_unparsed(self) -> None:
        dates, warning = parse_time_text("last 0 years", current_year=_CURRENT_YEAR)
        assert dates == []
        assert warning is not None
        assert warning.code == TIME_TEXT_NOT_PARSED

    def test_twenty_one_years_stays_unparsed(self) -> None:
        """Only "twenty" and below are recognised number words; a two-digit
        count above the cap this grammar accepts stays unparsed too, since
        _SPAN_COUNT's digit branch is capped at two digits and 21 is not a
        recognised number word."""
        dates, warning = parse_time_text("last twenty-one years", current_year=_CURRENT_YEAR)
        assert dates == []
        assert warning is not None
        assert warning.code == TIME_TEXT_NOT_PARSED


class TestUnparsed:
    @pytest.mark.parametrize(
        "text",
        [
            "garbled nonsense",
            "sometime in the 2010s",
            "202",
            "20155",
            "2015 until 2024",
            "last 0 decades",
            "the next year",
            "Q1 2015",
            "current year",
            "the current year",
            "now available",
        ],
    )
    def test_unrecognised_text_degrades(self, text: str) -> None:
        dates, warning = parse_time_text(text, current_year=_CURRENT_YEAR)
        assert dates == []
        assert isinstance(warning, Warning)
        assert warning.code == TIME_TEXT_NOT_PARSED
        assert warning.severity == "warn"

    def test_dates_to_request_treats_unparsed_as_dateless(self) -> None:
        """dates_to_request([]) already returns None -- unparseable
        time_text and no time_text at all degrade identically."""
        dates, _ = parse_time_text("garbled nonsense", current_year=_CURRENT_YEAR)
        assert dates_to_request(dates) is None

    def test_never_raises_on_fuzz(self) -> None:
        rng = random.Random(1234)
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 -–.,/\\'\""
        for _ in range(200):
            length = rng.randint(0, 40)
            text = "".join(rng.choice(alphabet) for _ in range(length)) or " "
            dates, warning = parse_time_text(text, current_year=_CURRENT_YEAR)
            assert isinstance(dates, list)
            if dates:
                assert warning is None
            else:
                assert warning is None or isinstance(warning, Warning)


class TestAcceptedForms:
    @pytest.mark.parametrize("form", TIME_TEXT_FORMS)
    def test_every_listed_form_parses(self, form: str) -> None:
        text = form.replace("YYYY", "2010", 1).replace("YYYY", "2020").replace("N", "5")
        dates, warning = parse_time_text(text, current_year=_CURRENT_YEAR)
        assert warning is None, text
        assert dates != []

    def test_the_not_parsed_warning_lists_every_form(self) -> None:
        _, warning = parse_time_text("garbled nonsense", current_year=_CURRENT_YEAR)
        assert warning is not None
        for form in TIME_TEXT_FORMS:
            assert repr(form) in warning.message


class TestPropertyFeedsDatesToRequest:
    _GOLDEN_INPUTS: ClassVar[list[str]] = [
        "2015",
        "2015 to 2024",
        "2015-2024",
        "2015–2024",
        "since 2015",
        "latest",
        "current",
        "now",
        "last 10 years",
        "last decade",
        "over the last decade",
    ]

    @pytest.mark.parametrize("text", _GOLDEN_INPUTS)
    def test_every_non_empty_result_feeds_dates_to_request(self, text: str) -> None:
        dates, warning = parse_time_text(text, current_year=_CURRENT_YEAR)
        assert warning is None
        assert dates != []
        request = dates_to_request(dates)
        assert isinstance(request, DateRequest)
