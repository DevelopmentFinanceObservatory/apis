"""Parse time phrases into the ``ExtractedDate`` form used by the parsed door.
The caller passes results to ``extract.dates_to_request`` for window semantics.

Input is stripped and casefolded. Four-digit years yield points. A bounded phrase is an
optional lead, a start year, a separator and an end, and the lead fixes the separator:
``between`` takes ``and``, ``from`` takes ``to`` or ``until``, and no lead takes ``to``,
``-`` or ``–``. A year end yields an ordered range; a latest end ("the latest", "most
recent year available") yields an open-ended range, as ``since``, ``depuis``, and
``desde`` do. The latest phrase alone yields a latest entry.

Wording measured from now resolves against the caller-supplied ``current_year``. Spans
back ("last 5 years", "la última década", "au cours des dix dernières années") yield
closed ranges that include the current year and carry their count of years
(`ExtractedDate.trailing_years`), so the answer reads that many years ending at the newest
year its data holds. "This decade" runs from the decade's first year to the current one. Spans ahead ("the next 5 years", "los próximos cinco
años") start the year after the current one, so a forecast over them targets their last
year and its window leaves the data years alone. "Last year", "this year", "so far this
year" and "next year" yield single years. "The past year" and "the last year" are the
last twelve months, the range from the year before the current one to the current year;
bare "last year" is the previous calendar year. A count of months reaches back as many
calendar years as it covers, part years rounded up, so "the last 12 months" is that
same range and "the last 18 months" starts two years back. A span ahead with no count
("the coming years") names no last year and stays unparsed. The relative rows read with
accents folded and after any leading "in", "during", "over", "for" or "within" and
their Spanish and French counterparts. An English "the" stays, since it separates "the
last year" from "last year".
``TIME_TEXT_FORMS`` lists every accepted form. Counts accept digits or number words
through twenty, and the fixed vocabulary covers English, French, and Spanish.

Other phrases, including reversed ranges, return an empty list and a
``TIME_TEXT_NOT_PARSED`` warning. The request stays valid and dateless.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from dfo_platform import WarningSeverity
from search.engine.extracted_date import ExtractedDate
from search.models import Warning

TIME_TEXT_NOT_PARSED: str = "time_text_not_parsed"

# "latest" and "most recent" take "the" and a "year"/"available" suffix; "current" and
# "now" stand alone.
_LATEST = r"(?:(?:the\s+)?(?:latest|most\s+recent)(?:\s+year)?(?:\s+available)?|current|now)"
_LATEST_RE = re.compile(rf"^{_LATEST}$")
_BOUNDED_RE = re.compile(
    r"^(?:(?P<lead>from|between|since|depuis|desde)\s+)?(?P<start>\d{4})"
    rf"(?:\s*(?P<separator>to|until|and|-|–)\s*(?P<end>\d{{4}}|{_LATEST}))?$"
)
# The separators each lead takes; `None` is a phrase that ends at its start year, a bare
# year without a lead and an open-ended range after a "since" lead.
_SEPARATORS_BY_LEAD: dict[str | None, frozenset[str | None]] = {
    None: frozenset({None, "to", "-", "–"}),
    "from": frozenset({"to", "until"}),
    "between": frozenset({"and"}),
    "since": frozenset({None}),
    "depuis": frozenset({None}),
    "desde": frozenset({None}),
}

# Every form the grammar accepts, as the not-parsed warning lists them. `YYYY` stands
# for a four-digit year and `N` for a count in digits or words up to twenty.
TIME_TEXT_FORMS: tuple[str, ...] = (
    "YYYY",
    "YYYY to YYYY",
    "YYYY-YYYY",
    "from YYYY to YYYY",
    "from YYYY until YYYY",
    "between YYYY and YYYY",
    "YYYY to latest",
    "from YYYY to the latest",
    "between YYYY and the latest",
    "since YYYY",
    "depuis YYYY",
    "desde YYYY",
    "latest",
    "the latest",
    "latest year available",
    "most recent",
    "current",
    "now",
    "last N years",
    "past N years",
    "in the last N years",
    "within the past N years",
    "last decade",
    "past decade",
    "over the last decade",
    "the past N decades",
    "recent years",
    "the last few years",
    "the past year",
    "the last year",
    "the last N months",
    "this decade",
    "the next N years",
    "the coming N years",
    "last year",
    "this year",
    "so far this year",
    "next year",
    "los últimos N años",
    "la última década",
    "las últimas N décadas",
    "los últimos años",
    "el último año",
    "los últimos N meses",
    "esta década",
    "los próximos N años",
    "el año pasado",
    "este año",
    "en lo que va del año",
    "el próximo año",
    "au cours des N dernières années",
    "les N derniers ans",
    "derniers N ans",
    "la dernière décennie",
    "ces dernières années",
    "la dernière année",
    "les N derniers mois",
    "cette décennie",
    "dans les N prochaines années",
    "l'année dernière",
    "l'année passée",
    "cette année",
    "depuis le début de l'année",
    "l'année prochaine",
)

# Number words one to twenty in English, Spanish and French, accents folded. Larger
# counts use digits.
_NUMBER_WORDS_BY_LANGUAGE: tuple[str, ...] = (
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty",
    "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince "
    "dieciseis diecisiete dieciocho diecinueve veinte",
    "un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze "
    "seize dix-sept dix-huit dix-neuf vingt",
)
_NUMBER_WORDS: dict[str, int] = {
    word: count
    for words in _NUMBER_WORDS_BY_LANGUAGE
    for count, word in enumerate(words.split(), start=1)
}
_COUNT = rf"(?P<count>\d{{1,2}}|{'|'.join(_NUMBER_WORDS)})"

# Words that may open a relative wording and add nothing to its years, in any number
# and order: "in the last 5 years", "durante los últimos 10 años", "au cours des dix
# dernières années". The English rows take "the" themselves.
_RELATIVE_LEADS_RE = re.compile(
    r"^(?:(?:in|during|over|for|within|en|durante|los|las|la|el"
    r"|au\s+cours\s+des?|pendant|durant|sur|dans|les|ces)\s+)*"
)

_DECADE_YEARS = 10
# "Recent years" states no count and reads as the last five years ending at the
# current year, as do "the last few years", "los últimos años" and "ces dernières
# années".
_RECENT_YEARS = 5
# Twelve months back from now cross into the year before, so "the past year", "the last
# year", "el último año" and "la dernière année" cover that year and the current one.
_TWELVE_MONTH_YEARS = 2
_THE = r"(?:the\s+)?"

# A span's first and last years from its count and the current year.
_SpanYears = Callable[[int, int], tuple[int, int]]


def _back(unit_years: int) -> _SpanYears:
    """Years counted back from now, the current year included: "last 10 years" from
    2026 is 2017 to 2026."""
    return lambda count, current_year: (current_year - count * unit_years + 1, current_year)


def _ahead(unit_years: int) -> _SpanYears:
    """Years counted on from the year after now: "the next 5 years" from 2026 is 2027 to
    2031, so a forecast over them targets 2031."""
    return lambda count, current_year: (current_year + 1, current_year + count * unit_years)


def _months_back(count: int, current_year: int) -> tuple[int, int]:
    """Months counted back from now reach as many years before the current one as they
    cover, a part year rounded up: 12 months from 2026 are 2025 to 2026, 18 are 2024 to
    2026."""
    return current_year - -(-count // 12), current_year


def _this_decade(_count: int, current_year: int) -> tuple[int, int]:
    """The current decade so far: 2020 to 2026 from 2026."""
    return current_year - current_year % _DECADE_YEARS, current_year


# Each row is a span measured from now, accents folded and leads removed, the years it
# covers, and whether it trails: a span counted back to now asks for its count of years
# ending at the newest year the answer's data holds (`ExtractedDate.trailing_years`). A
# row without a count covers one unit.
_RELATIVE_SPANS: tuple[tuple[re.Pattern[str], _SpanYears, bool], ...] = tuple(
    (re.compile(rf"^(?:{pattern})$"), span_years, trailing)
    for pattern, span_years, trailing in (
        (rf"{_THE}(?:last|past)\s+{_COUNT}\s+years?", _back(1), True),
        (rf"{_THE}(?:last|past)\s+decade", _back(_DECADE_YEARS), True),
        (rf"{_THE}(?:last|past)\s+{_COUNT}\s+decades", _back(_DECADE_YEARS), True),
        (rf"{_THE}(?:recent\s+years|(?:last|past)\s+few\s+years)", _back(_RECENT_YEARS), True),
        (rf"{_THE}past\s+year|the\s+last\s+year", _back(_TWELVE_MONTH_YEARS), True),
        (rf"{_THE}(?:last|past)\s+{_COUNT}\s+months?", _months_back, True),
        (r"this\s+decade", _this_decade, False),
        (rf"{_THE}(?:next|coming)\s+{_COUNT}\s+years?", _ahead(1), False),
        (rf"ultimos\s+{_COUNT}\s+anos", _back(1), True),
        (r"ultima\s+decada", _back(_DECADE_YEARS), True),
        (rf"ultimas\s+{_COUNT}\s+decadas", _back(_DECADE_YEARS), True),
        (r"ultimos\s+anos|anos\s+recientes", _back(_RECENT_YEARS), True),
        (r"ultimo\s+ano", _back(_TWELVE_MONTH_YEARS), True),
        (rf"ultimos\s+{_COUNT}\s+meses", _months_back, True),
        (r"esta\s+decada", _this_decade, False),
        (rf"proximos\s+{_COUNT}\s+anos", _ahead(1), False),
        (rf"{_COUNT}\s+(?:dernieres\s+annees|derniers\s+ans)", _back(1), True),
        (rf"dernieres\s+{_COUNT}\s+annees", _back(1), True),
        (rf"derniers\s+{_COUNT}\s+ans", _back(1), True),
        (r"derniere\s+decennie", _back(_DECADE_YEARS), True),
        (rf"{_COUNT}\s+dernieres\s+decennies", _back(_DECADE_YEARS), True),
        (r"dernieres\s+annees|annees\s+recentes", _back(_RECENT_YEARS), True),
        (r"derniere\s+annee", _back(_TWELVE_MONTH_YEARS), True),
        (rf"{_COUNT}\s+derniers\s+mois", _months_back, True),
        (r"cette\s+decennie", _this_decade, False),
        (rf"{_COUNT}\s+prochaines\s+annees", _ahead(1), False),
    )
)
# Each row is a single year measured from now, read like the spans, and its offset
# from the current year.
_RELATIVE_YEARS: tuple[tuple[re.Pattern[str], int], ...] = tuple(
    (re.compile(rf"^(?:{pattern})$"), offset)
    for pattern, offset in (
        (r"last\s+year", -1),
        (r"(?:so\s+far\s+)?this\s+year(?:\s+so\s+far)?", 0),
        (r"next\s+year", 1),
        (r"ano\s+pasado", -1),
        (r"este\s+ano|lo\s+que\s+va\s+del?\s+ano", 0),
        (r"proximo\s+ano|ano\s+(?:proximo|que\s+viene)", 1),
        (r"l'\s*(?:annee\s+(?:derniere|passee)|an\s+dernier)", -1),
        (r"cette\s+annee|depuis\s+le\s+debut\s+de\s+l'\s*annee", 0),
        (r"l'\s*(?:annee\s+prochaine|an\s+prochain)", 1),
    )
)

_WARN_SEVERITY: WarningSeverity = "warn"


def parse_time_text(text: str, *, current_year: int) -> tuple[list[ExtractedDate], Warning | None]:
    """Parse a free-form time phrase into zero or more ExtractedDate entries
    -- the same structured shape raw_text's LLM extraction produces -- so a
    caller feeds the result through extract.dates_to_request() unchanged.
    Pure and deterministic. Returns dates or a warning without I/O. ``current_year`` resolves
    wording measured from now: a span back ("last N years", "last decade", "recent years",
    "the past year", "this decade") is a closed range ending at that year, a span counted
    back carrying its count of years (`ExtractedDate.trailing_years`), a span ahead
    ("the next N years") a closed range starting the year after it, and "last year",
    "this year" and "next year" are the years before, at and after it."""
    normalised = text.strip().casefold()

    if _LATEST_RE.match(normalised):
        return [ExtractedDate(kind="latest", start=None, end=None)], None

    bounded_match = _BOUNDED_RE.match(normalised)
    if bounded_match:
        dates = _bounded_dates(
            lead=bounded_match.group("lead"),
            start=bounded_match.group("start"),
            separator=bounded_match.group("separator"),
            end=bounded_match.group("end"),
        )
        return (dates, None) if dates else ([], _not_parsed_warning(text))

    relative = _RELATIVE_LEADS_RE.sub("", _fold_accents(normalised), count=1)
    for pattern, span_years, trailing in _RELATIVE_SPANS:
        if span_match := pattern.match(relative):
            token = span_match.groupdict().get("count")
            count = 1 if token is None else _resolve_span_count(token)
            if count is None:
                return [], _not_parsed_warning(text)
            start_year, end_year = span_years(count, current_year)
            date = ExtractedDate(
                kind="range",
                start=str(start_year),
                end=str(end_year),
                trailing_years=end_year - start_year + 1 if trailing else None,
            )
            return [date], None
    for pattern, offset in _RELATIVE_YEARS:
        if pattern.match(relative):
            return [ExtractedDate(kind="point", start=str(current_year + offset))], None

    return [], _not_parsed_warning(text)


def _fold_accents(text: str) -> str:
    """`text` with combining accents dropped and a typographic apostrophe made plain, so
    "últimos años" reads as "ultimos anos" and "l’année" as "l'annee"."""
    decomposed = unicodedata.normalize("NFKD", text.replace("\u2019", "'"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _bounded_dates(
    *, lead: str | None, start: str, separator: str | None, end: str | None
) -> list[ExtractedDate]:
    """The dates a lead, start year, separator and end phrase yield, or an empty list when
    the lead does not take the separator or a year end precedes the start."""
    if separator not in _SEPARATORS_BY_LEAD[lead]:
        return []
    if separator is None and lead is None:
        return [ExtractedDate(kind="point", start=start, end=None)]
    # A "since" lead or a latest end leaves the range open.
    if end is None or not end.isdigit():
        return [ExtractedDate(kind="range", start=start, end=None)]
    if int(start) > int(end):
        return []
    return [ExtractedDate(kind="range", start=start, end=end)]


def _resolve_span_count(token: str) -> int | None:
    """A relative-span count token (a bare digit string or a number word up to twenty
    in English, Spanish or French) resolved to a positive int, or None for a zero count
    ("last 0 years" is well-formed grammar with no representable range)."""
    count = int(token) if token.isdigit() else _NUMBER_WORDS[token]
    return count if count > 0 else None


def _not_parsed_warning(text: str) -> Warning:
    return Warning(
        code=TIME_TEXT_NOT_PARSED,
        severity=_WARN_SEVERITY,
        message=(
            f"time_text {text!r} did not match any recognised date grammar "
            f"({'; '.join(repr(form) for form in TIME_TEXT_FORMS)}, where YYYY is a "
            "four-digit year and N a count); treated as dateless."
        ),
    )


# Emitted when the parsed door receives no time_text and defaults to all periods.
# Supplied text outside the grammar gets TIME_TEXT_NOT_PARSED.
TIME_WINDOW_DEFAULTED = "time_window_defaulted"


def time_window_defaulted_warning() -> Warning:
    """The time_window_defaulted warning for a part with no time_text at
    all, mirroring region/grounding.py's role_scope_defaulted_message
    wording for an undirected entity axis."""
    return Warning(
        code=TIME_WINDOW_DEFAULTED,
        severity=_WARN_SEVERITY,
        message="no time_text was given; the answer was resolved across every period with data.",
    )
