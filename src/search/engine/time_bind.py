"""Parse time phrases into the ``ExtractedDate`` form used by the parsed door.
The caller passes results to ``extract.dates_to_request`` for window semantics.

Input is stripped and casefolded. Four-digit years yield points. A bounded phrase is an
optional lead, a start year, a separator and an end, and the lead fixes the separator:
``between`` takes ``and``, ``from`` takes ``to`` or ``until``, and no lead takes ``to``,
``-`` or ``–``. A year end yields an ordered range; a latest end ("the latest", "most
recent year available") yields an open-ended range, as ``since``, ``depuis``, and
``desde`` do. The latest phrase alone yields a latest entry. ``last``/``past`` year spans
and decades yield closed ranges ending at caller-supplied ``current_year``.
``TIME_TEXT_FORMS`` lists every accepted form. Counts accept digits or
number words through twenty. Relative spans include the current year, and the fixed
vocabulary covers English, French, and Spanish.

Other phrases, including reversed ranges, return an empty list and a
``TIME_TEXT_NOT_PARSED`` warning. The request stays valid and dateless.
"""

from __future__ import annotations

import re

from dfo_platform import WarningSeverity
from search.engine.extract import ExtractedDate
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
    "last decade",
    "past decade",
    "over the last decade",
)

# Relative spans accept single number words through twenty. Larger counts use digits.
_NUMBER_WORDS: dict[str, int] = {
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
}
_SPAN_WORDS = r"(?:last|past)"
_SPAN_COUNT = rf"(\d{{1,2}}|{'|'.join(_NUMBER_WORDS)})"
_RELATIVE_YEARS_RE = re.compile(rf"^(?:over\s+the\s+)?{_SPAN_WORDS}\s+{_SPAN_COUNT}\s+years?$")
_RELATIVE_DECADE_RE = re.compile(rf"^(?:over\s+the\s+)?{_SPAN_WORDS}\s+decade$")
_DECADE_YEARS = 10

_WARN_SEVERITY: WarningSeverity = "warn"


def parse_time_text(text: str, *, current_year: int) -> tuple[list[ExtractedDate], Warning | None]:
    """Parse a free-form time phrase into zero or more ExtractedDate entries
    -- the same structured shape raw_text's LLM extraction produces -- so a
    caller feeds the result through extract.dates_to_request() unchanged.
    Pure and deterministic. Returns dates or a warning without I/O. ``current_year`` resolves the
    relative-span rows ("last N years", "last decade") against a closed
    range ending at that year."""
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

    years_match = _RELATIVE_YEARS_RE.match(normalised)
    if years_match:
        span_years = _resolve_span_count(years_match.group(1))
        if span_years is not None:
            return _relative_range(current_year, span_years), None
        return [], _not_parsed_warning(text)

    if _RELATIVE_DECADE_RE.match(normalised):
        return _relative_range(current_year, _DECADE_YEARS), None

    return [], _not_parsed_warning(text)


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
    """A relative-span count token (a bare digit string or a number word up
    to twenty) resolved to a positive int, or None for a zero count ("last 0
    years" is well-formed grammar with no representable range)."""
    count = int(token) if token.isdigit() else _NUMBER_WORDS[token]
    return count if count > 0 else None


def _relative_range(current_year: int, span_years: int) -> list[ExtractedDate]:
    """The closed range for a relative span ending at current_year and
    covering span_years years, current_year included (so "last 10 years"
    from 2026 is 2017-2026, ten years inclusive)."""
    start_year = current_year - span_years + 1
    return [ExtractedDate(kind="range", start=str(start_year), end=str(current_year))]


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
