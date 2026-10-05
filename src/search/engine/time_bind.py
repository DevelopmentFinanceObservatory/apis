"""Parse time phrases into the ``ExtractedDate`` form used by the parsed door.
The caller passes results to ``extract.dates_to_request`` for window semantics.

Input is stripped and casefolded. Four-digit years yield points; ordered ranges yield
ranges. ``Since``, ``depuis``, and ``desde`` yield open-ended ranges. ``latest``,
``current``, and ``now`` yield latest entries. ``last``/``past`` year spans and decades
yield closed ranges ending at caller-supplied ``current_year``. Counts accept digits or
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

_YEAR = r"(\d{4})"
_RANGE_SEP = r"\s*(?:to|-|–)\s*"
_RANGE_RE = re.compile(rf"^{_YEAR}{_RANGE_SEP}{_YEAR}$")
_SINCE_WORDS = r"(?:since|depuis|desde)"
_SINCE_RE = re.compile(rf"^{_SINCE_WORDS}\s+{_YEAR}$")
_POINT_RE = re.compile(rf"^{_YEAR}$")
_LATEST_WORDS = frozenset({"latest", "current", "now"})

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

    if normalised in _LATEST_WORDS:
        return [ExtractedDate(kind="latest", start=None, end=None)], None

    point_match = _POINT_RE.match(normalised)
    if point_match:
        year = point_match.group(1)
        return [ExtractedDate(kind="point", start=year, end=None)], None

    range_match = _RANGE_RE.match(normalised)
    if range_match:
        start, end = range_match.group(1), range_match.group(2)
        if int(start) <= int(end):
            return [ExtractedDate(kind="range", start=start, end=end)], None
        return [], _not_parsed_warning(text)

    since_match = _SINCE_RE.match(normalised)
    if since_match:
        start = since_match.group(1)
        return [ExtractedDate(kind="range", start=start, end=None)], None

    years_match = _RELATIVE_YEARS_RE.match(normalised)
    if years_match:
        span_years = _resolve_span_count(years_match.group(1))
        if span_years is not None:
            return _relative_range(current_year, span_years), None
        return [], _not_parsed_warning(text)

    if _RELATIVE_DECADE_RE.match(normalised):
        return _relative_range(current_year, _DECADE_YEARS), None

    return [], _not_parsed_warning(text)


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
            "(a bare year; 'YYYY to YYYY'/'YYYY-YYYY'; 'since YYYY'/'depuis YYYY'/"
            "'desde YYYY'; 'latest'/'current'/'now'; 'last N years'/'past N years'; "
            "'last decade'/'past decade'/'over the last decade'); treated as dateless."
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
