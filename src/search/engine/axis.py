"""Classify constraint properties by override, observed dates, or place-value share.

An exact override wins first. Otherwise, any date-shaped value selects ``when``. A place
namespace in at least 90% of values selects ``where``. Other samples and empty samples
select ``how``. The classifier uses only stdlib and has no side effects.
"""

from __future__ import annotations

from typing import Literal

Axis = Literal["what", "how", "where", "when"]

AXIS_OVERRIDES: dict[str, Axis] = {
    "comparisonRegion": "where",
    # Place-named properties with categorical enum values.
    "placeOfBirth": "how",
    "placeOfResidenceClassification": "how",
    "placeOfWork": "how",
    "placeCategory": "how",
    "locationType": "how",
    "jurisdiction": "how",
    "computerUsageLocation": "how",
    "internetUsageLocation": "how",
    # Time-named properties with duration or epoch-bucket values.
    "dateBuilt": "how",
    "periodOfMilitaryService": "how",
    "instrumentTerm": "how",
    "maturity": "how",
    "commuteTime": "how",
    "accumulationPeriod": "how",
    "extremesOverTime": "how",
}

_PLACE_NAMESPACES: frozenset[str] = frozenset(
    {
        "country",
        "geoId",
        "nuts",
        "wikidataId",
        "Earth",
        "continent",
        "undata-geo",
    }
)

# Matches YYYY, YYYY-MM, and YYYY-MM-DD.
import re as _re  # noqa: E402 — stdlib, after module-level constants

_DATE_RE = _re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def _dcid_namespace(dcid: str) -> str:
    """Return the prefix before the first '/', or the whole string if no '/'."""
    slash = dcid.find("/")
    return dcid[:slash] if slash != -1 else dcid


def classify_axis(property_dcid: str, observed_values: list[str]) -> Axis:
    """Classify by exact override, then any ISO-date value, then place-value share.

    A place namespace in at least 90% of non-empty samples selects ``where``. Other
    samples and empty samples select ``how`` unless an override matches.
    """
    if property_dcid in AXIS_OVERRIDES:
        return AXIS_OVERRIDES[property_dcid]

    if not observed_values:
        return "how"

    if any(_DATE_RE.match(v) for v in observed_values):
        return "when"

    place_count = sum(1 for v in observed_values if _dcid_namespace(v) in _PLACE_NAMESPACES)
    place_frac = place_count / len(observed_values)
    if place_frac >= 0.9:
        return "where"

    return "how"
