"""Convert `sourceReleaseFrequency` ISO 8601 durations to display labels.

Recognised tokens map to `_LABELS`. `known_cadence_label` returns `None` for
an unknown token, and `cadence_label` passes it through unchanged. The Graph API's
node response includes both the raw token and its label.
"""

from __future__ import annotations

_LABELS: dict[str, str] = {
    "P1D": "Daily",
    "P1W": "Weekly",
    "P1M": "Monthly",
    "P3M": "Quarterly",
    "P6M": "Semiannual",
    "P1Y": "Annual",
    "P2Y": "Biennial",
    "P5Y": "Quinquennial",
}


def known_cadence_label(token: str | None) -> str | None:
    """`_LABELS[token]` when the token is recognised, else `None`."""
    if token is None:
        return None
    return _LABELS.get(token)


def cadence_label(token: str | None) -> str | None:
    """`None` for `None` (nothing to convert). Otherwise the recognised label,
    else `token` itself, unchanged.
    """
    return known_cadence_label(token) or token
