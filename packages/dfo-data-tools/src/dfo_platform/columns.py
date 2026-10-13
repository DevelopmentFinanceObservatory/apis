"""The noun each column of a frame reads under, wherever a column is named for a reader: a
card table's headers and a download's column docs.

A role column takes the noun of its graph role name, so `financingRecipient` reads
"recipient" and `creditor` reads "creditor". A role this module does not name reads "entity".
A reserved frame column (`stat_var`, `time_period`, `value`, `unit`, `status`,
`measurementMethod`, `provenance`) takes its own noun. Graph role names and column keys never
reach a header. This standard-library module keeps `assembler` and `materialise` reading one
table.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

ROLE_NOUNS: Mapping[str, str] = MappingProxyType(
    {
        "observationAbout": "place",
        "provider": "provider",
        "financingProvider": "provider",
        "source": "provider",
        "recipient": "recipient",
        "financingRecipient": "recipient",
        "target": "recipient",
        "creditor": "creditor",
        "debtor": "debtor",
    }
)
"""The noun per graph role name, keyed by the role so a creditor reads "creditor" while
`dfo_platform.roles.ROLE_AXES` places it on the same axis as a provider."""

UNKNOWN_ROLE_NOUN = "entity"
"""The noun of a role column whose role `ROLE_NOUNS` does not name."""

RESERVED_COLUMN_NOUNS: Mapping[str, str] = MappingProxyType(
    {
        "stat_var": "variable",
        "time_period": "period",
        "value": "value",
        "unit": "unit",
        "status": "status",
        "measurementMethod": "measurement method",
        "provenance": "source dataset",
    }
)
"""The noun per reserved frame column, the frame grammar's own columns and facet identity
columns."""


def role_noun(role: str) -> str:
    """The noun for a role column, `UNKNOWN_ROLE_NOUN` when `ROLE_NOUNS` does not name `role`."""
    return ROLE_NOUNS.get(role, UNKNOWN_ROLE_NOUN)


def column_noun(column: str) -> str:
    """The noun for a frame column: a reserved column's own noun, else its role's noun."""
    return RESERVED_COLUMN_NOUNS.get(column) or role_noun(column)


def column_title(column: str) -> str:
    """The header for a frame column, its noun with a capital first letter, such as
    "Recipient" for `financingRecipient` and "Source dataset" for `provenance`."""
    return column_noun(column).capitalize()
