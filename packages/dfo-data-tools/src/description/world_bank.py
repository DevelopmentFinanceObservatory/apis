"""The definition a World Development Indicators StatVar's graph description gives.

The WDI import writes each series' World Bank note into its StatVar's `description` and
appends the series code in parentheses, sometimes with the note's source after it:
"... (WorldBank Series Code: NY.GDP.MKTP.CD)", "... (WorldBank Series Code: CC.EST, Source:
Detailed documentation of the WGI (...))". The code is no part of the definition, so it is
cut. WDI StatVars sit in the `worldBank/` namespace. World Bank IDS StatVars sit outside it,
and their descriptions are abbreviated series titles ("Debt buyback (current US$)"), so they
give no definition.
"""

from __future__ import annotations

import re
from typing import Final

WORLD_BANK_NAMESPACE: Final = "worldBank/"
"""The dcid prefix of every World Development Indicators StatVar."""

_SERIES_CODE: Final = re.compile(r"\s*\(WorldBank Series Code: [^()]*(?:\([^()]*\)[^()]*)*\)\s*$")
"""The parenthesised series code that ends a WDI description, holding at most one level of
nested parentheses, as a source note's can."""


def world_bank_definition(stat_var: str, description: str | None) -> str | None:
    """The definition `description`, a StatVar's graph `description`, gives `stat_var`.

    Returns:
        For a StatVar in the `worldBank/` namespace, `description` without the series code
        that ends it, stripped; `None` when nothing is left, as when the description holds
        only the code. `None` for any other StatVar and for no description.
    """
    if description is None or not stat_var.startswith(WORLD_BANK_NAMESPACE):
        return None
    return _SERIES_CODE.sub("", description).strip() or None
