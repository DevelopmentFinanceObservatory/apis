"""Build the sources response's `usage` object. Both entries are null for every dcid.

A runnable sample needs a StatVar, but no API maps a source dcid to its
variables. The graph's reverse `isPartOf` arc lists a Dataset's
Provenance children, not its StatVars. Finding those variables would need
an observation-wide scan.

SDMX `data()` and `availability()` also require `variableMeasured`.
`provenance` only narrows a variable query. It cannot enumerate variables
for a source. The API returns null usage entries, listed in
`UNCONDITIONAL_GAPS`.
"""

from __future__ import annotations

from graph_api.source_models import Usage


def usage_document() -> Usage:
    """Return `Usage()` with both entries null."""
    return Usage()
