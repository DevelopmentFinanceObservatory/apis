"""Build the sources response's `usage` object. Both entries are null for every dcid.

A runnable sample needs a StatVar. `GET /api/graph/node/{dcid}/members?relation=variables`
lists the variables the code index associates with a Dataset or Provenance, but the
documentation route does not read it, and a source the code index does not associate lists
none. The API returns null usage entries, listed in `UNCONDITIONAL_GAPS`.
"""

from __future__ import annotations

from graph_api.source_models import Usage


def usage_document() -> Usage:
    """Return `Usage()` with both entries null."""
    return Usage()
