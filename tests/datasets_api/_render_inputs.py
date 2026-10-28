"""The inputs a direct render or frame publish call passes: a first-mint instant, a site and an
enrichment lookup over an empty graph."""

from __future__ import annotations

from datetime import UTC, datetime

from dfo_platform.config import PublicSiteSettings
from materialise.enrichment import EnrichmentLookup

from ._graph_doubles import FakeMintGraphReader

PREPARED_AT = datetime(2026, 10, 4, 9, 0, tzinfo=UTC)
SITE = PublicSiteSettings(public_site_url="https://site.example")
ENRICHMENT = EnrichmentLookup(FakeMintGraphReader(), parties=lambda: {})
