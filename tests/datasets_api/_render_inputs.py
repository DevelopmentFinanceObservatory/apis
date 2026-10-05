"""The bundle inputs a direct `render_and_store` call passes: a first-mint instant and a site."""

from __future__ import annotations

from datetime import UTC, datetime

from dfo_platform.config import PublicSiteSettings

PREPARED_AT = datetime(2026, 10, 4, 9, 0, tzinfo=UTC)
SITE = PublicSiteSettings(public_site_url="https://site.example")
