"""The committed `geo_names.json` is what `scripts/export_geo_names.py` writes from the
resolvekit geo modules installed here, so a resolvekit upgrade that changes a country's,
region's or group's names fails until the file is regenerated."""

from __future__ import annotations

import json

from entity_resolution import pack
from scripts.export_geo_names import export_geo_names


def test_committed_geo_names_match_the_bundled_store():
    committed = json.loads(pack.geo_names_path().read_text(encoding="utf-8"))

    assert export_geo_names() == committed
