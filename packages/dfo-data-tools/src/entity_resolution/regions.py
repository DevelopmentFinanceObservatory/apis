"""Translate resolvekit IDs to graph dcids.

Custom-pack IDs carry the dcid after the namespace prefix. `m49/*` and `wikidataId/*` use
`region_ids.json`. Other IDs are graph dcids."""

from __future__ import annotations

import json
from functools import lru_cache

from entity_resolution import pack

_CONTAINER_PREFIXES = ("m49/", "wikidataId/")


@lru_cache(maxsize=1)
def _region_map() -> dict[str, str | None]:
    rows = json.loads(pack.region_ids_path().read_text(encoding="utf-8"))
    return {row["resolvekit_id"]: row["dcid"] for row in rows}


def identity_to_dcid(entity_id: str) -> str | None:
    """Return the graph dcid, or `None` for a container ID without a non-null map entry."""
    if entity_id.startswith(pack.NAMESPACE_PREFIX):
        return entity_id[len(pack.NAMESPACE_PREFIX) :]
    if entity_id.startswith(_CONTAINER_PREFIXES):
        return _region_map().get(entity_id)
    return entity_id
