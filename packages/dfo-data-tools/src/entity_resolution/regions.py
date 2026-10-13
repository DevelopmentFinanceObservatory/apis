"""Translate resolvekit IDs to graph dcids.

Custom-pack IDs carry the dcid after the namespace prefix. `m49/*` and `wikidataId/*` use
`region_ids.json`. A group whose membership has its own graph node takes that node's dcid from
`group_ids.json`, which retires the resolvekit id it replaces. Other IDs are graph dcids."""

from __future__ import annotations

import importlib.resources
import json
from functools import lru_cache

from entity_resolution import pack

_CONTAINER_PREFIXES = ("m49/", "wikidataId/")
_GROUP_IDS_FILENAME = "group_ids.json"


@lru_cache(maxsize=1)
def _region_map() -> dict[str, str | None]:
    rows = json.loads(pack.region_ids_path().read_text(encoding="utf-8"))
    return {row["resolvekit_id"]: row["dcid"] for row in rows}


@lru_cache(maxsize=1)
def _group_map() -> dict[str, str]:
    path = importlib.resources.files(__package__) / "data" / _GROUP_IDS_FILENAME
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["resolvekit_id"]: row["dcid"] for row in rows}


def identity_to_dcid(entity_id: str) -> str | None:
    """Return the graph dcid, or `None` for a container ID without a non-null map entry."""
    if entity_id.startswith(pack.NAMESPACE_PREFIX):
        return entity_id[len(pack.NAMESPACE_PREFIX) :]
    if entity_id.startswith(_CONTAINER_PREFIXES):
        return _region_map().get(entity_id)
    return _group_map().get(entity_id, entity_id)


def replacing_group_id(dcid: str) -> str | None:
    """The group id that replaced `dcid` when `dcid` is a retired resolvekit group id, such as
    `GroupOfSeven` for `groups/G7`. `None` for any other id."""
    return _group_map().get(dcid)
