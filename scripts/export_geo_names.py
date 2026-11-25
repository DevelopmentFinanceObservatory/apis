"""Regenerate `geo_names.json`, the names and aliases of every country, region and group.

Read from resolvekit's bundled geo modules (`entity_resolution.shared.GEO_MODULE_IDS`) offline:
every `geo.country`, every continent and subregion (`membership.REGION_CLASSES`) and every group
`known_groups()` names. Each entity's names are the ones the resolver's own record lists
(`resolver.entity(entity_id)`: its canonical name and each alias), keyed by the dcid Search binds
for it (`regions.identity_to_dcid`: `country/USA`, the region's graph node through
`region_ids.json`, the group's registry dcid through `group_ids.json`). Two resolvekit entities
mapping to one dcid (the continent Africa and the DAC's "Africa (DAC)", both `africa`) share one
entry holding both entities' names. An entity that maps to no dcid is left out.

Search reads these names at intake (`pack.geo_names`), so a resolvekit upgrade that changes them
needs the file regenerated: `uv run python scripts/export_geo_names.py`. Output JSON is stable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from entity_resolution.membership import COUNTRY_CLASS, GEO_DOMAIN, REGION_CLASSES
from entity_resolution.regions import identity_to_dcid
from entity_resolution.shared import GEO_MODULE_IDS

_DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "dfo-data-tools"
    / "src"
    / "entity_resolution"
    / "data"
    / "geo_names.json"
)


def export_geo_names() -> dict[str, list[str]]:
    """Each dcid's names, sorted, by dcid, sorted. The resolver opens and closes here, and
    every read runs on this thread, since resolvekit's stores are thread-affine."""
    os.environ["RESOLVEKIT_OFFLINE"] = "1"
    from resolvekit import Resolver
    from resolvekit.core.api import RoutingMode

    resolver = Resolver.lite(
        module_ids=list(GEO_MODULE_IDS),
        routing_mode=RoutingMode.EXPLICIT,
        cache_size=0,
        warm=False,
    )
    try:
        store = resolver.store_for(GEO_DOMAIN)
        entity_ids = [
            entity.entity_id
            for entity_type in (COUNTRY_CLASS, *REGION_CLASSES)
            for entity in store.list_entities_by_type(entity_type)
        ]
        for name in resolver.known_groups():
            entity_id = resolver.resolve_id(name)
            if entity_id is None:
                raise SystemExit(f"resolvekit group {name!r} resolves to no entity")
            entity_ids.append(entity_id)
        names: dict[str, set[str]] = {}
        for entity_id in entity_ids:
            dcid = identity_to_dcid(entity_id)
            record = resolver.entity(entity_id)
            if dcid is None or record is None:
                continue
            names.setdefault(dcid, set()).update(
                (record.canonical_name, *(name.value for name in record.names))
            )
    finally:
        resolver.close()
    return {dcid: sorted(names[dcid]) for dcid in sorted(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()
    names = export_geo_names()
    Path(args.out).write_text(
        json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(names)} entities to {args.out}")


if __name__ == "__main__":
    main()
