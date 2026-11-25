"""Regenerate `geo_names.json`, the names Search reads a bound party by in a variable phrase.

Each country, region and group in resolvekit's bundled geo modules
(`entity_resolution.shared.GEO_MODULE_IDS`) is read offline: every `geo.country`, every
continent and subregion (`membership.REGION_CLASSES`) but the World, the membership policy's
root, and every group `known_groups()` names. Each entity keeps the names its resolver record
reports as canonical (`resolver.entity(entity_id)`: its canonical name and each name of kind
`canonical`, in every language the record holds); resolvekit's other aliases ("the States",
"Murica", "IT" for Italy, "+256" for Uganda) are fuzzy reads and are left out. A name is also
left out when it holds a digit ("Group of 77", the dated EU snapshots), has fewer than two
letters, is a lower-case code, or is a name of two dcids ("Antarctica").

Entries are keyed by the dcid Search binds (`regions.identity_to_dcid`: `country/USA`, the
region's graph node through `region_ids.json`, the group's registry dcid through
`group_ids.json`). Two resolvekit entities mapping to one dcid (the continent Africa and
"Africa (DAC)", both `africa`) share one entry. The reviewed abbreviations clients write
(`geo_abbreviations.json`: "US", "UK", "G7", "LDCs") join their dcid's names.

Search reads the file at intake (`pack.geo_names`), so a resolvekit upgrade that changes these
names needs it regenerated: `uv run python scripts/export_geo_names.py`. Output JSON is stable.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from entity_resolution import pack
from entity_resolution.membership import (
    COUNTRY_CLASS,
    GEO_DOMAIN,
    REGION_CLASSES,
    policy_data,
)
from entity_resolution.regions import identity_to_dcid
from entity_resolution.shared import GEO_MODULE_IDS

if TYPE_CHECKING:
    from resolvekit.core.model import EntityRecord

_DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "dfo-data-tools"
    / "src"
    / "entity_resolution"
    / "data"
    / "geo_names.json"
)
_CANONICAL = "canonical"


def _kept(name: str) -> bool:
    """Whether `name` can stand as a party's name in a variable phrase: no digit, at least two
    letters, and not a lower-case code ("us", "fr")."""
    letters = [char for char in name if char.isalpha()]
    lower_case = any(char.islower() for char in letters) and not any(
        char.isupper() for char in letters
    )
    return len(letters) >= 2 and not any(char.isdigit() for char in name) and not lower_case


def _key(name: str) -> str:
    return " ".join(name.casefold().split())


def _without_shared(names: dict[str, set[str]]) -> dict[str, set[str]]:
    """`names` less each name, casing and spacing aside, that two dcids hold."""
    owners: dict[str, set[str]] = {}
    for dcid, held in names.items():
        for name in held:
            owners.setdefault(_key(name), set()).add(dcid)
    return {
        dcid: {name for name in held if len(owners[_key(name)]) == 1}
        for dcid, held in names.items()
    }


def _canonical_names(record: EntityRecord) -> Iterable[str]:
    return (record.canonical_name, *(n.value for n in record.names if n.kind == _CANONICAL))


def export_geo_names() -> dict[str, list[str]]:
    """Each dcid's names, sorted, by dcid, sorted. Needs `RESOLVEKIT_OFFLINE` set. The
    resolver opens and closes here, and every read runs on this thread, since resolvekit's
    stores are thread-affine."""
    from resolvekit import Resolver
    from resolvekit.core.api import RoutingMode

    root = policy_data().root
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
            if entity.entity_id != root
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
            names.setdefault(dcid, set()).update(filter(_kept, _canonical_names(record)))
    finally:
        resolver.close()
    names = _without_shared(names)
    for abbreviation, dcid in pack.geo_abbreviations().items():
        if dcid not in names:
            raise SystemExit(f"geo_abbreviations.json maps {abbreviation!r} to unknown {dcid}")
        names[dcid].add(abbreviation)
    return {dcid: sorted(names[dcid]) for dcid in sorted(names) if names[dcid]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(_DEFAULT_OUT))
    args = parser.parse_args()
    os.environ["RESOLVEKIT_OFFLINE"] = "1"
    names = export_geo_names()
    Path(args.out).write_text(
        json.dumps(names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {sum(map(len, names.values()))} names of {len(names)} entities to {args.out}")


if __name__ == "__main__":
    main()
