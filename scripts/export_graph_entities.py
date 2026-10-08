"""Regenerate graph-derived entity and region packs.

`entities` exports `Organization`, `DACRecipientUnallocatedByCountry` and `DACUnspecifiedParty`
nodes, sorted by dcid, then folds each `_MERGED_DCIDS` source record into its target. A record's
`dac_code` is the node's numerically lowest `dacCode`; its other codes join its aliases, since a
resolvekit entity holds one code per system and every code must resolve.
`regions` maps resolvekit subregions and continents to graph dcids via `/v2/resolve`.

Both commands require `SEARCH_GRAPH_BASE` and read the live graph. Output JSON is stable.
Maintainers author `curated_aliases.json` by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from dc_graph import GraphClient

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ENTITIES_OUT = (
    _REPO_ROOT
    / "packages"
    / "dfo-data-tools"
    / "src"
    / "entity_resolution"
    / "data"
    / "graph_entities.json"
)
_DEFAULT_REGIONS_OUT = (
    _REPO_ROOT
    / "packages"
    / "dfo-data-tools"
    / "src"
    / "entity_resolution"
    / "data"
    / "region_ids.json"
)

# `aggregate` covers both DAC residual classes: stand-ins for parties the DAC does not name.
_KIND_BY_TYPE = {
    "Organization": "organisation",
    "DACRecipientUnallocatedByCountry": "aggregate",
    "DACUnspecifiedParty": "aggregate",
}
_ENTITY_PROPERTIES = ("name", "alternateName", "dacCode")

# The graph holds two "Africa Finance Corporation" nodes. The debt StatVars name the base-schema node
# `Africa_Finance_Corporation` as lendingEntity. The DFO ontology node `AfricaFinanceCorporation`
# carries the "AFC" alias and no data. Delete this entry once ONEcampaign/dfo-ontology#38 mints AFC
# under the base-schema dcid; the export fails until then if either node disappears or is renamed.
_MERGED_DCIDS = {"AfricaFinanceCorporation": "Africa_Finance_Corporation"}

_SUBREGION_TYPE_OF = "UNGeoRegion"
_CONTINENT_TYPE_OF = "Continent"
_TYPE_OF_BY_ENTITY_TYPE = {
    "geo.subregion": _SUBREGION_TYPE_OF,
    "geo.continent": _CONTINENT_TYPE_OF,
}


def _graph_base_url() -> str:
    base = os.environ.get("SEARCH_GRAPH_BASE", "").strip()
    if not base:
        raise SystemExit(
            "SEARCH_GRAPH_BASE must be set -- this script always talks to the "
            "live graph, for both the entities and regions subcommands."
        )
    return base


def _single_name(values: list[str]) -> str:
    if not values:
        raise SystemExit("a node's own name property is empty -- cannot export a nameless entity")
    return values[0]


def _split_dac_codes(values: list[str]) -> tuple[str | None, list[str]]:
    """Return the numerically lowest code and the rest in numeric order."""
    codes = sorted({value for value in values if value}, key=_numeric_code)
    if not codes:
        return None, []
    return codes[0], codes[1:]


def _numeric_code(code: str) -> int:
    if not code.isdigit():
        raise SystemExit(f"dacCode {code!r} is not numeric -- cannot order a node's DAC codes")
    return int(code)


def export_entities(reader: GraphClient, *, max_results: int = 1000) -> list[dict]:
    """Fetch every `_KIND_BY_TYPE` population in dcid order.

    Raise `SystemExit` if any population exceeds `max_results` and would be truncated.
    """
    dcids_by_type: dict[str, tuple[str, ...]] = {}
    for graph_type in _KIND_BY_TYPE:
        page = reader.node_subjects(graph_type, property="typeOf", max_results=max_results)
        if page.truncated:
            raise SystemExit(
                f"{graph_type} population truncated at {max_results} dcids -- raise "
                "max_results, the export must read the full population"
            )
        dcids_by_type[graph_type] = page.dcids

    all_dcids = sorted({dcid for dcids in dcids_by_type.values() for dcid in dcids})
    kind_by_dcid = {
        dcid: _KIND_BY_TYPE[graph_type]
        for graph_type, dcids in dcids_by_type.items()
        for dcid in dcids
    }

    # A single ~160-dcid x 3-property batch trips a nextToken loop on this
    # deployment's mixer, a composition-dependent quirk. Chunking sidesteps it.
    arcs_by_dcid: dict[str, dict[str, tuple[str, ...]] | None] = {}
    chunk_size = 20
    for start in range(0, len(all_dcids), chunk_size):
        chunk = all_dcids[start : start + chunk_size]
        arcs_by_dcid.update(reader.node_arcs_many(chunk, properties=list(_ENTITY_PROPERTIES)))

    records = []
    for dcid in all_dcids:
        arcs = arcs_by_dcid.get(dcid)
        if arcs is None:
            raise SystemExit(
                f"{dcid} was enumerated via <-typeOf but the graph holds no node for it"
            )
        dac_code, other_codes = _split_dac_codes(list(arcs.get("dacCode", ())))
        aliases = [a for a in arcs.get("alternateName", ()) if a]
        records.append(
            {
                "dcid": dcid,
                "name": _single_name(list(arcs.get("name", ()))),
                "kind": kind_by_dcid[dcid],
                "aliases": list(dict.fromkeys([*aliases, *other_codes])),
                "dac_code": dac_code,
            }
        )
    return merge_records(records, _MERGED_DCIDS)


def merge_records(records: list[dict], merges: dict[str, str]) -> list[dict]:
    """Fold each source record into its target and drop the source.

    The target gains the source's aliases and, when it has none, its DAC code. Raise `SystemExit`
    if a source or target is absent, if the two differ in name or kind, or if both carry different
    DAC codes.
    """
    by_dcid = {record["dcid"]: record for record in records}
    for source_dcid, target_dcid in merges.items():
        source = by_dcid.pop(source_dcid, None)
        target = by_dcid.get(target_dcid)
        if source is None or target is None:
            missing = source_dcid if source is None else target_dcid
            raise SystemExit(
                f"pinned merge {source_dcid} -> {target_dcid}: {missing} is absent from the graph "
                "-- remove the entry from _MERGED_DCIDS"
            )
        if (source["name"], source["kind"]) != (target["name"], target["kind"]):
            raise SystemExit(
                f"pinned merge {source_dcid} -> {target_dcid}: the nodes no longer share a name "
                "and kind -- remove the entry from _MERGED_DCIDS"
            )
        if source["dac_code"] and target["dac_code"] and source["dac_code"] != target["dac_code"]:
            raise SystemExit(
                f"pinned merge {source_dcid} -> {target_dcid}: conflicting DAC codes "
                f"{source['dac_code']} and {target['dac_code']}"
            )
        by_dcid[target_dcid] = {
            **target,
            "aliases": list(dict.fromkeys([*target["aliases"], *source["aliases"]])),
            "dac_code": target["dac_code"] or source["dac_code"],
        }
    return [by_dcid[record["dcid"]] for record in records if record["dcid"] in by_dcid]


def _resolve_via_arc(client: httpx.Client, *, name: str, arc: str, type_of: str) -> str | None:
    body = client.post(
        "/v2/resolve", json={"nodes": [name], "property": f"<-{arc}{{typeOf:{type_of}}}->dcid"}
    )
    body.raise_for_status()
    payload = body.json()
    entities = payload.get("entities", [])
    if not entities:
        return None
    candidates = entities[0].get("candidates", [])
    if not candidates:
        return None
    return candidates[0].get("dcid")


def _iter_resolvekit_region_entities() -> list[tuple[str, str, str]]:
    """Return region rows from resolvekit's bundled modules, sorted by id.

    Each row is `(resolvekit_id, entity_type, canonical_name)`.
    """
    import sqlite3

    os.environ.setdefault("RESOLVEKIT_OFFLINE", "1")
    from resolvekit.core.module_registry import get_module_path

    rows: list[tuple[str, str, str]] = []
    for module_id in ("geo.regions", "geo.continents"):
        module_dir = get_module_path(module_id)
        db_files = sorted(module_dir.glob("*.sqlite"))
        if len(db_files) != 1:
            raise SystemExit(f"expected exactly one *.sqlite under {module_dir}, found {db_files}")
        conn = sqlite3.connect(db_files[0])
        try:
            cursor = conn.execute(
                "SELECT entity_id, entity_type, canonical_name FROM entities "
                "WHERE entity_type IN ('geo.subregion', 'geo.continent')"
            )
            rows.extend(cursor.fetchall())
        finally:
            conn.close()
    rows.sort(key=lambda row: row[0])
    return rows


def export_regions(*, graph_base_url: str, token: str | None) -> list[dict]:
    """Map each resolvekit region's canonical name to a graph dcid.

    Try the typed `description` arc, then `name`.

    Unresolved rows carry `dcid: null` for reporting.
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["x-graph-token"] = token
    client = httpx.Client(base_url=f"{graph_base_url}/core/api", headers=headers, timeout=30.0)
    rows = []
    try:
        for resolvekit_id, entity_type, name in _iter_resolvekit_region_entities():
            type_of = _TYPE_OF_BY_ENTITY_TYPE[entity_type]
            dcid = _resolve_via_arc(client, name=name, arc="description", type_of=type_of)
            if dcid is None:
                dcid = _resolve_via_arc(client, name=name, arc="name", type_of=type_of)
            rows.append({"resolvekit_id": resolvekit_id, "name": name, "dcid": dcid})
    finally:
        client.close()
    return rows


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cmd_entities(args: argparse.Namespace) -> None:
    base = _graph_base_url()
    reader = GraphClient(base_url=f"{base}/core/api", token=os.environ.get("SEARCH_GRAPH_TOKEN"))
    try:
        records = export_entities(reader)
    finally:
        reader.close()
    _write_json(Path(args.out), records)
    print(f"wrote {len(records)} entity records to {args.out}", file=sys.stderr)


def _cmd_regions(args: argparse.Namespace) -> None:
    base = _graph_base_url()
    rows = export_regions(graph_base_url=base, token=os.environ.get("SEARCH_GRAPH_TOKEN"))
    _write_json(Path(args.out), rows)
    null_rows = [row["name"] for row in rows if row["dcid"] is None]
    print(f"wrote {len(rows)} region rows to {args.out}", file=sys.stderr)
    if null_rows:
        print(f"unmapped region names ({len(null_rows)}): {null_rows}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    entities_parser = subparsers.add_parser("entities", help="export graph_entities.json")
    entities_parser.add_argument("--out", default=str(_DEFAULT_ENTITIES_OUT))
    entities_parser.set_defaults(func=_cmd_entities)

    regions_parser = subparsers.add_parser("regions", help="export region_ids.json")
    regions_parser.add_argument("--out", default=str(_DEFAULT_REGIONS_OUT))
    regions_parser.set_defaults(func=_cmd_regions)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
