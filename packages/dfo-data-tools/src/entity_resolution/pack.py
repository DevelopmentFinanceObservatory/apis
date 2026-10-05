"""Build the `dfo_graph` resolvekit pack from committed records and curated aliases.

The pack uses `domain="org"` and is registered with the shared resolver at startup."""

from __future__ import annotations

import importlib.resources
import json
import os
from importlib.resources.abc import Traversable
from pathlib import Path

MODULE_ID = "dfo_graph"
"""The pack's flat resolvekit module id. Flat because `namespace` (and so
`module_id`, which the builder writes as exactly `namespace`) is validated
against `^[a-zA-Z0-9][a-zA-Z0-9_-]*$` -- no dots -- so this cannot follow the
bundled packs' dotted convention (`geo.countries`, `org.data_sources`).
"""

NAMESPACE_PREFIX = f"{MODULE_ID}/"
"""resolvekit prefixes every custom entity id with its namespace. A caller
strips this prefix to recover the dcid a resolved entity id names."""

ENTITY_TYPE_ORGANISATION = "org.organisation"
ENTITY_TYPE_AGGREGATE = "org.dac_aggregate"

_PACKAGE = __package__
_DATA_DIR = "data"
_RECORDS_FILENAME = "graph_entities.json"
_CURATED_FILENAME = "curated_aliases.json"
_REGION_IDS_FILENAME = "region_ids.json"
_QUERY_ALIASES_FILENAME = "query_aliases.json"

_CODE_SYSTEM = "dac"


def records_path() -> Traversable:
    """The packaged `graph_entities.json` path, via `importlib.resources` so
    this works whether the package is unpacked on disk or served from a
    wheel/zip."""
    return importlib.resources.files(_PACKAGE) / _DATA_DIR / _RECORDS_FILENAME


def curated_path() -> Traversable:
    """The packaged `curated_aliases.json` path."""
    return importlib.resources.files(_PACKAGE) / _DATA_DIR / _CURATED_FILENAME


def region_ids_path() -> Traversable:
    """The packaged `region_ids.json` path."""
    return importlib.resources.files(_PACKAGE) / _DATA_DIR / _REGION_IDS_FILENAME


def query_aliases_path() -> Traversable:
    """The packaged `query_aliases.json` path."""
    return importlib.resources.files(_PACKAGE) / _DATA_DIR / _QUERY_ALIASES_FILENAME


class CuratedAliasError(ValueError):
    """A curated alias names a dcid absent from the committed records."""


def _load_records(path: Traversable) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_curated(path: Traversable) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity_type(kind: str) -> str:
    if kind == "organisation":
        return ENTITY_TYPE_ORGANISATION
    if kind == "aggregate":
        return ENTITY_TYPE_AGGREGATE
    raise ValueError(f"unknown record kind {kind!r}, expected 'organisation' or 'aggregate'")


def _deduped_aliases(*, name: str, raw_aliases: list[str], normalizer) -> list[str]:
    """Keep aliases in order, dropping empty or duplicate normalized names.

    `OrgNormalizer` catches legal-suffix collisions that casefolding misses."""
    seen = {normalizer.normalize_name(name)}
    kept = []
    for alias in raw_aliases:
        if not alias:
            continue
        key = normalizer.normalize_name(alias)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(alias)
    return kept


def _build_rows(*, records: list[dict], curated: dict[str, list[str]], normalizer) -> list[dict]:
    dcids = {record["dcid"] for record in records}
    unknown = sorted(set(curated) - dcids)
    if unknown:
        raise CuratedAliasError(
            f"curated_aliases.json names {len(unknown)} dcid(s) absent from the "
            f"committed records, refusing to build a pack with a dangling curated "
            f"alias: {unknown}"
        )

    rows = []
    for record in records:
        dcid = record["dcid"]
        raw_aliases = list(record.get("aliases") or []) + list(curated.get(dcid, []))
        aliases = _deduped_aliases(
            name=record["name"], raw_aliases=raw_aliases, normalizer=normalizer
        )
        row = {
            "id": dcid,
            "name": record["name"],
            "entity_type": _entity_type(record["kind"]),
            _CODE_SYSTEM: record.get("dac_code"),
        }
        for index, alias in enumerate(aliases):
            row[f"alias{index}"] = alias
        rows.append(row)
    return rows


def build_pack(
    *, records_path: Traversable | Path, curated_path: Traversable | Path, out_dir: Path
) -> Path:
    """Build the pack and return its `<out_dir>/byod/<content-hash>/` directory.

    Read `{dcid, name, kind, aliases, dac_code}` records and a `{dcid: [alias, ...]}` map.
    Set `RESOLVEKIT_OFFLINE=1` and `RESOLVEKIT_CACHE_DIR=out_dir`, and build with
    `cache=True`, so a pack already cached for the same inputs is returned unwritten. The
    returned directory is the one these inputs hash to, whatever other packs `out_dir`
    holds. Raise `CuratedAliasError` for unknown curated dcids."""
    # The build step `Resolver.from_records` runs before it loads the pack; it reports the
    # pack directory, which the public constructor does not. resolvekit is pinned exactly.
    from resolvekit.core.api._byod import prepare_standalone_pack
    from resolvekit.packs.org.normalizer import OrgNormalizer

    records = _load_records(records_path)
    curated = _load_curated(curated_path)
    normalizer = OrgNormalizer()
    rows = _build_rows(records=records, curated=curated, normalizer=normalizer)

    _FIXED_ROW_KEYS = 4  # id, name, entity_type, and the "dac" code column
    max_row_width = max((len(row) for row in rows), default=_FIXED_ROW_KEYS)
    # `from_records` declares alias columns for the whole pack, so size them to the widest row.
    alias_columns = [f"alias{i}" for i in range(max_row_width - _FIXED_ROW_KEYS)]

    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RESOLVEKIT_CACHE_DIR"] = str(out_dir)
    os.environ["RESOLVEKIT_OFFLINE"] = "1"

    outcome = prepare_standalone_pack(
        data=rows,
        domain="org",
        namespace=MODULE_ID,
        name="name",
        id="id",
        aliases=alias_columns,
        codes=[_CODE_SYSTEM],
        attrs=None,
        entity_type="entity_type",
        cache=True,
    )
    return Path(outcome.pack_dir)
