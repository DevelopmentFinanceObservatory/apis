"""Build the `dfo_graph` resolvekit pack from committed records and curated aliases.

The pack uses `domain="org"` and is registered with the shared resolver at startup."""

from __future__ import annotations

import functools
import importlib.resources
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resolvekit.core.model.result import CandidateSummary

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
_ALIAS_COLUMN_PREFIX = "alias"
_ORG_PACK_ID = "org"


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
            row[f"{_ALIAS_COLUMN_PREFIX}{index}"] = alias
        rows.append(row)
    return rows


def _rows(
    *, records_path: Traversable | Path, curated_path: Traversable | Path, normalizer
) -> list[dict]:
    return _build_rows(
        records=_load_records(records_path),
        curated=_load_curated(curated_path),
        normalizer=normalizer,
    )


@dataclass(frozen=True, slots=True)
class AliasIndex:
    """Every name the pack stores for a row, mapped to that row as an org-pack candidate.

    Keys and queries both pass through `normalize`, `OrgNormalizer.normalize_name`, which is
    the form the BYOD build stores as `value_norm` and the listed-name gate compares against."""

    owners: Mapping[str, tuple[CandidateSummary, ...]]
    normalize: Callable[[str], str]

    def owners_of(self, text: str) -> tuple[CandidateSummary, ...]:
        return self.owners.get(self.normalize(text), ())


def build_alias_index(
    *, records_path: Traversable | Path, curated_path: Traversable | Path
) -> AliasIndex:
    """Index the canonical name and every alias of each row `build_pack` writes.

    The rows come from the same `_build_rows` call on the same inputs, so the index holds
    exactly the names the pack holds. Each candidate carries no confidence."""
    from resolvekit.core.model.result import CandidateSummary
    from resolvekit.packs.org.normalizer import OrgNormalizer

    normalizer = OrgNormalizer()
    owners: dict[str, list[CandidateSummary]] = {}
    for row in _rows(records_path=records_path, curated_path=curated_path, normalizer=normalizer):
        candidate = CandidateSummary(
            entity_id=f"{NAMESPACE_PREFIX}{row['id']}",
            canonical_name=row["name"],
            entity_type=row["entity_type"],
            pack_id=_ORG_PACK_ID,
        )
        names = [row["name"]] + [
            value for column, value in row.items() if column.startswith(_ALIAS_COLUMN_PREFIX)
        ]
        for key in dict.fromkeys(normalizer.normalize_name(name) for name in names):
            if key:
                owners.setdefault(key, []).append(candidate)
    return AliasIndex(
        owners={key: tuple(candidates) for key, candidates in owners.items()},
        normalize=normalizer.normalize_name,
    )


@functools.cache
def _committed_alias_index() -> AliasIndex:
    return build_alias_index(records_path=records_path(), curated_path=curated_path())


def is_organisation_name(text: str) -> bool:
    """Whether `text` is a name the pack stores for an organisation ("IMF" for the
    International Monetary Fund), read from the committed records and curated aliases the
    shared resolver indexes, with no resolver started. Countries and other places are
    outside the pack, so a place name is none."""
    return any(
        candidate.entity_type == ENTITY_TYPE_ORGANISATION
        for candidate in _committed_alias_index().owners_of(text)
    )


@functools.cache
def organisation_dcids() -> frozenset[str]:
    """The dcids of the committed records of kind `organisation`: the nodes the graph types
    `Organization`, multilaterals and private foundations alike."""
    return frozenset(
        record["dcid"]
        for record in _load_records(records_path())
        if record["kind"] == "organisation"
    )


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

    rows = _rows(records_path=records_path, curated_path=curated_path, normalizer=OrgNormalizer())

    _FIXED_ROW_KEYS = 4  # id, name, entity_type, and the "dac" code column
    max_row_width = max((len(row) for row in rows), default=_FIXED_ROW_KEYS)
    # `from_records` declares alias columns for the whole pack, so size them to the widest row.
    alias_columns = [f"{_ALIAS_COLUMN_PREFIX}{i}" for i in range(max_row_width - _FIXED_ROW_KEYS)]

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
