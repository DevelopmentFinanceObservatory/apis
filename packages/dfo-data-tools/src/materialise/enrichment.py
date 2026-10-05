"""Batched graph lookups for materialisation's country-code, label and place-type enrichment
fields."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping

from materialise.graph import GraphReader

ENRICHMENT_PROPERTIES: Mapping[str, str] = {
    "iso3": "countryAlpha3Code",
    "iso2": "countryAlpha2Code",
    "label": "name",
    "type": "typeOf",
}
"""Enrichment field name to the graph out-arc property that supplies it. Every field comes from
one batched `node_arcs_many` read, so asking for `type` beside `label` costs no extra read."""

_TYPE_PROPERTY = "typeOf"

_PROPERTIES = tuple(ENRICHMENT_PROPERTIES.values())
_FIELDS_BY_PROPERTY = {property_: field for field, property_ in ENRICHMENT_PROPERTIES.items()}

_CACHE_CEILING = 50_000
"""Maximum cached refs. Bounds cache growth across the process lifetime."""

_INDIVIDUAL_PROBE_BUDGET = 200
"""Caps individual suspect probes in `node_arcs_many` at 200. Large frames can contain many
absent refs. The cap avoids one POST per ref while ordinary refs resolve in the batch and
its group retry."""


class EnrichmentLookup:
    """Resolves refs to available `iso3`, `iso2`, `label` and `type` fields in an LRU capped at
    `_CACHE_CEILING`. Each entry caches all fields. Lookups promote refs to most-recently used.

    The lock guards cache access only. Graph calls run unlocked, so concurrent misses may
    duplicate requests. Results use each call's own hits and fetches, even if another call
    evicts them. A lookup returns every requested ref, even when the request exceeds the cap."""

    def __init__(self, graph: GraphReader) -> None:
        self._graph = graph
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Mapping[str, str]] = OrderedDict()

    def lookup(
        self, refs: Iterable[str], *, deadline: float | None = None
    ) -> Mapping[str, Mapping[str, str]]:
        """Returns unique refs in input order, mapping available fields to values. Absent refs
        map to `{}`. Cached refs avoid graph calls. New refs are fetched in one unlocked batch,
        bounded by the absolute monotonic `deadline` when given."""
        requested = list(dict.fromkeys(refs))
        if not requested:
            return {}
        hits, missing = self._hits_and_missing(requested)
        if missing:
            fetched = self._graph.node_arcs_many(
                missing,
                properties=_PROPERTIES,
                individual_probe_budget=_INDIVIDUAL_PROBE_BUDGET,
                deadline=deadline,
            )
            resolved = {ref: _resolve_fields(fetched.get(ref)) for ref in missing}
            self._store(resolved)
            hits.update(resolved)
        return {ref: hits[ref] for ref in requested}

    def _hits_and_missing(
        self, requested: list[str]
    ) -> tuple[dict[str, Mapping[str, str]], list[str]]:
        with self._lock:
            hits: dict[str, Mapping[str, str]] = {}
            missing: list[str] = []
            for ref in requested:
                if ref in self._cache:
                    self._cache.move_to_end(ref)
                    hits[ref] = self._cache[ref]
                else:
                    missing.append(ref)
            return hits, missing

    def _store(self, resolved: Mapping[str, Mapping[str, str]]) -> None:
        with self._lock:
            for ref, fields in resolved.items():
                self._cache[ref] = fields
                self._cache.move_to_end(ref)
            while len(self._cache) > _CACHE_CEILING:
                self._cache.popitem(last=False)


def _resolve_fields(node: Mapping[str, tuple[str, ...]] | None) -> Mapping[str, str]:
    """Each field's first value, except `type`: every `typeOf` value of the place, sorted and
    comma-joined (`AdministrativeArea1,Country,EurostatNUTS2`), since a place can hold several
    levels at once."""
    if node is None:
        return {}
    fields: dict[str, str] = {}
    for property_, field in _FIELDS_BY_PROPERTY.items():
        values = node.get(property_, ())
        if values:
            fields[field] = ",".join(sorted(values)) if property_ == _TYPE_PROPERTY else values[0]
    return fields
