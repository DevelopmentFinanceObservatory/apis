"""Batched graph lookups for materialisation's country-code, label, place-type and party-kind
enrichment fields."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping

from code_index.types import PartyKind, PartyRow, party_kind
from materialise.graph import GraphReader

ENRICHMENT_PROPERTIES: Mapping[str, str] = {
    "iso3": "countryAlpha3Code",
    "iso2": "countryAlpha2Code",
    "label": "name",
    "type": "typeOf",
}
"""Enrichment field name to the graph out-arc property that supplies it. Every field comes from
one batched `node_arcs_many` read, so asking for `type` beside `label` costs no extra read."""

KIND_FIELD = "kind"
"""The field a lookup derives with no read of its own: the kind the code index declares for the
ref (`CodeIndex.parties`), else the kind the node's `typeOf` values give under the declared party
types (`code_index.types.party_kind`)."""

ENRICHMENT_FIELDS: tuple[str, ...] = (*ENRICHMENT_PROPERTIES, KIND_FIELD)
"""Every field a lookup can return."""

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
    """Resolves refs to available `iso3`, `iso2`, `label`, `type` and `kind` fields in an LRU
    capped at `_CACHE_CEILING`. Each entry caches all fields. Lookups promote refs to
    most-recently used.

    `parties` and `type_kinds` are the code index's party declarations (`CodeIndex.parties` and
    `CodeIndex.party_types`). A ref has the `kind` its `parties` row gives it, which can differ
    from its types' (a group of organisations is typed `Organization`); a ref with no row has
    the kind its graph types give under `type_kinds`, when they give one. With both empty, no ref
    has a kind.

    The lock guards cache access only. Graph calls run unlocked, so concurrent misses may
    duplicate requests. Results use each call's own hits and fetches, even if another call
    evicts them. A lookup returns every requested ref, even when the request exceeds the cap."""

    def __init__(
        self,
        graph: GraphReader,
        *,
        type_kinds: Mapping[str, PartyKind | None],
        parties: Mapping[str, PartyRow],
    ) -> None:
        self._graph = graph
        self._type_kinds = type_kinds
        self._parties = parties
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Mapping[str, str]] = OrderedDict()

    def lookup(
        self, refs: Iterable[str], *, deadline: float | None = None
    ) -> Mapping[str, Mapping[str, str]]:
        """Returns unique refs in input order, mapping available fields to values. Absent refs
        map to `{}`, or to the `kind` their `parties` row gives. Cached refs avoid graph calls.
        New refs are fetched in one unlocked batch, bounded by the absolute monotonic `deadline`
        when given."""
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
            resolved = {
                ref: _resolve_fields(
                    ref, fetched.get(ref), type_kinds=self._type_kinds, parties=self._parties
                )
                for ref in missing
            }
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


def _resolve_fields(
    ref: str,
    node: Mapping[str, tuple[str, ...]] | None,
    *,
    type_kinds: Mapping[str, PartyKind | None],
    parties: Mapping[str, PartyRow],
) -> Mapping[str, str]:
    """Each field's first value, except `type`: every `typeOf` value of the place, sorted and
    comma-joined (`AdministrativeArea1,Country,EurostatNUTS2`), since a place can hold several
    levels at once. `kind` is the kind `parties` gives `ref`, which holds with no `node`
    (the graph has no node for `ref`, or its read failed); else the kind the node's types give
    under `type_kinds`, absent when neither gives one. A `ref` with no `node` and no `parties`
    row maps to `{}`."""
    fields = {} if node is None else _node_fields(node)
    if (kind := _kind(ref, node, type_kinds=type_kinds, parties=parties)) is not None:
        fields[KIND_FIELD] = kind
    return fields


def _node_fields(node: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for property_, field in _FIELDS_BY_PROPERTY.items():
        values = node.get(property_, ())
        if values:
            fields[field] = ",".join(sorted(values)) if property_ == _TYPE_PROPERTY else values[0]
    return fields


def _kind(
    ref: str,
    node: Mapping[str, tuple[str, ...]] | None,
    *,
    type_kinds: Mapping[str, PartyKind | None],
    parties: Mapping[str, PartyRow],
) -> PartyKind | None:
    if (party := parties.get(ref)) is not None:
        return party.kind
    if node is None:
        return None
    return party_kind(node.get(_TYPE_PROPERTY, ()), type_kinds)
