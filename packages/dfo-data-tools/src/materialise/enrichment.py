"""Batched graph lookups for materialisation's country-code, label and place-type enrichment
fields, the party-kind field the code index gives, and `read_party_kinds`, Search's reader of a
party's kind. Every reader takes the kind from the code index alone (`party_kind_of`)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping

from code_index.types import PartyKind, PartyRow
from dfo_platform.roles import role_key_axis
from materialise.graph import GraphReader

ENRICHMENT_PROPERTIES: Mapping[str, str] = {
    "iso3": "countryAlpha3Code",
    "iso2": "countryAlpha2Code",
    "label": "name",
    "type": "typeOf",
}
"""Enrichment field name to the graph out-arc property that supplies it. Every field comes from
one batched `node_arcs_many` read, so asking for `type` beside `label` costs no extra read."""

PARTY_KIND_FIELD = "party_kind"
"""The field a lookup derives with no graph read: the party's kind (`party`, `total`,
`residual`) the code index declares (`party_kind_of`)."""

ENRICHMENT_FIELDS: tuple[str, ...] = (*ENRICHMENT_PROPERTIES, PARTY_KIND_FIELD)
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
    """Resolves refs to available `iso3`, `iso2`, `label`, `type` and `party_kind` fields in an
    LRU capped at `_CACHE_CEILING`. Each entry caches all fields. Lookups promote refs to
    most-recently used.

    `parties` is the code index's party rows (`CodeIndex.parties`), which give a ref its
    `party_kind` (`party_kind_of`). A ref they do not list has no kind, whatever its graph types.

    The lock guards cache access only. Graph calls run unlocked, so concurrent misses may
    duplicate requests. Results use each call's own hits and fetches, even if another call
    evicts them. A lookup returns every requested ref, even when the request exceeds the cap."""

    def __init__(
        self,
        graph: GraphReader,
        *,
        parties: Mapping[str, PartyRow],
    ) -> None:
        self._graph = graph
        self.parties = parties
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, Mapping[str, str]] = OrderedDict()

    def lookup(
        self, refs: Iterable[str], *, deadline: float | None = None
    ) -> Mapping[str, Mapping[str, str]]:
        """Returns unique refs in input order, mapping available fields to values. Absent refs
        map to `{}`, or to the `party_kind` their `parties` row gives. Cached refs avoid graph
        calls. New refs are fetched in one unlocked batch, bounded by the absolute monotonic
        `deadline` when given."""
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
                ref: _resolve_fields(ref, fetched.get(ref), parties=self.parties) for ref in missing
            }
            self._store(resolved)
            hits.update(resolved)
        return {ref: hits[ref] for ref in requested}

    def party_kind(self, ref: str) -> PartyKind | None:
        """The kind `parties` declares for `ref` (`party_kind_of`), read with no graph call, so
        it holds when a `lookup` fails. `lookup` gives `ref` the same kind."""
        return party_kind_of(ref, parties=self.parties)

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
    parties: Mapping[str, PartyRow],
) -> Mapping[str, str]:
    """Each field's first value, except `type`: every `typeOf` value of the place, sorted and
    comma-joined (`AdministrativeArea1,Country,EurostatNUTS2`), since a place can hold several
    levels at once. `party_kind` is `party_kind_of` the ref, absent when it gives none. A `ref`
    with no `node` and no `parties` row maps to `{}`."""
    fields = {} if node is None else _node_fields(node)
    if (kind := party_kind_of(ref, parties=parties)) is not None:
        fields[PARTY_KIND_FIELD] = kind
    return fields


def _node_fields(node: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for property_, field in _FIELDS_BY_PROPERTY.items():
        values = node.get(property_, ())
        if values:
            fields[field] = ",".join(sorted(values)) if property_ == _TYPE_PROPERTY else values[0]
    return fields


def party_kind_of(ref: str, *, parties: Mapping[str, PartyRow]) -> PartyKind | None:
    """The kind the code index's `parties` row declares for the party `ref`, which holds
    whatever the graph says (a group of organisations is typed `Organization` and declared
    `total`). `None` for a ref the rows do not list.

    Search, the options document and every download read this one source, since the index
    lists every party a declared dataset publishes. A coverage build carrying a party the index
    was not exported for leaves that party with no kind. That is index drift, which the
    code-index drift gate reports and a re-export of the index repairs."""
    party = parties.get(ref)
    return None if party is None else party.kind


def is_party_role(role: str) -> bool:
    """Whether an entity in `role` is a party of a flow: a role on a flow axis (`origin`,
    `destination`, or a graph role placed on one, `dfo_platform.roles.role_key_axis`). A subject
    role (`observationAbout`) holds no party."""
    return role_key_axis(role) is not None


def read_party_kinds(
    entities: Iterable[tuple[str, str]], *, parties: Mapping[str, PartyRow]
) -> dict[tuple[str, str], PartyKind]:
    """The kind of each `(role, ref)` entity whose role is a party role (`is_party_role`), keyed
    by the pair: `party_kind_of` over the code index's `parties` rows. An
    entity the rows do not list, or in a subject role, is absent."""
    kinds: dict[tuple[str, str], PartyKind] = {}
    for role, ref in dict.fromkeys(entities):
        if not is_party_role(role):
            continue
        kind = party_kind_of(ref, parties=parties)
        if kind is not None:
            kinds[role, ref] = kind
    return kinds
