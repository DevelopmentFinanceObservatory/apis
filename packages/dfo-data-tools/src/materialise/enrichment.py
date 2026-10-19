"""Batched graph lookups for materialisation's country-code, label and place-type enrichment
fields, the party-kind field the code index gives, and `read_party_kinds`, Search's reader of a
party's kind. Every reader takes the kind from the code index alone (`party_kind_of`). A frame's
`<role>_party_kind` cell is that kind, except that a whole-population total reads
`whole_total` (`party_kind_cell`)."""

from __future__ import annotations

import functools
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Literal

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
"""The field a lookup derives with no graph read: the party's kind cell (`party`, `private`,
`total`, `whole_total` or `residual`) the code index declares (`party_kind_cell`)."""

WHOLE_TOTAL_CELL = "whole_total"
"""The cell of a total that holds every party of its marker's kinds on its axis in the row's
dataset (`PartyRow.whole`, `code_index.types.WholeMarker`). It
is a `total` for every reader that cuts or splits totals, and the one total a pooled role may
sum alone (two whole totals on one axis hold disjoint kinds, so their sum counts no party
twice)."""

type PartyKindCell = PartyKind | Literal["whole_total"]
"""What a `<role>_party_kind` cell holds: a `PartyKind`, or `whole_total` for a whole total."""

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
    `party_kind` cell (`party_kind_cell`). A ref they do not list has no kind, whatever its graph
    types.

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

    def party_kind(
        self, ref: str, *, year: int | None = None, provenance: str | None = None
    ) -> PartyKindCell | None:
        """The `party_kind` cell `parties` gives `ref` (`party_kind_cell`, for a row of
        `provenance` in `year` when given), read with no graph call, so it holds when a `lookup`
        fails. `lookup` gives `ref` the cell for no dataset and year, so a whole total reads
        `total` there."""
        return party_kind_cell(ref, parties=self.parties, year=year, provenance=provenance)

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
    levels at once. `party_kind` is `party_kind_cell` of the ref for no dataset and year, absent when it
    gives none. A `ref` with no `node` and no `parties` row maps to `{}`."""
    fields = {} if node is None else _node_fields(node)
    if (kind := party_kind_cell(ref, parties=parties)) is not None:
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


def party_kind_cell(
    ref: str,
    *,
    parties: Mapping[str, PartyRow],
    year: int | None = None,
    provenance: str | None = None,
) -> PartyKindCell | None:
    """The `<role>_party_kind` cell of the party `ref` for a row of `provenance` in `year`: its
    kind (`party_kind_of`), except that a `total` with a marker for `provenance` reads
    `whole_total` when the marker applies. It applies in every year when the marker's `years`
    is `None`, and otherwise only in a `year` inside that window. A row with no `provenance`,
    or no `year` where a window is set, keeps `total`, the reading that holds in every dataset
    and year. `None` for a ref the rows do not list."""
    party = parties.get(ref)
    if party is None:
        return None
    marker = None if provenance is None else party.marker(provenance)
    if party.kind == "total" and marker is not None and _in_window(marker.years, year):
        return WHOLE_TOTAL_CELL
    return party.kind


def row_party_kind_cell(
    ref: object, *, parties: Mapping[str, PartyRow], period: object, provenance: object
) -> PartyKindCell | None:
    """The `<role>_party_kind` cell of a frame row whose party cell is `ref`, `time_period` cell
    `period` and `provenance` cell `provenance` (`None` for a table without that column): the
    `party_kind_cell` for the year of `period` (`period_year`) and that dataset. `None` for a
    `ref` that is not text. A `provenance` that is not text reads as no dataset."""
    if not isinstance(ref, str):
        return None
    return party_kind_cell(
        ref,
        parties=parties,
        year=period_year(period),
        provenance=provenance if isinstance(provenance, str) else None,
    )


_POINT = r"(\d{4})(?:-\d{2}(?:-\d{2})?)?"
_PERIOD = re.compile(rf"{_POINT}(?:/{_POINT})?")
"""A `time_period` cell as the frame grammar spells it: `YYYY`, `YYYY-MM` or `YYYY-MM-DD`, or two
such points joined by `/`. The years are captured; this module reads no more of the grammar."""


def period_year(cell: object) -> int | None:
    """The one year a row's `time_period` cell lies in, for `party_kind_cell`: the year of a
    point (`2021`, `2021-06`, `2021-06-30`) or of an interval inside one year. `None` for an
    interval across years, an empty cell and any other spelling, so such a row reads the `total`
    that holds in every year. An integer cell (a year already written as a number) is its own
    year."""
    if isinstance(cell, int) and not isinstance(cell, bool):
        return cell
    return _text_year(cell) if isinstance(cell, str) else None


@functools.cache
def _text_year(text: str) -> int | None:
    if (match := _PERIOD.fullmatch(text)) is None:
        return None
    start, end = match.groups()
    return int(start) if end is None or end == start else None


def _in_window(years: tuple[int, int] | None, year: int | None) -> bool:
    if years is None:
        return True
    first, last = years
    return year is not None and first <= year <= last


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
