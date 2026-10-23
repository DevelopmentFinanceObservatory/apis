"""Batched graph lookups for materialisation's country-code, label and place-type enrichment
fields, the party fields the code index gives, and `read_party_kinds`, Search's reader of a
party's kind. Every reader takes the kind from the code index alone (`party_kind_of`). A frame
row's party cells (`party_cell`) are its party's kind, its class, and whether it is a whole
total for that row's dataset and year (`is_whole_total`)."""

from __future__ import annotations

import functools
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping

from code_index.types import PartyKind, PartyRow
from dfo_platform.roles import role_key_axis
from graph_release.context import served_origin_key
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
"""The field a lookup derives with no graph read: the party's kind (`party`, `total` or
`residual`) the code index declares (`party_kind_of`)."""

PARTY_CLASS_FIELD = "party_class"
"""The party's class (`official`, `private` or `residual`, `PartyRow.party_class`), for a
total the one class its whole markers cover, empty otherwise."""

PARTY_WHOLE_FIELD = "party_whole"
"""`true` on a row whose party is a whole total for the row's dataset and year
(`is_whole_total`), `false` on a row where it is not, and empty where the row names no one
dataset and year that decides it (`whole_cell`)."""

PARTY_FIELDS: tuple[str, ...] = (PARTY_KIND_FIELD, PARTY_CLASS_FIELD, PARTY_WHOLE_FIELD)
"""The fields the code index's party rows fill with no graph read (`party_cell`). Each
qualifies a party role only, and a frame carries one only when a token asks for it."""

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
    """Resolves refs to available `iso3`, `iso2`, `label`, `type` and `party_kind` fields. The
    graph fields are cached per ref in an LRU capped at `_CACHE_CEILING`, and lookups promote
    refs to most-recently used. An entry also keys on the bound release (`served_origin_key`),
    which answers a DFO variable's `name` and `typeOf`.

    `parties` returns the code index's party rows (`CodeIndex.parties`), which give a ref its
    `party_kind` (`party_kind_of`). A ref they do not list has no kind, whatever its graph
    types. It is read on every lookup and the kind is computed from it each time, since the
    index can change between requests.

    The lock guards cache access only. Graph calls run unlocked, so concurrent misses may
    duplicate requests. Results use each call's own hits and fetches, even if another call
    evicts them. A lookup returns every requested ref, even when the request exceeds the cap."""

    def __init__(
        self,
        graph: GraphReader,
        *,
        parties: Callable[[], Mapping[str, PartyRow]],
    ) -> None:
        self._graph = graph
        self._parties = parties
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], Mapping[str, str]] = OrderedDict()

    @property
    def parties(self) -> Mapping[str, PartyRow]:
        """The party rows the `parties` callable returns on this read."""
        return self._parties()

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
        parties = self._parties()
        hits, missing = self._hits_and_missing(requested)
        if missing:
            fetched = self._graph.node_arcs_many(
                missing,
                properties=_PROPERTIES,
                individual_probe_budget=_INDIVIDUAL_PROBE_BUDGET,
                deadline=deadline,
            )
            resolved = {
                ref: {} if (node := fetched.get(ref)) is None else _node_fields(node)
                for ref in missing
            }
            self._store(resolved)
            hits.update(resolved)
        return {ref: _with_party_kind(ref, hits[ref], parties=parties) for ref in requested}

    def party_kind(self, ref: str) -> PartyKind | None:
        """The kind `parties` declares for `ref` (`party_kind_of`), read with no graph call, so
        it holds when a `lookup` fails. `lookup` gives `ref` the same kind."""
        return party_kind_of(ref, parties=self._parties())

    def party_cells(self, ref: str) -> dict[str, str]:
        """The party cells `parties` gives `ref` for no dataset and year (`party_cell` of each
        of `PARTY_FIELDS`), read with no graph call, so they hold when a `lookup` fails. Empty
        for a ref the rows do not list. With no dataset, `party_whole` is empty for a party that
        carries whole markers (`whole_cell`)."""
        cells = {field: party_cell(field, ref, parties=self._parties()) for field in PARTY_FIELDS}
        return {field: cell for field, cell in cells.items() if cell is not None}

    def _hits_and_missing(
        self, requested: list[str]
    ) -> tuple[dict[str, Mapping[str, str]], list[str]]:
        release = served_origin_key()
        with self._lock:
            hits: dict[str, Mapping[str, str]] = {}
            missing: list[str] = []
            for ref in requested:
                if (key := (ref, release)) in self._cache:
                    self._cache.move_to_end(key)
                    hits[ref] = self._cache[key]
                else:
                    missing.append(ref)
            return hits, missing

    def _store(self, resolved: Mapping[str, Mapping[str, str]]) -> None:
        release = served_origin_key()
        with self._lock:
            for ref, fields in resolved.items():
                self._cache[ref, release] = fields
                self._cache.move_to_end((ref, release))
            while len(self._cache) > _CACHE_CEILING:
                self._cache.popitem(last=False)


def _with_party_kind(
    ref: str, fields: Mapping[str, str], *, parties: Mapping[str, PartyRow]
) -> Mapping[str, str]:
    """`fields` with `party_kind` set to `party_kind_of` the ref, unchanged when it gives none.
    A copy, so the cached fields never carry a kind."""
    if (kind := party_kind_of(ref, parties=parties)) is None:
        return fields
    return {**fields, PARTY_KIND_FIELD: kind}


def _node_fields(node: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Each graph field's first value, except `type`: every `typeOf` value of the place, sorted
    and comma-joined (`AdministrativeArea1,Country,EurostatNUTS2`), since a place can hold
    several levels at once."""
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


def is_whole_total(party: PartyRow, *, year: int | None, provenance: str | None) -> bool:
    """Whether `party` is a whole total on a row of `provenance` in `year`: a `total` with a
    marker for `provenance` (`PartyRow.marker`) that applies in `year`. A marker applies in
    every year when its `years` is `None`, and otherwise only in a `year` inside that window. A
    row with no `provenance`, or no `year` where a window is set, reads as no whole total, the
    reading that holds in every dataset and year."""
    marker = None if provenance is None else party.marker(provenance)
    return party.kind == "total" and marker is not None and _in_window(marker.years, year)


def whole_cell(party: PartyRow, *, year: int | None, provenance: str | None) -> str:
    """The `party_whole` cell of `party` on a row of `provenance` in `year`: `true` when it is a
    whole total there (`is_whole_total`) and `false` when it is not. Empty when the row lacks
    what decides it: a party with whole markers on a row with no `provenance`, or on one with no
    `year` where that dataset's marker sets a window. A pivot's year-free cell, a period across
    years and a table without the column all read so. A party with no marker reads `false`,
    which holds in every dataset and year."""
    if party.whole and provenance is None:
        return ""
    marker = None if provenance is None else party.marker(provenance)
    if marker is not None and marker.years is not None and year is None:
        return ""
    return "true" if is_whole_total(party, year=year, provenance=provenance) else "false"


def party_cell(
    field: str,
    ref: str,
    *,
    parties: Mapping[str, PartyRow],
    year: int | None = None,
    provenance: str | None = None,
) -> str | None:
    """The `<role>_<field>` cell of the party `ref`, for `field` one of `PARTY_FIELDS`, on a
    row of `provenance` in `year`: its kind (`party_kind_of`), its class (empty when it has
    none), or whether it is a whole total there (`whole_cell`). `None` for a ref the rows do not
    list."""
    party = parties.get(ref)
    if party is None:
        return None
    if field == PARTY_KIND_FIELD:
        return party.kind
    if field == PARTY_CLASS_FIELD:
        return party.party_class or ""
    if field == PARTY_WHOLE_FIELD:
        return whole_cell(party, year=year, provenance=provenance)
    raise ValueError(f"{field!r} is not a party field; expected one of {PARTY_FIELDS!r}")


def row_party_cell(
    field: str, ref: object, *, parties: Mapping[str, PartyRow], period: object, provenance: object
) -> str | None:
    """The `<role>_<field>` cell of a frame row whose party cell is `ref`, `time_period` cell
    `period` and `provenance` cell `provenance` (`None` for a table without that column): the
    `party_cell` for the year of `period` (`period_year`) and that dataset. `None` for a `ref`
    that is not text. A `provenance` that is not text reads as no dataset."""
    if not isinstance(ref, str):
        return None
    return party_cell(
        field,
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
    """The one year a row's `time_period` cell lies in, for `party_cell`: the year of a point
    (`2021`, `2021-06`, `2021-06-30`) or of an interval inside one year. `None` for an interval
    across years, an empty cell and any other spelling, so such a row reads as no whole total,
    the reading that holds in every year. An integer cell (a year already written as a number)
    is its own year."""
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
