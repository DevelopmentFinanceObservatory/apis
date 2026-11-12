"""Batched graph lookups for materialisation's country-code, label and place-type enrichment
fields, the party fields the code index gives, and `read_party_kinds`, Search's reader of a
party's kind. Every reader takes the kind from the code index alone (`party_kind_of`), on the
axis of the role it reads and for the datasets it serves, since a party can take a different
kind in each (`PartyRow.dataset_kinds`). A frame row's party cells (`party_cell`) are its
party's kind and class on its column's axis in that row's dataset, and whether it is a whole
total for that row's dataset and year (`is_whole_total`)."""

from __future__ import annotations

import functools
import re
import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping

from code_index.types import AnyAxis, PartyKind, PartyRow
from dc_graph import label_or_self_name
from dfo_platform.roles import RoleAxis, role_key_axis
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
"""The party's kind (`party`, `total` or `residual`) the code index declares on the axis of the
column that holds it, for the row's dataset (`party_cell`), read with no graph read. A pivot
reads it over the datasets the frame serves the party from (`pivot_party_cell`). Where the row
or the pivot reads datasets in which the party takes different kinds on that axis the kind is
unavailable, as for a party the index does not list."""

PARTY_CLASS_FIELD = "party_class"
"""The party's class (`official`, `private` or `residual`, `PartyRow.class_of`) in the row's
dataset, for a total the one class its whole markers cover, empty otherwise. Unavailable where
the kind is (`PARTY_KIND_FIELD`)."""

PARTY_WHOLE_FIELD = "party_whole"
"""`true` on a row whose party is a whole total for the row's dataset and year
(`is_whole_total`), `false` on a row where it is not, and empty where the row names no one
dataset and year that decides it (`whole_cell`)."""

PARTY_FIELDS: tuple[str, ...] = (PARTY_KIND_FIELD, PARTY_CLASS_FIELD, PARTY_WHOLE_FIELD)
"""The fields the code index's party rows fill with no graph read (`party_cell`). Each
qualifies a party role only, and a frame carries one only when a token asks for it."""

ENRICHMENT_FIELDS: tuple[str, ...] = (*ENRICHMENT_PROPERTIES, PARTY_KIND_FIELD)
"""Every field an output profile counts coverage of: the graph fields `EnrichmentLookup.lookup`
returns, and `party_kind`, which the code index fills per column (`party_cell`)."""

_NAME_PROPERTY = "name"
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
    """Resolves refs to their available graph fields: `iso3`, `iso2`, `label` and `type`. The
    fields are cached per ref in an LRU capped at `_CACHE_CEILING`, and lookups promote refs to
    most-recently used. An entry also keys on the bound release (`served_origin_key`), which
    answers a DFO variable's `name` and `typeOf`.

    The lock guards cache access only. Graph calls run unlocked, so concurrent misses may
    duplicate requests. Results use each call's own hits and fetches, even if another call
    evicts them. A lookup returns every requested ref, even when the request exceeds the cap."""

    def __init__(self, graph: GraphReader) -> None:
        self._graph = graph
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], Mapping[str, str]] = OrderedDict()

    def lookup(
        self, refs: Iterable[str], *, deadline: float | None = None
    ) -> Mapping[str, Mapping[str, str]]:
        """Returns unique refs in input order, mapping available graph fields to values. Absent
        refs map to `{}`. The party fields depend on the axis of the column that holds a ref, so
        readers take them from `party_cell` per column. Cached refs avoid graph calls. New refs
        are fetched in one unlocked batch, bounded by the absolute monotonic `deadline` when
        given."""
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
                ref: {} if (node := fetched.get(ref)) is None else _node_fields(ref, node)
                for ref in missing
            }
            self._store(resolved)
            hits.update(resolved)
        return {ref: hits[ref] for ref in requested}

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


def _node_fields(dcid: str, node: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Each graph field's first value, except `label` and `type`. `label` is the node's
    `label_or_self_name`. `type` is every `typeOf` value of the place, sorted and comma-joined
    (`AdministrativeArea1,Country,EurostatNUTS2`), since a place can hold several levels at
    once."""
    fields: dict[str, str] = {}
    for property_, field in _FIELDS_BY_PROPERTY.items():
        values = node.get(property_, ())
        if property_ == _NAME_PROPERTY:
            value = label_or_self_name(values, dcid)
        elif property_ == _TYPE_PROPERTY:
            value = ",".join(sorted(values))
        else:
            value = values[0] if values else None
        if value:
            fields[field] = value
    return fields


def party_kind_of(
    ref: str,
    *,
    axis: RoleAxis | AnyAxis,
    parties: Mapping[str, PartyRow],
    datasets: Collection[str],
) -> PartyKind | None:
    """The kind the code index's `parties` row declares for the party `ref` on `axis`, the
    axis of the role the reader reads (`ANY_AXIS` for a reader of no role), in `datasets`, the
    provenances the reader serves (`PartyRow.kind_over`), which holds whatever the graph says
    (a group of organisations is typed `Organization` and declared `total`). Empty `datasets`
    reads every dataset that publishes the party on `axis`. `None` for a ref the rows do not
    list, for a party no dataset publishes on `axis`, and for a party that takes different
    kinds in those datasets.

    Search, the options document and every download read this one source, since the index
    lists every party a declared dataset publishes. A coverage build carrying a party the index
    was not exported for leaves that party with no kind. That is index drift, which the
    code-index drift gate reports and a re-export of the index repairs."""
    party = parties.get(ref)
    return None if party is None else party.kind_over(axis, datasets)


def party_count(
    dcids: Collection[str],
    *,
    axis: RoleAxis | AnyAxis,
    parties: Mapping[str, PartyRow],
    datasets: Collection[str],
) -> int:
    """How many of `dcids`, members of a role on `axis`, the default download keeps as
    parties: those the code index's `parties` declares no `total` on `axis` in `datasets`
    (`party_kind_of`), the test the Datasets cut of party totals applies
    (`datasets_api.party_totals`). When every member is a total the cut keeps them all, and so
    does the count."""
    kept = sum(
        party_kind_of(dcid, axis=axis, parties=parties, datasets=datasets) != "total"
        for dcid in dcids
    )
    return kept or len(dcids)


def is_whole_total(party: PartyRow, *, year: int | None, provenance: str | None) -> bool:
    """Whether `party` is a whole total on a row of `provenance` in `year`: a `total` with a
    marker for `provenance` (`PartyRow.marker`) that applies in `year`. A marker applies in
    every year when its `years` is `None`, and otherwise only in a `year` inside that window. A
    row with no `provenance`, or no `year` where a window is set, reads as no whole total, the
    reading that holds in every dataset and year."""
    if provenance is None or (marker := party.marker(provenance)) is None:
        return False
    return party.kind_in(provenance) == "total" and _in_window(marker.years, year)


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
    axis: RoleAxis | AnyAxis,
    parties: Mapping[str, PartyRow],
    year: int | None = None,
    provenance: str | None = None,
) -> str | None:
    """The `<role>_<field>` cell of the party `ref`, for `field` one of `PARTY_FIELDS`, on a
    row of `provenance` in `year`, for a role on `axis` (`ANY_AXIS` for a reader of no role):
    its kind there (`party_kind_of`), its class there (`PartyRow.class_of`, empty when it has
    none), or whether it is a whole total there (`whole_cell`). With no `provenance` the kind
    and class are those of every dataset that publishes it on `axis`. `None` for a ref the rows
    do not list, and for the kind and class of a party that takes no one kind for the row
    (`PartyRow.kind_over`), so a reader counts the cell unavailable."""
    return _party_cell(
        field,
        parties.get(ref),
        axis=axis,
        datasets=() if provenance is None else (provenance,),
        year=year,
        provenance=provenance,
    )


def pivot_party_cell(
    field: str,
    ref: object,
    *,
    role: str,
    parties: Mapping[str, PartyRow],
    datasets: Collection[str],
) -> str | None:
    """The `<role>_<field>` cell a pivot writes for the party `ref`, one cell for every row
    holding it in the `role` column: its kind and class on the role's axis over `datasets`,
    the datasets the frame serves it from (`PartyRow.kind_over`, the one kind they agree on),
    and its `party_whole` read for no dataset and year (`whole_cell`). `None` for a `ref`
    that is not text, a role on no flow axis, and wherever `party_cell` gives none."""
    if not isinstance(ref, str) or (axis := role_key_axis(role)) is None:
        return None
    return _party_cell(
        field, parties.get(ref), axis=axis, datasets=datasets, year=None, provenance=None
    )


def _party_cell(
    field: str,
    party: PartyRow | None,
    *,
    axis: RoleAxis | AnyAxis,
    datasets: Collection[str],
    year: int | None,
    provenance: str | None,
) -> str | None:
    """`party`'s `field` cell, its kind and class read over `datasets` and its `party_whole`
    for `provenance` in `year`."""
    if party is None:
        return None
    if field in (PARTY_KIND_FIELD, PARTY_CLASS_FIELD):
        kind = party.kind_over(axis, datasets)
        if kind is None:
            return None
        if field == PARTY_KIND_FIELD:
            return kind
        return party.class_of(kind) or ""
    if field == PARTY_WHOLE_FIELD:
        return whole_cell(party, year=year, provenance=provenance)
    raise ValueError(f"{field!r} is not a party field; expected one of {PARTY_FIELDS!r}")


def row_party_cell(
    field: str,
    ref: object,
    *,
    role: str,
    parties: Mapping[str, PartyRow],
    period: object,
    provenance: object,
) -> str | None:
    """The `<role>_<field>` cell of a frame row whose `role` cell is `ref`, `time_period` cell
    `period` and `provenance` cell `provenance` (`None` for a table without that column): the
    `party_cell` on the role's axis (`role_key_axis`) for the year of `period` (`period_year`)
    and that dataset. `None` for a `ref` that is not text and for a role on no flow axis. A
    `provenance` that is not text reads as no dataset."""
    if not isinstance(ref, str) or (axis := role_key_axis(role)) is None:
        return None
    return party_cell(
        field,
        ref,
        axis=axis,
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
    entities: Iterable[tuple[str, str]],
    *,
    parties: Mapping[str, PartyRow],
    datasets: Collection[str],
) -> dict[tuple[str, str], PartyKind]:
    """The kind of each `(role, ref)` entity whose role is a party role (`is_party_role`) in
    `datasets`, the provenances the reader serves, keyed by the pair: `party_kind_of` on the
    role's axis over the code index's `parties` rows. An entity the rows do not list, one that
    takes no one kind on that axis in `datasets`, or one in a subject role, is absent."""
    kinds: dict[tuple[str, str], PartyKind] = {}
    for role, ref in dict.fromkeys(entities):
        if (axis := role_key_axis(role)) is None:
            continue
        kind = party_kind_of(ref, axis=axis, parties=parties, datasets=datasets)
        if kind is not None:
            kinds[role, ref] = kind
    return kinds
