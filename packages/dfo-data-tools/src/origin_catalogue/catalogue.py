"""The origin catalogue in memory, and the admission rules that read it.

`CatalogueTables` holds one build's tables. `CatalogueSnapshot` pairs them
with the build's manifest and the UN origin setting, and answers every
admission question from them with no graph read. Request code reads the
`Catalogue` protocol, which `CatalogueSnapshot` and test doubles implement.

Admission works on provenance ids:

- a DFO provenance is always admitted.
- a base provenance is admitted when the provenance table lists it. The job
  lists each registered source's provenances minus DFO's and the entry's
  exclusions.
- a UN agency provenance is admitted when the table lists it and the UN
  origin is enabled.

Where a DFO facet is present for a variable and place, every other facet
there is withheld, so DFO's data wins over an upstream copy.

A search candidate is refused only when the catalogue knows its data at the
scan places sits under provenances it does not admit (`withheld_variables`).
A base variable the catalogue does not list passes, since every facet read
still admits facet by facet, and `data_commons` names base for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final, Protocol
from urllib.parse import urlsplit

from origin_catalogue.origins import BASE, DFO, UN_ID_PREFIX, UN_SYSTEM, DataCommons
from origin_catalogue.registry import UN_CROSS_AGENCY_TOPIC_FAMILIES
from origin_catalogue.schema import CatalogueManifest, ScanLevel
from origin_catalogue.topic_walk import TOPIC_EXPAND_MAX_MEMBERS

STALE_AFTER: Final[timedelta] = timedelta(days=14)
"""Answers carry a staleness warning once the loaded build is older than this."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A provenance's source: the `dc/s/*` node its `source` arc names on
    base Data Commons, the agency provenance itself on the UN deployment."""

    ref: str
    label: str


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """The `dc/d/*` Dataset a provenance's `isPartOf` arc names, labelled
    with the Dataset's `name`, else its dcid."""

    ref: str
    label: str


@dataclass(frozen=True, slots=True)
class ProvenanceRow:
    """One admitted provenance. `segment` is the UN agency's id segment and
    is set on UN rows only. `dataset` is the provenance's parent Dataset,
    `None` on UN rows and on rows with no `isPartOf` arc."""

    id: str
    data_commons: DataCommons
    source: SourceRef
    name: str
    url: str
    segment: str | None = None
    dataset: DatasetRef | None = None


def entry_ref(row: ProvenanceRow) -> str:
    """The source-index entry `row` belongs to: its parent Dataset, else the
    provenance itself. `CatalogueTables.dataset_facts` is keyed by it."""
    return row.dataset.ref if row.dataset is not None else row.id


@dataclass(frozen=True, slots=True)
class CardFacts:
    """The facts a source-index entry and its documentation page share, each
    `None` when the graph gives none (`origin_catalogue.card_facts`).
    `license_type` is the licence's label, the `name` of its `licenseType`
    node. `update_cadence` is the raw ISO 8601 `sourceReleaseFrequency`
    token, which readers map to a label."""

    description: str | None = None
    license_type: str | None = None
    update_cadence: str | None = None
    years_covered: str | None = None
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class BaseVariable:
    """A variable an admitted base provenance holds data for. Bit `i` of
    `places` is set when an admitted provenance has data at scan place `i`.
    Which of `provenances` has data at which place is not kept per variable;
    `CatalogueTables.provenance_places` keeps it per provenance."""

    provenances: tuple[str, ...]
    places: int


@dataclass(frozen=True, slots=True)
class SeriesAdmission:
    """The provenances admitted and withheld for one variable at one place,
    each in first-occurrence order."""

    admitted: tuple[str, ...]
    withheld: tuple[str, ...]


class CatalogueTableError(ValueError):
    """The tables break an invariant the admission rules rely on."""


@dataclass(frozen=True, slots=True)
class BaseTables:
    """The tables the catalogue job enumerates from base Data Commons: the
    base variable table, each admitted base provenance's places, the
    withheld variable set, the topic table, and the scan places and levels
    the bitsets index. Fields mean what `CatalogueTables`' fields of the
    same names mean."""

    base_variables: Mapping[str, BaseVariable]
    provenance_places: Mapping[str, int]
    withheld_variables: frozenset[str]
    topics: Mapping[str, tuple[str, ...]]
    scan_places: tuple[str, ...]
    scan_levels: Mapping[str, tuple[ScanLevel, ...]]


@dataclass(frozen=True, slots=True)
class CatalogueTables:
    """One build's tables.

    `provenances` maps every admitted provenance id to its row.
    `provenance_places` maps each base provenance holding a base variable to
    a bitset over `scan_places`: bit `i` is set when one of its variables
    has a facet under it at scan place `i`. `withheld_variables` holds each variable our mixer has a facet for at a
    scan place under a provenance that is neither an admitted base
    provenance nor DFO's, and none under an admitted base provenance.
    `topics` maps each Topic on our mixer to
    every leaf variable search's topic walk reaches from it
    (`topic_walk.walk_topics`, uncapped). `scan_places` orders the places the
    variable bitsets index. `scan_levels` records, per source ref, the
    levels the job scanned beyond every country, Earth and the regional
    aggregates. `dataset_facts` maps the `entry_ref` of every DFO and base
    row to its entry's `CardFacts`; UN entries carry none.

    `admitted_topics` is derived: each topic with at least one and at most
    `TOPIC_EXPAND_MAX_MEMBERS` admitted leaves, the cap search applies after
    the same admission, mapped to base when an admitted leaf is not a DFO-only
    variable and to DFO otherwise. A leaf is admitted when `dfo_variables`
    holds it or `withheld_variables` does not. The topic table depends on no
    DFO data, so a build after an ingestion carries it forward and admission
    follows the new DFO variable set.

    Construction refuses a base variable naming a provenance that is not a
    base row, a bitset reaching past `scan_places`, a base variable with a
    place none of its provenances' places holds, a provenance place no base
    variable naming it holds, a variable both in
    `base_variables` and `withheld_variables`, a UN row with no segment
    or a non-UN row with one, a UN topic or a topic with no leaves, and
    `dataset_facts` keys other than the DFO and base rows' entry refs.
    """

    provenances: Mapping[str, ProvenanceRow]
    dfo_variables: frozenset[str]
    base_variables: Mapping[str, BaseVariable]
    provenance_places: Mapping[str, int]
    withheld_variables: frozenset[str]
    topics: Mapping[str, tuple[str, ...]]
    scan_places: tuple[str, ...]
    scan_levels: Mapping[str, tuple[ScanLevel, ...]]
    dataset_facts: Mapping[str, CardFacts]
    un_blocked_places: frozenset[str] = frozenset()
    """Places the UN front door refused to have named in an observation GET
    when the job probed it: UN reads leave them out and answer them empty."""
    admitted_topics: Mapping[str, DataCommons] = field(init=False)
    dfo_provenances: frozenset[str] = field(init=False)
    un_segments: frozenset[str] = field(init=False)
    mixer_domains: tuple[str, ...] = field(init=False)
    """The URL hostnames of the admitted DFO and base provenances, sorted."""
    scan_place_set: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        for provenance_id, row in self.provenances.items():
            if row.id != provenance_id:
                raise CatalogueTableError(
                    f"provenance row {row.id!r} is keyed as {provenance_id!r}"
                )
            if (row.data_commons == UN_SYSTEM) != (row.segment is not None):
                raise CatalogueTableError(
                    f"provenance {provenance_id!r} ({row.data_commons.id}) has segment "
                    f"{row.segment!r}; UN rows carry a segment and no others do"
                )
        entry_refs = {
            entry_ref(row) for row in self.provenances.values() if row.data_commons != UN_SYSTEM
        }
        if entry_refs != self.dataset_facts.keys():
            missing = sorted(entry_refs - self.dataset_facts.keys())
            extra = sorted(self.dataset_facts.keys() - entry_refs)
            raise CatalogueTableError(
                f"dataset facts miss {len(missing)} entry ref(s), e.g. {missing[:3]!r}, and "
                f"hold {len(extra)} unknown one(s), e.g. {extra[:3]!r}"
            )
        base_ids = {pid for pid, row in self.provenances.items() if row.data_commons == BASE}
        limit = 1 << len(self.scan_places)
        held_by: dict[str, int] = {}
        for variable, entry in self.base_variables.items():
            unknown = [pid for pid in entry.provenances if pid not in base_ids]
            if unknown or not entry.provenances:
                raise CatalogueTableError(
                    f"base variable {variable!r} names {list(entry.provenances)!r}; each must be "
                    "a base provenance row and there must be at least one"
                )
            if not 0 <= entry.places < limit:
                raise CatalogueTableError(
                    f"base variable {variable!r} has place bits past the "
                    f"{len(self.scan_places)} scan places"
                )
            covered = 0
            for pid in entry.provenances:
                covered |= self.provenance_places.get(pid, 0)
                held_by[pid] = held_by.get(pid, 0) | entry.places
            if entry.places & ~covered:
                raise CatalogueTableError(
                    f"base variable {variable!r} has places none of its provenances "
                    f"{list(entry.provenances)!r} has"
                )
        for pid, places in self.provenance_places.items():
            if places & ~held_by.get(pid, 0) or not places:
                raise CatalogueTableError(
                    f"provenance {pid!r} has places no base variable naming it has, or none"
                )
        both = sorted(self.withheld_variables.intersection(self.base_variables))
        if both:
            raise CatalogueTableError(
                f"{len(both)} variable(s) are both admitted and withheld, e.g. {both[:3]!r}"
            )
        for topic, leaves in self.topics.items():
            if topic.startswith(UN_ID_PREFIX) or not leaves:
                raise CatalogueTableError(
                    f"topic {topic!r} has leaves {list(leaves)!r}; the topic table holds "
                    "our mixer's topics, each with at least one leaf"
                )
        object.__setattr__(self, "admitted_topics", self._admit_topics())
        object.__setattr__(
            self,
            "dfo_provenances",
            frozenset(pid for pid, row in self.provenances.items() if row.data_commons == DFO),
        )
        object.__setattr__(
            self,
            "un_segments",
            frozenset(row.segment for row in self.provenances.values() if row.segment is not None),
        )
        hostnames = (
            urlsplit(row.url).hostname
            for row in self.provenances.values()
            if row.data_commons != UN_SYSTEM
        )
        object.__setattr__(
            self, "mixer_domains", tuple(sorted({host for host in hostnames if host}))
        )
        object.__setattr__(self, "scan_place_set", frozenset(self.scan_places))

    def _admit_topics(self) -> dict[str, DataCommons]:
        admitted: dict[str, DataCommons] = {}
        for topic, leaves in self.topics.items():
            base = dfo_only = 0
            for leaf in leaves:
                if leaf in self.base_variables or not (
                    leaf in self.dfo_variables or leaf in self.withheld_variables
                ):
                    base += 1
                elif leaf in self.dfo_variables:
                    dfo_only += 1
            if 0 < base + dfo_only <= TOPIC_EXPAND_MAX_MEMBERS:
                admitted[topic] = BASE if base else DFO
        return admitted


class Catalogue(Protocol):
    """What request code reads from one catalogue version."""

    @property
    def build_id(self) -> str: ...

    @property
    def built_at(self) -> datetime: ...

    @property
    def un_system_enabled(self) -> bool:
        """Whether the UN origin is served. With it off, UN dcids are read nowhere."""
        ...

    @property
    def mixer_domains(self) -> tuple[str, ...] | None:
        """The hostnames sent as `filter.domains` on facet reads to our mixer,
        or `None` to send no filter."""
        ...

    @property
    def un_blocked_places(self) -> frozenset[str]:
        """Places the UN front door refuses to have named in an observation
        GET, as the catalogue job found them."""
        ...

    def is_stale(self, *, now: datetime | None = None) -> bool:
        """Whether the build is older than `STALE_AFTER`."""
        ...

    def is_scan_place(self, place: str) -> bool:
        """Whether the variable table's place bitsets cover `place`."""
        ...

    def provenance(self, provenance_id: str) -> ProvenanceRow | None:
        """The row for an admitted provenance id: origin, source, name and URL."""
        ...

    def provenance_rows(self) -> tuple[ProvenanceRow, ...]:
        """Every admitted provenance's row, in table order."""
        ...

    def dataset_facts(self, ref: str) -> CardFacts | None:
        """The facts of the source-index entry `ref` (an `entry_ref`), or
        `None` for a UN entry or one the build does not list."""
        ...

    def is_admitted(self, provenance_id: str) -> bool: ...

    def admit_series(self, provenance_ids: Iterable[str]) -> SeriesAdmission:
        """Admission for the facets of one variable at one place, with DFO precedence."""
        ...

    def admit_candidate(self, dcid: str) -> DataCommons | None:
        """The origin that admits a search candidate, or `None` when none does.
        A base candidate is refused only when the catalogue knows its data at
        the scan places to sit under provenances it does not admit."""
        ...

    def data_commons(self, variable: str) -> tuple[DataCommons, ...]:
        """Every origin holding admitted data for `variable`, in wire order. It
        names base for every variable `admit_candidate` admits as base."""
        ...

    def base_places(self, variable: str) -> tuple[str, ...] | None:
        """The scan places where admitted base provenances hold `variable`,
        or `None` when the variable table does not list it."""
        ...

    def variable_provenances(self, variable: str) -> tuple[ProvenanceRow, ...]:
        """The admitted provenance rows known to hold `variable`, in
        first-occurrence order. Empty for DFO variables and anything the
        catalogue does not list."""
        ...


@dataclass(frozen=True, slots=True)
class CatalogueSnapshot:
    """One loaded catalogue version. `un_system_enabled` gates UN admission;
    with it off, no UN provenance or candidate is admitted."""

    manifest: CatalogueManifest
    tables: CatalogueTables
    un_system_enabled: bool = True

    @property
    def build_id(self) -> str:
        return self.manifest.build_id

    @property
    def built_at(self) -> datetime:
        return self.manifest.built_at

    @property
    def mixer_domains(self) -> tuple[str, ...]:
        return self.tables.mixer_domains

    @property
    def un_blocked_places(self) -> frozenset[str]:
        return self.tables.un_blocked_places

    def is_stale(self, *, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) - self.manifest.built_at > STALE_AFTER

    def is_scan_place(self, place: str) -> bool:
        return place in self.tables.scan_place_set

    def provenance(self, provenance_id: str) -> ProvenanceRow | None:
        return self.tables.provenances.get(provenance_id)

    def provenance_rows(self) -> tuple[ProvenanceRow, ...]:
        """UN rows are left out when the UN origin is off, as they are not admitted."""
        return tuple(row for row in self.tables.provenances.values() if self.is_admitted(row.id))

    def dataset_facts(self, ref: str) -> CardFacts | None:
        return self.tables.dataset_facts.get(ref)

    def is_admitted(self, provenance_id: str) -> bool:
        row = self.tables.provenances.get(provenance_id)
        if row is None:
            return False
        return row.data_commons != UN_SYSTEM or self.un_system_enabled

    def admit_series(self, provenance_ids: Iterable[str]) -> SeriesAdmission:
        present = tuple(dict.fromkeys(provenance_ids))
        admitted = [pid for pid in present if self.is_admitted(pid)]
        dfo = self.tables.dfo_provenances
        if any(pid in dfo for pid in admitted):
            admitted = [pid for pid in admitted if pid in dfo]
        kept = set(admitted)
        return SeriesAdmission(
            admitted=tuple(admitted), withheld=tuple(pid for pid in present if pid not in kept)
        )

    def admit_candidate(self, dcid: str) -> DataCommons | None:
        if dcid.startswith(UN_ID_PREFIX):
            return UN_SYSTEM if self._is_admitted_un_candidate(dcid) else None
        if dcid in self.tables.dfo_variables:
            return DFO
        if self._is_admitted_base(dcid):
            return BASE
        return self.tables.admitted_topics.get(dcid)

    def data_commons(self, variable: str) -> tuple[DataCommons, ...]:
        if variable.startswith(UN_ID_PREFIX):
            return (UN_SYSTEM,) if self._is_admitted_un_candidate(variable) else ()
        held = (
            (DFO, variable in self.tables.dfo_variables),
            (BASE, self._is_admitted_base(variable)),
        )
        return tuple(origin for origin, holds in held if holds)

    def base_places(self, variable: str) -> tuple[str, ...] | None:
        entry = self.tables.base_variables.get(variable)
        if entry is None:
            return None
        places = self.tables.scan_places
        bits = entry.places
        return tuple(places[i] for i in range(bits.bit_length()) if bits >> i & 1)

    def variable_provenances(self, variable: str) -> tuple[ProvenanceRow, ...]:
        """Base variables name their rows in the variable table; an admitted
        `undata/<segment>/...` variable maps to the UN rows of its segment."""
        entry = self.tables.base_variables.get(variable)
        if entry is not None:
            return tuple(self.tables.provenances[pid] for pid in entry.provenances)
        if not variable.startswith(UN_ID_PREFIX) or not self._is_admitted_un_candidate(variable):
            return ()
        segment = variable[len(UN_ID_PREFIX) :].split("/")[0]
        return tuple(row for row in self.tables.provenances.values() if row.segment == segment)

    def _is_admitted_base(self, dcid: str) -> bool:
        """A variable the base table lists, or one no table places: not DFO's,
        not withheld and not a topic."""
        tables = self.tables
        return dcid in tables.base_variables or not (
            dcid in tables.dfo_variables
            or dcid in tables.withheld_variables
            or dcid in tables.topics
        )

    def _is_admitted_un_candidate(self, dcid: str) -> bool:
        """`undata/<segment>/<id>` or `undata/topic/<segment>/<id>` with a
        registered segment, or a topic in a cross-agency family."""
        if not self.un_system_enabled:
            return False
        parts = dcid[len(UN_ID_PREFIX) :].split("/")
        if parts[0] == "topic":
            if len(parts) < 3 or not parts[2]:
                return False
            return parts[1] in self.tables.un_segments or parts[1] in UN_CROSS_AGENCY_TOPIC_FAMILIES
        return len(parts) >= 2 and bool(parts[1]) and parts[0] in self.tables.un_segments


__all__ = [
    "STALE_AFTER",
    "BaseTables",
    "BaseVariable",
    "CardFacts",
    "Catalogue",
    "CatalogueSnapshot",
    "CatalogueTableError",
    "CatalogueTables",
    "DatasetRef",
    "ProvenanceRow",
    "SeriesAdmission",
    "SourceRef",
    "entry_ref",
]
