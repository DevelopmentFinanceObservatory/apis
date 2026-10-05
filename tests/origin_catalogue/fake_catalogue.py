"""Origin catalogue doubles for tests.

`FakeCatalogue` admits every provenance and candidate. The suite-wide
autouse fixture in `tests/conftest.py` binds one, so code that reads the
request's catalogue behaves as it did before admission existed, and the
process-wide catalogue service serves one through `fake_catalogue_service`,
so every surface's request binding does too. A test of admission binds a
restrictive `CatalogueSnapshot` built with `restrictive_catalogue` through
the `bind_catalogue` fixture.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from origin_catalogue import (
    BASE,
    DFO,
    UN_SYSTEM,
    BaseVariable,
    CardFacts,
    Catalogue,
    CatalogueManifest,
    CatalogueSnapshot,
    CatalogueTables,
    DataCommons,
    DatasetRef,
    ProvenanceRow,
    SeriesAdmission,
    SourceRef,
    entry_ref,
)
from origin_catalogue.origins import UN_ID_PREFIX
from origin_catalogue.service import CatalogueService

FAKE_BUILT_AT = datetime(2026, 9, 29, tzinfo=UTC)


def _origin_of(dcid: str, dfo: frozenset[str]) -> DataCommons:
    if dcid.startswith(UN_ID_PREFIX):
        return UN_SYSTEM
    return DFO if dcid in dfo else BASE


@dataclass(frozen=True)
class FakeCatalogue:
    """Admits everything. Origins come from the id: `undata/` is the UN,
    ids in `dfo_variables` are DFO, the rest base. Its provenance table
    lists only the rows in `provenances`, none by default, so wire members
    read from the table (labels, `source`, `data_commons`) stay empty and
    callers fall back as they do for a provenance the catalogue does not
    list. Admission stays permissive whatever `provenances` holds. `card_facts`
    answers `dataset_facts`, empty by default. Facet reads send no domain
    filter, and no place is a scan place, so presence is always read live."""

    build_id: str = "fake-catalogue"
    built_at: datetime = FAKE_BUILT_AT
    dfo_variables: frozenset[str] = frozenset()
    un_system_enabled: bool = True
    mixer_domains: tuple[str, ...] | None = None
    un_blocked_places: frozenset[str] = frozenset()
    provenances: Mapping[str, ProvenanceRow] = field(default_factory=dict, compare=False)
    card_facts: Mapping[str, CardFacts] = field(default_factory=dict, compare=False)

    def is_stale(self, *, now: datetime | None = None) -> bool:
        return False

    def is_scan_place(self, place: str) -> bool:
        return False

    def provenance(self, provenance_id: str) -> ProvenanceRow | None:
        return self.provenances.get(provenance_id)

    def provenance_rows(self) -> tuple[ProvenanceRow, ...]:
        return tuple(self.provenances.values())

    def dataset_facts(self, ref: str) -> CardFacts | None:
        return self.card_facts.get(ref)

    def is_admitted(self, provenance_id: str) -> bool:
        return True

    def admit_series(self, provenance_ids: Iterable[str]) -> SeriesAdmission:
        return SeriesAdmission(admitted=tuple(dict.fromkeys(provenance_ids)), withheld=())

    def admit_candidate(self, dcid: str) -> DataCommons:
        return _origin_of(dcid, self.dfo_variables)

    def data_commons(self, variable: str) -> tuple[DataCommons, ...]:
        return (_origin_of(variable, self.dfo_variables),)

    def base_places(self, variable: str) -> tuple[str, ...] | None:
        return None

    def variable_provenances(self, variable: str) -> tuple[ProvenanceRow, ...]:
        return ()


def fake_catalogue_service(catalogue: Catalogue | None = None) -> CatalogueService:
    """A catalogue service serving `catalogue`, a `FakeCatalogue` by default, with no
    artifact behind it and no recheck."""
    served = catalogue if catalogue is not None else FakeCatalogue()
    return CatalogueService(
        artifact_uri="fake://origin-catalogue", recheck_seconds=0, loader=lambda _uri: served
    )


def spread_provenance_places(base_variables: Mapping[str, BaseVariable]) -> dict[str, int]:
    """Each provenance's places as the union of the places of the variables
    naming it, for tables whose test does not say which provenance holds a
    variable where."""
    places: dict[str, int] = {}
    for entry in base_variables.values():
        for provenance in entry.provenances:
            places[provenance] = places.get(provenance, 0) | entry.places
    return {provenance: held for provenance, held in places.items() if held}


def restrictive_catalogue(
    *,
    provenances: Iterable[ProvenanceRow] = (),
    dfo_variables: Iterable[str] = (),
    base_variables: Mapping[str, tuple[Iterable[str], Iterable[str]]] | None = None,
    withheld_variables: Iterable[str] = (),
    topics: Mapping[str, Iterable[str]] | None = None,
    scan_places: Iterable[str] = (),
    un_blocked_places: Iterable[str] = (),
    un_system_enabled: bool = True,
    dataset_facts: Mapping[str, CardFacts] | None = None,
    build_id: str = "restrictive-catalogue",
    built_at: datetime = FAKE_BUILT_AT,
) -> CatalogueSnapshot:
    """A real snapshot admitting only what it is given. `base_variables`
    maps a variable to its admitted base provenances and the scan places
    holding it, and each provenance holds data at every place of every
    variable naming it. A base candidate it does not list is admitted unless
    `withheld_variables` names it. `dataset_facts` defaults to empty facts
    for every DFO and base row's entry (`empty_dataset_facts`)."""
    rows = {row.id: row for row in provenances}
    places = tuple(scan_places)
    index = {place: i for i, place in enumerate(places)}
    base = {
        variable: BaseVariable(
            provenances=tuple(held_by), places=sum(1 << index[p] for p in held_at)
        )
        for variable, (held_by, held_at) in (base_variables or {}).items()
    }
    tables = CatalogueTables(
        provenances=rows,
        dfo_variables=frozenset(dfo_variables),
        base_variables=base,
        provenance_places=spread_provenance_places(base),
        withheld_variables=frozenset(withheld_variables),
        topics={topic: tuple(leaves) for topic, leaves in (topics or {}).items()},
        scan_places=places,
        scan_levels={},
        dataset_facts=(
            empty_dataset_facts(rows.values()) if dataset_facts is None else dataset_facts
        ),
        un_blocked_places=frozenset(un_blocked_places),
    )
    manifest = CatalogueManifest(
        build_id=build_id,
        built_at=built_at,
        build_seconds=0.0,
        identity_inputs={},
        registry_hash="",
        row_counts={},
        resident_bytes=0,
        overlaps=(),
        files={},
    )
    return CatalogueSnapshot(manifest=manifest, tables=tables, un_system_enabled=un_system_enabled)


def empty_dataset_facts(rows: Iterable[ProvenanceRow]) -> dict[str, CardFacts]:
    """Empty `CardFacts` for the entry of every DFO and base row in `rows`, as
    `CatalogueTables` requires."""
    return {entry_ref(row): CardFacts() for row in rows if row.data_commons != UN_SYSTEM}


def provenance_row(
    provenance_id: str,
    origin: DataCommons,
    *,
    source: str | None = None,
    segment: str | None = None,
    dataset: DatasetRef | None = None,
) -> ProvenanceRow:
    """A provenance row labelled with its own id, for `restrictive_catalogue`."""
    ref = source or provenance_id
    return ProvenanceRow(
        id=provenance_id,
        data_commons=origin,
        source=SourceRef(ref=ref, label=ref),
        name=provenance_id,
        url=f"https://example.org/{provenance_id}",
        segment=segment,
        dataset=dataset,
    )


__all__ = [
    "FAKE_BUILT_AT",
    "FakeCatalogue",
    "empty_dataset_facts",
    "fake_catalogue_service",
    "provenance_row",
    "restrictive_catalogue",
    "spread_provenance_places",
]
