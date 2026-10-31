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

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from artifact_publication.jobs import Ingestion
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
    UnTables,
    entry_ref,
)
from origin_catalogue.arc_tables import EMPTY_ARC_TABLES, ArcTables
from origin_catalogue.catalogue import (
    EMPTY_UN_TABLES,
    SourceDocumentation,
    release_held_arcs,
    release_held_family,
)
from origin_catalogue.families import (
    FamilyKey,
    FamilySummary,
    FamilyTables,
    family_tables,
    summarise_table,
)
from origin_catalogue.frozen_arcs import FrozenArcs, served_arcs
from origin_catalogue.origins import UN_ID_PREFIX
from origin_catalogue.schema import UnAgencyCounts
from origin_catalogue.service import CatalogueService
from origin_catalogue.un_mirror import EMPTY_UN_MIRROR, UnArc, UnMirror, UnVariable

FAKE_BUILT_AT = datetime(2026, 9, 29, tzinfo=UTC)


def _fake_build_id(
    stem: str, *, rows: Mapping[str, ProvenanceRow], arcs: ArcTables, un: UnMirror
) -> str:
    """`stem` for a catalogue holding no rows, arc table or UN table, else `stem` and a digest
    of all three, so a cache keyed on the build id, such as the shared held-family decodes,
    never serves one fake's tables to another."""
    if not rows and arcs == EMPTY_ARC_TABLES and un == EMPTY_UN_MIRROR:
        return stem
    tables = repr((sorted(rows.items()), arcs, un))
    return f"{stem}-{hashlib.sha256(tables.encode()).hexdigest()[:12]}"


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
    answers `dataset_facts` and `documentation` answers `source_documentation`,
    both empty by default. Facet reads send no domain
    filter, and no place is a scan place, so presence is always read live.
    `un` holds no UN table by default, so every UN node is read from the UN
    deployment; a UN dcid outside the tables it is given is admitted too.
    `successors` maps a superseded variable to the UN variable standing for
    it, empty by default, and names none with the UN origin off.
    `arcs` holds no arc table by default, so every node of ours is read from
    our mixer, and it holds no topic table. `families` are the family tables
    of `arcs` and `un`, computed as the catalogue job computes them, and `base_families`
    the summaries of `arcs.base` alone. Its arcs and families answer as a snapshot's do: the
    bound graph release's DFO half joined to its own base and UN halves
    (`origin_catalogue.catalogue.release_held_arcs`, `release_held_family`), so its own DFO
    rows give way to the release's.
    `ingestion` is the ingestion the build read, none by default.
    `superseded` holds the superseded dcids with no successor. A superseded
    dcid is the one candidate it refuses, as a snapshot withholds it. `scope`, set
    by `scoped`, admits only the provenances it holds, as a snapshot's does. Left
    unset, `build_id` names the rows and tables it holds (`_fake_build_id`)."""

    build_id: str = ""
    built_at: datetime = FAKE_BUILT_AT
    ingestion: Ingestion | None = None
    dfo_variables: frozenset[str] = frozenset()
    un_system_enabled: bool = True
    mixer_domains: tuple[str, ...] | None = None
    un_blocked_places: frozenset[str] = frozenset()
    provenances: Mapping[str, ProvenanceRow] = field(default_factory=dict, compare=False)
    card_facts: Mapping[str, CardFacts] = field(default_factory=dict, compare=False)
    documentation: Mapping[str, SourceDocumentation] = field(default_factory=dict, compare=False)
    un: UnTables = field(default=EMPTY_UN_TABLES, compare=False)
    arcs: ArcTables = field(default=EMPTY_ARC_TABLES, compare=False)
    families: FamilyTables = field(init=False, compare=False, repr=False)
    base_families: Mapping[FamilyKey, FamilySummary] = field(init=False, compare=False, repr=False)
    successors: Mapping[str, str] = field(default_factory=dict, compare=False)
    superseded: frozenset[str] = frozenset()
    scope: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "families",
            family_tables(dfo=self.arcs.dfo, base=self.arcs.base, un=self.un.mirror),
        )
        object.__setattr__(self, "base_families", summarise_table(self.arcs.base))
        # Each set of tables names its own build, as a published catalogue's tables change only
        # with its build id, so a cache keyed on the build sees the tables it was built from.
        if not self.build_id:
            build_id = _fake_build_id(
                "fake-catalogue", rows=self.provenances, arcs=self.arcs, un=self.un.mirror
            )
            object.__setattr__(self, "build_id", build_id)

    def scoped(self, allowed: frozenset[str]) -> FakeCatalogue:
        return replace(self, scope=allowed if self.scope is None else self.scope & allowed)

    def is_stale(self, *, now: datetime | None = None) -> bool:
        return False

    def is_scan_place(self, place: str) -> bool:
        return False

    def source_documentation(self, ref: str) -> SourceDocumentation | None:
        return self.documentation.get(ref)

    def provenance(self, provenance_id: str) -> ProvenanceRow | None:
        return self.provenances.get(provenance_id)

    def provenance_rows(self) -> tuple[ProvenanceRow, ...]:
        return tuple(self.provenances.values())

    def dataset_facts(self, ref: str) -> CardFacts | None:
        return self.card_facts.get(ref)

    def is_admitted(self, provenance_id: str) -> bool:
        return True

    def admits_in_scope(self, provenance_id: str) -> bool:
        return self.scope is None or provenance_id in self.scope

    def admit_series(self, provenance_ids: Iterable[str]) -> SeriesAdmission:
        present = dict.fromkeys(provenance_ids)
        return SeriesAdmission(
            admitted=tuple(pid for pid in present if self.admits_in_scope(pid)),
            withheld=(),
        )

    def admit_candidate(self, dcid: str) -> DataCommons | None:
        if self.is_superseded(dcid):
            return None
        return _origin_of(dcid, self.dfo_variables)

    def data_commons(self, variable: str) -> tuple[DataCommons, ...]:
        return (_origin_of(variable, self.dfo_variables),)

    def base_places(self, variable: str) -> tuple[str, ...] | None:
        return None

    def variable_provenances(self, variable: str) -> tuple[ProvenanceRow, ...]:
        return ()

    def un_variable(self, dcid: str) -> UnVariable | None:
        return self.un.mirror.variables.get(dcid)

    def un_series_members(self, series: str) -> tuple[str, ...]:
        return self.un.series_members.get(series, ())

    def un_topic_leaves(self, topic: str) -> tuple[str, ...] | None:
        return self.un.mirror.topics.get(topic)

    def un_node(self, dcid: str) -> UnArc | None:
        return self.un.nodes.get(dcid)

    def held_arcs(self, dcid: str) -> tuple[UnArc, ...] | None:
        return release_held_arcs(self.arcs, dcid)

    def arc_answer(self, dcid: str, rows: tuple[UnArc, ...]) -> FrozenArcs:
        return served_arcs(rows)

    def held_family(self, key: FamilyKey) -> FamilySummary | None:
        return release_held_family(
            key,
            build_id=self.build_id,
            arcs=self.arcs,
            families=self.families,
            base_families=self.base_families,
            un_system_enabled=self.un_system_enabled,
        )

    def arc_target(self, dcid: str) -> UnArc | None:
        return self.arcs.targets.get(dcid)

    def arc_label(self, dcid: str) -> str | None:
        return self.arcs.labels.get(dcid)

    def mixer_topic_leaves(self, topic: str) -> tuple[str, ...] | None:
        return None

    def superseded_by(self, variable: str) -> str | None:
        return self.un.superseded_by(variable)

    def successor(self, dcid: str) -> str | None:
        return self.successors.get(dcid) if self.un_system_enabled else None

    def is_superseded(self, dcid: str) -> bool:
        return dcid in self.successors or dcid in self.superseded

    def un_excluded_series_name(self, series: str) -> str | None:
        return self.un.mirror.excluded_series_names.get(series)


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
    un_mirror: UnMirror = EMPTY_UN_MIRROR,
    arcs: ArcTables = EMPTY_ARC_TABLES,
    successors: Mapping[str, str | None] | None = None,
    un_system_enabled: bool = True,
    dataset_facts: Mapping[str, CardFacts] | None = None,
    source_documentation: Mapping[str, SourceDocumentation] | None = None,
    build_id: str | None = None,
    built_at: datetime = FAKE_BUILT_AT,
    ingestion: Ingestion | None = None,
) -> CatalogueSnapshot:
    """A real snapshot admitting only what it is given. `base_variables`
    maps a variable to its admitted base provenances and the scan places
    holding it, and each provenance holds data at every place of every
    variable naming it. A base candidate it does not list is admitted unless
    `withheld_variables` names it. A UN candidate is admitted when `un_mirror`
    holds it. `arcs` are its arc tables, whose `dfo` nodes with arcs must be listed in
    `dfo_variables`, and its family tables are those of `arcs` and `un_mirror`. `successors` maps a superseded variable, which must be
    withheld, to the UN variable standing for it, or to `None`. `dataset_facts` defaults to empty facts
    for every DFO and base row's entry (`empty_dataset_facts`),
    `source_documentation` to documentation covering no provenance for each
    (`empty_source_documentation`), and `build_id` to one naming its rows
    and tables (`_fake_build_id`). `ingestion` is the ingestion its manifest records. It carries
    the base summaries `held_family` joins to the bound release's (`CatalogueSnapshot.base_families`)."""
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
        source_documentation=(
            empty_source_documentation(rows.values())
            if source_documentation is None
            else source_documentation
        ),
        un_blocked_places=frozenset(un_blocked_places),
        superseded=dict(successors or {}),
    )
    manifest = CatalogueManifest(
        build_id=build_id
        or _fake_build_id("restrictive-catalogue", rows=rows, arcs=arcs, un=un_mirror),
        built_at=built_at,
        build_seconds=0.0,
        identity_inputs={},
        registry_hash="",
        row_counts={},
        resident_bytes=0,
        overlaps=(),
        files={},
        ingestion=ingestion,
    )
    return CatalogueSnapshot(
        manifest=manifest,
        tables=tables,
        un=UnTables(mirror=un_mirror),
        arcs=arcs,
        families=family_tables(dfo=arcs.dfo, base=arcs.base, un=un_mirror),
        un_system_enabled=un_system_enabled,
        base_families=summarise_table(arcs.base),
    )


def un_variable(
    series: str,
    *,
    name: str,
    constraints: Mapping[str, tuple[str, str, str]] | None = None,
    observation_properties: Sequence[str] = (),
) -> UnVariable:
    """A UN variable as the catalogue job copies one: a StatisticalVariable
    of `series` measuring `value`, named `name`. `constraints` maps each
    constraint property to its label, its value and the value's label."""
    arcs = [
        *(
            UnArc(property="constraintProperties", dcid=prop, name=label, types=("Property",))
            for prop, (label, _value, _value_label) in (constraints or {}).items()
        ),
        UnArc(property="measuredProperty", dcid="value", name="value", types=("Property",)),
        UnArc(property="name", value=name),
        *(
            UnArc(property="observationProperties", dcid=prop, types=("Property",))
            for prop in observation_properties
        ),
        UnArc(property="populationType", dcid=series, name=f"{series} name", types=("Class",)),
        UnArc(property="statType", dcid="measuredValue", name="measuredValue"),
        UnArc(property="typeOf", dcid="StatisticalVariable", name="StatisticalVariable"),
        *(
            UnArc(property=prop, dcid=value, name=value_label, types=("Enumeration",))
            for prop, (_label, value, value_label) in (constraints or {}).items()
        ),
    ]
    return UnVariable(series=series, arcs=tuple(arcs))


def un_mirror(
    variables: Mapping[str, UnVariable],
    *,
    excluded: Mapping[str, str] | None = None,
    topics: Mapping[str, Iterable[str]] | None = None,
    superseded_by: str = "provenance/OECDCreditorReportingSystem",
) -> UnMirror:
    """UN tables holding `variables`, the `excluded` variables by series and
    `topics`, counted under one agency. Every excluded series names
    `superseded_by`."""
    excluded = dict(excluded or {})
    exclusions = dict.fromkeys(excluded.values(), superseded_by)
    return UnMirror(
        variables=dict(variables),
        excluded_variables=excluded,
        topics={topic: tuple(leaves) for topic, leaves in (topics or {}).items()},
        exclusions=exclusions,
        excluded_series_names={series: f"{series} name" for series in exclusions},
        agencies={
            "agency": UnAgencyCounts(
                series=len({variable.series for variable in variables.values()} | set(exclusions)),
                variables=len(variables) + len(excluded),
                excluded_series=len(exclusions),
                excluded_variables=len(excluded),
            )
        },
    )


def empty_dataset_facts(rows: Iterable[ProvenanceRow]) -> dict[str, CardFacts]:
    """Empty `CardFacts` for the entry of every DFO and base row in `rows`, as
    `CatalogueTables` requires."""
    return {entry_ref(row): CardFacts() for row in rows if row.data_commons != UN_SYSTEM}


def empty_source_documentation(rows: Iterable[ProvenanceRow]) -> dict[str, SourceDocumentation]:
    """`SourceDocumentation` covering no provenance for the entry of every DFO
    and base row in `rows`, as `CatalogueTables` requires."""
    return {entry_ref(row): SourceDocumentation() for row in rows if row.data_commons != UN_SYSTEM}


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
    "empty_source_documentation",
    "fake_catalogue_service",
    "provenance_row",
    "restrictive_catalogue",
    "spread_provenance_places",
    "un_mirror",
    "un_variable",
]
