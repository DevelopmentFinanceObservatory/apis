"""Builds and publishes the origin catalogue: `python -m origin_catalogue.job`.

A run:

1. skips while `IngestionLock` is held, after checking that the identity
   the last unlocked run saw is published;
2. computes the build identity from the registry file's hash, the DFO
   fingerprint (the latest successful `IngestionHistory` row plus the row
   count and latest `last_update_timestamp` over every `Observation` row),
   the ISO week, the artifact format version and a hash of the job's own
   modules (`JOB_MODULES`), and skips when `current.json` already carries
   it, unless that build carried its UN stage forward, in which case the
   run retries the UN and publishes only if the UN answers;
3. takes the create-only lease `lease.json` at the format root, so
   overlapping scheduler executions do not both build, and skips while
   another run holds an unexpired one;
4. reads DFO's provenances and variables from Spanner through BigQuery;
5. builds the DFO and base rows of the provenance table: DFO rows from
   Spanner, base rows from `<-source` on every registered base source
   through our mixer (keeping `typeOf: Provenance`, minus DFO's and each
   entry's exclusions), and each row's source-index entry's card facts
   (`origin_catalogue.card_facts`);
6. copies the base variable, withheld variable and topic tables forward
   from the published build when (format version, enumeration code hash, registry hash,
   admitted base provenance set, DFO provenance set, ISO week) is unchanged, and builds them with
   `origin_catalogue.enumerate.build_base_tables` otherwise;
7. refuses to publish on any `Refusal`, naming each in the log and exiting
   non-zero, so the execution shows failed;
8. reads the UN stage: a row per registered agency, read from the UN
   deployment together with its root group `undata/g/<segment>`, the
   scan-set countries the UN front door refuses to have named in an
   observation GET, so the origin client never pays to find them, and the
   UN tables (`origin_catalogue.un_mirror`): every member variable of every
   agency with the arcs search derives a shape from, the member variables
   of each excluded series, and the UN topic table. The UN tables are
   copied from the published build when (format version, `UN_MIRROR_MODULES`
   code hash, registry hash, ISO week) is unchanged and that build read them
   from the UN, and enumerated otherwise. When a UN read fails or the
   enumeration is incomplete (a read that stops advancing, a variable count
   that differs from the UN deployment's own, a variable naming no series,
   a topic answering no arcs, a topic tree yielding no topic, tables that do
   not hold together),
   the build carries the published build's UN rows, refused places and UN
   tables forward whole and records it in the manifest, so a UN outage
   delays no DFO or base update and no build holds a partial UN table. With
   no published build to carry from, it fails. A UN row the checks reject,
   an excluded series the agency does not hold and a carried table that
   disagrees with the registry on excluded series or agencies refuse the
   build, as any refusal does;
9. derives the superseded variable table (`origin_catalogue.successors`)
   from the withheld variables and the UN tables the build publishes,
   enumerated, copied or carried forward: each withheld variable of a base
   provenance a UN agency supersedes, with the UN variable standing for it
   where one does. A successor the UN tables do not admit refuses the
   build;
10. discards the build if the lock or the fingerprint moved while it ran,
   and otherwise publishes it with a conditional `current.json` write.

Every run prunes old builds. `origin_catalogue_publish_stale` is logged
once an identity has gone unpublished for more than one cycle (reason
`identity_unpublished`), and once the published build is older than a week
plus one run and one cycle (reason `published_build_age`), since the ISO
week in the identity makes a healthy job publish at least weekly. Every run
that returns logs `origin_catalogue_job_heartbeat` last, naming the
published build and its age, so an alert on its absence catches a job that
stopped running.

Environment: `ORIGIN_CATALOGUE_URI` (`gs://` or `file://`),
`ORIGIN_CATALOGUE_BQ_CONNECTION` and `DATASETS_GRAPH_BASE_URL` (our mixer's
API root, including `/core/api`) are required. `UN_GRAPH_BASE_URL`, the UN
deployment's API root, defaults to `origin_catalogue.graph.UN_BASE_URL`. The
serving surfaces read the same variable with the same default, so the build
and serving read one UN deployment. `UN_SITE_BASE_URL`, the UN site root that serves the group counts, defaults
to `UN_GRAPH_BASE_URL` without its `/core/api`. `DATASETS_GRAPH_TOKEN` is optional.
`ORIGIN_CATALOGUE_BQ_PROJECT` defaults to the connection's project, `ORIGIN_CATALOGUE_GRAPH_TIMEOUT_SECONDS` to 120
and `ORIGIN_CATALOGUE_CYCLE_SECONDS` to 900. `--local-dir PATH` publishes
under a local directory in place of `ORIGIN_CATALOGUE_URI`.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Final, Literal, Protocol, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from artifact_publication.jobs import (
    AliasOnlySettings,
    BigQueryReader,
    LiveBigQuery,
    acquire_lease,
    emit,
    lock_held,
    new_build_id,
    observation_stats_sql,
    prune_versions,
    read_ingestion_fingerprint,
    release_lease,
    tracked_stamp,
    unpublished_seconds,
)
from artifact_publication.uri import join_uri, read_uri_bytes_and_generation
from dc_graph import UNRESOLVED_NODE, Facet, GraphClient, GraphReadError, SubjectPage
from origin_catalogue.artifact import (
    CatalogueArtifactError,
    encode_build,
    format_root,
    load_tables,
    publish_build,
    read_manifest,
    read_un_mirror,
)
from origin_catalogue.card_facts import (
    CHILD_PROPERTIES,
    LICENCE_PROPERTIES,
    PROVENANCE_TYPE,
    Arcs,
    card_facts_of,
    first_dataset_parent,
    licence_labels_of,
    node_label,
    without_unresolved,
)
from origin_catalogue.catalogue import (
    BaseTables,
    CardFacts,
    CatalogueTableError,
    CatalogueTables,
    DatasetRef,
    ProvenanceRow,
    SourceRef,
    entry_ref,
)
from origin_catalogue.enumerate import EnumerationError, build_base_tables
from origin_catalogue.graph import (
    UN_BASE_URL,
    UN_BASE_URL_ENV,
    UN_OBSERVATION_GET_URL_BUDGET_BYTES,
)
from origin_catalogue.origins import BASE, DFO, UN_SYSTEM, DataCommons
from origin_catalogue.registry import (
    BaseSource,
    Registry,
    RegistryError,
    UnSupersededProvenance,
    packaged_registry,
    parse_registry,
    registry_hash,
)
from origin_catalogue.schema import (
    CATALOGUE_FORMAT_VERSION,
    POINTER_FILE,
    CatalogueManifest,
    CataloguePointer,
    UnCarryForward,
    identity_key,
)
from origin_catalogue.successors import Successors, derive_successors, unheld_successors
from origin_catalogue.un_mirror import (
    EMPTY_UN_MIRROR,
    UnGroupCounts,
    UnMirror,
    UnMirrorFaultKind,
    UnSiteClient,
    build_un_mirror,
    registry_exclusions,
    un_root_group,
    un_site_url,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "catalogue_state.json"
_DEFAULT_CYCLE_SECONDS = 900.0  # Cloud Scheduler's 15-minute cadence.
_USER_AGENT = "dfo-origin-catalogue"

_MAX_RUN_SECONDS: Final[float] = 7200.0
"""The catalogue Cloud Run Job's maximum runtime, which its job template's
`timeout` must equal. A lease lapses after it, since Cloud Run has killed
the run holding it by then, and pruning uses it as its grace period."""

_WEEK_SECONDS: Final[float] = 7 * 86_400.0
"""How often a new ISO week changes the identity, the longest a healthy job
goes between publishes."""

_PUBLISH_RESERVE_SECONDS: Final[float] = 900.0
"""Time kept back from the UN stage's deadline: the base topic walk, whose
reads take no deadline, and the checks and writes that follow the build."""

_UN_STAGE_RESERVE_SECONDS: Final[float] = 1800.0
"""Time kept back from the base-table deadline for the UN stage, which
enumerates every agency in about 7 minutes and waits up to 100 seconds on
each read the UN front door rate-limits."""

_MAX_SOURCE_SUBJECTS: Final[int] = 5000
"""The most nodes a registered source's `<-source` read may return. A
source past it refuses the build, since a truncated list would silently
drop provenances."""

_UN_PROBE_VARIABLE: Final[str] = "undata/who/MORT_U5_RATE"
"""The UN variable the blocked-place probe reads. The UN front door refuses a
GET by the places it names, whatever the variable."""

_COUNTRY_PREFIX = "country/"

_PROVENANCE_PROPERTIES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(("typeOf", "name", "url", "source", "isPartOf", *CHILD_PROPERTIES))
)
"""A provenance's identity arcs and the arcs its card facts read."""

_PARENT_PROPERTIES: Final[tuple[str, ...]] = ("typeOf", "name", "url")
"""A parent's type, which decides whether it is a Dataset, and its name and URL."""


class CatalogueBuildError(Exception):
    """An input the catalogue cannot represent: no successful ingestion to
    fingerprint, an observation with no provenance, or a source whose
    `<-source` list is truncated."""


class CataloguePointerReadError(Exception):
    """`current.json` could not be read for a reason other than absence.
    Publication needs a reliable generation token."""


# --- Settings ----------------------------------------------------------------


class JobSettings(AliasOnlySettings):
    """Job settings from the environment, failing fast at construction. Each
    field's `alias` is the only environment variable name read; construct
    by field name to override one. `bq_project` defaults to
    `bq_connection`'s leading segment."""

    model_config = SettingsConfigDict(frozen=True)

    artifact_uri: str = Field(min_length=1, alias="ORIGIN_CATALOGUE_URI")
    bq_connection: str = Field(min_length=1, alias="ORIGIN_CATALOGUE_BQ_CONNECTION")
    bq_project: str = Field(default="", alias="ORIGIN_CATALOGUE_BQ_PROJECT")
    graph_base_url: str = Field(min_length=1, alias="DATASETS_GRAPH_BASE_URL")
    graph_token: str | None = Field(default=None, alias="DATASETS_GRAPH_TOKEN")
    un_base_url: str = Field(default=UN_BASE_URL, min_length=1, alias=UN_BASE_URL_ENV)
    un_site_url: str = Field(default="", alias="UN_SITE_BASE_URL")
    graph_timeout_seconds: float = Field(
        default=120.0, alias="ORIGIN_CATALOGUE_GRAPH_TIMEOUT_SECONDS"
    )
    cycle_seconds: float = Field(
        default=_DEFAULT_CYCLE_SECONDS, alias="ORIGIN_CATALOGUE_CYCLE_SECONDS"
    )

    @field_validator("graph_token", mode="before")
    @classmethod
    def _empty_token_to_none(cls, value: object) -> object:
        return value or None

    @model_validator(mode="after")
    def _default_bq_project(self) -> Self:
        if not self.bq_project:
            object.__setattr__(self, "bq_project", self.bq_connection.split(".")[0])
        return self

    @model_validator(mode="after")
    def _default_un_site_url(self) -> Self:
        if not self.un_site_url:
            derived = un_site_url(self.un_base_url)
            if derived is None:
                raise ValueError(
                    f"{UN_BASE_URL_ENV} {self.un_base_url!r} does not end in /core/api, so "
                    "UN_SITE_BASE_URL must name the UN site root"
                )
            object.__setattr__(self, "un_site_url", derived)
        return self


# --- Collaborators -----------------------------------------------------------


class CatalogueGraph(Protocol):
    """The node reads the provenance stage makes, which `dc_graph.GraphClient`
    provides."""

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage: ...

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]: ...


class UnGraph(CatalogueGraph, Protocol):
    """The UN deployment's reads: the provenance stage's node reads, the
    `/v2/node` POST the UN tables are enumerated with, and the observation
    GET the blocked-place probe sends, which `dc_graph.GraphClient` built
    with an observation GET budget provides. `clear` forgets the blocked
    places, so the probe can name them again."""

    def clear(self) -> None: ...

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict: ...

    def observation_facets_batch(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        domains: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, list[Facet]]]: ...

    def blocked_observation_entities(self) -> frozenset[str]: ...


class BaseTablesBuilder(Protocol):
    """`origin_catalogue.enumerate.build_base_tables`: enumerates the base
    variable, withheld variable and topic tables for the admitted base
    provenances and DFO's provenances through our mixer, finishing by the
    monotonic `deadline`."""

    def __call__(
        self,
        *,
        graph: GraphClient,
        registry: Registry,
        admitted_base_provenances: Mapping[str, ProvenanceRow],
        dfo_provenances: frozenset[str],
        deadline: float | None = None,
    ) -> BaseTables: ...


# --- Identity ----------------------------------------------------------------

JOB_MODULES: Final[tuple[str, ...]] = (
    "artifact.py",
    "card_facts.py",
    "enumerate.py",
    "job.py",
    "origins.py",
    "registry.py",
    "schema.py",
    "successors.py",
    "topic_walk.py",
    "un_mirror.py",
)
"""The `origin_catalogue` modules whose code shapes a build: the job, what it
enumerates and encodes with, the card-fact derivation, and the ids and
registry it writes. A change to
any of them changes the build identity. The serving modules (`catalogue`,
`context`, `graph`, `service`) are left out, so a serving change triggers
no build."""

ENUMERATION_MODULES: Final[tuple[str, ...]] = ("enumerate.py", "topic_walk.py")
"""The modules that build the base variable, withheld variable and topic
tables. A change to either rebuilds them, where an unchanged build carries them forward."""

UN_MIRROR_MODULES: Final[tuple[str, ...]] = ("topic_walk.py", "un_mirror.py")
"""The modules that enumerate the UN tables. A change to either enumerates
them again, where an unchanged build copies them from the published one."""


def iso_week(at: datetime) -> str:
    """`at`'s ISO 8601 week, e.g. `2026-W40`."""
    year, week, _ = at.astimezone(UTC).isocalendar()
    return f"{year}-W{week:02d}"


@cache
def source_digest(modules: tuple[str, ...]) -> str:
    """SHA-256 over the source of `modules`, files of the `origin_catalogue`
    package, each keyed by its name."""
    package = resources.files("origin_catalogue")
    digest = hashlib.sha256()
    for name in sorted(modules):
        data = package.joinpath(name).read_bytes()
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def identity_inputs(
    *,
    registry_digest: str,
    dfo_stamp: str,
    week: str,
    job_code: str,
    enumeration_code: str,
    un_mirror_code: str,
) -> dict[str, str]:
    """What a build's identity hashes. A registry edit, an ingestion, a new
    ISO week and a change to `JOB_MODULES` each change it, so each produces
    a new build. `enumeration_code` and `un_mirror_code` are recorded so the
    next build can recompute this one's `base_tables_key` and `un_mirror_key`."""
    return {
        "format_version": str(CATALOGUE_FORMAT_VERSION),
        "job_code": job_code,
        "enumeration_code": enumeration_code,
        "un_mirror_code": un_mirror_code,
        "registry": registry_digest,
        "dfo_fingerprint": dfo_stamp,
        "iso_week": week,
    }


def base_tables_key(
    *,
    registry_digest: str,
    base_provenances: Iterable[str],
    dfo_provenances: Iterable[str],
    week: str,
    enumeration_code: str,
) -> str:
    """The key under which the base variable, withheld variable and topic
    tables carry forward from one build to the next: every input the base-table build reads
    besides the graph, which the weekly rebuild refreshes. DFO's provenance
    ids are among them, since pass 2 withholds no variable on a DFO facet;
    DFO's variables and rows are not, so an ingestion that adds no provenance
    carries the tables forward. The code of `ENUMERATION_MODULES` is among
    them, so a fix to enumeration rebuilds them."""
    return identity_key(
        {
            "format_version": str(CATALOGUE_FORMAT_VERSION),
            "enumeration_code": enumeration_code,
            "registry": registry_digest,
            "base_provenances": "\n".join(sorted(base_provenances)),
            "dfo_provenances": "\n".join(sorted(dfo_provenances)),
            "iso_week": week,
        }
    )


def un_mirror_key(*, registry_digest: str, week: str, un_mirror_code: str) -> str:
    """The key under which the UN tables are copied from one build to the
    next: every input their enumeration reads besides the UN deployment,
    which the weekly enumeration refreshes."""
    return identity_key(
        {
            "format_version": str(CATALOGUE_FORMAT_VERSION),
            "un_mirror_code": un_mirror_code,
            "registry": registry_digest,
            "iso_week": week,
        }
    )


def _published_un_mirror_key(manifest: CatalogueManifest) -> str | None:
    """The published build's `un_mirror_key`, or `None` when it carried its
    UN tables forward, since no enumeration of its own week stands behind them."""
    week = manifest.identity_inputs.get("iso_week")
    un_mirror_code = manifest.identity_inputs.get("un_mirror_code")
    if week is None or un_mirror_code is None or manifest.un_carried_forward is not None:
        return None
    return un_mirror_key(
        registry_digest=manifest.registry_hash, week=week, un_mirror_code=un_mirror_code
    )


def _published_base_tables_key(manifest: CatalogueManifest, tables: CatalogueTables) -> str | None:
    week = manifest.identity_inputs.get("iso_week")
    enumeration_code = manifest.identity_inputs.get("enumeration_code")
    if week is None or enumeration_code is None:
        return None
    return base_tables_key(
        registry_digest=manifest.registry_hash,
        base_provenances=(
            pid for pid, row in tables.provenances.items() if row.data_commons == BASE
        ),
        dfo_provenances=tables.dfo_provenances,
        week=week,
        enumeration_code=enumeration_code,
    )


# --- DFO stage ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DfoHoldings:
    """What our Spanner holds: every provenance with observations, every
    variable with observations, and the count of distinct (variable,
    provenance) pairs."""

    provenances: frozenset[str]
    variables: frozenset[str]
    pairs: int


def read_dfo_holdings(bq: BigQueryReader, *, connection: str) -> DfoHoldings:
    """Reads the distinct (variable, provenance) pairs with observations.

    `Observation` carries no provenance; it joins to `TimeSeries` on
    `facet_id`. Spanner's root-partition rules reject `DISTINCT` inside
    `EXTERNAL_QUERY`, so both sides are plain projections and BigQuery
    aggregates. An observation whose facet has no `TimeSeries` provenance
    raises `CatalogueBuildError`.
    """
    sql = (
        "SELECT DISTINCT o.variable_measured AS variable, f.provenance AS provenance\n"
        f'FROM EXTERNAL_QUERY("{connection}", '
        '"SELECT variable_measured, facet_id FROM Observation") AS o\n'
        "LEFT JOIN (\n"
        "  SELECT DISTINCT facet_id, provenance\n"
        f'  FROM EXTERNAL_QUERY("{connection}", "SELECT facet_id, provenance FROM TimeSeries")\n'
        ") AS f USING (facet_id)"
    )
    rows = bq.query(sql)
    orphans = sorted({str(row["variable"]) for row in rows if not row["provenance"]})
    if orphans:
        raise CatalogueBuildError(
            f"{len(orphans)} variable(s) have observations whose facet has no TimeSeries "
            f"provenance, e.g. {orphans[:10]!r}"
        )
    return DfoHoldings(
        provenances=frozenset(str(row["provenance"]) for row in rows),
        variables=frozenset(str(row["variable"]) for row in rows),
        pairs=len(rows),
    )


# --- Provenance stage --------------------------------------------------------

type RefusalKind = (
    Literal[
        "base_source_without_variables",
        "exclusion_not_held",
        "dfo_provenance_under_unreviewed_source",
        "dfo_provenance_without_source",
        "provenance_without_url",
        "un_agency_root_group_missing",
        "un_carried_agency_mismatch",
        "un_carried_exclusions_mismatch",
        "un_carried_segment_mismatch",
        "un_exclusion_provenance_not_held",
        "un_successor_not_held",
        "un_successor_ambiguous",
        "un_successor_not_served",
    ]
    | UnMirrorFaultKind
)


@dataclass(frozen=True, slots=True)
class Refusal:
    """One reason the build may not publish. `subject` is the dcid at fault."""

    kind: RefusalKind
    subject: str
    detail: str


class CatalogueRefusedError(Exception):
    """The build found at least one `Refusal` and published nothing."""

    def __init__(self, refusals: Sequence[Refusal]) -> None:
        self.refusals = tuple(refusals)
        super().__init__(
            f"refused to publish on {len(self.refusals)} refusal(s): "
            + ", ".join(f"{r.kind} {r.subject}" for r in self.refusals)
        )


@dataclass(frozen=True, slots=True)
class ProvenanceStage:
    """The DFO and base rows of the provenance table, each registered base
    source's admitted provenances in registry order, the card facts of each
    row's source-index entry (keyed by `entry_ref`), and the refusals found
    building them."""

    rows: Mapping[str, ProvenanceRow]
    by_source: Mapping[str, tuple[str, ...]]
    dataset_facts: Mapping[str, CardFacts]
    refusals: tuple[Refusal, ...]

    @property
    def base_rows(self) -> dict[str, ProvenanceRow]:
        return {pid: row for pid, row in self.rows.items() if row.data_commons == BASE}

    @property
    def dfo_ids(self) -> frozenset[str]:
        return frozenset(pid for pid, row in self.rows.items() if row.data_commons == DFO)


def _first(arcs: Mapping[str, tuple[str, ...]] | None, prop: str) -> str | None:
    for value in (arcs or {}).get(prop, ()):
        if value and value != UNRESOLVED_NODE:
            return value
    return None


type _ArcsByDcid = Mapping[str, dict[str, tuple[str, ...]] | None]


def _read_parents(
    graph: CatalogueGraph, arcs: _ArcsByDcid
) -> dict[str, dict[str, tuple[str, ...]] | None]:
    """The type, name and URL of every `isPartOf` parent the nodes in `arcs`
    name, from one batched read."""
    parents = sorted({p for node in arcs.values() for p in (node or {}).get("isPartOf", ())})
    if not parents:
        return {}
    return without_unresolved(graph.node_arcs_many(parents, properties=_PARENT_PROPERTIES))


def _dataset(arcs: Arcs | None, parents: _ArcsByDcid) -> DatasetRef | None:
    """The provenance's first `isPartOf` parent typed Dataset
    (`first_dataset_parent`), labelled with its name, else its dcid. `None`
    for an orphan."""
    ref, parent_arcs = first_dataset_parent((arcs or {}).get("isPartOf", ()), parents)
    if ref is None:
        return None
    return DatasetRef(ref=ref, label=node_label(parent_arcs, ref) or ref)


def _title(
    provenance: str, arcs: Arcs | None, dataset: DatasetRef | None, parents: _ArcsByDcid
) -> str:
    """A provenance's display name, by the chain datasets citations use: its
    parent Dataset's name, else its own, else its dcid. A Provenance's own
    `name` is usually an import slug (`GeoBoundaries_Jordan`)."""
    return (
        (node_label(parents.get(dataset.ref), dataset.ref) if dataset is not None else None)
        or node_label(arcs, provenance)
        or provenance
    )


def _licence_labels(
    graph: CatalogueGraph, rows: Mapping[str, ProvenanceRow], arcs: _ArcsByDcid
) -> dict[str, str]:
    """The label of every `licenseType` node the rows name, from one batched
    read. Rows that name none read nothing."""
    licences = list(
        dict.fromkeys(
            licence
            for provenance in rows
            for licence in (arcs.get(provenance) or {}).get("licenseType", ())
        )
    )
    if not licences:
        return {}
    return licence_labels_of(
        without_unresolved(graph.node_arcs_many(licences, properties=LICENCE_PROPERTIES))
    )


def _dataset_facts(
    rows: Mapping[str, ProvenanceRow],
    arcs: _ArcsByDcid,
    parents: _ArcsByDcid,
    licence_labels: Mapping[str, str],
) -> dict[str, CardFacts]:
    """The card facts of each row's source-index entry. A Dataset entry's
    children are the rows typed `Provenance` that name the Dataset anywhere in
    `isPartOf`, the set the documentation reader reads for it. An orphan is
    its own sole child. `licence_labels` comes from `_licence_labels`."""
    children_of: dict[str, dict[str, Arcs]] = {}
    for provenance in rows:
        node = arcs.get(provenance) or {}
        if PROVENANCE_TYPE not in node.get("typeOf", ()):
            continue
        for parent in node.get("isPartOf", ()):
            children_of.setdefault(parent, {})[provenance] = node
    facts: dict[str, CardFacts] = {}
    for row in rows.values():
        ref = entry_ref(row)
        if ref in facts:
            continue
        if row.dataset is None:
            facts[ref] = card_facts_of(None, {row.id: arcs.get(row.id) or {}}, licence_labels)
        else:
            facts[ref] = card_facts_of(parents.get(ref), children_of.get(ref, {}), licence_labels)
    return facts


def build_provenance_table(
    *,
    registry: Registry,
    dfo_provenances: frozenset[str],
    graph: CatalogueGraph,
) -> ProvenanceStage:
    """Builds every admitted DFO and base provenance's row.

    Base rows: the `typeOf: Provenance` nodes whose `source` is a registered
    base source, minus DFO's provenances and the entry's exclusions, labelled
    with the registry's source label. DFO rows: each Spanner provenance with
    its URL and source. A row's `dataset` is its first parent typed Dataset
    (`isPartOf`), and its name the Dataset's name (`_title`).
    `dataset_facts` holds each row's entry's card facts (`_dataset_facts`).
    """
    refusals: list[Refusal] = []
    subjects: dict[str, tuple[str, ...]] = {}
    for entry in registry.base:
        page = graph.node_subjects(
            entry.source, property="source", max_results=_MAX_SOURCE_SUBJECTS
        )
        if page.truncated:
            raise CatalogueBuildError(
                f"{entry.source!r} has more than {_MAX_SOURCE_SUBJECTS} `<-source` subjects"
            )
        subjects[entry.source] = page.dcids
    to_read = sorted({dcid for dcids in subjects.values() for dcid in dcids} | dfo_provenances)
    arcs = without_unresolved(graph.node_arcs_many(to_read, properties=_PROVENANCE_PROPERTIES))
    parents = _read_parents(graph, arcs)

    rows: dict[str, ProvenanceRow] = {}
    by_source: dict[str, tuple[str, ...]] = {}
    for entry in registry.base:
        held = [
            dcid
            for dcid in subjects[entry.source]
            if PROVENANCE_TYPE in (arcs.get(dcid) or {}).get("typeOf", ())
        ]
        excluded = {exclusion.provenance for exclusion in entry.exclusions or ()}
        refusals.extend(
            Refusal(
                kind="exclusion_not_held",
                subject=provenance,
                detail=f"{entry.source} excludes {provenance}, which it does not hold",
            )
            for provenance in sorted(excluded - set(held))
        )
        admitted: list[str] = []
        for provenance in held:
            if provenance in dfo_provenances or provenance in excluded:
                continue
            if provenance in rows:
                logger.warning(
                    "%s is held by %s and %s; its row names the first",
                    provenance,
                    rows[provenance].source.ref,
                    entry.source,
                )
                continue
            dataset = _dataset(arcs.get(provenance), parents)
            row = _row(
                provenance,
                arcs.get(provenance),
                origin=BASE,
                source=SourceRef(ref=entry.source, label=entry.label),
                name=_title(provenance, arcs.get(provenance), dataset, parents),
                refusals=refusals,
                dataset=dataset,
            )
            if row is not None:
                rows[provenance] = row
                admitted.append(provenance)
        by_source[entry.source] = tuple(admitted)

    rows.update(
        _dfo_rows(
            registry,
            dfo_provenances,
            arcs=arcs,
            parents=parents,
            graph=graph,
            refusals=refusals,
        )
    )
    return ProvenanceStage(
        rows=rows,
        by_source=by_source,
        dataset_facts=_dataset_facts(rows, arcs, parents, _licence_labels(graph, rows, arcs)),
        refusals=tuple(refusals),
    )


def _hostname(url: str) -> str | None:
    try:
        return urlsplit(url).hostname or None
    except ValueError:
        return None


def _row(
    provenance: str,
    arcs: Mapping[str, tuple[str, ...]] | None,
    *,
    origin: DataCommons,
    source: SourceRef,
    name: str,
    refusals: list[Refusal],
    segment: str | None = None,
    dataset: DatasetRef | None = None,
) -> ProvenanceRow | None:
    """The provenance's row, or `None` with a refusal when it has no URL or
    its URL has no hostname, since the domain filter matches facets by
    their provenance URL's hostname."""
    url = _first(arcs, "url")
    if url is None or _hostname(url) is None:
        refusals.append(
            Refusal(
                kind="provenance_without_url",
                subject=provenance,
                detail=(
                    f"admitted provenance {provenance} has no url arc"
                    if url is None
                    else f"admitted provenance {provenance} has url {url!r}, which has no hostname"
                ),
            )
        )
        return None
    return ProvenanceRow(
        id=provenance,
        data_commons=origin,
        source=source,
        name=name,
        url=url,
        segment=segment,
        dataset=dataset,
    )


def _dfo_rows(
    registry: Registry,
    dfo_provenances: frozenset[str],
    *,
    arcs: _ArcsByDcid,
    parents: _ArcsByDcid,
    graph: CatalogueGraph,
    refusals: list[Refusal],
) -> dict[str, ProvenanceRow]:
    """DFO rows with the source label from the registry when the source is
    registered, and from the source node's `name` otherwise. A DFO
    provenance under a registered source whose entry states no exclusions
    is refused, since its overlap with that source was not reviewed."""
    entries: dict[str, BaseSource] = {entry.source: entry for entry in registry.base}
    sources = {p: _first(arcs.get(p), "source") for p in sorted(dfo_provenances)}
    unregistered = sorted({s for s in sources.values() if s is not None and s not in entries})
    names = graph.node_arcs_many(unregistered, properties=("name",)) if unregistered else {}
    rows: dict[str, ProvenanceRow] = {}
    for provenance, source in sources.items():
        if source is None:
            refusals.append(
                Refusal(
                    kind="dfo_provenance_without_source",
                    subject=provenance,
                    detail=f"DFO provenance {provenance} has no source arc",
                )
            )
            continue
        entry = entries.get(source)
        if entry is not None and entry.exclusions is None:
            refusals.append(
                Refusal(
                    kind="dfo_provenance_under_unreviewed_source",
                    subject=provenance,
                    detail=(
                        f"DFO provenance {provenance} is under {source}, whose registry entry "
                        "states no exclusions"
                    ),
                )
            )
        label = entry.label if entry is not None else _first(names.get(source), "name") or source
        dataset = _dataset(arcs.get(provenance), parents)
        row = _row(
            provenance,
            arcs.get(provenance),
            origin=DFO,
            source=SourceRef(ref=source, label=label),
            name=_title(provenance, arcs.get(provenance), dataset, parents),
            refusals=refusals,
            dataset=dataset,
        )
        if row is not None:
            rows[provenance] = row
    return rows


def sources_without_variables(
    by_source: Mapping[str, tuple[str, ...]], base: BaseTables
) -> list[Refusal]:
    """A refusal for each registered base source none of whose admitted
    provenances holds a variable in the base variable table."""
    holding = {pid for entry in base.base_variables.values() for pid in entry.provenances}
    return [
        Refusal(
            kind="base_source_without_variables",
            subject=source,
            detail=(
                f"{source} has {len(provenances)} admitted provenance(s) and none holds an "
                "admitted variable"
            ),
        )
        for source, provenances in by_source.items()
        if holding.isdisjoint(provenances)
    ]


# --- UN stage ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnStage:
    """The UN rows of the provenance table, the scan-set countries the UN
    front door refuses to have named in an observation GET, and the UN
    tables. `carried_forward` is set when all three come from an earlier
    build. `mirror_source` says where the UN tables come from: enumerated by
    this build, copied from the published build under an unchanged
    `un_mirror_key`, or carried forward after a failed UN read or an
    incomplete enumeration."""

    rows: Mapping[str, ProvenanceRow]
    blocked_places: frozenset[str]
    mirror: UnMirror = EMPTY_UN_MIRROR
    mirror_source: Literal["enumerated", "copied", "carried"] = "enumerated"
    carried_forward: UnCarryForward | None = None


class UnReadError(Exception):
    """A read of the UN deployment failed or its enumeration is incomplete,
    so the UN stage has no result."""


def read_un_stage(
    registry: Registry,
    *,
    un_graph: UnGraph,
    un_group_counts: UnGroupCounts,
    scan_places: Sequence[str],
    refusals: list[Refusal],
    published_mirror: Callable[[], UnMirror | None] | None = None,
    deadline: float | None = None,
) -> UnStage:
    """Reads the UN rows, appending a refusal for each row the checks reject,
    then, when none is refused, probes the UN front door over the scan-set
    countries and reads the UN tables: `published_mirror`'s when it returns
    them, and `build_un_mirror`'s otherwise. A fault the registry causes is
    a refusal, and beside one every other fault is too, so one run names
    them all. Raises `UnReadError` when a UN read fails, when `deadline`, an
    absolute `time.monotonic()` value, passes, and when every fault is an
    incomplete enumeration."""
    try:
        found: list[Refusal] = []
        rows = _un_rows(registry, un_graph=un_graph, refusals=found)
        refusals.extend(found)
        if found:
            return UnStage(rows=rows, blocked_places=frozenset())
        blocked = probe_un_blocked_places(un_graph, scan_places)
        copied = published_mirror() if published_mirror is not None else None
        if copied is not None:
            return UnStage(rows=rows, blocked_places=blocked, mirror=copied, mirror_source="copied")
        built = build_un_mirror(
            registry, graph=un_graph, group_counts=un_group_counts, deadline=deadline
        )
    except GraphReadError as exc:
        raise UnReadError(f"UN deployment read failed: {exc}") from exc
    if built.faults and all(fault.incomplete for fault in built.faults):
        faults = "; ".join(
            f"{fault.kind} at {fault.subject}: {fault.detail}" for fault in built.faults
        )
        raise UnReadError(f"UN enumeration incomplete: {faults}")
    refusals.extend(
        Refusal(kind=fault.kind, subject=fault.subject, detail=fault.detail)
        for fault in built.faults
    )
    return UnStage(rows=rows, blocked_places=blocked, mirror=built.mirror or EMPTY_UN_MIRROR)


def un_registry_refusals(
    registry: Registry, *, dfo_provenances: frozenset[str], un_rows: Mapping[str, ProvenanceRow]
) -> list[Refusal]:
    """A refusal for each excluded UN series whose `dfo_provenance` DFO does
    not hold, and for each base provenance superseded by a UN agency the
    build has no row for."""
    refusals = [
        Refusal(
            kind="un_exclusion_provenance_not_held",
            subject=exclusion.series,
            detail=(
                f"{agency.provenance} excludes {exclusion.series} as superseded by "
                f"{exclusion.dfo_provenance}, which DFO does not hold"
            ),
        )
        for agency in registry.un_system
        for exclusion in agency.exclusions
        if exclusion.dfo_provenance not in dfo_provenances
    ]
    refusals.extend(
        Refusal(
            kind="un_successor_not_served",
            subject=exclusion.provenance,
            detail=(
                f"{entry.source} excludes {exclusion.provenance} as superseded by "
                f"{exclusion.successor}, which has no row in this build"
            ),
        )
        for entry in registry.base
        for exclusion in entry.exclusions or ()
        if isinstance(exclusion, UnSupersededProvenance) and exclusion.successor not in un_rows
    )
    return refusals


def successor_refusals(successors: Successors, mirror: UnMirror) -> list[Refusal]:
    """A refusal for each successor row naming a variable `mirror` does not
    admit, one of an excluded series or one the UN tables do not hold, and
    for each superseded variable whose code and constraint pairs two UN
    variables share."""
    return [
        *(
            Refusal(
                kind="un_successor_ambiguous",
                subject=variable,
                detail=(
                    f"{len(matched)} variables of the successor agency hold {variable}'s "
                    f"series code and constraint pairs: {list(matched)!r}"
                ),
            )
            for variable, matched in successors.ambiguous.items()
        ),
        *(
            Refusal(
                kind="un_successor_not_held",
                subject=variable,
                detail=(
                    f"{variable} has successor {successors.rows[variable]}, which is not an "
                    "admitted variable of the build's UN tables"
                ),
            )
            for variable in unheld_successors(successors.rows, mirror)
        ),
    ]


def _un_rows(
    registry: Registry, *, un_graph: CatalogueGraph, refusals: list[Refusal]
) -> dict[str, ProvenanceRow]:
    """UN rows, each agency provenance serving as its own source and named
    by the registry's label, as the agency's parent Dataset on the UN
    deployment is the UN Statistics Division for every agency. An agency
    whose root group `undata/g/<segment>` is absent is refused."""
    groups = {agency.provenance: un_root_group(agency) for agency in registry.un_system}
    arcs = un_graph.node_arcs_many([*groups, *groups.values()], properties=("typeOf", "url"))
    rows: dict[str, ProvenanceRow] = {}
    for agency in registry.un_system:
        group = groups[agency.provenance]
        if arcs.get(group) is None:
            refusals.append(
                Refusal(
                    kind="un_agency_root_group_missing",
                    subject=agency.provenance,
                    detail=f"{agency.provenance}'s root group {group} is absent on the UN deployment",
                )
            )
        row = _row(
            agency.provenance,
            arcs.get(agency.provenance),
            origin=UN_SYSTEM,
            source=SourceRef(ref=agency.provenance, label=agency.label),
            name=agency.label,
            refusals=refusals,
            segment=agency.segment,
        )
        if row is not None:
            rows[agency.provenance] = row
    return rows


def probe_un_blocked_places(un_graph: UnGraph, scan_places: Sequence[str]) -> frozenset[str]:
    """The scan-set countries the UN front door refuses to have named in an
    observation GET.

    One read of `_UN_PROBE_VARIABLE` over the countries, through the GET
    path the origin client uses, halves a refused request until each
    refused place stands alone. The client then forgets what it found, and
    each found place is read alone once more. A place counts as refused only
    when the front door refuses it both times, so one stray 403 does not keep
    a place's UN data out of a build.
    """
    countries = [place for place in scan_places if place.startswith(_COUNTRY_PREFIX)]
    if not countries:
        return frozenset()
    un_graph.observation_facets_batch([_UN_PROBE_VARIABLE], countries)
    found = un_graph.blocked_observation_entities() & frozenset(countries)
    if not found:
        return frozenset()
    un_graph.clear()
    for place in sorted(found):
        un_graph.observation_facets_batch([_UN_PROBE_VARIABLE], [place])
    return un_graph.blocked_observation_entities() & found


# --- Publication -------------------------------------------------------------


def _read_published(root: str) -> tuple[CataloguePointer | None, str | None]:
    """Reads the pointer and its version token. A missing pointer returns
    `(None, None)`. A malformed one returns `(None, generation)`, so publication
    can replace it. Other read failures raise `CataloguePointerReadError`."""
    uri = join_uri(root, POINTER_FILE)
    try:
        found = read_uri_bytes_and_generation(uri)
    except Exception as exc:
        raise CataloguePointerReadError(f"couldn't read {uri!r}: {exc}") from exc
    if found is None:
        return None, None
    raw, generation = found
    try:
        return CataloguePointer.from_json(raw), generation
    except ValueError:
        logger.warning("origin_catalogue: current.json at %r is malformed; rebuilding", uri)
        return None, generation


@dataclass(frozen=True, slots=True)
class _CarriedUn:
    """The published build's UN rows and refused places, a reader of its UN
    tables, their `un_mirror_key` (`None` when that build carried them
    forward), and the build that last read all three from the UN deployment."""

    rows: Mapping[str, ProvenanceRow]
    blocked_places: frozenset[str]
    read_mirror: Callable[[], UnMirror]
    mirror_key: str | None
    from_build: str

    def copied_mirror(self, key: str) -> UnMirror | None:
        """The published UN tables when their key equals `key`, else `None`.
        Unreadable tables are enumerated again."""
        if self.mirror_key != key:
            return None
        try:
            return self.read_mirror()
        except CatalogueArtifactError as exc:
            logger.warning("origin_catalogue: published UN tables unreadable, enumerating: %s", exc)
            return None

    def stage(self, registry: Registry, *, reason: str, refusals: list[Refusal]) -> UnStage:
        """The carried rows of the agencies `registry` still lists, each
        keeping its own segment and taking the registry's current label,
        with the carried refused places and the carried UN tables, whole. A
        carried row whose segment is not the registry's is refused, since no
        UN read has checked the new segment's root group. Carried UN tables
        whose excluded series are not the registry's are refused, since
        their variables were admitted under other exclusions, and so are
        carried UN tables holding an agency segment the registry no longer
        lists or lists under another provenance, since admission of a UN
        candidate reads the tables."""
        rows: dict[str, ProvenanceRow] = {}
        for agency in registry.un_system:
            row = self.rows.get(agency.provenance)
            if row is None:
                continue
            if row.segment != agency.segment:
                refusals.append(
                    Refusal(
                        kind="un_carried_segment_mismatch",
                        subject=agency.provenance,
                        detail=(
                            f"{agency.provenance} carries segment {row.segment!r} from build "
                            f"{self.from_build}, and the registry names {agency.segment!r}"
                        ),
                    )
                )
                continue
            rows[agency.provenance] = replace(
                row, source=SourceRef(ref=agency.provenance, label=agency.label), name=agency.label
            )
        try:
            mirror = self.read_mirror()
        except CatalogueArtifactError as exc:
            raise CatalogueBuildError(
                f"{reason}, and the published build's UN tables are unreadable: {exc}"
            ) from exc
        carried, current = mirror.exclusions, registry_exclusions(registry)
        refusals.extend(
            Refusal(
                kind="un_carried_exclusions_mismatch",
                subject=series,
                detail=(
                    f"the UN tables carried from build {self.from_build} exclude {series} as "
                    f"superseded by {carried.get(series)!r}, and the registry names "
                    f"{current.get(series)!r}"
                ),
            )
            for series in sorted(carried.keys() | current.keys())
            if carried.get(series) != current.get(series)
        )
        listed = {agency.segment: agency.provenance for agency in registry.un_system}
        carried_provenance = {
            row.segment: provenance for provenance, row in self.rows.items() if row.segment
        }
        refusals.extend(
            Refusal(
                kind="un_carried_agency_mismatch",
                subject=segment,
                detail=(
                    f"the UN tables carried from build {self.from_build} hold agency segment "
                    f"{segment!r} under provenance {carried_provenance.get(segment)!r}, and the "
                    f"registry lists {listed.get(segment)!r} for it"
                ),
            )
            for segment in sorted(mirror.agencies)
            if listed.get(segment) is None or listed[segment] != carried_provenance.get(segment)
        )
        return UnStage(
            rows=rows,
            blocked_places=self.blocked_places,
            mirror=mirror,
            mirror_source="carried",
            carried_forward=UnCarryForward(from_build=self.from_build, reason=reason),
        )


def _published_carry(base_uri: str, *, key: str) -> tuple[BaseTables | None, _CarriedUn | None]:
    """What the published build can carry forward: its base tables when
    their carry-forward key equals `key`, and its UN rows, refused places
    and UN tables. Only these outlive the call, and the UN tables are read
    and decoded when a build takes them and not before, so a base-table
    build does not hold the whole
    published build in memory. Carried tables are re-encoded into
    the new build directory, so the new build does not reference the old
    one, which a later prune deletes."""
    try:
        manifest, tables = load_tables(base_uri)
    except CatalogueArtifactError as exc:
        logger.warning("origin_catalogue: published build unreadable, building afresh: %s", exc)
        return None, None
    base = None
    if _published_base_tables_key(manifest, tables) == key:
        base = BaseTables(
            base_variables=tables.base_variables,
            provenance_places=tables.provenance_places,
            withheld_variables=tables.withheld_variables,
            topics=tables.topics,
            scan_places=tables.scan_places,
            scan_levels=tables.scan_levels,
        )
    earlier = manifest.un_carried_forward
    un = _CarriedUn(
        rows={pid: row for pid, row in tables.provenances.items() if row.data_commons == UN_SYSTEM},
        blocked_places=tables.un_blocked_places,
        read_mirror=lambda: read_un_mirror(base_uri, manifest),
        mirror_key=_published_un_mirror_key(manifest),
        from_build=manifest.build_id if earlier is None else earlier.from_build,
    )
    return base, un


def prune_old_versions(base_uri: str, *, now: Callable[[], float] = time.time) -> tuple[str, ...]:
    """Deletes and returns the build ids safe to remove from this format
    root, with `_MAX_RUN_SECONDS` as the grace period."""
    return prune_versions(
        format_root(base_uri),
        pointer_build_id=lambda raw: CataloguePointer.from_json(raw).build_id,
        grace_seconds=_MAX_RUN_SECONDS,
        now=now,
    )


def _prune(base_uri: str, clock: Callable[[], float]) -> None:
    pruned = prune_old_versions(base_uri, now=clock)
    if pruned:
        emit("origin_catalogue_job_pruned", builds=list(pruned))


def _track_stale(
    settings: JobSettings, *, identity: str, published: str | None, clock: Callable[[], float]
) -> None:
    """Logs `origin_catalogue_publish_stale` once `identity` has gone
    unpublished for more than one cycle."""
    if published == identity:
        return
    behind = unpublished_seconds(
        format_root(settings.artifact_uri), state_file=_STATE_FILE, stamp=identity, now=clock()
    )
    if behind > settings.cycle_seconds:
        emit(
            "origin_catalogue_publish_stale",
            reason="identity_unpublished",
            identity=identity,
            published_identity=published,
            seconds_behind=round(behind, 1),
        )


def _track_stale_under_lock(settings: JobSettings, *, clock: Callable[[], float]) -> None:
    """`_track_stale` for a run that finds `IngestionLock` held. The
    fingerprint moves during a load, so the run checks the identity the
    last unlocked run saw in place of its own, and a lock held past a cycle
    with that identity unpublished still logs `origin_catalogue_publish_stale`."""
    root = format_root(settings.artifact_uri)
    pending = tracked_stamp(root, state_file=_STATE_FILE)
    if pending is None:
        return
    published, _ = _read_published(root)
    _track_stale(
        settings,
        identity=pending,
        published=published.identity if published is not None else None,
        clock=clock,
    )


def _heartbeat(settings: JobSettings, *, clock: Callable[[], float]) -> None:
    """Logs `origin_catalogue_job_heartbeat` with the published build and
    its age in seconds, both `None` when nothing is published and the age
    `None` when the build's manifest is unreadable. Before it, logs
    `origin_catalogue_publish_stale` when the age passes a week plus one
    run and one cycle. The first run of an ISO week may wait a cycle for a
    lease and then build for up to `_MAX_RUN_SECONDS`."""
    published, _ = _read_published(format_root(settings.artifact_uri))
    if published is None:
        emit("origin_catalogue_job_heartbeat", build_id=None, built_at=None, age_seconds=None)
        return
    try:
        built_at = read_manifest(settings.artifact_uri, published.build_id).built_at
    except CatalogueArtifactError as exc:
        emit(
            "origin_catalogue_job_heartbeat",
            severity="WARNING",
            build_id=published.build_id,
            built_at=None,
            age_seconds=None,
            error=str(exc),
        )
        return
    age = round(clock() - built_at.timestamp(), 1)
    limit = _WEEK_SECONDS + _MAX_RUN_SECONDS + settings.cycle_seconds
    if age > limit:
        emit(
            "origin_catalogue_publish_stale",
            reason="published_build_age",
            build_id=published.build_id,
            age_seconds=age,
            limit_seconds=limit,
        )
    emit(
        "origin_catalogue_job_heartbeat",
        build_id=published.build_id,
        built_at=built_at.isoformat(),
        age_seconds=age,
    )


# --- Orchestration -----------------------------------------------------------


def _un_stage_or_carried(
    registry: Registry,
    *,
    un_graph: UnGraph,
    un_group_counts: UnGroupCounts,
    scan_places: Sequence[str],
    carried: _CarriedUn | None,
    mirror_key: str,
    refusals: list[Refusal],
    deadline: float,
) -> UnStage:
    """The UN stage read afresh, or, when a UN read fails or the enumeration
    is incomplete, the published build's UN stage carried forward. A fresh
    stage copies the published UN tables when their key equals `mirror_key`.
    Refusals from either land in `refusals`. With no published build to
    carry from, either fails the build."""
    try:
        return read_un_stage(
            registry,
            un_graph=un_graph,
            un_group_counts=un_group_counts,
            scan_places=scan_places,
            refusals=refusals,
            published_mirror=(
                None if carried is None else lambda: carried.copied_mirror(mirror_key)
            ),
            deadline=deadline,
        )
    except UnReadError as exc:
        if carried is None:
            raise CatalogueBuildError(
                f"{exc}, and no published build holds UN rows to carry forward"
            ) from exc
        return carried.stage(registry, reason=str(exc), refusals=refusals)


def _published_carries_un(base_uri: str, build_id: str) -> bool:
    """Whether the published build carried its UN stage forward, so a run
    with its identity must retry the UN. An unreadable manifest counts as
    carried, so the run rebuilds."""
    try:
        return read_manifest(base_uri, build_id).un_carried_forward is not None
    except CatalogueArtifactError as exc:
        logger.warning("origin_catalogue: published manifest unreadable, rebuilding: %s", exc)
        return True


def run(
    settings: JobSettings,
    *,
    bq: BigQueryReader,
    graph: GraphClient,
    un_graph: UnGraph,
    un_group_counts: UnGroupCounts,
    build_base_tables: BaseTablesBuilder,
    registry_raw: bytes,
    clock: Callable[[], float] = time.time,
) -> None:
    """One job run over `registry_raw`, the registry file's bytes. Raises
    `CatalogueRefusedError` after logging a refused build. A skip, a discard
    or a lost publish race returns, since a later run publishes. A run that
    returns logs its heartbeat last (`_heartbeat`)."""
    _build(
        settings,
        bq=bq,
        graph=graph,
        un_graph=un_graph,
        un_group_counts=un_group_counts,
        build_base_tables=build_base_tables,
        registry_raw=registry_raw,
        clock=clock,
    )
    _heartbeat(settings, clock=clock)


def _build(
    settings: JobSettings,
    *,
    bq: BigQueryReader,
    graph: GraphClient,
    un_graph: UnGraph,
    un_group_counts: UnGroupCounts,
    build_base_tables: BaseTablesBuilder,
    registry_raw: bytes,
    clock: Callable[[], float],
) -> None:
    """`run` without the heartbeat."""
    connection = settings.bq_connection
    if lock_held(bq, connection=connection):
        emit("origin_catalogue_job_skipped", reason="ingestion_lock_held")
        _track_stale_under_lock(settings, clock=clock)
        return

    registry = parse_registry(registry_raw)
    registry_digest = registry_hash(registry_raw)
    fingerprint = read_ingestion_fingerprint(
        bq, connection=connection, observation_sql=observation_stats_sql(connection)
    )
    if fingerprint is None:
        raise CatalogueBuildError(
            "IngestionHistory has no row with IngestionFailure=false and Status='SUCCESS' "
            "to fingerprint from"
        )
    week = iso_week(datetime.fromtimestamp(clock(), UTC))
    enumeration_code = source_digest(ENUMERATION_MODULES)
    un_mirror_code = source_digest(UN_MIRROR_MODULES)
    inputs = identity_inputs(
        registry_digest=registry_digest,
        dfo_stamp=fingerprint.stamp,
        week=week,
        job_code=source_digest(JOB_MODULES),
        enumeration_code=enumeration_code,
        un_mirror_code=un_mirror_code,
    )
    identity = identity_key(inputs)

    root = format_root(settings.artifact_uri)
    published, published_generation = _read_published(root)
    published_identity = published.identity if published is not None else None
    _track_stale(settings, identity=identity, published=published_identity, clock=clock)
    retrying_un = False
    if published is not None and published.identity == identity:
        if not _published_carries_un(settings.artifact_uri, published.build_id):
            emit("origin_catalogue_job_noop", identity=identity)
            _prune(settings.artifact_uri, clock)
            return
        retrying_un = True

    build_id = new_build_id()
    lease = acquire_lease(root, build_id=build_id, ttl_seconds=_MAX_RUN_SECONDS, now=clock)
    if lease is None:
        emit("origin_catalogue_job_skipped", reason="lease_held", identity=identity)
        return
    started = time.monotonic()
    try:
        dfo = read_dfo_holdings(bq, connection=connection)
        stage = build_provenance_table(
            registry=registry, dfo_provenances=dfo.provenances, graph=graph
        )
        base_rows = stage.base_rows
        key = base_tables_key(
            registry_digest=registry_digest,
            base_provenances=base_rows,
            dfo_provenances=stage.dfo_ids,
            week=week,
            enumeration_code=enumeration_code,
        )
        base, carried_un = (
            _published_carry(settings.artifact_uri, key=key) if published else (None, None)
        )
        carried = base is not None
        if base is None:
            base = build_base_tables(
                graph=graph,
                registry=registry,
                admitted_base_provenances=base_rows,
                dfo_provenances=stage.dfo_ids,
                deadline=(
                    started
                    + _MAX_RUN_SECONDS
                    - _PUBLISH_RESERVE_SECONDS
                    - _UN_STAGE_RESERVE_SECONDS
                ),
            )
        refusals = list(stage.refusals)
        un = _un_stage_or_carried(
            registry,
            un_graph=un_graph,
            un_group_counts=un_group_counts,
            scan_places=base.scan_places,
            carried=carried_un,
            mirror_key=un_mirror_key(
                registry_digest=registry_digest, week=week, un_mirror_code=un_mirror_code
            ),
            refusals=refusals,
            deadline=started + _MAX_RUN_SECONDS - _PUBLISH_RESERVE_SECONDS,
        )
        refusals.extend(
            un_registry_refusals(registry, dfo_provenances=dfo.provenances, un_rows=un.rows)
        )
        refusals.extend(sources_without_variables(stage.by_source, base))
        successors = derive_successors(
            registry,
            withheld_variables=base.withheld_variables,
            dfo_variables=dfo.variables,
            mirror=un.mirror,
        )
        refusals.extend(successor_refusals(successors, un.mirror))
        if refusals:
            emit(
                "origin_catalogue_job_refused",
                identity=identity,
                build_id=build_id,
                refusals=[asdict(refusal) for refusal in refusals],
            )
            raise CatalogueRefusedError(refusals)
        if un.carried_forward is not None:
            emit(
                "origin_catalogue_job_un_retry_failed"
                if retrying_un
                else "origin_catalogue_job_un_carried_forward",
                severity="WARNING",
                identity=identity,
                build_id=build_id,
                from_build=un.carried_forward.from_build,
                reason=un.carried_forward.reason,
            )
            if retrying_un:
                # The published build already carries this UN stage.
                return

        if lock_held(bq, connection=connection):
            emit(
                "origin_catalogue_job_discarded",
                reason="ingestion_lock_acquired_during_build",
                identity=identity,
            )
            return
        current = read_ingestion_fingerprint(
            bq, connection=connection, observation_sql=observation_stats_sql(connection)
        )
        if current is None or current.stamp != fingerprint.stamp:
            emit(
                "origin_catalogue_job_discarded",
                reason="fingerprint_moved_during_build",
                identity=identity,
            )
            return

        tables = CatalogueTables(
            provenances={**stage.rows, **un.rows},
            dfo_variables=dfo.variables,
            base_variables=base.base_variables,
            provenance_places=base.provenance_places,
            withheld_variables=base.withheld_variables,
            topics=base.topics,
            scan_places=base.scan_places,
            scan_levels=base.scan_levels,
            dataset_facts=stage.dataset_facts,
            un_blocked_places=un.blocked_places,
            superseded=successors.rows,
        )
        build = encode_build(
            tables,
            build_id=build_id,
            built_at=datetime.fromtimestamp(clock(), UTC),
            build_seconds=time.monotonic() - started,
            identity_inputs=inputs,
            registry_hash=registry_digest,
            un_carried_forward=un.carried_forward,
            un_mirror=un.mirror,
        )
        if not publish_build(
            settings.artifact_uri, build, expected_generation=published_generation
        ):
            emit(
                "origin_catalogue_job_lost_publish_race",
                identity=identity,
                build_id=build_id,
                published_generation=published_generation,
            )
            return
        emit(
            "origin_catalogue_job_published",
            identity=identity,
            build_id=build_id,
            base_tables="carried" if carried else "built",
            row_counts=dict(build.manifest.row_counts),
            dfo_pairs=dfo.pairs,
            un_blocked_places=sorted(un.blocked_places),
            un_carried_from=(
                un.carried_forward.from_build if un.carried_forward is not None else None
            ),
            un_mirror=un.mirror_source,
            un_row_counts=dict(build.manifest.un_mirror.row_counts),
            un_resident_bytes=build.manifest.un_mirror.resident_bytes,
            successors={
                name: count
                for name, count in asdict(successors).items()
                if name not in {"rows", "ambiguous"}
            },
            overlaps=len(build.manifest.overlaps),
            resident_bytes=build.manifest.resident_bytes,
            build_seconds=round(build.manifest.build_seconds, 1),
        )
    finally:
        if not release_lease(root, lease):
            logger.warning("origin_catalogue: lease for %s was reclaimed before release", build_id)
    _prune(settings.artifact_uri, clock)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m origin_catalogue.job")
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="publish under this local directory in place of ORIGIN_CATALOGUE_URI",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    overrides = {"artifact_uri": args.local_dir.resolve().as_uri()} if args.local_dir else {}
    settings = JobSettings(**overrides)

    bq = LiveBigQuery(project=settings.bq_project)
    graph = GraphClient(
        base_url=settings.graph_base_url,
        token=settings.graph_token,
        timeout=settings.graph_timeout_seconds,
        user_agent=_USER_AGENT,
    )
    un_graph = GraphClient(
        base_url=settings.un_base_url,
        timeout=settings.graph_timeout_seconds,
        user_agent=_USER_AGENT,
        observation_get_url_budget_bytes=UN_OBSERVATION_GET_URL_BUDGET_BYTES,
    )
    un_site = UnSiteClient(
        site_url=settings.un_site_url,
        timeout=settings.graph_timeout_seconds,
        user_agent=_USER_AGENT,
    )
    with graph, un_graph, closing(un_site):
        try:
            run(
                settings,
                bq=bq,
                graph=graph,
                un_graph=un_graph,
                un_group_counts=un_site,
                build_base_tables=build_base_tables,
                registry_raw=packaged_registry().read_bytes(),
            )
        except (
            CatalogueBuildError,
            CataloguePointerReadError,
            CatalogueTableError,
            EnumerationError,
            GraphReadError,
            RegistryError,
        ) as exc:
            emit("origin_catalogue_job_failed", error=str(exc))
            logger.error("origin catalogue job failed: %s", exc)
            return 1
        except CatalogueRefusedError as exc:
            logger.error("origin catalogue job %s", exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ENUMERATION_MODULES",
    "JOB_MODULES",
    "UN_MIRROR_MODULES",
    "BaseTablesBuilder",
    "CatalogueBuildError",
    "CatalogueGraph",
    "CataloguePointerReadError",
    "CatalogueRefusedError",
    "DfoHoldings",
    "JobSettings",
    "ProvenanceStage",
    "Refusal",
    "RefusalKind",
    "UnGraph",
    "UnReadError",
    "UnStage",
    "base_tables_key",
    "build_provenance_table",
    "identity_inputs",
    "iso_week",
    "main",
    "probe_un_blocked_places",
    "prune_old_versions",
    "read_dfo_holdings",
    "read_un_stage",
    "run",
    "source_digest",
    "sources_without_variables",
    "successor_refusals",
    "un_mirror_key",
    "un_registry_refusals",
]
