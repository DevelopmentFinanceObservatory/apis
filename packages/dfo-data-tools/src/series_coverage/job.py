"""Builds and publishes the coverage artifact served by Search and Datasets.

Artifacts live under the image's format root. `current.json` selects an
immutable `versions/<build_id>/` directory. Each image reads and prunes its
own format root. BigQuery queries send root-partitionable projections to
Spanner through `EXTERNAL_QUERY` and aggregate in BigQuery.

The job refuses to publish while `IngestionLock` is held, if the source
fingerprint changes during a build, if a date is invalid or more than 50
years ahead, or if SDMX reconciliation differs. It writes build files
create-only and conditionally replaces `current.json` using its version
token. Concurrent runs can leave an unreferenced build, which pruning
removes after `_MAX_RUN_SECONDS` (1800 s). A zero exit also covers skips and
discarded builds, so deploys run `python -m series_coverage.verify` to load
the artifact with the serving loader before moving services to the image.

The build covers every variable whose Spanner `Edge` rows declare one or two
`observationProperties` roles. A two-role observation stores its entities in
`entity1` and `extra_entities_id`, which become the `donor` and `recipient`
columns, and `resolve_role_names` matches SDMX role values against them to
name each column's role. A one-role observation stores its entity in
`entity1` with an empty `extra_entities_id`. Its row puts that entity in the
column of its role's flow axis, `donor` for an origin-axis role and
`recipient` for a destination-axis one, and `ABSENT_ENTITY` in the other. The
build refuses a row whose `extra_entities_id` disagrees with its variable's
role count, a one-role variable whose role is on neither axis, and a row
outside that population that still carries a role signal (a set
`extra_entities_id`, or three or more declared roles).

After each publish the job loads the new build with the serving loader and
checks it against the code index its image packages
(`series_coverage.compatibility`). A compatible build moves that index's
compatibility pointer, `compatible/<content_hash>.json`; an incompatible one
emits `coverage_index_incompatible`, since services running that index keep
serving the pointer's build until the index is re-exported. Prune keeps every
build a compatibility pointer names. It deletes the pointer of an index no
serving revision runs (`_expired_pointers`) once that pointer is
`_COMPATIBLE_POINTER_RETENTION_SECONDS` (seven days) old, and a build only that
pointer named goes on a later run.

Reconciliation compares per-variable row counts with SDMX `/data` for a
sample of variables, which bounds the rows the SDMX service streams per
build. The sample takes the smallest variable of each donor/recipient role
group, one-role groups included, then adds variables in
`sha256(stamp + variable)` order while their expected rows fit
`COVERAGE_RECONCILE_MAX_ROWS` (default 250,000). One data version always
gets the same sample, and a new version gets a new one. A group's smallest
variable can exceed that cap on its own, so each group's sampled variables
are requested in chunks that expect at most `_RECONCILE_ROW_BUDGET`
(250,000) rows each and fit the SDMX client's URL budget. The mixer times
out on, or truncates, a single `/data` response of a few million rows.

Facet metadata comes from `TimeSeries`, keyed by `facet_id`. Each facet id
of the role-bearing population has one provenance, unit, measurement method,
and observation period tuple, and the build refuses a facet id with more than
one. A variable can carry several facets, such as a recorded series and an
`Estimated` projection of the same measure. The query does not select
`scalingFactor`, so the job writes `None` for it.

Environment: `COVERAGE_ARTIFACT_URI` and `COVERAGE_BQ_CONNECTION` are
required. `DATASETS_SDMX_BASE_URL` is the mixer's API root, including
`/core/api`. `DATASETS_GRAPH_TOKEN` is optional. `COVERAGE_BQ_PROJECT`
defaults to the connection's project. `COVERAGE_SDMX_TIMEOUT_SECONDS`
defaults to 90 and `COVERAGE_CYCLE_SECONDS` to 900. The stale-publish
threshold is one cycle. `DATASETS_GRAPH_BASE_URL` is unused.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, Self, cast

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from artifact_publication.jobs import (
    AliasOnlySettings,
    BigQueryReader,
    Fingerprint,
    Ingestion,
    LiveBigQuery,
    emit,
    lock_held,
    new_build_id,
    prune_versions,
    read_ingestion_fingerprint,
    unpublished_seconds,
)
from artifact_publication.uri import (
    delete_uri_if_unchanged,
    join_uri,
    list_uri_object_times,
    read_uri_bytes_and_generation,
    write_uri_bytes_if_unchanged,
)
from code_index.query import CodeIndex
from code_index.shared import get_index
from dc_sdmx import plan_entity_chunks, plan_request_chunks
from dc_sdmx.client import SdmxClient
from dc_sdmx.errors import SdmxClientError, SdmxQueryError
from dc_sdmx.results import VARIABLE_COLUMN as SDMX_VARIABLE_COLUMN
from dc_sdmx.results import SdmxDataResult
from dfo_platform.roles import ROLE_AXES
from materialise.roles import axis_for_role
from series_coverage._bits import popcount64
from series_coverage.compatibility import read_serving_index, record_if_compatible
from series_coverage.io import (
    write_coverage_table,
    write_facet_metadata,
    write_manifest,
    write_role_names,
)
from series_coverage.loader import CoverageArtifactError, format_root, load_coverage
from series_coverage.schema import (
    ABSENT_ENTITY,
    BUILD_EPOCH_YEAR,
    COMPATIBLE_PREFIX,
    DONOR_COLUMN,
    FACET_COLUMN,
    FIRST_YEAR_COLUMN,
    LAST_YEAR_COLUMN,
    MASK_WORD_BITS,
    N_OBS_COLUMN,
    POINTER_FILE,
    RECIPIENT_COLUMN,
    VARIABLE_COLUMN,
    VERSIONS_PREFIX,
    CoverageManifest,
    CoveragePointer,
    FacetMetadata,
    RoleNames,
    YearHorizon,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "coverage_state.json"
_DEFAULT_CYCLE_SECONDS = 900.0  # Cloud Scheduler's own 15-minute cadence.
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF

_MAX_RUN_SECONDS: Final[float] = 1800.0
"""The coverage Cloud Run Job's own maximum runtime (dfo-infra's
`coverage.tf` sets the job template's `timeout` to `"1800s"`), and the
grace period `prune_old_versions` applies. A build directory whose newest
file is younger than this could still be another invocation's in-progress
`publish()`, which Cloud Run kills by this age. A pointer written less than
this long ago may have named another build moments earlier, and a reader
that resolved that older pointer (a load takes seconds) could still be
reading it."""

_COMPATIBLE_POINTER_RETENTION_SECONDS: Final[float] = 7 * 86_400.0
"""How long prune keeps the compatibility pointer of an index no serving
revision runs after it was last written, and with it the build it names. The
window covers a rollback to that index's revision in the Cloud Run console."""


_OBSERVATION_PROPERTIES_INNER = (
    "SELECT subject_id, object_id FROM Edge WHERE predicate = 'observationProperties'"
)


_ORIGIN_ROLES_SQL = ", ".join(
    f"'{role}'" for role, axis in sorted(ROLE_AXES.items()) if axis == "origin"
)
"""The role names `dfo_platform.roles.ROLE_AXES` places on the origin axis, as
a SQL list."""


def _with_role_bearing(connection: str) -> str:
    """A `WITH` clause defining `declared_roles(variable, n_roles, lone_origin)`,
    every variable whose `Edge` rows declare any `observationProperties` role,
    and `role_bearing(variable, n_roles, lone_origin)`, the ones declaring one
    or two. `lone_origin` holds for a variable whose one role is on the origin
    axis. Queries join Spanner projections to them in BigQuery, so each
    `EXTERNAL_QUERY` stays root-partitionable."""
    return (
        "WITH declared_roles AS (\n"
        "  SELECT subject_id AS variable, COUNT(DISTINCT object_id) AS n_roles,\n"
        "    COUNT(DISTINCT object_id) = 1\n"
        f"      AND LOGICAL_AND(object_id IN ({_ORIGIN_ROLES_SQL})) AS lone_origin\n"
        f'  FROM EXTERNAL_QUERY("{connection}", "{_OBSERVATION_PROPERTIES_INNER}")\n'
        "  GROUP BY subject_id\n"
        "),\n"
        "role_bearing AS (\n"
        "  SELECT variable, n_roles, lone_origin\n"
        "  FROM declared_roles WHERE n_roles BETWEEN 1 AND 2\n"
        ")\n"
    )


_IS_ROLE_BEARING = "variable_measured IN (SELECT variable FROM role_bearing)"


class CoverageFingerprintError(Exception):
    """`IngestionHistory` has no successful row to fingerprint from."""


class CoverageBuildError(Exception):
    """The BigQuery build query found data this schema can't represent, or
    an invariant the published artifact must hold (`n_obs` equal to the
    year-mask set-bit count, at most one observation per series-year, an
    `extra_entities_id` that is empty exactly for a one-role variable) is
    violated. The job refuses to publish.
    """


class CoverageRoleNameError(Exception):
    """Raised when donor/recipient role names are missing or ambiguous, or a
    one-role variable's role is not on the destination axis."""


class CoverageFacetMetadataError(Exception):
    """Raised for conflicting metadata or a coverage facet absent from `TimeSeries`."""


class CoverageReconciliationError(Exception):
    """The built coverage table's per-variable row counts disagree with
    SDMX `/data`, for at least one role-bearing StatVar. The job refuses to
    publish.
    """


class CoveragePointerReadError(Exception):
    """Raised for pointer read failures such as network, permission, or quota errors.

    Publication needs a reliable generation token. An absent pointer is a
    valid first-build state.
    """


# --- Settings ----------------------------------------------------------------


class JobSettings(AliasOnlySettings):
    """Loads job settings from the environment and fails fast at
    construction. An unparseable, missing or empty required value raises
    `pydantic.ValidationError` naming the field. `artifact_uri`,
    `bq_connection` and `sdmx_base_url` have no safe default. `bq_project`
    defaults to `bq_connection`'s leading segment when unset or empty.

    Each field's `alias` is the only environment variable name this class
    reads. A caller can still construct one by field name
    (`JobSettings(artifact_uri=..., ...)`, as every test in this package
    does).
    """

    model_config = SettingsConfigDict(frozen=True)

    artifact_uri: str = Field(min_length=1, alias="COVERAGE_ARTIFACT_URI")
    bq_connection: str = Field(min_length=1, alias="COVERAGE_BQ_CONNECTION")
    bq_project: str = Field(default="", alias="COVERAGE_BQ_PROJECT")
    sdmx_base_url: str = Field(min_length=1, alias="DATASETS_SDMX_BASE_URL")
    sdmx_token: str | None = Field(default=None, alias="DATASETS_GRAPH_TOKEN")
    sdmx_timeout_seconds: float = Field(default=90.0, alias="COVERAGE_SDMX_TIMEOUT_SECONDS")
    cycle_seconds: float = Field(default=_DEFAULT_CYCLE_SECONDS, alias="COVERAGE_CYCLE_SECONDS")
    reconcile_max_rows: int = Field(default=250_000, gt=0, alias="COVERAGE_RECONCILE_MAX_ROWS")

    @field_validator("sdmx_token", mode="before")
    @classmethod
    def _empty_token_to_none(cls, value: object) -> object:
        return value or None

    @model_validator(mode="after")
    def _default_bq_project(self) -> Self:
        if not self.bq_project:
            object.__setattr__(self, "bq_project", self.bq_connection.split(".")[0])
        return self

    @classmethod
    def from_env(cls) -> JobSettings:
        return cls()


# --- SDMX ----------------------------------------------------------------


class CoverageSdmxClient(Protocol):
    """The SDMX methods used for role resolution and reconciliation.

    Reconciliation sets `roles=` to keep one request per chunk, without the
    stream reader's header pre-scan, and plans each request's query string
    within `url_budget_bytes`, the client's own limit.
    """

    @property
    def url_budget_bytes(self) -> int: ...

    def data(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, Sequence[str]],
        facet_ids: Sequence[str] | None = None,
    ) -> SdmxDataResult: ...

    def stream_data_batches(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, Sequence[str]],
        roles: Sequence[str] | None = None,
        time_period: Any = None,
    ) -> Any: ...


# --- Fingerprint -----------------------------------------------------------


def read_fingerprint(bq: BigQueryReader, *, connection: str) -> Fingerprint:
    """Reads the latest successful ingestion and role-bearing row counts.
    `stamp` is the data version recorded in the manifest and `current.json`.
    Missing successful history raises `CoverageFingerprintError`.
    """
    observation_sql = (
        _with_role_bearing(connection)
        + "SELECT COUNT(*) AS row_count, MAX(last_update_timestamp) AS max_ts\n"
        f'FROM EXTERNAL_QUERY("{connection}", '
        '"SELECT variable_measured, last_update_timestamp FROM Observation")\n'
        f"WHERE {_IS_ROLE_BEARING}"
    )
    fingerprint = read_ingestion_fingerprint(
        bq, connection=connection, observation_sql=observation_sql
    )
    if fingerprint is None:
        raise CoverageFingerprintError(
            "IngestionHistory has no row with IngestionFailure=false and Status='SUCCESS' "
            "to fingerprint from"
        )
    return fingerprint


def _read_published(root: str) -> tuple[CoveragePointer | None, str | None]:
    """Reads the pointer and version token together for the no-op check and
    conditional publish. Missing pointers return `(None, None)`.
    Malformed ones return `(None, generation)` so publish can replace that version.
    Other read failures raise `CoveragePointerReadError`.
    """
    uri = join_uri(root, POINTER_FILE)
    try:
        found = read_uri_bytes_and_generation(uri)
    except Exception as exc:
        raise CoveragePointerReadError(f"couldn't read {uri!r}: {exc}") from exc
    if found is None:
        return None, None
    raw, generation = found
    try:
        return CoveragePointer.from_json(raw), generation
    except ValueError:
        logger.warning("series_coverage: current.json at %r is malformed; rebuilding", uri)
        return None, generation


# --- Build -------------------------------------------------------------------

_BUILD_INNER_SQL = (
    "SELECT variable_measured, entity1, extra_entities_id, facet_id, date FROM Observation"
)


_YEAR_SQL = "SAFE_CAST(date AS INT64)"


@dataclass(frozen=True, slots=True)
class CoverageBuild:
    """`build_coverage_table`'s result: the table and the `YearHorizon`
    its year-mask columns encode, which `publish` records in the
    manifest."""

    table: pa.Table
    horizon: YearHorizon


_OUTSIDE_EXAMPLES: Final[int] = 10
"""How many excluded variables a role-population refusal names."""


def _year_horizon(bq: BigQueryReader, *, connection: str) -> YearHorizon:
    """Builds the horizon from `BUILD_EPOCH_YEAR` through the latest row year.

    `SAFE_CAST` lets the query count malformed dates. Unparseable or pre-epoch
    dates raise before the build query. Years more than 50 years ahead also
    raise. A bad year such as 9999 would otherwise require 126 64-bit mask
    words per row.

    The same scan refuses an Observation row that carries a role signal
    outside the one- and two-role population: a set `extra_entities_id` on a
    variable that declares no role, or any row of a variable declaring three
    or more. Every other query filters to `role_bearing` and would drop such
    rows without a trace.
    """
    from datetime import UTC, datetime

    sql = (
        _with_role_bearing(connection) + "SELECT\n"
        "  COUNTIF(NOT in_population) AS n_outside_rows,\n"
        "  ARRAY_AGG(DISTINCT IF(in_population, NULL, variable_measured) IGNORE NULLS "
        f"LIMIT {_OUTSIDE_EXAMPLES}) AS outside_examples,\n"
        "  COUNTIF(in_population AND year IS NULL) AS n_unparsed,\n"
        f"  COUNTIF(in_population AND year < {BUILD_EPOCH_YEAR}) AS n_early,\n"
        "  MAX(IF(in_population, year, NULL)) AS max_year\n"
        "FROM (\n"
        f"  SELECT o.variable_measured, {_YEAR_SQL} AS year,\n"
        "    IFNULL(d.n_roles BETWEEN 1 AND 2, FALSE) AS in_population\n"
        f'  FROM EXTERNAL_QUERY("{connection}", '
        '"SELECT variable_measured, extra_entities_id, date FROM Observation") AS o\n'
        "  LEFT JOIN declared_roles AS d ON o.variable_measured = d.variable\n"
        "  WHERE d.n_roles IS NOT NULL OR IFNULL(o.extra_entities_id, '') != ''\n"
        ")"
    )
    [row] = bq.query(sql)
    outside = int(row["n_outside_rows"])
    if outside:
        raise CoverageBuildError(
            f"{outside} Observation row(s) carry a role signal but their variable declares "
            "neither one nor two observationProperties roles (a set extra_entities_id with "
            "no declared role, or three or more declared roles), e.g. "
            f"{list(row['outside_examples'] or ())!r}; refusing to build"
        )
    unparsed = int(row["n_unparsed"])
    if unparsed:
        raise CoverageBuildError(
            f"{unparsed} role-bearing Observation row(s) have a `date` that doesn't parse "
            "as a year; refusing to build"
        )
    early = int(row["n_early"])
    if early:
        raise CoverageBuildError(
            f"{early} role-bearing Observation row(s) have a `date` year before "
            f"{BUILD_EPOCH_YEAR}; refusing to build"
        )
    max_year = row["max_year"]
    last_year = BUILD_EPOCH_YEAR if max_year is None else int(max_year)
    max_allowed_year = datetime.now(UTC).year + 50
    if last_year > max_allowed_year:
        raise CoverageBuildError(
            f"a role-bearing Observation row has date year {last_year}, more than 50 years "
            f"past today (ceiling {max_allowed_year}); refusing to build"
        )
    return YearHorizon.covering(epoch_year=BUILD_EPOCH_YEAR, last_year=last_year)


def build_coverage_table(bq: BigQueryReader, *, connection: str) -> CoverageBuild:
    """Builds one row per `(variable, facet, donor, recipient)` in BigQuery.

    Spanner receives a root-partitionable projection through
    `EXTERNAL_QUERY`, which BigQuery joins to `role_bearing`. A two-role
    row takes `entity1` as donor and `extra_entities_id` as recipient. A
    one-role row takes `entity1` as donor when its role is on the origin axis
    (`lone_origin`), else as recipient, and `ABSENT_ENTITY` in the other
    column. `resolve_role_names` names the same column from the same
    `ROLE_AXES`, and refuses a lone role on neither axis.
    BigQuery computes counts, bounds, and 64-year mask words. Since BigQuery
    `INT64` is signed, each word is masked to uint64 before writing. Dates
    outside the horizon, duplicate observations in a year, an
    `extra_entities_id` whose emptiness disagrees with the role count, and
    zero result rows raise `CoverageBuildError`.

    Rows sort by `(variable, facet_id)` because `GROUP BY` has no order
    guarantee. The index uses that order to skip a full-table sort on load.
    """
    horizon = _year_horizon(bq, connection=connection)
    word_columns = []
    for word, name in enumerate(horizon.mask_columns):
        lo = horizon.epoch_year + MASK_WORD_BITS * word
        hi = lo + MASK_WORD_BITS - 1
        word_columns.append(
            f"  BIT_OR(IF({_YEAR_SQL} BETWEEN {lo} AND {hi}, "
            f"CAST(1 AS INT64) << ({_YEAR_SQL} - {lo}), 0)) AS {name}"
        )
    sql = (
        _with_role_bearing(connection) + "SELECT\n"
        "  o.variable_measured AS variable,\n"
        "  o.facet_id AS facet_id,\n"
        f"  IF(r.n_roles = 1 AND NOT r.lone_origin, '{ABSENT_ENTITY}', o.entity1) AS donor,\n"
        "  CASE WHEN r.n_roles = 2 THEN o.extra_entities_id\n"
        f"    WHEN r.lone_origin THEN '{ABSENT_ENTITY}' ELSE o.entity1 END AS recipient,\n"
        "  LOGICAL_AND((IFNULL(o.extra_entities_id, '') = '') = (r.n_roles = 1))"
        " AS entities_match_roles,\n"
        "  COUNT(*) AS n_obs,\n"
        f"  MIN({_YEAR_SQL}) AS first_year,\n"
        f"  MAX({_YEAR_SQL}) AS last_year,\n" + ",\n".join(word_columns) + "\n"
        f'FROM EXTERNAL_QUERY("{connection}", "{_BUILD_INNER_SQL}") AS o\n'
        "JOIN role_bearing AS r ON o.variable_measured = r.variable\n"
        "GROUP BY variable, facet_id, donor, recipient"
    )
    rows = bq.query(sql)
    if not rows:
        raise CoverageBuildError("the BigQuery build query returned zero role-bearing series")
    mismatched = sorted({str(r["variable"]) for r in rows if not r["entities_match_roles"]})
    if mismatched:
        raise CoverageBuildError(
            f"{len(mismatched)} variable(s) have Observation rows whose extra_entities_id is "
            "empty for a two-role variable or set for a one-role variable, e.g. "
            f"{mismatched[:10]!r}; refusing to build"
        )
    # `GROUP BY` leaves row order unspecified. Sorting by `(variable, facet_id)`
    # lets `CoverageIndex.__init__` combine chunks without a full-table sort.
    rows = sorted(rows, key=lambda r: (str(r["variable"]), str(r["facet_id"])))

    table = pa.table(
        {
            VARIABLE_COLUMN: [str(r["variable"]) for r in rows],
            FACET_COLUMN: [str(r["facet_id"]) for r in rows],
            DONOR_COLUMN: [str(r["donor"]) for r in rows],
            RECIPIENT_COLUMN: [str(r["recipient"]) for r in rows],
            N_OBS_COLUMN: pa.array([int(r["n_obs"]) for r in rows], type=pa.uint32()),
            FIRST_YEAR_COLUMN: pa.array([int(r["first_year"]) for r in rows], type=pa.uint16()),
            LAST_YEAR_COLUMN: pa.array([int(r["last_year"]) for r in rows], type=pa.uint16()),
            **{
                name: pa.array([int(r[name]) & _UINT64_MASK for r in rows], type=pa.uint64())
                for name in horizon.mask_columns
            },
        },
        schema=horizon.schema,
    )
    _check_popcount_invariant(table, horizon)
    return CoverageBuild(table=table, horizon=horizon)


def _check_popcount_invariant(table: pa.Table, horizon: YearHorizon) -> None:
    import numpy as np

    n_obs = np.array(table[N_OBS_COLUMN].to_pylist(), dtype=np.uint64)
    bits = np.zeros(table.num_rows, dtype=np.uint64)
    for name in horizon.mask_columns:
        bits += popcount64(np.array(table[name].to_pylist(), dtype=np.uint64))
    mismatched = n_obs != bits
    count = int(mismatched.sum())
    if count:
        idx = int(np.flatnonzero(mismatched)[0])
        bad_variable = cast("list[str]", table[VARIABLE_COLUMN].to_pylist())[idx]
        raise CoverageBuildError(
            f"{count} built row(s) have n_obs != the year-mask bit count (e.g. variable "
            f"{bad_variable!r}) -- more than one observation in a year for that series; "
            "refusing to build"
        )


# --- Role names --------------------------------------------------------------


def _read_role_property_sets(bq: BigQueryReader, *, connection: str) -> dict[str, tuple[str, ...]]:
    """Maps each variable to its alphabetically sorted declared role names.

    Spanner's `Edge` rows store an unordered role set.
    `_resolve_donor_recipient` maps names to donor and recipient slots.
    """
    sql = (
        "SELECT subject_id AS variable, object_id AS role\n"
        f'FROM EXTERNAL_QUERY("{connection}", "{_OBSERVATION_PROPERTIES_INNER}")'
    )
    rows = bq.query(sql)
    by_variable: dict[str, set[str]] = {}
    for row in rows:
        by_variable.setdefault(str(row["variable"]), set()).add(str(row["role"]))
    return {variable: tuple(sorted(roles)) for variable, roles in by_variable.items()}


def _first_row_for_variable(table: pa.Table, variable: str) -> tuple[str, str, str] | None:
    mask = pc.equal(table[VARIABLE_COLUMN], pa.scalar(variable, type=pa.string()))
    rows = table.filter(mask)
    if rows.num_rows == 0:
        return None
    return (
        cast("list[str]", rows[DONOR_COLUMN].to_pylist())[0],
        cast("list[str]", rows[RECIPIENT_COLUMN].to_pylist())[0],
        cast("list[str]", rows[FACET_COLUMN].to_pylist())[0],
    )


def _resolve_donor_recipient(
    sdmx: CoverageSdmxClient,
    *,
    table: pa.Table,
    props: tuple[str, ...],
    group_variables: Sequence[str],
) -> tuple[str, str]:
    """Maps the two role names to `donor` and `recipient` slots.

    Tries both assignments against each candidate variable's known Spanner
    dcids, matching returned `role_values`. The first SDMX `/data` result
    confirms the mapping. If no candidate succeeds, raises
    `CoverageRoleNameError`.
    """
    role_a, role_b = props
    for variable in group_variables:
        row = _first_row_for_variable(table, variable)
        if row is None:
            continue
        donor_dcid, recipient_dcid, facet_id = row
        for guess_donor, guess_recipient in ((role_a, role_b), (role_b, role_a)):
            try:
                result = sdmx.data(
                    variables=[variable],
                    entities={guess_donor: [donor_dcid], guess_recipient: [recipient_dcid]},
                    facet_ids=[facet_id],
                )
            except SdmxClientError:
                continue
            if result.rows:
                return guess_donor, guess_recipient
    raise CoverageRoleNameError(
        f"could not determine donor/recipient assignment for role-name pair {props!r} "
        f"(tried {len(group_variables)} candidate variable(s) against live SDMX)"
    )


def _lone_role_names(role: str, *, example: str) -> RoleNames:
    """The `RoleNames` of a one-role variable: its role on the column of the
    role's flow axis, `donor` for origin and `recipient` for destination, where
    the build puts its entity. Refuses a role on neither axis, which no column
    holds."""
    match axis_for_role(role):
        case "origin":
            return RoleNames(donor=role, recipient=None)
        case "destination":
            return RoleNames(donor=None, recipient=role)
        case None:
            raise CoverageRoleNameError(
                f"variable {example!r} declares the single role {role!r}, which is on no "
                "flow axis; a one-role coverage row stores its entity on its role's axis"
            )


def resolve_role_names(
    bq: BigQueryReader, sdmx: CoverageSdmxClient, *, connection: str, table: pa.Table
) -> dict[str, RoleNames]:
    """Resolves role names for each variable, grouping by its declared role set.

    Each two-role group gets one donor/recipient probe. A one-role group
    needs none: its role's flow axis names its column (`_lone_role_names`).
    Missing role declarations, a count other than one or two, or a one-role
    variable whose role is on no flow axis raises `CoverageRoleNameError`.
    """
    variables = sorted(set(cast("list[str]", table[VARIABLE_COLUMN].to_pylist())))
    property_sets = _read_role_property_sets(bq, connection=connection)
    groups: dict[tuple[str, ...], list[str]] = {}
    for variable in variables:
        props = property_sets.get(variable)
        if props is None or len(props) not in (1, 2):
            raise CoverageRoleNameError(
                f"variable {variable!r} has coverage rows but its Spanner Edge "
                f"observationProperties are {props!r}, not one or two role names"
            )
        groups.setdefault(props, []).append(variable)

    role_names: dict[str, RoleNames] = {}
    for props, group_variables in groups.items():
        if len(props) == 1:
            names = _lone_role_names(props[0], example=group_variables[0])
        else:
            donor_role, recipient_role = _resolve_donor_recipient(
                sdmx, table=table, props=props, group_variables=group_variables
            )
            names = RoleNames(donor=donor_role, recipient=recipient_role)
        for variable in group_variables:
            role_names[variable] = names
    return role_names


# --- Facet metadata ------------------------------------------------------

_FACET_METADATA_INNER_SQL = (
    "SELECT facet_id, provenance, unit, measurement_method, observation_period, "
    "variable_measured FROM TimeSeries"
)


def resolve_facet_metadata(
    bq: BigQueryReader, *, connection: str, table: pa.Table
) -> dict[str, FacetMetadata]:
    """Reads metadata for every coverage facet from Spanner `TimeSeries`.

    BigQuery performs grouping outside the root-partitionable
    `EXTERNAL_QUERY`. `scaling_factor` is always `None`. Raises
    `CoverageFacetMetadataError` for conflicting metadata or a coverage
    facet missing from `TimeSeries`.
    """
    sql = (
        _with_role_bearing(connection)
        + "SELECT facet_id, provenance, unit, measurement_method, observation_period, "
        "COUNT(*) AS n\n"
        f'FROM EXTERNAL_QUERY("{connection}", "{_FACET_METADATA_INNER_SQL}")\n'
        f"WHERE {_IS_ROLE_BEARING}\n"
        "GROUP BY facet_id, provenance, unit, measurement_method, observation_period"
    )
    rows = bq.query(sql)
    metadata: dict[str, FacetMetadata] = {}
    combos_by_facet: dict[str, set[tuple[str | None, str | None, str | None, str | None]]] = {}
    for row in rows:
        facet_id = str(row["facet_id"])
        combo = (
            row["provenance"],
            row["unit"],
            row["measurement_method"],
            row["observation_period"],
        )
        combos_by_facet.setdefault(facet_id, set()).add(combo)
        metadata[facet_id] = FacetMetadata(
            provenance_id=combo[0],
            measurement_method=combo[2],
            unit=combo[1],
            scaling_factor=None,
            observation_period=combo[3],
        )
    inconsistent = {
        facet_id: combos for facet_id, combos in combos_by_facet.items() if len(combos) > 1
    }
    if inconsistent:
        sample_facet_id, sample_combos = next(iter(inconsistent.items()))
        raise CoverageFacetMetadataError(
            f"{len(inconsistent)} facet_id(s) map to more than one (provenance, unit, "
            f"measurement_method, observation_period) tuple on TimeSeries, e.g. facet_id {sample_facet_id!r}: "
            f"{sorted(sample_combos, key=repr)!r}"
        )
    table_facet_ids = set(cast("list[str]", table[FACET_COLUMN].to_pylist()))
    missing = sorted(table_facet_ids - metadata.keys())
    if missing:
        raise CoverageFacetMetadataError(
            f"{len(missing)} facet_id(s) in the built coverage table have no TimeSeries row "
            f"at all, e.g. {missing[:10]!r}"
        )
    return metadata


# --- Reconciliation ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    duration_seconds: float
    request_count: int
    variables_checked: int
    rows_expected: int


def _per_variable_n_obs(table: pa.Table) -> dict[str, int]:
    # A tuple preserves each aggregate function's Literal type for pyarrow-stubs.
    grouped = table.group_by([VARIABLE_COLUMN]).aggregate(((N_OBS_COLUMN, "sum"),))
    variables = cast("list[str]", grouped[VARIABLE_COLUMN].to_pylist())
    sums = cast("list[int]", grouped[f"{N_OBS_COLUMN}_sum"].to_pylist())
    return dict(zip(variables, sums, strict=True))


_RECONCILE_ROW_BUDGET: Final[int] = 250_000
"""The most coverage rows one reconciliation request expects. On dev, requests
of up to 351K rows (the largest one-role chunk of the International Debt
Statistics load) read in about 7 s, and a 166K-row CRS request read in under
6 s. Requests planned on the URL budget alone reached 1.9M rows, which timed
out, and 2.66M rows, which the mixer answered with a 503 or with a body cut
after 267K rows."""


@dataclass(frozen=True, slots=True)
class _ReconcileRequest:
    variables: tuple[str, ...]
    entities: dict[str, tuple[str, ...]]


def _plan_group_requests(
    table: pa.Table,
    *,
    names: RoleNames,
    variables: Sequence[str],
    expected_counts: Mapping[str, int],
    row_budget: int,
    url_budget_bytes: int,
) -> list[_ReconcileRequest]:
    """Plans one role-name group's requests so each expects at most
    `row_budget` rows and has a query string within `url_budget_bytes`.

    Variables within the row budget share one filter: every covered dcid
    of the group's smaller role axis (its one axis for a one-role group),
    leaving the other axis open, so a row on a dcid another
    sampled variable covers still counts. `_plan_shared_filter` splits that
    filter into runs within the full URL budget. A dcid that does not fit
    beside the longest variable there raises `CoverageReconciliationError`.
    A plan with runs of half the URL budget replaces it when it needs fewer
    requests, since a filter that nearly fills the URL leaves room for only
    one or two variables per request.

    A variable over the row budget gets requests of its own. Their filters
    split every covered dcid of the group on one axis into consecutive runs
    within the row and URL budgets, so a row on a dcid another sampled
    variable covers still counts. The axis is whichever gives the variable
    the smallest per-dcid count, and a dcid the variable does not cover
    weighs nothing against the row budget. A dcid whose rows alone pass the
    row budget raises `CoverageReconciliationError`.
    """
    selected = table.filter(
        pc.is_in(table[VARIABLE_COLUMN], value_set=pa.array(variables, type=pa.string()))
    )
    axes = [
        (role, column)
        for role, column in ((names.donor, DONOR_COLUMN), (names.recipient, RECIPIENT_COLUMN))
        if role is not None
    ]
    axis_dcids = {
        column: tuple(sorted(set(cast("list[str]", selected[column].to_pylist()))))
        for _role, column in axes
    }
    # counts[column][variable][dcid] is the variable's rows on that dcid.
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for _role, axis_column in axes:
        grouped = selected.group_by([VARIABLE_COLUMN, axis_column]).aggregate(
            ((N_OBS_COLUMN, "sum"),)
        )
        by_variable = counts.setdefault(axis_column, {})
        for variable, dcid, n_obs in zip(
            cast("list[str]", grouped[VARIABLE_COLUMN].to_pylist()),
            cast("list[str]", grouped[axis_column].to_pylist()),
            cast("list[int]", grouped[f"{N_OBS_COLUMN}_sum"].to_pylist()),
            strict=True,
        ):
            by_variable.setdefault(variable, {})[dcid] = n_obs

    requests: list[_ReconcileRequest] = []
    within = [v for v in variables if expected_counts[v] <= row_budget]
    if within:
        role, column = min(axes, key=lambda axis: len(axis_dcids[axis[1]]))
        shared = functools.partial(
            _plan_shared_filter,
            within,
            role=role,
            dcids=axis_dcids[column],
            counts=counts[column],
            row_budget=row_budget,
            url_budget_bytes=url_budget_bytes,
        )
        try:
            plan = shared(entity_budget=url_budget_bytes)
        except SdmxQueryError as exc:
            raise CoverageReconciliationError(
                f"the shared {role!r} filter of {len(within)} variable(s) does not split "
                f"into reconciliation requests: {exc}"
            ) from exc
        # Only a candidate for fewer requests: a dcid that fits beside the
        # longest variable in the full budget can still pass half of it.
        with contextlib.suppress(SdmxQueryError):
            half = shared(entity_budget=url_budget_bytes // 2)
            if len(half) < len(plan):
                plan = half
        requests.extend(plan)

    for variable in (v for v in variables if expected_counts[v] > row_budget):
        split_role, split_column = min(
            axes, key=lambda axis: max(counts[axis[1]][variable].values())
        )
        own_counts = counts[split_column][variable]
        try:
            own_runs = plan_entity_chunks(
                axis_dcids[split_column],
                role=split_role,
                variables=(variable,),
                url_budget_bytes=url_budget_bytes,
                row_weights={d: own_counts.get(d, 0) for d in axis_dcids[split_column]},
                row_budget=row_budget,
            )
        except SdmxQueryError as exc:
            raise CoverageReconciliationError(
                f"variable {variable!r} does not split into reconciliation requests: {exc}"
            ) from exc
        requests.extend(
            _ReconcileRequest(variables=(variable,), entities={split_role: run}) for run in own_runs
        )
    return requests


def _plan_shared_filter(
    variables: Sequence[str],
    *,
    role: str,
    dcids: Sequence[str],
    counts: Mapping[str, Mapping[str, int]],
    row_budget: int,
    url_budget_bytes: int,
    entity_budget: int,
) -> list[_ReconcileRequest]:
    """Plans requests that read every one of `variables` on every dcid of
    one role's filter.

    `plan_entity_chunks` splits `dcids` into runs that fit `entity_budget`
    beside the longest variable. Each run then packs `variables` within
    `url_budget_bytes` and on their rows within the run, taken from
    `counts[variable][dcid]`, so a variable's per-run counts sum to its total.
    """
    requests: list[_ReconcileRequest] = []
    for run in plan_entity_chunks(
        dcids, role=role, variables=variables, url_budget_bytes=entity_budget
    ):
        run_rows = {v: sum(counts[v].get(dcid, 0) for dcid in run) for v in variables}
        requests.extend(
            _ReconcileRequest(variables=chunk, entities={role: run})
            for chunk in plan_request_chunks(
                variables,
                entities={role: run},
                url_budget_bytes=url_budget_bytes,
                row_weights=run_rows,
                row_budget=row_budget,
            )
        )
    return requests


def _sample_variables(
    groups: Mapping[RoleNames, Sequence[str]],
    *,
    expected_counts: Mapping[str, int],
    stamp: str,
    max_rows: int,
) -> set[str]:
    """Picks the variables `reconcile` checks.

    Each role-name group contributes its variable with the fewest expected
    rows (ties broken by name), even when that variable alone exceeds
    `max_rows`. The remaining variables follow in `sha256(stamp + variable)`
    order, and each joins while the running total of expected rows stays
    within `max_rows`.
    """
    sample: set[str] = set()
    for group_variables in groups.values():
        sample.add(min(group_variables, key=lambda v: (expected_counts[v], v)))
    total = sum(expected_counts[v] for v in sample)

    def _order(variable: str) -> str:
        return hashlib.sha256(f"{stamp}{variable}".encode()).hexdigest()

    for variable in sorted(expected_counts.keys() - sample, key=lambda v: (_order(v), v)):
        count = expected_counts[variable]
        if total + count <= max_rows:
            sample.add(variable)
            total += count
    return sample


def reconcile(
    sdmx: CoverageSdmxClient,
    *,
    table: pa.Table,
    role_names: Mapping[str, RoleNames],
    stamp: str,
    max_rows: int,
    row_budget: int = _RECONCILE_ROW_BUDGET,
) -> ReconciliationResult:
    """Compares per-variable coverage counts with SDMX `/data` counts for a
    sample of variables.

    The sample holds the variable with the fewest expected rows from each
    `(donor_role, recipient_role)` group, so every role mapping is checked,
    then further variables in `sha256(stamp + variable)` order while the
    expected rows fit `max_rows`. `stamp` is the fingerprint the build
    publishes under, so one data version always gets the same sample.

    Variables group by `RoleNames`, and `_plan_group_requests` plans each
    group's sampled variables into requests so none expects more than
    `row_budget` rows, the table's own `n_obs` sums, or has a query string
    over the client's `url_budget_bytes`. Each request passes `roles=` to avoid the stream reader's extra
    pre-scan request.
    Mismatches or missing SDMX rows raise `CoverageReconciliationError` and
    block publication. A transport, HTTP or truncated-body error from
    dc_sdmx propagates and also blocks it.
    """
    variables = sorted(set(cast("list[str]", table[VARIABLE_COLUMN].to_pylist())))
    if not variables:
        raise CoverageReconciliationError("coverage table has no role-bearing rows to reconcile")

    groups: dict[RoleNames, list[str]] = {}
    for variable in variables:
        groups.setdefault(role_names[variable], []).append(variable)

    expected_counts = _per_variable_n_obs(table)
    sample = _sample_variables(
        groups, expected_counts=expected_counts, stamp=stamp, max_rows=max_rows
    )
    sampled = sorted(sample)
    observed_counts: dict[str, int] = dict.fromkeys(sampled, 0)
    total_requests = 0
    start = time.monotonic()
    for names, group_variables in groups.items():
        roles = names.declared
        requests = _plan_group_requests(
            table,
            names=names,
            variables=sorted(v for v in group_variables if v in sample),
            expected_counts=expected_counts,
            row_budget=row_budget,
            url_budget_bytes=sdmx.url_budget_bytes,
        )
        total_requests += len(requests)
        for request in requests:
            with sdmx.stream_data_batches(
                variables=request.variables,
                entities=request.entities,
                roles=roles,
                time_period=None,
            ) as (_roles, batches):
                for batch in batches:
                    for variable in cast(
                        "list[str]", batch.column(SDMX_VARIABLE_COLUMN).to_pylist()
                    ):
                        observed_counts[variable] = observed_counts.get(variable, 0) + 1
    duration = time.monotonic() - start

    mismatches = [
        (variable, expected_counts[variable], observed_counts.get(variable, 0))
        for variable in sampled
        if expected_counts[variable] != observed_counts.get(variable, 0)
    ]
    if mismatches:
        examples = mismatches[:10]
        raise CoverageReconciliationError(
            f"{len(mismatches)} of {len(sampled)} sampled role-bearing StatVar(s) disagree "
            "between the coverage table and SDMX /data (variable, expected, observed), "
            f"e.g. {examples!r}"
        )
    return ReconciliationResult(
        duration_seconds=duration,
        request_count=total_requests,
        variables_checked=len(sampled),
        rows_expected=sum(expected_counts[v] for v in sampled),
    )


# --- Publication -----------------------------------------------------------


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def publish(
    *,
    base_uri: str,
    stamp: str,
    build_id: str,
    table: pa.Table,
    horizon: YearHorizon,
    role_names: Mapping[str, RoleNames],
    facet_metadata: Mapping[str, FacetMetadata],
    source: str,
    ingestion: Ingestion,
    expected_generation: str | None,
) -> bool:
    """Writes files to a new build directory, then conditionally commits
    `current.json` using `expected_generation`. The manifest records
    `ingestion`, the `IngestionHistory` row the build's fingerprint read.
    Build files are create-only.
    An existing file raises `CoverageBuildError`. Returns whether the pointer
    write landed. A lost race leaves an unreferenced build for pruning.
    """
    root = format_root(base_uri)
    build_prefix = join_uri(root, VERSIONS_PREFIX, build_id)
    with tempfile.TemporaryDirectory(prefix="coverage-build-") as tmp:
        local = Path(tmp)
        write_coverage_table(table, local / "coverage.parquet", horizon=horizon)
        write_role_names(dict(role_names), local / "role_names.json")
        write_facet_metadata(dict(facet_metadata), local / "facets.json")
        manifest = CoverageManifest(
            stamp=stamp,
            build_id=build_id,
            built_at=_utc_now_iso(),
            source=source,
            row_counts={
                "rows": table.num_rows,
                "variables": len(set(cast("list[str]", table[VARIABLE_COLUMN].to_pylist()))),
            },
            files={
                "coverage": "coverage.parquet",
                "role_names": "role_names.json",
                "facets": "facets.json",
            },
            horizon=horizon,
            ingestion=ingestion,
        )
        write_manifest(manifest, local / "manifest.json")

        # manifest.json last: a build directory holding one is complete.
        for name in ("coverage.parquet", "role_names.json", "facets.json", "manifest.json"):
            uri = join_uri(build_prefix, name)
            if not write_uri_bytes_if_unchanged(
                uri, (local / name).read_bytes(), expected_generation=None
            ):
                raise CoverageBuildError(
                    f"{uri} already exists; a build directory is written once, by the run "
                    "that created it"
                )
    return write_uri_bytes_if_unchanged(
        join_uri(root, POINTER_FILE),
        CoveragePointer(stamp=stamp, build_id=build_id).to_json().encode("utf-8"),
        expected_generation=expected_generation,
    )


def prune_old_versions(
    base_uri: str, *, now: Callable[[], float] = time.time
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deletes the build ids safe to remove from this format root and the
    compatibility pointers that have expired, and returns both, sorted.

    Builds go under `prune_versions` with `_MAX_RUN_SECONDS` as the grace
    period, keeping every build `current.json` or a compatibility pointer
    names. Pointers expire under `_expired_pointers`, after the builds, so
    the build one named stays until a later run, and each is deleted only if
    unchanged since it was read.
    """
    root = format_root(base_uri)
    builds = prune_versions(
        root,
        pointer_build_id=lambda raw: CoveragePointer.from_json(raw).build_id,
        grace_seconds=_MAX_RUN_SECONDS,
        now=now,
        pointer_prefix=COMPATIBLE_PREFIX,
    )
    expired: list[str] = []
    for name in _expired_pointers(base_uri, now=now()):
        uri = join_uri(root, COMPATIBLE_PREFIX, name)
        held = read_uri_bytes_and_generation(uri)
        if held is not None and delete_uri_if_unchanged(uri, expected_generation=held[1]):
            expired.append(name.removesuffix(".json"))
    return builds, tuple(sorted(expired))


def _expired_pointers(base_uri: str, *, now: float) -> list[str]:
    """The `compatible/` pointer names whose index no serving revision runs, last written
    `_COMPATIBLE_POINTER_RETENTION_SECONDS` ago or more.

    The deploy records the serving index (`record_serving_index`) only after both services
    run it, so every revision then serving runs that index, and a later deploy that moved only
    some services wrote its own index's pointer after the record. A pointer is therefore
    expired only when it is not the serving index's own and was last written before the
    record. With no record, nothing expires.
    """
    serving = read_serving_index(base_uri)
    if serving is None:
        return []
    own = f"{serving.content_hash}.json"
    return [
        name
        for name, written in list_uri_object_times(
            join_uri(format_root(base_uri), COMPATIBLE_PREFIX)
        ).items()
        if name != own
        and written < serving.recorded_at
        and now - written >= _COMPATIBLE_POINTER_RETENTION_SECONDS
    ]


def _prune(base_uri: str) -> None:
    builds, pointers = prune_old_versions(base_uri)
    if builds:
        emit("coverage_job_pruned", builds=list(builds))
    if pointers:
        emit("coverage_job_expired_compatibility_pointers", indexes=list(pointers))


def _record_compatibility(base_uri: str, *, stamp: str, build_id: str, index: CodeIndex) -> None:
    """Loads the published build with the serving loader and moves `index`'s compatibility
    pointer to it when it carries the index's provenances; emits the outcome."""
    loaded = load_coverage(base_uri, pointer=CoveragePointer(stamp=stamp, build_id=build_id))
    differences = record_if_compatible(base_uri, index, loaded)
    if not differences:
        emit(
            "coverage_index_compatible",
            build_id=build_id,
            index_content_hash=index.stamp.content_hash,
        )
        return
    emit(
        "coverage_index_incompatible",
        severity="ERROR",
        build_id=build_id,
        index_content_hash=index.stamp.content_hash,
        index_coverage_build=index.stamp.coverage_build,
        differing_variables=len(differences.variables),
        datasets_without_rows=list(differences.datasets_without_rows),
        variables=sorted(differences.variables)[:20],
        unlisted_members=list(differences.unlisted_members[:20]),
    )


# --- Stale-publish alerting --------------------------------------------------


def _update_stale_state(
    settings: JobSettings, *, fingerprint: Fingerprint, published_stamp: str | None, clock
) -> None:
    """Tracks unpublished time in `coverage_state.json` at the format root.

    Emits `coverage_publish_stale` after one cycle. A new fingerprint starts
    its own clock. Each format root has separate state.
    """
    if published_stamp == fingerprint.stamp:
        return
    behind_seconds = unpublished_seconds(
        format_root(settings.artifact_uri),
        state_file=_STATE_FILE,
        stamp=fingerprint.stamp,
        now=clock(),
    )
    if behind_seconds > settings.cycle_seconds:
        emit(
            "coverage_publish_stale",
            fingerprint_stamp=fingerprint.stamp,
            published_stamp=published_stamp,
            seconds_behind=round(behind_seconds, 1),
        )


# --- Orchestration -----------------------------------------------------------


def run(
    settings: JobSettings,
    *,
    bq: BigQueryReader,
    sdmx: CoverageSdmxClient,
    index: CodeIndex,
    clock=time.time,
) -> None:
    """Publishes a build when the fingerprint has moved, records it against `index`, the
    code index this image packages, and prunes."""
    if lock_held(bq, connection=settings.bq_connection):
        emit("coverage_job_skipped", reason="ingestion_lock_held")
        return

    fingerprint = read_fingerprint(bq, connection=settings.bq_connection)
    published, published_generation = _read_published(format_root(settings.artifact_uri))
    published_stamp = published.stamp if published is not None else None
    _update_stale_state(
        settings, fingerprint=fingerprint, published_stamp=published_stamp, clock=clock
    )
    if published_stamp == fingerprint.stamp:
        emit("coverage_job_noop", stamp=fingerprint.stamp)
        _prune(settings.artifact_uri)
        return

    build_start = time.monotonic()
    build = build_coverage_table(bq, connection=settings.bq_connection)
    table = build.table
    role_names = resolve_role_names(bq, sdmx, connection=settings.bq_connection, table=table)
    facet_metadata = resolve_facet_metadata(bq, connection=settings.bq_connection, table=table)
    reconciliation = reconcile(
        sdmx,
        table=table,
        role_names=role_names,
        stamp=fingerprint.stamp,
        max_rows=settings.reconcile_max_rows,
    )
    build_seconds = time.monotonic() - build_start

    if lock_held(bq, connection=settings.bq_connection):
        emit(
            "coverage_job_discarded",
            reason="ingestion_lock_acquired_during_build",
            stamp=fingerprint.stamp,
        )
        return
    current_fingerprint = read_fingerprint(bq, connection=settings.bq_connection)
    if current_fingerprint.stamp != fingerprint.stamp:
        emit(
            "coverage_job_discarded",
            reason="fingerprint_moved_during_build",
            stamp=fingerprint.stamp,
        )
        return

    build_id = new_build_id()
    committed = publish(
        base_uri=settings.artifact_uri,
        stamp=fingerprint.stamp,
        build_id=build_id,
        table=table,
        horizon=build.horizon,
        role_names=role_names,
        facet_metadata=facet_metadata,
        source="series_coverage.job",
        ingestion=fingerprint.ingestion,
        expected_generation=published_generation,
    )
    if not committed:
        emit(
            "coverage_job_lost_publish_race",
            stamp=fingerprint.stamp,
            build_id=build_id,
            published_generation=published_generation,
        )
        return
    emit(
        "coverage_job_published",
        stamp=fingerprint.stamp,
        build_id=build_id,
        rows=table.num_rows,
        variables=len(role_names),
        build_seconds=round(build_seconds, 1),
        reconciliation_seconds=round(reconciliation.duration_seconds, 1),
        reconciliation_requests=reconciliation.request_count,
        reconciliation_variables=reconciliation.variables_checked,
        reconciliation_rows_expected=reconciliation.rows_expected,
    )
    _record_compatibility(
        settings.artifact_uri, stamp=fingerprint.stamp, build_id=build_id, index=index
    )
    _prune(settings.artifact_uri)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    settings = JobSettings.from_env()
    bq = LiveBigQuery(project=settings.bq_project)
    sdmx = SdmxClient(
        base_url=settings.sdmx_base_url,
        token=settings.sdmx_token,
        timeout=settings.sdmx_timeout_seconds,
    )
    try:
        run(settings, bq=bq, sdmx=sdmx, index=get_index())
    except (
        CoverageArtifactError,
        CoverageBuildError,
        CoverageFingerprintError,
        CoveragePointerReadError,
        CoverageRoleNameError,
        CoverageFacetMetadataError,
        CoverageReconciliationError,
    ) as exc:
        emit("coverage_job_failed", error=str(exc))
        logger.error("coverage job failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BigQueryReader",
    "CoverageBuildError",
    "CoverageFacetMetadataError",
    "CoverageFingerprintError",
    "CoveragePointerReadError",
    "CoverageReconciliationError",
    "CoverageRoleNameError",
    "CoverageSdmxClient",
    "Fingerprint",
    "JobSettings",
    "LiveBigQuery",
    "ReconciliationResult",
    "build_coverage_table",
    "lock_held",
    "main",
    "new_build_id",
    "prune_old_versions",
    "publish",
    "read_fingerprint",
    "reconcile",
    "resolve_facet_metadata",
    "resolve_role_names",
    "run",
]
