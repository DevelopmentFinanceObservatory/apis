"""Load and cache currency-conversion rate and deflator tables.

A table is everything one StatVar serves under its table provenance. The
loader lists the StatVar's `observationAbout` entities with one
`/availability` read, then fetches their full history with one `/data` read.
Exchange-rate rows are about currency nodes (`CurrencyEUR`). Deflator rows
are about countries and aggregates such as `DACCountriesTotal`.

Actual and `IMFProjection` facets are resolved per entity-year, with actuals
preferred and rows from other provenances dropped. Missing survivors are
coverage gaps. Admission requires annual rows, one row per entity-year,
finite positive values, and a `CurrencyUSD` exchange rate within `1e-6` of
one.

`lastDataRefreshDate` on the table provenance is the change signal. The
loader reads it before and after both fetches. A mismatch retries once, then
raises `UpstreamVersionChangedError`. The admitted table's version is a
SHA-256 digest of its rows, values included, so two loads of the same served
rows carry the same version whatever the refresh date says. The table also
carries the refresh date it was admitted under, which head reuse compares with
the live date. The thread-safe cache keys tables by StatVar, provenance, and
refresh date, so recipes that read one table share one fetch.

`RateTableCache.for_request` returns a `RequestRateTables` view. The view pins
the first table admitted for each StatVar and provenance, and every later load
in that request returns the pinned table without reading the refresh date
again, so all members of one request cite one version of each table.
`load_table` is the public entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, cast, override

import pyarrow as pa

from dc_sdmx import SdmxClient, SdmxQueryError
from dc_sdmx.results import SdmxObservation
from materialise.errors import (
    MaterialisationError,
    UpstreamResponseError,
    UpstreamVersionChangedError,
)
from materialise.graph import GraphReader

# Requests prefix StatVars with `dcid:`; graph and SDMX reads use bare dcids.
_STAT_VAR_REF_PREFIX = "dcid:"

# The one role a rate or deflator StatVar declares.
_SUBJECT_ROLE = "observationAbout"

# SDMX marks projected rate facets with `measurementMethod="IMFProjection"`.
# Actual facets carry `None` or another method.
_PROJECTION_MEASUREMENT_METHOD = "IMFProjection"

# The one property `load_table` reads off a table provenance to learn whether
# its rows may have changed.
_REFRESH_PROPERTY = "lastDataRefreshDate"

# The exchange-rate table's own anchor: `X[USD, t]` is one by definition.
# `_validate_table` refuses admission when the loaded table disagrees
# (subject to `_USD_ANCHOR_RELATIVE_TOLERANCE`, below).
_USD_ENTITY = "CurrencyUSD"

# Published rate ratios are rounded, so the USD identity needs a 1e-6
# tolerance. A 0.01% deviation (1e-4) still fails admission.
_USD_ANCHOR_RELATIVE_TOLERANCE = 1e-6

# Annual SDMX-CSV `TIME_PERIOD` values are bare four-digit years; quarters,
# months, and ranges fail admission.
_ANNUAL_TIME_PERIOD_RE = re.compile(r"^\d{4}$")

RateTableKind = Literal["exchange_rate", "deflator"]

RATE_TABLE_SCHEMA = pa.schema(
    [
        ("entity", pa.string()),
        ("year", pa.int32()),
        ("value", pa.float64()),
        ("facet", pa.string()),
    ]
)
"""The admitted table's own Arrow contract: one row per entity-year per
series, `facet` one of `"actual"`/`"projected"` (`_union_facets`'s own
output vocabulary). `entity` is a currency node in an exchange-rate table and
a country or aggregate in a deflator table. Every `RateTable.table` is built
under this schema."""


class RateTableInvalidError(MaterialisationError):
    """501 `capability_unavailable` when a table lacks an annual period, has
    duplicate entity-year rows from its provenance, contains a non-finite or
    non-positive value, or an exchange-rate table fails the USD anchor check.
    It is also raised when the provenance has no `lastDataRefreshDate`, when
    the StatVar serves no entities, or when its entity list cannot fit one
    `/data` request. Every conversion reading that table then refuses with
    this code.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail, code="capability_unavailable", status=501, detail=detail)


@dataclass(frozen=True)
class RateTable:
    """An immutable admitted table. IDs are bare dcids; rows follow
    `RATE_TABLE_SCHEMA`, sorted by `(entity, year)` for deterministic bytes.
    `version` is the SHA-256 digest of every row (`_table_digest`), the
    content version. `refresh_date` is the provenance's `lastDataRefreshDate`
    read on both sides of the fetch, the change signal.

    `lookup` and `entities` are built on first use and kept with the table, so every
    conversion and coverage check reading a cached table shares one build.
    """

    kind: RateTableKind
    stat_var: str
    provenance_id: str
    version: str
    refresh_date: str
    table: pa.Table

    @cached_property
    def lookup(self) -> dict[tuple[str, int], tuple[float, str]]:
        """`(entity, year) -> (value, facet)` over every row."""
        entities = cast("list[str]", self.table["entity"].to_pylist())
        years = cast("list[int]", self.table["year"].to_pylist())
        values = cast("list[float]", self.table["value"].to_pylist())
        facets = cast("list[str]", self.table["facet"].to_pylist())
        return {
            (entity, year): (value, facet)
            for entity, year, value, facet in zip(entities, years, values, facets, strict=True)
        }

    @cached_property
    def entities(self) -> frozenset[str]:
        """Every entity the table holds a row for."""
        return frozenset(cast("list[str]", self.table["entity"].to_pylist()))


def _table_digest(table: pa.Table) -> str:
    """SHA-256 over every row in `(entity, year, facet)` order, each row
    serialised as the JSON array `[entity, year, facet, value]`. A changed
    value, an added or dropped row, or an actual turned projection moves the
    digest. A reload of identical rows leaves it where it was.
    """
    rows = sorted(
        zip(
            table["entity"].to_pylist(),
            table["year"].to_pylist(),
            table["facet"].to_pylist(),
            table["value"].to_pylist(),
            strict=True,
        ),
        key=lambda row: (row[0], row[1], row[2]),
    )
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(json.dumps(list(row)).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


class RateTableCache:
    """Thread-safe process-wide cache keyed by StatVar, table provenance, and
    `lastDataRefreshDate`. Admitted tables remain cached for the process
    lifetime. When two loads race on one key, the first admitted table wins
    and both callers receive it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str, str], RateTable] = {}

    def pinned(self, *, stat_var: str, provenance_id: str) -> RateTable | None:
        """The table this request already holds for the pair. The process
        cache spans requests and pins nothing, so it returns `None`.
        """
        return None

    def get(self, *, stat_var: str, provenance_id: str, refresh_date: str) -> RateTable | None:
        with self._lock:
            return self._entries.get((stat_var, provenance_id, refresh_date))

    def admit(self, table: RateTable) -> RateTable:
        """Cache `table` under its own refresh date unless that key already
        holds a table, and return the cached one.
        """
        key = (table.stat_var, table.provenance_id, table.refresh_date)
        with self._lock:
            return self._entries.setdefault(key, table)

    def for_request(self) -> RateTableCache:
        """A `RequestRateTables` view over this cache for one request."""
        return RequestRateTables(self)


class RequestRateTables(RateTableCache):
    """One request's view of a process-wide `RateTableCache`.

    Admission goes through the shared cache, then pins the result by StatVar
    and provenance. The first table pinned for a pair wins, including when
    two members race, so every member of the request converts against the
    same table even if the provenance refreshes mid-request. `for_request`
    returns the view itself, so nested entry points keep one scope.
    """

    def __init__(self, shared: RateTableCache) -> None:
        self._shared = shared
        self._lock = threading.Lock()
        self._pins: dict[tuple[str, str], RateTable] = {}

    @property
    def shared(self) -> RateTableCache:
        """The process-wide cache this view admits through."""
        return self._shared

    @override
    def pinned(self, *, stat_var: str, provenance_id: str) -> RateTable | None:
        with self._lock:
            return self._pins.get((stat_var, provenance_id))

    @override
    def get(self, *, stat_var: str, provenance_id: str, refresh_date: str) -> RateTable | None:
        return self._shared.get(
            stat_var=stat_var, provenance_id=provenance_id, refresh_date=refresh_date
        )

    @override
    def admit(self, table: RateTable) -> RateTable:
        cached = self._shared.admit(table)
        with self._lock:
            return self._pins.setdefault((cached.stat_var, cached.provenance_id), cached)

    @override
    def for_request(self) -> RateTableCache:
        return self


def load_table(
    *,
    kind: RateTableKind,
    stat_var: str,
    provenance_id: str,
    sdmx: SdmxClient,
    graph: GraphReader,
    cache: RateTableCache,
    deadline: float | None = None,
) -> RateTable:
    """Load, validate, and cache the whole table `stat_var` serves under
    `provenance_id`.

    A table the request already pinned returns with no read. Otherwise a
    cache hit on the refresh date read before fetching returns without a
    fetch. On a miss, one `/availability` read lists the StatVar's
    `observationAbout` entities and one `/data` read fetches their full
    history, including any `IMFProjection` facet. Both sit between the two
    refresh-date reads. Every returned table passes through `cache.admit`, so
    a request view pins it. `deadline` reaches the graph reads and both SDMX
    reads.
    """
    bare_stat_var = stat_var.removeprefix(_STAT_VAR_REF_PREFIX)
    pinned = cache.pinned(stat_var=bare_stat_var, provenance_id=provenance_id)
    if pinned is not None:
        return pinned
    before = _read_refresh_date(graph, provenance_id, deadline=deadline)
    if before is not None:
        cached = cache.get(stat_var=bare_stat_var, provenance_id=provenance_id, refresh_date=before)
        if cached is not None:
            return cache.admit(cached)

    def _fetch() -> tuple[SdmxObservation, ...]:
        return _fetch_rows(sdmx=sdmx, stat_var=bare_stat_var, deadline=deadline)

    rows, refresh_date = _admit_with_version_guard(
        graph=graph,
        provenance_id=provenance_id,
        before=before,
        fetch=_fetch,
        deadline=deadline,
    )
    table = _union_and_validate(
        rows, kind=kind, bare_stat_var=bare_stat_var, provenance_id=provenance_id
    )
    result = RateTable(
        kind=kind,
        stat_var=bare_stat_var,
        provenance_id=provenance_id,
        version=_table_digest(table),
        refresh_date=refresh_date,
        table=table,
    )
    return cache.admit(result)


def _read_refresh_date(
    graph: GraphReader, provenance_id: str, *, deadline: float | None
) -> str | None:
    """The provenance's own `lastDataRefreshDate`, or `None` when the graph
    omits the provenance or it carries no such arc. One `node_arcs_many`
    call, scoped to just this property. This module reads only
    `lastDataRefreshDate`; see `facets._probe_provenance_refresh` for
    reading both properties.
    """
    arcs_by_id = graph.node_arcs_many(
        [provenance_id], properties=(_REFRESH_PROPERTY,), deadline=deadline
    )
    arcs = arcs_by_id.get(provenance_id)
    if arcs is None:
        return None
    values = arcs.get(_REFRESH_PROPERTY, ())
    return values[0] if values else None


def _admit_with_version_guard(
    *,
    graph: GraphReader,
    provenance_id: str,
    before: str | None,
    fetch: Callable[[], tuple[SdmxObservation, ...]],
    deadline: float | None,
) -> tuple[tuple[SdmxObservation, ...], str]:
    """Fetch once and compare refresh dates. On a mismatch, fetch once more
    using the first `after` value as the retry's `before`; a second mismatch
    raises `UpstreamVersionChangedError`. Matching `None` dates raise
    `RateTableInvalidError` because the table has no cache key.
    """
    for _attempt in range(2):
        rows = fetch()
        after = _read_refresh_date(graph, provenance_id, deadline=deadline)
        if before == after:
            if after is None:
                raise RateTableInvalidError(
                    f"provenance {provenance_id!r} carries no lastDataRefreshDate; its "
                    "rate table cannot be versioned"
                )
            return rows, after
        before = after

    raise UpstreamVersionChangedError(
        f"provenance {provenance_id!r}'s lastDataRefreshDate changed while its rate "
        "table was loading, and the retry did not converge either"
    )


def _fetch_rows(
    *, sdmx: SdmxClient, stat_var: str, deadline: float | None
) -> tuple[SdmxObservation, ...]:
    """List the StatVar's `observationAbout` entities, then fetch them all in
    one `sdmx.data` call. Raises `RateTableInvalidError` when the StatVar
    serves no entity or its entity list overflows the client's URL budget,
    and `UpstreamResponseError` if the response reports a different role
    shape. A StatVar reachable through financing roles is outside this
    loader's scope.
    """
    availability = sdmx.availability(
        variables=[stat_var], component=_SUBJECT_ROLE, deadline=deadline
    )
    entities = availability.components.get(_SUBJECT_ROLE, ())
    if not entities:
        raise RateTableInvalidError(
            f"rate/deflator StatVar {stat_var!r} serves no {_SUBJECT_ROLE} entities"
        )
    try:
        result = sdmx.data(
            variables=[stat_var], entities={_SUBJECT_ROLE: list(entities)}, deadline=deadline
        )
    except SdmxQueryError as exc:
        raise RateTableInvalidError(
            f"rate/deflator StatVar {stat_var!r} serves {len(entities)} entities, which "
            f"no single /data request can carry: {exc}"
        ) from exc
    if result.roles != (_SUBJECT_ROLE,):
        raise UpstreamResponseError(
            f"rate/deflator StatVar {stat_var!r} returned entity roles {result.roles!r}, "
            "not the expected ('observationAbout',); this loader only handles a "
            "subject-only rate/deflator StatVar"
        )
    return result.rows


def _union_and_validate(
    rows: Sequence[SdmxObservation], *, kind: RateTableKind, bare_stat_var: str, provenance_id: str
) -> pa.Table:
    """`_union_facets`, then `_validate_table` over its result. Split out
    from `load_table` so a test can drive it directly over hand-built rows,
    without a fetch or a version guard in the way.
    """
    unioned = _union_facets(rows, bare_stat_var=bare_stat_var, provenance_id=provenance_id)
    return _validate_table(unioned, kind=kind)


@dataclass(frozen=True)
class _UnionedRow:
    entity: str
    year: int
    value: float
    facet: Literal["actual", "projected"]
    provenance: str | None


def _resolve_facet_duplicates(
    candidates: list[_UnionedRow], *, provenance_id: str
) -> list[_UnionedRow]:
    """Keep rows from this load's provenance and drop stale re-imports, even
    when a stale row has no current replacement. No match means a coverage
    gap; multiple matches survive for `_validate_table` to reject.
    """
    return [row for row in candidates if row.provenance == provenance_id]


def _union_facets(
    rows: Sequence[SdmxObservation], *, bare_stat_var: str, provenance_id: str
) -> list[_UnionedRow]:
    """Choose actuals over projections for each `(entity, year)`; retain a
    projected row marked `projected` when no actual exists. Drop missing
    values and stale provenance rows. Unexpected variables or missing
    `observationAbout` values raise `UpstreamResponseError`; duplicate
    current rows reach `_validate_table` for refusal.
    """
    by_key: dict[tuple[str, int], dict[str, list[_UnionedRow]]] = {}
    for row in rows:
        if row.variable.removeprefix(_STAT_VAR_REF_PREFIX) != bare_stat_var:
            raise UpstreamResponseError(
                f"rate/deflator fetch for {bare_stat_var!r} returned a row for "
                f"variable {row.variable!r}"
            )
        entity = row.role_values.get(_SUBJECT_ROLE)
        if entity is None:
            raise UpstreamResponseError(
                f"rate/deflator fetch for {bare_stat_var!r} returned a row with no "
                "observationAbout role value"
            )
        value = row.value_as_float()
        if value is None:
            continue
        year = _parse_annual_year(row)
        facet: Literal["actual", "projected"] = (
            "projected" if row.measurement_method == _PROJECTION_MEASUREMENT_METHOD else "actual"
        )
        group = by_key.setdefault((entity, year), {"actual": [], "projected": []})
        group[facet].append(
            _UnionedRow(
                entity=entity, year=year, value=value, facet=facet, provenance=row.provenance
            )
        )

    unioned: list[_UnionedRow] = []
    for by_facet in by_key.values():
        actual = _resolve_facet_duplicates(by_facet["actual"], provenance_id=provenance_id)
        projected = _resolve_facet_duplicates(by_facet["projected"], provenance_id=provenance_id)
        if actual:
            unioned.extend(actual)
        else:
            unioned.extend(projected)
    return unioned


def _parse_annual_year(row: SdmxObservation) -> int:
    """Return the year as an `int`. Raise `RateTableInvalidError` unless
    `observationPeriod` is absent or `P1Y` and `TIME_PERIOD` is a bare
    four-digit year; the period check applies even when the cadence is absent.
    """
    if row.observation_period is not None and row.observation_period != "P1Y":
        raise RateTableInvalidError(
            f"non-annual period {row.observation_period!r} for entity "
            f"{row.role_values.get(_SUBJECT_ROLE)!r}, TIME_PERIOD {row.time_period!r}"
        )
    time_period = row.time_period
    if time_period is None or not _ANNUAL_TIME_PERIOD_RE.match(time_period):
        raise RateTableInvalidError(
            f"non-annual TIME_PERIOD {time_period!r} for entity "
            f"{row.role_values.get(_SUBJECT_ROLE)!r}"
        )
    return int(time_period)


def _validate_table(rows: Sequence[_UnionedRow], *, kind: RateTableKind) -> pa.Table:
    """Build `RATE_TABLE_SCHEMA` or raise `RateTableInvalidError` for
    duplicate entity-years or non-finite or non-positive values.
    Exchange-rate tables also require a `CurrencyUSD` anchor within
    tolerance. Sort by `(entity, year)` for deterministic output.
    """
    by_key: dict[tuple[str, int], list[_UnionedRow]] = {}
    for row in rows:
        by_key.setdefault((row.entity, row.year), []).append(row)

    for (entity, year), group in by_key.items():
        if len(group) > 1:
            raise RateTableInvalidError(
                f"duplicate entity-year: {len(group)} rows for entity {entity!r}, year {year}"
            )
        value = group[0].value
        if not math.isfinite(value) or value <= 0:
            raise RateTableInvalidError(
                f"non-finite or non-positive value {value!r} for entity {entity!r}, year {year}"
            )

    if kind == "exchange_rate":
        usd_rows = [row for row in rows if row.entity == _USD_ENTITY]
        if not usd_rows:
            raise RateTableInvalidError(
                f"missing USD anchor: no {_USD_ENTITY!r} row in the exchange-rate table"
            )
        for row in usd_rows:
            if not math.isclose(row.value, 1.0, rel_tol=_USD_ANCHOR_RELATIVE_TOLERANCE):
                raise RateTableInvalidError(
                    f"USD anchor {_USD_ENTITY!r} carries rate {row.value!r} in year "
                    f"{row.year}, not within {_USD_ANCHOR_RELATIVE_TOLERANCE!r} relative "
                    "tolerance of 1.0"
                )

    ordered = sorted(rows, key=lambda row: (row.entity, row.year))
    return pa.Table.from_pylist(
        [
            {"entity": row.entity, "year": row.year, "value": row.value, "facet": row.facet}
            for row in ordered
        ],
        schema=RATE_TABLE_SCHEMA,
    )
