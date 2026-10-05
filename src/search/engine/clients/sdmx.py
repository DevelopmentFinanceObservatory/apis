"""SDMX observation and availability clients for Search.

``LiveSdmxClient`` wraps ``dc_sdmx.SdmxClient`` on the graph deployment's
SDMX endpoint. ``role_series`` returns real Facets filtered by non-empty
facet-id sets. Oversized OR-lists are split to fit the client's URL budget.
``RoleFacetCandidate`` comes from the coverage artifact and carries no
observation count or dates. All methods return the ``Facet`` shape from
``clients.graph``.

Invalid role parameters raise ``ValueError`` before HTTP or cache access.
Missing and unsupported filters map to ``RoleNotDeclaredError``. Incompatible
variables and role ceilings map to ``RoleSignatureError``. Transport,
response, containment, and HTTP errors map to ``GraphInfraError``.
``SdmxQueryError`` and ``SdmxCapExceededError`` propagate. This module must
not import ``search.eval``.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import httpx

import search.engine.config as config  # module-attribute access for runtime-resolved values
from dc_sdmx import (
    ContainedIn,
    EntityFilter,
    SdmxClient,
    SdmxContainmentFilterError,
    SdmxHttpError,
    SdmxIncompatibleVariablesError,
    SdmxMissingFilterError,
    SdmxObservation,
    SdmxResponseError,
    SdmxRoleCeilingError,
    SdmxTransportError,
    SdmxUnsupportedFilterError,
    place_slices,
)

# Reuse dc_sdmx query builders and packing helpers so URL estimates match the wire.
# Read the budget from the wrapped client to honour per-instance settings.
from dc_sdmx.client import (
    _entity_params,
    _or_list,
    _pack_variable_chunks,
    _time_period_param,
)
from origin_catalogue import DFO, current_catalogue
from origin_catalogue.origins import UN_ID_PREFIX
from search.engine.clients.graph import Facet
from search.engine.errors import GraphInfraError, RoleNotDeclaredError, RoleSignatureError

# A role_tuple is the sorted tuple of (role, dcid) pairs a row's role_values
# carried. It is the same shape role_series keys its return dict by, and
# RolePairs.head yields.
RoleTuple = tuple[tuple[str, str], ...]

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class RolePairs(Protocol):
    """Expose ordered role tuples and totals without materialising the population."""

    def head(
        self,
        limit: int,
        *,
        first_year: int | None,
        last_year: int | None,
        with_data: bool = False,
    ) -> list[tuple[RoleTuple, int]]:
        """Return the first ``limit`` tuples in RoleTuple order, counted within the requested and build windows.

        A tuple with zero observations keeps its place with count ``0``, unless
        ``with_data`` skips it before the cut.
        """
        ...

    def n_obs(self, *, first_year: int | None, last_year: int | None) -> int:
        """Summed observation count across every role tuple, counted the
        same way `head` counts one."""
        ...

    def n_with_data(self, *, first_year: int | None, last_year: int | None) -> int:
        """How many full role tuples have an observation inside the window: the series a
        read returns, one per tuple, including a role the pairs leave open."""
        ...

    def max_n_obs(self, *, first_year: int | None, last_year: int | None) -> int:
        """The largest one full role tuple's observation count inside the window."""
        ...

    def year_span(self, *, first_year: int | None, last_year: int | None) -> tuple[int, int] | None:
        """The earliest and latest years any tuple has an observation inside the window."""
        ...

    def members_with_data(
        self, *, first_year: int | None, last_year: int | None, role: str | None = None
    ) -> frozenset[str]:
        """The distinct dcids on `role`, one of the tuples' graph roles, among the tuples
        with an observation inside the window. None reads the candidate's member role."""
        ...

    def excluding(self, *, role: str, dcids: frozenset[str]) -> RolePairs:
        """These pairs without the tuples binding one of `dcids` on `role`, one of the
        tuples' graph roles, so every count, span and member read sees only the rest."""
        ...


@dataclass
class RoleFacetCandidate:
    """Summary for one facet id across all role tuples it covers.

    Coverage produces one candidate per ``(variable, facet_id)`` row.
    Coverage rows provide pair counts and year ranges. Observation values,
    real counts, and earliest dates come from ``role_series`` Facets.
    ``members_by_role`` maps each role to its distinct dcids. ``pair_count`` counts tuples,
    and ``pairs`` reads them on demand. ``latest_date`` is the latest of the tuples' latest
    observations.

    Provenance, method, unit, and scaling factor are shared by every tuple
    under a facet id because the coverage artifact keys metadata by that id.
    """

    facet_id: str
    members_by_role: Mapping[str, frozenset[str]]
    pair_count: int
    latest_date: str | None
    pairs: RolePairs
    provenance_id: str | None = None
    measurement_method: str | None = None
    unit: str | None = None
    scaling_factor: str | None = None


# {variable: [RoleFacetCandidate, ...]}, one candidate per distinct facet_id
# a role index build returned for that variable, in first-seen order.
RoleFacetIndex = dict[str, list[RoleFacetCandidate]]


type RoleTimePeriod = Sequence[str] | Literal["LATEST"] | None
"""A role read's time filter: exact periods, `LATEST` for each series' latest point, or
None for full history."""


def _validate_role_time_period(*, method: str, time_period: RoleTimePeriod) -> None:
    """Reject the bare strings, other than `LATEST`, that structurally satisfy
    ``Sequence[str]``."""
    if isinstance(time_period, (str, bytes)) and time_period != "LATEST":
        raise ValueError(
            f"{method} time_period got a bare string/bytes ({time_period!r}); "
            'pass a sequence of explicit period strings, "LATEST", or None'
        )


def _validate_role_variables(*, method: str, variables: Sequence[str]) -> None:
    """Reject bare-string, empty, or repeated variable lists.

    Strings satisfy ``Sequence[str]`` and would become character filters.
    Empty and repeated lists can behave differently with caching, so all
    implementations reject them with ``ValueError`` before any read.
    """
    if isinstance(variables, (str, bytes)):
        raise ValueError(
            f"{method} variables got a bare string/bytes ({variables!r}); "
            "pass a sequence of variable dcids"
        )
    if not variables:
        raise ValueError(
            f"{method} requires at least one variable; an empty variable list "
            "has no answer to cache or to read"
        )
    seen: set[str] = set()
    for variable in variables:
        if variable in seen:
            raise ValueError(
                f"{method} variables repeats {variable!r} "
                f"({variables.count(variable)} times); a variable must appear "
                "at most once per request"
            )
        seen.add(variable)


def _validate_role_series_params(
    *,
    variables: Sequence[str],
    facet_ids: AbstractSet[str],
    time_period: RoleTimePeriod,
) -> None:
    """Shared ``role_series`` validation for live, test, and cached clients.

    Facet ids and variables are non-empty, and variables are unique.
    A bare ``str`` or ``bytes`` time period other than ``"LATEST"`` raises ``ValueError``.
    ``None`` selects full history, ``"LATEST"`` each series' latest point, and a sequence
    exact periods. Invalid inputs raise ``ValueError`` before reads or cache keys.
    """
    if not facet_ids:
        raise ValueError("role_series requires a non-empty facet_ids set")
    _validate_role_variables(method="role_series", variables=variables)
    _validate_role_time_period(method="role_series", time_period=time_period)


class UpstreamVariableRoutedToSdmx(RuntimeError):
    """An SDMX read named a variable our mixer's SDMX service does not serve. That covers a UN
    System Data Commons variable on any read, and on a role-series read a variable the origin
    catalogue places on Data Commons by Google alone. The mixer answers such a variable with
    an empty result that reads as no data, so every search SDMX entry point refuses it."""


def assert_no_un_variables(*, method: str, variables: Sequence[str]) -> None:
    """Raise `UpstreamVariableRoutedToSdmx` when `variables` holds a UN System Data Commons
    variable, whose rows only the origin graph client reads."""
    un = [variable for variable in variables if variable.startswith(UN_ID_PREFIX)]
    if un:
        raise UpstreamVariableRoutedToSdmx(
            f"{method} was asked for UN System Data Commons variables {un!r}; they are read "
            "through the origin graph client"
        )


def assert_dfo_role_variables(variables: Sequence[str]) -> None:
    """Raise `UpstreamVariableRoutedToSdmx` when the bound catalogue places any of
    `variables` upstream, meaning a UN variable or a base-table variable the DFO set does not
    hold."""
    catalogue = current_catalogue()
    upstream = [
        variable
        for variable in variables
        if variable.startswith(UN_ID_PREFIX)
        or (
            catalogue.base_places(variable) is not None
            and DFO not in catalogue.data_commons(variable)
        )
    ]
    if upstream:
        raise UpstreamVariableRoutedToSdmx(
            f"role_series was asked for upstream variables {upstream!r}; SDMX serves DFO's "
            "role-bearing variables only"
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EngineSdmxClient(Protocol):
    def role_series(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, Sequence[str]],
        facet_ids: AbstractSet[str],
        time_period: RoleTimePeriod = None,
        max_rows: int,
        max_bytes: int,
        deadline: float | None = None,
    ) -> dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]]:
        """Fetch real Facets grouped as ``{variable: {role_tuple: [Facet]}}``.

        Each bound role maps to a non-empty sequence of dcid strings. Omit
        unbound roles. Empty values raise ``SdmxQueryError``. ``None`` raises
        ``TypeError``. ``facet_ids`` must be a non-empty set. The wire accepts
        a comma OR-list, split into requests when it exceeds the URL budget.
        Rows retain their own ``facet_id`` for attribution. Explicit
        ``time_period`` values select exact periods. ``None`` selects full
        history. ``"LATEST"`` selects each series' latest point, computed by the mixer per
        role tuple and facet. Row and byte caps apply per request.

        Invalid parameters raise ``ValueError`` before HTTP or cache access.
        A returned facet outside the request raises ``GraphInfraError``.
        Other typed errors follow the module contract; ``SdmxCapExceededError`` propagates
        uncaught, so a caller wanting partial results catches it.
        """
        ...

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        time_period: str | Sequence[str] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        """Return available dcids for an entity role or SDMX component.

        Filters can name roles, ``observationAbout``, ``unit``,
        ``provenance``, or ``measurementMethod``. Role values follow the
        ``role_series`` dcid-sequence rules, or are a ``ContainedIn``
        containment filter. A bare ``time_period`` string
        selects one exact period. A sequence is an exact-period OR-list.
        The endpoint serves exact periods. ``LATEST`` and ranges raise
        ``RoleNotDeclaredError``. Caps raise ``SdmxCapExceededError``.
        Deadline or ``read_seconds`` expiry raises ``TimeoutError``. Empty results are ``[]``.
        """
        ...

    def source_key(self) -> str:
        """Return the backend the client reads, which the default place scope's memo keys on
        beside the graph stamp (`materialise.place_scope`)."""
        ...

    def subject_observations(
        self,
        *,
        variables: Sequence[str],
        places: Sequence[str],
        facet_ids: Sequence[str] | None = None,
        latest_only: bool = True,
        max_rows: int | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[SdmxObservation]]]:
        """Fetch subject-axis observations for each variable at each of ``places``.

        Omitting ``facet_ids`` fetches every facet. Pass ids to select one shared
        facet across entities. ``latest_only`` returns the latest row per
        variable/entity/facet. ``False`` fetches full history. Results group as
        ``{variable: {entity: [SdmxObservation]}}``. A row with no
        ``observationAbout`` raises ``GraphInfraError``. Row and byte caps bound the
        fetch. Typed errors follow the module contract. ``SdmxQueryError`` and
        ``SdmxCapExceededError`` propagate.
        """
        ...


# ---------------------------------------------------------------------------
# Exception translation
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _translate_sdmx_errors():
    """Map typed SDMX errors to engine errors. Query and cap errors propagate.

    Catch role-specific errors before the generic ``SdmxHttpError`` base.
    """
    try:
        yield
    except (SdmxMissingFilterError, SdmxUnsupportedFilterError) as exc:
        raise RoleNotDeclaredError(str(exc)) from exc
    except (SdmxIncompatibleVariablesError, SdmxRoleCeilingError) as exc:
        raise RoleSignatureError(str(exc)) from exc
    except (
        SdmxTransportError,
        SdmxResponseError,
        SdmxContainmentFilterError,
        SdmxHttpError,
    ) as exc:
        raise GraphInfraError(
            str(exc), upstream_status=getattr(exc, "upstream_status", None)
        ) from exc


# ---------------------------------------------------------------------------
# Group Facets by their wire facet_id.
# ---------------------------------------------------------------------------


def _check_facet_consistency(facet_id: str | None, rows: list[SdmxObservation]) -> None:
    """Raise ``GraphInfraError`` if rows for one facet disagree on unit, method, provenance, or scale."""
    for field_name in ("unit", "measurement_method", "provenance", "scaling_factor"):
        values = {getattr(row, field_name) for row in rows}
        if len(values) > 1:
            raise GraphInfraError(
                f"SDMX rows sharing facet_id {facet_id!r} disagree on "
                f"{field_name!r}: {sorted(v for v in values if v is not None)!r}"
            )


def _facets_from_sdmx_rows(rows: Sequence[SdmxObservation]) -> list[Facet]:
    """Group homogeneous rows into one ``Facet`` per facet id, in first-seen order.

    Dates and values follow sorted ``time_period``. Rows without a period are
    skipped, while ``obs_count`` includes every row. Metadata comes from the
    shared row fields, and ``facet_id`` is kept for role-series attribution.
    """
    by_facet: dict[str | None, list[SdmxObservation]] = {}
    for row in rows:
        by_facet.setdefault(row.facet_id, []).append(row)

    facets: list[Facet] = []
    for facet_id, group in by_facet.items():
        _check_facet_consistency(facet_id, group)
        sorted_group = sorted(group, key=lambda r: r.time_period or "")
        dates: list[str] = []
        values: list[float | None] = []
        for row in sorted_group:
            if not row.time_period:
                continue
            dates.append(row.time_period)
            values.append(row.value_as_float())
        facets.append(
            Facet(
                earliest_date=dates[0] if dates else None,
                latest_date=dates[-1] if dates else None,
                obs_count=len(group),
                dates=tuple(dates),
                values=tuple(values),
                provenance_id=group[0].provenance,
                measurement_method=group[0].measurement_method,
                unit=group[0].unit,
                scaling_factor=group[0].scaling_factor,
                facet_id=facet_id,
            )
        )
    return facets


def _split_facet_ids_for_request(
    facet_ids: AbstractSet[str],
    *,
    variables: Sequence[str],
    entities: Mapping[str, Sequence[str]],
    time_period: RoleTimePeriod,
    budget: int,
) -> list[tuple[str, ...]]:
    """Pack sorted facet ids into the fewest requests that fit ``budget``.

    The budget comes from the wrapped client. ``SdmxClient.data`` chunks
    variables separately. Let headroom be the budget less entity/period bytes.
    The reservation is ``max(longest-variable bytes, headroom // 2)``.
    A single-id chunk uses the exact joint query length without reservation.
    The packer rejects it when that query exceeds the budget with the longest variable.

    ``httpx.QueryParams`` measures the encoded facet, variable, entity, and
    period filters together. The shared ``dc_sdmx`` packer and query helpers
    keep estimates aligned with the wire. Sorting makes chunk assignment
    deterministic. Row and byte caps are enforced separately per request.
    """
    fixed_params: dict[str, str] = dict(_entity_params(entities))
    if time_period is not None:
        fixed_params["c[TIME_PERIOD]"] = _time_period_param(time_period)
    fixed_len = len(str(httpx.QueryParams(fixed_params)))
    headroom = budget - fixed_len

    def _single_variable_len(variable: str) -> int:
        params = {"c[variableMeasured]": _or_list("variableMeasured", [variable])}
        return len(str(httpx.QueryParams(params)))

    longest_variable = max(variables, key=_single_variable_len, default=None)
    longest_variable_len = (
        _single_variable_len(longest_variable) if longest_variable is not None else 0
    )
    reservation = max(longest_variable_len, headroom // 2)
    # Never negative: added only for chunks longer than one facet id, and
    # `reservation` is already >= `longest_variable_len` by construction.
    extra = reservation - longest_variable_len

    def _joint_len(chunk: Sequence[str]) -> int:
        params: dict[str, str] = {"c[facetId]": _or_list("facetId", chunk)}
        if longest_variable is not None:
            params["c[variableMeasured]"] = _or_list("variableMeasured", [longest_variable])
        params.update(fixed_params)
        return len(str(httpx.QueryParams(params)))

    def _query_len(chunk: Sequence[str]) -> int:
        length = _joint_len(chunk)
        if len(chunk) > 1:
            length += extra
        return length

    return _pack_variable_chunks(
        sorted(facet_ids),
        budget=budget,
        query_len=_query_len,
        label="facet id",
    )


# ---------------------------------------------------------------------------
# Live implementation
# ---------------------------------------------------------------------------


SDMX_TOTAL_SEMAPHORE = threading.BoundedSemaphore(config.settings().sdmx_total_concurrency)
"""Process-wide SDMX GET limit shared by Search's two ``SdmxClient`` instances.

Both live and materialise clients receive this semaphore, sized by
``SEARCH_SDMX_TOTAL_CONCURRENCY``. Its module-level identity survives config
reloads. Tests can replace this attribute. ``threading.BoundedSemaphore``
has no resize method.
"""


class LiveSdmxClient:
    """Synchronous ``EngineSdmxClient`` using one ``dc_sdmx.SdmxClient``.

    Uses the graph client's base URL, token, and timeout. Both Search SDMX
    clients share ``SDMX_TOTAL_SEMAPHORE``.
    """

    def __init__(
        self,
        *,
        base: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        _base = (base or config.settings().graph_base).rstrip("/")
        _timeout = timeout if timeout is not None else config.settings().graph_timeout_s
        self._client = SdmxClient(
            base_url=f"{_base}/core/api",
            # Read via module attribute so the value resolves at construction
            # time, after env loading, matching clients.graph.LiveGraphClient.
            token=config.settings().graph_token or None,
            timeout=_timeout,
            transport=transport,
            chunk_concurrency=config.settings().sdmx_chunk_concurrency,
            total_concurrency=SDMX_TOTAL_SEMAPHORE,
        )

    def close(self) -> None:
        """Close the underlying httpx client. Safe to call more than once."""
        self._client.close()

    def source_key(self) -> str:
        """The wrapped client's API root."""
        return self._client.source_key()

    def role_series(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, Sequence[str]],
        facet_ids: AbstractSet[str],
        time_period: RoleTimePeriod = None,
        max_rows: int,
        max_bytes: int,
        deadline: float | None = None,
    ) -> dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]]:
        """Fetch Facets filtered by facet ids and optional exact periods.

        The mixer accepts ``time_period`` as its sole time filter. Facet ids
        are split to fit the URL budget. ``SdmxClient.data`` chunks variables
        separately. Row and byte
        caps apply to each data request.

        The client checks each returned ``facet_id`` and raises
        ``GraphInfraError`` for an unrequested id. Shared validation rejects
        duplicate variables to prevent double-counted observations. An
        upstream variable raises ``UpstreamVariableRoutedToSdmx``
        (`assert_dfo_role_variables`).
        """
        _validate_role_series_params(
            variables=variables, facet_ids=facet_ids, time_period=time_period
        )
        assert_dfo_role_variables(variables)
        entities_arg = {role: list(dcids) for role, dcids in entities.items()}
        chunks = _split_facet_ids_for_request(
            facet_ids,
            variables=variables,
            entities=entities_arg,
            time_period=time_period,
            budget=self._client.url_budget_bytes,
        )

        by_variable: dict[str, dict[tuple[tuple[str, str], ...], list[SdmxObservation]]] = {}
        for chunk in chunks:
            chunk_ids = set(chunk)
            with _translate_sdmx_errors():
                result = self._client.data(
                    variables=list(variables),
                    entities=entities_arg,
                    facet_ids=list(chunk),
                    time_period=(
                        time_period
                        if time_period is None or time_period == "LATEST"
                        else list(time_period)
                    ),
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    deadline=deadline,
                )

            dropped = 0
            for row in result.rows:
                if row.facet_id not in chunk_ids:
                    dropped += 1
                    continue
                role_tuple = tuple(sorted(row.role_values.items()))
                by_variable.setdefault(row.variable, {}).setdefault(role_tuple, []).append(row)
            if dropped:
                raise GraphInfraError(
                    f"role_series requested facet_ids {sorted(chunk_ids)!r} but "
                    f"the wire returned {dropped} row(s) under a different "
                    "facet_id; its c[facetId] filter is not authoritative"
                )

        return {
            variable: {
                role_tuple: _facets_from_sdmx_rows(rows) for role_tuple, rows in role_groups.items()
            }
            for variable, role_groups in by_variable.items()
        }

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        time_period: str | Sequence[str] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        """Return distinct component values. Payload size follows component cardinality.

        ``max_values`` and ``max_bytes`` narrow the bound, and ``read_seconds`` bounds the read
        from when it holds its GET slot (``dc_sdmx.SdmxClient.availability``). A UN System Data
        Commons variable raises ``UpstreamVariableRoutedToSdmx``.
        """
        assert_no_un_variables(method="availability_values", variables=variables)
        filters_arg = (
            {
                role: filt if isinstance(filt, ContainedIn) else list(filt)
                for role, filt in filters.items()
            }
            if filters
            else None
        )
        if time_period is None or isinstance(time_period, str):
            time_period_arg = time_period
        else:
            time_period_arg = list(time_period)
        with _translate_sdmx_errors():
            result = self._client.availability(
                variables=list(variables),
                component=component,
                filters=filters_arg,
                time_period=time_period_arg,
                max_values=max_values,
                max_bytes=max_bytes,
                deadline=deadline,
                read_seconds=read_seconds,
            )
        return list(result.components.get(component, ()))

    def subject_observations(
        self,
        *,
        variables: Sequence[str],
        places: Sequence[str],
        facet_ids: Sequence[str] | None = None,
        latest_only: bool = True,
        max_rows: int | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[SdmxObservation]]]:
        """Fetch the subject scope with buffered ``data()`` reads naming ``places``, in
        URL-budget slices (``dc_sdmx.place_slices``) one after another.

        The coverage probe bounds the expected rows to roughly one per entity. ``max_rows``
        bounds the rows of every slice together and ``max_bytes`` each slice's body. A row or
        byte cap trip aborts the read. A UN System Data Commons variable raises
        ``UpstreamVariableRoutedToSdmx``.
        """
        assert_no_un_variables(method="subject_observations", variables=variables)
        time_period: Literal["LATEST"] | None = "LATEST" if latest_only else None
        rows: list[SdmxObservation] = []
        with _translate_sdmx_errors():
            for place_slice in place_slices(
                variables,
                role="observationAbout",
                places=places,
                facet_ids=facet_ids,
                time_period=time_period,
                url_budget_bytes=self._client.url_budget_bytes,
            ):
                result = self._client.data(
                    variables=list(variables),
                    entities={"observationAbout": list(place_slice)},
                    facet_ids=list(facet_ids) if facet_ids else None,
                    time_period=time_period,
                    max_rows=None if max_rows is None else max(max_rows - len(rows), 0),
                    max_bytes=max_bytes,
                )
                rows.extend(result.rows)
        by_variable: dict[str, dict[str, list[SdmxObservation]]] = {}
        for row in rows:
            entity = row.role_values.get("observationAbout")
            if entity is None:
                # A subject row names its place, so a row without one is malformed.
                raise GraphInfraError(
                    f"an SDMX subject row for {row.variable!r} carries no observationAbout"
                )
            by_variable.setdefault(row.variable, {}).setdefault(entity, []).append(row)
        return by_variable
