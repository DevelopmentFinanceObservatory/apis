"""Bounded, generation-keyed single-flight cache for ``role_series``.

Search wraps its shared ``EngineSdmxClient`` here. The cache stores real
Facets per ``(variable, facet_id)`` and keys requests by role entities,
periods, and row/byte caps. Hits require an unchanged graph generation and
an unexpired entry. ``availability_values`` and ``subject_observations``
pass through uncached. This module must not import ``search.eval``.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet

import search.engine.config as config  # module-attribute access for runtime-resolved values
from dc_sdmx import EntityFilter, SdmxObservation
from dfo_platform.single_flight_cache import SingleFlightCache
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import (
    EngineSdmxClient,
    RoleTimePeriod,
    RoleTuple,
    _validate_role_series_params,
)
from search.engine.errors import GraphInfraError

# Matches the deadline wording in dc_sdmx.client. A waiter whose deadline
# expires before the fetch raises the same TimeoutError as an uncached read.
_WAITER_DEADLINE_EXCEEDED_DETAIL = "the SDMX read exceeded its absolute caller deadline"

# SDMX role filters use set membership, so the sorted, de-duplicated key ignores order.
_EntitiesKey = tuple[tuple[str, tuple[str, ...]], ...]

# Exact-period OR filters are order-independent. None keys full history separately.
_TimePeriodKey = tuple[str, ...] | str | None

# One cache entry per variable and facet, even when a wire request fetches several.
_SeriesKey = tuple[str, _EntitiesKey, str, _TimePeriodKey, int | None, int | None]


def _entities_key(entities: Mapping[str, Sequence[str]]) -> _EntitiesKey:
    return tuple(sorted((role, tuple(sorted(set(dcids)))) for role, dcids in entities.items()))


def _time_period_key(time_period: RoleTimePeriod) -> _TimePeriodKey:
    if time_period is None or time_period == "LATEST":
        return time_period
    return tuple(sorted(set(time_period)))


class CachingSdmxClient:
    """Cache ``role_series`` Facets across requests, keyed by graph generation.

    One ``SingleFlightCache`` backs ``stats()``, ``clear()``, and the
    per-variable/facet entries. Concurrent misses share a fetch.
    """

    def __init__(
        self,
        inner: EngineSdmxClient,
        *,
        maxsize: int | None = None,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        _maxsize = config.settings().sdmx_facet_cache_size if maxsize is None else maxsize
        self._series_cache: SingleFlightCache[_SeriesKey, dict[RoleTuple, list[Facet]]] = (
            SingleFlightCache(
                maxsize=_maxsize,
                generation=generation,
                clock=clock,
                timeout_message=_WAITER_DEADLINE_EXCEEDED_DETAIL,
            )
        )

    def close(self) -> None:
        """Delegate to the wrapped client's own close(), when it has one.
        Safe to call more than once (LiveSdmxClient.close() itself is)."""
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def source_key(self) -> str:
        """The wrapped client's backend."""
        return self._inner.source_key()

    def stats(self) -> Mapping[str, int]:
        """Hit, miss, eviction, and entry counts for graph-generation registration."""
        return self._series_cache.stats()

    def clear(self) -> None:
        """Clears the `role_series` cache."""
        self._series_cache.clear()

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
    ) -> dict[str, dict[RoleTuple, list[Facet]]]:
        """Cache each variable/facet cell while batching compatible misses.

        Cells with the same missing-variable list share one inner call. Results
        Facets are attributed by their ``facet_id``. An unrequested id raises
        ``GraphInfraError``. Sorted facet ids make merges deterministic.
        Empty cells are cached. Shared validation runs before lookup, so invalid
        parameters raise ``ValueError`` under every cache setting.
        """
        _validate_role_series_params(
            variables=variables, facet_ids=facet_ids, time_period=time_period
        )
        if not self._series_cache.enabled:
            return self._inner.role_series(
                variables=variables,
                entities=entities,
                facet_ids=facet_ids,
                time_period=time_period,
                max_rows=max_rows,
                max_bytes=max_bytes,
                deadline=deadline,
            )
        entities_key = _entities_key(entities)
        time_key = _time_period_key(time_period)
        sorted_facet_ids = sorted(facet_ids)
        keys: list[_SeriesKey] = [
            (variable, entities_key, facet_id, time_key, max_rows, max_bytes)
            for facet_id in sorted_facet_ids
            for variable in variables
        ]

        def fetch(
            missing_keys: list[_SeriesKey],
        ) -> dict[_SeriesKey, dict[RoleTuple, list[Facet]]]:
            # Preserve variable order within each facet's missing cells.
            missing_variables_by_facet: dict[str, list[str]] = {}
            for variable, _entities, facet_id, _time, _max_rows, _max_bytes in missing_keys:
                missing_variables_by_facet.setdefault(facet_id, []).append(variable)

            # Fetch facets with identical missing-variable sets together.
            groups: dict[tuple[str, ...], list[str]] = {}
            for facet_id, missing_variables in missing_variables_by_facet.items():
                groups.setdefault(tuple(missing_variables), []).append(facet_id)

            resolved: dict[_SeriesKey, dict[RoleTuple, list[Facet]]] = {}
            for missing_variables, group_facet_ids in groups.items():
                group_facet_id_set = set(group_facet_ids)
                result = self._inner.role_series(
                    variables=list(missing_variables),
                    entities=entities,
                    facet_ids=group_facet_id_set,
                    time_period=time_period,
                    max_rows=max_rows,
                    max_bytes=max_bytes,
                    deadline=deadline,
                )
                by_variable_facet: dict[tuple[str, str], dict[RoleTuple, list[Facet]]] = {}
                for variable, role_groups in result.items():
                    for role_tuple, facets in role_groups.items():
                        for facet in facets:
                            if facet.facet_id not in group_facet_id_set:
                                raise GraphInfraError(
                                    f"role_series requested facet_ids "
                                    f"{sorted(group_facet_id_set)!r} for variable "
                                    f"{variable!r} but the wire returned facet_id "
                                    f"{facet.facet_id!r}, outside that request; its "
                                    "facet attribution is not authoritative"
                                )
                            cell = by_variable_facet.setdefault((variable, facet.facet_id), {})
                            cell.setdefault(role_tuple, []).append(facet)
                for variable in missing_variables:
                    for facet_id in group_facet_ids:
                        key = (variable, entities_key, facet_id, time_key, max_rows, max_bytes)
                        resolved[key] = by_variable_facet.get((variable, facet_id), {})
            return resolved

        resolved = self._series_cache.get_or_fetch_many(
            keys, fetch, deadline=deadline, copy_result=copy.deepcopy
        )
        merged: dict[str, dict[RoleTuple, list[Facet]]] = {}
        for facet_id in sorted_facet_ids:
            for variable in variables:
                key = (variable, entities_key, facet_id, time_key, max_rows, max_bytes)
                role_groups = resolved[key]
                if not role_groups:
                    continue
                variable_entry = merged.setdefault(variable, {})
                for role_tuple, facets in role_groups.items():
                    variable_entry.setdefault(role_tuple, []).extend(facets)

        return merged

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
        """Uncached passthrough to the wrapped client."""
        return self._inner.availability_values(
            variables=variables,
            component=component,
            filters=filters,
            time_period=time_period,
            max_values=max_values,
            max_bytes=max_bytes,
            deadline=deadline,
            read_seconds=read_seconds,
        )

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
        """Uncached passthrough to the wrapped client."""
        return self._inner.subject_observations(
            variables=variables,
            places=places,
            facet_ids=facet_ids,
            latest_only=latest_only,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )
