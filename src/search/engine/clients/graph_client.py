"""Search adapter over the origin graph client, which routes each read between our mixer and the
UN System Data Commons and admits every facet against the request's origin catalogue.

The shared client supplies primitive, observation-facet and `materialise.graph` reads. This adapter
adds Search's recall over our resolve and the UN's (`detect_svs`), the `force_full_facets` upgrade
on observation reads, observation counts, subject-page coverage, and the packaged code index's
label override. Search uses one attempt by default and substitutes `GraphReadFailure`,
`GraphDeadlineFailure` and `GraphCapExceededError`, catchable as both Search and `dc_graph` errors.

The adapter also satisfies `materialise.graph.GraphReader`. Separate node-arc pools keep role
refresh capacity free during large options enumerations. `node_arc_total_concurrency`, configured
by `SEARCH_NODE_ARC_TOTAL_CONCURRENCY`, bounds role refreshes and `node_arcs_batch`.
`node_arc_many_total_concurrency` bounds `node_arcs_many` and `node_names_many` at the graph
client's default. The adapter also satisfies the eval graph protocol. This module must not import
`search.eval`.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final, Self

import cachetools
import pydantic

import dc_graph
from code_index.labels import normalize_label
from code_index.shared import get_index
from dc_sdmx import SdmxClient
from dfo_platform.request_deadline import submit_detached
from graph_generation import UNKNOWN_GENERATION, GenerationProvider, GenerationStamp
from graph_generation.stamp import ttu_from_stamp_key
from origin_catalogue.context import current_catalogue
from origin_catalogue.graph import (
    ARCS_CACHE_SIZE,
    LABEL_TYPE_CACHE_SIZE,
    OBS_CACHE_SIZE,
    OBS_PROPERTIES_CACHE_SIZE,
    UN_BASE_URL,
    OriginGraphClient,
    UnAvailabilityClient,
    build_graph_client,
)
from search.engine.clients.graph import (
    DetectedSV,
    Detection,
    DetectLegs,
    Facet,
    SubjectsCoverage,
    SubjectsPage,
    _force_full_facets,
    admit_un_candidates,
    recall_hits,
    settled_un,
    stand_candidates,
)
from search.engine.config import USER_AGENT, SearchSettings
from search.engine.config import settings as search_settings
from search.engine.errors import (
    GraphCapExceededError,
    GraphDeadlineFailure,
    GraphInfraError,
    GraphReadFailure,
)

logger = logging.getLogger(__name__)

_ttu_from_stamp_key = ttu_from_stamp_key

_SUBJECTS_CACHE_MAXSIZE: Final[int] = ARCS_CACHE_SIZE
"""Complete subject pages kept, by lead, property and cap, the size of the default arcs cache.
A resubmit checks each posted group's members against one, so a warm resubmit reads none."""

_UN_RESOLVE_WORKERS: Final[int] = 16
"""Threads that run UN System Data Commons resolves, process-wide. A resolve holds one for at
most `SEARCH_UN_RESOLVE_TIMEOUT_S` (3 s by default), so 16 sustain about five resolves a
second when every one runs to its deadline. Past that, new ones queue. A queued resolve spends
its deadline in the queue and fails without a request once it has run out."""

_UN_RESOLVE_POOL: Final = ThreadPoolExecutor(
    max_workers=_UN_RESOLVE_WORKERS, thread_name_prefix="detect-un"
)


def _first_type_of(raw_type_of: object) -> str | None:
    if isinstance(raw_type_of, list):
        first = raw_type_of[0] if raw_type_of else None
        return first if isinstance(first, str) else None
    if isinstance(raw_type_of, str):
        return raw_type_of
    return None


def _parse_detect_candidates(body: dict) -> list[DetectedSV]:
    try:
        entities = body.get("entities", [])
        if not isinstance(entities, list):
            raise GraphInfraError(f"graph resolve returned non-list entities: {entities!r}")
        result: list[DetectedSV] = []
        for entity in entities:
            candidates = entity.get("candidates", [])
            if not isinstance(candidates, list):
                raise GraphInfraError(f"graph resolve returned non-list candidates: {candidates!r}")
            for cand in candidates:
                dcid = cand.get("dcid")
                if not dcid:
                    continue
                metadata = cand.get("metadata", {}) or {}
                if not isinstance(metadata, dict):
                    raise GraphInfraError(
                        f"graph resolve candidate metadata is not a dict: {metadata!r}"
                    )
                raw_score = metadata.get("score")
                try:
                    score = float(raw_score)  # ty: ignore[invalid-argument-type]
                    if not math.isfinite(score):
                        raise ValueError("score is not finite")
                except (TypeError, ValueError):
                    raise GraphInfraError(
                        f"graph resolve candidate {dcid!r} has a missing/non-numeric score: "
                        f"{raw_score!r}"
                    ) from None
                result.append(
                    DetectedSV(
                        dcid=dcid,
                        score=score,
                        type_of=_first_type_of(cand.get("typeOf")),
                        sentence=metadata.get("sentence"),
                    )
                )
        return result
    except GraphInfraError:
        raise
    except (AttributeError, TypeError, pydantic.ValidationError) as exc:
        raise GraphInfraError(f"graph resolve returned malformed body: {exc}") from exc


def _build_dc_graph_client(
    *,
    base: str,
    token: str,
    un_base_url: str,
    timeout: float | None,
    max_attempts: int,
    generation: GenerationProvider,
    timer: Callable[[], float],
    arcs_cache_size: int,
    obs_properties_cache_size: int,
    observation_properties_flight_timeout_s: float | None,
    obs_cache_size: int,
    node_arc_total_concurrency: int | None,
    node_arc_call_concurrency: int | None,
) -> OriginGraphClient:
    """Build the wrapped client, omitting optional settings when their value is `None` so its
    defaults remain authoritative. `base` is a service root, to which `/core/api` is appended.
    `un_base_url` is the UN deployment's API root, passed as given, which also serves the UN
    SDMX availability read.
    """
    kwargs: dict[str, object] = {
        "base_url": f"{base.rstrip('/')}/core/api",
        "token": token or None,
        "un_base_url": un_base_url,
        "un_availability": SdmxClient(base_url=un_base_url)
        if timeout is None
        else SdmxClient(base_url=un_base_url, timeout=timeout),
        "max_attempts": max_attempts,
        "user_agent": USER_AGENT,
        "generation": generation,
        "timer": timer,
        "label_cache_size": LABEL_TYPE_CACHE_SIZE,
        "type_cache_size": LABEL_TYPE_CACHE_SIZE,
        "arcs_cache_size": arcs_cache_size,
        "obs_properties_cache_size": obs_properties_cache_size,
        "obs_cache_size": obs_cache_size,
        "single_value_cache_enabled": True,
        "read_error_cls": GraphReadFailure,
        "deadline_error_cls": GraphDeadlineFailure,
        "cap_error_cls": GraphCapExceededError,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if observation_properties_flight_timeout_s is not None:
        kwargs["observation_properties_flight_timeout_s"] = observation_properties_flight_timeout_s
    if node_arc_total_concurrency is not None:
        kwargs["node_arc_total_concurrency"] = node_arc_total_concurrency
    if node_arc_call_concurrency is not None:
        kwargs["node_arc_call_concurrency"] = node_arc_call_concurrency
    # A dict lets None-valued options be omitted to match the wrapped client's strict types.
    return build_graph_client(**kwargs)  # ty: ignore[invalid-argument-type]


class LiveGraphClient:
    """Synchronous Search graph client. Async callers use `asyncio.to_thread`. Appends `/core/api`
    to `base`, while `un_base_url` is already an API root. `max_attempts` defaults to 1, while
    `dc_graph.GraphClient` defaults to 3. The UN backend's caches key on the request's bound
    catalogue build. `observation_properties_flight_timeout_s` set to `None` keeps
    `dc_graph.GraphClient`'s default.
    """

    def __init__(
        self,
        *,
        base: str,
        timeout: float | None = None,
        token: str = "",
        un_base_url: str = UN_BASE_URL,
        max_attempts: int = 1,
        generation: GenerationProvider = UNKNOWN_GENERATION,
        timer: Callable[[], float] | None = None,
        obs_cache_size: int = OBS_CACHE_SIZE,
        detect_cache_size: int = 256,
        arcs_cache_size: int = ARCS_CACHE_SIZE,
        obs_properties_cache_size: int = OBS_PROPERTIES_CACHE_SIZE,
        observation_properties_flight_timeout_s: float | None = None,
        node_arc_total_concurrency: int | None = None,
        node_arc_call_concurrency: int | None = None,
        detect_k: int = 80,
        resolve_target: str = "base_and_custom",
        un_resolve_timeout_s: float = 3.0,
    ) -> None:
        self._detect_k = detect_k
        self._resolve_target = resolve_target
        self._un_resolve_timeout_s = un_resolve_timeout_s
        self._generation = generation
        self._timer: Callable[[], float] = timer if timer is not None else time.monotonic
        self._client = _build_dc_graph_client(
            base=base,
            token=token,
            un_base_url=un_base_url,
            timeout=timeout,
            max_attempts=max_attempts,
            generation=generation,
            timer=self._timer,
            arcs_cache_size=arcs_cache_size,
            obs_properties_cache_size=obs_properties_cache_size,
            observation_properties_flight_timeout_s=observation_properties_flight_timeout_s,
            obs_cache_size=obs_cache_size,
            node_arc_total_concurrency=node_arc_total_concurrency,
            node_arc_call_concurrency=node_arc_call_concurrency,
        )
        self._detect_cache: cachetools.TLRUCache[
            tuple[str, str, bool, frozenset[str] | None, str, str, int], Detection
        ] = cachetools.TLRUCache(
            maxsize=detect_cache_size, ttu=_ttu_from_stamp_key, timer=self._timer
        )
        self._subjects_cache: cachetools.TLRUCache[tuple[str, str, int, str], SubjectsPage] = (
            cachetools.TLRUCache(
                maxsize=_SUBJECTS_CACHE_MAXSIZE, ttu=_ttu_from_stamp_key, timer=self._timer
            )
        )
        self._cache_lock = threading.Lock()
        # UN resolves queued or running, by query, so concurrent detections of one query share one.
        self._un_resolves: dict[str, Future[tuple[DetectedSV, ...] | None]] = {}
        self._detect_hits = 0
        self._detect_misses = 0

    @classmethod
    def from_search_settings(
        cls,
        settings: SearchSettings | None = None,
        *,
        generation: GenerationProvider = UNKNOWN_GENERATION,
    ) -> LiveGraphClient:
        """Build from Search settings, or from the supplied `SearchSettings` instance. A shared
        observation-property read gets the longest hard cap of a request that makes one, a
        `dataset_resubmit`'s: `resubmit_full_read_timeout_s` plus `finalise_grace_s`."""
        s = settings if settings is not None else search_settings()
        return cls(
            base=s.graph_base,
            timeout=s.graph_timeout_s,
            token=s.graph_token,
            un_base_url=s.un_graph_base_url,
            max_attempts=1,
            generation=generation,
            obs_cache_size=s.obs_cache_size,
            detect_cache_size=s.detect_cache_size,
            arcs_cache_size=s.arcs_cache_size,
            obs_properties_cache_size=s.obs_properties_cache_size,
            observation_properties_flight_timeout_s=(
                s.resubmit_full_read_timeout_s + s.finalise_grace_s
            ),
            node_arc_total_concurrency=s.node_arc_total_concurrency,
            node_arc_call_concurrency=s.node_arc_call_concurrency,
            detect_k=s.detect_k,
            resolve_target=s.resolve_target,
            un_resolve_timeout_s=s.un_resolve_timeout_s,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def node_type(self, dcid: str) -> str | None:
        return self._client.node_type(dcid)

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        return self._client.node_types_batch(dcids)

    def node_arcs(self, dcid: str) -> dict | None:
        return self._client.node_arcs(dcid)

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        return self._client.node_arcs_batch(dcids, deadline=deadline)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        return self._client.node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        return self._client.node_names_many(dcids, deadline=deadline)

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._client.resolve_by_arc(names, arc=arc, deadline=deadline)

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._client.child_places_batch(parent_dcids, child_type, deadline=deadline)

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        return self._client.node_observation_properties_batch(dcids)

    def child_dcids(self, parent_dcid: str) -> list[str]:
        return self._client.child_dcids(parent_dcid)

    def enum_members(self, property_dcid: str) -> list[str]:
        return self._client.enum_members(property_dcid)

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        return self._client.currency_units(deadline=deadline)

    def entities_with_data(
        self, variable_dcids: list[str], entity_dcids: list[str], *, max_bytes: int | None = None
    ) -> list[str]:
        return self._client.entities_with_data(variable_dcids, entity_dcids, max_bytes=max_bytes)

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return self._client.entities_with_data_by_variable(
            variable_dcids, entity_dcids, deadline=deadline
        )

    def exists(self, dcid: str) -> bool:
        return self._client.exists(dcid)

    def exists_batch(self, dcids: list[str]) -> set[str]:
        return self._client.exists_batch(dcids)

    def variable_facets_batch(
        self, *, stat_var: str, entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, list[dc_graph.Facet]]:
        return self._client.variable_facets_batch(
            stat_var=stat_var, entities=entities, deadline=deadline
        )

    def observation_facets_batch_many(
        self, *, stat_vars: Sequence[str], entities: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, list[dc_graph.Facet]]]:
        return self._client.observation_facets_batch_many(
            stat_vars=stat_vars, entities=entities, deadline=deadline
        )

    def un_observation_reader(self) -> OriginGraphClient:
        """The wrapped origin client, which reads UN rows."""
        return self._client

    def un_availability(self) -> UnAvailabilityClient | None:
        """The wrapped origin client's UN availability read."""
        return self._client.un_availability()

    def generation_stamp(self) -> GenerationStamp:
        """The wrapped origin client's stamp, which its caches key on."""
        return self._client.generation_stamp()

    def source_key(self) -> str:
        """The wrapped origin client's backends."""
        return self._client.source_key()

    def declares_facet_defaults(self) -> bool:
        return self._client.declares_facet_defaults()

    def default_facet_policy(
        self, *, stat_var: str, axis: str
    ) -> dc_graph.FacetDefaultPolicy | None:
        return self._client.default_facet_policy(stat_var=stat_var, axis=axis)

    def node_arc_values(
        self, dcid: str, *, properties: Sequence[str]
    ) -> dict[str, tuple[str, ...]]:
        return self._client.node_arc_values(dcid, properties=properties)

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> dc_graph.SubjectPage:
        return self._client.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )

    def node_label(self, dcid: str) -> str | None:
        raw_label = self._client.node_label(dcid)
        if raw_label is None:
            return None
        return normalize_label(dcid, raw_label, index=get_index())

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        raw = self._client.node_labels_batch(dcids)
        index = get_index()
        return {dcid: normalize_label(dcid, label, index=index) for dcid, label in raw.items()}

    def withheld_provenances(
        self, variables: Sequence[str], entities: Sequence[str], *, deadline: float | None = None
    ) -> tuple[str, ...]:
        """Provenances holding data for `variables` at `entities` that the request's
        catalogue does not admit, from one read per backend bounded by `deadline`
        (`OriginGraphClient.withheld_provenances`)."""
        return self._client.withheld_provenances(variables, entities, deadline=deadline)

    def detect_svs(self, query: str) -> Detection:
        """Our resolve, and the UN resolve started beside it, each admitted by the request's
        catalogue (`admit_candidates`). Returns once ours finishes; `Detection.un` settles when
        the UN resolve does.

        The cache keys on the generation stamp and the catalogue's build, UN setting and
        source scope, since admission ran before the cut and the scope adds rows past it
        (`with_rival_families`). A detection enters the cache when its UN resolve
        succeeds, before `Detection.un` settles, so a failed or unfinished one is never served
        from it.
        """
        catalogue = current_catalogue()
        stamp = self._generation.stamp()
        cache_key = (
            stamp.key,
            catalogue.build_id,
            catalogue.un_system_enabled,
            catalogue.scope,
            query,
            self._resolve_target,
            self._detect_k,
        )
        with self._cache_lock:
            if cache_key in self._detect_cache:
                self._detect_hits += 1
                return self._detect_cache[cache_key]
            self._detect_misses += 1

        resolved = self._start_un(query)
        # A running future cannot be cancelled, so a waiter that gives up (`asyncio.wrap_future`
        # cancels its source) leaves it for `settle` to finish.
        un: Future[tuple[DetectedSV, ...] | None] = Future()
        un.set_running_or_notify_cancel()
        hits = recall_hits(self._resolve_ours(query), catalogue=catalogue, detect_k=self._detect_k)
        detection = Detection(
            candidates=stand_candidates(hits, catalogue=catalogue), un=un, hits=hits
        )

        def settle(done: Future[tuple[DetectedSV, ...] | None]) -> None:
            try:
                admitted = admit_un_candidates(
                    done.result(), catalogue=catalogue, detect_k=self._detect_k
                )
            except Exception:
                logger.exception("UN System Data Commons resolve admission raised")
                admitted = None
            if admitted is not None:
                with self._cache_lock:
                    self._detect_cache[cache_key] = detection
            un.set_result(admitted)

        resolved.add_done_callback(settle)
        return detection

    def detect_legs(self, query: str) -> DetectLegs:
        """Both resolves, unadmitted, once both finish (`detect_svs` for the resolves)."""
        un = self._start_un(query)
        ours = self._resolve_ours(query)
        return DetectLegs(ours=ours, un=un.result())

    def _start_un(self, query: str) -> Future[tuple[DetectedSV, ...] | None]:
        """The UN resolve of `query`, unadmitted, on the process-wide UN resolve pool in the
        caller's context less its request deadline (`submit_detached`), without waiting for it,
        when the request's catalogue has the UN origin on, and an empty resolve when it is off. A detection of a query whose resolve is still
        queued or running shares it, since the resolve posts only `query` and no other request
        state changes its answer. Its deadline starts at submission, so time in the queue counts
        against it.
        """
        if not current_catalogue().un_system_enabled:
            return settled_un(())
        with self._cache_lock:
            resolve = self._un_resolves.get(query)
            if resolve is not None:
                return resolve
            resolve = submit_detached(
                _UN_RESOLVE_POOL,
                self._resolve_un,
                query,
                deadline=time.monotonic() + self._un_resolve_timeout_s,
            )
            self._un_resolves[query] = resolve
        resolve.add_done_callback(lambda done: self._forget_un(query, done))
        return resolve

    def _forget_un(self, query: str, done: Future[tuple[DetectedSV, ...] | None]) -> None:
        with self._cache_lock:
            if self._un_resolves.get(query) is done:
                del self._un_resolves[query]

    def _resolve_ours(self, query: str) -> tuple[DetectedSV, ...]:
        body = self._client.resolve_indicators(query, target=self._resolve_target)
        return tuple(_parse_detect_candidates(body))

    def _resolve_un(self, query: str, *, deadline: float) -> tuple[DetectedSV, ...] | None:
        """The UN resolve's candidates, or `None` when it fails. The resolve may outlive the
        request, so an unexpected error is logged with its traceback and returns `None`."""
        try:
            body = self._client.resolve_indicators(
                query, target="custom_only", un=True, deadline=deadline
            )
            return tuple(_parse_detect_candidates(body))
        except GraphInfraError as exc:
            logger.warning("UN System Data Commons resolve failed for %r: %s", query, exc)
            return None
        except Exception:
            logger.exception("UN System Data Commons resolve raised")
            return None

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        return (
            self.observation_facets_batch([stat_var], [entity], needs_dates)
            .get(stat_var, {})
            .get(entity, [])
        )

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        return self._client.observation_facets_batch(
            stat_vars,
            entities,
            needs_dates or _force_full_facets.get(),
            max_bytes=max_bytes,
            deadline=deadline,
        )

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, SubjectsPage]:
        """Map the shared client's subject pages and coverage flags to Search's result types.

        A `complete` page is cached under the generation stamp, so a later read of the same
        lead, property and cap reads nothing. A capped or budget-cut page is read again,
        since another batch may complete it."""
        stamp_key = self._generation.stamp().key
        with self._cache_lock:
            cached = {
                dcid: page
                for dcid in dcids
                if (page := self._subjects_cache.get((stamp_key, property, max_results, dcid)))
                is not None
            }
        missing = [dcid for dcid in dcids if dcid not in cached]
        if not missing:
            return cached
        result = self._client.node_subjects_many(
            missing, property=property, max_results=max_results, deadline=deadline
        )
        read = {
            dcid: SubjectsPage(dcids=page_dcids, coverage=SubjectsCoverage(coverage))
            for dcid, (page_dcids, coverage) in result.items()
        }
        with self._cache_lock:
            for dcid, page in read.items():
                if page.coverage is SubjectsCoverage.complete:
                    self._subjects_cache[(stamp_key, property, max_results, dcid)] = page
        return {**cached, **read}

    # -- stats/clear: this instance's own detect cache, merged with
    # dc_graph.GraphClient's own (label, type, arcs, observations, observation
    # properties, single-value memos). --

    def stats(self) -> Mapping[str, int]:
        with self._cache_lock:
            own = {
                "detect_hits": self._detect_hits,
                "detect_misses": self._detect_misses,
                "detect_entries": len(self._detect_cache),
                "subjects_entries": len(self._subjects_cache),
                "un_resolves_in_flight": len(self._un_resolves),
            }
        return {**self._client.stats(), **own}

    def clear(self) -> None:
        with self._cache_lock:
            self._detect_cache.clear()
            self._subjects_cache.clear()
        self._client.clear()
