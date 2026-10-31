"""SQLite read-through cache for graph calls made during prompt-ladder captures.

One capture supplies a point-in-time graph snapshot for cell replays. WAL rows
retain their write-time schema. `node_subjects_many` validates `coverage` and raises
`StaleCaptureError` when that field is missing. The cache proxies engine and
evaluation graph surfaces, with unknown methods failing closed.

Negative results are cached for cross-cell determinism. Inner-client transport,
HTTP, and JSON errors propagate. A 200 with an empty body is cached as a negative,
indistinguishable from true absence. Facet keys use resolved `needs_dates`, including
`force_full_facets` context. A bounded semaphore of 4 guards inner calls. One lock
guards the SQLite connection. Read-only mode raises `CacheMiss` on every miss and
lets `ladder report --rescore` rebuild scores from stored calls and cached reads.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import TypeAdapter

from dc_graph import ArcTarget, FacetDefaultPolicy, parse_arc_targets
from graph_generation import GenerationStamp
from origin_catalogue import current_catalogue
from origin_catalogue.graph import UnAvailabilityClient
from search.engine.clients.graph import (
    DetectedSV,
    Detection,
    DetectLegs,
    EngineGraphClient,
    Facet,
    SubjectsCoverage,
    SubjectsPage,
    _force_full_facets,
    admit_legs,
)
from search.engine.config import settings

_DETECTED_SVS_ADAPTER: TypeAdapter[list[DetectedSV]] = TypeAdapter(list[DetectedSV])


class _InnerGraphClient(EngineGraphClient, Protocol):
    """Engine graph client plus `detect_legs`. The cache derives `exists_batch` from
    labels and singleton arc reads."""

    def detect_legs(self, query: str) -> DetectLegs: ...


class CacheMiss(Exception):
    """Raised by a read_only GraphCache on any cache miss, naming the key."""


class StaleCaptureError(Exception):
    """A capture's `node_subjects_many` row lacks `coverage`. Callers must recapture."""


def _key(method: str, *args: object) -> str:
    return method + "\x01" + json.dumps(args, sort_keys=True, separators=(",", ":"))


def _facet_to_json(facet: Facet) -> dict[str, Any]:
    return dataclasses.asdict(facet)


def _facet_from_json(data: dict[str, Any]) -> Facet:
    # JSON returns the tuple fields as lists.
    facet = Facet(**data)
    return dataclasses.replace(facet, dates=tuple(facet.dates), values=tuple(facet.values))


class _Store:
    """SQLite-backed key/value table for one capture's graph replay data."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS graph_cache ("
                "key TEXT PRIMARY KEY, method TEXT NOT NULL, "
                "value TEXT NOT NULL, written_at REAL NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> tuple[bool, Any]:
        """Return `(hit, value)` with the stored JSON decoded. Callers interpret its type."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM graph_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return False, None
        return True, json.loads(row[0])

    def put(self, key: str, method: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO graph_cache (key, method, value, written_at) "
                "VALUES (?, ?, ?, ?)",
                (key, method, json.dumps(value), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class GraphCache:
    """Read-through SQLite cache. `inner` answers misses for live captures or tests."""

    def __init__(
        self,
        inner: _InnerGraphClient,
        *,
        path: Path,
        read_only: bool = False,
    ) -> None:
        self._inner = inner
        self._store = _Store(path)
        self._read_only = read_only
        self._semaphore = threading.BoundedSemaphore(4)

    def __enter__(self) -> GraphCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._store.close()

    def _lookup[T](self, method: str, key_args: tuple[object, ...], fetch: Callable[[], T]) -> T:
        """Read or fetch and cache JSON-native results. Structured graph types are
        serialized separately."""
        key = _key(method, *key_args)
        hit, raw = self._store.get(key)
        if hit:
            return raw
        if self._read_only:
            raise CacheMiss(key)
        with self._semaphore:
            value = fetch()
        self._store.put(key, method, value)
        return value

    # -- EngineGraphClient surface ------------------------------------

    def node_label(self, dcid: str) -> str | None:
        return self._lookup("node_label", (dcid,), lambda: self._inner.node_label(dcid))

    def node_type(self, dcid: str) -> str | None:
        return self._lookup("node_type", (dcid,), lambda: self._inner.node_type(dcid))

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        # Passed through uncached: the live client answers from no wire read, and captures
        # recorded before resolve asked hold no rows for it.
        return self._inner.default_facet_policy(stat_var=stat_var, axis=axis)

    def node_arcs(self, dcid: str) -> dict | None:
        return self.node_arcs_batch([dcid]).get(dcid)

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        if not dcids:
            return {}
        unique = list(dict.fromkeys(dcids))
        result: dict[str, dict | None] = {}
        misses: list[str] = []
        for dcid in unique:
            hit, arcs = self._store.get(_key("node_arcs", dcid))
            if hit:
                result[dcid] = arcs
            else:
                misses.append(dcid)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("node_arcs_batch", *sorted(misses)))
            with self._semaphore:
                fetched = self._inner.node_arcs_batch(misses, deadline=deadline)
            for dcid in misses:
                arcs = fetched.get(dcid)
                self._store.put(_key("node_arcs", dcid), "node_arcs", arcs)
                result[dcid] = arcs
        return result

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        if not dcids:
            return {}
        unique = list(dict.fromkeys(dcids))
        result: dict[str, str] = {}
        misses: list[str] = []
        for dcid in unique:
            hit, label = self._store.get(_key("node_label", dcid))
            if hit:
                if label is not None:
                    result[dcid] = label
            else:
                misses.append(dcid)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("node_labels_batch", *sorted(misses)))
            with self._semaphore:
                fetched = self._inner.node_labels_batch(misses)
            for dcid in misses:
                label = fetched.get(dcid)
                self._store.put(_key("node_label", dcid), "node_label", label)
                if label is not None:
                    result[dcid] = label
        return result

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        if not dcids:
            return {}
        unique = list(dict.fromkeys(dcids))
        result: dict[str, str] = {}
        misses: list[str] = []
        for dcid in unique:
            hit, type_of = self._store.get(_key("node_type", dcid))
            if hit:
                if type_of is not None:
                    result[dcid] = type_of
            else:
                misses.append(dcid)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("node_types_batch", *sorted(misses)))
            with self._semaphore:
                fetched = self._inner.node_types_batch(misses)
            for dcid in misses:
                type_of = fetched.get(dcid)
                self._store.put(_key("node_type", dcid), "node_type", type_of)
                if type_of is not None:
                    result[dcid] = type_of
        return result

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        if not dcids:
            return {}
        unique = list(dict.fromkeys(dcids))
        result: dict[str, tuple[str, ...]] = {}
        misses: list[str] = []
        for dcid in unique:
            hit, targets = self._store.get(_key("node_observation_properties", dcid))
            if hit:
                result[dcid] = tuple(targets)
            else:
                misses.append(dcid)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("node_observation_properties_batch", *sorted(misses)))
            with self._semaphore:
                fetched = self._inner.node_observation_properties_batch(misses)
            for dcid in misses:
                targets = fetched[dcid]
                self._store.put(
                    _key("node_observation_properties", dcid),
                    "node_observation_properties",
                    list(targets),
                )
                result[dcid] = targets
        return result

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        # Not part of this cache's own key: a replay's inner client is a
        # point-in-time capture, so the deadline a caller passes changes
        # nothing about what the capture returns.
        del deadline
        if not dcids:
            return {}
        unique = list(dict.fromkeys(dcids))
        sorted_properties = sorted(properties)
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        misses: list[str] = []
        for dcid in unique:
            hit, arcs = self._store.get(_key("node_arcs_many", dcid, sorted_properties))
            if hit:
                result[dcid] = {k: tuple(v) for k, v in arcs.items()} if arcs is not None else None
            else:
                misses.append(dcid)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("node_arcs_many", *sorted(misses), sorted_properties))
            with self._semaphore:
                fetched = self._inner.node_arcs_many(misses, properties=properties)
            for dcid in misses:
                entry = fetched.get(dcid)
                self._store.put(
                    _key("node_arcs_many", dcid, sorted_properties),
                    "node_arcs_many",
                    {k: list(v) for k, v in entry.items()} if entry is not None else None,
                )
                result[dcid] = entry
        return result

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        """Cache each dcid's targets under its sorted `properties`, as `node_arcs_many` does.

        Read-only, misses are answered from the capture's whole-node rows when those give the
        full answer (`_captured_arc_targets`), so a capture made before this read existed
        replays it as live serving would, else raise `CacheMiss`."""
        del deadline
        if not dcids:
            return {}
        sorted_properties = sorted(properties)
        result: dict[str, dict[str, tuple[ArcTarget, ...]] | None] = {}
        misses: list[str] = []
        for dcid in dict.fromkeys(dcids):
            hit, arcs = self._store.get(_key("node_arc_targets_many", dcid, sorted_properties))
            if hit:
                result[dcid] = (
                    None
                    if arcs is None
                    else {
                        prop: tuple(
                            ArcTarget(value=value, name=name, types=tuple(types))
                            for value, name, types in targets
                        )
                        for prop, targets in arcs.items()
                    }
                )
            else:
                misses.append(dcid)
        if misses and self._read_only:
            captured = self._captured_arc_targets(misses, properties)
            if captured is None:
                raise CacheMiss(_key("node_arc_targets_many", *sorted(misses), sorted_properties))
            result.update(captured)
            misses = []
        if misses:
            with self._semaphore:
                fetched = self._inner.node_arc_targets_many(misses, properties=properties)
            for dcid in misses:
                entry = fetched.get(dcid)
                self._store.put(
                    _key("node_arc_targets_many", dcid, sorted_properties),
                    "node_arc_targets_many",
                    None
                    if entry is None
                    else {
                        prop: [[t.value, t.name, list(t.types)] for t in targets]
                        for prop, targets in entry.items()
                    },
                )
                result[dcid] = entry
        return result

    def _captured_arc_targets(
        self, dcids: Sequence[str], properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None] | None:
        """Each of `dcids`' `properties` from its stored `node_arcs` row, the whole node a
        `->*` read answered, with each `isPartOf` parent's name and types from the row or from
        the parent's stored `node_label` and `node_type` rows. `None` for a dcid stored with no
        arcs, and `None` overall when a dcid has no row or a parent lacks its name or types,
        since the answer would then differ from the live read's."""
        answered: dict[str, dict[str, tuple[ArcTarget, ...]] | None] = {}
        for dcid in dcids:
            hit, arcs = self._store.get(_key("node_arcs", dcid))
            if not hit:
                return None
            if arcs is None:
                answered[dcid] = None
                continue
            targets = parse_arc_targets(arcs, properties)
            parents: list[ArcTarget] = []
            for parent in targets.get("isPartOf", ()):
                if (named := self._captured_parent(parent)) is None:
                    return None
                parents.append(named)
            if "isPartOf" in targets:
                targets["isPartOf"] = tuple(parents)
            answered[dcid] = targets
        return answered

    def _captured_parent(self, parent: ArcTarget) -> ArcTarget | None:
        """`parent` with the name and types its stored rows give, or `None` without them."""
        name, types = parent.name, parent.types
        if name is None:
            hit, name = self._store.get(_key("node_label", parent.value))
            if not hit or name is None:
                return None
        if not types:
            hit, type_of = self._store.get(_key("node_type", parent.value))
            if not hit or type_of is None:
                return None
            types = (type_of,)
        return ArcTarget(value=parent.value, name=name, types=types)

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Cache each dcid's name tuple separately, so cached dcids bypass inner reads.
        `deadline` follows `node_arcs_many`."""
        del deadline
        if not dcids:
            return {}
        unique = list(dict.fromkeys(dcids))
        result: dict[str, tuple[str, ...]] = {}
        misses: list[str] = []
        for dcid in unique:
            hit, names = self._store.get(_key("node_names_many", dcid))
            if hit:
                result[dcid] = tuple(names)
            else:
                misses.append(dcid)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("node_names_many", *sorted(misses)))
            with self._semaphore:
                fetched = self._inner.node_names_many(misses)
            for dcid in misses:
                names = tuple(fetched.get(dcid, ()))
                self._store.put(_key("node_names_many", dcid), "node_names_many", list(names))
                result[dcid] = names
        return result

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, SubjectsPage]:
        if not dcids:
            return {}
        unique = list(dict.fromkeys(dcids))
        result: dict[str, SubjectsPage] = {}
        misses: list[str] = []
        for dcid in unique:
            # Include the cap because a larger request needs a longer prefix of the
            # full subject set.
            hit, page = self._store.get(_key("node_subjects_many", dcid, property, max_results))
            if hit:
                if "coverage" not in page:
                    raise StaleCaptureError(
                        f"{self._store.path} has a node_subjects_many row with no "
                        "coverage field, so this capture cannot be replayed; recapture it"
                    )
                result[dcid] = SubjectsPage(
                    dcids=tuple(page["dcids"]), coverage=SubjectsCoverage(page["coverage"])
                )
            else:
                misses.append(dcid)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("node_subjects_many", *sorted(misses), property, max_results))
            with self._semaphore:
                fetched = self._inner.node_subjects_many(
                    misses, property=property, max_results=max_results, deadline=deadline
                )
            for dcid in misses:
                page = fetched[dcid]
                self._store.put(
                    _key("node_subjects_many", dcid, property, max_results),
                    "node_subjects_many",
                    {"dcids": list(page.dcids), "coverage": page.coverage.value},
                )
                result[dcid] = page
        return result

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        del deadline
        if not names:
            return {}
        result: dict[str, list[str]] = {}
        misses: list[str] = []
        for name in names:
            hit, dcids = self._store.get(_key("resolve_by_arc", name, arc))
            if hit:
                result[name] = dcids
            else:
                misses.append(name)
        if misses:
            if self._read_only:
                raise CacheMiss(_key("resolve_by_arc", *sorted(misses), arc))
            with self._semaphore:
                fetched = self._inner.resolve_by_arc(misses, arc=arc)
            for name in misses:
                dcids = fetched.get(name, [])
                self._store.put(_key("resolve_by_arc", name, arc), "resolve_by_arc", dcids)
                result[name] = dcids
        return result

    def un_availability(self) -> UnAvailabilityClient | None:
        """The inner client's UN availability read, uncached."""
        return self._inner.un_availability()

    def generation_stamp(self) -> GenerationStamp:
        """The inner client's stamp. The default-scope memo keys on it."""
        return self._inner.generation_stamp()

    def source_key(self) -> str:
        """The inner client's backends."""
        return self._inner.source_key()

    def child_places_batch(
        self, parent_dcids: list[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        del deadline
        if not parent_dcids:
            return {}
        return self._lookup(
            "child_places_batch",
            (sorted(parent_dcids), child_type),
            lambda: self._inner.child_places_batch(parent_dcids, child_type),
        )

    def detect_svs(self, query: str) -> Detection:
        """Admit the cached resolves under the bound catalogue, as the live client does."""
        return admit_legs(
            self.detect_legs(query), catalogue=current_catalogue(), detect_k=settings().detect_k
        )

    def detect_legs(self, query: str) -> DetectLegs:
        """Each resolve as captured, a failed UN resolve included."""
        un_enabled = current_catalogue().un_system_enabled
        key = _key("detect_legs", query, settings().resolve_target, un_enabled)
        hit, raw = self._store.get(key)
        if hit:
            un = raw["un"]
            return DetectLegs(
                ours=tuple(_DETECTED_SVS_ADAPTER.validate_python(raw["ours"])),
                un=None if un is None else tuple(_DETECTED_SVS_ADAPTER.validate_python(un)),
            )
        if self._read_only:
            raise CacheMiss(key)
        with self._semaphore:
            legs = self._inner.detect_legs(query)
        self._store.put(
            key,
            "detect_legs",
            {
                "ours": _DETECTED_SVS_ADAPTER.dump_python(list(legs.ours), mode="json"),
                "un": None
                if legs.un is None
                else _DETECTED_SVS_ADAPTER.dump_python(list(legs.un), mode="json"),
            },
        )
        return legs

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        resolved = needs_dates or _force_full_facets.get()
        key = _key("obs", stat_var, entity, resolved)
        hit, raw = self._store.get(key)
        if hit:
            return [_facet_from_json(f) for f in raw]
        if self._read_only:
            raise CacheMiss(key)
        with self._semaphore:
            facets = self._inner.observation_facets(
                stat_var=stat_var, entity=entity, needs_dates=needs_dates
            )
        self._store.put(key, "obs", [_facet_to_json(f) for f in facets])
        return facets

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        resolved = needs_dates or _force_full_facets.get()
        if not stat_vars or not entities:
            return {}
        result: dict[str, dict[str, list[Facet]]] = {sv: {} for sv in stat_vars}
        miss_pairs: set[tuple[str, str]] = set()
        for sv in stat_vars:
            for entity in entities:
                hit, raw = self._store.get(_key("obs", sv, entity, resolved))
                if hit:
                    result[sv][entity] = [_facet_from_json(f) for f in raw]
                else:
                    miss_pairs.add((sv, entity))
        if not miss_pairs:
            return result
        if self._read_only:
            pair_names = sorted(f"{sv}|{entity}" for sv, entity in miss_pairs)
            raise CacheMiss(_key("observation_facets_batch", *pair_names))
        # One fetch covers the full missing-variable x missing-entity product. Cache
        # every pair it returns, including overlapping hits, for later replays.
        miss_svs = list(dict.fromkeys(sv for sv, _ in miss_pairs))
        miss_entities = list(dict.fromkeys(entity for _, entity in miss_pairs))
        with self._semaphore:
            fetched = self._inner.observation_facets_batch(
                miss_svs,
                miss_entities,
                needs_dates=needs_dates,
                max_bytes=max_bytes,
                deadline=deadline,
            )
        for sv in miss_svs:
            for entity in miss_entities:
                facets = fetched.get(sv, {}).get(entity, [])
                self._store.put(
                    _key("obs", sv, entity, resolved), "obs", [_facet_to_json(f) for f in facets]
                )
                result[sv][entity] = facets
        return result

    def observation_read_groups(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> list[list[str]]:
        """The inner client's request groups for the read, cached so a replay of an
        early-stopping read stops where its capture did."""
        resolved = needs_dates or _force_full_facets.get()
        return self._lookup(
            "observation_read_groups",
            (stat_vars, entities, resolved),
            lambda: self._inner.observation_read_groups(stat_vars, entities, needs_dates),
        )

    def child_dcids(self, parent_dcid: str) -> list[str]:
        return self._lookup(
            "child_dcids", (parent_dcid,), lambda: self._inner.child_dcids(parent_dcid)
        )

    def enum_members(self, property_dcid: str) -> list[str]:
        return self._lookup(
            "enum_members", (property_dcid,), lambda: self._inner.enum_members(property_dcid)
        )

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        return self._lookup(
            "currency_units", (), lambda: self._inner.currency_units(deadline=deadline)
        )

    def entities_with_data(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        max_bytes: int | None = None,
    ) -> list[str]:
        if not variable_dcids or not entity_dcids:
            return []
        present = self._entities_with_data_present(
            variable_dcids, entity_dcids, max_bytes=max_bytes
        )
        return [e for e in entity_dcids if e in present]

    def _entities_with_data_present(
        self, variable_dcids: list[str], entity_dcids: list[str], *, max_bytes: int | None
    ) -> set[str]:
        """Cache the qualifying entity set order-insensitively, then restore the caller's
        input order. `max_bytes` applies only to the inner fetch on a miss."""
        key = _key("entities_with_data", sorted(variable_dcids), sorted(entity_dcids))
        hit, raw = self._store.get(key)
        if hit:
            return set(raw)
        if self._read_only:
            raise CacheMiss(key)
        with self._semaphore:
            present = set(
                self._inner.entities_with_data(variable_dcids, entity_dcids, max_bytes=max_bytes)
            )
        self._store.put(key, "entities_with_data", sorted(present))
        return present

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        if not variable_dcids or not entity_dcids:
            return {}
        key = _key("entities_with_data_by_variable", sorted(variable_dcids), sorted(entity_dcids))
        hit, raw = self._store.get(key)
        if hit:
            present_by_var = {var: set(members) for var, members in raw.items()}
        else:
            if self._read_only:
                raise CacheMiss(key)
            with self._semaphore:
                fetched = self._inner.entities_with_data_by_variable(
                    variable_dcids, entity_dcids, deadline=deadline
                )
            present_by_var = {var: set(fetched.get(var, [])) for var in variable_dcids}
            self._store.put(
                key,
                "entities_with_data_by_variable",
                {var: sorted(members) for var, members in present_by_var.items()},
            )
        return {
            var: [e for e in entity_dcids if e in present_by_var.get(var, set())]
            for var in variable_dcids
        }

    # -- eval-compatible surface: exists_batch ------------------------

    def exists_batch(self, dcids: list[str]) -> set[str]:
        """Confirm each batch name miss with singleton `->*` arcs, since real nodes can
        be unnamed. In read-only mode, raise `CacheMiss` when singleton evidence is
        uncached."""
        unique_dcids = list(dict.fromkeys(dcids))
        if not unique_dcids:
            return set()

        known = set(self.node_labels_batch(unique_dcids))
        for dcid in unique_dcids:
            if dcid in known:
                continue
            # A non-empty cached arc set confirms existence. Composition-sensitive
            # cached None still requires a singleton read.
            arcs_hit, cached_arcs = self._store.get(_key("node_arcs", dcid))
            if arcs_hit and cached_arcs is not None:
                known.add(dcid)
                continue
            singleton_key = _key("exists_singleton_arcs", dcid)
            singleton_hit, arcs = self._store.get(singleton_key)
            if singleton_hit:
                if arcs is not None:
                    known.add(dcid)
                continue
            if self._read_only:
                raise CacheMiss(singleton_key)
            with self._semaphore:
                arcs = self._inner.node_arcs(dcid)
            self._store.put(singleton_key, "exists_singleton_arcs", arcs)
            if arcs is not None:
                known.add(dcid)
        return known

    # -- fail-closed method surface ------------------------------------

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError(
            f"GraphCache has no proxied method {name!r}. Its method surface is "
            "the EngineGraphClient methods plus exists_batch and detect_legs, "
            "enumerated explicitly rather than inherited, "
            "so a graph method added to the engine later cannot silently "
            "bypass this cache."
        )
