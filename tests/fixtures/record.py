"""Record Search LLM, graph, and SDMX fixtures from live services with credentials stripped."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from dc_graph import ArcTarget, FacetDefaultPolicy, display_name
from dc_graph.client import _dedup_wire_arcs
from origin_catalogue import bound_catalogue, current_catalogue
from search.engine.clients.graph import Detection, DetectLegs, SubjectsPage, admit_legs
from search.engine.config import settings
from search.engine.retrieval.facts import _dcids_for
from tests.engine._fake_sdmx import filter_text
from tests.fixtures import PINNED_DATE, _arc_type_of, detect_legs_to_json
from tests.fixtures.keys import fixture_key
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

if TYPE_CHECKING:
    from graph_generation import GenerationStamp
    from origin_catalogue.graph import UnAvailabilityClient

_FIXTURES_DIR = Path(__file__).parent

# Pause 0.3 seconds between graph reads by default. Heavy queries (60+ SVs) fire one node
# read per SV, and without a pause the staging node endpoint returns 503.
# SEARCH_RECORD_THROTTLE_S overrides the delay.
_RECORD_THROTTLE_S = float(os.getenv("SEARCH_RECORD_THROTTLE_S", "0.3"))

_T = TypeVar("_T", bound=BaseModel)

_SECRET_RE = re.compile(
    r"api[_-]?key|authorization|x-goog-api-key|token|credential|secret",
    re.IGNORECASE,
)


def _strip_secrets(obj: Any) -> Any:
    """Remove dict keys matching API-key or authentication patterns."""
    if isinstance(obj, dict):
        return {k: _strip_secrets(v) for k, v in obj.items() if not _SECRET_RE.search(k)}
    if isinstance(obj, list):
        return [_strip_secrets(i) for i in obj]
    return obj


def _facet_to_graph_obs_dict(f) -> dict:
    """Build a graph_obs entry with summary fields and present facet metadata.

    Dates and values appear only when the Facet carries observations.
    """
    return {
        "earliestDate": f.earliest_date,
        "latestDate": f.latest_date,
        "obsCount": f.obs_count,
        **({"dates": list(f.dates), "values": list(f.values)} if f.dates else {}),
        **({"provenanceId": f.provenance_id} if f.provenance_id else {}),
        **({"measurementMethod": f.measurement_method} if f.measurement_method else {}),
        **({"unit": f.unit} if f.unit else {}),
        **({"facetId": f.facet_id} if f.facet_id else {}),
    }


_IDENTITY_FIELDS = (
    ("provenanceId", "provenance_id"),
    ("measurementMethod", "measurement_method"),
    ("unit", "unit"),
)
"""graph_obs keys and Facet attributes used to match legacy entries."""


def _facets_agree_on_identity(stored_entry: dict, f) -> bool:
    """Require at least one shared identity field and no conflicting shared values.

    Identity fields are provenanceId, measurementMethod, and unit. Facet order
    can change between select sets, so legacy entries need identity evidence instead.
    """
    corroborated = False
    for json_key, attr in _IDENTITY_FIELDS:
        stored_value = stored_entry.get(json_key)
        incoming_value = getattr(f, attr)
        if stored_value is None or incoming_value is None:
            continue
        if stored_value != incoming_value:
            return False
        corroborated = True
    return corroborated


def _summary_fields_agree(stored_entry: dict, f) -> bool:
    """Require matching earliestDate, latestDate, and obsCount before a legacy upgrade.

    A stored entry with none of the three fields can never pass, so the identity fallback
    cannot pair it; _claim_facets_for_key reports that case as its own refusal.
    """
    return (
        stored_entry.get("earliestDate") == f.earliest_date
        and stored_entry.get("latestDate") == f.latest_date
        and stored_entry.get("obsCount") == f.obs_count
    )


def _claim_facets_for_key(
    candidates: list[dict], facets: list
) -> dict[int, tuple[Any | None, str | None]]:
    """Pair date-less stored entries with incoming facets, claiming each facet at most once.

    Unique facetId matches run first and reserve their facets. Legacy entries then match
    unclaimed facets by identity and summary, uniquely on both sides.

    Return entry id to (facet, None), or (None, refusal reason). Refused entries stay unchanged.
    """
    result: dict[int, tuple[Any | None, str | None]] = {}
    fid_candidates = [c for c in candidates if c.get("facetId")]
    legacy_candidates = [c for c in candidates if not c.get("facetId")]

    by_fid: dict[str, list[dict]] = {}
    for c in fid_candidates:
        by_fid.setdefault(c["facetId"], []).append(c)

    claimed_facet_ids: set[int] = set()
    for fid, sharing in by_fid.items():
        fid_facets = [f for f in facets if f.facet_id == fid]
        if len(sharing) > 1:
            reason = f"ambiguous identity: facetId {fid!r} is stored on {len(sharing)} entries"
            for c in sharing:
                result[id(c)] = (None, reason)
        elif len(fid_facets) != 1:
            reason = (
                f"no corroborating match: no incoming facet carries facetId {fid!r}"
                if not fid_facets
                else f"ambiguous identity: {len(fid_facets)} incoming facets carry facetId {fid!r}"
            )
            result[id(sharing[0])] = (None, reason)
        else:
            result[id(sharing[0])] = (fid_facets[0], None)
            claimed_facet_ids.add(id(fid_facets[0]))

    available = [f for f in facets if id(f) not in claimed_facet_ids]
    tentative: dict[int, Any] = {}
    for c in legacy_candidates:
        agreeing = [f for f in available if _facets_agree_on_identity(c, f)]
        if len(agreeing) == 1:
            tentative[id(c)] = agreeing[0]
        else:
            reason = (
                "no corroborating match: no incoming facet agrees on identity"
                if not agreeing
                else f"ambiguous identity: {len(agreeing)} incoming facets agree on identity"
            )
            result[id(c)] = (None, reason)

    genuine_claimant_ids: dict[int, list[int]] = {}
    for c in legacy_candidates:
        f = tentative.get(id(c))
        if f is not None and _summary_fields_agree(c, f):
            genuine_claimant_ids.setdefault(id(f), []).append(id(c))

    for c in legacy_candidates:
        if id(c) in result:
            continue
        f = tentative[id(c)]
        claimant_ids = genuine_claimant_ids.get(id(f), [])
        if len(claimant_ids) > 1:
            reason = (
                "ambiguous identity: identity-agreeing facet also matches "
                f"{len(claimant_ids) - 1} other stored entry(ies) whose summary agrees with it"
            )
            result[id(c)] = (None, reason)
        elif id(c) not in claimant_ids:
            reason = (
                "disagreeing summary: no earliestDate/latestDate/obsCount to compare"
                if not any(k in c for k in ("earliestDate", "latestDate", "obsCount"))
                else "disagreeing summary: earliestDate/latestDate/obsCount don't match "
                "the identity-agreeing facet"
            )
            result[id(c)] = (None, reason)
        else:
            result[id(c)] = (f, None)

    return result


class _RecordingLLM:
    """Wraps the live LLM and records every call to the fixture dict."""

    def __init__(self, live_llm, store: dict):
        self._llm = live_llm
        self._store = store

    async def generate_structured(
        self, *, prompt: str, system: str, schema: type[_T]
    ) -> tuple[_T, dict | None]:
        result, usage = await self._llm.generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        key = fixture_key(schema, system, prompt)
        self._store[key] = result.model_dump(mode="json")
        return result, usage


def _merge_arcs(existing: dict, new: dict) -> dict:
    """Union partial arc properties and node values, keeping the first occurrence.

    Live node reads can return partial slices. Unioning preserves earlier values.
    """
    merged = {prop: {"nodes": list(data.get("nodes", []))} for prop, data in existing.items()}
    for prop, data in new.items():
        merged.setdefault(prop, {"nodes": []})["nodes"].extend(data.get("nodes", []))
    return _dedup_wire_arcs(merged)


class _RecordingGraph:
    """Record live graph calls in fixture dictionaries."""

    def __init__(
        self,
        live_graph,
        nodes: dict,
        obs: dict,
        detect: dict,
        resolve: dict,
        places: dict | None = None,
        children: dict | None = None,
        enum_members: dict | None = None,
        currency_units: dict | None = None,
    ):
        self._graph = live_graph
        self._nodes = nodes
        self._obs = obs
        self._detect = detect
        self._resolve = resolve
        # Region and child maps are populated on demand.
        self._places = places if places is not None else {}
        self._children = children if children is not None else {}
        # Enum members are recorded on demand.
        self._enum_members = enum_members if enum_members is not None else {}
        # Currency units are recorded on demand.
        self._currency_units = currency_units if currency_units is not None else {}

    def _merge_node(
        self,
        dcid: str,
        *,
        label: str | None = None,
        type_of: str | None = None,
        arcs: dict | None = None,
    ) -> None:
        """Update labels/types from observed values and merge arcs by property and node value."""
        node = self._nodes.setdefault(dcid, {"label": None, "type": None})
        if label is not None:
            node["label"] = label
        if type_of is not None:
            node["type"] = type_of
        if arcs is not None:
            node["arcs"] = _merge_arcs(node.get("arcs", {}), arcs)

    def node_label(self, dcid: str) -> str | None:
        result = self._graph.node_label(dcid)
        self._record_node(dcid)
        return result

    def node_arcs(self, dcid: str) -> dict | None:
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.node_arcs(dcid)
        if result is not None:
            label = self._graph.node_label(dcid)
            node_type = self._graph.node_type(dcid)
            self._merge_node(dcid, label=label, type_of=node_type, arcs=_strip_secrets(result))
        return result

    def node_type(self, dcid: str) -> str | None:
        return self._graph.node_type(dcid)

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        # Passed through unrecorded: the replaying `FakeGraph` declares none, and a declared
        # default refuses resolve before any recorded row is read.
        return self._graph.default_facet_policy(stat_var=stat_var, axis=axis)

    def generation_stamp(self) -> GenerationStamp:
        return self._graph.generation_stamp()

    def source_key(self) -> str:
        return self._graph.source_key()

    def un_availability(self) -> UnAvailabilityClient | None:
        # Passed through unrecorded: the replaying `FakeGraph` answers the UN deployment's
        # availability from the recorded observations (`FakeGraph.un_availability`).
        return self._graph.un_availability()

    def _record_node(self, dcid: str) -> None:
        if dcid not in self._nodes:
            time.sleep(_RECORD_THROTTLE_S)
            arcs = self._graph.node_arcs(dcid)
            if arcs is not None:
                label = self._graph.node_label(dcid)
                node_type = self._graph.node_type(dcid)
                self._merge_node(dcid, label=label, type_of=node_type, arcs=_strip_secrets(arcs))

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Record observed types and names. Merge arcs with the existing node arcs."""
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.node_arcs_many(dcids, properties=properties, deadline=deadline)
        for dcid, arcs in result.items():
            if arcs is None:
                continue
            type_values = arcs.get("typeOf", ())
            name_values = arcs.get("name", ())
            self._merge_node(
                dcid,
                label=name_values[0] if name_values else None,
                type_of=type_values[0] if type_values else None,
            )
        return result

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        """Record what `FakeGraph.node_arc_targets_many` replays: each node's display name as
        its label, its first type, its other properties as arcs, and each node target's name
        and first type on that target's own node."""
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.node_arc_targets_many(dcids, properties=properties, deadline=deadline)
        for dcid, targets in result.items():
            if targets is None:
                continue
            type_values = targets.get("typeOf", ())
            arcs: dict[str, dict[str, list[dict[str, str]]]] = {}
            for prop, values in targets.items():
                if prop in ("name", "typeOf"):
                    continue
                arcs[prop] = {
                    "nodes": [
                        {"dcid": t.value} if t.name or t.types else {"value": t.value}
                        for t in values
                    ]
                }
                for target in values:
                    if target.name or target.types:
                        self._merge_node(
                            target.value,
                            label=target.name,
                            type_of=target.types[0] if target.types else None,
                        )
            self._merge_node(
                dcid,
                label=display_name((t.value for t in targets.get("name", ())), dcid),
                type_of=type_values[0].value if type_values else None,
                arcs=arcs or None,
            )
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Pass through. Raw node arcs carry alternate names."""
        return self._graph.node_names_many(dcids, deadline=deadline)

    def resolve_by_arc(
        self,
        names: Sequence[str],
        *,
        arc: str,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.resolve_by_arc(names, arc=arc, deadline=deadline)
        type_of = _arc_type_of(arc)
        for name, dcids in result.items():
            if not dcids:
                continue
            dcid = dcids[0]
            if type_of == "Country":
                self._resolve[name] = dcid
            elif type_of in ("Continent", "UNGeoRegion"):
                self._places[name] = {"dcid": dcid, "type_of": type_of}
            # Bare-rung resolutions are not recorded.
        return result

    def child_places_batch(
        self,
        parent_dcids: list[str],
        child_type: str,
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.child_places_batch(parent_dcids, child_type, deadline=deadline)
        for parent, children in result.items():
            self._children.setdefault(parent, {})[child_type] = children
        return result

    def detect_svs(self, query: str) -> Detection:
        """Record each resolve separately and admit them as the live client does."""
        return admit_legs(
            self.detect_legs(query), catalogue=current_catalogue(), detect_k=settings().detect_k
        )

    def detect_legs(self, query: str) -> DetectLegs:
        time.sleep(_RECORD_THROTTLE_S)
        legs = self._graph.detect_legs(query)
        self._detect[query] = detect_legs_to_json(legs)
        return legs

    def observation_facets(self, *, stat_var: str, entity: str, needs_dates: bool = False):
        time.sleep(_RECORD_THROTTLE_S)
        # Preserve date-aware facet data.
        result = self._graph.observation_facets(
            stat_var=stat_var, entity=entity, needs_dates=needs_dates
        )
        self._upsert_obs(f"{stat_var}|{entity}", result)
        return result

    def _upsert_obs(self, key: str, facets: list) -> None:
        """Keep existing graph_obs entries. Upgrade only date-less facets.

        FacetId matches replace entries, dropping stored fields the incoming facet lacks.
        Legacy matches require unique identity and matching summaries, then add dates and
        values only. To refresh a recorded key outright, delete it from graph_obs.json.
        Refusals print key and reason to stderr.
        """
        if key not in self._obs:
            self._obs[key] = [_facet_to_graph_obs_dict(f) for f in facets]
            return
        candidates = [entry for entry in self._obs[key] if "dates" not in entry]
        claims = _claim_facets_for_key(candidates, facets)
        for stored_entry in candidates:
            f, reason = claims[id(stored_entry)]
            if f is None:
                print(f"refusing upgrade for {key!r}: {reason}", file=sys.stderr)
                continue
            if not f.dates:
                continue
            if stored_entry.get("facetId"):
                stored_entry.clear()
                stored_entry.update(_facet_to_graph_obs_dict(f))
                continue
            stored_entry["dates"] = list(f.dates)
            stored_entry["values"] = list(f.values)

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ):
        """Fetch and record each StatVar/entity pair in one POST. Empty inputs return {}."""
        if not stat_vars or not entities:
            return {}
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.observation_facets_batch(
            stat_vars, entities, needs_dates, max_bytes=max_bytes, deadline=deadline
        )
        for sv, by_entity in result.items():
            for entity, facets in by_entity.items():
                self._upsert_obs(f"{sv}|{entity}", facets)
        return result

    def observation_read_groups(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> list[list[str]]:
        return self._graph.observation_read_groups(stat_vars, entities, needs_dates)

    def entities_with_data(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        max_bytes: int | None = None,
    ) -> list[str]:
        """Check every entity against variable_dcids via one
        observation_facets_batch call, recording every pair in the
        variable_dcids x entity_dcids cartesian product. That is more pairs
        than a union short-circuit would need, but presence per pair is
        unchanged, so replay yields the same answer and a fuller
        graph_obs.json can only reduce missing-key failures. Empty
        variable_dcids or entity_dcids returns [] without any calls.
        max_bytes is accepted for signature parity with EngineGraphClient
        and unused: this recorder always fetches every pair to build a
        complete fixture, regardless of any per-chunk cap LiveGraphClient
        itself applies.
        Returns the union across variable_dcids, in entity_dcids' input
        order, where an entity qualifies if it has data for at least one.
        """
        if not variable_dcids or not entity_dcids:
            return []
        facets = self.observation_facets_batch(variable_dcids, entity_dcids)
        return [
            entity for entity in entity_dcids if any(facets[var][entity] for var in variable_dcids)
        ]

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Check every entity against variable_dcids via one
        observation_facets_batch call, recording every pair in the
        variable_dcids x entity_dcids cartesian product. Empty variable_dcids
        or entity_dcids returns {} without any calls. A variable_dcid maps to
        the entity_dcids that have data for it, in entity_dcids' input order.
        """
        if not variable_dcids or not entity_dcids:
            return {}
        facets = self.observation_facets_batch(variable_dcids, entity_dcids)
        return {
            var: [entity for entity in entity_dcids if facets[var][entity]]
            for var in variable_dcids
        }

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.node_arcs_batch(dcids, deadline=deadline)
        for dcid, arcs in result.items():
            if arcs is not None:
                self._merge_node(dcid, arcs=_strip_secrets(arcs))
        return result

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.node_labels_batch(dcids)
        for dcid, label in result.items():
            self._merge_node(dcid, label=label)
        return result

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.node_types_batch(dcids)
        for dcid, type_of in result.items():
            self._merge_node(dcid, type_of=type_of)
        return result

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        """Record missing full arcs and derive observationProperties from the stored nodes."""
        missing = [
            dcid
            for dcid in dcids
            if self._nodes.get(dcid) is None or "arcs" not in self._nodes[dcid]
        ]
        if missing:
            time.sleep(_RECORD_THROTTLE_S)
            fetched = self._graph.node_arcs_batch(missing)
            for dcid, arcs in fetched.items():
                if arcs is not None:
                    self._merge_node(dcid, arcs=_strip_secrets(arcs))
        result: dict[str, tuple[str, ...]] = {}
        for dcid in dcids:
            node = self._nodes.get(dcid)
            arcs = node.get("arcs") if node is not None else None
            result[dcid] = tuple(_dcids_for(arcs, "observationProperties")) if arcs else ()
        return result

    def exists(self, dcid: str) -> bool:
        return self._graph.exists(dcid)

    def child_dcids(self, parent_dcid: str) -> list[str]:
        time.sleep(_RECORD_THROTTLE_S)
        children = self._graph.child_dcids(parent_dcid)
        # Record child arcs so FakeGraph can reverse-scan isPartOf during replay.
        for dcid in children:
            self._record_node(dcid)
        return children

    def enum_members(self, property_dcid: str) -> list[str]:
        time.sleep(_RECORD_THROTTLE_S)
        members = self._graph.enum_members(property_dcid)
        self._enum_members[property_dcid] = members
        return members

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        time.sleep(_RECORD_THROTTLE_S)
        units = self._graph.currency_units(deadline=deadline)
        self._currency_units.update(units)
        return units

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, SubjectsPage]:
        time.sleep(_RECORD_THROTTLE_S)
        result = self._graph.node_subjects_many(
            dcids, property=property, max_results=max_results, deadline=deadline
        )
        # Record subject arcs for FakeGraph's reverse scan during replay.
        for page in result.values():
            for subject_dcid in page.dcids:
                self._record_node(subject_dcid)
        return result


def _facet_to_dict(facet) -> dict:
    """Build the sdmx_obs facet shape, including facetId for filtered replay."""
    return {
        "earliestDate": facet.earliest_date,
        "latestDate": facet.latest_date,
        "obsCount": facet.obs_count,
        "dates": list(facet.dates),
        "values": list(facet.values),
        **({"provenanceId": facet.provenance_id} if facet.provenance_id else {}),
        **({"measurementMethod": facet.measurement_method} if facet.measurement_method else {}),
        **({"unit": facet.unit} if facet.unit else {}),
        **({"scalingFactor": facet.scaling_factor} if facet.scaling_factor else {}),
        **({"facetId": facet.facet_id} if facet.facet_id else {}),
    }


def _availability_key(
    component: str,
    variables: Sequence[str],
    filters: Mapping[str, Sequence[str]] | None,
    time_period: str | Sequence[str] | None = None,
) -> str:
    """Build the canonical availability key from component, variables, filters, and period."""
    key = f"{component}|variables={','.join(sorted(set(variables)))}"
    if filters:
        parts = ";".join(f"{role}={filter_text(filt)}" for role, filt in sorted(filters.items()))
        key = f"{key}|{parts}"
    if time_period is not None:
        period_text = time_period if isinstance(time_period, str) else ",".join(time_period)
        key = f"{key}@time_period={period_text}"
    return key


def _role_tuple_key(variable: str, role_tuple: Sequence[tuple[str, str]]) -> str:
    """Build the role-series key with role segments sorted by role name."""
    parts = ",".join(f"{role}={dcid}" for role, dcid in sorted(role_tuple))
    return f"{variable}|{parts}"


class _RecordingSdmx:
    """Record role-series facets by id and availability values by canonical request key.

    Matching facet ids merge dates. Incoming values win on overlaps.
    Distinct ids stay separate, and facets without ids are skipped. Subject-axis reads pass through.
    """

    def __init__(self, live_sdmx, obs: dict, availability: dict):
        self._sdmx = live_sdmx
        self._obs = obs
        self._availability = availability

    def source_key(self) -> str:
        return self._sdmx.source_key()

    def _upsert_role_facets(self, key: str, facets) -> None:
        """Merge returned facets into the role-series fixture by facet id."""
        stored = self._obs.setdefault(key, [])
        for f in facets:
            if f.facet_id is None:
                continue
            existing_idx = next(
                (i for i, entry in enumerate(stored) if entry.get("facetId") == f.facet_id),
                None,
            )
            if existing_idx is None:
                stored.append(_facet_to_dict(f))
                continue

            existing = stored[existing_idx]
            observations = dict(
                zip(existing.get("dates", []), existing.get("values", []), strict=True)
            )
            observations.update(zip(f.dates, f.values, strict=True))
            dates = sorted(observations)
            merged = _facet_to_dict(f)
            merged["dates"] = dates
            merged["values"] = [observations[date] for date in dates]
            merged["obsCount"] = len(dates)
            merged["earliestDate"] = dates[0] if dates else None
            merged["latestDate"] = dates[-1] if dates else None
            stored[existing_idx] = merged

    def role_series(
        self,
        *,
        variables,
        entities,
        facet_ids,
        time_period=None,
        max_rows,
        max_bytes,
        deadline=None,
    ):
        time.sleep(_RECORD_THROTTLE_S)
        result = self._sdmx.role_series(
            variables=variables,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
            max_rows=max_rows,
            max_bytes=max_bytes,
            deadline=deadline,
        )
        for variable, role_groups in result.items():
            for role_tuple, facets in role_groups.items():
                self._upsert_role_facets(_role_tuple_key(variable, role_tuple), facets)
        return result

    def availability_values(
        self,
        *,
        variables,
        component,
        filters=None,
        time_period=None,
        max_values=None,
        max_bytes=None,
        deadline=None,
        read_seconds=None,
    ):
        time.sleep(_RECORD_THROTTLE_S)
        result = self._sdmx.availability_values(
            variables=variables,
            component=component,
            filters=filters,
            time_period=time_period,
            max_values=max_values,
            max_bytes=max_bytes,
            deadline=deadline,
            read_seconds=read_seconds,
        )
        self._availability[_availability_key(component, variables, filters, time_period)] = list(
            result
        )
        return result

    def subject_observations(
        self,
        *,
        variables,
        places,
        facet_ids=None,
        latest_only: bool = True,
        max_rows=None,
        max_bytes=None,
    ):
        """Pass through subject-axis observations. Offline fixtures return no rows."""
        time.sleep(_RECORD_THROTTLE_S)
        return self._sdmx.subject_observations(
            variables=variables,
            places=places,
            facet_ids=facet_ids,
            latest_only=latest_only,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)
    print(f"  Wrote {path}")


def main() -> None:
    import argparse

    from search.engine.clients.graph_client import LiveGraphClient
    from search.engine.clients.llm import LLM
    from search.eval.cases import load_case_files
    from search.eval.runner import build_request
    from search.models import ResolveRequest

    parser = argparse.ArgumentParser(description="Search fixture recorder")
    parser.add_argument(
        "--file",
        default="standard",
        help="Corpus case file to record, by stem (default: standard; other values: 'sdg').",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated case ids to record (intersected with --file).",
    )
    args = parser.parse_args()
    stem = args.file
    only_ids = {i.strip() for i in args.ids.split(",")} if args.ids else None

    print(f"Search fixture recorder (file={stem!r}; requires GEMINI_API_KEY and graph access)")

    case_files = load_case_files()
    if stem not in case_files:
        raise SystemExit(f"no corpus case file {stem!r}; choose from {sorted(case_files)}")
    cases = [c for c in case_files[stem] if only_ids is None or c.id in only_ids]
    if only_ids:
        print(f"Recording only ids: {sorted(only_ids)}")
    print(f"Found {len(cases)} cases in {stem!r} to record")

    # Preserve existing entries while adding new recordings.
    def _load_existing(name: str) -> dict:
        path = _FIXTURES_DIR / name
        if path.exists():
            with open(path) as _f:
                return json.load(_f)
        return {}

    llm_store: dict = _load_existing("llm_responses.json")
    nodes_store: dict = _load_existing("graph_nodes.json")
    obs_store: dict = _load_existing("graph_obs.json")
    detect_store: dict = _load_existing("graph_detect.json")
    resolve_store: dict = _load_existing("graph_resolve.json")
    places_store: dict = _load_existing("graph_places.json")
    children_store: dict = _load_existing("graph_children.json")
    enum_members_store: dict = _load_existing("graph_enum_members.json")
    currency_units_store: dict = _load_existing("graph_currency_units.json")
    sdmx_obs_store: dict = _load_existing("sdmx_obs.json")
    sdmx_availability_store: dict = _load_existing("sdmx_availability.json")

    # Recording binds the permissive catalogue replay binds, so every read replay makes is
    # recorded, and each resolve is recorded before admission.
    catalogue = FakeCatalogue()
    live_graph = LiveGraphClient.from_search_settings()
    live_llm = LLM()
    rec_graph = _RecordingGraph(
        live_graph,
        nodes_store,
        obs_store,
        detect_store,
        resolve_store,
        places_store,
        children_store,
        enum_members_store,
        currency_units_store,
    )
    rec_llm = _RecordingLLM(live_llm, llm_store)

    from search.engine.clients.sdmx import LiveSdmxClient

    live_sdmx = LiveSdmxClient()
    rec_sdmx = _RecordingSdmx(live_sdmx, sdmx_obs_store, sdmx_availability_store)

    try:
        import asyncio
        from datetime import date
        from unittest.mock import patch

        from search.engine.core import resolve_async

        # Pin the current year so recorded relative dates and forecast years match replay.

        for case in cases:
            print(f"  Recording {case.id}: {case.query!r}")
            _max_attempts = 5
            for _attempt in range(1, _max_attempts + 1):
                try:
                    request = ResolveRequest.model_validate(build_request(case))
                    with (
                        bound_catalogue(catalogue),
                        patch("search.engine.extract.date") as mock_date,
                    ):
                        mock_date.today.return_value = PINNED_DATE
                        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
                        asyncio.run(
                            resolve_async(request, graph=rec_graph, llm=rec_llm, sdmx=rec_sdmx)
                        )
                    break
                except Exception as exc:
                    if _attempt < _max_attempts:
                        _wait = 2**_attempt
                        print(f"    attempt {_attempt} FAIL ({exc}); retrying in {_wait}s")
                        time.sleep(_wait)
                    else:
                        print(f"    ERROR on {case.id} (all {_max_attempts} attempts): {exc}")
    except ImportError:
        print("  search.engine.core not yet available; recording graph probes only.")

    print("\nWriting fixture files...")
    _write_json(_FIXTURES_DIR / "llm_responses.json", llm_store)
    _write_json(_FIXTURES_DIR / "graph_nodes.json", nodes_store)
    _write_json(_FIXTURES_DIR / "graph_obs.json", obs_store)
    _write_json(_FIXTURES_DIR / "graph_detect.json", detect_store)
    _write_json(_FIXTURES_DIR / "graph_resolve.json", resolve_store)
    _write_json(_FIXTURES_DIR / "graph_places.json", places_store)
    _write_json(_FIXTURES_DIR / "graph_children.json", children_store)
    _write_json(_FIXTURES_DIR / "graph_enum_members.json", enum_members_store)
    _write_json(_FIXTURES_DIR / "graph_currency_units.json", currency_units_store)
    _write_json(_FIXTURES_DIR / "sdmx_obs.json", sdmx_obs_store)
    _write_json(_FIXTURES_DIR / "sdmx_availability.json", sdmx_availability_store)

    print("\nDone. Run 'uv run pytest -q tests/fixtures tests/engine' to verify.")


if __name__ == "__main__":
    main()
