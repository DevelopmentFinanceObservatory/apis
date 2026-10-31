"""Offline FakeLLM and FakeGraph clients backed by local JSON fixtures."""

from __future__ import annotations

import itertools
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel

import entity_resolution.shared as shared_module
from code_index.shared import get_index
from dc_graph import ArcTarget, FacetDefaultPolicy
from dc_graph.client import UNRESOLVED_NODE, _dedup_wire_arcs
from entity_resolution import EntityResolver, GraphLookup, acquire_shared
from entity_resolution.membership import (
    COUNTRY_CLASS,
    Authority,
    Membership,
    MembershipPolicy,
    Row,
    Sovereignty,
)
from entity_resolution.shared import shared_memberships
from graph_generation import FakeGeneration, GenerationStamp
from origin_catalogue import current_catalogue
from search.engine.clients.graph import (
    DetectedSV,
    Detection,
    DetectLegs,
    Facet,
    SubjectsCoverage,
    SubjectsPage,
    admit_legs,
)
from search.engine.config import settings
from search.engine.errors import GraphInfraError, GraphReadFailure
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT
from search.engine.retrieval.facts import _dcids_for
from tests.engine._fake_sdmx import scope_availability_of
from tests.fixtures.keys import fixture_key
from tests.fixtures.provenance import facet_provenance
from tests.llm_miss_guard import HARVEST_MISS_PATH_ENV, note_miss
from tests.materialise._scope_availability import ScopeAvailability

_T = TypeVar("_T", bound=BaseModel)

_FIXTURES_DIR = Path(__file__).parent


def make_resolver(graph: GraphLookup) -> EntityResolver:
    """Build an EntityResolver over `graph` using the session's shared resolver and the code
    index production Search passes, so a node's kind follows `entity_resolution.kinds.classify`."""
    return EntityResolver(graph=graph, shared=acquire_shared(), code_index=get_index)


def declare_regions(
    monkeypatch: pytest.MonkeyPatch,
    regions: Mapping[str, Sequence[str]],
    *,
    omitted: Sequence[str] = (),
) -> None:
    """Make the process's membership policy hold `regions`, each a dcid mapped to
    its member countries, and omit `omitted`, until `monkeypatch` undoes it (`with_regions`).
    Search reads a region's members from that policy, so a fake graph's regions are declared
    here."""
    held = shared_memberships()
    policy = with_regions(held.policy, regions, omitted=omitted)
    monkeypatch.setattr(shared_module, "_memberships", replace(held, policy=policy))


def with_regions(
    policy: MembershipPolicy,
    regions: Mapping[str, Sequence[str]],
    *,
    omitted: Sequence[str] = (),
) -> MembershipPolicy:
    """`policy` with each region dcid in `regions` holding the listed countries, replacing
    any region of the same dcid and keeping its class, and each dcid in `omitted` omitted by
    rule. A region the policy did not hold is a subregion. Every listed country is a
    present-day UN member, so a declared region's `countries` projection is its list."""
    members = {member for listed in regions.values() for member in listed}
    sovereignty = Sovereignty(
        states={member: (Row(member=member, member_class=COUNTRY_CLASS),) for member in members},
        dissolved=frozenset(),
        other_states=frozenset(),
    )
    held_classes = {m.ref: m.region_class for m in policy if m.region_class is not None}
    declared = [
        Membership(
            ref=dcid,
            entity_id=dcid,
            label=dcid,
            kind="region",
            authority=Authority(name="test regions", version="1"),
            dated=False,
            rows=tuple(Row(member=member, member_class=COUNTRY_CLASS) for member in listed),
            sovereignty=sovereignty,
            region_class=held_classes.get(dcid, "geo.subregion"),
        )
        for dcid, listed in regions.items()
    ]
    return MembershipPolicy(
        [*(m for m in policy if m.ref not in regions), *declared],
        omitted_regions=(policy.omitted_regions - regions.keys()) | set(omitted),
    )


# Extract the typeOf constraint from an arc expression. Bare arcs return None.
_ARC_TYPE_OF_RE = re.compile(r"\{typeOf:([^}]+)\}")


def _arc_type_of(arc: str) -> str | None:
    match = _ARC_TYPE_OF_RE.search(arc)
    return match.group(1) if match else None


PINNED_DATE = date(2026, 6, 23)
"""The `date.today()` tests pin, so `extract` resolves relative dates and forecast years
against one current year (2026) in recording and replay."""


def _load_json(name: str) -> dict:
    path = _FIXTURES_DIR / name
    with open(path) as f:
        return json.load(f)


def fixture_provenance_names(*extra: str) -> dict[str, str]:
    """Each provenance the fixture graph's observations carry (`graph_obs.json`), and each in
    `extra`, named by its fixture node's label, else by its dcid's last segment, the import
    name the graph names a base provenance by. A test lists them in its catalogue
    (`tests.origin_catalogue.fake_catalogue.listing`), as production's lists every provenance
    it admits."""
    found = re.findall(r'"provenanceId": "([^"]+)"', (_FIXTURES_DIR / "graph_obs.json").read_text())
    nodes = _load_json("graph_nodes.json")
    return {
        dcid: (nodes.get(dcid) or {}).get("label") or dcid.rsplit("/", 1)[-1]
        for dcid in dict.fromkeys([*found, *extra])
    }


def detect_leg_from_json(entry: Mapping) -> tuple[DetectedSV, ...]:
    """One recorded resolve: `svs` in recorded order, `cosine_scores` parallel to them (1.0
    each when absent or of another length), and optional `types` and `sentences` maps."""
    svs: list[str] = entry.get("svs", [])
    raw_scores: list[float] = entry.get("cosine_scores", [])
    scores = raw_scores if raw_scores and len(raw_scores) == len(svs) else [1.0] * len(svs)
    types: dict[str, str | None] = entry.get("types", {})
    sentences: dict[str, str | None] = entry.get("sentences", {})
    return tuple(
        DetectedSV(dcid=sv, score=score, type_of=types.get(sv), sentence=sentences.get(sv))
        for sv, score in zip(svs, scores, strict=True)
    )


def detect_leg_to_json(candidates: Sequence[DetectedSV]) -> dict:
    """The fixture form of one resolve, read back by `detect_leg_from_json`."""
    return {
        "svs": [c.dcid for c in candidates],
        "cosine_scores": [c.score for c in candidates],
        "types": {c.dcid: c.type_of for c in candidates if c.type_of is not None},
        "sentences": {c.dcid: c.sentence for c in candidates if c.sentence is not None},
    }


def detect_legs_to_json(legs: DetectLegs) -> dict:
    """The `graph_detect.json` entry for one query: each resolve separately, the UN one as
    `null` when it failed."""
    return {
        "ours": detect_leg_to_json(legs.ours),
        "un": None if legs.un is None else detect_leg_to_json(legs.un),
    }


def _load_detect_fixture() -> tuple[dict[str, dict], dict[str, dict | None]]:
    """`graph_detect.json` split into our resolves and the UN's, each keyed by query."""
    entries = {
        query: entry
        for query, entry in _load_json("graph_detect.json").items()
        if isinstance(entry, dict)
    }
    return (
        {query: entry["ours"] for query, entry in entries.items()},
        {query: entry["un"] for query, entry in entries.items()},
    )


# Admission reads these graph schema arcs in every FakeGraph.
# Test nodes override defined fields. The merge supplies missing schema arcs.
_SCHEMA_NODE_DCIDS = (
    "Country",
    "State",
    "City",
    "AdministrativeArea",
    "Continent",
    "Landform",
    "UNGeoRegion",
    "ContinentalUnion",
    "GeoRegion",
    "Place",
    "Organization",
    "DACRecipientUnallocatedByCountry",
    "Thing",
    "observationAbout",
    "financingProvider",
    "financingRecipient",
)
_schema_nodes_cache: dict | None = None

# Synthetic role properties use financingProvider/financingRecipient ranges.
# These ranges admit Country, DACRecipientUnallocatedByCountry, and Organization.
_SYNTHETIC_ROLE_PROPERTY_DCIDS = ("provider", "recipient", "creditor", "debtor", "contributor")


def _schema_nodes() -> dict:
    global _schema_nodes_cache
    if _schema_nodes_cache is None:
        all_nodes = _load_json("graph_nodes.json")
        nodes = {dcid: all_nodes[dcid] for dcid in _SCHEMA_NODE_DCIDS if dcid in all_nodes}
        range_includes = all_nodes["financingProvider"]["arcs"]["rangeIncludes"]
        for dcid in _SYNTHETIC_ROLE_PROPERTY_DCIDS:
            nodes[dcid] = {
                "label": dcid,
                "type": "Property",
                "arcs": {"rangeIncludes": range_includes},
            }
        _schema_nodes_cache = nodes
    return _schema_nodes_cache


def _merge_schema_nodes(nodes: dict) -> dict:
    """Merge schema nodes under test nodes, with custom fields taking precedence.

    Arcs merge by property so partial fixtures retain schema hierarchy and ranges.
    """
    schema = _schema_nodes()
    merged = dict(nodes)
    for dcid, schema_node in schema.items():
        custom_node = nodes.get(dcid)
        if custom_node is None:
            merged[dcid] = schema_node
            continue
        merged_arcs = {**schema_node.get("arcs", {}), **custom_node.get("arcs", {})}
        merged[dcid] = {
            **schema_node,
            **custom_node,
            **({"arcs": merged_arcs} if merged_arcs else {}),
        }
    return merged


class MissingFixtureError(KeyError):
    """Raised when FakeLLM lacks a response, while preserving KeyError handling.

    `fixture_key` identifies the miss for the harvest outcome plugin.
    """

    def __init__(self, message: str, *, fixture_key: str) -> None:
        super().__init__(message)
        self.fixture_key = fixture_key


def _append_harvest_line(
    path: str,
    *,
    outcome: str,
    key: str,
    schema_name: str | None = None,
    system: str | None = None,
    prompt: str | None = None,
) -> None:
    """Append one JSON record per call. Misses include prompts. Hits include key and outcome.

    A single append write keeps concurrent records on separate lines.
    """
    record: dict[str, str] = {"key": key, "outcome": outcome}
    if outcome == "miss":
        record["schema_name"] = schema_name or ""
        record["system"] = system or ""
        record["prompt"] = prompt or ""
    line = json.dumps(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


_YEAR_PREFIX, _, _YEAR_SUFFIX = _EXTRACTION_SYSTEM_PROMPT.partition("[[YEAR]]")


def _recorded_system(system: str) -> str:
    """`system` as recorded fixtures key it. An extraction prompt with a four-digit year in
    place of `[[YEAR]]` keys as the template, so a recorded extraction replays in any year;
    every other prompt keys as sent."""
    if len(system) != len(_YEAR_PREFIX) + 4 + len(_YEAR_SUFFIX):
        return system
    if not (system.startswith(_YEAR_PREFIX) and system.endswith(_YEAR_SUFFIX)):
        return system
    year = system[len(_YEAR_PREFIX) : len(_YEAR_PREFIX) + 4]
    return _EXTRACTION_SYSTEM_PROMPT if year.isdigit() else system


class FakeLLM:
    """Replay structured outputs from JSON fixtures.

    Missing keys raise MissingFixtureError, carrying the schema name, key, and full system and
    user prompt so the fixture can be written without a second run. Optional harvest logging
    records each call.
    Keys come from `fixture_key` over the system prompt as `_recorded_system` keys it.
    `recorded` is whether it replays `llm_responses.json` rather than the `responses` a test
    scripted.
    """

    def __init__(self, responses: dict | None = None):
        self.recorded = responses is None
        self._responses: dict = (
            responses if responses is not None else _load_json("llm_responses.json")
        )

    async def generate_structured(
        self,
        *,
        prompt: str,
        system: str,
        schema: type[_T],
    ) -> tuple[_T, dict | None]:
        schema_name = schema.__name__
        key = fixture_key(schema, _recorded_system(system), prompt)
        harvest_path = os.environ.get(HARVEST_MISS_PATH_ENV)
        if key not in self._responses:
            note_miss(key)
            if harvest_path:
                _append_harvest_line(
                    harvest_path,
                    outcome="miss",
                    key=key,
                    schema_name=schema_name,
                    system=system,
                    prompt=prompt,
                )
            raise MissingFixtureError(
                f"FakeLLM: missing fixture for schema={schema_name!r}, key={key!r}\n"
                f"  system prompt:\n{system}\n"
                f"  user prompt:\n{prompt}\n"
                f"Add an entry to tests/fixtures/llm_responses.json with the key above.",
                fixture_key=key,
            )
        if harvest_path:
            _append_harvest_line(harvest_path, outcome="hit", key=key)
        parsed = schema.model_validate(self._responses[key])
        # Fake usage counts are zero.
        usage: dict | None = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        return parsed, usage


class _RaiseOnAny:
    """Graph double that raises GraphInfraError on every read."""

    def node_label(self, dcid: str) -> str | None:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {dcid!r}")

    def node_arcs(self, dcid: str) -> dict | None:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {dcid!r}")

    def node_type(self, dcid: str) -> str | None:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {dcid!r}")

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        # No transport read to fail: the live client answers from no wire call
        # (`dc_graph.GraphClient.default_facet_policy`), so paths that read no graph still
        # run resolve's declared-default check.
        return None

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {dcids!r}")

    def node_arc_targets_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        raise GraphReadFailure(f"FakeGraph(raise=True): simulated transport error for {dcids!r}")

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {dcids!r}")

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, SubjectsPage]:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {dcids!r}")

    def resolve_by_arc(
        self,
        names: Sequence[str],
        *,
        arc: str,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {names!r}")

    def child_places_batch(
        self,
        parent_dcids: list[str],
        child_type: str,
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        raise GraphInfraError(
            f"FakeGraph(raise=True): simulated transport error for {parent_dcids!r}"
        )

    def detect_svs(self, query: str) -> Detection:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {query!r}")

    def detect_legs(self, query: str) -> DetectLegs:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {query!r}")

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def observation_read_groups(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> list[list[str]]:
        return [list(stat_vars)]

    def un_availability(self) -> None:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def child_dcids(self, parent_dcid: str) -> list[str]:
        raise GraphInfraError(
            f"FakeGraph(raise=True): simulated transport error for {parent_dcid!r}"
        )

    def enum_members(self, property_dcid: str) -> list[str]:
        raise GraphInfraError(
            f"FakeGraph(raise=True): simulated transport error for {property_dcid!r}"
        )

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def entities_with_data(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        max_bytes: int | None = None,
    ) -> list[str]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        raise GraphInfraError("FakeGraph(raise=True): simulated transport error")

    def exists(self, dcid: str) -> bool:
        raise GraphInfraError(f"FakeGraph(raise=True): simulated transport error for {dcid!r}")


_PLACE_NODE: dict = {"type": "Place"}

# Each fake graph has a generation token of its own, so no fake shares another's memoised
# default scopes (`materialise.place_scope`).
_FAKE_GENERATIONS = itertools.count()


class FakeGraph:
    """Offline graph client backed by recorded graph and observation fixtures.

    Missing dcids have no label and fail `exists`. `raise_on_call` raises GraphInfraError.
    `node_arcs_many` reads a place `obs` holds observations at as a node typed `Place` unless
    `nodes` holds it, as a live place holding observations is a graph node. Each instance has
    a generation token of its own (`generation_stamp`), and `scope_availability` answers the
    default place scope's availability read from `obs`.

    Recall replays each resolve separately. `detect` holds our resolves and `un_detect` the UN
    System Data Commons' by query, in `detect_leg_from_json`'s form; a `None` UN entry is a
    failed UN resolve. With neither given, both come from `graph_detect.json`. With `detect`
    given, the UN resolves are `un_detect`, or none.
    """

    def __init__(
        self,
        *,
        nodes: dict | None = None,
        obs: dict | None = None,
        detect: dict | None = None,
        un_detect: dict | None = None,
        resolve: dict | None = None,
        places: dict | None = None,
        children: dict | None = None,
        enum_members: dict | None = None,
        currency_units: dict | None = None,
        resolve_any: dict | None = None,
        facet_policies: Mapping[tuple[str, str], FacetDefaultPolicy] | None = None,
        raise_on_call: bool = False,
    ):
        self._generation = FakeGeneration(token=f"fake-graph-{next(_FAKE_GENERATIONS)}")
        if raise_on_call:
            self._impl = _RaiseOnAny()
            return
        self._impl = None
        self._nodes: dict = (
            _load_json("graph_nodes.json") if nodes is None else _merge_schema_nodes(nodes)
        )
        self._obs: dict = obs if obs is not None else _load_json("graph_obs.json")
        self._observed_places = frozenset(key.partition("|")[2] for key in self._obs) - {""}
        # Declared default facets keyed by (`dcid:`-prefixed StatVar ref, facet axis).
        self._facet_policies = dict(facet_policies or {})
        if detect is None:
            self._detect, recorded_un = _load_detect_fixture()
            self._un_detect = recorded_un if un_detect is None else un_detect
        else:
            self._detect = detect
            self._un_detect = un_detect if un_detect is not None else {}
        self._resolve: dict = resolve if resolve is not None else _load_json("graph_resolve.json")
        # Region-name mappings used by constrained and bare fallback rungs.
        self._places: dict = places if places is not None else _load_json("graph_places.json")
        # Test-injected answers for bare-rung names absent from _resolve and _places.
        self._resolve_any: dict = resolve_any if resolve_any is not None else {}
        # Child dcids keyed by parent and type.
        self._children: dict = (
            children if children is not None else _load_json("graph_children.json")
        )
        # Closed-enum members keyed by property dcid.
        self._enum_members: dict = (
            enum_members if enum_members is not None else _load_json("graph_enum_members.json")
        )
        # CurrencyUnit dcid to ISO 4217 alpha-3 code.
        self._currency_units: dict = (
            currency_units
            if currency_units is not None
            else _load_json("graph_currency_units.json")
        )

    def node_label(self, dcid: str) -> str | None:
        if self._impl is not None:
            return self._impl.node_label(dcid)
        node = self._nodes.get(dcid)
        if node is None:
            return None
        return node.get("label")

    def node_arcs(self, dcid: str) -> dict | None:
        if self._impl is not None:
            return self._impl.node_arcs(dcid)
        node = self._nodes.get(dcid)
        if node is None:
            return None
        arcs = node.get("arcs")
        return _dedup_wire_arcs(arcs) if arcs else None

    def node_type(self, dcid: str) -> str | None:
        if self._impl is not None:
            return self._impl.node_type(dcid)
        node = self._nodes.get(dcid)
        if node is None:
            return None
        return node.get("type")

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._generation.stamp()

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        if self._impl is not None:
            return self._impl.default_facet_policy(stat_var=stat_var, axis=axis)
        return self._facet_policies.get((stat_var, axis))

    def resolve_by_arc(
        self,
        names: Sequence[str],
        *,
        arc: str,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        """Replay constrained and bare fallback rungs from fixture tables.

        The bare rung consults _resolve, then _places, then _resolve_any. Every requested name
        is keyed. Misses map to [].
        """
        if self._impl is not None:
            return self._impl.resolve_by_arc(names, arc=arc)
        type_of = _arc_type_of(arc)
        return {name: self._resolve_by_arc_one(name, type_of=type_of) for name in names}

    def _resolve_by_arc_one(self, name: str, *, type_of: str | None) -> list[str]:
        if type_of == "Country":
            dcid = self._resolve.get(name)
            return [dcid] if dcid else []
        if type_of in ("Continent", "UNGeoRegion"):
            entry = self._places.get(name)
            if entry is not None and entry["type_of"] == type_of:
                return [entry["dcid"]]
            return []
        # Bare description rung.
        dcid = self._resolve.get(name)
        if dcid:
            return [dcid]
        entry = self._places.get(name)
        if entry is not None:
            return [entry["dcid"]]
        any_dcid = self._resolve_any.get(name)
        return [any_dcid] if any_dcid else []

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Read typeOf from type, name from label, and other properties from arcs.

        Missing nodes map to None. Absent properties map to ().
        """
        if self._impl is not None:
            return self._impl.node_arcs_many(dcids, properties=properties)
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            node = self._nodes.get(dcid, _PLACE_NODE if dcid in self._observed_places else None)
            if node is None:
                result[dcid] = None
                continue
            arcs = node.get("arcs", {})
            entry: dict[str, tuple[str, ...]] = {}
            for prop in properties:
                if prop == "typeOf":
                    type_of = node.get("type")
                    entry[prop] = (type_of,) if type_of else ()
                elif prop == "name":
                    label = node.get("label")
                    entry[prop] = (label,) if label else ()
                else:
                    nodes = arcs.get(prop, {}).get("nodes", [])
                    entry[prop] = tuple(
                        dict.fromkeys(
                            v for n in nodes if (v := (n.get("dcid") or n.get("value"))) is not None
                        )
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
        """`node_arcs_many`'s values, each node target carrying the label and type this fake
        holds for it, as the mixer answers beside an arc's targets."""
        arcs = self.node_arcs_many(dcids, properties=properties, deadline=deadline)
        result: dict[str, dict[str, tuple[ArcTarget, ...]] | None] = {}
        for dcid, entry in arcs.items():
            if entry is None:
                result[dcid] = None
                continue
            result[dcid] = {}
            for prop, values in entry.items():
                targets = []
                for value in values:
                    node = self._nodes.get(value) if prop not in ("name", "typeOf") else None
                    type_of = node.get("type") if node is not None else None
                    targets.append(
                        ArcTarget(
                            value=value,
                            name=node.get("label") if node is not None else None,
                            types=(type_of,) if type_of else (),
                        )
                    )
                result[dcid][prop] = tuple(targets)
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return the label followed by deduplicated alternate names. Missing nodes map to ()."""
        if self._impl is not None:
            return self._impl.node_names_many(dcids)
        result: dict[str, tuple[str, ...]] = {}
        for dcid in dict.fromkeys(dcids):
            node = self._nodes.get(dcid)
            if node is None:
                result[dcid] = ()
                continue
            label = node.get("label")
            arcs = node.get("arcs") or {}
            alternates = [
                v
                for n in arcs.get("alternateName", {}).get("nodes", [])
                if (v := (n.get("dcid") or n.get("value"))) is not None
            ]
            result[dcid] = tuple(dict.fromkeys(([label] if label else []) + alternates))
        return result

    def child_places_batch(
        self,
        parent_dcids: list[str],
        child_type: str,
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        """Batch child-place lookup from the _children fixture."""
        if self._impl is not None:
            return self._impl.child_places_batch(parent_dcids, child_type)
        return {
            parent: list(dict.fromkeys(self._children.get(parent, {}).get(child_type, [])))
            for parent in parent_dcids
        }

    def detect_svs(self, query: str) -> Detection:
        """Admit the recorded resolves as `LiveGraphClient.detect_svs` does."""
        if self._impl is not None:
            return self._impl.detect_svs(query)
        return admit_legs(
            self.detect_legs(query), catalogue=current_catalogue(), detect_k=settings().detect_k
        )

    def detect_legs(self, query: str) -> DetectLegs:
        """The recorded resolves for `query`; a query never recorded resolves to nothing. The UN
        resolve is skipped when the bound catalogue has the UN origin off."""
        if self._impl is not None:
            return self._impl.detect_legs(query)
        ours = detect_leg_from_json(self._detect.get(query, {}))
        if not current_catalogue().un_system_enabled:
            return DetectLegs(ours=ours, un=())
        un = self._un_detect.get(query, {})
        return DetectLegs(ours=ours, un=None if un is None else detect_leg_from_json(un))

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        if self._impl is not None:
            return self._impl.observation_facets(
                stat_var=stat_var, entity=entity, needs_dates=needs_dates
            )
        key = f"{stat_var}|{entity}"
        raw = self._obs.get(key, [])
        return [
            Facet(
                earliest_date=f.get("earliestDate"),
                latest_date=f.get("latestDate"),
                obs_count=f.get("obsCount", 0),
                dates=tuple(
                    f.get(
                        "dates",
                        [o["date"] for o in f.get("observations", []) if o.get("date")],
                    )
                ),
                values=tuple(
                    f.get(
                        "values",
                        [o.get("value") for o in f.get("observations", []) if o.get("date")],
                    )
                ),
                provenance_id=facet_provenance(stat_var, f),
                measurement_method=f.get("measurementMethod"),
                unit=f.get("unit"),
                # The live client reads the facet's `observationPeriod` too.
                observation_period=f.get("observationPeriod"),
                # The mixer names every facet. A fixture without `facetId` takes the position
                # id `materialise_double_from_obs` gives the same facet.
                facet_id=f.get("facetId") or f"f{i}",
            )
            for i, f in enumerate(raw)
        ]

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        """Fetch fixture labels, preferring fixture index CodeRow labels when available."""
        if self._impl is not None:
            return self._impl.node_labels_batch(dcids)
        index = get_index()
        result: dict[str, str] = {}
        for dcid in dcids:
            node = self._nodes.get(dcid)
            if node is None:
                continue
            label = node.get("label")
            if not label:
                continue
            row = index.code_by_dcid(dcid)
            result[dcid] = row.label if row is not None else label
        return result

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        """Return types for typed fixture nodes."""
        if self._impl is not None:
            return self._impl.node_types_batch(dcids)
        result: dict[str, str] = {}
        for dcid in dcids:
            node = self._nodes.get(dcid)
            if node is None:
                continue
            type_of = node.get("type")
            if type_of:
                result[dcid] = type_of
        return result

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        """Return arcs for fixture nodes. Missing or empty arcs map to None."""
        if self._impl is not None:
            return self._impl.node_arcs_batch(dcids)
        result: dict[str, dict | None] = {}
        for dcid in dcids:
            node = self._nodes.get(dcid)
            if node is None:
                result[dcid] = None
                continue
            arcs = node.get("arcs")
            result[dcid] = _dedup_wire_arcs(arcs) if arcs else None
        return result

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        """Return observationProperties dcids from recorded arcs. Missing arcs return ()."""
        if self._impl is not None:
            return self._impl.node_observation_properties_batch(dcids)
        result: dict[str, tuple[str, ...]] = {}
        for dcid in dcids:
            node = self._nodes.get(dcid)
            arcs = node.get("arcs") if node is not None else None
            result[dcid] = (
                tuple(dict.fromkeys(_dcids_for(arcs, "observationProperties"))) if arcs else ()
            )
        return result

    def child_dcids(self, parent_dcid: str) -> list[str]:
        """Reverse-scan recorded arcs for children of parent_dcid."""
        if self._impl is not None:
            return self._impl.child_dcids(parent_dcid)
        result = []
        for dcid, node in self._nodes.items():
            arcs = node.get("arcs", {})
            parent_nodes = arcs.get("isPartOf", {}).get("nodes", [])
            if any(n.get("dcid") == parent_dcid for n in parent_nodes):
                result.append(dcid)
        return result

    def member_arcs(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every node's arcs in the tuple shape the mint's member walk reads, each value its
        dcid or literal, or `UNRESOLVED_NODE` for a target holding neither (a provisional
        node), as the live client reads it, with `typeOf` read from `type` when the arcs
        omit it, or `StatisticalVariable` for a node carrying `measuredProperty`."""
        result: dict[str, dict[str, tuple[str, ...]]] = {}
        if getattr(self, "_impl", None) is not None:
            # A graph that raises on every call holds no node.
            return result
        for dcid, node in getattr(self, "_nodes", {}).items():
            if not isinstance(node, dict):
                continue
            arcs = {
                prop: tuple(
                    dict.fromkeys(
                        n.get("dcid") or n.get("value") or UNRESOLVED_NODE
                        for n in arc.get("nodes", [])
                    )
                )
                for prop, arc in (node.get("arcs") or {}).items()
            }
            if "typeOf" not in arcs and node.get("type"):
                arcs["typeOf"] = (node["type"],)
            elif "typeOf" not in arcs and "measuredProperty" in arcs:
                # Only a StatVar carries measuredProperty, and fixtures written before the
                # walk read `typeOf` leave it out.
                arcs["typeOf"] = ("StatisticalVariable",)
            result[dcid] = arcs
        return result

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, SubjectsPage]:
        """Reverse-scan arcs for each requested dcid and cap results at max_results.

        Every request gets a key. Empty results have complete coverage. `deadline` does not
        limit fixture reads.
        """
        if self._impl is not None:
            return self._impl.node_subjects_many(dcids, property=property, max_results=max_results)
        result: dict[str, SubjectsPage] = {}
        for dcid in dcids:
            subjects: list[str] = []
            seen: set[str] = set()
            for subject_dcid, node in self._nodes.items():
                # The fixture's `_comment` entry is no node.
                if not isinstance(node, dict):
                    continue
                arcs = node.get("arcs") or {}
                leads = arcs.get(property, {}).get("nodes", [])
                if any(n.get("dcid") == dcid for n in leads) and subject_dcid not in seen:
                    seen.add(subject_dcid)
                    subjects.append(subject_dcid)
            capped = len(subjects) > max_results
            result[dcid] = SubjectsPage(
                dcids=tuple(subjects[:max_results]) if capped else tuple(subjects),
                coverage=SubjectsCoverage.capped if capped else SubjectsCoverage.complete,
            )
        return result

    def enum_members(self, property_dcid: str) -> list[str]:
        """Return deduplicated enum members from the fixture."""
        if self._impl is not None:
            return self._impl.enum_members(property_dcid)
        return list(dict.fromkeys(self._enum_members.get(property_dcid, [])))

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        """Return a copy of the currency-unit map."""
        if self._impl is not None:
            return self._impl.currency_units(deadline=deadline)
        return dict(self._currency_units)

    def entities_with_data(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        max_bytes: int | None = None,
    ) -> list[str]:
        """Return entities with a non-empty facet for any variable, preserving input order.

        `max_bytes` preserves interface compatibility and does not limit fixture reads.
        """
        if self._impl is not None:
            return self._impl.entities_with_data(variable_dcids, entity_dcids, max_bytes=max_bytes)
        if not variable_dcids or not entity_dcids:
            return []
        return [
            entity
            for entity in entity_dcids
            if any(self.observation_facets(stat_var=var, entity=entity) for var in variable_dcids)
        ]

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Return a separate ordered entity list for each variable."""
        if self._impl is not None:
            return self._impl.entities_with_data_by_variable(
                variable_dcids, entity_dcids, deadline=deadline
            )
        if not variable_dcids or not entity_dcids:
            return {}
        result: dict[str, list[str]] = {}
        for var in variable_dcids:
            result[var] = [
                entity
                for entity in entity_dcids
                if self.observation_facets(stat_var=var, entity=entity)
            ]
        return result

    @property
    def scope_availability(self) -> ScopeAvailability:
        """The default place scope's availability read, answered from this graph's observations:
        every place holding a facet for a StatVar, whatever its type. A property, so a recorder
        of the double's public methods does not log the test's own wiring as a request."""
        return scope_availability_of(self._obs)

    def un_availability(self) -> ScopeAvailability | None:
        """The UN deployment's availability read, answered from this graph's observations."""
        if self._impl is not None:
            return self._impl.un_availability()
        return self.scope_availability

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        """Read every StatVar/entity pair from the fixture. Neither max_bytes nor deadline
        limits fixture data."""
        if self._impl is not None:
            return self._impl.observation_facets_batch(
                stat_vars, entities, needs_dates=needs_dates, max_bytes=max_bytes
            )
        result: dict[str, dict[str, list[Facet]]] = {}
        for sv in stat_vars:
            result[sv] = {}
            for entity in entities:
                result[sv][entity] = self.observation_facets(
                    stat_var=sv, entity=entity, needs_dates=needs_dates
                )
        return result

    def observation_read_groups(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> list[list[str]]:
        """One group: the fixture reads every StatVar of a read together, as a POST does."""
        if self._impl is not None:
            return self._impl.observation_read_groups(stat_vars, entities, needs_dates)
        return [list(stat_vars)]

    def exists(self, dcid: str) -> bool:
        if self._impl is not None:
            return self._impl.exists(dcid)
        return dcid in self._nodes


# India's five nominal GDP facets in `graph_obs.json` carry no provenance there, as the
# `resolved_sources` tests need. `select_facet` refuses a facet without one, so a test that
# needs the preview to serve India's GDP names each facet's import, in fixture order.
GDP_INDIA_PROVENANCES: Mapping[str, Sequence[str]] = {
    "Amount_EconomicActivity_GrossDomesticProduction_Nominal|country/IND": (
        "dc/base/WorldDevelopmentIndicators",
        "dc/base/IMF_WorldEconomicOutlook",
        "dc/base/India_NationalAccountsStatistics",
        "dc/base/OECD_NationalAccounts",
        "dc/base/India_QuarterlyGDP",
    )
}


def fake_graph_with_provenances(provenances: Mapping[str, Sequence[str]]) -> FakeGraph:
    """`FakeGraph` over `graph_obs.json` with each listed series' facets carrying the given
    provenance ids, in facet order."""
    obs = _load_json("graph_obs.json")
    for key, provenance_ids in provenances.items():
        rows = obs[key]
        assert len(rows) == len(provenance_ids), key
        obs[key] = [
            {**row, "provenanceId": provenance_id}
            for row, provenance_id in zip(rows, provenance_ids, strict=True)
        ]
    return FakeGraph(obs=obs)
