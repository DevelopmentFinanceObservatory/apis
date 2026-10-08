"""A UN deployment for tests: StatVarGroups, variables and topics over `/v2/node`,
and the site's group counts."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from dc_graph import FrontDoorRefused, GraphReadError

WHO = "undata/p/WHO"
WHO_ROOT = "undata/g/who"
MORT_GROUP = "undata/g/who/MORT"
MORT_BY_SEX = "undata/g/who/MORT.001"
ODA_GROUP = "undata/g/who/ODA"

MORT_SERIES = "undata/who/SERIES-MORT"
ODA_SERIES = "undata/who/SERIES-ODA"
MORT = "undata/who/MORT"
MORT_FEMALE = "undata/who/MORT.SEX--F"
MORT_MALE = "undata/who/MORT.SEX--M"
ODA = "undata/who/ODA_1"

TOPIC_ROOT = "undata/topic/Root"
WHO_TOPIC = "undata/topic/who"
MORT_TOPIC = "undata/topic/who/MORT"
ODA_TOPIC = "undata/topic/who/ODA"
MORT_PEERS = "undata/svpg/who/MORT.001"

type Arcs = dict[str, list[dict[str, Any]]]


def target(dcid: str, name: str | None = None, *types: str) -> dict[str, Any]:
    """One arc target as `/v2/node` answers it."""
    node: dict[str, Any] = {"dcid": dcid, "provenanceId": WHO}
    if name is not None:
        node["name"] = name
    if types:
        node["types"] = list(types)
    return node


def group(parent: str, name: str) -> Arcs:
    return {
        "typeOf": [target("StatVarGroup", "StatVarGroup", "Class")],
        "name": [{"value": name, "provenanceId": WHO}],
        "specializationOf": [target(parent, None, "StatVarGroup")],
    }


def variable(
    *, name: str, series: str | None, member_of: str, sex: tuple[str, str] | None = None
) -> Arcs:
    arcs: Arcs = {
        "typeOf": [target("StatisticalVariable", "StatisticalVariable", "Class")],
        "name": [{"value": name, "provenanceId": WHO}],
        "measuredProperty": [target("value", "value", "Property")],
        "statType": [target("measuredValue", "measuredValue", "Property")],
        "memberOf": [target(member_of, None, "StatVarGroup")],
    }
    if series is not None:
        arcs["populationType"] = [target(series, f"{series} name", "undata/who/SERIES")]
    if sex is not None:
        arcs["constraintProperties"] = [target("undataSex", "Sex", "Property")]
        arcs["undataSex"] = [target(sex[0], sex[1], "undata/SEX")]
    return arcs


def topic(*children: dict[str, Any], arc: str = "relevantVariable", kind: str = "Topic") -> Arcs:
    return {"typeOf": [target(kind, kind, "Class")], arc: list(children)}


def sample_nodes() -> dict[str, Arcs]:
    """WHO with two series: mortality, whose by-sex variables sit in a group of
    their own, and an aid series. The topic tree reaches every variable."""
    statvar = "StatisticalVariable"
    return {
        WHO_ROOT: group("undata/g/Root", "WHO (UN Data)"),
        MORT_GROUP: group(WHO_ROOT, "Mortality"),
        MORT_BY_SEX: group(MORT_GROUP, "Mortality, by Sex"),
        ODA_GROUP: group(WHO_ROOT, "Aid for health"),
        MORT: variable(name="Mortality", series=MORT_SERIES, member_of=MORT_GROUP),
        MORT_FEMALE: variable(
            name="Mortality [Sex=Female]",
            series=MORT_SERIES,
            member_of=MORT_BY_SEX,
            sex=("undata/SEX-F", "Female"),
        ),
        MORT_MALE: variable(
            name="Mortality [Sex=Male]",
            series=MORT_SERIES,
            member_of=MORT_BY_SEX,
            sex=("undata/SEX-M", "Male"),
        ),
        ODA: variable(name="Aid for health", series=ODA_SERIES, member_of=ODA_GROUP),
        TOPIC_ROOT: topic(target(WHO_TOPIC, None, "Topic")),
        WHO_TOPIC: topic(target(MORT_TOPIC, None, "Topic"), target(ODA_TOPIC, None, "Topic")),
        MORT_TOPIC: topic(
            target(MORT, None, statvar), target(MORT_PEERS, None, "StatVarPeerGroup")
        ),
        MORT_PEERS: topic(
            target(MORT_FEMALE, None, statvar),
            target(MORT_MALE, None, statvar),
            arc="member",
            kind="StatVarPeerGroup",
        ),
        ODA_TOPIC: topic(target(ODA, None, statvar)),
    }


class FakeUn:
    """`UnMirrorGraph` and `UnGroupCounts` over `nodes`, each node's out-arcs.

    A `<-property` read answers `page_size` subjects per node and page. An
    `->*` read answers whole nodes up to `arc_targets_per_page` targets, then
    a token; one node over it pages through its targets.

    Hazards: a `<-` read naming a node in `repeating` that needs a second
    page answers the same token from then on, and one naming a node in
    `idle` answers a fresh token and the first page again. With
    `batches_repeat`, a `<-` read naming several nodes does as one naming a
    `repeating` node does, and each of its nodes reads well alone. The next
    `replaying[node]` reads naming the node with a token answer the first
    page and that token again, `<-` and `->*` alike. A node in `truncating`
    answers only that many subjects and no token. `counts` overrides a
    group's count, `None` removing it. A read whose nodes include every
    dcid of one set in `refused` is refused as the front door refuses it,
    with the `GraphReadError` `GraphClient.post` raises from
    `FrontDoorRefused`. `failing` fails every read, and the first
    `rate_limited` reads are answered with HTTP 429.
    """

    def __init__(self, nodes: Mapping[str, Arcs] | None = None, *, page_size: int = 500) -> None:
        self.nodes = dict(sample_nodes() if nodes is None else nodes)
        self.page_size = page_size
        self.arc_targets_per_page = 500
        self.repeating: set[str] = set()
        self.idle: set[str] = set()
        self.batches_repeat = False
        self.replaying: dict[str, int] = {}
        self.truncating: dict[str, int] = {}
        self.counts: dict[str, int | None] = {}
        self.refused: list[set[str]] = []
        self.failing = False
        self.rate_limited = 0
        self.delay = 0.0
        self.requests: list[dict] = []
        self.count_reads: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        return None

    # --- /v2/node -------------------------------------------------------------

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict:
        assert path == "/v2/node"
        with self._lock:
            self.requests.append(payload)
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if any(dcids <= set(payload["nodes"]) for dcids in self.refused):
                raise GraphReadError(
                    f"graph returned HTTP 403 for {path!r}", upstream_status=403
                ) from FrontDoorRefused(path)
            if self.failing:
                raise GraphReadError("graph returned HTTP 503 for /v2/node", upstream_status=503)
            with self._lock:
                limited = self.rate_limited > 0
                self.rate_limited -= limited
            if limited:
                raise GraphReadError("graph returned HTTP 429 for /v2/node", upstream_status=429)
            arc = payload["property"]
            if arc == "->*":
                return self._out_arcs(payload["nodes"], payload.get("nextToken"))
            assert arc.startswith("<-")
            return self._subjects(payload["nodes"], arc[2:], payload.get("nextToken"))
        finally:
            with self._lock:
                self._in_flight -= 1

    def subjects_of(self, node: str, property: str) -> list[str]:
        found = sorted(
            dcid
            for dcid, arcs in self.nodes.items()
            if any(entry.get("dcid") == node for entry in arcs.get(property, []))
        )
        return found[: self.truncating.get(node, len(found))]

    def _replays(self, nodes: list[str], token: str | None) -> bool:
        """Whether this read answers its first page and its own token again."""
        if token is None:
            return False
        with self._lock:
            node = next((node for node in nodes if self.replaying.get(node, 0) > 0), None)
            if node is not None:
                self.replaying[node] -= 1
        return node is not None

    def _subjects(self, nodes: list[str], property: str, token: str | None) -> dict:
        if self._replays(nodes, token):
            return {**self._subjects(nodes, property, None), "nextToken": token}
        page = int(token) if token is not None else 0
        idle = bool(self.idle.intersection(nodes))
        stuck = bool(self.repeating.intersection(nodes)) or (len(nodes) > 1 and self.batches_repeat)
        shown = 0 if idle else page
        data: dict[str, dict] = {}
        more = False
        for node in nodes:
            subjects = self.subjects_of(node, property)
            rows = subjects[shown * self.page_size : (shown + 1) * self.page_size]
            more = more or len(subjects) > (page + 1) * self.page_size
            data[node] = (
                {"arcs": {property: {"nodes": [target(dcid) for dcid in rows]}}} if rows else {}
            )
        body: dict[str, Any] = {"data": data}
        if idle:
            body["nextToken"] = str(page + 1)
        elif stuck and (more or page):
            body["nextToken"] = "1"
        elif more:
            body["nextToken"] = str(page + 1)
        return body

    def _out_arcs(self, nodes: list[str], token: str | None) -> dict:
        if self._replays(nodes, token):
            return {**self._out_arcs(nodes, None), "nextToken": token}
        limit = self.arc_targets_per_page
        if len(nodes) == 1:
            node = nodes[0]
            targets = [
                (property, entry)
                for property, entries in self.nodes.get(node, {}).items()
                for entry in entries
            ]
            start = int(token) if token is not None else 0
            arcs: dict[str, dict] = {}
            for property, entry in targets[start : start + limit]:
                arcs.setdefault(property, {"nodes": []})["nodes"].append(entry)
            body: dict[str, Any] = {"data": {node: {"arcs": arcs} if arcs else {}}}
            if start + limit < len(targets):
                body["nextToken"] = str(start + limit)
            return body
        data: dict[str, dict] = {}
        used = 0
        for node in nodes:
            arcs_of = self.nodes.get(node, {})
            used += sum(len(entries) for entries in arcs_of.values())
            if used > limit:
                return {"data": data, "nextToken": "more"}
            data[node] = (
                {"arcs": {property: {"nodes": entries} for property, entries in arcs_of.items()}}
                if arcs_of
                else {}
            )
        return {"data": data}

    # --- Group counts ---------------------------------------------------------

    def descendants(self, group_dcid: str) -> set[str]:
        """The variables of `group_dcid` and every group under it, untruncated."""
        variables: set[str] = set()
        stack = [group_dcid]
        while stack:
            current = stack.pop()
            for dcid, arcs in self.nodes.items():
                if any(entry.get("dcid") == current for entry in arcs.get("memberOf", [])):
                    variables.add(dcid)
                if any(entry.get("dcid") == current for entry in arcs.get("specializationOf", [])):
                    stack.append(dcid)
        return variables

    def __call__(self, group: str) -> dict[str, int]:
        self.count_reads.append(group)
        if self.failing:
            raise GraphReadError("UN group count read failed", upstream_status=503)
        children = [
            dcid
            for dcid, arcs in self.nodes.items()
            if any(entry.get("dcid") == group for entry in arcs.get("specializationOf", []))
        ]
        counts: dict[str, int | None] = {
            dcid: len(self.descendants(dcid)) for dcid in (group, *children)
        }
        counts.update(self.counts)
        return {dcid: count for dcid, count in counts.items() if count is not None}
