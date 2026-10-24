"""Reads `/v2/node` arcs and subjects of many nodes, by the rules the UN
tables (`origin_catalogue.un_mirror`) and the arc tables
(`origin_catalogue.arc_reads`) are both read by.

The UN front door refuses some `/v2/node` POSTs by their text, with an HTML 403
(`dc_graph.FrontDoorRefused`). A body naming three particular topics together is
refused, and each topic alone passes. A refused read naming several nodes is
read again as two halves, each halved again while refused, and a node refused
on its own is an `un_read_refused` fault.

A read naming several nodes sometimes leaves one out of its answer, or
answers it with no arcs, depending on which others share it. Only a read
naming one node and answering it with no arcs says the node has none, so a
node a read naming several leaves out or answers empty is read again alone,
and a node left out of a read naming it alone is an `un_node_omitted` fault.
A caller that knows from elsewhere that the empty answers are right reads
with `reread_empty=False`, and only the left-out nodes are read again alone.

The graph sometimes answers a page request with a token the read has
already been given, beside the first page again, and answers the same request
correctly a second later. Such a page is requested again after 1, 2 and 4
seconds, and only one that repeats a token on all four attempts stops the
read. A read that stops is an `un_enumeration_stalled` fault. A read answered
with HTTP 429 is repeated after a wait, three times at most.

A fault raises `NodeReadFault`. `iter_node_arcs` and `iter_node_labels`, the
reads of our mixer, raise it as `dc_graph.GraphReadError`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

from dc_graph import FrontDoorRefused, GraphReadError, label_or_self_name

READ_CONCURRENCY: Final[int] = 4
"""The most requests `iter_node_arcs` and `iter_node_labels` keep in flight."""

NODES_PER_REQUEST: Final[int] = 25
"""Nodes named in one read. A `->*` answer pages at about 500 arc targets,
and a variable carries up to about 15."""

_NODE_PATH: Final[str] = "/v2/node"

_LABEL_NODES_PER_REQUEST: Final[int] = 500
"""Nodes named in one `->name` read, as many as `dc_graph.GraphClient.node_labels_batch`
names in one."""

_MAX_PAGES: Final[int] = 400
"""Pages one read may take. The largest group and topic lists end well inside it."""

_STALLED_PAGE_LIMIT: Final[int] = 3
"""Consecutive pages that add nothing before a read counts as stalled."""

_REPEATED_PAGE_WAITS_SECONDS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0)
"""The waits before each further attempt at a page the graph answered
with a token the read had already been given. A page that still repeats one
after the last wait, 7 seconds in all, stops the read."""

_RATE_LIMITED: Final[int] = 429
_RATE_LIMIT_WAITS_SECONDS: Final[tuple[float, ...]] = (10.0, 30.0, 60.0)
"""The waits before each further attempt at a read the UN front door
answered with HTTP 429. It rate-limits a client that has sent it tens of
thousands of requests within the hour."""

type NodeReadFaultKind = Literal["un_enumeration_stalled", "un_node_omitted", "un_read_refused"]


class NodeGraph(Protocol):
    """The `/v2/node` POST, which `dc_graph.GraphClient` provides."""

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict: ...


class NodeReadFault(Exception):
    """A read that cannot be answered whole: `subject` is the node at fault."""

    def __init__(self, kind: NodeReadFaultKind, subject: str, detail: str) -> None:
        super().__init__(detail)
        self.kind: NodeReadFaultKind = kind
        self.subject = subject
        self.detail = detail


class _NotAdvancing(Exception):
    """A paged read repeated a token on every attempt at one page, added
    nothing for `_STALLED_PAGE_LIMIT` pages, or passed `_MAX_PAGES`."""


class _Refused(Exception):
    """The UN front door refused a `/v2/node` POST by its text (`FrontDoorRefused`)."""


@dataclass
class NodeRun:
    """One run of reads: its graph, worker pool, deadline, request and page
    retry counters and the strings its answers share."""

    graph: NodeGraph
    pool: ThreadPoolExecutor
    deadline: float | None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _requests: int = 0
    _page_retries: int = 0
    _strings: dict[str, str] = field(default_factory=dict)

    def post(self, payload: dict) -> dict:
        """One `/v2/node` read. An HTTP 429 is read again after each of
        `_RATE_LIMIT_WAITS_SECONDS`, and raises once they are spent. The
        front door's refusal raises `_Refused`, which `_halving` turns into
        reads of the halves."""
        waits = iter(_RATE_LIMIT_WAITS_SECONDS)
        while True:
            with self._lock:
                self._requests += 1
            try:
                return self.graph.post(_NODE_PATH, payload, deadline=self.deadline)
            except GraphReadError as exc:
                if isinstance(exc.__cause__, FrontDoorRefused):
                    raise _Refused from exc
                wait = next(waits, None)
                if exc.upstream_status != _RATE_LIMITED or wait is None:
                    raise
            time.sleep(wait)

    def page(self, payload: dict, given: set[str]) -> dict:
        """One page of a paged read, whose answer carries no token or one
        outside `given`, the tokens the read has been given so far. An
        answer repeating one of them is dropped and the page requested again
        after each of `_REPEATED_PAGE_WAITS_SECONDS`. Raises `_NotAdvancing`
        once they are spent, or when the next wait would pass the deadline."""
        waits = iter(_REPEATED_PAGE_WAITS_SECONDS)
        while True:
            body = self.post(payload)
            if body.get("nextToken") not in given:
                return body
            wait = next(waits, None)
            if wait is None or (
                self.deadline is not None and time.monotonic() + wait >= self.deadline
            ):
                raise _NotAdvancing
            with self._lock:
                self._page_retries += 1
            time.sleep(wait)

    def requests(self) -> int:
        with self._lock:
            return self._requests

    def page_retries(self) -> int:
        with self._lock:
            return self._page_retries

    def map[T, R](self, read: Callable[[T], R], items: Sequence[T]) -> Iterator[R]:
        """`read` over `items` on the pool, in order. Each result is let go
        once yielded, so a caller that keeps less than a read returns holds
        less than every read. A failure cancels the reads not yet started."""
        futures = deque(self.pool.submit(read, item) for item in items)
        try:
            while futures:
                yield futures.popleft().result()
        finally:
            for future in futures:
                future.cancel()

    def string(self, value: str) -> str:
        return self._strings.setdefault(value, value)


def chunks[T](items: Sequence[T], size: int = NODES_PER_REQUEST) -> list[Sequence[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _halving[R](
    read: Callable[[Sequence[str]], dict[str, R]], nodes: Sequence[str], arc: str
) -> dict[str, R]:
    """`read(nodes)`, or, when the front door refuses it, the merged reads of
    the two halves, each through `_halving` again. The halves are read in the
    calling worker, since waiting on the pool from inside it can take every
    worker. A refused read naming one node is an `un_read_refused` fault."""
    try:
        return read(nodes)
    except _Refused:
        if len(nodes) == 1:
            raise NodeReadFault(
                "un_read_refused",
                nodes[0],
                f"the UN front door refused a {_NODE_PATH} read of {arc} of {nodes[0]}",
            ) from None
    middle = len(nodes) // 2
    return {**_halving(read, nodes[:middle], arc), **_halving(read, nodes[middle:], arc)}


def _stalled(node: str, arc: str) -> NodeReadFault:
    return NodeReadFault(
        "un_enumeration_stalled",
        node,
        f"the graph stopped advancing while paging {arc} of {node}",
    )


def _omitted(node: str, arc: str) -> NodeReadFault:
    return NodeReadFault(
        "un_node_omitted",
        node,
        f"the graph left {node} out of a {_NODE_PATH} read of {arc} naming it alone",
    )


# --- Subjects ----------------------------------------------------------------


def _subjects_paged(
    run: NodeRun, nodes: Sequence[str], property: str
) -> dict[str, dict[str, None]]:
    """Every `<-property` subject of each node, from one paged read naming
    them all. Raises `_NotAdvancing` when the read stops advancing and
    `_Refused` when the front door refuses a page."""
    found: dict[str, dict[str, None]] = {node: {} for node in nodes}
    tokens: set[str] = set()
    token: str | None = None
    idle_pages = 0
    for _ in range(_MAX_PAGES):
        payload: dict = {"nodes": list(nodes), "property": f"<-{property}"}
        if token is not None:
            payload["nextToken"] = token
        body = run.page(payload, tokens)
        before = sum(len(subjects) for subjects in found.values())
        data = body.get("data") or {}
        for node in nodes:
            arcs = (data.get(node) or {}).get("arcs") or {}
            for entry in (arcs.get(property) or {}).get("nodes", []):
                if entry.get("dcid"):
                    found[node][run.string(entry["dcid"])] = None
        token = body.get("nextToken")
        if token is None:
            return found
        idle_pages = (
            idle_pages + 1 if sum(len(subjects) for subjects in found.values()) == before else 0
        )
        if idle_pages >= _STALLED_PAGE_LIMIT:
            raise _NotAdvancing
        tokens.add(token)
    raise _NotAdvancing


def read_subjects(run: NodeRun, nodes: Sequence[str], property: str) -> dict[str, tuple[str, ...]]:
    """Every `<-property` subject of each node. A read naming several nodes
    that stops advancing is read again one node at a time, and a single
    node's read that stops advancing is a fault. A read the front door
    refuses is read again in halves (`_halving`), each with the same
    fallback."""
    arc = f"<-{property}"

    def read(chunk: Sequence[str]) -> dict[str, dict[str, None]]:
        try:
            return _subjects_paged(run, chunk, property)
        except _NotAdvancing:
            if len(chunk) == 1:
                raise _stalled(chunk[0], arc) from None
        found: dict[str, dict[str, None]] = {}
        for node in chunk:
            try:
                found.update(_subjects_paged(run, [node], property))
            except _NotAdvancing:
                raise _stalled(node, arc) from None
        return found

    subjects: dict[str, tuple[str, ...]] = {}
    for found in run.map(lambda chunk: _halving(read, chunk, arc), chunks(nodes)):
        subjects.update({node: tuple(dcids) for node, dcids in found.items()})
    return subjects


# --- Arcs --------------------------------------------------------------------


def _arc_pages(run: NodeRun, node: str, first: dict, arc: str) -> dict[str, list[dict]]:
    """One node's `arc` targets by property, following `first`'s token to the end."""
    merged: dict[str, list[dict]] = {}
    tokens: set[str] = set()
    body = first
    for _ in range(_MAX_PAGES):
        arcs = ((body.get("data") or {}).get(node) or {}).get("arcs") or {}
        for property, targets in arcs.items():
            merged.setdefault(property, []).extend(targets.get("nodes", []))
        token = body.get("nextToken")
        if token is None:
            return merged
        tokens.add(token)
        try:
            body = run.page({"nodes": [node], "property": arc, "nextToken": token}, tokens)
        except _NotAdvancing:
            break
    raise _stalled(node, arc)


def read_arcs(
    run: NodeRun, nodes: Sequence[str], arc: str = "->*", *, reread_empty: bool = True
) -> dict[str, dict[str, list[dict]]]:
    """Each node's `arc` targets by property, `->*` by default. A read naming
    several nodes that pages, or that the front door refuses, is split in
    half, and a single node's pages are followed. A single node whose first
    or later page the front door refuses is an `un_read_refused` fault. A
    node a read naming several leaves out is read again alone, and so is one
    it answers empty unless `reread_empty` is false."""
    return _halving(lambda chunk: _arcs_read(run, chunk, arc, reread_empty), nodes, arc)


def _arcs_read(
    run: NodeRun, nodes: Sequence[str], arc: str, reread_empty: bool
) -> dict[str, dict[str, list[dict]]]:
    """`read_arcs` for one read naming `nodes`, raising `_Refused` when the
    front door refuses it or a later page. A node a read naming it alone
    leaves out is an `un_node_omitted` fault."""
    body = run.post({"nodes": list(nodes), "property": arc})
    if body.get("nextToken") is not None:
        if len(nodes) == 1:
            return {nodes[0]: _arc_pages(run, nodes[0], body, arc)}
        middle = len(nodes) // 2
        return {
            **read_arcs(run, nodes[:middle], arc, reread_empty=reread_empty),
            **read_arcs(run, nodes[middle:], arc, reread_empty=reread_empty),
        }
    data = body.get("data") or {}
    if len(nodes) == 1 and nodes[0] not in data:
        raise _omitted(nodes[0], arc)
    found = {node: _answered_arcs(data.get(node)) for node in nodes}
    if len(nodes) > 1:
        for node in nodes:
            if node not in data or (reread_empty and not found[node]):
                found.update(read_arcs(run, [node], arc))
    return found


def _answered_arcs(answer: Mapping | None) -> dict[str, list[dict]]:
    """One node's `->*` answer as its targets by property."""
    return {
        property: targets.get("nodes", [])
        for property, targets in ((answer or {}).get("arcs") or {}).items()
    }


def _labels(run: NodeRun, nodes: Sequence[str]) -> dict[str, str | None]:
    """Each node's label from its `->name` arcs (`read_arcs`), as
    `dc_graph.GraphClient.node_label` picks it (`label_or_self_name`), `None`
    for a node a read naming it alone answers with none."""
    found: dict[str, str | None] = {}
    for node, arcs in read_arcs(run, nodes, "->name").items():
        names = [value for entry in arcs.get("name", ()) if (value := entry.get("value"))]
        label = label_or_self_name(names, node)
        found[node] = None if label is None else run.string(label)
    return found


# --- Our mixer ---------------------------------------------------------------


def iter_node_arcs(
    graph: NodeGraph,
    nodes: Sequence[str],
    *,
    deadline: float | None = None,
    arc: str = "->*",
    per_request: int = NODES_PER_REQUEST,
    reread_empty: bool = True,
) -> Iterator[tuple[str, dict[str, dict[str, list[dict]]]]]:
    """Each of `nodes` with its `arc` arcs, `->*` by default, each property's
    `{"nodes": [...]}` as `dc_graph.GraphClient.node_arcs_batch` returns them
    and `{}` for a node answered with no arcs, read by `read_arcs`' rules:
    `per_request` nodes a read, `READ_CONCURRENCY` reads in flight, a read
    that carries a `nextToken` or that the front door refuses split in half,
    a node a read naming several leaves out read again alone, as is one it
    answers empty unless `reread_empty` is false, and a single node's pages
    followed. Nodes come read by read, in the order given, and a read's
    answers are let go once yielded. Raises `GraphReadError` when a read
    fails, `deadline` passes, the front door refuses a node alone, a read
    naming a node alone leaves it out, or a node's pages stop advancing."""
    with ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as pool:
        run = NodeRun(graph=graph, pool=pool, deadline=deadline)
        try:
            for read in run.map(
                lambda chunk: read_arcs(run, chunk, arc, reread_empty=reread_empty),
                chunks(list(nodes), per_request),
            ):
                for node, arcs in read.items():
                    yield node, {property: {"nodes": targets} for property, targets in arcs.items()}
        except NodeReadFault as exc:
            raise GraphReadError(exc.detail) from exc


def iter_node_labels(
    graph: NodeGraph, nodes: Sequence[str], *, deadline: float | None = None
) -> Iterator[tuple[str, str | None]]:
    """Each of `nodes` with the label `dc_graph.GraphClient.node_label`
    gives it (`dc_graph.label_or_self_name` over its `->name` arcs), or
    `None` for a node with no name. Nodes are read `_LABEL_NODES_PER_REQUEST` a
    request, as `node_labels_batch` reads them, `READ_CONCURRENCY` reads in
    flight, by `read_arcs`' rules, so a node a read naming several leaves
    out or answers with no name is read again alone. Nodes come read by
    read, in the order given. Raises `GraphReadError` when a read fails,
    `deadline` passes, the front door refuses a node alone, a read naming a
    node alone leaves it out, or a node's pages stop advancing."""
    with ThreadPoolExecutor(max_workers=READ_CONCURRENCY) as pool:
        run = NodeRun(graph=graph, pool=pool, deadline=deadline)
        try:
            for found in run.map(
                lambda chunk: _labels(run, chunk), chunks(list(nodes), _LABEL_NODES_PER_REQUEST)
            ):
                yield from found.items()
        except NodeReadFault as exc:
            raise GraphReadError(exc.detail) from exc


__all__ = [
    "NODES_PER_REQUEST",
    "READ_CONCURRENCY",
    "NodeGraph",
    "NodeReadFault",
    "NodeReadFaultKind",
    "NodeRun",
    "chunks",
    "iter_node_arcs",
    "iter_node_labels",
    "read_arcs",
    "read_subjects",
]
