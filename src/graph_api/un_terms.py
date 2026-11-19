"""Build the list of the Classes and Properties the UN System Data Commons declares.

`build_un_terms` reads `Class <-typeOf` and then `Property <-typeOf` from the UN deployment's
own `dc_graph.GraphClient`, page by page through its `post`, one listing after the other, so the
build has one request in flight against the UN deployment. Two failures are retried on the
page they hit, and the pages already read are kept:

- The UN front door rate-limits a busy client with HTTP 429. The page is requested again after
  each of `UN_TERMS_RATE_LIMIT_WAITS_SECONDS` (5 and 15 s), one budget shared by both listings,
  so a build waits 20 s at most for 429s, inside its 60 s flight. A 429 past the budget, or one
  whose next wait would end past the build's deadline, raises its `GraphReadError`.
- The graph sometimes answers a page with a token the listing has already been given. The page
  is requested again after each of `origin_catalogue.node_reads.REPEATED_PAGE_WAITS_SECONDS`
  (1, 2 and 4 s), while the wait ends before the deadline, and then raises
  `dc_graph.GraphNotAdvancingError`, as does a listing that adds no subject for
  `STALLED_PAGE_LIMIT` pages or passes `MAX_PAGES`.

Each wait logs one line. A wait ends early when the build's `stop` event is set, as the Graph
lifespan sets it at shutdown, and the build then raises `GraphReadError`.

A term is kept when its listing entries carry a provenance outside base Data Commons, meaning
one whose origin (`graph_api.schema_document.provenance_origin`) is not `"base"`. Every
provenance the UN deployment serves outside `dc/base/` is a UN declaration, so the `"dfo"`
fallback for an unknown prefix, and a term with base and UN provenances both, count as
`un_system`. A UN term declared under `undata/p/UNData` with no `undata` prefix, such as
`unCode` or `unConcept`, is kept, and so is a base-named Class the UN deployment declares again,
such as `Dataset`. A term whose entries carry only base provenances is the base term the default
list serves, and is left out. A term whose entries carry none is left out and logged. The UN
deployment lists 2,105 Classes and 6,144 Properties, and declares 219 and 215 of them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Final, Protocol

from dc_graph import ArcSubject, GraphNotAdvancingError, GraphReadError
from graph_api.schema_document import (
    TERM_LISTING_CAP,
    SchemaListingTruncatedError,
    TermKind,
    declared_by,
    fold_listings,
    schema_term,
)
from graph_api.schema_models import SchemaTerms
from origin_catalogue import Catalogue
from origin_catalogue.node_reads import (
    MAX_PAGES,
    REPEATED_PAGE_WAITS_SECONDS,
    STALLED_PAGE_LIMIT,
)

logger = logging.getLogger(__name__)

UN_TERMS_RATE_LIMIT_WAITS_SECONDS: Final[tuple[float, ...]] = (5.0, 15.0)
"""The waits before each further request for a page the UN front door answered with HTTP 429,
one budget for the whole build. Both end inside the 60 s flight with time left to read."""

_NODE_PATH: Final[str] = "/v2/node"
_RATE_LIMITED: Final[int] = 429
_KINDS: Final[tuple[TermKind, ...]] = ("Class", "Property")
_LOGGED_DCIDS: Final[int] = 10


class UnTermsReader(Protocol):
    """The `/v2/node` POST of the UN deployment's `dc_graph.GraphClient`."""

    def post(self, path: str, payload: dict, *, deadline: float | None = None) -> dict: ...


class _Pages:
    """The page requests of one build, with its shared 429 budget, deadline and stop event."""

    def __init__(self, reader: UnTermsReader, *, deadline: float, stop: threading.Event) -> None:
        self._reader = reader
        self._deadline = deadline
        self._stop = stop
        self._rate_limit_waits: Iterator[float] = iter(UN_TERMS_RATE_LIMIT_WAITS_SECONDS)

    def page(self, kind: str, token: str | None, given: set[str]) -> dict:
        """One page of `kind <-typeOf`, whose answer carries no token or one outside `given`."""
        payload: dict = {"nodes": [kind], "property": "<-typeOf"}
        if token is not None:
            payload["nextToken"] = token
        repeat_waits = iter(REPEATED_PAGE_WAITS_SECONDS)
        while True:
            body = self._post(kind, payload)
            if body.get("nextToken") not in given:
                return body
            wait = next(repeat_waits, None)
            if wait is None or not self._fits(wait):
                raise GraphNotAdvancingError(
                    f"the UN deployment repeated a nextToken paging {kind} <-typeOf",
                    dcid=kind,
                    property="typeOf",
                )
            logger.info("UN terms: %s listing repeated a page token, waiting %.0f s", kind, wait)
            self._sleep(wait)

    def _post(self, kind: str, payload: dict) -> dict:
        while True:
            try:
                return self._reader.post(_NODE_PATH, payload, deadline=self._deadline)
            except GraphReadError as exc:
                if exc.upstream_status != _RATE_LIMITED:
                    raise
                wait = next(self._rate_limit_waits, None)
                if wait is None or not self._fits(wait):
                    raise
                logger.info("UN terms: %s listing answered 429, waiting %.0f s", kind, wait)
            self._sleep(wait)

    def _fits(self, wait: float) -> bool:
        return time.monotonic() + wait < self._deadline

    def _sleep(self, wait: float) -> None:
        if self._stop.wait(wait):
            raise GraphReadError("the UN terms build stopped for shutdown")


def build_un_terms(
    *,
    reader: UnTermsReader,
    catalogue: Catalogue,
    deadline: float,
    stop: threading.Event | None = None,
) -> SchemaTerms:
    """Every Class and Property the UN deployment declares, sorted by `ref`, with a `null`
    generation, since the UN deployment serves no generation token.

    Args:
        reader: the UN deployment's own client, read past any origin routing.
        catalogue: the origin catalogue whose provenance table names the origin of the
            provenances it lists.
        deadline: the absolute monotonic deadline passed to every read.
        stop: an event that ends any wait at once and fails the build.

    Raises:
        SchemaListingTruncatedError: a listing passed `TERM_LISTING_CAP`.
        GraphReadError: a listing failed, stopped advancing, or still answered 429.
    """
    pages = _Pages(reader, deadline=deadline, stop=stop if stop is not None else threading.Event())
    listed = fold_listings((kind, _list(pages, kind)) for kind in _KINDS)
    unsourced = sorted(dcid for dcid, term in listed.items() if not term.provenances)
    if unsourced:
        logger.warning(
            "UN terms left out %d terms with no listed provenance, among them %s",
            len(unsourced),
            unsourced[:_LOGGED_DCIDS],
        )
    terms = [
        schema_term(dcid, term, declared_by="un_system")
        for dcid, term in sorted(listed.items())
        if term.provenances and declared_by(term.provenances, catalogue=catalogue) != "base"
    ]
    return SchemaTerms(generation=None, terms=terms)


def _list(pages: _Pages, kind: TermKind) -> list[ArcSubject]:
    """The distinct subjects of `kind <-typeOf`, each with every name and provenance of its
    entries.

    Raises:
        SchemaListingTruncatedError: the listing passed `TERM_LISTING_CAP` subjects.
        GraphNotAdvancingError: the listing added no subject for `STALLED_PAGE_LIMIT` pages or
            passed `MAX_PAGES`.
    """
    names: dict[str, set[str]] = {}
    provenances: dict[str, set[str]] = {}
    given: set[str] = set()
    token: str | None = None
    idle_pages = 0
    for _ in range(MAX_PAGES):
        body = pages.page(kind, token, given)
        before = len(names)
        node = (body.get("data") or {}).get(kind) or {}
        for entry in ((node.get("arcs") or {}).get("typeOf") or {}).get("nodes", []):
            if not (dcid := entry.get("dcid")):
                continue
            names.setdefault(dcid, set())
            provenances.setdefault(dcid, set())
            if name := entry.get("name"):
                names[dcid].add(name)
            if provenance := entry.get("provenanceId"):
                provenances[dcid].add(provenance)
        if len(names) > TERM_LISTING_CAP:
            raise SchemaListingTruncatedError(
                f"the UN deployment's {kind} <-typeOf lists more than {TERM_LISTING_CAP} subjects"
            )
        token = body.get("nextToken")
        if token is None:
            return [
                ArcSubject(
                    dcid=dcid,
                    names=tuple(sorted(names[dcid])),
                    provenances=tuple(sorted(provenances[dcid])),
                )
                for dcid in names
            ]
        idle_pages = idle_pages + 1 if len(names) == before else 0
        if idle_pages >= STALLED_PAGE_LIMIT:
            break
        given.add(token)
    raise GraphNotAdvancingError(
        f"the UN deployment's {kind} <-typeOf stopped advancing",
        dcid=kind,
        property="typeOf",
    )
