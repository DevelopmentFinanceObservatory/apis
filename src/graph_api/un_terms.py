"""Build the list of the Classes and Properties the UN System Data Commons declares.

`build_un_terms` reads `Class <-typeOf` and then `Property <-typeOf` from the UN deployment's
own `dc_graph.GraphClient`, page by page through its `post`, one listing after the other, so the
build has one request in flight against the UN deployment. Two failures are retried on the
page they hit, and the pages already read are kept:

- The UN front door rate-limits a busy client with HTTP 429. The page is requested again after
  each of `UN_TERMS_RATE_LIMIT_WAITS_SECONDS` (5 and 15 s), one budget shared by both listings,
  or after the response's `Retry-After` where that is longer, so a build with no `Retry-After`
  waits 20 s at most for 429s, inside its 60 s flight. A 429 past the budget, or one whose
  next wait would not leave `UN_TERMS_MIN_READ_SECONDS` before the build's deadline for the
  page's read, raises `UnTermsRateLimitedError`, which carries the response's `Retry-After`.
- The graph sometimes answers a page with a token the listing has already been given. The page
  is requested again after each of `origin_catalogue.node_reads.REPEATED_PAGE_WAITS_SECONDS`
  (1, 2 and 4 s), while the wait leaves `UN_TERMS_MIN_READ_SECONDS` before the deadline, and
  then raises `dc_graph.GraphNotAdvancingError`, as does a listing that passes `MAX_PAGES` or
  whose pages add no subject and no new provenance of a subject for `STALLED_PAGE_LIMIT` pages.

Each wait logs one line. The build's `stop` event, which the Graph lifespan sets at shutdown,
ends a wait at once and stops the build before its next request, and the build then raises
`GraphReadError`.

A listing that names no subject, or none the UN origin declares, raises
`UnTermsListingEmptyError`: the UN deployment declares 219 Classes and 215 Properties, so an
empty one is a deployment mid-reload, which answers `{"data": {"Property": {}}}`.

A term is kept when its listing entries carry a provenance outside base Data Commons, meaning
one whose origin (`graph_api.schema_document.provenance_origin`) is not `"base"`. Every
provenance the UN deployment serves outside `dc/base/` is a UN declaration, so the `"dfo"`
fallback for an unknown prefix, and a term with base and UN provenances both, count as
`un_system`. A UN term declared under `undata/p/UNData` with no `undata` prefix, such as
`unCode` or `unConcept`, is kept, and so is a base-named Class the UN deployment declares again,
such as `Dataset`. A term whose entries carry only base provenances is the base term the default
list serves, and is left out. A term whose entries carry none is left out and logged. The UN
deployment lists 2,105 Classes and 6,129 Properties, and declares 219 and 215 of them. The
mixer's `GET /api/graph/schema/terms` lists every one of the other 1,886 Classes and 5,914
Properties, so leaving them out drops no term from both routes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Collection, Iterator
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

UN_TERMS_MIN_READ_SECONDS: Final[float] = 5.0
"""The time a wait must leave before the build's deadline for the page's read that follows it.
A wait that leaves less is not taken, since the read would fail on the deadline."""

_NODE_PATH: Final[str] = "/v2/node"
_RATE_LIMITED: Final[int] = 429
_KINDS: Final[tuple[TermKind, ...]] = ("Class", "Property")
_LOGGED_DCIDS: Final[int] = 10


class UnTermsRateLimitedError(GraphReadError):
    """The UN front door still answered HTTP 429 when the build's 429 waits were spent or the
    next would not fit before its deadline. `retry_after` is the seconds the last response's
    `Retry-After` named, `None` when it named none."""

    def __init__(self, *args: object, retry_after: float | None = None) -> None:
        super().__init__(*args, upstream_status=_RATE_LIMITED)
        self.retry_after = retry_after


class UnTermsListingEmptyError(GraphReadError):
    """A `Class` or `Property` listing named no subject, or none the UN origin declares."""

    def __init__(self, kind: TermKind) -> None:
        super().__init__(f"the UN deployment's {kind} <-typeOf lists no term the UN declares")
        self.kind = kind


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
            if self._stop.is_set():
                raise GraphReadError("the UN terms build stopped for shutdown")
            try:
                return self._reader.post(_NODE_PATH, payload, deadline=self._deadline)
            except GraphReadError as exc:
                if exc.upstream_status != _RATE_LIMITED:
                    raise
                budget = next(self._rate_limit_waits, None)
                wait = max(budget, exc.retry_after or 0.0) if budget is not None else None
                if wait is None or not self._fits(wait):
                    raise UnTermsRateLimitedError(
                        f"the UN deployment answered 429 paging {kind} <-typeOf",
                        retry_after=exc.retry_after,
                    ) from exc
                logger.info("UN terms: %s listing answered 429, waiting %.0f s", kind, wait)
            self._sleep(wait)

    def _fits(self, wait: float) -> bool:
        return time.monotonic() + wait + UN_TERMS_MIN_READ_SECONDS < self._deadline

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
        stop: an event that ends any wait at once, stops the build before its next request,
            and fails the build.

    Raises:
        SchemaListingTruncatedError: a listing passed `TERM_LISTING_CAP`.
        UnTermsListingEmptyError: a listing named no term the UN declares.
        UnTermsRateLimitedError: a listing still answered 429.
        GraphReadError: a listing failed or stopped advancing.
    """
    pages = _Pages(reader, deadline=deadline, stop=stop if stop is not None else threading.Event())
    listings = [(kind, _list(pages, kind)) for kind in _KINDS]
    for kind, subjects in listings:
        if not any(_declared_by_un(subject.provenances, catalogue) for subject in subjects):
            raise UnTermsListingEmptyError(kind)
    listed = fold_listings(listings)
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
        if _declared_by_un(term.provenances, catalogue)
    ]
    return SchemaTerms(generation=None, terms=terms)


def _declared_by_un(provenances: Collection[str], catalogue: Catalogue) -> bool:
    return bool(provenances) and declared_by(provenances, catalogue=catalogue) != "base"


def _list(pages: _Pages, kind: TermKind) -> list[ArcSubject]:
    """The distinct subjects of `kind <-typeOf`, each with every name and provenance of its
    entries.

    Raises:
        SchemaListingTruncatedError: the listing passed `TERM_LISTING_CAP` subjects.
        GraphNotAdvancingError: the listing added no subject and no new provenance of a subject
            for `STALLED_PAGE_LIMIT` pages, or passed `MAX_PAGES`.
    """
    names: dict[str, set[str]] = {}
    provenances: dict[str, set[str]] = {}
    given: set[str] = set()
    token: str | None = None
    idle_pages = 0
    for _ in range(MAX_PAGES):
        body = pages.page(kind, token, given)
        advanced = False
        node = (body.get("data") or {}).get(kind) or {}
        for entry in ((node.get("arcs") or {}).get("typeOf") or {}).get("nodes", []):
            if not (dcid := entry.get("dcid")):
                continue
            if dcid not in names:
                names[dcid] = set()
                provenances[dcid] = set()
                advanced = True
            if name := entry.get("name"):
                names[dcid].add(name)
            provenance = entry.get("provenanceId")
            if provenance and provenance not in provenances[dcid]:
                provenances[dcid].add(provenance)
                advanced = True
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
        idle_pages = 0 if advanced else idle_pages + 1
        if idle_pages >= STALLED_PAGE_LIMIT:
            break
        given.add(token)
    raise GraphNotAdvancingError(
        f"the UN deployment's {kind} <-typeOf stopped advancing",
        dcid=kind,
        property="typeOf",
    )
