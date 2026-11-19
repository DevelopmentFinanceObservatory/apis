"""Build the list of the Classes and Properties the UN System Data Commons declares.

`build_un_terms` reads `Class <-typeOf` and then `Property <-typeOf` from the UN deployment's
own `dc_graph.GraphClient`, one listing after the other, so the build has one request in
flight against the UN deployment. The UN front door rate-limits a busy client with HTTP 429, so
a listing it answers that way is read again after each of `UN_RATE_LIMIT_WAITS_SECONDS`, the
waits `origin_catalogue.node_reads` uses, while the wait ends before the build's deadline. The
last 429 raises its `GraphReadError`.

A term is kept when the provenances of its listing entries name the UN origin by
`graph_api.schema_document.declared_by`, the rule of the default terms list. A UN term
declared under `undata/p/UNData` with no `undata` prefix, such as `unCode` or `unConcept`, is
kept, and so is a base-named Class the UN deployment declares again, such as `Dataset`. A term
whose entries carry only base Data Commons provenances is the base term the default list
serves, and is left out, as is one whose entries carry none. The UN deployment lists 2,105
Classes and 6,144 Properties, and declares 219 and 215 of them.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Final, Literal, Protocol

from dc_graph import ArcSubjectPage, GraphReadError, display_name
from graph_api.schema_document import (
    TERM_LISTING_CAP,
    SchemaListingTruncatedError,
    declared_by,
)
from graph_api.schema_models import SchemaTerm, SchemaTerms
from origin_catalogue import Catalogue

UN_RATE_LIMIT_WAITS_SECONDS: Final[tuple[float, ...]] = (10.0, 30.0, 60.0)
"""The waits before each further attempt at a listing the UN front door answered with HTTP
429."""

_RATE_LIMITED: Final[int] = 429
type _TermKind = Literal["Class", "Property"]

_KINDS: Final[tuple[_TermKind, ...]] = ("Class", "Property")


class UnTermsReader(Protocol):
    """The reverse-arc listing of the UN deployment's `dc_graph.GraphClient`."""

    def node_arc_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> ArcSubjectPage: ...


def build_un_terms(
    *,
    reader: UnTermsReader,
    catalogue: Catalogue,
    deadline: float,
) -> SchemaTerms:
    """Every Class and Property the UN deployment declares, sorted by `ref`, with a `null`
    generation, since the UN deployment serves no generation token.

    Args:
        reader: the UN deployment's own client, read past any origin routing.
        catalogue: the origin catalogue whose provenance table names the origin of the
            provenances it lists.
        deadline: the absolute monotonic deadline passed to every read.

    Raises:
        SchemaListingTruncatedError: a listing reached `TERM_LISTING_CAP`.
        GraphReadError: a listing failed, or the UN front door still answered 429.
    """
    names: dict[str, set[str]] = {}
    kinds: dict[str, set[_TermKind]] = {}
    provenances: dict[str, set[str]] = {}
    for kind in _KINDS:
        page = _list(reader, kind, deadline=deadline)
        if page.truncated:
            raise SchemaListingTruncatedError(
                f"the UN deployment's {kind} <-typeOf lists more than {TERM_LISTING_CAP} subjects"
            )
        for subject in page.subjects:
            names.setdefault(subject.dcid, set()).update(subject.names)
            kinds.setdefault(subject.dcid, set()).add(kind)
            provenances.setdefault(subject.dcid, set()).update(subject.provenances)
    terms = [
        SchemaTerm(
            ref=dcid,
            label=display_name(names[dcid], dcid),
            types=sorted(kinds[dcid]),
            provenances=sorted(provenances[dcid]),
            declared_by="un_system",
        )
        for dcid in sorted(provenances)
        if provenances[dcid] and declared_by(provenances[dcid], catalogue=catalogue) == "un_system"
    ]
    return SchemaTerms(generation=None, terms=terms)


def _list(reader: UnTermsReader, kind: str, *, deadline: float) -> ArcSubjectPage:
    """`kind <-typeOf`, read again after each 429 while the next wait ends before
    `deadline`."""
    waits = iter(UN_RATE_LIMIT_WAITS_SECONDS)
    while True:
        try:
            return reader.node_arc_subjects(
                kind, property="typeOf", max_results=TERM_LISTING_CAP, deadline=deadline
            )
        except GraphReadError as exc:
            wait = next(waits, None)
            if (
                exc.upstream_status != _RATE_LIMITED
                or wait is None
                or time.monotonic() + wait >= deadline
            ):
                raise
        time.sleep(wait)
