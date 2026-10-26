"""Search graph contracts, value types, and request-scoped facet controls.

Transport, HTTP, JSON, and ordinary timeout failures raise ``GraphInfraError``. Deadline
reads may raise ``TimeoutError``. Byte caps raise ``GraphCapExceededError``. ``node_label``
and ``node_arcs`` return ``None`` only for a confirmed absent or empty node. This module
stays outside ``search.eval``.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pydantic

from code_index.query import CodeIndex, IndexFamily
from code_index.shared import get_index, per_index
from code_index.types import StatVarRow
from dc_graph import Facet, FacetDefaultPolicy
from origin_catalogue import Catalogue
from origin_catalogue.origins import UN_ID_PREFIX
from search.scope.basis import convention_measures, names_a_basis
from statvar_members import FiveTuple

if TYPE_CHECKING:
    from graph_generation import GenerationStamp
    from origin_catalogue.graph import UnAvailabilityClient

# Let coverage probes and later observation reads share a cache key within one request.
# ContextVar keeps concurrent requests' settings isolated.
_force_full_facets: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "search_force_full_facets", default=False
)


@contextlib.contextmanager
def force_full_facets(enabled: bool):
    """Control ``needs_dates`` for facet calls in this context.

    ``True`` upgrades every call. ``False`` lets nested probes opt out of an outer upgrade.
    Restore the previous setting on exit.
    """
    token = _force_full_facets.set(enabled)
    try:
        yield
    finally:
        _force_full_facets.reset(token)


@pydantic.dataclasses.dataclass
class DetectedSV:
    """One ``/v2/resolve`` candidate, validated as a Pydantic field on ``interpret.Recall``.

    Keep every graph candidate. Later stages use ``score`` and ``type_of`` to handle weak or
    non-SV matches.
    """

    dcid: str
    score: float
    type_of: str | None  # candidate.typeOf, used to decide Topic expansion.
    sentence: str | None  # candidate.metadata.sentence, carried for diagnostics and labels.
    # A row `with_rival_families` adds for its five-tuple, which the resolve did not return.
    rival: bool = False


@dataclass(frozen=True, slots=True)
class DetectLegs:
    """The two resolves behind one recall, as each deployment answered them.

    `ours` is our mixer's resolve under ``SEARCH_RESOLVE_TARGET``. `un` is the UN System Data
    Commons' ``custom_only`` resolve: empty when the request's catalogue has the UN origin off,
    and ``None`` when the resolve failed or timed out.
    """

    ours: tuple[DetectedSV, ...]
    un: tuple[DetectedSV, ...] | None


@dataclass(frozen=True, slots=True)
class Detection:
    """Recall's candidates per resolve, each admitted, in descending score and capped at
    ``SEARCH_DETECT_K``, ours with the rival-family rows `recall_hits` adds past the cap.
    `candidates` come from our resolve. `un` settles to the UN System Data
    Commons' candidates, or to ``None`` when that resolve failed; it may still be running when
    the detection returns. A part whose scope allows that origin derives the UN shapes
    beside its own, and only a part whose own pass ends no_data or answers on a pick its
    binder doubted (`region.pipeline._fill_gap`) waits for them, at most
    ``SEARCH_UN_FIRST_PASS_WAIT_S`` from the start of derivation
    (`region.phase_intake.un_pool_shapes`)."""

    candidates: tuple[DetectedSV, ...]
    un: Future[tuple[DetectedSV, ...] | None]
    # Our resolve's hits inside the cut, under their own dcids, superseded ones included
    # (`cut_hits`), and the rival rows past it. Derivation confirms the hits inside the cut
    # and the rival rows in one arc read (`derive.derive_shapes`). `None` reads as
    # `candidates`.
    hits: tuple[DetectedSV, ...] | None = None


def stand_in(dcid: str, *, catalogue: Catalogue) -> str | None:
    """The dcid an admitted or superseded variable stands as in Search: its successor
    (`Catalogue.successor`) when it is superseded, `None` for a superseded one with no
    successor, as with the UN origin off, and the dcid itself otherwise."""
    if catalogue.is_superseded(dcid):
        return catalogue.successor(dcid)
    return dcid


def standing_dcid(dcid: str, *, catalogue: Catalogue) -> str | None:
    """The dcid a recalled or Topic-expanded variable stands as in Search (`stand_in`), or
    `None` for one `catalogue` neither admits nor supersedes."""
    if catalogue.is_superseded(dcid) or catalogue.admit_candidate(dcid) is not None:
        return stand_in(dcid, catalogue=catalogue)
    return None


def cut_hits(
    pool: Iterable[DetectedSV], *, catalogue: Catalogue, detect_k: int
) -> tuple[DetectedSV, ...]:
    """`pool`'s hits that `catalogue` admits or supersedes, in descending score and cut at
    `detect_k`, each under its own dcid.

    A rejected hit leaves its slot to the next one; a superseded hit
    (`Catalogue.is_superseded`) holds its slot whatever it stands as. So the cut is the cut
    the resolve got while the superseded provenance was admitted, less the hits refused for
    other reasons."""
    kept = [
        c
        for c in pool
        if catalogue.is_superseded(c.dcid) or catalogue.admit_candidate(c.dcid) is not None
    ]
    kept.sort(key=lambda c: c.score, reverse=True)
    return tuple(kept[:detect_k])


def _family_roles(family: IndexFamily) -> frozenset[str]:
    """The roles any row of `family` declares."""
    return frozenset(role for row in family.rows for role in row.observation_properties)


@per_index
def _rows_broadest_first(
    index: CodeIndex,
) -> dict[tuple[FiveTuple, frozenset[str]], tuple[StatVarRow, ...]]:
    """Each index family's rows, keyed by five-tuple and family properties, fewest constraint
    properties first and in index order within a count."""
    return {
        (family.five_tuple, family.properties): tuple(
            sorted(family.rows, key=lambda row: len(row.constraint_properties))
        )
        for families in index.families_by_five_tuple.values()
        for family in families
    }


def with_rival_families(
    hits: Sequence[DetectedSV], *, catalogue: Catalogue, index: CodeIndex
) -> tuple[DetectedSV, ...]:
    """`hits` and, for each five-tuple an index row among them belongs to, one row for each
    set of roles that a family publishing it declares and that falls strictly inside the
    roles of a family among the hits (`CodeIndex.families_by_five_tuple`), in descending
    score.

    A family on fewer roles publishes the totals over the axes the other breaks down: DAC1
    holds a donor's grant-equivalent ODA, the CRS the same by donor and recipient. The
    resolve ranks sentences, so the parts can fill the cut and keep the whole out: fifty CRS
    variables fill it for "official development assistance grant equivalents" and no DAC1
    variable reaches it. Which of the two answers turns on the roles the query names
    (`place_roles.without_unasked_axes`), which a sentence score cannot see. Families on the
    same roles differ in wording, which the score does rank, and a query naming the axis the
    whole lacks recalls the parts by its own words, so the cut stands for both. Among
    families on one set of roles the first in serving order stands for them.

    Search builds one shape per family it confirms a member of (`derive.derive_shapes`), and
    an index family's shape reads its slots from every row of the family, so one row stands
    for it. The row is the family's first in index order among those `catalogue` admits with
    the fewest constraint properties, as DAC1's `DAC1AidType1010` total, and takes the best
    score of the five-tuple's hits. A family with no admitted row gives way to the next on
    its roles.

    A row added here is marked `DetectedSV.rival`. It adds to what the cut yields and
    displaces none of it: derivation confirms it beside every recalled hit, and its shape
    displaces only shapes of its own five-tuple (`without_unasked_axes`). The query did not
    recall it, so it holds no claim against a table of another five-tuple.

    Each five-tuple adds at most one row per role set its families declare, less one. In
    today's index three five-tuples gain a row, each a DAC1 total by donor: grant-equivalent
    flows and FinancialObligation amounts, which the CRS publishes by donor and recipient,
    and flow amounts per GNI, which DAC2a does. The FinancialFlow amount gains none, since
    DAC1 declares `financingProvider` and `financingRecipient` there, as the CRS, DAC2a and
    DAC2b do.

    Under a source scope (`Catalogue.scope`), a five-tuple among the hits none of whose
    recalled families a provenance in the scope publishes gains a row of each of its
    families that one does, whatever its roles. A part naming DAC2a recalls the DAC2a family
    of a five-tuple whose CRS variables filled the cut. A five-tuple with a recalled family
    in scope gains none, so a part naming the DAC tables answers from the DAC2a family it
    recalled as the unscoped question does, and an exclusion that removes no recalled family
    leaves recall as it is without the exclusion.

    A family among the hits that a dataset the measure-basis conventions read publishes
    gains a row of each family of that dataset measuring a basis the conventions count it on
    (`scope.basis.convention_measures`) on the hit's statistic, qualifier and denominator,
    when no family among the hits measures that basis there. The conventions elect the basis
    from the part's shape and window, so its family must reach the menu whatever words
    recalled the other: "ODA" from the United States recalls DAC1's ODA amounts and none of
    its grant equivalents, the headline basis from 2018. A row added so is marked
    `DetectedSV.rival` and takes its hit family's best score."""
    scope = catalogue.scope
    best: dict[FiveTuple, float] = {}
    present: dict[FiveTuple, set[frozenset[str]]] = {}
    held: set[int] = set()
    recalled_in_scope: set[FiveTuple] = set()
    recalled: list[IndexFamily] = []
    for hit in hits:
        family = index.family_by_dcid.get(hit.dcid)
        if family is None:
            continue
        if id(family) not in held:
            recalled.append(family)
        held.add(id(family))
        present.setdefault(family.five_tuple, set()).add(_family_roles(family))
        best[family.five_tuple] = max(best.get(family.five_tuple, hit.score), hit.score)
        if scope is not None and not scope.isdisjoint(family.datasets):
            recalled_in_scope.add(family.five_tuple)

    rivals: list[DetectedSV] = []
    for five_tuple, score in best.items():
        covered = present[five_tuple]
        for family in index.families_by_five_tuple[five_tuple]:
            if id(family) in held:
                continue
            roles = _family_roles(family)
            rival = roles not in covered and any(roles < whole for whole in covered)
            in_scope = (
                scope is not None
                and five_tuple not in recalled_in_scope
                and not scope.isdisjoint(family.datasets)
            )
            if not (rival or in_scope):
                continue
            row = _broadest_admitted_row(family, catalogue=catalogue, index=index)
            if row is not None:
                covered.add(roles)
                held.add(id(family))
                rivals.append(
                    DetectedSV(
                        dcid=row.dcid,
                        score=score,
                        type_of="StatisticalVariable",
                        sentence=None,
                        rival=True,
                    )
                )
    for sibling, score in _basis_siblings(recalled, best, index=index):
        if id(sibling) in held:
            continue
        row = _broadest_admitted_row(sibling, catalogue=catalogue, index=index)
        if row is not None:
            held.add(id(sibling))
            rivals.append(
                DetectedSV(
                    dcid=row.dcid,
                    score=score,
                    type_of="StatisticalVariable",
                    sentence=None,
                    rival=True,
                )
            )
    if not rivals:
        return tuple(hits)
    return tuple(sorted((*hits, *rivals), key=lambda c: c.score, reverse=True))


def _broadest_admitted_row(
    family: IndexFamily, *, catalogue: Catalogue, index: CodeIndex
) -> StatVarRow | None:
    """`family`'s first row in index order among those `catalogue` admits with the fewest
    constraint properties, None when it admits none."""
    rows = _rows_broadest_first(index)[family.five_tuple, family.properties]
    return next((r for r in rows if catalogue.admit_candidate(r.dcid) is not None), None)


def _basis_siblings(
    recalled: Sequence[IndexFamily], best: Mapping[FiveTuple, float], *, index: CodeIndex
) -> list[tuple[IndexFamily, float]]:
    """The families measuring a convention basis that `recalled`, the families among the
    hits, leave out, each with the best score (`best`) of the recalled family it is a
    sibling of (`with_rival_families`), in recalled order and each once."""
    measured = {(dataset, family.five_tuple) for family in recalled for dataset in family.datasets}
    found: dict[int, tuple[IndexFamily, float]] = {}
    for family in recalled:
        five_tuple = family.five_tuple
        if not names_a_basis(five_tuple.population_type, five_tuple.measured_property):
            continue
        for dataset in family.datasets:
            for measure in convention_measures(dataset):
                sibling = dataclasses.replace(
                    five_tuple,
                    population_type=measure.population_type,
                    measured_property=measure.measured_property,
                )
                if (dataset, sibling) in measured:
                    continue
                for candidate in index.families_by_five_tuple.get(sibling, ()):
                    if dataset in candidate.datasets and id(candidate) not in found:
                        found[id(candidate)] = (candidate, best[five_tuple])
    return list(found.values())


def recall_hits(
    pool: Iterable[DetectedSV],
    *,
    catalogue: Catalogue,
    detect_k: int,
    index: CodeIndex | None = None,
) -> tuple[DetectedSV, ...]:
    """`pool`'s hits cut at `detect_k` (`cut_hits`), with a row of each index family the cut
    left out of a five-tuple it holds (`with_rival_families`)."""
    hits = cut_hits(pool, catalogue=catalogue, detect_k=detect_k)
    return with_rival_families(hits, catalogue=catalogue, index=index or get_index())


def stand_candidates(hits: Iterable[DetectedSV], *, catalogue: Catalogue) -> tuple[DetectedSV, ...]:
    """`hits`, in descending score, as they stand (`stand_in`): a superseded hit as its
    successor at the score the resolve gave it, one with no successor dropped, and several
    standing as one dcid kept once at their best score."""
    best: dict[str, DetectedSV] = {}
    for hit in sorted(hits, key=lambda c: c.score, reverse=True):
        dcid = stand_in(hit.dcid, catalogue=catalogue)
        if dcid is not None and dcid not in best:
            best[dcid] = hit if dcid == hit.dcid else dataclasses.replace(hit, dcid=dcid)
    return tuple(best.values())


def admit_candidates(
    pool: Iterable[DetectedSV], *, catalogue: Catalogue, detect_k: int
) -> tuple[DetectedSV, ...]:
    """`pool`'s candidates as they stand, cut at `detect_k` hits (`cut_hits`,
    `stand_candidates`): supersession changes and removes candidates and brings in none the
    cut would have left out."""
    hits = cut_hits(pool, catalogue=catalogue, detect_k=detect_k)
    return stand_candidates(hits, catalogue=catalogue)


def admit_un_candidates(
    un: Iterable[DetectedSV] | None, *, catalogue: Catalogue, detect_k: int
) -> tuple[DetectedSV, ...] | None:
    """The UN resolve's admitted ``undata/`` candidates, the ones the origin client reads from
    the UN deployment (`admit_candidates`), or ``None`` for a failed resolve."""
    if un is None:
        return None
    return admit_candidates(
        (c for c in un if c.dcid.startswith(UN_ID_PREFIX)), catalogue=catalogue, detect_k=detect_k
    )


def settled_un(un: tuple[DetectedSV, ...] | None) -> Future[tuple[DetectedSV, ...] | None]:
    """A `Detection.un` already holding `un`."""
    future: Future[tuple[DetectedSV, ...] | None] = Future()
    future.set_result(un)
    return future


def admit_legs(
    legs: DetectLegs, *, catalogue: Catalogue, detect_k: int, index: CodeIndex | None = None
) -> Detection:
    """Both finished resolves, each admitted through `catalogue` and cut at `detect_k`, ours
    with the rival families of its five-tuples (`recall_hits`)."""
    hits = recall_hits(legs.ours, catalogue=catalogue, detect_k=detect_k, index=index)
    return Detection(
        candidates=stand_candidates(hits, catalogue=catalogue),
        un=settled_un(admit_un_candidates(legs.un, catalogue=catalogue, detect_k=detect_k)),
        hits=hits,
    )


class SubjectsCoverage(StrEnum):
    """The three mutually-exclusive end states of one lead dcid's subject
    collection in node_subjects_many. Exactly one applies to a given
    SubjectsPage; the states cannot combine, since a lead that reached its
    cap did not run out of page budget, and one cut off by the page
    budget never reached its cap.
    """

    complete = "complete"
    """The lead's distinct subject set is fully collected: either it was
    read to exhaustion under max_results, or the batch's page budget ran
    out with this lead's own set already known-exhausted. dcids is the
    real subject set."""

    capped = "capped"
    """A (max_results + 1)th distinct subject was confirmed for this
    dcid: dcids holds only the first max_results of them, a known-partial
    prefix of a subject set confirmed to hold more than the cap. Reaching
    the cap is itself a final determination -- a page beyond it can hold
    only duplicates of subjects already collected -- so this state is
    never revised by budget exhaustion."""

    budget_exhausted = "budget_exhausted"
    """The batch's page budget ran out before this dcid either capped or
    had its subject set confirmed exhausted. dcids holds whatever was
    collected so far, a lower bound on the true set rather than the set
    itself; an empty dcids in this state proves nothing, unlike an empty
    dcids under complete."""


@dataclass(frozen=True)
class SubjectsPage:
    """Distinct incoming subjects for one lead dcid, in first-occurrence order, with coverage.

    ``capped`` gives a known prefix, ``budget_exhausted`` a lower bound, and ``complete`` the
    full subject set.
    """

    dcids: tuple[str, ...]
    coverage: SubjectsCoverage


@runtime_checkable
class EngineGraphClient(Protocol):
    """Graph methods used by Search stages.

    ``node_arcs_many``, ``node_names_many``, and ``resolve_by_arc`` also satisfy
    ``entity_resolution.GraphLookup``. ``LiveGraphClient`` adds ``exists_batch`` for
    ``search.eval``.
    """

    def node_label(self, dcid: str) -> str | None:
        """Return the fuller last name value, or the first when it is the only one.

        ``None`` means confirmed absence. Transport and HTTP failures raise ``GraphInfraError``.
        """
        ...

    def node_arcs(self, dcid: str) -> dict | None:
        """Return the full ``->*`` arcs, or ``None`` for a confirmed absence.

        Values are distinct in first-occurrence order across provenance entries. Transport
        and HTTP failures raise ``GraphInfraError``.
        """
        ...

    def node_type(self, dcid: str) -> str | None:
        """Return the first typeOf dcid for a node, or None if absent."""
        ...

    def default_facet_policy(self, *, stat_var: str, axis: str) -> FacetDefaultPolicy | None:
        """Return the default facet the graph declares for `stat_var` on `axis`, or None."""
        ...

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Fetch named out-arcs for each dcid via ``POST /v2/node``.

        Return every requested dcid. Missing nodes map to ``None``. Held nodes map each
        requested property to a tuple, with ``()`` for a missing arc. Raise
        ``GraphInfraError`` on transport or HTTP failure, or when a dcid remains omitted
        after an individual retry.
        """
        ...

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Fetch each node's ``name`` and ``alternateName`` arcs.

        Return every unique input dcid as a key. Names combine both arcs in first-occurrence
        order, with duplicates removed. Missing nodes and nameless nodes map to ``()``. Empty
        input returns ``{}`` without a request. Transport or HTTP failures, and dcids omitted
        from both a batch and individual retry, raise ``GraphInfraError``.
        """
        ...

    def node_subjects_many(
        self,
        dcids: Sequence[str],
        *,
        property: str,
        max_results: int,
        deadline: float | None = None,
    ) -> dict[str, SubjectsPage]:
        """Fetch each requested dcid's incoming ``<-{property}`` subjects.

        Return every dcid as a ``SubjectsPage``. Missing nodes and absent arcs are complete
        empty sets. Subjects are distinct and ordered by first occurrence, up to
        ``max_results``. A confirmed ``max_results + 1``th subject marks that lead as capped.

        The top-level ``nextToken`` pages the whole batch. Repeated consecutive tokens raise
        ``GraphInfraError``. Capped leads leave later page requests while the rest continue.
        If the page budget ends first, mark only unfinished leads ``budget_exhausted``. Capped
        leads retain their final ``capped`` state.

        Empty reverse-arc results are valid leaves and receive no retry. Transport and HTTP
        failures raise ``GraphInfraError``. One absolute ``deadline`` bounds every page.
        """
        ...

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Resolve names through ``POST /core/api/v2/resolve`` using a full property expression.

        Return every name as a key, with distinct dcids in mixer order. Missing names map to
        ``[]``. Empty input returns ``{}`` without a request. Transport and HTTP failures
        raise ``GraphInfraError``.
        """
        ...

    def child_places_batch(
        self, parent_dcids: list[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Fetch each parent's descendant places of ``child_type`` in one POST.

        Return every parent as a key. Missing parents map to ``[]``. Child dcids are distinct
        in first-occurrence order. Transport and HTTP failures raise ``GraphInfraError``.
        """
        ...

    def un_availability(self) -> UnAvailabilityClient | None:
        """Return the UN deployment's SDMX ``/availability`` read, which places a UN variable
        for a query that names no place, or ``None`` when this client reaches no UN deployment.
        """
        ...

    def generation_stamp(self) -> GenerationStamp:
        """Return the stamp the client's caches key on, without I/O. The default place scope's
        memo keys on it (`materialise.place_scope`)."""
        ...

    def source_key(self) -> str:
        """Return the backends the client reads, which the default place scope's memo keys on
        beside the stamp (`materialise.place_scope`)."""
        ...

    def detect_svs(self, query: str) -> Detection:
        """Return each resolve's admitted candidates in descending score, capped at
        ``SEARCH_DETECT_K`` (`admit_candidates`).

        The graph applies no relevance floor. Later stages handle weak or non-SV candidates,
        and the engine confirms each StatVar with ``node_arcs`` before using it. The UN
        resolve may still be running on return; a failed one settles `Detection.un` to
        ``None``. A failed resolve on our mixer raises.
        """
        ...

    def observation_facets(
        self, *, stat_var: str, entity: str, needs_dates: bool = False
    ) -> list[Facet]:
        """Return ordered facets for a StatVar/entity pair, or ``[]`` when none exist.

        ``needs_dates=False`` omits per-observation dates and values while retaining summary
        dates and counts. Transport failures raise ``GraphInfraError``.
        """
        ...

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        """Fetch confirmed display labels in one ``POST /v2/node``.

        Omit missing nodes. Use the release's code index label when available, which strips the
        graph name's parenthetical code-system suffix. Otherwise return the graph name.
        Transport and HTTP failures raise ``GraphInfraError``.
        """
        ...

    def node_types_batch(self, dcids: list[str]) -> dict[str, str]:
        """Fetch the first ``->typeOf`` dcid for each node in one ``POST /v2/node``.

        Omit missing and type-less nodes. Reads share a cache with ``node_type``.
        Transport and HTTP failures raise ``GraphInfraError``.
        """
        ...

    def node_observation_properties_batch(self, dcids: list[str]) -> dict[str, tuple[str, ...]]:
        """Fetch ``->observationProperties`` for each dcid.

        Return every unique input dcid with distinct targets in first-occurrence order.
        Missing nodes or arcs map to ``()``. A present empty arc is trusted. Retry omitted
        response keys once, then raise ``GraphInfraError`` if any remain absent.

        A ``declared_role_refresh_deadline`` scope stops the live client's further chunk
        requests at its deadline and raises ``TimeoutError``. Other implementations may
        ignore that scheduling scope. Transport, HTTP, and malformed-body failures raise
        ``GraphInfraError``.
        """
        ...

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        """Fetch full ``->*`` arcs for each dcid, returning every requested key.

        Retry absent or empty batch entries once, except a present-empty result for a
        single-dcid batch. An originally absent single-dcid entry still retries. A key
        omitted from the retry raises ``GraphInfraError``. A returned empty node maps to
        ``None``. Values are distinct per property in first-occurrence order across
        provenance entries. Transport and HTTP failures raise ``GraphInfraError``. One
        absolute ``deadline`` bounds every request; a read past it raises
        ``GraphInfraError``.
        """
        ...

    def observation_facets_batch(
        self,
        stat_vars: list[str],
        entities: list[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        """Fetch ordered facets for requested StatVar/entity pairs.

        Missing pairs map to ``[]``. ``max_bytes`` caps the miss-subset response while it is
        read, raising ``GraphCapExceededError`` before the remaining body is consumed.
        ``None`` leaves the read unbounded. ``deadline`` is an absolute monotonic bound on
        the miss-subset read; a read past it raises ``GraphInfraError``.

        The process-wide pair cache can serve results from an earlier uncapped read without
        applying this cap. Callers needing a hard receive limit must check the returned
        facets themselves. Transport failures raise ``GraphInfraError``.
        """
        ...

    def observation_read_groups(
        self, stat_vars: list[str], entities: list[str], needs_dates: bool = False
    ) -> list[list[str]]:
        """Split ``stat_vars``, in order, into the groups an ``observation_facets_batch``
        read of them at ``entities`` sends apart: one group where a single request names
        them all, and one per request where the deployment serving them reads by GET under
        a URL budget. A caller that may stop reading early reads a group at a time. Makes
        no request.
        """
        ...

    def child_dcids(self, parent_dcid: str) -> list[str]:
        """Return distinct dcids pointing to the parent via ``<-isPartOf``.

        An absent parent returns ``[]``. Results follow first-occurrence order. Transport
        failures raise ``GraphInfraError``.
        """
        ...

    def enum_members(self, property_dcid: str) -> list[str]:
        """Return distinct members of the property's declared enum ranges in API order.

        Follow ``->rangeIncludes`` to enum classes, then ``<-typeOf`` to members. Return
        ``[]`` when no classes or members exist. Caller seeds can then drift from the live
        enum. Transport and HTTP failures raise ``GraphInfraError``.
        """
        ...

    def currency_units(self, *, deadline: float | None = None) -> dict[str, str]:
        """Map each CurrencyUnit dcid to its ISO 4217 alpha-3 code.

        Follow ``<-typeOf`` to members, then read ``->iso4217AlphaCode``. Log and omit
        members with zero or multiple codes. Unit references reject them. No members is an
        error. Cache the first successful result for the process lifetime. Failed reads leave
        the cache untouched for retries. Transport and HTTP failures raise ``GraphInfraError``.
        One absolute ``deadline`` bounds every page; a read past it raises
        ``GraphDeadlineFailure``.
        """
        ...

    def entities_with_data(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        max_bytes: int | None = None,
    ) -> list[str]:
        """Return entities with data for any requested variable, as a union and presence check.

        This backs the options door without fetching observation series. In this deployment,
        ``entity.expression`` with an entity/variable-only select returns no per-entity rows.
        adding facets collapses results to one aggregate key, while date/value selects pull
        full series. Explicit ``entity.dcids`` with entity/variable fields gives one result per
        entity that has data.

        Read variables in fixed-size chunks using ``dc_graph.GraphClient``'s
        ``entities_with_data_variable_chunk_size``.
        Derived populations can contain thousands of StatVars, and ``/v2/observation`` has no
        row cap. An unchunked read can exceed the 32 MiB response cap. Stop when every entity
        has data for some variable, since the union cannot grow further.

        ``max_bytes`` applies independently to each chunk and raises ``GraphCapExceededError``
        while reading, before parsing. ``None`` leaves each chunk unbounded. Return entities
        in input order. Empty inputs return ``[]`` without a request. Transport and HTTP
        failures raise ``GraphInfraError``.
        """
        ...

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Return entities with data for each variable, without unioning variables.

        Use explicit ``entity.dcids`` and one ``entity``/``variable`` presence request. Caller
        variable lists are bounded, so this sends one POST for the full list. Missing response
        variables map to ``[]``. Results preserve entity input order. Empty inputs return
        ``{}`` without a request. Transport and HTTP failures raise ``GraphInfraError``.
        """
        ...
