"""Probes and selects materialisation facets.

Subject-only StatVars use one graph batch across bound entities. A subject-only request that
binds no entity is probed at its default place scope, every place where it holds data or its
countries with data (`materialise.place_scope`), and the probe returns those places for the
fetch to name. Members probed together are each probed and fetched at their own places, so a
member's scope does not depend on its companions. Role-bearing StatVars, one-role ones
included, use a caller-supplied `CoverageIndex` because `/v2/observation` filters subject
entities, and an unbound role covers every entity in the index. Uncovered StatVars return no
candidates. Each axis resolves from the shared candidate set. Bound refs and graph-declared
defaults must match an available candidate. Deployment fallback ranks a whole candidate after
narrowing by bound axes, currency-native preference and the requested year range. Cross-axis
compatibility is checked after per-axis selection. Axis failures are raised separately and
aggregated by the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Literal, Protocol

from dc_graph import CURRENCY_UNIT_OF_MEASURE_ISO_CODES
from dc_graph import earliest_date as _earliest
from dc_graph import latest_date as _latest
from dc_sdmx import ContainedIn, EntityFilter
from dfo_platform.problem import Alternative
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DatasetDisagreementError,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationUsageError,
)
from materialise.graph import (
    Facet,
    FacetDefaultPolicy,
    GraphReader,
    GraphReadError,
    ProvenanceRefresh,
)
from materialise.place_scope import (
    AvailabilityReader,
    DefaultPlaceScope,
    read_default_presence,
    scope_of_places,
)
from materialise.recipes import NonCurrencyUnitError
from materialise.revision import RevisionReader, RevisionSnapshot
from materialise.window import RequestedWindow
from origin_catalogue import (
    UnUnserved,
    current_catalogue,
    record_un_read_failure,
    recording_un_blocked_places,
)
from origin_catalogue.graph import is_un_dcid
from origin_catalogue.origins import BASE, DFO
from series_coverage.coverage_types import BoundRoles, Role
from series_coverage.facets import FacetCandidate as _CoverageFacetCandidate
from series_coverage.facets import facet_candidates as _coverage_facet_candidates
from series_coverage.facets import rank_candidates as _coverage_rank_candidates

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest

    # Keep `CoverageIndex` annotation-only. Importing its module adds NumPy to the pure install.
    from series_coverage.index import CoverageIndex

# Strip `dcid:` here so coverage keys use bare StatVar ids.
_STAT_VAR_REF_PREFIX = "dcid:"

# Default for a StatVar with no `observationProperties` arc. Other roles use coverage probes.
_SUBJECT_ROLE = "observationAbout"
_SUBJECT_ROLE_ONLY: tuple[str, ...] = (_SUBJECT_ROLE,)

FACET_AXES: tuple[str, ...] = ("provenance", "measurementMethod", "unit")

# `probe_collection_facets`' own batched `node_arcs_many` read on every
# distinct provenance id its facet probe returned (`_probe_provenance_refresh`).
_PROVENANCE_REFRESH_PROPERTIES: tuple[str, ...] = ("lastDataRefreshDate", "nextDataRefreshDate")

# Fallback when the graph declares no axis default. Its id and version travel with each selection.
DEPLOYMENT_FACET_POLICY = "dfo/facet-policy/most-observations"
DEPLOYMENT_FACET_POLICY_VERSION = "1"

# Maps each axis to the `Facet` field that carries its value.
_AXIS_FIELDS: dict[str, str] = {
    "provenance": "provenance_id",
    "measurementMethod": "measurement_method",
    "unit": "unit",
}

# Shared with `group_facets._as_issue`, which imports this name and applies
# the same cap again as a backstop after aggregation. Defined here rather
# than in group_facets.py because group_facets.py already imports from this
# module, and the reverse import would cycle.
_MAX_ALTERNATIVES_PER_ISSUE = 10

# Caps summed distinct StatVar/entity pairs across subject-only scope groups. A supported
# shared scope is 100 members x 1,024 entities = 102,400 pairs, below the 200,000 cap. A member
# that binds no place counts its own places, at most `place_scope.MAX_EVERY_PLACE` (5,000), so
# 40 members at a full scope fit and a group past the cap refuses.
_MAX_COLLECTION_PROBE_PAIRS = 200_000

# `CurrencyUnit` refs use `Currency<ISO>`. A unit request resolves to such a ref or to a
# `CURRENCY_UNIT_OF_MEASURE_ISO_CODES` key, else raises `NonCurrencyUnitError`.
_CURRENCY_REF_PREFIX = "Currency"


def native_currency_iso(ref: str | None) -> str | None:
    """Return the ISO 4217 code a native facet unit is denominated in, or None.

    `Currency<ISO>` refs carry the code in the dcid. The base Data Commons currencies typed
    `CurrencyUnitOfMeasure` (`USDollar` and three others) carry it in
    `CURRENCY_UNIT_OF_MEASURE_ISO_CODES`. Any other ref is not a currency.
    """
    if ref is None:
        return None
    if ref in CURRENCY_UNIT_OF_MEASURE_ISO_CODES:
        return CURRENCY_UNIT_OF_MEASURE_ISO_CODES[ref]
    if ref.startswith(_CURRENCY_REF_PREFIX):
        return ref.removeprefix(_CURRENCY_REF_PREFIX)
    return None


def _currency_native_matches(candidates: Sequence[Facet], axis: str, currency: str) -> list[Facet]:
    """Candidates whose unit is denominated in `currency`: the `Currency<T>` ones when any
    exist, else those in a `CURRENCY_UNIT_OF_MEASURE_ISO_CODES` unit for `currency`."""
    exact_ref = f"{_CURRENCY_REF_PREFIX}{currency}"
    exact = [c for c in candidates if facet_axis_value(c, axis) == exact_ref]
    if exact:
        return exact
    return [c for c in candidates if native_currency_iso(facet_axis_value(c, axis)) == currency]


@dataclass(frozen=True)
class UnitRequest:
    """Unit conversion request. `currency` is an ISO 4217 code or spec role key. `prices`
    is `current` or `constant`, with `base_year` for constant prices. Mirrors the identity
    request without importing `search` or `dfo_frame_grammar`. The identity core validates
    the field pairing."""

    currency: str
    prices: Literal["current", "constant"]
    base_year: int | None = None


@dataclass(frozen=True)
class FacetBinding:
    """Binding for one `scope.facets` axis. `bound` carries a string `ref`, which callers
    must validate, or `None` for the facets that declare no value on the axis. Selection
    trusts it and matches candidates by `facet_axis_value` equality. A `request` binding
    carries `UnitRequest` and leaves `ref` unset. It applies only to `unit`.
    `__post_init__` enforces consistent fields."""

    state: Literal["bound", "unbound", "request"]
    ref: str | None = None
    request: UnitRequest | None = None

    def __post_init__(self) -> None:
        if self.state == "request":
            if self.ref is not None:
                raise MaterialisationUsageError("a request-state facet binding cannot carry a ref")
            if self.request is None:
                raise MaterialisationUsageError(
                    "a request-state facet binding must carry a request"
                )
        elif self.request is not None:
            raise MaterialisationUsageError(
                f"a {self.state!r}-state facet binding cannot carry a request; "
                "only state='request' does"
            )


@dataclass(frozen=True)
class ElectedDataset:
    """The dataset elected for an index-resolved family, as one member carries it.

    `provenance` is the elected dataset's, shared by every member of the family.
    `member_provenances` holds every dataset the code index records the member under."""

    provenance: str
    member_provenances: tuple[str, ...]

    def publishes(self, provenance_id: str | None) -> bool:
        """Whether a facet of `provenance_id` is the elected dataset's. The mint
        (`dataset_candidates`) and a Search read plan
        (`search.engine.retrieval.read_plan.in_elected_datasets`) both keep a member's
        candidates by this test before any ranking, so they elect within one dataset."""
        return provenance_id == self.provenance

    def published[C: DatasetCandidate](self, stat_var: str, candidates: Iterable[C]) -> list[C]:
        """The `candidates` of member `stat_var` this dataset publishes, in order: every one
        the member holds at the bound entities.

        A member the code index also records under another dataset can hold observations
        under that one alone, and is then unpopulated. Raises `DatasetDisagreementError`
        when no dispatchable candidate is this dataset's and one carries a provenance the
        code index does not record for the member. The mint (`dataset_candidates`) keeps
        candidates through this method, and Search runs it over the members a mint of its
        answer serves (`search.engine.retrieval.elected_dataset.check_elected_datasets`), so
        both refuse the same members."""
        candidates = list(candidates)
        kept = [c for c in candidates if self.publishes(c.provenance_id)]
        if not any(c.facet_id is not None for c in kept):
            unrecorded = sorted(
                {
                    c.provenance_id
                    for c in candidates
                    if c.facet_id is not None
                    and c.provenance_id is not None
                    and c.provenance_id not in self.member_provenances
                }
            )
            if unrecorded:
                raise DatasetDisagreementError(
                    f"{stat_var!r} is a member of dataset {self.provenance!r}, and the "
                    f"code index records it under {list(self.member_provenances)!r}; its "
                    f"observations at the bound entities are published by {unrecorded!r} alone",
                    dataset=self.provenance,
                    stat_var=stat_var,
                )
        return kept


class DatasetCandidate(Protocol):
    """A facet candidate as `ElectedDataset.published` reads it."""

    @property
    def facet_id(self) -> str | None: ...

    @property
    def provenance_id(self) -> str | None: ...


def dataset_candidates(request: MaterialisationRequest, candidates: Sequence[Facet]) -> list[Facet]:
    """Returns the candidates the request's elected dataset publishes
    (`ElectedDataset.published`), or every candidate for a request with no elected dataset.

    Selection reads only these and takes the provenance it selects from one of them, and
    rows are filtered to the selection, so a member of an index-resolved family is served
    from the elected dataset alone and a member with no candidate under it is unpopulated.

    Raises `DatasetDisagreementError` when no dispatchable candidate is the elected
    dataset's and one carries a provenance the code index does not record for the member."""
    dataset = request.dataset
    if dataset is None:
        return list(candidates)
    return dataset.published(request.stat_var, candidates)


@dataclass(frozen=True)
class FacetSelection:
    axis: str
    # `None` means the selected facet lacks this axis value. Dispatch filters for null values
    # to keep rows aligned with the selection.
    ref: str | None
    # Distinguishes this deployment's fallback policy from a graph-declared default.
    basis: Literal["bound", "declared_default", "deployment_fallback"]
    default_policy: str | None
    default_policy_version: str | None
    selection_reason: str | None
    dispatch_facet_id: str
    # Unit conversion request, or `None` for non-request and non-unit selections.
    request: UnitRequest | None = None

    def as_grammar_facet(self) -> dict[str, object]:
        """Projects to `content_version.facet_preimage`'s `{axis, ref,
        basis}` shape. The caller must skip an entry whose `ref is None`.
        The grammar rejects null refs and maps `deployment_fallback` to `declared_default`.
        Content-version identity uses one default basis for graph and deployment policies.

        A unit request uses its requested currency, prices, and base year as `ref`, with
        `basis="bound"`. The trace keeps the resolved native unit in `ref`. The preimage
        tracks the requested conversion.
        """
        if self.request is not None:
            return {"axis": self.axis, "ref": _unit_request_ref(self.request), "basis": "bound"}
        grammar_basis = "declared_default" if self.basis == "deployment_fallback" else self.basis
        return {"axis": self.axis, "ref": self.ref, "basis": grammar_basis}

    def as_trace_entry(self) -> dict[str, object]:
        """The full 7-key wire shape."""
        return {
            "axis": self.axis,
            "ref": self.ref,
            "basis": self.basis,
            "default_policy": self.default_policy,
            "default_policy_version": self.default_policy_version,
            "selection_reason": self.selection_reason,
            "dispatch_facet_id": self.dispatch_facet_id,
        }


def _bound_roles_for_coverage(
    role_bindings: Mapping[str, EntityFilter], *, coverage: CoverageIndex, stat_var: str
) -> BoundRoles | None:
    """Maps declared graph-role bindings to coverage's `donor`/`recipient` labels.
    Returns `None` when `stat_var` has no `RoleNames` entry. `ContainedIn` is unbound because
    its expansion requires a graph read. This copy keeps `materialise` independent of
    `datasets_api`."""
    names = coverage.role_names.get(stat_var)
    if names is None:
        return None
    bound: dict[Role, frozenset[str]] = {}
    for role, declared_role in (("donor", names.donor), ("recipient", names.recipient)):
        refs = role_bindings.get(declared_role)
        if refs is None or isinstance(refs, ContainedIn):
            continue
        bound[role] = frozenset(refs)
    return bound


def _facet_from_coverage_candidate(candidate: _CoverageFacetCandidate) -> Facet:
    """Maps a coverage candidate's `n_obs`, `first_year` and `last_year` to facet count and
    year-grain dates, and its years with data (from the row year masks) to `dates`, so
    `overlaps_window` tests the points a window holds. Coverage has no cadence, so
    `observation_period` is `None`. Four-digit years sort lexically through 9999. This
    artifact's horizon starts in 1960."""
    return Facet(
        dates=tuple(str(year) for year in candidate.years),
        facet_id=candidate.facet_id,
        provenance_id=candidate.provenance_id,
        measurement_method=candidate.measurement_method,
        unit=candidate.unit,
        obs_count=candidate.n_obs,
        earliest_date=str(candidate.first_year),
        latest_date=str(candidate.last_year),
        scaling_factor=candidate.scaling_factor,
        observation_period=None,
    )


def role_bearing_facets(
    *,
    stat_var: str,
    role_bindings: Mapping[str, EntityFilter],
    coverage: CoverageIndex,
    first_year: int | None = None,
    last_year: int | None = None,
) -> list[Facet] | None:
    """A role-bearing StatVar's facet candidates: coverage rows filtered by the bound roles
    and the `[first_year, last_year]` window, joined to facet metadata and ranked. A role
    `role_bindings` omits stays open. `obs_count` counts in-window observations, and a facet
    with none in the window is absent. Returns `None` when `stat_var` lacks `RoleNames`, and
    `[]` when it is covered but no rows match. Materialisation probes with no window, and
    search's facet options with the query's window, so both read one candidate set."""
    bound = _bound_roles_for_coverage(role_bindings, coverage=coverage, stat_var=stat_var)
    if bound is None:
        return None
    return _coverage_facets(
        stat_var=stat_var,
        bound=bound,
        coverage=coverage,
        first_year=first_year,
        last_year=last_year,
    )


def _coverage_facets(
    *,
    stat_var: str,
    bound: BoundRoles,
    coverage: CoverageIndex,
    first_year: int | None = None,
    last_year: int | None = None,
) -> list[Facet]:
    """`stat_var`'s coverage rows under `bound` and the window, as ranked facets."""
    variable_coverage = coverage.filter_candidates(
        variables=[stat_var], bound=bound, first_year=first_year, last_year=last_year
    )
    facet_coverage = variable_coverage[0].facets if variable_coverage else ()
    candidates = _coverage_facet_candidates(facet_coverage, metadata=coverage.facet_metadata)
    ranked = _coverage_rank_candidates(candidates)
    return [_facet_from_coverage_candidate(c) for c in ranked]


def coverage_probe(*, stat_var: str, bound: BoundRoles, coverage: CoverageIndex) -> MemberProbe:
    """A role-bearing member's probe from coverage under `bound`, in coverage's role labels:
    its candidates across every bound role tuple and every year, ranked
    (`_coverage_facets`), and each facet's series count for a window's years
    (`CoverageIndex.series_counts`), one series per donor and recipient pair holding data
    there. A mint probes here (`probe_facets`), and a Search read plan elects each member's
    group from the same probe, so both rank the same candidates by the same counts."""

    def breadth(window: RequestedWindow | None) -> Mapping[str, int]:
        first, last = (
            (window.start_year, window.end_year)
            if window is not None and window.kind == "range"
            else (None, None)
        )
        return coverage.series_counts(
            variable=stat_var, bound=bound, first_year=first, last_year=last
        )

    return MemberProbe(
        candidates=tuple(_coverage_facets(stat_var=stat_var, bound=bound, coverage=coverage)),
        admission=CoverageAdmission(),
        breadth=breadth,
    )


def _role_bearing_probe(
    *, stat_var: str, role_bindings: Mapping[str, EntityFilter], coverage: CoverageIndex
) -> MemberProbe:
    """`coverage_probe` for a request's role bindings. A StatVar lacking `RoleNames` has no
    candidates."""
    bound = _bound_roles_for_coverage(role_bindings, coverage=coverage, stat_var=stat_var)
    if bound is None:
        return MemberProbe(candidates=(), admission=CoverageAdmission())
    return coverage_probe(stat_var=stat_var, bound=bound, coverage=coverage)


type SeriesBreadth = Callable[[RequestedWindow | None], Mapping[str, int]]
"""Each facet id's series count under a window: the places, or the role tuples, at which
the facet holds a point in it. `rank_candidates` leads with it."""


def series_breadth(
    by_series: Mapping[str, Sequence[Facet]], *, window: RequestedWindow | None
) -> dict[str, int]:
    """How many of `by_series`' series each facet id holds a point at inside `window`
    (`overlaps_window`). A facet absent from the result holds none."""
    counts: dict[str, int] = {}
    for facets in by_series.values():
        held = {
            f.facet_id
            for f in facets
            if f.facet_id is not None and f.obs_count > 0 and overlaps_window(f, window=window)
        }
        for facet_id in held:
            counts[facet_id] = counts.get(facet_id, 0) + 1
    return counts


@dataclass(frozen=True)
class PlaceAdmission:
    """The probed places at which origin admission passed each candidate facet id, for one
    subject-only StatVar. These are the places the request binds, or the default place scope's
    when it binds none. The graph probe reads admitted facets per place, and admission withholds
    every other facet at a place where DFO's facet is present
    (`CatalogueSnapshot.admit_series`), so a facet id admitted at one place can be withheld
    at another. A fetched row serves only when its `facetId` was admitted at its
    `observationAbout` (`columnar.admission_mask`)."""

    places_by_facet: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class CoverageAdmission:
    """Admission for a role-bearing StatVar, whose candidates come from the coverage index.
    The index holds DFO's data alone, which admission never withholds, so its rows serve
    unmasked."""


RowAdmission = PlaceAdmission | CoverageAdmission


@dataclass(frozen=True)
class MemberProbe:
    """One member's facet candidates, merged across every place or role tuple the request
    binds, the admission its fetched rows must pass, and each facet's series count under a
    window (`breadth`), which leads the election (`rank_candidates`). `breadth` is None for
    candidates read at one series, where every facet counts alike. `places` holds the places
    its fetch names when the request binds no place, its default place scope
    (`fetch_role_bindings`), and is empty for every other request. `every_place` is false when
    those places are the countries with data because the full list of places was too large or
    unavailable (`place_scope.DefaultPlaceScope.every_place`)."""

    candidates: tuple[Facet, ...]
    admission: RowAdmission
    places: tuple[str, ...] = ()
    every_place: bool = True
    breadth: SeriesBreadth | None = field(default=None, compare=False)

    def breadth_in(self, window: RequestedWindow | None) -> Mapping[str, int] | None:
        """Each facet id's series count under `window`, or None when the probe counts none."""
        return self.breadth(window) if self.breadth is not None else None

    @property
    def scope(self) -> DefaultPlaceScope:
        """`places` and `every_place` as the default place scope that resolved them."""
        return DefaultPlaceScope(places=self.places, every_place=self.every_place)


# A subject-only member the probe read nothing for. It has no candidate and admits no row.
_NO_SUBJECT_PROBE = MemberProbe(candidates=(), admission=PlaceAdmission(places_by_facet={}))


def admitted_candidates(
    candidates: Sequence[Facet], candidate_filter: Callable[[Facet], bool] | None
) -> list[Facet]:
    """The probed candidates a caller's facet filter admits, which every election then
    ranks. A filter that admits none of a non-empty probe raises `FacetNotAvailableError`,
    the refusal a bound ref with no observations gets. No filter admits every candidate."""
    if candidate_filter is None or not candidates:
        return list(candidates)
    admitted = [c for c in candidates if candidate_filter(c)]
    if not admitted:
        raise FacetNotAvailableError(
            "no probed facet satisfies the request's facet filter", pointer="/facets"
        )
    return admitted


def probe_facets(
    *,
    stat_var: str,
    bound_entities: Sequence[str],
    graph: GraphReader,
    availability: AvailabilityReader,
    settle: bool,
    declared_roles: tuple[str, ...] = _SUBJECT_ROLE_ONLY,
    role_bindings: Mapping[str, EntityFilter] | None = None,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
) -> MemberProbe:
    """Probes candidates for the request's bound entities. The pipeline calls this once per
    generation and shares candidates across all three axes.

    Subject-only StatVars use one `graph.variable_facets_batch` call across all entities.
    Results merge by `facet_id`, summing counts and widening date
    ranges, and the probe records the places each facet id was admitted at. A subject-only
    request with no `role_bindings` and no `bound_entities` is probed at its default place
    scope (`place_scope.read_default_presence`, read through `availability`), and the probe
    returns those places as `places`. A mint passes `settle=True`, so its scope does not depend
    on how busy the settle pool is; Search passes `settle=False` and answers fast.
    Role-bearing StatVars require
    `role_bindings` and `coverage` because `/v2/observation` filters subject entities, and an
    unbound role covers every entity in the index. Missing inputs raise
    `MaterialisationUsageError`. An uncovered StatVar returns no candidates without an SDMX call.

    Only plain DCID bindings contribute. A subject-only `ContainedIn` binding has no known
    expansion and returns no candidates under an empty place admission. Graph and
    availability reads receive `deadline`, and their errors propagate."""
    if declared_roles == _SUBJECT_ROLE_ONLY:
        if not bound_entities:
            if role_bindings:
                return _NO_SUBJECT_PROBE
            scope = read_default_presence(
                graph=graph,
                availability=availability,
                stat_vars=[stat_var],
                deadline=deadline,
                settle=settle,
            ).scope(stat_var)
            by_entity = (
                graph.variable_facets_batch(
                    stat_var=stat_var, entities=scope.places, deadline=deadline
                )
                if scope.places
                else {}
            )
            return _scoped_probe(by_entity, scope)
        by_entity = graph.variable_facets_batch(
            stat_var=stat_var, entities=bound_entities, deadline=deadline
        )
        return subject_probe(by_entity, bound_entities)
    if role_bindings is None:
        raise MaterialisationUsageError(
            f"the facet probe for stat_var {stat_var!r} declares roles {declared_roles!r}, "
            "a role-bearing shape whose facet candidates require role_bindings; none was given"
        )
    if coverage is None:
        raise MaterialisationUsageError(
            f"the facet probe for stat_var {stat_var!r} declares roles {declared_roles!r}, "
            "a role-bearing shape whose facet candidates require coverage; none was given"
        )
    return _role_bearing_probe(stat_var=stat_var, role_bindings=role_bindings, coverage=coverage)


def _scoped_probe(
    by_entity: Mapping[str, Sequence[Facet]], scope: DefaultPlaceScope
) -> MemberProbe:
    """`subject_probe` at a default place scope's places, carrying the scope."""
    return replace(
        subject_probe(by_entity, scope.places),
        places=scope.places,
        every_place=scope.every_place,
    )


def probe_from_candidates(
    candidates: Sequence[Facet],
    *,
    declared_roles: tuple[str, ...],
    bound_entities: Sequence[str],
) -> MemberProbe:
    """A member probe for candidates a caller already read for one request, admitting rows by
    the rule `probe_facets` applies: a subject-only StatVar's candidates, read at its one
    bound place, admit their facet ids there (`subject_probe`); a role-bearing StatVar's come
    from coverage and serve unmasked (`CoverageAdmission`). A subject-only request binding
    other than one place raises `MaterialisationUsageError`, since its candidates cannot say
    where each facet was admitted."""
    if declared_roles != _SUBJECT_ROLE_ONLY:
        return MemberProbe(candidates=tuple(candidates), admission=CoverageAdmission())
    if len(bound_entities) != 1:
        raise MaterialisationUsageError(
            f"precomputed candidates for a subject-only request bind {len(bound_entities)} "
            "places; they can say where each facet was admitted only for one"
        )
    return subject_probe({bound_entities[0]: candidates}, bound_entities)


def subject_probe(
    by_entity: Mapping[str, Sequence[Facet]], bound_entities: Sequence[str]
) -> MemberProbe:
    """Merges a subject-only probe's per-entity facets, keeps the bound places each facet id
    was admitted at, and counts the places each facet holds a point at under a window
    (`series_breadth`). A facet without an id never dispatches, so it admits no row. A mint
    builds its probe here, and a Search read plan each member's from the facets it read at
    the places an answer names."""
    places_by_facet: dict[str, set[str]] = {}
    by_place = {entity: tuple(by_entity.get(entity, ())) for entity in bound_entities}
    for entity, facets in by_place.items():
        for facet in facets:
            if facet.facet_id is not None:
                places_by_facet.setdefault(facet.facet_id, set()).add(entity)
    return MemberProbe(
        candidates=tuple(merge_facets_by_series(by_place, bound_entities)),
        admission=PlaceAdmission(
            places_by_facet={
                facet_id: frozenset(places) for facet_id, places in places_by_facet.items()
            }
        ),
        breadth=lambda window: series_breadth(by_place, window=window),
    )


def merge_facets_by_series(
    by_series: Mapping[str, Sequence[Facet]], series: Sequence[str]
) -> list[Facet]:
    """Merges each series' facets by `facet_id` in first-seen order over `series`
    (`merge_facets`), so `observation_period` comes from the first series. A mint merges its
    bound places here, and a Search read plan the places an answer names, so both rank one
    candidate per facet across them (`rank_candidates`)."""
    grouped: dict[str | None, list[Facet]] = {}
    for key in series:
        for facet in by_series.get(key, []):
            grouped.setdefault(facet.facet_id, []).append(facet)
    return [merge_facets(facets) for facets in grouped.values()]


def entity_filter_key(role_bindings: Mapping[str, EntityFilter]) -> tuple[object, ...]:
    """Builds a scope key preserving role and filter order.

    SDMX applies each `c[role]` filter to every variable in a request, so
    members with different scopes cannot share one.
    """

    def _filter_key(filt: EntityFilter) -> tuple[object, ...]:
        if isinstance(filt, ContainedIn):
            return ("contained_in", filt.parent, filt.child_type)
        return ("dcids", tuple(filt))

    return tuple(sorted((role, _filter_key(filt)) for role, filt in role_bindings.items()))


def request_scope_key(
    request: MaterialisationRequest,
) -> tuple[tuple[str, ...], tuple[object, ...]]:
    """Keys a request by its declared roles and entity scope.

    Requests sharing a key are probed together (`probe_collection_candidates`), and
    `generation.request_batch_key` embeds this key, so each fetch group lies inside one probe
    group. The roles are part of the key because an empty binding has the same entity scope
    under every role shape."""
    return request.declared_roles, entity_filter_key(request.role_bindings)


def binds_no_place(
    role_bindings: Mapping[str, EntityFilter], declared_roles: tuple[str, ...]
) -> bool:
    """Whether a request is subject-only and binds no entity, so the default place scope
    supplies its places."""
    return declared_roles == _SUBJECT_ROLE_ONLY and not role_bindings


def fetch_role_bindings(
    role_bindings: Mapping[str, EntityFilter],
    declared_roles: tuple[str, ...],
    *,
    places: Sequence[str],
) -> Mapping[str, EntityFilter]:
    """The entity filters a fetch sends: `places` on the subject role for a request
    `binds_no_place` matches, and `role_bindings` unchanged otherwise. `places` is the member
    probe's own (`MemberProbe.places`), so the fetch names the places the probe read."""
    if binds_no_place(role_bindings, declared_roles):
        return {_SUBJECT_ROLE: places}
    return role_bindings


def default_scope_of(members: Sequence[MemberProbe]) -> DefaultPlaceScope:
    """The places a request binding no place is fetched at, across `members`, in the order
    Search states a default scope's places (`place_scope.scope_places`), and whether they are
    every place with data, which holds only when each member's are. Empty when every member
    binds a place."""
    return scope_of_places(
        (place for member in members for place in member.places),
        every_place=all(member.every_place for member in members),
    )


def merge_facets(facets: Sequence[Facet]) -> Facet:
    """The first of `facets` with the others' points added: counts summed, date bounds
    widened, and dated points concatenated in order. The merged dates are complete while
    each facet's dates account for its observations, so a facet read without dates leaves
    them incomplete. Metadata stays the first facet's.

    Each facet's points are copied once, so merging a facet across many series costs the
    points they hold, and a lone facet is returned as it is. `facets` is non-empty."""
    first = facets[0]
    if len(facets) == 1:
        return first
    dates = tuple(chain.from_iterable(f.dates for f in facets))
    return replace(
        first,
        obs_count=sum(f.obs_count for f in facets),
        earliest_date=reduce(_earliest, (f.earliest_date for f in facets)),
        latest_date=reduce(_latest, (f.latest_date for f in facets)),
        dates=dates,
        values=tuple(chain.from_iterable(f.values for f in facets)),
        dates_complete=all(
            _dates_account_for_points(f) if dates else f.dates_complete for f in facets
        ),
    )


def _dates_account_for_points(facet: Facet) -> bool:
    """Whether `facet`'s dates hold every one of its observations."""
    return facet.dates_complete and (bool(facet.dates) or facet.obs_count == 0)


@dataclass(frozen=True)
class CollectionFacetProbe:
    """Member probes for a request group, in request order.
    Subject-only groups use one graph batch. Role-bearing groups use coverage. Each
    `members[i]` matches `probe_facets` for request `i`. Reuse as `precomputed_probe`
    requires the same request order. A generation that reuses this probe fetches each request
    that binds no entity at its member's `places`.

    `un_unserved` is what the probe's UN reads could not serve. A materialisation that reuses
    the probe adds it to the caller's `recording_un_blocked_places` record, so a probe made
    before that record opened still reports the places the UN refused.

    `revision` is read before the first probe call so the caller's drift guard covers the
    probe-to-generation gap. It is `None` without a reader.

    `provenance_refresh` maps distinct candidate provenance ids to refresh dates from one
    batched `node_arcs_many` call. It is empty when no candidate has an id. Currency checks
    use these dates.
    """

    members: tuple[MemberProbe, ...]
    revision: RevisionSnapshot | None = None
    provenance_refresh: Mapping[str, ProvenanceRefresh] = field(default_factory=dict)
    un_unserved: UnUnserved = field(default_factory=UnUnserved)

    @property
    def candidates(self) -> tuple[tuple[Facet, ...], ...]:
        """Each member's candidates, in request order."""
        return tuple(member.candidates for member in self.members)

    @property
    def default_scope(self) -> tuple[str, ...]:
        """The places of `default_scope_of` over the members."""
        return default_scope_of(self.members).places


def probe_collection_facets(
    requests: Sequence[MaterialisationRequest],
    *,
    graph: GraphReader,
    availability: AvailabilityReader,
    settle: bool,
    coverage: CoverageIndex | None = None,
    revision_reader: RevisionReader | None = None,
    deadline: float | None = None,
) -> CollectionFacetProbe:
    """Probes a request group and returns its candidates, revision snapshot, and provenance
    refresh dates. Reads `revision_reader` before any probe call. Then fetches distinct
    provenance dates in one `node_arcs_many` call. That method has no `deadline` parameter.
    Empty `requests` returns empty candidates and refresh dates without network calls.
    For grouping and refusals, see `probe_collection_candidates`.

    The probe runs under its own `recording_un_blocked_places` record and returns what that
    record held as `un_unserved`. A UN read that fails beside a read of our variables
    therefore joins `un_unserved` and leaves the UN members without candidates."""
    revision = revision_reader.read() if revision_reader is not None else None
    with recording_un_blocked_places() as un_record:
        members = probe_collection_candidates(
            requests,
            graph=graph,
            availability=availability,
            settle=settle,
            coverage=coverage,
            deadline=deadline,
        )
    provenance_ids = sorted(
        {
            facet.provenance_id
            for member in members
            for facet in member.candidates
            if facet.provenance_id is not None
        }
    )
    provenance_refresh = _probe_provenance_refresh(provenance_ids, graph=graph)
    return CollectionFacetProbe(
        members=members,
        revision=revision,
        provenance_refresh=provenance_refresh,
        un_unserved=un_record.snapshot(),
    )


def probe_collection_candidates(
    requests: Sequence[MaterialisationRequest],
    *,
    graph: GraphReader,
    availability: AvailabilityReader,
    settle: bool,
    coverage: CoverageIndex | None = None,
    deadline: float | None = None,
) -> tuple[MemberProbe, ...]:
    """Returns one member probe per request, in input order. Members with the same declared
    roles and entity scope (`request_scope_key`) share one probe. Subject-only groups make one
    graph batch across distinct StatVars and entities. Role-bearing groups look up coverage,
    where an unbound role covers every entity in the index. Missing `RoleNames` yields no
    candidates. Results are merged per member and include only that member's bound entities.

    Subject-only members that bind no entity take their scopes from one presence read for all
    of them (`place_scope.read_default_presence`, read through `availability`, settling with
    `settle`, as `probe_facets` does). Each member's scope is its own, every place where it
    holds data or its countries with data (`DefaultPresence.scope` of that StatVar alone).
    Members are probed at their own places, with one graph batch per distinct place list
    (`_probe_at_own_places`), and each member's probe carries its places
    (`MemberProbe.places`), so a member of a group gets the scope,
    candidates, rows and `content_version` it gets alone.

    Empty requests return no members. A subject-only group whose bindings name no plain
    dcid gets no candidates under an empty place admission and makes no call. `ContainedIn`
    bindings are skipped because expansion requires a graph read.

    Before probing, role-bearing groups require `coverage` or raise `MaterialisationUsageError`.
    The cap counts `len(group_stat_vars) * len(group_entities)` across subject-only groups that
    bind entities, so coverage members count nothing, plus each distinct StatVar's own places
    for members that bind none, after the presence read resolves them. Totals above
    `_MAX_COLLECTION_PROBE_PAIRS` raise `CollectionProbeCeilingExceededError`, whose detail
    names the remedy, and no member's scope shrinks to fit.

    Graph and availability reads receive `deadline`. `GraphDeadlineExceededError` or transport
    errors fail the probe at the first failing group in first-seen order. Coverage lookups have
    no deadline. This function reads no revision or refresh data. `probe_collection_facets`
    adds both."""
    if not requests:
        return ()

    member_specs: list[tuple[str, tuple[str, ...]]] = []
    # Members sharing a key share their declared roles and bindings, so the first one's stand
    # for the group.
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, request in enumerate(requests):
        bare_stat_var = request.stat_var.removeprefix(_STAT_VAR_REF_PREFIX)
        member_specs.append((bare_stat_var, _flatten_bound_entities(request.role_bindings)))
        groups.setdefault(request_scope_key(request), []).append(index)

    if coverage is None and any(
        requests[indices[0]].declared_roles != _SUBJECT_ROLE_ONLY for indices in groups.values()
    ):
        raise MaterialisationUsageError(
            "the group facet probe got a role-bearing entity-scope group whose facet "
            "candidates require a coverage index; none was given"
        )

    # Every subject-only member that binds no entity shares one key, so one group holds them.
    scopes: dict[str, DefaultPlaceScope] = {}
    group_plans: list[tuple[list[int], list[str], list[str]]] = []
    total_pairs = 0
    for indices in groups.values():
        first = requests[indices[0]]
        group_stat_vars = list(dict.fromkeys(member_specs[index][0] for index in indices))
        group_entities = list(
            dict.fromkeys(dcid for index in indices for dcid in member_specs[index][1])
        )
        # Coverage filtering does not scale with entity count. The cap bounds graph batches.
        if binds_no_place(first.role_bindings, first.declared_roles):
            presence = read_default_presence(
                graph=graph,
                availability=availability,
                stat_vars=group_stat_vars,
                deadline=deadline,
                settle=settle,
            )
            scopes = {stat_var: presence.scope(stat_var) for stat_var in group_stat_vars}
            total_pairs += sum(len(scope.places) for scope in scopes.values())
        elif first.declared_roles == _SUBJECT_ROLE_ONLY:
            total_pairs += len(group_stat_vars) * len(group_entities)
        group_plans.append((indices, group_stat_vars, group_entities))

    if total_pairs > _MAX_COLLECTION_PROBE_PAIRS:
        raise CollectionProbeCeilingExceededError(
            f"the request's entity-scope groups would probe {total_pairs} "
            f"stat_var/entity pairs, over the cap of {_MAX_COLLECTION_PROBE_PAIRS}; a member "
            "that binds no place counts every place where it holds data, so name the places "
            "to cover or request fewer members"
        )

    members_by_index: list[MemberProbe] = [
        _NO_SUBJECT_PROBE
        if request.declared_roles == _SUBJECT_ROLE_ONLY
        else MemberProbe(candidates=(), admission=CoverageAdmission())
        for request in requests
    ]
    for indices, group_stat_vars, group_entities in group_plans:
        first = requests[indices[0]]
        if binds_no_place(first.role_bindings, first.declared_roles):
            at_own_places = _probe_at_own_places(scopes, graph=graph, deadline=deadline)
            for index in indices:
                bare_stat_var = member_specs[index][0]
                members_by_index[index] = _scoped_probe(
                    at_own_places.get(bare_stat_var, {}), scopes[bare_stat_var]
                )
            continue
        if first.declared_roles == _SUBJECT_ROLE_ONLY:
            if not group_entities:
                continue
            by_stat_var = graph.observation_facets_batch_many(
                stat_vars=group_stat_vars, entities=group_entities, deadline=deadline
            )
            for index in indices:
                bare_stat_var, bound_entities = member_specs[index]
                members_by_index[index] = subject_probe(
                    by_stat_var.get(bare_stat_var, {}), bound_entities
                )
            continue
        # Missing `RoleNames` yields no candidates.
        # The earlier group check guarantees `coverage` is present.
        assert coverage is not None
        resolved: dict[str, MemberProbe] = {}
        for bare_stat_var in group_stat_vars:
            resolved[bare_stat_var] = _role_bearing_probe(
                stat_var=bare_stat_var, role_bindings=first.role_bindings, coverage=coverage
            )
        for index in indices:
            members_by_index[index] = resolved[member_specs[index][0]]

    return tuple(members_by_index)


def _probe_at_own_places(
    scopes: Mapping[str, DefaultPlaceScope], *, graph: GraphReader, deadline: float | None
) -> dict[str, Mapping[str, Sequence[Facet]]]:
    """Each StatVar's facets at its own places, one graph batch per distinct place list.

    A batch of UN variables alone that fails with `GraphReadError` is a UN read alongside ours
    when `scopes` also names our variables, so `origin_catalogue.record_un_read_failure`
    records it and it reads as empty, as a UN read inside a batch with ours does in
    `origin_catalogue.graph.OriginGraphClient`. Otherwise it raises."""
    stat_vars_by_places: dict[tuple[str, ...], list[str]] = {}
    for stat_var, scope in scopes.items():
        if scope.places:
            stat_vars_by_places.setdefault(scope.places, []).append(stat_var)
    names_ours = not all(map(is_un_dcid, scopes))
    by_stat_var: dict[str, Mapping[str, Sequence[Facet]]] = {}
    for places, stat_vars in stat_vars_by_places.items():
        try:
            by_stat_var.update(
                graph.observation_facets_batch_many(
                    stat_vars=stat_vars, entities=places, deadline=deadline
                )
            )
        except GraphReadError as exc:
            if not all(map(is_un_dcid, stat_vars)) or not record_un_read_failure(
                exc, variables=stat_vars, places=places, alongside_ours=names_ours
            ):
                raise
    return by_stat_var


def _probe_provenance_refresh(
    provenance_ids: Sequence[str], *, graph: GraphReader
) -> dict[str, ProvenanceRefresh]:
    """Fetches last and next refresh dates in one graph batch. A missing node or arcs yields
    `ProvenanceRefresh(None, None)`, leaving currency checks with no date to compare. Empty
    `provenance_ids` returns `{}` without a network call."""
    if not provenance_ids:
        return {}
    arcs_by_id = graph.node_arcs_many(provenance_ids, properties=_PROVENANCE_REFRESH_PROPERTIES)
    refresh_by_id: dict[str, ProvenanceRefresh] = {}
    for provenance_id in provenance_ids:
        arcs = arcs_by_id.get(provenance_id)
        last_values = arcs.get("lastDataRefreshDate", ()) if arcs is not None else ()
        next_values = arcs.get("nextDataRefreshDate", ()) if arcs is not None else ()
        refresh_by_id[provenance_id] = ProvenanceRefresh(
            last_data_refresh_date=last_values[0] if last_values else None,
            next_data_refresh_date=next_values[0] if next_values else None,
        )
    return refresh_by_id


def _flatten_bound_entities(role_bindings: Mapping[str, EntityFilter]) -> tuple[str, ...]:
    """Returns unique plain dcids in first-seen order. Skips `ContainedIn` because expansion
    requires a graph read. Kept local because `generation` imports this module."""
    seen: set[str] = set()
    entities: list[str] = []
    for filt in role_bindings.values():
        if isinstance(filt, ContainedIn):
            continue
        for dcid in filt:
            if dcid not in seen:
                seen.add(dcid)
                entities.append(dcid)
    return tuple(entities)


def _leading_year(date: str | None) -> int | None:
    """Returns the four-digit year that opens a graph date, or `None` when absent or unparseable."""
    if date is None:
        return None
    try:
        return int(date[:4])
    except ValueError:
        return None


def _date_in_range(date: str, window: RequestedWindow) -> bool:
    """Whether `date`'s year falls inside `window`'s year bounds. A date with no leading
    year counts as inside, so an unparseable date never excludes its facet."""
    year = _leading_year(date)
    if year is None:
        return True
    return (window.start_year is None or year >= window.start_year) and (
        window.end_year is None or year <= window.end_year
    )


def overlaps_window(candidate: Facet, *, window: RequestedWindow | None) -> bool:
    """Whether the candidate holds a point in the window's range: one of its `dates` when it
    carries them all (`dates_complete`), else its span (`spans_window`), so a facet with a
    gap across the window does not overlap it when its dates are known. An undated candidate
    counts as holding a point when its span reaches the window. Every election narrows its
    pool by this (`narrow_to_window`), and a read plan decides which series it serves by it."""
    if window is None or window.kind != "range":
        return True
    if candidate.dates and candidate.dates_complete:
        return any(_date_in_range(date, window) for date in candidate.dates)
    return spans_window(candidate, window=window)


def spans_window(candidate: Facet, *, window: RequestedWindow | None) -> bool:
    """Whether the candidate's `[earliest_date, latest_date]` span reaches the window's range.

    Only a `range` window with a year bound can exclude a candidate. `all`, `latest` and
    `None` admit every candidate. An unknown date bound never excludes, so a candidate with no
    `earliest_date` reaches the window when its `latest_date` reaches the window start."""
    if window is None or window.kind != "range":
        return True
    earliest = _leading_year(candidate.earliest_date)
    latest = _leading_year(candidate.latest_date)
    if window.start_year is not None and latest is not None and latest < window.start_year:
        return False
    return not (window.end_year is not None and earliest is not None and earliest > window.end_year)


def years_in_window(candidate: Facet, *, window: RequestedWindow | None) -> int:
    """The window years the candidate's span covers, the measure every election ranks by
    under a range: the years from the later of the span's and the window's starts to the
    earlier of their ends.

    It reads `earliest_date` and `latest_date` alone, which a graph facet read with or
    without dates, a coverage candidate and a bulk candidate all carry, so a search answer
    and a mint or datasets read of its Spec rank the same candidates alike. A facet with a
    gap inside its span counts the gap's years. Zero for a span that misses the window
    (`spans_window`) and for `all`, `latest` and `None`, which rank by count alone. A bound
    unknown on both the span and the window takes the other end's year."""
    if window is None or window.kind != "range" or not spans_window(candidate, window=window):
        return 0
    starts = [
        y for y in (_leading_year(candidate.earliest_date), window.start_year) if y is not None
    ]
    ends = [y for y in (_leading_year(candidate.latest_date), window.end_year) if y is not None]
    if not starts and not ends:
        return 1
    start = max(starts) if starts else min(ends)
    end = min(ends) if ends else max(starts)
    return max(end - start + 1, 1)


def narrow_to_window(candidates: Sequence[Facet], *, window: RequestedWindow | None) -> list[Facet]:
    """Keeps the candidates known to hold a point in `window` (`overlaps_window`), or all of
    them when none does.

    Deployment fallback applies this before ranking and before `dfo_precedence`, so the window
    changes which facet wins and the selection reasons describe the narrowed pool. A window no
    candidate reaches leaves the pool whole, and the lifetime ranking picks the facet as it
    would without a window."""
    overlapping = [c for c in candidates if overlaps_window(c, window=window)]
    return overlapping or list(candidates)


def fallback_pool(
    candidates: Sequence[Facet],
    declared: Mapping[str, str | None],
    *,
    window: RequestedWindow | None,
) -> list[Facet]:
    """The pool a deployment-fallback axis ranks: `candidates` carrying every declared default
    in `declared` (axis to resolved ref), then narrowed to `window` (`narrow_to_window`).

    A declared default wins before window narrowing, so a fallback axis takes its value from a
    facet that carries the declared refs, never from one the window reaches but the declared
    default excludes. When no candidate carries them all, the whole pool stays and the
    caller's joint-coherence check refuses. A declared facet with no point in the window keeps
    its place, as `narrow_to_window` keeps a pool no candidate reaches. Member and group
    selection both rank this pool."""
    consistent = [
        c for c in candidates if all(facet_axis_value(c, a) == ref for a, ref in declared.items())
    ]
    return narrow_to_window(consistent or list(candidates), window=window)


def declared_default_policies(
    bindings: Mapping[str, FacetBinding], *, graph: GraphReader, stat_var: str
) -> dict[str, FacetDefaultPolicy | None]:
    """Each axis `bindings` leaves to selection (unbound, or a unit request), mapped to its
    graph-declared default policy or `None`. Bound axes are absent. One read per axis, shared
    by every axis's `select_facet` call."""
    return {
        axis: graph.default_facet_policy(stat_var=stat_var, axis=axis)
        for axis in FACET_AXES
        if bindings.get(axis, FacetBinding(state="unbound")).state != "bound"
    }


def _declared_refs(
    policies: Mapping[str, FacetDefaultPolicy | None],
    dispatchable: Sequence[Facet],
    all_bindings: Mapping[str, FacetBinding],
    *,
    except_axis: str,
) -> dict[str, str | None]:
    """The declared defaults that resolve on the axes other than `except_axis`: each policy
    some candidate carries, matched in the pool `select_facet` matches it in (every candidate
    for an unbound axis, the bound-narrowed pool for a unit request)."""
    refs: dict[str, str | None] = {}
    for axis, policy in policies.items():
        if axis == except_axis or policy is None:
            continue
        state = all_bindings.get(axis, FacetBinding(state="unbound")).state
        pool = (
            dispatchable
            if state == "unbound"
            else _narrow_by_bound_axes(dispatchable, all_bindings)
        )
        if _match_policy(policy, pool, axis) is not None:
            refs[axis] = policy.resolved_ref
    return refs


def rank_candidates(
    candidates: Sequence[Facet],
    *,
    window: RequestedWindow | None,
    precedence: frozenset[str] = frozenset(),
    breadth: Mapping[str, int] | None = None,
) -> tuple[Facet, ...]:
    """Orders candidates holding a point in `window` first (`overlaps_window`), then those whose
    provenance is in `precedence` (`dfo_precedence`), then by descending `breadth`, descending
    window years covered (`years_in_window`), descending `obs_count`, descending
    `latest_date`, then `provenance_id`, `measurement_method`, `unit`, and `facet_id`. A
    `None` tiebreak value sorts last. Stable sorts make the total order deterministic.

    `breadth` maps a `facet_id` to the number of series the facet holds a point at in
    `window`: the places, or the role tuples, a request binds (`MemberProbe.breadth`,
    `series_breadth`). One facet serves a variable across all of them, so the facet covering
    the most leads, ahead of the years and the count. A mint, a datasets read, a unit
    decision, a group selection and Search's read plans, scope and bulk elections all pass
    it. A facet it omits counts zero. `None`, for a single series, skips the step.

    `window` is required so every ranking site states whether a requested year range applies.
    Under a range the election is the top span-ranked facet among those known to hold a
    point in the window: the facet whose span covers more of the window's years leads, so
    one spanning 2020 to 2022 ranks above one spanning 1970 to 2021 for a 2020 to 2022
    request, and lifetime count breaks a tie. `None`, `all` and `latest` windows rank by
    count. The top candidate is the top of `narrow_to_window`'s pool.

    A reader holding dated candidates and one holding undated ones rank alike, and differ
    only in the pool: dates show that a facet with a gap across the whole window holds no
    point there, which a span cannot. An undated read of such a pool elects the gapped facet
    when its span covers the most window years."""
    ranked = list(candidates)
    ranked.sort(key=lambda c: _none_last(c.facet_id))
    ranked.sort(key=lambda c: _none_last(c.unit))
    ranked.sort(key=lambda c: _none_last(c.measurement_method))
    ranked.sort(key=lambda c: _none_last(c.provenance_id))
    # `latest_date` sorts descending (most recent first) with `None` last:
    # treating `None` as `""`, the empty string, sorts lowest in ascending
    # order and therefore last once the whole pass is reversed.
    ranked.sort(key=lambda c: c.latest_date or "", reverse=True)
    ranked.sort(key=lambda c: c.obs_count, reverse=True)
    ranked.sort(key=lambda c: years_in_window(c, window=window), reverse=True)
    if breadth is not None:
        ranked.sort(key=lambda c: breadth.get(c.facet_id or "", 0), reverse=True)
    ranked.sort(key=lambda c: c.provenance_id not in precedence)
    ranked.sort(key=lambda c: not overlaps_window(c, window=window))
    return tuple(ranked)


@dataclass(frozen=True)
class RankedPool:
    """The fallback pool (`pool`, `fallback_pool`), the `dfo_precedence` set read on that pool,
    and the pool in deployment-policy order (`ranked`), top first."""

    pool: tuple[Facet, ...]
    precedence: frozenset[str]
    ranked: tuple[Facet, ...]


def rank_pool(
    candidates: Sequence[Facet],
    *,
    stat_vars: Iterable[str],
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None = None,
    declared: Mapping[str, str | None] | None = None,
) -> RankedPool:
    """Keeps the `candidates` carrying every declared default in `declared` and narrows them
    to `window` (`fallback_pool`), then ranks them with DFO's first for a variable in the
    origin catalogue's overlap set (`dfo_precedence`), by `breadth`, by the window years
    each span covers and by count (`rank_candidates`, `years_in_window`).

    This is the one deployment-fallback election. A single member (`select_facet`), a unit
    request (`_select_unit_request`, `decide_member_unit`), a group (`select_group_facets`)
    and Search's read plan, previews and bulk elections rank their pools here over each
    candidate's span and count, the fields every probe carries. A search answer and a mint
    or datasets read of its Spec under the same window therefore serve the same facet,
    except where only dates show that the top-ranked facet holds no point in the window
    (`rank_candidates`)."""
    pool = tuple(fallback_pool(candidates, declared or {}, window=window))
    precedence = dfo_precedence(stat_vars, pool)
    return RankedPool(
        pool=pool,
        precedence=precedence,
        ranked=rank_candidates(pool, window=window, precedence=precedence, breadth=breadth),
    )


def served_facets(
    candidates: Sequence[Facet],
    *,
    stat_vars: Iterable[str],
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None = None,
) -> tuple[Facet, ...]:
    """The facets a preview for `stat_vars` under `window` serves from `candidates` when no
    axis is bound and the graph declares no default: the top of `rank_pool` over the
    dispatchable candidates, and every dispatchable candidate carrying the top's provenance,
    measurement method and unit (`joint_dispatch_facet_ids` over the selections those values
    form), since dispatch fetches each of them and a `latest` read keeps the latest period
    across them. Empty when no candidate carries a `facet_id`, since only those dispatch.

    A top carrying no provenance serves too: unbound, `select_facet` refuses it, so the
    preview binds its provenance axis to the null ref (`FacetBinding(state="bound",
    ref=None)`), which selects it and dispatches its group by facet id, as a bulk read
    dispatches its facet."""
    dispatchable = [c for c in candidates if c.facet_id is not None]
    if not dispatchable:
        return ()
    top = rank_pool(dispatchable, stat_vars=stat_vars, window=window, breadth=breadth).ranked[0]
    refs = {axis: facet_axis_value(top, axis) for axis in FACET_AXES}
    return tuple(c for c in dispatchable if _matches_refs(c, refs))


def dfo_precedence(stat_vars: Iterable[str], candidates: Sequence[Facet]) -> frozenset[str]:
    """DFO's provenances among `candidates` when the bound origin catalogue lists any of
    `stat_vars` under both DFO and base Data Commons (its overlap set), and empty
    otherwise.

    Admission withholds a base facet only at places where DFO holds data, so a base facet
    that wins on observation count would be fetched at DFO's places too. Ranking DFO's facet
    first whenever DFO has data at any requested place serves DFO's data there, and places
    only the base source covers read as a coverage gap. Outside the overlap set every
    candidate comes from one origin and the ranking is unchanged."""
    catalogue = current_catalogue()
    if not any(
        DFO in (held := catalogue.data_commons(stat_var.removeprefix(_STAT_VAR_REF_PREFIX)))
        and BASE in held
        for stat_var in stat_vars
    ):
        return frozenset()
    return frozenset(
        provenance_id
        for candidate in candidates
        if (provenance_id := candidate.provenance_id) is not None
        and (row := catalogue.provenance(provenance_id)) is not None
        and row.data_commons == DFO
    )


def _decided_by_precedence(top: Facet, viable: Sequence[Facet], precedence: frozenset[str]) -> bool:
    """Whether `dfo_precedence` put `top` ahead of a viable candidate from another origin."""
    return top.provenance_id in precedence and any(
        c.provenance_id not in precedence for c in viable
    )


def _none_last(value: str | None) -> tuple[bool, str]:
    return (value is None, value or "")


def select_facet(
    axis: str,
    binding: FacetBinding,
    *,
    candidates: Sequence[Facet],
    graph: GraphReader,
    stat_var: str,
    window: RequestedWindow | None,
    all_bindings: Mapping[str, FacetBinding] = {},
    policies: Mapping[str, FacetDefaultPolicy | None],
    breadth: Mapping[str, int] | None = None,
) -> FacetSelection:
    """Resolves one axis and raises at most one refusal. The pipeline aggregates axis errors.
    Only candidates with `facet_id` dispatch.

    A bound ref resolves through `select_bound_facet`. A request binding
    resolves the native unit and applies only to `unit`. Other axes raise
    `MaterialisationUsageError`.

    An unbound axis first tries the graph's declared default. A matching ref returns
    `basis="declared_default"`. An unavailable declared ref raises `FacetNotAvailableError`.
    Without a declared policy, candidates are narrowed by bound axes, then to those overlapping
    `window` (`narrow_to_window`), and ranked by deployment policy, led by `breadth`, each
    facet's series count under `window` (`MemberProbe.breadth_in`). An empty pool raises
    `DefaultFacetUndefinedError`. A bound ref or declared default ignores `window`.

    `policies` (`declared_default_policies`) holds every unresolved axis's declared default,
    read once for all axes. This axis's policy comes from the map, and a fallback pool keeps
    only the facets carrying the other axes' declared defaults (`fallback_pool`), so a
    declared default wins before window narrowing on every axis.

    Fallback records `sole_candidate`, `most_observations`, `dfo_precedence` when
    `dfo_precedence` ranked DFO's facet over another origin's, or
    `most_observations_over_disagreement`, which triggers a warning. Each reason describes the
    window-narrowed pool. About two in five observed pairs lack `measurementMethod` or `unit`,
    so a missing value returns `ref=None` on those axes. Missing provenance raises
    `DefaultFacetUndefinedError`. The
    caller checks cross-axis fit."""
    dispatchable = [c for c in candidates if c.facet_id is not None]

    if binding.state == "bound":
        return select_bound_facet(axis, binding.ref, candidates=dispatchable, graph=graph)

    if binding.state == "request":
        if axis != "unit":
            raise MaterialisationUsageError(
                f"a request-state facet binding only applies to the unit axis, not {axis!r}"
            )
        return _select_unit_request(
            axis,
            binding,
            dispatchable=dispatchable,
            all_bindings=all_bindings,
            graph=graph,
            stat_var=stat_var,
            window=window,
            policies=policies,
            breadth=breadth,
        )

    policy = policies.get(axis)
    if policy is not None:
        match = _match_policy(policy, dispatchable, axis)
        if match is not None:
            return _selection(
                axis,
                policy.resolved_ref,
                basis="declared_default",
                dispatch_facet_id=match.facet_id,
                default_policy=policy.policy_node,
                default_policy_version=policy.policy_version,
                selection_reason=policy.selection_reason,
            )
        alternatives, alternatives_truncated = _alternatives(dispatchable, axis, graph)
        raise FacetNotAvailableError(
            f"facet axis {axis!r}'s declared default {policy.resolved_ref!r} has no observations",
            pointer=f"/facets/{axis}",
            alternatives=alternatives,
            alternatives_truncated=alternatives_truncated,
        )

    viable = _narrow_by_bound_axes(dispatchable, all_bindings)
    if not viable:
        raise DefaultFacetUndefinedError(
            f"facet axis {axis!r} is unbound, no default policy is declared, and no "
            "probed candidate has any observations",
            pointer=f"/facets/{axis}",
        )
    declared = _declared_refs(policies, dispatchable, all_bindings, except_axis=axis)
    ranked_pool = rank_pool(
        viable, stat_vars=[stat_var], window=window, declared=declared, breadth=breadth
    )
    viable, precedence, top = ranked_pool.pool, ranked_pool.precedence, ranked_pool.ranked[0]
    value = facet_axis_value(top, axis)
    if value is None:
        if axis == "provenance":
            raise DefaultFacetUndefinedError(
                "facet axis 'provenance' is unbound, no default policy is declared, and "
                "the winning candidate carries no provenance",
                pointer="/facets/provenance",
            )
        return _selection(
            axis,
            None,
            basis="deployment_fallback",
            dispatch_facet_id=top.facet_id,
            default_policy=DEPLOYMENT_FACET_POLICY,
            default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
            selection_reason="absent_on_selected_facet",
        )
    if len(viable) == 1:
        reason = "sole_candidate"
    elif all(facet_axis_value(c, axis) == value for c in viable):
        reason = "most_observations"
    elif _decided_by_precedence(top, viable, precedence):
        reason = "dfo_precedence"
    else:
        reason = "most_observations_over_disagreement"
    return _selection(
        axis,
        value,
        basis="deployment_fallback",
        dispatch_facet_id=top.facet_id,
        default_policy=DEPLOYMENT_FACET_POLICY,
        default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
        selection_reason=reason,
    )


def select_bound_facet(
    axis: str, ref: str | None, *, candidates: Sequence[Facet], graph: GraphReader
) -> FacetSelection:
    """Resolves an axis bound to `ref` (`None` for the facets that declare no value) to a
    candidate carrying it, with `basis="bound"`. A bound ref reads no declared default and
    ignores the window. An unavailable ref raises `FacetNotAvailableError` with distinct
    alternatives capped at `_MAX_ALTERNATIVES_PER_ISSUE`. Only candidates with `facet_id`
    dispatch."""
    dispatchable = [c for c in candidates if c.facet_id is not None]
    match = next((c for c in dispatchable if facet_axis_value(c, axis) == ref), None)
    if match is not None:
        return _selection(axis, ref, basis="bound", dispatch_facet_id=match.facet_id)
    alternatives, alternatives_truncated = _alternatives(dispatchable, axis, graph)
    bound_to = repr(ref) if ref is not None else "no declared value"
    raise FacetNotAvailableError(
        f"facet axis {axis!r} bound to {bound_to}, which has no observations",
        pointer=f"/facets/{axis}",
        alternatives=alternatives,
        alternatives_truncated=alternatives_truncated,
    )


def _select_unit_request(
    axis: str,
    binding: FacetBinding,
    *,
    dispatchable: Sequence[Facet],
    all_bindings: Mapping[str, FacetBinding],
    graph: GraphReader,
    stat_var: str,
    window: RequestedWindow | None,
    policies: Mapping[str, FacetDefaultPolicy | None],
    breadth: Mapping[str, int] | None,
) -> FacetSelection:
    """Resolves a unit request from candidates narrowed by every bound axis, then by the other
    axes' declared defaults and `window` (`fallback_pool`). This shared pool keeps unit
    selection consistent with sibling fallback choices. A declared default is matched before
    the window narrows the pool, as `select_facet` ignores `window` for a declared default.

    An empty pool raises `FacetNotAvailableError` with alternatives from the original
    dispatchable candidates.

    A current-price ISO request prefers a candidate denominated in that currency. Otherwise
    the declared or ranked default applies. The result must be a currency ref
    (`native_currency_iso`). Missing or other refs raise `NonCurrencyUnitError`
    (`code="facet_binding_unsupported"`).
    `member_conversion.decide_member_conversion` verifies that the ref's `Currency<ISO>`
    node is a live `CurrencyUnit`.
    """
    request = binding.request
    assert request is not None  # `FacetBinding.__post_init__` guarantees this.
    viable = _narrow_by_bound_axes(dispatchable, all_bindings)
    if not viable:
        alternatives, alternatives_truncated = _alternatives(dispatchable, axis, graph)
        raise FacetNotAvailableError(
            f"facet axis {axis!r} requests a unit conversion, but narrowing by the "
            "request's other bound axes leaves no probed candidate",
            pointer=f"/facets/{axis}",
            alternatives=alternatives,
            alternatives_truncated=alternatives_truncated,
        )
    declared = _declared_refs(policies, dispatchable, all_bindings, except_axis=axis)
    ranked_pool = rank_pool(
        viable, stat_vars=[stat_var], window=window, declared=declared, breadth=breadth
    )
    in_window, precedence = ranked_pool.pool, ranked_pool.precedence
    selection = _preferred_native_unit_selection(
        axis, request, in_window, window=window, precedence=precedence, breadth=breadth
    )
    if selection is None:
        selection = _select_unit_request_default(
            axis,
            viable,
            in_window,
            request,
            policy=policies.get(axis),
            window=window,
            precedence=precedence,
            breadth=breadth,
        )
    _require_currency_native_unit(selection.ref)
    return selection


def _is_iso_currency(currency: str) -> bool:
    """Recognises three-letter uppercase ISO 4217 codes. Spec role keys use a different
    shape. Mirrors the identity pattern without importing `search`."""
    return len(currency) == 3 and currency.isalpha() and currency.isupper()


def _unit_request_ref(request: UnitRequest) -> dict[str, object]:
    """Builds the grammar's `{currency, prices, base_year}` unit ref. ISO currencies pass
    through. Role keys become `{"role": ...}`. Includes `base_year` only when present, as
    current-price requests forbid it."""
    currency: object = (
        request.currency if _is_iso_currency(request.currency) else {"role": request.currency}
    )
    ref: dict[str, object] = {"currency": currency, "prices": request.prices}
    if request.base_year is not None:
        ref["base_year"] = request.base_year
    return ref


def _preferred_native_unit_selection(
    axis: str,
    request: UnitRequest,
    candidates: Sequence[Facet],
    *,
    window: RequestedWindow | None,
    precedence: frozenset[str],
    breadth: Mapping[str, int] | None,
) -> FacetSelection | None:
    """For a current-price ISO request, selects the top-ranked candidate denominated in
    `T`, preferring `Currency<T>` over a `CurrencyUnitOfMeasure` unit such as `USDollar`.
    Returns `None` for role currencies, constant prices, or no match, so the caller uses the
    default. Shared by member and group selection."""
    if request.prices != "current" or not _is_iso_currency(request.currency):
        return None
    matching = _currency_native_matches(candidates, axis, request.currency)
    if not matching:
        return None
    top = rank_candidates(matching, window=window, precedence=precedence, breadth=breadth)[0]
    return _selection(
        axis,
        facet_axis_value(top, axis),
        basis="deployment_fallback",
        dispatch_facet_id=top.facet_id,
        default_policy=DEPLOYMENT_FACET_POLICY,
        default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
        selection_reason="native_currency_preferred",
        request=request,
    )


def _select_unit_request_default(
    axis: str,
    viable: Sequence[Facet],
    in_window: Sequence[Facet],
    request: UnitRequest,
    *,
    policy: FacetDefaultPolicy | None,
    window: RequestedWindow | None,
    precedence: frozenset[str],
    breadth: Mapping[str, int] | None,
) -> FacetSelection:
    """Selects the declared `policy` from `viable`, narrowed by bound axes, or else the
    top-ranked default from `in_window`, `viable`'s fallback pool ranked with `precedence`.
    Every result carries the `UnitRequest`."""
    if policy is not None:
        match = _match_policy(policy, viable, axis)
        if match is not None:
            return _selection(
                axis,
                policy.resolved_ref,
                basis="declared_default",
                dispatch_facet_id=match.facet_id,
                default_policy=policy.policy_node,
                default_policy_version=policy.policy_version,
                selection_reason=policy.selection_reason,
                request=request,
            )
    top = rank_candidates(in_window, window=window, precedence=precedence, breadth=breadth)[0]
    value = facet_axis_value(top, axis)
    if value is None:
        reason = "absent_on_selected_facet"
    elif len(in_window) == 1:
        reason = "sole_candidate"
    elif all(facet_axis_value(c, axis) == value for c in in_window):
        reason = "most_observations"
    elif _decided_by_precedence(top, in_window, precedence):
        reason = "dfo_precedence"
    else:
        reason = "most_observations_over_disagreement"
    return _selection(
        axis,
        value,
        basis="deployment_fallback",
        dispatch_facet_id=top.facet_id,
        default_policy=DEPLOYMENT_FACET_POLICY,
        default_policy_version=DEPLOYMENT_FACET_POLICY_VERSION,
        selection_reason=reason,
        request=request,
    )


def _require_currency_native_unit(ref: str | None) -> None:
    """Raises `NonCurrencyUnitError` for a ref `native_currency_iso` does not map to a currency.
    Live `CurrencyUnit` validation happens downstream without another graph read here."""
    if native_currency_iso(ref) is not None:
        return
    raise NonCurrencyUnitError(
        f"unit request resolved to native unit {ref!r}, which is not a currency",
        unit=ref if ref is not None else "",
        pointer="/facets/unit",
    )


def _narrow_by_bound_axes_only(
    dispatchable: Sequence[Facet], all_bindings: Mapping[str, FacetBinding]
) -> list[Facet]:
    """Filters by bound-axis refs only. Group selection intersects member candidates before
    currency preference. Applying preference per member could remove a facet the group shares."""
    viable = list(dispatchable)
    for bound_axis in FACET_AXES:
        binding = all_bindings.get(bound_axis, FacetBinding(state="unbound"))
        if binding.state == "bound":
            viable = [c for c in viable if facet_axis_value(c, bound_axis) == binding.ref]
    return viable


def _narrow_by_currency_native_request(
    viable: Sequence[Facet], all_bindings: Mapping[str, FacetBinding]
) -> list[Facet]:
    """For a current-price ISO unit request, keeps the candidates denominated in that
    currency when any exist (`_currency_native_matches`). Otherwise returns `viable`
    unchanged. Groups apply this after member pools intersect."""
    unit_binding = all_bindings.get("unit", FacetBinding(state="unbound"))
    if unit_binding.state == "request":
        request = unit_binding.request
        assert request is not None  # `FacetBinding.__post_init__` guarantees this.
        if request.prices == "current" and _is_iso_currency(request.currency):
            currency_matched = _currency_native_matches(viable, "unit", request.currency)
            if currency_matched:
                return currency_matched
    return list(viable)


def _narrow_by_bound_axes(
    dispatchable: Sequence[Facet], all_bindings: Mapping[str, FacetBinding]
) -> list[Facet]:
    """Filters by bound-axis refs, then applies any currency-native unit preference. All
    deployment-fallback axes use this pool, so they choose a facet compatible with the unit
    request. Group selection splits the steps and applies currency preference after member
    pools intersect."""
    viable = _narrow_by_bound_axes_only(dispatchable, all_bindings)
    return _narrow_by_currency_native_request(viable, all_bindings)


def joint_dispatch_facet_ids(
    candidates: Sequence[Facet], selections: Mapping[str, FacetSelection]
) -> tuple[str, ...]:
    """Returns sorted unique ids for candidates matching every selection at once. Independent
    per-axis witnesses can miss a candidate that matches all axes, excluding its rows before
    post-fetch filtering. Only candidates with `facet_id` dispatch. An empty tuple signals no
    joint match."""
    refs = {axis: selection.ref for axis, selection in selections.items()}
    matching_ids = {
        candidate.facet_id
        for candidate in candidates
        if candidate.facet_id is not None and _matches_refs(candidate, refs)
    }
    return tuple(sorted(matching_ids))


def _matches_refs(candidate: Facet, refs: Mapping[str, str | None]) -> bool:
    """Whether `candidate` carries every axis value in `refs`."""
    return all(facet_axis_value(candidate, axis) == ref for axis, ref in refs.items())


def facet_axis_value(facet: Facet, axis: str) -> str | None:
    """`facet`'s value on one of `FACET_AXES`, or `None` when the facet declares none. A
    bound ref selects the candidates whose value equals it."""
    return getattr(facet, _AXIS_FIELDS[axis])


def _match_policy(
    policy: FacetDefaultPolicy, candidates: Sequence[Facet], axis: str
) -> Facet | None:
    """Returns the first candidate matching `policy.resolved_ref`, or `None`."""
    return next((c for c in candidates if facet_axis_value(c, axis) == policy.resolved_ref), None)


def _selection(
    axis: str,
    ref: str | None,
    *,
    basis: Literal["bound", "declared_default", "deployment_fallback"],
    dispatch_facet_id: str | None,
    default_policy: str | None = None,
    default_policy_version: str | None = None,
    selection_reason: str | None = None,
    request: UnitRequest | None = None,
) -> FacetSelection:
    # A null `ref` is valid. Each selection names its concrete dispatch facet.
    assert dispatch_facet_id is not None
    return FacetSelection(
        axis=axis,
        ref=ref,
        basis=basis,
        default_policy=default_policy,
        default_policy_version=default_policy_version,
        selection_reason=selection_reason,
        dispatch_facet_id=dispatch_facet_id,
        request=request,
    )


def _alternatives(
    candidates: Sequence[Facet], axis: str, graph: GraphReader
) -> tuple[list[Alternative], bool]:
    """Returns capped alternatives and whether the cap cut the distinct value list. Dedupes
    before capping, so truncation is known without extra label lookups."""
    values = list(
        dict.fromkeys(v for c in candidates if (v := facet_axis_value(c, axis)) is not None)
    )
    truncated = len(values) > _MAX_ALTERNATIVES_PER_ISSUE
    # Capped before the label lookups below: candidates are only
    # deduplicated by axis value in the line above, so capping any earlier
    # could drop distinct values a later duplicate would have reintroduced.
    values = values[:_MAX_ALTERNATIVES_PER_ISSUE]
    return [Alternative(ref=v, label=graph.node_label(v) or v) for v in values], truncated
